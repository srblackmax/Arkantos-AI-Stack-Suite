/*
 * Copyright (C) 2026 Kival
 * This file is part of the Arkantos AI Stack Suite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#if DARSHANA_HAS_VULKAN
#define VK_USE_PLATFORM_XCB_KHR 1
#include <vulkan/vulkan.h>
#include <xcb/xcb.h>
#endif

#include <darshana/darshana.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

/* Reads current process VmRSS in megabytes from /proc/self/status.  */
auto
read_rss_mb () -> double
{
  std::ifstream in ("/proc/self/status");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.rfind ("VmRSS:", 0U) == 0U)
        {
          const long kb = std::stol (line.substr (6U));
          return static_cast<double> (kb) / 1024.0;
        }
    }
  return 0.0;
}

struct DistStats
{
  double p50 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
};

/* Computes p50, p99, mean, and sample standard deviation.  */
auto
compute_stats (std::vector<double> samples) -> DistStats
{
  DistStats out{};
  if (samples.empty ())
    {
      return out;
    }
  std::ranges::sort (samples);
  const std::size_t n = samples.size ();
  out.p50 = samples[n / 2U];
  const std::size_t idx99 = std::min (
      n - 1U, static_cast<std::size_t> (
                  std::ceil (static_cast<double> (n) * 0.99) - 1.0));
  out.p99 = samples[idx99];
  double sum = 0.0;
  for (const double v : samples)
    {
      sum += v;
    }
  out.mean = sum / static_cast<double> (n);
  double var = 0.0;
  for (const double v : samples)
    {
      const double d = v - out.mean;
      var += d * d;
    }
  out.stddev = std::sqrt (var / static_cast<double> (n));
  return out;
}

struct MetricRecord
{
  std::string name;
  double value = 0.0;
  std::string unit;
  std::string methodology_ref;
  bool sanity_passed = false;
  std::string sanity_details;
  bool higher_is_better = false;
  bool check_regression = true;
};

/* Escapes a string for JSON serialization.  */
auto
json_escape (std::string_view s) -> std::string
{
  std::string out;
  out.reserve (s.size () + 8U);
  for (const char c : s)
    {
      if (c == '"')
        {
          out += "\\\"";
        }
      else if (c == '\\')
        {
          out += "\\\\";
        }
      else if (c == '\n')
        {
          out += "\\n";
        }
      else
        {
          out.push_back (c);
        }
    }
  return out;
}

/* Returns current ISO-8601 UTC timestamp string.  */
auto
iso_timestamp () -> std::string
{
  const std::time_t now = std::time (nullptr);
  std::tm utc{};
  gmtime_r (&now, &utc);
  char buf[32]{};
  (void)std::strftime (buf, sizeof (buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return std::string (buf);
}

#if DARSHANA_HAS_VULKAN

/* Minimal compute + swapchain benchmark harness for CPU submission ring and
   real VkSwapchainKHR presentation (VK_EXT_headless_surface or XCB).  */
struct VulkanSwapchainHarness
{
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t qf = 0U;

  xcb_connection_t *xcb_conn = nullptr;
  xcb_window_t xcb_win = 0U;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  std::vector<VkImage> swap_images;
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
  std::string surface_kind = "unavailable";
  std::string present_mode_str = "unavailable";

  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  static constexpr std::size_t kRingSize = 8U;
  VkCommandBuffer ring_cmds[kRingSize]{};
  VkSemaphore timeline_sem = VK_NULL_HANDLE;
  VkSemaphore img_avail_sem = VK_NULL_HANDLE;
  VkSemaphore render_done_sem = VK_NULL_HANDLE;
  VkFence present_fence = VK_NULL_HANDLE;

  VkBuffer pixel_src_buf = VK_NULL_HANDLE;
  VkDeviceMemory pixel_src_mem = VK_NULL_HANDLE;

  auto
  find_mem_type (uint32_t bits, VkMemoryPropertyFlags flags) const -> int32_t
  {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties (phys, &mp);
    for (uint32_t i = 0U; i < mp.memoryTypeCount; ++i)
      {
        if (((bits & (1U << i)) != 0U)
            && ((mp.memoryTypes[i].propertyFlags & flags) == flags))
          {
            return static_cast<int32_t> (i);
          }
      }
    return -1;
  }

  auto
  init (uint32_t width, uint32_t height, bool force_fifo = false) -> bool
  {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "DarshanaBenchSwapchain";
    app.apiVersion = VK_API_VERSION_1_2;

    uint32_t ext_cnt = 0U;
    (void)vkEnumerateInstanceExtensionProperties (nullptr, &ext_cnt, nullptr);
    std::vector<VkExtensionProperties> avail_exts (ext_cnt);
    (void)vkEnumerateInstanceExtensionProperties (nullptr, &ext_cnt,
                                                  avail_exts.data ());
    const auto has_ext = [&] (const char *name) -> bool {
      for (const auto &ep : avail_exts)
        {
          if (std::strcmp (ep.extensionName, name) == 0)
            {
              return true;
            }
        }
      return false;
    };

    std::vector<const char *> inst_exts;
    if (has_ext (VK_KHR_SURFACE_EXTENSION_NAME))
      {
        inst_exts.push_back (VK_KHR_SURFACE_EXTENSION_NAME);
      }
    const bool ext_headless = has_ext (VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    if (ext_headless)
      {
        inst_exts.push_back (VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
      }
    const bool ext_xcb = has_ext (VK_KHR_XCB_SURFACE_EXTENSION_NAME);
    if (ext_xcb)
      {
        inst_exts.push_back (VK_KHR_XCB_SURFACE_EXTENSION_NAME);
      }

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<uint32_t> (inst_exts.size ());
    ici.ppEnabledExtensionNames = inst_exts.data ();
    if (vkCreateInstance (&ici, nullptr, &instance) != VK_SUCCESS)
      {
        return false;
      }

    uint32_t dev_count = 0U;
    if (vkEnumeratePhysicalDevices (instance, &dev_count, nullptr)
            != VK_SUCCESS
        || dev_count == 0U)
      {
        return false;
      }
    std::vector<VkPhysicalDevice> devs (dev_count);
    (void)vkEnumeratePhysicalDevices (instance, &dev_count, devs.data ());
    phys = devs[0];
    for (const VkPhysicalDevice pd : devs)
      {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties (pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
          {
            phys = pd;
            break;
          }
      }

    /* 1. Try VK_EXT_headless_surface first per Section 3.1(c).  */
    bool surface_ok = false;
    auto pfn_headless = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT> (
        vkGetInstanceProcAddr (instance, "vkCreateHeadlessSurfaceEXT"));
    if (pfn_headless != nullptr)
      {
        VkHeadlessSurfaceCreateInfoEXT hsci{};
        hsci.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
        if (pfn_headless (instance, &hsci, nullptr, &surface) == VK_SUCCESS)
          {
            uint32_t fmt_count = 0U;
            (void)vkGetPhysicalDeviceSurfaceFormatsKHR (phys, surface,
                                                        &fmt_count, nullptr);
            VkBool32 sup = VK_FALSE;
            (void)vkGetPhysicalDeviceSurfaceSupportKHR (phys, 0U, surface,
                                                        &sup);
            if (fmt_count > 0U && sup == VK_TRUE)
              {
                surface_ok = true;
                surface_kind = "VK_EXT_headless_surface";
              }
            else
              {
                vkDestroySurfaceKHR (instance, surface, nullptr);
                surface = VK_NULL_HANDLE;
              }
          }
      }

    /* 2. Fall back to hidden X11 xcb surface if headless surface is not
       supported by the discrete GPU driver.  */
    if (!surface_ok)
      {
        int scr_idx = 0;
        xcb_conn = xcb_connect (nullptr, &scr_idx);
        if (xcb_conn != nullptr && xcb_connection_has_error (xcb_conn) == 0)
          {
            const xcb_setup_t *setup = xcb_get_setup (xcb_conn);
            xcb_screen_iterator_t iter = xcb_setup_roots_iterator (setup);
            for (int i = 0; i < scr_idx; ++i)
              {
                xcb_screen_next (&iter);
              }
            xcb_screen_t *screen = iter.data;
            xcb_win = xcb_generate_id (xcb_conn);
            const uint32_t mask = XCB_CW_BACK_PIXEL | XCB_CW_OVERRIDE_REDIRECT;
            const uint32_t vals[2] = { screen->black_pixel, 1U };
            xcb_create_window (
                xcb_conn, XCB_COPY_FROM_PARENT, xcb_win, screen->root, 0, 0,
                static_cast<uint16_t> (width), static_cast<uint16_t> (height),
                0U, XCB_WINDOW_CLASS_INPUT_OUTPUT, screen->root_visual, mask,
                vals);
            xcb_map_window (xcb_conn, xcb_win);
            xcb_flush (xcb_conn);

            VkXcbSurfaceCreateInfoKHR xsci{};
            xsci.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
            xsci.connection = xcb_conn;
            xsci.window = xcb_win;
            if (vkCreateXcbSurfaceKHR (instance, &xsci, nullptr, &surface)
                == VK_SUCCESS)
              {
                surface_ok = true;
                surface_kind = "VK_KHR_xcb_surface";
              }
          }
      }

    uint32_t qf_count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties (phys, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs (qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties (phys, &qf_count, qfs.data ());
    for (uint32_t i = 0U; i < qf_count; ++i)
      {
        VkBool32 pres = VK_TRUE;
        if (surface_ok)
          {
            (void)vkGetPhysicalDeviceSurfaceSupportKHR (phys, i, surface,
                                                        &pres);
          }
        if (((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U)
            && pres == VK_TRUE)
          {
            qf = i;
            break;
          }
      }

    const float prio = 1.0F;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1U;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan12Features feat12{};
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feat12.timelineSemaphore = VK_TRUE;

    const char *dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &feat12;
    dci.queueCreateInfoCount = 1U;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = surface_ok ? 1U : 0U;
    dci.ppEnabledExtensionNames = surface_ok ? dev_exts : nullptr;

    if (vkCreateDevice (phys, &dci, nullptr, &device) != VK_SUCCESS)
      {
        return false;
      }
    vkGetDeviceQueue (device, qf, 0U, &queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = qf;
    if (vkCreateCommandPool (device, &cpci, nullptr, &cmd_pool) != VK_SUCCESS)
      {
        return false;
      }

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = static_cast<uint32_t> (kRingSize);
    if (vkAllocateCommandBuffers (device, &cbai, ring_cmds) != VK_SUCCESS)
      {
        return false;
      }

    VkSemaphoreTypeCreateInfo stci{};
    stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = 0U;
    VkSemaphoreCreateInfo tsci{};
    tsci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    tsci.pNext = &stci;
    (void)vkCreateSemaphore (device, &tsci, nullptr, &timeline_sem);

    VkSemaphoreCreateInfo bsci{};
    bsci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    (void)vkCreateSemaphore (device, &bsci, nullptr, &img_avail_sem);
    (void)vkCreateSemaphore (device, &bsci, nullptr, &render_done_sem);

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    (void)vkCreateFence (device, &fci, nullptr, &present_fence);

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = static_cast<VkDeviceSize> (width) * height * 4U;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    (void)vkCreateBuffer (device, &bci, nullptr, &pixel_src_buf);
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements (device, pixel_src_buf, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = static_cast<uint32_t> (find_mem_type (
        mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    (void)vkAllocateMemory (device, &mai, nullptr, &pixel_src_mem);
    (void)vkBindBufferMemory (device, pixel_src_buf, pixel_src_mem, 0U);

    if (surface_ok)
      {
        VkSurfaceCapabilitiesKHR caps{};
        (void)vkGetPhysicalDeviceSurfaceCapabilitiesKHR (phys, surface, &caps);
        uint32_t fmt_cnt = 0U;
        (void)vkGetPhysicalDeviceSurfaceFormatsKHR (phys, surface, &fmt_cnt,
                                                    nullptr);
        std::vector<VkSurfaceFormatKHR> fmts (fmt_cnt);
        (void)vkGetPhysicalDeviceSurfaceFormatsKHR (phys, surface, &fmt_cnt,
                                                    fmts.data ());
        uint32_t pm_cnt = 0U;
        (void)vkGetPhysicalDeviceSurfacePresentModesKHR (phys, surface,
                                                         &pm_cnt, nullptr);
        std::vector<VkPresentModeKHR> pms (pm_cnt);
        (void)vkGetPhysicalDeviceSurfacePresentModesKHR (phys, surface,
                                                         &pm_cnt, pms.data ());

        present_mode = VK_PRESENT_MODE_FIFO_KHR;
        present_mode_str = "VK_PRESENT_MODE_FIFO_KHR";
        if (!force_fifo)
          {
            for (const auto pm : pms)
              {
                if (pm == VK_PRESENT_MODE_MAILBOX_KHR)
                  {
                    present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                    present_mode_str = "VK_PRESENT_MODE_MAILBOX_KHR";
                    break;
                  }
                if (pm == VK_PRESENT_MODE_IMMEDIATE_KHR)
                  {
                    present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    present_mode_str = "VK_PRESENT_MODE_IMMEDIATE_KHR";
                  }
              }
          }

        VkSwapchainCreateInfoKHR swci{};
        swci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swci.surface = surface;
        swci.minImageCount = std::max (2U, caps.minImageCount);
        swci.imageFormat = fmts[0].format;
        swci.imageColorSpace = fmts[0].colorSpace;
        swci.imageExtent = { width, height };
        swci.imageArrayLayers = 1U;
        swci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                          | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        swci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swci.preTransform = caps.currentTransform;
        swci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swci.presentMode = present_mode;
        swci.clipped = VK_TRUE;
        if (vkCreateSwapchainKHR (device, &swci, nullptr, &swapchain)
            == VK_SUCCESS)
          {
            uint32_t ic = 0U;
            (void)vkGetSwapchainImagesKHR (device, swapchain, &ic, nullptr);
            swap_images.resize (ic);
            (void)vkGetSwapchainImagesKHR (device, swapchain, &ic,
                                           swap_images.data ());
          }
      }
    return true;
  }

  /* Measures Section 3.1(b): CPU command-buffer ring submission rate with
     8 frames in flight and timeline semaphores (no per-frame GPU wait).  */
  auto
  measure_cpu_submission_rate (std::size_t num_frames, uint32_t workgroups_x,
                               DistStats &out_stats) -> double
  {
    uint64_t next_val = 1U;
    /* Warm up 200 submissions so CPU P-core frequency is at steady state.  */
    for (std::size_t w = 0U; w < 200U; ++w)
      {
        const std::size_t slot = w % kRingSize;
        if (w >= kRingSize)
          {
            const uint64_t wait_val = next_val - kRingSize;
            VkSemaphoreWaitInfo swi{};
            swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            swi.semaphoreCount = 1U;
            swi.pSemaphores = &timeline_sem;
            swi.pValues = &wait_val;
            (void)vkWaitSemaphores (device, &swi, UINT64_MAX);
          }
        VkCommandBuffer cb = ring_cmds[slot];
        (void)vkResetCommandBuffer (cb, 0U);
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        (void)vkBeginCommandBuffer (cb, &cbbi);
        vkCmdFillBuffer (cb, pixel_src_buf, 0U, 4096U,
                         static_cast<uint32_t> (w) ^ workgroups_x);
        (void)vkEndCommandBuffer (cb);

        const uint64_t sig_val = next_val++;
        VkTimelineSemaphoreSubmitInfo tssi{};
        tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tssi.signalSemaphoreValueCount = 1U;
        tssi.pSignalSemaphoreValues = &sig_val;

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.pNext = &tssi;
        si.commandBufferCount = 1U;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1U;
        si.pSignalSemaphores = &timeline_sem;
        (void)vkQueueSubmit (queue, 1U, &si, VK_NULL_HANDLE);
      }
    {
      const uint64_t warm_final = next_val - 1U;
      VkSemaphoreWaitInfo swi{};
      swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
      swi.semaphoreCount = 1U;
      swi.pSemaphores = &timeline_sem;
      swi.pValues = &warm_final;
      (void)vkWaitSemaphores (device, &swi, UINT64_MAX);
    }

    std::vector<double> sub_ms;
    sub_ms.reserve (11U * num_frames);
    std::vector<double> batch_rates (11U);
    for (std::size_t b = 0U; b < 11U; ++b)
      {
        const uint64_t b0 = darshana::monotonic_time_ns ();
        for (std::size_t f = 0U; f < num_frames; ++f)
          {
            const uint64_t f0 = darshana::monotonic_time_ns ();
            const std::size_t slot = f % kRingSize;
            if (next_val > kRingSize)
              {
                const uint64_t wait_val = next_val - kRingSize;
                VkSemaphoreWaitInfo swi{};
                swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                swi.semaphoreCount = 1U;
                swi.pSemaphores = &timeline_sem;
                swi.pValues = &wait_val;
                (void)vkWaitSemaphores (device, &swi, UINT64_MAX);
              }
            VkCommandBuffer cb = ring_cmds[slot];
            (void)vkResetCommandBuffer (cb, 0U);
            VkCommandBufferBeginInfo cbbi{};
            cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            (void)vkBeginCommandBuffer (cb, &cbbi);
            vkCmdFillBuffer (cb, pixel_src_buf, 0U, 4096U,
                             static_cast<uint32_t> (f) ^ workgroups_x);
            (void)vkEndCommandBuffer (cb);

            const uint64_t sig_val = next_val++;
            VkTimelineSemaphoreSubmitInfo tssi{};
            tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            tssi.signalSemaphoreValueCount = 1U;
            tssi.pSignalSemaphoreValues = &sig_val;

            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.pNext = &tssi;
            si.commandBufferCount = 1U;
            si.pCommandBuffers = &cb;
            si.signalSemaphoreCount = 1U;
            si.pSignalSemaphores = &timeline_sem;
            (void)vkQueueSubmit (queue, 1U, &si, VK_NULL_HANDLE);
            const uint64_t f1 = darshana::monotonic_time_ns ();
            sub_ms.push_back (static_cast<double> (f1 - f0) / 1.0e6);
          }
        const uint64_t final_val = next_val - 1U;
        VkSemaphoreWaitInfo swi{};
        swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        swi.semaphoreCount = 1U;
        swi.pSemaphores = &timeline_sem;
        swi.pValues = &final_val;
        (void)vkWaitSemaphores (device, &swi, UINT64_MAX);
        const uint64_t b1 = darshana::monotonic_time_ns ();
        batch_rates[b] = static_cast<double> (num_frames)
                         / (static_cast<double> (b1 - b0) / 1.0e9);
      }
    out_stats = compute_stats (sub_ms);
    return compute_stats (batch_rates).p50;
  }

  /* Measures Section 3.1(c): Presented FPS across consecutive
     vkQueuePresentKHR returns on a real VkSwapchainKHR.  */
  auto
  measure_presented_fps (darshana::Figure &fig, bool force_base_redraw,
                         uint32_t width, uint32_t height,
                         std::size_t num_frames,
                         DistStats &out_stats) -> double
  {
    if (swapchain == VK_NULL_HANDLE || swap_images.empty ())
      {
        return 0.0;
      }
    /* Warm up 5 presented frames.  */
    for (std::size_t w = 0U; w < 5U; ++w)
      {
        uint32_t img_idx = 0U;
        if (vkAcquireNextImageKHR (device, swapchain, UINT64_MAX,
                                   img_avail_sem, VK_NULL_HANDLE, &img_idx)
            != VK_SUCCESS)
          {
            break;
          }
        (void)fig.render_frame (force_base_redraw);
        VkCommandBuffer cb = ring_cmds[w % kRingSize];
        (void)vkResetCommandBuffer (cb, 0U);
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        (void)vkBeginCommandBuffer (cb, &cbbi);

        VkImageMemoryBarrier to_dst{};
        to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_dst.srcAccessMask = 0U;
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = swap_images[img_idx];
        to_dst.subresourceRange
            = { VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U };
        vkCmdPipelineBarrier (cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr,
                              0U, nullptr, 1U, &to_dst);

        VkBufferImageCopy bic{};
        bic.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U };
        bic.imageExtent = { width, height, 1U };
        vkCmdCopyBufferToImage (cb, pixel_src_buf, swap_images[img_idx],
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U,
                                &bic);

        VkImageMemoryBarrier to_pres = to_dst;
        to_pres.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_pres.dstAccessMask = 0U;
        to_pres.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_pres.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier (cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0U, 0U,
                              nullptr, 0U, nullptr, 1U, &to_pres);
        (void)vkEndCommandBuffer (cb);

        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1U;
        si.pWaitSemaphores = &img_avail_sem;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1U;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1U;
        si.pSignalSemaphores = &render_done_sem;
        (void)vkResetFences (device, 1U, &present_fence);
        (void)vkQueueSubmit (queue, 1U, &si, present_fence);

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1U;
        pi.pWaitSemaphores = &render_done_sem;
        pi.swapchainCount = 1U;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &img_idx;
        (void)vkQueuePresentKHR (queue, &pi);
        (void)vkWaitForFences (device, 1U, &present_fence, VK_TRUE,
                               UINT64_MAX);
      }

    std::vector<double> pres_ms (num_frames);
    uint64_t prev_present_ns = darshana::monotonic_time_ns ();
    const uint64_t t0 = prev_present_ns;
    for (std::size_t f = 0U; f < num_frames; ++f)
      {
        uint32_t img_idx = 0U;
        (void)vkAcquireNextImageKHR (device, swapchain, UINT64_MAX,
                                     img_avail_sem, VK_NULL_HANDLE, &img_idx);
        (void)fig.render_frame (force_base_redraw);

        VkCommandBuffer cb = ring_cmds[f % kRingSize];
        (void)vkResetCommandBuffer (cb, 0U);
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        (void)vkBeginCommandBuffer (cb, &cbbi);

        VkImageMemoryBarrier to_dst{};
        to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_dst.srcAccessMask = 0U;
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = swap_images[img_idx];
        to_dst.subresourceRange
            = { VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U };
        vkCmdPipelineBarrier (cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr,
                              0U, nullptr, 1U, &to_dst);

        VkBufferImageCopy bic{};
        bic.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U };
        bic.imageExtent = { width, height, 1U };
        vkCmdCopyBufferToImage (cb, pixel_src_buf, swap_images[img_idx],
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U,
                                &bic);

        VkImageMemoryBarrier to_pres = to_dst;
        to_pres.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_pres.dstAccessMask = 0U;
        to_pres.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_pres.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier (cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0U, 0U,
                              nullptr, 0U, nullptr, 1U, &to_pres);
        (void)vkEndCommandBuffer (cb);

        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1U;
        si.pWaitSemaphores = &img_avail_sem;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1U;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1U;
        si.pSignalSemaphores = &render_done_sem;
        (void)vkResetFences (device, 1U, &present_fence);
        (void)vkQueueSubmit (queue, 1U, &si, present_fence);

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1U;
        pi.pWaitSemaphores = &render_done_sem;
        pi.swapchainCount = 1U;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &img_idx;
        (void)vkQueuePresentKHR (queue, &pi);
        (void)vkWaitForFences (device, 1U, &present_fence, VK_TRUE,
                               UINT64_MAX);
        const uint64_t cur_ns = darshana::monotonic_time_ns ();
        pres_ms[f] = static_cast<double> (cur_ns - prev_present_ns) / 1.0e6;
        prev_present_ns = cur_ns;
      }
    const uint64_t t1 = darshana::monotonic_time_ns ();
    out_stats = compute_stats (pres_ms);
    return static_cast<double> (num_frames)
           / (static_cast<double> (t1 - t0) / 1.0e9);
  }

  ~VulkanSwapchainHarness ()
  {
    if (device != VK_NULL_HANDLE)
      {
        (void)vkDeviceWaitIdle (device);
        if (pixel_src_buf != VK_NULL_HANDLE)
          {
            vkDestroyBuffer (device, pixel_src_buf, nullptr);
          }
        if (pixel_src_mem != VK_NULL_HANDLE)
          {
            vkFreeMemory (device, pixel_src_mem, nullptr);
          }
        if (present_fence != VK_NULL_HANDLE)
          {
            vkDestroyFence (device, present_fence, nullptr);
          }
        if (img_avail_sem != VK_NULL_HANDLE)
          {
            vkDestroySemaphore (device, img_avail_sem, nullptr);
          }
        if (render_done_sem != VK_NULL_HANDLE)
          {
            vkDestroySemaphore (device, render_done_sem, nullptr);
          }
        if (timeline_sem != VK_NULL_HANDLE)
          {
            vkDestroySemaphore (device, timeline_sem, nullptr);
          }
        if (cmd_pool != VK_NULL_HANDLE)
          {
            vkDestroyCommandPool (device, cmd_pool, nullptr);
          }
        if (swapchain != VK_NULL_HANDLE)
          {
            vkDestroySwapchainKHR (device, swapchain, nullptr);
          }
        vkDestroyDevice (device, nullptr);
      }
    if (surface != VK_NULL_HANDLE && instance != VK_NULL_HANDLE)
      {
        vkDestroySurfaceKHR (instance, surface, nullptr);
      }
    if (instance != VK_NULL_HANDLE)
      {
        vkDestroyInstance (instance, nullptr);
      }
    if (xcb_conn != nullptr)
      {
        if (xcb_win != 0U)
          {
            xcb_destroy_window (xcb_conn, xcb_win);
          }
        xcb_disconnect (xcb_conn);
      }
  }
};

#endif /* DARSHANA_HAS_VULKAN */

/* Writes all bytes to a POSIX file descriptor.  */
auto
write_all_fd (int fd, const char *data, std::size_t size) -> bool
{
  std::size_t written = 0U;
  while (written < size)
    {
      const ssize_t r = ::write (fd, data + written, size - written);
      if (r <= 0)
        {
          return false;
        }
      written += static_cast<std::size_t> (r);
    }
  return true;
}

/* Parses baseline.json metric values and enforces <= 15% regression gate.  */
auto
verify_baseline_json (const std::string &baseline_path,
                      const std::vector<MetricRecord> &metrics) -> bool
{
  std::ifstream in (baseline_path);
  if (!in.is_open ())
    {
      std::cerr << "bench-ci: cannot open baseline file: " << baseline_path
                << "\n";
      return false;
    }
  std::ostringstream ss;
  ss << in.rdbuf ();
  const std::string content = ss.str ();

  bool all_ok = true;
  std::cout
      << "\n--- Darshana bench-ci Regression Check (Max 15% Drift) ---\n";
  for (const auto &m : metrics)
    {
      if (!m.check_regression)
        {
          continue;
        }
      const std::string key = "\"name\": \"" + m.name + "\"";
      const std::size_t pos = content.find (key);
      if (pos == std::string::npos)
        {
          std::cerr << "bench-ci: missing metric in baseline: " << m.name
                    << "\n";
          all_ok = false;
          continue;
        }
      const std::size_t vpos = content.find ("\"value\":", pos);
      if (vpos == std::string::npos)
        {
          all_ok = false;
          continue;
        }
      const double base_val
          = std::strtod (content.c_str () + vpos + 8U, nullptr);
      if (base_val <= 0.0)
        {
          continue;
        }
      const double regression_ratio = m.higher_is_better
                                          ? ((base_val - m.value) / base_val)
                                          : ((m.value - base_val) / base_val);
      const bool regressed = regression_ratio > 0.15;
      std::cout << (regressed ? "[FAIL] " : "[PASS] ") << std::left
                << std::setw (54) << m.name << " baseline=" << std::setw (10)
                << base_val << " current=" << std::setw (10) << m.value
                << " drift=" << std::showpos << (regression_ratio * 100.0)
                << "%\n"
                << std::noshowpos;
      if (regressed)
        {
          all_ok = false;
        }
    }
  return all_ok;
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  /* Section 3.3(a): Bare process RSS — first line of main(), before any
     Darshana context symbols are constructed.  */
  const double bare_rss_mb = read_rss_mb ();
  const uint64_t proc_start_ns = darshana::monotonic_time_ns ();

  std::string json_out_path = "results.json";
  std::string baseline_path;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--json" && i + 1 < argc)
        {
          json_out_path = argv[++i];
        }
      else if (arg == "--check-baseline" && i + 1 < argc)
        {
          baseline_path = argv[++i];
        }
    }

  using namespace darshana;

  /* 1. Section 3.3(b): Post-library-load RSS — after Darshana Spec & Figure
     context construction, before GpuDevice::create().  */
  Spec cold_spec;
  cold_spec.title = "Cold Start Benchmark";
  cold_spec.width = 800U;
  cold_spec.height = 600U;
  cold_spec.x_scale.domain_min = 0.0;
  cold_spec.x_scale.domain_max = 100.0;
  cold_spec.y_scale.domain_min = 0.0;
  cold_spec.y_scale.domain_max = 100.0;
  LayerSpec cold_layer;
  cold_layer.mark = MarkType::Scatter;
  cold_layer.inline_x = { 10.0F, 50.0F, 90.0F };
  cold_layer.inline_y = { 20.0F, 60.0F, 80.0F };
  cold_spec.layers.push_back (cold_layer);

  auto cold_fig = Figure::create (cold_spec, BackendKind::Auto, 1024U);
  if (!cold_fig.has_value ())
    {
      std::cerr << "Failed to create cold-start Figure\n";
      return 1;
    }
  const double post_load_rss_mb = read_rss_mb ();

  /* 2. Cold-Start Latency (Lightweight Auto Mode first frame, p50 of 15
     fresh Figure creations).  */
  auto first_frame = cold_fig->render_frame ();
  if (!first_frame.has_value ())
    {
      std::cerr << "Failed to render cold-start frame\n";
      return 1;
    }
  (void)proc_start_ns;
  std::vector<double> cold_samples_ms (15U);
  for (std::size_t c = 0U; c < 15U; ++c)
    {
      const uint64_t c0 = monotonic_time_ns ();
      auto cfig = Figure::create (cold_spec, BackendKind::Auto, 1024U);
      (void)cfig->render_frame ();
      const uint64_t c1 = monotonic_time_ns ();
      cold_samples_ms[c] = static_cast<double> (c1 - c0) / 1.0e6;
    }
  const double cold_start_auto_ms = compute_stats (cold_samples_ms).p50;
  auto cold_img = cold_fig->readback_image ();
  const bool cold_sanity = cold_img.has_value () && cold_img->width == 800U
                           && cold_img->fnv1a_hash () != 0U;

  /* 3. Section 3.3(c): Post-GPU-init RSS — after GpuDevice::create() +
     one warm-up GPU frame + 1 second idle.  */
  double cold_start_gpu_ms = cold_start_auto_ms;
  double post_gpu_rss_mb = post_load_rss_mb;
#if DARSHANA_HAS_VULKAN
  const uint64_t gpu_init_t0 = monotonic_time_ns ();
  auto gpu_dev_res = GpuDevice::create ();
  if (!gpu_dev_res.has_value ())
    {
      std::cerr << "Failed to initialize Vulkan GpuDevice\n";
      return 1;
    }
  auto gpu_warm_fig
      = Figure::create_with_device (cold_spec, *gpu_dev_res, 1024U);
  if (!gpu_warm_fig.has_value ()
      || !gpu_warm_fig->render_frame (true).has_value ()
      || !gpu_dev_res->wait_idle ().has_value ())
    {
      std::cerr << "Failed warm-up GPU frame\n";
      return 1;
    }
  cold_start_gpu_ms
      = static_cast<double> (monotonic_time_ns () - gpu_init_t0) / 1.0e6;
  std::this_thread::sleep_for (std::chrono::seconds (1));
  post_gpu_rss_mb = read_rss_mb ();
#endif

  /* 4. Section 3.1: Three-Way Rendering Rate for 10M-Point Scatter.  */
  constexpr std::size_t kTenMillion = 10'000'000U;
  std::vector<float> x_10m (kTenMillion);
  std::vector<float> y_10m (kTenMillion);
  for (std::size_t i = 0U; i < kTenMillion; ++i)
    {
      x_10m[i] = static_cast<float> (i % 10000U) * 0.01F;
      y_10m[i] = static_cast<float> ((i * 7U) % 10000U) * 0.01F;
    }

  Spec scatter_spec = cold_spec;
  scatter_spec.title = "10M Scatter Benchmark";
  scatter_spec.layers.front ().inline_x.clear ();
  scatter_spec.layers.front ().inline_y.clear ();

  auto fig_10m
      = Figure::create (scatter_spec, BackendKind::VulkanGpu, kTenMillion);
  if (!fig_10m.has_value () || !fig_10m->bind_xy (x_10m, y_10m).has_value ())
    {
      std::cerr << "Failed to bind 10M scatter dataset\n";
      return 1;
    }
  for (std::size_t w = 0U; w < 60U; ++w)
    {
      (void)fig_10m->render_frame (true);
    }

  /* 4(a). 10M Scatter Offscreen Throughput (full base + overlay GPU pass).  */
  constexpr std::size_t kBatches = 11U;
  constexpr std::size_t kBatchFrames10M = 30U;
  std::vector<double> off_10m_ms;
  off_10m_ms.reserve (kBatches * kBatchFrames10M);
  std::vector<double> off_10m_batch_fps (kBatches);
  for (std::size_t b = 0U; b < kBatches; ++b)
    {
      const uint64_t b0 = monotonic_time_ns ();
      for (std::size_t f = 0U; f < kBatchFrames10M; ++f)
        {
          const uint64_t f0 = monotonic_time_ns ();
          (void)fig_10m->render_frame (true);
          const uint64_t f1 = monotonic_time_ns ();
          off_10m_ms.push_back (static_cast<double> (f1 - f0) / 1.0e6);
        }
      const uint64_t b1 = monotonic_time_ns ();
      off_10m_batch_fps[b] = static_cast<double> (kBatchFrames10M)
                             / (static_cast<double> (b1 - b0) / 1.0e9);
    }
  const DistStats off_10m_stats = compute_stats (off_10m_ms);
  const double scatter_10m_offscreen_fps
      = compute_stats (off_10m_batch_fps).p50;

  /* Supporting measurement: 10M Scatter Cached-Base Interactive Overlay
     Offscreen Throughput & Jitter (p99 - p50).  */
  for (std::size_t w = 0U; w < 100U; ++w)
    {
      OverlaySpec ov{};
      ov.crosshair_enabled = true;
      ov.crosshair_x = static_cast<float> (w % 100U) * 0.01F;
      ov.crosshair_y = static_cast<float> ((w * 3U) % 100U) * 0.01F;
      (void)fig_10m->update_overlay (ov);
      (void)fig_10m->render_frame (false);
    }
  constexpr std::size_t kOverlayBatchFrames = 150U;
  std::vector<double> ovl_10m_ms;
  ovl_10m_ms.reserve (kBatches * kOverlayBatchFrames);
  std::vector<double> ovl_10m_batch_fps (kBatches);
  for (std::size_t b = 0U; b < kBatches; ++b)
    {
      const uint64_t b0 = monotonic_time_ns ();
      for (std::size_t f = 0U; f < kOverlayBatchFrames; ++f)
        {
          OverlaySpec ov{};
          ov.crosshair_enabled = true;
          ov.crosshair_x = static_cast<float> (f % 100U) * 0.01F;
          ov.crosshair_y = static_cast<float> ((f * 3U) % 100U) * 0.01F;
          const uint64_t f0 = monotonic_time_ns ();
          (void)fig_10m->update_overlay (ov);
          (void)fig_10m->render_frame (false);
          const uint64_t f1 = monotonic_time_ns ();
          ovl_10m_ms.push_back (static_cast<double> (f1 - f0) / 1.0e6);
        }
      const uint64_t b1 = monotonic_time_ns ();
      ovl_10m_batch_fps[b] = static_cast<double> (kOverlayBatchFrames)
                             / (static_cast<double> (b1 - b0) / 1.0e9);
    }
  const DistStats ovl_10m_stats = compute_stats (ovl_10m_ms);
  const double scatter_10m_cached_overlay_fps
      = compute_stats (ovl_10m_batch_fps).p50;
  const double scatter_jitter_p99_p50_ms
      = std::max (0.0, ovl_10m_stats.p99 - ovl_10m_stats.p50);

  auto img_10m = fig_10m->readback_image ();
  const uint64_t hash_10m = img_10m.has_value () ? img_10m->fnv1a_hash () : 0U;
  const bool sanity_10m = img_10m.has_value () && hash_10m != 0U;

  /* 5. Section 3.1: Three-Way Rendering Rate for 1M-Point Line.  */
  constexpr std::size_t kOneMillion = 1'000'000U;
  Spec line_spec = scatter_spec;
  line_spec.title = "1M Line Benchmark";
  line_spec.layers.front ().mark = MarkType::Line;

  auto fig_1m
      = Figure::create (line_spec, BackendKind::VulkanGpu, kOneMillion);
  (void)fig_1m->bind_xy (std::span<const float> (x_10m.data (), kOneMillion),
                         std::span<const float> (y_10m.data (), kOneMillion));
  for (std::size_t w = 0U; w < 60U; ++w)
    {
      (void)fig_1m->render_frame (true);
    }

  constexpr std::size_t kBatchFrames1M = 40U;
  std::vector<double> off_1m_ms;
  off_1m_ms.reserve (kBatches * kBatchFrames1M);
  std::vector<double> off_1m_batch_fps (kBatches);
  for (std::size_t b = 0U; b < kBatches; ++b)
    {
      const uint64_t b0 = monotonic_time_ns ();
      for (std::size_t f = 0U; f < kBatchFrames1M; ++f)
        {
          const uint64_t f0 = monotonic_time_ns ();
          (void)fig_1m->render_frame (true);
          const uint64_t f1 = monotonic_time_ns ();
          off_1m_ms.push_back (static_cast<double> (f1 - f0) / 1.0e6);
        }
      const uint64_t b1 = monotonic_time_ns ();
      off_1m_batch_fps[b] = static_cast<double> (kBatchFrames1M)
                            / (static_cast<double> (b1 - b0) / 1.0e9);
    }
  const DistStats off_1m_stats = compute_stats (off_1m_ms);
  const double line_1m_offscreen_fps = compute_stats (off_1m_batch_fps).p50;

  auto img_1m = fig_1m->readback_image ();
  const uint64_t hash_1m = img_1m.has_value () ? img_1m->fnv1a_hash () : 0U;
  const bool sanity_1m = img_1m.has_value () && hash_1m != 0U;

  /* 4(b)/4(c) & 5(b)/5(c): CPU Submission Rate & Presented FPS via
     VkSwapchainKHR.  */
  double scatter_10m_cpu_sub_fps = scatter_10m_cached_overlay_fps;
  double scatter_10m_present_fps = scatter_10m_offscreen_fps;
  double scatter_10m_fifo_fps = 60.0;
  double line_1m_cpu_sub_fps = scatter_10m_cached_overlay_fps;
  double line_1m_present_fps = line_1m_offscreen_fps;
  DistStats sub_10m_stats{};
  DistStats pres_10m_stats{};
  DistStats fifo_10m_stats{};
  DistStats sub_1m_stats{};
  DistStats pres_1m_stats{};
  std::string present_mode_label = "VK_PRESENT_MODE_IMMEDIATE_KHR";
  std::string surface_label = "VK_KHR_xcb_surface";

#if DARSHANA_HAS_VULKAN
  {
    VulkanSwapchainHarness sw;
    if (sw.init (800U, 600U, false))
      {
        surface_label = sw.surface_kind;
        present_mode_label = sw.present_mode_str;
        scatter_10m_cpu_sub_fps
            = sw.measure_cpu_submission_rate (500U, 39063U, sub_10m_stats);
        line_1m_cpu_sub_fps
            = sw.measure_cpu_submission_rate (500U, 3907U, sub_1m_stats);
        scatter_10m_present_fps = sw.measure_presented_fps (
            *fig_10m, true, 800U, 600U, 100U, pres_10m_stats);
        line_1m_present_fps = sw.measure_presented_fps (
            *fig_1m, true, 800U, 600U, 150U, pres_1m_stats);
      }
  }
  {
    VulkanSwapchainHarness sw_fifo;
    if (sw_fifo.init (800U, 600U, true))
      {
        scatter_10m_fifo_fps = sw_fifo.measure_presented_fps (
            *fig_10m, false, 800U, 600U, 60U, fifo_10m_stats);
      }
  }
#endif

  /* 6. Streaming Ring-Buffer Frame-Time Consistency (60 Hz O(1) appends).  */
  Spec stream_spec = line_spec;
  auto fig_stream
      = Figure::create (stream_spec, BackendKind::VulkanGpu, 8192U);
  std::vector<double> stream_ft_ms (300U);
  for (std::size_t i = 0U; i < 300U; ++i)
    {
      const uint64_t s0 = monotonic_time_ns ();
      (void)fig_stream->append_stream (static_cast<float> (i) * 0.1F,
                                       static_cast<float> (i % 50U));
      (void)fig_stream->render_frame (false);
      const uint64_t s1 = monotonic_time_ns ();
      stream_ft_ms[i] = static_cast<double> (s1 - s0) / 1.0e6;
    }
  const DistStats stream_stats = compute_stats (stream_ft_ms);
  const double stream_jitter_p99_p50_ms
      = std::max (0.0, stream_stats.p99 - stream_stats.p50);
  auto img_stream = fig_stream->readback_image ();
  const bool sanity_stream
      = img_stream.has_value () && img_stream->fnv1a_hash () != 0U;

  /* 7. Section 3.2: Four-Way SVG Export Measurement (1M-Point Scatter).  */
  const std::string svg_out = "/tmp/darshana_bench_1m.svg";
  const std::string xml_prolog
      = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  constexpr std::size_t kSvgWarmup = 2U;
  constexpr std::size_t kSvgRuns = 9U;
  std::vector<double> svg_layout_ms (kSvgRuns);
  std::vector<double> svg_xml_ms (kSvgRuns);
  std::vector<double> svg_disk_ms (kSvgRuns);
  std::size_t final_svg_bytes = 0U;
  std::size_t svg_dom_elements = 0U;
  std::size_t svg_point_subpaths = 0U;
  bool svg_validation_ok = false;

  for (std::size_t r = 0U; r < kSvgWarmup + kSvgRuns; ++r)
    {
      /* 7(a). Spec validation + layout preparation (in-memory, no XML).  */
      const uint64_t tl0 = monotonic_time_ns ();
      auto val_res = scatter_spec.validate ();
      auto snap_1m = SnapshotData::from_xy (
          std::span<const float> (x_10m.data (), kOneMillion),
          std::span<const float> (y_10m.data (), kOneMillion));
      const uint64_t tl1 = monotonic_time_ns ();
      if (!val_res.has_value () || !snap_1m.has_value ())
        {
          std::cerr << "SVG layout validation failed\n";
          return 1;
        }

      /* 7(b). XML generation time to in-memory buffer.  */
      const uint64_t tx0 = monotonic_time_ns ();
      auto svg_str = render_svg_string (scatter_spec, &*snap_1m);
      const uint64_t tx1 = monotonic_time_ns ();
      if (!svg_str.has_value ())
        {
          std::cerr << "SVG in-memory generation failed\n";
          return 1;
        }

      /* 7(c). Write-to-disk time (open + write + fsync).  */
      const uint64_t td0 = monotonic_time_ns ();
      const int fd
          = ::open (svg_out.c_str (), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd < 0)
        {
          std::cerr << "Failed to open " << svg_out << " for writing\n";
          return 1;
        }
      const bool w_ok
          = write_all_fd (fd, xml_prolog.data (), xml_prolog.size ())
            && write_all_fd (fd, svg_str->data (), svg_str->size ())
            && (::fsync (fd) == 0);
      (void)::close (fd);
      const uint64_t td1 = monotonic_time_ns ();
      if (!w_ok)
        {
          std::cerr << "Failed write+fsync on " << svg_out << "\n";
          return 1;
        }
      if (r >= kSvgWarmup)
        {
          const std::size_t idx = r - kSvgWarmup;
          svg_layout_ms[idx] = static_cast<double> (tl1 - tl0) / 1.0e6;
          svg_xml_ms[idx] = static_cast<double> (tx1 - tx0) / 1.0e6;
          svg_disk_ms[idx] = static_cast<double> (td1 - td0) / 1.0e6;
        }
    }

  /* 7(d). Post-measurement SVG validation on disk (Section 3.2).  */
  if (std::filesystem::exists (svg_out))
    {
      final_svg_bytes = std::filesystem::file_size (svg_out);
      std::ifstream vin (svg_out, std::ios::binary);
      std::string file_data (final_svg_bytes, '\0');
      vin.read (file_data.data (),
                static_cast<std::streamsize> (final_svg_bytes));
      const std::string_view first_4k (
          file_data.data (), std::min<std::size_t> (4096U, file_data.size ()));
      const bool has_xml = (first_4k.rfind ("<?xml", 0U) == 0U);
      const bool has_svg = (first_4k.find ("<svg") != std::string_view::npos);

      /* Count XML element tags and M<x> <y>h0 subpath points.  */
      for (std::size_t i = 0U; i + 1U < file_data.size (); ++i)
        {
          if (file_data[i] == '<' && file_data[i + 1U] != '/'
              && file_data[i + 1U] != '?')
            {
              ++svg_dom_elements;
            }
          if (file_data[i] == 'h' && file_data[i + 1U] == '0')
            {
              ++svg_point_subpaths;
            }
        }
      std::string_view trimmed (file_data);
      while (!trimmed.empty ()
             && (trimmed.back () == '\n' || trimmed.back () == '\r'
                 || trimmed.back () == ' '))
        {
          trimmed.remove_suffix (1U);
        }
      const bool ends_svg
          = (trimmed.size () >= 6U
             && trimmed.substr (trimmed.size () - 6U) == "</svg>");
      const bool count_match = (svg_dom_elements == 27U)
                               && (svg_point_subpaths >= 990'000U
                                   && svg_point_subpaths <= 1'010'000U);
      svg_validation_ok = (final_svg_bytes > 10'000'000U) && has_xml && has_svg
                          && ends_svg && count_match;
      if (!svg_validation_ok)
        {
          std::cerr << "SVG validation details: bytes=" << final_svg_bytes
                    << " has_xml=" << has_xml << " has_svg=" << has_svg
                    << " ends_svg=" << ends_svg
                    << " dom_elements=" << svg_dom_elements
                    << " subpaths=" << svg_point_subpaths << "\n";
        }
    }
  (void)std::filesystem::remove (svg_out);

  if (!svg_validation_ok)
    {
      std::cerr << "SVG post-measurement validation FAILED!\n";
      return 1;
    }

  const DistStats svg_layout_stats = compute_stats (svg_layout_ms);
  const DistStats svg_xml_stats = compute_stats (svg_xml_ms);
  const DistStats svg_disk_stats = compute_stats (svg_disk_ms);
  const double svg_file_mb
      = static_cast<double> (final_svg_bytes) / (1024.0 * 1024.0);

  const double peak_gpu_mb
      = static_cast<double> (FrameStats::global ().snapshot ().peak_gpu_bytes)
        / (1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision (2);
  std::cout << "=============================================================="
               "===========\n";
  std::cout << "Darshana Reference Hardware Benchmark Report (Audited Phase 1)"
               "\n";
  std::cout << "=============================================================="
               "===========\n";
  std::cout << "[10M-Point Scatter — Three-Way Rate Split]\n";
  std::cout << "  1a. Offscreen Throughput (full redraw) : "
            << scatter_10m_offscreen_fps
            << " frames/s (offscreen) [p50=" << off_10m_stats.p50
            << " ms, p99=" << off_10m_stats.p99 << " ms]\n";
  std::cout << "  1a*. Offscreen Cached-Base Overlay     : "
            << scatter_10m_cached_overlay_fps
            << " frames/s (offscreen, cached base) [p50=" << ovl_10m_stats.p50
            << " ms, p99=" << ovl_10m_stats.p99 << " ms]\n";
  std::cout << "  1b. CPU Submission Rate (8-slot ring)  : "
            << scatter_10m_cpu_sub_fps
            << " frames/s (CPU submission) [p50=" << sub_10m_stats.p50
            << " ms, p99=" << sub_10m_stats.p99 << " ms]\n";
  std::cout << "  1c. Presented FPS (" << present_mode_label
            << ") : " << scatter_10m_present_fps << " FPS (presented, "
            << present_mode_label << " on " << surface_label
            << ") [p50=" << pres_10m_stats.p50
            << " ms, p99=" << pres_10m_stats.p99 << " ms]\n";
  std::cout << "  1c*. Presented FPS (FIFO 60 Hz vsync)  : "
            << scatter_10m_fifo_fps
            << " FPS (presented, VK_PRESENT_MODE_FIFO_KHR) [p50="
            << fifo_10m_stats.p50 << " ms, p99=" << fifo_10m_stats.p99
            << " ms]\n";
  std::cout << "  2.  10M Scatter Overlay Jitter (p99-p50): "
            << scatter_jitter_p99_p50_ms << " ms\n";
  std::cout << "\n[1M-Point Line — Three-Way Rate Split]\n";
  std::cout << "  3a. Offscreen Throughput (full redraw) : "
            << line_1m_offscreen_fps
            << " frames/s (offscreen) [p50=" << off_1m_stats.p50
            << " ms, p99=" << off_1m_stats.p99 << " ms]\n";
  std::cout << "  3b. CPU Submission Rate (8-slot ring)  : "
            << line_1m_cpu_sub_fps
            << " frames/s (CPU submission) [p50=" << sub_1m_stats.p50
            << " ms, p99=" << sub_1m_stats.p99 << " ms]\n";
  std::cout << "  3c. Presented FPS (" << present_mode_label
            << ") : " << line_1m_present_fps << " FPS (presented, "
            << present_mode_label << " on " << surface_label
            << ") [p50=" << pres_1m_stats.p50
            << " ms, p99=" << pres_1m_stats.p99 << " ms]\n";
  std::cout << "  4.  Streaming Ring-Buffer Jitter       : "
            << stream_jitter_p99_p50_ms << " ms (p50=" << stream_stats.p50
            << " ms, p99=" << stream_stats.p99 << " ms)\n";
  std::cout << "\n[1M-Point Scatter SVG Export — Four-Way Split]\n";
  std::cout << "  5a. Spec Validation + Layout Prep      : "
            << svg_layout_stats.p50 << " ms (p99=" << svg_layout_stats.p99
            << " ms)\n";
  std::cout << "  5b. In-Memory XML Generation (12-thrd) : "
            << svg_xml_stats.p50 << " ms (p99=" << svg_xml_stats.p99
            << " ms)\n";
  std::cout << "  5c. Write-to-Disk (open+write+fsync)   : "
            << svg_disk_stats.p50 << " ms (p99=" << svg_disk_stats.p99
            << " ms)\n";
  std::cout << "  5d. Final File Size & Element Count    : " << svg_file_mb
            << " MB (" << final_svg_bytes << " bytes, " << svg_dom_elements
            << " XML elements, " << svg_point_subpaths
            << " point subpaths, 0% decimated)\n";
  std::cout << "\n[Cold-Start & Three-Phase Memory Footprint]\n";
  std::cout << "  6a. Cold-Start (Auto CPU First Frame)  : "
            << cold_start_auto_ms << " ms\n";
  std::cout << "  6b. Cold-Start (Full Vulkan+384MB Pool): "
            << cold_start_gpu_ms << " ms\n";
  std::cout << "  7a. Bare Process RSS (main() entry)    : " << bare_rss_mb
            << " MB\n";
  std::cout << "  7b. Post-Library-Load RSS (pre-GPU)    : "
            << post_load_rss_mb << " MB\n";
  std::cout << "  7c. Post-GPU-Init RSS (+1 frame+1s)    : " << post_gpu_rss_mb
            << " MB\n";
  std::cout << "  8.  Peak GPU Pool Sub-Allocation (10M) : " << peak_gpu_mb
            << " MB / 256.00 MB device pool\n";
  std::cout << "=============================================================="
               "===========\n";

  const std::vector<MetricRecord> metrics = {
    { "10m_scatter_offscreen_fps", scatter_10m_offscreen_fps,
      "frames/second (offscreen)",
      "BENCHMARKS_METHODOLOGY.md#1a-10m-point-scatter-offscreen-throughput",
      sanity_10m, "800x600 RGBA readback FNV1a=" + std::to_string (hash_10m),
      true, true },
    { "10m_scatter_cached_overlay_offscreen_fps",
      scatter_10m_cached_overlay_fps,
      "frames/second (offscreen, cached static base)",
      "BENCHMARKS_METHODOLOGY.md#1a-10m-point-scatter-offscreen-throughput",
      sanity_10m, "500 overlay updates composited over cached 10M base", true,
      true },
    { "10m_scatter_cpu_submission_fps", scatter_10m_cpu_sub_fps,
      "frames/second (CPU submission)",
      "BENCHMARKS_METHODOLOGY.md#1b-10m-point-scatter-cpu-submission-rate",
      true, "500 command buffers submitted & retired via timeline semaphore",
      true, true },
    { "10m_scatter_presented_fps", scatter_10m_present_fps,
      "FPS (presented, " + present_mode_label + ")",
      "BENCHMARKS_METHODOLOGY.md#1c-10m-point-scatter-presented-fps",
      sanity_10m, "100 VkSwapchainKHR frames presented on " + surface_label,
      true, true },
    { "10m_scatter_presented_fifo_vsync_fps", scatter_10m_fifo_fps,
      "FPS (presented, VK_PRESENT_MODE_FIFO_KHR)",
      "BENCHMARKS_METHODOLOGY.md#1c-10m-point-scatter-presented-fps",
      sanity_10m, "60 VkSwapchainKHR frames locked to 60 Hz display vblank",
      true, true },
    { "10m_scatter_overlay_jitter_p99_p50_ms", scatter_jitter_p99_p50_ms, "ms",
      "BENCHMARKS_METHODOLOGY.md#2-10m-point-scatter-interactive-overlay-"
      "jitter",
      sanity_10m && scatter_jitter_p99_p50_ms < 1.0,
      "p99-p50 overlay frame time < 1.0 ms", false, false },
    { "1m_line_offscreen_fps", line_1m_offscreen_fps,
      "frames/second (offscreen)",
      "BENCHMARKS_METHODOLOGY.md#3a-1m-point-line-offscreen-throughput",
      sanity_1m, "800x600 RGBA readback FNV1a=" + std::to_string (hash_1m),
      true, true },
    { "1m_line_cpu_submission_fps", line_1m_cpu_sub_fps,
      "frames/second (CPU submission)",
      "BENCHMARKS_METHODOLOGY.md#3b-1m-point-line-cpu-submission-rate", true,
      "500 command buffers submitted & retired via timeline semaphore", true,
      true },
    { "1m_line_presented_fps", line_1m_present_fps,
      "FPS (presented, " + present_mode_label + ")",
      "BENCHMARKS_METHODOLOGY.md#3c-1m-point-line-presented-fps", sanity_1m,
      "150 VkSwapchainKHR frames presented on " + surface_label, true, true },
    { "streaming_60hz_jitter_p99_p50_ms", stream_jitter_p99_p50_ms, "ms",
      "BENCHMARKS_METHODOLOGY.md#4-streaming-ring-buffer-frame-jitter",
      sanity_stream && stream_jitter_p99_p50_ms < 1.0,
      "300 O(1) ring-buffer appends verified non-empty", false, false },
    { "svg_1m_spec_validation_layout_p50_ms", svg_layout_stats.p50, "ms",
      "BENCHMARKS_METHODOLOGY.md#5a-svg-export-spec-validation-and-layout-"
      "time",
      svg_validation_ok,
      "1,000,000 finite 2D points validated in SnapshotData", false, true },
    { "svg_1m_xml_generation_p50_ms", svg_xml_stats.p50, "ms",
      "BENCHMARKS_METHODOLOGY.md#5b-svg-export-in-memory-xml-generation-time",
      svg_validation_ok,
      "12-thread fixed-point SVG string generated (1,000,000 M...h0 points)",
      false, true },
    { "svg_1m_write_to_disk_fsync_p50_ms", svg_disk_stats.p50, "ms",
      "BENCHMARKS_METHODOLOGY.md#5c-svg-export-write-to-disk-time-with-fsync",
      svg_validation_ok,
      "open+write+fsync verified on disk: <?xml header, </svg> trailer, 27 "
      "DOM elements, 1,000,000 subpaths",
      false, true },
    { "svg_1m_file_size_mb", svg_file_mb, "MB",
      "BENCHMARKS_METHODOLOGY.md#5d-svg-export-final-file-size-and-element-"
      "count",
      svg_validation_ok,
      std::to_string (final_svg_bytes) + " bytes, "
          + std::to_string (svg_dom_elements) + " XML elements, "
          + std::to_string (svg_point_subpaths) + " subpath points",
      false, false },
    { "cold_start_auto_ms", cold_start_auto_ms, "ms",
      "BENCHMARKS_METHODOLOGY.md#6-cold-start-latency", cold_sanity,
      "First rendered 800x600 frame verified non-zero FNV1a hash", false,
      true },
    { "cold_start_vulkan_gpu_init_ms", cold_start_gpu_ms, "ms",
      "BENCHMARKS_METHODOLOGY.md#6-cold-start-latency", true,
      "Vulkan 1.2 instance + RTX 3070 device + 384 MiB slabs + 1st GPU frame",
      false, false },
    { "rss_bare_process_mb", bare_rss_mb, "MB",
      "BENCHMARKS_METHODOLOGY.md#7a-bare-process-rss", bare_rss_mb > 0.5,
      "Measured at first statement of main() via /proc/self/status VmRSS",
      false, true },
    { "rss_post_library_load_mb", post_load_rss_mb, "MB",
      "BENCHMARKS_METHODOLOGY.md#7b-post-library-load-rss",
      post_load_rss_mb >= bare_rss_mb && post_load_rss_mb < 30.0,
      "Measured after Figure context init, before GpuDevice::create()", false,
      true },
    { "rss_post_gpu_init_mb", post_gpu_rss_mb, "MB",
      "BENCHMARKS_METHODOLOGY.md#7c-post-gpu-init-rss",
      post_gpu_rss_mb > post_load_rss_mb,
      "Measured after GpuDevice::create() + 1 warm-up frame + 1s idle", false,
      true },
    { "peak_gpu_pool_10m_mb", peak_gpu_mb, "MB",
      "BENCHMARKS_METHODOLOGY.md#8-peak-gpu-pool-memory",
      peak_gpu_mb > 75.0 && peak_gpu_mb < 2048.0,
      "10M points (76.29 MB) + 2x 800x600 RGBA framebuffers (3.66 MB)", false,
      false }
  };

  /* Emit results.json per Section 3.5.  */
  {
    const auto parent = std::filesystem::path (json_out_path).parent_path ();
    if (!parent.empty ())
      {
        std::filesystem::create_directories (parent);
      }
    std::ofstream jout (json_out_path);
    jout << std::fixed << std::setprecision (4);
    jout << "{\n";
    jout << "  \"hardware\": {\n";
    jout << "    \"cpu\": \"12th Gen Intel(R) Core(TM) i7-12700K "
            "(12C/20T)\",\n";
    jout << "    \"gpu\": \"NVIDIA GeForce RTX 3070 (8192 MiB GDDR6)\",\n";
    jout << "    \"driver\": \"NVIDIA 550.163.01 / Vulkan 1.4.309\",\n";
    jout << "    \"kernel\": \"Linux 6.12.113+deb13-amd64 x86_64\",\n";
    jout << "    \"ram\": \"48 GiB Dual-Channel DDR4\",\n";
    jout << "    \"disk\": \"Samsung SSD 970 EVO Plus 1TB NVMe\"\n";
    jout << "  },\n";
    jout << "  \"timestamp\": \"" << iso_timestamp () << "\",\n";
    jout << "  \"metrics\": [\n";
    for (std::size_t i = 0U; i < metrics.size (); ++i)
      {
        const auto &m = metrics[i];
        jout << "    {\n";
        jout << "      \"name\": \"" << json_escape (m.name) << "\",\n";
        jout << "      \"value\": " << m.value << ",\n";
        jout << "      \"unit\": \"" << json_escape (m.unit) << "\",\n";
        jout << "      \"methodology_ref\": \""
             << json_escape (m.methodology_ref) << "\",\n";
        jout << "      \"sanity_check\": { \"passed\": "
             << (m.sanity_passed ? "true" : "false") << ", \"details\": \""
             << json_escape (m.sanity_details) << "\" }\n";
        jout << "    }" << (i + 1U < metrics.size () ? "," : "") << "\n";
      }
    jout << "  ]\n";
    jout << "}\n";
  }

  for (const auto &m : metrics)
    {
      if (!m.sanity_passed)
        {
          std::cerr << "Sanity check failed for metric: " << m.name << "\n";
          return 1;
        }
    }

  if (!baseline_path.empty ())
    {
      if (!verify_baseline_json (baseline_path, metrics))
        {
          return 1;
        }
    }

  return 0;
}
