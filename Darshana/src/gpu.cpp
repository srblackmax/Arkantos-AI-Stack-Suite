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

#include <darshana/gpu.hpp>
#include <darshana/observability.hpp>

#include "gpu_internal.hpp"
#include "gpu_shaders.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if DARSHANA_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace darshana::inline v1
{

namespace
{

constexpr std::size_t kPoolAlign = 256U;

/* Aligns a byte size upward to a power-of-two boundary.  */
constexpr auto
align_up (std::size_t value, std::size_t align) noexcept -> std::size_t
{
  return (value + align - 1U) & ~(align - 1U);
}

struct FreeBlock
{
  std::size_t offset = 0U;
  std::size_t size = 0U;
};

#if DARSHANA_HAS_VULKAN

/* Finds a Vulkan memory type index matching type_bits and properties.  */
auto
find_memory_type (VkPhysicalDevice phys, uint32_t type_bits,
                  VkMemoryPropertyFlags props) noexcept -> int32_t
{
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties (phys, &mem_props);
  for (uint32_t i = 0U; i < mem_props.memoryTypeCount; ++i)
    {
      if (((type_bits & (1U << i)) != 0U)
          && ((mem_props.memoryTypes[i].propertyFlags & props) == props))
        {
          return static_cast<int32_t> (i);
        }
    }
  return -1;
}

/* Push constant layout for clear/grid, scatter, and line compute passes.  */
struct BasePushConstants
{
  float scale_x = 1.0F;
  float offset_x = 0.0F;
  float scale_y = 1.0F;
  float offset_y = 0.0F;
  uint32_t width = 800U;
  uint32_t height = 600U;
  uint32_t aux0 = 0U;
  uint32_t aux1 = 0U;
};

/* Push constant layout for interactive overlay compositing compute pass.  */
struct OverlayPushConstants
{
  uint32_t width = 800U;
  uint32_t height = 600U;
  uint32_t cross_x = 0xFFFFFFFFU;
  uint32_t cross_y = 0xFFFFFFFFU;
  uint32_t brush_x0 = 0U;
  uint32_t brush_y0 = 0U;
  uint32_t brush_x1 = 0U;
  uint32_t brush_y1 = 0U;
};

/* Shared hardware Vulkan context holding pipelines and memory slabs.  */
struct VulkanHardwareEngine
{
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0U;
  std::string device_name;

  VkSemaphore timeline_sem = VK_NULL_HANDLE;
  std::atomic<uint64_t> timeline_value{ 0U };
  std::mutex queue_mutex;

  VkBuffer device_slab_buf = VK_NULL_HANDLE;
  VkDeviceMemory device_slab_mem = VK_NULL_HANDLE;
  std::size_t device_pool_capacity = 0U;

  VkBuffer staging_slab_buf = VK_NULL_HANDLE;
  VkDeviceMemory staging_slab_mem = VK_NULL_HANDLE;
  void *staging_mapped = nullptr;
  std::size_t staging_pool_capacity = 0U;
  std::atomic<std::size_t> staging_cursor{ 0U };

  VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
  VkShaderModule mod_clear = VK_NULL_HANDLE;
  VkShaderModule mod_scatter = VK_NULL_HANDLE;
  VkShaderModule mod_line = VK_NULL_HANDLE;
  VkShaderModule mod_overlay = VK_NULL_HANDLE;
  VkPipeline pipe_clear = VK_NULL_HANDLE;
  VkPipeline pipe_scatter = VK_NULL_HANDLE;
  VkPipeline pipe_line = VK_NULL_HANDLE;
  VkPipeline pipe_overlay = VK_NULL_HANDLE;

  struct WorkerSlot
  {
    std::mutex mutex;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
  };
  static constexpr std::size_t kNumWorkerSlots = 16U;
  WorkerSlot worker_slots[kNumWorkerSlots];
  std::atomic<uint32_t> next_slot{ 0U };

  std::mutex pool_mutex;
  std::vector<FreeBlock> free_blocks;

  VulkanHardwareEngine () noexcept = default;
  VulkanHardwareEngine (const VulkanHardwareEngine &) = delete;
  auto
  operator= (const VulkanHardwareEngine &) -> VulkanHardwareEngine & = delete;
  VulkanHardwareEngine (VulkanHardwareEngine &&) = delete;
  auto operator= (VulkanHardwareEngine &&) -> VulkanHardwareEngine & = delete;

  ~VulkanHardwareEngine () noexcept
  {
    if (device != VK_NULL_HANDLE)
      {
        (void)vkDeviceWaitIdle (device);
        for (auto &slot : worker_slots)
          {
            if (slot.desc_pool != VK_NULL_HANDLE)
              {
                vkDestroyDescriptorPool (device, slot.desc_pool, nullptr);
              }
            if (slot.cmd_pool != VK_NULL_HANDLE)
              {
                vkDestroyCommandPool (device, slot.cmd_pool, nullptr);
              }
          }
        if (pipe_clear != VK_NULL_HANDLE)
          {
            vkDestroyPipeline (device, pipe_clear, nullptr);
          }
        if (pipe_scatter != VK_NULL_HANDLE)
          {
            vkDestroyPipeline (device, pipe_scatter, nullptr);
          }
        if (pipe_line != VK_NULL_HANDLE)
          {
            vkDestroyPipeline (device, pipe_line, nullptr);
          }
        if (pipe_overlay != VK_NULL_HANDLE)
          {
            vkDestroyPipeline (device, pipe_overlay, nullptr);
          }
        if (mod_clear != VK_NULL_HANDLE)
          {
            vkDestroyShaderModule (device, mod_clear, nullptr);
          }
        if (mod_scatter != VK_NULL_HANDLE)
          {
            vkDestroyShaderModule (device, mod_scatter, nullptr);
          }
        if (mod_line != VK_NULL_HANDLE)
          {
            vkDestroyShaderModule (device, mod_line, nullptr);
          }
        if (mod_overlay != VK_NULL_HANDLE)
          {
            vkDestroyShaderModule (device, mod_overlay, nullptr);
          }
        if (pipe_layout != VK_NULL_HANDLE)
          {
            vkDestroyPipelineLayout (device, pipe_layout, nullptr);
          }
        if (desc_layout != VK_NULL_HANDLE)
          {
            vkDestroyDescriptorSetLayout (device, desc_layout, nullptr);
          }
        if (staging_mapped != nullptr && staging_slab_mem != VK_NULL_HANDLE)
          {
            vkUnmapMemory (device, staging_slab_mem);
          }
        if (staging_slab_buf != VK_NULL_HANDLE)
          {
            vkDestroyBuffer (device, staging_slab_buf, nullptr);
          }
        if (staging_slab_mem != VK_NULL_HANDLE)
          {
            vkFreeMemory (device, staging_slab_mem, nullptr);
          }
        if (device_slab_buf != VK_NULL_HANDLE)
          {
            vkDestroyBuffer (device, device_slab_buf, nullptr);
          }
        if (device_slab_mem != VK_NULL_HANDLE)
          {
            vkFreeMemory (device, device_slab_mem, nullptr);
          }
        if (timeline_sem != VK_NULL_HANDLE)
          {
            vkDestroySemaphore (device, timeline_sem, nullptr);
          }
        vkDestroyDevice (device, nullptr);
      }
#if !defined(__SANITIZE_THREAD__)
    if (instance != VK_NULL_HANDLE)
      {
        vkDestroyInstance (instance, nullptr);
      }
#endif
  }
};

/* Creates a single compute shader module and pipeline.  */
auto
create_compute_pipeline (VkDevice dev, VkPipelineLayout layout,
                         const uint32_t *spv_code, std::size_t spv_bytes,
                         VkShaderModule &out_mod,
                         VkPipeline &out_pipe) noexcept -> Result<void>
{
  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spv_bytes;
  smci.pCode = spv_code;
  DARSHANA_VK_CHECK (vkCreateShaderModule (dev, &smci, nullptr, &out_mod),
                     GpuErrorCode::PipelineCreationFailed,
                     "Failed to create Vulkan SPIR-V shader module");

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = out_mod;
  cpci.stage.pName = "main";
  cpci.layout = layout;
  DARSHANA_VK_CHECK (vkCreateComputePipelines (dev, VK_NULL_HANDLE, 1U, &cpci,
                                               nullptr, &out_pipe),
                     GpuErrorCode::PipelineCreationFailed,
                     "Failed to create Vulkan compute pipeline");
  return {};
}

/* Initializes the VulkanHardwareEngine singleton or custom instance.  */
auto
init_hardware_engine (const GpuDeviceConfig &config) noexcept
    -> Result<std::shared_ptr<VulkanHardwareEngine>>
{
#if defined(__SANITIZE_THREAD__)
  const char *tsan_icd = "/usr/share/vulkan/icd.d/intel_icd.json";
  if (std::filesystem::exists (tsan_icd))
    {
      (void)::setenv ("VK_DRIVER_FILES", tsan_icd, 1);
    }
#else
  if (std::getenv ("VK_DRIVER_FILES") == nullptr
      && std::getenv ("VK_ICD_FILENAMES") == nullptr)
    {
      const char *nv_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json";
      if (std::filesystem::exists (nv_icd))
        {
          (void)::setenv ("VK_DRIVER_FILES", nv_icd, 0);
        }
    }
#endif

  auto eng = std::make_shared<VulkanHardwareEngine> ();

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "Darshana";
  app.applicationVersion = VK_MAKE_VERSION (1, 0, 0);
  app.pEngineName = "Darshana";
  app.engineVersion = VK_MAKE_VERSION (1, 0, 0);
  app.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  DARSHANA_VK_CHECK (vkCreateInstance (&ici, nullptr, &eng->instance),
                     GpuErrorCode::InitializationFailed,
                     "Failed to create Vulkan 1.2 instance");

  uint32_t dev_count = 0U;
  DARSHANA_VK_CHECK (
      vkEnumeratePhysicalDevices (eng->instance, &dev_count, nullptr),
      GpuErrorCode::InitializationFailed,
      "Failed to enumerate Vulkan physical devices");
  if (dev_count == 0U)
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::InitializationFailed, -3,
          "No Vulkan-capable GPU physical devices found on system"));
    }

  std::vector<VkPhysicalDevice> phys_list (dev_count);
  DARSHANA_VK_CHECK (vkEnumeratePhysicalDevices (eng->instance, &dev_count,
                                                 phys_list.data ()),
                     GpuErrorCode::InitializationFailed,
                     "Failed to read Vulkan physical device handles");

  eng->phys_device = phys_list[0];
  for (const VkPhysicalDevice pd : phys_list)
    {
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties (pd, &props);
      if (config.prefer_discrete_gpu
          && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
          eng->phys_device = pd;
          break;
        }
    }

  VkPhysicalDeviceProperties chosen_props{};
  vkGetPhysicalDeviceProperties (eng->phys_device, &chosen_props);
  eng->device_name = chosen_props.deviceName;

  uint32_t qf_count = 0U;
  vkGetPhysicalDeviceQueueFamilyProperties (eng->phys_device, &qf_count,
                                            nullptr);
  std::vector<VkQueueFamilyProperties> qf_props (qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties (eng->phys_device, &qf_count,
                                            qf_props.data ());
  for (uint32_t i = 0U; i < qf_count; ++i)
    {
      if ((qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U)
        {
          eng->queue_family = i;
          break;
        }
    }

  const float prio = 1.0F;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = eng->queue_family;
  qci.queueCount = 1U;
  qci.pQueuePriorities = &prio;

  VkPhysicalDeviceVulkan12Features feat12{};
  feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  feat12.timelineSemaphore = VK_TRUE;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pNext = &feat12;
  dci.queueCreateInfoCount = 1U;
  dci.pQueueCreateInfos = &qci;

  DARSHANA_VK_CHECK (
      vkCreateDevice (eng->phys_device, &dci, nullptr, &eng->device),
      GpuErrorCode::InitializationFailed,
      "Failed to create Vulkan 1.2 logical device");
  vkGetDeviceQueue (eng->device, eng->queue_family, 0U, &eng->queue);

  VkSemaphoreTypeCreateInfo stci{};
  stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  stci.initialValue = 0U;

  VkSemaphoreCreateInfo sci{};
  sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  sci.pNext = &stci;
  DARSHANA_VK_CHECK (
      vkCreateSemaphore (eng->device, &sci, nullptr, &eng->timeline_sem),
      GpuErrorCode::InitializationFailed,
      "Failed to create Vulkan timeline semaphore");

  eng->device_pool_capacity = align_up (config.device_pool_bytes, kPoolAlign);
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = eng->device_pool_capacity;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  DARSHANA_VK_CHECK (
      vkCreateBuffer (eng->device, &bci, nullptr, &eng->device_slab_buf),
      GpuErrorCode::OutOfMemory,
      "Failed to create pre-allocated device-local storage buffer");

  VkMemoryRequirements dev_reqs{};
  vkGetBufferMemoryRequirements (eng->device, eng->device_slab_buf, &dev_reqs);
  int32_t dev_mem_idx
      = find_memory_type (eng->phys_device, dev_reqs.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (dev_mem_idx < 0)
    {
      dev_mem_idx
          = find_memory_type (eng->phys_device, dev_reqs.memoryTypeBits, 0U);
    }
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = dev_reqs.size;
  mai.memoryTypeIndex = static_cast<uint32_t> (dev_mem_idx);
  DARSHANA_VK_CHECK (
      vkAllocateMemory (eng->device, &mai, nullptr, &eng->device_slab_mem),
      GpuErrorCode::OutOfMemory,
      "Failed to allocate pre-allocated device-local memory pool");
  DARSHANA_VK_CHECK (vkBindBufferMemory (eng->device, eng->device_slab_buf,
                                         eng->device_slab_mem, 0U),
                     GpuErrorCode::OutOfMemory,
                     "Failed to bind device-local memory pool");

  eng->free_blocks.push_back (FreeBlock{ 0U, eng->device_pool_capacity });

  eng->staging_pool_capacity
      = align_up (config.staging_pool_bytes, kPoolAlign);
  VkBufferCreateInfo sbci{};
  sbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  sbci.size = eng->staging_pool_capacity;
  sbci.usage
      = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  sbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  DARSHANA_VK_CHECK (
      vkCreateBuffer (eng->device, &sbci, nullptr, &eng->staging_slab_buf),
      GpuErrorCode::OutOfMemory, "Failed to create host staging buffer");

  VkMemoryRequirements stg_reqs{};
  vkGetBufferMemoryRequirements (eng->device, eng->staging_slab_buf,
                                 &stg_reqs);
  const int32_t stg_mem_idx
      = find_memory_type (eng->phys_device, stg_reqs.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkMemoryAllocateInfo smai{};
  smai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  smai.allocationSize = stg_reqs.size;
  smai.memoryTypeIndex = static_cast<uint32_t> (stg_mem_idx);
  DARSHANA_VK_CHECK (
      vkAllocateMemory (eng->device, &smai, nullptr, &eng->staging_slab_mem),
      GpuErrorCode::OutOfMemory, "Failed to allocate host staging memory");
  DARSHANA_VK_CHECK (vkBindBufferMemory (eng->device, eng->staging_slab_buf,
                                         eng->staging_slab_mem, 0U),
                     GpuErrorCode::OutOfMemory,
                     "Failed to bind host staging memory");
  DARSHANA_VK_CHECK (vkMapMemory (eng->device, eng->staging_slab_mem, 0U,
                                  eng->staging_pool_capacity, 0U,
                                  &eng->staging_mapped),
                     GpuErrorCode::OutOfMemory,
                     "Failed to map persistent host staging buffer");

  VkDescriptorSetLayoutBinding bindings[3]{};
  for (uint32_t i = 0U; i < 3U; ++i)
    {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1U;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 3U;
  dslci.pBindings = bindings;
  DARSHANA_VK_CHECK (vkCreateDescriptorSetLayout (eng->device, &dslci, nullptr,
                                                  &eng->desc_layout),
                     GpuErrorCode::PipelineCreationFailed,
                     "Failed to create Vulkan descriptor set layout");

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0U;
  pcr.size = sizeof (BasePushConstants);

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1U;
  plci.pSetLayouts = &eng->desc_layout;
  plci.pushConstantRangeCount = 1U;
  plci.pPushConstantRanges = &pcr;
  DARSHANA_VK_CHECK (
      vkCreatePipelineLayout (eng->device, &plci, nullptr, &eng->pipe_layout),
      GpuErrorCode::PipelineCreationFailed,
      "Failed to create Vulkan pipeline layout");

  if (auto r = create_compute_pipeline (
          eng->device, eng->pipe_layout, detail::kClearGridSpv,
          sizeof (detail::kClearGridSpv), eng->mod_clear, eng->pipe_clear);
      !r.has_value ())
    {
      return std::unexpected (r.error ());
    }
  if (auto r = create_compute_pipeline (eng->device, eng->pipe_layout,
                                        detail::kScatterSplatSpv,
                                        sizeof (detail::kScatterSplatSpv),
                                        eng->mod_scatter, eng->pipe_scatter);
      !r.has_value ())
    {
      return std::unexpected (r.error ());
    }
  if (auto r = create_compute_pipeline (
          eng->device, eng->pipe_layout, detail::kLineSdfSpv,
          sizeof (detail::kLineSdfSpv), eng->mod_line, eng->pipe_line);
      !r.has_value ())
    {
      return std::unexpected (r.error ());
    }
  if (auto r = create_compute_pipeline (eng->device, eng->pipe_layout,
                                        detail::kOverlayCompSpv,
                                        sizeof (detail::kOverlayCompSpv),
                                        eng->mod_overlay, eng->pipe_overlay);
      !r.has_value ())
    {
      return std::unexpected (r.error ());
    }

  for (auto &slot : eng->worker_slots)
    {
      VkCommandPoolCreateInfo cpci{};
      cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      cpci.queueFamilyIndex = eng->queue_family;
      DARSHANA_VK_CHECK (
          vkCreateCommandPool (eng->device, &cpci, nullptr, &slot.cmd_pool),
          GpuErrorCode::InitializationFailed,
          "Failed to create worker command pool");

      VkCommandBufferAllocateInfo cbai{};
      cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      cbai.commandPool = slot.cmd_pool;
      cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      cbai.commandBufferCount = 1U;
      DARSHANA_VK_CHECK (
          vkAllocateCommandBuffers (eng->device, &cbai, &slot.cmd_buf),
          GpuErrorCode::InitializationFailed,
          "Failed to allocate worker command buffer");

      VkDescriptorPoolSize pool_size{};
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = 3U;
      VkDescriptorPoolCreateInfo dpci{};
      dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      dpci.maxSets = 1U;
      dpci.poolSizeCount = 1U;
      dpci.pPoolSizes = &pool_size;
      DARSHANA_VK_CHECK (vkCreateDescriptorPool (eng->device, &dpci, nullptr,
                                                 &slot.desc_pool),
                         GpuErrorCode::InitializationFailed,
                         "Failed to create worker descriptor pool");

      VkDescriptorSetAllocateInfo dsai{};
      dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      dsai.descriptorPool = slot.desc_pool;
      dsai.descriptorSetCount = 1U;
      dsai.pSetLayouts = &eng->desc_layout;
      DARSHANA_VK_CHECK (
          vkAllocateDescriptorSets (eng->device, &dsai, &slot.desc_set),
          GpuErrorCode::InitializationFailed,
          "Failed to allocate worker descriptor set");
    }

  return eng;
}

/* Returns or initializes the default cached hardware engine.  */
auto
get_default_engine (const GpuDeviceConfig &config) noexcept
    -> Result<std::shared_ptr<VulkanHardwareEngine>>
{
  static std::mutex s_mutex;
  static std::shared_ptr<VulkanHardwareEngine> s_engine;
  std::lock_guard<std::mutex> lock (s_mutex);
  if (!s_engine || s_engine->device_pool_capacity < config.device_pool_bytes)
    {
      s_engine.reset ();
      auto res = init_hardware_engine (config);
      if (!res.has_value ())
        {
          return std::unexpected (res.error ());
        }
      s_engine = std::move (*res);
    }
  return s_engine;
}

#endif /* DARSHANA_HAS_VULKAN */

} /* namespace */

struct GpuDevice::Impl
{
#if DARSHANA_HAS_VULKAN
  std::shared_ptr<VulkanHardwareEngine> engine;
#endif
  std::string name = "CPU-Fallback";
  std::size_t capacity = 0U;
  std::atomic<std::size_t> allocated{ 0U };
  std::atomic<std::size_t> peak_allocated{ 0U };
  std::atomic<bool> device_lost{ false };
};

struct GpuBuffer::Impl
{
  std::shared_ptr<GpuDevice::Impl> dev_impl;
  std::size_t offset = 0U;
  std::size_t capacity = 0U;

  Impl () noexcept = default;
  Impl (const Impl &) = delete;
  auto operator= (const Impl &) -> Impl & = delete;
  Impl (Impl &&) = delete;
  auto operator= (Impl &&) -> Impl & = delete;

  ~Impl () noexcept
  {
    if (dev_impl && capacity > 0U)
      {
#if DARSHANA_HAS_VULKAN
        if (dev_impl->engine)
          {
            auto &eng = *dev_impl->engine;
            std::lock_guard<std::mutex> lock (eng.pool_mutex);
            eng.free_blocks.push_back (FreeBlock{ offset, capacity });
            std::ranges::sort (eng.free_blocks,
                               [] (const FreeBlock &a, const FreeBlock &b) {
                                 return a.offset < b.offset;
                               });
            std::vector<FreeBlock> merged;
            merged.reserve (eng.free_blocks.size ());
            for (const auto &blk : eng.free_blocks)
              {
                if (!merged.empty ()
                    && merged.back ().offset + merged.back ().size
                           == blk.offset)
                  {
                    merged.back ().size += blk.size;
                  }
                else
                  {
                    merged.push_back (blk);
                  }
              }
            eng.free_blocks = std::move (merged);
          }
#endif
        dev_impl->allocated.fetch_sub (capacity, std::memory_order_relaxed);
      }
  }
};

/* Default-constructs an empty GpuDevice handle.  */
GpuDevice::GpuDevice () noexcept = default;

/* Constructs a GpuDevice from an internal implementation handle.  */
GpuDevice::GpuDevice (std::shared_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

GpuDevice::GpuDevice (GpuDevice &&other) noexcept = default;

auto
GpuDevice::operator= (GpuDevice &&other) noexcept -> GpuDevice & = default;

GpuDevice::~GpuDevice () noexcept = default;

/* Initializes a Vulkan 1.2+ device and pre-allocated memory pools.  */
auto
GpuDevice::create (const GpuDeviceConfig &config) noexcept -> Result<GpuDevice>
{
  if (config.force_device_failure)
    {
      Logger::log (LogLevel::Warn, "gpu",
                   "Vulkan device creation forced to fail by configuration");
      return std::unexpected (Error::gpu (
          GpuErrorCode::InitializationFailed, -3,
          "Vulkan device creation failed due to forced failure flag"));
    }
  if (config.device_pool_bytes == 0U || config.staging_pool_bytes == 0U)
    {
      return std::unexpected (
          Error::gpu (GpuErrorCode::InitializationFailed, -1,
                      "GpuDevice pool capacities must be greater than zero"));
    }
#if DARSHANA_HAS_VULKAN
  auto eng_res = get_default_engine (config);
  if (!eng_res.has_value ())
    {
      Logger::log (LogLevel::Warn, "gpu", eng_res.error ().message ());
      return std::unexpected (eng_res.error ());
    }
  auto impl = std::make_shared<Impl> ();
  impl->engine = std::move (*eng_res);
  impl->name = impl->engine->device_name;
  impl->capacity = config.device_pool_bytes;
  return GpuDevice (std::move (impl));
#else
  Logger::log (LogLevel::Warn, "gpu",
               "Darshana was built without Vulkan support; GPU unavailable");
  return std::unexpected (Error::gpu (
      GpuErrorCode::InitializationFailed, -3,
      "Vulkan backend is disabled in release-headless build configuration"));
#endif
}

/* Checks whether this GpuDevice owns an active Vulkan context.  */
auto
GpuDevice::is_valid () const noexcept -> bool
{
  return impl_ != nullptr
         && !impl_->device_lost.load (std::memory_order_acquire);
}

/* Retrieves the physical GPU device name.  */
auto
GpuDevice::device_name () const noexcept -> std::string_view
{
  return impl_ ? std::string_view (impl_->name) : std::string_view ("Invalid");
}

/* Queries bytes currently sub-allocated from the GPU pool.  */
auto
GpuDevice::allocated_bytes () const noexcept -> std::size_t
{
  return impl_ ? impl_->allocated.load (std::memory_order_relaxed) : 0U;
}

/* Queries peak bytes sub-allocated from the GPU pool.  */
auto
GpuDevice::peak_allocated_bytes () const noexcept -> std::size_t
{
  return impl_ ? impl_->peak_allocated.load (std::memory_order_relaxed) : 0U;
}

/* Queries total pre-allocated device-local pool capacity.  */
auto
GpuDevice::pool_capacity_bytes () const noexcept -> std::size_t
{
  return impl_ ? impl_->capacity : 0U;
}

/* Waits for all in-flight GPU command buffers to complete.  */
auto
GpuDevice::wait_idle () const noexcept -> Result<void>
{
  if (!is_valid ())
    {
      return std::unexpected (
          Error::gpu (GpuErrorCode::DeviceLost, -4,
                      "Cannot wait_idle on an invalid or lost GpuDevice"));
    }
#if DARSHANA_HAS_VULKAN
  auto &eng = *impl_->engine;
  std::lock_guard<std::mutex> lock (eng.queue_mutex);
  DARSHANA_VK_CHECK (vkQueueWaitIdle (eng.queue), GpuErrorCode::DeviceLost,
                     "vkQueueWaitIdle failed on Vulkan queue");
#endif
  return {};
}

/* Injects a simulated VK_ERROR_DEVICE_LOST fault for testing.  */
void
GpuDevice::simulate_device_loss () noexcept
{
  if (impl_)
    {
      impl_->device_lost.store (true, std::memory_order_release);
    }
}

/* Accesses the opaque internal implementation pointer.  */
auto
GpuDevice::native_handle () const noexcept -> void *
{
  return impl_.get ();
}

/* Default-constructs an empty GpuBuffer handle.  */
GpuBuffer::GpuBuffer () noexcept = default;

/* Constructs a GpuBuffer from an internal implementation pointer.  */
GpuBuffer::GpuBuffer (std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

GpuBuffer::GpuBuffer (GpuBuffer &&other) noexcept = default;

auto
GpuBuffer::operator= (GpuBuffer &&other) noexcept -> GpuBuffer & = default;

GpuBuffer::~GpuBuffer () noexcept = default;

/* Sub-allocates a GpuBuffer from the GpuDevice's pre-allocated pool.  */
auto
GpuBuffer::allocate (GpuDevice &device,
                     std::size_t capacity_bytes) noexcept -> Result<GpuBuffer>
{
  if (!device.is_valid ())
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::DeviceLost, -4,
          "Cannot sub-allocate GpuBuffer from an invalid or lost GpuDevice"));
    }
  if (capacity_bytes == 0U)
    {
      return std::unexpected (
          Error::gpu (GpuErrorCode::OutOfMemory, -2,
                      "GpuBuffer::allocate requested zero bytes"));
    }
  auto *dev_impl = static_cast<GpuDevice::Impl *> (device.native_handle ());
  const std::size_t aligned_req = align_up (capacity_bytes, kPoolAlign);
  const std::size_t cur_alloc
      = dev_impl->allocated.load (std::memory_order_relaxed);
  if (cur_alloc + aligned_req > dev_impl->capacity)
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::OutOfMemory, -2,
          "GpuBuffer sub-allocation exceeds GpuDevice pool capacity"));
    }

  std::size_t chosen_offset = 0U;
#if DARSHANA_HAS_VULKAN
  auto &eng = *dev_impl->engine;
  {
    std::lock_guard<std::mutex> lock (eng.pool_mutex);
    bool found = false;
    for (std::size_t i = 0U; i < eng.free_blocks.size (); ++i)
      {
        if (eng.free_blocks[i].size >= aligned_req)
          {
            chosen_offset = eng.free_blocks[i].offset;
            eng.free_blocks[i].offset += aligned_req;
            eng.free_blocks[i].size -= aligned_req;
            if (eng.free_blocks[i].size == 0U)
              {
                eng.free_blocks.erase (eng.free_blocks.begin ()
                                       + static_cast<std::ptrdiff_t> (i));
              }
            found = true;
            break;
          }
      }
    if (!found)
      {
        return std::unexpected (Error::gpu (
            GpuErrorCode::OutOfMemory, -2,
            "Vulkan pre-allocated device memory pool is exhausted"));
      }
  }
#endif

  auto b_impl = std::make_unique<Impl> ();
  b_impl->dev_impl
      = std::shared_ptr<GpuDevice::Impl> (dev_impl, [] (GpuDevice::Impl *) {});
  b_impl->offset = chosen_offset;
  b_impl->capacity = aligned_req;

  const std::size_t new_alloc
      = dev_impl->allocated.fetch_add (aligned_req, std::memory_order_relaxed)
        + aligned_req;
  std::size_t prev_peak
      = dev_impl->peak_allocated.load (std::memory_order_relaxed);
  while (prev_peak < new_alloc
         && !dev_impl->peak_allocated.compare_exchange_weak (
             prev_peak, new_alloc, std::memory_order_relaxed))
    {
    }
  FrameStats::global ().record_gpu_memory (new_alloc,
                                           dev_impl->peak_allocated.load ());
  return GpuBuffer (std::move (b_impl));
}

/* Asynchronously uploads 2D points from host to device via staging buffer.  */
auto
GpuBuffer::upload_points_async (std::span<const DataPoint2D> points,
                                std::size_t offset_points) noexcept
    -> Result<uint64_t>
{
  if (!impl_ || !impl_->dev_impl
      || impl_->dev_impl->device_lost.load (std::memory_order_acquire))
    {
      return std::unexpected (
          Error::gpu (GpuErrorCode::DeviceLost, -4,
                      "Cannot upload points to an invalid or lost GpuBuffer"));
    }
  if (points.empty ())
    {
      return 0U;
    }
  const std::size_t byte_offset = offset_points * sizeof (DataPoint2D);
  const std::size_t byte_size = points.size () * sizeof (DataPoint2D);
  if (byte_offset + byte_size > impl_->capacity)
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::OutOfMemory, -2,
          "Point upload exceeds pre-allocated GpuBuffer capacity"));
    }

#if DARSHANA_HAS_VULKAN
  auto &eng = *impl_->dev_impl->engine;
  const std::size_t aligned_bytes = align_up (byte_size, kPoolAlign);
  if (aligned_bytes > eng.staging_pool_capacity)
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::OutOfMemory, -2,
          "Upload size exceeds pre-allocated staging pool capacity"));
    }

  std::size_t stg_off
      = eng.staging_cursor.fetch_add (aligned_bytes, std::memory_order_relaxed)
        % eng.staging_pool_capacity;
  if (stg_off + byte_size > eng.staging_pool_capacity)
    {
      stg_off = 0U;
    }

  std::memcpy (static_cast<std::byte *> (eng.staging_mapped) + stg_off,
               points.data (), byte_size);

  const uint32_t slot_idx
      = eng.next_slot.fetch_add (1U, std::memory_order_relaxed)
        % VulkanHardwareEngine::kNumWorkerSlots;
  auto &slot = eng.worker_slots[slot_idx];
  std::lock_guard<std::mutex> slot_lock (slot.mutex);

  DARSHANA_VK_CHECK (vkResetCommandBuffer (slot.cmd_buf, 0U),
                     GpuErrorCode::TransferFailed,
                     "Failed to reset transfer command buffer");
  VkCommandBufferBeginInfo cbbi{};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  DARSHANA_VK_CHECK (vkBeginCommandBuffer (slot.cmd_buf, &cbbi),
                     GpuErrorCode::TransferFailed,
                     "Failed to begin transfer command buffer");

  VkBufferCopy copy_region{};
  copy_region.srcOffset = stg_off;
  copy_region.dstOffset = impl_->offset + byte_offset;
  copy_region.size = byte_size;
  vkCmdCopyBuffer (slot.cmd_buf, eng.staging_slab_buf, eng.device_slab_buf, 1U,
                   &copy_region);

  VkMemoryBarrier bar{};
  bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier (slot.cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &bar, 0U,
                        nullptr, 0U, nullptr);

  DARSHANA_VK_CHECK (vkEndCommandBuffer (slot.cmd_buf),
                     GpuErrorCode::TransferFailed,
                     "Failed to end transfer command buffer");

  uint64_t signal_val = 0U;
  {
    std::lock_guard<std::mutex> q_lock (eng.queue_mutex);
    signal_val
        = eng.timeline_value.fetch_add (1U, std::memory_order_relaxed) + 1U;
    VkTimelineSemaphoreSubmitInfo tssi{};
    tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tssi.signalSemaphoreValueCount = 1U;
    tssi.pSignalSemaphoreValues = &signal_val;

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tssi;
    si.commandBufferCount = 1U;
    si.pCommandBuffers = &slot.cmd_buf;
    si.signalSemaphoreCount = 1U;
    si.pSignalSemaphores = &eng.timeline_sem;
    DARSHANA_VK_CHECK (vkQueueSubmit (eng.queue, 1U, &si, VK_NULL_HANDLE),
                       GpuErrorCode::TransferFailed,
                       "Failed to submit async DMA transfer to Vulkan queue");
  }

  VkSemaphoreWaitInfo swi{};
  swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  swi.semaphoreCount = 1U;
  swi.pSemaphores = &eng.timeline_sem;
  swi.pValues = &signal_val;
  DARSHANA_VK_CHECK (vkWaitSemaphores (eng.device, &swi, UINT64_MAX),
                     GpuErrorCode::TransferFailed,
                     "Failed waiting on Vulkan timeline semaphore for upload");
  return signal_val;
#else
  return 1U;
#endif
}

/* Waits on the Vulkan timeline semaphore for an async upload to finish.  */
auto
GpuBuffer::wait_timeline (uint64_t timeline_value) const noexcept
    -> Result<void>
{
  if (!impl_ || !impl_->dev_impl
      || impl_->dev_impl->device_lost.load (std::memory_order_acquire))
    {
      return std::unexpected (
          Error::gpu (GpuErrorCode::DeviceLost, -4,
                      "Cannot wait_timeline on an invalid or lost GpuBuffer"));
    }
  if (timeline_value == 0U)
    {
      return {};
    }
#if DARSHANA_HAS_VULKAN
  auto &eng = *impl_->dev_impl->engine;
  VkSemaphoreWaitInfo swi{};
  swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  swi.semaphoreCount = 1U;
  swi.pSemaphores = &eng.timeline_sem;
  swi.pValues = &timeline_value;
  DARSHANA_VK_CHECK (vkWaitSemaphores (eng.device, &swi, UINT64_MAX),
                     GpuErrorCode::DeviceLost,
                     "vkWaitSemaphores failed on timeline semaphore");
#endif
  return {};
}

/* Queries the pre-allocated capacity of this buffer in bytes.  */
auto
GpuBuffer::capacity_bytes () const noexcept -> std::size_t
{
  return impl_ ? impl_->capacity : 0U;
}

/* Queries the byte offset of this buffer within the device pool.  */
auto
GpuBuffer::pool_offset () const noexcept -> std::size_t
{
  return impl_ ? impl_->offset : 0U;
}

/* Accesses the opaque internal buffer handle.  */
auto
GpuBuffer::native_handle () const noexcept -> void *
{
  return impl_.get ();
}

/* Checks whether a Vulkan 1.2+ capable GPU is available at runtime.  */
auto
is_vulkan_available () noexcept -> bool
{
#if DARSHANA_HAS_VULKAN
  GpuDeviceConfig cfg;
  auto res = get_default_engine (cfg);
  return res.has_value ();
#else
  return false;
#endif
}

namespace detail
{

/* Acquires or returns the process-wide shared default GpuDevice.  */
auto
acquire_shared_gpu_device () noexcept -> Result<GpuDevice>
{
  return GpuDevice::create ();
}

/* Executes the static base-layer compute pass (clear/grid + scatter/line).  */
auto
gpu_execute_base_pass (
    GpuDevice &device, const GpuBuffer &points_buf, GpuBuffer &base_pixels_buf,
    const GpuBasePassParams &params) noexcept -> Result<void>
{
  if (!device.is_valid ())
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::DeviceLost, -4,
          "Cannot execute base GPU pass on a lost or invalid GpuDevice"));
    }
#if DARSHANA_HAS_VULKAN
  auto *dev_impl = static_cast<GpuDevice::Impl *> (device.native_handle ());
  auto &eng = *dev_impl->engine;

  const uint32_t slot_idx
      = eng.next_slot.fetch_add (1U, std::memory_order_relaxed)
        % VulkanHardwareEngine::kNumWorkerSlots;
  auto &slot = eng.worker_slots[slot_idx];
  std::lock_guard<std::mutex> slot_lock (slot.mutex);

  VkDescriptorBufferInfo dbis[3]{};
  dbis[0].buffer = eng.device_slab_buf;
  dbis[0].offset = points_buf.pool_offset ();
  dbis[0].range = std::max<std::size_t> (points_buf.capacity_bytes (), 256U);

  dbis[1].buffer = eng.device_slab_buf;
  dbis[1].offset = base_pixels_buf.pool_offset ();
  dbis[1].range = base_pixels_buf.capacity_bytes ();

  dbis[2].buffer = eng.device_slab_buf;
  dbis[2].offset = base_pixels_buf.pool_offset ();
  dbis[2].range = base_pixels_buf.capacity_bytes ();

  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0U; i < 3U; ++i)
    {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = slot.desc_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1U;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &dbis[i];
    }
  vkUpdateDescriptorSets (eng.device, 3U, writes, 0U, nullptr);

  DARSHANA_VK_CHECK (vkResetCommandBuffer (slot.cmd_buf, 0U),
                     GpuErrorCode::DeviceLost,
                     "Failed to reset base pass command buffer");
  VkCommandBufferBeginInfo cbbi{};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  DARSHANA_VK_CHECK (vkBeginCommandBuffer (slot.cmd_buf, &cbbi),
                     GpuErrorCode::DeviceLost,
                     "Failed to begin base pass command buffer");

  vkCmdBindDescriptorSets (slot.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                           eng.pipe_layout, 0U, 1U, &slot.desc_set, 0U,
                           nullptr);

  BasePushConstants clear_pc{};
  clear_pc.scale_x = params.scale_x;
  clear_pc.offset_x = params.offset_x;
  clear_pc.scale_y = params.scale_y;
  clear_pc.offset_y = params.offset_y;
  clear_pc.width = params.width;
  clear_pc.height = params.height;
  clear_pc.aux0 = params.bg_color;
  clear_pc.aux1 = params.grid_color;

  vkCmdBindPipeline (slot.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                     eng.pipe_clear);
  vkCmdPushConstants (slot.cmd_buf, eng.pipe_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof (clear_pc),
                      &clear_pc);
  vkCmdDispatch (slot.cmd_buf, (params.width + 15U) / 16U,
                 (params.height + 15U) / 16U, 1U);

  if (params.point_count > 0U)
    {
      VkMemoryBarrier bar{};
      bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      bar.dstAccessMask
          = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier (slot.cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &bar,
                            0U, nullptr, 0U, nullptr);

      BasePushConstants draw_pc = clear_pc;
      draw_pc.aux0 = params.point_count;
      draw_pc.aux1 = params.mark_color;

      vkCmdBindPipeline (slot.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                         params.is_line_plot ? eng.pipe_line
                                             : eng.pipe_scatter);
      vkCmdPushConstants (slot.cmd_buf, eng.pipe_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof (draw_pc),
                          &draw_pc);
      const uint32_t groups = (params.point_count + 255U) / 256U;
      vkCmdDispatch (slot.cmd_buf, groups, 1U, 1U);
    }

  VkMemoryBarrier end_bar{};
  end_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  end_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  end_bar.dstAccessMask
      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier (slot.cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                            | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0U, 1U, &end_bar, 0U, nullptr, 0U, nullptr);

  DARSHANA_VK_CHECK (vkEndCommandBuffer (slot.cmd_buf),
                     GpuErrorCode::DeviceLost,
                     "Failed to end base pass command buffer");

  uint64_t signal_val = 0U;
  {
    std::lock_guard<std::mutex> q_lock (eng.queue_mutex);
    signal_val
        = eng.timeline_value.fetch_add (1U, std::memory_order_relaxed) + 1U;
    VkTimelineSemaphoreSubmitInfo tssi{};
    tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tssi.signalSemaphoreValueCount = 1U;
    tssi.pSignalSemaphoreValues = &signal_val;

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tssi;
    si.commandBufferCount = 1U;
    si.pCommandBuffers = &slot.cmd_buf;
    si.signalSemaphoreCount = 1U;
    si.pSignalSemaphores = &eng.timeline_sem;
    DARSHANA_VK_CHECK (vkQueueSubmit (eng.queue, 1U, &si, VK_NULL_HANDLE),
                       GpuErrorCode::DeviceLost,
                       "Failed to submit base pass command buffer");
  }

  VkSemaphoreWaitInfo swi{};
  swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  swi.semaphoreCount = 1U;
  swi.pSemaphores = &eng.timeline_sem;
  swi.pValues = &signal_val;
  DARSHANA_VK_CHECK (vkWaitSemaphores (eng.device, &swi, UINT64_MAX),
                     GpuErrorCode::DeviceLost,
                     "Failed waiting on base pass timeline semaphore");
#else
  (void)points_buf;
  (void)base_pixels_buf;
  (void)params;
#endif
  return {};
}

/* Executes the overlay compositing compute pass onto out_pixels_buf.  */
auto
gpu_execute_overlay_pass (GpuDevice &device, const GpuBuffer &base_pixels_buf,
                          GpuBuffer &out_pixels_buf,
                          const GpuOverlayPassParams &params) noexcept
    -> Result<void>
{
  if (!device.is_valid ())
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::DeviceLost, -4,
          "Cannot execute overlay GPU pass on a lost or invalid GpuDevice"));
    }
#if DARSHANA_HAS_VULKAN
  auto *dev_impl = static_cast<GpuDevice::Impl *> (device.native_handle ());
  auto &eng = *dev_impl->engine;

  const uint32_t slot_idx
      = eng.next_slot.fetch_add (1U, std::memory_order_relaxed)
        % VulkanHardwareEngine::kNumWorkerSlots;
  auto &slot = eng.worker_slots[slot_idx];
  std::lock_guard<std::mutex> slot_lock (slot.mutex);

  VkDescriptorBufferInfo dbis[3]{};
  dbis[0].buffer = eng.device_slab_buf;
  dbis[0].offset = base_pixels_buf.pool_offset ();
  dbis[0].range = base_pixels_buf.capacity_bytes ();

  dbis[1].buffer = eng.device_slab_buf;
  dbis[1].offset = base_pixels_buf.pool_offset ();
  dbis[1].range = base_pixels_buf.capacity_bytes ();

  dbis[2].buffer = eng.device_slab_buf;
  dbis[2].offset = out_pixels_buf.pool_offset ();
  dbis[2].range = out_pixels_buf.capacity_bytes ();

  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0U; i < 3U; ++i)
    {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = slot.desc_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1U;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &dbis[i];
    }
  vkUpdateDescriptorSets (eng.device, 3U, writes, 0U, nullptr);

  DARSHANA_VK_CHECK (vkResetCommandBuffer (slot.cmd_buf, 0U),
                     GpuErrorCode::DeviceLost,
                     "Failed to reset overlay pass command buffer");
  VkCommandBufferBeginInfo cbbi{};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  DARSHANA_VK_CHECK (vkBeginCommandBuffer (slot.cmd_buf, &cbbi),
                     GpuErrorCode::DeviceLost,
                     "Failed to begin overlay pass command buffer");

  vkCmdBindDescriptorSets (slot.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                           eng.pipe_layout, 0U, 1U, &slot.desc_set, 0U,
                           nullptr);
  vkCmdBindPipeline (slot.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                     eng.pipe_overlay);

  OverlayPushConstants opc{};
  opc.width = params.width;
  opc.height = params.height;
  opc.cross_x = params.cross_x;
  opc.cross_y = params.cross_y;
  opc.brush_x0 = params.brush_x0;
  opc.brush_y0 = params.brush_y0;
  opc.brush_x1 = params.brush_x1;
  opc.brush_y1 = params.brush_y1;

  vkCmdPushConstants (slot.cmd_buf, eng.pipe_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof (opc), &opc);
  vkCmdDispatch (slot.cmd_buf, (params.width + 15U) / 16U,
                 (params.height + 15U) / 16U, 1U);

  VkMemoryBarrier end_bar{};
  end_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  end_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  end_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier (slot.cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 1U, &end_bar, 0U,
                        nullptr, 0U, nullptr);

  DARSHANA_VK_CHECK (vkEndCommandBuffer (slot.cmd_buf),
                     GpuErrorCode::DeviceLost,
                     "Failed to end overlay pass command buffer");

  uint64_t signal_val = 0U;
  {
    std::lock_guard<std::mutex> q_lock (eng.queue_mutex);
    signal_val
        = eng.timeline_value.fetch_add (1U, std::memory_order_relaxed) + 1U;
    VkTimelineSemaphoreSubmitInfo tssi{};
    tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tssi.signalSemaphoreValueCount = 1U;
    tssi.pSignalSemaphoreValues = &signal_val;

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tssi;
    si.commandBufferCount = 1U;
    si.pCommandBuffers = &slot.cmd_buf;
    si.signalSemaphoreCount = 1U;
    si.pSignalSemaphores = &eng.timeline_sem;
    DARSHANA_VK_CHECK (vkQueueSubmit (eng.queue, 1U, &si, VK_NULL_HANDLE),
                       GpuErrorCode::DeviceLost,
                       "Failed to submit overlay pass command buffer");
  }

  VkSemaphoreWaitInfo swi{};
  swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  swi.semaphoreCount = 1U;
  swi.pSemaphores = &eng.timeline_sem;
  swi.pValues = &signal_val;
  DARSHANA_VK_CHECK (vkWaitSemaphores (eng.device, &swi, UINT64_MAX),
                     GpuErrorCode::DeviceLost,
                     "Failed waiting on overlay pass timeline semaphore");
#else
  (void)base_pixels_buf;
  (void)out_pixels_buf;
  (void)params;
#endif
  return {};
}

/* Reads back RGBA pixels from a device GpuBuffer into host memory.  */
auto
gpu_readback_pixels (GpuDevice &device, const GpuBuffer &pixels_buf,
                     std::span<uint8_t> dst_rgba) noexcept -> Result<void>
{
  if (!device.is_valid ())
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::DeviceLost, -4,
          "Cannot readback pixels from a lost or invalid GpuDevice"));
    }
#if DARSHANA_HAS_VULKAN
  auto *dev_impl = static_cast<GpuDevice::Impl *> (device.native_handle ());
  auto &eng = *dev_impl->engine;
  const std::size_t bytes = dst_rgba.size ();
  if (bytes == 0U || bytes > pixels_buf.capacity_bytes ())
    {
      return std::unexpected (Error::gpu (
          GpuErrorCode::TransferFailed, -1,
          "Readback buffer size does not match GpuBuffer capacity"));
    }

  const std::size_t aligned_bytes = align_up (bytes, kPoolAlign);
  std::size_t stg_off
      = eng.staging_cursor.fetch_add (aligned_bytes, std::memory_order_relaxed)
        % eng.staging_pool_capacity;
  if (stg_off + bytes > eng.staging_pool_capacity)
    {
      stg_off = 0U;
    }

  const uint32_t slot_idx
      = eng.next_slot.fetch_add (1U, std::memory_order_relaxed)
        % VulkanHardwareEngine::kNumWorkerSlots;
  auto &slot = eng.worker_slots[slot_idx];
  std::lock_guard<std::mutex> slot_lock (slot.mutex);

  DARSHANA_VK_CHECK (vkResetCommandBuffer (slot.cmd_buf, 0U),
                     GpuErrorCode::TransferFailed,
                     "Failed to reset readback command buffer");
  VkCommandBufferBeginInfo cbbi{};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  DARSHANA_VK_CHECK (vkBeginCommandBuffer (slot.cmd_buf, &cbbi),
                     GpuErrorCode::TransferFailed,
                     "Failed to begin readback command buffer");

  VkBufferCopy copy_region{};
  copy_region.srcOffset = pixels_buf.pool_offset ();
  copy_region.dstOffset = stg_off;
  copy_region.size = bytes;
  vkCmdCopyBuffer (slot.cmd_buf, eng.device_slab_buf, eng.staging_slab_buf, 1U,
                   &copy_region);

  DARSHANA_VK_CHECK (vkEndCommandBuffer (slot.cmd_buf),
                     GpuErrorCode::TransferFailed,
                     "Failed to end readback command buffer");

  uint64_t signal_val = 0U;
  {
    std::lock_guard<std::mutex> q_lock (eng.queue_mutex);
    signal_val
        = eng.timeline_value.fetch_add (1U, std::memory_order_relaxed) + 1U;
    VkTimelineSemaphoreSubmitInfo tssi{};
    tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tssi.signalSemaphoreValueCount = 1U;
    tssi.pSignalSemaphoreValues = &signal_val;

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tssi;
    si.commandBufferCount = 1U;
    si.pCommandBuffers = &slot.cmd_buf;
    si.signalSemaphoreCount = 1U;
    si.pSignalSemaphores = &eng.timeline_sem;
    DARSHANA_VK_CHECK (vkQueueSubmit (eng.queue, 1U, &si, VK_NULL_HANDLE),
                       GpuErrorCode::TransferFailed,
                       "Failed to submit readback command buffer");
  }

  VkSemaphoreWaitInfo swi{};
  swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  swi.semaphoreCount = 1U;
  swi.pSemaphores = &eng.timeline_sem;
  swi.pValues = &signal_val;
  DARSHANA_VK_CHECK (vkWaitSemaphores (eng.device, &swi, UINT64_MAX),
                     GpuErrorCode::TransferFailed,
                     "Failed waiting on readback timeline semaphore");

  std::memcpy (dst_rgba.data (),
               static_cast<const std::byte *> (eng.staging_mapped) + stg_off,
               bytes);
#else
  (void)pixels_buf;
  (void)dst_rgba;
#endif
  return {};
}

} /* namespace detail */

} /* namespace darshana::inline v1 */
