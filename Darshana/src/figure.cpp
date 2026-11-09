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

#include <darshana/figure.hpp>
#include <darshana/observability.hpp>

#include "gpu_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace darshana::inline v1
{

namespace
{

/* Alpha-blends a foreground ColorRgba over a packed ABGR pixel.  */
auto
blend_pixel (uint32_t dst_abgr, const ColorRgba &src,
             float cov) noexcept -> uint32_t
{
  const float alpha
      = std::clamp ((static_cast<float> (src.a) / 255.0F) * cov, 0.0F, 1.0F);
  const float inv = 1.0F - alpha;
  const auto dr = static_cast<float> (dst_abgr & 0xFFU);
  const auto dg = static_cast<float> ((dst_abgr >> 8U) & 0xFFU);
  const auto db = static_cast<float> ((dst_abgr >> 16U) & 0xFFU);
  const auto r = static_cast<uint32_t> (std::clamp (
      dr * inv + static_cast<float> (src.r) * alpha, 0.0F, 255.0F));
  const auto g = static_cast<uint32_t> (std::clamp (
      dg * inv + static_cast<float> (src.g) * alpha, 0.0F, 255.0F));
  const auto b = static_cast<uint32_t> (std::clamp (
      db * inv + static_cast<float> (src.b) * alpha, 0.0F, 255.0F));
  return 0xFF000000U | (b << 16U) | (g << 8U) | r;
}

/* Evaluates 2D signed-distance field for a MarkerShape centered at origin.  */
auto
marker_sdf (MarkerShape shape, float dx, float dy,
            float radius) noexcept -> float
{
  switch (shape)
    {
    case MarkerShape::Circle:
      return std::sqrt (dx * dx + dy * dy) - radius;
    case MarkerShape::Square:
      return std::max (std::abs (dx), std::abs (dy)) - radius;
    case MarkerShape::Diamond:
      return (std::abs (dx) + std::abs (dy)) * 0.70710678F - radius;
    case MarkerShape::Triangle:
      {
        const float ay = dy + radius * 0.35F;
        return std::max (std::abs (dx) * 0.8660254F + ay * 0.5F, -ay)
               - radius * 0.65F;
      }
    case MarkerShape::Cross:
      {
        const float arm
            = std::min (std::abs (dx), std::abs (dy)) - radius * 0.30F;
        const float box = std::max (std::abs (dx), std::abs (dy)) - radius;
        return std::max (arm, box);
      }
    }
  return std::sqrt (dx * dx + dy * dy) - radius;
}

/* Checks whether parametric arc-length s is active for a DashPattern.  */
auto
dash_active (DashPattern pat, float s) noexcept -> bool
{
  const float m = std::fmod (std::abs (s), 16.0F);
  switch (pat)
    {
    case DashPattern::Solid:
      return true;
    case DashPattern::Dashed:
      return m < 10.0F;
    case DashPattern::Dotted:
      return m < 3.0F || (m >= 8.0F && m < 11.0F);
    case DashPattern::DashDot:
      return m < 9.0F || (m >= 12.0F && m < 14.0F);
    }
  return true;
}

/* Renders an anti-aliased SDF marker into a packed ABGR framebuffer.  */
void
draw_sdf_marker (std::vector<uint32_t> &fb, uint32_t w, uint32_t h, float cx,
                 float cy, const AestheticSpec &aes) noexcept
{
  const float rad = std::max (1.0F, aes.marker_size);
  const float aa = std::max (0.75F, aes.sdf_aa_width);
  const float ext = rad + aa + 1.0F;
  const int x0 = std::max (0, static_cast<int> (std::floor (cx - ext)));
  const int x1 = std::min (static_cast<int> (w) - 1,
                           static_cast<int> (std::ceil (cx + ext)));
  const int y0 = std::max (0, static_cast<int> (std::floor (cy - ext)));
  const int y1 = std::min (static_cast<int> (h) - 1,
                           static_cast<int> (std::ceil (cy + ext)));
  for (int y = y0; y <= y1; ++y)
    {
      for (int x = x0; x <= x1; ++x)
        {
          const float d
              = marker_sdf (aes.marker_shape, static_cast<float> (x) - cx,
                            static_cast<float> (y) - cy, rad);
          if (d < aa)
            {
              const float cov = std::clamp (0.5F - d / (2.0F * aa), 0.0F, 1.0F)
                                * aes.opacity;
              const std::size_t idx = static_cast<std::size_t> (y) * w
                                      + static_cast<std::size_t> (x);
              fb[idx] = blend_pixel (fb[idx], aes.fill_color, cov);
            }
        }
    }
}

/* Renders an anti-aliased SDF line segment with dash pattern modulation.  */
void
draw_sdf_segment (std::vector<uint32_t> &fb, uint32_t w, uint32_t h, float x0,
                  float y0, float x1, float y1,
                  const AestheticSpec &aes) noexcept
{
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float len_sq = dx * dx + dy * dy;
  const float len = std::sqrt (len_sq);
  const float half_w = std::max (0.6F, aes.stroke_width * 0.5F);
  const float aa = std::max (0.75F, aes.sdf_aa_width);
  const float pad = half_w + aa + 1.0F;

  const int bx0
      = std::max (0, static_cast<int> (std::floor (std::min (x0, x1) - pad)));
  const int bx1
      = std::min (static_cast<int> (w) - 1,
                  static_cast<int> (std::ceil (std::max (x0, x1) + pad)));
  const int by0
      = std::max (0, static_cast<int> (std::floor (std::min (y0, y1) - pad)));
  const int by1
      = std::min (static_cast<int> (h) - 1,
                  static_cast<int> (std::ceil (std::max (y0, y1) + pad)));

  for (int y = by0; y <= by1; ++y)
    {
      for (int x = bx0; x <= bx1; ++x)
        {
          const float px = static_cast<float> (x) - x0;
          const float py = static_cast<float> (y) - y0;
          const float t
              = (len_sq > 1e-6F)
                    ? std::clamp ((px * dx + py * dy) / len_sq, 0.0F, 1.0F)
                    : 0.0F;
          if (!dash_active (aes.dash_pattern, t * len))
            {
              continue;
            }
          const float proj_x = px - t * dx;
          const float proj_y = py - t * dy;
          const float d
              = std::sqrt (proj_x * proj_x + proj_y * proj_y) - half_w;
          if (d < aa)
            {
              const float cov = std::clamp (0.5F - d / (2.0F * aa), 0.0F, 1.0F)
                                * aes.opacity;
              const std::size_t idx = static_cast<std::size_t> (y) * w
                                      + static_cast<std::size_t> (x);
              fb[idx] = blend_pixel (fb[idx], aes.stroke_color, cov);
            }
        }
    }
}

/* Renders a filled axis-aligned rectangle into the packed ABGR buffer.  */
void
draw_rect (std::vector<uint32_t> &fb, uint32_t w, uint32_t h, int x0, int y0,
           int x1, int y1, const ColorRgba &fill, float opacity) noexcept
{
  const int rx0 = std::clamp (std::min (x0, x1), 0, static_cast<int> (w) - 1);
  const int rx1 = std::clamp (std::max (x0, x1), 0, static_cast<int> (w) - 1);
  const int ry0 = std::clamp (std::min (y0, y1), 0, static_cast<int> (h) - 1);
  const int ry1 = std::clamp (std::max (y0, y1), 0, static_cast<int> (h) - 1);
  for (int y = ry0; y <= ry1; ++y)
    {
      for (int x = rx0; x <= rx1; ++x)
        {
          const std::size_t idx = static_cast<std::size_t> (y) * w
                                  + static_cast<std::size_t> (x);
          fb[idx] = blend_pixel (fb[idx], fill, opacity);
        }
    }
}

} /* namespace */

/* Computes the fraction of pixels matching reference within tolerance.  */
auto
ImageBuffer::pixel_match_ratio (const ImageBuffer &reference,
                                uint8_t tolerance) const noexcept
    -> Result<double>
{
  if (width == 0U || height == 0U || width != reference.width
      || height != reference.height || rgba.size () != reference.rgba.size ()
      || rgba.size () != static_cast<std::size_t> (width) * height * 4U)
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "ImageBuffer",
          "ImageBuffer dimensions do not match golden reference"));
    }
  const std::size_t num_pixels = static_cast<std::size_t> (width) * height;
  std::size_t matching = 0U;
  for (std::size_t i = 0U; i < num_pixels; ++i)
    {
      const std::size_t base = i * 4U;
      const int dr = std::abs (static_cast<int> (rgba[base])
                               - static_cast<int> (reference.rgba[base]));
      const int dg = std::abs (static_cast<int> (rgba[base + 1U])
                               - static_cast<int> (reference.rgba[base + 1U]));
      const int db = std::abs (static_cast<int> (rgba[base + 2U])
                               - static_cast<int> (reference.rgba[base + 2U]));
      const int da = std::abs (static_cast<int> (rgba[base + 3U])
                               - static_cast<int> (reference.rgba[base + 3U]));
      const int max_d = std::max ({ dr, dg, db, da });
      if (max_d <= static_cast<int> (tolerance))
        {
          ++matching;
        }
    }
  return static_cast<double> (matching) / static_cast<double> (num_pixels);
}

/* Computes the maximum per-channel difference against reference.  */
auto
ImageBuffer::max_channel_diff (const ImageBuffer &reference) const noexcept
    -> Result<uint8_t>
{
  if (width == 0U || height == 0U || width != reference.width
      || height != reference.height || rgba.size () != reference.rgba.size ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "ImageBuffer",
          "ImageBuffer dimensions do not match golden reference"));
    }
  uint8_t max_diff = 0U;
  for (std::size_t i = 0U; i < rgba.size (); ++i)
    {
      const auto diff = static_cast<uint8_t> (std::abs (
          static_cast<int> (rgba[i]) - static_cast<int> (reference.rgba[i])));
      max_diff = std::max (max_diff, diff);
    }
  return max_diff;
}

/* Computes a 64-bit FNV-1a checksum of the RGBA pixel buffer.  */
auto
ImageBuffer::fnv1a_hash () const noexcept -> uint64_t
{
  uint64_t hash = 14695981039346656037ULL;
  for (const uint8_t b : rgba)
    {
      hash ^= static_cast<uint64_t> (b);
      hash *= 1099511628211ULL;
    }
  return hash;
}

struct Figure::Impl
{
  Spec spec;
  BackendKind requested_backend = BackendKind::Auto;
  BackendKind active_backend = BackendKind::CpuVector;
  std::size_t max_points = 65536U;

  ScratchArena arena{ 131072U };
  FrameStats stats;
  SnapshotData snapshot;
  StreamingBuffer stream_buf;
  bool use_stream = false;

  GpuDevice gpu_device;
  GpuBuffer gpu_points_buf;
  GpuBuffer gpu_base_buf;
  GpuBuffer gpu_out_buf;
  bool gpu_initialized = false;
  bool explicit_device = false;
  uint32_t uploaded_point_count = 0U;

  std::vector<uint32_t> cpu_base_pixels;
  std::vector<uint32_t> cpu_out_pixels;

  bool base_dirty = true;
  bool overlay_dirty = true;
  bool stream_gpu_dirty = false;

  /* Lazily initializes Vulkan buffers from the pre-allocated device pool.  */
  auto
  ensure_gpu_resources () noexcept -> Result<void>
  {
    if (active_backend != BackendKind::VulkanGpu)
      {
        return {};
      }
    if (gpu_initialized)
      {
        if (!gpu_device.is_valid ())
          {
            return std::unexpected (
                Error::gpu (GpuErrorCode::DeviceLost, -4,
                            "Vulkan GpuDevice experienced device loss"));
          }
        return {};
      }
    if (!gpu_device.is_valid ())
      {
        if (explicit_device)
          {
            Logger::log (
                LogLevel::Warn, "figure",
                "Provided GpuDevice is invalid; falling back to CpuVector");
            active_backend = BackendKind::CpuVector;
            return {};
          }
        auto dev_res = detail::acquire_shared_gpu_device ();
        if (!dev_res.has_value ())
          {
            Logger::log (LogLevel::Warn, "figure",
                         "Vulkan initialization failed; degrading gracefully "
                         "to CpuVector backend");
            active_backend = BackendKind::CpuVector;
            return {};
          }
        gpu_device = std::move (*dev_res);
      }

    const std::size_t pts_bytes
        = std::max<std::size_t> (max_points, 256U) * sizeof (DataPoint2D);
    const std::size_t fb_bytes = static_cast<std::size_t> (spec.width)
                                 * spec.height * sizeof (uint32_t);

    auto p_res = GpuBuffer::allocate (gpu_device, pts_bytes);
    auto b_res = GpuBuffer::allocate (gpu_device, fb_bytes);
    auto o_res = GpuBuffer::allocate (gpu_device, fb_bytes);
    if (!p_res.has_value () || !b_res.has_value () || !o_res.has_value ())
      {
        Logger::log (LogLevel::Warn, "figure",
                     "GPU pool sub-allocation failed; falling back to "
                     "CpuVector backend");
        active_backend = BackendKind::CpuVector;
        return {};
      }
    gpu_points_buf = std::move (*p_res);
    gpu_base_buf = std::move (*b_res);
    gpu_out_buf = std::move (*o_res);
    gpu_initialized = true;
    return {};
  }

  /* Renders the static base layer on CPU using analytical SDF primitives.  */
  void
  render_cpu_base ()
  {
    const uint32_t w = spec.width;
    const uint32_t h = spec.height;
    const std::size_t total_px = static_cast<std::size_t> (w) * h;
    if (cpu_base_pixels.size () != total_px)
      {
        cpu_base_pixels.resize (total_px);
      }
    const uint32_t bg = spec.background.pack_abgr ();
    std::ranges::fill (cpu_base_pixels, bg);

    const uint32_t margin_l = std::min (60U, w / 4U);
    const uint32_t margin_r = w > 30U ? w - 30U : w;
    const uint32_t margin_t = std::min (40U, h / 4U);
    const uint32_t margin_b = h > 50U ? h - 50U : h;
    const float plot_w
        = static_cast<float> (margin_r > margin_l ? margin_r - margin_l : 1U);
    const float plot_h
        = static_cast<float> (margin_b > margin_t ? margin_b - margin_t : 1U);

    if (spec.show_grid && margin_r > margin_l && margin_b > margin_t)
      {
        for (uint32_t y = margin_t; y <= margin_b; ++y)
          {
            for (uint32_t x = margin_l; x <= margin_r; ++x)
              {
                if (x == margin_l || x == margin_r || y == margin_t
                    || y == margin_b)
                  {
                    cpu_base_pixels[static_cast<std::size_t> (y) * w + x]
                        = 0xFF333333U;
                  }
                else if (((x - margin_l) * 5U) % (margin_r - margin_l) < 5U
                         || ((y - margin_t) * 5U) % (margin_b - margin_t) < 5U)
                  {
                    cpu_base_pixels[static_cast<std::size_t> (y) * w + x]
                        = 0xFFE6E6E6U;
                  }
              }
          }
      }

    const auto map_x = [&] (double vx) -> float {
      auto m = spec.x_scale.map (vx);
      const double t = m.has_value () ? *m : 0.5;
      return static_cast<float> (margin_l) + static_cast<float> (t) * plot_w;
    };
    const auto map_y = [&] (double vy) -> float {
      auto m = spec.y_scale.map (vy);
      const double t = m.has_value () ? *m : 0.5;
      return static_cast<float> (margin_b) - static_cast<float> (t) * plot_h;
    };

    for (const auto &layer : spec.layers)
      {
        if (layer.mark == MarkType::Heatmap && layer.matrix_rows > 0U
            && layer.matrix_cols > 0U
            && layer.matrix_values.size ()
                   == layer.matrix_rows * layer.matrix_cols)
          {
            const float cell_w
                = plot_w / static_cast<float> (layer.matrix_cols);
            const float cell_h
                = plot_h / static_cast<float> (layer.matrix_rows);
            for (std::size_t r = 0U; r < layer.matrix_rows; ++r)
              {
                for (std::size_t c = 0U; c < layer.matrix_cols; ++c)
                  {
                    const float val = std::clamp (
                        layer.matrix_values[r * layer.matrix_cols + c], -1.0F,
                        1.0F);
                    const auto red = static_cast<uint8_t> (
                        std::clamp ((val + 1.0F) * 127.5F, 0.0F, 255.0F));
                    const auto blue = static_cast<uint8_t> (
                        std::clamp ((1.0F - val) * 127.5F, 0.0F, 255.0F));
                    const ColorRgba cell_col{ red, 90U, blue, 255U };
                    const int x0
                        = static_cast<int> (static_cast<float> (margin_l)
                                            + static_cast<float> (c) * cell_w);
                    const int x1 = static_cast<int> (
                        static_cast<float> (margin_l)
                        + static_cast<float> (c + 1U) * cell_w);
                    const int y0
                        = static_cast<int> (static_cast<float> (margin_t)
                                            + static_cast<float> (r) * cell_h);
                    const int y1 = static_cast<int> (
                        static_cast<float> (margin_t)
                        + static_cast<float> (r + 1U) * cell_h);
                    draw_rect (cpu_base_pixels, w, h, x0, y0, x1, y1, cell_col,
                               layer.aesthetics.opacity);
                  }
              }
            continue;
          }

        std::vector<DataPoint2D> temp_pts;
        std::span<const DataPoint2D> pts;
        if (!layer.inline_x.empty ()
            && layer.inline_x.size () == layer.inline_y.size ())
          {
            temp_pts.resize (layer.inline_x.size ());
            for (std::size_t i = 0U; i < layer.inline_x.size (); ++i)
              {
                temp_pts[i]
                    = DataPoint2D{ layer.inline_x[i], layer.inline_y[i] };
              }
            pts = temp_pts;
          }
        else if (use_stream && stream_buf.size () > 0U)
          {
            temp_pts.resize (stream_buf.size ());
            (void)stream_buf.copy_ordered (temp_pts);
            pts = temp_pts;
          }
        else if (snapshot.point_count () > 0U)
          {
            pts = snapshot.points ();
          }

        if (pts.empty ())
          {
            continue;
          }

        if (layer.mark == MarkType::Bar || layer.mark == MarkType::Histogram)
          {
            const int bar_half = std::max (
                2, static_cast<int> (plot_w
                                     / static_cast<float> (pts.size () * 3U)));
            const int y_base = static_cast<int> (
                map_y (std::max (0.0, spec.y_scale.domain_min)));
            for (const auto &pt : pts)
              {
                const int cx = static_cast<int> (map_x (pt.x));
                const int cy = static_cast<int> (map_y (pt.y));
                draw_rect (cpu_base_pixels, w, h, cx - bar_half, cy,
                           cx + bar_half, y_base, layer.aesthetics.fill_color,
                           layer.aesthetics.opacity);
              }
          }
        else if (layer.mark == MarkType::Box || layer.mark == MarkType::Violin)
          {
            const int box_half = std::max (
                4, static_cast<int> (plot_w
                                     / static_cast<float> (pts.size () * 4U)));
            for (std::size_t i = 0U; i < pts.size (); ++i)
              {
                const int cx = static_cast<int> (map_x (pts[i].x));
                const int y_med = static_cast<int> (map_y (pts[i].y));
                const float q1_v = i < layer.inline_q1.size ()
                                       ? layer.inline_q1[i]
                                       : (pts[i].y - 0.5F);
                const float q3_v = i < layer.inline_q3.size ()
                                       ? layer.inline_q3[i]
                                       : (pts[i].y + 0.5F);
                const int y_q1 = static_cast<int> (map_y (q1_v));
                const int y_q3 = static_cast<int> (map_y (q3_v));
                draw_rect (cpu_base_pixels, w, h, cx - box_half, y_q3,
                           cx + box_half, y_q1, layer.aesthetics.fill_color,
                           layer.aesthetics.opacity);
                draw_sdf_segment (
                    cpu_base_pixels, w, h, static_cast<float> (cx - box_half),
                    static_cast<float> (y_med),
                    static_cast<float> (cx + box_half),
                    static_cast<float> (y_med), layer.aesthetics);
              }
          }
        else if (layer.mark == MarkType::ConfidenceBand
                 || layer.mark == MarkType::Area)
          {
            for (std::size_t i = 0U; i + 1U < pts.size (); ++i)
              {
                const int x0 = static_cast<int> (map_x (pts[i].x));
                const int x1 = static_cast<int> (map_x (pts[i + 1U].x));
                const int y_lo = static_cast<int> (map_y (pts[i].y));
                const float y2_v
                    = i < layer.inline_y2.size ()
                          ? layer.inline_y2[i]
                          : static_cast<float> (spec.y_scale.domain_min);
                const int y_hi = static_cast<int> (map_y (y2_v));
                draw_rect (cpu_base_pixels, w, h, x0, y_hi, x1, y_lo,
                           layer.aesthetics.fill_color,
                           layer.aesthetics.opacity);
              }
          }
        else if (layer.mark == MarkType::ErrorBar)
          {
            for (std::size_t i = 0U; i < pts.size (); ++i)
              {
                const float cx = map_x (pts[i].x);
                const float y0 = map_y (pts[i].y);
                const float y1
                    = map_y (i < layer.inline_y2.size () ? layer.inline_y2[i]
                                                         : pts[i].y);
                draw_sdf_segment (cpu_base_pixels, w, h, cx, y0, cx, y1,
                                  layer.aesthetics);
              }
          }
        else if (layer.mark == MarkType::Line || layer.mark == MarkType::Kde
                 || layer.mark == MarkType::Ecdf
                 || layer.mark == MarkType::RegressionLine)
          {
            for (std::size_t i = 0U; i + 1U < pts.size (); ++i)
              {
                draw_sdf_segment (cpu_base_pixels, w, h, map_x (pts[i].x),
                                  map_y (pts[i].y), map_x (pts[i + 1U].x),
                                  map_y (pts[i + 1U].y), layer.aesthetics);
              }
          }
        else
          {
            if (pts.size () <= 8192U)
              {
                for (const auto &pt : pts)
                  {
                    draw_sdf_marker (cpu_base_pixels, w, h, map_x (pt.x),
                                     map_y (pt.y), layer.aesthetics);
                  }
              }
            else
              {
                const uint32_t col = layer.aesthetics.fill_color.pack_abgr ();
                for (const auto &pt : pts)
                  {
                    const int ix = static_cast<int> (map_x (pt.x));
                    const int iy = static_cast<int> (map_y (pt.y));
                    if (ix >= 0 && ix < static_cast<int> (w) && iy >= 0
                        && iy < static_cast<int> (h))
                      {
                        cpu_base_pixels[static_cast<std::size_t> (iy) * w
                                        + static_cast<std::size_t> (ix)]
                            = col;
                      }
                  }
              }
          }
      }
  }

  /* Composites interactive overlays onto the cached CPU base buffer.  */
  void
  render_cpu_overlay ()
  {
    const uint32_t w = spec.width;
    const uint32_t h = spec.height;
    cpu_out_pixels = cpu_base_pixels;

    if (spec.overlay.brush_enabled)
      {
        const int bx0 = static_cast<int> (spec.overlay.brush_x0
                                          * static_cast<float> (w));
        const int by0 = static_cast<int> (spec.overlay.brush_y0
                                          * static_cast<float> (h));
        const int bx1 = static_cast<int> (spec.overlay.brush_x1
                                          * static_cast<float> (w));
        const int by1 = static_cast<int> (spec.overlay.brush_y1
                                          * static_cast<float> (h));
        draw_rect (cpu_out_pixels, w, h, bx0, by0, bx1, by1,
                   ColorRgba{ 0U, 136U, 255U, 70U }, 1.0F);
      }
    if (spec.overlay.crosshair_enabled)
      {
        const auto cx = static_cast<uint32_t> (
            std::clamp (spec.overlay.crosshair_x * static_cast<float> (w),
                        0.0F, static_cast<float> (w - 1U)));
        const auto cy = static_cast<uint32_t> (
            std::clamp (spec.overlay.crosshair_y * static_cast<float> (h),
                        0.0F, static_cast<float> (h - 1U)));
        for (uint32_t y = 0U; y < h; ++y)
          {
            cpu_out_pixels[static_cast<std::size_t> (y) * w + cx]
                = 0xFF2222DDU;
          }
        for (uint32_t x = 0U; x < w; ++x)
          {
            cpu_out_pixels[static_cast<std::size_t> (cy) * w + x]
                = 0xFF2222DDU;
          }
      }
    if (spec.overlay.tooltip_enabled)
      {
        const int tx = static_cast<int> (spec.overlay.tooltip_x
                                         * static_cast<float> (w));
        const int ty = static_cast<int> (spec.overlay.tooltip_y
                                         * static_cast<float> (h));
        draw_rect (cpu_out_pixels, w, h, tx, ty - 18, tx + 64, ty,
                   ColorRgba{ 30U, 30U, 30U, 210U }, 1.0F);
      }
  }
};

/* Default-constructs an empty Figure handle.  */
Figure::Figure () noexcept = default;

/* Constructs a Figure from an internal implementation pointer.  */
Figure::Figure (std::unique_ptr<Impl> impl) noexcept : impl_ (std::move (impl))
{
}

Figure::Figure (Figure &&other) noexcept = default;

auto Figure::operator= (Figure &&other) noexcept -> Figure & = default;

Figure::~Figure () noexcept = default;

/* Creates a Figure from a declarative Spec and backend preference.  */
auto
Figure::create (const Spec &spec, BackendKind backend,
                std::size_t max_points) noexcept -> Result<Figure>
{
  if (auto val = spec.validate (); !val.has_value ())
    {
      return std::unexpected (val.error ());
    }
  if (max_points == 0U)
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidSpec, "max_points",
          "Figure::create requires max_points greater than zero"));
    }
  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->spec = spec;
      impl->requested_backend = backend;
      impl->max_points = max_points;

      const std::size_t stream_cap
          = std::min<std::size_t> (max_points, 65536U);
      auto st_res = StreamingBuffer::create (stream_cap);
      if (!st_res.has_value ())
        {
          return std::unexpected (st_res.error ());
        }
      impl->stream_buf = std::move (*st_res);

      if (backend == BackendKind::CpuVector)
        {
          impl->active_backend = BackendKind::CpuVector;
        }
      else
        {
#if DARSHANA_HAS_VULKAN
          impl->active_backend = BackendKind::VulkanGpu;
#else
          Logger::log (LogLevel::Warn, "figure",
                       "Vulkan disabled at build time; falling back to "
                       "CpuVector backend");
          impl->active_backend = BackendKind::CpuVector;
#endif
        }
      return Figure (std::move (impl));
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory while constructing Figure"));
    }
}

/* Creates a Figure bound to an explicit GpuDevice instance.  */
auto
Figure::create_with_device (const Spec &spec, GpuDevice &device,
                            std::size_t max_points) noexcept -> Result<Figure>
{
  auto fig_res = create (spec, BackendKind::VulkanGpu, max_points);
  if (!fig_res.has_value ())
    {
      return std::unexpected (fig_res.error ());
    }
  fig_res->impl_->explicit_device = true;
  if (!device.is_valid ())
    {
      Logger::log (LogLevel::Warn, "figure",
                   "Explicit GpuDevice is invalid; falling back to CpuVector");
      fig_res->impl_->active_backend = BackendKind::CpuVector;
      return fig_res;
    }

  const std::size_t pts_bytes
      = std::max<std::size_t> (max_points, 256U) * sizeof (DataPoint2D);
  const std::size_t fb_bytes = static_cast<std::size_t> (spec.width)
                               * spec.height * sizeof (uint32_t);
  auto p_res = GpuBuffer::allocate (device, pts_bytes);
  auto b_res = GpuBuffer::allocate (device, fb_bytes);
  auto o_res = GpuBuffer::allocate (device, fb_bytes);
  if (!p_res.has_value () || !b_res.has_value () || !o_res.has_value ())
    {
      Logger::log (LogLevel::Warn, "figure",
                   "GpuDevice pool sub-allocation failed; falling back to "
                   "CpuVector");
      fig_res->impl_->active_backend = BackendKind::CpuVector;
      return fig_res;
    }
  fig_res->impl_->gpu_device = GpuDevice::create ().value_or (GpuDevice{});
  fig_res->impl_->gpu_points_buf = std::move (*p_res);
  fig_res->impl_->gpu_base_buf = std::move (*b_res);
  fig_res->impl_->gpu_out_buf = std::move (*o_res);
  fig_res->impl_->gpu_initialized = true;
  return fig_res;
}

/* Replaces the plot Spec and marks the static base layer dirty.  */
auto
Figure::set_spec (const Spec &spec) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  if (auto val = spec.validate (); !val.has_value ())
    {
      return std::unexpected (val.error ());
    }
  impl_->spec = spec;
  impl_->base_dirty = true;
  impl_->overlay_dirty = true;
  return {};
}

/* Retrieves the current declarative plot Spec.  */
auto
Figure::spec () const noexcept -> const Spec &
{
  return impl_->spec;
}

/* Binds a static SnapshotData dataset and schedules async GPU upload.  */
auto
Figure::bind_snapshot (const SnapshotData &data) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  if (data.point_count () == 0U)
    {
      return std::unexpected (
          Error::data (DataErrorCode::EmptyDataset, "data",
                       "Cannot bind an empty SnapshotData dataset to Figure"));
    }
  if (data.point_count () > impl_->max_points)
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "data",
          "SnapshotData point_count exceeds Figure pre-allocated max_points"));
    }
  impl_->snapshot = data;
  impl_->use_stream = false;
  impl_->base_dirty = true;
  impl_->overlay_dirty = true;

  if (impl_->active_backend == BackendKind::VulkanGpu)
    {
      if (auto r = impl_->ensure_gpu_resources (); !r.has_value ())
        {
          return std::unexpected (r.error ());
        }
      if (impl_->active_backend == BackendKind::VulkanGpu)
        {
          ScopedFrameTimer up_timer (impl_->stats, FrameStage::GpuUpload,
                                     "bind_snapshot_upload");
          auto up_res
              = impl_->gpu_points_buf.upload_points_async (data.points (), 0U);
          if (!up_res.has_value ())
            {
              return std::unexpected (up_res.error ());
            }
          impl_->uploaded_point_count
              = static_cast<uint32_t> (data.point_count ());
        }
    }
  return {};
}

/* Binds raw x and y coordinate spans as a snapshot dataset.  */
auto
Figure::bind_xy (std::span<const float> x,
                 std::span<const float> y) noexcept -> Result<void>
{
  auto snap_res = SnapshotData::from_xy (x, y);
  if (!snap_res.has_value ())
    {
      return std::unexpected (snap_res.error ());
    }
  return bind_snapshot (*snap_res);
}

/* Appends a single streaming data point in O(1) to the ring buffer.  */
auto
Figure::append_stream (float x, float y) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  if (auto r = impl_->stream_buf.push (x, y); !r.has_value ())
    {
      return std::unexpected (r.error ());
    }
  impl_->use_stream = true;
  impl_->base_dirty = true;
  impl_->overlay_dirty = true;
  impl_->stream_gpu_dirty = true;
  return {};
}

/* Appends a batch of streaming data points to the O(1) ring buffer.  */
auto
Figure::append_stream_batch (std::span<const float> x,
                             std::span<const float> y) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  if (auto r = impl_->stream_buf.push_batch (x, y); !r.has_value ())
    {
      return std::unexpected (r.error ());
    }
  impl_->use_stream = true;
  impl_->base_dirty = true;
  impl_->overlay_dirty = true;
  impl_->stream_gpu_dirty = true;
  return {};
}

/* Updates interactive overlays WITHOUT invalidating static base layer.  */
auto
Figure::update_overlay (const OverlaySpec &overlay) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  if (!std::isfinite (overlay.crosshair_x)
      || !std::isfinite (overlay.crosshair_y)
      || !std::isfinite (overlay.brush_x0) || !std::isfinite (overlay.brush_y0)
      || !std::isfinite (overlay.brush_x1)
      || !std::isfinite (overlay.brush_y1))
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidSpec, "overlay",
          "OverlaySpec coordinates must be finite floating-point numbers"));
    }
  impl_->spec.overlay = overlay;
  impl_->overlay_dirty = true;
  return {};
}

/* Updates the axis viewport domain for interactive pan/zoom.  */
auto
Figure::set_viewport (double x_min, double x_max, double y_min,
                      double y_max) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  if (!std::isfinite (x_min) || !std::isfinite (x_max) || x_min >= x_max
      || !std::isfinite (y_min) || !std::isfinite (y_max) || y_min >= y_max)
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidScaleDomain, "viewport",
                       "set_viewport requires finite bounds with min < max"));
    }
  impl_->spec.x_scale.domain_min = x_min;
  impl_->spec.x_scale.domain_max = x_max;
  impl_->spec.y_scale.domain_min = y_min;
  impl_->spec.y_scale.domain_max = y_max;
  impl_->base_dirty = true;
  impl_->overlay_dirty = true;
  return {};
}

/* Renders a frame using the layered GPU or CPU pipeline.  */
auto
Figure::render_frame (bool force_base_redraw) noexcept
    -> Result<FrameStatsSnapshot>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  const uint64_t frame_start = monotonic_time_ns ();
  impl_->arena.reset ();
  {
    ScopedFrameTimer prep_timer (impl_->stats, FrameStage::CpuPrep,
                                 "cpu_prep");
    auto scratch_res = impl_->arena.allocate (256U, 16U);
    if (scratch_res.has_value ())
      {
        impl_->stats.record_arena_usage (impl_->arena.used_bytes ());
      }
  }

  const bool need_gpu_now
      = (impl_->requested_backend == BackendKind::VulkanGpu)
        || impl_->gpu_initialized || impl_->use_stream
        || (impl_->snapshot.point_count () > 0U);

  if (impl_->active_backend == BackendKind::VulkanGpu && need_gpu_now)
    {
      if (auto r = impl_->ensure_gpu_resources (); !r.has_value ())
        {
          return std::unexpected (r.error ());
        }
    }

  const bool need_base = impl_->base_dirty || force_base_redraw;
  const bool need_overlay = impl_->overlay_dirty || need_base;

  if (impl_->active_backend == BackendKind::VulkanGpu
      && impl_->gpu_initialized)
    {
      if (impl_->use_stream && impl_->stream_gpu_dirty)
        {
          ScopedFrameTimer up_timer (impl_->stats, FrameStage::GpuUpload,
                                     "stream_upload");
          const auto raw = impl_->stream_buf.raw_storage ();
          if (!raw.empty ())
            {
              auto up_res
                  = impl_->gpu_points_buf.upload_points_async (raw, 0U);
              if (!up_res.has_value ())
                {
                  return std::unexpected (up_res.error ());
                }
              impl_->uploaded_point_count
                  = static_cast<uint32_t> (raw.size ());
            }
          impl_->stream_gpu_dirty = false;
        }
      else if (!impl_->use_stream && impl_->uploaded_point_count == 0U
               && !impl_->spec.layers.empty ())
        {
          const auto &l0 = impl_->spec.layers.front ();
          const std::size_t n
              = std::min (l0.inline_x.size (), l0.inline_y.size ());
          if (n > 0U && n <= impl_->max_points)
            {
              ScopedFrameTimer up_timer (impl_->stats, FrameStage::GpuUpload,
                                         "inline_upload");
              std::vector<DataPoint2D> pts (n);
              for (std::size_t i = 0U; i < n; ++i)
                {
                  pts[i] = DataPoint2D{ l0.inline_x[i], l0.inline_y[i] };
                }
              auto up_res
                  = impl_->gpu_points_buf.upload_points_async (pts, 0U);
              if (!up_res.has_value ())
                {
                  return std::unexpected (up_res.error ());
                }
              impl_->uploaded_point_count = static_cast<uint32_t> (n);
            }
        }

      if (need_base)
        {
          ScopedFrameTimer draw_timer (impl_->stats, FrameStage::GpuDraw,
                                       "gpu_base_pass");
          const uint32_t w = impl_->spec.width;
          const uint32_t h = impl_->spec.height;
          const float margin_l = 60.0F;
          const float margin_r = w > 90U ? static_cast<float> (w - 30U)
                                         : static_cast<float> (w);
          const float margin_t = 40.0F;
          const float margin_b = h > 90U ? static_cast<float> (h - 50U)
                                         : static_cast<float> (h);
          const float plot_w = std::max (1.0F, margin_r - margin_l);
          const float plot_h = std::max (1.0F, margin_b - margin_t);
          const double dom_w = impl_->spec.x_scale.domain_max
                               - impl_->spec.x_scale.domain_min;
          const double dom_h = impl_->spec.y_scale.domain_max
                               - impl_->spec.y_scale.domain_min;

          detail::GpuBasePassParams bp{};
          bp.width = w;
          bp.height = h;
          bp.point_count = impl_->uploaded_point_count;
          bp.is_line_plot
              = !impl_->spec.layers.empty ()
                && (impl_->spec.layers.front ().mark == MarkType::Line
                    || impl_->spec.layers.front ().mark == MarkType::Kde
                    || impl_->spec.layers.front ().mark
                           == MarkType::RegressionLine);
          bp.scale_x
              = static_cast<float> (static_cast<double> (plot_w) / dom_w);
          bp.offset_x = margin_l
                        - static_cast<float> (impl_->spec.x_scale.domain_min)
                              * bp.scale_x;
          bp.scale_y
              = static_cast<float> (-static_cast<double> (plot_h) / dom_h);
          bp.offset_y = margin_b
                        - static_cast<float> (impl_->spec.y_scale.domain_min)
                              * bp.scale_y;
          bp.bg_color = impl_->spec.background.pack_abgr ();
          bp.grid_color = impl_->spec.show_grid ? 0xFFE6E6E6U : 0U;
          bp.mark_color = !impl_->spec.layers.empty ()
                              ? impl_->spec.layers.front ()
                                    .aesthetics.fill_color.pack_abgr ()
                              : 0xFFB4771FU;

          if (auto r = detail::gpu_execute_base_pass (impl_->gpu_device,
                                                      impl_->gpu_points_buf,
                                                      impl_->gpu_base_buf, bp);
              !r.has_value ())
            {
              return std::unexpected (r.error ());
            }
          impl_->stats.record_base_redraw ();
          impl_->base_dirty = false;
        }

      if (need_overlay)
        {
          ScopedFrameTimer pres_timer (impl_->stats, FrameStage::Present,
                                       "gpu_overlay_pass");
          const uint32_t w = impl_->spec.width;
          const uint32_t h = impl_->spec.height;
          detail::GpuOverlayPassParams op{};
          op.width = w;
          op.height = h;
          if (impl_->spec.overlay.crosshair_enabled)
            {
              op.cross_x = static_cast<uint32_t> (std::clamp (
                  impl_->spec.overlay.crosshair_x * static_cast<float> (w),
                  0.0F, static_cast<float> (w - 1U)));
              op.cross_y = static_cast<uint32_t> (std::clamp (
                  impl_->spec.overlay.crosshair_y * static_cast<float> (h),
                  0.0F, static_cast<float> (h - 1U)));
            }
          if (impl_->spec.overlay.brush_enabled)
            {
              op.brush_x0 = static_cast<uint32_t> (std::clamp (
                  impl_->spec.overlay.brush_x0 * static_cast<float> (w), 0.0F,
                  static_cast<float> (w)));
              op.brush_y0 = static_cast<uint32_t> (std::clamp (
                  impl_->spec.overlay.brush_y0 * static_cast<float> (h), 0.0F,
                  static_cast<float> (h)));
              op.brush_x1 = static_cast<uint32_t> (std::clamp (
                  impl_->spec.overlay.brush_x1 * static_cast<float> (w), 0.0F,
                  static_cast<float> (w)));
              op.brush_y1 = static_cast<uint32_t> (std::clamp (
                  impl_->spec.overlay.brush_y1 * static_cast<float> (h), 0.0F,
                  static_cast<float> (h)));
            }

          if (auto r = detail::gpu_execute_overlay_pass (
                  impl_->gpu_device, impl_->gpu_base_buf, impl_->gpu_out_buf,
                  op);
              !r.has_value ())
            {
              return std::unexpected (r.error ());
            }
          impl_->stats.record_overlay_redraw ();
          impl_->overlay_dirty = false;
        }
    }
  else
    {
      if (need_base)
        {
          ScopedFrameTimer draw_timer (impl_->stats, FrameStage::GpuDraw,
                                       "cpu_base_pass");
          impl_->render_cpu_base ();
          impl_->stats.record_base_redraw ();
          impl_->base_dirty = false;
        }
      if (need_overlay)
        {
          ScopedFrameTimer pres_timer (impl_->stats, FrameStage::Present,
                                       "cpu_overlay_pass");
          impl_->render_cpu_overlay ();
          impl_->stats.record_overlay_redraw ();
          impl_->overlay_dirty = false;
        }
    }

  const uint64_t elapsed = monotonic_time_ns () - frame_start;
  impl_->stats.record_frame (elapsed);
  return impl_->stats.snapshot ();
}

/* Reads back the composited RGBA framebuffer into host memory.  */
auto
Figure::readback_image () noexcept -> Result<ImageBuffer>
{
  if (!impl_)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0, "Figure is moved-from"));
    }
  if (impl_->base_dirty || impl_->overlay_dirty)
    {
      if (auto r = render_frame (false); !r.has_value ())
        {
          return std::unexpected (r.error ());
        }
    }
  try
    {
      ImageBuffer img;
      img.width = impl_->spec.width;
      img.height = impl_->spec.height;
      const std::size_t bytes
          = static_cast<std::size_t> (img.width) * img.height * 4U;
      img.rgba.resize (bytes);

      if (impl_->active_backend == BackendKind::VulkanGpu)
        {
          const bool simple_mark
              = impl_->spec.layers.size () == 1U
                && (impl_->spec.layers.front ().mark == MarkType::Scatter
                    || impl_->spec.layers.front ().mark == MarkType::Line)
                && impl_->spec.layers.front ().aesthetics.marker_shape
                       == MarkerShape::Circle
                && impl_->spec.layers.front ().aesthetics.dash_pattern
                       == DashPattern::Solid
                && impl_->uploaded_point_count > 8192U;
          if (simple_mark)
            {
              if (auto r = detail::gpu_readback_pixels (
                      impl_->gpu_device, impl_->gpu_out_buf, img.rgba);
                  !r.has_value ())
                {
                  return std::unexpected (r.error ());
                }
              return img;
            }
        }

      impl_->render_cpu_base ();
      impl_->render_cpu_overlay ();
      std::memcpy (img.rgba.data (), impl_->cpu_out_pixels.data (), bytes);
      return img;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory during Figure::readback_image"));
    }
}

/* Asynchronously exports the Figure to SVG or PDF without blocking.  */
auto
Figure::export_async (std::string_view file_path, ExportFormat format,
                      ExportCallback callback) const noexcept
    -> std::future<Result<ExportResult>>
{
  return export_file_async (impl_->spec, std::string (file_path), format,
                            impl_->snapshot, impl_->stream_buf,
                            std::move (callback));
}

/* Queries the active rendering backend for this Figure.  */
auto
Figure::active_backend () const noexcept -> BackendKind
{
  return impl_ ? impl_->active_backend : BackendKind::CpuVector;
}

/* Accesses the per-figure FrameStats telemetry accumulator.  */
auto
Figure::stats () const noexcept -> const FrameStats &
{
  return impl_->stats;
}

/* Queries how many times the static base data layer was redrawn.  */
auto
Figure::base_redraw_count () const noexcept -> uint64_t
{
  return impl_ ? impl_->stats.snapshot ().base_layer_redraws : 0U;
}

/* Queries how many times the interactive overlay layer was composited.  */
auto
Figure::overlay_redraw_count () const noexcept -> uint64_t
{
  return impl_ ? impl_->stats.snapshot ().overlay_layer_redraws : 0U;
}

} /* namespace darshana::inline v1 */
