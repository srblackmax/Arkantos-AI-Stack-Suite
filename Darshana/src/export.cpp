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

#include <darshana/export.hpp>
#include <darshana/observability.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

namespace darshana::inline v1
{

namespace
{

/* Formats a ColorRgba into a 7-character #RRGGBB hex string.  */
auto
color_to_svg_hex (const ColorRgba &c) -> std::string
{
  constexpr char kHex[] = "0123456789abcdef";
  std::string out = "#000000";
  out[1] = kHex[(c.r >> 4U) & 0x0FU];
  out[2] = kHex[c.r & 0x0FU];
  out[3] = kHex[(c.g >> 4U) & 0x0FU];
  out[4] = kHex[c.g & 0x0FU];
  out[5] = kHex[(c.b >> 4U) & 0x0FU];
  out[6] = kHex[c.b & 0x0FU];
  return out;
}

/* Fast branchless append of a non-negative 1-decimal fixed-point coordinate.
 */
inline void
append_coord_tenths (std::string &buf, float coord)
{
  int32_t tenths = static_cast<int32_t> (coord * 10.0F + 0.5F);
  if (tenths < 0)
    {
      tenths = 0;
    }
  uint32_t whole = static_cast<uint32_t> (tenths) / 10U;
  const uint32_t frac = static_cast<uint32_t> (tenths) % 10U;
  char digits[12];
  std::size_t len = 0U;
  do
    {
      digits[len++] = static_cast<char> ('0' + (whole % 10U));
      whole /= 10U;
    }
  while (whole > 0U);
  for (std::size_t i = len; i > 0U; --i)
    {
      buf.push_back (digits[i - 1U]);
    }
  buf.push_back ('.');
  buf.push_back (static_cast<char> ('0' + frac));
}

/* Escapes XML special characters for SVG text nodes.  */
void
append_xml_escaped (std::string &out, std::string_view text)
{
  for (const char ch : text)
    {
      switch (ch)
        {
        case '&':
          out += "&amp;";
          break;
        case '<':
          out += "&lt;";
          break;
        case '>':
          out += "&gt;";
          break;
        case '"':
          out += "&quot;";
          break;
        default:
          out.push_back (ch);
          break;
        }
    }
}

/* Extracts primary dataset points from snapshot, stream, or inline layer.  */
auto
resolve_points (const Spec &spec, const SnapshotData *snapshot,
                const StreamingBuffer *stream,
                std::vector<DataPoint2D> &scratch)
    -> std::span<const DataPoint2D>
{
  if (snapshot != nullptr && snapshot->point_count () > 0U)
    {
      return snapshot->points ();
    }
  if (stream != nullptr && stream->size () > 0U)
    {
      scratch.resize (stream->size ());
      (void)stream->copy_ordered (scratch);
      return std::span<const DataPoint2D> (scratch.data (), scratch.size ());
    }
  if (!spec.layers.empty ())
    {
      const auto &l = spec.layers.front ();
      const std::size_t n = std::min (l.inline_x.size (), l.inline_y.size ());
      scratch.resize (n);
      for (std::size_t i = 0U; i < n; ++i)
        {
          scratch[i] = DataPoint2D{ l.inline_x[i], l.inline_y[i] };
        }
      return std::span<const DataPoint2D> (scratch.data (), scratch.size ());
    }
  return {};
}

/* Maps a DashPattern to an SVG stroke-dasharray attribute fragment.  */
auto
dash_to_svg_attr (DashPattern pat) -> std::string_view
{
  switch (pat)
    {
    case DashPattern::Dashed:
      return " stroke-dasharray=\"8,4\"";
    case DashPattern::Dotted:
      return " stroke-dasharray=\"2,4\"";
    case DashPattern::DashDot:
      return " stroke-dasharray=\"8,4,2,4\"";
    case DashPattern::Solid:
    default:
      return "";
    }
}

/* Appends a 10-digit zero-padded PDF xref offset entry.  */
void
append_pdf_xref_entry (std::string &pdf, std::size_t offset)
{
  char digits[10];
  std::size_t val = offset;
  for (int i = 9; i >= 0; --i)
    {
      digits[i] = static_cast<char> ('0' + (val % 10U));
      val /= 10U;
    }
  pdf.append (digits, 10U);
  pdf += " 00000 n \n";
}

} /* namespace */

/* Generates an in-memory SVG 1.1 document from a Spec and dataset.  */
auto
render_svg_string (const Spec &spec, const SnapshotData *snapshot,
                   const StreamingBuffer *stream) noexcept
    -> Result<std::string>
{
  if (auto val = spec.validate (); !val.has_value ())
    {
      return std::unexpected (val.error ());
    }
  try
    {
      std::vector<DataPoint2D> scratch;
      const auto pts = resolve_points (spec, snapshot, stream, scratch);
      const std::size_t n_pts = pts.size ();

      const float margin_l = 60.0F;
      const float margin_r = spec.width > 90U
                                 ? static_cast<float> (spec.width - 30U)
                                 : static_cast<float> (spec.width);
      const float margin_t = 40.0F;
      const float margin_b = spec.height > 90U
                                 ? static_cast<float> (spec.height - 50U)
                                 : static_cast<float> (spec.height);
      const float plot_w = std::max (1.0F, margin_r - margin_l);
      const float plot_h = std::max (1.0F, margin_b - margin_t);

      const double dom_w = spec.x_scale.domain_max - spec.x_scale.domain_min;
      const double dom_h = spec.y_scale.domain_max - spec.y_scale.domain_min;
      const float scale_x
          = static_cast<float> (static_cast<double> (plot_w) / dom_w);
      const float offset_x
          = margin_l - static_cast<float> (spec.x_scale.domain_min) * scale_x;
      const float scale_y
          = static_cast<float> (-static_cast<double> (plot_h) / dom_h);
      const float offset_y
          = margin_b - static_cast<float> (spec.y_scale.domain_min) * scale_y;
      const float w_lim = static_cast<float> (spec.width);
      const float h_lim = static_cast<float> (spec.height);

      std::string svg;
      svg.reserve (1536U + n_pts * 15U);
      svg += "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"";
      svg += std::to_string (spec.width);
      svg += "\" height=\"";
      svg += std::to_string (spec.height);
      svg += "\" viewBox=\"0 0 ";
      svg += std::to_string (spec.width);
      svg += " ";
      svg += std::to_string (spec.height);
      svg += "\">\n<rect width=\"100%\" height=\"100%\" fill=\"";
      svg += color_to_svg_hex (spec.background);
      svg += "\"/>\n";

      if (spec.show_grid)
        {
          svg += "<g stroke=\"#e6e6e6\" stroke-width=\"1\">\n";
          for (int g = 1; g < 5; ++g)
            {
              const float gx
                  = margin_l + plot_w * (static_cast<float> (g) / 5.0F);
              const float gy
                  = margin_t + plot_h * (static_cast<float> (g) / 5.0F);
              svg += "<line x1=\"";
              append_coord_tenths (svg, gx);
              svg += "\" y1=\"";
              append_coord_tenths (svg, margin_t);
              svg += "\" x2=\"";
              append_coord_tenths (svg, gx);
              svg += "\" y2=\"";
              append_coord_tenths (svg, margin_b);
              svg += "\"/>\n<line x1=\"";
              append_coord_tenths (svg, margin_l);
              svg += "\" y1=\"";
              append_coord_tenths (svg, gy);
              svg += "\" x2=\"";
              append_coord_tenths (svg, margin_r);
              svg += "\" y2=\"";
              append_coord_tenths (svg, gy);
              svg += "\"/>\n";
            }
          svg += "</g>\n";
        }

      svg += "<rect x=\"";
      append_coord_tenths (svg, margin_l);
      svg += "\" y=\"";
      append_coord_tenths (svg, margin_t);
      svg += "\" width=\"";
      append_coord_tenths (svg, plot_w);
      svg += "\" height=\"";
      append_coord_tenths (svg, plot_h);
      svg += "\" fill=\"none\" stroke=\"#333333\" stroke-width=\"1\"/>\n";

      if (!spec.title.empty ())
        {
          svg += "<text x=\"";
          svg += std::to_string (spec.width / 2U);
          svg += "\" y=\"24\" text-anchor=\"middle\" font-family=\"sans-serif"
                 "\" font-size=\"14\">";
          append_xml_escaped (svg, spec.title);
          svg += "</text>\n";
        }
      if (!spec.x_label.empty ())
        {
          svg += "<text x=\"";
          svg += std::to_string (spec.width / 2U);
          svg += "\" y=\"";
          svg += std::to_string (spec.height > 12U ? spec.height - 12U
                                                   : spec.height);
          svg += "\" text-anchor=\"middle\" font-family=\"sans-serif\" "
                 "font-size=\"11\">";
          append_xml_escaped (svg, spec.x_label);
          svg += "</text>\n";
        }
      if (!spec.y_label.empty ())
        {
          svg += "<text x=\"16\" y=\"";
          svg += std::to_string (spec.height / 2U);
          svg += "\" text-anchor=\"middle\" font-family=\"sans-serif\" "
                 "font-size=\"11\">";
          append_xml_escaped (svg, spec.y_label);
          svg += "</text>\n";
        }

      const auto &layer0 = spec.layers.front ();
      const std::string stroke_hex0
          = color_to_svg_hex (layer0.aesthetics.stroke_color);

      if (n_pts >= 8192U)
        {
          const std::size_t hw = std::clamp<std::size_t> (
              std::thread::hardware_concurrency (), 2U, 12U);
          std::vector<std::string> chunks (hw);
          {
            std::vector<std::jthread> workers;
            workers.reserve (hw);
            const std::size_t per_worker = (n_pts + hw - 1U) / hw;
            const DataPoint2D *raw_pts = pts.data ();

            for (std::size_t t = 0U; t < hw; ++t)
              {
                const std::size_t start = std::min (n_pts, t * per_worker);
                const std::size_t end
                    = std::min (n_pts, (t + 1U) * per_worker);
                workers.emplace_back (
                    [&, t, start, end, raw_pts, w_lim, h_lim] () {
                      if (start >= end)
                        {
                          return;
                        }
                      std::string &cbuf = chunks[t];
                      cbuf.reserve ((end - start) * 14U + 96U);
                      cbuf += "<path fill=\"none\" stroke=\"";
                      cbuf += stroke_hex0;
                      cbuf += "\" stroke-width=\"2\" stroke-linecap=\"round\" "
                              "d=\"";
                      for (std::size_t i = start; i < end; ++i)
                        {
                          const float px = std::clamp (
                              raw_pts[i].x * scale_x + offset_x, 0.0F, w_lim);
                          const float py = std::clamp (
                              raw_pts[i].y * scale_y + offset_y, 0.0F, h_lim);
                          cbuf.push_back ('M');
                          append_coord_tenths (cbuf, px);
                          cbuf.push_back (' ');
                          append_coord_tenths (cbuf, py);
                          cbuf += "h0";
                        }
                      cbuf += "\"/>\n";
                    });
              }
          }
          for (const auto &c : chunks)
            {
              svg += c;
            }
        }
      else
        {
          for (std::size_t li = 0U; li < spec.layers.size (); ++li)
            {
              const auto &layer = spec.layers[li];
              const std::string f_hex
                  = color_to_svg_hex (layer.aesthetics.fill_color);
              const std::string s_hex
                  = color_to_svg_hex (layer.aesthetics.stroke_color);

              if ((layer.mark == MarkType::Area
                   || layer.mark == MarkType::ConfidenceBand
                   || layer.mark == MarkType::Violin)
                  && layer.inline_x.size () >= 2U
                  && layer.inline_y.size () >= layer.inline_x.size ())
                {
                  const std::size_t m = layer.inline_x.size ();
                  svg += "<polygon fill=\"";
                  svg += f_hex;
                  svg += "\" fill-opacity=\"0.35\" stroke=\"";
                  svg += s_hex;
                  svg += "\" stroke-width=\"1\" points=\"";
                  for (std::size_t i = 0U; i < m; ++i)
                    {
                      const float px = std::clamp (
                          layer.inline_x[i] * scale_x + offset_x, 0.0F, w_lim);
                      const float py = std::clamp (
                          layer.inline_y[i] * scale_y + offset_y, 0.0F, h_lim);
                      append_coord_tenths (svg, px);
                      svg.push_back (',');
                      append_coord_tenths (svg, py);
                      svg.push_back (' ');
                    }
                  for (std::size_t i = m; i > 0U; --i)
                    {
                      const std::size_t idx = i - 1U;
                      const float y_low
                          = idx < layer.inline_y2.size ()
                                ? layer.inline_y2[idx]
                                : static_cast<float> (spec.y_scale.domain_min);
                      const float px = std::clamp (
                          layer.inline_x[idx] * scale_x + offset_x, 0.0F,
                          w_lim);
                      const float py = std::clamp (y_low * scale_y + offset_y,
                                                   0.0F, h_lim);
                      append_coord_tenths (svg, px);
                      svg.push_back (',');
                      append_coord_tenths (svg, py);
                      if (i > 1U)
                        {
                          svg.push_back (' ');
                        }
                    }
                  svg += "\"/>\n";
                }
              else if (layer.mark == MarkType::Bar
                       || layer.mark == MarkType::Histogram
                       || layer.mark == MarkType::Box)
                {
                  const std::size_t m = std::min (layer.inline_x.size (),
                                                  layer.inline_y.size ());
                  const float half_w = std::max (
                      2.0F, plot_w
                                / static_cast<float> (
                                    std::max<std::size_t> (m * 3U, 6U)));
                  for (std::size_t i = 0U; i < m; ++i)
                    {
                      const float cx = std::clamp (
                          layer.inline_x[i] * scale_x + offset_x, 0.0F, w_lim);
                      const float y_top = std::clamp (
                          layer.inline_y[i] * scale_y + offset_y, 0.0F, h_lim);
                      const float y_bot
                          = i < layer.inline_y2.size ()
                                ? std::clamp (layer.inline_y2[i] * scale_y
                                                  + offset_y,
                                              0.0F, h_lim)
                                : margin_b;
                      const float ry = std::min (y_top, y_bot);
                      const float rh
                          = std::max (1.0F, std::abs (y_bot - y_top));
                      svg += "<rect x=\"";
                      append_coord_tenths (svg, std::max (0.0F, cx - half_w));
                      svg += "\" y=\"";
                      append_coord_tenths (svg, ry);
                      svg += "\" width=\"";
                      append_coord_tenths (svg, half_w * 2.0F);
                      svg += "\" height=\"";
                      append_coord_tenths (svg, rh);
                      svg += "\" fill=\"";
                      svg += f_hex;
                      svg += "\" stroke=\"";
                      svg += s_hex;
                      svg += "\" stroke-width=\"1\"/>\n";
                    }
                }
              else if (layer.mark == MarkType::Heatmap
                       && layer.matrix_rows > 0U && layer.matrix_cols > 0U)
                {
                  const float cell_w
                      = plot_w / static_cast<float> (layer.matrix_cols);
                  const float cell_h
                      = plot_h / static_cast<float> (layer.matrix_rows);
                  for (uint32_t r = 0U; r < layer.matrix_rows; ++r)
                    {
                      for (uint32_t c = 0U; c < layer.matrix_cols; ++c)
                        {
                          const std::size_t idx = static_cast<std::size_t> (r)
                                                      * layer.matrix_cols
                                                  + c;
                          const float v
                              = idx < layer.inline_y.size ()
                                    ? std::clamp ((layer.inline_y[idx] + 1.0F)
                                                      * 0.5F,
                                                  0.0F, 1.0F)
                                    : 0.5F;
                          const ColorRgba cell_col{
                            static_cast<uint8_t> (68.0F + v * 180.0F),
                            static_cast<uint8_t> (1.0F + v * 220.0F),
                            static_cast<uint8_t> (84.0F + (1.0F - v) * 140.0F),
                            255U
                          };
                          svg += "<rect x=\"";
                          append_coord_tenths (
                              svg, margin_l + static_cast<float> (c) * cell_w);
                          svg += "\" y=\"";
                          append_coord_tenths (
                              svg, margin_t + static_cast<float> (r) * cell_h);
                          svg += "\" width=\"";
                          append_coord_tenths (svg, cell_w);
                          svg += "\" height=\"";
                          append_coord_tenths (svg, cell_h);
                          svg += "\" fill=\"";
                          svg += color_to_svg_hex (cell_col);
                          svg += "\"/>\n";
                        }
                    }
                }
              else
                {
                  std::vector<DataPoint2D> layer_scratch;
                  std::span<const DataPoint2D> lpts = pts;
                  if (li > 0U || (snapshot == nullptr && stream == nullptr))
                    {
                      const std::size_t m = std::min (layer.inline_x.size (),
                                                      layer.inline_y.size ());
                      layer_scratch.resize (m);
                      for (std::size_t i = 0U; i < m; ++i)
                        {
                          layer_scratch[i] = DataPoint2D{ layer.inline_x[i],
                                                          layer.inline_y[i] };
                        }
                      lpts = layer_scratch;
                    }
                  if (!lpts.empty ())
                    {
                      svg += "<path fill=\"none\" stroke=\"";
                      svg += s_hex;
                      svg += "\" stroke-width=\"2\" stroke-linecap=\"round\"";
                      svg += dash_to_svg_attr (layer.aesthetics.dash_pattern);
                      svg += " d=\"";
                      const char cmd
                          = (layer.mark == MarkType::Line
                             || layer.mark == MarkType::Kde
                             || layer.mark == MarkType::Ecdf
                             || layer.mark == MarkType::RegressionLine
                             || layer.mark == MarkType::ErrorBar)
                                ? 'L'
                                : 'M';
                      for (std::size_t i = 0U; i < lpts.size (); ++i)
                        {
                          const float px = std::clamp (
                              lpts[i].x * scale_x + offset_x, 0.0F, w_lim);
                          const float py = std::clamp (
                              lpts[i].y * scale_y + offset_y, 0.0F, h_lim);
                          svg.push_back (i == 0U ? 'M' : cmd);
                          append_coord_tenths (svg, px);
                          svg.push_back (' ');
                          append_coord_tenths (svg, py);
                          if (cmd == 'M')
                            {
                              svg += "h0";
                            }
                        }
                      svg += "\"/>\n";
                    }
                }
            }
        }

      svg += "</svg>\n";
      return svg;
    }
  catch (...)
    {
      return std::unexpected (
          Error::export_err (ExportErrorCode::BufferOverflow, "",
                             "Out of memory while generating SVG document"));
    }
}

/* Generates an in-memory PDF 1.4 vector document from a Spec and dataset.  */
auto
render_pdf_string (const Spec &spec, const SnapshotData *snapshot,
                   const StreamingBuffer *stream) noexcept
    -> Result<std::string>
{
  if (auto val = spec.validate (); !val.has_value ())
    {
      return std::unexpected (val.error ());
    }
  try
    {
      std::vector<DataPoint2D> scratch;
      const auto pts = resolve_points (spec, snapshot, stream, scratch);
      const std::size_t limit = std::min<std::size_t> (pts.size (), 25000U);

      const float margin_l = 60.0F;
      const float margin_r = spec.width > 90U
                                 ? static_cast<float> (spec.width - 30U)
                                 : static_cast<float> (spec.width);
      const float margin_b = 50.0F;
      const float margin_t = spec.height > 90U
                                 ? static_cast<float> (spec.height - 40U)
                                 : static_cast<float> (spec.height);
      const float plot_w = std::max (1.0F, margin_r - margin_l);
      const float plot_h = std::max (1.0F, margin_t - margin_b);
      const double dom_w = spec.x_scale.domain_max - spec.x_scale.domain_min;
      const double dom_h = spec.y_scale.domain_max - spec.y_scale.domain_min;
      const float scale_x
          = static_cast<float> (static_cast<double> (plot_w) / dom_w);
      const float offset_x
          = margin_l - static_cast<float> (spec.x_scale.domain_min) * scale_x;
      const float scale_y
          = static_cast<float> (static_cast<double> (plot_h) / dom_h);
      const float offset_y
          = margin_b - static_cast<float> (spec.y_scale.domain_min) * scale_y;
      const float w_lim = static_cast<float> (spec.width);
      const float h_lim = static_cast<float> (spec.height);

      std::string stream_body;
      stream_body.reserve (256U + limit * 24U);
      stream_body += "0.20 0.20 0.20 RG 1.0 w\n";
      append_coord_tenths (stream_body, margin_l);
      stream_body.push_back (' ');
      append_coord_tenths (stream_body, margin_b);
      stream_body.push_back (' ');
      append_coord_tenths (stream_body, plot_w);
      stream_body.push_back (' ');
      append_coord_tenths (stream_body, plot_h);
      stream_body += " re S\n";

      stream_body += "0.12 0.46 0.70 RG 1.5 w\n";
      for (std::size_t i = 0U; i < limit; ++i)
        {
          const float px
              = std::clamp (pts[i].x * scale_x + offset_x, 0.0F, w_lim);
          const float py
              = std::clamp (pts[i].y * scale_y + offset_y, 0.0F, h_lim);
          append_coord_tenths (stream_body, px);
          stream_body.push_back (' ');
          append_coord_tenths (stream_body, py);
          stream_body += (i == 0U) ? " m\n" : " l\n";
        }
      if (limit > 0U)
        {
          stream_body += "S\n";
        }

      std::string pdf;
      pdf.reserve (768U + stream_body.size ());
      pdf += "%PDF-1.4\n";
      const std::size_t off1 = pdf.size ();
      pdf += "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n";
      const std::size_t off2 = pdf.size ();
      pdf += "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n";
      const std::size_t off3 = pdf.size ();
      pdf += "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 ";
      pdf += std::to_string (spec.width);
      pdf += " ";
      pdf += std::to_string (spec.height);
      pdf += "] /Contents 4 0 R >> endobj\n";
      const std::size_t off4 = pdf.size ();
      pdf += "4 0 obj << /Length ";
      pdf += std::to_string (stream_body.size ());
      pdf += " >>\nstream\n";
      pdf += stream_body;
      pdf += "endstream\nendobj\n";
      const std::size_t xref_off = pdf.size ();
      pdf += "xref\n0 5\n0000000000 65535 f \n";
      append_pdf_xref_entry (pdf, off1);
      append_pdf_xref_entry (pdf, off2);
      append_pdf_xref_entry (pdf, off3);
      append_pdf_xref_entry (pdf, off4);
      pdf += "trailer << /Size 5 /Root 1 0 R >>\nstartxref\n";
      pdf += std::to_string (xref_off);
      pdf += "\n%%EOF\n";
      return pdf;
    }
  catch (...)
    {
      return std::unexpected (
          Error::export_err (ExportErrorCode::BufferOverflow, "",
                             "Out of memory while generating PDF document"));
    }
}

/* Synchronously exports a plot to disk in SVG or PDF format.  */
auto
export_file_sync (const Spec &spec, std::string_view file_path,
                  ExportFormat format, const SnapshotData *snapshot,
                  const StreamingBuffer *stream) noexcept
    -> Result<ExportResult>
{
  if (file_path.empty ())
    {
      return std::unexpected (Error::export_err (
          ExportErrorCode::IoFailure, "",
          "Export destination file_path must not be empty"));
    }
  if (format != ExportFormat::Svg && format != ExportFormat::Pdf)
    {
      return std::unexpected (Error::export_err (
          ExportErrorCode::UnsupportedFormat, std::string (file_path),
          "Unsupported ExportFormat specified for plot export"));
    }

  const uint64_t t0 = monotonic_time_ns ();
  auto content_res = (format == ExportFormat::Svg)
                         ? render_svg_string (spec, snapshot, stream)
                         : render_pdf_string (spec, snapshot, stream);
  if (!content_res.has_value ())
    {
      return std::unexpected (content_res.error ());
    }

  const std::string path_str (file_path);
  FILE *fp = std::fopen (path_str.c_str (), "wb");
  if (fp == nullptr)
    {
      return std::unexpected (Error::export_err (
          ExportErrorCode::IoFailure, path_str,
          "Failed to open destination file for plot export"));
    }
  const std::size_t written
      = std::fwrite (content_res->data (), 1U, content_res->size (), fp);
  (void)std::fclose (fp);
  if (written != content_res->size ())
    {
      return std::unexpected (Error::export_err (
          ExportErrorCode::IoFailure, path_str,
          "Short write occurred while exporting plot to disk"));
    }

  std::size_t pts_count = 0U;
  if (snapshot != nullptr && snapshot->point_count () > 0U)
    {
      pts_count = snapshot->point_count ();
    }
  else if (stream != nullptr && stream->size () > 0U)
    {
      pts_count = stream->size ();
    }
  else if (!spec.layers.empty ())
    {
      pts_count = spec.layers.front ().inline_x.size ();
    }

  ExportResult res;
  res.format = format;
  res.file_path = path_str;
  res.bytes_written = written;
  res.elapsed_ns = monotonic_time_ns () - t0;
  res.points_exported = pts_count;
  return res;
}

/* Asynchronously exports a plot to disk without blocking the caller.  */
auto
export_file_async (Spec spec, std::string file_path, ExportFormat format,
                   SnapshotData snapshot, StreamingBuffer stream,
                   ExportCallback callback) noexcept
    -> std::future<Result<ExportResult>>
{
  return std::async (
      std::launch::async,
      [s = std::move (spec), p = std::move (file_path), format,
       snap = std::move (snapshot), strm = std::move (stream),
       cb = std::move (callback)] () -> Result<ExportResult> {
        const SnapshotData *snap_ptr
            = snap.point_count () > 0U ? &snap : nullptr;
        const StreamingBuffer *strm_ptr = strm.size () > 0U ? &strm : nullptr;
        auto res = export_file_sync (s, p, format, snap_ptr, strm_ptr);
        if (cb)
          {
            cb (res);
          }
        return res;
      });
}

} /* namespace darshana::inline v1 */
