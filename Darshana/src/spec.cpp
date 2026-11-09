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

#include <darshana/spec.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

namespace darshana::inline v1
{

namespace
{

/* Appends a JSON-escaped string literal with surrounding quotes.  */
void
append_json_str (std::string &out, std::string_view s)
{
  out.push_back ('"');
  for (const char ch : s)
    {
      if (ch == '"')
        {
          out += "\\\"";
        }
      else if (ch == '\\')
        {
          out += "\\\\";
        }
      else if (ch == '\n')
        {
          out += "\\n";
        }
      else if (ch == '\r')
        {
          out += "\\r";
        }
      else if (ch == '\t')
        {
          out += "\\t";
        }
      else
        {
          out.push_back (ch);
        }
    }
  out.push_back ('"');
}

/* Appends a lossless floating-point representation using std::to_chars.  */
template <typename FloatT>
void
append_json_float (std::string &out, FloatT val)
{
  char buf[64]{};
  auto [ptr, ec] = std::to_chars (buf, buf + sizeof (buf), val);
  if (ec == std::errc{})
    {
      out.append (buf, static_cast<std::size_t> (ptr - buf));
    }
  else
    {
      out += "0";
    }
}

/* Appends a ColorRgba as a 4-element JSON integer array.  */
void
append_json_color (std::string &out, const ColorRgba &c)
{
  out.push_back ('[');
  out += std::to_string (c.r);
  out.push_back (',');
  out += std::to_string (c.g);
  out.push_back (',');
  out += std::to_string (c.b);
  out.push_back (',');
  out += std::to_string (c.a);
  out.push_back (']');
}

/* Appends a vector of floats as a JSON array.  */
void
append_json_fvec (std::string &out, const std::vector<float> &v)
{
  out.push_back ('[');
  for (std::size_t i = 0U; i < v.size (); ++i)
    {
      if (i > 0U)
        {
          out.push_back (',');
        }
      append_json_float (out, v[i]);
    }
  out.push_back (']');
}

/* Appends a ScaleSpec as a JSON object.  */
void
append_json_scale (std::string &out, const ScaleSpec &s)
{
  out += "{\"type\":";
  out += std::to_string (static_cast<uint32_t> (s.type));
  out += ",\"domain_min\":";
  append_json_float (out, s.domain_min);
  out += ",\"domain_max\":";
  append_json_float (out, s.domain_max);
  out += ",\"range_min\":";
  append_json_float (out, s.range_min);
  out += ",\"range_max\":";
  append_json_float (out, s.range_max);
  out += ",\"auto_domain\":";
  out += (s.auto_domain ? "true" : "false");
  out += ",\"clamp\":";
  out += (s.clamp ? "true" : "false");
  out += ",\"categories\":[";
  for (std::size_t i = 0U; i < s.categories.size (); ++i)
    {
      if (i > 0U)
        {
          out.push_back (',');
        }
      append_json_str (out, s.categories[i]);
    }
  out += "]}";
}

/* Recursive descent JSON reader for Spec deserialization.  */
struct JsonReader
{
  std::string_view src;
  std::size_t pos = 0U;

  void
  skip_ws () noexcept
  {
    while (pos < src.size ())
      {
        const char ch = src[pos];
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
          {
            ++pos;
          }
        else
          {
            break;
          }
      }
  }

  auto
  consume (char expected) noexcept -> bool
  {
    skip_ws ();
    if (pos < src.size () && src[pos] == expected)
      {
        ++pos;
        return true;
      }
    return false;
  }

  auto
  read_string (std::string &out) -> bool
  {
    skip_ws ();
    if (pos >= src.size () || src[pos] != '"')
      {
        return false;
      }
    ++pos;
    out.clear ();
    while (pos < src.size ())
      {
        const char ch = src[pos++];
        if (ch == '"')
          {
            return true;
          }
        if (ch == '\\')
          {
            if (pos >= src.size ())
              {
                return false;
              }
            const char esc = src[pos++];
            switch (esc)
              {
              case '"':
                out.push_back ('"');
                break;
              case '\\':
                out.push_back ('\\');
                break;
              case 'n':
                out.push_back ('\n');
                break;
              case 'r':
                out.push_back ('\r');
                break;
              case 't':
                out.push_back ('\t');
                break;
              default:
                out.push_back (esc);
                break;
              }
          }
        else
          {
            out.push_back (ch);
          }
      }
    return false;
  }

  auto
  read_bool (bool &out) noexcept -> bool
  {
    skip_ws ();
    if (src.substr (pos).starts_with ("true"))
      {
        pos += 4U;
        out = true;
        return true;
      }
    if (src.substr (pos).starts_with ("false"))
      {
        pos += 5U;
        out = false;
        return true;
      }
    return false;
  }

  template <typename IntT>
  auto
  read_uint (IntT &out) noexcept -> bool
  {
    skip_ws ();
    if (pos >= src.size ())
      {
        return false;
      }
    uint64_t val = 0U;
    auto [ptr, ec]
        = std::from_chars (src.data () + pos, src.data () + src.size (), val);
    if (ec != std::errc{})
      {
        return false;
      }
    pos = static_cast<std::size_t> (ptr - src.data ());
    out = static_cast<IntT> (val);
    return true;
  }

  template <typename FloatT>
  auto
  read_float (FloatT &out) noexcept -> bool
  {
    skip_ws ();
    if (pos >= src.size ())
      {
        return false;
      }
    auto [ptr, ec]
        = std::from_chars (src.data () + pos, src.data () + src.size (), out);
    if (ec != std::errc{})
      {
        return false;
      }
    pos = static_cast<std::size_t> (ptr - src.data ());
    return true;
  }

  auto
  read_color (ColorRgba &c) noexcept -> bool
  {
    return consume ('[') && read_uint (c.r) && consume (',') && read_uint (c.g)
           && consume (',') && read_uint (c.b) && consume (',')
           && read_uint (c.a) && consume (']');
  }

  auto
  read_fvec (std::vector<float> &v) -> bool
  {
    v.clear ();
    if (!consume ('['))
      {
        return false;
      }
    if (consume (']'))
      {
        return true;
      }
    while (true)
      {
        float val = 0.0F;
        if (!read_float (val))
          {
            return false;
          }
        v.push_back (val);
        if (consume (']'))
          {
            return true;
          }
        if (!consume (','))
          {
            return false;
          }
      }
  }

  auto
  read_scale (ScaleSpec &s) -> bool
  {
    if (!consume ('{'))
      {
        return false;
      }
    if (consume ('}'))
      {
        return true;
      }
    while (true)
      {
        std::string key;
        if (!read_string (key) || !consume (':'))
          {
            return false;
          }
        if (key == "type")
          {
            uint32_t t = 0U;
            if (!read_uint (t) || t > 2U)
              {
                return false;
              }
            s.type = static_cast<ScaleType> (t);
          }
        else if (key == "domain_min")
          {
            if (!read_float (s.domain_min))
              {
                return false;
              }
          }
        else if (key == "domain_max")
          {
            if (!read_float (s.domain_max))
              {
                return false;
              }
          }
        else if (key == "range_min")
          {
            if (!read_float (s.range_min))
              {
                return false;
              }
          }
        else if (key == "range_max")
          {
            if (!read_float (s.range_max))
              {
                return false;
              }
          }
        else if (key == "auto_domain")
          {
            if (!read_bool (s.auto_domain))
              {
                return false;
              }
          }
        else if (key == "clamp")
          {
            if (!read_bool (s.clamp))
              {
                return false;
              }
          }
        else if (key == "categories")
          {
            s.categories.clear ();
            if (!consume ('['))
              {
                return false;
              }
            if (!consume (']'))
              {
                while (true)
                  {
                    std::string cat;
                    if (!read_string (cat))
                      {
                        return false;
                      }
                    s.categories.push_back (std::move (cat));
                    if (consume (']'))
                      {
                        break;
                      }
                    if (!consume (','))
                      {
                        return false;
                      }
                  }
              }
          }
        else
          {
            return false;
          }
        if (consume ('}'))
          {
            return true;
          }
        if (!consume (','))
          {
            return false;
          }
      }
  }

  auto
  read_overlay (OverlaySpec &ov) -> bool
  {
    if (!consume ('{'))
      {
        return false;
      }
    if (consume ('}'))
      {
        return true;
      }
    while (true)
      {
        std::string key;
        if (!read_string (key) || !consume (':'))
          {
            return false;
          }
        if (key == "crosshair_enabled")
          {
            if (!read_bool (ov.crosshair_enabled))
              {
                return false;
              }
          }
        else if (key == "crosshair_x")
          {
            if (!read_float (ov.crosshair_x))
              {
                return false;
              }
          }
        else if (key == "crosshair_y")
          {
            if (!read_float (ov.crosshair_y))
              {
                return false;
              }
          }
        else if (key == "tooltip_enabled")
          {
            if (!read_bool (ov.tooltip_enabled))
              {
                return false;
              }
          }
        else if (key == "tooltip_x")
          {
            if (!read_float (ov.tooltip_x))
              {
                return false;
              }
          }
        else if (key == "tooltip_y")
          {
            if (!read_float (ov.tooltip_y))
              {
                return false;
              }
          }
        else if (key == "tooltip_text")
          {
            if (!read_string (ov.tooltip_text))
              {
                return false;
              }
          }
        else if (key == "brush_enabled")
          {
            if (!read_bool (ov.brush_enabled))
              {
                return false;
              }
          }
        else if (key == "brush_x0")
          {
            if (!read_float (ov.brush_x0))
              {
                return false;
              }
          }
        else if (key == "brush_y0")
          {
            if (!read_float (ov.brush_y0))
              {
                return false;
              }
          }
        else if (key == "brush_x1")
          {
            if (!read_float (ov.brush_x1))
              {
                return false;
              }
          }
        else if (key == "brush_y1")
          {
            if (!read_float (ov.brush_y1))
              {
                return false;
              }
          }
        else
          {
            return false;
          }
        if (consume ('}'))
          {
            return true;
          }
        if (!consume (','))
          {
            return false;
          }
      }
  }

  auto
  read_layer (LayerSpec &layer) -> bool
  {
    if (!consume ('{'))
      {
        return false;
      }
    if (consume ('}'))
      {
        return true;
      }
    while (true)
      {
        std::string key;
        if (!read_string (key) || !consume (':'))
          {
            return false;
          }
        if (key == "mark")
          {
            uint32_t m = 0U;
            if (!read_uint (m) || m > 12U)
              {
                return false;
              }
            layer.mark = static_cast<MarkType> (m);
          }
        else if (key == "x_field")
          {
            if (!read_string (layer.encoding.x_field))
              {
                return false;
              }
          }
        else if (key == "y_field")
          {
            if (!read_string (layer.encoding.y_field))
              {
                return false;
              }
          }
        else if (key == "y2_field")
          {
            if (!read_string (layer.encoding.y2_field))
              {
                return false;
              }
          }
        else if (key == "color_field")
          {
            if (!read_string (layer.encoding.color_field))
              {
                return false;
              }
          }
        else if (key == "size_field")
          {
            if (!read_string (layer.encoding.size_field))
              {
                return false;
              }
          }
        else if (key == "label")
          {
            if (!read_string (layer.encoding.label))
              {
                return false;
              }
          }
        else if (key == "stroke_color")
          {
            if (!read_color (layer.aesthetics.stroke_color))
              {
                return false;
              }
          }
        else if (key == "fill_color")
          {
            if (!read_color (layer.aesthetics.fill_color))
              {
                return false;
              }
          }
        else if (key == "stroke_width")
          {
            if (!read_float (layer.aesthetics.stroke_width))
              {
                return false;
              }
          }
        else if (key == "marker_size")
          {
            if (!read_float (layer.aesthetics.marker_size))
              {
                return false;
              }
          }
        else if (key == "marker_shape")
          {
            uint32_t s = 0U;
            if (!read_uint (s) || s > 4U)
              {
                return false;
              }
            layer.aesthetics.marker_shape = static_cast<MarkerShape> (s);
          }
        else if (key == "dash_pattern")
          {
            uint32_t d = 0U;
            if (!read_uint (d) || d > 3U)
              {
                return false;
              }
            layer.aesthetics.dash_pattern = static_cast<DashPattern> (d);
          }
        else if (key == "opacity")
          {
            if (!read_float (layer.aesthetics.opacity))
              {
                return false;
              }
          }
        else if (key == "sdf_aa_width")
          {
            if (!read_float (layer.aesthetics.sdf_aa_width))
              {
                return false;
              }
          }
        else if (key == "inline_x")
          {
            if (!read_fvec (layer.inline_x))
              {
                return false;
              }
          }
        else if (key == "inline_y")
          {
            if (!read_fvec (layer.inline_y))
              {
                return false;
              }
          }
        else if (key == "inline_y2")
          {
            if (!read_fvec (layer.inline_y2))
              {
                return false;
              }
          }
        else if (key == "inline_q1")
          {
            if (!read_fvec (layer.inline_q1))
              {
                return false;
              }
          }
        else if (key == "inline_q3")
          {
            if (!read_fvec (layer.inline_q3))
              {
                return false;
              }
          }
        else if (key == "matrix_rows")
          {
            if (!read_uint (layer.matrix_rows))
              {
                return false;
              }
          }
        else if (key == "matrix_cols")
          {
            if (!read_uint (layer.matrix_cols))
              {
                return false;
              }
          }
        else if (key == "matrix_values")
          {
            if (!read_fvec (layer.matrix_values))
              {
                return false;
              }
          }
        else
          {
            return false;
          }
        if (consume ('}'))
          {
            return true;
          }
        if (!consume (','))
          {
            return false;
          }
      }
  }
};

} /* namespace */

/* Maps a continuous domain value to the target range.  */
auto
ScaleSpec::map (double value) const noexcept -> Result<double>
{
  if (!std::isfinite (value) || !std::isfinite (domain_min)
      || !std::isfinite (domain_max) || !std::isfinite (range_min)
      || !std::isfinite (range_max))
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidScaleDomain, "scale",
          "ScaleSpec::map requires finite domain, range, and input values"));
    }
  double t = 0.0;
  if (type == ScaleType::Linear)
    {
      if (domain_min == domain_max)
        {
          return std::unexpected (Error::data (
              DataErrorCode::InvalidScaleDomain, "domain",
              "Linear scale requires distinct domain_min and domain_max"));
        }
      t = (value - domain_min) / (domain_max - domain_min);
    }
  else if (type == ScaleType::Log)
    {
      if (domain_min <= 0.0 || domain_max <= 0.0 || domain_min == domain_max
          || value <= 0.0)
        {
          return std::unexpected (Error::data (
              DataErrorCode::InvalidScaleDomain, "domain",
              "Logarithmic scale requires strictly positive domain and "
              "value"));
        }
      const double log_min = std::log10 (domain_min);
      const double log_max = std::log10 (domain_max);
      t = (std::log10 (value) - log_min) / (log_max - log_min);
    }
  else
    {
      if (!categories.empty ())
        {
          const auto n = static_cast<double> (categories.size ());
          t = (value + 0.5) / n;
        }
      else if (domain_max != domain_min)
        {
          t = (value - domain_min) / (domain_max - domain_min);
        }
      else
        {
          return std::unexpected (Error::data (
              DataErrorCode::InvalidScaleDomain, "categories",
              "Categorical scale requires non-empty categories or valid "
              "domain"));
        }
    }

  if (clamp)
    {
      t = std::clamp (t, 0.0, 1.0);
    }
  return range_min + t * (range_max - range_min);
}

/* Maps a discrete category label to a normalized range coordinate.  */
auto
ScaleSpec::map_category (std::string_view category) const noexcept
    -> Result<double>
{
  if (categories.empty ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidScaleDomain, "categories",
          "ScaleSpec::map_category requires a non-empty categories list"));
    }
  for (std::size_t i = 0U; i < categories.size (); ++i)
    {
      if (categories[i] == category)
        {
          const double t = (static_cast<double> (i) + 0.5)
                           / static_cast<double> (categories.size ());
          return range_min + t * (range_max - range_min);
        }
    }
  return std::unexpected (
      Error::data (DataErrorCode::InvalidSpec, std::string (category),
                   "Category label was not found in ScaleSpec categories"));
}

/* Validates structural and numerical invariants of the Spec.  */
auto
Spec::validate () const noexcept -> Result<void>
{
  if (width == 0U || height == 0U || width > 16384U || height > 16384U)
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "width/height",
                       "Spec viewport dimensions must lie in [1, 16384]"));
    }
  if (!std::isfinite (dpi) || dpi <= 0.0F)
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "dpi",
                       "Spec dpi must be a positive finite number"));
    }
  if (layers.empty ())
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "layers",
                       "Spec must contain at least one visual mark layer"));
    }
  if (!std::isfinite (x_scale.domain_min)
      || !std::isfinite (x_scale.domain_max)
      || x_scale.domain_min == x_scale.domain_max)
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidScaleDomain, "x_scale",
          "Spec x_scale must have finite distinct domain bounds"));
    }
  if (x_scale.type == ScaleType::Log
      && (x_scale.domain_min <= 0.0 || x_scale.domain_max <= 0.0))
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidScaleDomain, "x_scale",
          "Logarithmic x_scale requires strictly positive domain bounds"));
    }
  if (!std::isfinite (y_scale.domain_min)
      || !std::isfinite (y_scale.domain_max)
      || y_scale.domain_min == y_scale.domain_max)
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidScaleDomain, "y_scale",
          "Spec y_scale must have finite distinct domain bounds"));
    }
  if (y_scale.type == ScaleType::Log
      && (y_scale.domain_min <= 0.0 || y_scale.domain_max <= 0.0))
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidScaleDomain, "y_scale",
          "Logarithmic y_scale requires strictly positive domain bounds"));
    }
  for (const auto &layer : layers)
    {
      if (!std::isfinite (layer.aesthetics.stroke_width)
          || layer.aesthetics.stroke_width < 0.0F
          || !std::isfinite (layer.aesthetics.marker_size)
          || layer.aesthetics.marker_size <= 0.0F
          || !std::isfinite (layer.aesthetics.opacity)
          || layer.aesthetics.opacity < 0.0F
          || layer.aesthetics.opacity > 1.0F)
        {
          return std::unexpected (Error::data (
              DataErrorCode::InvalidSpec, "aesthetics",
              "Layer aesthetics contain out-of-range stroke, marker, or "
              "opacity values"));
        }
    }
  return {};
}

/* Serializes this Spec into a deterministic JSON string.  */
auto
Spec::to_json () const noexcept -> Result<std::string>
{
  if (auto res = validate (); !res.has_value ())
    {
      return std::unexpected (res.error ());
    }
  try
    {
      std::string out;
      out.reserve (1024U);
      out += "{\"title\":";
      append_json_str (out, title);
      out += ",\"subtitle\":";
      append_json_str (out, subtitle);
      out += ",\"x_label\":";
      append_json_str (out, x_label);
      out += ",\"y_label\":";
      append_json_str (out, y_label);
      out += ",\"width\":";
      out += std::to_string (width);
      out += ",\"height\":";
      out += std::to_string (height);
      out += ",\"dpi\":";
      append_json_float (out, dpi);
      out += ",\"background\":";
      append_json_color (out, background);
      out += ",\"show_grid\":";
      out += (show_grid ? "true" : "false");
      out += ",\"show_legend\":";
      out += (show_legend ? "true" : "false");
      out += ",\"x_scale\":";
      append_json_scale (out, x_scale);
      out += ",\"y_scale\":";
      append_json_scale (out, y_scale);
      out += ",\"overlay\":{\"crosshair_enabled\":";
      out += (overlay.crosshair_enabled ? "true" : "false");
      out += ",\"crosshair_x\":";
      append_json_float (out, overlay.crosshair_x);
      out += ",\"crosshair_y\":";
      append_json_float (out, overlay.crosshair_y);
      out += ",\"tooltip_enabled\":";
      out += (overlay.tooltip_enabled ? "true" : "false");
      out += ",\"tooltip_x\":";
      append_json_float (out, overlay.tooltip_x);
      out += ",\"tooltip_y\":";
      append_json_float (out, overlay.tooltip_y);
      out += ",\"tooltip_text\":";
      append_json_str (out, overlay.tooltip_text);
      out += ",\"brush_enabled\":";
      out += (overlay.brush_enabled ? "true" : "false");
      out += ",\"brush_x0\":";
      append_json_float (out, overlay.brush_x0);
      out += ",\"brush_y0\":";
      append_json_float (out, overlay.brush_y0);
      out += ",\"brush_x1\":";
      append_json_float (out, overlay.brush_x1);
      out += ",\"brush_y1\":";
      append_json_float (out, overlay.brush_y1);
      out += "},\"layers\":[";
      for (std::size_t i = 0U; i < layers.size (); ++i)
        {
          if (i > 0U)
            {
              out.push_back (',');
            }
          const auto &l = layers[i];
          out += "{\"mark\":";
          out += std::to_string (static_cast<uint32_t> (l.mark));
          out += ",\"x_field\":";
          append_json_str (out, l.encoding.x_field);
          out += ",\"y_field\":";
          append_json_str (out, l.encoding.y_field);
          out += ",\"y2_field\":";
          append_json_str (out, l.encoding.y2_field);
          out += ",\"color_field\":";
          append_json_str (out, l.encoding.color_field);
          out += ",\"size_field\":";
          append_json_str (out, l.encoding.size_field);
          out += ",\"label\":";
          append_json_str (out, l.encoding.label);
          out += ",\"stroke_color\":";
          append_json_color (out, l.aesthetics.stroke_color);
          out += ",\"fill_color\":";
          append_json_color (out, l.aesthetics.fill_color);
          out += ",\"stroke_width\":";
          append_json_float (out, l.aesthetics.stroke_width);
          out += ",\"marker_size\":";
          append_json_float (out, l.aesthetics.marker_size);
          out += ",\"marker_shape\":";
          out += std::to_string (
              static_cast<uint32_t> (l.aesthetics.marker_shape));
          out += ",\"dash_pattern\":";
          out += std::to_string (
              static_cast<uint32_t> (l.aesthetics.dash_pattern));
          out += ",\"opacity\":";
          append_json_float (out, l.aesthetics.opacity);
          out += ",\"sdf_aa_width\":";
          append_json_float (out, l.aesthetics.sdf_aa_width);
          out += ",\"inline_x\":";
          append_json_fvec (out, l.inline_x);
          out += ",\"inline_y\":";
          append_json_fvec (out, l.inline_y);
          out += ",\"inline_y2\":";
          append_json_fvec (out, l.inline_y2);
          out += ",\"inline_q1\":";
          append_json_fvec (out, l.inline_q1);
          out += ",\"inline_q3\":";
          append_json_fvec (out, l.inline_q3);
          out += ",\"matrix_rows\":";
          out += std::to_string (l.matrix_rows);
          out += ",\"matrix_cols\":";
          out += std::to_string (l.matrix_cols);
          out += ",\"matrix_values\":";
          append_json_fvec (out, l.matrix_values);
          out.push_back ('}');
        }
      out += "]}";
      return out;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory while serializing Spec to JSON"));
    }
}

/* Deserializes a Spec from a JSON string representation.  */
auto
Spec::from_json (std::string_view json) noexcept -> Result<Spec>
{
  if (json.empty ())
    {
      return std::unexpected (
          Error::data (DataErrorCode::JsonParseError, "json",
                       "Cannot deserialize Spec from an empty JSON string"));
    }
  try
    {
      JsonReader r{ json, 0U };
      Spec spec;
      if (!r.consume ('{'))
        {
          return std::unexpected (
              Error::data (DataErrorCode::JsonParseError, "json",
                           "Expected opening brace at root of Spec JSON"));
        }
      if (!r.consume ('}'))
        {
          while (true)
            {
              std::string key;
              if (!r.read_string (key) || !r.consume (':'))
                {
                  return std::unexpected (
                      Error::data (DataErrorCode::JsonParseError, "key",
                                   "Malformed key-value pair in Spec JSON"));
                }
              bool ok = false;
              if (key == "title")
                {
                  ok = r.read_string (spec.title);
                }
              else if (key == "subtitle")
                {
                  ok = r.read_string (spec.subtitle);
                }
              else if (key == "x_label")
                {
                  ok = r.read_string (spec.x_label);
                }
              else if (key == "y_label")
                {
                  ok = r.read_string (spec.y_label);
                }
              else if (key == "width")
                {
                  ok = r.read_uint (spec.width);
                }
              else if (key == "height")
                {
                  ok = r.read_uint (spec.height);
                }
              else if (key == "dpi")
                {
                  ok = r.read_float (spec.dpi);
                }
              else if (key == "background")
                {
                  ok = r.read_color (spec.background);
                }
              else if (key == "show_grid")
                {
                  ok = r.read_bool (spec.show_grid);
                }
              else if (key == "show_legend")
                {
                  ok = r.read_bool (spec.show_legend);
                }
              else if (key == "x_scale")
                {
                  ok = r.read_scale (spec.x_scale);
                }
              else if (key == "y_scale")
                {
                  ok = r.read_scale (spec.y_scale);
                }
              else if (key == "overlay")
                {
                  ok = r.read_overlay (spec.overlay);
                }
              else if (key == "layers")
                {
                  spec.layers.clear ();
                  if (r.consume ('['))
                    {
                      if (r.consume (']'))
                        {
                          ok = true;
                        }
                      else
                        {
                          while (true)
                            {
                              LayerSpec l;
                              if (!r.read_layer (l))
                                {
                                  ok = false;
                                  break;
                                }
                              spec.layers.push_back (std::move (l));
                              if (r.consume (']'))
                                {
                                  ok = true;
                                  break;
                                }
                              if (!r.consume (','))
                                {
                                  ok = false;
                                  break;
                                }
                            }
                        }
                    }
                }
              if (!ok)
                {
                  return std::unexpected (Error::data (
                      DataErrorCode::JsonParseError, key,
                      "Failed to parse value for field in Spec JSON"));
                }
              if (r.consume ('}'))
                {
                  break;
                }
              if (!r.consume (','))
                {
                  return std::unexpected (Error::data (
                      DataErrorCode::JsonParseError, key,
                      "Expected comma or closing brace in Spec JSON"));
                }
            }
        }
      if (auto val_res = spec.validate (); !val_res.has_value ())
        {
          return std::unexpected (val_res.error ());
        }
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::data (DataErrorCode::JsonParseError, "json",
                       "Unexpected exception while parsing Spec JSON"));
    }
}

} /* namespace darshana::inline v1 */
