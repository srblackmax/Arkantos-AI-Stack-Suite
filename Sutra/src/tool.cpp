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

#include <sutra/tool.hpp>

#include <cctype>
#include <charconv>
#include <memory>
#include <mutex>

namespace sutra::inline v1
{

namespace
{

/* Skips ASCII whitespace characters starting at pos.  */
[[nodiscard]] constexpr auto
skip_ws (std::string_view sv, std::size_t pos) noexcept -> std::size_t
{
  while (pos < sv.size ()
         && (sv[pos] == ' ' || sv[pos] == '\t' || sv[pos] == '\n'
             || sv[pos] == '\r'))
    {
      ++pos;
    }
  return pos;
}

/* Checks whether a JSON object string has balanced top-level braces and
 * quotes.  */
[[nodiscard]] auto
is_well_formed_json_object (std::string_view sv) noexcept -> bool
{
  const std::size_t start = skip_ws (sv, 0U);
  if (start >= sv.size () || sv[start] != '{')
    {
      return false;
    }

  int brace_depth = 0;
  int bracket_depth = 0;
  bool in_quotes = false;
  bool escaped = false;
  std::size_t last_non_ws = start;

  for (std::size_t i = start; i < sv.size (); ++i)
    {
      const char ch = sv[i];
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
        {
          last_non_ws = i;
        }
      if (escaped)
        {
          escaped = false;
          continue;
        }
      if (ch == '\\' && in_quotes)
        {
          escaped = true;
          continue;
        }
      if (ch == '"')
        {
          in_quotes = !in_quotes;
          continue;
        }
      if (!in_quotes)
        {
          if (ch == '{')
            {
              ++brace_depth;
            }
          else if (ch == '}')
            {
              --brace_depth;
              if (brace_depth < 0)
                {
                  return false;
                }
            }
          else if (ch == '[')
            {
              ++bracket_depth;
            }
          else if (ch == ']')
            {
              --bracket_depth;
              if (bracket_depth < 0)
                {
                  return false;
                }
            }
        }
    }

  return !in_quotes && !escaped && (brace_depth == 0) && (bracket_depth == 0)
         && (sv[last_non_ws] == '}');
}

/* Finds a top-level key in a JSON object and returns the value start index. */
[[nodiscard]] auto
find_top_level_json_key (std::string_view obj,
                         std::string_view key) noexcept -> std::size_t
{
  const std::size_t start = skip_ws (obj, 0U);
  if (start >= obj.size () || obj[start] != '{')
    {
      return std::string_view::npos;
    }

  int brace_depth = 0;
  int bracket_depth = 0;
  bool in_quotes = false;
  bool escaped = false;

  for (std::size_t i = start; i < obj.size (); ++i)
    {
      const char ch = obj[i];
      if (escaped)
        {
          escaped = false;
          continue;
        }
      if (ch == '\\' && in_quotes)
        {
          escaped = true;
          continue;
        }
      if (ch == '"')
        {
          if (!in_quotes && brace_depth == 1 && bracket_depth == 0)
            {
              const std::size_t key_start = i + 1U;
              std::size_t key_end = key_start;
              while (key_end < obj.size ())
                {
                  if (obj[key_end] == '\\')
                    {
                      key_end += 2U;
                      continue;
                    }
                  if (obj[key_end] == '"')
                    {
                      break;
                    }
                  ++key_end;
                }
              if (key_end >= obj.size ())
                {
                  return std::string_view::npos;
                }
              const std::string_view candidate
                  = obj.substr (key_start, key_end - key_start);
              const std::size_t after_quote = skip_ws (obj, key_end + 1U);
              if (after_quote < obj.size () && obj[after_quote] == ':')
                {
                  if (candidate == key)
                    {
                      return skip_ws (obj, after_quote + 1U);
                    }
                }
              i = key_end;
              continue;
            }
          in_quotes = !in_quotes;
          continue;
        }
      if (!in_quotes)
        {
          if (ch == '{')
            {
              ++brace_depth;
            }
          else if (ch == '}')
            {
              --brace_depth;
            }
          else if (ch == '[')
            {
              ++bracket_depth;
            }
          else if (ch == ']')
            {
              --bracket_depth;
            }
        }
    }

  return std::string_view::npos;
}

/* Parses a JSON type string into JsonType.  */
[[nodiscard]] auto
string_to_json_type (std::string_view str) noexcept -> Result<JsonType>
{
  if (str == "string")
    {
      return JsonType::String;
    }
  if (str == "integer")
    {
      return JsonType::Integer;
    }
  if (str == "number")
    {
      return JsonType::Number;
    }
  if (str == "boolean")
    {
      return JsonType::Boolean;
    }
  if (str == "array")
    {
      return JsonType::Array;
    }
  if (str == "object")
    {
      return JsonType::Object;
    }
  return std::unexpected (AgentError::tool (
      ToolErrorCode::SchemaViolation, "schema_validator",
      "Unrecognized JSON Schema type keyword in tool schema"));
}

} /* namespace */

/* Extracts and validates a field token from a JSON object.  */
auto
extract_json_field_token (
    std::string_view json_object, std::string_view field_name,
    JsonType expected_type) noexcept -> Result<std::string>
{
  if (!is_well_formed_json_object (json_object))
    {
      return std::unexpected (AgentError::tool (
          ToolErrorCode::InvalidInput, std::string (field_name),
          "Malformed JSON object provided to tool invocation"));
    }

  const std::size_t val_pos
      = find_top_level_json_key (json_object, field_name);
  if (val_pos == std::string_view::npos || val_pos >= json_object.size ())
    {
      std::string msg = "Missing required tool argument: ";
      msg.append (field_name);
      return std::unexpected (AgentError::tool (ToolErrorCode::SchemaViolation,
                                                std::string (field_name),
                                                std::move (msg)));
    }

  const char first_ch = json_object[val_pos];
  if (expected_type == JsonType::String)
    {
      if (first_ch != '"')
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, std::string (field_name),
              "Expected JSON string value for tool parameter"));
        }
      std::string out;
      std::size_t i = val_pos + 1U;
      while (i < json_object.size ())
        {
          if (json_object[i] == '\\')
            {
              if (i + 1U >= json_object.size ())
                {
                  return std::unexpected (AgentError::tool (
                      ToolErrorCode::InvalidInput, std::string (field_name),
                      "Unterminated escape sequence in JSON string"));
                }
              const char esc = json_object[i + 1U];
              if (esc == 'n')
                {
                  out.push_back ('\n');
                }
              else if (esc == 'r')
                {
                  out.push_back ('\r');
                }
              else if (esc == 't')
                {
                  out.push_back ('\t');
                }
              else
                {
                  out.push_back (esc);
                }
              i += 2U;
              continue;
            }
          if (json_object[i] == '"')
            {
              return out;
            }
          out.push_back (json_object[i]);
          ++i;
        }
      return std::unexpected (AgentError::tool (
          ToolErrorCode::InvalidInput, std::string (field_name),
          "Unterminated string literal in JSON tool argument"));
    }

  if (expected_type == JsonType::Array || expected_type == JsonType::Object)
    {
      const char open_ch = (expected_type == JsonType::Array) ? '[' : '{';
      const char close_ch = (expected_type == JsonType::Array) ? ']' : '}';
      if (first_ch != open_ch)
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, std::string (field_name),
              "Expected composite JSON container for tool parameter"));
        }
      int depth = 0;
      bool in_quotes = false;
      for (std::size_t i = val_pos; i < json_object.size (); ++i)
        {
          if (json_object[i] == '"'
              && (i == 0U || json_object[i - 1U] != '\\'))
            {
              in_quotes = !in_quotes;
            }
          if (!in_quotes)
            {
              if (json_object[i] == open_ch)
                {
                  ++depth;
                }
              else if (json_object[i] == close_ch)
                {
                  --depth;
                  if (depth == 0)
                    {
                      return std::string (
                          json_object.substr (val_pos, i - val_pos + 1U));
                    }
                }
            }
        }
      return std::unexpected (AgentError::tool (
          ToolErrorCode::InvalidInput, std::string (field_name),
          "Unterminated composite value in JSON tool argument"));
    }

  std::size_t end_pos = val_pos;
  while (end_pos < json_object.size () && json_object[end_pos] != ','
         && json_object[end_pos] != '}' && json_object[end_pos] != ' '
         && json_object[end_pos] != '\t' && json_object[end_pos] != '\n'
         && json_object[end_pos] != '\r')
    {
      ++end_pos;
    }
  const std::string_view token
      = json_object.substr (val_pos, end_pos - val_pos);
  if (token.empty ())
    {
      return std::unexpected (AgentError::tool (
          ToolErrorCode::InvalidInput, std::string (field_name),
          "Empty value token in JSON tool argument"));
    }

  if (expected_type == JsonType::Boolean)
    {
      if (token != "true" && token != "false")
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, std::string (field_name),
              "Expected boolean literal true or false for tool parameter"));
        }
      return std::string (token);
    }

  if (expected_type == JsonType::Integer)
    {
      int64_t parsed = 0;
      const char *const first_ptr = std::to_address (token.begin ());
      const char *const last_ptr = std::to_address (token.end ());
      const auto [ptr, ec] = std::from_chars (first_ptr, last_ptr, parsed);
      if (ec != std::errc{} || ptr != last_ptr)
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, std::string (field_name),
              "Expected integer value for tool parameter"));
        }
      return std::string (token);
    }

  if (expected_type == JsonType::Number)
    {
      if (first_ch == '"' || token == "true" || token == "false")
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, std::string (field_name),
              "Expected numeric value for tool parameter"));
        }
      double parsed = 0.0;
      const char *const first_ptr = std::to_address (token.begin ());
      const char *const last_ptr = std::to_address (token.end ());
      const auto [ptr, ec] = std::from_chars (first_ptr, last_ptr, parsed);
      if (ec != std::errc{} || ptr != last_ptr)
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, std::string (field_name),
              "Invalid floating-point number in tool parameter"));
        }
      return std::string (token);
    }

  return std::string (token);
}

template <>
auto
parse_json_token_as<std::string> (std::string_view token) noexcept
    -> Result<std::string>
{
  return std::string (token);
}

template <>
auto
parse_json_token_as<int64_t> (std::string_view token) noexcept
    -> Result<int64_t>
{
  int64_t val = 0;
  const char *const first_ptr = std::to_address (token.begin ());
  const char *const last_ptr = std::to_address (token.end ());
  const auto [ptr, ec] = std::from_chars (first_ptr, last_ptr, val);
  if (ec != std::errc{} || ptr != last_ptr)
    {
      return std::unexpected (
          AgentError::tool (ToolErrorCode::SchemaViolation, "parser",
                            "Failed to parse integer parameter token"));
    }
  return val;
}

template <>
auto
parse_json_token_as<int> (std::string_view token) noexcept -> Result<int>
{
  auto res = parse_json_token_as<int64_t> (token);
  if (!res)
    {
      return std::unexpected (res.error ());
    }
  return static_cast<int> (*res);
}

template <>
auto
parse_json_token_as<double> (std::string_view token) noexcept -> Result<double>
{
  double val = 0.0;
  const char *const first_ptr = std::to_address (token.begin ());
  const char *const last_ptr = std::to_address (token.end ());
  const auto [ptr, ec] = std::from_chars (first_ptr, last_ptr, val);
  if (ec != std::errc{} || ptr != last_ptr)
    {
      return std::unexpected (
          AgentError::tool (ToolErrorCode::SchemaViolation, "parser",
                            "Failed to parse floating-point parameter token"));
    }
  return val;
}

template <>
auto
parse_json_token_as<float> (std::string_view token) noexcept -> Result<float>
{
  auto res = parse_json_token_as<double> (token);
  if (!res)
    {
      return std::unexpected (res.error ());
    }
  return static_cast<float> (*res);
}

template <>
auto
parse_json_token_as<bool> (std::string_view token) noexcept -> Result<bool>
{
  if (token == "true")
    {
      return true;
    }
  if (token == "false")
    {
      return false;
    }
  return std::unexpected (
      AgentError::tool (ToolErrorCode::SchemaViolation, "parser",
                        "Failed to parse boolean parameter token"));
}

/* Generates a canonical JSON Schema string from parameter descriptors.  */
auto
build_json_schema (std::span<const ToolParamSchema> params) noexcept
    -> Result<std::string>
{
  try
    {
      std::string out = R"({"type":"object","properties":{)";
      std::vector<std::string_view> required_names;
      required_names.reserve (params.size ());

      for (std::size_t i = 0U; i < params.size (); ++i)
        {
          const ToolParamSchema &p = params[i];
          if (p.name.empty ())
            {
              return std::unexpected (AgentError::tool (
                  ToolErrorCode::SchemaViolation, "schema_builder",
                  "Tool parameter name must not be empty"));
            }
          if (i > 0U)
            {
              out.push_back (',');
            }
          out.push_back ('"');
          out.append (p.name);
          out.append (R"(":{"type":")");
          out.append (json_type_to_string (p.type));
          out.append (R"("})");
          if (p.required)
            {
              required_names.push_back (p.name);
            }
        }

      out.append (R"(},"required":[)");
      for (std::size_t i = 0U; i < required_names.size (); ++i)
        {
          if (i > 0U)
            {
              out.push_back (',');
            }
          out.push_back ('"');
          out.append (required_names.at (i));
          out.push_back ('"');
        }
      out.append ("]}");
      return out;
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while generating tool JSON Schema"));
    }
}

/* Validates and parses a JSON Schema object string for round-trip checks.  */
auto
validate_and_parse_schema (std::string_view schema_json) noexcept
    -> Result<ParsedToolSchema>
{
  auto type_tok
      = extract_json_field_token (schema_json, "type", JsonType::String);
  if (!type_tok || *type_tok != "object")
    {
      return std::unexpected (
          AgentError::tool (ToolErrorCode::SchemaViolation, "schema_validator",
                            "Root JSON Schema must declare type as object"));
    }

  auto props_tok
      = extract_json_field_token (schema_json, "properties", JsonType::Object);
  if (!props_tok)
    {
      return std::unexpected (AgentError::tool (
          ToolErrorCode::SchemaViolation, "schema_validator",
          "Root JSON Schema must contain a properties object"));
    }

  ParsedToolSchema parsed;
  const std::string_view props = *props_tok;
  std::size_t pos = skip_ws (props, 1U);

  while (pos < props.size () && props[pos] != '}')
    {
      if (props[pos] == ',')
        {
          pos = skip_ws (props, pos + 1U);
          continue;
        }
      if (props[pos] != '"')
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, "schema_validator",
              "Expected quoted property name in JSON Schema properties"));
        }
      const std::size_t name_end = props.find ('"', pos + 1U);
      if (name_end == std::string_view::npos)
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, "schema_validator",
              "Unterminated property name in JSON Schema"));
        }
      std::string prop_name (props.substr (pos + 1U, name_end - pos - 1U));
      if (prop_name.empty ())
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, "schema_validator",
              "Empty property name is invalid in JSON Schema"));
        }

      const std::size_t colon_pos = skip_ws (props, name_end + 1U);
      if (colon_pos >= props.size () || props[colon_pos] != ':')
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, "schema_validator",
              "Missing colon after property name in JSON Schema"));
        }

      const std::size_t sub_start = skip_ws (props, colon_pos + 1U);
      if (sub_start >= props.size () || props[sub_start] != '{')
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, "schema_validator",
              "Property descriptor in JSON Schema must be an object"));
        }

      int depth = 0;
      std::size_t sub_end = sub_start;
      for (; sub_end < props.size (); ++sub_end)
        {
          if (props[sub_end] == '{')
            {
              ++depth;
            }
          else if (props[sub_end] == '}')
            {
              --depth;
              if (depth == 0)
                {
                  break;
                }
            }
        }
      if (sub_end >= props.size ())
        {
          return std::unexpected (AgentError::tool (
              ToolErrorCode::SchemaViolation, "schema_validator",
              "Unterminated property object in JSON Schema"));
        }

      const std::string_view sub_obj
          = props.substr (sub_start, sub_end - sub_start + 1U);
      auto sub_type
          = extract_json_field_token (sub_obj, "type", JsonType::String);
      if (!sub_type)
        {
          return std::unexpected (sub_type.error ());
        }
      auto jt_res = string_to_json_type (*sub_type);
      if (!jt_res)
        {
          return std::unexpected (jt_res.error ());
        }

      parsed.properties.push_back (
          ToolParamSchema{ std::move (prop_name), *jt_res, true });
      pos = skip_ws (props, sub_end + 1U);
    }

  return parsed;
}

/* Validates tool call arguments against parameter schema requirements.  */
auto
validate_tool_arguments (std::string_view args_json,
                         std::span<const ToolParamSchema> params) noexcept
    -> Result<void>
{
  if (!is_well_formed_json_object (args_json))
    {
      return std::unexpected (AgentError::tool (
          ToolErrorCode::InvalidInput, "validator",
          "Tool arguments payload is not a valid JSON object"));
    }

  for (const ToolParamSchema &param : params)
    {
      const auto tok_res
          = extract_json_field_token (args_json, param.name, param.type);
      if (!tok_res && param.required)
        {
          return std::unexpected (tok_res.error ());
        }
    }
  return {};
}

/* Private delegating copy constructor for ToolRegistry.  */
ToolRegistry::ToolRegistry (
    const ToolRegistry &other,
    const std::shared_lock<std::shared_mutex> & /*lock*/)
    : tools_ (other.tools_)
{
}

/* Copy constructor for ToolRegistry.  */
ToolRegistry::ToolRegistry (const ToolRegistry &other)
    : ToolRegistry (other, std::shared_lock (other.mtx_))
{
}

/* Copy assignment for ToolRegistry.  */
auto
ToolRegistry::operator= (const ToolRegistry &other) -> ToolRegistry &
{
  if (this != &other)
    {
      std::unique_lock lhs_lock (mtx_, std::defer_lock);
      std::shared_lock rhs_lock (other.mtx_, std::defer_lock);
      std::lock (lhs_lock, rhs_lock);
      tools_ = other.tools_;
    }
  return *this;
}

/* Private delegating move constructor for ToolRegistry.  */
ToolRegistry::ToolRegistry (
    ToolRegistry &&other,
    const std::unique_lock<std::shared_mutex> & /*lock*/) noexcept
    : tools_ (std::move (other).tools_)
{
}

/* Move constructor for ToolRegistry.  */
ToolRegistry::ToolRegistry (ToolRegistry &&other) noexcept
    : ToolRegistry (std::move (other), std::unique_lock (other.mtx_))
{
}

/* Move assignment for ToolRegistry.  */
auto
ToolRegistry::operator= (ToolRegistry &&other) noexcept -> ToolRegistry &
{
  if (this != &other)
    {
      std::scoped_lock lock (mtx_, other.mtx_);
      tools_ = std::move (other.tools_);
    }
  return *this;
}

/* Validates and registers a ToolDescriptor.  */
auto
ToolRegistry::register_tool (ToolDescriptor descriptor) noexcept
    -> Result<void>
{
  if (descriptor.name.empty () || descriptor.dispatch_fn == nullptr)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                                 "ToolDescriptor requires a non-empty name "
                                 "and non-null dispatch_fn"));
    }

  auto parsed_res = validate_and_parse_schema (descriptor.schema_json);
  if (!parsed_res)
    {
      return std::unexpected (parsed_res.error ());
    }
  if (descriptor.parameters.empty () && !parsed_res->properties.empty ())
    {
      descriptor.parameters = std::move (parsed_res->properties);
    }

  try
    {
      const std::unique_lock lock (mtx_);
      if (tools_.contains (descriptor.name))
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Duplicate tool registration name in ToolRegistry"));
        }
      std::string key = descriptor.name;
      tools_.emplace (std::move (key), std::move (descriptor));
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while registering tool in ToolRegistry"));
    }
}

/* Looks up, validates, and dispatches a tool call without copying history.  */
auto
ToolRegistry::invoke (std::string_view tool_name, const ToolContext &ctx,
                      std::string_view args_json) const noexcept
    -> Result<std::string>
{
  ToolDescriptor desc;
  {
    const std::shared_lock lock (mtx_);
    const auto it = tools_.find (std::string (tool_name));
    if (it == tools_.end ())
      {
        std::string msg = "Requested tool is not registered: ";
        msg.append (tool_name);
        return std::unexpected (AgentError::tool (ToolErrorCode::ToolNotFound,
                                                  std::string (tool_name),
                                                  std::move (msg)));
      }
    desc = it->second;
  }

  const auto val_res = validate_tool_arguments (args_json, desc.parameters);
  if (!val_res)
    {
      return std::unexpected (val_res.error ());
    }

  return desc.dispatch_fn (ctx, args_json);
}

/* Checks whether tool_name exists in the registry.  */
auto
ToolRegistry::has_tool (std::string_view tool_name) const noexcept -> bool
{
  const std::shared_lock lock (mtx_);
  return tools_.contains (std::string (tool_name));
}

/* Retrieves a copy of the registered ToolDescriptor.  */
auto
ToolRegistry::find_tool (std::string_view tool_name) const noexcept
    -> Result<ToolDescriptor>
{
  const std::shared_lock lock (mtx_);
  const auto it = tools_.find (std::string (tool_name));
  if (it == tools_.end ())
    {
      return std::unexpected (AgentError::tool (
          ToolErrorCode::ToolNotFound, std::string (tool_name),
          "Tool not found in ToolRegistry::find_tool"));
    }
  return it->second;
}

/* Returns the count of registered tools.  */
auto
ToolRegistry::size () const noexcept -> std::size_t
{
  const std::shared_lock lock (mtx_);
  return tools_.size ();
}

/* Exports registered tools in Model Context Protocol (MCP) JSON format.  */
auto
ToolRegistry::export_mcp_tools_json () const -> std::string
{
  const std::shared_lock lock (mtx_);
  std::string out = "[";
  bool first = true;
  for (const auto &[name, desc] : tools_)
    {
      if (!first)
        {
          out.push_back (',');
        }
      first = false;
      out.append (R"({"name":")");
      out.append (name);
      out.append (R"(","description":")");
      out.append (desc.description);
      out.append (R"(","inputSchema":)");
      out.append (desc.schema_json);
      out.push_back ('}');
    }
  out.push_back (']');
  return out;
}

} /* namespace sutra::inline v1 */
