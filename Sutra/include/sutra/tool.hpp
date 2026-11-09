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

#ifndef SUTRA_TOOL_HPP
#define SUTRA_TOOL_HPP

#include <sutra/error.hpp>
#include <sutra/memory.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sutra::inline v1
{

/**
 * @brief Lightweight zero-copy handle passed to every tool invocation.
 * @pre Underlying ShortTermMemory and InferenceArena must outlive invocation.
 * @post Never copies or serializes conversation state (eliminates O(N^2)
 * growth).
 * @note Thread-safety: Read-only access to conversation_view is thread-safe.
 */
struct ToolContext
{
  ConversationSpanView conversation_view;
  std::pmr::memory_resource *scratch_arena = nullptr;
  uint64_t run_id = 0U;
  std::size_t step_index = 0U;
};

/**
 * @brief Supported JSON Schema primitive and composite value types.
 * @note Thread-safety: Immutable enum type.
 */
enum class JsonType : uint8_t
{
  String = 0,
  Integer = 1,
  Number = 2,
  Boolean = 3,
  Array = 4,
  Object = 5
};

/**
 * @brief Converts a JsonType enumerator to its standard JSON Schema keyword.
 * @param type JsonType to convert.
 * @pre None.
 * @post State is unchanged.
 * @returns Lowercase JSON Schema type string.
 */
[[nodiscard]] constexpr auto
json_type_to_string (JsonType type) noexcept -> std::string_view
{
  switch (type)
    {
    case JsonType::String:
      return "string";
    case JsonType::Integer:
      return "integer";
    case JsonType::Number:
      return "number";
    case JsonType::Boolean:
      return "boolean";
    case JsonType::Array:
      return "array";
    case JsonType::Object:
      return "object";
    }
  return "string";
}

namespace detail
{

template <typename T> struct IsStdVector : std::false_type
{
};

template <typename U, typename Alloc>
struct IsStdVector<std::vector<U, Alloc>> : std::true_type
{
};

} /* namespace detail */

/**
 * @brief Concept satisfied by C++ types reflectable into JSON Schema types.
 */
template <typename T>
concept JsonReflectable
    = std::same_as<std::remove_cvref_t<T>, bool>
      || (std::integral<std::remove_cvref_t<T>>
          && !std::same_as<std::remove_cvref_t<T>, bool>)
      || std::floating_point<std::remove_cvref_t<T>>
      || std::same_as<std::remove_cvref_t<T>, std::string>
      || std::same_as<std::remove_cvref_t<T>, std::string_view>
      || detail::IsStdVector<std::remove_cvref_t<T>>::value;

/**
 * @brief Compile-time reflection mapping C++ parameter types to JsonType.
 * @tparam T C++ parameter type satisfying JsonReflectable.
 * @returns Corresponding JsonType at compile time.
 */
template <JsonReflectable T>
[[nodiscard]] consteval auto
reflect_json_type () noexcept -> JsonType
{
  using CleanT = std::remove_cvref_t<T>;
  if constexpr (std::same_as<CleanT, bool>)
    {
      return JsonType::Boolean;
    }
  else if constexpr (std::integral<CleanT>)
    {
      return JsonType::Integer;
    }
  else if constexpr (std::floating_point<CleanT>)
    {
      return JsonType::Number;
    }
  else if constexpr (std::same_as<CleanT, std::string>
                     || std::same_as<CleanT, std::string_view>)
    {
      return JsonType::String;
    }
  else
    {
      return JsonType::Array;
    }
}

/**
 * @brief Describes a single reflected parameter in a tool's JSON Schema.
 * @note Thread-safety: Value type.
 */
struct ToolParamSchema
{
  std::string name;
  JsonType type = JsonType::String;
  bool required = true;
};

/**
 * @brief Parsed representation of a JSON Schema for round-trip verification.
 * @note Thread-safety: Value type.
 */
struct ParsedToolSchema
{
  std::vector<ToolParamSchema> properties;
};

/**
 * @brief Non-erased function pointer type for tool dispatch.
 */
using ToolDispatchFn
    = auto (*) (const ToolContext &,
                std::string_view) noexcept -> Result<std::string>;

/**
 * @brief Complete registration entry for a Sutra tool.
 * @note Thread-safety: Immutable once registered in ToolRegistry.
 */
struct ToolDescriptor
{
  std::string name;
  std::string description;
  std::string schema_json;
  std::vector<ToolParamSchema> parameters;
  ToolDispatchFn dispatch_fn = nullptr;
};

/**
 * @brief Extracts a typed parameter value from a JSON object string.
 * @param json_object Raw JSON object payload.
 * @param field_name Property key to extract.
 * @param expected_type Expected JsonType for validation.
 * @pre json_object is a valid JSON object.
 * @post State is unchanged.
 * @returns Result<std::string> raw token or ToolError on type/missing
 * mismatch.
 */
[[nodiscard]] auto extract_json_field_token (
    std::string_view json_object, std::string_view field_name,
    JsonType expected_type) noexcept -> Result<std::string>;

/**
 * @brief Parses a validated JSON token into a concrete C++ value of type T.
 * @tparam T Target C++ parameter type satisfying JsonReflectable.
 * @param token Validated token string returned by extract_json_field_token.
 * @returns Result<std::remove_cvref_t<T>> or ToolError on conversion failure.
 */
template <JsonReflectable T>
[[nodiscard]] auto parse_json_token_as (std::string_view token) noexcept
    -> Result<std::remove_cvref_t<T>>;

template <>
[[nodiscard]] auto parse_json_token_as<std::string> (
    std::string_view token) noexcept -> Result<std::string>;

template <>
[[nodiscard]] auto parse_json_token_as<int64_t> (
    std::string_view token) noexcept -> Result<int64_t>;

template <>
[[nodiscard]] auto
parse_json_token_as<int> (std::string_view token) noexcept -> Result<int>;

template <>
[[nodiscard]] auto parse_json_token_as<double> (
    std::string_view token) noexcept -> Result<double>;

template <>
[[nodiscard]] auto
parse_json_token_as<float> (std::string_view token) noexcept -> Result<float>;

template <>
[[nodiscard]] auto
parse_json_token_as<bool> (std::string_view token) noexcept -> Result<bool>;

template <JsonReflectable T>
auto
parse_json_token_as (std::string_view token) noexcept
    -> Result<std::remove_cvref_t<T>>
{
  using CleanT = std::remove_cvref_t<T>;
  if constexpr (std::same_as<CleanT, std::string_view>)
    {
      return std::string_view (token);
    }
  else if constexpr (std::integral<CleanT> && !std::same_as<CleanT, bool>)
    {
      auto res = parse_json_token_as<int64_t> (token);
      if (!res)
        {
          return std::unexpected (res.error ());
        }
      return static_cast<CleanT> (*res);
    }
  else if constexpr (detail::IsStdVector<CleanT>::value)
    {
      (void)token;
      return CleanT{};
    }
  else
    {
      return CleanT{};
    }
}

namespace detail
{

template <typename T> struct StorageTypeFor
{
  using Type = std::remove_cvref_t<T>;
};

template <> struct StorageTypeFor<std::string_view>
{
  using Type = std::string;
};

template <> struct StorageTypeFor<const std::string_view>
{
  using Type = std::string;
};

template <> struct StorageTypeFor<const std::string_view &>
{
  using Type = std::string;
};

template <typename Fn> struct FunctionTraits;

/* Specialization for functions taking (const ToolContext &, Args...).  */
template <typename R, typename... Args>
struct FunctionTraits<R (*) (const ToolContext &, Args...)>
{
  static_assert ((JsonReflectable<Args> && ...),
                 "All tool parameter types must satisfy JsonReflectable");
  using ReturnType = R;
  using ArgsTuple = std::tuple<typename StorageTypeFor<Args>::Type...>;
  static constexpr std::size_t kArity = sizeof...(Args);
  static constexpr bool kTakesContext = true;

  static consteval auto
  reflected_types () noexcept -> std::array<JsonType, sizeof...(Args)>
  {
    return { reflect_json_type<Args> ()... };
  }
};

template <typename R, typename... Args>
struct FunctionTraits<R (*) (const ToolContext &, Args...) noexcept>
    : FunctionTraits<R (*) (const ToolContext &, Args...)>
{
};

/* Specialization for plain functions taking (Args...) without ToolContext.  */
template <typename R, typename... Args> struct FunctionTraits<R (*) (Args...)>
{
  static_assert ((JsonReflectable<Args> && ...),
                 "All tool parameter types must satisfy JsonReflectable");
  using ReturnType = R;
  using ArgsTuple = std::tuple<typename StorageTypeFor<Args>::Type...>;
  static constexpr std::size_t kArity = sizeof...(Args);
  static constexpr bool kTakesContext = false;

  static consteval auto
  reflected_types () noexcept -> std::array<JsonType, sizeof...(Args)>
  {
    return { reflect_json_type<Args> ()... };
  }
};

template <typename R, typename... Args>
struct FunctionTraits<R (*) (Args...) noexcept>
    : FunctionTraits<R (*) (Args...)>
{
};

template <auto FnPtr, std::size_t... Is>
[[nodiscard]] auto
invoke_with_unpacked_json (
    const ToolContext &ctx, [[maybe_unused]] std::string_view args_json,
    [[maybe_unused]] std::span<const std::string_view, sizeof...(Is)>
        param_names,
    std::index_sequence<Is...>) noexcept -> Result<std::string>
{
  using Traits = FunctionTraits<decltype (FnPtr)>;
  using TupleT = typename Traits::ArgsTuple;
  [[maybe_unused]] constexpr auto kTypes = Traits::reflected_types ();

  std::tuple<Result<std::tuple_element_t<Is, TupleT>>...> parsed_results{
    [&] () -> Result<std::tuple_element_t<Is, TupleT>> {
      auto tok_res = extract_json_field_token (args_json, param_names[Is],
                                               std::get<Is> (kTypes));
      if (!tok_res)
        {
          return std::unexpected (tok_res.error ());
        }
      return parse_json_token_as<std::tuple_element_t<Is, TupleT>> (*tok_res);
    }()...
  };

  const AgentError *first_err = nullptr;
  (
      [&] {
        if (first_err == nullptr
            && !std::get<Is> (parsed_results).has_value ())
          {
            first_err = &std::get<Is> (parsed_results).error ();
          }
      }(),
      ...);

  if (first_err != nullptr)
    {
      return std::unexpected (*first_err);
    }

  try
    {
      if constexpr (Traits::kTakesContext)
        {
          if constexpr (std::same_as<typename Traits::ReturnType,
                                     Result<std::string>>)
            {
              return FnPtr (ctx,
                            std::move (*std::get<Is> (parsed_results))...);
            }
          else
            {
              return std::string (
                  FnPtr (ctx, std::move (*std::get<Is> (parsed_results))...));
            }
        }
      else
        {
          (void)ctx;
          if constexpr (std::same_as<typename Traits::ReturnType,
                                     Result<std::string>>)
            {
              return FnPtr (std::move (*std::get<Is> (parsed_results))...);
            }
          else
            {
              return std::string (
                  FnPtr (std::move (*std::get<Is> (parsed_results))...));
            }
        }
    }
  catch (...)
    {
      return std::unexpected (AgentError::tool (
          ToolErrorCode::ExecutionFailure, "reflected_tool",
          "Unhandled exception thrown inside registered C++ tool function"));
    }
}

} /* namespace detail */

/**
 * @brief Generates a canonical JSON Schema string from parameter descriptors.
 * @param params Span of parameter descriptors.
 * @pre All parameter names must be non-empty.
 * @post Returned JSON Schema is valid and round-trippable.
 * @returns Result<std::string> containing the JSON Schema object.
 */
[[nodiscard]] auto build_json_schema (
    std::span<const ToolParamSchema> params) noexcept -> Result<std::string>;

/**
 * @brief Validates and parses a JSON Schema object string for round-tripping.
 * @param schema_json JSON Schema string to validate.
 * @pre None.
 * @post Verifies "type":"object", "properties", and "required" consistency.
 * @returns Result<ParsedToolSchema> or ToolErrorCode::SchemaViolation.
 */
[[nodiscard]] auto validate_and_parse_schema (
    std::string_view schema_json) noexcept -> Result<ParsedToolSchema>;

/**
 * @brief Validates a JSON argument payload against a list of parameter
 * schemas.
 * @param args_json Incoming JSON object string for a tool invocation.
 * @param params Expected parameter schemas.
 * @pre None.
 * @post State is unchanged.
 * @returns Result<void> or ToolError (InvalidInput or SchemaViolation).
 */
[[nodiscard]] auto validate_tool_arguments (
    std::string_view args_json,
    std::span<const ToolParamSchema> params) noexcept -> Result<void>;

/**
 * @brief Thread-safe registry for compile-time reflected and dynamic tools.
 * @pre Tool names must be unique and non-empty.
 * @post Validates schemas at registration time and arguments at dispatch time.
 * @note Thread-safety: Thread-safe via internal std::shared_mutex.
 */
class ToolRegistry
{
public:
  ToolRegistry () = default;
  ToolRegistry (const ToolRegistry &other);
  auto operator= (const ToolRegistry &other) -> ToolRegistry &;
  ToolRegistry (ToolRegistry &&other) noexcept;
  auto operator= (ToolRegistry &&other) noexcept -> ToolRegistry &;
  ~ToolRegistry () noexcept = default;

  /**
   * @brief Registers a tool descriptor and validates its JSON Schema.
   * @param descriptor Complete ToolDescriptor with dispatch function.
   * @pre !descriptor.name.empty() and descriptor.dispatch_fn != nullptr.
   * @post Tool is available for invoke() and MCP export.
   * @returns Result<void> or AgentError if schema is invalid or duplicate.
   * @note Thread-safety: Acquires exclusive write lock.
   */
  [[nodiscard]] auto
  register_tool (ToolDescriptor descriptor) noexcept -> Result<void>;

  /**
   * @brief Registers a C++ function pointer using compile-time reflection.
   * @tparam FnPtr Compile-time function pointer to reflect.
   * @tparam N Number of tool parameters (verified at compile time).
   * @param name Unique tool identifier.
   * @param description Human-readable description for the LLM planner.
   * @param param_names Fixed array of parameter names matching FnPtr arity.
   * @pre sizeof...(Args) == N (enforced by static_assert at compile time).
   * @post Generates JSON Schema and non-erased dispatch trampoline.
   * @returns Result<void> or AgentError on duplicate name.
   */
  template <auto FnPtr, std::size_t N>
  [[nodiscard]] auto
  register_function (std::string name, std::string description,
                     const std::array<std::string_view, N>
                         &param_names) noexcept -> Result<void>
  {
    using Traits = detail::FunctionTraits<decltype (FnPtr)>;
    static_assert (Traits::kArity == N,
                   "Parameter name count must match C++ function arity");

    constexpr auto kReflectedTypes = Traits::reflected_types ();
    std::vector<ToolParamSchema> params;
    params.reserve (N);
    for (std::size_t i = 0U; i < N; ++i)
      {
        params.push_back (ToolParamSchema{ std::string (param_names.at (i)),
                                           kReflectedTypes.at (i), true });
      }

    auto schema_res = build_json_schema (params);
    if (!schema_res)
      {
        return std::unexpected (schema_res.error ());
      }

    static const std::array<std::string_view, N> kSavedNames = param_names;
    const ToolDispatchFn trampoline
        = +[] (const ToolContext &ctx,
               std::string_view args_json) noexcept -> Result<std::string> {
      return detail::invoke_with_unpacked_json<FnPtr> (
          ctx, args_json, std::span<const std::string_view, N> (kSavedNames),
          std::make_index_sequence<N>{});
    };

    ToolDescriptor desc;
    desc.name = std::move (name);
    desc.description = std::move (description);
    desc.schema_json = std::move (*schema_res);
    desc.parameters = std::move (params);
    desc.dispatch_fn = trampoline;
    return register_tool (std::move (desc));
  }

  /**
   * @brief Invokes a registered tool after validating JSON arguments.
   * @param tool_name Name of the registered tool.
   * @param ctx Zero-copy ToolContext borrowing conversation history.
   * @param args_json JSON object string containing tool arguments.
   * @pre tool_name is registered and args_json conforms to tool schema.
   * @post Returns tool output or typed ToolError without throwing.
   * @returns Result<std::string> tool output or ToolError.
   * @note Thread-safety: Acquires shared read lock during lookup, then
   *       executes tool function outside lock.
   */
  [[nodiscard]] auto
  invoke (std::string_view tool_name, const ToolContext &ctx,
          std::string_view args_json) const noexcept -> Result<std::string>;

  /**
   * @brief Checks whether a tool is registered.
   * @param tool_name Identifier to query.
   * @pre None.
   * @post State is unchanged.
   * @returns True if tool_name exists in registry.
   */
  [[nodiscard]] auto
  has_tool (std::string_view tool_name) const noexcept -> bool;

  /**
   * @brief Retrieves a copy of a tool's descriptor by name.
   * @param tool_name Identifier to query.
   * @pre None.
   * @post State is unchanged.
   * @returns Result<ToolDescriptor> or ToolErrorCode::ToolNotFound.
   */
  [[nodiscard]] auto find_tool (std::string_view tool_name) const noexcept
      -> Result<ToolDescriptor>;

  /**
   * @brief Returns the number of registered tools.
   * @pre None.
   * @post State is unchanged.
   * @returns Registered tool count.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Exports all registered tools in Model Context Protocol (MCP) JSON.
   * @pre None.
   * @post State is unchanged.
   * @returns JSON array string of MCP tool objects with inputSchema.
   */
  [[nodiscard]] auto export_mcp_tools_json () const -> std::string;

private:
  ToolRegistry (const ToolRegistry &other,
                const std::shared_lock<std::shared_mutex> &lock);
  ToolRegistry (ToolRegistry &&other,
                const std::unique_lock<std::shared_mutex> &lock) noexcept;

  mutable std::shared_mutex mtx_;
  std::unordered_map<std::string, ToolDescriptor> tools_;
};

/**
 * @brief Macro registering a C++ function with compile-time schema reflection.
 */
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SUTRA_REGISTER_TOOL(registry, fn_ptr, tool_name, tool_desc, ...)      \
  (registry).register_function<(fn_ptr)> (                                    \
      (tool_name), (tool_desc),                                               \
      std::to_array<std::string_view> ({ __VA_ARGS__ }))

} /* namespace sutra::inline v1 */

#endif /* SUTRA_TOOL_HPP */
