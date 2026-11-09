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

#include <sutra/sutra.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

#define SUTRA_ASSERT(...)                                                     \
  do                                                                          \
    {                                                                         \
      if (!(__VA_ARGS__))                                                     \
        {                                                                     \
          std::cerr << __FILE__ << ":" << __LINE__                            \
                    << ": assertion failed: " #__VA_ARGS__ "\n";              \
          std::abort ();                                                      \
        }                                                                     \
    }                                                                         \
  while (false)

auto
fn_0_args () noexcept -> sutra::Result<std::string>
{
  return std::string ("ok");
}

auto
fn_2_args (int64_t x, bool flag) noexcept -> sutra::Result<std::string>
{
  return std::to_string (x) + (flag ? "_t" : "_f");
}

auto
fn_4_args (const sutra::ToolContext & /*ctx*/, std::string s, int a, double b,
           std::vector<int> /*v*/) noexcept -> sutra::Result<std::string>
{
  return s + std::to_string (a) + std::to_string (static_cast<int> (b));
}

/* Property 1: Tool schema generation and randomized round-tripping.  */
void
test_property_tool_schema_round_trip ()
{
  sutra::ToolRegistry reg;
  SUTRA_ASSERT (reg.register_function<&fn_0_args, 0U> (
                       "fn0", "Zero args", std::array<std::string_view, 0U>{})
                    .has_value ());
  SUTRA_ASSERT (
      SUTRA_REGISTER_TOOL (reg, &fn_2_args, "fn2", "Two args", "x", "flag")
          .has_value ());
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (reg, &fn_4_args, "fn4", "Four args", "s",
                                     "a", "b", "v")
                    .has_value ());

  for (const std::string_view name : { "fn0", "fn2", "fn4" })
    {
      const auto desc = reg.find_tool (name);
      SUTRA_ASSERT (desc.has_value ());
      const auto parsed = sutra::validate_and_parse_schema (desc->schema_json);
      SUTRA_ASSERT (parsed.has_value ());
      const auto rebuilt = sutra::build_json_schema (parsed->properties);
      SUTRA_ASSERT (rebuilt.has_value ());
      SUTRA_ASSERT (*rebuilt == desc->schema_json);
    }

  // Randomized property-based schema round-tripping across 250 random schemas.
  std::mt19937 rng (0x53555452U);
  const std::array<sutra::JsonType, 6U> all_types{
    sutra::JsonType::String, sutra::JsonType::Integer,
    sutra::JsonType::Number, sutra::JsonType::Boolean,
    sutra::JsonType::Array,  sutra::JsonType::Object
  };

  for (int trial = 0; trial < 250; ++trial)
    {
      const std::size_t arity = static_cast<std::size_t> (rng () % 10U);
      std::vector<sutra::ToolParamSchema> params;
      params.reserve (arity);
      for (std::size_t p = 0U; p < arity; ++p)
        {
          sutra::ToolParamSchema ps;
          ps.name
              = "arg_" + std::to_string (p) + "_t" + std::to_string (trial);
          ps.type = all_types.at (rng () % all_types.size ());
          ps.required = true;
          params.push_back (std::move (ps));
        }

      const auto schema_str = sutra::build_json_schema (params);
      SUTRA_ASSERT (schema_str.has_value ());
      const auto parsed = sutra::validate_and_parse_schema (*schema_str);
      SUTRA_ASSERT (parsed.has_value ());
      SUTRA_ASSERT (parsed->properties.size () == params.size ());

      for (std::size_t p = 0U; p < arity; ++p)
        {
          SUTRA_ASSERT (parsed->properties.at (p).name == params.at (p).name);
          SUTRA_ASSERT (parsed->properties.at (p).type == params.at (p).type);
        }

      const auto round_trip = sutra::build_json_schema (parsed->properties);
      SUTRA_ASSERT (round_trip.has_value () && *round_trip == *schema_str);
    }
}

/* Independent graph-theoretic oracle using Warshall transitive closure to
   detect cycles in an adjacency matrix.  */
auto
independent_has_cycle (const std::vector<std::vector<bool>> &adj) -> bool
{
  const std::size_t n = adj.size ();
  std::vector<std::vector<bool>> reach = adj;
  for (std::size_t k = 0U; k < n; ++k)
    {
      for (std::size_t i = 0U; i < n; ++i)
        {
          for (std::size_t j = 0U; j < n; ++j)
            {
              if (reach.at (i).at (k) && reach.at (k).at (j))
                {
                  reach.at (i).at (j) = true;
                }
            }
        }
    }
  for (std::size_t i = 0U; i < n; ++i)
    {
      if (reach.at (i).at (i))
        {
          return true;
        }
    }
  return false;
}

/* Property 2: Randomized graph generation and PlanLinter oracle verification.
 */
void
test_property_plan_linter_random_graphs ()
{
  std::mt19937 rng (0xABCDEF01U);

  for (int trial = 0; trial < 300; ++trial)
    {
      const std::size_t n = 2U + (rng () % 10U);
      sutra::WorkflowGraph graph;
      std::vector<sutra::JsonType> node_out_types (n);
      std::vector<sutra::JsonType> node_in_types (n);

      const bool inject_schema_mismatch = (trial % 7 == 0);
      const bool inject_dangling = (trial % 11 == 0);

      for (std::size_t i = 0U; i < n; ++i)
        {
          node_in_types.at (i) = sutra::JsonType::Integer;
          node_out_types.at (i) = (inject_schema_mismatch && i == 0U)
                                      ? sutra::JsonType::String
                                      : sutra::JsonType::Integer;

          std::vector<sutra::ToolParamSchema> in_p{
            { "val", node_in_types.at (i), true }
          };
          std::vector<sutra::ToolParamSchema> out_p{
            { "val", node_out_types.at (i), true }
          };

          sutra::WorkflowNode node;
          node.id = "n_" + std::to_string (i);
          node.input_schema_json = *sutra::build_json_schema (in_p);
          node.output_schema_json = *sutra::build_json_schema (out_p);
          SUTRA_ASSERT (graph.add_node (std::move (node)).has_value ());
        }

      std::vector<std::vector<bool>> adj (n, std::vector<bool> (n, false));
      std::vector<std::vector<std::size_t>> incoming (n);

      // Generate random edges (allowing cycles in ~40% of trials).
      const bool allow_back_edges = (trial % 3 == 0);
      for (std::size_t u = 0U; u < n; ++u)
        {
          for (std::size_t v = 0U; v < n; ++v)
            {
              if (!allow_back_edges && u >= v)
                {
                  continue;
                }
              if ((rng () % 4U) == 0U)
                {
                  adj.at (u).at (v) = true;
                  incoming.at (v).push_back (u);
                  SUTRA_ASSERT (graph
                                    .add_edge ("n_" + std::to_string (u),
                                               "n_" + std::to_string (v))
                                    .has_value ());
                }
            }
        }

      if (inject_dangling)
        {
          SUTRA_ASSERT (
              graph.add_edge ("n_0", "non_existent_node").has_value ());
        }

      const bool oracle_cyclic = independent_has_cycle (adj);
      bool oracle_schema_mismatch = false;
      for (std::size_t v = 0U; v < n; ++v)
        {
          for (const std::size_t u : incoming.at (v))
            {
              if (node_out_types.at (u) != node_in_types.at (v))
                {
                  oracle_schema_mismatch = true;
                }
            }
        }

      const bool oracle_valid
          = !inject_dangling && !oracle_cyclic && !oracle_schema_mismatch;
      const auto lint_res = sutra::PlanLinter::validate (graph);

      SUTRA_ASSERT (lint_res.has_value () == oracle_valid);
      if (!lint_res.has_value ())
        {
          const auto code
              = std::get<sutra::PlanningError> (lint_res.error ().variant ())
                    .code;
          if (inject_dangling)
            {
              SUTRA_ASSERT (code
                            == sutra::PlanningErrorCode::DanglingDependency);
            }
          else if (oracle_cyclic)
            {
              SUTRA_ASSERT (code == sutra::PlanningErrorCode::CycleDetected);
            }
          else if (oracle_schema_mismatch)
            {
              SUTRA_ASSERT (code == sutra::PlanningErrorCode::SchemaMismatch);
            }
        }
    }
}

/* Property 3: Strict isolation across Short-Term, Long-Term, and Semantic
   memory tiers.  */
void
test_property_memory_tier_isolation ()
{
  auto mem_res = sutra::MemorySystem::create (16U, 4U);
  SUTRA_ASSERT (mem_res.has_value ());
  sutra::MemorySystem &mem = *mem_res;

  // Populate all three tiers with initial baseline state.
  sutra::Message init_msg;
  init_msg.role = sutra::MessageRole::System;
  init_msg.content = "system_prompt";
  SUTRA_ASSERT (mem.short_term ().append (init_msg).has_value ());
  SUTRA_ASSERT (mem.long_term ().put ("k1", "v1").has_value ());
  const std::array<float, 4U> emb{ 0.5F, 0.5F, 0.5F, 0.5F };
  SUTRA_ASSERT (mem.semantic ().add_document ("d1", "doc1", emb).has_value ());

  // 100 random operations on Tier 1 (ShortTerm) must never alter Tier 2 or 3.
  for (int i = 0; i < 100; ++i)
    {
      sutra::Message m;
      m.content = "turn_" + std::to_string (i);
      SUTRA_ASSERT (mem.short_term ().append (std::move (m)).has_value ());
      SUTRA_ASSERT (mem.long_term ().size () == 1U);
      SUTRA_ASSERT (*mem.long_term ().get ("k1") == "v1");
      SUTRA_ASSERT (mem.semantic ().size () == 1U);
    }

  mem.short_term ().clear ();
  SUTRA_ASSERT (mem.short_term ().size () == 0U);
  SUTRA_ASSERT (mem.long_term ().size () == 1U);
  SUTRA_ASSERT (mem.semantic ().size () == 1U);

  // Operations on Tier 2 (LongTerm) must never alter Tier 1 or Tier 3.
  for (int i = 0; i < 50; ++i)
    {
      SUTRA_ASSERT (
          mem.long_term ()
              .put ("key_" + std::to_string (i), "val_" + std::to_string (i))
              .has_value ());
    }
  mem.long_term ().clear ();
  SUTRA_ASSERT (mem.long_term ().size () == 0U);
  SUTRA_ASSERT (mem.semantic ().size () == 1U);

  // Operations on Tier 3 (Semantic) must never alter Tier 1 or Tier 2.
  SUTRA_ASSERT (mem.long_term ().put ("persist", "true").has_value ());
  mem.semantic ().clear ();
  SUTRA_ASSERT (mem.semantic ().size () == 0U);
  SUTRA_ASSERT (mem.long_term ().size () == 1U);
  SUTRA_ASSERT (*mem.long_term ().get ("persist") == "true");
}

} /* namespace */

auto
main () -> int
{
  test_property_tool_schema_round_trip ();
  test_property_plan_linter_random_graphs ();
  test_property_memory_tier_isolation ();
  std::cout << "All Sutra property-based tests PASSED.\n";
  return 0;
}
