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

#include <sutra/observability.hpp>
#include <sutra/workflow.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <utility>

namespace sutra::inline v1
{

namespace
{

/* Merges multiple upstream JSON objects into a single JSON object payload.  */
[[nodiscard]] auto
merge_upstream_json_outputs (std::span<const std::string> payloads)
    -> std::string
{
  if (payloads.empty ())
    {
      return "{}";
    }
  if (payloads.size () == 1U)
    {
      return payloads[0];
    }

  std::string merged = "{";
  bool first_entry = true;
  for (const std::string &obj : payloads)
    {
      const std::size_t open_brace = obj.find ('{');
      const std::size_t close_brace = obj.rfind ('}');
      if (open_brace == std::string::npos || close_brace == std::string::npos
          || close_brace <= open_brace + 1U)
        {
          continue;
        }
      const std::string_view inner = std::string_view (obj).substr (
          open_brace + 1U, close_brace - open_brace - 1U);
      bool all_ws = true;
      for (const char ch : inner)
        {
          if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
            {
              all_ws = false;
              break;
            }
        }
      if (all_ws)
        {
          continue;
        }
      if (!first_entry)
        {
          merged.push_back (',');
        }
      first_entry = false;
      merged.append (inner);
    }
  merged.push_back ('}');
  return merged;
}

} /* namespace */

/* Adds a stage node to the graph after checking for duplicate IDs.  */
auto
WorkflowGraph::add_node (WorkflowNode node) noexcept -> Result<void>
{
  if (node.id.empty ())
    {
      return std::unexpected (
          AgentError::planning (PlanningErrorCode::NoValidPlan, nodes_.size (),
                                "WorkflowNode id must not be empty"));
    }
  for (const WorkflowNode &existing : nodes_)
    {
      if (existing.id == node.id)
        {
          return std::unexpected (AgentError::planning (
              PlanningErrorCode::NoValidPlan, nodes_.size (),
              "Duplicate WorkflowNode id in WorkflowGraph"));
        }
    }
  try
    {
      nodes_.push_back (std::move (node));
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while adding node to WorkflowGraph"));
    }
}

/* Adds a directed dependency edge between two node IDs.  */
auto
WorkflowGraph::add_edge (std::string from_node,
                         std::string to_node) noexcept -> Result<void>
{
  if (from_node.empty () || to_node.empty ())
    {
      return std::unexpected (AgentError::planning (
          PlanningErrorCode::DanglingDependency, edges_.size (),
          "WorkflowEdge endpoints must not be empty"));
    }
  try
    {
      edges_.push_back (
          WorkflowEdge{ std::move (from_node), std::move (to_node) });
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while adding edge to WorkflowGraph"));
    }
}

/* Returns a span of all registered nodes.  */
auto
WorkflowGraph::nodes () const noexcept -> std::span<const WorkflowNode>
{
  return nodes_;
}

/* Returns a span of all registered directed edges.  */
auto
WorkflowGraph::edges () const noexcept -> std::span<const WorkflowEdge>
{
  return edges_;
}

/* Validates a WorkflowGraph for dangling dependencies, cycles, and schema
   compatibility.  */
auto
PlanLinter::validate (const WorkflowGraph &graph) noexcept
    -> Result<LintReport>
{
  const ScopedStageTimer timer (AgentStats::global (), StageKind::Planning,
                                "plan_linter_validate");

  const std::span<const WorkflowNode> nodes = graph.nodes ();
  const std::span<const WorkflowEdge> edges = graph.edges ();

  if (nodes.empty ())
    {
      return std::unexpected (AgentError::planning (
          PlanningErrorCode::NoValidPlan, 0U,
          "WorkflowGraph must contain at least one WorkflowNode"));
    }

  try
    {
      const std::size_t n = nodes.size ();
      std::unordered_map<std::string, std::size_t> id_to_idx;
      id_to_idx.reserve (n);

      std::vector<ParsedToolSchema> in_schemas (n);
      std::vector<ParsedToolSchema> out_schemas (n);

      for (std::size_t i = 0U; i < n; ++i)
        {
          if (nodes[i].id.empty () || id_to_idx.contains (nodes[i].id))
            {
              return std::unexpected (AgentError::planning (
                  PlanningErrorCode::NoValidPlan, i,
                  "Invalid or duplicate node ID found in WorkflowGraph"));
            }
          id_to_idx.emplace (nodes[i].id, i);

          auto in_res = validate_and_parse_schema (nodes[i].input_schema_json);
          if (!in_res)
            {
              return std::unexpected (AgentError::planning (
                  PlanningErrorCode::SchemaMismatch, i,
                  "Invalid input_schema_json on workflow node"));
            }
          auto out_res
              = validate_and_parse_schema (nodes[i].output_schema_json);
          if (!out_res)
            {
              return std::unexpected (AgentError::planning (
                  PlanningErrorCode::SchemaMismatch, i,
                  "Invalid output_schema_json on workflow node"));
            }
          in_schemas.at (i) = std::move (*in_res);
          out_schemas.at (i) = std::move (*out_res);
        }

      std::vector<std::vector<std::size_t>> adj (n);
      std::vector<std::vector<std::size_t>> incoming (n);
      std::vector<std::size_t> in_degree (n, 0U);

      for (std::size_t e_idx = 0U; e_idx < edges.size (); ++e_idx)
        {
          const WorkflowEdge &edge = edges[e_idx];
          const auto u_it = id_to_idx.find (edge.from_node);
          const auto v_it = id_to_idx.find (edge.to_node);
          if (u_it == id_to_idx.end () || v_it == id_to_idx.end ())
            {
              return std::unexpected (AgentError::planning (
                  PlanningErrorCode::DanglingDependency, e_idx,
                  "WorkflowEdge references a non-existent node ID"));
            }
        }

      for (std::size_t e_idx = 0U; e_idx < edges.size (); ++e_idx)
        {
          const WorkflowEdge &edge = edges[e_idx];
          const std::size_t u = id_to_idx.at (edge.from_node);
          const std::size_t v = id_to_idx.at (edge.to_node);
          if (u == v)
            {
              return std::unexpected (AgentError::planning (
                  PlanningErrorCode::CycleDetected, e_idx,
                  "Self-loop cycle detected in WorkflowGraph"));
            }

          adj.at (u).push_back (v);
          incoming.at (v).push_back (u);
          ++in_degree.at (v);
        }

      // Topological sort (Kahn's algorithm) for cycle detection and wave
      // partitioning.
      std::vector<std::size_t> current_wave;
      for (std::size_t i = 0U; i < n; ++i)
        {
          if (in_degree.at (i) == 0U)
            {
              current_wave.push_back (i);
            }
        }

      std::vector<std::vector<std::string>> waves;
      std::size_t visited = 0U;

      while (!current_wave.empty ())
        {
          std::vector<std::string> wave_ids;
          wave_ids.reserve (current_wave.size ());
          std::vector<std::size_t> next_wave;

          for (const std::size_t u : current_wave)
            {
              ++visited;
              wave_ids.push_back (nodes[u].id);
              for (const std::size_t v : adj.at (u))
                {
                  --in_degree.at (v);
                  if (in_degree.at (v) == 0U)
                    {
                      next_wave.push_back (v);
                    }
                }
            }
          waves.push_back (std::move (wave_ids));
          current_wave = std::move (next_wave);
        }

      if (visited != n)
        {
          return std::unexpected (AgentError::planning (
              PlanningErrorCode::CycleDetected, 0U,
              "Directed cycle detected in WorkflowGraph by PlanLinter"));
        }

      // Verify schema compatibility for every node with incoming edges.
      for (std::size_t v = 0U; v < n; ++v)
        {
          if (incoming.at (v).empty ())
            {
              continue;
            }

          for (const ToolParamSchema &req_prop : in_schemas.at (v).properties)
            {
              bool found_matching_producer = false;
              for (const std::size_t u : incoming.at (v))
                {
                  for (const ToolParamSchema &out_prop :
                       out_schemas.at (u).properties)
                    {
                      if (out_prop.name == req_prop.name)
                        {
                          if (out_prop.type != req_prop.type)
                            {
                              return std::unexpected (AgentError::planning (
                                  PlanningErrorCode::SchemaMismatch, v,
                                  "Property type mismatch between upstream "
                                  "output schema and downstream input "
                                  "schema"));
                            }
                          found_matching_producer = true;
                        }
                    }
                }

              if (req_prop.required && !found_matching_producer)
                {
                  return std::unexpected (AgentError::planning (
                      PlanningErrorCode::SchemaMismatch, v,
                      "Required downstream input property is not produced by "
                      "any incoming upstream node"));
                }
            }
        }

      LintReport report;
      report.execution_waves = std::move (waves);
      report.total_nodes = n;
      report.total_edges = edges.size ();
      return report;
    }
  catch (...)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                                 "Out of memory during PlanLinter::validate"));
    }
}

/* Validates and executes the WorkflowGraph in parallel topological waves.  */
auto
WorkflowGraph::execute (const ToolContext &ctx, WorkStealingPool &pool,
                        std::string_view initial_input_json) const noexcept
    -> Result<std::unordered_map<std::string, std::string>>
{
  auto lint_res = PlanLinter::validate (*this);
  if (!lint_res)
    {
      return std::unexpected (lint_res.error ());
    }

  try
    {
      const std::size_t n = nodes_.size ();
      std::unordered_map<std::string, std::size_t> id_to_idx;
      id_to_idx.reserve (n);
      for (std::size_t i = 0U; i < n; ++i)
        {
          id_to_idx.emplace (nodes_.at (i).id, i);
        }

      std::vector<std::vector<std::size_t>> incoming (n);
      for (const WorkflowEdge &edge : edges_)
        {
          const std::size_t u = id_to_idx.at (edge.from_node);
          const std::size_t v = id_to_idx.at (edge.to_node);
          incoming.at (v).push_back (u);
        }

      std::vector<Result<std::string>> node_outputs (n);

      for (const std::vector<std::string> &wave : lint_res->execution_waves)
        {
          const std::size_t wave_size = wave.size ();
          std::vector<std::string> wave_inputs (wave_size);

          for (std::size_t w = 0U; w < wave_size; ++w)
            {
              const std::size_t node_idx = id_to_idx.at (wave.at (w));
              if (incoming.at (node_idx).empty ())
                {
                  wave_inputs.at (w) = std::string (initial_input_json);
                }
              else
                {
                  std::vector<std::string> up_payloads;
                  up_payloads.reserve (incoming.at (node_idx).size ());
                  for (const std::size_t u : incoming.at (node_idx))
                    {
                      up_payloads.push_back (*node_outputs.at (u));
                    }
                  wave_inputs.at (w)
                      = merge_upstream_json_outputs (up_payloads);
                }
            }

          if (wave_size == 1U)
            {
              const std::size_t node_idx = id_to_idx.at (wave.at (0U));
              const WorkflowNode &node = nodes_.at (node_idx);
              if (!node.action)
                {
                  node_outputs.at (node_idx) = wave_inputs.at (0U);
                }
              else
                {
                  node_outputs.at (node_idx)
                      = node.action (ctx, wave_inputs.at (0U));
                }
            }
          else
            {
              std::atomic<std::size_t> next_item{ 0U };
              std::atomic<std::size_t> tasks_done{ 0U };

              for (std::size_t w = 0U; w < wave_size; ++w)
                {
                  pool.submit ([this, &ctx, &id_to_idx, &wave, &wave_inputs,
                                &node_outputs, &next_item, &tasks_done,
                                wave_size] () noexcept {
                    for (;;)
                      {
                        const std::size_t idx = next_item.fetch_add (
                            1U, std::memory_order_relaxed);
                        if (idx >= wave_size)
                          {
                            break;
                          }
                        const std::size_t node_idx
                            = id_to_idx.at (wave.at (idx));
                        const WorkflowNode &node = nodes_.at (node_idx);
                        if (!node.action)
                          {
                            node_outputs.at (node_idx) = wave_inputs.at (idx);
                          }
                        else
                          {
                            node_outputs.at (node_idx)
                                = node.action (ctx, wave_inputs.at (idx));
                          }
                      }
                    tasks_done.fetch_add (1U, std::memory_order_release);
                  });
                }

              for (;;)
                {
                  const std::size_t idx
                      = next_item.fetch_add (1U, std::memory_order_relaxed);
                  if (idx >= wave_size)
                    {
                      break;
                    }
                  const std::size_t node_idx = id_to_idx.at (wave.at (idx));
                  const WorkflowNode &node = nodes_.at (node_idx);
                  if (!node.action)
                    {
                      node_outputs.at (node_idx) = wave_inputs.at (idx);
                    }
                  else
                    {
                      node_outputs.at (node_idx)
                          = node.action (ctx, wave_inputs.at (idx));
                    }
                }

              while (tasks_done.load (std::memory_order_acquire) < wave_size)
                {
                  if (!pool.try_run_one ())
                    {
                      std::this_thread::yield ();
                    }
                }
            }

          // Check whether any node in this wave returned an error.
          for (std::size_t w = 0U; w < wave_size; ++w)
            {
              const std::size_t node_idx = id_to_idx.at (wave.at (w));
              if (!node_outputs.at (node_idx).has_value ())
                {
                  return std::unexpected (node_outputs.at (node_idx).error ());
                }
            }
        }

      std::unordered_map<std::string, std::string> final_map;
      final_map.reserve (n);
      for (std::size_t i = 0U; i < n; ++i)
        {
          final_map.emplace (nodes_.at (i).id,
                             std::move (*node_outputs.at (i)));
        }
      return final_map;
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while executing WorkflowGraph"));
    }
}

} /* namespace sutra::inline v1 */
