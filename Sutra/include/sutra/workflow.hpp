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

#ifndef SUTRA_WORKFLOW_HPP
#define SUTRA_WORKFLOW_HPP

#include <sutra/error.hpp>
#include <sutra/scheduler.hpp>
#include <sutra/tool.hpp>

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sutra::inline v1
{

/**
 * @brief Callable signature for a workflow stage node.
 */
using WorkflowNodeFn = std::function<Result<std::string> (const ToolContext &,
                                                          std::string_view)>;

/**
 * @brief Represents an agent or tool stage within a WorkflowGraph.
 * @note Thread-safety: Value type.
 */
struct WorkflowNode
{
  std::string id;
  std::string input_schema_json
      = R"({"type":"object","properties":{},"required":[]})";
  std::string output_schema_json
      = R"({"type":"object","properties":{},"required":[]})";
  WorkflowNodeFn action;
};

/**
 * @brief Represents a directed data-dependency edge between two workflow
 * nodes.
 * @note Thread-safety: Value type.
 */
struct WorkflowEdge
{
  std::string from_node;
  std::string to_node;
};

/**
 * @brief Result produced by PlanLinter upon successful static DAG validation.
 * @note Thread-safety: Value type.
 */
struct LintReport
{
  std::vector<std::vector<std::string>> execution_waves;
  std::size_t total_nodes = 0U;
  std::size_t total_edges = 0U;
};

/**
 * @brief Directed acyclic workflow graph orchestrating multi-agent/tool
 * stages.
 * @pre Graph must pass PlanLinter::validate() prior to execution.
 * @post Independent branches within each topological wave execute in parallel.
 * @note Thread-safety: Const execution methods are safe for concurrent runs.
 */
class WorkflowGraph
{
public:
  WorkflowGraph () = default;

  /**
   * @brief Adds a stage node to the workflow graph.
   * @param node WorkflowNode with non-empty id and valid schemas.
   * @pre !node.id.empty().
   * @post Node is appended to the graph definition.
   * @returns Result<void> or PlanningError on duplicate or empty node ID.
   */
  [[nodiscard]] auto add_node (WorkflowNode node) noexcept -> Result<void>;

  /**
   * @brief Adds a directed dependency edge from from_node to to_node.
   * @param from_node Upstream producer node ID.
   * @param to_node Downstream consumer node ID.
   * @pre !from_node.empty() and !to_node.empty().
   * @post Edge is recorded for validation by PlanLinter.
   * @returns Result<void> or PlanningError on empty endpoint ID.
   */
  [[nodiscard]] auto add_edge (std::string from_node,
                               std::string to_node) noexcept -> Result<void>;

  /**
   * @brief Returns a read-only span of all nodes in the graph.
   * @returns Span of WorkflowNode entries.
   */
  [[nodiscard]] auto nodes () const noexcept -> std::span<const WorkflowNode>;

  /**
   * @brief Returns a read-only span of all directed edges in the graph.
   * @returns Span of WorkflowEdge entries.
   */
  [[nodiscard]] auto edges () const noexcept -> std::span<const WorkflowEdge>;

  /**
   * @brief Validates and executes the workflow graph across WorkStealingPool.
   * @param ctx Zero-copy ToolContext passed to each node stage.
   * @param pool Work-stealing thread pool for parallel branch execution.
   * @param initial_input_json JSON payload supplied to root (in-degree 0)
   * nodes.
   * @pre Graph must be a valid DAG with compatible schemas.
   * @post Returns map from node ID to its output JSON payload.
   * @returns Result<std::unordered_map<std::string, std::string>> or
   * AgentError.
   */
  [[nodiscard]] auto execute (const ToolContext &ctx, WorkStealingPool &pool,
                              std::string_view initial_input_json = "{}")
      const noexcept -> Result<std::unordered_map<std::string, std::string>>;

private:
  std::vector<WorkflowNode> nodes_;
  std::vector<WorkflowEdge> edges_;
};

/**
 * @brief Static verifier catching cycles, dangling edges, and schema
 * mismatches.
 * @note Thread-safety: Stateless validator; safe for concurrent calls.
 */
class PlanLinter
{
public:
  /**
   * @brief Validates a WorkflowGraph for structural and schema correctness.
   * @param graph The WorkflowGraph to lint.
   * @pre None.
   * @post Guarantees absence of cycles, dangling edges, and schema mismatches.
   * @returns Result<LintReport> with parallel topological waves or
   * PlanningError.
   * @note Thread-safety: Pure static method; thread-safe.
   */
  [[nodiscard]] static auto
  validate (const WorkflowGraph &graph) noexcept -> Result<LintReport>;
};

} /* namespace sutra::inline v1 */

#endif /* SUTRA_WORKFLOW_HPP */
