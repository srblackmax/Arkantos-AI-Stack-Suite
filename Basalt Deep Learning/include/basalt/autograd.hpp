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

#ifndef BASALT_AUTOGRAD_HPP
#define BASALT_AUTOGRAD_HPP

/* Dynamic reverse-mode automatic differentiation engine supporting arbitrary
   higher-order gradients (create_graph), thread-local no_grad and enable_grad
   RAII scopes, activation checkpointing, and numerical finite-difference
   gradient verification.  */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/tensor.hpp>

namespace basalt
{

class Autograd_Node;

/* Directed edge in the reverse-mode computational DAG connecting an operation
   output to a predecessor Autograd_Node and its input slot index.  */
struct BASALT_API Edge
{
  std::shared_ptr<Autograd_Node> function;
  size_t input_nr = 0;

  bool
  defined (void) const noexcept
  {
    return function != nullptr;
  }
};

/* Snapshot of a Tensor saved for backward evaluation that verifies its
   in-place mutation version counter upon unpacking.  */
class BASALT_API Saved_Variable
{
public:
  Saved_Variable (void) noexcept = default;
  explicit Saved_Variable (const Tensor &t, bool is_output = false);

  Tensor
  unpack (void) const;

  bool
  defined (void) const noexcept
  {
    return data_.defined ();
  }

private:
  Tensor data_;
  uint64_t saved_version_ = 0;
  bool is_output_ = false;
};

/* Base class for all differentiable nodes in the dynamic computation graph.
   Subclasses implement apply() using differentiable Tensor operations so
   executing backward with create_graph = true constructs a second-order
   graph for higher-order derivatives.  */
class BASALT_API Autograd_Node
  : public std::enable_shared_from_this<Autograd_Node>
{
public:
  Autograd_Node (void) = default;
  virtual ~Autograd_Node (void) noexcept;

  virtual std::vector<Tensor>
  apply (const std::vector<Tensor> &grad_outputs) = 0;

  virtual const char *
  name (void) const noexcept
  {
    return "Autograd_Node";
  }

  void
  set_next_edges (std::vector<Edge> edges)
  {
    next_edges_ = std::move (edges);
  }

  const std::vector<Edge> &
  next_edges (void) const noexcept
  {
    return next_edges_;
  }

  size_t
  num_inputs (void) const noexcept
  {
    return next_edges_.size ();
  }

  virtual void
  release_saved_variables (void) noexcept
  {
  }

  void
  set_output_meta (std::weak_ptr<Autograd_Meta> meta) noexcept
  {
    output_meta_ = std::move (meta);
  }

  std::shared_ptr<Autograd_Meta>
  output_meta (void) const noexcept
  {
    return output_meta_.lock ();
  }

private:
  std::vector<Edge> next_edges_;
  std::weak_ptr<Autograd_Meta> output_meta_;
};

/* Thread-local gradient recording state.  */
class BASALT_API Grad_Mode
{
public:
  static bool
  is_enabled (void) noexcept;

  static void
  set_enabled (bool enabled) noexcept;
};

/* RAII scope that disables gradient tracking while active.  */
class BASALT_API No_Grad_Guard
{
public:
  No_Grad_Guard (void) noexcept
    : prev_ (Grad_Mode::is_enabled ())
  {
    Grad_Mode::set_enabled (false);
  }

  ~No_Grad_Guard (void) noexcept
  {
    Grad_Mode::set_enabled (prev_);
  }

  No_Grad_Guard (const No_Grad_Guard &) = delete;
  No_Grad_Guard &operator= (const No_Grad_Guard &) = delete;

private:
  bool prev_;
};

/* RAII scope that enables gradient tracking while active.  */
class BASALT_API Enable_Grad_Guard
{
public:
  explicit Enable_Grad_Guard (bool enable = true) noexcept
    : prev_ (Grad_Mode::is_enabled ())
  {
    Grad_Mode::set_enabled (enable);
  }

  ~Enable_Grad_Guard (void) noexcept
  {
    Grad_Mode::set_enabled (prev_);
  }

  Enable_Grad_Guard (const Enable_Grad_Guard &) = delete;
  Enable_Grad_Guard &operator= (const Enable_Grad_Guard &) = delete;

private:
  bool prev_;
};

namespace autograd
{

/* Compute the edge connecting T to its gradient producer (creating an
   Accumulate_Grad sink node on demand for leaf tensors).  */
BASALT_API Edge
gradient_edge (const Tensor &t);

/* Collect gradient edges for a parameter pack or list of input tensors.  */
BASALT_API std::vector<Edge>
collect_next_edges (const std::vector<Tensor> &inputs);

/* Execute reverse-mode backpropagation from ROOT.  */
BASALT_API void
backward (const Tensor &root,
          const Tensor &grad_output = Tensor (),
          bool retain_graph = false,
          bool create_graph = false);

/* Compute and return gradients of OUTPUTS with respect to INPUTS without
   mutating leaf .grad() fields.  Setting create_graph = true makes the
   returned gradients differentiable for higher-order derivatives.  */
BASALT_API std::vector<Tensor>
grad (const std::vector<Tensor> &outputs,
      const std::vector<Tensor> &inputs,
      const std::vector<Tensor> &grad_outputs = {},
      bool retain_graph = false,
      bool create_graph = false,
      bool allow_unused = false);

/* Trade compute for memory by running FN without saving intermediate
   activations during the forward pass and recomputing the subgraph on the
   fly during the backward pass.  */
BASALT_API Tensor
checkpoint (std::function<Tensor (const std::vector<Tensor> &)> fn,
            const std::vector<Tensor> &inputs);

/* Numerical gradient verification against central finite differences.  */
BASALT_API bool
gradcheck (std::function<Tensor (const std::vector<Tensor> &)> fn,
           const std::vector<Tensor> &inputs,
           double eps = 1e-5,
           double atol = 1e-4,
           double rtol = 1e-3);

/* Numerical verification of second-order derivatives against finite
   differences applied to analytical first derivatives.  */
BASALT_API bool
gradgradcheck (std::function<Tensor (const std::vector<Tensor> &)> fn,
               const std::vector<Tensor> &inputs,
               double eps = 1e-4,
               double atol = 1e-3,
               double rtol = 1e-2);

} /* namespace autograd */

/* Convenience aliases in namespace basalt.  */
using autograd::backward;
using autograd::checkpoint;
using autograd::grad;
using autograd::gradcheck;
using autograd::gradgradcheck;

} /* namespace basalt */

#endif /* BASALT_AUTOGRAD_HPP */
