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

#include "config.h"
#include <basalt/autograd.hpp>
#include <basalt/ops.hpp>
#include "tensor/tensor_internal.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace basalt
{

Saved_Variable::Saved_Variable (const Tensor &t, bool is_output)
  : data_ (t.defined () ? t : Tensor ()),
    saved_version_ (t.version ()),
    is_output_ (is_output)
{
}

Tensor
Saved_Variable::unpack (void) const
{
  if (!data_.defined ())
    {
      return Tensor ();
    }
  if (data_.version () != saved_version_)
    {
      BASALT_THROW (
        Autograd_Error,
        "one of the variables needed for gradient computation has been "
        "modified by an in-place operation");
    }
  return data_;
}

Autograd_Node::~Autograd_Node (void) noexcept = default;

namespace
{

bool &
tls_grad_enabled (void) noexcept
{
  thread_local bool enabled = true;
  return enabled;
}

/* Leaf sink node that accumulates incoming gradients into a leaf Tensor's
   .grad field.  */
class Accumulate_Grad : public Autograd_Node
{
public:
  explicit Accumulate_Grad (std::weak_ptr<Autograd_Meta> meta)
    : meta_ (std::move (meta))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Accumulate_Grad";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grad_outputs) override
  {
    auto m = meta_.lock ();
    if (m != nullptr && !grad_outputs.empty () && grad_outputs[0].defined ())
      {
        std::lock_guard<std::mutex> lock (m->mutex);
        if (m->grad.defined ())
          {
            m->grad = ops::add (m->grad, grad_outputs[0]);
          }
        else
          {
            m->grad = grad_outputs[0];
          }
      }
    return {};
  }

private:
  std::weak_ptr<Autograd_Meta> meta_;
};

/* Activation checkpointing backward node that re-runs the forward function
   under Enable_Grad_Guard during backward evaluation.  */
class Checkpoint_Node : public Autograd_Node
{
public:
  Checkpoint_Node (std::function<Tensor (const std::vector<Tensor> &)> fn,
                   std::vector<Saved_Variable> saved_inputs)
    : fn_ (std::move (fn)),
      saved_inputs_ (std::move (saved_inputs))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Checkpoint_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grad_outputs) override
  {
    bool prev_grad = Grad_Mode::is_enabled ();
    Enable_Grad_Guard enable (true);

    std::vector<Tensor> detached_inputs;
    detached_inputs.reserve (saved_inputs_.size ());
    for (const auto &sv : saved_inputs_)
      {
        Tensor inp = sv.unpack ().detach ();
        inp.set_requires_grad (true);
        detached_inputs.push_back (inp);
      }

    Tensor recomputed = fn_ (detached_inputs);
    return autograd::grad ({recomputed}, detached_inputs, grad_outputs,
                           prev_grad, prev_grad, true);
  }

  void
  release_saved_variables (void) noexcept override
  {
    saved_inputs_.clear ();
  }

private:
  std::function<Tensor (const std::vector<Tensor> &)> fn_;
  std::vector<Saved_Variable> saved_inputs_;
};

/* Build reverse topological ordering of reachable Autograd_Nodes starting
   from ROOTS.  */
void
topo_dfs (Autograd_Node *node,
          std::unordered_set<Autograd_Node *> &visited,
          std::vector<Autograd_Node *> &post_order)
{
  if (node == nullptr || visited.count (node) > 0)
    {
      return;
    }
  visited.insert (node);
  for (const Edge &edge : node->next_edges ())
    {
      if (edge.defined ())
        {
          topo_dfs (edge.function.get (), visited, post_order);
        }
    }
  post_order.push_back (node);
}

} /* namespace */

bool
Grad_Mode::is_enabled (void) noexcept
{
  return tls_grad_enabled ();
}

void
Grad_Mode::set_enabled (bool enabled) noexcept
{
  tls_grad_enabled () = enabled;
}

namespace autograd
{

Edge
gradient_edge (const Tensor &t)
{
  if (!t.defined () || !t.requires_grad ())
    {
      return Edge{};
    }
  if (t.grad_fn () != nullptr)
    {
      return Edge{t.grad_fn (), 0};
    }
  auto meta = t.autograd_meta ();
  if (meta != nullptr)
    {
      std::lock_guard<std::mutex> lock (meta->mutex);
      if (meta->grad_accumulator == nullptr)
        {
          meta->grad_accumulator
            = std::make_shared<Accumulate_Grad> (meta);
        }
      return Edge{meta->grad_accumulator, 0};
    }
  return Edge{};
}

std::vector<Edge>
collect_next_edges (const std::vector<Tensor> &inputs)
{
  std::vector<Edge> edges;
  edges.reserve (inputs.size ());
  for (const Tensor &t : inputs)
    {
      edges.push_back (gradient_edge (t));
    }
  return edges;
}

void
backward (const Tensor &root,
          const Tensor &grad_output,
          bool retain_graph,
          bool create_graph)
{
  BASALT_ASSERT (root.defined (), "backward() called on undefined Tensor");
  BASALT_ASSERT (root.requires_grad (),
                 "backward() called on Tensor that does not require_grad");

  Tensor initial_grad = grad_output;
  if (!initial_grad.defined ())
    {
      BASALT_ASSERT_SHAPE (
        root.numel () == 1,
        "backward() grad_output can be omitted only for scalar outputs");
      initial_grad = ones (root.shape (),
                           root.options ().with_requires_grad (create_graph));
    }

  Edge root_edge = gradient_edge (root);
  if (!root_edge.defined ())
    {
      return;
    }

  std::unordered_set<Autograd_Node *> visited;
  std::vector<Autograd_Node *> post_order;
  topo_dfs (root_edge.function.get (), visited, post_order);
  std::reverse (post_order.begin (), post_order.end ());

  std::unordered_map<Autograd_Node *, std::vector<Tensor>> ready_grads;
  ready_grads[root_edge.function.get ()] = {initial_grad};

  Enable_Grad_Guard grad_guard (create_graph);

  for (Autograd_Node *node : post_order)
    {
      auto it = ready_grads.find (node);
      if (it == ready_grads.end ())
        {
          continue;
        }
      const std::vector<Tensor> &node_out_grads = it->second;

      /* If the non-leaf tensor that produced NODE requested retain_grad(),
         record its accumulated gradient before propagating to inputs.  */
      auto out_meta = node->output_meta ();
      if (out_meta != nullptr && out_meta->retain_grad
          && !node_out_grads.empty () && node_out_grads[0].defined ())
        {
          std::lock_guard<std::mutex> lock (out_meta->mutex);
          if (out_meta->grad.defined ())
            {
              out_meta->grad = ops::add (out_meta->grad, node_out_grads[0]);
            }
          else
            {
              out_meta->grad = node_out_grads[0];
            }
        }

      std::vector<Tensor> input_grads = node->apply (node_out_grads);
      const std::vector<Edge> &edges = node->next_edges ();
      size_t count = std::min (input_grads.size (), edges.size ());

      for (size_t i = 0; i < count; ++i)
        {
          const Edge &edge = edges[i];
          const Tensor &ig = input_grads[i];
          if (!edge.defined () || !ig.defined ())
            {
              continue;
            }
          std::vector<Tensor> &slot_vec = ready_grads[edge.function.get ()];
          if (slot_vec.size () <= edge.input_nr)
            {
              slot_vec.resize (edge.input_nr + 1);
            }
          if (slot_vec[edge.input_nr].defined ())
            {
              slot_vec[edge.input_nr]
                = ops::add (slot_vec[edge.input_nr], ig);
            }
          else
            {
              slot_vec[edge.input_nr] = ig;
            }
        }

      if (!retain_graph && !create_graph)
        {
          node->release_saved_variables ();
        }
    }
}

std::vector<Tensor>
grad (const std::vector<Tensor> &outputs,
      const std::vector<Tensor> &inputs,
      const std::vector<Tensor> &grad_outputs,
      bool retain_graph,
      bool create_graph,
      bool allow_unused)
{
  BASALT_ASSERT_ARG (!outputs.empty (), "grad() outputs cannot be empty");
  BASALT_ASSERT_ARG (!inputs.empty (), "grad() inputs cannot be empty");

  std::vector<Tensor> init_grads (outputs.size ());
  if (grad_outputs.empty ())
    {
      for (size_t i = 0; i < outputs.size (); ++i)
        {
          BASALT_ASSERT_SHAPE (outputs[i].numel () == 1,
                               "grad() requires scalar outputs when "
                               "grad_outputs is omitted");
          init_grads[i] = ones (
            outputs[i].shape (),
            outputs[i].options ().with_requires_grad (create_graph));
        }
    }
  else
    {
      BASALT_ASSERT_ARG (grad_outputs.size () == outputs.size (),
                         "grad() grad_outputs size mismatch");
      init_grads = grad_outputs;
    }

  /* Map each target input's gradient edge function pointer to its index in
     INPUTS so we intercept gradients without mutating leaf .grad fields.  */
  std::unordered_map<Autograd_Node *, std::vector<size_t>> target_nodes;
  for (size_t i = 0; i < inputs.size (); ++i)
    {
      BASALT_ASSERT_ARG (inputs[i].requires_grad (),
                         "grad() input does not require_grad");
      Edge e = gradient_edge (inputs[i]);
      if (e.defined ())
        {
          target_nodes[e.function.get ()].push_back (i);
        }
    }

  std::unordered_set<Autograd_Node *> visited;
  std::vector<Autograd_Node *> post_order;
  std::unordered_map<Autograd_Node *, std::vector<Tensor>> ready_grads;

  for (size_t i = 0; i < outputs.size (); ++i)
    {
      Edge e = gradient_edge (outputs[i]);
      if (e.defined ())
        {
          topo_dfs (e.function.get (), visited, post_order);
          std::vector<Tensor> &slot_vec = ready_grads[e.function.get ()];
          if (slot_vec.empty ())
            {
              slot_vec.push_back (init_grads[i]);
            }
          else
            {
              slot_vec[0] = ops::add (slot_vec[0], init_grads[i]);
            }
        }
    }
  std::reverse (post_order.begin (), post_order.end ());

  std::vector<Tensor> result (inputs.size ());
  Enable_Grad_Guard grad_guard (create_graph);

  for (Autograd_Node *node : post_order)
    {
      auto it = ready_grads.find (node);
      if (it == ready_grads.end ())
        {
          continue;
        }
      const std::vector<Tensor> &node_out_grads = it->second;

      auto t_it = target_nodes.find (node);
      if (t_it != target_nodes.end () && !node_out_grads.empty ()
          && node_out_grads[0].defined ())
        {
          for (size_t inp_idx : t_it->second)
            {
              if (result[inp_idx].defined ())
                {
                  result[inp_idx]
                    = ops::add (result[inp_idx], node_out_grads[0]);
                }
              else
                {
                  result[inp_idx] = node_out_grads[0];
                }
            }
        }

      /* Do not invoke Accumulate_Grad::apply during autograd::grad so leaf
         .grad() attributes remain untouched.  */
      if (dynamic_cast<Accumulate_Grad *> (node) != nullptr)
        {
          continue;
        }

      std::vector<Tensor> input_grads = node->apply (node_out_grads);
      const std::vector<Edge> &edges = node->next_edges ();
      size_t count = std::min (input_grads.size (), edges.size ());

      for (size_t i = 0; i < count; ++i)
        {
          const Edge &edge = edges[i];
          const Tensor &ig = input_grads[i];
          if (!edge.defined () || !ig.defined ())
            {
              continue;
            }
          std::vector<Tensor> &slot_vec = ready_grads[edge.function.get ()];
          if (slot_vec.size () <= edge.input_nr)
            {
              slot_vec.resize (edge.input_nr + 1);
            }
          if (slot_vec[edge.input_nr].defined ())
            {
              slot_vec[edge.input_nr]
                = ops::add (slot_vec[edge.input_nr], ig);
            }
          else
            {
              slot_vec[edge.input_nr] = ig;
            }
        }

      if (!retain_graph && !create_graph)
        {
          node->release_saved_variables ();
        }
    }

  for (size_t i = 0; i < inputs.size (); ++i)
    {
      if (!result[i].defined ())
        {
          if (!allow_unused)
            {
              BASALT_THROW (Autograd_Error,
                            "grad() input was not used in computing output");
            }
          result[i] = zeros_like (inputs[i]);
        }
    }
  return result;
}

Tensor
checkpoint (std::function<Tensor (const std::vector<Tensor> &)> fn,
            const std::vector<Tensor> &inputs)
{
  bool any_req = false;
  std::vector<Saved_Variable> saved;
  saved.reserve (inputs.size ());
  for (const Tensor &t : inputs)
    {
      any_req = any_req || t.requires_grad ();
      saved.emplace_back (t, false);
    }

  Tensor out;
  {
    No_Grad_Guard no_grad;
    out = fn (inputs);
  }

  if (Grad_Mode::is_enabled () && any_req
      && is_floating_point (out.dtype ()))
    {
      auto node = std::make_shared<Checkpoint_Node> (
        std::move (fn), std::move (saved));
      node->set_next_edges (collect_next_edges (inputs));
      out.set_requires_grad (true);
      out.set_grad_fn (std::move (node));
    }
  return out;
}

bool
gradcheck (std::function<Tensor (const std::vector<Tensor> &)> fn,
           const std::vector<Tensor> &inputs,
           double eps,
           double atol,
           double rtol)
{
  std::vector<Tensor> clean_inputs;
  clean_inputs.reserve (inputs.size ());
  for (const Tensor &inp : inputs)
    {
      Tensor c = inp.clone ().detach ();
      c.set_requires_grad (inp.requires_grad ());
      clean_inputs.push_back (c);
    }

  Tensor out;
  {
    Enable_Grad_Guard enable (true);
    out = fn (clean_inputs);
    if (out.numel () > 1)
      {
        out = ops::sum (out);
      }
  }

  std::vector<Tensor> analytical = grad ({out}, clean_inputs, {},
                                         false, false, true);

  No_Grad_Guard no_grad;
  for (size_t idx = 0; idx < clean_inputs.size (); ++idx)
    {
      if (!clean_inputs[idx].requires_grad ())
        {
          continue;
        }
      Tensor x = clean_inputs[idx];
      const Tensor &ag = analytical[idx];
      size_t n = x.numel ();
      for (size_t elem = 0; elem < n; ++elem)
        {
          double orig = x.read_scalar_at (elem);

          x.write_scalar_at (elem, orig + eps);
          Tensor out_pos = fn (clean_inputs);
          if (out_pos.numel () > 1)
            {
              out_pos = ops::sum (out_pos);
            }
          double f_pos = out_pos.item_double ();

          x.write_scalar_at (elem, orig - eps);
          Tensor out_neg = fn (clean_inputs);
          if (out_neg.numel () > 1)
            {
              out_neg = ops::sum (out_neg);
            }
          double f_neg = out_neg.item_double ();

          x.write_scalar_at (elem, orig);

          double num_grad = (f_pos - f_neg) / (2.0 * eps);
          double ana_grad = ag.read_scalar_at (elem);
          double diff = std::fabs (num_grad - ana_grad);
          double tol = atol + rtol * std::max (std::fabs (num_grad),
                                               std::fabs (ana_grad));
          if (diff > tol)
            {
              return false;
            }
        }
    }
  return true;
}

bool
gradgradcheck (std::function<Tensor (const std::vector<Tensor> &)> fn,
               const std::vector<Tensor> &inputs,
               double eps,
               double atol,
               double rtol)
{
  auto first_grad_sum = [fn] (const std::vector<Tensor> &inps) -> Tensor
    {
      Enable_Grad_Guard enable (true);
      Tensor y = fn (inps);
      if (y.numel () > 1)
        {
          y = ops::sum (y);
        }
      std::vector<Tensor> g1 = grad ({y}, inps, {}, true, true, true);
      Tensor total = ops::sum (g1[0]);
      for (size_t i = 1; i < g1.size (); ++i)
        {
          total = ops::add (total, ops::sum (g1[i]));
        }
      return total;
    };

  return gradcheck (first_grad_sum, inputs, eps, atol, rtol);
}

} /* namespace autograd */
} /* namespace basalt */
