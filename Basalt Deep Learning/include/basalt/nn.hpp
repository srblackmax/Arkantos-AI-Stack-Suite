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

#ifndef BASALT_NN_HPP
#define BASALT_NN_HPP

/* Stateful neural network module hierarchy (Linear, Conv1d/2d/3d, pooling,
   Batch/Layer/Group/RMS normalization, activations, Multihead_Attention,
   Transformer blocks, Embedding, Dropout, and Sequential containers).  */

#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/functional.hpp>
#include <basalt/tensor.hpp>

namespace basalt::nn
{

using State_Dict = std::map<std::string, Tensor>;

namespace init
{

BASALT_API Tensor &
zeros_ (Tensor &tensor);

BASALT_API Tensor &
ones_ (Tensor &tensor);

BASALT_API Tensor &
constant_ (Tensor &tensor, double val);

BASALT_API Tensor &
uniform_ (Tensor &tensor, double low = 0.0, double high = 1.0);

BASALT_API Tensor &
normal_ (Tensor &tensor, double mean = 0.0, double std_dev = 1.0);

BASALT_API Tensor &
xavier_uniform_ (Tensor &tensor, double gain = 1.0);

BASALT_API Tensor &
xavier_normal_ (Tensor &tensor, double gain = 1.0);

BASALT_API Tensor &
kaiming_uniform_ (Tensor &tensor, double a = 0.0, bool fan_in_mode = true);

BASALT_API Tensor &
kaiming_normal_ (Tensor &tensor, double a = 0.0, bool fan_in_mode = true);

BASALT_API Tensor &
orthogonal_ (Tensor &tensor, double gain = 1.0);

} /* namespace init */

/* Base class for all neural network layers and composite models.
   Provides recursive parameter, buffer, and submodule registration,
   training/evaluation mode propagation, device/dtype transfer, and
   state_dict serialization.  */
class BASALT_API Module : public std::enable_shared_from_this<Module>
{
public:
  Module (void) = default;
  virtual ~Module (void) noexcept;

  virtual Tensor
  forward (const Tensor &input);

  Tensor
  operator() (const Tensor &input)
  {
    return forward (input);
  }

  Tensor &
  register_parameter (const std::string &name, Tensor tensor,
                      bool requires_grad = true);

  Tensor &
  register_buffer (const std::string &name, Tensor tensor);

  template <typename SubmoduleT>
  std::shared_ptr<SubmoduleT>
  register_module (const std::string &name,
                   std::shared_ptr<SubmoduleT> submodule)
  {
    register_module_base (name, submodule);
    return submodule;
  }

  std::vector<Tensor>
  parameters (bool recurse = true) const;

  std::vector<std::pair<std::string, Tensor>>
  named_parameters (bool recurse = true,
                    const std::string &prefix = "") const;

  std::vector<Tensor>
  buffers (bool recurse = true) const;

  std::vector<std::pair<std::string, Tensor>>
  named_buffers (bool recurse = true,
                 const std::string &prefix = "") const;

  State_Dict
  state_dict (const std::string &prefix = "") const;

  void
  load_state_dict (const State_Dict &dict, bool strict = true);

  virtual void
  train (bool mode = true);

  void
  eval (void)
  {
    train (false);
  }

  bool
  is_training (void) const noexcept
  {
    return training_;
  }

  virtual void
  to (Device device);

  virtual void
  to (Dtype dtype);

  virtual void
  to (Device device, Dtype dtype);

  void
  zero_grad (void);

  size_t
  num_parameters (bool only_trainable = true) const;

protected:
  void
  register_module_base (const std::string &name,
                        std::shared_ptr<Module> submodule);

  bool training_ = true;
  std::vector<std::pair<std::string, Tensor>> parameters_;
  std::vector<std::pair<std::string, Tensor>> buffers_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
};

/* Ordered container that chains a sequence of Modules in forward().  */
class BASALT_API Sequential : public Module
{
public:
  Sequential (void) = default;
  Sequential (std::initializer_list<std::shared_ptr<Module>> modules);
  explicit Sequential (std::vector<std::shared_ptr<Module>> modules);

  void
  add (std::shared_ptr<Module> module);

  void
  add (const std::string &name, std::shared_ptr<Module> module);

  size_t
  size (void) const noexcept
  {
    return layers_.size ();
  }

  std::shared_ptr<Module>
  operator[] (size_t index) const
  {
    return layers_.at (index);
  }

  Tensor
  forward (const Tensor &input) override;

private:
  std::vector<std::shared_ptr<Module>> layers_;
};

/* Indexed list of registered submodules.  */
class BASALT_API Module_List : public Module
{
public:
  Module_List (void) = default;
  explicit Module_List (std::vector<std::shared_ptr<Module>> modules);

  void
  append (std::shared_ptr<Module> module);

  size_t
  size (void) const noexcept
  {
    return modules_.size ();
  }

  std::shared_ptr<Module>
  operator[] (size_t index) const
  {
    return modules_.at (index);
  }

private:
  std::vector<std::shared_ptr<Module>> modules_;
};

using ModuleList = Module_List;

/* String-keyed dictionary of registered submodules.  */
class BASALT_API Module_Dict : public Module
{
public:
  Module_Dict (void) = default;

  void
  insert (const std::string &key, std::shared_ptr<Module> module);

  std::shared_ptr<Module>
  at (const std::string &key) const;

  bool
  contains (const std::string &key) const;

private:
  std::map<std::string, std::shared_ptr<Module>> dict_;
};

using ModuleDict = Module_Dict;

/* Affine linear transformation y = x * W^T + b.  */
class BASALT_API Linear : public Module
{
public:
  Linear (size_t in_features, size_t out_features, bool use_bias = true,
          Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  size_t
  in_features (void) const noexcept
  {
    return in_features_;
  }

  size_t
  out_features (void) const noexcept
  {
    return out_features_;
  }

  Tensor weight;
  Tensor bias;

private:
  size_t in_features_;
  size_t out_features_;
  bool use_bias_;
};

/* 1-D convolution over (N, C_in, L) inputs.  */
class BASALT_API Conv1d : public Module
{
public:
  Conv1d (size_t in_channels, size_t out_channels, size_t kernel_size,
          size_t stride = 1, size_t padding = 0,
          size_t dilation = 1, size_t groups = 1, bool use_bias = true,
          Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;

private:
  size_t stride_;
  size_t padding_;
  size_t dilation_;
  size_t groups_;
};

/* 2-D convolution over (N, C_in, H, W) inputs.  */
class BASALT_API Conv2d : public Module
{
public:
  Conv2d (size_t in_channels, size_t out_channels, size_t kernel_size,
          size_t stride = 1, size_t padding = 0,
          size_t dilation = 1, size_t groups = 1, bool use_bias = true,
          Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;

private:
  size_t stride_;
  size_t padding_;
  size_t dilation_;
  size_t groups_;
};

/* 3-D convolution over (N, C_in, D, H, W) inputs.  */
class BASALT_API Conv3d : public Module
{
public:
  Conv3d (size_t in_channels, size_t out_channels, size_t kernel_size,
          size_t stride = 1, size_t padding = 0,
          size_t dilation = 1, size_t groups = 1, bool use_bias = true,
          Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;

private:
  size_t stride_;
  size_t padding_;
  size_t dilation_;
  size_t groups_;
};

/* Spatial pooling modules.  */
class BASALT_API Max_Pool1d : public Module
{
public:
  explicit Max_Pool1d (size_t kernel_size, size_t stride = 0,
                       size_t padding = 0);

  Tensor
  forward (const Tensor &input) override;

private:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
};

class BASALT_API Max_Pool2d : public Module
{
public:
  explicit Max_Pool2d (size_t kernel_size, size_t stride = 0,
                       size_t padding = 0);

  Tensor
  forward (const Tensor &input) override;

private:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
};

class BASALT_API Max_Pool3d : public Module
{
public:
  explicit Max_Pool3d (size_t kernel_size, size_t stride = 0,
                       size_t padding = 0);

  Tensor
  forward (const Tensor &input) override;

private:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
};

class BASALT_API Avg_Pool1d : public Module
{
public:
  explicit Avg_Pool1d (size_t kernel_size, size_t stride = 0,
                       size_t padding = 0);

  Tensor
  forward (const Tensor &input) override;

private:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
};

class BASALT_API Avg_Pool2d : public Module
{
public:
  explicit Avg_Pool2d (size_t kernel_size, size_t stride = 0,
                       size_t padding = 0);

  Tensor
  forward (const Tensor &input) override;

private:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
};

class BASALT_API Avg_Pool3d : public Module
{
public:
  explicit Avg_Pool3d (size_t kernel_size, size_t stride = 0,
                       size_t padding = 0);

  Tensor
  forward (const Tensor &input) override;

private:
  size_t kernel_size_;
  size_t stride_;
  size_t padding_;
};

class BASALT_API Adaptive_Avg_Pool2d : public Module
{
public:
  Adaptive_Avg_Pool2d (size_t output_h, size_t output_w);
  explicit Adaptive_Avg_Pool2d (size_t output_size);

  Tensor
  forward (const Tensor &input) override;

private:
  size_t output_h_;
  size_t output_w_;
};

class BASALT_API Flatten : public Module
{
public:
  explicit Flatten (ptrdiff_t start_dim = 1, ptrdiff_t end_dim = -1);

  Tensor
  forward (const Tensor &input) override;

private:
  ptrdiff_t start_dim_;
  ptrdiff_t end_dim_;
};

using MaxPool1d = Max_Pool1d;
using MaxPool2d = Max_Pool2d;
using MaxPool3d = Max_Pool3d;
using AvgPool1d = Avg_Pool1d;
using AvgPool2d = Avg_Pool2d;
using AvgPool3d = Avg_Pool3d;
using AdaptiveAvgPool2d = Adaptive_Avg_Pool2d;

/* Normalization layers: Batch, Layer, Group, and RMS normalization.  */
class BASALT_API Batch_Norm : public Module
{
public:
  explicit Batch_Norm (size_t num_features, double eps = 1e-5,
                       double momentum = 0.1, bool affine = true,
                       Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;
  Tensor running_mean;
  Tensor running_var;

private:
  double eps_;
  double momentum_;
};

using Batch_Norm1d = Batch_Norm;
using Batch_Norm2d = Batch_Norm;
using Batch_Norm3d = Batch_Norm;
using BatchNorm1d = Batch_Norm;
using BatchNorm2d = Batch_Norm;
using BatchNorm3d = Batch_Norm;

class BASALT_API Layer_Norm : public Module
{
public:
  explicit Layer_Norm (Shape_Vector normalized_shape, double eps = 1e-5,
                       bool elementwise_affine = true,
                       Tensor_Options options = Tensor_Options ());

  explicit Layer_Norm (size_t normalized_dim, double eps = 1e-5,
                       bool elementwise_affine = true,
                       Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;

private:
  Shape_Vector normalized_shape_;
  double eps_;
};

using LayerNorm = Layer_Norm;

class BASALT_API Group_Norm : public Module
{
public:
  Group_Norm (size_t num_groups, size_t num_channels, double eps = 1e-5,
              bool affine = true,
              Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;

private:
  size_t num_groups_;
  size_t num_channels_;
  double eps_;
};

using GroupNorm = Group_Norm;

class BASALT_API RMS_Norm : public Module
{
public:
  explicit RMS_Norm (Shape_Vector normalized_shape, double eps = 1e-6,
                     bool elementwise_affine = true,
                     Tensor_Options options = Tensor_Options ());

  explicit RMS_Norm (size_t normalized_dim, double eps = 1e-6,
                     bool elementwise_affine = true,
                     Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;

private:
  Shape_Vector normalized_shape_;
  double eps_;
};

using RMSNorm = RMS_Norm;

/* Activation modules.  */
class BASALT_API ReLU : public Module
{
public:
  Tensor
  forward (const Tensor &input) override;
};

class BASALT_API Leaky_ReLU : public Module
{
public:
  explicit Leaky_ReLU (double negative_slope = 0.01);

  Tensor
  forward (const Tensor &input) override;

private:
  double negative_slope_;
};

using LeakyReLU = Leaky_ReLU;

class BASALT_API ELU : public Module
{
public:
  explicit ELU (double alpha = 1.0);

  Tensor
  forward (const Tensor &input) override;

private:
  double alpha_;
};

class BASALT_API GELU : public Module
{
public:
  Tensor
  forward (const Tensor &input) override;
};

class BASALT_API SiLU : public Module
{
public:
  Tensor
  forward (const Tensor &input) override;
};

class BASALT_API Mish : public Module
{
public:
  Tensor
  forward (const Tensor &input) override;
};

class BASALT_API Sigmoid : public Module
{
public:
  Tensor
  forward (const Tensor &input) override;
};

class BASALT_API Tanh : public Module
{
public:
  Tensor
  forward (const Tensor &input) override;
};

class BASALT_API Softmax : public Module
{
public:
  explicit Softmax (ptrdiff_t dim = -1);

  Tensor
  forward (const Tensor &input) override;

private:
  ptrdiff_t dim_;
};

class BASALT_API Log_Softmax : public Module
{
public:
  explicit Log_Softmax (ptrdiff_t dim = -1);

  Tensor
  forward (const Tensor &input) override;

private:
  ptrdiff_t dim_;
};

using LogSoftmax = Log_Softmax;

/* Embedding table and Dropout regularization.  */
class BASALT_API Embedding : public Module
{
public:
  Embedding (size_t num_embeddings, size_t embedding_dim,
             ptrdiff_t padding_idx = -1,
             Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &indices) override;

  Tensor weight;

private:
  size_t num_embeddings_;
  size_t embedding_dim_;
  ptrdiff_t padding_idx_;
};

class BASALT_API Dropout : public Module
{
public:
  explicit Dropout (double p = 0.5);

  Tensor
  forward (const Tensor &input) override;

private:
  double p_;
};

using Dropout2d = Dropout;

/* Multi-head scaled dot-product attention module.
   Accepts (batch, seq_len, embed_dim) inputs.  */
class BASALT_API Multihead_Attention : public Module
{
public:
  Multihead_Attention (size_t embed_dim, size_t num_heads,
                       double dropout_p = 0.0, bool use_bias = true,
                       Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor
  forward (const Tensor &query, const Tensor &key, const Tensor &value,
           const Tensor &attn_mask = Tensor (), bool is_causal = false);

  std::shared_ptr<Linear> q_proj;
  std::shared_ptr<Linear> k_proj;
  std::shared_ptr<Linear> v_proj;
  std::shared_ptr<Linear> out_proj;

private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;
  double dropout_p_;
};

using MultiheadAttention = Multihead_Attention;

/* Pre-norm / Post-norm Transformer Encoder Layer.  */
class BASALT_API Transformer_Encoder_Layer : public Module
{
public:
  Transformer_Encoder_Layer (size_t d_model, size_t nhead,
                             size_t dim_feedforward = 2048,
                             double dropout_p = 0.1,
                             bool norm_first = true,
                             Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &src) override;

  Tensor
  forward (const Tensor &src, const Tensor &src_mask, bool is_causal = false);

  std::shared_ptr<Multihead_Attention> self_attn;
  std::shared_ptr<Linear> linear1;
  std::shared_ptr<Linear> linear2;
  std::shared_ptr<Layer_Norm> norm1;
  std::shared_ptr<Layer_Norm> norm2;
  std::shared_ptr<Dropout> dropout;

private:
  bool norm_first_;
};

using TransformerEncoderLayer = Transformer_Encoder_Layer;

/* Transformer Decoder Layer with self-attention and cross-attention.  */
class BASALT_API Transformer_Decoder_Layer : public Module
{
public:
  Transformer_Decoder_Layer (size_t d_model, size_t nhead,
                             size_t dim_feedforward = 2048,
                             double dropout_p = 0.1,
                             bool norm_first = true,
                             Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &tgt) override;

  Tensor
  forward (const Tensor &tgt, const Tensor &memory,
           const Tensor &tgt_mask = Tensor (),
           const Tensor &memory_mask = Tensor (),
           bool tgt_is_causal = false);

  std::shared_ptr<Multihead_Attention> self_attn;
  std::shared_ptr<Multihead_Attention> multihead_attn;
  std::shared_ptr<Linear> linear1;
  std::shared_ptr<Linear> linear2;
  std::shared_ptr<Layer_Norm> norm1;
  std::shared_ptr<Layer_Norm> norm2;
  std::shared_ptr<Layer_Norm> norm3;
  std::shared_ptr<Dropout> dropout;

private:
  bool norm_first_;
};

using TransformerDecoderLayer = Transformer_Decoder_Layer;

/* Stack of N Transformer_Encoder_Layer blocks.  */
class BASALT_API Transformer_Encoder : public Module
{
public:
  Transformer_Encoder (size_t d_model, size_t nhead, size_t num_layers,
                       size_t dim_feedforward = 2048,
                       double dropout_p = 0.1,
                       bool norm_first = true,
                       Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &src) override;

  Tensor
  forward (const Tensor &src, const Tensor &mask, bool is_causal = false);

private:
  std::vector<std::shared_ptr<Transformer_Encoder_Layer>> layers_;
  std::shared_ptr<Layer_Norm> final_norm_;
};

using TransformerEncoder = Transformer_Encoder;

/* Stack of N Transformer_Decoder_Layer blocks.  */
class BASALT_API Transformer_Decoder : public Module
{
public:
  Transformer_Decoder (size_t d_model, size_t nhead, size_t num_layers,
                       size_t dim_feedforward = 2048,
                       double dropout_p = 0.1,
                       bool norm_first = true,
                       Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &tgt) override;

  Tensor
  forward (const Tensor &tgt, const Tensor &memory,
           const Tensor &tgt_mask = Tensor (),
           const Tensor &memory_mask = Tensor (),
           bool tgt_is_causal = false);

private:
  std::vector<std::shared_ptr<Transformer_Decoder_Layer>> layers_;
  std::shared_ptr<Layer_Norm> final_norm_;
};

using TransformerDecoder = Transformer_Decoder;

/* Recurrent sequence models: Elman RNN, LSTM, and GRU.  */
class BASALT_API RNN : public Module
{
public:
  RNN (size_t input_size, size_t hidden_size, size_t num_layers = 1,
       bool batch_first = true, double dropout_p = 0.0,
       Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  std::pair<Tensor, Tensor>
  forward_with_state (const Tensor &input, const Tensor &h0 = Tensor ());

private:
  size_t input_size_;
  size_t hidden_size_;
  size_t num_layers_;
  bool batch_first_;
  double dropout_p_;
  std::vector<std::shared_ptr<Linear>> w_ih_;
  std::vector<std::shared_ptr<Linear>> w_hh_;
};

class BASALT_API LSTM : public Module
{
public:
  LSTM (size_t input_size, size_t hidden_size, size_t num_layers = 1,
        bool batch_first = true, double dropout_p = 0.0,
        Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  std::tuple<Tensor, Tensor, Tensor>
  forward_with_state (const Tensor &input,
                      const Tensor &h0 = Tensor (),
                      const Tensor &c0 = Tensor ());

private:
  size_t input_size_;
  size_t hidden_size_;
  size_t num_layers_;
  bool batch_first_;
  double dropout_p_;
  std::vector<std::shared_ptr<Linear>> w_ih_;
  std::vector<std::shared_ptr<Linear>> w_hh_;
};

class BASALT_API GRU : public Module
{
public:
  GRU (size_t input_size, size_t hidden_size, size_t num_layers = 1,
       bool batch_first = true, double dropout_p = 0.0,
       Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  std::pair<Tensor, Tensor>
  forward_with_state (const Tensor &input, const Tensor &h0 = Tensor ());

private:
  size_t input_size_;
  size_t hidden_size_;
  size_t num_layers_;
  bool batch_first_;
  double dropout_p_;
  std::vector<std::shared_ptr<Linear>> w_ih_;
  std::vector<std::shared_ptr<Linear>> w_hh_;
};

/* Rotary Position Embedding (RoPE) for LLM / Transformer attention.  */
class BASALT_API Rotary_Embedding : public Module
{
public:
  Rotary_Embedding (size_t dim, size_t max_seq_len = 2048,
                    double base = 10000.0,
                    Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor cos_cached;
  Tensor sin_cached;

private:
  size_t dim_;
  size_t max_seq_len_;
  double base_;
};

using RoPE = Rotary_Embedding;

/* Pre-allocated autoregressive key-value cache for Transformer decoding.  */
class BASALT_API KV_Cache
{
public:
  KV_Cache (size_t batch_size, size_t num_heads, size_t max_seq_len,
            size_t head_dim, Tensor_Options options = Tensor_Options ());

  std::pair<Tensor, Tensor>
  update (const Tensor &k_step, const Tensor &v_step);

  void
  reset (void) noexcept;

  size_t
  seq_len (void) const noexcept;

  Tensor k_cache;
  Tensor v_cache;

private:
  size_t max_seq_len_;
  size_t seq_pos_;
};

/* 2-D transposed convolution module.  */
class BASALT_API Conv_Transpose2d : public Module
{
public:
  Conv_Transpose2d (size_t in_channels, size_t out_channels,
                    size_t kernel_size,
                    size_t stride = 1, size_t padding = 0,
                    size_t output_padding = 0,
                    size_t dilation = 1, size_t groups = 1,
                    bool use_bias = true,
                    Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;

private:
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  size_t out_pad_h_;
  size_t out_pad_w_;
  size_t dilation_h_;
  size_t dilation_w_;
  size_t groups_;
  bool use_bias_;
};

using ConvTranspose2d = Conv_Transpose2d;

/* Instance normalization over spatial dimensions of (N, C, ...).  */
class BASALT_API Instance_Norm : public Module
{
public:
  Instance_Norm (size_t num_features,
                 double eps = 1e-5,
                 bool affine = false,
                 Tensor_Options options = Tensor_Options ());

  Tensor
  forward (const Tensor &input) override;

  Tensor weight;
  Tensor bias;

private:
  size_t num_features_;
  double eps_;
  bool affine_;
};

using Instance_Norm1d = Instance_Norm;
using Instance_Norm2d = Instance_Norm;
using Instance_Norm3d = Instance_Norm;
using InstanceNorm1d = Instance_Norm;
using InstanceNorm2d = Instance_Norm;
using InstanceNorm3d = Instance_Norm;

} /* namespace basalt::nn */

#endif /* BASALT_NN_HPP */
