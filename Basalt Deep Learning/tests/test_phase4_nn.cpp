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

#include <basalt/autograd.hpp>
#include <basalt/error.hpp>
#include <basalt/nn.hpp>
#include <basalt/ops.hpp>
#include <basalt/random.hpp>
#include <basalt/tensor.hpp>
#include <cmath>
#include <memory>
#include <vector>

namespace
{

void
test_linear_conv_and_pooling (void)
{
  using namespace basalt;

  seed_everything (42ULL);
  Tensor_Options f64 = Tensor_Options ().with_dtype (Dtype::Float64);

  nn::Linear lin (4, 3, true, f64);
  Tensor x = randn ({2, 4}, f64.with_requires_grad (true));
  Tensor y = lin (x);
  BASALT_ASSERT (y.size (0) == 2 && y.size (1) == 3, "Linear output shape");
  sum (y).backward ();
  BASALT_ASSERT (lin.weight.grad ().defined () && lin.bias.grad ().defined (),
                 "Linear parameter grads must be populated");

  /* Conv1d, Conv2d, Conv3d + Pooling.  */
  nn::Conv1d c1 (2, 4, 3, 1, 1, 1, 1, true, f64);
  nn::Max_Pool1d mp1 (2);
  Tensor x1 = randn ({2, 2, 6}, f64.with_requires_grad (true));
  Tensor y1 = mp1 (c1 (x1));
  BASALT_ASSERT (y1.shape () == Shape_Vector ({2, 4, 3}),
                 "Conv1d + Max_Pool1d shape");
  sum (y1).backward ();
  BASALT_ASSERT (x1.grad ().defined (), "Conv1d input grad");

  nn::Conv2d c2 (2, 4, 3, 1, 1, 1, 2, true, f64);
  nn::Avg_Pool2d ap2 (2);
  nn::Adaptive_Avg_Pool2d aap2 (1, 1);
  Tensor x2 = randn ({2, 2, 4, 4}, f64.with_requires_grad (true));
  Tensor y2 = aap2 (ap2 (c2 (x2)));
  BASALT_ASSERT (y2.shape () == Shape_Vector ({2, 4, 1, 1}),
                 "Conv2d + Avg_Pool2d + Adaptive_Avg_Pool2d shape");
  sum (y2).backward ();
  BASALT_ASSERT (c2.weight.grad ().defined (), "Conv2d weight grad");

  nn::Conv3d c3 (1, 2, 2, 1, 0, 1, 1, true, f64);
  nn::Max_Pool3d mp3 (2);
  Tensor x3 = randn ({1, 1, 3, 3, 3}, f64.with_requires_grad (true));
  Tensor y3 = mp3 (c3 (x3));
  BASALT_ASSERT (y3.shape () == Shape_Vector ({1, 2, 1, 1, 1}),
                 "Conv3d + Max_Pool3d shape");
  sum (y3).backward ();
  BASALT_ASSERT (c3.weight.grad ().defined (), "Conv3d weight grad");

  /* Finite-difference gradcheck on functional::conv2d.  */
  Tensor gx_in = randn ({1, 2, 3, 3}, f64.with_requires_grad (true));
  Tensor gw_in = randn ({2, 2, 2, 2}, f64.with_requires_grad (true));
  Tensor gb_in = randn ({2}, f64.with_requires_grad (true));
  auto conv_fn = [] (const std::vector<Tensor> &inps) -> Tensor
    {
      return nn::functional::conv2d (inps[0], inps[1], inps[2], 1, 1, 0, 0);
    };
  BASALT_ASSERT (gradcheck (conv_fn, {gx_in, gw_in, gb_in}, 1e-6, 1e-5, 1e-4),
                 "conv2d finite-difference gradcheck failed");
}

void
test_normalizations_and_initializers (void)
{
  using namespace basalt;

  seed_everything (77ULL);
  Tensor_Options f64 = Tensor_Options ().with_dtype (Dtype::Float64);

  nn::Batch_Norm2d bn (4, 1e-5, 0.1, true, f64);
  Tensor x = randn ({2, 4, 3, 3}, f64.with_requires_grad (true));
  Tensor y_train = bn (x);
  BASALT_ASSERT (y_train.shape () == x.shape (), "BatchNorm2d shape");
  sum (y_train).backward ();
  bn.eval ();
  Tensor y_eval = bn (x);
  BASALT_ASSERT (y_eval.shape () == x.shape (), "BatchNorm2d eval shape");

  nn::Layer_Norm ln (6, 1e-5, true, f64);
  nn::RMS_Norm rms (6, 1e-6, true, f64);
  Tensor xl = randn ({3, 6}, f64.with_requires_grad (true));
  BASALT_ASSERT (ln (xl).shape () == xl.shape (), "LayerNorm shape");
  BASALT_ASSERT (rms (xl).shape () == xl.shape (), "RMSNorm shape");

  nn::Group_Norm gn (2, 4, 1e-5, true, f64);
  BASALT_ASSERT (gn (x).shape () == x.shape (), "GroupNorm shape");

  /* Verify orthogonal_ QR initialization produces Q * Q^T == I.  */
  Tensor q = empty ({4, 4}, f64);
  nn::init::orthogonal_ (q, 1.0);
  Tensor qtq = matmul (q, q.t ());
  BASALT_ASSERT (allclose (qtq, eye (4, 4, f64), 1e-10, 1e-10),
                 "orthogonal_ initialization must satisfy Q * Q^T = I");
}

void
test_attention_transformer_and_state_dict (void)
{
  using namespace basalt;

  seed_everything (101ULL);
  Tensor_Options f64 = Tensor_Options ().with_dtype (Dtype::Float64);

  nn::Embedding emb (16, 8, 0, f64);
  Tensor tokens = tensor ({0, 2, 5, 1, 3, 4},
                          Tensor_Options ().with_dtype (Dtype::Int64))
                    .reshape ({2, 3});
  Tensor tok_emb = emb (tokens);
  BASALT_ASSERT (tok_emb.shape () == Shape_Vector ({2, 3, 8}),
                 "Embedding output shape");
  /* Padding index 0 must produce all zeros at [0, 0, :].  */
  for (size_t d = 0; d < 8; ++d)
    {
      BASALT_ASSERT (tok_emb.read_scalar_at (d) == 0.0,
                     "padding_idx row must be zero");
    }

  nn::Transformer_Encoder enc (8, 2, 2, 16, 0.0, true, f64);
  nn::Transformer_Decoder dec (8, 2, 2, 16, 0.0, true, f64);
  Tensor memory = enc.forward (tok_emb, Tensor (), false);
  Tensor dec_out = dec.forward (tok_emb, memory, Tensor (), Tensor (), true);
  BASALT_ASSERT (dec_out.shape () == Shape_Vector ({2, 3, 8}),
                 "Transformer_Decoder output shape");
  sum (dec_out).backward ();
  BASALT_ASSERT (emb.weight.grad ().defined (),
                 "gradients must flow through Transformer to Embedding");

  /* Verify state_dict round-trip.  */
  nn::State_Dict sd = enc.state_dict ();
  nn::Transformer_Encoder enc2 (8, 2, 2, 16, 0.0, true, f64);
  enc2.load_state_dict (sd, true);
  Tensor memory2 = enc2.forward (tok_emb, Tensor (), false);
  BASALT_ASSERT (allclose (memory, memory2, 1e-12, 1e-12),
                 "state_dict round-trip must produce identical outputs");
}

void
test_recurrent_rope_and_kv_cache (void)
{
  using namespace basalt;
  seed_everything (99);

  /* RNN, LSTM, GRU forward + backward.  */
  Tensor x = randn ({2, 4, 6}, Tensor_Options ().with_requires_grad (true));

  nn::RNN rnn (6, 8, 2, true, 0.0);
  auto [rnn_out, rnn_hn] = rnn.forward_with_state (x);
  BASALT_ASSERT (rnn_out.shape () == Shape_Vector ({2, 4, 8}), "RNN out");
  BASALT_ASSERT (rnn_hn.shape () == Shape_Vector ({2, 2, 8}), "RNN hn");
  ops::sum (rnn_out).backward ();
  BASALT_ASSERT (x.grad ().defined (), "RNN backward grad");

  x.set_grad (Tensor ());
  nn::LSTM lstm (6, 8, 2, true, 0.0);
  auto [lstm_out, lstm_hn, lstm_cn] = lstm.forward_with_state (x);
  BASALT_ASSERT (lstm_out.shape () == Shape_Vector ({2, 4, 8}), "LSTM out");
  BASALT_ASSERT (lstm_hn.shape () == Shape_Vector ({2, 2, 8}), "LSTM hn");
  BASALT_ASSERT (lstm_cn.shape () == Shape_Vector ({2, 2, 8}), "LSTM cn");
  ops::sum (lstm_out).backward ();
  BASALT_ASSERT (x.grad ().defined (), "LSTM backward grad");

  x.set_grad (Tensor ());
  nn::GRU gru (6, 8, 2, true, 0.0);
  auto [gru_out, gru_hn] = gru.forward_with_state (x);
  BASALT_ASSERT (gru_out.shape () == Shape_Vector ({2, 4, 8}), "GRU out");
  BASALT_ASSERT (gru_hn.shape () == Shape_Vector ({2, 2, 8}), "GRU hn");
  ops::sum (gru_out).backward ();
  BASALT_ASSERT (x.grad ().defined (), "GRU backward grad");

  /* Rotary_Embedding (RoPE) and KV_Cache + FlashAttention backward.  */
  nn::Rotary_Embedding rope (8, 64);
  Tensor q = randn ({1, 2, 32, 8},
                    Tensor_Options ().with_requires_grad (true));
  Tensor k = randn ({1, 2, 32, 8},
                    Tensor_Options ().with_requires_grad (true));
  Tensor v = randn ({1, 2, 32, 8},
                    Tensor_Options ().with_requires_grad (true));
  Tensor q_rot = rope.forward (q);
  Tensor k_rot = rope.forward (k);
  BASALT_ASSERT (q_rot.shape () == Shape_Vector ({1, 2, 32, 8}), "RoPE shape");

  Tensor attn = nn::functional::scaled_dot_product_attention (
    q_rot, k_rot, v, Tensor (), 0.0, true, false);
  ops::sum (attn).backward ();
  BASALT_ASSERT (q.grad ().defined () && k.grad ().defined ()
                 && v.grad ().defined (),
                 "Flash_Attention_Backward + RoPE gradients");

  nn::KV_Cache cache (1, 2, 64, 8);
  auto [kc1, vc1] = cache.update (k_rot.detach (), v.detach ());
  BASALT_ASSERT (cache.seq_len () == 32
                 && kc1.shape () == Shape_Vector ({1, 2, 32, 8}),
                 "KV_Cache prefill shape");
  Tensor k_step = randn ({1, 2, 1, 8});
  Tensor v_step = randn ({1, 2, 1, 8});
  auto [kc2, vc2] = cache.update (k_step, v_step);
  BASALT_ASSERT (cache.seq_len () == 33
                 && kc2.shape () == Shape_Vector ({1, 2, 33, 8})
                 && vc2.shape () == Shape_Vector ({1, 2, 33, 8}),
                 "KV_Cache decode step shape");

  /* Conv_Transpose2d, Instance_Norm2d, and cosine_similarity.  */
  nn::Conv_Transpose2d deconv (2, 4, 3, 2, 1, 1);
  Tensor img = randn ({2, 2, 4, 4},
                      Tensor_Options ().with_requires_grad (true));
  Tensor up = deconv.forward (img);
  BASALT_ASSERT (up.shape () == Shape_Vector ({2, 4, 8, 8}),
                 "Conv_Transpose2d upsampled shape");
  nn::Instance_Norm2d inorm (4, 1e-5, true);
  Tensor normed = inorm.forward (up);
  BASALT_ASSERT (normed.shape () == Shape_Vector ({2, 4, 8, 8}),
                 "Instance_Norm2d shape");
  ops::sum (normed).backward ();
  BASALT_ASSERT (img.grad ().defined () && deconv.weight.grad ().defined ()
                 && deconv.bias.grad ().defined ()
                 && inorm.weight.grad ().defined (),
                 "Conv_Transpose2d + Instance_Norm2d backward gradients");

  Tensor v1 = tensor ({1.0, 0.0, 0.0}).reshape ({1, 3});
  Tensor v2 = tensor ({1.0, 0.0, 0.0}).reshape ({1, 3});
  Tensor csim = nn::functional::cosine_similarity (v1, v2, 1);
  BASALT_ASSERT (std::fabs (csim.item_double () - 1.0) < 1e-5,
                 "cosine_similarity identical vectors");
}

} /* namespace */

int
main (void)
{
  test_linear_conv_and_pooling ();
  test_normalizations_and_initializers ();
  test_attention_transformer_and_state_dict ();
  test_recurrent_rope_and_kv_cache ();
  return 0;
}
