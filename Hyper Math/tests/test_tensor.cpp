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

#include "hyper_math/hyper_math.hpp"
#include "hyper_math/hyper_math_c.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#define CHECK_TRUE(COND)                                                      \
  do                                                                          \
    {                                                                         \
      if (!(COND))                                                            \
        {                                                                     \
          std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: "       \
                    << #COND << "\n";                                         \
          return 1;                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

#define CHECK_NEAR(A, B, TOL)                                                 \
  do                                                                          \
    {                                                                         \
      const double _va = static_cast<double> (A);                             \
      const double _vb = static_cast<double> (B);                             \
      if (std::abs (_va - _vb) > (TOL))                                       \
        {                                                                     \
          std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: "       \
                    << #A << " (" << _va << ") != " << #B << " (" << _vb      \
                    << ")\n";                                                 \
          return 1;                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

int
run_tensor_tests ()
{
  using namespace hyper_math;
  using namespace hyper_math::tensor;

  std::cout << "[Suite 1/3] Core & Hyper-Lattice Tensor Engine (NumPy)...\n";

  /* 1. 64-byte aligned slab_buffer and reference counting.  */
  {
    slab_buffer<double> buf (128, 3.5);
    CHECK_TRUE (buf.size () == 128);
    CHECK_TRUE (buf.use_count () == 1);
    const auto addr = reinterpret_cast<std::uintptr_t> (buf.data ());
    CHECK_TRUE ((addr % cache_line_bytes) == 0);

    slab_buffer<double> shared = buf;
    CHECK_TRUE (buf.use_count () == 2);
    slab_buffer<double> cloned = buf.clone ();
    CHECK_TRUE (cloned.use_count () == 1);
    CHECK_NEAR (cloned[63], 3.5, 1e-12);
  }

  /* 2. 64-bit word validity_bitmap and hardware popcount.  */
  {
    validity_bitmap bm (130, true);
    CHECK_TRUE (bm.size () == 130);
    CHECK_TRUE (bm.valid_count () == 130);
    bm.set_valid (0, false);
    bm.set_valid (64, false);
    bm.set_valid (129, false);
    CHECK_TRUE (bm.valid_count () == 127);
    CHECK_TRUE (bm.null_count () == 3);
    validity_bitmap inv = ~bm;
    CHECK_TRUE (inv.valid_count () == 3);
    CHECK_TRUE ((bm & inv).valid_count () == 0);
    CHECK_TRUE ((bm | inv).valid_count () == 130);
  }

  /* 3. Tensor creation, views, slicing, transpose, and broadcasting.  */
  {
    auto a = arange<double> (0.0, 12.0, 1.0).reshape ({3, 4});
    CHECK_TRUE (a.ndim () == 2);
    CHECK_NEAR (a (2, 3), 11.0, 1e-12);

    auto at = a.t ();
    CHECK_TRUE (at.shape () == (shape_type {4, 3}));
    CHECK_NEAR (at (3, 2), 11.0, 1e-12);
    CHECK_TRUE (a.storage ().use_count () >= 2);

    auto sub = a.slice ({slice_range (1, 3, 1), slice_range (1, 4, 2)});
    CHECK_TRUE (sub.shape () == (shape_type {2, 2}));
    CHECK_NEAR (sub (0, 0), 5.0, 1e-12);
    CHECK_NEAR (sub (0, 1), 7.0, 1e-12);
    CHECK_NEAR (sub (1, 0), 9.0, 1e-12);
    CHECK_NEAR (sub (1, 1), 11.0, 1e-12);

    /* N-D Broadcasting: (3, 1) + (1, 4) -> (3, 4).  */
    nd_tensor<double> col_vec ({3, 1}, {10.0, 20.0, 30.0});
    nd_tensor<double> row_vec ({1, 4}, {1.0, 2.0, 3.0, 4.0});
    nd_tensor<double> bcast = col_vec + row_vec;
    CHECK_TRUE (bcast.shape () == (shape_type {3, 4}));
    CHECK_NEAR (bcast (0, 0), 11.0, 1e-12);
    CHECK_NEAR (bcast (2, 3), 34.0, 1e-12);
  }

  /* 4. Lazily fused ufuncs, where, clip, boolean masking.  */
  {
    auto x = linspace<double> (0.0, constants::pi, 5);
    nd_tensor<double> trig_identity =
      pow (sin (x), 2.0) + pow (cos (x), 2.0);
    for (std::size_t i = 0; i < trig_identity.size (); ++i)
      {
        CHECK_NEAR (trig_identity[i], 1.0, 1e-12);
      }

    nd_tensor<double> v = {-2.0, -0.5, 0.0, 1.5, 4.0};
    auto masked = v.filter (v > 0.0);
    CHECK_TRUE (masked.size () == 2);
    CHECK_NEAR (masked[0], 1.5, 1e-12);
    CHECK_NEAR (masked[1], 4.0, 1e-12);

    auto w = where (v >= 0.0, v, -v);
    CHECK_NEAR (w[0], 2.0, 1e-12);
    CHECK_NEAR (w[1], 0.5, 1e-12);

    auto cl = clip (v, -1.0, 2.0);
    CHECK_NEAR (cl[0], -1.0, 1e-12);
    CHECK_NEAR (cl[4], 2.0, 1e-12);
  }

  /* 5. Reductions, Welford variance, percentiles, matmul, and einsum.  */
  {
    nd_tensor<double> m = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    CHECK_NEAR (sum (m), 21.0, 1e-12);
    CHECK_NEAR (mean (m), 3.5, 1e-12);
    CHECK_NEAR (median (m), 3.5, 1e-12);

    auto col_sums = sum (m, 0);
    CHECK_TRUE (col_sums.shape () == (shape_type {3}));
    CHECK_NEAR (col_sums[0], 5.0, 1e-12);
    CHECK_NEAR (col_sums[1], 7.0, 1e-12);
    CHECK_NEAR (col_sums[2], 9.0, 1e-12);

    auto row_means = mean (m, 1);
    CHECK_TRUE (row_means.shape () == (shape_type {2}));
    CHECK_NEAR (row_means[0], 2.0, 1e-12);
    CHECK_NEAR (row_means[1], 5.0, 1e-12);

    /* Matrix multiplication and Einstein summation equivalence.  */
    nd_tensor<double> b = {{1.0, 0.0}, {0.0, 2.0}, {3.0, 1.0}};
    auto prod_mm = matmul (m, b);
    auto prod_ein = einsum<double> ("ij,jk->ik", {m, b});
    CHECK_TRUE (allclose (prod_mm, prod_ein));
    CHECK_NEAR (prod_mm (0, 0), 10.0, 1e-12);
    CHECK_NEAR (prod_mm (1, 1), 16.0, 1e-12);

    /* Einsum trace and outer product.  */
    nd_tensor<double> sq = {{2.0, 5.0}, {7.0, 3.0}};
    auto tr_ein = einsum<double> ("ii->", {sq});
    CHECK_NEAR (tr_ein[0], trace (sq), 1e-12);
  }

  /* 6. Manipulation, meshgrid, pad, rot90, tril/triu, histogram, cov/corrcoef,
     and polynomial least-squares fitting (polyfit / polyval).  */
  {
    auto [xx, yy] = meshgrid (nd_tensor<double> {1.0, 2.0, 3.0},
                              nd_tensor<double> {10.0, 20.0}, "xy");
    CHECK_TRUE (xx.shape () == (shape_type {2, 3}));
    CHECK_TRUE (yy.shape () == (shape_type {2, 3}));
    CHECK_NEAR (xx (1, 2), 3.0, 1e-12);
    CHECK_NEAR (yy (1, 2), 20.0, 1e-12);

    nd_tensor<double> base_arr = {1.0, 2.0, 3.0};
    auto p_edge = pad (base_arr, 2, 1, "edge");
    CHECK_TRUE (p_edge.size () == 6);
    CHECK_NEAR (p_edge[0], 1.0, 1e-12);
    CHECK_NEAR (p_edge[5], 3.0, 1e-12);

    auto rolled = roll (base_arr, 1);
    CHECK_NEAR (rolled[0], 3.0, 1e-12);
    CHECK_NEAR (rolled[1], 1.0, 1e-12);

    nd_tensor<double> mat2 = {{1.0, 2.0}, {3.0, 4.0}};
    auto r90 = rot90 (mat2, 1);
    CHECK_NEAR (r90 (0, 0), 2.0, 1e-12);
    CHECK_NEAR (r90 (1, 0), 1.0, 1e-12);

    auto lower = tril (mat2);
    CHECK_NEAR (lower (0, 1), 0.0, 1e-12);
    CHECK_NEAR (lower (1, 0), 3.0, 1e-12);

    auto upper = triu (mat2);
    CHECK_NEAR (upper (1, 0), 0.0, 1e-12);
    CHECK_NEAR (upper (0, 1), 2.0, 1e-12);

    /* Histogram & searchsorted.  */
    nd_tensor<double> samples = {0.5, 1.5, 2.5, 3.5};
    auto hist = histogram (samples, 2, std::make_pair (0.0, 4.0));
    CHECK_TRUE (hist.counts[0] == 2 && hist.counts[1] == 2);

    auto ins = searchsorted (samples, nd_tensor<double> {0.0, 2.0, 4.0});
    CHECK_TRUE (ins[0] == 0 && ins[1] == 2 && ins[2] == 4);

    /* Gradient of f(x) = x^2 on uniform grid.  */
    auto xg = linspace<double> (0.0, 4.0, 5);
    auto fg = pow (xg, 2.0);
    auto grad = gradient (fg, 1.0);
    CHECK_NEAR (grad[1], 2.0, 1e-12);
    CHECK_NEAR (grad[2], 4.0, 1e-12);
    CHECK_NEAR (grad[3], 6.0, 1e-12);

    /* Covariance & correlation matrix.  */
    nd_tensor<double> obs = {{1.0, 2.0, 3.0, 4.0},
                             {2.0, 4.0, 6.0, 8.0}};
    auto corr_m = corrcoef (obs);
    CHECK_NEAR (corr_m (0, 0), 1.0, 1e-12);
    CHECK_NEAR (corr_m (0, 1), 1.0, 1e-12);

    /* Polynomial fit: y = 3*x^2 - 2*x + 5.  */
    auto xp = linspace<double> (-2.0, 3.0, 10);
    nd_tensor<double> yp = 3.0 * pow (xp, 2.0) - 2.0 * xp + 5.0;
    auto p_coeffs = polyfit (xp, yp, 2);
    CHECK_NEAR (p_coeffs[0], 3.0, 1e-9);
    CHECK_NEAR (p_coeffs[1], -2.0, 1e-9);
    CHECK_NEAR (p_coeffs[2], 5.0, 1e-9);
    CHECK_NEAR (polyval (p_coeffs, 2.0), 13.0, 1e-9);

    /* Self-aliasing compound assignment: m += m.t().  */
    nd_tensor<double> alias_m = {{1.0, 2.0},
                                 {3.0, 4.0}};
    alias_m += alias_m.t ();
    CHECK_NEAR (alias_m (0, 0), 2.0, 1e-12);
    CHECK_NEAR (alias_m (0, 1), 5.0, 1e-12);
    CHECK_NEAR (alias_m (1, 0), 5.0, 1e-12);
    CHECK_NEAR (alias_m (1, 1), 8.0, 1e-12);

    /* Nested unary/binary expression with non-contiguous transposed view
       and broadcasting.  */
    nd_tensor<double> base_m = {{1.0, 2.0, 3.0},
                                {4.0, 5.0, 6.0}};
    nd_tensor<double> bias_v = {10.0, 20.0};
    nd_tensor<double> nested_eval = abs (-(base_m.t () + bias_v));
    CHECK_TRUE (nested_eval.shape () == (shape_type {3, 2}));
    CHECK_NEAR (nested_eval (0, 0), 11.0, 1e-12);
    CHECK_NEAR (nested_eval (2, 1), 26.0, 1e-12);

    /* Phase 3: Parallel thread_pool, nansum/nanmean, kron, tensordot,
       stack/split, and NumPy .npy binary roundtrip.  */
    const int par_sum = parallel_reduce (
      0, 1000, 100, 0,
      [] (std::size_t i) { return static_cast<int> (i); },
      [] (int a, int b) { return a + b; });
    CHECK_TRUE (par_sum == 499500);

    nd_tensor<double> with_nan = {1.0, constants::nan, 3.0, 5.0};
    CHECK_NEAR (nansum (with_nan), 9.0, 1e-12);
    CHECK_NEAR (nanmean (with_nan), 3.0, 1e-12);
    CHECK_NEAR (nanmin (with_nan), 1.0, 1e-12);
    CHECK_NEAR (nanmax (with_nan), 5.0, 1e-12);
    CHECK_NEAR (nanmedian (with_nan), 3.0, 1e-12);

    auto k_prod = kron (eye<double> (2), nd_tensor<double> {{1.0, 2.0}, {3.0, 4.0}});
    CHECK_TRUE (k_prod.shape () == (shape_type {4, 4}));
    CHECK_NEAR (trace (k_prod), 10.0, 1e-12);

    auto td_res = tensordot (ones<double> ({2, 3, 4}), ones<double> ({3, 4, 5}), 2);
    CHECK_TRUE (td_res.shape () == (shape_type {2, 5}));
    CHECK_NEAR (td_res (0, 0), 12.0, 1e-12);

    auto stk = stack<double> ({nd_tensor<double> {1.0, 2.0},
                               nd_tensor<double> {3.0, 4.0}}, 0);
    CHECK_TRUE (stk.shape () == (shape_type {2, 2}));
    auto parts = split (arange<double> (0.0, 6.0, 1.0), 3);
    CHECK_TRUE (parts.size () == 3 && parts[1][0] == 2.0);

    const std::string npy_path = "/tmp/test_hyper_math.npy";
    save_npy (npy_path, stk);
    auto loaded_npy = load_npy (npy_path);
    CHECK_TRUE (loaded_npy.shape () == (shape_type {2, 2}));
    CHECK_TRUE (allclose (stk, loaded_npy, 1e-12, 1e-12));

    /* Pure C ABI Wrapper Verification (hyper_math_c.h).  */
    CHECK_TRUE (std::strcmp (hm_version (), "0.1.0") == 0);
    const std::size_t c_shape[2] = {2, 2};
    const double c_data[4] = {4.0, 1.0, 1.0, 3.0};
    hm_tensor_t *ct_a = hm_tensor_from_data (c_shape, 2, c_data);
    CHECK_TRUE (ct_a != nullptr);
    CHECK_TRUE (hm_tensor_size (ct_a) == 4);
    CHECK_TRUE (hm_tensor_ndim (ct_a) == 2);
    CHECK_NEAR (hm_tensor_sum (ct_a), 9.0, 1e-12);
    CHECK_NEAR (hm_tensor_mean (ct_a), 2.25, 1e-12);
    CHECK_NEAR (hm_tensor_min (ct_a), 1.0, 1e-12);
    CHECK_NEAR (hm_tensor_max (ct_a), 4.0, 1e-12);
    CHECK_NEAR (hm_linalg_det (ct_a), 11.0, 1e-12);

    const std::size_t b_shape[1] = {2};
    const double b_data[2] = {9.0, 5.0};
    hm_tensor_t *ct_b = hm_tensor_from_data (b_shape, 1, b_data);
    hm_tensor_t *ct_x = hm_linalg_solve (ct_a, ct_b);
    CHECK_NEAR (hm_tensor_get_flat (ct_x, 0), 2.0, 1e-12);
    CHECK_NEAR (hm_tensor_get_flat (ct_x, 1), 1.0, 1e-12);

    hm_tensor_t *ct_inv = hm_linalg_inv (ct_a);
    hm_tensor_t *ct_prod = hm_tensor_matmul (ct_a, ct_inv);
    CHECK_NEAR (hm_tensor_get_flat (ct_prod, 0), 1.0, 1e-12);
    CHECK_NEAR (hm_tensor_get_flat (ct_prod, 1), 0.0, 1e-12);
    CHECK_NEAR (hm_tensor_get_flat (ct_prod, 3), 1.0, 1e-12);

    hm_tensor_free (ct_prod);
    hm_tensor_free (ct_inv);
    hm_tensor_free (ct_x);
    hm_tensor_free (ct_b);
    hm_tensor_free (ct_a);

    /* Phase 6: take, compress, nan_to_num, unwrap, and lexsort.  */
    nd_tensor<double> src_arr = {10.0, 20.0, 30.0, 40.0, 50.0};
    auto taken = take (src_arr, nd_tensor<std::size_t> {4, 1, 0});
    CHECK_NEAR (taken[0], 50.0, 1e-12);
    CHECK_NEAR (taken[1], 20.0, 1e-12);
    CHECK_NEAR (taken[2], 10.0, 1e-12);

    auto comp = compress (nd_tensor<std::uint8_t> {0, 1, 0, 1, 1}, src_arr);
    CHECK_TRUE (comp.size () == 3);
    CHECK_NEAR (comp[0], 20.0, 1e-12);
    CHECK_NEAR (comp[2], 50.0, 1e-12);

    nd_tensor<double> dirty = {constants::nan, constants::inf, -constants::inf, 7.5};
    auto cleaned = nan_to_num (dirty, 0.0, 999.0, -999.0);
    CHECK_NEAR (cleaned[0], 0.0, 1e-12);
    CHECK_NEAR (cleaned[1], 999.0, 1e-12);
    CHECK_NEAR (cleaned[2], -999.0, 1e-12);
    CHECK_NEAR (cleaned[3], 7.5, 1e-12);

    nd_tensor<double> wrapped = {0.0, 0.5 * constants::pi, constants::pi,
                                 -0.5 * constants::pi, 0.0};
    auto unwrapped = unwrap (wrapped);
    CHECK_NEAR (unwrapped[3], 1.5 * constants::pi, 1e-12);
    CHECK_NEAR (unwrapped[4], 2.0 * constants::pi, 1e-12);

    nd_tensor<double> sec_key = {2.0, 1.0, 2.0, 1.0};
    nd_tensor<double> pri_key = {10.0, 20.0, 10.0, 10.0};
    auto lex_idx = lexsort<double> ({sec_key, pri_key});
    CHECK_TRUE (lex_idx[0] == 3);
    CHECK_TRUE (lex_idx[1] == 0);
    CHECK_TRUE (lex_idx[2] == 2);
    CHECK_TRUE (lex_idx[3] == 1);

    /* Phase 7: sinc, deg2rad, rad2deg, isclose, ediff1d, and 1D set ops.  */
    nd_tensor<double> sc = sinc (nd_tensor<double> {0.0, 0.5, 1.0});
    CHECK_NEAR (sc[0], 1.0, 1e-12);
    CHECK_NEAR (sc[1], 2.0 / constants::pi, 1e-12);
    CHECK_NEAR (sc[2], 0.0, 1e-12);

    nd_tensor<double> degs = {0.0, 90.0, 180.0};
    nd_tensor<double> rads = deg2rad (degs);
    CHECK_NEAR (rads[1], constants::half_pi, 1e-12);
    nd_tensor<double> back_degs = rad2deg (rads);
    CHECK_TRUE (allclose (degs, back_degs, 1e-12, 1e-12));
    CHECK_TRUE (isclose (degs, back_degs)[2] == 1);

    auto ed = ediff1d (nd_tensor<double> {1.0, 3.0, 7.0, 13.0});
    CHECK_NEAR (ed[0], 2.0, 1e-12);
    CHECK_NEAR (ed[1], 4.0, 1e-12);
    CHECK_NEAR (ed[2], 6.0, 1e-12);

    nd_tensor<int> set_a = {1, 2, 3, 4};
    nd_tensor<int> set_b = {3, 4, 5, 6};
    CHECK_TRUE (intersect1d (set_a, set_b).size () == 2);
    CHECK_TRUE (union1d (set_a, set_b).size () == 6);
    CHECK_TRUE (setdiff1d (set_a, set_b).size () == 2);
    CHECK_TRUE (isin (set_a, set_b)[2] == 1 && isin (set_a, set_b)[0] == 0);

    /* Phase 8: polyadd, polysub, polymul, polyder, polyint, geomspace,
       bartlett, kaiser, and flatnonzero.  */
    nd_tensor<double> p1 = {1.0, -2.0};
    nd_tensor<double> p2 = {1.0, 2.0};
    auto pm = polymul (p1, p2); /* (x - 2)(x + 2) = x^2 - 4.  */
    CHECK_NEAR (pm[0], 1.0, 1e-12);
    CHECK_NEAR (pm[1], 0.0, 1e-12);
    CHECK_NEAR (pm[2], -4.0, 1e-12);

    auto p_sum = polyadd (pm, nd_tensor<double> {4.0});
    CHECK_NEAR (p_sum[2], 0.0, 1e-12);

    auto p_der = polyder (nd_tensor<double> {3.0, 2.0, 5.0}); /* 6x + 2.  */
    CHECK_NEAR (p_der[0], 6.0, 1e-12);
    CHECK_NEAR (p_der[1], 2.0, 1e-12);

    auto p_int = polyint (p_der, 1, 5.0);
    CHECK_NEAR (p_int[0], 3.0, 1e-12);
    CHECK_NEAR (p_int[2], 5.0, 1e-12);

    auto gs = geomspace (1.0, 1000.0, 4);
    CHECK_NEAR (gs[1], 10.0, 1e-12);
    CHECK_NEAR (gs[2], 100.0, 1e-12);

    auto bw_win = bartlett (5);
    CHECK_NEAR (bw_win[0], 0.0, 1e-12);
    CHECK_NEAR (bw_win[2], 1.0, 1e-12);

    auto kw_win = kaiser (5, 5.0);
    CHECK_NEAR (kw_win[2], 1.0, 1e-12);
    CHECK_TRUE (kw_win[0] < kw_win[2]);
    CHECK_TRUE (flatnonzero (nd_tensor<double> {0.0, 4.0, 0.0, 7.0}).size () == 2);

    /* Phase 9: squeeze, expand_dims, swapaxes, moveaxis, atleast_1d/2d/3d,
       cbrt, exp2, expm1, log1p, logaddexp, heaviside, and interp.  */
    nd_tensor<double> base_vec = {8.0, 27.0};
    auto exp_d = expand_dims (base_vec, 0);
    CHECK_TRUE (exp_d.shape () == (shape_type {1, 2}));
    auto sq_d = squeeze (exp_d);
    CHECK_TRUE (sq_d.shape () == (shape_type {2}));

    auto sw_d = swapaxes (exp_d, 0, 1);
    CHECK_TRUE (sw_d.shape () == (shape_type {2, 1}));
    auto mv_d = moveaxis (atleast_3d (base_vec), 1, 0);
    CHECK_TRUE (mv_d.shape () == (shape_type {2, 1, 1}));

    nd_tensor<double> cb_res = cbrt (base_vec);
    CHECK_NEAR (cb_res[0], 2.0, 1e-12);
    CHECK_NEAR (cb_res[1], 3.0, 1e-12);

    nd_tensor<double> e2_res = exp2 (nd_tensor<double> {3.0, 4.0});
    CHECK_NEAR (e2_res[0], 8.0, 1e-12);
    CHECK_NEAR (e2_res[1], 16.0, 1e-12);

    nd_tensor<double> lae =
      logaddexp (nd_tensor<double> {0.0}, nd_tensor<double> {0.0});
    CHECK_NEAR (lae[0], constants::ln2, 1e-12);

    nd_tensor<double> hv = heaviside (nd_tensor<double> {-1.0, 0.0, 2.0}, 0.5);
    CHECK_NEAR (hv[0], 0.0, 1e-12);
    CHECK_NEAR (hv[1], 0.5, 1e-12);
    CHECK_NEAR (hv[2], 1.0, 1e-12);

    auto lin_interp = interp (nd_tensor<double> {0.5, 1.5, 5.0},
                              nd_tensor<double> {0.0, 1.0, 2.0},
                              nd_tensor<double> {10.0, 20.0, 40.0});
    CHECK_NEAR (lin_interp[0], 15.0, 1e-12);
    CHECK_NEAR (lin_interp[1], 30.0, 1e-12);
    CHECK_NEAR (lin_interp[2], 40.0, 1e-12);

    /* Phase 10: unravel_index, ravel_multi_index, diagflat, fill_diagonal,
       tri, nanargmin, and nanargmax.  */
    const shape_type coords = unravel_index (5, {2, 3});
    CHECK_TRUE (coords == (shape_type {1, 2}));
    CHECK_TRUE (ravel_multi_index (coords, {2, 3}) == 5);

    auto df_mat = diagflat (nd_tensor<double> {3.0, 7.0}, 1);
    CHECK_TRUE (df_mat.shape () == (shape_type {3, 3}));
    CHECK_NEAR (df_mat (0, 1), 3.0, 1e-12);
    CHECK_NEAR (df_mat (1, 2), 7.0, 1e-12);

    fill_diagonal (df_mat, 9.0);
    CHECK_NEAR (df_mat (2, 2), 9.0, 1e-12);

    auto tri_m = tri<double> (3, 3, 0);
    CHECK_NEAR (sum (tri_m), 6.0, 1e-12);

    nd_tensor<double> nan_ext = {constants::nan, 5.0, -2.0, 9.0};
    CHECK_TRUE (nanargmin (nan_ext) == 2);
    CHECK_TRUE (nanargmax (nan_ext) == 3);

    /* Phase 11: ptp, trapz (uniform and non-uniform), block_2d, and C ABI
       extensions (hm_linalg_pinv, hm_optimize_linear_sum_assignment).  */
    CHECK_NEAR (ptp (nd_tensor<double> {3.0, -4.0, 8.0}), 12.0, 1e-12);
    CHECK_NEAR (trapz (nd_tensor<double> {0.0, 2.0, 4.0}, 0.5), 2.0, 1e-12);
    CHECK_NEAR (trapz (nd_tensor<double> {0.0, 2.0, 6.0},
                       nd_tensor<double> {0.0, 1.0, 3.0}),
                9.0, 1e-12);

    auto blk = block_2d<double> ({{eye<double> (2), zeros<double> ({2, 2})},
                                  {ones<double> ({2, 2}), eye<double> (2)}});
    CHECK_TRUE (blk.shape () == (shape_type {4, 4}));
    CHECK_NEAR (trace (blk), 4.0, 1e-12);

    const std::size_t c_dims[2] = {2, 2};
    const double c_cost[4] = {4.0, 1.0, 2.0, 3.0};
    hm_tensor_t *ct = hm_tensor_from_data (c_dims, 2, c_cost);
    CHECK_NEAR (hm_optimize_linear_sum_assignment (ct, 0), 3.0, 1e-12);
    hm_tensor_t *cpinv = hm_linalg_pinv (ct);
    CHECK_TRUE (hm_tensor_size (cpinv) == 4);
    hm_tensor_free (cpinv);
    hm_tensor_free (ct);

    /* Phase 12: logaddexp2, isnan, isinf, isfinite, signbit, average,
       cross_3d, tril_indices, and triu_indices.  */
    nd_tensor<double> lae2 =
      logaddexp2 (nd_tensor<double> {1.0}, nd_tensor<double> {1.0});
    CHECK_NEAR (lae2[0], 2.0, 1e-12);

    nd_tensor<double> pred_in = {constants::nan, constants::inf, -3.5};
    auto nan_mask = isnan (pred_in);
    auto inf_mask = isinf (pred_in);
    auto fin_mask = isfinite (pred_in);
    auto sgn_mask = signbit (pred_in);
    CHECK_TRUE (nan_mask[0] == 1);
    CHECK_TRUE (inf_mask[1] == 1);
    CHECK_TRUE (fin_mask[2] == 1);
    CHECK_TRUE (sgn_mask[2] == 1);

    CHECK_NEAR (average (nd_tensor<double> {10.0, 20.0, 30.0},
                         nd_tensor<double> {1.0, 1.0, 2.0}),
                22.5, 1e-12);

    auto c3 = cross_3d (nd_tensor<double> {1.0, 0.0, 0.0},
                        nd_tensor<double> {0.0, 1.0, 0.0});
    CHECK_NEAR (c3[0], 0.0, 1e-12);
    CHECK_NEAR (c3[1], 0.0, 1e-12);
    CHECK_NEAR (c3[2], 1.0, 1e-12);

    auto [tril_r, tril_c] = tril_indices (3, 0);
    CHECK_TRUE (tril_r.size () == 6 && tril_c.size () == 6);
    auto [triu_r, triu_c] = triu_indices (3, 1);
    CHECK_TRUE (triu_r.size () == 3 && triu_c.size () == 3);

    /* Phase 13: floor_divide, remainder, nextafter, ldexp, frexp,
       nancumsum, nancumprod, nanptp, diag_indices, rollaxis, trim_zeros.  */
    nd_tensor<double> fd =
      floor_divide (nd_tensor<double> {-7.0, 7.0}, nd_tensor<double> {3.0, 3.0});
    CHECK_NEAR (fd[0], -3.0, 1e-12);
    CHECK_NEAR (fd[1], 2.0, 1e-12);

    nd_tensor<double> rem =
      remainder (nd_tensor<double> {-7.0, 7.0}, nd_tensor<double> {3.0, 3.0});
    CHECK_NEAR (rem[0], 2.0, 1e-12);
    CHECK_NEAR (rem[1], 1.0, 1e-12);

    nd_tensor<double> na_res =
      nextafter (nd_tensor<double> {1.0}, nd_tensor<double> {2.0});
    CHECK_TRUE (na_res[0] > 1.0 && (na_res[0] - 1.0) < 1e-15);

    auto [mant, exps] = frexp (nd_tensor<double> {8.0, -3.0});
    CHECK_NEAR (mant[0], 0.5, 1e-12);
    CHECK_TRUE (exps[0] == 4);
    nd_tensor<double> ld_back = ldexp (mant, exps);
    CHECK_NEAR (ld_back[0], 8.0, 1e-12);
    CHECK_NEAR (ld_back[1], -3.0, 1e-12);

    nd_tensor<double> nan_seq = {2.0, constants::nan, 3.0, -4.0};
    auto ncs = nancumsum (nan_seq);
    auto ncp = nancumprod (nan_seq);
    CHECK_NEAR (ncs[1], 2.0, 1e-12);
    CHECK_NEAR (ncs[3], 1.0, 1e-12);
    CHECK_NEAR (ncp[2], 6.0, 1e-12);
    CHECK_NEAR (nanptp (nan_seq), 7.0, 1e-12);

    auto [dr, dc] = diag_indices (4);
    CHECK_TRUE (dr.size () == 4 && dc[3] == 3);

    auto ra_t = rollaxis (zeros<double> ({2, 3, 4}), 2, 0);
    CHECK_TRUE (ra_t.shape () == (shape_type {4, 2, 3}));

    auto tz = trim_zeros (nd_tensor<double> {0.0, 0.0, 5.0, 0.0, 9.0, 0.0});
    CHECK_TRUE (tz.size () == 3);
    CHECK_NEAR (tz[0], 5.0, 1e-12);
    CHECK_NEAR (tz[2], 9.0, 1e-12);

    /* Phase 14: modf, divmod, reciprocal, square, positive, negative,
       delete_elements, insert_elements, and append_elements.  */
    auto [mf_frac, mf_int] = modf (nd_tensor<double> {3.25, -2.5});
    CHECK_NEAR (mf_frac[0], 0.25, 1e-12);
    CHECK_NEAR (mf_int[0], 3.0, 1e-12);
    CHECK_NEAR (mf_frac[1], -0.5, 1e-12);
    CHECK_NEAR (mf_int[1], -2.0, 1e-12);

    auto [dm_q, dm_r] =
      divmod (nd_tensor<double> {17.0}, nd_tensor<double> {5.0});
    CHECK_NEAR (dm_q[0], 3.0, 1e-12);
    CHECK_NEAR (dm_r[0], 2.0, 1e-12);

    nd_tensor<double> rec_t = reciprocal (nd_tensor<double> {2.0, 4.0});
    nd_tensor<double> sq_t = square (nd_tensor<double> {-3.0, 5.0});
    nd_tensor<double> pos_t = positive (nd_tensor<double> {7.0});
    nd_tensor<double> neg_t = negative (nd_tensor<double> {7.0});
    CHECK_NEAR (rec_t[1], 0.25, 1e-12);
    CHECK_NEAR (sq_t[0], 9.0, 1e-12);
    CHECK_NEAR (pos_t[0], 7.0, 1e-12);
    CHECK_NEAR (neg_t[0], -7.0, 1e-12);

    auto del_t =
      delete_elements (nd_tensor<double> {10.0, 20.0, 30.0, 40.0}, {1, 3});
    CHECK_TRUE (del_t.size () == 2);
    CHECK_NEAR (del_t[0], 10.0, 1e-12);
    CHECK_NEAR (del_t[1], 30.0, 1e-12);

    auto ins_t = insert_elements (del_t, 1, nd_tensor<double> {99.0});
    CHECK_NEAR (ins_t[1], 99.0, 1e-12);

    auto app_t = append_elements (ins_t, nd_tensor<double> {77.0});
    CHECK_TRUE (app_t.size () == 4);
    CHECK_NEAR (app_t[3], 77.0, 1e-12);

    /* Phase 15: asinh, acosh, atanh, rint, spacing, histogram2d, setxor1d.  */
    nd_tensor<double> hyp_in = {0.0, 2.0, 0.5};
    CHECK_NEAR (nd_tensor<double> (asinh (hyp_in))[0], 0.0, 1e-12);
    CHECK_NEAR (nd_tensor<double> (acosh (hyp_in))[1], std::acosh (2.0), 1e-12);
    CHECK_NEAR (nd_tensor<double> (atanh (hyp_in))[2], std::atanh (0.5), 1e-12);

    nd_tensor<double> rint_t = rint (nd_tensor<double> {2.3, 3.8});
    CHECK_NEAR (rint_t[0], 2.0, 1e-12);
    CHECK_NEAR (rint_t[1], 4.0, 1e-12);

    nd_tensor<double> sp_t = spacing (nd_tensor<double> {1.0});
    CHECK_NEAR (sp_t[0], std::numeric_limits<double>::epsilon (), 1e-30);

    auto h2d = histogram2d (nd_tensor<double> {0.1, 0.2, 0.9},
                            nd_tensor<double> {0.1, 0.2, 0.9},
                            2, 2);
    CHECK_NEAR (h2d.counts (0, 0), 2.0, 1e-12);
    CHECK_NEAR (h2d.counts (1, 1), 1.0, 1e-12);

    auto sx = setxor1d (nd_tensor<double> {1.0, 2.0, 3.0},
                        nd_tensor<double> {2.0, 3.0, 4.0});
    CHECK_TRUE (sx.size () == 2);
    CHECK_NEAR (sx[0], 1.0, 1e-12);
    CHECK_NEAR (sx[1], 4.0, 1e-12);

    /* Phase 16: trunc, fix, gcd, lcm, logical_and/or/xor/not,
       sliding_window_view_1d, hanning, hamming, blackman.  */
    nd_tensor<double> tr_v = trunc (nd_tensor<double> {-2.9, 2.9});
    nd_tensor<double> fx_v = fix (nd_tensor<double> {-2.9, 2.9});
    CHECK_NEAR (tr_v[0], -2.0, 1e-12);
    CHECK_NEAR (fx_v[1], 2.0, 1e-12);

    nd_tensor<double> g_v =
      gcd (nd_tensor<double> {12.0, 15.0}, nd_tensor<double> {18.0, 25.0});
    nd_tensor<double> l_v =
      lcm (nd_tensor<double> {4.0, 6.0}, nd_tensor<double> {6.0, 8.0});
    CHECK_NEAR (g_v[0], 6.0, 1e-12);
    CHECK_NEAR (g_v[1], 5.0, 1e-12);
    CHECK_NEAR (l_v[0], 12.0, 1e-12);
    CHECK_NEAR (l_v[1], 24.0, 1e-12);

    nd_tensor<double> lb_a = {1.0, 1.0, 0.0, 0.0};
    nd_tensor<double> lb_b = {1.0, 0.0, 1.0, 0.0};
    CHECK_TRUE (logical_and (lb_a, lb_b)[0] == 1
                && logical_and (lb_a, lb_b)[1] == 0);
    CHECK_TRUE (logical_or (lb_a, lb_b)[2] == 1
                && logical_or (lb_a, lb_b)[3] == 0);
    CHECK_TRUE (logical_xor (lb_a, lb_b)[0] == 0
                && logical_xor (lb_a, lb_b)[1] == 1);
    CHECK_TRUE (logical_not (lb_a)[3] == 1);

    auto sw = sliding_window_view_1d (nd_tensor<double> {1.0, 2.0, 3.0, 4.0}, 2);
    CHECK_TRUE (sw.shape () == (shape_type {3, 2}));
    CHECK_NEAR (sw (2, 1), 4.0, 1e-12);

    CHECK_NEAR (hanning (5)[2], 1.0, 1e-12);
    CHECK_NEAR (hamming (5)[2], 1.0, 1e-12);
    CHECK_NEAR (blackman (5)[2], 1.0, 1e-12);

    /* Bug-audit regression tests: NaN propagation, broadcast +=, reflect pad,
       allclose/isclose inf/NaN, and axis min/max with infinities.  */
    nd_tensor<double> nan_v = {constants::nan, 1.0};
    CHECK_TRUE (std::isnan (nd_tensor<double> (sign (nan_v))[0]));
    CHECK_TRUE (std::isnan (nd_tensor<double> (heaviside (nan_v, 0.5))[0]));
    CHECK_TRUE (std::isnan (nd_tensor<double> (maximum (nan_v, nd_tensor<double> {2.0, 2.0}))[0]));
    CHECK_TRUE (!allclose (nan_v, nan_v));
    CHECK_TRUE (allclose (nd_tensor<double> {constants::inf},
                          nd_tensor<double> {constants::inf}));
    CHECK_TRUE (!allclose (nd_tensor<double> {constants::inf},
                           nd_tensor<double> {-constants::inf}));

    nd_tensor<double> bcast_src ({1}, {5.0});
    nd_tensor<double> bcast_v = bcast_src.broadcast_to ({4});
    bcast_v += 2.0;
    CHECK_NEAR (bcast_v[0], 7.0, 1e-12);

    nd_tensor<double> ref_in = {1.0, 2.0, 3.0, 4.0};
    auto ref_pad = pad (ref_in, 3, 3, "reflect");
    CHECK_NEAR (ref_pad[0], 4.0, 1e-12);
    CHECK_NEAR (ref_pad[9], 1.0, 1e-12);

    nd_tensor<double> inf_mat ({{constants::inf, constants::inf}});
    CHECK_TRUE (std::isinf (min (inf_mat, 1)[0]));

    /* Phase 17: bitwise ufuncs, diag/tril/triu_indices_from, ix_, select,
       piecewise, place, polydiv, and savetxt / loadtxt.  */
    nd_tensor<std::int64_t> bw_a = {6, 12};
    nd_tensor<std::int64_t> bw_b = {3, 5};
    CHECK_TRUE (nd_tensor<std::int64_t> (bitwise_and (bw_a, bw_b))[0] == 2);
    CHECK_TRUE (nd_tensor<std::int64_t> (bitwise_or (bw_a, bw_b))[0] == 7);
    CHECK_TRUE (nd_tensor<std::int64_t> (bitwise_xor (bw_a, bw_b))[0] == 5);
    CHECK_TRUE (nd_tensor<std::int64_t> (bitwise_not (bw_a))[0] == ~6LL);
    nd_tensor<std::int64_t> sh_amt = {1, 2};
    CHECK_TRUE (nd_tensor<std::int64_t> (left_shift (bw_a, sh_amt))[0] == 12);
    CHECK_TRUE (nd_tensor<std::int64_t> (right_shift (bw_a, sh_amt))[1] == 3);

    nd_tensor<double> sq3 = nd_tensor<double>::eye (3);
    auto [dfr, dfc] = diag_indices_from (sq3);
    CHECK_TRUE (dfr.size () == 3 && dfr[2] == 2 && dfc[2] == 2);
    auto [tlr, tlc] = tril_indices_from (sq3);
    CHECK_TRUE (tlr.size () == 6);
    auto [tur, tuc] = triu_indices_from (sq3, 1);
    CHECK_TRUE (tur.size () == 3);

    auto [ix_r, ix_c] = ix_ (nd_tensor<double> {1.0, 2.0},
                             nd_tensor<double> {10.0, 20.0, 30.0});
    CHECK_TRUE (ix_r.shape () == (shape_type {2, 1})
                && ix_c.shape () == (shape_type {1, 3}));

    nd_tensor<double> sel_x = {-2.0, 0.0, 5.0};
    auto sel_out = select<double> (
      {sel_x < 0.0, sel_x > 0.0},
      {nd_tensor<double> {-10.0, -10.0, -10.0},
       nd_tensor<double> {10.0, 10.0, 10.0}},
      0.0);
    CHECK_NEAR (sel_out[0], -10.0, 1e-12);
    CHECK_NEAR (sel_out[1], 0.0, 1e-12);
    CHECK_NEAR (sel_out[2], 10.0, 1e-12);

    auto pw_out = piecewise<double> (
      sel_x,
      {sel_x < 0.0, sel_x >= 0.0},
      {[] (double v) { return -v; }, [] (double v) { return v * 2.0; }});
    CHECK_NEAR (pw_out[0], 2.0, 1e-12);
    CHECK_NEAR (pw_out[2], 10.0, 1e-12);

    nd_tensor<double> pl_arr = {1.0, 2.0, 3.0, 4.0};
    place (pl_arr, pl_arr > 2.0, nd_tensor<double> {99.0});
    CHECK_NEAR (pl_arr[2], 99.0, 1e-12);
    CHECK_NEAR (pl_arr[3], 99.0, 1e-12);

    /* (x^2 - 1) / (x - 1) = (x + 1) remainder 0.  */
    auto [p_quot, p_rem] = polydiv (nd_tensor<double> {1.0, 0.0, -1.0},
                                    nd_tensor<double> {1.0, -1.0});
    CHECK_TRUE (p_quot.size () == 2);
    CHECK_NEAR (p_quot[0], 1.0, 1e-12);
    CHECK_NEAR (p_quot[1], 1.0, 1e-12);
    CHECK_NEAR (p_rem[0], 0.0, 1e-12);

    const std::string txt_path = "test_tensor_io.txt";
    nd_tensor<double> io_mat = {{1.25, 2.5}, {3.75, 5.0}};
    savetxt (txt_path, io_mat, ',');
    nd_tensor<double> io_loaded = loadtxt (txt_path, ',');
    std::remove (txt_path.c_str ());
    CHECK_TRUE (allclose (io_mat, io_loaded, 1e-12));

    /* Phase 18: masked_tensor, masked_where, masked_invalid, masked_equal,
       and save_npz / load_npz.  */
    {
      nd_tensor<double> ma_src = {1.0, -999.0, 3.0, 4.0};
      auto ma_eq = masked_equal (ma_src, -999.0);
      CHECK_TRUE (ma_eq.count () == 3);
      CHECK_NEAR (ma_eq.sum (), 8.0, 1e-12);
      CHECK_NEAR (ma_eq.mean (), 8.0 / 3.0, 1e-12);
      auto ma_comp = ma_eq.compressed ();
      CHECK_TRUE (ma_comp.size () == 3);
      CHECK_NEAR (ma_comp[1], 3.0, 1e-12);
      auto ma_fill = ma_eq.filled (0.0);
      CHECK_NEAR (ma_fill[1], 0.0, 1e-12);

      nd_tensor<double> ma_inv_src = {2.0, constants::nan, 6.0, constants::inf};
      auto ma_inv = masked_invalid (ma_inv_src);
      CHECK_TRUE (ma_inv.count () == 2);
      CHECK_NEAR (ma_inv.mean (), 4.0, 1e-12);

      auto ma_w = masked_where (ma_src < 0.0, ma_src);
      CHECK_TRUE (ma_w.count () == 3);

      const std::string npz_path = "test_archive.npz";
      save_npz (npz_path, {
        {"mat", io_mat},
        {"vec", nd_tensor<double> {10.0, 20.0, 30.0}}
      });
      auto loaded_npz = load_npz (npz_path);
      std::remove (npz_path.c_str ());
      CHECK_TRUE (loaded_npz.size () == 2);
      CHECK_TRUE (allclose (loaded_npz.at ("mat"), io_mat, 1e-12));
      CHECK_NEAR (loaded_npz.at ("vec")[2], 30.0, 1e-12);
    }

    /* Phase 19: random distributions/permutations, complex ufuncs,
       quantile/nanquantile, partition/argpartition, take_along_axis,
       put_along_axis, packbits, and unpackbits.  */
    {
      random_engine rng (42);
      auto r_int = random_randint ({50}, 10, 20, rng);
      auto r_exp = random_exponential ({100}, 2.0, rng);
      auto r_poi = random_poisson ({100}, 5.0, rng);
      auto r_bin = random_binomial ({100}, 10, 0.5, rng);
      auto r_gam = random_gamma ({100}, 2.0, 1.0, rng);
      auto r_bet = random_beta ({100}, 2.0, 3.0, rng);
      auto r_chi = random_chisquare ({100}, 4.0, rng);
      CHECK_TRUE (min (r_int) >= 10 && max (r_int) < 20);
      CHECK_TRUE (min (r_exp) > 0.0 && min (r_poi) >= 0.0
                  && max (r_bin) <= 10.0);
      CHECK_TRUE (min (r_gam) > 0.0 && min (r_bet) >= 0.0
                  && max (r_bet) <= 1.0 && min (r_chi) > 0.0);

      auto perm = random_permutation (10, rng);
      CHECK_TRUE (perm.size () == 10 && sum (perm) == 45);
      nd_tensor<double> shuf_v = {1.0, 2.0, 3.0, 4.0};
      random_shuffle (shuf_v, rng);
      CHECK_NEAR (sum (shuf_v), 10.0, 1e-12);
      auto ch = random_choice (shuf_v, 2, false, rng);
      CHECK_TRUE (ch.size () == 2);

      nd_tensor<std::complex<double>> c_arr = {
        std::complex<double> (3.0, 4.0),
        std::complex<double> (0.0, 1.0)
      };
      CHECK_NEAR (real (c_arr)[0], 3.0, 1e-12);
      CHECK_NEAR (imag (c_arr)[0], 4.0, 1e-12);
      CHECK_NEAR (conj (c_arr)[0].imag (), -4.0, 1e-12);
      CHECK_NEAR (complex_abs (c_arr)[0], 5.0, 1e-12);
      CHECK_NEAR (angle (c_arr)[1], constants::half_pi, 1e-12);

      nd_tensor<double> q_arr = {0.0, 10.0, 20.0, 30.0, 40.0};
      CHECK_NEAR (quantile (q_arr, 0.25), 10.0, 1e-12);
      nd_tensor<double> nq_arr = {0.0, 10.0, constants::nan, 30.0, 40.0};
      CHECK_NEAR (nanquantile (nq_arr, 0.5), 20.0, 1e-12);

      nd_tensor<double> unp = {9.0, 1.0, 8.0, 2.0, 7.0};
      auto part_v = partition (unp, 2);
      CHECK_NEAR (part_v[2], 7.0, 1e-12);
      auto argp_v = argpartition (unp, 2);
      CHECK_NEAR (unp[argp_v[2]], 7.0, 1e-12);

      nd_tensor<double> taa_src = {{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}};
      nd_tensor<std::size_t> taa_idx = {{2, 0}, {1, 2}};
      auto taa_out = take_along_axis (taa_src, taa_idx, 1);
      CHECK_NEAR (taa_out (0, 0), 30.0, 1e-12);
      CHECK_NEAR (taa_out (1, 0), 50.0, 1e-12);

      put_along_axis (taa_src, taa_idx,
                      nd_tensor<double> {{-1.0, -2.0}, {-3.0, -4.0}}, 1);
      CHECK_NEAR (taa_src (0, 2), -1.0, 1e-12);
      CHECK_NEAR (taa_src (0, 0), -2.0, 1e-12);

      nd_tensor<std::uint8_t> bits = {1, 0, 1, 1, 0, 0, 0, 1};
      auto packed = packbits (bits);
      CHECK_TRUE (packed.size () == 1 && packed[0] == 0xB1);
      auto unpacked = unpackbits (packed);
      CHECK_TRUE (unpacked[0] == 1 && unpacked[1] == 0 && unpacked[7] == 1);
    }

    /* Phase 20: CODATA SI constants & temperature conversion, extended random
       distributions, grid/mask constructors, complex predicates, 2D quantile
       & multi_dot, extended pad modes, orthogonal polynomials,
       apply_along_axis, sliding_window_view_2d, histogram_bin_edges,
       and histogramdd.  */
    {
      CHECK_NEAR (constants::speed_of_light, 299792458.0, 1e-6);
      CHECK_NEAR (constants::convert_temperature (100.0, "Celsius", "Kelvin"),
                  373.15, 1e-9);
      CHECK_NEAR (constants::convert_temperature (32.0, "Fahrenheit", "Celsius"),
                  0.0, 1e-9);

      random_engine rng (123);
      CHECK_TRUE (min (random_lognormal ({20}, 0.0, 1.0, rng)) > 0.0);
      CHECK_TRUE (min (random_weibull ({20}, 1.5, rng)) >= 0.0);
      CHECK_TRUE (min (random_pareto ({20}, 2.0, rng)) >= 0.0);
      CHECK_TRUE (random_laplace ({20}, 0.0, 1.0, rng).size () == 20);
      CHECK_TRUE (random_logistic ({20}, 0.0, 1.0, rng).size () == 20);
      CHECK_TRUE (random_gumbel ({20}, 0.0, 1.0, rng).size () == 20);
      CHECK_TRUE (min (random_rayleigh ({20}, 1.0, rng)) >= 0.0);
      CHECK_TRUE (min (random_triangular ({20}, 0.0, 0.5, 1.0, rng)) >= 0.0);
      CHECK_TRUE (min (random_wald ({20}, 1.0, 1.0, rng)) > 0.0);
      CHECK_TRUE (random_vonmises ({20}, 0.0, 1.0, rng).size () == 20);
      CHECK_TRUE (min (random_geometric ({20}, 0.5, rng)) >= 1);
      CHECK_TRUE (min (random_hypergeometric ({20}, 10, 10, 5, rng)) >= 0);
      CHECK_TRUE (min (random_zipf ({20}, 2.0, rng)) >= 1);
      CHECK_TRUE (random_standard_cauchy ({20}, rng).size () == 20);
      CHECK_TRUE (random_standard_t ({20}, 5.0, rng).size () == 20);
      CHECK_TRUE (min (random_f ({20}, 4.0, 6.0, rng)) >= 0.0);
      auto dir_s = random_dirichlet ({1.0, 2.0, 3.0}, 5, rng);
      CHECK_NEAR (sum (dir_s), 5.0, 1e-9);
      auto mult_s = random_multinomial (10, {0.2, 0.3, 0.5}, 4, rng);
      CHECK_TRUE (sum (mult_s) == 40);

      auto ff = fromfunction (
        [] (const std::vector<std::size_t> &c) {
          return static_cast<double> (10 * c[0] + c[1]);
        },
        shape_type {2, 3});
      CHECK_NEAR (ff (1, 2), 12.0, 1e-12);
      auto ind_grid = indices ({2, 3});
      CHECK_TRUE (ind_grid (0, 1, 2) == 1 && ind_grid (1, 1, 2) == 2);
      auto m_ind = mask_indices (3, [] (std::size_t r, std::size_t c) {
        return r < c;
      });
      CHECK_TRUE (m_ind.first.size () == 3);

      nd_tensor<std::complex<double>> cp = {
        std::complex<double> (2.0, 0.0),
        std::complex<double> (3.0, 0.5)
      };
      CHECK_TRUE (isreal (cp)[0] == 1 && iscomplex (cp)[1] == 1);
      nd_tensor<std::complex<double>> cp_real = {
        std::complex<double> (2.0, 1e-15),
        std::complex<double> (3.0, -1e-15)
      };
      CHECK_NEAR (real_if_close (cp_real)[0], 2.0, 1e-12);

      nd_tensor<double> q2d = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
      CHECK_NEAR (quantile (q2d, 0.5, 1)[0], 2.0, 1e-12);
      CHECK_NEAR (nanquantile (q2d, 0.5, 0)[1], 3.5, 1e-12);

      auto md = multi_dot<double> ({
        nd_tensor<double>::eye (2),
        nd_tensor<double> {{2.0, 0.0}, {0.0, 3.0}},
        nd_tensor<double>::eye (2)
      });
      CHECK_NEAR (md (1, 1), 3.0, 1e-12);

      nd_tensor<double> p_src = {1.0, 2.0, 3.0};
      CHECK_NEAR (pad (p_src, 1, 1, "symmetric")[0], 1.0, 1e-12);
      CHECK_NEAR (pad (p_src, 1, 1, "mean")[0], 2.0, 1e-12);
      CHECK_NEAR (pad (p_src, 1, 1, "median")[0], 2.0, 1e-12);
      CHECK_NEAR (pad (p_src, 1, 1, "minimum")[0], 1.0, 1e-12);
      CHECK_NEAR (pad (p_src, 1, 1, "maximum")[4], 3.0, 1e-12);

      nd_tensor<double> px = {-1.0, -0.5, 0.0, 0.5, 1.0};
      nd_tensor<double> py = {1.0, -0.5, -1.0, -0.5, 1.0}; /* T_2(x) = 2x^2 - 1 */
      auto c_cheb = chebfit (px, py, 2);
      CHECK_NEAR (chebval (0.5, c_cheb), -0.5, 1e-9);
      auto c_cheb_d = chebder (c_cheb);
      CHECK_NEAR (chebval (0.5, c_cheb_d), 2.0, 1e-9);
      auto c_cheb_i = chebint (c_cheb_d);
      CHECK_NEAR (chebval (0.5, c_cheb_i) - chebval (0.0, c_cheb_i), 0.5, 1e-9);

      auto c_leg = legfit (px, py, 2);
      CHECK_NEAR (legval (0.5, c_leg), -0.5, 1e-9);
      CHECK_NEAR (legval (0.5, legder (c_leg)), 2.0, 1e-9);
      CHECK_NEAR (legval (0.5, legint (legder (c_leg))), 0.5, 1e-9);

      auto c_herm = hermfit (px, py, 2);
      CHECK_NEAR (hermval (0.5, c_herm), -0.5, 1e-9);
      auto c_lag = lagfit (px, py, 2);
      CHECK_NEAR (lagval (0.5, c_lag), -0.5, 1e-9);

      auto aaa = apply_along_axis ([] (const nd_tensor<double> &v) {
        return sum (v);
      }, 1, q2d);
      CHECK_NEAR (aaa[0], 6.0, 1e-12);

      auto sw2d = sliding_window_view_2d (q2d, 2, 2);
      CHECK_TRUE (sw2d.shape ()[0] == 1 && sw2d.shape ()[1] == 2);

      auto h_edges = histogram_bin_edges (px, "sturges");
      CHECK_TRUE (h_edges.size () >= 2);
      auto hdd = histogramdd (q2d, {2, 2, 2});
      CHECK_NEAR (sum (hdd.hist), 2.0, 1e-12);
    }
  }

  std::cout << "  -> Suite 1 passed.\n";
  return 0;
}
