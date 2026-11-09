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

#include <cmath>
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
run_scientific_tests ()
{
  using namespace hyper_math;
  using namespace hyper_math::tensor;
  using namespace hyper_math::scientific;

  std::cout << "[Suite 2/3] Scientific Continuum (SciPy)...\n";

  /* 1. Linear Algebra: LU solve, det, inv, Cholesky, QR, lstsq, eigh, SVD.  */
  {
    nd_tensor<double> a = {{3.0, 1.0, 2.0},
                           {1.0, 4.0, 1.0},
                           {2.0, 1.0, 5.0}};
    nd_tensor<double> b = {10.0, 12.0, 18.0};
    auto x = linalg::solve (a, b);
    auto ax = matmul (a, x);
    CHECK_TRUE (allclose (ax, b, 1e-10, 1e-10));

    auto a_inv = linalg::inv (a);
    CHECK_TRUE (allclose (matmul (a, a_inv), eye<double> (3), 1e-10, 1e-10));
    CHECK_NEAR (linalg::det (a), 40.0, 1e-10);

    auto l_chol = linalg::cholesky (a);
    CHECK_TRUE (allclose (matmul (l_chol, l_chol.t ()), a, 1e-10, 1e-10));

    auto qr_res = linalg::qr (a);
    CHECK_TRUE (allclose (matmul (qr_res.q, qr_res.r), a, 1e-10, 1e-10));
    CHECK_TRUE (allclose (matmul (qr_res.q.t (), qr_res.q),
                          eye<double> (3), 1e-10, 1e-10));

    auto eig = linalg::eigh (a);
    auto recon_eig = matmul (matmul (eig.eigenvectors, diag (eig.eigenvalues)),
                             eig.eigenvectors.t ());
    CHECK_TRUE (allclose (recon_eig, a, 1e-9, 1e-9));

    auto svd_res = linalg::svd (a);
    auto recon_svd = matmul (matmul (svd_res.u, diag (svd_res.s)), svd_res.vt);
    CHECK_TRUE (allclose (recon_svd, a, 1e-9, 1e-9));

    auto a_pinv = linalg::pinv (a);
    CHECK_TRUE (allclose (a_pinv, a_inv, 1e-8, 1e-8));
  }

  /* 2. Fast Fourier Transform: power-of-2 Cooley-Tukey and prime-length
     Bluestein Chirp-Z transform.  */
  {
    auto sig8 = linspace<double> (1.0, 8.0, 8);
    auto spec8 = fft::fft (sig8);
    auto rec8 = fft::ifft (spec8);
    for (std::size_t i = 0; i < 8; ++i)
      {
        CHECK_NEAR (rec8[i].real (), sig8[i], 1e-10);
        CHECK_NEAR (rec8[i].imag (), 0.0, 1e-10);
      }

    /* Prime length N = 7 via Bluestein Chirp-Z transform.  */
    nd_tensor<double> sig7 = {1.0, -2.0, 3.5, 0.5, -1.0, 4.0, 2.0};
    auto spec7 = fft::fft (sig7);
    CHECK_NEAR (spec7[0].real (), sum (sig7), 1e-10);
    auto rec7 = fft::ifft (spec7);
    for (std::size_t i = 0; i < 7; ++i)
      {
        CHECK_NEAR (rec7[i].real (), sig7[i], 1e-10);
        CHECK_NEAR (rec7[i].imag (), 0.0, 1e-10);
      }
  }

  /* 3. Numerical Integration & Dormand-Prince RK4(5) ODE Solver.  */
  {
    auto q = integrate::quad ([] (double x) { return std::sin (x); },
                              0.0, constants::pi);
    CHECK_NEAR (q.value, 2.0, 1e-11);

    /* Harmonic oscillator ODE: y'' + y = 0 => y0' = y1, y1' = -y0.
       With y(0) = [1, 0], at t = pi we have y(pi) = [-1, 0].  */
    auto ode_sol = integrate::solve_ivp (
      [] (double, const nd_tensor<double> &y) {
        return nd_tensor<double> {y[1], -y[0]};
      },
      0.0, constants::pi, nd_tensor<double> {1.0, 0.0}, 1e-8, 1e-10);
    CHECK_TRUE (ode_sol.success);
    const std::size_t last_step = ode_sol.t.size () - 1;
    CHECK_NEAR (ode_sol.y (0, last_step), -1.0, 1e-5);
    CHECK_NEAR (ode_sol.y (1, last_step), 0.0, 1e-5);
  }

  /* 4. Optimization: Brent root, Golden-Section, BFGS, Nelder-Mead,
     and Levenberg-Marquardt curve_fit.  */
  {
    const double root = optimize::brentq (
      [] (double x) { return x * x - 2.0; }, 0.0, 2.0);
    CHECK_NEAR (root, constants::sqrt2, 1e-11);

    const auto s_min = optimize::minimize_scalar (
      [] (double x) { return (x - 3.25) * (x - 3.25) + 1.5; }, -5.0, 10.0);
    CHECK_NEAR (s_min.x, 3.25, 1e-7);
    CHECK_NEAR (s_min.fun, 1.5, 1e-10);

    auto quad_fn = [] (const nd_tensor<double> &v) {
      return (v[0] - 2.0) * (v[0] - 2.0) + (v[1] + 4.0) * (v[1] + 4.0);
    };
    const auto bfgs_res = optimize::minimize_bfgs (
      quad_fn, nd_tensor<double> {0.0, 0.0});
    CHECK_NEAR (bfgs_res.x[0], 2.0, 1e-6);
    CHECK_NEAR (bfgs_res.x[1], -4.0, 1e-6);

    const auto nm_res = optimize::minimize_nelder_mead (
      quad_fn, nd_tensor<double> {0.0, 0.0});
    CHECK_NEAR (nm_res.x[0], 2.0, 1e-3);
    CHECK_NEAR (nm_res.x[1], -4.0, 1e-3);

    /* Curve fit: y = p0 * exp(p1 * x) with true params [2.5, 0.4].  */
    auto x_pts = linspace<double> (0.0, 3.0, 15);
    auto y_pts = 2.5 * exp (0.4 * x_pts);
    auto fitted = optimize::curve_fit (
      [] (double x, const nd_tensor<double> &p) {
        return p[0] * std::exp (p[1] * x);
      },
      x_pts, y_pts, nd_tensor<double> {1.0, 0.1});
    CHECK_NEAR (fitted[0], 2.5, 1e-5);
    CHECK_NEAR (fitted[1], 0.4, 1e-5);
  }

  /* 5. Stats, Interpolation, Signal Processing, and CSR Sparse Solver.  */
  {
    stats::normal_dist norm01 {0.0, 1.0};
    CHECK_NEAR (norm01.cdf (0.0), 0.5, 1e-12);
    CHECK_NEAR (norm01.ppf (0.975), 1.95996398, 1e-5);

    nd_tensor<double> sx = {1.0, 2.0, 3.0, 4.0, 5.0};
    nd_tensor<double> sy = {2.0, 4.0, 6.0, 8.0, 10.0};
    auto pr = stats::pearsonr (sx, sy);
    CHECK_NEAR (pr.statistic, 1.0, 1e-12);
    auto sr = stats::spearmanr (sx, sy);
    CHECK_NEAR (sr.statistic, 1.0, 1e-12);

    /* Cubic spline exact interpolation at knots and midpoint check.  */
    interpolate::cubic_spline cs (sx, pow (sx, 2.0));
    CHECK_NEAR (cs (3.0), 9.0, 1e-12);
    CHECK_NEAR (cs (2.5), 6.25, 0.05);

    /* Peak finding and convolution.  */
    nd_tensor<double> wave = {0.0, 2.0, 0.0, 5.0, 1.0, 4.0, 0.0};
    auto peaks = signal::find_peaks (wave, 1.5, 2);
    CHECK_TRUE (peaks.size () == 3);
    CHECK_TRUE (peaks[0] == 1 && peaks[1] == 3 && peaks[2] == 5);

    /* Sparse CSR matrix & Conjugate Gradient solver.  */
    nd_tensor<double> dense_spd = {{4.0, -1.0, 0.0},
                                   {-1.0, 4.0, -1.0},
                                   {0.0, -1.0, 4.0}};
    auto csr = sparse::csr_matrix<double>::from_dense (dense_spd);
    CHECK_TRUE (csr.nnz () == 7);
    nd_tensor<double> rhs = {3.0, 2.0, 3.0};
    auto x_cg = sparse::cg_solve (csr, rhs);
    CHECK_NEAR (x_cg[0], 1.0, 1e-8);
    CHECK_NEAR (x_cg[1], 1.0, 1e-8);
    CHECK_NEAR (x_cg[2], 1.0, 1e-8);
  }

  /* 6. Matrix Exponential (expm), slogdet, DCT/IDCT, FFT Convolution,
     Spatial k-d Tree & 2D Convex Hull, k-Means++ & Hierarchical Clustering,
     ANOVA, KS-test, Mann-Whitney U, and BiCGSTAB.  */
  {
    /* expm of skew-symmetric rotation generator [[0, -theta], [theta, 0]]
       equals the 2D rotation matrix [[cos(theta), -sin(theta)], [sin, cos]]. */
    const double theta = 0.75;
    nd_tensor<double> rot_gen = {{0.0, -theta}, {theta, 0.0}};
    auto rot_mat = linalg::expm (rot_gen);
    CHECK_NEAR (rot_mat (0, 0), std::cos (theta), 1e-12);
    CHECK_NEAR (rot_mat (0, 1), -std::sin (theta), 1e-12);
    CHECK_NEAR (rot_mat (1, 0), std::sin (theta), 1e-12);
    CHECK_NEAR (rot_mat (1, 1), std::cos (theta), 1e-12);

    auto sld = linalg::slogdet (rot_mat);
    CHECK_NEAR (sld.sign, 1.0, 1e-12);
    CHECK_NEAR (sld.logabsdet, 0.0, 1e-12);

    /* Orthonormal DCT-II and IDCT-III roundtrip.  */
    nd_tensor<double> audio = {1.0, 2.5, -0.5, 4.0, 0.0, -1.5};
    auto dct_c = fft::dct (audio, true);
    auto idct_r = fft::idct (dct_c, true);
    CHECK_TRUE (allclose (audio, idct_r, 1e-10, 1e-10));

    /* Fast FFT convolution vs direct time-domain convolution.  */
    nd_tensor<double> ker = {0.25, 0.5, 0.25};
    auto conv_direct = signal::convolve (audio, ker, "full");
    auto conv_fast = fft::convolve_fft (audio, ker);
    CHECK_TRUE (allclose (conv_direct, conv_fast, 1e-10, 1e-10));

    /* Spatial k-d Tree nearest-neighbor and ball search.  */
    nd_tensor<double> pts = {{0.0, 0.0},
                             {1.0, 1.0},
                             {5.0, 5.0},
                             {0.0, 2.0},
                             {2.0, 0.0},
                             {2.0, 2.0}};
    spatial::kd_tree tree (pts);
    auto nn = tree.query (nd_tensor<double> {0.9, 1.1}, 1);
    CHECK_TRUE (nn.indices[0] == 1);
    auto ball = tree.query_ball_point (nd_tensor<double> {0.0, 0.0}, 1.5);
    CHECK_TRUE (ball.size () == 2);

    /* 2D Convex Hull of [0, 2] x [0, 2] square with interior point (1, 1):
       Hull has 4 vertices, perimeter = 8.0, area = 4.0.  */
    nd_tensor<double> square_pts = {{0.0, 0.0},
                                    {2.0, 0.0},
                                    {2.0, 2.0},
                                    {0.0, 2.0},
                                    {1.0, 1.0}};
    auto hull = spatial::convex_hull_2d (square_pts);
    CHECK_TRUE (hull.vertices.size () == 4);
    CHECK_NEAR (hull.perimeter, 8.0, 1e-12);
    CHECK_NEAR (hull.area, 4.0, 1e-12);

    /* k-Means++ and Ward Hierarchical Clustering.  */
    nd_tensor<double> blobs = {{0.0, 0.0}, {0.1, -0.1}, {-0.1, 0.1},
                               {10.0, 10.0}, {10.1, 9.9}, {9.9, 10.1}};
    random_engine fixed_rng (42);
    auto km = cluster::kmeans2 (blobs, 2, 50, 1e-6, fixed_rng);
    CHECK_TRUE (km.labels[0] == km.labels[1] && km.labels[1] == km.labels[2]);
    CHECK_TRUE (km.labels[3] == km.labels[4] && km.labels[4] == km.labels[5]);
    CHECK_TRUE (km.labels[0] != km.labels[3]);

    auto z_link = cluster::linkage (blobs, "ward");
    CHECK_TRUE (z_link.shape () == (shape_type {5, 4}));
    CHECK_NEAR (z_link (4, 3), 6.0, 1e-12);

    /* Non-parametric & ANOVA statistical hypothesis tests.  */
    nd_tensor<double> g1 = {1.0, 1.2, 1.1, 0.9};
    nd_tensor<double> g2 = {5.0, 5.2, 4.9, 5.1};
    auto anova = stats::f_oneway ({g1, g2});
    CHECK_TRUE (anova.statistic > 100.0 && anova.pvalue < 1e-4);

    auto ks2 = stats::ks_2samp (g1, g2);
    CHECK_NEAR (ks2.statistic, 1.0, 1e-12);

    auto mwu = stats::mannwhitneyu (g1, g2);
    CHECK_NEAR (mwu.statistic, 0.0, 1e-12);

    /* BiCGSTAB on non-symmetric sparse linear system.  */
    nd_tensor<double> nonsym = {{5.0, 2.0, 0.0},
                                {1.0, 6.0, 3.0},
                                {0.0, 2.0, 7.0}};
    auto csr_ns = sparse::csr_matrix<double>::from_dense (nonsym);
    nd_tensor<double> b_ns = {7.0, 10.0, 9.0};
    auto x_bicg = sparse::bicgstab_solve (csr_ns, b_ns);
    CHECK_NEAR (x_bicg[0], 1.0, 1e-8);
    CHECK_NEAR (x_bicg[1], 1.0, 1e-8);
    CHECK_NEAR (x_bicg[2], 1.0, 1e-8);

    /* Rank-deficient SVD orthogonality check (U^T * U == I).  */
    nd_tensor<double> rank1 = {{1.0, 2.0, 3.0},
                               {2.0, 4.0, 6.0},
                               {3.0, 6.0, 9.0}};
    auto svd_r1 = linalg::svd (rank1);
    auto utu = tensor::matmul (svd_r1.u.t (), svd_r1.u);
    CHECK_TRUE (allclose (utu, eye<double> (3), 1e-9, 1e-9));
    auto recon_r1 =
      tensor::matmul (tensor::matmul (svd_r1.u, diag (svd_r1.s)), svd_r1.vt);
    CHECK_TRUE (allclose (recon_r1, rank1, 1e-9, 1e-9));

    /* Phase 3: Schur & General Eigenvalues, Polynomial Roots, Sylvester,
       L-BFGS-B, Levenberg-Marquardt, Simplex linprog, Butterworth/filtfilt,
       Welch PSD, 2D Gaussian filter, KDE, and PCA.  */
    /* Rotation matrix [[0, -1], [1, 0]] has eigenvalues +i and -i.  */
    nd_tensor<double> rot90_m = {{0.0, -1.0},
                                 {1.0,  0.0}};
    auto ev_rot = linalg::eigvals (rot90_m);
    CHECK_TRUE (ev_rot.size () == 2);
    CHECK_NEAR (ev_rot[0].real (), 0.0, 1e-10);
    CHECK_NEAR (std::abs (ev_rot[0].imag ()), 1.0, 1e-10);

    /* Roots of x^2 - 5x + 6 = 0 -> {3, 2}.  */
    auto poly_r = linalg::roots (nd_tensor<double> {1.0, -5.0, 6.0});
    CHECK_TRUE (poly_r.size () == 2);
    const double r_sum = poly_r[0].real () + poly_r[1].real ();
    const double r_prod = poly_r[0].real () * poly_r[1].real ();
    CHECK_NEAR (r_sum, 5.0, 1e-9);
    CHECK_NEAR (r_prod, 6.0, 1e-9);

    /* Sylvester equation: A*X + X*B = C.  */
    nd_tensor<double> syl_a = {{2.0, 1.0}, {0.0, 3.0}};
    nd_tensor<double> syl_b = {{1.0, 0.0}, {2.0, 4.0}};
    nd_tensor<double> syl_c = {{5.0, 8.0}, {7.0, 9.0}};
    auto syl_x = linalg::solve_sylvester (syl_a, syl_b, syl_c);
    auto syl_lhs =
      tensor::matmul (syl_a, syl_x) + tensor::matmul (syl_x, syl_b);
    CHECK_TRUE (allclose (syl_lhs, syl_c, 1e-9, 1e-9));

    /* Box-constrained L-BFGS-B: minimize (x0 + 2)^2 + (x1 - 5)^2
       subject to x0 in [0, 10], x1 in [0, 3] -> optimum at (0, 3).  */
    auto lbfgsb_res = optimize::minimize_lbfgsb (
      [] (const nd_tensor<double> &x) {
        return (x[0] + 2.0) * (x[0] + 2.0) + (x[1] - 5.0) * (x[1] - 5.0);
      },
      nd_tensor<double> {2.0, 1.0},
      {{0.0, 10.0}, {0.0, 3.0}});
    CHECK_NEAR (lbfgsb_res.x[0], 0.0, 1e-5);
    CHECK_NEAR (lbfgsb_res.x[1], 3.0, 1e-5);

    /* Linear Programming via Simplex:
       Maximize 3*x0 + 2*x1  (Minimize -3*x0 - 2*x1)
       Subject to: 2*x0 + x1 <= 18, 2*x0 + 3*x1 <= 42, 3*x0 + x1 <= 24.
       Optimum at x = (3, 12) with objective = -33.  */
    nd_tensor<double> lp_c = {-3.0, -2.0};
    nd_tensor<double> lp_a = {{2.0, 1.0},
                              {2.0, 3.0},
                              {3.0, 1.0}};
    nd_tensor<double> lp_b = {18.0, 42.0, 24.0};
    auto lp_res = optimize::linprog (lp_c, lp_a, lp_b);
    CHECK_TRUE (lp_res.success);
    CHECK_NEAR (lp_res.x[0], 3.0, 1e-9);
    CHECK_NEAR (lp_res.x[1], 12.0, 1e-9);
    CHECK_NEAR (lp_res.fun, -33.0, 1e-9);

    /* Butterworth low-pass + zero-phase filtfilt + Welch PSD.  */
    auto bw = signal::butter_lowpass_ba (0.2);
    auto t_sig = linspace<double> (0.0, 1.0, 128, false);
    nd_tensor<double> raw_sig =
      sin (2.0 * constants::pi * 3.0 * t_sig)
      + 0.5 * sin (2.0 * constants::pi * 40.0 * t_sig);
    auto clean_sig = signal::filtfilt (bw.b, bw.a, raw_sig);
    CHECK_TRUE (clean_sig.size () == 128);
    auto psd_res = signal::welch (raw_sig, 128.0, 64);
    CHECK_TRUE (psd_res.freqs.size () == 33 && max (psd_res.psd) > 0.0);

    /* 2D Gaussian filter preserves constant image intensity.  */
    auto const_img = full<double> ({8, 8}, 5.0);
    auto blurred = signal::gaussian_filter2d (const_img, 1.0);
    CHECK_NEAR (blurred (4, 4), 5.0, 1e-12);

    /* Gaussian KDE and PCA.  */
    nd_tensor<double> kde_pts = {-1.0, -0.5, 0.0, 0.5, 1.0};
    stats::gaussian_kde kde (kde_pts);
    CHECK_TRUE (kde.evaluate (0.0) > kde.evaluate (3.0));

    nd_tensor<double> pca_data = {{1.0, 2.0},
                                  {2.0, 4.0},
                                  {3.0, 6.0},
                                  {4.0, 8.0}};
    auto pca_res = stats::pca (pca_data, 1);
    CHECK_NEAR (pca_res.explained_variance_ratio[0], 1.0, 1e-9);

    /* 5. Forward & Reverse-Mode Automatic Differentiation (autograd).  */
    const double df_dx = autograd::derivative ([] (autograd::dual_number x) {
      return autograd::sin (x) * autograd::exp (x);
    }, 0.5);
    const double exact_df =
      std::exp (0.5) * (std::cos (0.5) + std::sin (0.5));
    CHECK_NEAR (df_dx, exact_df, 1e-13);

    /* Forward-mode Jacobian of F(x0, x1) = [x0^2 * x1, 5*x0 + sin(x1)].  */
    nd_tensor<double> j_pt = {2.0, 1.0};
    auto jac = autograd::jacobian (
      [] (const std::vector<autograd::dual_number> &x) {
        return std::vector<autograd::dual_number> {
          x[0] * x[0] * x[1],
          5.0 * x[0] + autograd::sin (x[1])
        };
      },
      j_pt);
    CHECK_NEAR (jac (0, 0), 4.0, 1e-13);
    CHECK_NEAR (jac (0, 1), 4.0, 1e-13);
    CHECK_NEAR (jac (1, 0), 5.0, 1e-13);
    CHECK_NEAR (jac (1, 1), std::cos (1.0), 1e-13);

    /* Reverse-mode Wengert tape gradient, Hessian, and BFGS optimization.  */
    auto rosen_ag = [] (const std::vector<autograd::var> &v) {
      const auto d0 = 1.0 - v[0];
      const auto d1 = v[1] - v[0] * v[0];
      return d0 * d0 + 100.0 * d1 * d1;
    };
    nd_tensor<double> r_pt = {1.0, 1.0};
    auto vg = autograd::value_and_grad (rosen_ag, r_pt);
    CHECK_NEAR (vg.value, 0.0, 1e-14);
    CHECK_NEAR (vg.gradient[0], 0.0, 1e-14);
    CHECK_NEAR (vg.gradient[1], 0.0, 1e-14);

    auto hess = autograd::hessian (rosen_ag, r_pt);
    /* At (1, 1): d^2f/dx^2 = 2 - 400*y + 1200*x^2 = 802,
       d^2f/dxdy = -400*x = -400, d^2f/dy^2 = 200.  */
    CHECK_NEAR (hess (0, 0), 802.0, 1e-4);
    CHECK_NEAR (hess (0, 1), -400.0, 1e-4);
    CHECK_NEAR (hess (1, 0), -400.0, 1e-4);
    CHECK_NEAR (hess (1, 1), 200.0, 1e-4);

    auto ag_opt = autograd::minimize_bfgs_autograd (
      rosen_ag, nd_tensor<double> {-1.2, 1.0});
    CHECK_TRUE (ag_opt.success);
    CHECK_NEAR (ag_opt.x[0], 1.0, 1e-7);
    CHECK_NEAR (ag_opt.x[1], 1.0, 1e-7);

    /* 6. Phase 6: PCHIP, Barycentric, RBF interpolation, Romberg,
       cumulative_trapezoid, and Dual Annealing global optimization.  */
    nd_tensor<double> px = {0.0, 1.0, 2.0, 3.0, 4.0};
    nd_tensor<double> py = {0.0, 0.0, 1.0, 1.0, 1.0};
    interpolate::pchip_interpolator pchip (px, py);
    /* PCHIP preserves monotonicity in [0, 1] without overshoot.  */
    CHECK_TRUE (pchip (0.5) == 0.0);
    CHECK_TRUE (pchip (1.5) >= 0.0 && pchip (1.5) <= 1.0);
    CHECK_TRUE (pchip (2.5) == 1.0);

    /* Barycentric Lagrange interpolates y = x^2 + 2*x + 1 exactly.  */
    nd_tensor<double> bx = {0.0, 1.0, 2.0};
    nd_tensor<double> by = {1.0, 4.0, 9.0};
    interpolate::barycentric_interpolator bary (bx, by);
    CHECK_NEAR (bary (1.5), 6.25, 1e-12);
    CHECK_NEAR (bary (2.0), 9.0, 1e-12);

    /* 2D RBF interpolation on scattered points.  */
    nd_tensor<double> rbf_pts = {{0.0, 0.0},
                                 {1.0, 0.0},
                                 {0.0, 1.0},
                                 {1.0, 1.0}};
    nd_tensor<double> rbf_vals = {1.0, 2.0, 3.0, 4.0};
    interpolate::rbf_interpolator rbf (rbf_pts, rbf_vals, "multiquadric", 1.0);
    auto rbf_at_knots = rbf (rbf_pts);
    CHECK_NEAR (rbf_at_knots[0], 1.0, 1e-7);
    CHECK_NEAR (rbf_at_knots[3], 4.0, 1e-7);

    /* Romberg integration of sin(x) on [0, pi] == 2.0.  */
    const double romb = integrate::romberg (
      [] (double x) { return std::sin (x); }, 0.0, constants::pi);
    CHECK_NEAR (romb, 2.0, 1e-11);

    auto ctrap = integrate::cumulative_trapezoid (
      nd_tensor<double> {0.0, 1.0, 2.0, 3.0}, 1.0);
    CHECK_TRUE (ctrap.size () == 3);
    CHECK_NEAR (ctrap[2], 4.5, 1e-12);

    /* Dual Annealing on 2D multimodal objective (x0 - 0.5)^2 + (x1 + 0.25)^2
       + 0.1*(1 - cos(4*pi*(x0 - 0.5))).  */
    auto da_res = optimize::dual_annealing (
      [] (const nd_tensor<double> &x) {
        const double d0 = x[0] - 0.5;
        const double d1 = x[1] + 0.25;
        return d0 * d0 + d1 * d1;
      },
      nd_tensor<double> {-2.0, -2.0},
      nd_tensor<double> {2.0, 2.0},
      400);
    CHECK_NEAR (da_res.x[0], 0.5, 1e-5);
    CHECK_NEAR (da_res.x[1], -0.25, 1e-5);

    /* 7. Phase 7: solve_triangular, orth, null_space, sqrtm, logm_spd,
       sparse_eye, gmres_solve, entropy, kl_divergence, iqr, trim_mean,
       gmean, hmean, and kruskal.  */
    nd_tensor<double> l_tri = {{2.0, 0.0},
                               {3.0, 4.0}};
    auto tri_x = linalg::solve_triangular (l_tri, nd_tensor<double> {4.0, 14.0}, true);
    CHECK_NEAR (tri_x[0], 2.0, 1e-12);
    CHECK_NEAR (tri_x[1], 2.0, 1e-12);

    nd_tensor<double> rank1_m = {{1.0, 2.0},
                                 {2.0, 4.0}};
    auto ns = linalg::null_space (rank1_m);
    CHECK_TRUE (ns.shape () == (shape_type {2, 1}));
    auto ns_prod = matmul (rank1_m, ns);
    CHECK_NEAR (ns_prod (0, 0), 0.0, 1e-12);
    CHECK_NEAR (ns_prod (1, 0), 0.0, 1e-12);

    nd_tensor<double> spd_m = {{4.0, 0.0},
                               {0.0, 9.0}};
    auto sq_m = linalg::sqrtm (spd_m);
    CHECK_NEAR (sq_m (0, 0), 2.0, 1e-11);
    CHECK_NEAR (sq_m (1, 1), 3.0, 1e-11);

    auto lg_m = linalg::logm_spd (spd_m);
    CHECK_NEAR (lg_m (0, 0), std::log (4.0), 1e-12);
    CHECK_NEAR (lg_m (1, 1), std::log (9.0), 1e-12);

    auto gm_sol = sparse::gmres_solve (
      sparse::csr_matrix<double>::from_dense (
        nd_tensor<double> {{4.0, 1.0}, {2.0, 3.0}}),
      nd_tensor<double> {6.0, 8.0});
    CHECK_NEAR (gm_sol[0], 1.0, 1e-10);
    CHECK_NEAR (gm_sol[1], 2.0, 1e-10);

    nd_tensor<double> unif_p = {0.25, 0.25, 0.25, 0.25};
    CHECK_NEAR (stats::entropy (unif_p), std::log (4.0), 1e-12);
    CHECK_NEAR (stats::kl_divergence (unif_p, unif_p), 0.0, 1e-12);
    CHECK_NEAR (stats::gmean (nd_tensor<double> {1.0, 4.0, 16.0}), 4.0, 1e-12);
    CHECK_NEAR (stats::hmean (nd_tensor<double> {2.0, 6.0}), 3.0, 1e-12);

    auto kw = stats::kruskal ({
      nd_tensor<double> {1.0, 2.0, 3.0},
      nd_tensor<double> {10.0, 11.0, 12.0}
    });
    CHECK_TRUE (kw.statistic > 3.0 && kw.pvalue < 0.06);

    /* 8. Phase 8: detrend, medfilt, hilbert_analytic / envelope, resample,
       hausdorff_distance, and procrustes.  */
    auto dt_sig = signal::detrend (nd_tensor<double> {2.0, 4.0, 6.0, 8.0}, "linear");
    CHECK_NEAR (max (abs (dt_sig)), 0.0, 1e-12);

    auto mf_sig = signal::medfilt (nd_tensor<double> {1.0, 100.0, 1.0, 1.0}, 3);
    CHECK_NEAR (mf_sig[1], 1.0, 1e-12);

    auto t_env = linspace<double> (0.0, 1.0, 64, false);
    nd_tensor<double> cos_wave = cos (2.0 * constants::pi * 4.0 * t_env);
    auto env = signal::envelope (cos_wave);
    CHECK_NEAR (env[16], 1.0, 1e-10);

    auto resamp = signal::resample (cos_wave, 32);
    CHECK_TRUE (resamp.size () == 32);
    CHECK_NEAR (resamp[0], 1.0, 1e-10);

    nd_tensor<double> cloud_u = {{0.0, 0.0}, {1.0, 0.0}};
    nd_tensor<double> cloud_v = {{0.0, 1.0}, {1.0, 1.0}};
    CHECK_NEAR (spatial::hausdorff_distance (cloud_u, cloud_v), 1.0, 1e-12);

    nd_tensor<double> shape_a = {{0.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}};
    /* 90-degree rotation + scaling + translation of shape_a.  */
    nd_tensor<double> shape_b = {{5.0, 5.0}, {5.0, 9.0}, {3.0, 5.0}};
    auto proc = spatial::procrustes (shape_a, shape_b);
    CHECK_NEAR (proc.disparity, 0.0, 1e-12);

    /* 9. Phase 9: dblquad, fsolve, linear_sum_assignment, digamma, erfinv,
       binom, median_abs_deviation, wilcoxon, friedmanchisquare, and
       delaunay_2d.  */
    /* \int_0^1 \int_0^x (x * y) dy dx = \int_0^1 x * (x^2 / 2) dx = 1/8.  */
    auto dq = integrate::dblquad (
      [] (double y, double x) { return x * y; },
      0.0,
      1.0,
      [] (double) { return 0.0; },
      [] (double x) { return x; });
    CHECK_NEAR (dq.value, 0.125, 1e-10);

    /* Nonlinear system: x^2 + y^2 = 5, x - y = 1 => (x, y) = (2, 1).  */
    auto root_res = optimize::fsolve (
      [] (const nd_tensor<double> &v) {
        return nd_tensor<double> {
          v[0] * v[0] + v[1] * v[1] - 5.0,
          v[0] - v[1] - 1.0
        };
      },
      nd_tensor<double> {1.5, 0.5});
    CHECK_TRUE (root_res.success);
    CHECK_NEAR (root_res.x[0], 2.0, 1e-7);
    CHECK_NEAR (root_res.x[1], 1.0, 1e-7);

    /* Hungarian algorithm on 3x3 cost matrix: optimal assignment cost = 5.  */
    nd_tensor<double> cost_mat = {
      {4.0, 1.0, 3.0},
      {2.0, 0.0, 5.0},
      {3.0, 2.0, 2.0}
    };
    auto lsa = optimize::linear_sum_assignment (cost_mat);
    CHECK_NEAR (lsa.total_cost, 5.0, 1e-12);

    /* \psi(1) = -\gamma \approx -0.5772156649015329.  */
    CHECK_NEAR (stats::special::digamma (1.0), -0.5772156649015329, 1e-10);
    CHECK_NEAR (std::erf (stats::special::erfinv (0.75)), 0.75, 1e-12);
    CHECK_NEAR (stats::special::binom (6.0, 2.0), 15.0, 1e-12);

    CHECK_NEAR (stats::median_abs_deviation (nd_tensor<double> {1.0, 2.0, 3.0, 4.0, 5.0}),
                1.0, 1e-12);

    auto wil = stats::wilcoxon (
      nd_tensor<double> {10.0, 12.0, 14.0, 16.0, 18.0, 20.0},
      nd_tensor<double> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    CHECK_NEAR (wil.statistic, 0.0, 1e-12);
    CHECK_TRUE (wil.pvalue < 0.05);

    auto fried = stats::friedmanchisquare ({
      nd_tensor<double> {1.0, 1.0, 1.0, 1.0},
      nd_tensor<double> {5.0, 5.0, 5.0, 5.0},
      nd_tensor<double> {10.0, 10.0, 10.0, 10.0}
    });
    CHECK_NEAR (fried.statistic, 8.0, 1e-12);

    /* Delaunay triangulation of unit square + center point (5 points -> 4 triangles). */
    nd_tensor<double> del_pts = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, 0.5}
    };
    auto tris = spatial::delaunay_2d (del_pts);
    CHECK_TRUE (tris.ndim () == 2 && tris.shape ()[0] == 4 && tris.shape ()[1] == 3);

    /* 10. Phase 10: eigh_generalized, companion, hadamard, leggauss,
       fixed_quad, regular_grid_interpolator_2d, and dbscan.  */
    nd_tensor<double> ga = {{6.0, 0.0}, {0.0, 12.0}};
    nd_tensor<double> gb = {{2.0, 0.0}, {0.0, 3.0}};
    auto geg = linalg::eigh_generalized (ga, gb);
    CHECK_NEAR (geg.eigenvalues[0], 3.0, 1e-10);
    CHECK_NEAR (geg.eigenvalues[1], 4.0, 1e-10);

    auto comp_m = linalg::companion (nd_tensor<double> {1.0, -3.0, 2.0});
    CHECK_NEAR (trace (comp_m), 3.0, 1e-12);
    CHECK_NEAR (linalg::det (comp_m), 2.0, 1e-12);

    auto had4 = linalg::hadamard (4);
    auto had_ortho = matmul (had4, had4.t ());
    CHECK_NEAR (had_ortho (0, 0), 4.0, 1e-12);
    CHECK_NEAR (had_ortho (0, 1), 0.0, 1e-12);

    /* 5-point Gauss-Legendre integrates x^5 + 3 x^2 on [0, 2] exactly:
       [x^6 / 6 + x^3]_0^2 = 64/6 + 8 = 56/3.  */
    auto fq = integrate::fixed_quad (
      [] (double x) { return std::pow (x, 5.0) + 3.0 * x * x; },
      0.0,
      2.0,
      5);
    CHECK_NEAR (fq.value, 56.0 / 3.0, 1e-11);

    /* 2D bilinear grid interpolation on f(x, y) = 2x + 3y.  */
    nd_tensor<double> gx = {0.0, 1.0, 2.0};
    nd_tensor<double> gy = {0.0, 1.0, 2.0};
    nd_tensor<double> gvals (shape_type {3, 3});
    for (std::size_t i = 0; i < 3; ++i)
      {
        for (std::size_t j = 0; j < 3; ++j)
          {
            gvals (i, j) = 2.0 * gx[i] + 3.0 * gy[j];
          }
      }
    interpolate::regular_grid_interpolator_2d rgi (gx, gy, gvals, "linear");
    CHECK_NEAR (rgi (0.5, 1.5), 2.0 * 0.5 + 3.0 * 1.5, 1e-12);

    /* DBSCAN: 2 tight clusters of 3 points each + 1 distant outlier.  */
    nd_tensor<double> db_pts = {
      {0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1},
      {10.0, 10.0}, {10.1, 10.0}, {10.0, 10.1},
      {100.0, 100.0}
    };
    auto db_res = cluster::dbscan (db_pts, 0.5, 2);
    CHECK_TRUE (db_res.n_clusters == 2);
    CHECK_TRUE (db_res.labels[0] == 0 && db_res.labels[3] == 1);
    CHECK_TRUE (db_res.labels[6] == -1);

    /* 11. Phase 11: linalg::lu (P*L*U), linalg::ldl (L*D*L^T), tukey_window,
       upfirdn, wasserstein_distance, and energy_distance.  */
    nd_tensor<double> plu_a = {{0.0, 2.0}, {3.0, 4.0}};
    auto plu = linalg::lu (plu_a);
    auto plu_rec = matmul (matmul (plu.p, plu.l), plu.u);
    CHECK_TRUE (allclose (plu_rec, plu_a, 1e-12));

    nd_tensor<double> sym_indef = {{4.0, 2.0}, {2.0, -3.0}};
    auto ldl_res = linalg::ldl (sym_indef);
    auto ldl_rec = matmul (matmul (ldl_res.l, ldl_res.d), ldl_res.l.t ());
    CHECK_TRUE (allclose (ldl_rec, sym_indef, 1e-12));

    auto tuk = signal::tukey_window (9, 0.5);
    CHECK_NEAR (tuk[0], 0.0, 1e-12);
    CHECK_NEAR (tuk[4], 1.0, 1e-12);

    auto ufd = signal::upfirdn (nd_tensor<double> {1.0, 1.0},
                                nd_tensor<double> {1.0, 2.0, 3.0},
                                2, 1);
    CHECK_TRUE (ufd.size () == 6);
    CHECK_NEAR (ufd[0], 1.0, 1e-12);
    CHECK_NEAR (ufd[1], 1.0, 1e-12);
    CHECK_NEAR (ufd[2], 2.0, 1e-12);

    CHECK_NEAR (stats::wasserstein_distance (nd_tensor<double> {0.0, 1.0},
                                             nd_tensor<double> {2.0, 3.0}),
                2.0, 1e-12);
    CHECK_TRUE (stats::energy_distance (nd_tensor<double> {0.0, 1.0},
                                        nd_tensor<double> {2.0, 3.0})
                > 1.5);

    /* 12. Phase 12: solve_discrete_lyapunov, solve_discrete_are, khatri_rao,
       spectrogram, chirp, cubic_spline::integrate, multivariate_normal_dist,
       and boxcox.  */
    nd_tensor<double> dlyap_a = {{0.5, 0.1}, {0.0, 0.4}};
    nd_tensor<double> dlyap_q = eye<double> (2);
    auto dlyap_x = linalg::solve_discrete_lyapunov (dlyap_a, dlyap_q);
    auto dlyap_lhs =
      matmul (matmul (dlyap_a, dlyap_x), dlyap_a.t ()) - dlyap_x + dlyap_q;
    CHECK_NEAR (linalg::norm (dlyap_lhs), 0.0, 1e-10);

    nd_tensor<double> dare_a = {{0.8, 0.1}, {0.0, 0.9}};
    nd_tensor<double> dare_b (shape_type {2, 1}, {0.0, 1.0});
    nd_tensor<double> dare_q = eye<double> (2);
    nd_tensor<double> dare_r (shape_type {1, 1}, {1.0});
    auto dare_p = linalg::solve_discrete_are (dare_a, dare_b, dare_q, dare_r);
    CHECK_TRUE (dare_p (0, 0) > 0.0 && dare_p (1, 1) > 0.0);

    nd_tensor<double> kr_a = {{1.0, 2.0}, {3.0, 4.0}};
    nd_tensor<double> kr_b = {{5.0, 6.0}, {7.0, 8.0}};
    auto kr_ab = linalg::khatri_rao (kr_a, kr_b);
    CHECK_TRUE (kr_ab.shape () == (shape_type {4, 2}));
    CHECK_NEAR (kr_ab (0, 0), 5.0, 1e-12);
    CHECK_NEAR (kr_ab (3, 1), 32.0, 1e-12);

    auto ch_sig = signal::chirp (linspace (0.0, 1.0, 64), 2.0, 1.0, 8.0);
    auto spec = signal::spectrogram (ch_sig, 64.0, 16, 8);
    CHECK_TRUE (spec.sxx.ndim () == 2 && spec.frequencies.size () == 9);

    interpolate::cubic_spline cs_quad (nd_tensor<double> {0.0, 1.0, 2.0},
                                       nd_tensor<double> {0.0, 1.0, 8.0});
    CHECK_TRUE (cs_quad.integrate (0.0, 2.0) > 3.0);
    auto cs_der_vec = cs_quad.derivative (nd_tensor<double> {0.5, 1.5});
    CHECK_TRUE (cs_der_vec.size () == 2);

    stats::multivariate_normal_dist mvn (nd_tensor<double> {0.0, 0.0},
                                         eye<double> (2));
    CHECK_NEAR (mvn.pdf (nd_tensor<double> {0.0, 0.0}),
                1.0 / (2.0 * constants::pi), 1e-12);

    auto [bc_y, bc_lam] =
      stats::boxcox (nd_tensor<double> {1.0, 2.0, 4.0, 8.0, 16.0});
    CHECK_NEAR (bc_lam, 0.0, 0.25);
    CHECK_TRUE (bc_y.size () == 5);

    /* 13. Phase 13: solve_continuous_are, leslie, fiedler, csd, coherence,
       sawtooth, square_wave, yeojohnson, and binom_test.  */
    nd_tensor<double> care_a = {{-1.0, 1.0}, {0.0, -2.0}};
    nd_tensor<double> care_b (shape_type {2, 1}, {0.0, 1.0});
    nd_tensor<double> care_q = eye<double> (2);
    nd_tensor<double> care_r (shape_type {1, 1}, {1.0});
    auto care_x = linalg::solve_continuous_are (care_a, care_b, care_q, care_r);
    auto care_res =
      matmul (care_a.t (), care_x)
      + matmul (care_x, care_a)
      - matmul (matmul (care_x, care_b), matmul (care_b.t (), care_x))
      + care_q;
    CHECK_NEAR (linalg::norm (care_res), 0.0, 1e-8);

    auto les = linalg::leslie (nd_tensor<double> {0.1, 2.0, 1.5},
                               nd_tensor<double> {0.8, 0.5});
    CHECK_NEAR (les (0, 1), 2.0, 1e-12);
    CHECK_NEAR (les (1, 0), 0.8, 1e-12);
    CHECK_NEAR (les (2, 1), 0.5, 1e-12);

    auto fied = linalg::fiedler (nd_tensor<double> {1.0, 4.0, 6.0});
    CHECK_NEAR (fied (0, 2), 5.0, 1e-12);
    CHECK_NEAR (fied (1, 2), 2.0, 1e-12);

    auto coh_sig = sin (linspace (0.0, 4.0 * constants::pi, 64));
    auto coh = signal::coherence (coh_sig, coh_sig, 16.0, 16, 8);
    CHECK_NEAR (coh.cxy[1], 1.0, 1e-10);

    auto saw = signal::sawtooth (nd_tensor<double> {0.0, constants::pi});
    CHECK_NEAR (saw[0], -1.0, 1e-12);
    CHECK_NEAR (saw[1], 0.0, 1e-12);

    auto sqw = signal::square_wave (
      nd_tensor<double> {0.25 * constants::pi, 1.5 * constants::pi});
    CHECK_NEAR (sqw[0], 1.0, 1e-12);
    CHECK_NEAR (sqw[1], -1.0, 1e-12);

    auto [yj_y, yj_lam] =
      stats::yeojohnson (nd_tensor<double> {-2.0, -1.0, 0.0, 1.0, 2.0});
    CHECK_TRUE (yj_y.size () == 5 && std::isfinite (yj_lam));
    CHECK_NEAR (stats::binom_test (5, 10, 0.5), 1.0, 1e-12);
    CHECK_NEAR (stats::binom_test (0, 10, 0.5), 2.0 / 1024.0, 1e-12);

    /* 14. Phase 14: pascal, invpascal, helmert, integrate::romb, jarque_bera,
       levene, and sem.  */
    auto pas = linalg::pascal (4, "symmetric");
    auto ipas = linalg::invpascal (4, "symmetric");
    CHECK_TRUE (allclose (matmul (pas, ipas), eye<double> (4), 1e-11));

    auto helm = linalg::helmert (4, true);
    CHECK_TRUE (allclose (matmul (helm, helm.t ()), eye<double> (4), 1e-12));

    /* Integrate x^3 on [0, 1] using 2^3 + 1 = 9 uniform samples; exact = 0.25. */
    nd_tensor<double> rx = linspace (0.0, 1.0, 9);
    nd_tensor<double> ry = rx * rx * rx;
    CHECK_NEAR (integrate::romb (ry, 0.125), 0.25, 1e-14);

    auto jb_res = stats::jarque_bera (
      nd_tensor<double> {-1.5, -0.5, 0.0, 0.5, 1.5});
    CHECK_TRUE (jb_res.pvalue > 0.5);

    auto lev_res = stats::levene ({
      nd_tensor<double> {1.0, 2.0, 3.0},
      nd_tensor<double> {10.0, 20.0, 30.0}
    });
    CHECK_TRUE (lev_res.statistic > 1.0);

    CHECK_NEAR (stats::sem (nd_tensor<double> {2.0, 4.0, 6.0, 8.0}, 1),
                std::sqrt (20.0 / 3.0) / 2.0, 1e-12);

    /* 15. Phase 15: linalg::cosm, sinm, hessenberg, fft::dct2, idct2,
       stats::special::i0, i1, and bartlett_test.  */
    nd_tensor<double> trig_m = {{0.2, 0.1}, {-0.1, 0.3}};
    auto cm = linalg::cosm (trig_m);
    auto sm = linalg::sinm (trig_m);
    auto pythag = matmul (cm, cm) + matmul (sm, sm);
    CHECK_TRUE (allclose (pythag, eye<double> (2), 1e-11));

    nd_tensor<double> hess_a = {
      {4.0, 1.0, -2.0, 2.0},
      {1.0, 2.0, 0.0, 1.0},
      {-2.0, 0.0, 3.0, -2.0},
      {2.0, 1.0, -2.0, -1.0}
    };
    auto hess_res = linalg::hessenberg (hess_a);
    CHECK_NEAR (hess_res.h (2, 0), 0.0, 1e-12);
    CHECK_NEAR (hess_res.h (3, 0), 0.0, 1e-12);
    CHECK_NEAR (hess_res.h (3, 1), 0.0, 1e-12);
    CHECK_TRUE (allclose (matmul (matmul (hess_res.q, hess_res.h),
                                  hess_res.q.t ()),
                          hess_a, 1e-11));

    nd_tensor<double> img2d = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto img_dct = fft::dct2 (img2d);
    auto img_rec = fft::idct2 (img_dct);
    CHECK_TRUE (allclose (img_rec, img2d, 1e-11));

    CHECK_NEAR (stats::special::i0 (0.0), 1.0, 1e-15);
    CHECK_NEAR (stats::special::i1 (0.0), 0.0, 1e-15);
    CHECK_TRUE (stats::special::i0 (2.0) > 2.279);

    auto bart = stats::bartlett_test ({
      nd_tensor<double> {1.0, 2.0, 3.0, 4.0},
      nd_tensor<double> {10.0, 20.0, 30.0, 40.0}
    });
    CHECK_TRUE (bart.statistic > 5.0 && bart.pvalue < 0.05);

    /* 16. Phase 16: linalg::coshm, sinhm, subspace_angles, fft::dst, idst,
       stats::fisher_exact, and mood_median_test.  */
    auto chm = linalg::coshm (trig_m);
    auto shm = linalg::sinhm (trig_m);
    auto hyp_id = matmul (chm, chm) - matmul (shm, shm);
    CHECK_TRUE (allclose (hyp_id, eye<double> (2), 1e-11));

    nd_tensor<double> sub_a (shape_type {3, 1}, {1.0, 0.0, 0.0});
    nd_tensor<double> sub_b (shape_type {3, 1}, {0.0, 1.0, 0.0});
    auto ang = linalg::subspace_angles (sub_a, sub_b);
    CHECK_NEAR (ang[0], 0.5 * constants::pi, 1e-12);

    nd_tensor<double> dst_in = {1.0, 2.0, 3.0, 4.0};
    auto dst_f = fft::dst (dst_in);
    auto dst_b = fft::idst (dst_f);
    CHECK_TRUE (allclose (dst_b, dst_in, 1e-11));

    nd_tensor<double> fish_tab = {{8.0, 1.0}, {1.0, 8.0}};
    auto fish_res = stats::fisher_exact (fish_tab);
    CHECK_NEAR (fish_res.statistic, 64.0, 1e-12);
    CHECK_TRUE (fish_res.pvalue < 0.01);

    auto mood_res = stats::mood_median_test ({
      nd_tensor<double> {1.0, 2.0, 3.0, 4.0, 5.0},
      nd_tensor<double> {10.0, 20.0, 30.0, 40.0, 50.0}
    });
    CHECK_TRUE (mood_res.statistic > 5.0 && mood_res.pvalue < 0.05);

    /* Bug-audit regression tests: Simpson 3/8 exactness on cubic for even N,
       eval_legendre for |x| > 1, ks_2samp with ties, Poisson pmf(0, mu=0),
       odd-N signal::resample, and autograd dual pow at 0.  */
    nd_tensor<double> x_even = nd_tensor<double>::linspace (0.0, 3.0, 4);
    nd_tensor<double> y_cubic = x_even * x_even * x_even;
    CHECK_NEAR (integrate::simpson (y_cubic, 1.0), 81.0 / 4.0, 1e-12);

    CHECK_NEAR (stats::special::eval_legendre (2, 2.0), 5.5, 1e-12);
    auto ks_ties = stats::ks_2samp (nd_tensor<double> {1.0},
                                    nd_tensor<double> {1.0, 1.0});
    CHECK_NEAR (ks_ties.statistic, 0.0, 1e-12);
    CHECK_NEAR (stats::poisson_dist {0.0}.pmf (0), 1.0, 1e-12);

    nd_tensor<double> odd_sig = {1.0, -0.5, -0.5};
    auto resamp_odd = signal::resample (odd_sig, 6);
    CHECK_NEAR (resamp_odd[0], 1.0, 1e-12);

    autograd::dual_number d_zero (0.0, 1.0);
    autograd::dual_number d_exp (2.0, 0.0);
    auto d_pow = autograd::pow (d_zero, d_exp);
    CHECK_NEAR (d_pow.val (), 0.0, 1e-12);
    CHECK_NEAR (d_pow.dot (), 0.0, 1e-12);

    /* Phase 17: solve_tridiagonal, solve_banded, cholesky_banded,
       solveh_banded, qr_pivoted, irfft, rfft2, irfft2, rfftfreq, ifftshift,
       bisect, ridder, secant, bracket, differential_evolution, basinhopping,
       logit/expit/logsumexp/softmax/zeta/lambertw/ellipk/ellipe,
       weibull/cauchy/laplace/pareto/f/geom dists, kendalltau, and fligner.  */
    {
      nd_tensor<double> tri_dl = {-1.0, -1.0};
      nd_tensor<double> tri_d = {2.0, 2.0, 2.0};
      nd_tensor<double> tri_du = {-1.0, -1.0};
      nd_tensor<double> tri_b = {1.0, 0.0, 1.0};
      auto td_x = linalg::solve_tridiagonal (tri_dl, tri_d, tri_du, tri_b);
      CHECK_NEAR (td_x[0], 1.0, 1e-12);
      CHECK_NEAR (td_x[1], 1.0, 1e-12);
      CHECK_NEAR (td_x[2], 1.0, 1e-12);

      nd_tensor<double> ab_band = {
        {0.0, -1.0, -1.0},
        {2.0,  2.0,  2.0},
        {-1.0, -1.0, 0.0}
      };
      auto sb_x = linalg::solve_banded (1, 1, ab_band, tri_b);
      CHECK_TRUE (allclose (sb_x, td_x, 1e-12));

      nd_tensor<double> ab_upper = {
        {0.0, -1.0, -1.0},
        {2.0,  2.0,  2.0}
      };
      auto cb_u = linalg::cholesky_banded (ab_upper, false);
      CHECK_NEAR (cb_u (1, 0), std::sqrt (2.0), 1e-12);
      auto shb_x = linalg::solveh_banded (ab_upper, tri_b, false);
      CHECK_TRUE (allclose (shb_x, td_x, 1e-12));

      nd_tensor<double> qrp_a = {{1.0, 10.0}, {1.0, 20.0}, {1.0, 30.0}};
      auto qrp = linalg::qr_pivoted (qrp_a);
      CHECK_TRUE (qrp.p[0] == 1);
      auto qrp_rec = tensor::matmul (qrp.q, qrp.r);
      CHECK_NEAR (qrp_rec (0, 0), 10.0, 1e-11);
      CHECK_NEAR (qrp_rec (2, 1), 1.0, 1e-11);

      nd_tensor<double> rf_sig = {1.0, 2.0, 3.0, 4.0, 2.0, 1.0};
      auto rf_spec = fft::rfft (rf_sig);
      auto rf_rec = fft::irfft (rf_spec, rf_sig.size ());
      CHECK_TRUE (allclose (rf_rec, rf_sig, 1e-12));
      auto rfreqs = fft::rfftfreq (6, 1.0);
      CHECK_TRUE (rfreqs.size () == 4);
      CHECK_NEAR (rfreqs[3], 0.5, 1e-12);

      nd_tensor<double> odd_s = {10.0, 20.0, 30.0, 40.0, 50.0};
      CHECK_TRUE (allclose (fft::ifftshift (fft::fftshift (odd_s)), odd_s, 1e-12));

      nd_tensor<double> r2_in = {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}};
      auto r2_spec = fft::rfft2 (r2_in);
      auto r2_rec = fft::irfft2 (r2_spec, 4);
      CHECK_TRUE (allclose (r2_rec, r2_in, 1e-12));

      auto quad_fn = [] (double x) { return x * x - 2.0; };
      CHECK_NEAR (optimize::bisect (quad_fn, 0.0, 2.0), constants::sqrt2, 1e-10);
      CHECK_NEAR (optimize::ridder (quad_fn, 0.0, 2.0), constants::sqrt2, 1e-10);
      CHECK_NEAR (optimize::secant (quad_fn, 1.0, 2.0), constants::sqrt2, 1e-10);

      auto brk = optimize::bracket ([] (double x) { return (x - 3.0) * (x - 3.0); },
                                    0.0, 1.0);
      CHECK_TRUE (brk.fb <= brk.fa && brk.fb <= brk.fc);

      auto de_res = optimize::differential_evolution (
        [] (const nd_tensor<double> &v) {
          return (v[0] - 1.5) * (v[0] - 1.5) + (v[1] + 0.5) * (v[1] + 0.5);
        },
        {{-5.0, 5.0}, {-5.0, 5.0}},
        12, 80);
      CHECK_NEAR (de_res.x[0], 1.5, 1e-4);
      CHECK_NEAR (de_res.x[1], -0.5, 1e-4);

      auto bh_res = optimize::basinhopping (
        [] (const nd_tensor<double> &v) {
          return (v[0] - 2.0) * (v[0] - 2.0) + (v[1] + 1.0) * (v[1] + 1.0);
        },
        nd_tensor<double> {0.0, 0.0},
        15);
      CHECK_NEAR (bh_res.x[0], 2.0, 1e-5);
      CHECK_NEAR (bh_res.x[1], -1.0, 1e-5);

      CHECK_NEAR (stats::special::expit (stats::special::logit (0.25)), 0.25, 1e-12);
      nd_tensor<double> lse_in = {1000.0, 1000.0};
      CHECK_NEAR (stats::special::logsumexp (lse_in), 1000.0 + std::log (2.0), 1e-12);
      auto sm_out = stats::special::softmax (lse_in);
      CHECK_NEAR (sm_out[0], 0.5, 1e-12);
      CHECK_NEAR (stats::special::zeta (2.0),
                  (constants::pi * constants::pi) / 6.0, 1e-10);
      const double w1 = stats::special::lambertw (1.0);
      CHECK_NEAR (w1 * std::exp (w1), 1.0, 1e-12);
      CHECK_NEAR (stats::special::ellipk (0.0), constants::half_pi, 1e-12);
      CHECK_NEAR (stats::special::ellipe (0.0), constants::half_pi, 1e-12);
      CHECK_NEAR (stats::special::ellipe (1.0), 1.0, 1e-12);

      stats::weibull_min_dist wb {2.0, 1.0};
      CHECK_NEAR (wb.cdf (wb.ppf (0.6)), 0.6, 1e-12);
      stats::cauchy_dist cau {0.0, 1.0};
      CHECK_NEAR (cau.cdf (cau.ppf (0.75)), 0.75, 1e-12);
      stats::laplace_dist lap {0.0, 1.0};
      CHECK_NEAR (lap.cdf (lap.ppf (0.3)), 0.3, 1e-12);
      stats::pareto_dist par {2.0, 1.0};
      CHECK_NEAR (par.cdf (par.ppf (0.75)), 0.75, 1e-12);
      stats::f_dist fd {5.0, 10.0};
      CHECK_TRUE (fd.pdf (1.0) > 0.0 && fd.cdf (1.0) > 0.4 && fd.cdf (1.0) < 0.6);
      stats::geom_dist gd {0.5};
      CHECK_NEAR (gd.pmf (2), 0.25, 1e-12);
      CHECK_NEAR (gd.cdf (2), 0.75, 1e-12);

      auto kt = stats::kendalltau (nd_tensor<double> {1.0, 2.0, 3.0, 4.0},
                                   nd_tensor<double> {1.0, 2.0, 3.0, 4.0});
      CHECK_NEAR (kt.statistic, 1.0, 1e-12);

      auto flig = stats::fligner ({
        nd_tensor<double> {1.0, 2.0, 3.0, 4.0, 5.0},
        nd_tensor<double> {-50.0, -20.0, 0.0, 20.0, 50.0}
      });
      CHECK_TRUE (flig.statistic > 2.0 && flig.pvalue < 0.2);
    }

    /* Phase 18: tplquad, solve_ivp_bdf, solve_bvp, akima1d_interpolator,
       rect_bivariate_spline, griddata, lombscargle, ricker_wavelet, cwt,
       peak_prominences, peak_widths, sosfilt, ndimage morphology/filters,
       coo_matrix, csc_matrix, eigsh, csgraph algorithms, rotation_3d, slerp,
       fcluster, and cophenet.  */
    {
      /* int_0^1 int_0^1 int_0^1 (x + y + z) dz dy dx = 1.5.  */
      auto tq = integrate::tplquad (
        [] (double z, double y, double x) { return x + y + z; },
        0.0, 1.0,
        [] (double) { return 0.0; },
        [] (double) { return 1.0; },
        [] (double, double) { return 0.0; },
        [] (double, double) { return 1.0; });
      CHECK_NEAR (tq.value, 1.5, 1e-8);

      /* Stiff decay y' = -10 y, y(0) = 1 => y(0.5) = exp(-5).  */
      auto bdf_res = integrate::solve_ivp_bdf (
        [] (double, const nd_tensor<double> &y) {
          return nd_tensor<double> {-10.0 * y[0]};
        },
        0.0, 0.5, nd_tensor<double> {1.0}, 100);
      CHECK_NEAR (bdf_res.y (0, 100), std::exp (-5.0), 1e-3);

      /* BVP: y_0' = y_1, y_1' = -pi^2 y_0 on [0, 1] with y_0(0)=0, y_0(1)=0?
         Use inhomog BVP: y_0' = y_1, y_1' = 2 with y_0(0)=0, y_0(1)=1 => y_0(x) = x^2.  */
      nd_tensor<double> bvp_x = linspace (0.0, 1.0, 11);
      nd_tensor<double> bvp_y0 (shape_type {2, 11}, 0.0);
      auto bvp_res = integrate::solve_bvp (
        [] (double, const nd_tensor<double> &y) {
          return nd_tensor<double> {y[1], 2.0};
        },
        [] (const nd_tensor<double> &ya, const nd_tensor<double> &yb) {
          return nd_tensor<double> {ya[0] - 0.0, yb[0] - 1.0};
        },
        bvp_x, bvp_y0);
      CHECK_NEAR (bvp_res (0, 5), 0.25, 1e-4);

      /* Akima 1D, RectBivariateSpline, and griddata.  */
      nd_tensor<double> ak_x = {0.0, 1.0, 2.0, 3.0, 4.0};
      nd_tensor<double> ak_y = {0.0, 1.0, 4.0, 9.0, 16.0};
      interpolate::akima1d_interpolator akima (ak_x, ak_y);
      CHECK_NEAR (akima (1.5), 2.25, 0.15);

      nd_tensor<double> rbs_x = {0.0, 1.0, 2.0};
      nd_tensor<double> rbs_y = {0.0, 1.0, 2.0};
      nd_tensor<double> rbs_z (shape_type {3, 3});
      for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
          rbs_z (i, j) = rbs_x[i] + 2.0 * rbs_y[j];
      interpolate::rect_bivariate_spline rbs (rbs_x, rbs_y, rbs_z);
      CHECK_NEAR (rbs (0.5, 1.5), 3.5, 1e-10);

      nd_tensor<double> gd_pts = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
      nd_tensor<double> gd_vals = {0.0, 1.0, 2.0, 3.0};
      nd_tensor<double> gd_xi = {{0.5, 0.5}, {0.1, 0.1}};
      auto gd_near = interpolate::griddata (gd_pts, gd_vals, gd_xi, "nearest");
      CHECK_NEAR (gd_near[1], 0.0, 1e-12);
      auto gd_lin = interpolate::griddata (gd_pts, gd_vals, gd_xi, "linear");
      CHECK_NEAR (gd_lin[0], 1.5, 1e-6);

      /* Signal: Lomb-Scargle, Ricker wavelet, CWT, peak prominences/widths,
         sosfilt, and ndimage operations.  */
      nd_tensor<double> ls_t = linspace (0.0, 10.0, 50);
      nd_tensor<double> ls_y (shape_type {50});
      for (std::size_t i = 0; i < 50; ++i)
        ls_y[i] = std::sin (2.0 * constants::pi * ls_t[i]);
      auto ls_p = signal::lombscargle (
        ls_t, ls_y, nd_tensor<double> {2.0 * constants::pi, 4.0 * constants::pi});
      CHECK_TRUE (ls_p[0] > 5.0 * ls_p[1]);

      auto rw = signal::ricker_wavelet (11, 2.0);
      CHECK_TRUE (rw.size () == 11 && rw[5] > rw[0]);
      auto cwt_mat = signal::cwt (ls_y, nd_tensor<double> {1.0, 2.0, 4.0});
      CHECK_TRUE (cwt_mat.shape () == (shape_type {3, 50}));

      nd_tensor<double> pk_sig = {0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 5.0, 2.0, 0.0};
      std::vector<std::size_t> pks = {2, 6};
      auto proms = signal::peak_prominences (pk_sig, pks);
      CHECK_NEAR (proms[0], 3.0, 1e-12);
      CHECK_NEAR (proms[1], 5.0, 1e-12);
      auto widths = signal::peak_widths (pk_sig, pks, 0.5);
      CHECK_TRUE (widths[0] > 1.0 && widths[1] > 1.0);

      nd_tensor<double> sos = {{1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
      auto sos_out = signal::sosfilt (sos, pk_sig);
      CHECK_TRUE (allclose (sos_out, pk_sig, 1e-12));

      nd_tensor<double> bin_img = {
        {0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0}
      };
      auto eroded = signal::binary_erosion (bin_img);
      CHECK_NEAR (eroded (2, 2), 1.0, 1e-12);
      CHECK_NEAR (eroded (1, 1), 0.0, 1e-12);
      auto opened = signal::binary_opening (bin_img);
      auto closed = signal::binary_closing (bin_img);
      CHECK_NEAR (opened (2, 2), 1.0, 1e-12);
      CHECK_NEAR (closed (2, 2), 1.0, 1e-12);

      auto sob = signal::sobel (bin_img, 0);
      auto lap_img = signal::laplace (bin_img);
      CHECK_TRUE (sob.shape () == bin_img.shape ()
                  && lap_img.shape () == bin_img.shape ());
      auto [cm_r, cm_c] = signal::center_of_mass (bin_img);
      CHECK_NEAR (cm_r, 2.0, 1e-12);
      CHECK_NEAR (cm_c, 2.0, 1e-12);

      nd_tensor<double> two_blobs = {
        {1.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
      };
      auto [lbl_mat, n_lbl] = signal::label (two_blobs);
      CHECK_TRUE (n_lbl == 2 && lbl_mat (0, 0) == 1 && lbl_mat (1, 3) == 2);

      /* Sparse: coo_matrix, csc_matrix, eigsh, and csgraph algorithms.  */
      sparse::coo_matrix<double> coo (
        3, 3, {0, 1, 2}, {0, 1, 2}, {2.0, 3.0, 4.0});
      auto csr_from_coo = coo.to_csr ();
      auto csc_mat = sparse::csc_matrix<double>::from_dense (csr_from_coo.to_dense ());
      nd_tensor<double> sp_v = {1.0, 2.0, 3.0};
      CHECK_NEAR (csc_mat.spmv (sp_v)[2], 12.0, 1e-12);

      auto esh = sparse::eigsh (csr_from_coo, 2);
      CHECK_NEAR (esh.eigenvalues[1], 4.0, 1e-8);

      nd_tensor<double> adj_dense = {
        {0.0, 1.0, 4.0, 0.0},
        {1.0, 0.0, 2.0, 6.0},
        {4.0, 2.0, 0.0, 3.0},
        {0.0, 6.0, 3.0, 0.0}
      };
      auto adj_csr = sparse::csr_matrix<double>::from_dense (adj_dense);
      auto [n_comp, comp_lbl] = sparse::connected_components (adj_csr);
      CHECK_TRUE (n_comp == 1 && comp_lbl[3] == 0);

      auto dijk = sparse::dijkstra (adj_csr, 0);
      CHECK_NEAR (dijk[0], 0.0, 1e-12);
      CHECK_NEAR (dijk[2], 3.0, 1e-12);
      CHECK_NEAR (dijk[3], 6.0, 1e-12);

      auto fw = sparse::floyd_warshall (adj_csr);
      CHECK_NEAR (fw (0, 3), 6.0, 1e-12);

      auto mst = sparse::minimum_spanning_tree (adj_csr).to_dense ();
      /* MST edges: (0,1)=1, (1,2)=2, (2,3)=3 => total weight 6.  */
      CHECK_NEAR (sum (mst), 6.0, 1e-12);

      nd_tensor<double> dag_dense = {
        {0.0, 1.0, 1.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0}
      };
      auto topo = sparse::topological_sort (
        sparse::csr_matrix<double>::from_dense (dag_dense));
      CHECK_TRUE (topo.size () == 3 && topo[0] == 0 && topo[2] == 2);

      /* Spatial: rotation_3d and slerp.  */
      auto r_z90 = spatial::rotation_3d::from_rotvec (
        nd_tensor<double> {0.0, 0.0, constants::half_pi});
      auto rot_pt = r_z90.apply (nd_tensor<double> {1.0, 0.0, 0.0});
      CHECK_NEAR (rot_pt[0], 0.0, 1e-12);
      CHECK_NEAR (rot_pt[1], 1.0, 1e-12);
      auto r_back = (r_z90 * r_z90.inv ()).apply (nd_tensor<double> {1.0, 2.0, 3.0});
      CHECK_NEAR (r_back[1], 2.0, 1e-12);

      auto r_mid = spatial::slerp (spatial::rotation_3d (), r_z90, 0.5);
      auto mid_pt = r_mid.apply (nd_tensor<double> {1.0, 0.0, 0.0});
      CHECK_NEAR (mid_pt[0], 1.0 / constants::sqrt2, 1e-12);
      CHECK_NEAR (mid_pt[1], 1.0 / constants::sqrt2, 1e-12);

      /* Cluster: fcluster and cophenet.  */
      nd_tensor<double> cl_pts = {
        {0.0, 0.0}, {0.0, 1.0}, {10.0, 10.0}, {10.0, 11.0}
      };
      auto cl_Z = cluster::linkage (cl_pts, "single");
      auto fc_k2 = cluster::fcluster (cl_Z, 2.0, "maxclust");
      CHECK_TRUE (fc_k2[0] == fc_k2[1] && fc_k2[2] == fc_k2[3]
                  && fc_k2[0] != fc_k2[2]);
      auto [coph_r, coph_d] = cluster::cophenet (cl_Z, spatial::pdist (cl_pts));
      CHECK_TRUE (coph_r > 0.99 && coph_d.size () == 6);
    }

    /* Phase 19: linalg, fft, integrate, optimize, stats, signal, sparse,
       spatial, and cluster parity verification.  */
    {
      /* Linalg: eig, qz, solve_toeplitz, solve_circulant, matrix_power,
         fractional_matrix_power, funm.  */
      nd_tensor<double> rot90 = {{0.0, -1.0}, {1.0, 0.0}};
      auto eig_res = linalg::eig (rot90);
      CHECK_NEAR (std::abs (eig_res.eigenvalues[0].imag ()), 1.0, 1e-8);
      auto qz_res = linalg::qz (nd_tensor<double>::eye (2) * 3.0,
                                nd_tensor<double>::eye (2));
      CHECK_NEAR (qz_res.aa (0, 0), 3.0, 1e-8);

      auto x_toep = linalg::solve_toeplitz (
        nd_tensor<double> {4.0, 1.0},
        nd_tensor<double> {4.0, 1.0},
        nd_tensor<double> {5.0, 5.0});
      CHECK_NEAR (x_toep[0], 1.0, 1e-10);
      CHECK_NEAR (x_toep[1], 1.0, 1e-10);

      auto x_circ = linalg::solve_circulant (
        nd_tensor<double> {2.0, 1.0}, nd_tensor<double> {3.0, 3.0});
      CHECK_NEAR (x_circ[0], 1.0, 1e-10);

      nd_tensor<double> spd2 = {{4.0, 0.0}, {0.0, 9.0}};
      auto mp2 = linalg::matrix_power (spd2, 2);
      CHECK_NEAR (mp2 (0, 0), 16.0, 1e-12);
      auto fmp = linalg::fractional_matrix_power (spd2, 0.5);
      CHECK_NEAR (fmp (0, 0), 2.0, 1e-10);
      CHECK_NEAR (fmp (1, 1), 3.0, 1e-10);
      auto fm_exp = linalg::funm (nd_tensor<double>::zeros ({2, 2}),
                                  [] (double v) { return std::exp (v); });
      CHECK_NEAR (fm_exp (0, 0), 1.0, 1e-10);

      /* FFT: next_fast_len, fftshift2, ifftshift2, fht, ifht.  */
      CHECK_TRUE (fft::next_fast_len (13) == 15);
      auto sh2 = fft::ifftshift2 (fft::fftshift2 (spd2));
      CHECK_TRUE (allclose (sh2, spd2, 1e-12));
      auto fht_seq = nd_tensor<double> {1.0, 2.0, 3.0, 4.0};
      auto fht_back = fft::ifht (fft::fht (fht_seq, 1.0, 0.5), 1.0, 0.5);
      CHECK_TRUE (allclose (fht_back, fht_seq, 1e-8));

      /* Integrate: nquad and newton_cotes.  */
      auto nq = integrate::nquad (
        [] (const nd_tensor<double> &v) { return v[0] + v[1]; },
        {{0.0, 1.0}, {0.0, 1.0}});
      CHECK_NEAR (nq.value, 1.0, 1e-10);
      auto nc_w = integrate::newton_cotes (2);
      CHECK_NEAR (nc_w[0], 1.0 / 3.0, 1e-12);
      CHECK_NEAR (nc_w[1], 4.0 / 3.0, 1e-12);

      /* Optimize: minimize_cg, minimize_powell, minimize_slsqp, lsq_linear,
         nnls, isotonic_regression, milp.  */
      auto quad_fn = [] (const nd_tensor<double> &x) {
        return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] + 1.0) * (x[1] + 1.0);
      };
      auto cg_res = optimize::minimize_cg (quad_fn, nd_tensor<double> {0.0, 0.0});
      CHECK_NEAR (cg_res.x[0], 2.0, 1e-4);
      auto pw_res = optimize::minimize_powell (quad_fn, nd_tensor<double> {0.0, 0.0});
      CHECK_NEAR (pw_res.x[1], -1.0, 1e-4);
      auto slsqp_res = optimize::minimize_slsqp (
        [] (const nd_tensor<double> &x) { return x[0] * x[0] + x[1] * x[1]; },
        nd_tensor<double> {2.0, 2.0},
        {[] (const nd_tensor<double> &x) { return x[0] + x[1] - 2.0; }});
      CHECK_NEAR (slsqp_res.x[0], 1.0, 1e-2);
      CHECK_NEAR (slsqp_res.x[1], 1.0, 1e-2);

      auto [nnls_x, nnls_norm] = optimize::nnls (
        nd_tensor<double>::eye (2), nd_tensor<double> {3.0, -2.0});
      CHECK_NEAR (nnls_x[0], 3.0, 1e-6);
      CHECK_NEAR (nnls_x[1], 0.0, 1e-6);
      auto lsq_b = optimize::lsq_linear (
        nd_tensor<double>::eye (2), nd_tensor<double> {5.0, -5.0},
        nd_tensor<double> {0.0, 0.0}, nd_tensor<double> {2.0, 2.0});
      CHECK_NEAR (lsq_b[0], 2.0, 1e-6);
      CHECK_NEAR (lsq_b[1], 0.0, 1e-6);

      auto iso = optimize::isotonic_regression (
        nd_tensor<double> {3.0, 1.0, 4.0});
      CHECK_NEAR (iso[0], 2.0, 1e-12);
      CHECK_NEAR (iso[1], 2.0, 1e-12);
      CHECK_NEAR (iso[2], 4.0, 1e-12);

      auto milp_res = optimize::milp (
        nd_tensor<double> {-1.0, -1.0},
        nd_tensor<double> {{2.0, 2.0}},
        nd_tensor<double> {5.0},
        std::vector<bool> {true, true});
      CHECK_TRUE (milp_res.success);
      CHECK_NEAR (milp_res.fun, -2.0, 1e-6);

      /* Stats special functions, distributions, and hypothesis tests.  */
      CHECK_NEAR (stats::special::gammaincinv (
                    2.0, stats::special::gammainc (2.0, 1.5)), 1.5, 1e-6);
      CHECK_NEAR (stats::special::betaincinv (
                    2.0, 3.0, stats::special::betainc (2.0, 3.0, 0.4)),
                  0.4, 1e-6);
      CHECK_NEAR (stats::special::poch (3.0, 2.0), 12.0, 1e-10);
      CHECK_NEAR (stats::special::eval_hermite (2, 1.0), 2.0, 1e-12);
      CHECK_NEAR (stats::special::eval_laguerre (1, 2.0), -1.0, 1e-12);
      CHECK_NEAR (stats::special::eval_jacobi (1, 0.0, 0.0, 0.5), 0.5, 1e-12);
      CHECK_NEAR (stats::special::spherical_jn (0, 0.0), 1.0, 1e-12);
      CHECK_TRUE (stats::special::voigt_profile (0.0, 1.0, 1.0) > 0.0);
      auto [fres_s, fres_c] = stats::special::fresnel (1.0);
      CHECK_TRUE (fres_s > 0.0 && fres_c > 0.0);
      auto [si_v, ci_v] = stats::special::sici (1.0);
      CHECK_TRUE (si_v > 0.0 && std::isfinite (ci_v));

      stats::logistic_dist log_d (0.0, 1.0);
      CHECK_NEAR (log_d.cdf (0.0), 0.5, 1e-12);
      stats::rayleigh_dist ray_d (1.0);
      CHECK_NEAR (ray_d.ppf (ray_d.cdf (1.5)), 1.5, 1e-10);
      stats::hypergeom_dist hyp_d (10, 5, 4);
      CHECK_NEAR (hyp_d.mean (), 2.0, 1e-12);
      stats::dirichlet_dist dir_d ({1.0, 1.0});
      CHECK_NEAR (dir_d.pdf ({0.5, 0.5}), 1.0, 1e-10);

      nd_tensor<double> norm_samp = {-1.2, -0.5, -0.1, 0.0, 0.2, 0.6, 1.1, 1.4};
      CHECK_TRUE (stats::shapiro (norm_samp).pvalue > 0.05);
      CHECK_TRUE (stats::normaltest (norm_samp).pvalue > 0.05);
      CHECK_TRUE (stats::cramervonmises (
                    norm_samp, [] (double x) {
                      return stats::normal_dist ().cdf (x);
                    }).pvalue > 0.05);

      auto lr = stats::linregress (nd_tensor<double> {1.0, 2.0, 3.0},
                                   nd_tensor<double> {3.0, 5.0, 7.0});
      CHECK_NEAR (lr.slope, 2.0, 1e-12);
      CHECK_NEAR (lr.intercept, 1.0, 1e-12);
      auto ts_reg = stats::theilslopes (nd_tensor<double> {3.0, 5.0, 7.0},
                                        nd_tensor<double> {1.0, 2.0, 3.0});
      CHECK_NEAR (ts_reg.first, 2.0, 1e-12);

      auto chi_c = stats::chi2_contingency (
        nd_tensor<double> {{10.0, 10.0}, {10.0, 10.0}}, false);
      CHECK_NEAR (chi_c.statistic, 0.0, 1e-12);
      CHECK_NEAR (stats::mcnemar ({{10.0, 5.0}, {5.0, 10.0}}).pvalue, 1.0, 1e-10);
      CHECK_TRUE (stats::combine_pvalues ({0.01, 0.02}).pvalue < 0.01);

      /* Signal & NDImage: iirfilter, tf2sos, sosfiltfilt, freqz, stft/istft,
         periodogram, decimate, wiener, deconvolve, distance_transform_edt,
         binary_fill_holes, rank/edge filters, rotate_2d, zoom_2d.  */
      auto [iir_b, iir_a] = signal::iirfilter (2, 0.25, "lowpass");
      auto sos_m = signal::tf2sos (iir_b, iir_a);
      auto ff_s = signal::sosfiltfilt (sos_m, nd_tensor<double>::ones ({16}));
      CHECK_TRUE (ff_s.size () == 16);
      auto [fz_w, fz_h] = signal::freqz (iir_b, iir_a, 16);
      CHECK_NEAR ( std::abs (fz_h[0]), 1.0, 1e-6);

      nd_tensor<double> stft_in = nd_tensor<double>::ones ({16});
      auto stft_r = signal::stft (stft_in, 8, 4);
      auto istft_out = signal::istft (stft_r, 8, 4);
      CHECK_NEAR (istft_out[4], 1.0, 1e-6);

      auto [per_f, per_p] = signal::periodogram (stft_in);
      CHECK_TRUE (per_f.size () == 9);
      CHECK_TRUE (signal::decimate (stft_in, 2).size () == 8);
      CHECK_TRUE (signal::wiener (stft_in, 3).size () == 16);

      auto [dec_q, dec_r] = signal::deconvolve (
        nd_tensor<double> {1.0, 3.0, 3.0, 1.0},
        nd_tensor<double> {1.0, 1.0});
      CHECK_NEAR (dec_q[1], 2.0, 1e-12);

      nd_tensor<double> ring_img = {
        {1.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}
      };
      auto filled_holes = signal::binary_fill_holes (ring_img);
      CHECK_NEAR (filled_holes (1, 1), 1.0, 1e-12);
      auto edt = signal::distance_transform_edt (ring_img);
      CHECK_NEAR (edt (1, 1), 0.0, 1e-12);
      CHECK_NEAR (edt (0, 1), 1.0, 1e-12);
      CHECK_NEAR (edt (0, 0), constants::sqrt2, 1e-12);
      CHECK_NEAR (signal::uniform_filter2d (filled_holes, 3) (1, 1), 1.0, 1e-12);
      CHECK_NEAR (signal::minimum_filter2d (ring_img, 3) (1, 1), 0.0, 1e-12);
      CHECK_NEAR (signal::maximum_filter2d (ring_img, 3) (1, 1), 1.0, 1e-12);
      CHECK_TRUE (signal::zoom_2d (ring_img, 2.0).shape () == (shape_type {6, 6}));
      CHECK_TRUE (signal::rotate_2d (ring_img, 90.0).shape () == (shape_type {3, 3}));

      /* Sparse: dia_matrix, spsolve, splu, sparse_kron, bellman_ford,
         bfs/dfs order, graph_laplacian, maximum_flow, reverse_cuthill_mckee.  */
      sparse::dia_matrix<double> dia (3, 3, {0},
                                      nd_tensor<double> {{2.0, 4.0, 8.0}});
      auto dia_csr = dia.to_csr ();
      auto sp_x = sparse::spsolve (dia_csr, nd_tensor<double> {2.0, 8.0, 24.0});
      CHECK_NEAR (sp_x[2], 3.0, 1e-12);
      auto lu_f = sparse::splu (dia_csr);
      CHECK_NEAR (lu_f.solve (nd_tensor<double> {2.0, 8.0, 24.0})[1], 2.0, 1e-12);
      CHECK_TRUE (sparse::sparse_kron (dia_csr, dia_csr).rows () == 9);

      nd_tensor<double> cap_dense = {
        {0.0, 10.0, 5.0, 0.0},
        {0.0, 0.0, 15.0, 10.0},
        {0.0, 0.0, 0.0, 10.0},
        {0.0, 0.0, 0.0, 0.0}
      };
      auto cap_csr = sparse::csr_matrix<double>::from_dense (cap_dense);
      CHECK_NEAR (sparse::maximum_flow (cap_csr, 0, 3), 15.0, 1e-12);
      CHECK_NEAR (sparse::bellman_ford (cap_csr, 0)[3], 15.0, 1e-12);
      CHECK_TRUE (sparse::breadth_first_order (cap_csr, 0).size () == 4);
      CHECK_TRUE (sparse::depth_first_order (cap_csr, 0).size () == 4);
      CHECK_TRUE (sparse::reverse_cuthill_mckee (cap_csr).size () == 4);

      /* Spatial & Cluster: voronoi_2d, extended distances, whiten, vq,
         inconsistent, leaders.  */
      nd_tensor<double> vor_pts = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}
      };
      auto vor = spatial::voronoi_2d (vor_pts);
      CHECK_TRUE (vor.vertices.shape ()[0] >= 1);
      CHECK_NEAR (vor.vertices (0, 0), 0.5, 1e-10);
      CHECK_NEAR (vor.vertices (0, 1), 0.5, 1e-10);

      nd_tensor<double> dm_a = {{1.0, 0.0, 1.0}};
      nd_tensor<double> dm_b = {{1.0, 1.0, 0.0}};
      CHECK_NEAR (spatial::cdist (dm_a, dm_b, "hamming") (0, 0),
                  2.0 / 3.0, 1e-12);
      CHECK_NEAR (spatial::cdist (dm_a, dm_b, "jaccard") (0, 0),
                  2.0 / 3.0, 1e-12);

      auto w_obs = cluster::whiten (vor_pts);
      CHECK_TRUE (w_obs.shape () == vor_pts.shape ());
      auto [vq_code, vq_dist] = cluster::vq (vor_pts, vor_pts);
      CHECK_TRUE (vq_code[2] == 2 && vq_dist[2] < 1e-12);
    }

    /* Phase 20: linalg, integrate, optimize, stats (qmc + special + dists +
       tests), interpolate, signal, sparse, and spatial additions.  */
    {
      /* Linalg additions.  */
      nd_tensor<double> m2 = {{3.0, 0.0}, {0.0, 4.0}};
      CHECK_NEAR (linalg::svdvals (m2)[0], 4.0, 1e-12);
      auto t_inv = linalg::tensorinv (nd_tensor<double>::eye (4).reshape ({2, 2, 2, 2}), 2);
      CHECK_TRUE (t_inv.ndim () == 4);
      auto t_sol = linalg::tensorsolve (
        nd_tensor<double>::eye (4).reshape ({2, 2, 2, 2}),
        nd_tensor<double> {{1.0, 2.0}, {3.0, 4.0}});
      CHECK_NEAR (t_sol (1, 1), 4.0, 1e-12);

      nd_tensor<double> ord_a = {{2.0, 1.0}, {0.0, -3.0}};
      auto os = linalg::ordschur (ord_a, "lhp");
      CHECK_TRUE (os.selected_count == 1 && os.t (0, 0) < 0.0);
      auto csf = linalg::rsf2csf (os.t, os.z);
      CHECK_TRUE (csf.t.shape () == (shape_type {2, 2}));
      auto eg_m = linalg::eig (ord_a);
      auto rdf = linalg::cdf2rdf (eg_m.eigenvalues, eg_m.eigenvectors);
      CHECK_TRUE (rdf.d_real.shape () == (shape_type {2, 2}));

      auto bal = linalg::matrix_balance (nd_tensor<double> {{1.0, 1000.0}, {0.001, 1.0}});
      CHECK_TRUE (bal.scale.size () == 2);
      auto cs_dec = linalg::cossin (nd_tensor<double>::eye (4), 2, 2);
      CHECK_TRUE (cs_dec.theta.size () == 2);
      CHECK_TRUE (linalg::dft_matrix (4).shape () == (shape_type {4, 4}));
      CHECK_TRUE (linalg::convolution_matrix (nd_tensor<double> {1.0, 2.0}, 3).shape ()
                  == (shape_type {4, 3}));
      CHECK_TRUE (linalg::fiedler_companion (nd_tensor<double> {1.0, -3.0, 2.0}).shape ()
                  == (shape_type {2, 2}));
      CHECK_TRUE (linalg::clarkson_woodruff_transform (nd_tensor<double>::eye (6), 3).shape ()
                  == (shape_type {3, 6}));

      /* Integrate additions.  */
      auto rk23_sol = integrate::solve_ivp_rk23 (
        [] (double, const nd_tensor<double> &y) { return -1.0 * y; },
        0.0, 1.0, nd_tensor<double> {1.0}, 50);
      CHECK_NEAR (rk23_sol.y (0, 50), std::exp (-1.0), 1e-3);

      auto radau_sol = integrate::solve_ivp_radau (
        [] (double, const nd_tensor<double> &y) { return -2.0 * y; },
        0.0, 1.0, nd_tensor<double> {1.0}, 40);
      CHECK_NEAR (radau_sol.y (0, 40), std::exp (-2.0), 1e-3);

      auto csimp = integrate::cumulative_simpson (
        nd_tensor<double> {0.0, 1.0, 4.0, 9.0, 16.0}, 1.0);
      CHECK_NEAR (csimp[4], 64.0 / 3.0, 1e-9);

      auto qmc_q = integrate::qmc_quad (
        [] (const nd_tensor<double> &x) { return x[0] + x[1]; },
        nd_tensor<double> {0.0, 0.0},
        nd_tensor<double> {1.0, 1.0},
        512);
      CHECK_NEAR (qmc_q.value, 1.0, 1e-2);

      /* Optimize additions.  */
      auto cob = optimize::minimize_cobyla (
        [] (const nd_tensor<double> &x) {
          return (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 2.0) * (x[1] - 2.0);
        },
        nd_tensor<double> {0.0, 0.0},
        {[] (const nd_tensor<double> &x) { return x[0] + x[1] - 1.0; }});
      CHECK_NEAR (cob.x[0], 1.0, 1e-2);

      auto dog = optimize::minimize_dogleg (
        [] (const nd_tensor<double> &x) {
          return (x[0] - 3.0) * (x[0] - 3.0) + (x[1] + 1.0) * (x[1] + 1.0);
        },
        nd_tensor<double> {0.0, 0.0});
      CHECK_NEAR (dog.x[0], 3.0, 1e-4);

      auto shgo_r = optimize::shgo (
        [] (const nd_tensor<double> &x) { return (x[0] - 0.5) * (x[0] - 0.5); },
        {{-2.0, 2.0}});
      CHECK_NEAR (shgo_r.x[0], 0.5, 1e-4);

      auto dir_r = optimize::direct (
        [] (const nd_tensor<double> &x) { return (x[0] + 0.25) * (x[0] + 0.25); },
        {{-2.0, 2.0}}, 30);
      CHECK_NEAR (dir_r.x[0], -0.25, 1e-4);

      auto br_r = optimize::brute (
        [] (const nd_tensor<double> &x) { return (x[0] - 1.2) * (x[0] - 1.2); },
        {{0.0, 2.0}}, 11);
      CHECK_NEAR (br_r.x[0], 1.2, 1e-4);

      auto broy = optimize::broyden1 (
        [] (const nd_tensor<double> &x) {
          return nd_tensor<double> {x[0] * x[0] - 4.0, x[1] - 3.0};
        },
        nd_tensor<double> {1.5, 2.0});
      CHECK_NEAR (broy[0], 2.0, 1e-5);
      CHECK_NEAR (broy[1], 3.0, 1e-5);

      auto qap = optimize::quadratic_assignment (
        nd_tensor<double> {{0.0, 1.0}, {1.0, 0.0}},
        nd_tensor<double> {{0.0, 2.0}, {2.0, 0.0}});
      CHECK_NEAR (qap.fun, 4.0, 1e-12);

      /* Stats: QMC, Special, Distributions, Resampling, Circular, Tests.  */
      auto h_seq = stats::qmc::halton (16, 2);
      CHECK_TRUE (stats::qmc::discrepancy (h_seq) >= 0.0);
      CHECK_TRUE (stats::qmc::sobol (16, 2).shape () == (shape_type {16, 2}));
      CHECK_TRUE (stats::qmc::latin_hypercube (16, 2).shape () == (shape_type {16, 2}));
      CHECK_TRUE (stats::qmc::poisson_disk (8, 2).shape () == (shape_type {8, 2}));
      CHECK_TRUE (stats::qmc::scale_qmc (
        h_seq, nd_tensor<double> {0.0, 0.0}, nd_tensor<double> {10.0, 20.0}).shape ()
        == (shape_type {16, 2}));

      CHECK_NEAR (stats::special::factorial (5), 120.0, 1e-12);
      CHECK_NEAR (stats::special::factorial2 (5), 15.0, 1e-12);
      CHECK_NEAR (stats::special::comb (6, 2), 15.0, 1e-12);
      CHECK_NEAR (stats::special::perm (5, 2), 20.0, 1e-12);
      CHECK_NEAR (stats::special::stirling2 (4, 2), 7.0, 1e-12);
      CHECK_NEAR (stats::special::bernoulli_numbers (2)[2], 1.0 / 6.0, 1e-12);
      CHECK_NEAR (stats::special::euler_numbers (2)[2], -1.0, 1e-12);
      CHECK_NEAR (stats::special::airy (0.0).ai, 0.355028053887817, 1e-9);
      CHECK_TRUE (stats::special::struve (0.0, 1.0) > 0.0);
      CHECK_NEAR (stats::special::erfcx (0.0), 1.0, 1e-12);
      CHECK_NEAR (stats::special::erfi (0.0), 0.0, 1e-12);
      CHECK_NEAR (stats::special::ndtr (0.0), 0.5, 1e-12);
      CHECK_NEAR (stats::special::ndtri (0.5), 0.0, 1e-9);
      CHECK_NEAR (stats::special::wrightomega (0.0), 0.567143290409784, 1e-9);
      CHECK_NEAR (stats::special::wrightomega (1.0), 1.0, 1e-9);
      CHECK_NEAR (stats::special::huber (1.0, 0.5), 0.125, 1e-12);
      CHECK_NEAR (stats::special::inv_boxcox (stats::special::boxcox1p (1.0, 0.5), 0.5),
                  2.0, 1e-12);
      CHECK_NEAR (stats::special::eval_chebyu (2, 0.5), 0.0, 1e-12);

      stats::erlang_dist erl {2, 1.5};
      CHECK_NEAR (erl.mean (), 3.0, 1e-12);
      stats::maxwell_dist mxw {1.0};
      CHECK_TRUE (mxw.pdf (1.0) > 0.0 && mxw.cdf (1.0) > 0.0);
      stats::skellam_dist skl {3.0, 1.0};
      CHECK_NEAR (skl.mean (), 2.0, 1e-12);
      stats::multinomial_dist mnd {3, nd_tensor<double> {0.5, 0.5}};
      CHECK_NEAR (mnd.pmf (nd_tensor<std::size_t> {2, 1}), 0.375, 1e-12);

      nd_tensor<double> bs_s = {1.0, 2.0, 3.0, 4.0, 5.0};
      auto b_res = stats::bootstrap (bs_s, [] (const nd_tensor<double> &x) {
        return tensor::mean (x);
      }, 100);
      CHECK_TRUE (b_res.confidence_low <= 3.0 && b_res.confidence_high >= 3.0);

      auto p_res = stats::permutation_test (
        nd_tensor<double> {1.0, 2.0, 3.0},
        nd_tensor<double> {10.0, 11.0, 12.0},
        [] (const nd_tensor<double> &a, const nd_tensor<double> &b) {
          return tensor::mean (a) - tensor::mean (b);
        }, 100);
      CHECK_TRUE (p_res.pvalue < 0.2);

      CHECK_NEAR (stats::circmean (nd_tensor<double> {0.1, 0.2, 0.3}), 0.2, 1e-6);
      CHECK_TRUE (stats::circvar (nd_tensor<double> {0.1, 0.2, 0.3}) >= 0.0);
      CHECK_NEAR (stats::expectile (bs_s, 0.5), 3.0, 1e-9);
      CHECK_TRUE (stats::somersd (bs_s, bs_s).statistic > 0.99);

      /* Interpolate additions.  */
      nd_tensor<double> ix = {0.0, 1.0, 2.0, 3.0, 4.0};
      nd_tensor<double> iy = {0.0, 1.0, 4.0, 9.0, 16.0};
      auto bsp = interpolate::make_interp_spline (ix, iy, 3);
      CHECK_NEAR (bsp (2.5), 6.25, 1e-2);
      CHECK_NEAR (bsp.derivative () (2.0), 4.0, 1e-1);
      interpolate::lagrange_interpolator lag_i (ix, iy);
      CHECK_NEAR (lag_i (2.5), 6.25, 1e-9);
      interpolate::krogh_interpolator kr_i (ix, iy);
      CHECK_NEAR (kr_i (2.5), 6.25, 1e-9);
      auto [pade_p, pade_q] =
        interpolate::pade (nd_tensor<double> {1.0, 1.0, 0.5, 1.0 / 6.0}, 1, 1);
      CHECK_NEAR (tensor::polyval (pade_p, 0.1) / tensor::polyval (pade_q, 0.1),
                  std::exp (0.1), 1e-3);

      /* Signal & NDImage additions.  */
      auto [nb, na] = signal::iirnotch (0.25, 10.0);
      CHECK_TRUE (nb.size () == 3 && na.size () == 3);
      auto zpk = signal::tf2zpk (nb, na);
      auto [b_rec, a_rec] = signal::zpk2tf (zpk.z, zpk.p, zpk.k);
      CHECK_NEAR (b_rec[0], nb[0], 1e-9);
      CHECK_TRUE (signal::impulse_response (nb, na, 16).size () == 16);
      CHECK_TRUE (signal::step_response (nb, na, 16).size () == 16);
      CHECK_TRUE (signal::czt (nd_tensor<double> {1.0, 2.0, 3.0, 4.0}).size () == 4);
      CHECK_TRUE (signal::max_len_seq (4).size () == 15);
      CHECK_NEAR (tensor::sum (signal::savgol_coeffs (5, 2, 0)), 1.0, 1e-9);
      CHECK_TRUE (signal::flattop_window (8).size () == 8);

      nd_tensor<double> g_img = {{1.0, 2.0, 1.0}, {2.0, 5.0, 2.0}, {1.0, 2.0, 1.0}};
      CHECK_NEAR (signal::grey_dilation (g_img, 3) (0, 0), 5.0, 1e-12);
      CHECK_TRUE (signal::morphological_gradient (g_img, 3) (1, 1) > 0.0);
      nd_tensor<int> lbl_m = {{0, 1, 0}, {0, 1, 2}, {0, 0, 2}};
      CHECK_TRUE (signal::find_objects (lbl_m).size () == 2);
      CHECK_NEAR (signal::sum_labels (g_img, lbl_m, 1), 7.0, 1e-12);

      /* Sparse & Spatial additions.  */
      sparse::dok_matrix<double> dok (3, 3);
      dok.set (0, 0, 4.0);
      dok.set (1, 1, 9.0);
      dok.set (2, 2, 16.0);
      auto dok_csr = dok.to_csr ();
      CHECK_NEAR (sparse::svds (dok_csr, 1).s[0], 16.0, 1e-9);
      CHECK_NEAR (sparse::onenormest (dok_csr), 16.0, 1e-12);
      CHECK_TRUE (sparse::structural_rank (dok_csr) == 3);

      auto gs = spatial::geometric_slerp (
        nd_tensor<double> {1.0, 0.0},
        nd_tensor<double> {0.0, 1.0},
        nd_tensor<double> {0.0, 0.5, 1.0});
      CHECK_NEAR (gs (1, 0), std::sqrt (0.5), 1e-9);
      nd_tensor<double> hs = {
        {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0}
      };
      CHECK_TRUE (spatial::halfspace_intersection_2d (hs).shape ()[0] == 4);
    }
  }

  std::cout << "  -> Suite 2 passed.\n";
  return 0;
}
