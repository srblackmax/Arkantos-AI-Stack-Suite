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

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

void
print_version ()
{
  std::cout << hyper_math::get_version_string () << "\n"
            << hyper_math::get_license_notice () << "\n\n"
            << "Written by Kival and the Kival.\n";
}

void
print_help (const char *prog_name)
{
  std::cout
    << "Usage: " << prog_name << " [OPTION]...\n"
    << "Unified C++20 engine for N-D Tensors (NumPy), Scientific Computing (SciPy),\n"
    << "Automatic Differentiation, and Columnar/Lazy DataFrames (Pandas & Polars).\n\n"
    << "Options:\n"
    << "  -d, --demo             run a comprehensive demonstration of all engine pillars\n"
    << "  -r, --repl             launch interactive Read-Eval-Print Loop shell\n"
    << "  -c, --csv=FILE         load CSV file and display head(10) & describe() summary\n"
    << "  -j, --ndjson=FILE      load NDJSON file and display head(10) & describe() summary\n"
    << "  -e, --eval=CMD         evaluate command ('roots', 'hilbert', 'svd', 'fft',\n"
    << "                         'stats', 'grad', 'quad', 'ode', 'qmc', 'sql',\n"
    << "                         'constants')\n"
    << "  -h, --help             display this help and exit\n"
    << "  -v, --version          output version information and exit\n\n"
    << "Report bugs to: <bug-hyper-math@gnu.org>\n";
}

std::vector<double>
parse_csv_doubles (const std::string &s)
{
  std::vector<double> out;
  std::stringstream ss (s);
  std::string item;
  while (std::getline (ss, item, ','))
    {
      if (!item.empty ())
        {
          out.push_back (std::stod (item));
        }
    }
  return out;
}

void
run_demo ();

int
run_eval_command (const std::string &cmd)
{
  using namespace hyper_math;
  std::stringstream ss (cmd);
  std::string op;
  std::string arg;
  ss >> op;
  std::getline (ss >> std::ws, arg);

  if (op.empty ())
    {
      return 0;
    }

  try
    {
      if (op == "roots")
        {
          const auto coeffs = parse_csv_doubles (arg);
          tensor::nd_tensor<double> c (
            tensor::shape_type {coeffs.size ()}, coeffs);
          const auto r = scientific::linalg::roots (c);
          std::cout << "roots(" << c << ") = [";
          for (std::size_t i = 0; i < r.size (); ++i)
            {
              if (i > 0)
                {
                  std::cout << ", ";
                }
              std::cout << r[i].real ();
              if (std::abs (r[i].imag ()) > 1e-12)
                {
                  std::cout << (r[i].imag () >= 0.0 ? "+" : "")
                            << r[i].imag () << "i";
                }
            }
          std::cout << "]\n";
          return 0;
        }
      else if (op == "hilbert")
        {
          if (!arg.empty () && arg[0] == '-')
            {
              throw hyper_math_error ("hilbert order must be positive");
            }
          const std::size_t n =
            arg.empty () ? 3 : static_cast<std::size_t> (std::stoull (arg));
          const auto h = scientific::linalg::hilbert (n);
          std::cout << "hilbert(" << n << ") =\n" << h
                    << "\ndet = " << scientific::linalg::det (h)
                    << ", cond = " << scientific::linalg::cond (h) << "\n";
          return 0;
        }
      else if (op == "fft")
        {
          const auto vals = parse_csv_doubles (arg);
          tensor::nd_tensor<double> sig (
            tensor::shape_type {vals.size ()}, vals);
          const auto spec = scientific::fft::fft (sig);
          std::cout << "fft(" << sig << ") = [";
          for (std::size_t i = 0; i < spec.size (); ++i)
            {
              if (i > 0)
                {
                  std::cout << ", ";
                }
              std::cout << "(" << spec[i].real () << ", "
                        << spec[i].imag () << "i)";
            }
          std::cout << "]\n";
          return 0;
        }
      else if (op == "stats")
        {
          const auto vals = parse_csv_doubles (arg);
          tensor::nd_tensor<double> x (
            tensor::shape_type {vals.size ()}, vals);
          std::cout << "n      = " << x.size () << "\n"
                    << "mean   = " << tensor::mean (x) << "\n"
                    << "std    = " << tensor::std_dev (x) << "\n"
                    << "median = " << tensor::median (x) << "\n"
                    << "min    = " << tensor::min (x) << "\n"
                    << "max    = " << tensor::max (x) << "\n"
                    << "skew   = " << scientific::stats::skew (x) << "\n"
                    << "kurt   = " << scientific::stats::kurtosis (x) << "\n";
          return 0;
        }
      else if (op == "grad")
        {
          const auto vals = parse_csv_doubles (arg);
          tensor::nd_tensor<double> pt =
            (vals.size () >= 2)
              ? tensor::nd_tensor<double> {vals[0], vals[1]}
              : tensor::nd_tensor<double> {-1.2, 1.0};
          auto rosen = [] (const std::vector<scientific::autograd::var> &v) {
            const auto d0 = 1.0 - v[0];
            const auto d1 = v[1] - v[0] * v[0];
            return d0 * d0 + 100.0 * d1 * d1;
          };
          const auto vg = scientific::autograd::value_and_grad (rosen, pt);
          const auto h = scientific::autograd::hessian (rosen, pt);
          std::cout << "Rosenbrock at x = " << pt << ":\n"
                    << "  f(x)      = " << vg.value << "\n"
                    << "  grad f(x) = " << vg.gradient << "\n"
                    << "  hess f(x) = " << h << "\n";
          return 0;
        }
      else if (op == "quad")
        {
          const auto vals = parse_csv_doubles (arg);
          const double a = vals.size () >= 1 ? vals[0] : 0.0;
          const double b = vals.size () >= 2 ? vals[1] : 1.0;
          const auto q = scientific::integrate::quad (
            [] (double x) { return std::exp (-x * x); }, a, b);
          std::cout << "quad(exp(-x^2), " << a << ", " << b << ") = "
                    << std::setprecision (12) << q.value
                    << " (error ~ " << q.error_estimate << ")\n";
          return 0;
        }
      else if (op == "svd")
        {
          const std::size_t n =
            arg.empty () ? 3 : static_cast<std::size_t> (std::stoull (arg));
          const auto h = scientific::linalg::hilbert (n);
          const auto s = scientific::linalg::svdvals (h);
          std::cout << "svdvals(hilbert(" << n << ")) = " << s << "\n";
          return 0;
        }
      else if (op == "qmc")
        {
          const auto vals = parse_csv_doubles (arg);
          const std::size_t n =
            vals.size () >= 1 ? static_cast<std::size_t> (vals[0]) : 16;
          const std::size_t d =
            vals.size () >= 2 ? static_cast<std::size_t> (vals[1]) : 2;
          const auto h = scientific::stats::qmc::halton (n, d);
          const double disc = scientific::stats::qmc::discrepancy (h);
          std::cout << "qmc::halton(n=" << n << ", d=" << d
                    << ") L2 discrepancy = " << disc << "\n";
          return 0;
        }
      else if (op == "ode")
        {
          const double t_end = arg.empty () ? 1.0 : std::stod (arg);
          const auto sol = scientific::integrate::solve_ivp (
            [] (double, const tensor::nd_tensor<double> &y) {
              return tensor::nd_tensor<double> {-y[0]};
            },
            0.0, t_end, tensor::nd_tensor<double> {1.0});
          std::cout << "solve_ivp(y' = -y, y(0)=1, t=" << t_end << ") -> y(t) = "
                    << std::setprecision (12) << sol.y[sol.y.size () - 1]
                    << " (exact = " << std::exp (-t_end) << ")\n";
          return 0;
        }
      else if (op == "constants")
        {
          std::cout << "CODATA Physical Constants:\n"
                    << "  c     = " << constants::c << " m/s\n"
                    << "  h     = " << constants::h << " J*s\n"
                    << "  k_B   = " << constants::k_boltzmann << " J/K\n"
                    << "  N_A   = " << constants::n_avogadro << " mol^-1\n"
                    << "  alpha = " << constants::alpha_fine_structure << "\n"
                    << "  100 C = "
                    << constants::convert_temperature (100.0, "Celsius", "Kelvin")
                    << " K ("
                    << constants::convert_temperature (100.0, "Celsius", "Fahrenheit")
                    << " F)\n";
          return 0;
        }
      else if (op == "sql")
        {
          tabular::data_frame df ({
            tabular::series ("dept", std::vector<std::string> {
              "AI", "AI", "Quant", "Quant", "Systems"
            }),
            tabular::series ("salary", std::vector<double> {
              150000.0, 180000.0, 210000.0, 190000.0, 140000.0
            }),
            tabular::series ("bonus", std::vector<double> {
              20000.0, 30000.0, 45000.0, 35000.0, 15000.0
            })
          });
          const std::string q =
            arg.empty ()
              ? "SELECT dept, salary, bonus FROM employees WHERE salary >= 180000 ORDER BY salary DESC"
              : arg;
          tabular::sql_context ctx;
          ctx.register_table ("employees", tabular::lazy_frame (df));
          std::cout << ctx.execute (q);
          return 0;
        }
      else if (op == "demo")
        {
          run_demo ();
          return 0;
        }
      else if (op == "help")
        {
          std::cout
            << "Available commands:\n"
            << "  roots a,b,c       Polynomial companion-matrix roots\n"
            << "  hilbert N         N x N Hilbert matrix, det, and condition number\n"
            << "  svd N             Singular values of N x N Hilbert matrix\n"
            << "  fft x1,x2,...     1D Fast Fourier Transform\n"
            << "  stats x1,x2,...   Summary statistics (mean, std, median, skew, kurt)\n"
            << "  grad x,y          Reverse-mode autograd value, gradient & Hessian\n"
            << "  quad a,b          Gauss-Kronrod G7-K15 integral of exp(-x^2)\n"
            << "  ode T             Dormand-Prince RK4(5) ODE solution of y'=-y at t=T\n"
            << "  qmc N,D           Halton Quasi-Monte Carlo sequence & discrepancy\n"
            << "  sql QUERY         Execute SQL SELECT query over demo employees table\n"
            << "  constants         Display CODATA physical constants & unit conversions\n"
            << "  demo              Run full 4-pillar + Autograd + SQL demonstration\n"
            << "  quit / exit       Exit interactive shell\n";
          return 0;
        }
    }
  catch (const std::exception &ex)
    {
      std::cerr << "hyper-math error: " << ex.what () << "\n";
      return 1;
    }

  std::cerr << "hyper-math: unknown command '" << op
            << "' (type 'help' for supported commands)\n";
  return 1;
}

int
run_repl ()
{
  std::cout << hyper_math::get_version_string ()
            << " Interactive Shell (type 'help' or 'quit')\n";
  std::string line;
  while (true)
    {
      std::cout << "hyper-math> " << std::flush;
      if (!std::getline (std::cin, line))
        {
          std::cout << "\n";
          break;
        }
      std::stringstream ss (line);
      std::string first;
      ss >> first;
      if (first.empty ())
        {
          continue;
        }
      if (first == "quit" || first == "exit")
        {
          break;
        }
      run_eval_command (line);
    }
  return 0;
}

int
inspect_tabular_file (const std::string &filepath, bool is_ndjson)
{
  using namespace hyper_math::tabular;
  try
    {
      data_frame df =
        is_ndjson ? data_frame::read_ndjson (filepath)
                  : data_frame::read_csv (filepath);
      std::cout << "=== File: " << filepath << " (" << df.nrows () << " rows x "
                << df.ncols () << " cols) ===\n"
                << df.head (10)
                << "\n=== Summary Statistics (describe) ===\n"
                << df.describe ();
      return 0;
    }
  catch (const std::exception &ex)
    {
      std::cerr << "hyper-math error: " << ex.what () << "\n";
      return 1;
    }
}

void
run_demo ()
{
  using namespace hyper_math;
  std::cout << "==============================================================\n"
            << " " << get_version_string () << " -- Full Engine Demonstration\n"
            << "==============================================================\n\n";

  /* 1. NumPy-equivalent N-D Tensor & Einstein Summation.  */
  std::cout << "[1] Tensor Engine (NumPy layer): Broadcasting & Einsum\n";
  tensor::nd_tensor<double> a = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  tensor::nd_tensor<double> b ({2, 1}, {10.0, 20.0});
  tensor::nd_tensor<double> c = tensor::sin (a) + b * 2.0;
  std::cout << "  Broadcasted sin(A) + 2*b: " << c << "\n";
  auto trace_ab = tensor::einsum<double> ("ij,ji->", {a, a.t ()});
  std::cout << "  einsum(\"ij,ji->\", A, A^T) = " << trace_ab[0] << "\n\n";

  /* 2. SciPy-equivalent Linear Algebra, FFT, Quadrature, Optimization, Autograd. */
  std::cout << "[2] Scientific Continuum (SciPy + Autograd layer)\n";
  tensor::nd_tensor<double> spd = {{4.0, 1.0}, {1.0, 3.0}};
  auto eig = scientific::linalg::eigh (spd);
  std::cout << "  Eigenvalues of [[4, 1], [1, 3]]: " << eig.eigenvalues << "\n";

  auto q_res = scientific::integrate::quad (
    [] (double x) { return std::exp (-x * x); }, 0.0, 1.0);
  std::cout << "  Gauss-Kronrod G7-K15 int_0^1 exp(-x^2) dx = "
            << std::setprecision (10) << q_res.value
            << " (err ~ " << q_res.error_estimate << ")\n";

  auto rosen_ag = [] (const std::vector<scientific::autograd::var> &v) {
    const auto d0 = 1.0 - v[0];
    const auto d1 = v[1] - v[0] * v[0];
    return d0 * d0 + 100.0 * d1 * d1;
  };
  auto opt = scientific::autograd::minimize_bfgs_autograd (
    rosen_ag, tensor::nd_tensor<double> {-1.2, 1.0});
  std::cout << "  Autograd-BFGS Rosenbrock minimum at x = " << opt.x
            << " (f = " << opt.fun << ", nit = " << opt.nit << ")\n";
  auto qmc_pts = scientific::stats::qmc::halton (64, 2);
  std::cout << "  Quasi-Monte Carlo Halton(64, 2) L2 discrepancy = "
            << scientific::stats::qmc::discrepancy (qmc_pts) << "\n\n";

  /* 3. Pandas/Polars-equivalent Dual-Bitmask DataFrame & Lazy Optimizer.  */
  std::cout << "[3] Tabular & Lazy Engine (Pandas & Polars layer)\n";
  tabular::data_frame df ({
    tabular::series ("dept", std::vector<std::string> {
      "AI", "AI", "Quant", "Quant", "Systems"
    }),
    tabular::series ("salary", std::vector<double> {
      150000.0, 180000.0, 210000.0, 190000.0, 140000.0
    }),
    tabular::series ("bonus", std::vector<double> {
      20000.0, 30000.0, 45000.0, 35000.0, 15000.0
    }),
    tabular::series ("unused_col", std::vector<double> {
      1.0, 2.0, 3.0, 4.0, 5.0
    })
  });

  using tabular::col;
  using tabular::lit;

  tabular::lazy_frame lf =
    tabular::lazy_frame (df)
      .with_columns ({(col ("salary") + col ("bonus")).alias ("total_comp")})
      .filter (col ("salary") > lit (100000.0) + lit (45000.0))
      .sort ("total_comp", false);

  std::cout << "  -- Unoptimized Logical Plan --\n"
            << lf.explain (false)
            << "  -- Optimized Logical Plan (Predicate & Projection Pushdown) --\n"
            << lf.explain (true)
            << "  -- Materialized Result --\n"
            << lf.collect (true)
            << "\n  -- SQL Context Query (SELECT dept, salary FROM employees WHERE salary >= 180000) --\n"
            << df.sql_query ("SELECT dept, salary FROM self WHERE salary >= 180000 ORDER BY salary DESC");
}

} // namespace

int
main (int argc, char **argv)
{
  if (argc <= 1)
    {
      run_demo ();
      return 0;
    }

  for (int i = 1; i < argc; ++i)
    {
      const std::string arg (argv[i]);
      if (arg == "--version" || arg == "-v")
        {
          print_version ();
          return 0;
        }
      if (arg == "--help" || arg == "-h")
        {
          print_help (argv[0]);
          return 0;
        }
      if (arg == "--demo" || arg == "-d")
        {
          run_demo ();
          return 0;
        }
      if (arg == "--repl" || arg == "-r")
        {
          return run_repl ();
        }
      if (arg.rfind ("--csv=", 0) == 0)
        {
          return inspect_tabular_file (arg.substr (6), false);
        }
      if (arg == "--csv" || arg == "-c")
        {
          if (i + 1 >= argc)
            {
              std::cerr << argv[0] << ": option '" << arg
                        << "' requires an argument\n";
              return 1;
            }
          return inspect_tabular_file (argv[++i], false);
        }
      if (arg.rfind ("--ndjson=", 0) == 0)
        {
          return inspect_tabular_file (arg.substr (9), true);
        }
      if (arg == "--ndjson" || arg == "-j")
        {
          if (i + 1 >= argc)
            {
              std::cerr << argv[0] << ": option '" << arg
                        << "' requires an argument\n";
              return 1;
            }
          return inspect_tabular_file (argv[++i], true);
        }
      if (arg.rfind ("--eval=", 0) == 0)
        {
          return run_eval_command (arg.substr (7));
        }
      if (arg == "--eval" || arg == "-e")
        {
          if (i + 1 >= argc)
            {
              std::cerr << argv[0] << ": option '" << arg
                        << "' requires an argument\n";
              return 1;
            }
          return run_eval_command (argv[++i]);
        }

      std::cerr << argv[0] << ": unrecognized option '" << argv[i] << "'\n"
                << "Try '" << argv[0] << " --help' for more information.\n";
      return 1;
    }

  return 0;
}
