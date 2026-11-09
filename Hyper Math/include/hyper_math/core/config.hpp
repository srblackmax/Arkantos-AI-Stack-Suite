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

#ifndef HYPER_MATH_CORE_CONFIG_HPP
#define HYPER_MATH_CORE_CONFIG_HPP

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#define HYPER_MATH_VERSION_MAJOR 0
#define HYPER_MATH_VERSION_MINOR 1
#define HYPER_MATH_VERSION_PATCH 0
#define HYPER_MATH_VERSION_STRING "0.1.0"
#define HYPER_MATH_PACKAGE_STRING "GNU Hyper Math 0.1.0"
#define HYPER_MATH_NODISCARD [[nodiscard]]

#if defined(__GNUC__) || defined(__clang__)
#  define HYPER_MATH_ALWAYS_INLINE inline __attribute__ ((always_inline))
#  define HYPER_MATH_PURE __attribute__ ((pure))
#  define HYPER_MATH_HOT __attribute__ ((hot))
#  define HYPER_MATH_RESTRICT __restrict__
#else
#  define HYPER_MATH_ALWAYS_INLINE inline
#  define HYPER_MATH_PURE
#  define HYPER_MATH_HOT
#  define HYPER_MATH_RESTRICT
#endif

namespace hyper_math
{

/* Cache-line alignment boundary in bytes (512-bit SIMD ready).  */
inline constexpr std::size_t cache_line_bytes = 64;

/* Fundamental mathematical and CODATA physical constants in double precision. */
namespace constants
{
  inline constexpr double pi = 3.141592653589793238462643383279502884;
  inline constexpr double two_pi = 6.283185307179586476925286766559005768;
  inline constexpr double half_pi = 1.570796326794896619231321691639751442;
  inline constexpr double e = 2.718281828459045235360287471352662498;
  inline constexpr double golden_ratio = 1.618033988749894848204586834365638118;
  inline constexpr double sqrt2 = 1.414213562373095048801688724209698079;
  inline constexpr double sqrt_two_pi = 2.506628274631000502415765284811045253;
  inline constexpr double euler_gamma = 0.577215664901532860606512090082402431;
  inline constexpr double ln2 = 0.693147180559945309417232121458176568;
  inline constexpr double nan = std::numeric_limits<double>::quiet_NaN ();
  inline constexpr double inf = std::numeric_limits<double>::infinity ();

  /* CODATA SI physical constants (scipy.constants parity).  */
  inline constexpr double c = 299792458.0;
  inline constexpr double speed_of_light = 299792458.0;
  inline constexpr double h = 6.62607015e-34;
  inline constexpr double planck = 6.62607015e-34;
  inline constexpr double hbar = 1.0545718176461565e-34;
  inline constexpr double elementary_charge = 1.602176634e-19;
  inline constexpr double m_e = 9.1093837015e-31;
  inline constexpr double m_p = 1.67262192369e-27;
  inline constexpr double m_n = 1.67492749804e-27;
  inline constexpr double n_avogadro = 6.02214076e23;
  inline constexpr double k_boltzmann = 1.380649e-23;
  inline constexpr double r_gas = 8.31446261815324;
  inline constexpr double g_grav = 6.67430e-11;
  inline constexpr double g_standard = 9.80665;
  inline constexpr double epsilon_0 = 8.8541878128e-12;
  inline constexpr double mu_0 = 1.25663706212e-6;
  inline constexpr double sigma_sb = 5.670374419e-8;
  inline constexpr double alpha_fine_structure = 7.2973525693e-3;
  inline constexpr double rydberg = 10973731.568160;
  inline constexpr double atomic_mass = 1.66053906660e-27;
  inline constexpr double zero_celsius = 273.15;

  /* SI metric prefixes.  */
  inline constexpr double yotta = 1e24;
  inline constexpr double zetta = 1e21;
  inline constexpr double exa = 1e18;
  inline constexpr double peta = 1e15;
  inline constexpr double tera = 1e12;
  inline constexpr double giga = 1e9;
  inline constexpr double mega = 1e6;
  inline constexpr double kilo = 1e3;
  inline constexpr double hecto = 1e2;
  inline constexpr double deca = 1e1;
  inline constexpr double deci = 1e-1;
  inline constexpr double centi = 1e-2;
  inline constexpr double milli = 1e-3;
  inline constexpr double micro = 1e-6;
  inline constexpr double nano = 1e-9;
  inline constexpr double pico = 1e-12;
  inline constexpr double femto = 1e-15;
  inline constexpr double atto = 1e-18;
  inline constexpr double zepto = 1e-21;
  inline constexpr double yocto = 1e-24;
} // namespace constants

/* C++20 scalar concepts for type-safe mathematical templates.  */
template <typename T>
concept arithmetic_scalar = std::is_arithmetic_v<T>;

template <typename T>
concept floating_scalar = std::floating_point<T>;

template <typename T>
concept integral_scalar = std::integral<T>;

/* Base exception class formatted according to GNU diagnostic conventions.  */
class hyper_math_error : public std::runtime_error
{
public:
  explicit
  hyper_math_error (const std::string &category, const std::string &message)
    : std::runtime_error ("hyper-math: " + category + ": " + message)
  {
  }

  explicit
  hyper_math_error (const std::string &message)
    : std::runtime_error ("hyper-math: error: " + message)
  {
  }
};

class shape_error : public hyper_math_error
{
public:
  explicit
  shape_error (const std::string &message)
    : hyper_math_error ("shape_error", message)
  {
  }
};

class index_error : public hyper_math_error
{
public:
  explicit
  index_error (const std::string &message)
    : hyper_math_error ("index_error", message)
  {
  }
};

class singular_matrix_error : public hyper_math_error
{
public:
  explicit
  singular_matrix_error (const std::string &message)
    : hyper_math_error ("singular_matrix_error", message)
  {
  }
};

class convergence_error : public hyper_math_error
{
public:
  explicit
  convergence_error (const std::string &message)
    : hyper_math_error ("convergence_error", message)
  {
  }
};

class schema_error : public hyper_math_error
{
public:
  explicit
  schema_error (const std::string &message)
    : hyper_math_error ("schema_error", message)
  {
  }
};

class parse_error : public hyper_math_error
{
public:
  explicit
  parse_error (const std::string &message)
    : hyper_math_error ("parse_error", message)
  {
  }
};

class value_error : public hyper_math_error
{
public:
  explicit
  value_error (const std::string &message)
    : hyper_math_error ("value_error", message)
  {
  }
};

/* Runtime version query functions implemented in libhypermath.cpp.  */
const char *
get_version_string () noexcept;

const char *
get_license_notice () noexcept;

namespace constants
{
  /* Temperature conversion between "C", "K", "F", and "R".  */
  inline double
  convert_temperature (double val,
                       const std::string &old_scale,
                       const std::string &new_scale)
  {
    double kelvin = 0.0;
    if (old_scale == "K" || old_scale == "Kelvin")
      {
        kelvin = val;
      }
    else if (old_scale == "C" || old_scale == "Celsius")
      {
        kelvin = val + zero_celsius;
      }
    else if (old_scale == "F" || old_scale == "Fahrenheit")
      {
        kelvin = (val - 32.0) * (5.0 / 9.0) + zero_celsius;
      }
    else if (old_scale == "R" || old_scale == "Rankine")
      {
        kelvin = val * (5.0 / 9.0);
      }
    else
      {
        throw hyper_math_error ("unknown temperature scale: " + old_scale);
      }

    if (new_scale == "K" || new_scale == "Kelvin")
      {
        return kelvin;
      }
    if (new_scale == "C" || new_scale == "Celsius")
      {
        return kelvin - zero_celsius;
      }
    if (new_scale == "F" || new_scale == "Fahrenheit")
      {
        return (kelvin - zero_celsius) * (9.0 / 5.0) + 32.0;
      }
    if (new_scale == "R" || new_scale == "Rankine")
      {
        return kelvin * (9.0 / 5.0);
      }
    throw hyper_math_error ("unknown temperature scale: " + new_scale);
  }
} // namespace constants

} // namespace hyper_math

#endif /* HYPER_MATH_CORE_CONFIG_HPP */
