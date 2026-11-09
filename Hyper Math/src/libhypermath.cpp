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

namespace hyper_math
{

const char *
get_version_string () noexcept
{
  return HYPER_MATH_PACKAGE_STRING;
}

const char *
get_license_notice () noexcept
{
  return "Copyright (C) 2026 Kival.\n"
         "License GPLv3+: GNU GPL version 3 or later "
         "<https://gnu.org/licenses/gpl.html>.\n"
         "This is free software: you are free to change and redistribute it.\n"
         "There is NO WARRANTY, to the extent permitted by law.";
}

/* Explicit template instantiations for core tensor types.  */
template class slab_buffer<double>;
template class slab_buffer<float>;
template class slab_buffer<std::int64_t>;
template class slab_buffer<std::uint8_t>;
template class tensor::nd_tensor<double>;
template class tensor::nd_tensor<std::int64_t>;
template class tensor::nd_tensor<std::uint8_t>;

} // namespace hyper_math

/* Pure C ABI Wrapper Implementation.  */
struct hm_tensor_s
{
  hyper_math::tensor::nd_tensor<double> inner;
};

struct hm_dataframe_s
{
  hyper_math::tabular::data_frame inner;
};

extern "C" {

const char *
hm_version (void)
{
  return HYPER_MATH_VERSION_STRING;
}

hm_tensor_t *
hm_tensor_zeros (const size_t *shape, size_t ndim)
{
  if (!shape || ndim == 0)
    {
      return nullptr;
    }
  try
    {
      hyper_math::tensor::shape_type dims (shape, shape + ndim);
      return new hm_tensor_s {
        hyper_math::tensor::zeros<double> (std::move (dims))
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_tensor_t *
hm_tensor_from_data (const size_t *shape, size_t ndim, const double *data)
{
  if (!shape || ndim == 0 || !data)
    {
      return nullptr;
    }
  try
    {
      hyper_math::tensor::shape_type dims (shape, shape + ndim);
      std::size_t total = 1;
      for (std::size_t d : dims)
        {
          if (d != 0
              && total > std::numeric_limits<std::size_t>::max () / d)
            {
              return nullptr;
            }
          total *= d;
        }
      std::vector<double> vec (data, data + total);
      return new hm_tensor_s {
        hyper_math::tensor::nd_tensor<double> (std::move (dims), vec)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

void
hm_tensor_free (hm_tensor_t *t)
{
  delete t;
}

size_t
hm_tensor_size (const hm_tensor_t *t)
{
  return t ? t->inner.size () : 0;
}

size_t
hm_tensor_ndim (const hm_tensor_t *t)
{
  return t ? t->inner.ndim () : 0;
}

double
hm_tensor_get_flat (const hm_tensor_t *t, size_t idx)
{
  if (!t || idx >= t->inner.size ())
    {
      return 0.0;
    }
  try
    {
      return t->inner.eval_at (idx);
    }
  catch (...)
    {
      return 0.0;
    }
}

void
hm_tensor_set_flat (hm_tensor_t *t, size_t idx, double val)
{
  if (t && idx < t->inner.size ())
    {
      try
        {
          t->inner[idx] = val;
        }
      catch (...)
        {
        }
    }
}

double *
hm_tensor_raw_data (hm_tensor_t *t)
{
  return t ? t->inner.raw_data () : nullptr;
}

double
hm_tensor_sum (const hm_tensor_t *t)
{
  if (!t)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::tensor::sum (t->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_tensor_mean (const hm_tensor_t *t)
{
  if (!t)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::tensor::mean (t->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_tensor_std (const hm_tensor_t *t)
{
  if (!t)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::tensor::std_dev (t->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_tensor_min (const hm_tensor_t *t)
{
  if (!t)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::tensor::min (t->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_tensor_max (const hm_tensor_t *t)
{
  if (!t)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::tensor::max (t->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_tensor_dot (const hm_tensor_t *a, const hm_tensor_t *b)
{
  if (!a || !b)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::tensor::dot (a->inner, b->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_linalg_det (const hm_tensor_t *mat)
{
  if (!mat)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::scientific::linalg::det (mat->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_linalg_cond (const hm_tensor_t *mat)
{
  if (!mat)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::scientific::linalg::cond (mat->inner);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

hm_tensor_t *
hm_tensor_matmul (const hm_tensor_t *a, const hm_tensor_t *b)
{
  if (!a || !b)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::tensor::matmul (a->inner, b->inner)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_tensor_t *
hm_linalg_solve (const hm_tensor_t *a, const hm_tensor_t *b)
{
  if (!a || !b)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::scientific::linalg::solve (a->inner, b->inner)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_tensor_t *
hm_linalg_inv (const hm_tensor_t *a)
{
  if (!a)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::scientific::linalg::inv (a->inner)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_tensor_t *
hm_linalg_pinv (const hm_tensor_t *a)
{
  if (!a)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::scientific::linalg::pinv (a->inner)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_tensor_t *
hm_linalg_lstsq (const hm_tensor_t *a, const hm_tensor_t *b)
{
  if (!a || !b)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::scientific::linalg::lstsq (a->inner, b->inner)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_tensor_t *
hm_linalg_expm (const hm_tensor_t *a)
{
  if (!a)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::scientific::linalg::expm (a->inner)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

int
hm_tensor_save_npy (const hm_tensor_t *t, const char *filepath)
{
  if (!t || !filepath)
    {
      return -1;
    }
  try
    {
      hyper_math::tensor::save_npy (filepath, t->inner);
      return 0;
    }
  catch (...)
    {
      return -1;
    }
}

hm_tensor_t *
hm_tensor_load_npy (const char *filepath)
{
  if (!filepath)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::tensor::load_npy (filepath)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

double
hm_optimize_linear_sum_assignment (const hm_tensor_t *cost_matrix, int maximize)
{
  if (!cost_matrix)
    {
      return 0.0;
    }
  try
    {
      return hyper_math::scientific::optimize::linear_sum_assignment (
               cost_matrix->inner, maximize != 0)
        .total_cost;
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

hm_dataframe_t *
hm_dataframe_read_csv (const char *filepath)
{
  if (!filepath)
    {
      return nullptr;
    }
  try
    {
      return new hm_dataframe_s {
        hyper_math::tabular::data_frame::read_csv (filepath)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

int
hm_dataframe_write_csv (const hm_dataframe_t *df, const char *filepath)
{
  if (!df || !filepath)
    {
      return -1;
    }
  try
    {
      df->inner.write_csv (filepath);
      return 0;
    }
  catch (...)
    {
      return -1;
    }
}

hm_dataframe_t *
hm_dataframe_read_ndjson (const char *filepath)
{
  if (!filepath)
    {
      return nullptr;
    }
  try
    {
      return new hm_dataframe_s {
        hyper_math::tabular::data_frame::read_ndjson (filepath)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

int
hm_dataframe_write_ndjson (const hm_dataframe_t *df, const char *filepath)
{
  if (!df || !filepath)
    {
      return -1;
    }
  try
    {
      df->inner.write_ndjson (filepath);
      return 0;
    }
  catch (...)
    {
      return -1;
    }
}

hm_dataframe_t *
hm_dataframe_read_hmb (const char *filepath)
{
  if (!filepath)
    {
      return nullptr;
    }
  try
    {
      return new hm_dataframe_s {
        hyper_math::tabular::data_frame::read_hmb (filepath)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

int
hm_dataframe_write_hmb (const hm_dataframe_t *df, const char *filepath)
{
  if (!df || !filepath)
    {
      return -1;
    }
  try
    {
      df->inner.write_hmb (filepath);
      return 0;
    }
  catch (...)
    {
      return -1;
    }
}

size_t
hm_dataframe_nrows (const hm_dataframe_t *df)
{
  return df ? df->inner.nrows () : 0;
}

size_t
hm_dataframe_ncols (const hm_dataframe_t *df)
{
  return df ? df->inner.ncols () : 0;
}

double
hm_dataframe_col_sum (const hm_dataframe_t *df, const char *col_name)
{
  if (!df || !col_name || !df->inner.has_column (col_name))
    {
      return 0.0;
    }
  try
    {
      return df->inner[col_name].sum ();
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

double
hm_dataframe_col_mean (const hm_dataframe_t *df, const char *col_name)
{
  if (!df || !col_name || !df->inner.has_column (col_name))
    {
      return 0.0;
    }
  try
    {
      return df->inner[col_name].mean ();
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

void
hm_dataframe_free (hm_dataframe_t *df)
{
  delete df;
}

hm_tensor_t *
hm_linalg_svdvals (const hm_tensor_t *a)
{
  if (!a)
    {
      return nullptr;
    }
  try
    {
      return new hm_tensor_s {
        hyper_math::scientific::linalg::svdvals (a->inner)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_tensor_t *
hm_fft_rfft_abs (const hm_tensor_t *x)
{
  if (!x)
    {
      return nullptr;
    }
  try
    {
      const auto spec = hyper_math::scientific::fft::rfft (x->inner);
      return new hm_tensor_s {
        hyper_math::tensor::complex_abs (spec)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

double
hm_special_gamma (double x)
{
  return hyper_math::scientific::stats::special::gamma (x);
}

double
hm_special_erf (double x)
{
  return hyper_math::scientific::stats::special::erf (x);
}

double
hm_special_bessel_j0 (double x)
{
  return hyper_math::scientific::stats::special::j0 (x);
}

double
hm_special_lambertw (double x)
{
  return hyper_math::scientific::stats::special::lambertw (x);
}

double
hm_constants_convert_temperature (double val,
                                  const char *from_scale,
                                  const char *to_scale)
{
  if (!from_scale || !to_scale)
    {
      return hyper_math::constants::nan;
    }
  try
    {
      return hyper_math::constants::convert_temperature (
        val, from_scale, to_scale);
    }
  catch (...)
    {
      return hyper_math::constants::nan;
    }
}

hm_tensor_t *
hm_qmc_halton (size_t n, size_t d)
{
  try
    {
      return new hm_tensor_s {
        hyper_math::scientific::stats::qmc::halton (n, d)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

hm_dataframe_t *
hm_dataframe_sql_query (const hm_dataframe_t *df, const char *sql)
{
  if (!df || !sql)
    {
      return nullptr;
    }
  try
    {
      return new hm_dataframe_s {
        df->inner.sql_query (sql)
      };
    }
  catch (...)
    {
      return nullptr;
    }
}

} /* extern "C" */
