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

#ifndef HYPER_MATH_C_H
#define HYPER_MATH_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types for C / FFI consumers (Python ctypes, Rust, Guile).  */
typedef struct hm_tensor_s hm_tensor_t;
typedef struct hm_dataframe_s hm_dataframe_t;

const char *hm_version (void);

/* Tensor lifecycle & element access.  */
hm_tensor_t *hm_tensor_zeros (const size_t *shape, size_t ndim);
hm_tensor_t *hm_tensor_from_data (const size_t *shape,
                                  size_t ndim,
                                  const double *data);
void hm_tensor_free (hm_tensor_t *t);
size_t hm_tensor_size (const hm_tensor_t *t);
size_t hm_tensor_ndim (const hm_tensor_t *t);
double hm_tensor_get_flat (const hm_tensor_t *t, size_t idx);
void hm_tensor_set_flat (hm_tensor_t *t, size_t idx, double val);
double *hm_tensor_raw_data (hm_tensor_t *t);

/* Tensor reductions & linear algebra.  */
double hm_tensor_sum (const hm_tensor_t *t);
double hm_tensor_mean (const hm_tensor_t *t);
double hm_tensor_std (const hm_tensor_t *t);
double hm_tensor_min (const hm_tensor_t *t);
double hm_tensor_max (const hm_tensor_t *t);
double hm_tensor_dot (const hm_tensor_t *a, const hm_tensor_t *b);
double hm_linalg_det (const hm_tensor_t *mat);
double hm_linalg_cond (const hm_tensor_t *mat);
hm_tensor_t *hm_tensor_matmul (const hm_tensor_t *a, const hm_tensor_t *b);
hm_tensor_t *hm_linalg_solve (const hm_tensor_t *a, const hm_tensor_t *b);
hm_tensor_t *hm_linalg_inv (const hm_tensor_t *a);
hm_tensor_t *hm_linalg_pinv (const hm_tensor_t *a);
hm_tensor_t *hm_linalg_lstsq (const hm_tensor_t *a, const hm_tensor_t *b);
hm_tensor_t *hm_linalg_expm (const hm_tensor_t *a);
hm_tensor_t *hm_linalg_svdvals (const hm_tensor_t *a);
hm_tensor_t *hm_fft_rfft_abs (const hm_tensor_t *x);
int hm_tensor_save_npy (const hm_tensor_t *t, const char *filepath);
hm_tensor_t *hm_tensor_load_npy (const char *filepath);
double hm_optimize_linear_sum_assignment (const hm_tensor_t *cost_matrix,
                                          int maximize);

/* Special functions, CODATA unit conversions & Quasi-Monte Carlo.  */
double hm_special_gamma (double x);
double hm_special_erf (double x);
double hm_special_bessel_j0 (double x);
double hm_special_lambertw (double x);
double hm_constants_convert_temperature (double val,
                                         const char *from_scale,
                                         const char *to_scale);
hm_tensor_t *hm_qmc_halton (size_t n, size_t d);

/* Tabular DataFrame CSV / NDJSON / HMB I/O, SQL & inspection.  */
hm_dataframe_t *hm_dataframe_read_csv (const char *filepath);
int hm_dataframe_write_csv (const hm_dataframe_t *df, const char *filepath);
hm_dataframe_t *hm_dataframe_read_ndjson (const char *filepath);
int hm_dataframe_write_ndjson (const hm_dataframe_t *df, const char *filepath);
hm_dataframe_t *hm_dataframe_read_hmb (const char *filepath);
int hm_dataframe_write_hmb (const hm_dataframe_t *df, const char *filepath);
hm_dataframe_t *hm_dataframe_sql_query (const hm_dataframe_t *df,
                                        const char *sql);
size_t hm_dataframe_nrows (const hm_dataframe_t *df);
size_t hm_dataframe_ncols (const hm_dataframe_t *df);
double hm_dataframe_col_sum (const hm_dataframe_t *df, const char *col_name);
double hm_dataframe_col_mean (const hm_dataframe_t *df, const char *col_name);
void hm_dataframe_free (hm_dataframe_t *df);

#ifdef __cplusplus
}
#endif

#endif /* HYPER_MATH_C_H */
