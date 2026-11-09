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

#ifndef BASALT_C_H
#define BASALT_C_H

/* Stable C ABI for Basalt Deep Learning (libbasalt-c).
   All symbols use the basalt_ prefix and all macros use the BASALT_ prefix.
   C++ exceptions are caught at the ABI boundary and reported through
   thread-local error strings accessible via basalt_last_error().  */

#include <stddef.h>
#include <stdint.h>
#include <basalt/version.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum basalt_status
{
  BASALT_STATUS_OK = 0,
  BASALT_STATUS_INVALID_ARGUMENT = 1,
  BASALT_STATUS_SHAPE_ERROR = 2,
  BASALT_STATUS_DTYPE_ERROR = 3,
  BASALT_STATUS_DEVICE_ERROR = 4,
  BASALT_STATUS_IO_ERROR = 5,
  BASALT_STATUS_AUTOGRAD_ERROR = 6,
  BASALT_STATUS_RUNTIME_ERROR = 7
} basalt_status_t;

typedef enum basalt_dtype
{
  BASALT_DTYPE_FLOAT32 = 0,
  BASALT_DTYPE_FLOAT64 = 1,
  BASALT_DTYPE_FLOAT16 = 2,
  BASALT_DTYPE_BFLOAT16 = 3,
  BASALT_DTYPE_INT32 = 4,
  BASALT_DTYPE_INT64 = 5,
  BASALT_DTYPE_UINT8 = 6,
  BASALT_DTYPE_BOOL = 7
} basalt_dtype_t;

typedef enum basalt_device_type
{
  BASALT_DEVICE_CPU = 0,
  BASALT_DEVICE_CUDA = 1
} basalt_device_type_t;

typedef struct basalt_tensor_handle *basalt_tensor_t;
typedef struct basalt_mlp_handle *basalt_mlp_t;

/* Return the last error message on the calling thread.  */
const char *
basalt_last_error (void);

/* Seed all thread-local and global deterministic random generators.  */
basalt_status_t
basalt_seed_everything (uint64_t seed);

/* Tensor lifecycle and metadata queries.  */
basalt_status_t
basalt_tensor_create (const size_t *shape,
                      size_t ndim,
                      basalt_dtype_t dtype,
                      basalt_device_type_t device_type,
                      size_t device_index,
                      int requires_grad,
                      basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_zeros (const size_t *shape,
                     size_t ndim,
                     basalt_dtype_t dtype,
                     basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_ones (const size_t *shape,
                    size_t ndim,
                    basalt_dtype_t dtype,
                    basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_from_data (const void *host_data,
                         const size_t *shape,
                         size_t ndim,
                         basalt_dtype_t dtype,
                         int requires_grad,
                         basalt_tensor_t *out_tensor);

void
basalt_tensor_destroy (basalt_tensor_t tensor);

size_t
basalt_tensor_ndim (const basalt_tensor_t tensor);

size_t
basalt_tensor_numel (const basalt_tensor_t tensor);

basalt_status_t
basalt_tensor_shape (const basalt_tensor_t tensor,
                     size_t *out_dims,
                     size_t max_dims);

basalt_dtype_t
basalt_tensor_dtype (const basalt_tensor_t tensor);

basalt_status_t
basalt_tensor_set_requires_grad (basalt_tensor_t tensor, int requires_grad);

basalt_status_t
basalt_tensor_copy_to_host (const basalt_tensor_t tensor,
                            void *dst_host,
                            size_t max_bytes);

basalt_status_t
basalt_tensor_item (const basalt_tensor_t tensor, double *out_value);

/* Differentiable operations and reverse-mode autograd.  */
basalt_status_t
basalt_tensor_add (const basalt_tensor_t a,
                   const basalt_tensor_t b,
                   basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_mul (const basalt_tensor_t a,
                   const basalt_tensor_t b,
                   basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_matmul (const basalt_tensor_t a,
                      const basalt_tensor_t b,
                      basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_relu (const basalt_tensor_t x,
                    basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_sum (const basalt_tensor_t x,
                   basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_mean (const basalt_tensor_t x,
                    basalt_tensor_t *out_tensor);

basalt_status_t
basalt_tensor_backward (const basalt_tensor_t loss);

basalt_status_t
basalt_tensor_grad (const basalt_tensor_t tensor,
                    basalt_tensor_t *out_grad);

/* High-level C MLP model creation, training, and serialization.  */
basalt_status_t
basalt_mlp_create (size_t in_features,
                   size_t hidden_features,
                   size_t out_features,
                   double learning_rate,
                   basalt_mlp_t *out_mlp);

void
basalt_mlp_destroy (basalt_mlp_t mlp);

basalt_status_t
basalt_mlp_forward (basalt_mlp_t mlp,
                    const basalt_tensor_t input,
                    basalt_tensor_t *out_pred);

basalt_status_t
basalt_mlp_train_step (basalt_mlp_t mlp,
                       const basalt_tensor_t inputs,
                       const basalt_tensor_t targets,
                       double *out_loss);

basalt_status_t
basalt_mlp_save (basalt_mlp_t mlp, const char *path);

basalt_status_t
basalt_mlp_load (basalt_mlp_t mlp, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* BASALT_C_H */
