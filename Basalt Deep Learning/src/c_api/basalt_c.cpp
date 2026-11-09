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

#include <basalt/basalt_c.h>

#include <cstring>
#include <memory>
#include <string>
#include <basalt/basalt.hpp>

struct basalt_tensor_handle
{
  basalt::Tensor tensor;
};

struct basalt_mlp_handle
{
  std::shared_ptr<basalt::nn::Sequential> model;
  std::unique_ptr<basalt::optim::Adam> optimizer;
};

namespace
{

thread_local std::string tls_last_error;

basalt_status_t
translate_exception (void) noexcept
{
  try
    {
      throw;
    }
  catch (const basalt::Invalid_Argument_Error &e)
    {
      tls_last_error = e.what ();
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  catch (const basalt::Shape_Error &e)
    {
      tls_last_error = e.what ();
      return BASALT_STATUS_SHAPE_ERROR;
    }
  catch (const basalt::Dtype_Error &e)
    {
      tls_last_error = e.what ();
      return BASALT_STATUS_DTYPE_ERROR;
    }
  catch (const basalt::Device_Error &e)
    {
      tls_last_error = e.what ();
      return BASALT_STATUS_DEVICE_ERROR;
    }
  catch (const basalt::IO_Error &e)
    {
      tls_last_error = e.what ();
      return BASALT_STATUS_IO_ERROR;
    }
  catch (const basalt::Autograd_Error &e)
    {
      tls_last_error = e.what ();
      return BASALT_STATUS_AUTOGRAD_ERROR;
    }
  catch (const std::exception &e)
    {
      tls_last_error = e.what ();
      return BASALT_STATUS_RUNTIME_ERROR;
    }
  catch (...)
    {
      tls_last_error = "Unknown C++ exception in libbasalt-c";
      return BASALT_STATUS_RUNTIME_ERROR;
    }
}

basalt::Dtype
to_cpp_dtype (basalt_dtype_t dt)
{
  return static_cast<basalt::Dtype> (dt);
}

basalt::Device
to_cpp_device (basalt_device_type_t dev_type, size_t idx)
{
  if (dev_type == BASALT_DEVICE_CUDA)
    return basalt::Device::cuda (idx);
  return basalt::Device::cpu ();
}

} /* anonymous namespace */

extern "C" {

const char *
basalt_last_error (void)
{
  return tls_last_error.c_str ();
}

basalt_status_t
basalt_seed_everything (uint64_t seed)
{
  try
    {
      basalt::seed_everything (seed);
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_create (const size_t *shape,
                      size_t ndim,
                      basalt_dtype_t dtype,
                      basalt_device_type_t device_type,
                      size_t device_index,
                      int requires_grad,
                      basalt_tensor_t *out_tensor)
{
  if (out_tensor == nullptr || (ndim > 0 && shape == nullptr))
    {
      tls_last_error = "basalt_tensor_create: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::Shape_Vector sv (shape, shape + ndim);
      basalt::Tensor_Options opts =
        basalt::Tensor_Options ()
          .with_dtype (to_cpp_dtype (dtype))
          .with_device (to_cpp_device (device_type, device_index))
          .with_requires_grad (requires_grad != 0);
      auto *h = new basalt_tensor_handle{basalt::zeros (sv, opts)};
      *out_tensor = h;
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_zeros (const size_t *shape,
                     size_t ndim,
                     basalt_dtype_t dtype,
                     basalt_tensor_t *out_tensor)
{
  return basalt_tensor_create (
    shape, ndim, dtype, BASALT_DEVICE_CPU, 0, 0, out_tensor);
}

basalt_status_t
basalt_tensor_ones (const size_t *shape,
                    size_t ndim,
                    basalt_dtype_t dtype,
                    basalt_tensor_t *out_tensor)
{
  if (out_tensor == nullptr || (ndim > 0 && shape == nullptr))
    {
      tls_last_error = "basalt_tensor_ones: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::Shape_Vector sv (shape, shape + ndim);
      basalt::Tensor_Options opts =
        basalt::Tensor_Options ().with_dtype (to_cpp_dtype (dtype));
      auto *h = new basalt_tensor_handle{basalt::ones (sv, opts)};
      *out_tensor = h;
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_from_data (const void *host_data,
                         const size_t *shape,
                         size_t ndim,
                         basalt_dtype_t dtype,
                         int requires_grad,
                         basalt_tensor_t *out_tensor)
{
  if (host_data == nullptr || out_tensor == nullptr
      || (ndim > 0 && shape == nullptr))
    {
      tls_last_error = "basalt_tensor_from_data: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::Shape_Vector sv (shape, shape + ndim);
      basalt::Tensor_Options opts =
        basalt::Tensor_Options ()
          .with_dtype (to_cpp_dtype (dtype))
          .with_requires_grad (requires_grad != 0);
      auto *h = new basalt_tensor_handle{
        basalt::from_blob (host_data, sv, opts)};
      *out_tensor = h;
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

void
basalt_tensor_destroy (basalt_tensor_t tensor)
{
  delete tensor;
}

size_t
basalt_tensor_ndim (const basalt_tensor_t tensor)
{
  if (tensor == nullptr)
    return 0;
  return tensor->tensor.ndim ();
}

size_t
basalt_tensor_numel (const basalt_tensor_t tensor)
{
  if (tensor == nullptr)
    return 0;
  return tensor->tensor.numel ();
}

basalt_status_t
basalt_tensor_shape (const basalt_tensor_t tensor,
                     size_t *out_dims,
                     size_t max_dims)
{
  if (tensor == nullptr || out_dims == nullptr)
    {
      tls_last_error = "basalt_tensor_shape: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  const auto &sh = tensor->tensor.shape ();
  size_t count = sh.size () < max_dims ? sh.size () : max_dims;
  for (size_t i = 0; i < count; ++i)
    out_dims[i] = sh[i];
  return BASALT_STATUS_OK;
}

basalt_dtype_t
basalt_tensor_dtype (const basalt_tensor_t tensor)
{
  if (tensor == nullptr)
    return BASALT_DTYPE_FLOAT32;
  return static_cast<basalt_dtype_t> (tensor->tensor.dtype ());
}

basalt_status_t
basalt_tensor_set_requires_grad (basalt_tensor_t tensor, int requires_grad)
{
  if (tensor == nullptr)
    {
      tls_last_error = "basalt_tensor_set_requires_grad: null tensor";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      tensor->tensor.set_requires_grad (requires_grad != 0);
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_copy_to_host (const basalt_tensor_t tensor,
                            void *dst_host,
                            size_t max_bytes)
{
  if (tensor == nullptr || dst_host == nullptr)
    {
      tls_last_error = "basalt_tensor_copy_to_host: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::Tensor cpu_t =
        tensor->tensor.to (basalt::Device::cpu ()).contiguous ();
      size_t nbytes =
        cpu_t.numel () * basalt::dtype_size (cpu_t.dtype ());
      if (max_bytes < nbytes)
        {
          tls_last_error = "basalt_tensor_copy_to_host: buffer too small";
          return BASALT_STATUS_INVALID_ARGUMENT;
        }
      if (nbytes > 0)
        std::memcpy (dst_host, cpu_t.raw_data_ptr (), nbytes);
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_item (const basalt_tensor_t tensor, double *out_value)
{
  if (tensor == nullptr || out_value == nullptr)
    {
      tls_last_error = "basalt_tensor_item: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      *out_value = tensor->tensor.item_double ();
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_add (const basalt_tensor_t a,
                   const basalt_tensor_t b,
                   basalt_tensor_t *out_tensor)
{
  if (a == nullptr || b == nullptr || out_tensor == nullptr)
    {
      tls_last_error = "basalt_tensor_add: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      *out_tensor = new basalt_tensor_handle{
        basalt::ops::add (a->tensor, b->tensor)};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_mul (const basalt_tensor_t a,
                   const basalt_tensor_t b,
                   basalt_tensor_t *out_tensor)
{
  if (a == nullptr || b == nullptr || out_tensor == nullptr)
    {
      tls_last_error = "basalt_tensor_mul: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      *out_tensor = new basalt_tensor_handle{
        basalt::ops::mul (a->tensor, b->tensor)};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_matmul (const basalt_tensor_t a,
                      const basalt_tensor_t b,
                      basalt_tensor_t *out_tensor)
{
  if (a == nullptr || b == nullptr || out_tensor == nullptr)
    {
      tls_last_error = "basalt_tensor_matmul: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      *out_tensor = new basalt_tensor_handle{
        basalt::ops::matmul (a->tensor, b->tensor)};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_relu (const basalt_tensor_t x,
                    basalt_tensor_t *out_tensor)
{
  if (x == nullptr || out_tensor == nullptr)
    {
      tls_last_error = "basalt_tensor_relu: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      *out_tensor = new basalt_tensor_handle{basalt::ops::relu (x->tensor)};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_sum (const basalt_tensor_t x,
                   basalt_tensor_t *out_tensor)
{
  if (x == nullptr || out_tensor == nullptr)
    {
      tls_last_error = "basalt_tensor_sum: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      *out_tensor = new basalt_tensor_handle{basalt::ops::sum (x->tensor)};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_mean (const basalt_tensor_t x,
                    basalt_tensor_t *out_tensor)
{
  if (x == nullptr || out_tensor == nullptr)
    {
      tls_last_error = "basalt_tensor_mean: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      *out_tensor = new basalt_tensor_handle{basalt::ops::mean (x->tensor)};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_backward (const basalt_tensor_t loss)
{
  if (loss == nullptr)
    {
      tls_last_error = "basalt_tensor_backward: null tensor";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      loss->tensor.backward ();
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_tensor_grad (const basalt_tensor_t tensor,
                    basalt_tensor_t *out_grad)
{
  if (tensor == nullptr || out_grad == nullptr)
    {
      tls_last_error = "basalt_tensor_grad: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::Tensor g = tensor->tensor.grad ();
      if (!g.defined ())
        {
          *out_grad = nullptr;
          return BASALT_STATUS_OK;
        }
      *out_grad = new basalt_tensor_handle{g};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_mlp_create (size_t in_features,
                   size_t hidden_features,
                   size_t out_features,
                   double learning_rate,
                   basalt_mlp_t *out_mlp)
{
  if (out_mlp == nullptr)
    {
      tls_last_error = "basalt_mlp_create: null out_mlp pointer";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      auto seq = std::make_shared<basalt::nn::Sequential> ();
      seq->add ("fc1", std::make_shared<basalt::nn::Linear> (
                         in_features, hidden_features));
      seq->add ("relu", std::make_shared<basalt::nn::ReLU> ());
      seq->add ("fc2", std::make_shared<basalt::nn::Linear> (
                         hidden_features, out_features));
      auto opt = std::make_unique<basalt::optim::Adam> (
        seq->parameters (), learning_rate);
      auto *h = new basalt_mlp_handle{std::move (seq), std::move (opt)};
      *out_mlp = h;
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

void
basalt_mlp_destroy (basalt_mlp_t mlp)
{
  delete mlp;
}

basalt_status_t
basalt_mlp_forward (basalt_mlp_t mlp,
                    const basalt_tensor_t input,
                    basalt_tensor_t *out_pred)
{
  if (mlp == nullptr || input == nullptr || out_pred == nullptr)
    {
      tls_last_error = "basalt_mlp_forward: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::Tensor y = mlp->model->forward (input->tensor);
      *out_pred = new basalt_tensor_handle{std::move (y)};
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_mlp_train_step (basalt_mlp_t mlp,
                       const basalt_tensor_t inputs,
                       const basalt_tensor_t targets,
                       double *out_loss)
{
  if (mlp == nullptr || inputs == nullptr || targets == nullptr)
    {
      tls_last_error = "basalt_mlp_train_step: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      mlp->optimizer->zero_grad ();
      basalt::Tensor pred = mlp->model->forward (inputs->tensor);
      basalt::Tensor loss = basalt::loss::mse_loss (pred, targets->tensor);
      double lval = loss.item_double ();
      loss.backward ();
      mlp->optimizer->step ();
      if (out_loss != nullptr)
        *out_loss = lval;
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_mlp_save (basalt_mlp_t mlp, const char *path)
{
  if (mlp == nullptr || path == nullptr)
    {
      tls_last_error = "basalt_mlp_save: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::io::save_model (path, *mlp->model);
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

basalt_status_t
basalt_mlp_load (basalt_mlp_t mlp, const char *path)
{
  if (mlp == nullptr || path == nullptr)
    {
      tls_last_error = "basalt_mlp_load: null pointer argument";
      return BASALT_STATUS_INVALID_ARGUMENT;
    }
  try
    {
      basalt::io::load_model (path, *mlp->model);
      return BASALT_STATUS_OK;
    }
  catch (...)
    {
      return translate_exception ();
    }
}

} /* extern "C" */
