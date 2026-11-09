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

/* Optional pybind11 Python extension module for Basalt Deep Learning.
   Built only when ./configure --enable-python is specified.  */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <basalt/basalt.hpp>

namespace py = pybind11;

PYBIND11_MODULE (_basalt, m)
{
  m.doc () = "Basalt Deep Learning C++20 extension module";
  m.attr ("__version__") = basalt_version_string ();

  py::enum_<basalt::Dtype> (m, "Dtype")
    .value ("Float32", basalt::Dtype::Float32)
    .value ("Float64", basalt::Dtype::Float64)
    .value ("Float16", basalt::Dtype::Float16)
    .value ("BFloat16", basalt::Dtype::BFloat16)
    .value ("Int32", basalt::Dtype::Int32)
    .value ("Int64", basalt::Dtype::Int64)
    .value ("UInt8", basalt::Dtype::UInt8)
    .value ("Bool", basalt::Dtype::Bool);

  py::class_<basalt::Device> (m, "Device")
    .def_static ("cpu", &basalt::Device::cpu)
    .def_static ("cuda", &basalt::Device::cuda, py::arg ("index") = 0)
    .def ("is_cpu", &basalt::Device::is_cpu)
    .def ("is_cuda", &basalt::Device::is_cuda)
    .def ("to_string", &basalt::Device::to_string);

  py::class_<basalt::Tensor> (m, "Tensor")
    .def (py::init<> ())
    .def ("ndim", &basalt::Tensor::ndim)
    .def ("numel", &basalt::Tensor::numel)
    .def ("shape", &basalt::Tensor::shape)
    .def ("dtype", &basalt::Tensor::dtype)
    .def ("requires_grad", &basalt::Tensor::requires_grad)
    .def ("set_requires_grad", &basalt::Tensor::set_requires_grad)
    .def ("backward",
          [] (const basalt::Tensor &t)
          {
            t.backward ();
          })
    .def ("grad", &basalt::Tensor::grad)
    .def ("item", &basalt::Tensor::item_double)
    .def ("to_list", &basalt::Tensor::to_vector_double)
    .def ("__repr__", &basalt::Tensor::to_string);

  m.def ("seed_everything", &basalt::seed_everything);
  m.def ("zeros",
         [] (const basalt::Shape_Vector &shape)
         {
           return basalt::zeros (shape);
         });
  m.def ("ones",
         [] (const basalt::Shape_Vector &shape)
         {
           return basalt::ones (shape);
         });
  m.def ("randn",
         [] (const basalt::Shape_Vector &shape)
         {
           return basalt::randn (shape);
         });
  m.def ("add", &basalt::ops::add);
  m.def ("mul", &basalt::ops::mul);
  m.def ("matmul", &basalt::ops::matmul);
  m.def ("relu", &basalt::ops::relu);
  m.def ("mse_loss", &basalt::loss::mse_loss,
         py::arg ("input"), py::arg ("target"),
         py::arg ("reduction") = basalt::loss::Reduction::Mean);

  py::class_<basalt::nn::Module, std::shared_ptr<basalt::nn::Module>> (
    m, "Module")
    .def ("forward", &basalt::nn::Module::forward)
    .def ("parameters", &basalt::nn::Module::parameters,
          py::arg ("recurse") = true)
    .def ("zero_grad", &basalt::nn::Module::zero_grad);

  py::class_<basalt::nn::Linear, basalt::nn::Module,
             std::shared_ptr<basalt::nn::Linear>> (m, "Linear")
    .def (py::init<size_t, size_t, bool> (),
          py::arg ("in_features"),
          py::arg ("out_features"),
          py::arg ("bias") = true);

  py::class_<basalt::nn::Sequential, basalt::nn::Module,
             std::shared_ptr<basalt::nn::Sequential>> (m, "Sequential")
    .def (py::init<> ())
    .def ("add",
          py::overload_cast<std::shared_ptr<basalt::nn::Module>> (
            &basalt::nn::Sequential::add));
}
