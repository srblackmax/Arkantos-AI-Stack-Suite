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

#include "config.h"
#include <basalt/data.hpp>
#include <basalt/error.hpp>
#include <basalt/ops.hpp>
#include <basalt/random.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/preprocess/scaling.hpp>
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <queue>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace basalt::data
{

Batch
default_collate (const std::vector<Batch> &samples)
{
  BASALT_ASSERT_ARG (!samples.empty (),
                     "default_collate() requires non-empty sample list");
  std::vector<Tensor> inputs;
  std::vector<Tensor> targets;
  inputs.reserve (samples.size ());
  targets.reserve (samples.size ());

  bool has_targets = samples[0].targets.defined ();
  for (const Batch &s : samples)
    {
      inputs.push_back (s.inputs);
      if (has_targets && s.targets.defined ())
        {
          targets.push_back (s.targets);
        }
    }

  Batch out;
  out.inputs = ops::stack (inputs, 0);
  if (has_targets && targets.size () == samples.size ())
    {
      Tensor stacked_t = ops::stack (targets, 0);
      if (stacked_t.ndim () == 2 && stacked_t.size (1) == 1
          && !is_floating_point (stacked_t.dtype ()))
        {
          stacked_t = stacked_t.squeeze (1);
        }
      out.targets = stacked_t;
    }
  return out;
}

Transform::~Transform (void) noexcept = default;

Normalize_Transform::Normalize_Transform (double mean, double std_dev)
  : mean_ (mean),
    inv_std_ (1.0 / std_dev)
{
  BASALT_ASSERT_ARG (std_dev > 0.0,
                     "Normalize_Transform requires std_dev > 0");
}

Batch
Normalize_Transform::apply (const Batch &sample) const
{
  Batch out;
  out.inputs = ops::mul_scalar (ops::sub_scalar (sample.inputs, mean_),
                                inv_std_);
  out.targets = sample.targets;
  return out;
}

Compose_Transform::Compose_Transform (
  std::vector<std::shared_ptr<Transform>> transforms)
  : transforms_ (std::move (transforms))
{
}

Batch
Compose_Transform::apply (const Batch &sample) const
{
  Batch cur = sample;
  for (const auto &t : transforms_)
    {
      if (t != nullptr)
        {
          cur = t->apply (cur);
        }
    }
  return cur;
}

Dataset::~Dataset (void) noexcept = default;

Tensor_Dataset::Tensor_Dataset (Tensor inputs, Tensor targets)
  : inputs_ (std::move (inputs)),
    targets_ (std::move (targets)),
    num_samples_ (0)
{
  BASALT_ASSERT_ARG (inputs_.defined () && inputs_.ndim () >= 1,
                     "Tensor_Dataset requires defined inputs with ndim >= 1");
  num_samples_ = inputs_.size (0);
  if (targets_.defined ())
    {
      BASALT_ASSERT_SHAPE (targets_.ndim () >= 1
                           && targets_.size (0) == num_samples_,
                           "Tensor_Dataset inputs/targets length mismatch");
    }
}

Batch
Tensor_Dataset::get (size_t index) const
{
  BASALT_ASSERT_ARG (index < num_samples_,
                     "Tensor_Dataset index out of bounds");
  Batch b;
  b.inputs = inputs_.select (0, static_cast<ptrdiff_t> (index));
  if (targets_.defined ())
    {
      b.targets = targets_.select (0, static_cast<ptrdiff_t> (index));
    }
  return b;
}

Tabular_Dataset::Tabular_Dataset (const bellows::data_table &features,
                                  const std::vector<double> &targets,
                                  bool standardize_with_bellows,
                                  Dtype dtype)
{
  BASALT_ASSERT_SHAPE (features.row_count () == targets.size (),
                       "Tabular_Dataset row count and target length mismatch");
  bellows::data_table proc = features;
  if (standardize_with_bellows && features.row_count () > 0)
    {
      bellows::moment_normaliser norm;
      norm.calibrate (features);
      proc = norm.project (features);
    }
  inputs_ = from_bellows_table (proc, Tensor_Options ().with_dtype (dtype));
  targets_ = Tensor ({targets.size (), 1},
                     Tensor_Options ().with_dtype (dtype));
  for (size_t i = 0; i < targets.size (); ++i)
    {
      targets_.write_scalar_at (i, targets[i]);
    }
}

Batch
Tabular_Dataset::get (size_t index) const
{
  BASALT_ASSERT_ARG (index < size (), "Tabular_Dataset index out of bounds");
  Batch b;
  b.inputs = inputs_.select (0, static_cast<ptrdiff_t> (index));
  b.targets = targets_.select (0, static_cast<ptrdiff_t> (index));
  return b;
}

/* Mmap_Dataset binary format:
   [uint32_t magic = 0x424D4D50 ("BMMP")]
   [uint64_t num_samples]
   [uint64_t in_ndim] [uint64_t in_shape[in_ndim]]
   [uint64_t tgt_ndim] [uint64_t tgt_shape[tgt_ndim]]
   followed by num_samples * (in_elems + tgt_elems) float64 values.  */

namespace
{
constexpr uint32_t k_mmap_magic = 0x424D4D50U;
}

void
Mmap_Dataset::write_binary_file (const std::string &path,
                                 const Tensor &inputs,
                                 const Tensor &targets)
{
  BASALT_ASSERT_ARG (inputs.defined () && targets.defined ()
                     && inputs.size (0) == targets.size (0),
                     "Mmap_Dataset::write_binary_file invalid tensors");
  std::ofstream ofs (path, std::ios::binary | std::ios::trunc);
  if (!ofs)
    {
      BASALT_THROW (IO_Error, "cannot open file for writing: " + path);
    }

  uint32_t magic = k_mmap_magic;
  uint64_t n = inputs.size (0);
  uint64_t in_ndim = inputs.ndim () - 1;
  uint64_t tgt_ndim = targets.ndim () - 1;

  ofs.write (reinterpret_cast<const char *> (&magic), sizeof (magic));
  ofs.write (reinterpret_cast<const char *> (&n), sizeof (n));
  ofs.write (reinterpret_cast<const char *> (&in_ndim), sizeof (in_ndim));
  for (size_t i = 1; i < inputs.ndim (); ++i)
    {
      uint64_t dim_v = inputs.shape ()[i];
      ofs.write (reinterpret_cast<const char *> (&dim_v), sizeof (dim_v));
    }
  ofs.write (reinterpret_cast<const char *> (&tgt_ndim), sizeof (tgt_ndim));
  for (size_t i = 1; i < targets.ndim (); ++i)
    {
      uint64_t dim_v = targets.shape ()[i];
      ofs.write (reinterpret_cast<const char *> (&dim_v), sizeof (dim_v));
    }

  size_t in_stride = inputs.numel () / n;
  size_t tgt_stride = targets.numel () / n;
  for (size_t s = 0; s < n; ++s)
    {
      for (size_t j = 0; j < in_stride; ++j)
        {
          double v = inputs.read_scalar_at (s * in_stride + j);
          ofs.write (reinterpret_cast<const char *> (&v), sizeof (v));
        }
      for (size_t j = 0; j < tgt_stride; ++j)
        {
          double v = targets.read_scalar_at (s * tgt_stride + j);
          ofs.write (reinterpret_cast<const char *> (&v), sizeof (v));
        }
    }
  if (!ofs.good ())
    {
      BASALT_THROW (IO_Error, "failed writing binary mmap file: " + path);
    }
}

Mmap_Dataset::Mmap_Dataset (const std::string &path)
{
  fd_ = ::open (path.c_str (), O_RDONLY);
  if (fd_ < 0)
    {
      BASALT_THROW (IO_Error, "cannot open mmap dataset file: " + path);
    }
  struct stat st{};
  if (::fstat (fd_, &st) != 0 || st.st_size < 28)
    {
      ::close (fd_);
      fd_ = -1;
      BASALT_THROW (IO_Error, "invalid or truncated mmap file: " + path);
    }
  mapped_bytes_ = static_cast<size_t> (st.st_size);
  mapped_ptr_ = ::mmap (nullptr, mapped_bytes_, PROT_READ, MAP_PRIVATE,
                        fd_, 0);
  if (mapped_ptr_ == MAP_FAILED)
    {
      mapped_ptr_ = nullptr;
      ::close (fd_);
      fd_ = -1;
      BASALT_THROW (IO_Error, "POSIX mmap failed for: " + path);
    }

  const uint8_t *ptr = static_cast<const uint8_t *> (mapped_ptr_);
  size_t pos = 0;
  auto fail_close = [&] (const std::string &msg)
  {
    ::munmap (mapped_ptr_, mapped_bytes_);
    mapped_ptr_ = nullptr;
    ::close (fd_);
    fd_ = -1;
    BASALT_THROW (IO_Error, msg + ": " + path);
  };

  uint32_t magic = 0;
  std::memcpy (&magic, ptr + pos, sizeof (magic));
  pos += sizeof (magic);
  if (magic != k_mmap_magic)
    {
      fail_close ("corrupt mmap magic in");
    }

  uint64_t n = 0;
  uint64_t in_ndim = 0;
  std::memcpy (&n, ptr + pos, sizeof (n));
  pos += sizeof (n);
  std::memcpy (&in_ndim, ptr + pos, sizeof (in_ndim));
  pos += sizeof (in_ndim);
  num_samples_ = static_cast<size_t> (n);

  if (in_ndim > 32 || pos + in_ndim * sizeof (uint64_t) > mapped_bytes_)
    {
      fail_close ("truncated mmap input shape header in");
    }

  input_elems_ = 1;
  for (size_t i = 0; i < in_ndim; ++i)
    {
      uint64_t d = 0;
      std::memcpy (&d, ptr + pos, sizeof (d));
      pos += sizeof (d);
      input_sample_shape_.push_back (static_cast<size_t> (d));
      input_elems_ *= static_cast<size_t> (d);
    }
  if (input_sample_shape_.empty ())
    {
      input_sample_shape_.push_back (1);
    }

  if (pos + sizeof (uint64_t) > mapped_bytes_)
    {
      fail_close ("truncated mmap target rank header in");
    }

  uint64_t tgt_ndim = 0;
  std::memcpy (&tgt_ndim, ptr + pos, sizeof (tgt_ndim));
  pos += sizeof (tgt_ndim);
  if (tgt_ndim > 32 || pos + tgt_ndim * sizeof (uint64_t) > mapped_bytes_)
    {
      fail_close ("truncated mmap target shape header in");
    }

  target_elems_ = 1;
  for (size_t i = 0; i < tgt_ndim; ++i)
    {
      uint64_t d = 0;
      std::memcpy (&d, ptr + pos, sizeof (d));
      pos += sizeof (d);
      target_sample_shape_.push_back (static_cast<size_t> (d));
      target_elems_ *= static_cast<size_t> (d);
    }
  if (target_sample_shape_.empty ())
    {
      target_sample_shape_.push_back (1);
    }
  header_bytes_ = pos;
  size_t record_bytes = (input_elems_ + target_elems_) * sizeof (double);
  if (num_samples_ > 0
      && (record_bytes == 0
          || num_samples_ > (mapped_bytes_ - header_bytes_) / record_bytes))
    {
      fail_close ("truncated mmap sample payload in");
    }
}

Mmap_Dataset::~Mmap_Dataset (void) noexcept
{
  if (mapped_ptr_ != nullptr)
    {
      ::munmap (mapped_ptr_, mapped_bytes_);
      mapped_ptr_ = nullptr;
    }
  if (fd_ >= 0)
    {
      ::close (fd_);
      fd_ = -1;
    }
}

Batch
Mmap_Dataset::get (size_t index) const
{
  BASALT_ASSERT_ARG (index < num_samples_,
                     "Mmap_Dataset index out of bounds");
  size_t record_doubles = input_elems_ + target_elems_;
  const uint8_t *base = static_cast<const uint8_t *> (mapped_ptr_)
                        + header_bytes_
                        + index * record_doubles * sizeof (double);

  Batch b;
  b.inputs = Tensor (input_sample_shape_,
                     Tensor_Options ().with_dtype (Dtype::Float32));
  for (size_t i = 0; i < input_elems_; ++i)
    {
      double v = 0.0;
      std::memcpy (&v, base + i * sizeof (double), sizeof (double));
      b.inputs.write_scalar_at (i, v);
    }

  b.targets = Tensor (target_sample_shape_,
                      Tensor_Options ().with_dtype (Dtype::Float32));
  for (size_t i = 0; i < target_elems_; ++i)
    {
      double v = 0.0;
      std::memcpy (&v, base + (input_elems_ + i) * sizeof (double),
                   sizeof (double));
      b.targets.write_scalar_at (i, v);
    }
  return b;
}

/* Data_Loader implementation.  */

Data_Loader::Data_Loader (std::shared_ptr<Dataset> dataset,
                          size_t batch_size,
                          bool shuffle,
                          size_t num_workers,
                          size_t prefetch_factor,
                          bool drop_last,
                          uint64_t seed,
                          Transform_Fn transform,
                          Collate_Fn collate)
  : dataset_ (std::move (dataset)),
    batch_size_ (batch_size),
    shuffle_ (shuffle),
    num_workers_ (num_workers),
    prefetch_factor_ (std::max<size_t> (1, prefetch_factor)),
    drop_last_ (drop_last),
    seed_ (seed),
    epoch_ (0),
    transform_ (std::move (transform)),
    collate_ (collate ? std::move (collate) : default_collate)
{
  BASALT_ASSERT_ARG (dataset_ != nullptr && batch_size_ >= 1,
                     "Data_Loader requires valid dataset and batch_size >= 1");
}

size_t
Data_Loader::num_batches (void) const noexcept
{
  size_t n = dataset_->size ();
  if (n == 0)
    {
      return 0;
    }
  if (drop_last_)
    {
      return n / batch_size_;
    }
  return (n + batch_size_ - 1) / batch_size_;
}

void
Data_Loader::for_each_batch (
  const std::function<void (size_t, Batch &)> &fn)
{
  size_t total_samples = dataset_->size ();
  size_t total_batches = num_batches ();
  if (total_batches == 0)
    {
      ++epoch_;
      return;
    }

  /* Deterministic index permutation for the current epoch.  */
  std::vector<size_t> indices;
  if (shuffle_)
    {
      Random_Generator gen (seed_ ^ (static_cast<uint64_t> (epoch_ + 1U)
                                     * 0x9E3779B97F4A7C15ULL));
      indices = gen.permutation (total_samples);
    }
  else
    {
      indices.resize (total_samples);
      for (size_t i = 0; i < total_samples; ++i)
        {
          indices[i] = i;
        }
    }
  ++epoch_;

  auto build_batch = [&] (size_t b_idx) -> Batch
    {
      size_t start = b_idx * batch_size_;
      size_t end = std::min (start + batch_size_, total_samples);
      std::vector<Batch> samples;
      samples.reserve (end - start);
      for (size_t i = start; i < end; ++i)
        {
          Batch s = dataset_->get (indices[i]);
          if (transform_)
            {
              s = transform_ (s);
            }
          samples.push_back (std::move (s));
        }
      return collate_ (samples);
    };

  if (num_workers_ == 0)
    {
      for (size_t b_idx = 0; b_idx < total_batches; ++b_idx)
        {
          Batch batch = build_batch (b_idx);
          fn (b_idx, batch);
        }
      return;
    }

  /* Multi-threaded bounded prefetch queue preserving exact batch order.  */
  size_t max_in_flight = num_workers_ * prefetch_factor_;
  std::mutex mtx;
  std::condition_variable cv_prod;
  std::condition_variable cv_cons;
  std::map<size_t, Batch> ready_batches;
  size_t next_to_assign = 0;
  size_t next_to_consume = 0;
  std::exception_ptr worker_exc = nullptr;

  auto worker_loop = [&] (void)
    {
      while (true)
        {
          size_t my_batch = 0;
          {
            std::unique_lock<std::mutex> lock (mtx);
            cv_prod.wait (lock, [&] ()
              {
                return worker_exc != nullptr
                       || next_to_assign >= total_batches
                       || (next_to_assign - next_to_consume) < max_in_flight;
              });
            if (worker_exc != nullptr || next_to_assign >= total_batches)
              {
                return;
              }
            my_batch = next_to_assign++;
          }

          try
            {
              Batch b = build_batch (my_batch);
              std::lock_guard<std::mutex> lock (mtx);
              ready_batches[my_batch] = std::move (b);
              cv_cons.notify_all ();
            }
          catch (...)
            {
              std::lock_guard<std::mutex> lock (mtx);
              if (worker_exc == nullptr)
                {
                  worker_exc = std::current_exception ();
                }
              cv_cons.notify_all ();
              cv_prod.notify_all ();
              return;
            }
        }
    };

  std::vector<std::thread> workers;
  workers.reserve (num_workers_);
  for (size_t w = 0; w < num_workers_; ++w)
    {
      workers.emplace_back (worker_loop);
    }

  while (next_to_consume < total_batches)
    {
      Batch batch;
      {
        std::unique_lock<std::mutex> lock (mtx);
        cv_cons.wait (lock, [&] ()
          {
            return worker_exc != nullptr
                   || ready_batches.find (next_to_consume)
                      != ready_batches.end ();
          });
        if (worker_exc != nullptr)
          {
            break;
          }
        auto it = ready_batches.find (next_to_consume);
        batch = std::move (it->second);
        ready_batches.erase (it);
        ++next_to_consume;
        cv_prod.notify_all ();
      }
      fn (next_to_consume - 1, batch);
    }

  for (auto &t : workers)
    {
      if (t.joinable ())
        {
          t.join ();
        }
    }
  if (worker_exc != nullptr)
    {
      std::rethrow_exception (worker_exc);
    }
}

std::vector<Batch>
Data_Loader::collect_epoch (void)
{
  std::vector<Batch> batches;
  batches.reserve (num_batches ());
  for_each_batch ([&] (size_t, Batch &b)
    {
      batches.push_back (std::move (b));
    });
  return batches;
}

} /* namespace basalt::data */
