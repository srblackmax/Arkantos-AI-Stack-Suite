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

#ifndef BASALT_DATA_HPP
#define BASALT_DATA_HPP

/* Data pipeline abstractions: Dataset, Tensor_Dataset, Tabular_Dataset
   (bridging Bellows data_table and preprocessors), POSIX memory-mapped
   Mmap_Dataset, composable Transforms, Collate functions, and deterministic
   multi-threaded prefetching Data_Loader.  */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/tensor.hpp>

namespace basalt::data
{

/* A single sample or collated mini-batch pairing input features and
   targets.  */
struct BASALT_API Batch
{
  Tensor inputs;
  Tensor targets;
};

using Transform_Fn = std::function<Batch (const Batch &)>;
using Collate_Fn = std::function<Batch (const std::vector<Batch> &)>;

/* Default collate function that stacks individual sample tensors along
   leading batch dimension 0.  */
BASALT_API Batch
default_collate (const std::vector<Batch> &samples);

/* Composable sample/batch transformation pipeline.  */
class BASALT_API Transform
{
public:
  Transform (void) = default;
  virtual ~Transform (void) noexcept;

  virtual Batch
  apply (const Batch &sample) const = 0;

  Batch
  operator() (const Batch &sample) const
  {
    return apply (sample);
  }
};

/* Standardize input features by subtracting MEAN and dividing by STD.  */
class BASALT_API Normalize_Transform : public Transform
{
public:
  Normalize_Transform (double mean, double std_dev);

  Batch
  apply (const Batch &sample) const override;

private:
  double mean_;
  double inv_std_;
};

/* Chain multiple Transform instances sequentially.  */
class BASALT_API Compose_Transform : public Transform
{
public:
  explicit Compose_Transform (
    std::vector<std::shared_ptr<Transform>> transforms);

  Batch
  apply (const Batch &sample) const override;

private:
  std::vector<std::shared_ptr<Transform>> transforms_;
};

/* Abstract random-access Dataset interface.  */
class BASALT_API Dataset
{
public:
  Dataset (void) = default;
  virtual ~Dataset (void) noexcept;

  virtual size_t
  size (void) const noexcept = 0;

  virtual Batch
  get (size_t index) const = 0;
};

/* In-memory dataset wrapping aligned input and target Tensors.  */
class BASALT_API Tensor_Dataset : public Dataset
{
public:
  Tensor_Dataset (Tensor inputs, Tensor targets = Tensor ());

  size_t
  size (void) const noexcept override
  {
    return num_samples_;
  }

  Batch
  get (size_t index) const override;

private:
  Tensor inputs_;
  Tensor targets_;
  size_t num_samples_;
};

using TensorDataset = Tensor_Dataset;

/* Tabular dataset backed by bellows::data_table with optional Bellows
   z-score moment normalization.  */
class BASALT_API Tabular_Dataset : public Dataset
{
public:
  Tabular_Dataset (const bellows::data_table &features,
                   const std::vector<double> &targets,
                   bool standardize_with_bellows = false,
                   Dtype dtype = Dtype::Float32);

  size_t
  size (void) const noexcept override
  {
    return inputs_.defined () ? inputs_.size (0) : 0;
  }

  Batch
  get (size_t index) const override;

private:
  Tensor inputs_;
  Tensor targets_;
};

using TabularDataset = Tabular_Dataset;

/* Zero-copy POSIX memory-mapped binary dataset reader.
   Reads fixed-stride float32/float64 sample records directly from an OS
   virtual memory mapping without loading the entire file onto the heap.  */
class BASALT_API Mmap_Dataset : public Dataset
{
public:
  explicit Mmap_Dataset (const std::string &path);
  ~Mmap_Dataset (void) noexcept override;

  Mmap_Dataset (const Mmap_Dataset &) = delete;
  Mmap_Dataset &operator= (const Mmap_Dataset &) = delete;

  size_t
  size (void) const noexcept override
  {
    return num_samples_;
  }

  Batch
  get (size_t index) const override;

  static void
  write_binary_file (const std::string &path,
                     const Tensor &inputs,
                     const Tensor &targets);

private:
  int fd_ = -1;
  void *mapped_ptr_ = nullptr;
  size_t mapped_bytes_ = 0;
  size_t num_samples_ = 0;
  Shape_Vector input_sample_shape_;
  Shape_Vector target_sample_shape_;
  size_t input_elems_ = 0;
  size_t target_elems_ = 0;
  size_t header_bytes_ = 0;
};

using MmapDataset = Mmap_Dataset;

/* Deterministic, multi-threaded prefetching Data_Loader.
   Guarantees identical batch ordering across thread counts for any given
   seed and epoch.  */
class BASALT_API Data_Loader
{
public:
  Data_Loader (std::shared_ptr<Dataset> dataset,
               size_t batch_size = 32,
               bool shuffle = false,
               size_t num_workers = 0,
               size_t prefetch_factor = 2,
               bool drop_last = false,
               uint64_t seed = 0x853C49E6748FEA9BULL,
               Transform_Fn transform = nullptr,
               Collate_Fn collate = default_collate);

  size_t
  num_batches (void) const noexcept;

  size_t
  batch_size (void) const noexcept
  {
    return batch_size_;
  }

  void
  set_epoch (size_t epoch) noexcept
  {
    epoch_ = epoch;
  }

  /* Materialize or stream all batches for the current epoch using the
     configured worker pool and prefetch queue, then increment epoch.  */
  std::vector<Batch>
  collect_epoch (void);

  /* Execute FN(batch_idx, batch) for each prefetched batch in the epoch.  */
  void
  for_each_batch (const std::function<void (size_t, Batch &)> &fn);

private:
  std::shared_ptr<Dataset> dataset_;
  size_t batch_size_;
  bool shuffle_;
  size_t num_workers_;
  size_t prefetch_factor_;
  bool drop_last_;
  uint64_t seed_;
  size_t epoch_ = 0;
  Transform_Fn transform_;
  Collate_Fn collate_;
};

using DataLoader = Data_Loader;

} /* namespace basalt::data */

#endif /* BASALT_DATA_HPP */
