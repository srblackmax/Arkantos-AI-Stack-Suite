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

#ifndef YANTRA_INDEX_HPP
#define YANTRA_INDEX_HPP

#include <yantra/error.hpp>
#include <yantra/gpu.hpp>
#include <yantra/observability.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace yantra::inline v1
{

/**
 * @brief Scalar metadata attached to an indexed vector for hybrid search.
 */
struct VectorMetadata
{
  uint64_t id = 0U;
  uint64_t timestamp = 0U;
  int64_t category = 0;
  double score_attr = 0.0;
};

/**
 * @brief Scalar metadata filter predicate supporting automatic pre-filter vs
 * post-filter execution based on estimated selectivity.
 */
struct MetadataFilter
{
  uint64_t min_timestamp = 0U;
  uint64_t max_timestamp = UINT64_MAX;
  int64_t required_category = -1; /* -1 = any category */
  std::function<bool (const VectorMetadata &)> custom_predicate;

  [[nodiscard]] auto
  matches (const VectorMetadata &meta) const noexcept -> bool
  {
    if (meta.timestamp < min_timestamp || meta.timestamp > max_timestamp)
      {
        return false;
      }
    if (required_category >= 0 && meta.category != required_category)
      {
        return false;
      }
    if (custom_predicate)
      {
        return custom_predicate (meta);
      }
    return true;
  }
};

/**
 * @brief Exact brute-force vector index accelerated with runtime-dispatched
 * AVX-512 / AVX2+FMA SIMD kernels and double-precision ground-truth mode.
 * @note Thread-safety: Concurrent readers are safe; writers are serialized.
 */
class FlatIndex
{
public:
  explicit FlatIndex (std::size_t dim) noexcept;
  FlatIndex (const FlatIndex &) = delete;
  auto operator= (const FlatIndex &) -> FlatIndex & = delete;
  FlatIndex (FlatIndex &&other) noexcept;
  auto operator= (FlatIndex &&other) noexcept -> FlatIndex &;
  ~FlatIndex () noexcept;

  /**
   * @brief Incrementally adds a single vector and optional metadata.
   * @param vec Vector span of length dimension().
   * @param id External vector identifier.
   * @param meta Optional scalar metadata for filtered search.
   * @returns Result<void> or IndexErrorCode::DimensionMismatch.
   */
  [[nodiscard]] auto
  add (std::span<const float> vec, uint64_t id,
       VectorMetadata meta = {}) noexcept -> Result<void>;

  /**
   * @brief Adds a contiguous batch of vectors [num_vectors * dim].
   * @param vectors Contiguous fp32 vectors.
   * @param num_vectors Number of vectors in the batch.
   * @returns Result<void> or IndexErrorCode::DimensionMismatch.
   */
  [[nodiscard]] auto
  add_batch (std::span<const float> vectors,
             std::size_t num_vectors) noexcept -> Result<void>;

  /**
   * @brief Performs exact k-NN search using SIMD-accelerated fp32 L2 distance.
   * @param query Query vector of length dimension().
   * @param k Number of nearest neighbors to return.
   * @returns Result<std::vector<NeighborResult>> sorted by ascending distance.
   */
  [[nodiscard]] auto
  search (std::span<const float> query,
          std::size_t k) const noexcept -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Performs exact k-NN search using double-precision fp64 L2
   * accumulation for benchmark recall ground-truth verification.
   * @param query Query vector of length dimension().
   * @param k Number of nearest neighbors to return.
   * @returns Result<std::vector<NeighborResult>> sorted by ascending distance.
   */
  [[nodiscard]] auto
  search_exact_fp64 (std::span<const float> query,
                     std::size_t k) const noexcept
      -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Performs metadata-filtered exact k-NN search.
   * @param query Query vector of length dimension().
   * @param k Number of nearest neighbors to return.
   * @param filter Scalar metadata filter predicate.
   * @returns Result<std::vector<NeighborResult>>.
   */
  [[nodiscard]] auto
  search_filtered (std::span<const float> query, std::size_t k,
                   const MetadataFilter &filter) const noexcept
      -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Queries the number of vectors in the index.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Queries the vector dimensionality.
   */
  [[nodiscard]] auto dimension () const noexcept -> std::size_t;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Construction and search hyperparameters for HnswIndex.
 */
struct HnswParams
{
  std::size_t m = 16U;
  std::size_t ef_construction = 200U;
  std::size_t ef_search = 64U;
};

/**
 * @brief Hierarchical Navigable Small World (HNSW) approximate nearest-
 * neighbor index with AVX-512/AVX2 acceleration, incremental insertion,
 * pre/post metadata filtering, and zero-copy mmap(2) persistence.
 * @pre Vector dimension > 0.
 * @post Memory overhead is <= 1.5x raw vector storage; mmap persistence uses
 * RAII page-cache mapping without loading the entire index into heap RAM.
 * @note Thread-safety: Concurrent search() readers are lock-free / shared-lock
 * safe using per-thread std::pmr scratch arenas; writers acquire unique lock.
 */
class HnswIndex
{
public:
  explicit HnswIndex (std::size_t dim, HnswParams params = {}) noexcept;
  HnswIndex (const HnswIndex &) = delete;
  auto operator= (const HnswIndex &) -> HnswIndex & = delete;
  HnswIndex (HnswIndex &&other) noexcept;
  auto operator= (HnswIndex &&other) noexcept -> HnswIndex &;
  ~HnswIndex () noexcept;

  /**
   * @brief Incrementally inserts a single vector into the HNSW graph without
   * rebuilding existing nodes.
   * @param vec Vector span of length dimension().
   * @param id External identifier.
   * @param meta Optional scalar metadata.
   * @returns Result<void> or IndexErrorCode::DimensionMismatch.
   */
  [[nodiscard]] auto
  add (std::span<const float> vec, uint64_t id,
       VectorMetadata meta = {}) noexcept -> Result<void>;

  /**
   * @brief Builds or extends the HNSW graph over a batch of contiguous vectors
   * [num_vectors * dim] using parallel multi-threaded insertion.
   * @param vectors Contiguous fp32 vector data.
   * @param num_vectors Number of vectors in the batch.
   * @returns Result<void> or IndexErrorCode::DimensionMismatch.
   */
  [[nodiscard]] auto
  add_batch (std::span<const float> vectors,
             std::size_t num_vectors) noexcept -> Result<void>;

  /**
   * @brief Searches for the top-k approximate nearest neighbors of query.
   * @param query Query vector of length dimension().
   * @param k Number of nearest neighbors to return.
   * @param ef_search_override Optional override for efSearch (0 uses default).
   * @returns Result<std::vector<NeighborResult>> sorted by ascending distance.
   */
  [[nodiscard]] auto
  search (std::span<const float> query, std::size_t k,
          std::size_t ef_search_override = 0U) const noexcept
      -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Searches for the top-k nearest neighbors matching a scalar metadata
   * predicate, automatically choosing pre-filtering or post-filtering based on
   * predicate selectivity.
   * @param query Query vector of length dimension().
   * @param k Number of nearest neighbors to return.
   * @param filter Scalar metadata filter predicate.
   * @param ef_search_override Optional override for efSearch.
   * @returns Result<std::vector<NeighborResult>>.
   */
  [[nodiscard]] auto
  search_filtered (std::span<const float> query, std::size_t k,
                   const MetadataFilter &filter,
                   std::size_t ef_search_override = 0U) const noexcept
      -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Persists the HNSW graph and vector storage to a binary file with
   * a CRC32C-protected header suitable for direct mmap(2) loading.
   * @param path Target file path.
   * @returns Result<void> or IndexErrorCode::PersistenceCorruption.
   */
  [[nodiscard]] auto
  save (const std::filesystem::path &path) const noexcept -> Result<void>;

  /**
   * @brief Memory-maps a persisted HNSW index from disk via mmap(2) for fast
   * cold-start loading without copying vectors into heap RAM.
   * @param path Path to persisted HNSW index file.
   * @returns Result<HnswIndex> or IndexErrorCode::PersistenceCorruption.
   */
  [[nodiscard]] static auto
  load_mmap (const std::filesystem::path &path) noexcept -> Result<HnswIndex>;

  /**
   * @brief Exports this HNSW index's vectors and level-0 navigable graph to a
   * CUDA-accelerated GpuIndex handle.
   * @returns Result<GpuIndex> or IndexErrorCode::GpuOutOfMemory.
   */
  [[nodiscard]] auto to_gpu_index () const noexcept -> Result<GpuIndex>;

  /**
   * @brief Queries the number of indexed vectors.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Queries the vector dimensionality.
   */
  [[nodiscard]] auto dimension () const noexcept -> std::size_t;

  /**
   * @brief Queries the number of level-0 neighbors for a given internal node.
   * @param node_index Internal 0-based node index.
   * @returns Number of valid level-0 graph links for node_index.
   */
  [[nodiscard]] auto
  layer0_neighbor_count (std::size_t node_index) const noexcept -> std::size_t;

  /**
   * @brief Queries the level-0 neighbor indices for a given internal node.
   * @param node_index Internal 0-based node index.
   * @returns Span of valid level-0 neighbor node indices.
   */
  [[nodiscard]] auto
  layer0_neighbors (std::size_t node_index) const noexcept
      -> std::span<const uint32_t>;

  /**
   * @brief Queries the maximum layer index in the HNSW hierarchy.
   */
  [[nodiscard]] auto max_level () const noexcept -> std::size_t;

  /**
   * @brief Queries the maximum layer assigned to a given internal node.
   * @param node_index Internal 0-based node index.
   */
  [[nodiscard]] auto
  node_level (std::size_t node_index) const noexcept -> std::size_t;

  /**
   * @brief Queries total memory usage (vectors + graph links) in bytes.
   */
  [[nodiscard]] auto memory_usage_bytes () const noexcept -> std::size_t;

  /**
   * @brief Queries the configured HnswParams.
   */
  [[nodiscard]] auto params () const noexcept -> HnswParams;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Hyperparameters for Inverted File with Product Quantization (IVF-PQ).
 */
struct IvfPqParams
{
  std::size_t nlist = 128U;
  std::size_t m_subspaces = 16U;
  std::size_t nprobe = 12U;
  std::size_t rerank_k = 48U;
};

/**
 * @brief Inverted File with Product Quantization (IVF-PQ) vector index for
 * memory-constrained, sub-millisecond approximate search.
 * @note Thread-safety: Concurrent search() calls are safe; writers lock.
 */
class IvfPqIndex
{
public:
  explicit IvfPqIndex (std::size_t dim, IvfPqParams params = {}) noexcept;
  IvfPqIndex (const IvfPqIndex &) = delete;
  auto operator= (const IvfPqIndex &) -> IvfPqIndex & = delete;
  IvfPqIndex (IvfPqIndex &&other) noexcept;
  auto operator= (IvfPqIndex &&other) noexcept -> IvfPqIndex &;
  ~IvfPqIndex () noexcept;

  /**
   * @brief Trains coarse IVF centroids and PQ codebooks and indexes a batch of
   * vectors [num_vectors * dim].
   * @param vectors Contiguous fp32 vectors.
   * @param num_vectors Number of vectors in the batch.
   * @returns Result<void> or IndexError.
   */
  [[nodiscard]] auto
  build (std::span<const float> vectors,
         std::size_t num_vectors) noexcept -> Result<void>;

  /**
   * @brief Incrementally encodes and inserts a single vector into the trained
   * IVF-PQ index without rebuilding codebooks.
   * @param vec Vector of length dimension().
   * @param id External vector identifier.
   * @returns Result<void> or IndexErrorCode::IndexNotBuilt.
   */
  [[nodiscard]] auto
  add (std::span<const float> vec, uint64_t id) noexcept -> Result<void>;

  /**
   * @brief Searches for the top-k nearest neighbors using Asymmetric Distance
   * Computation (ADC) lookup tables across nprobe inverted lists.
   * @param query Query vector of length dimension().
   * @param k Number of nearest neighbors to return.
   * @param nprobe_override Optional override for nprobe (0 uses default).
   * @returns Result<std::vector<NeighborResult>>.
   */
  [[nodiscard]] auto
  search (std::span<const float> query, std::size_t k,
          std::size_t nprobe_override = 0U) const noexcept
      -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Queries the number of indexed vectors.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Queries the vector dimensionality.
   */
  [[nodiscard]] auto dimension () const noexcept -> std::size_t;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

namespace index
{
using yantra::v1::FlatIndex;
using yantra::v1::GpuIndex;
using yantra::v1::HnswIndex;
using yantra::v1::HnswParams;
using yantra::v1::IvfPqIndex;
using yantra::v1::IvfPqParams;
using yantra::v1::MetadataFilter;
using yantra::v1::NeighborResult;
using yantra::v1::VectorMetadata;
} /* namespace index */

} /* namespace yantra::inline v1 */

#endif /* YANTRA_INDEX_HPP */
