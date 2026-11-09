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

#include <yantra/yantra.hpp>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace
{

void
test_tracking_and_wal ()
{
  const fs::path tmp_dir = fs::temp_directory_path () / "yantra_unit_track";
  std::error_code ec;
  fs::remove_all (tmp_dir, ec);

  auto store_res = yantra::ExperimentStore::open (tmp_dir);
  assert (store_res.has_value ());
  auto &store = *store_res;
  assert (store.run_count () == 0U);

  /* Boundary: query over zero runs returns empty vector.  */
  yantra::QueryFilter empty_q{};
  auto q0 = store.query (empty_q);
  assert (q0.has_value () && q0->empty ());

  /* Error path: invalid query filter range.  */
  yantra::QueryFilter bad_q{};
  bad_q.has_metric_filter = true;
  bad_q.metric_key = "loss";
  bad_q.metric_min = 10.0;
  bad_q.metric_max = 1.0;
  auto q_bad = store.query (bad_q);
  assert (!q_bad.has_value () && q_bad.error ().is_tracking_error ());

  auto run_res = store.create_run ("exp_alpha");
  assert (run_res.has_value ());
  auto &run = *run_res;
  assert (run.state () == yantra::RunState::Created);
  assert (run.log_param ("lr", "0.001").has_value ());
  assert (run.log_tag ("team", "vision").has_value ());

  /* Logging first metric transitions run to RUNNING and locks parameters.  */
  assert (run.log_metric (1U, "loss", 0.75).has_value ());
  assert (run.state () == yantra::RunState::Running);

  auto immut_err = run.log_param ("batch_size", "64");
  assert (!immut_err.has_value () && immut_err.error ().is_tracking_error ());

  yantra::HistogramSummary hist{};
  hist.count = 100U;
  hist.min_value = -1.0;
  hist.max_value = 1.0;
  hist.sum_value = 0.25;
  assert (run.log_histogram (2U, "grad_norm", hist).has_value ());

  const std::array<std::byte, 8> blob_bytes
      = { std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 },
          std::byte{ 5 }, std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 } };
  auto blob_hash = run.log_blob (3U, "weights_delta", blob_bytes);
  assert (blob_hash.has_value () && blob_hash->size () == 64U);

  auto loaded_blob = store.load_artifact (*blob_hash);
  assert (loaded_blob.has_value () && loaded_blob->size () == 8U);

  assert (run.finish ().has_value ());
  assert (run.state () == yantra::RunState::Completed);

  auto comp_res = store.compact ();
  assert (comp_res.has_value () && *comp_res >= 1U);

  yantra::QueryFilter filter{};
  filter.required_params["lr"] = "0.001";
  filter.has_metric_filter = true;
  filter.metric_key = "loss";
  filter.metric_min = 0.5;
  filter.metric_max = 1.0;
  auto q_matches = store.query (filter);
  assert (q_matches.has_value () && q_matches->size () == 1U);
  assert ((*q_matches)[0].run_id == "exp_alpha");

  /* Test WAL corruption detection.  */
  const fs::path bad_wal = tmp_dir / "corrupt_wal.bin";
  {
    auto wal_res = yantra::WalEngine::open (bad_wal, true);
    assert (wal_res.has_value ());
    yantra::WalRecord rec{};
    rec.scalar_value = 42.0;
    assert (wal_res->append_record (rec, true).has_value ());
  }
  {
    std::fstream f (bad_wal, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp (72);
    const char bad_byte = static_cast<char> (0xFF);
    f.write (&bad_byte, 1);
  }
  auto rec_corrupt = yantra::WalEngine::recover (bad_wal);
  assert (!rec_corrupt.has_value ()
          && rec_corrupt.error ().is_tracking_error ());

  fs::remove_all (tmp_dir, ec);
}

void
test_model_registry ()
{
  const fs::path tmp_dir = fs::temp_directory_path () / "yantra_unit_reg";
  std::error_code ec;
  fs::remove_all (tmp_dir, ec);

  auto reg_res = yantra::ModelRegistry::open (tmp_dir);
  assert (reg_res.has_value ());
  auto &reg = *reg_res;

  std::vector<std::byte> ckpt (1024U, std::byte{ 0x42 });
  const char magic[4] = { 'Y', 'M', 'C', 'P' };
  const uint32_t tcount = 12U;
  std::memcpy (ckpt.data (), magic, 4U);
  std::memcpy (ckpt.data () + 4U, &tcount, 4U);

  yantra::ModelLineage lin1{};
  lin1.run_id = "run_1";
  lin1.dataset_hash = "ds_hash_01";
  lin1.code_commit_hash = "abc1234";

  auto v1_res
      = reg.register_version ("basalt-resnet18", "v1.0.0", ckpt, lin1, 12U);
  assert (v1_res.has_value () && v1_res->version == 1U);
  assert (v1_res->tensor_count == 12U);

  /* Optimistic concurrency failure test.  */
  auto cas_fail = reg.register_version_cas ("basalt-resnet18", 99U, "v1.1.0",
                                            ckpt, lin1);
  assert (!cas_fail.has_value () && cas_fail.error ().is_registry_error ());

  /* Lineage cycle rejection test.  */
  yantra::ModelLineage cyclic_lin{};
  cyclic_lin.parent_model_versions = { "basalt-resnet18:v2.0.0" };
  auto cyc_fail = reg.register_version ("basalt-resnet18", "v2.0.0", ckpt,
                                        cyclic_lin, 12U);
  assert (!cyc_fail.has_value () && cyc_fail.error ().is_registry_error ());

  /* Valid child version registration and lineage query.  */
  yantra::ModelLineage lin2{};
  lin2.run_id = "run_2";
  lin2.parent_model_versions = { "basalt-resnet18:v1.0.0" };
  auto v2_res = reg.register_version_cas ("basalt-resnet18", 1U, "v1.2.0",
                                          ckpt, lin2);
  assert (v2_res.has_value () && v2_res->version == 2U);

  auto lk = reg.lookup ("basalt-resnet18:v1.2.0");
  assert (lk.has_value () && lk->version == 2U);

  auto lineage_chain = reg.query_lineage ("basalt-resnet18", 2U);
  assert (lineage_chain.has_value () && lineage_chain->size () == 2U);

  /* Stage transition validation: empty justification rejected.  */
  auto bad_just = reg.transition_stage ("basalt-resnet18", 2U,
                                        yantra::ModelStage::Production, "");
  assert (!bad_just.has_value () && bad_just.error ().is_registry_error ());

  assert (reg.transition_stage ("basalt-resnet18", 2U,
                                yantra::ModelStage::Production,
                                "Passed validation recall SLA")
              .has_value ());
  assert (reg.transition_stage ("basalt-resnet18", 2U,
                                yantra::ModelStage::Archived,
                                "Superseded by v2")
              .has_value ());

  /* Archived -> Production transition rejected.  */
  auto bad_stage = reg.transition_stage (
      "basalt-resnet18", 2U, yantra::ModelStage::Production, "Re-promote");
  assert (!bad_stage.has_value () && bad_stage.error ().is_registry_error ());

  auto warm_load = reg.load_artifact_warm_mmap ("basalt-resnet18", 2U);
  assert (warm_load.has_value () && warm_load->tensor_count == 12U
          && warm_load->sha256_hex == v2_res->sha256_hex);

  auto cold_load = reg.load_artifact_cold_nvme ("basalt-resnet18", 2U);
  assert (cold_load.has_value ()
          && cold_load->sha256_hex == v2_res->sha256_hex);

  fs::remove_all (tmp_dir, ec);
}

void
test_dag_orchestrator ()
{
  yantra::DagScheduler sched (4U, { 8U, 4ULL * 1024ULL * 1024ULL * 1024ULL });

  /* Boundary: empty pipeline rejected.  */
  yantra::PipelineDefinition empty_pipe;
  auto empty_res = sched.execute (empty_pipe);
  assert (!empty_res.has_value () && empty_res.error ().is_dag_error ());

  /* Cycle detection test.  */
  yantra::PipelineDefinition cyc_pipe;
  yantra::Task ta{};
  ta.id = "A";
  ta.dependencies = { "B" };
  yantra::Task tb{};
  tb.id = "B";
  tb.dependencies = { "A" };
  assert (cyc_pipe.add_task (ta).has_value ());
  assert (cyc_pipe.add_task (tb).has_value ());
  auto cyc_res = sched.execute (cyc_pipe);
  assert (!cyc_res.has_value () && cyc_res.error ().is_dag_error ());

  /* Resource deadlock detection test.  */
  yantra::PipelineDefinition dl_pipe;
  yantra::Task huge{};
  huge.id = "huge_gpu";
  huge.resources.cpu_cores = 64U;
  assert (dl_pipe.add_task (huge).has_value ());
  auto dl_res = sched.execute (dl_pipe);
  assert (!dl_res.has_value () && dl_res.error ().is_dag_error ());

  /* Retries, conditional edges, and failure skip propagation.  */
  yantra::PipelineDefinition pipe;
  std::atomic<int> flaky_tries{ 0 };

  yantra::Task t_flaky{};
  t_flaky.id = "flaky";
  t_flaky.retry_policy.max_retries = 2U;
  t_flaky.retry_policy.initial_backoff_ns = 100000ULL;
  t_flaky.run = [&] (yantra::TaskContext &)
      -> yantra::Result<std::vector<std::string>> {
    if (flaky_tries.fetch_add (1) < 1)
      {
        return std::unexpected (yantra::Error::dag (
            yantra::DagErrorCode::TaskExecutionFailed, "flaky",
            "Simulated transient failure on first attempt"));
      }
    return std::vector<std::string>{ "art_hash_1" };
  };

  yantra::Task t_cond{};
  t_cond.id = "conditional_skip";
  t_cond.dependencies = { "flaky" };
  t_cond.condition = [] (const yantra::TaskContext &) { return false; };
  t_cond.run = [] (yantra::TaskContext &)
      -> yantra::Result<std::vector<std::string>> {
    return std::vector<std::string>{};
  };

  assert (pipe.add_task (t_flaky).has_value ());
  assert (pipe.add_task (t_cond).has_value ());

  const std::string json_str = pipe.to_json ();
  auto parsed_pipe = yantra::PipelineDefinition::from_json (json_str);
  assert (parsed_pipe.has_value () && parsed_pipe->tasks ().size () == 2U);

  auto exec_res = sched.execute (pipe);
  assert (exec_res.has_value ());
  assert (exec_res->results["flaky"].status == yantra::TaskStatus::Completed);
  assert (exec_res->results["flaky"].attempts == 2U);
  assert (exec_res->results["conditional_skip"].status
          == yantra::TaskStatus::Skipped);
}

void
test_vector_indices ()
{
  constexpr std::size_t kDim = 32U;
  constexpr std::size_t kNum = 1000U;

  std::mt19937 rng (42U);
  std::uniform_real_distribution<float> dist (-1.0F, 1.0F);
  std::vector<float> data (kNum * kDim);
  for (float &v : data)
    {
      v = dist (rng);
    }

  yantra::FlatIndex flat (kDim);
  yantra::HnswIndex hnsw (kDim, { 16U, 100U, 64U });
  yantra::IvfPqIndex ivfpq (kDim, { 16U, 8U, 8U, 32U });

  /* Boundary: search before build fails cleanly.  */
  std::vector<float> q (data.begin (), data.begin () + kDim);
  assert (!hnsw.search (q, 5U).has_value ());
  assert (!ivfpq.search (q, 5U).has_value ());

  /* Boundary: single-element index.  */
  assert (hnsw.add (q, 0U, { 0U, 100U, 1, 0.5 }).has_value ());
  auto single_res = hnsw.search (q, 5U);
  assert (single_res.has_value () && single_res->size () == 1U
          && (*single_res)[0].id == 0U);

  /* Build full 1,000-vector indices and verify recall & persistence.  */
  yantra::HnswIndex hnsw_full (kDim, { 16U, 120U, 64U });
  assert (flat.add_batch (data, kNum).has_value ());
  assert (hnsw_full.add_batch (data, kNum).has_value ());
  assert (ivfpq.build (data, kNum).has_value ());

  /* Incremental insertion into HNSW and IVF-PQ.  */
  assert (hnsw_full.add (q, kNum, { kNum, 500U, 7, 1.0 }).has_value ());
  assert (ivfpq.add (q, kNum).has_value ());

  auto gt = flat.search_exact_fp64 (q, 10U);
  auto h_res = hnsw_full.search (q, 10U);
  auto pq_res = ivfpq.search (q, 10U);
  assert (gt.has_value () && h_res.has_value () && pq_res.has_value ());

  /* Filtered search test.  */
  yantra::MetadataFilter mf{};
  mf.required_category = 7;
  auto filt_res = hnsw_full.search_filtered (q, 5U, mf);
  assert (filt_res.has_value () && !filt_res->empty ()
          && (*filt_res)[0].id == kNum);

  /* Persistence save + mmap load test.  */
  const fs::path idx_path
      = fs::temp_directory_path () / "yantra_unit_hnsw.bin";
  assert (hnsw_full.save (idx_path).has_value ());
  auto mmap_idx = yantra::HnswIndex::load_mmap (idx_path);
  assert (mmap_idx.has_value () && mmap_idx->size () == kNum + 1U);
  auto mmap_search = mmap_idx->search (q, 10U);
  assert (mmap_search.has_value () && mmap_search->size () == 10U);

  /* GPU index export and search if CUDA is available.  */
  if (yantra::is_cuda_available ())
    {
      auto gpu_idx = mmap_idx->to_gpu_index ();
      assert (gpu_idx.has_value ());
      auto g_res = gpu_idx->search (q, 10U);
      assert (g_res.has_value () && g_res->size () == 10U);
    }

  std::error_code ec;
  fs::remove (idx_path, ec);
}

void
test_fairness_and_drift ()
{
  constexpr std::size_t kRows = 10000U;
  std::vector<double> scores (kRows);
  std::vector<uint8_t> labels (kRows);
  std::vector<uint32_t> groups (kRows);

  for (std::size_t i = 0U; i < kRows; ++i)
    {
      const uint32_t g = static_cast<uint32_t> (i % 2U);
      groups[i] = g;
      labels[i] = static_cast<uint8_t> ((i % 3U == 0U) ? 1U : 0U);
      const double base = (labels[i] != 0U) ? 0.72 : 0.28;
      const double bias = (g == 0U) ? 0.12 : -0.12;
      scores[i] = std::clamp (
          base + bias + 0.05 * std::sin (static_cast<double> (i)), 0.01, 0.99);
    }

  yantra::DatasetView view{};
  view.prediction_scores = scores;
  view.true_labels = labels;
  view.sensitive_groups = groups;
  view.num_groups = 2U;

  auto rep_fast = yantra::FairnessAuditor::audit (view);
  auto rep_ref = yantra::FairnessAuditor::audit_scalar_reference (view);
  assert (rep_fast.has_value () && rep_ref.has_value ());
  assert (rep_fast->total_rows == kRows);
  assert (std::abs (rep_fast->demographic_parity_difference
                    - rep_ref->demographic_parity_difference)
          < 1.0e-12);

  auto opt_res
      = yantra::FairnessAuditor::optimize_thresholds (view, 0.05, 80U);
  assert (opt_res.has_value ());
  assert (opt_res->iterations <= 80U);
  assert (opt_res->optimized_equalized_odds_diff
          <= opt_res->initial_equalized_odds_diff);

  std::vector<double> prod_scores (kRows);
  for (std::size_t i = 0U; i < kRows; ++i)
    {
      prod_scores[i] = std::clamp (scores[i] + 0.08, 0.0, 1.0);
    }
  auto drift_fast = yantra::DriftDetector::compute_drift (scores, prod_scores);
  auto drift_ref = yantra::DriftDetector::compute_drift_scalar_reference (
      scores, prod_scores);
  assert (drift_fast.has_value () && drift_ref.has_value ());
  assert (drift_fast->psi > 0.0 && drift_fast->ks_statistic > 0.0
          && drift_fast->ks_statistic <= 1.0);
  assert (std::abs (drift_fast->psi - drift_ref->psi) < 1.0e-12);
}

} /* namespace */

auto
main () -> int
{
  test_tracking_and_wal ();
  test_model_registry ();
  test_dag_orchestrator ();
  test_vector_indices ();
  test_fairness_and_drift ();
  std::cout << "All Yantra unit tests PASSED.\n";
  return 0;
}
