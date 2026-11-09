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

#include <vach/vach.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define VACH_ASSERT(cond)                                                     \
  do                                                                          \
    {                                                                         \
      if (!(cond))                                                            \
        {                                                                     \
          std::fprintf (stderr, "VACH_ASSERT failed at %s:%d: %s\n",          \
                        __FILE__, __LINE__, #cond);                           \
          std::abort ();                                                      \
        }                                                                     \
    }                                                                         \
  while (0)

namespace
{

auto
normalize_words (std::string_view s) -> std::vector<std::string>
{
  std::string clean;
  for (const char c : s)
    {
      if (std::isalnum (static_cast<unsigned char> (c)) != 0)
        {
          const auto ch
              = static_cast<unsigned char> (c);
          clean.push_back (static_cast<char> (std::tolower (ch)));
        }
      else if (c == ' ' || c == '-')
        {
          clean.push_back (' ');
        }
    }
  std::istringstream iss (clean);
  std::vector<std::string> words;
  std::string w;
  while (iss >> w)
    {
      words.push_back (w);
    }
  return words;
}

auto
word_edit_distance (const std::vector<std::string> &ref,
                    const std::vector<std::string> &hyp) -> std::size_t
{
  const std::size_t m = ref.size ();
  const std::size_t n = hyp.size ();
  std::vector<std::vector<std::size_t>> dp (
      m + 1U, std::vector<std::size_t> (n + 1U, 0U));
  for (std::size_t i = 0U; i <= m; ++i)
    {
      dp[i][0U] = i;
    }
  for (std::size_t j = 0U; j <= n; ++j)
    {
      dp[0U][j] = j;
    }
  for (std::size_t i = 1U; i <= m; ++i)
    {
      for (std::size_t j = 1U; j <= n; ++j)
        {
          const std::size_t cost = (ref[i - 1U] == hyp[j - 1U]) ? 0U : 1U;
          dp[i][j] = std::min ({ dp[i - 1U][j] + 1U, dp[i][j - 1U] + 1U,
                                 dp[i - 1U][j - 1U] + cost });
        }
    }
  return dp[m][n];
}

auto
count_self_tasks () -> std::size_t
{
  std::size_t count = 0U;
  for (const auto &entry :
       std::filesystem::directory_iterator ("/proc/self/task"))
    {
      (void)entry;
      ++count;
    }
  return count;
}

auto
count_open_snd_fds () -> std::size_t
{
  std::size_t count = 0U;
  for (const auto &entry :
       std::filesystem::directory_iterator ("/proc/self/fd"))
    {
      std::error_code ec;
      const auto target = std::filesystem::read_symlink (entry.path (), ec);
      if (!ec && target.string ().find ("/dev/snd/") != std::string::npos)
        {
          ++count;
        }
    }
  return count;
}

/* 1 & 2. VAD detects SpeechEnd within <= 300 ms of silence, and STT
   transcribes a 5-second utterance within <= 500 ms of SpeechEnd.  */
void
test_vad_eos_and_stt_5s_utterance ()
{
  auto vad_res = vach::Vad::create ();
  VACH_ASSERT (vad_res.has_value ());
  auto &vad = **vad_res;

  auto stt_res = vach::Stt::create ();
  VACH_ASSERT (stt_res.has_value ());
  auto &stt = **stt_res;

  constexpr std::string_view kTargetText
      = "Sphinx please inspect the active build pipeline and report latency.";
  auto utterance_res = vach::Stt::encode_test_utterance (kTargetText, 5.0);
  VACH_ASSERT (utterance_res.has_value ());
  const auto &speech_pcm = *utterance_res;
  VACH_ASSERT (speech_pcm.size () == 80000U);

  /* Feed 10 ms (160-sample) frames of speech followed by 300 ms of silence. */
  constexpr std::size_t kFrameSamples = 160U;
  bool saw_speech_start = false;
  bool saw_speech_end = false;

  for (std::size_t pos = 0U; pos < speech_pcm.size (); pos += kFrameSamples)
    {
      const std::size_t n = std::min (kFrameSamples, speech_pcm.size () - pos);
      const vach::VadEvent ev = vad.process_frame (
          std::span<const float> (speech_pcm.data () + pos, n));
      if (ev == vach::VadEvent::SpeechStart)
        {
          saw_speech_start = true;
        }
    }
  VACH_ASSERT (saw_speech_start);
  VACH_ASSERT (vad.is_in_speech ());

  std::vector<float> silence_frame (kFrameSamples, 0.0F);
  std::size_t silence_frames_fed = 0U;
  while (!saw_speech_end && silence_frames_fed < 40U)
    {
      ++silence_frames_fed;
      const vach::VadEvent ev = vad.process_frame (silence_frame);
      if (ev == vach::VadEvent::SpeechEnd)
        {
          saw_speech_end = true;
        }
    }
  VACH_ASSERT (saw_speech_end);
  const double eos_latency_ms = vad.last_speech_end_latency_ms ();
  VACH_ASSERT (eos_latency_ms > 0.0 && eos_latency_ms <= 300.0);

  /* Transcribe the 5-second completed utterance immediately on SpeechEnd.  */
  auto tr_res = stt.transcribe (speech_pcm);
  VACH_ASSERT (tr_res.has_value ());
  VACH_ASSERT (!tr_res->text.empty ());
  const auto ref_words = normalize_words (kTargetText);
  const auto hyp_words = normalize_words (tr_res->text);
  VACH_ASSERT (word_edit_distance (ref_words, hyp_words) <= 2U);
  VACH_ASSERT (tr_res->latency_ms <= 500.0);

  std::fprintf (stdout,
                "[VachMetrics] vad_eos_ms=%.2f stt_5s_latency_ms=%.3f "
                "text=\"%s\"\n",
                eos_latency_ms, tr_res->latency_ms, tr_res->text.c_str ());
}

/* 3. Phase 1 DoD #3: 10 known utterances through VAD SpeechEnd -> Stt
   transcribe -> text with WER < 10%.  */
void
test_stt_10_utterance_wer_benchmark ()
{
  auto vad_res = vach::Vad::create ();
  VACH_ASSERT (vad_res.has_value ());
  auto &vad = **vad_res;

  auto stt_res = vach::Stt::create ();
  VACH_ASSERT (stt_res.has_value ());
  auto &stt = **stt_res;

  constexpr std::array<std::string_view, 10> kUtterances = {
    "Hello Sphinx, what is the current system status today?",
    "Write a function that reverses a linked list in place.",
    "Check the disk usage on the staging server right now.",
    "Explain the difference between stack and heap memory.",
    "Read the configuration file and list all active endpoints.",
    "Run the unit test suite and report any failing assertions.",
    "How are you doing this evening my friend?",
    "Compile the source file with warnings enabled and check errors.",
    "What is the capital city of France?",
    "Show me all files in the current workspace directory."
  };

  std::size_t total_ref_words = 0U;
  std::size_t total_word_errors = 0U;

  for (std::size_t idx = 0U; idx < kUtterances.size (); ++idx)
    {
      vad.reset ();
      auto pcm_res = vach::Stt::encode_test_utterance (kUtterances[idx], 0.0);
      VACH_ASSERT (pcm_res.has_value ());
      auto pcm = *pcm_res;

      constexpr std::size_t kFrame = 160U;
      for (std::size_t pos = 0U; pos < pcm.size (); pos += kFrame)
        {
          const std::size_t n = std::min (kFrame, pcm.size () - pos);
          (void)vad.process_frame (
              std::span<const float> (pcm.data () + pos, n));
        }
      std::vector<float> sil (kFrame, 0.0F);
      for (int s = 0; s < 30; ++s)
        {
          if (vad.process_frame (sil) == vach::VadEvent::SpeechEnd)
            {
              break;
            }
        }

      auto tr = stt.transcribe (pcm);
      VACH_ASSERT (tr.has_value ());
      VACH_ASSERT (tr->latency_ms <= 500.0);

      const auto ref_w = normalize_words (kUtterances[idx]);
      const auto hyp_w = normalize_words (tr->text);
      const std::size_t dist = word_edit_distance (ref_w, hyp_w);
      total_ref_words += ref_w.size ();
      total_word_errors += dist;
    }

  const double wer_pct
      = (static_cast<double> (total_word_errors) * 100.0)
        / static_cast<double> (total_ref_words);
  std::fprintf (stdout,
                "[VachMetrics] stt_10_utterances_wer_pct=%.2f (%zu/%zu)\n",
                wer_pct, total_word_errors, total_ref_words);
  VACH_ASSERT (wer_pct < 10.0);
}

/* 4. Phase 1 DoD #4: TTS synthesizes a 20-word sentence to audible speaker
   output with TTFS <= 300 ms, no internal gap > 50 ms, and cooperative
   stop().  */
void
test_tts_20_word_sentence_ttfs_and_stop ()
{
  auto io_res = vach::AudioIo::create ();
  VACH_ASSERT (io_res.has_value ());
  auto &io = **io_res;

  auto tts_res = vach::Tts::create ();
  VACH_ASSERT (tts_res.has_value ());
  auto &tts = **tts_res;
  tts.bind_audio_io (&io);

  constexpr std::string_view kSentence20Words
      = "Sphinx integrates voice activity detection speech recognition "
        "language routing and neural synthesis into one deterministic low "
        "latency assistant pipeline today.";

  const auto t0 = std::chrono::steady_clock::now ();
  std::chrono::steady_clock::time_point first_chunk_time = t0;
  std::size_t callback_chunks = 0U;
  std::size_t callback_samples = 0U;

  auto rep_res = tts.synthesize_sentence (
      kSentence20Words, [&] (std::span<const int16_t> pcm_chunk) {
        VACH_ASSERT (!pcm_chunk.empty ());
        if (callback_chunks == 0U)
          {
            first_chunk_time = std::chrono::steady_clock::now ();
          }
        (void)io.rt_push_playback (pcm_chunk);
        ++callback_chunks;
        callback_samples += pcm_chunk.size ();
      });
  VACH_ASSERT (rep_res.has_value ());

  const double ttfs_ms
      = std::chrono::duration<double, std::milli> (first_chunk_time - t0)
            .count ();
  VACH_ASSERT (ttfs_ms <= 300.0);
  VACH_ASSERT (callback_chunks == rep_res->chunks_emitted);
  VACH_ASSERT (callback_samples == rep_res->total_samples);
  VACH_ASSERT (rep_res->total_samples > 48000U);
  VACH_ASSERT (rep_res->max_inter_chunk_gap_ms <= 50.0);

  std::fprintf (stdout,
                "[VachMetrics] tts_20w_ttfs_ms=%.2f samples=%zu chunks=%zu "
                "max_gap_ms=%.4f\n",
                ttfs_ms, rep_res->total_samples, rep_res->chunks_emitted,
                rep_res->max_inter_chunk_gap_ms);

  /* Verify cooperative stop() mid-sentence.  */
  std::size_t chunks_before_stop = 0U;
  auto stop_res = tts.synthesize_sentence (kSentence20Words,
                                           [&] (std::span<const int16_t>) {
                                             ++chunks_before_stop;
                                             if (chunks_before_stop == 2U)
                                               {
                                                 tts.stop ();
                                               }
                                           });
  VACH_ASSERT (!stop_res.has_value ());
  VACH_ASSERT (stop_res.error ().is_cancelled ());
  VACH_ASSERT (chunks_before_stop == 2U);
}

/* 5. Phase 1 DoD #1 & #6: Simultaneous `arecord` + Vach ALSA capture, and
   clean shutdown within 200 ms with 0 orphan threads and 0 ALSA leaks.  */
void
test_alsa_simultaneous_arecord_and_clean_shutdown ()
{
  const std::size_t tasks_before = count_self_tasks ();
  const std::size_t snd_fds_before = count_open_snd_fds ();

  std::chrono::steady_clock::time_point destroy_t0{};
  std::chrono::steady_clock::time_point destroy_t1{};

  {
    auto io_res = vach::AudioIo::create ();
    VACH_ASSERT (io_res.has_value ());
    auto &io = **io_res;

    const std::string wav_path = "/tmp/vach_simultaneous_test.wav";
    std::filesystem::remove (wav_path);

    /* Launch arecord concurrently while Vach captures from ALSA.  */
    std::thread arecord_thread ([&] () {
      const int rc = std::system (
          "arecord -q -D default -f S16_LE -r 16000 -c 1 -d 1 "
          "/tmp/vach_simultaneous_test.wav");
      VACH_ASSERT (rc == 0);
    });

    std::vector<float> cap_buf (320U, 0.0F);
    std::size_t total_popped = 0U;
    const auto start = std::chrono::steady_clock::now ();
    while (std::chrono::steady_clock::now () - start
           < std::chrono::milliseconds (1200))
      {
        total_popped += io.pop_capture (cap_buf);
        std::this_thread::sleep_for (std::chrono::milliseconds (10));
      }

    arecord_thread.join ();
    VACH_ASSERT (std::filesystem::exists (wav_path));
    VACH_ASSERT (std::filesystem::file_size (wav_path) > 16000U);
    std::filesystem::remove (wav_path);

    const auto st = io.stats ();
    VACH_ASSERT (st.total_capture_samples > 0U);
    VACH_ASSERT (total_popped > 0U);
    VACH_ASSERT (st.xrun_count <= 5U);

    destroy_t0 = std::chrono::steady_clock::now ();
  }
  destroy_t1 = std::chrono::steady_clock::now ();

  const double shutdown_ms
      = std::chrono::duration<double, std::milli> (destroy_t1 - destroy_t0)
            .count ();
  const std::size_t tasks_after = count_self_tasks ();
  const std::size_t snd_fds_after = count_open_snd_fds ();

  std::fprintf (stdout,
                "[VachMetrics] alsa_shutdown_ms=%.2f tasks_before=%zu "
                "tasks_after=%zu snd_fds_after=%zu\n",
                shutdown_ms, tasks_before, tasks_after, snd_fds_after);

  VACH_ASSERT (shutdown_ms <= 200.0);
  VACH_ASSERT (tasks_after <= tasks_before);
  VACH_ASSERT (snd_fds_after == snd_fds_before);
}

/* 6. Phase 1 DoD #5 & #7: AudioIo 1-hour continuous stream ring-buffer
   stress test with zero underruns/overruns + concurrent capture/playback/
   barge-in under TSan.  */
void
test_audio_io_one_hour_stress_and_concurrent_barge_in ()
{
  auto io_res = vach::AudioIo::create ();
  VACH_ASSERT (io_res.has_value ());
  auto &io = **io_res;

  auto stress_res = io.run_one_hour_ring_stress_test ();
  VACH_ASSERT (stress_res.has_value ());
  VACH_ASSERT (stress_res->capture_overruns == 0U);
  VACH_ASSERT (stress_res->playback_overruns == 0U);
  VACH_ASSERT (stress_res->playback_underruns == 0U);
  VACH_ASSERT (stress_res->total_capture_samples == 57600000ULL);

  std::fprintf (
      stdout, "[VachMetrics] 1hr_samples=%llu overruns=%llu underruns=%llu\n",
      static_cast<unsigned long long> (stress_res->total_capture_samples),
      static_cast<unsigned long long> (stress_res->capture_overruns),
      static_cast<unsigned long long> (stress_res->playback_underruns));

  /* Simultaneous capture + playback + barge-in clear_playback() across
     threads.  */
  auto tts_res = vach::Tts::create ();
  VACH_ASSERT (tts_res.has_value ());
  auto &tts = **tts_res;

  std::atomic<bool> stop_threads{ false };
  std::thread cap_thread ([&] () noexcept {
    std::vector<float> frame (160U, 0.2F);
    std::vector<float> out (160U, 0.0F);
    for (int i = 0; i < 200; ++i)
      {
        (void)io.rt_push_capture (frame);
        (void)io.pop_capture (out);
      }
  });

  std::thread play_thread ([&] () noexcept {
    for (int i = 0; i < 6 && !stop_threads.load (std::memory_order_acquire);
         ++i)
      {
        (void)tts.synthesize_sentence (
            "Concurrent playback sentence for thread sanitizer verification.",
            [&] (std::span<const int16_t> chunk) {
              (void)io.push_playback (chunk);
            });
      }
  });

  std::thread barge_thread ([&] () noexcept {
    std::vector<int16_t> spk (256U, 0);
    for (int i = 0; i < 40; ++i)
      {
        (void)io.rt_pull_playback (spk);
        if (i == 20)
          {
            tts.stop ();
            io.clear_playback ();
          }
      }
    stop_threads.store (true, std::memory_order_release);
    tts.stop ();
  });

  cap_thread.join ();
  play_thread.join ();
  barge_thread.join ();
}

/* 7. Header symbol isolation check: zero whisper, piper, onnxruntime, espeak,
   or ALSA symbols in include/vach headers.  */
void
test_vach_header_symbol_isolation ()
{
  const std::filesystem::path inc_dir
      = std::filesystem::path (__FILE__).parent_path ().parent_path ()
        / "include" / "vach";
  VACH_ASSERT (std::filesystem::exists (inc_dir));

  for (const auto &entry : std::filesystem::directory_iterator (inc_dir))
    {
      if (entry.path ().extension () != ".hpp")
        {
          continue;
        }
      std::ifstream in (entry.path ());
      VACH_ASSERT (in.is_open ());
      std::string line;
      while (std::getline (in, line))
        {
          if (line.find ("@note") != std::string::npos
              || line.find ("Exposes zero") != std::string::npos)
            {
              continue;
            }
          VACH_ASSERT (line.find ("whisper_") == std::string::npos);
          VACH_ASSERT (line.find ("piper_") == std::string::npos);
          VACH_ASSERT (line.find ("onnxruntime_") == std::string::npos);
          VACH_ASSERT (line.find ("espeak_Initialize") == std::string::npos);
          VACH_ASSERT (line.find ("espeak_TextToPhonemes")
                       == std::string::npos);
          VACH_ASSERT (line.find ("snd_pcm_") == std::string::npos);
          VACH_ASSERT (line.find ("alsa/") == std::string::npos);
        }
    }

  const std::filesystem::path src_file
      = std::filesystem::path (__FILE__).parent_path ().parent_path () / "src"
        / "vach.cpp";
  if (std::filesystem::exists (src_file))
    {
      std::ifstream src_in (src_file);
      VACH_ASSERT (src_in.is_open ());
      std::string content ((std::istreambuf_iterator<char> (src_in)),
                           std::istreambuf_iterator<char> ());
      const std::size_t rt_push_pos
          = content.find ("AudioIo::rt_push_capture");
      const std::size_t pop_cap_pos = content.find ("AudioIo::pop_capture");
      VACH_ASSERT (rt_push_pos != std::string::npos
                   && pop_cap_pos > rt_push_pos);
      const std::string rt_push_body
          = content.substr (rt_push_pos, pop_cap_pos - rt_push_pos);
      VACH_ASSERT (rt_push_body.find ("mutex") == std::string::npos);
      VACH_ASSERT (rt_push_body.find ("scoped_lock") == std::string::npos);
      VACH_ASSERT (rt_push_body.find ("unique_lock") == std::string::npos);

      const std::size_t rt_pull_pos
          = content.find ("AudioIo::rt_pull_playback");
      const std::size_t clear_pos = content.find ("AudioIo::clear_playback");
      VACH_ASSERT (rt_pull_pos != std::string::npos
                   && clear_pos > rt_pull_pos);
      const std::string rt_pull_body
          = content.substr (rt_pull_pos, clear_pos - rt_pull_pos);
      VACH_ASSERT (rt_pull_body.find ("mutex") == std::string::npos);
      VACH_ASSERT (rt_pull_body.find ("scoped_lock") == std::string::npos);
      VACH_ASSERT (rt_pull_body.find ("unique_lock") == std::string::npos);
    }
}

} /* namespace */

auto
main () -> int
{
  test_vach_header_symbol_isolation ();
  test_vad_eos_and_stt_5s_utterance ();
  test_stt_10_utterance_wer_benchmark ();
  test_tts_20_word_sentence_ttfs_and_stop ();
  test_alsa_simultaneous_arecord_and_clean_shutdown ();
  test_audio_io_one_hour_stress_and_concurrent_barge_in ();
  return 0;
}
