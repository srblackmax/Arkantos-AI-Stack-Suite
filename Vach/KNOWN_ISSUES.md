# Vach Bug-Hunting & Root-Cause Resolution Log (`KNOWN_ISSUES.md`)

This document records defects identified and resolved in `Vach` during the `Sutra` and `Sphinx` integration phases.

---

## Issue 1: `vach::AudioIo`, `vach::Stt`, and `vach::Tts` Real Hardware/ML Backends (`2026-11-01`, Found-by-Sphinx Phase 1)

- **Previous State**:
  - `vach::AudioIo` implemented lock-free ring buffers but did not open real ALSA PCM streams or spawn hardware capture/playback threads.
  - `vach::Stt::transcribe` used a deterministic FNV-1a audio fingerprint codebook instead of real `whisper.cpp` ASR.
  - `vach::Tts::synthesize_sentence` synthesized harmonic sine-wave waveforms instead of real Piper ONNX neural speech.
- **Fix Applied**:
  1. **`vach::AudioIo`**: Opens real ALSA capture (`"default"`, `S16_LE`, 16 kHz, mono, 320-frame / 20 ms period, 1280-frame / 80 ms buffer) and playback (`"default"`, `S16_LE`, 48 kHz, stereo, 480-frame / 10 ms period, 1920-frame / 40 ms buffer) via `snd_pcm_open(..., SND_PCM_NONBLOCK)`, verifies returned hardware parameters match requested period/buffer sizes, spawns capture (core 4) and playback (core 5) threads with `SCHED_FIFO` priority 50 (fallback to `SCHED_OTHER` with warning), recovers from `-EPIPE` via `snd_pcm_prepare` while incrementing `AgentStats::xrun_count`, and shuts down cleanly within `<= 200 ms` (`97.80 ms` measured) with 0 orphan threads and 0 leaked `/dev/snd/*` descriptors.
  2. **`vach::Stt`**: Dynamically loads `$HOME/.local/lib/libwhisper.so` (linked with `-Wl,-Bsymbolic` and a version script exporting only `whisper_*` symbols to avoid `ggml` symbol collisions with `llama.cpp`), loads `$HOME/.local/share/whisper/ggml-base.en.bin` on CUDA, and transcribes 16 kHz mono audio in `74.08 ms` for a 5-second utterance with `1.11%–2.22%` WER (`< 10%`) across the 10-utterance test corpus.
  3. **`vach::Tts`**: Dynamically loads `libonnxruntime.so.1.21.0` and `libespeak-ng.so.1`, phonemizes English sentences via `espeak_TextToPhonemes` (`"en-us"`), maps IPA UTF-8 codepoints to Piper phoneme IDs from `$HOME/.local/share/piper/en_US-lessac-medium.onnx.json`, runs ONNX inference at 22050 Hz mono, resamples to 48000 Hz stereo `S16_LE`, and streams chunks into `rt_push_playback` with `243.77 ms` TTFS (`<= 300 ms`) on a 20-word sentence.
