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

#include "cortex_a5_engine.hpp"

namespace {

uint32_t parse_u32_arg(const char* s) {
    if (s == nullptr) {
        return 0;
    }
    uint32_t v = 0;
    while (*s >= '0' && *s <= '9') {
        v = (v * 10u) + static_cast<uint32_t>(*s - '0');
        ++s;
    }
    return v;
}

void print_banner() {
    singularity::print_str(
        "========================================================================\n"
        " Singularity v1.0 — Bare-Metal ARMv7-A (Cortex-A5 NEON/VFPv4) Engine\n"
        " Target: LG Optimus L5 (LG-E612 \"Billy\") | Qualcomm MSM7225A @ 800 MHz\n"
        " License: GPLv3 | ABI: arm-linux-gnueabihf | Zero-libstdc++ Freestanding\n"
        " Milda 5-Language Router: Esperanto | English | Espanol | Русский | 简体中文\n"
        "========================================================================\n"
    );
}

void print_usage() {
    singularity::print_str(
        "Usage: ./singularity [options]\n"
        "  --chat, -i              Start interactive multi-turn conversation REPL\n"
        "  --model <file.gguf>     Map and run an external GGUF v2/v3 model via sys_mmap2\n"
        "  --gen-gguf <file.gguf>  Emit a Milda 5-Language GGUF v3 file and mmap it\n"
        "  --prompt <utf8-text>    Run inference on a UTF-8 prompt (persists across sessions)\n"
        "  --tokens <count>        Number of tokens to generate per turn (default: 16)\n"
        "  --session <file.bin>    Custom persistent session file (default: /opt/singularity/session.bin)\n"
        "  --memory                Display saved cross-session memory & conversation history\n"
        "  --reset-session         Clear persistent cross-session memory on disk\n"
        "  --no-session            Disable cross-session persistence for this run\n"
        "  --all-langs             Run 5-language Milda verification pass (default)\n"
        "  --help                  Display this help message\n"
    );
}

void print_turn_stats(
    const singularity::CortexA5Engine& engine,
    const singularity::InferenceStats& stats
) {
    using namespace singularity;
    print_str("  [Session #");
    print_u32(engine.session_count() > 0 ? engine.session_count() : 1u);
    print_str(" | Turn #");
    print_u32(engine.turn_count());
    print_str(" | Lang: ");
    print_str(CortexA5Engine::language_name(stats.dominant_lang));
    print_str(" | Mask: ");
    print_hex32(stats.script_mask);
    print_str(" | KV-Pos: ");
    print_u32(engine.session_pos());
    print_str("/");
    print_u32(engine.config().max_seq_len);
    print_str(" | Tokens: ");
    print_u32(stats.tokens_generated);
    print_str(" | RAM: ");
    print_u32(stats.available_ram_kb >> 10);
    print_str(" MB]\n");
}

void run_interactive_chat(singularity::CortexA5Engine& engine, uint32_t max_tokens) {
    using namespace singularity;
    print_str(
        "[Multi-Turn Chat Mode] Commands: /memory, /remember <fact>, /forget, /status, /lang <eo|en|es|ru|zh|auto>, /exit\n"
        "------------------------------------------------------------------------\n"
    );

    char line_buf[512];
    while (true) {
        print_str("\nYou [S");
        print_u32(engine.session_count() > 0 ? engine.session_count() : 1u);
        print_str(":T");
        print_u32(engine.turn_count() + 1u);
        print_str("] > ");

        const int32_t nread = read_stdin_line(line_buf, sizeof(line_buf));
        if (nread < 0) {
            print_str("\n[Session saved & closed]\n");
            break;
        }
        if (nread == 0) {
            continue;
        }

        if (str_eq(line_buf, "/exit") || str_eq(line_buf, "/quit") || str_eq(line_buf, "exit")) {
            engine.save_session_state();
            print_str("[Session saved to disk. Exiting Singularity]\n");
            break;
        }
        if (str_eq(line_buf, "/reset") || str_eq(line_buf, "/clear") || str_eq(line_buf, "/forget")) {
            engine.clear_session_state();
            print_str("[Cross-session memory & fp16 KV cache cleared]\n");
            continue;
        }
        if (str_eq(line_buf, "/memory") || str_eq(line_buf, "/history")) {
            engine.print_memory_summary();
            continue;
        }
        if (str_starts_with(line_buf, "/remember ")) {
            engine.set_user_fact(line_buf + 10);
            engine.save_session_state();
            print_str("[Saved to persistent cross-session memory: ");
            print_str(engine.user_fact());
            print_str("]\n");
            continue;
        }
        if (str_eq(line_buf, "/status") || str_eq(line_buf, "/mem")) {
            print_str("[Status] Session #");
            print_u32(engine.session_count());
            print_str(" | Turns: ");
            print_u32(engine.turn_count());
            print_str(" | KV-Pos: ");
            print_u32(engine.session_pos());
            print_str("/");
            print_u32(engine.config().max_seq_len);
            print_str(" | Available RAM: ");
            print_u32(engine.available_ram_kb() >> 10);
            print_str(" MB | MemFree: ");
            print_u32(engine.free_ram_kb() >> 10);
            print_str(" MB\n");
            continue;
        }
        if (str_starts_with(line_buf, "/lang ")) {
            const char* code = line_buf + 6;
            if (str_eq(code, "eo")) {
                engine.set_forced_language(LANG_ESPERANTO);
                print_str("[Language locked to Esperanto (eo) across sessions]\n");
            } else if (str_eq(code, "en")) {
                engine.set_forced_language(LANG_ENGLISH);
                print_str("[Language locked to English (en) across sessions]\n");
            } else if (str_eq(code, "es")) {
                engine.set_forced_language(LANG_SPANISH);
                print_str("[Language locked to Spanish (es) across sessions]\n");
            } else if (str_eq(code, "ru")) {
                engine.set_forced_language(LANG_RUSSIAN);
                print_str("[Language locked to Russian (ru) across sessions]\n");
            } else if (str_eq(code, "zh")) {
                engine.set_forced_language(LANG_MANDARIN);
                print_str("[Language locked to Simplified Mandarin Chinese (zh) across sessions]\n");
            } else {
                engine.set_forced_language(-1);
                print_str("[Language set to automatic NEON vtbl.8 detection]\n");
            }
            engine.save_session_state();
            continue;
        }
        if (str_eq(line_buf, "/help")) {
            print_str(
                "Commands:\n"
                "  /memory              Show persistent cross-session memory & turn history\n"
                "  /remember <fact>     Store a persistent fact across sessions\n"
                "  /forget, /reset      Clear persistent session memory & fp16 KV cache\n"
                "  /status              Display RAM, OOM guard & KV cache telemetry\n"
                "  /lang <eo|en|es|ru|zh|auto> Set or auto-detect Milda language\n"
                "  /exit                Save session and exit interactive conversation\n"
            );
            continue;
        }

        InferenceStats stats{};
        print_str("Singularity > ");
        if (!engine.chat_turn(line_buf, max_tokens, true, &stats)) {
            print_str("\n[ERROR] Invalid UTF-8 input stream rejected by NEON vtbl.8 validator.\n");
            continue;
        }
        print_turn_stats(engine, stats);
    }
}

} // namespace

int main(int argc, char** argv) {
    using namespace singularity;

    // Ensure SCHED_NORMAL on single-core MSM7225A so kernel threads (kswapd0,
    // mmcqd/0, WCN1314 Wi-Fi VOS_MC_Thread) are never starved by chrt -f 99.
    singularity_sys_sched_normal();

    const char* model_path    = nullptr;
    const char* gen_gguf_path = nullptr;
    const char* user_prompt   = nullptr;
    const char* session_path  = DEFAULT_SESSION_PATH;
    uint32_t    max_tokens    = 4;
    bool        tokens_set    = false;
    bool        chat_mode     = false;
    bool        use_session   = true;
    bool        reset_sess    = false;
    bool        show_memory   = false;

    for (int i = 1; i < argc; ++i) {
        if (str_eq(argv[i], "--help") || str_eq(argv[i], "-h")) {
            print_banner();
            print_usage();
            return 0;
        } else if (str_eq(argv[i], "--chat") || str_eq(argv[i], "-i")) {
            chat_mode = true;
        } else if (str_eq(argv[i], "--no-session")) {
            use_session = false;
        } else if (str_eq(argv[i], "--reset-session")) {
            reset_sess = true;
        } else if (str_eq(argv[i], "--memory")) {
            show_memory = true;
        } else if (str_eq(argv[i], "--session") && (i + 1) < argc) {
            session_path = argv[++i];
        } else if (str_eq(argv[i], "--model") && (i + 1) < argc) {
            model_path = argv[++i];
        } else if (str_eq(argv[i], "--gen-gguf") && (i + 1) < argc) {
            gen_gguf_path = argv[++i];
        } else if (str_eq(argv[i], "--prompt") && (i + 1) < argc) {
            user_prompt = argv[++i];
        } else if (str_eq(argv[i], "--tokens") && (i + 1) < argc) {
            max_tokens = parse_u32_arg(argv[++i]);
            tokens_set = true;
            if (max_tokens == 0) {
                max_tokens = 4;
            }
        }
    }

    if (chat_mode && !tokens_set) {
        max_tokens = 16;
    }

    print_banner();

    CortexA5Engine engine;
    bool ok = false;
    if (model_path != nullptr) {
        print_str("[mmap2] Mapping external GGUF model: ");
        print_str(model_path);
        print_str("\n");
        ok = engine.load_gguf_file(model_path);
    } else {
        ok = engine.init_milda_5lang_model(gen_gguf_path);
    }

    if (!ok) {
        print_str("[ERROR] Failed to initialize GGUF model or Linux 4.x memory arenas.\n");
        return 1;
    }

    const ModelConfig& cfg = engine.config();
    print_str("[Arena] Scratch Used: ");
    print_u32(static_cast<uint32_t>(engine.scratch_arena_used()));
    print_str(" B (Budget <= 16 MB) | fp16 KV Cache: ");
    print_u32(static_cast<uint32_t>(engine.kv_arena_used()));
    print_str(" B (Budget <= 35 MB)\n");

    print_str("[MemGuard] /proc/meminfo Available RAM: ");
    print_u32(engine.available_ram_kb() >> 10);
    print_str(" MB (OOM Guard Floor: 48 MB)\n");

    print_str("[GGUF v3] dim=");
    print_u32(cfg.dim);
    print_str(", hidden_dim=");
    print_u32(cfg.hidden_dim);
    print_str(", layers=");
    print_u32(cfg.n_layers);
    print_str(", heads=");
    print_u32(cfg.n_heads);
    print_str(", vocab=");
    print_u32(cfg.vocab_size);
    print_str("\n------------------------------------------------------------------------\n");

    if (reset_sess) {
        engine.set_session_file_path(session_path);
        engine.clear_session_state(session_path);
        print_str("[Memory] Cleared persistent cross-session memory: ");
        print_str(session_path);
        print_str("\n");
        if (user_prompt == nullptr && !chat_mode && !show_memory) {
            return 0;
        }
    }

    if (use_session && (user_prompt != nullptr || chat_mode || show_memory)) {
        if (engine.load_session_state(session_path)) {
            print_str("[Memory] Restored Session #");
            print_u32(engine.session_count());
            print_str(" (Prior Turns: ");
            print_u32(engine.turn_count());
            print_str(", Restored fp16 KV-Pos: ");
            print_u32(engine.session_pos());
            print_str("/");
            print_u32(cfg.max_seq_len);
            if (engine.user_fact()[0] != '\0') {
                print_str(", Fact: \"");
                print_str(engine.user_fact());
                print_str("\"");
            }
            print_str(")\n");
        } else {
            print_str("[Memory] Initialized new persistent Session #1 (");
            print_str(session_path);
            print_str(")\n");
        }
    }

    if (show_memory) {
        engine.print_memory_summary();
        if (user_prompt == nullptr && !chat_mode) {
            return 0;
        }
    }

    if (user_prompt != nullptr) {
        InferenceStats stats{};
        print_str("You [S");
        print_u32(engine.session_count() > 0 ? engine.session_count() : 1u);
        print_str(":T");
        print_u32(engine.turn_count() + 1u);
        print_str("] > ");
        print_str(user_prompt);
        print_str("\nSingularity > ");
        if (!engine.chat_turn(user_prompt, max_tokens, true, &stats)) {
            print_str("\n[ERROR] UTF-8 stream validation failed (invalid UTF-8 input).\n");
            return 1;
        }
        print_turn_stats(engine, stats);
        if (chat_mode) {
            run_interactive_chat(engine, max_tokens);
        }
        return 0;
    }

    if (chat_mode) {
        run_interactive_chat(engine, max_tokens);
        return 0;
    }

    // Default: Run all 5 Milda languages through the full NEON Assembly pipeline
    const char* const kPrompts[5] = {
        "Saluton! Eĥoŝanĝo ĉiuĵaŭde",
        "Hello! Bare-metal Cortex-A5",
        "¡Hola! ¿Cómo estás, niño?",
        "Привет, мир! Сингулярность",
        "你好，世界！奇点推理引擎"
    };

    for (int i = 0; i < 5; ++i) {
        InferenceStats stats{};
        print_str("[Prompt ");
        print_i32(i + 1);
        print_str("/5] ");
        print_str(kPrompts[i]);
        print_str("\n  -> [NEON vtbl.8 Route: ");
        engine.generate_stream(kPrompts[i], max_tokens, false, &stats);
        print_str(CortexA5Engine::language_name(stats.dominant_lang));
        print_str(" | Mask ");
        print_hex32(stats.script_mask);
        print_str("]: ");
        engine.generate_stream(kPrompts[i], max_tokens, true, &stats);
    }

    print_str("========================================================================\n");
    return 0;
}
