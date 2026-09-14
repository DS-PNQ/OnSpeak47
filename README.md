# OnSpeak47 — Omni Voice

**On-device speech translation for Vietnamese ↔ English and Vietnamese ↔ Chinese.**

Edge AI Challenge, Phase 2 | Public Services domain | Android-first, wearable-ready

> Snapshot: this README describes the **streaming-first** project state —
> input is Start/Stop streaming toggle with live Zipformer partials and
> VAD endpointing. There is no Whisper path and no WAV staging.

---

## Pipeline

```
[User speaks]
    → Streaming Zipformer (VI/EN/ZH auto-router) — live partials + endpoint finals
    → Hy-MT1.5-1.8B-1.25bit GGUF — translates text between VN/EN/CN
    → MMS-TTS — converts translated text back to speech
[Translated speech plays]
```

Three models, three stages. Source language is auto-detected by the
streaming router — the UI only selects the translation target.

**ASR is true streaming**: `AudioRecord` 16 kHz → 20 ms frames → ring
buffer → Silero VAD → 160 ms scheduler → active Zipformer → partial
transcript → async LID (400–600 ms) → router → 640 ms rollback +
candidate verification on code-switch. P50 target 250–350 ms, P95 <500 ms
on the Snapdragon target — see `docs/streaming_asr.md`.

**Translation runs on a single GGUF backend**: Tencent Hy-MT1.5-1.8B quantized
to 1.25-bit STQ (Sherry, 440 MB), decoded on the mobile CPU by llama.cpp +
the STQ kernel (llama.cpp PR #22836). Same base model as RTranslator v3
(`tencent/HY-MT1.5-1.8B`), different deployment (GGUF/llama.cpp vs INT8
ONNX/ORT). Decode is **greedy** (deterministic MT, no sampling cost), stops on
the real end-of-assistant token (`120020`), and short ASR transcripts take a
**single LLM call** (one prefill) with a 64–128 token budget.

## Project Structure

```
OnSpeak47/
├── backend/                    # Python pipeline modules
│   ├── streaming_asr/         # Zipformer streaming reference (ring/VAD/LID/router/rollback/partial/metrics/pipeline)
│   ├── translation_hymt.py    # Hy-MT1.5 translation (greedy, single-call)
│   ├── tts_mms.py             # MMS-TTS speech synthesis
│   └── orchestrator.py        # Streaming FINAL → Translation → TTS pipeline
│
├── tests_local/               # Local quality tests (no device needed)
│   ├── conftest.py            # Fixtures for all test modules
│   ├── test_02_translation.py # Hy-MT translation scoring
│   ├── test_04_vizh_corpus.py # Large-corpus VI↔ZH evaluation
│   ├── test_05_pipeline.py    # Streaming-final → Translation → TTS tests
│   ├── test_07_streaming_asr.py # Streaming ASR state-machine gate (36 tests)
│   ├── baselines/             # Recorded regression baseline (auto-created)
│   ├── output/                # Latest gate/parity results (JSON)
│   └── data/
│       ├── parallel_sentences.json     # 40 hand-curated test sentences
│       └── audio_samples/              # Test fixtures
│
├── optimize/                  # Model asset preparation
│   ├── 11_fetch_streaming_zipformer.py # Fetch Zipformer VI/EN/ZH + Silero VAD
│   ├── 07_prepare_hymt_gguf.py       # Fetch Hy-MT 1.25-bit GGUF + typefix
│   ├── 08_export_hymt_onnx_int4.py   # Reference: INT4 ONNX export (ORT GenAI)
│   ├── hymt_gguf_typefix.py          # Remap legacy STQ tensor-type codes
│   ├── 07_preoptimize.py     # Offline ORT graph pre-opt (*.opt.onnx)
│   └── export_mms_tts.py     # MMS-TTS export
│
├── android/                   # Android app (OmniVoice)
│   ├── app/src/main/
│   │   ├── java/com/omnivoice/onspeak47/
│   │   │   ├── OmniVoiceApp.java
│   │   │   ├── LoadingActivity.java
│   │   │   ├── TranslationActivity.java   # Start/Stop streaming toggle + auto translate
│   │   │   ├── pipeline/     # Translation (GGUF), HyMtGgufJNI bridge,
│   │   │   │                 # TTS, PipelineOrchestrator, Tokenizer
│   │   │   ├── asr/          # Streaming Zipformer (AudioCapture, Pipeline,
│   │   │   │                 # VAD, LID, Router, Rollback, ModelManager)
│   │   │   ├── audio/        # AudioPlayer (no recorder — mic via AudioCapture)
│   │   │   └── util/         # LanguageConfig, OrtSessionConfig, TensorUtils
│   │   ├── cpp/              # hymt_gguf_jni.cpp + CMakeLists (vendored llama.cpp)
│   │   └── res/              # Layouts, values, raw language XMLs
│   └── build.gradle
│
├── docs/
│   ├── architecture.md           # Pipeline architecture overview
│   └── optimization_results.md   # Runtime/RAM + streaming-ASR optimization log
│
└── requirements.txt           # Python dependencies
```

## Environment Setup (Python — local quality tests)

Windows PowerShell, from the `OnSpeak47/` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```powershell
python -c "import torch, transformers, sentencepiece, soundfile, librosa, scipy, jiwer, sacrebleu; print('ok')"
```

## Running Tests

### Translation quality (Hy-MT)
```powershell
python -m pytest tests_local/test_02_translation.py -v
```

### VI↔ZH large corpus evaluation
```powershell
python -m pytest tests_local/test_04_vizh_corpus.py -v -s
```

### Full pipeline (ASR → Translation → TTS)
```powershell
python -m pytest tests_local/test_05_pipeline.py -v -s
```

### All tests
```powershell
python -m pytest tests_local/ -v -s
```

### Streaming ASR gate (no models/network needed)
```powershell
python -m pytest tests_local/test_07_streaming_asr.py -v
python -m backend.streaming_asr.bench
```

## Translation Backend (Hy-MT1.5-1.8B-1.25bit)

Both the Python backend (`backend/translation_hymt.py`) and the Android app
(`pipeline/TranslationModule.java` + `cpp/hymt_gguf_jni.cpp`) share one design,
tuned against RTranslator v3's HY-MT ONNX path:

- **Greedy decode everywhere** — `temperature=0/top_k=1` (Python),
  `llama_sampler_init_greedy()` (JNI). The HY-MT report sampling values
  (`top_k=20/top_p=0.6/temp=0.7`) are reference-only; sampling adds RNG cost
  and longer, non-deterministic outputs for MT.
- **Exact chat framing** — BOS `120000` (`<｜hy_begin▁of▁sentence｜>`) prepended
  by hand (the `gpt2-pre` tokenizer never adds it), stop on EOS `120020`
  (`<｜hy_place▁holder▁no▁2｜>`), plus a repeat-run guard and whitespace/
  terminator cleanup ported from RTranslator.
- **One call per transcript** — ASR outputs (≤400 chars) go through a single
  prefill; only long paragraphs fall back to per-sentence calls. Token budget
  scales with input (64–128, 256 cap) instead of a flat 256.
- **Small context** — `n_ctx=1024`, `n_batch/ubatch=512`: a translation prompt
  (<200 tok) plus output never needs 2048.
- **Optimized native build** — `cpp/CMakeLists.txt` forces `-O2` for Debug
  (AGP always passes `Debug`, so an unguarded default silently compiled ggml
  at `-O0`, 5–10× slower) and `-O3` for Release. Optional NEON dot-product
  kernels via `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod` on capable SoCs.
- **Threads `cores-2`** (min 2), reserving headroom for ASR/TTS/UI.

Per-call timing is logged (`HyMtGgufJNI`: `prompt=N tok (prefill …) + M tok
(gen …, tok/s)`) for direct comparison against RTranslator.

## GGUF Asset Preparation

`*.gguf` files are **gitignored** — a fresh checkout must fetch the model
before building the app (the STQ typefix below is size-neutral, so it is
revision-gated, not size-gated):

```powershell
python optimize/07_prepare_hymt_gguf.py   # download GGUF → assets + onnx_models/
```

This verifies the 440 MB file, remaps legacy STQ tensor-type codes in place
(`hymt_gguf_typefix.py`), and stages copies for the app and the backend.

## Android App

The Android app is in `android/`. To build:

1. Ensure the ONNX/GGUF assets exist in `android/app/src/main/assets/`
   (gitignored — regenerate via `optimize/`; on a machine that already ran
   the scripts a plain rebuild is enough)
2. Open the `android/` folder in **Android Studio** — the project uses **Gradle 9.5**
   and requires **JDK 17+**
3. Clean `.cxx` after any `cpp/CMakeLists.txt` change
   (`Remove-Item -Recurse -Force android/app/.cxx`), then
   Build → Make Project (or Run ▶ on a device)

> The APK is built for **arm64-v8a only** (`abiFilters 'arm64-v8a'`).
> Input is **streaming toggle**: tap Start to open the mic (live partials),
> tap Stop to close it. Each VAD endpoint FINAL auto-translates → TTS.
> Fetch streaming assets first:
> `python optimize/11_fetch_streaming_zipformer.py --all --stage-assets`.

Required assets:
- `sherpa-onnx-1.13.4.aar` in `android/app/libs/` (gitignored — fetch from
  https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-1.13.4.aar)
- `Hy-MT1.5-1.8B-1.25bit.gguf` (translation, post-typefix revision)
- `zipformer_vi_encoder.onnx` / `zipformer_vi_decoder.onnx` / `zipformer_vi_joiner.onnx` / `zipformer_vi_tokens.txt`
- `zipformer_en_encoder.onnx` / `zipformer_en_decoder.onnx` / `zipformer_en_joiner.onnx` / `zipformer_en_tokens.txt`
- `zipformer_zh_encoder.int8.onnx` / `zipformer_zh_decoder.onnx` / `zipformer_zh_joiner.int8.onnx` / `zipformer_zh_tokens.txt`
- `silero_vad.onnx` (VAD; missing → energy-gate fallback)
- `mms_tts_vi.onnx` + `mms_tts_vi_vocab.json` (MMS-TTS Vietnamese)
- `mms_tts_en.onnx` + `mms_tts_en_config.json` + `mms_tts_en_vocab.json` (MMS-TTS English)

> APK size notes: the ~1.2 GB HuggingFace `hf_cache/` folder (a build-time
> byproduct the app never reads) is excluded via `aaptOptions.ignoreAssetsPattern
> "hf_cache"`. Only arm64-v8a native libs are included.

> There is **no `mms_tts_zh` asset** — Meta never released an MMS-TTS Mandarin
> checkpoint, so Chinese TTS uses the device's Android **system**
> TextToSpeech engine (`synthesizeToFile` → played like any other output).
> Requirements: a TTS engine with Chinese voice data installed (usually Google
> TTS → Install voice data → 普通话); on emulators use a Google APIs / Play
> Store image (plain AOSP ships no TTS engine).

> The `mms_tts_*` files are optional — if a model isn't bundled, `TTSModule`
> falls back to the Android system TextToSpeech engine for that language (which
> is always the path used for Chinese).

## Benchmarking On-Device

```powershell
adb logcat -s HyMtGgufJNI TranslationModule PipelineOrchestrator StreamingPipeline LanguageRouter AsrMetrics
```

- `decode: prompt=N tok (prefill …ms) + M tok (gen …ms, …tok/s)` — translation
- `Translation: "…" (…ms)` / `MEM[stage] totalPss=…MB` — per-stage pipeline stats

## Architecture

- **Modular pipeline stages** — clear I/O contracts between ASR, Translation, TTS
- **Decoder-only translation** — instruction-prompted Hy-MT1.5 (no seq2seq
  encoder graphs, no SentencePiece BPE, no beam search)
- **True streaming ASR** — Zipformer transducer with incremental state,
  160 ms scheduler, async LID + router + 640 ms rollback (no WAV staging)
- **sherpa-onnx** for streaming ASR; **ONNX Runtime** for VAD/TTS;
  **llama.cpp** for translation
- **Streaming toggle input** — mic stays open while live; VAD endpoints
  finalize utterances; single-thread executor with latest-wins cancel
- **Two-backend TTS** — neural MMS-TTS (ONNX) for VN/EN when bundled, with the
  Android system TextToSpeech engine as an automatic fallback (and the only
  backend for Chinese)

## Known Issues & Limitations (streaming ASR)

Device-validated unless noted; logic gates (`test_07`, 36 tests, bench A–D) green.

1. **UNKNOWN loop without an acoustic LID model (open, root cause known).**
   Every utterance starts `UND` and bootstrap LID must commit from audio
   alone — but the interim heuristic is capped below the 0.70 gate by
   design, so all traffic falls into the dual-candidate fallback. That
   fallback decodes a **cold 400 ms window with no drain decodes**, which on
   the real streaming Zipformer almost always yields empty hypotheses →
   `speculative candidates inconclusive, staying UNKNOWN` repeats, endpoint
   finalizes empty, and full English sentences produce no output (only
   dictionary-strong tokens like `HELLO` escape via the single-unit lexical
   exception). Landed mitigations (in this tree, device retest pending):
   provisional VI speculative decode while UNKNOWN with adopt-or-rollback on
   commit, rotating candidate pairs, endpoint backstop (pair → leftover →
   full-utterance verification). Remaining plan: feed candidates up to ~2 s
   of left-context + drain loop, then retune the 0.35/0.30/0.50 bars down
   (gated by the anti-garbage tests). A trained tiny LID remains the real
   fix (roadmap in `docs/streaming_asr.md`).
2. **sherpa `GetFrames` race → native crash (open, fix planned).** Observed
   once after long use: `features.cc:GetFrames:188 / 6208 + 45 > 6248`
   (fatal, uncatchable) → `Channel is unrecoverably broken` → process death.
   Cause: `OnlineStream` is not thread-safe, but the audio thread
   (`decodeChunk`/`decodeProvisional`/endpoint) and the LID worker
   (`onCommitted`/`activateBootstrap`/`verifyCandidate`) touch the shared
   resident engines concurrently. Planned fix: serialize all engine access
   behind one lock (no thread-architecture change). Not observed in the
   16:46 session below (clean stop, no `GetFrames` warning).
3. **Full-English accuracy still weak (open).** Short EN works through the VI
   model (`YOU` @0.47 → translated `BẠN.`), but fluent EN sentences end
   UNKNOWN/empty per (1). Same remediation as (1).
4. **Endpoint trade-off (2000 ms silence).** Natural pauses no longer split
   sentences, but hands-free finals arrive ~2 s after speech stops (plus
   translation). Tapping Stop flushes immediately.
5. **sherpa/ORT version lockstep.** The app pins sherpa-onnx v1.13.4
   (vendored `android/app/libs/*.aar`, gitignored) with
   onnxruntime-android 1.27.0. Both ship `libonnxruntime.so` with ELF
   versioned symbols — bumping either side alone crashes at startup
   (`cannot locate symbol OrtGetApiBase`). Related rule: never decode
   without `isReadyToDecode()` — sherpa aborts the whole process on an
   under-buffered decode (no exception is thrown).
6. **Memory & startup.** Full stack (VI+EN+ZH + HyMT + TTS) measures
   ~1.35 GB PSS, stable across sessions (no leak observed); the 6 GB / 4 GB
   lazy buckets are designed but less validated on-device. Cold start loads
   the three Zipformer recognizers sequentially (~2.6 s VI→EN→ZH on-device),
   covered by the `LoadingActivity` screen.
7. **Metrics caveat.** `utterances` counts VAD endpoints *including* empty
   UND finals, so it overstates successfully decoded utterances; pair it
   with non-empty FINAL logs when measuring. `partial_p50 ≈ 160 ms`
   (meets targets) only reflects committed-stream partials.
8. **No ZH TTS asset** (unchanged) — see note under Required assets above;
   `No bundled MMS-TTS asset for [zh]` in logcat is expected, system TTS
   covers Chinese.
9. **Upstream caveat.** ORT 1.27.0 is reported to miscompute the zipformer2
   int8 encoder on Snapdragon 8 Elite Gen 5 (k2-fsa/sherpa-onnx#3845);
   fixed upstream in 1.28.0, which no sherpa Android release bundles yet.

### Field-log notes 2026-09-14 (build pid 485, OnePlus/Oppo device)

- Session starts `active=UND`, Silero VAD loads natively in ms, models
  resident (`>=8GB: VI+EN+ZH`), TTS published for vi/en, zh falls back to
  system TTS — all as designed.
- The UNKNOWN loop in this log comes from a **pre-fix APK**: no
  `provisional`/`bootstrap adopted` lines exist (the Option-A provisional
  decode in this tree had not been built yet) — retest with a fresh APK
  before concluding anything about current code.
- Safe to ignore in logcat: `Oplus*` / `PopupWindow` / `HWUI` /
  `SchedAssist open sharedFd Permission denied` / `AppOps attributionTag
  not declared` / `predictive settings is disabled` / `unregisterSystemUI…
  Receiver not registered` — OEM framework and system_server noise, not app
  faults. Useful filters stay: `StreamingPipeline AsrMetrics
  PipelineOrchestrator TranslationModule HyMtGgufJNI TTSModule`.
