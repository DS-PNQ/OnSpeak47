# OnSpeak47 — Omni Voice

**On-device speech translation for Vietnamese ↔ English and Vietnamese ↔ Chinese.**

Edge AI Challenge, Phase 2 | Public Services domain | Android-first, wearable-ready

> Snapshot: this README describes the project state **before VAD auto-listen
> was added** — input is push-to-talk only, with a post-record RMS silence
> gate. Translation runs on the Hy-MT1.5 GGUF path below.

---

## Pipeline

```
[User speaks]
    → Whisper Small — transcribes speech to text
    → Hy-MT1.5-1.8B-1.25bit GGUF — translates text between VN/EN/CN
    → MMS-TTS — converts translated text back to speech
[Translated speech plays]
```

Three models, three stages. No language-branching — every input goes through the same path.

**ASR runs streaming-class fast**: the Whisper decoder is a custom KV-cache
export (constant per-token cost instead of O(n²)) and the encoder accepts the
audio's real length (a 3 s clip no longer pays a full 30 s encoder window).
End-to-end ASR on ~3 s clips measured **~10× faster** (en 5515→533 ms,
vi 5160→570 ms, desktop ort 1.22) with quality equal or better — see
`docs/optimization_results.md`.

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
│   ├── asr_whisper.py         # Whisper Small ASR wrapper
│   ├── translation_hymt.py    # Hy-MT1.5 translation (greedy, single-call)
│   ├── tts_mms.py             # MMS-TTS speech synthesis
│   └── orchestrator.py        # ASR → Translation → TTS pipeline
│
├── tests_local/               # Local quality tests (no device needed)
│   ├── conftest.py            # Fixtures for all test modules
│   ├── test_01_asr.py         # Whisper transcription accuracy
│   ├── test_02_translation.py # Hy-MT translation scoring
│   ├── test_04_vizh_corpus.py # Large-corpus VI↔ZH evaluation
│   ├── test_05_pipeline.py    # End-to-end pipeline tests
│   ├── test_06_onnx_parity.py # ONNX parity GATE (Whisper assets)
│   ├── gen_parity_reference.py     # Generates the PyTorch greedy reference
│   ├── verify_preprocess_onnx.py   # whisper_preprocess.onnx sanity check
│   ├── diagnose_whisper_paths.py   # Decodes two fixtures on every asset variant
│   ├── baselines/             # Recorded regression baseline (auto-created)
│   ├── output/                # Latest gate/parity results (JSON)
│   └── data/
│       ├── parallel_sentences.json     # 40 hand-curated test sentences
│       └── audio_samples/              # ASR parity fixtures (WAV + reference text)
│
├── optimize/                  # Model asset preparation
│   ├── 07_prepare_hymt_gguf.py       # Fetch Hy-MT 1.25-bit GGUF + typefix
│   ├── 08_export_hymt_onnx_int4.py   # Reference: INT4 ONNX export (ORT GenAI)
│   ├── hymt_gguf_typefix.py          # Remap legacy STQ tensor-type codes
│   ├── 01_export_onnx.py     # Whisper ONNX export
│   ├── 07_preoptimize.py     # Offline ORT graph pre-opt (*.opt.onnx)
│   ├── 09_export_whisper_decoder_kv.py # KV-cached Whisper decoder (CURRENT)
│   ├── 10_export_whisper_encoder_dyn.py # Dynamic-length Whisper encoder (CURRENT)
│   └── export_mms_tts.py     # MMS-TTS export
│
├── android/                   # Android app (OmniVoice)
│   ├── app/src/main/
│   │   ├── java/com/omnivoice/onspeak47/
│   │   │   ├── OmniVoiceApp.java
│   │   │   ├── LoadingActivity.java
│   │   │   ├── TranslationActivity.java   # Push-to-talk UI + RMS silence gate
│   │   │   ├── pipeline/     # ASR, Translation (GGUF), HyMtGgufJNI bridge,
│   │   │   │                 # TTS, PipelineOrchestrator, Tokenizer
│   │   │   ├── audio/        # AudioRecorder (manual record), AudioPlayer
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

### ONNX parity GATE (Whisper assets)
Runs the exact on-device inference scheme against the bundled assets.
Use a python with **onnxruntime==1.22.0** (the version the app ships):

```powershell
python -m venv .venv-ort122
.venv-ort122\Scripts\pip install "onnxruntime==1.22.0" numpy pytest jiwer sacrebleu sentencepiece
.venv-ort122\Scripts\python -m pytest tests_local\test_06_onnx_parity.py -v
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
> Input is **push-to-talk**: hold the button to record; recordings below the
> RMS silence threshold are rejected before ASR (no VAD auto-listen yet).

Required assets:
- `Hy-MT1.5-1.8B-1.25bit.gguf` (translation, post-typefix revision)
- `whisper_encoder.opt.onnx` (dynamic-length int8)
- `whisper_decoder.opt.onnx` (KV-cached int8)
- `whisper_preprocess.onnx` / `whisper_postprocess.onnx` / `whisper_vocab.json`
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
adb logcat -s HyMtGgufJNI TranslationModule PipelineOrchestrator ASRModule
```

- `decode: prompt=N tok (prefill …ms) + M tok (gen …ms, …tok/s)` — translation
- `Translation: "…" (…ms)` / `MEM[stage] totalPss=…MB` — per-stage pipeline stats

## Architecture

- **Modular pipeline stages** — clear I/O contracts between ASR, Translation, TTS
- **Decoder-only translation** — instruction-prompted Hy-MT1.5 (no seq2seq
  encoder graphs, no SentencePiece BPE, no beam search)
- **Streaming-class ASR decode** — KV-cached zero-copy greedy decode with
  whole-sequence fallback; row-only logits argmax
- **Short-window encoder** — dynamic-length encoder input; short clips feed only
  their real mel frames (~10× less encoder compute for a 3 s clip)
- **ONNX Runtime** for Whisper inference (offline pre-optimized `*.opt.onnx`
  loaded with NO_OPT); **llama.cpp** for translation
- **Push-to-talk input** — single-thread pipeline executor with latest-wins
  cancel; silence rejected by an RMS energy gate after recording
- **Two-backend TTS** — neural MMS-TTS (ONNX) for VN/EN when bundled, with the
  Android system TextToSpeech engine as an automatic fallback (and the only
  backend for Chinese)
