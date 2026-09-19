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
buffer → lossless 512-sample VAD FIFO → Silero VAD (sample-clocked
endpointing) → 160 ms scheduler → active Zipformer → partial
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
│   ├── test_07_streaming_asr.py # Streaming ASR state-machine gate (51 tests)
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
│   │   │   │                 # VAD + VadWindowBuffer framing, LID, Router,
│   │   │   │                 # Rollback, ModelManager)
│   │   │   ├── audio/        # AudioPlayer (no recorder — mic via AudioCapture)
│   │   │   └── util/         # LanguageConfig, OrtSessionConfig, TensorUtils
│   │   ├── cpp/              # hymt_gguf_jni.cpp + CMakeLists (vendored llama.cpp)
│   │   └── res/              # Layouts, values, raw language XMLs
│   ├── app/src/test/java/    # JVM unit tests (40: streaming state machine,
│   │                         # VAD framing + endpoint)
│   ├── app/src/androidTest/  # Instrumented EN reset probe (device only)
│   └── build.gradle
│
├── docs/
│   ├── architecture.md           # Pipeline architecture overview
│   ├── streaming_asr.md          # Streaming ASR design + latency targets
│   ├── voxlingua_lid_diagnosis.md # LID bootstrap diagnosis (v1/v2 field logs)
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

### Android VAD/streaming unit tests (JVM — no device, no model assets)
```powershell
cd android
gradle testDebugUnitTest --offline    # Gradle 9.5 + JDK 17; Android Studio's Gradle pane works too
```

40 tests, all green: `StreamingAsrUnitTest` (32) plus the VAD suites
(`VadWindowBufferTest` 3, `VadEndpointTest` 5). The VAD suites read the real
16 kHz WAV fixture via the `onspeak.testAudio` system property wired in
`app/build.gradle`, so framing is exercised with actual PCM instead of
synthetic arrays. The instrumented EN reset probe in `app/src/androidTest/`
needs a connected device (`gradle connectedDebugAndroidTest`) and stages the
same fixture as an APK asset.

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
- `zipformer_en_zh_mixed_encoder.int8.onnx` / `zipformer_en_zh_mixed_decoder.onnx` / `zipformer_en_zh_mixed_joiner.int8.onnx` / `zipformer_en_zh_mixed_tokens.txt`
  — **one bilingual EN+ZH model** (`k2fsa-zipformer-chinese-english-mixed`,
  csukuangfj on HuggingFace), loaded as **one session** and keyed by
  `AsrModelType.EN_ZH`, so a LID flip between en and zh neither loads a model
  nor restarts a stream. Replaced the separate EN 2023-06-26 (70 MB) and ZH
  int8 2025-06-30 (161 MB) packages: one 80 MB encoder instead of two sessions
  and 231 MB of assets, and code-switched speech no longer needs a model
  switch (the shared BPE vocabulary handles it inside the model).
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
- **Two ASR models, not three** — `AsrModelType{VI, EN_ZH}`: the bilingual
  EN+ZH encoder serves both labels from **one** session, and the router only
  guards VI ↔ EN_ZH, so code-switched speech (`en`↔`zh` inside one utterance)
  triggers no rollback and no model switch
- **sherpa-onnx** for streaming ASR; **ONNX Runtime** for VAD/TTS;
  **llama.cpp** for translation
- **Streaming toggle input** — mic stays open while live; VAD endpoints
  finalize utterances; single-thread executor with latest-wins cancel
- **Two-backend TTS** — neural MMS-TTS (ONNX) for VN/EN when bundled, with the
  Android system TextToSpeech engine as an automatic fallback (and the only
  backend for Chinese)

## Known Issues & Limitations (streaming ASR)

Logic gates (`test_07`, 51 tests, bench A–D) green; Android JVM suites 51/51
green (including 11 tests pinning the `AsrModelType` bucket behaviour).
**On-device state is still not demo-clean**: language routing is no
longer the blocker (LID commits `en` on English audio), but the transcript
stays empty, so the 2026-09-17 session produced no usable output — see 3 and
the 2026-09-17 field log below.

1. **VoxLingua LID bootstrap: works on-device for `en`, sub-gate tuning
    open.** Diagnosed in `docs/voxlingua_lid_diagnosis.md` (v1 + v2 field
    logs): the fixed 600 ms window sat below the ECAPA evidence point, the
    107-class argmax gate rejected every short-window EN/ZH result, and the
    0.30 candidate bar committed Vietnamese on no evidence. Current state
    (this tree): growing onset-anchored window (1500 ms live cap, 2500 ms at
    endpoints), two-tier gate — FAST (absolute mass `≥ 0.45`) for clean audio,
    SLOW (relative `0.70/0.15` + 3-hop unanimity + foreign-argmax guard) for
    noisy device-mic audio where supported abs sits at ~0.01 with a correct
    relative ranking. The 2026-09-16 field log showed exactly that regime
    (`en-rel 0.77–0.87`, `enAbs 0.01–0.02` → UNKNOWN forever under the
    abs-only bar); the 2026-09-17 log shows the gate now **committing**
    (`en=0.9996 → bootstrap committed en conf=0.9998`). Follow-latest
    smoother, evidence-steered provisional (switch-to-resident / quiet),
    candidate bar `0.65 + 0.10` with non-VI-first tie-break, and a
    post-weak-commit acoustic referee round out the pipeline. Remaining:
    commit latency/accuracy on real VI/EN/ZH speech still varies by run
    (same English video produced `en 0.9996`, `en 0.8565`, `en 0.7408` and one
    leading `vi 0.7308`), so the gate must keep surviving that spread — tune
    (`VOXLINGUA_MIN_SUPPORTED_ABS_SCORE`, `FOREIGN_TOP_REJECT`, windows) on
    measured device distributions; clean-sample numbers do NOT transfer.
2. **sherpa `GetFrames` race → native crash (fix landed, soak retest
    pending).** Observed once after long use: `features.cc:GetFrames:188 /
    6208 + 45 > 6248` (fatal, uncatchable) → `Channel is unrecoverably
    broken` → process death. Cause: `OnlineStream` is not thread-safe, but
    the audio thread and the LID worker touched the shared resident engines
    concurrently. Fix in this tree: every engine call in `StreamingPipeline`
    is serialized on one leaf lock (`engineLock`, no thread-architecture
    change — live decode, bootstrap replay, candidate shadow decodes,
    endpoint verification, flushes and finals all go through it).
3. **Full-English accuracy still weak (open, same remediation as 1).**
    Short EN works through the VI model (`YOU` @0.47 → translated `BẠN.`),
    but fluent EN sentences end UNKNOWN/empty while bootstrap stays
    uncommitted. A 2026-09-16 field session also showed a late false VI
    commit (`adopted provisional vi conf=0.79` on English audio) — single
    windows can flip either way, which is why the slow path requires
    unanimity instead of just a lower bar. The 2026-09-17 log (below) isolates
    the remaining symptom more precisely: LID now commits `en` correctly
    (`voxlingua … en=0.9996`, `bootstrap committed en conf=0.9998`) yet every
    decode of that English video still logs an empty `text=` — so this is a
    **decode/transcript** problem, not a language-routing one. Beware the log
    reading: when the transcript is empty, `StreamingPipeline` reuses the
    previous `conf` value and `reset()` does not zero it either, so a repeated
    `conf=0.0638` does **not** prove a frozen decoder. A hypothesis that
    `reset()` leaves the native stream unusable was **not confirmed**: the
    Python reference (`sherpa_onnx`) produces normal text after
    `recognizer.reset(stream)`, and the dedicated
    `StreamingAsrResetInstrumentedTest` (identical PCM before/after reset) has
    **not been run yet — no device was attached when it was written**.
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
6. **Memory & startup.** The ~1.35 GB PSS figure was measured with the old
   separate EN (70 MB) + ZH (161 MB) encoders, each in its own session. The
   swap replaces both with **one** 80 MB int8 bilingual encoder, and the model
   pool is now keyed by `AsrModelType` (`VI`, `EN_ZH`) instead of by language —
   so EN and ZH share **one** session instead of allocating two over the same
   file. Assets ~231 MB → ~80 MB; resident sessions 3 → 2; EN↔ZH label flips
   no longer restart a stream. The new PSS baseline has **not** been
   re-measured on-device yet. Cold start loads two recognizers (VI + the shared
   mixed model), covered by the `LoadingActivity` screen.
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
10. **VAD framing + endpoint clock (fixed, unit-tested; device retest open).**
    `VadEngine` used to stretch each 20 ms capture frame (320 samples) onto the
    512-sample Silero window by nearest-neighbour index scaling, so duplicated
    and skipped samples reached the model and the model never saw contiguous
    audio; the helper also dropped any residual samples, so window boundaries
    could never align with capture boundaries. The energy fallback path (no
    native session, no injected scorer) still scores each capture frame
    directly, so asset-less/test behaviour is unchanged. Now `VadWindowBuffer`
    is a lossless FIFO:
    only complete, contiguous 512-sample windows are scored, the partial tail
    waits for the next frame, and `reset()` clears it. Endpoint decisions are
    counted in **captured samples** rather than `System.currentTimeMillis()`,
    so a stalled worker thread or a wall-clock jump can no longer finalize an
    utterance early; `VadEngine.reset()` also clears the leftover FIFO, the
    last probability and the gate state. Covered by 8 JVM tests
    (`VadWindowBufferTest` 3, `VadEndpointTest` 5) — 40/40 green. **Not
    proven on-device:** the endpoint suites inject a deterministic scorer to
    test framing/timing, so they say nothing about Silero accuracy, and the
    empty-EN-transcript symptom (3) is untouched by this fix.
11. **Reading the `audioDBG` heartbeat.** `speechRmsAvg` and `speechMaxVad` are
    **per-heartbeat** values (`logDbg()` zeroes the running sum and peak every
    2 s) while `frames`/`speech` are cumulative — so a heartbeat that added no
    new speech-classified frame prints `speechRmsAvg=0.0000 speechMaxVad=0.0`
    with `speech` unchanged. That is the VAD gate not re-triggering, not a dead
    microphone. Likewise `ysProbs n=N min=… max=…` is a per-decode joiner-stat
    trace, not an error.

### Field-log notes 2026-09-17 (pid 1090, OnePlus/Oppo device, same English test video)

- Startup is unchanged and clean: three Zipformer recognizers load in ~1 s
  each, HyMT GGUF in 396 ms, MMS-TTS published for vi/en (zh → system TTS),
  `Ready!` ~4 s after launch.
- **LID is no longer the blocker.** On the English video the first VoxLingua
  inference is decisive (`vi=0.0004 en=0.9996`, `enAbs=0.0100`) and the
  pipeline commits it: `bootstrap committed en conf=0.9999 replayMs=1060`;
  the 50th inference still reports `en=0.9922`. The language stays `en` for the
  whole session.
- **Yet every decode is empty.** After the commit, `decode lang=en
  conf=0.16350022 text=` repeats ~8×/s for ~20 s with no text at all, while
  the VAD heartbeat of that window shows real speech
  (`frames=100 speech=48 decodes=6 vadProb=0.9982 speechRmsAvg=0.0417
  speechMaxVad=0.9987`). Text only appears at 21:51:09 — and it starts
  **mid-word**: `NG` → `NGERS` → `NGERS OF`, unchanged for the last 10 s of
  the session.
- That fragment is what got translated (`Translation: "NGERS OF." (574 ms)` →
  TTS → playback at 21:51:24). So the pipeline's plumbing works end to end;
  the ASR text is the defect.
- The second run reproduces the failure in a cleaner form: LID commits
  `en conf=0.7408 replayMs=860`, seven empty decodes, then
  `endpoint reached, active=en` at 21:51:26.998 — an **empty final**, so
  neither Translation nor TTS ran (`utterances=1` with no output). The
  heartbeats afterwards show the expected post-endpoint silence shape:
  `frames` keeps counting, `speech=106`, `decodes=12` and `conf` frozen.
- Session-to-session variance is real: an earlier run in the same log booked
  `bootstrap committed en conf=0.7202 replayMs=2960` and another
  `en conf=0.8565 replayMs=1060`, and one run's first inference was
  `vi=0.7308` on the same English audio — the gate's job is now to survive
  that, not to be bypassed.
- Reading this log safely: an empty transcript keeps the **previous** `conf`
  value (`conf=0.0638` repeated for minutes is stale, not a measurement), and
  `reset()` does not clear it. See Known Issues 3 for why the mid-life-reset
  hypothesis was not adopted, and 11 for the heartbeat fields.
- Safe to ignore (same OEM noise as 09-16): `Oplus*`, `HWUI`,
  `AppOps attributionTag not declared`, `unregisterSystemUIBroadcastReceiver
  failed`, `predictive settings is disabled`.

### Field-log notes 2026-09-16 (pid 12666, OnePlus/Oppo device, post-v1 APK)

- VoxLingua session loads fine (`featureInput=features wavLensInput=wav_lens
  outputs=[probabilities]`), but live inferences report flat 107-way mass
  with correct relative ranking (`en 0.77–0.87`, `enAbs 0.01–0.02`,
  `globalTop=nn 0.09–0.26`) → abs-only gate never opens → `speculative
  candidates inconclusive, staying UNKNOWN` repeats → empty finals. This
  motivated the v2 slow path (relative + unanimity).
- One utterance committed a late false VI (`adopted provisional vi
  conf=0.79` on English audio, final `HEO.` → translated) — single-window
  evidence flips both ways; persistence is required, not just a lower bar.
- End-to-end cost in this log (`Pipeline total: 2661 ms = Translation
  737 ms + TTS 1787 ms`, plus the 2000 ms endpoint silence) sits mostly
  outside ASR once commits flow early again; MT/TTS are the next latency
  lever after LID accuracy.

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
