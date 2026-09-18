# Streaming Multilingual ASR — VI / EN / ZH (implementation notes)

Reference spec: `D:\vids\AIC2026-week2\realtime_multilingual_asr_vi_en_zh_android.md`
(engineering targets, not Snapdragon benchmarks — re-measure on device).

This document describes what was built, how the pieces map to the spec, and
what remains device work. Python reference + Android implementation share the
same thresholds and state machines.

## Module map (spec §23)

| Android (`android/.../asr/`) | Python (`backend/streaming_asr/`) | Spec |
|---|---|---|
| `AudioCapture.java` | — (mic only exists on device) | §7 |
| `AudioRingBuffer.java` | `ring_buffer.py` | MUST 3, §14 |
| `VadEngine.java` | `vad.py` | §8 |
| `StreamingAsrEngine.java` | `model_manager.py` (`FakeEngine`) | MUST 1/2, §9 |
| `ZipformerModelManager.java` | `model_manager.py` | §5, MUST 7 |
| `LanguageIdEngine.java` | `lid.py` | §10–12, §16–17 |
| `LanguageRouter.java` | `router.py` | MUST 5, §12–15, §25 |
| `RollbackManager.java` | `rollback.py` | MUST 6, §14 |
| `PartialTranscriptManager.java` | `partial.py` | §18 |
| `AsrState.java` + `AsrLanguage.java` + `RouterState` | `config.py` | §6/9/11/12/19 |
| `AsrMetrics.java` | `metrics.py` | §26, §28 |
| `StreamingPipeline.java` | `pipeline.py` | §24 |
| `StreamingController.java` | — (Android lifecycle glue) | Phase 1–6 |

Streaming Zipformer is the production ASR path (Whisper removed in
`migrate/zipformer-only`).

## Key pipeline facts

- 16 kHz mono, 20 ms frontend frames, 160 ms ASR scheduler (320 ms
  `setAdaptiveChunk` fallback, OPTIONAL 5), Zipformer chunk16 baseline.
- LID every 600 ms stable / 400 ms uncertain, on a worker thread, never
  blocking ASR (spec §11, OPTIONAL 6).
- Router: `0.45·acoustic + 0.30·text + 0.15·confidence + 0.10·history`,
  switch needs `> 0.72`, `> active + 0.20`, persistent 200 ms (spec §12).
- Rollback 640 ms default (up to 960), candidate-only shadow decode, 1500 ms
  cooldown after a rejected candidate (avoids decode storms on stuck LID).
- Every utterance starts UNKNOWN: bootstrap acoustic LID commits the first
  model from audio alone via a two-tier gate (2026-09-16 v2) — FAST
  (relative `0.70` / margin `0.15` + absolute supported mass `≥ 0.45`, clean
  audio) or SLOW (same relative gate on the EMA + a full smoother depth of
  UNANIMOUS raw windows + foreign-argmax guard at `0.60`). The old
  "107-class argmax must be en/vi/zh" rule is gone: field logs showed device
  audio keeps supported abs at ~0.01 with a correct relative ranking, so the
  abs bar alone locked every utterance to UNKNOWN. Windows are
  onset-anchored (no trailing-silence dilution) and GROW with the utterance
  (1500 ms baseline → 1500 ms live cap, 200 ms hops, ≤6 submits per phase;
  endpoints use a 2500 ms window), then the buffered audio is replayed into
  the winner — no VI default, no transcript circularity. Uncertain → retries,
  then a one-shot ≤2-model speculative decode (0.65 + 0.10 margin bar,
  800 ms cooldown, rotating pair so the 3rd language is tried next pass;
  endpoint also tries the leftover).
- Option A provisional decode: while UNKNOWN the pipeline ALSO decodes
  continuously for live SPECULATIVE partials (~160 ms, never committed). It
  starts on resident VI but uncertain LID evidence steers it: a persistent
  lean to a resident engine moves the stream there (adopt works for EN/ZH
  too), a lean to a non-resident engine quiets the UI instead of rendering
  VI guesses. Same-language commit on a complete stream adopts it (no
  replay); otherwise discard + replay into the winner. Endpoint while UNKNOWN
  flushes the provisional residue, then falls back to pair + leftover with an
  acoustic-confidence backstop — utterances never get stuck with zero output,
  and weak evidence finalizes empty on UND instead of forcing VI.
- Referee mode (§6.1): after a WEAK commit (< 0.70) the growing-window
  acoustic LID keeps re-evaluating on an 800 ms speech cadence (fresh budget
  per commit, ≤6 passes). Agreement firms the decision; disagreement goes
  through the runtime hysteresis/rollback gate — never a hard switch.
- Threading: every streaming-engine call is serialized on one lock
  (`engineLock`); the audio thread and the LID worker share the resident
  engines, and unsynchronized access used to abort the process in sherpa
  `GetFrames` (fatal, uncatchable).
- Endpoint switches use a lower bar (`0.60`, margin `0.10`): boundaries
  rewrite nothing and spend no rollback decode (spec §13.1).
- Endpoint candidate verification: at each VAD endpoint (2000 ms silence,
  raised 600 → 1000 → 2000 so mid-sentence pauses don't split utterances;
  a deliberate Stop flushes immediately) the utterance audio (≤6 s) is replayed through the non-active models; a
  candidate wins on non-trivial text + 0.15 confidence margin. This is the
  inter-utterance escape hatch while no acoustic LID model is bundled —
  without it the flat acoustic prior caps every fused score below the 0.72
  intra gate and the active language can never change.
- Endpoint flush: sub-chunk scheduler residue is folded into the stream,
  then trailing silence is padded until a full chunk is decodable
  (bounded), and one guarded decode runs. Every decode stays behind
  `isReadyToDecode`: sherpa ABORTS the process on an under-buffered decode
  (its `GetFrames` logs a warning then exits — not an exception — so an
  unconditional flush decode is a crash bug, not an optimization).
  The padding doubles as right-context so held-out tail words emit into
  their own utterance instead of leaking into the next one.

## Roadmap — acoustic LID / Option C (intra-utterance switching)

Status: VOXLINGUA REFEREE LANDED (2026-09-16 fix; Java + Python mirror).
`VoxLinguaAcousticLidEngine` (107-class ECAPA ONNX, `optimize/12_fetch_voxlingua_lid.py`)
is the bootstrap + runtime referee; `HeuristicAcousticLidEngine` remains the
no-asset fallback. Field-log root causes fixed
(`docs/voxlingua_lid_diagnosis.md`):

- Fixed 600 ms window → growing window (1500 ms baseline → 2500 ms cap).
- Argmax-must-be-supported gate (+ max-sticky smoother) → absolute-mass gate
  (`isBootstrapConfident`: relative `0.70/0.15` AND supported abs `≥ 0.45`);
  global top now follows the latest window.
- VI-biased fallback → evidence-steered provisional (switch-to-resident /
  quiet), candidate bar `0.65 + 0.10` margin, non-VI-first tie-break,
  post-weak-commit referee through the runtime gate.
- Audio-thread / LID-worker engine sharing → serialized on one lock
  (GetFrames abort fix).

Mechanics (unchanged): `LanguageIdEngine.classifyBootstrap` is audio-only
(`0.90·acoustic + 0.10·prior`, side-effect free); `LanguageRouter` starts
`UND`/`UNKNOWN` with split bootstrap/runtime paths; `StreamingPipeline`
replays buffered audio into the winner, guards stale results with
`utteranceSeq`, runs ≤2 speculative candidates with 800 ms cooldown, and
resets to UNKNOWN per utterance. Python mirror + `test_07` + `bench` A–D
green (logic only — device numbers still required).

Remaining device work:

1. Validate FBank/ONNX parity with the SpeechBrain reference, then measure
   VI↔EN confusion on the §26 set (gates retune: abs bar, 1500/2500 windows).
2. Benchmark ECAPA + Zipformer latency/RAM/thermal on the Snapdragon target;
   confirm P50/P95 and the 0.72 intra gate with no VI lock-in.
3. Decide INT8 for ECAPA only after the FP32 parity check passes.
4. Acceptance: intra-utterance VI→EN switch <1.5 s audio-time; ZH whole
   utterances detected without CJK text evidence; no stale-callback or
   oscillation regressions in logcat.
- Transcript tiers SPECULATIVE → STABLE (N=2 survival) → FINAL (spec §18).

## Tuning notes (found by `test_07` + `bench`, kept in code)

1. **EMA gates latency, not just noise.** Gating the router on EMA-smoothed
   fused scores pushed switch latency into seconds. EMA now feeds the
   telemetry view (`smoothedScores()`); the router gates on fused raw
   scores — persistence + margin + shadow verification already reject flaps.
2. **Strip the committed prefix.** Transducer partials re-emit the whole
   hypothesis; without prefix subtraction the committed head was
   double-counted (caught as duplicated display text in tests).
3. **Window the LID text.** Scoring the full committed transcript pins
   evidence to the utterance's first language. Both LID inputs now cover the
   same recent span: 480 ms audio + last ~6 text tokens.
4. Simulated intra-utterance switch latency: **~820 ms audio-time**
   (`bench` scenario C).

## Models (spec §4) — fetch, don't commit

```powershell
python optimize/11_fetch_streaming_zipformer.py --list
python optimize/11_fetch_streaming_zipformer.py --all --stage-assets
```

| Lang | Asset set | Notes |
|---|---|---|
| VI | `hynt/Zipformer-30M-RNNT-Streaming-6000h` (HF) → `zipformer_vi_*` | streaming 30M, chunk 16/32/64; `model_type=zipformer2` |
| EN + ZH | `csukuangfj/k2fsa-zipformer-chinese-english-mixed` (HF) → `zipformer_mixed_*` | **one bilingual model for both slots**; INT8 encoder ~80 MB (fp32 decoder), `model_type=zipformer` (v1) — see below |
| VAD | `silero_vad.onnx` (release asset) | missing → energy-gate fallback |

`model_type` is **per asset set, never global** (`AsrState.transducerModelType`):
the VI export is zipformer v2 (`query_head_dims` present) while the mixed export
is zipformer v1 (`attention_dims` only). Decoding the mixed model with
`"zipformer2"` **aborts the process natively** — there is no Java exception to
catch (verified 2026-09-18 with sherpa-onnx Python from a clean process).
The EN and ZH router slots stay separate; they just resolve to the same files.

RAM buckets (§5): ≥8 GB preload VI+EN+ZH; 6 GB VI+EN (+ZH lazy);
≤4 GB VI (+EN/ZH lazy). One active decoder; candidate only in the rollback
window; `trimTo(active, candidate)` under pressure (never 3 parallel).

## Enablement (device validation order)

1. Fetch models (`optimize/11_fetch_streaming_zipformer.py --all --stage-assets`)
   and the sherpa AAR (`android/app/libs/sherpa-onnx-1.13.4.aar` from
   https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-1.13.4.aar),
   rebuild arm64. Do NOT bump sherpa/ORT independently — both sides must
   carry the same ELF symbol versions (see `app/build.gradle`).
2. `adb logcat -s StreamingPipeline LanguageRouter AsrMetrics` — check
   partial cadence ≈160 ms, `summaryJson` P50/P95.
3. Toggle Start in `TranslationActivity` — partials render live, each endpoint
   FINAL auto-translates → TTS (no WAV staging, no Whisper path).
4. Tune weights/thresholds on the §26 dataset; re-run scenarios A–D.

## Verification (no device needed)

```powershell
pip install pytest
python -m pytest tests_local/test_07_streaming_asr.py -v   # 22 tests
python -m backend.streaming_asr.bench                      # scenarios A–D
```

JVM mirror: `android/app/src/test/.../StreamingAsrUnitTest.java`
(`./gradlew :app:testDebugUnitTest` on a machine with the SDK).

## Definition of Done (spec §29) — status

- [x] 16 kHz streaming frontend, no WAV staging (`AudioCapture`, `StreamingPipeline`)
- [x] Continuous partials, SPECULATIVE/STABLE/FINAL tiers
- [x] VI/EN/ZH engine slots + RAM-bucket preload/lazy (`ZipformerModelManager`)
- [x] Async LID (400–600 ms) + router state machine + 640 ms rollback + shadow verification
- [x] Inter-utterance switching (endpoint bar) + intra-utterance candidate verification
- [x] Logic benchmark A–D green (`bench.py`); 22 python + 13 JVM tests green
- [ ] Real-model check: sherpa AAR + assets on Snapdragon target
- [ ] Device P50 < 350 ms / P95 < 500 ms, no dropped frames, stable memory (`AsrMetrics` log)
- [ ] 10–20 min thermal soak, no significant throttling
- [ ] Tune fusion weights/thresholds on the §26 VI/EN/ZH/code-switch set
- [ ] OPTIONAL 1/2: INT8 VI, QNN/NNAPI A/B only if it beats the CPU baseline
