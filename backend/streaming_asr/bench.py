#!/usr/bin/env python3
"""Logic benchmark for the streaming VI/EN/ZH pipeline (spec §26-§27).

Simulates scenarios A–D with scripted FakeEngines on a virtual clock and
reports *audio-time* telemetry: partial cadence, switch-detection latency,
endpoint behavior. This validates the state machine, NOT on-device RTF —
real P50/P95/RTF/CPU/RSS numbers must come from adb + the AsrMetrics log on
the Snapdragon target (see docs/streaming_asr.md).

Usage:
    python -m backend.streaming_asr.bench
    python backend/streaming_asr/bench.py
"""
from __future__ import annotations

import statistics

from .config import SCHEDULER_MS
from .model_manager import FakeEngine
from .pipeline import StreamingPipeline

STEP_MS = 20
SPEECH = [0.2] * 320
SILENCE = [0.0] * 320


def _run(p: StreamingPipeline, now: list, frames: int, frame) -> None:
    t = now[0]
    for _ in range(frames):
        t += STEP_MS
        now[0] = t
        p.on_frame(list(frame), t)


def _partials(p: StreamingPipeline):
    return [e for e in p.events if e[0] == "partial"]


def _partial_intervals_ms(p: StreamingPipeline) -> list:
    # Scheduler emits at most one partial per 160 ms of speech; count events
    # per 160 ms bucket as the cadence signal.
    return [SCHEDULER_MS] * max(0, len(_partials(p)) - 1)


def scenario_a_mono() -> dict:
    now = [0]
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            # Hook-driven (survives the bootstrap replay reset): text appears
            # only when audio was actually fed, like a real transducer.
            e.decode_hook = lambda fed: ("xin chào mọi người", 0.9) if fed >= 1600 else ("", 0.5)
        return e

    def acoustic_vi(audio):
        return {"vi": 0.9, "en": 0.05, "zh": 0.05}

    p = StreamingPipeline(clock=lambda: now[0], factory=factory,
                          acoustic_scorer=acoustic_vi)
    _run(p, now, 250, SPEECH)  # 5 s VI
    n = len(_partials(p))
    langs = {e[3] for e in _partials(p)}
    return {"partials_5s": n, "expected_min": 25,
            "ok": n >= 25 and langs == {"vi"}, "utterances": p.metrics.utterances}


def scenario_b_inter_utterance() -> dict:
    now = [0]
    engines = {}
    current = ["vi"]
    texts = {"vi": "hôm nay tôi đi học",
             "en": "now lets discuss the project",
             "zh": "然后我们开始"}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        # Candidate re-decode (endpoint verification resets engines, so the
        # scripted hypothesis must come from a decode hook, not injected
        # state): each engine decodes fluently only when fed its own
        # language, and mumbles low-confidence garbage otherwise — the
        # margin that lets verification tell them apart.
        e.decode_hook = (lambda fed, _l=lang: (texts[_l], 0.9)
                         if current[0] == _l else ("o fre e o", 0.3))
        return e

    def acoustic(audio):
        lang = current[0]
        return {l: (0.9 if l == lang else 0.05) for l in ("vi", "en", "zh")}

    p = StreamingPipeline(clock=lambda: now[0], factory=factory,
                          acoustic_scorer=acoustic)
    final_langs = []
    for lang, text in (("vi", texts["vi"]), ("en", texts["en"]),
                       ("zh", texts["zh"])):
        current[0] = lang
        # No manual inject: each engine decodes fluently only when fed its
        # own language (decode hooks above), and bootstrap acoustic LID
        # commits the right model from the audio at ~200 ms — the fakedemo2
        # fix (no VI lock, no endpoint escape hatch needed).
        _run(p, now, 150, SPEECH)   # 3 s speech
        _run(p, now, 120, SILENCE)  # 2400 ms endpoint (2000 ms bar)
    finals = [e for e in p.events if e[0] == "final"]
    final_langs = [f[2] for f in finals]
    return {"finals": len(finals), "expected_finals": 3,
            "final_langs": final_langs, "expected_langs": ["vi", "en", "zh"],
            "ok": len(finals) == 3 and final_langs == ["vi", "en", "zh"]}


def scenario_c_intra_utterance() -> dict:
    now = [0]
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "en":
            e.decode_hook = lambda fed: ("meeting with team to discuss project schedule", 0.9)
        return e

    def acoustic(audio):
        if now[0] >= 2000:
            return {"en": 0.95, "vi": 0.03, "zh": 0.02}
        return {"vi": 0.8, "en": 0.1, "zh": 0.1}

    p = StreamingPipeline(clock=lambda: now[0], factory=factory, acoustic_scorer=acoustic)
    t = 0
    for _ in range(100):  # 2 s VI
        t += STEP_MS
        now[0] = t
        engines["vi"].inject_partial("hôm nay tôi có", 0.85)
        p.on_frame(list(SPEECH), t)
    switch_at = None
    for _ in range(300):  # up to 6 s mixed
        t += STEP_MS
        now[0] = t
        engines["vi"].inject_partial("meeting with team to discuss project schedule", 0.25)
        engines["en"].inject_partial("meeting with team to discuss project schedule", 0.9)
        p.on_frame(list(SPEECH), t)
        if p.active_lang == "en":
            switch_at = t
            break
    latency = None if switch_at is None else switch_at - 2000
    return {"switched": switch_at is not None, "switch_latency_audio_ms": latency,
            "ok": switch_at is not None and latency <= 2000}


def scenario_d_noisy_silence() -> dict:
    now = [0]
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            e.decode_hook = lambda fed: ("xin chào", 0.9) if fed >= 1600 else ("", 0.5)
        return e

    def acoustic_vi(audio):
        return {"vi": 0.9, "en": 0.05, "zh": 0.05}

    p = StreamingPipeline(clock=lambda: now[0], factory=factory,
                          acoustic_scorer=acoustic_vi)
    _run(p, now, 50, [0.003] * 320)  # sub-threshold hum: VAD must suppress
    suppressed = len(_partials(p)) == 0
    _run(p, now, 50, SPEECH)         # real speech decodes
    return {"silence_suppressed": suppressed, "partials_after_speech": len(_partials(p)) > 0,
            "ok": suppressed and len(_partials(p)) > 0}


def main() -> int:
    results = {
        "A mono-language 5s VI": scenario_a_mono(),
        "B inter-utterance VI->EN->ZH": scenario_b_inter_utterance(),
        "C intra-utterance VI->EN": scenario_c_intra_utterance(),
        "D noisy silence suppression": scenario_d_noisy_silence(),
    }
    print("Streaming ASR logic benchmark (virtual clock, scripted engines)")
    print("=" * 64)
    all_ok = True
    for name, r in results.items():
        status = "PASS" if r.pop("ok") else "FAIL"
        all_ok = all_ok and status == "PASS"
        detail = ", ".join(f"{k}={v}" for k, v in r.items())
        print(f"[{status}] {name}: {detail}")
    print("=" * 64)
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
