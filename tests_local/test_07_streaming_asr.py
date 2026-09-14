"""Streaming multilingual ASR tests — spec §26-§29 (no models/network needed).

Covers the MUST-path with scripted FakeEngines:
ring buffer/rollback window, VAD endpointing, staged LID, router hysteresis,
partial tiers, metrics percentiles, RAM-bucket preload, and the §24-§25
pipeline (mono partials, inter-utterance switch, intra-utterance rollback).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.streaming_asr import (  # noqa: E402
    SAMPLE_RATE, SCHEDULER_SAMPLES, SWITCH_PERSIST_MS,
    AudioRingBuffer, VadEngine, LanguageIdEngine, LidResult, LanguageRouter,
    PartialTranscriptManager, RollbackManager, AsrMetrics,
    ZipformerModelManager, FakeEngine, StreamingPipeline,
)


def make_clock():
    now = [0]
    return now, lambda: now[0]


# --- MUST 3: ring buffer + rollback window -------------------------------

def test_ring_buffer_last_window_sizes():
    ring = AudioRingBuffer.with_capacity_ms(30_000)
    ring.append([0.1] * SAMPLE_RATE)  # 1 s
    assert ring.available() == SAMPLE_RATE
    w640 = ring.last_ms(640)
    w960 = ring.last_ms(960)
    assert len(w640) == SAMPLE_RATE * 640 // 1000 == 10240
    assert len(w960) == SAMPLE_RATE * 960 // 1000
    assert ring.dropped_count() == 0


def test_ring_buffer_overwrite_counts_dropped():
    ring = AudioRingBuffer(1000)
    ring.append([0.0] * 1500)
    assert ring.available() == 1000
    assert ring.dropped_count() == 500
    assert len(ring.last(640)) == 640


def test_rollback_window_defaults_to_640ms():
    ring = AudioRingBuffer.with_capacity_ms(30_000)
    ring.append([0.2] * SAMPLE_RATE * 2)
    rb = RollbackManager()
    assert rb.window_ms == 640
    audio = rb.start_rollback(ring, "en", now_ms=1000)
    assert len(audio) == SAMPLE_RATE * 640 // 1000
    assert rb.has_pending and rb.candidate == "en"


# --- VAD: speech gating + endpointing -------------------------------------

def test_vad_speech_and_endpoint():
    vad = VadEngine()
    t = 0
    speech = [0.2] * 320
    r = vad.process(speech, t)
    assert r.is_speech and r.speech_start
    # 1000 ms trailing silence finalizes the utterance — the endpoint fires
    # exactly once mid-silence, so observe every frame, not just the last.
    silence = [0.0] * 320
    saw_end = False
    for i in range(120):  # 2400 ms
        t += 20
        end = vad.process(silence, t)
        saw_end = saw_end or end.speech_end
    assert saw_end
    assert not vad.in_speech


def test_vad_silence_suppresses_decode():
    vad = VadEngine()
    r = vad.process([0.0] * 320, 0)
    assert not r.is_speech


# --- LID: CJK + lexical evidence, adaptive cadence ------------------------

def test_lid_zh_boosted_by_cjk_characters():
    lid = LanguageIdEngine()
    r = lid.classify([0.01] * 100, "你好我今天去开会", 0.5, "vi")
    assert r.language == "zh"
    assert r.scores["zh"] > 0.5


def test_lid_en_lexical_evidence():
    lid = LanguageIdEngine()
    lid.classify([0.01] * 100, "hôm nay tôi có meeting với team", 0.5, "vi")
    r = lid.classify([0.01] * 100, "the final report tomorrow please", 0.8, "vi")
    assert r.scores["en"] > r.scores["vi"]


def test_lid_interval_adaptive():
    assert LanguageIdEngine.lid_interval_ms(False) == 600
    assert LanguageIdEngine.lid_interval_ms(True) == 400


# --- Router: hysteresis + state machine ------------------------------------

def test_router_rejects_weak_candidate():
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    router.set_active("vi", 0.5)
    d = router.on_lid_result(LidResult("en", 0.5, {"en": 0.5, "vi": 0.3, "zh": 0.2}))
    assert d == "HOLD"


def test_router_requires_margin_over_active():
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    router.set_active("vi", 0.65)
    # 0.72 threshold ok but margin 0.20 not met (0.72 < 0.65+0.20).
    d = router.on_lid_result(LidResult("en", 0.72, {"en": 0.72, "vi": 0.2, "zh": 0.08}))
    assert d == "HOLD"


def test_router_persistence_then_rollback_then_commit():
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    router.set_active("vi", 0.3)
    lid = LidResult("en", 0.85, {"en": 0.85, "vi": 0.1, "zh": 0.05})
    assert router.on_lid_result(lid) == "OBSERVE"
    now[0] += SWITCH_PERSIST_MS - 50
    assert router.on_lid_result(lid) == "OBSERVE"  # not persistent yet
    now[0] += 100
    assert router.on_lid_result(lid) == "START_ROLLBACK"
    router.set_verifier(lambda c, audio: __import__("backend.streaming_asr.router", fromlist=["Verification"]).Verification(True, "is very beautiful", 0.8))
    assert router.verify_and_commit([0.0] * 100) == "COMMITTED"
    assert router.active == "en"
    assert router.switch_count == 1


def test_router_discards_failed_verification():
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    router.set_active("vi", 0.3)
    lid = LidResult("en", 0.9, {"en": 0.9, "vi": 0.05, "zh": 0.05})
    router.on_lid_result(lid)
    now[0] += SWITCH_PERSIST_MS + 10
    assert router.on_lid_result(lid) == "START_ROLLBACK"
    router.set_verifier(lambda c, audio: __import__("backend.streaming_asr.router", fromlist=["Verification"]).Verification(False, "", 0.1))
    assert router.verify_and_commit([0.0] * 100) == "DISCARDED"
    assert router.active == "vi"


def test_router_endpoint_switch_no_rollback():
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    router.set_active("vi", 0.4)
    assert router.on_endpoint({"en": 0.8, "vi": 0.1, "zh": 0.1}) is True
    assert router.active == "en"


def test_router_endpoint_bar_lower_than_intra_gate():
    # 0.65 would never pass the intra-utterance 0.72 + persistence gate, but
    # at a VAD boundary switching is cheap (spec §13.1) so it commits.
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    router.set_active("vi", 0.4)
    assert router.on_lid_result(LidResult("en", 0.65, {"en": 0.65, "vi": 0.25, "zh": 0.1})) == "HOLD"
    assert router.on_endpoint({"en": 0.65, "vi": 0.25, "zh": 0.1}) is True
    assert router.active == "en"


# --- Partial tiers ---------------------------------------------------------

def test_partial_stabilizes_after_two_updates():
    pm = PartialTranscriptManager()
    pm.update_speculative("xin")
    assert pm.tier == "SPECULATIVE"
    pm.update_speculative("xin")
    pm.update_speculative("xin chào")
    pm.update_speculative("xin chào")
    assert "xin" in pm.committed  # head promoted after N=2 survivals
    assert pm.display.startswith("xin")


def test_partial_rollback_and_commit():
    pm = PartialTranscriptManager()
    pm.update_speculative("hôm nay trời")
    pm.rollback_speculative()
    assert pm.speculative == ""
    pm.commit("is very beautiful")
    assert pm.committed == "is very beautiful"
    pm.finalize("Hôm nay trời is very beautiful")
    assert pm.tier == "FINAL"


# --- Metrics ---------------------------------------------------------------

def test_metrics_percentiles_and_targets():
    m = AsrMetrics()
    for v in [200, 250, 300, 350, 400]:
        m.add_partial_latency(v)
    assert m.partial_p(50) == 300
    assert 200 <= m.partial_p(90) <= 400
    assert m.meets_targets() is True  # p50=300<350, p95≈390<500
    m2 = AsrMetrics()
    for v in [400, 500, 600, 700, 800]:
        m2.add_partial_latency(v)
    assert m2.meets_targets() is False


# --- Model manager buckets --------------------------------------------------

def test_model_manager_ram_buckets():
    assert {e.lang for e in ZipformerModelManager(8.0).preload_for_device()} == {"vi", "en", "zh"}
    assert {e.lang for e in ZipformerModelManager(6.0).preload_for_device()} == {"vi", "en"}
    assert [e.lang for e in ZipformerModelManager(4.0).preload_for_device()] == ["vi"]


def test_model_manager_trims_to_active_plus_candidate():
    mm = ZipformerModelManager(8.0)
    mm.preload_for_device()
    mm.trim_to("vi", "en")
    assert mm.is_resident("vi") and mm.is_resident("en")
    assert not mm.is_resident("zh")  # never 3 parallel decodes (spec §31)


# --- Pipeline: scheduler cadence --------------------------------------------

def _speech_frame(amp=0.2):
    return [amp] * 320


def test_pipeline_emits_partials_every_160ms():
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        return e

    # Confident VI acoustic LID: bootstrap commits VI at ~200 ms, then live
    # decoding proceeds on the VI engine (fakedemo2 §7: audio decides, and
    # the buffered 200 ms are replayed — not dropped).
    p = StreamingPipeline(clock=clock, factory=factory,
                          acoustic_scorer=lambda audio: {"vi": 0.9, "en": 0.05, "zh": 0.05})
    assert p.active_lang == "und"  # UNKNOWN until bootstrap (§6)
    vi = engines["vi"]
    # 1 s of speech: 50 frames x 20 ms → ~6 scheduler chunks → ≥4 partials.
    t = 0
    n_partials = 0
    for i in range(50):
        t += 20
        now[0] = t
        if i == 20:
            vi.inject_partial("xin chào", 0.8)
        p.on_frame(_speech_frame(), t)
        n_partials = len([e for e in p.events if e[0] == "partial"])
    assert n_partials >= 4
    assert p.metrics.partial_latencies, "expected partial latency samples"


def test_pipeline_intra_utterance_switch_with_rollback():
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "en":
            # Shadow decode: after reset(), decoding the rollback window
            # yields the English span with high confidence (what a real
            # Zipformer candidate would produce on code-switched audio).
            e.decode_hook = lambda fed: ("meeting with team to discuss project schedule", 0.9)
        return e

    # Acoustic scorer forces EN after t=1200 ms to simulate code-switch.
    def acoustic(audio):
        if now[0] >= 1200:
            return {"en": 0.95, "vi": 0.03, "zh": 0.02}
        return {"vi": 0.8, "en": 0.1, "zh": 0.1}

    p = StreamingPipeline(clock=clock, factory=factory, acoustic_scorer=acoustic)
    t = 0
    # Phase 1: VI speech.
    for _ in range(30):  # 600 ms
        t += 20
        now[0] = t
        engines["vi"].inject_partial("hôm nay tôi có", 0.85)
        p.on_frame(_speech_frame(), t)
    assert p.active_lang == "vi"
    # Phase 2: code-switched audio. The English span dominates the partial
    # ("...meeting with team to discuss project schedule") so acoustic LID,
    # text evidence and the confidence drop all agree — the sustained,
    # unanimous signal the 0.72 + 0.20 hysteresis gate requires. The
    # candidate EN model decodes the 640 ms rollback window.
    engines["en"].inject_partial("meeting with team to discuss project schedule", 0.9)
    switched = False
    for _ in range(200):  # up to 4000 ms; LID cadence 400 ms under uncertainty
        t += 20
        now[0] = t
        engines["vi"].inject_partial("meeting with team to discuss project schedule", 0.25)
        engines["en"].inject_partial("meeting with team to discuss project schedule", 0.9)
        p.on_frame(_speech_frame(), t)
        if p.active_lang == "en":
            switched = True
            break
    assert switched, "router should commit EN via rollback verification"
    kinds = [e[0] for e in p.events]
    assert "switch" in kinds


def test_pipeline_endpoint_finalizes_utterance():
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        return e

    p = StreamingPipeline(clock=clock, factory=factory)
    t = 0
    for _ in range(10):
        t += 20
        now[0] = t
        engines["vi"].inject_partial("xin chào", 0.9)
        p.on_frame(_speech_frame(), t)
    # 2400 ms silence → endpoint (2000 ms bar) → FINAL.
    for _ in range(120):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals, "expected a FINAL event after endpoint silence"
    assert p.metrics.utterances >= 1


# --- Bootstrap acoustic LID (fakedemo2 §6–§9, §46 acceptance) ---------------

def test_router_starts_unknown_and_gates_bootstrap():
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    # §6: never ACTIVE+VI at start — UNKNOWN until bootstrap commits.
    assert router.active == "und"
    assert router.is_bootstrapping()
    # Runtime hysteresis must not fire while bootstrapping (§23).
    assert router.on_lid_result(
        LidResult("en", 0.95, {"en": 0.95, "vi": 0.03, "zh": 0.02})) == "HOLD"
    # Confident VI (0.82, margin 0.69) commits.
    d = router.on_bootstrap_lid_result(
        LidResult("vi", 0.82, {"vi": 0.82, "en": 0.13, "zh": 0.05}))
    assert d == "vi" and router.active == "vi"
    assert not router.is_bootstrapping()


def test_router_bootstrap_uncertain_stays_unknown():
    now, clock = make_clock()
    router = LanguageRouter(clock=clock)
    # VI 0.46 / EN 0.43: must NOT force VI on max-probability (§8).
    d = router.on_bootstrap_lid_result(
        LidResult("vi", 0.46, {"vi": 0.46, "en": 0.43, "zh": 0.11}))
    assert d == "und" and router.active == "und"
    assert router.is_bootstrapping()
    # Below-threshold top1 also stays UNKNOWN.
    router.reset()
    d = router.on_bootstrap_lid_result(
        LidResult("en", 0.65, {"en": 0.65, "vi": 0.25, "zh": 0.10}))
    assert d == "und"


def test_bootstrap_uses_audio_not_transcript():
    # Even a strongly VI transcript must not sway bootstrap: audio decides.
    lid = LanguageIdEngine(
        acoustic_scorer=lambda audio: {"en": 0.9, "vi": 0.05, "zh": 0.05})
    r = lid.classify_bootstrap([0.2] * 4800)
    assert r.language == "en" and r.confidence >= 0.70
    # No scorer → flat audio prior → uncertain (extend/candidates, §17).
    flat = LanguageIdEngine(acoustic_scorer=None)
    # NOTE: default pipeline wires the heuristic; a bare None scorer is the
    # "no acoustic model" unit case and must stay flat, never VI-locked.
    flat.acoustic_scorer = None
    r = flat.classify_bootstrap([0.2] * 4800)
    assert abs(r.scores["vi"] - r.scores["en"]) < 1e-6


def test_pipeline_en_utterance_starts_with_en_recognizer():
    """§46: EN-only utterance starts with the EN recognizer (no VI smear)."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "en":
            e.decode_hook = lambda fed: ("let us start the meeting", 0.9) if fed >= 1600 else ("", 0.5)
        if lang == "vi":
            e.decode_hook = lambda fed: ("let gi ta mit ting", 0.3)
        return e

    p = StreamingPipeline(clock=clock, factory=factory,
                          acoustic_scorer=lambda audio: {"en": 0.86, "vi": 0.09, "zh": 0.05})
    t = 0
    for _ in range(15):  # 300 ms — bootstrap commits at ~200 ms
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    assert p.active_lang == "en", "bootstrap must commit EN from audio"
    for _ in range(15):
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    for _ in range(120):  # endpoint
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and "meeting" in finals[-1][1]
    assert "let gi" not in finals[-1][1], "VI phonetic smear leaked into final"
    assert finals[-1][2] == "en"


def test_pipeline_uncertain_bootstrap_uses_at_most_two_models():
    """§10/§32: uncertain bootstrap decodes ≤2 candidates, never 3."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            e.decode_hook = lambda fed: ("xin chào mọi người", 0.85) if fed >= 1600 else ("", 0.5)
        return e

    p = StreamingPipeline(clock=clock, factory=factory)  # heuristic: uncertain by design
    t = 0
    for _ in range(40):  # 800 ms of VI speech
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    # Live phase only (before endpoint): at most 2 engines may have decoded.
    fed = {lang: e.fed_samples for lang, e in engines.items()}
    assert sum(1 for v in fed.values() if v > 0) <= 2, fed
    for _ in range(120):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and "xin chào" in finals[-1][1]


def test_pipeline_short_blip_never_forces_vi():
    """§35: a 60 ms blip with no evidence finalizes empty on UND."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        return e

    p = StreamingPipeline(clock=clock, factory=factory)
    t = 0
    for _ in range(3):  # 60 ms — below BOOTSTRAP_MIN_MS
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    for _ in range(120):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals, "expected a FINAL after endpoint silence"
    assert finals[-1][1] == "" and finals[-1][2] == "und"
    assert p.active_lang == "und"


# --- Provisional speculative decode (Option A hotfix) --------------------

def test_provisional_partials_flow_while_unknown():
    """Partials must be visible ~160 ms after speech starts even though the
    language is not committed yet (the stuck-UNKNOWN regression)."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            e.decode_hook = lambda fed: ("xin chào", 0.8) if fed >= 1600 else ("", 0.5)
        return e

    p = StreamingPipeline(clock=clock, factory=factory)
    t = 0
    for _ in range(9):  # 180 ms — below BOOTSTRAP_MIN_MS, still UNKNOWN
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    assert p.active_lang == "und"
    partials = [e for e in p.events if e[0] == "partial"]
    assert partials, "provisional decode must emit partials while UNKNOWN"
    assert partials[-1][2] == "xin chào"
    # ... but nothing is committed yet (§11).
    assert p.transcripts.committed == ""


def test_provisional_adopted_on_same_language_commit():
    """Bootstrap VI adopts the live provisional stream: final keeps the
    early words (no replay gap, no duplication)."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            e.decode_hook = lambda fed: ("xin chào mọi người", 0.9) if fed >= 1600 else ("", 0.5)
        return e

    p = StreamingPipeline(clock=clock, factory=factory,
                          acoustic_scorer=lambda audio: {"vi": 0.9, "en": 0.05, "zh": 0.05})
    t = 0
    for _ in range(30):
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    assert p.active_lang == "vi"
    for _ in range(120):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and finals[-1][1] == "xin chào mọi người"
    assert finals[-1][2] == "vi"


def test_provisional_discarded_on_other_language_commit():
    """Provisional VI garbage must leave no trace in an EN final."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            e.decode_hook = lambda fed: ("o a e o", 0.3) if fed >= 1600 else ("", 0.5)
        if lang == "en":
            e.decode_hook = lambda fed: ("good morning team", 0.9) if fed >= 1600 else ("", 0.5)
        return e

    p = StreamingPipeline(clock=clock, factory=factory,
                          acoustic_scorer=lambda audio: {"en": 0.9, "vi": 0.05, "zh": 0.05})
    t = 0
    for _ in range(30):
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    assert p.active_lang == "en"
    for _ in range(120):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and "good morning" in finals[-1][1]
    assert "o a e" not in finals[-1][1]
    assert finals[-1][2] == "en"


def _run_utterance(p, now, engines, partials, silence_frames=120):
    """Feed speech partials then trailing silence; returns final time."""
    t = now[0]
    for text, conf in partials:
        t += 20
        now[0] = t
        engines["vi"].inject_partial(text, conf)
        p.on_frame(_speech_frame(), t)
    for _ in range(silence_frames):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    return t


def test_endpoint_verification_switches_to_better_candidate():
    """Whole English utterance bootstraps straight to EN from audio.

    Before the fakedemo2 fix the pipeline decoded everything with the VI
    engine first and needed endpoint replay as an escape hatch. Now the
    bootstrap acoustic LID commits EN at ~200 ms from the audio alone, so
    the utterance never locks to the wrong model in the first place.
    """
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "en":
            e.decode_hook = lambda fed: ("now lets discuss the project plan", 0.9)
        return e

    def acoustic_en(audio):
        return {"en": 0.9, "vi": 0.05, "zh": 0.05}

    p = StreamingPipeline(clock=clock, factory=factory, acoustic_scorer=acoustic_en)
    assert p.active_lang == "und"
    # VI decodes English audio as low-confidence garbage.
    partials = [("o fre e o", 0.3)] * 30  # 600 ms of speech
    _run_utterance(p, now, engines, partials)
    assert p.active_lang == "und", "next utterance restarts UNKNOWN (§45)"
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and "discuss" in finals[-1][1]
    assert finals[-1][2] == "en", "bootstrap must commit EN from audio"


def test_endpoint_verification_keeps_active_when_candidate_weak():
    """No confident candidate → VI utterance finalizes VI, no switch."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "en":
            e.decode_hook = lambda fed: ("", 0.2)
        return e

    p = StreamingPipeline(clock=clock, factory=factory,
                          acoustic_scorer=lambda audio: {"vi": 0.9, "en": 0.05, "zh": 0.05})
    assert p.active_lang == "und"
    partials = [("xin chào mọi người", 0.9)] * 30
    _run_utterance(p, now, engines, partials)
    assert p.active_lang == "und", "next utterance restarts UNKNOWN (§45)"
    assert not [e for e in p.events if e[0] == "switch"]
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and "xin chào" in finals[-1][1]
    assert finals[-1][2] == "vi"


def test_endpoint_lexical_switch_single_english_word():
    """A lone 'HELLO' resolves to EN via the speculative dual-candidate.

    No confident acoustic LID (default heuristic stays uncertain by design),
    so the pipeline decodes the bootstrap window on VI+EN once (§32): EN's
    'HELLO' @0.78 beats VI's 'HEO' @0.47 on confidence + lexical fit, and
    the utterance commits EN instead of locking VI.
    """
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "en":
            e.decode_hook = lambda fed: ("HELLO", 0.78)
        if lang == "vi":
            e.decode_hook = lambda fed: ("HEO", 0.47)
        return e

    p = StreamingPipeline(clock=clock, factory=factory)
    assert p.active_lang == "und"
    partials = [("HEO", 0.47)] * 30
    _run_utterance(p, now, engines, partials)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and finals[-1][1] == "HELLO"
    assert finals[-1][2] == "en"


def test_endpoint_loanword_stays_vietnamese():
    """A fluent VI sentence with an English loanword must NOT flip to EN."""
    now, clock = make_clock()
    engines = {}

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            e.decode_hook = lambda fed: ("Hôm nay tôi có meeting", 0.8)
        if lang == "en":
            e.decode_hook = lambda fed: ("hom nay toi co", 0.35)
        return e

    p = StreamingPipeline(clock=clock, factory=factory)
    assert p.active_lang == "und"
    partials = [("Hôm nay tôi có meeting", 0.8)] * 30
    _run_utterance(p, now, engines, partials)
    assert not [e for e in p.events if e[0] == "switch"]
    finals = [e for e in p.events if e[0] == "final"]
    assert finals and "meeting" in finals[-1][1]
    assert finals[-1][2] == "vi"


def test_endpoint_flushes_scheduler_residue_without_leak():
    """Tail frames (< one 160 ms chunk) join their own utterance's final.

    Five speech frames (100 ms) never trigger a live decode, so without an
    endpoint flush the tail words are lost — and a naive post-reset decode
    would leak them into the next utterance instead.
    """
    now, clock = make_clock()
    engines = {}
    phase = ["một"]

    def factory(lang):
        e = FakeEngine(lang)
        engines[lang] = e
        if lang == "vi":
            # Hook-driven (no inject): text appears only when audio was
            # actually fed through decode, like a real transducer.
            e.decode_hook = lambda fed: (phase[0], 0.9) if fed >= 1600 else ("", 0.5)
        return e

    p = StreamingPipeline(clock=clock, factory=factory)
    t = 0
    # Utterance 1: 100 ms of speech, then endpoint silence.
    for _ in range(5):
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    for _ in range(120):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals, "expected a FINAL after endpoint silence"
    assert finals[-1][1] == "một", "tail residue must be recovered, got %r" % finals[-1][1]

    # Utterance 2 must start clean — no words from utterance 1.
    phase[0] = "hai"
    for _ in range(5):
        t += 20
        now[0] = t
        p.on_frame(_speech_frame(), t)
    for _ in range(120):
        t += 20
        now[0] = t
        p.on_frame([0.0] * 320, t)
    finals = [e for e in p.events if e[0] == "final"]
    assert finals[-1][1] == "hai", "tail leaked across utterances: %r" % finals[-1][1]
