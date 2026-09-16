"""Streaming pipeline glue — mirrors StreamingPipeline.java (spec §24-§25).

Synchronous reference implementation: the caller feeds 20 ms frames;
ASR decodes every 160 ms; LID runs every 400–600 ms inline (the Android
build moves it to a worker thread — same cadence, same thresholds).
"""
import time
from .config import (FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, SCHEDULER_MS,
                      SCHEDULER_SAMPLES, LID_WINDOW_MS, ENDPOINT_VERIFY_MS,
                      ENDPOINT_VERIFY_MARGIN, ENDPOINT_VERIFY_LEX_W,
                      ENDPOINT_VERIFY_STRONG_LEX, ENDPOINT_VERIFY_MIN_TOKENS,
                      BOOTSTRAP_MIN_MS, BOOTSTRAP_LID_WINDOW_MS,
                      BOOTSTRAP_HOP_MS, BOOTSTRAP_MAX_MS,
                      BOOTSTRAP_GIVE_UP_MS,
                      SPECULATIVE_CANDIDATE_COOLDOWN_MS)
from .ring_buffer import AudioRingBuffer
from .vad import VadEngine
from .lid import (LanguageIdEngine, text_units, lexical_fit,
                  heuristic_acoustic_scorer)
from .router import LanguageRouter
from .partial import PartialTranscriptManager
from .rollback import RollbackManager
from .metrics import AsrMetrics
from .model_manager import ZipformerModelManager


class StreamingPipeline:
    def __init__(self, total_mem_gb: float = 8.0, clock=None, factory=None,
                 acoustic_scorer=None):
        self.now_ms = [0]
        self.clock = clock or (lambda: self.now_ms[0])
        self.ring = AudioRingBuffer.with_capacity_ms(30_000)
        self.vad = VadEngine()
        self.models = ZipformerModelManager(total_mem_gb, factory)
        # Default to the interim audio-derived heuristic (mirrors
        # StreamingPipeline.defaultLidEngine): a null scorer would fall back
        # to the transcript heuristic — the circular VI lock (§3–§4).
        if acoustic_scorer is None:
            acoustic_scorer = heuristic_acoustic_scorer
        self.lid = LanguageIdEngine(acoustic_scorer=acoustic_scorer)
        self.router = LanguageRouter(clock=self.clock)
        self.router.set_verifier(self._verify_candidate)
        self.rollback = RollbackManager()
        self.transcripts = PartialTranscriptManager()
        self.metrics = AsrMetrics()
        # fakedemo2 §6, §12: never start as "vi" — start UNKNOWN; bootstrap
        # acoustic LID commits the first model per utterance.
        self.active_lang = "und"
        self.active_asr = None
        self.models.preload_for_device()
        self.router.set_active("und", 1.0 / 3)
        self._sched_buf = []
        self._last_lid_ms = -10 ** 9
        self._utterance_start = -1
        # --- Bootstrap state (§7, §24, §27) ---
        self._utterance_speech_ms = 0
        self._utterance_seq = 0
        self._last_bootstrap_speech_ms = -1
        self._last_candidate_speech_ms = -1
        self._last_candidate_pair = None
        self._utterance_start_sample = -1
        # --- Provisional speculative decode (Option A hotfix) ---
        # While UNKNOWN the pipeline ALSO decodes continuously on VI so
        # partials flow after ~160 ms. Provisional text stays outside the
        # shared transcript manager: adopted on same-language commit,
        # discarded without a trace otherwise (§11).
        self._provisional_asr = None
        self._provisional_lang = "vi"
        self._provisional_text = ""
        self._provisional_conf = 1.0 / 3
        self._last_posted_provisional = ""
        self._reset_provisional()
        self.events = []  # ("partial"|"final"|"switch", payload...)

    # -- public ------------------------------------------------------
    def on_frame(self, frame, now_ms: int = None):
        now = self.clock() if now_ms is None else now_ms
        self.ring.append(frame)
        v = self.vad.process(frame, now)
        if v.speech_end:
            self._on_endpoint()
            return
        if not v.is_speech:
            return
        if self._utterance_start < 0:
            self._utterance_start = now
            # The frame was already appended: rewind to its first sample so
            # replay starts at the true speech onset (§15).
            self._utterance_start_sample = self.ring.write_pos - len(frame)
        self._utterance_speech_ms += FRAME_MS
        # --- Bootstrap phase (Option A): provisional speculative decode keeps
        # partials flowing while bootstrap acoustic LID decides in parallel.
        # Provisional text is SPECULATIVE-only — never committed itself (§11).
        if self.active_lang == "und" or self.active_asr is None:
            self._try_bootstrap_lid()
            self._accumulate_scheduler(frame, now, provisional=True)
            return
        # 160 ms scheduler accumulation on the active stream.
        self._accumulate_scheduler(frame, now, provisional=False)
        uncertain = self.router.state != "ACTIVE"
        has_candidate = self.router.candidate is not None
        interval = LanguageIdEngine.lid_interval_ms(uncertain, has_candidate)
        if now - self._last_lid_ms >= interval:
            self._last_lid_ms = now
            window = self.ring.last_ms(LID_WINDOW_MS)
            # Text evidence covers the same recent span as the 480 ms audio
            # window (last ~6 tokens). Scoring the whole committed transcript
            # would pin the evidence to the utterance's first language and
            # blind the router to intra-utterance switches.
            r = self.lid.classify(window, self._recent_text(6),
                                  self.active_asr.confidence, self.active_lang)
            d = self.router.on_lid_result(r)
            if d == "START_ROLLBACK":
                w = self.rollback.start_rollback(self.ring, r.language, now)
                out = self.router.verify_and_commit(w)
                if out == "COMMITTED":
                    self._on_committed(r.language, w)
                else:
                    self.transcripts.rollback_speculative()

    # -- bootstrap acoustic LID (§7–§9, §14–§15, §24–§27, §32) --------
    def _try_bootstrap_lid(self) -> None:
        """One sync bootstrap attempt when enough speech has accumulated."""
        if self.active_lang != "und":
            return
        if self._utterance_speech_ms < BOOTSTRAP_MIN_MS:
            return
        if (self._last_bootstrap_speech_ms >= 0
                and (self._utterance_speech_ms - self._last_bootstrap_speech_ms
                     < BOOTSTRAP_HOP_MS)):
            return
        self._last_bootstrap_speech_ms = self._utterance_speech_ms
        window = self.ring.last_ms(BOOTSTRAP_LID_WINDOW_MS)
        r = self.lid.classify_bootstrap(window)
        decided = self.router.on_bootstrap_lid_result(r)
        if decided != "und":
            self._activate_bootstrap(decided, r.confidence)
            return
        # Uncertain: collect up to BOOTSTRAP_MAX_MS (§9 cách 1), then ≤2
        # speculative candidates (§32) — never force VI on max-probability.
        if self._utterance_speech_ms < BOOTSTRAP_MAX_MS:
            return
        # Cooldown: a candidate pass costs 2 shadow decodes — rerun at most
        # every SPECULATIVE_CANDIDATE_COOLDOWN_MS of additional audio.
        since_cand = self._utterance_speech_ms - self._last_candidate_speech_ms
        if self._last_candidate_speech_ms >= 0 and since_cand < SPECULATIVE_CANDIDATE_COOLDOWN_MS:
            return
        # Give-up: past BOOTSTRAP_GIVE_UP_MS without a commit, stop burning
        # candidate decodes; hidden provisional keeps flowing and endpoint
        # recovery still gets one final chance.
        if self._utterance_speech_ms > BOOTSTRAP_GIVE_UP_MS:
            return
        self._last_candidate_speech_ms = self._utterance_speech_ms
        self._run_speculative_candidates(r)

    def _utterance_audio(self, max_ms: int):
        """Utterance audio from speech onset, oldest-first, capped at max_ms.

        Falls back to last_ms when the start mark is missing.
        """
        max_samples = 16000 * max_ms // 1000
        if self._utterance_start_sample >= 0:
            return self.ring.since(self._utterance_start_sample, max_samples)
        return self.ring.last_ms(max_ms)

    def _activate_bootstrap(self, lang: str, confidence: float) -> None:
        """Commit the bootstrap winner — adopt or rollback.

        Winner == provisional language → adopt the live stream as-is (no
        reset, no replay). Otherwise discard the provisional text without a
        trace and REPLAY the buffered utterance audio into the winner
        (§15, §25).
        """
        if self.active_lang != "und" or not lang or lang == "und":
            return
        if lang == self._provisional_lang and self._provisional_asr is not None:
            self.active_lang = lang
            self.active_asr = self._provisional_asr
            self._provisional_asr = None
            self.router.commit_bootstrap(lang, confidence)
            self.transcripts.update_speculative(self._provisional_text)
            self.events.append(("partial", self.transcripts.committed,
                                self.transcripts.speculative, lang))
            self._provisional_text = ""
            self._last_posted_provisional = ""
            return
        self._provisional_text = ""
        self._last_posted_provisional = ""
        self._provisional_asr = None
        self._sched_buf = []  # replay below re-covers this audio from the ring
        self.active_lang = lang
        self.active_asr = self.models.get(lang)
        self.active_asr.reset()
        self.router.commit_bootstrap(lang, confidence)
        # Replay from the utterance start (§15, §25) — not last_ms, so a
        # mid-utterance pause can't shift the window onto silence.
        buffered = self._utterance_audio(30000)
        for off in range(0, len(buffered), SCHEDULER_SAMPLES):
            self.active_asr.accept_audio(buffered[off:off + SCHEDULER_SAMPLES])
            if self.active_asr.is_ready_to_decode():
                self.active_asr.decode_available()
        text, _ = self.active_asr.partial()
        # Still SPECULATIVE (§11): language chosen, transcript firms up later.
        self.transcripts.update_speculative(text or "")
        self.events.append(("partial", self.transcripts.committed,
                            self.transcripts.speculative, lang))

    def _run_speculative_candidates(self, r) -> None:
        """Uncertain fallback (§32): one-shot decode on at most top-2 models."""
        window = self.ring.last_ms(BOOTSTRAP_MAX_MS)
        # Too-short window → decode would be garbage + burn 2 engines.
        if not window or len(window) < SAMPLE_RATE * 400 // 1000:
            return
        self._run_speculative_candidates_on(
            r, window,
            self._choose_candidate_pair(r.scores if r is not None else None))

    def _choose_candidate_pair(self, scores) -> list:
        """Rotating pair (≤2 decodes per pass, §10): acoustic top-2 normally;
        after an inconclusive pass the leftover 3rd language pairs with the
        top-1, so all three are tried within ~1.6 s without ever running 3
        models at once."""
        top = self._top_two(scores)
        last = self._last_candidate_pair
        if not last:
            return top
        leftover = next((l for l in ("vi", "en", "zh") if l not in last), None)
        if leftover is None or leftover == top[0]:
            return top
        return [top[0], leftover]

    @staticmethod
    def _leftovers_of(pair) -> list:
        return [l for l in ("vi", "en", "zh") if not pair or l not in pair]

    def _run_speculative_candidates_on(self, r, window, pair=None) -> None:
        """One-shot decode of `window` on the given pair (max 2)."""
        if self.active_lang != "und":
            return
        scores = r.scores if r is not None else None
        pair = pair if pair is not None else self._top_two(scores)
        if not window:
            return
        self._last_candidate_pair = list(pair)
        winner, winner_score, winner_conf = None, -1.0, 0.0
        for cand in pair:
            if not cand or cand == "und":
                continue
            s = self._score_speculative_candidate(cand, window)
            if s is None:
                continue
            text, conf = s
            if not text:
                continue
            combined = 0.65 * conf + 0.35 * lexical_fit(text, cand)
            if combined > winner_score:
                winner, winner_score, winner_conf = cand, combined, conf
        if winner is not None and winner_score >= 0.30:
            self._activate_bootstrap(winner, min(winner_score, 0.69))

    @staticmethod
    def _top_two(scores) -> list:
        order = ["vi", "en", "zh"]
        if scores:
            order = sorted(order, key=lambda l: -(scores.get(l) or 0.0))
        return order[:2]

    def _score_speculative_candidate(self, lang: str, window):
        """One-shot decode of the bootstrap window.

        The live provisional engine is read WITHOUT reset — wiping it would
        destroy the speculative stream it is still decoding.
        """
        try:
            cand = self.models.get(lang)
            if cand is not self._provisional_asr:
                cand.reset()
                for off in range(0, len(window), SCHEDULER_SAMPLES):
                    cand.accept_audio(window[off:off + SCHEDULER_SAMPLES])
                    if cand.is_ready_to_decode():
                        cand.decode_available()
            text, conf = cand.partial()
            return (text or "").strip(), conf
        except Exception:
            return None

    def _accumulate_scheduler(self, frame, now: int, provisional: bool) -> None:
        """Shared 160 ms scheduler. While UNKNOWN chunks go to the provisional
        decoder; otherwise to the active stream. Residue stays in _sched_buf
        either way, so adopting the provisional engine keeps one continuous
        stream with no duplication."""
        self._sched_buf.extend(frame)
        while len(self._sched_buf) >= SCHEDULER_SAMPLES:
            chunk = self._sched_buf[:SCHEDULER_SAMPLES]
            self._sched_buf = self._sched_buf[SCHEDULER_SAMPLES:]
            # Latency is measured from the START of the audio window, not
            # from when its last frame arrived (else ~160 ms goes missing).
            if provisional:
                self._decode_provisional(chunk, now - SCHEDULER_MS)
            else:
                self._decode_chunk(chunk, now - SCHEDULER_MS)

    def _reset_provisional(self) -> None:
        """(Re)point the provisional decoder at resident VI, fresh state."""
        self._provisional_lang = "vi"
        self._provisional_text = ""
        self._provisional_conf = 1.0 / 3
        self._last_posted_provisional = ""
        try:
            self._provisional_asr = self.models.get("vi")
            self._provisional_asr.reset()
        except Exception:
            self._provisional_asr = None

    def _decode_provisional(self, chunk, audio_time_ms: int) -> None:
        """Continuous speculative partials on VI while UNKNOWN (Option A).

        The decode keeps running (adopt fast-path needs live stream state),
        but UI posts are hidden speculative-only and de-duplicated.
        """
        if self._provisional_asr is None:
            try:
                self._provisional_asr = self.models.get("vi")
                self._provisional_asr.reset()
            except Exception:
                return
        self._provisional_asr.accept_audio(chunk)
        if self._provisional_asr.is_ready_to_decode():
            self._provisional_asr.decode_available()
        text, conf = self._provisional_asr.partial()
        self._provisional_text = text or ""
        self._provisional_conf = conf
        self.metrics.add_partial_latency(max(0, self.clock() - audio_time_ms))
        if self._provisional_text and self._provisional_text != self._last_posted_provisional:
            self._last_posted_provisional = self._provisional_text
            self.events.append(("partial", "", self._provisional_text,
                                self._provisional_lang))
    def _decode_chunk(self, chunk, audio_time_ms: int):
        t0 = self.clock()
        self.active_asr.accept_audio(chunk)
        self.active_asr.decode_available()
        text, conf = self.active_asr.partial()
        self.transcripts.update_speculative(text)
        self.router.on_partial_confidence(conf)
        self.metrics.add_partial_latency(max(0, self.clock() - audio_time_ms))
        self.events.append(("partial", self.transcripts.committed,
                            self.transcripts.speculative, self.active_lang))

    def _verify_candidate(self, candidate, rollback_audio):
        cand = self.models.get(candidate)
        cand.reset()
        for off in range(0, len(rollback_audio), SCHEDULER_SAMPLES):
            cand.accept_audio(rollback_audio[off:off + SCHEDULER_SAMPLES])
            if cand.is_ready_to_decode():
                cand.decode_available()
        for _ in range(4):
            if not cand.is_ready_to_decode():
                break
            cand.decode_available()
        text, conf = cand.partial()
        ok = bool(text and text.strip()) and conf >= 0.35
        from .router import Verification
        return Verification(ok, text, conf)

    def _on_committed(self, nxt: str, window) -> None:
        prev = self.active_lang
        self.transcripts.rollback_speculative()
        cand = self.models.get(nxt)
        text, conf = cand.partial()
        self.transcripts.commit(text)
        self.active_lang = nxt
        self.active_asr = cand
        self.router.set_active(nxt, conf)
        self.active_asr.accept_audio(window)
        self.events.append(("switch", prev, nxt))

    def _flush_scheduler(self) -> None:
        """Fold scheduler residue into the stream, pad trailing silence
        until a full chunk is decodable (bounded), then decode once.

        Mirrors StreamingPipeline.flushScheduler (Java): every decode stays
        behind is_ready_to_decode because sherpa ABORTS the process on an
        under-buffered decode. The silence padding doubles as the
        right-context held-out tail words need to emit.
        """
        if self.active_asr is None:
            self._sched_buf = []
            return
        text = self._flush_engine(self.active_asr)
        if text is not None:
            self.transcripts.update_speculative(text)

    def _flush_engine(self, engine):
        """Flush residue + padded silence into `engine`, decode once.

        Returns the fresh partial text, or None when nothing became
        decodable. Shared by the active flush and the provisional endpoint
        flush.
        """
        if engine is None:
            self._sched_buf = []
            return None
        if self._sched_buf:
            engine.accept_audio(self._sched_buf)
            self._sched_buf = []
        padded, cap = 0, 960 * SAMPLE_RATE // 1000
        while not engine.is_ready_to_decode() and padded < cap:
            engine.accept_audio([0.0] * SCHEDULER_SAMPLES)
            padded += SCHEDULER_SAMPLES
        if not engine.is_ready_to_decode():
            return None
        engine.decode_available()
        text, _ = engine.partial()
        return text

    def _on_endpoint(self) -> None:
        # Short utterance that never bootstrapped (§35): one synchronous
        # audio-only bootstrap, then finalize. Never force a language.
        if self.active_asr is None or self.active_lang == "und":
            self._finish_unbootstrapped_endpoint()
            return
        # Flush residue first so tail words join THIS utterance instead of
        # leaking into the next one (see _flush_scheduler).
        self._flush_scheduler()
        text, conf = self.active_asr.final()
        base_text = self._merge(text)
        # Endpoint candidate verification (spec §13.1, kept as backstop in
        # fakedemo2 Phase 5): replay the utterance through the non-active
        # models. Bootstrap usually commits the right model already, but a
        # mid-utterance change the runtime gate missed still gets one
        # correction chance here. Mirrors StreamingPipeline.java.
        verdict = self._verify_endpoint_language(base_text, conf)
        if verdict is not None and verdict["lang"] != self.active_lang:
            prev = self.active_lang
            if verdict["text"]:
                base_text = verdict["text"]
            self.active_lang = verdict["lang"]
            self.active_asr = self.models.get(self.active_lang)
            self.router.commit_endpoint_switch(self.active_lang,
                                               verdict["confidence"])
            self.events.append(("switch", prev, self.active_lang))
        else:
            # Confirmed-active or inconclusive: the text-router endpoint bar
            # stays as backstop for lexical cases the re-decode missed.
            if verdict is not None:
                self.router.set_active(self.active_lang,
                                       self.active_asr.confidence)
            window = self.ring.last_ms(LID_WINDOW_MS)
            r = self.lid.classify(window, self.transcripts.committed,
                                  self.active_asr.confidence, self.active_lang)
            if self.router.on_endpoint(r.scores):
                self.active_lang = self.router.active
                self.active_asr = self.models.get(self.active_lang)
        self.transcripts.finalize(base_text)
        self.events.append(("final", self.transcripts.committed, self.active_lang))
        self.metrics.utterances += 1
        self._reset_utterance_state()

    def _finish_unbootstrapped_endpoint(self) -> None:
        """Endpoint before any commit (§35). Recovery in order: (1) sync
        audio-only bootstrap on the utterance onset; (2) one speculative pair
        pass + the leftover language over the utterance audio; (3)
        endpoint-verification backstop with the flushed provisional hypothesis
        as base. Never forces a language.
        """
        try:
            # Flush the provisional residue first so the short utterance is
            # fully decoded before it is judged (mirrors _flush_scheduler).
            if self._provisional_asr is not None:
                flushed = self._flush_engine(self._provisional_asr)
                if flushed and flushed.strip():
                    self._provisional_text = flushed.strip()
                    self._provisional_conf = self._provisional_asr.confidence
            # Read from the utterance start: at an endpoint last_ms would
            # return mostly the trailing silence (§35 short utterances).
            spoken = self._utterance_audio(ENDPOINT_VERIFY_MS)
            window = (spoken[:16000 * BOOTSTRAP_LID_WINDOW_MS // 1000]
                      if len(spoken) > 16000 * BOOTSTRAP_LID_WINDOW_MS // 1000
                      else spoken)
            r = self.lid.classify_bootstrap(window)
            decided = self.router.on_bootstrap_lid_result(r)
            if decided != "und":
                self._activate_bootstrap(decided, r.confidence)
                decided = self.active_lang
            else:
                pair = self._choose_candidate_pair(
                    r.scores if r is not None else None)
                self._run_speculative_candidates_on(r, spoken, pair)
                decided = self.active_lang
                if decided == "und":
                    rest = self._leftovers_of(pair)
                    if rest:
                        self._run_speculative_candidates_on(
                            r, spoken, [rest[0], "und"])
                        decided = self.active_lang
            if decided == "und":
                # Final backstop: full-utterance verification with the
                # provisional hypothesis as base (old-system behavior).
                base = (self._provisional_text or "").strip()
                verdict = self._verify_endpoint_language(
                    base, self._provisional_conf, self._provisional_lang)
                if verdict is not None:
                    if verdict["lang"] != self._provisional_lang:
                        self._activate_bootstrap(verdict["lang"],
                                                 verdict["confidence"])
                    else:
                        self._adopt_provisional(verdict["confidence"])
                    decided = self.active_lang
            if decided != "und" and self.active_asr is not None:
                self._flush_scheduler()
                text, _ = self.active_asr.final()
                self.transcripts.finalize(self._merge(text))
                self.events.append(("final", self.transcripts.committed,
                                    self.active_lang))
                self.metrics.utterances += 1
            else:
                self.transcripts.finalize("")
                self.events.append(("final", "", "und"))
                self.metrics.utterances += 1
        except Exception:
            try:
                self.transcripts.finalize("")
                self.events.append(("final", "", "und"))
            except Exception:
                pass
        finally:
            self._reset_utterance_state()

    def _reset_utterance_state(self) -> None:
        """Clear per-utterance state; next utterance bootstraps from UNKNOWN.

        Resident engines stay cached (§29 — switch = pointer change, never
        load/unload); only the active pointer is cleared.
        """
        self.transcripts.reset()
        if self.active_asr is not None:
            try:
                self.active_asr.reset()
            except Exception:
                pass
        self.active_asr = None
        self.active_lang = "und"
        self.router.reset()
        try:
            self.lid.reset()
        except Exception:
            pass
        self._sched_buf = []
        self._utterance_start = -1
        self._utterance_speech_ms = 0
        self._last_bootstrap_speech_ms = -1
        self._last_candidate_speech_ms = -1
        self._last_candidate_pair = None
        self._utterance_start_sample = -1
        self._utterance_seq += 1
        self._reset_provisional()

    def _adopt_provisional(self, confidence: float) -> None:
        """Adopt the provisional stream as committed active (endpoint backstop
        confirmed it). No reset, no replay — the engine holds full state."""
        if (self.active_lang != "und" or self._provisional_asr is None):
            return
        self.active_lang = self._provisional_lang
        self.active_asr = self._provisional_asr
        self._provisional_asr = None
        self.router.commit_bootstrap(self.active_lang, confidence)
        self.transcripts.update_speculative(self._provisional_text)
        self._provisional_text = ""
        self._last_posted_provisional = ""

    def _verify_endpoint_language(self, base_text: str, base_conf: float,
                                  base_lang: str = None):
        """Replay the utterance through non-base models; winner/match/None.

        Compares COMBINED scores (confidence + lexical fit): a
        cross-lingual near-miss ("HELLO" via the VI model) can carry
        middling confidence, and a pure-confidence margin locks that in.
        base_lang is the hypothesis owner (active model, or the provisional
        model on the unbootstrapped path). Mirrors
        StreamingPipeline.verifyEndpointLanguage (Java).
        """
        if base_lang is None:
            base_lang = self.active_lang
        now = self.clock()
        span = (ENDPOINT_VERIFY_MS if self._utterance_start < 0
                else min(max(now - self._utterance_start, 500),
                         ENDPOINT_VERIFY_MS))
        audio = self.ring.last_ms(span)
        if len(audio) < SAMPLE_RATE // 2:
            return None
        clean_base = (base_text or "").strip()
        conf_w = 1.0 - ENDPOINT_VERIFY_LEX_W
        base_score = (conf_w * base_conf + ENDPOINT_VERIFY_LEX_W
                      * lexical_fit(clean_base, base_lang))
        best, best_score, best_text, best_conf = None, 0.0, "", 0.0
        for lang in ("vi", "en", "zh"):
            if lang == base_lang:
                continue
            s = self._score_candidate(lang, audio)
            if s is None or not s[0]:
                continue
            text, conf = s
            lex_cand = lexical_fit(text, lang)
            min_units = (1 if lex_cand >= ENDPOINT_VERIFY_STRONG_LEX
                         else ENDPOINT_VERIFY_MIN_TOKENS)
            if text_units(text) < min_units:
                continue
            cand_score = conf_w * conf + ENDPOINT_VERIFY_LEX_W * lex_cand
            if (cand_score >= base_score + ENDPOINT_VERIFY_MARGIN
                    and cand_score > best_score):
                best, best_score, best_text, best_conf = lang, cand_score, text, conf
        if best is not None:
            return {"lang": best, "text": best_text, "confidence": best_conf}
        # Base unbeaten: confirmed when fluent, or a single dictionary-strong
        # token like a lone "HELLO"/"một" (§35 short utterances).
        if base_conf >= 0.5 and (
                text_units(clean_base) >= 2
                or (text_units(clean_base) >= 1
                    and lexical_fit(clean_base, base_lang)
                    >= ENDPOINT_VERIFY_STRONG_LEX)):
            return {"lang": base_lang, "text": clean_base,
                    "confidence": base_conf}
        return None

    def _score_candidate(self, lang: str, audio):
        """Decode audio on a non-active engine; (text, conf) or None.

        Feeds 160 ms slices exactly like live pacing so the hypothesis
        covers the whole window (mirrors StreamingPipeline.java).
        """
        try:
            cand = self.models.get(lang)
            cand.reset()
            for off in range(0, len(audio), SCHEDULER_SAMPLES):
                cand.accept_audio(audio[off:off + SCHEDULER_SAMPLES])
                if cand.is_ready_to_decode():
                    cand.decode_available()
            for _ in range(4):
                if not cand.is_ready_to_decode():
                    break
                cand.decode_available()
            text, conf = cand.partial()
            return (text or "").strip(), conf
        except Exception:
            return None

    def _recent_text(self, max_tokens: int = 10) -> str:
        toks = self.transcripts.display.split()
        return " ".join(toks[-max_tokens:]) if len(toks) > max_tokens else " ".join(toks)

    def _merge(self, fresh: str) -> str:
        c = self.transcripts.committed
        f = (fresh or "").strip()
        if not f:
            return c
        if c and f.startswith(c):
            return f
        return f if not c else c + " " + f

    def feed_silence(self, ms: int, step_ms: int = 20) -> None:
        t = self.clock()
        n = [0.0] * (16000 * step_ms // 1000)
        while ms > 0:
            self.on_frame(n, t)
            t += step_ms
            if isinstance(self.now_ms, list):
                self.now_ms[0] = t
            ms -= step_ms

    def feed_speech(self, ms: int, step_ms: int = 20, amp: float = 0.2) -> None:
        t = self.clock()
        n = [amp] * (16000 * step_ms // 1000)
        while ms > 0:
            self.on_frame(n, t)
            t += step_ms
            if isinstance(self.now_ms, list):
                self.now_ms[0] = t
            ms -= step_ms
