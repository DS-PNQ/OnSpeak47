"""Streaming pipeline glue — mirrors StreamingPipeline.java (spec §24-§25).

Synchronous reference implementation: the caller feeds 20 ms frames;
ASR decodes every 160 ms; LID runs every 400–600 ms inline (the Android
build moves it to a worker thread — same cadence, same thresholds).
"""
import time
from .config import (FRAME_SAMPLES, SAMPLE_RATE, SCHEDULER_MS,
                      SCHEDULER_SAMPLES, LID_WINDOW_MS, ENDPOINT_VERIFY_MS,
                      ENDPOINT_VERIFY_MARGIN, ENDPOINT_VERIFY_LEX_W,
                      ENDPOINT_VERIFY_STRONG_LEX, ENDPOINT_VERIFY_MIN_TOKENS)
from .ring_buffer import AudioRingBuffer
from .vad import VadEngine
from .lid import LanguageIdEngine, text_units, lexical_fit
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
        self.lid = LanguageIdEngine(acoustic_scorer=acoustic_scorer)
        self.router = LanguageRouter(clock=self.clock)
        self.router.set_verifier(self._verify_candidate)
        self.rollback = RollbackManager()
        self.transcripts = PartialTranscriptManager()
        self.metrics = AsrMetrics()
        self.active_lang = "vi"
        self.models.preload_for_device()
        self.active_asr = self.models.get(self.active_lang)
        self.router.set_active(self.active_lang, 1.0 / 3)
        self._sched_buf = []
        self._last_lid_ms = -10 ** 9
        self._utterance_start = -1
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
        self._sched_buf.extend(frame)
        while len(self._sched_buf) >= SCHEDULER_SAMPLES:
            chunk = self._sched_buf[:SCHEDULER_SAMPLES]
            self._sched_buf = self._sched_buf[SCHEDULER_SAMPLES:]
            # Latency is measured from the START of the audio window, not
            # from when its last frame arrived (else ~160 ms goes missing).
            self._decode_chunk(chunk, now - SCHEDULER_MS)
        uncertain = self.router.state != "ACTIVE"
        interval = LanguageIdEngine.lid_interval_ms(uncertain)
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

    # -- internals ---------------------------------------------------
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
        if self._sched_buf:
            self.active_asr.accept_audio(self._sched_buf)
            self._sched_buf = []
        padded, cap = 0, 960 * SAMPLE_RATE // 1000
        while not self.active_asr.is_ready_to_decode() and padded < cap:
            self.active_asr.accept_audio([0.0] * SCHEDULER_SAMPLES)
            padded += SCHEDULER_SAMPLES
        if self.active_asr.is_ready_to_decode():
            self.active_asr.decode_available()
            text, _ = self.active_asr.partial()
            self.transcripts.update_speculative(text)

    def _on_endpoint(self) -> None:
        # Flush residue first so tail words join THIS utterance instead of
        # leaking into the next one (see _flush_scheduler).
        self._flush_scheduler()
        text, conf = self.active_asr.final()
        base_text = self._merge(text)
        # Endpoint candidate verification (spec §13.1, mirrors
        # StreamingPipeline.java): replay the utterance through the
        # non-active models so a whole utterance in another language cannot
        # lock to the wrong model.
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
        self.transcripts.reset()
        self.active_asr.reset()
        self._utterance_start = -1

    def _verify_endpoint_language(self, base_text: str, base_conf: float):
        """Replay the utterance through non-active models; winner/match/None.

        Compares COMBINED scores (confidence + lexical fit): a
        cross-lingual near-miss ("HELLO" via the VI model) can carry
        middling confidence, and a pure-confidence margin locks that in.
        Mirrors StreamingPipeline.verifyEndpointLanguage (Java).
        """
        now = self.clock()
        span = (ENDPOINT_VERIFY_MS if self._utterance_start < 0
                else min(max(now - self._utterance_start, 500),
                         ENDPOINT_VERIFY_MS))
        audio = self.ring.last_ms(span)
        if len(audio) < SAMPLE_RATE // 2:
            return None
        clean_base = (base_text or "").strip()
        conf_w = 1.0 - ENDPOINT_VERIFY_LEX_W
        active_score = (conf_w * base_conf + ENDPOINT_VERIFY_LEX_W
                        * lexical_fit(clean_base, self.active_lang))
        best, best_score, best_text, best_conf = None, 0.0, "", 0.0
        for lang in ("vi", "en", "zh"):
            if lang == self.active_lang:
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
            if (cand_score >= active_score + ENDPOINT_VERIFY_MARGIN
                    and cand_score > best_score):
                best, best_score, best_text, best_conf = lang, cand_score, text, conf
        if best is not None:
            return {"lang": best, "text": best_text, "confidence": best_conf}
        if text_units(clean_base) >= 2 and base_conf >= 0.5:
            return {"lang": self.active_lang, "text": clean_base,
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
