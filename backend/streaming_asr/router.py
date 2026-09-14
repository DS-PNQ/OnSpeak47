"""Router state machine — mirrors LanguageRouter.java (spec §10-§15, MUST 5/6)."""
from dataclasses import dataclass
from .config import (SWITCH_THRESHOLD, SWITCH_MARGIN, SWITCH_PERSIST_MS, EMA_ALPHA,
                      ENDPOINT_THRESHOLD, ENDPOINT_MARGIN,
                      BOOTSTRAP_THRESHOLD, BOOTSTRAP_MARGIN)


class RouterState:
    UNKNOWN = "UNKNOWN"
    BOOTSTRAPPING = "BOOTSTRAPPING"
    ACTIVE = "ACTIVE"
    CANDIDATE_SWITCH = "CANDIDATE_SWITCH"
    ROLLBACK = "ROLLBACK"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"


@dataclass
class Verification:
    supports_switch: bool
    tokens: str = ""
    confidence: float = 0.0


class LanguageRouter:
    def __init__(self, switch_threshold=SWITCH_THRESHOLD, switch_margin=SWITCH_MARGIN,
                 persist_ms=SWITCH_PERSIST_MS, clock=None, discard_cooldown_ms=1500):
        import time
        self.switch_threshold = switch_threshold
        self.switch_margin = switch_margin
        self.persist_ms = persist_ms
        self.discard_cooldown_ms = discard_cooldown_ms
        self.clock = clock or (lambda: int(time.time() * 1000))
        # fakedemo2 §6, §22: never start as "vi" — start UNKNOWN; bootstrap
        # acoustic LID commits the first language per utterance.
        self.active = "und"
        self.active_confidence = 1.0 / 3
        self.candidate = None
        self._candidate_since = -1
        self.state = RouterState.UNKNOWN
        self.verifier = None
        self.switch_count = 0
        self.last_switch_ms = -1
        self._last_discard_ms = -10 ** 12
        self._last_discarded = None

    def set_verifier(self, fn) -> None:
        self.verifier = fn

    def set_active(self, lang: str, conf: float) -> None:
        self.active = lang or "und"
        self.active_confidence = max(0.0, min(1.0, conf))
        self._clear_candidate()
        self.state = (RouterState.UNKNOWN if self.active == "und"
                      else RouterState.ACTIVE)

    def is_bootstrapping(self) -> bool:
        """True while no language is committed (fakedemo2 §22)."""
        return self.active == "und"

    def commit_bootstrap(self, lang: str, conf: float) -> None:
        """Commit the bootstrap decision; never called with 'und'."""
        if not lang or lang == "und":
            return
        self.active = lang
        self.active_confidence = max(0.0, min(1.0, conf))
        self._clear_candidate()
        self.state = RouterState.ACTIVE

    def on_bootstrap_lid_result(self, lid) -> str:
        """Bootstrap gate (§22, §42): top1 >= 0.70 AND top1-top2 >= 0.15.

        Returns the committed language, or 'und' while still uncertain
        (caller extends the window or runs ≤2 speculative candidates —
        never forces VI on a bare max-probability).
        """
        if lid is None or not getattr(lid, "language", None) or lid.language == "und":
            self.state = RouterState.BOOTSTRAPPING
            return "und"
        if self.active != "und":
            return self.active
        top1 = lid.confidence
        top2 = 0.0
        for lang, score in (lid.scores or {}).items():
            if lang == lid.language or lang == "und" or score is None:
                continue
            top2 = max(top2, score)
        if top1 >= BOOTSTRAP_THRESHOLD and (top1 - top2) >= BOOTSTRAP_MARGIN:
            self.commit_bootstrap(lid.language, top1)
            return self.active
        self.state = RouterState.BOOTSTRAPPING
        return "und"

    def on_partial_confidence(self, token_conf: float) -> None:
        c = max(0.0, min(1.0, token_conf))
        self.active_confidence = EMA_ALPHA * c + (1 - EMA_ALPHA) * self.active_confidence

    def on_lid_result(self, lid) -> str:
        """Returns HOLD | OBSERVE | START_ROLLBACK | COMMITTED | DISCARDED."""
        if lid is None or lid.language is None:
            return "HOLD"
        # Bootstrap owns the UND phase (§23): runtime hysteresis must not
        # vote while no active model exists.
        if self.active == "und":
            return "HOLD"
        cand = lid.language
        if cand == self.active or cand == "und":
            self._clear_candidate()
            self.state = RouterState.ACTIVE
            return "HOLD"
        if lid.confidence < self.switch_threshold:
            self._clear_candidate()
            if self.state == RouterState.CANDIDATE_SWITCH:
                self.state = RouterState.ACTIVE
            return "HOLD"
        if lid.confidence < self.active_confidence + self.switch_margin:
            return "OBSERVE" if self.state == RouterState.CANDIDATE_SWITCH else "HOLD"
        now = self.clock()
        if self.candidate != cand:
            # Cooldown after a failed verification: don't re-run a costly
            # shadow decode for the same rejected candidate immediately.
            if (cand == self._last_discarded
                    and now - self._last_discard_ms < self.discard_cooldown_ms):
                return "HOLD"
            self.candidate = cand
            self._candidate_since = now
            self.state = RouterState.CANDIDATE_SWITCH
            return "OBSERVE"
        if now - self._candidate_since < self.persist_ms:
            self.state = RouterState.CANDIDATE_SWITCH
            return "OBSERVE"
        self.state = RouterState.ROLLBACK
        return "START_ROLLBACK"

    def verify_and_commit(self, rollback_audio) -> str:
        if self.state != RouterState.ROLLBACK or self.candidate is None:
            return "HOLD"
        self.state = RouterState.VERIFY
        v = self.verifier(self.candidate, rollback_audio) if self.verifier else Verification(False)
        ok = bool(v and v.supports_switch)
        return self.on_verification(ok, v.tokens if v else "", v.confidence if v else 0.0)

    def on_verification(self, supports: bool, tokens: str = "", conf: float = 0.0) -> str:
        if self.state not in (RouterState.VERIFY, RouterState.ROLLBACK):
            return "HOLD"
        if supports and self.candidate is not None:
            self.active = self.candidate
            self.active_confidence = max(0.0, min(1.0, conf))
            self.last_switch_ms = self.clock()
            self.switch_count += 1
            self._clear_candidate()
            self.state = RouterState.ACTIVE
            return "COMMITTED"
        self._last_discarded = self.candidate
        self._last_discard_ms = self.clock()
        self._clear_candidate()
        self.state = RouterState.ACTIVE
        return "DISCARDED"

    def on_endpoint(self, scores: dict) -> bool:
        # Endpoint bar is deliberately lower than the intra-utterance gate:
        # at a VAD boundary no committed text is rewritten and no rollback
        # decode is spent, so starting the next utterance on the LID winner
        # is cheap (spec §13.1).
        if not scores:
            return False
        best, best_score = self.active, -1.0
        for lang, score in scores.items():
            if lang == "und" or score is None:
                continue
            if score > best_score:
                best, best_score = lang, score
        active_score = scores.get(self.active, 0.0) or 0.0
        if (best != self.active and best_score >= ENDPOINT_THRESHOLD
                and best_score > active_score + ENDPOINT_MARGIN):
            self.active = best
            self.active_confidence = max(0.0, min(1.0, best_score))
            self.last_switch_ms = self.clock()
            self.switch_count += 1
            self._clear_candidate()
            self.state = RouterState.ACTIVE
            return True
        self._clear_candidate()
        return False

    def commit_endpoint_switch(self, lang: str, conf: float) -> None:
        """Record a switch decided by endpoint candidate verification.

        The evidence (candidate re-decode of the utterance) was already
        compared by the caller; this commits it and counts it like any
        other switch so telemetry stays in one place.
        """
        self.active = lang
        self.active_confidence = max(0.0, min(1.0, conf))
        self.last_switch_ms = self.clock()
        self.switch_count += 1
        self._clear_candidate()
        self.state = RouterState.ACTIVE

    def reset(self) -> None:
        self.active = "und"
        self.active_confidence = 1.0 / 3
        self._clear_candidate()
        self.state = RouterState.UNKNOWN
        self.switch_count = 0
        self.last_switch_ms = -1
        self._last_discard_ms = -10 ** 12
        self._last_discarded = None

    def _clear_candidate(self) -> None:
        self.candidate = None
        self._candidate_since = -1
