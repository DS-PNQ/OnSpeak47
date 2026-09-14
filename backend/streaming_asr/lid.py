"""Stage-1 LID — mirrors LanguageIdEngine.java (spec §10-§12, §16-§17)."""
from dataclasses import dataclass, field
from .config import (
    W_ACOUSTIC, W_TEXT, W_CONFIDENCE, W_HISTORY, EMA_ALPHA,
    LID_INTERVAL_STABLE_MS, LID_INTERVAL_UNCERTAIN_MS,
)

LANGS = ("vi", "en", "zh")

EN_WORDS = {"meeting", "project", "system", "beautiful", "discussion",
            "report", "team", "final", "tomorrow", "today", "please",
            "thanks", "hello", "discuss", "schedule", "deadline", "email"}
VI_WORDS = {"hôm", "nay", "trời", "rất", "đẹp", "mình", "tôi", "đang",
            "không", "của", "và", "là", "một", "người", "với", "để",
            "xin", "chào", "cảm", "ơn", "gửi"}

# High-frequency English function words (word-boundary matched).
EN_COMMON = {"you", "the", "and", "are", "was", "were", "have", "has",
             "will", "would", "what", "when", "this", "that", "with",
             "from", "hi"}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF):
            cjk += 1
    return cjk / total if total else 0.0


def cjk_count(text: str) -> int:
    """Number of CJK Unified Ideograph characters (no spaces in Chinese)."""
    if not text:
        return 0
    n = 0
    for ch in text:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF):
            n += 1
    return n


def text_units(text: str) -> int:
    """Length gate for transcripts: whitespace tokens + CJK chars/2.

    Chinese has no word spaces, so plain token counting would reject any
    fluent ZH hypothesis (e.g. "然后我们开始" is one 'token').
    """
    if not text or not text.strip():
        return 0
    return len(text.split()) + cjk_count(text) // 2


def lexical_fit(text: str, lang: str) -> float:
    """Lexical fit of a transcript to one language in [0,1].

    Mirrors LanguageIdEngine.lexicalFit (Java): word-boundary matched, for
    endpoint verification re-ranking (Option A).
    """
    if not text or not text.strip() or lang not in LANGS:
        return 0.0
    words = set(text.lower().split())
    en_hits = sum(1 for w in EN_WORDS | EN_COMMON
                  if w and " " not in w and w in words)
    vi_hits = sum(1 for w in VI_WORDS
                  if w and " " not in w and w in words)
    cjk = cjk_ratio(text)
    if lang == "en":
        return _clamp01(0.20 + 0.30 * min(en_hits, 3) - 0.15 * min(vi_hits, 3))
    if lang == "vi":
        return _clamp01(0.20 + 0.30 * min(vi_hits, 3) - 0.15 * min(en_hits, 3))
    if lang == "zh":
        return _clamp01(0.10 + 0.85 * cjk)
    return 0.0


def _normalize(d: dict) -> dict:
    s = sum(d.values())
    if s <= 0:
        return {k: 1.0 / len(d) for k in d}
    return {k: v / s for k, v in d.items()}


@dataclass
class LidResult:
    language: str
    confidence: float
    scores: dict = field(default_factory=dict)


class LanguageIdEngine:
    """Cheap text-evidence LID with EMA smoothing (acoustic hook optional)."""

    def __init__(self, acoustic_scorer=None, ema_alpha: float = EMA_ALPHA):
        self.acoustic_scorer = acoustic_scorer
        self.ema_alpha = ema_alpha
        self.smoothed = {l: 1.0 / 3 for l in LANGS}
        self.history = {l: 1.0 / 3 for l in LANGS}

    def text_evidence(self, text: str) -> dict:
        out = {l: 1.0 / 3 for l in LANGS}
        if not text or not text.strip():
            return out
        cjk = cjk_ratio(text)
        lower = text.lower()
        en_hits = sum(1 for w in EN_WORDS if w in lower)
        vi_hits = sum(1 for w in VI_WORDS if w in lower)
        out = {
            "vi": 0.34 + 0.12 * min(vi_hits, 3) - 0.10 * min(en_hits, 3),
            "en": 0.33 + 0.12 * min(en_hits, 3) - 0.10 * min(vi_hits, 3),
            "zh": 0.33 + 0.60 * cjk - 0.05 * min(vi_hits + en_hits, 3),
        }
        out = {k: max(0.01, v) for k, v in out.items()}
        return _normalize(out)

    def _acoustic(self, audio, partial_text: str) -> dict:
        if self.acoustic_scorer is not None and audio:
            try:
                s = self.acoustic_scorer(audio)
                if s:
                    return _normalize({l: max(0.0, min(1.0, s.get(l, 1 / 3))) for l in LANGS})
            except Exception:
                pass
        cjk = cjk_ratio(partial_text)
        return _normalize({
            "zh": 0.15 + 0.7 * cjk,
            "vi": 0.425 - 0.35 * cjk,
            "en": 0.425 - 0.35 * cjk,
        })

    def classify(self, audio, partial_text: str, token_conf: float, active: str) -> LidResult:
        acoustic = self._acoustic(audio, partial_text)
        text_ev = self.text_evidence(partial_text)
        combined = {}
        for lang in LANGS:
            conf_w = token_conf if lang == active else (1.0 - token_conf) / 2.0
            combined[lang] = (W_ACOUSTIC * acoustic[lang] + W_TEXT * text_ev[lang]
                              + W_CONFIDENCE * conf_w + W_HISTORY * self.history[lang])
        combined = _normalize(combined)
        # EMA (spec OPTIONAL 7) smooths only the telemetry view: the router's
        # hysteresis gate (threshold + margin + 200 ms persistence, spec §12)
        # already rejects single-window flaps, so gating on the EMA as well
        # would double-damp and push switch latency into seconds.
        for lang in LANGS:
            prev = self.smoothed[lang]
            self.smoothed[lang] = self.ema_alpha * combined[lang] + (1 - self.ema_alpha) * prev
        best = max(LANGS, key=lambda l: combined[l])
        for lang in LANGS:
            self.history[lang] = 0.9 * self.history[lang] + 0.1 * (1.0 if lang == best else 0.0)
        self.history = _normalize(self.history)
        return LidResult(best, combined[best], dict(combined))

    def smoothed_scores(self) -> dict:
        """EMA-stability view for UI/telemetry (not the router gate)."""
        return dict(self.smoothed)

    @staticmethod
    def lid_interval_ms(uncertain: bool) -> int:
        return LID_INTERVAL_UNCERTAIN_MS if uncertain else LID_INTERVAL_STABLE_MS
