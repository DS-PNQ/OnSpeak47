"""Stage-1 LID — mirrors LanguageIdEngine.java (spec §10-§12, §16-§17)."""
from dataclasses import dataclass, field
from .config import (
    W_ACOUSTIC, W_TEXT, W_CONFIDENCE, W_HISTORY, EMA_ALPHA,
    LID_INTERVAL_STABLE_MS, LID_INTERVAL_UNCERTAIN_MS, LID_INTERVAL_CANDIDATE_MS,
    BOOTSTRAP_W_ACOUSTIC, BOOTSTRAP_W_PRIOR,
    RUNTIME_W_ACOUSTIC, RUNTIME_W_TEXT, RUNTIME_W_CONFIDENCE, RUNTIME_W_HISTORY,
    VOXLINGUA_MIN_GLOBAL_SCORE,
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


def _voiced_ratio(samples) -> float:
    """Fraction of 20 ms frames with strong voicing (autocorrelation F0)."""
    frame = 16000 * 20 // 1000
    voiced = total = 0
    n = len(samples)
    for start in range(0, n - frame + 1, frame):
        total += 1
        seg = samples[start:start + frame]
        e0 = sum(s * s for s in seg)
        if e0 < 1e-6:
            continue
        best = 0.0
        for lag in range(40, 201, 4):
            if start + lag + frame > n:
                break
            corr = sum(seg[i] * samples[start + i + lag]
                       for i in range(frame - lag))
            e1 = sum(samples[start + i + lag] ** 2 for i in range(frame - lag))
            denom = (e0 * e1) ** 0.5
            if denom > 0:
                best = max(best, corr / denom)
        if best > 0.45:
            voiced += 1
    return voiced / total if total else 0.5


def _sibilant_ratio(samples) -> float:
    """Fraction of frames dominated by zero-crossings (frication)."""
    frame = 16000 * 20 // 1000
    sib = total = 0
    for start in range(0, len(samples) - frame + 1, frame):
        total += 1
        seg = samples[start:start + frame]
        zc = sum(1 for i in range(1, frame)
                 if (seg[i] >= 0) != (seg[i - 1] >= 0))
        rms = (sum(s * s for s in seg) / frame) ** 0.5
        if zc / frame > 0.28 and rms > 0.015:
            sib += 1
    return sib / total if total else 0.0


def heuristic_acoustic_scorer(audio) -> dict:
    """Interim audio-derived LID (mirrors HeuristicAcousticLidEngine).

    Prosodic time-domain cues only; deliberately UNCERTAIN by design (top
    capped at 0.58, below the 0.70 bootstrap bar) so utterances fall through
    to the dual-candidate path instead of being forced into VI. Swap for the
    trained tiny classifier (§33) without touching callers.
    """
    if not audio or len(audio) < 16000 // 10:
        return {"vi": 1.0 / 3, "en": 1.0 / 3, "zh": 1.0 / 3}
    # Silence carries no language information: score flat instead of letting
    # a voicing bias vote on near-zero audio (e.g. an endpoint window that is
    # mostly trailing silence).
    energy = (sum(s * s for s in audio) / len(audio)) ** 0.5
    if energy < 0.008:
        return {"vi": 1.0 / 3, "en": 1.0 / 3, "zh": 1.0 / 3}
    voiced = _voiced_ratio(list(audio))
    sib = _sibilant_ratio(list(audio))
    vi = 1.0 + 0.35 * (voiced - 0.5) - 0.30 * sib
    en = 1.0 - 0.25 * (voiced - 0.5) + 0.35 * sib
    zh = 1.0 + 0.18 * (voiced - 0.5) - 0.10 * sib
    s = vi + en + zh
    out = {"vi": vi / s, "en": en / s, "zh": zh / s}
    top = max(out.values())
    if top > 0.58:
        winner = max(out, key=lambda k: out[k])
        excess = top - 0.58
        out[winner] = 0.58
        for k in out:
            if k != winner:
                out[k] += excess / 2
    return out


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

    def _acoustic_audio_only(self, audio) -> dict:
        """Audio-only scores for BOOTSTRAP (§17, §19): never the transcript.

        With no scorer this returns a flat prior (UNCERTAIN → extend window
        / dual-candidate) instead of the text-derived fallback — CJK == 0
        must not rule out ZH before any transcript exists.
        """
        if self.acoustic_scorer is not None and audio:
            try:
                s = self.acoustic_scorer(audio)
                if s:
                    return _normalize({l: max(0.0, min(1.0, s.get(l, 1 / 3)))
                                       for l in LANGS})
            except Exception:
                pass
        return {l: 1.0 / 3 for l in LANGS}

    def has_real_acoustic(self) -> bool:
        """True when a real VoxLingua session is ready."""
        return (self.acoustic_scorer is not None
                and hasattr(self.acoustic_scorer, "is_ready")
                and self.acoustic_scorer.is_ready())

    def reset(self) -> None:
        """Clear smoothed telemetry, history, and acoustic engine state."""
        self.smoothed = {l: 1.0 / 3 for l in LANGS}
        self.history = {l: 1.0 / 3 for l in LANGS}
        if hasattr(self.acoustic_scorer, "reset"):
            try:
                self.acoustic_scorer.reset()
            except Exception:
                pass

    def classify_bootstrap(self, audio) -> LidResult:
        """Bootstrap classification (VoxLingua §10, §14–§15): audio PRIMARY.

        bootstrapScore = 0.90 * acoustic + 0.10 * uniform prior. Text,
        history and ASR confidence are disabled; side-effect free (must not
        drift runtime history/EMA).
        """
        if not audio:
            return LidResult("und", 0.0, {l: 1.0 / 3 for l in LANGS})

        # Detailed VoxLingua path (§14): global-top policy first.
        if hasattr(self.acoustic_scorer, "classify_detailed"):
            try:
                detailed = self.acoustic_scorer.classify_detailed(audio)
                if detailed is not None and getattr(detailed, "num_windows", 0) > 0:
                    if not detailed.is_supported_top():
                        return LidResult("und", 0.0, {l: 1.0 / 3 for l in LANGS})
                    if detailed.global_top_score < VOXLINGUA_MIN_GLOBAL_SCORE:
                        return LidResult("und", 0.0, {l: 1.0 / 3 for l in LANGS})
                    fused = _normalize({
                        "vi": BOOTSTRAP_W_ACOUSTIC * detailed.vi + BOOTSTRAP_W_PRIOR * (1.0 / 3),
                        "en": BOOTSTRAP_W_ACOUSTIC * detailed.en + BOOTSTRAP_W_PRIOR * (1.0 / 3),
                        "zh": BOOTSTRAP_W_ACOUSTIC * detailed.zh + BOOTSTRAP_W_PRIOR * (1.0 / 3),
                    })
                    best = max(LANGS, key=lambda l: fused[l])
                    return LidResult(best, fused[best], dict(fused))
            except Exception:
                pass

        acoustic = self._acoustic_audio_only(audio)
        fused = _normalize({l: BOOTSTRAP_W_ACOUSTIC * acoustic[l]
                            + BOOTSTRAP_W_PRIOR * (1.0 / 3) for l in LANGS})
        best = max(LANGS, key=lambda l: fused[l])
        return LidResult(best, fused[best], dict(fused))

    def classify(self, audio, partial_text: str, token_conf: float, active: str) -> LidResult:
        acoustic = self._acoustic(audio, partial_text)
        text_ev = self.text_evidence(partial_text)

        real_acoustic = self.has_real_acoustic()
        w_a = RUNTIME_W_ACOUSTIC if real_acoustic else W_ACOUSTIC
        w_t = RUNTIME_W_TEXT if real_acoustic else W_TEXT
        w_c = RUNTIME_W_CONFIDENCE if real_acoustic else W_CONFIDENCE
        w_h = RUNTIME_W_HISTORY if real_acoustic else W_HISTORY

        combined = {}
        for lang in LANGS:
            conf_w = token_conf if lang == active else (1.0 - token_conf) / 2.0
            combined[lang] = (w_a * acoustic[lang] + w_t * text_ev[lang]
                              + w_c * conf_w + w_h * self.history[lang])
        combined = _normalize(combined)
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
    def lid_interval_ms(uncertain: bool, has_candidate: bool = False) -> int:
        if has_candidate:
            return LID_INTERVAL_CANDIDATE_MS
        return LID_INTERVAL_UNCERTAIN_MS if uncertain else LID_INTERVAL_STABLE_MS
