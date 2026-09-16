"""VoxLingua107 ECAPA acoustic LID — mirrors Android VoxLingua implementation.

Role per VoxLingua pipeline doc §4:
  - ECAPA answers "What language is being spoken?" (107 classes, 16 kHz mono,
    60-bin FBank, ECAPA-TDNN).
  - Zipformer answers "What was said?"
  - Router answers "When should the active ASR model change?"
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    BOOTSTRAP_THRESHOLD,
    BOOTSTRAP_MARGIN,
    LID_EMA_ALPHA,
    VOXLINGUA_FOREIGN_TOP_REJECT,
    VOXLINGUA_MIN_SUPPORTED_ABS_SCORE,
    VOXLINGUA_N_MELS,
    VOXLINGUA_MIN_GLOBAL_SCORE,
    VOXLINGUA_MODEL,
    VOXLINGUA_LABELS_JSON,
)


class VoxLinguaLabels:
    """107 language label table for speechbrain/lang-id-voxlingua107-ecapa."""

    NUM_LANGUAGES = 107
    IDX_EN = 20
    IDX_VI = 102
    IDX_ZH = 106

    CODES: Tuple[str, ...] = (
        "ab", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn",
        "bo", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el",
        "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl",
        "gn", "gu", "gv", "ha", "haw", "hi", "hr", "ht", "hu", "hy",
        "ia", "id", "is", "it", "iw", "ja", "jw", "ka", "kk", "km",
        "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi",
        "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
        "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sco",
        "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv",
        "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk",
        "ur", "uz", "vi", "war", "yi", "yo", "zh",
    )

    @classmethod
    def code_at(cls, index: int) -> str:
        if 0 <= index < len(cls.CODES):
            return cls.CODES[index]
        return "unk"

    @classmethod
    def index_of(cls, code: str) -> int:
        if not code:
            return -1
        try:
            return cls.CODES.index(code)
        except ValueError:
            return -1

    @classmethod
    def is_supported(cls, code: str) -> bool:
        return code in ("en", "vi", "zh")

    @classmethod
    def to_asr_language(cls, code: str) -> str:
        if code in ("en", "vi", "zh"):
            return code
        return "und"


@dataclass
class LanguageScores:
    """VoxLingua bootstrap and runtime score carrier (§31).

    vi/en/zh are SUPPORTED-RELATIVE posteriors (renormalized among the
    three, sum ≈ 1) used for the 0.70/0.15 relative gate. vi_abs/en_abs/zh_abs
    are ABSOLUTE 107-way posteriors (sum with the other 104 classes ≈ 1) used
    for the absolute-evidence gate: on short windows the 107-class argmax is
    regularly an unrelated language while the relative evidence is already
    decisive, so the gate must read absolute mass, not the argmax label.
    """

    vi: float
    en: float
    zh: float
    global_top_language: str
    global_top_index: int
    global_top_score: float
    num_windows: int = 1
    vi_abs: float = 0.0
    en_abs: float = 0.0
    zh_abs: float = 0.0
    # Smoother-stamped flag (2026-09-16 v2): True when the smoother's full
    # depth of raw windows unanimously agrees on the supported top. Set by
    # VoxLinguaTemporalSmoother.add only — never by the engine.
    unanimous: bool = False

    @classmethod
    def flat(cls) -> "LanguageScores":
        return cls(
            vi=1.0 / 3,
            en=1.0 / 3,
            zh=1.0 / 3,
            global_top_language="unk",
            global_top_index=-1,
            global_top_score=1.0 / 107,
            num_windows=0,
            vi_abs=1.0 / 107,
            en_abs=1.0 / 107,
            zh_abs=1.0 / 107,
        )

    def is_supported_top(self) -> bool:
        """True when the global 107-class winner is one of VI/EN/ZH."""
        return VoxLinguaLabels.is_supported(self.global_top_language)

    def top_supported(self) -> str:
        if self.vi >= self.en and self.vi >= self.zh:
            return "vi"
        if self.en >= self.vi and self.en >= self.zh:
            return "en"
        return "zh"

    def top_supported_score(self) -> float:
        top = self.top_supported()
        if top == "vi":
            return self.vi
        if top == "en":
            return self.en
        return self.zh

    def second_supported_score(self) -> float:
        top = self.top_supported()
        if top == "vi":
            return max(self.en, self.zh)
        if top == "en":
            return max(self.vi, self.zh)
        return max(self.vi, self.en)

    def top_supported_abs(self) -> float:
        top = self.top_supported()
        if top == "vi":
            return self.vi_abs
        if top == "en":
            return self.en_abs
        return self.zh_abs

    def _relative_gate(self) -> bool:
        """Shared relative part: top >= 0.70 AND top - second >= 0.15."""
        top = self.top_supported_score()
        second = self.second_supported_score()
        return (top >= BOOTSTRAP_THRESHOLD) and (
            (top - second) >= BOOTSTRAP_MARGIN)

    def is_bootstrap_confident(self) -> bool:
        """FAST bootstrap gate (§15, 2026-09-16 fix).

        Relative gate PLUS absolute posterior mass (replaces the old
        "107-class argmax must be en/vi/zh" rule). Commits clean audio
        quickly; on noisy device-mic audio the absolute mass stays tiny and
        the SLOW path below takes over instead.
        """
        if not self._relative_gate():
            return False
        return self.top_supported_abs() >= VOXLINGUA_MIN_SUPPORTED_ABS_SCORE

    def is_slow_bootstrap_confident(self) -> bool:
        """SLOW bootstrap gate (2026-09-16 v2, field-log fix).

        Device-mic audio yields flat 107-way distributions (supported abs
        ≈ 0.01) with a CORRECT relative ranking — the absolute bar never
        opens there. The slow path trusts the relative gate once a full
        smoother depth UNANIMOUSLY agrees (single-window flips, which do
        happen in both directions, cannot commit alone). Residue of the §14
        policy: a STRONG unsupported argmax still blocks (true foreign
        audio waits); weak/flat argmax tops do not.
        """
        if not self.unanimous:
            return False
        if not self._relative_gate():
            return False
        if (not self.is_supported_top()
                and self.global_top_score >= VOXLINGUA_FOREIGN_TOP_REJECT):
            return False
        return True

    def to_map(self) -> Dict[str, float]:
        return {"vi": self.vi, "en": self.en, "zh": self.zh}


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filters(n_mels: int, n_fft: int, sr: int, f_min: float, f_max: float) -> np.ndarray:
    n_bins = n_fft // 2 + 1  # 201
    fb = np.zeros((n_bins, n_mels), dtype=np.float32)
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    band = hz_points[1:] - hz_points[:-1]
    f_central = hz_points[1:-1]
    band = band[:-1]
    all_freqs = np.linspace(0, sr / 2, n_bins)
    for m in range(n_mels):
        fc = f_central[m]
        b = band[m]
        slope = (all_freqs - fc) / b
        tri = np.maximum(0.0, np.minimum(slope + 1.0, -slope + 1.0))
        fb[:, m] = tri
    return fb


class VoxLinguaFbankExtractor:
    """Dedicated 60-bin FBank frontend for VoxLingua107 ECAPA (§13)."""

    SAMPLE_RATE = 16000
    N_MELS = 60
    WIN_LENGTH = 400
    HOP_LENGTH = 160
    N_FFT = 400
    F_MIN = 0.0
    F_MAX = 8000.0
    LOG_FLOOR = 1e-10
    TOP_DB = 80.0

    def __init__(self):
        self.mel_filters = _build_mel_filters(
            self.N_MELS, self.N_FFT, self.SAMPLE_RATE, self.F_MIN, self.F_MAX
        )
        indices = np.arange(self.WIN_LENGTH)
        self.window = (0.54 - 0.46 * np.cos(2 * np.pi * indices / (self.WIN_LENGTH - 1))).astype(np.float32)

    @classmethod
    def num_frames_for(cls, num_samples: int) -> int:
        if num_samples < cls.WIN_LENGTH:
            return 0
        pad = cls.N_FFT // 2
        padded_len = num_samples + 2 * pad
        return 1 + (padded_len - cls.WIN_LENGTH) // cls.HOP_LENGTH

    def extract(self, pcm16k_mono: np.ndarray | list) -> np.ndarray:
        """Extract [num_frames, 60] sentence-mean normalized log-mel features."""
        if pcm16k_mono is None:
            return np.empty((0, self.N_MELS), dtype=np.float32)
        pcm = np.asarray(pcm16k_mono, dtype=np.float32)
        if len(pcm) < self.WIN_LENGTH:
            return np.empty((0, self.N_MELS), dtype=np.float32)

        pad = self.N_FFT // 2
        padded = np.pad(pcm, (pad, pad), mode="reflect")
        num_frames = 1 + (len(padded) - self.WIN_LENGTH) // self.HOP_LENGTH
        feats = np.zeros((num_frames, self.N_MELS), dtype=np.float32)

        for t in range(num_frames):
            offset = t * self.HOP_LENGTH
            frame_pcm = padded[offset : offset + self.WIN_LENGTH] * self.window
            fft_res = np.fft.rfft(frame_pcm, n=self.N_FFT)
            power = (fft_res.real ** 2 + fft_res.imag ** 2).astype(np.float32)
            linear_fb = np.dot(power, self.mel_filters)
            linear_fb = np.maximum(linear_fb, self.LOG_FLOOR)
            feats[t] = 10.0 * np.log10(linear_fb)

        max_db = np.max(feats) - self.TOP_DB
        feats = np.maximum(feats, max_db)
        mean = np.mean(feats, axis=0, keepdims=True)
        feats -= mean
        return feats


class VoxLinguaTemporalSmoother:
    """Temporal smoothing across rolling windows (§16).

    Fuses the last N window results with EMA (relative + absolute supported
    posteriors) plus a majority-vote stabilizer on the relative top.

    2026-09-16 fix: the global-top label now FOLLOWS THE LATEST window. The
    old "keep the highest score ever seen" rule latched one noisy window
    (e.g. `lo 0.474` at 400 ms) and locked the whole utterance to UND, since
    later lower-scoring windows could never replace it. The bootstrap gate no
    longer reads the argmax label (it reads absolute mass), so a single noisy
    window can no longer jam the pipeline.
    """

    DEFAULT_DEPTH = 3
    DEFAULT_ALPHA = 0.5

    def __init__(self, depth: int = DEFAULT_DEPTH, alpha: float = DEFAULT_ALPHA):
        self.depth = max(1, depth)
        self.alpha = min(1.0, max(0.0, alpha))
        self.history: collections.deque[LanguageScores] = collections.deque(maxlen=self.depth)
        self.ema: Optional[LanguageScores] = None

    def add(self, raw: Optional[LanguageScores]) -> LanguageScores:
        if raw is None:
            raw = LanguageScores.flat()
        self.history.append(raw)

        if self.ema is None:
            self.ema = raw
        else:
            a = self.alpha
            vi = a * raw.vi + (1 - a) * self.ema.vi
            en = a * raw.en + (1 - a) * self.ema.en
            zh = a * raw.zh + (1 - a) * self.ema.zh
            vi_abs = a * raw.vi_abs + (1 - a) * self.ema.vi_abs
            en_abs = a * raw.en_abs + (1 - a) * self.ema.en_abs
            zh_abs = a * raw.zh_abs + (1 - a) * self.ema.zh_abs

            # Follow the latest window's global top (never latch the max:
            # that locked UND for the whole utterance on one noisy window).
            if raw.num_windows > 0:
                g_top = raw.global_top_language
                g_idx = raw.global_top_index
                g_score = raw.global_top_score
            else:
                g_top = self.ema.global_top_language
                g_idx = self.ema.global_top_index
                g_score = self.ema.global_top_score

            self.ema = LanguageScores(
                vi=vi,
                en=en,
                zh=zh,
                global_top_language=g_top,
                global_top_index=g_idx,
                global_top_score=g_score,
                num_windows=self.ema.num_windows + 1,
                vi_abs=vi_abs,
                en_abs=en_abs,
                zh_abs=zh_abs,
            )

        if len(self.history) >= self.depth:
            vote = self._majority_top()
            if vote and vote == self.ema.top_supported():
                boost = 0.05
                vi = self.ema.vi + (boost if vote == "vi" else 0.0)
                en = self.ema.en + (boost if vote == "en" else 0.0)
                zh = self.ema.zh + (boost if vote == "zh" else 0.0)
                s = vi + en + zh
                self.ema = LanguageScores(
                    vi=vi / s,
                    en=en / s,
                    zh=zh / s,
                    global_top_language=self.ema.global_top_language,
                    global_top_index=self.ema.global_top_index,
                    global_top_score=self.ema.global_top_score,
                    num_windows=self.ema.num_windows,
                    vi_abs=self.ema.vi_abs,
                    en_abs=self.ema.en_abs,
                    zh_abs=self.ema.zh_abs,
                )

        # Stamp unanimity for the SLOW bootstrap gate (2026-09-16 v2): a full
        # depth of raw windows agreeing on the supported top. The majority
        # boost above never changes history, so this stays exact.
        self.ema.unanimous = (
            len(self.history) >= self.depth and self._all_agree())

        return self.ema

    def smoothed(self) -> LanguageScores:
        return self.ema if self.ema is not None else LanguageScores.flat()

    @property
    def size(self) -> int:
        return len(self.history)

    def reset(self) -> None:
        self.history.clear()
        self.ema = None

    def _majority_top(self) -> Optional[str]:
        counts = {"vi": 0, "en": 0, "zh": 0}
        for s in self.history:
            top = s.top_supported()
            if top in counts:
                counts[top] += 1
        need = self.depth // 2 + 1
        for lang, count in counts.items():
            if count >= need:
                return lang
        return None

    def _all_agree(self) -> bool:
        """True when every buffered raw window shares one supported top."""
        tops = {s.top_supported() for s in self.history}
        return len(tops) == 1


def scores_from_logits(logits: np.ndarray | list) -> LanguageScores:
    """Convert raw 107-class logits (or output probabilities) to LanguageScores."""
    vals = np.asarray(logits, dtype=np.float64)
    # Check if values are already probabilities (sum ~= 1.0 and all >= -1e-4)
    if np.all(vals >= -1e-4) and abs(float(np.sum(vals)) - 1.0) < 0.05:
        probs = np.maximum(0.0, vals)
    else:
        m = np.max(vals)
        exp_logits = np.exp(vals - m)
        probs = exp_logits / np.sum(exp_logits)

    top_idx = int(np.argmax(probs))
    top_code = VoxLinguaLabels.code_at(top_idx)
    top_score = float(probs[top_idx])

    p_en = float(probs[VoxLinguaLabels.IDX_EN])
    p_vi = float(probs[VoxLinguaLabels.IDX_VI])
    p_zh = float(probs[VoxLinguaLabels.IDX_ZH])
    s_sum = p_en + p_vi + p_zh

    if s_sum <= 1e-12:
        vi = en = zh = 1.0 / 3
    else:
        vi = float(p_vi / s_sum)
        en = float(p_en / s_sum)
        zh = float(p_zh / s_sum)

    return LanguageScores(
        vi=vi,
        en=en,
        zh=zh,
        global_top_language=top_code,
        global_top_index=top_idx,
        global_top_score=top_score,
        num_windows=1,
        vi_abs=p_vi,
        en_abs=p_en,
        zh_abs=p_zh,
    )


class VoxLinguaAcousticLidEngine:
    """VoxLingua107 ECAPA acoustic LID engine."""

    MIN_SAMPLES = 16000 * 400 // 1000  # 400 ms @ 16 kHz

    def __init__(self, model_path: Optional[str | Path] = None):
        self.fbank = VoxLinguaFbankExtractor()
        self.smoother = VoxLinguaTemporalSmoother(
            alpha=LID_EMA_ALPHA)
        self.session = None
        self.feature_input = None
        self.wav_lens_input = None
        # Back-compat alias (older callers read .input_name).
        self.input_name = None
        self.ready = False
        self.load_error = None
        self.inference_count = 0

        if model_path:
            self._init_session(Path(model_path))

    def _init_session(self, path: Path) -> None:
        if not path.exists():
            self.load_error = f"Model file not found: {path}"
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(path), sess_options=opts)
            # Resolve by role, not by order: ['features' [1,T,60],
            # 'wav_lens' [1]]. Feeding only features crashes 100% of runs.
            feat = None
            lens = None
            for inp in self.session.get_inputs():
                n = inp.name or ""
                if n.lower() == "wav_lens":
                    lens = inp.name
                elif feat is None:
                    feat = inp.name
            if feat is None:
                feat = self.session.get_inputs()[0].name
            self.feature_input = feat
            self.wav_lens_input = lens
            self.input_name = feat
            self.ready = True
        except Exception as e:
            self.load_error = str(e)

    def is_ready(self) -> bool:
        return self.ready and self.session is not None

    def classify(self, audio) -> Dict[str, float]:
        s = self.classify_detailed(audio)
        return s.to_map()

    def classify_detailed(self, audio) -> LanguageScores:
        raw = self._infer_raw(audio)
        return self.smoother.add(raw)

    def reset(self) -> None:
        self.smoother.reset()

    def smoothed(self) -> LanguageScores:
        """Last smoothed aggregate (mirrors AcousticLidEngine.smoothed)."""
        return self.smoother.smoothed()

    def _infer_raw(self, audio) -> LanguageScores:
        if audio is None or len(audio) < self.MIN_SAMPLES:
            return LanguageScores.flat()
        if not self.is_ready():
            return LanguageScores.flat()

        feats = self.fbank.extract(audio)
        if len(feats) < 10:
            return LanguageScores.flat()

        feats_input = feats[np.newaxis, :, :].astype(np.float32)
        try:
            feat_key = self.feature_input or self.input_name
            if feat_key is None:
                # Last-resort role resolution (mirrors Java runSession).
                names = [inp.name for inp in self.session.get_inputs()]
                feat_key = next(
                    (n for n in names if (n or "").lower() != "wav_lens"),
                    names[0],
                )
            feed = {feat_key: feats_input}
            input_names = [inp.name for inp in self.session.get_inputs()]
            lens_key = self.wav_lens_input
            if lens_key is None:
                lens_key = next(
                    (n for n in input_names if (n or "").lower() == "wav_lens"),
                    None,
                )
            if lens_key is not None:
                # Relative length in batch (batch=1 → always 1.0).
                # Missing it → OrtException on every run → flat forever.
                feed[lens_key] = np.array([1.0], dtype=np.float32)
            out = self.session.run(None, feed)
            logits = out[0].squeeze()
            if len(logits) != VoxLinguaLabels.NUM_LANGUAGES:
                return LanguageScores.flat()
            scores = scores_from_logits(logits)
            self.inference_count += 1
            return scores
        except Exception:
            return LanguageScores.flat()
