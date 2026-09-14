"""Energy VAD + endpointing — mirrors VadEngine.java (spec §8)."""
import math
from dataclasses import dataclass
from .config import VAD_SPEECH_THRESHOLD, ENDPOINT_SILENCE_MS

ENERGY_SPEECH_RMS = 0.01
ENERGY_SILENCE_RMS = 0.006


@dataclass
class VadResult:
    is_speech: bool
    prob: float
    speech_start: bool
    speech_end: bool


class VadEngine:
    def __init__(self):
        self.in_speech = False
        self._silence_since_ms = -1
        self.last_prob = 0.0

    def _energy_prob(self, frame) -> float:
        if not frame:
            return 0.0
        mean_sq = sum(s * s for s in frame) / len(frame)
        rms = math.sqrt(mean_sq)
        thr = ENERGY_SILENCE_RMS if self.in_speech else ENERGY_SPEECH_RMS
        if rms >= thr:
            return min(1.0, 0.55 + rms * 20.0)
        return max(0.0, rms * 20.0)

    def process(self, frame, now_ms: int) -> VadResult:
        prob = self._energy_prob(frame)
        self.last_prob = prob
        speech = prob >= VAD_SPEECH_THRESHOLD
        start = end = False
        if speech:
            if not self.in_speech:
                self.in_speech = True
                start = True
            self._silence_since_ms = -1
        elif self.in_speech:
            if self._silence_since_ms < 0:
                self._silence_since_ms = now_ms
            if now_ms - self._silence_since_ms >= ENDPOINT_SILENCE_MS:
                self.in_speech = False
                end = True
                self._silence_since_ms = -1
        return VadResult(bool(self.in_speech or speech), prob, start, end)

    def reset(self) -> None:
        self.in_speech = False
        self._silence_since_ms = -1
