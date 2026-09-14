"""Rollback window — mirrors RollbackManager.java (spec §14-§15)."""
import time
from .config import ROLLBACK_DEFAULT_MS, ROLLBACK_MAX_MS, ROLLBACK_MIN_MS, SAMPLE_RATE


class RollbackManager:
    def __init__(self, window_ms: int = ROLLBACK_DEFAULT_MS):
        self.window_ms = min(max(window_ms, ROLLBACK_MIN_MS), ROLLBACK_MAX_MS)
        self.rollback_start_ms = -1
        self.rollback_audio = None
        self.candidate = None

    def start_rollback(self, ring, candidate: str, now_ms: int = None):
        self.candidate = candidate
        self.rollback_start_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.rollback_audio = ring.last_ms(self.window_ms)
        return self.rollback_audio

    def expected_samples(self) -> int:
        return SAMPLE_RATE * self.window_ms // 1000

    def clear(self) -> None:
        self.rollback_audio = None
        self.candidate = None
        self.rollback_start_ms = -1

    @property
    def has_pending(self) -> bool:
        return self.rollback_audio is not None and self.candidate is not None
