"""Ring buffer — mirrors AudioRingBuffer.java (spec §21 MUST 3, §14)."""
from .config import SAMPLE_RATE


class AudioRingBuffer:
    def __init__(self, capacity_samples: int):
        if capacity_samples <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity_samples
        self._buf = [0.0] * capacity_samples
        self.write_pos = 0

    @classmethod
    def with_capacity_ms(cls, ms: int) -> "AudioRingBuffer":
        return cls(SAMPLE_RATE * ms // 1000)

    def append(self, samples) -> None:
        for s in samples:
            self._buf[self.write_pos % self.capacity] = float(s)
            self.write_pos += 1

    def available(self) -> int:
        return min(self.write_pos, self.capacity)

    def dropped_count(self) -> int:
        return max(0, self.write_pos - self.capacity)

    def last(self, n: int):
        n = min(n, self.available())
        start = self.write_pos - n
        return [self._buf[(start + i) % self.capacity] for i in range(n)]

    def last_ms(self, ms: int):
        return self.last(SAMPLE_RATE * ms // 1000)

    def since(self, start_pos: int, max_samples: int):
        """Samples appended since start_pos, oldest-first, capped.

        Mirrors AudioRingBuffer.since (Java): replays the utterance from its
        speech start (§15, §25) — last_ms would return trailing silence when
        called at an endpoint after seconds of silence.
        """
        end = self.write_pos
        start = max(start_pos, end - self.capacity)
        start = max(start, end - max(0, max_samples))
        n = max(0, end - start)
        return [self._buf[(start + i) % self.capacity] for i in range(n)]

    def clear(self) -> None:
        self.write_pos = 0
