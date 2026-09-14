"""Metrics — mirrors AsrMetrics.java (spec §26-§28)."""
import statistics

AUDIO_RECEIVED = "timestamp_audio_received"
VAD = "timestamp_vad"
ASR_START = "timestamp_asr_start"
ASR_END = "timestamp_asr_end"
PARTIAL_VISIBLE = "timestamp_partial_visible"
LID_START = "timestamp_lid_start"
LID_END = "timestamp_lid_end"
SWITCH_DETECTED = "timestamp_switch_detected"
ROLLBACK_START = "timestamp_rollback_start"
ROLLBACK_END = "timestamp_rollback_end"
COMMIT = "timestamp_commit"


def percentile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    s = sorted(sorted_vals)
    if len(s) == 1:
        return float(s[0])
    rank = p / 100.0 * (len(s) - 1)
    lo, hi = int(rank // 1), int(-(-rank // 1))
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return s[lo] + frac * (s[hi] - s[lo])


class AsrMetrics:
    def __init__(self):
        self.events = {}
        self.partial_latencies = []
        self.switch_latencies = []
        self.rtf_samples = []
        self.dropped_frames = 0
        self.utterances = 0

    def mark(self, event: str, at_ms: int = None) -> None:
        import time
        self.events[event] = at_ms if at_ms is not None else int(time.time() * 1000)

    def add_partial_latency(self, ms: float) -> None:
        self.partial_latencies.append(ms)

    def add_switch_latency(self, ms: float) -> None:
        self.switch_latencies.append(ms)

    def add_rtf(self, rtf: float) -> None:
        self.rtf_samples.append(rtf)

    def add_dropped_frames(self, n: int) -> None:
        self.dropped_frames += n

    def partial_p(self, p: float) -> float:
        return percentile(self.partial_latencies, p)

    def mean_rtf(self) -> float:
        if not self.rtf_samples:
            return float("nan")
        return sum(self.rtf_samples) / len(self.rtf_samples)

    def meets_targets(self, p50_target=350, p95_target=500) -> bool:
        import math
        p50, p95 = self.partial_p(50), self.partial_p(95)
        if math.isnan(p50) or math.isnan(p95):
            return False
        return p50 < p50_target and p95 < p95_target

    def summary(self) -> dict:
        return {
            "partial_p50": self.partial_p(50),
            "partial_p90": self.partial_p(90),
            "partial_p95": self.partial_p(95),
            "partial_p99": self.partial_p(99),
            "mean_rtf": self.mean_rtf(),
            "dropped_frames": self.dropped_frames,
            "utterances": self.utterances,
            "meets_targets": self.meets_targets(),
        }
