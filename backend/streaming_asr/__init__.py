"""Streaming multilingual ASR mirror (VI/EN/ZH) — Python reference for local tests.

Mirrors android/.../asr/ (spec realtime_multilingual_asr_vi_en_zh_android.md).
Dependency-free (stdlib only; numpy optional in vad helpers) so
tests_local/test_07_streaming_asr.py runs without model downloads.
"""
from .config import (
    SAMPLE_RATE, FRAME_MS, FRAME_SAMPLES, SCHEDULER_MS, SCHEDULER_SAMPLES,
    LID_INTERVAL_STABLE_MS, LID_INTERVAL_UNCERTAIN_MS, SWITCH_THRESHOLD,
    SWITCH_MARGIN, SWITCH_PERSIST_MS, ENDPOINT_THRESHOLD, ENDPOINT_MARGIN,
    ROLLBACK_MIN_MS, ROLLBACK_MAX_MS,
    ROLLBACK_DEFAULT_MS, TOKEN_STABLE_UPDATES, EMA_ALPHA,
)
from .ring_buffer import AudioRingBuffer
from .vad import VadEngine, VadResult
from .lid import LanguageIdEngine, LidResult
from .router import LanguageRouter, RouterState
from .partial import PartialTranscriptManager
from .rollback import RollbackManager
from .metrics import AsrMetrics
from .model_manager import ZipformerModelManager, FakeEngine, StreamingAsrEngine
from .pipeline import StreamingPipeline

__all__ = [
    "SAMPLE_RATE", "FRAME_MS", "FRAME_SAMPLES", "SCHEDULER_MS", "SCHEDULER_SAMPLES",
    "LID_INTERVAL_STABLE_MS", "LID_INTERVAL_UNCERTAIN_MS", "SWITCH_THRESHOLD",
    "SWITCH_MARGIN", "SWITCH_PERSIST_MS", "ENDPOINT_THRESHOLD", "ENDPOINT_MARGIN",
    "ROLLBACK_MIN_MS", "ROLLBACK_MAX_MS",
    "ROLLBACK_DEFAULT_MS", "TOKEN_STABLE_UPDATES", "EMA_ALPHA",
    "AudioRingBuffer", "VadEngine", "VadResult", "LanguageIdEngine", "LidResult",
    "LanguageRouter", "RouterState", "PartialTranscriptManager", "RollbackManager",
    "AsrMetrics", "ZipformerModelManager", "FakeEngine", "StreamingAsrEngine",
    "StreamingPipeline",
]
