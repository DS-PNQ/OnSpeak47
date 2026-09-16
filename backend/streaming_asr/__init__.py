"""Streaming multilingual ASR mirror (VI/EN/ZH) — Python reference for local tests.

Mirrors android/.../asr/ (spec realtime_multilingual_asr_vi_en_zh_android.md).
Dependency-free (stdlib only; numpy optional in vad helpers) so
tests_local/test_07_streaming_asr.py runs without model downloads.
"""
from .config import (
    SAMPLE_RATE, FRAME_MS, FRAME_SAMPLES, SCHEDULER_MS, SCHEDULER_SAMPLES,
    LID_INTERVAL_STABLE_MS, LID_INTERVAL_UNCERTAIN_MS,
    LID_INTERVAL_CANDIDATE_MS, SWITCH_THRESHOLD,
    SWITCH_MARGIN, SWITCH_PERSIST_MS, ENDPOINT_THRESHOLD, ENDPOINT_MARGIN,
    ROLLBACK_MIN_MS, ROLLBACK_MAX_MS,
    ROLLBACK_DEFAULT_MS, TOKEN_STABLE_UPDATES, EMA_ALPHA,
    BOOTSTRAP_MIN_MS, BOOTSTRAP_LID_WINDOW_MS, BOOTSTRAP_HOP_MS,
    BOOTSTRAP_MAX_MS, BOOTSTRAP_GIVE_UP_MS,
    SPECULATIVE_CANDIDATE_COOLDOWN_MS,
    BOOTSTRAP_THRESHOLD, BOOTSTRAP_MARGIN,
    CANDIDATE_COMMIT_SCORE, CANDIDATE_MARGIN,
    PROVISIONAL_MIN_SUPPORTED_REL, REFEREE_MAX_ATTEMPTS,
    BOOTSTRAP_W_ACOUSTIC, BOOTSTRAP_W_PRIOR,
    RUNTIME_W_ACOUSTIC, RUNTIME_W_TEXT,
    RUNTIME_W_CONFIDENCE, RUNTIME_W_HISTORY,
    BOOTSTRAP_ENDPOINT_WINDOW_MS,
    VOXLINGUA_MIN_SUPPORTED_ABS_SCORE, VOXLINGUA_FOREIGN_TOP_REJECT,
    LID_EMA_ALPHA,
)
from .ring_buffer import AudioRingBuffer
from .vad import VadEngine, VadResult
from .lid import LanguageIdEngine, LidResult, heuristic_acoustic_scorer
from .router import LanguageRouter, RouterState
from .partial import PartialTranscriptManager
from .rollback import RollbackManager
from .metrics import AsrMetrics
from .model_manager import ZipformerModelManager, FakeEngine, StreamingAsrEngine
from .pipeline import StreamingPipeline

__all__ = [
    "SAMPLE_RATE", "FRAME_MS", "FRAME_SAMPLES", "SCHEDULER_MS", "SCHEDULER_SAMPLES",
    "LID_INTERVAL_STABLE_MS", "LID_INTERVAL_UNCERTAIN_MS",
    "LID_INTERVAL_CANDIDATE_MS", "SWITCH_THRESHOLD",
    "SWITCH_MARGIN", "SWITCH_PERSIST_MS", "ENDPOINT_THRESHOLD", "ENDPOINT_MARGIN",
    "ROLLBACK_MIN_MS", "ROLLBACK_MAX_MS",
    "ROLLBACK_DEFAULT_MS", "TOKEN_STABLE_UPDATES", "EMA_ALPHA",
    "BOOTSTRAP_MIN_MS", "BOOTSTRAP_LID_WINDOW_MS", "BOOTSTRAP_HOP_MS",
    "BOOTSTRAP_MAX_MS", "BOOTSTRAP_GIVE_UP_MS",
    "SPECULATIVE_CANDIDATE_COOLDOWN_MS",
    "BOOTSTRAP_THRESHOLD", "BOOTSTRAP_MARGIN",
    "CANDIDATE_COMMIT_SCORE", "CANDIDATE_MARGIN",
    "PROVISIONAL_MIN_SUPPORTED_REL", "REFEREE_MAX_ATTEMPTS",
    "BOOTSTRAP_ENDPOINT_WINDOW_MS",
    "VOXLINGUA_MIN_SUPPORTED_ABS_SCORE", "VOXLINGUA_FOREIGN_TOP_REJECT",
    "LID_EMA_ALPHA",
    "BOOTSTRAP_W_ACOUSTIC", "BOOTSTRAP_W_PRIOR",
    "RUNTIME_W_ACOUSTIC", "RUNTIME_W_TEXT",
    "RUNTIME_W_CONFIDENCE", "RUNTIME_W_HISTORY",
    "AudioRingBuffer", "VadEngine", "VadResult", "LanguageIdEngine", "LidResult",
    "heuristic_acoustic_scorer",
    "LanguageRouter", "RouterState", "PartialTranscriptManager", "RollbackManager",
    "AsrMetrics", "ZipformerModelManager", "FakeEngine", "StreamingAsrEngine",
    "StreamingPipeline",
]
