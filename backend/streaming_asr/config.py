"""Shared constants — mirrors AsrState.java (spec §6/§9/§11/§12/§14/§19)."""
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
SCHEDULER_MS = 160
SCHEDULER_SAMPLES = SAMPLE_RATE * SCHEDULER_MS // 1000  # 2560
SCHEDULER_FALLBACK_MS = 320
CHUNK_SIZE = 16
CHUNK_SIZE_FALLBACK = 32

LID_INTERVAL_STABLE_MS = 600
LID_INTERVAL_UNCERTAIN_MS = 400
LID_INTERVAL_CANDIDATE_MS = 250
LID_WINDOW_MS = 480

W_ACOUSTIC = 0.45
W_TEXT = 0.30
W_CONFIDENCE = 0.15
W_HISTORY = 0.10

# --- Bootstrap acoustic LID (VoxLingua pipeline §5, §7-§8, §17): audio-only,
# never the active-ASR transcript. Thresholds are starting points for tuning.
BOOTSTRAP_MIN_MS = 400
BOOTSTRAP_LID_WINDOW_MS = 600
BOOTSTRAP_HOP_MS = 200
BOOTSTRAP_MAX_MS = 1000
# Give-up cap: past this much speech without a commit, stop burning
# speculative-candidate decodes and keep only the hidden provisional stream
# (mirrors AsrState.BOOTSTRAP_GIVE_UP_MS — avoids CPU decode storms).
BOOTSTRAP_GIVE_UP_MS = 3000
# Cooldown between speculative-candidate passes (each pass = ≤2 decodes).
SPECULATIVE_CANDIDATE_COOLDOWN_MS = 800
BOOTSTRAP_THRESHOLD = 0.70
BOOTSTRAP_MARGIN = 0.15
BOOTSTRAP_W_ACOUSTIC = 0.90
BOOTSTRAP_W_PRIOR = 0.10

# --- Runtime LID weights (fakedemo2 §18, starting point; W_* above stay
# authoritative until retuned against the real acoustic scorer).
RUNTIME_W_ACOUSTIC = 0.65
RUNTIME_W_TEXT = 0.20
RUNTIME_W_CONFIDENCE = 0.05
RUNTIME_W_HISTORY = 0.10

SWITCH_THRESHOLD = 0.72
SWITCH_MARGIN = 0.20
SWITCH_PERSIST_MS = 200

# Endpoint (inter-utterance) bar, spec §13.1: boundaries are cheap switching
# points — no rollback, no committed-text rewrite — so the gate is lower.
ENDPOINT_THRESHOLD = 0.60
ENDPOINT_MARGIN = 0.10

ROLLBACK_MIN_MS = 640
ROLLBACK_MAX_MS = 960
ROLLBACK_DEFAULT_MS = 640

TOKEN_STABLE_UPDATES = 2
EMA_ALPHA = 0.4

VAD_SPEECH_THRESHOLD = 0.5
# Raised 600 → 1000 → 2000 ms: natural mid-sentence pauses must not split
# one sentence into separately translated fragments. A deliberate Stop
# flushes immediately, so this only delays hands-free endpointing.
ENDPOINT_SILENCE_MS = 2000

# Endpoint candidate verification (spec §13.1) — mirrors AsrState.java.
ENDPOINT_VERIFY_MS = 6000
# Switch margin on the COMBINED score (confidence + lexical fit).
ENDPOINT_VERIFY_MARGIN = 0.05
ENDPOINT_VERIFY_LEX_W = 0.35
ENDPOINT_VERIFY_STRONG_LEX = 0.45
ENDPOINT_VERIFY_MIN_TOKENS = 2

LATENCY_P50_TARGET_MS = 350
LATENCY_P95_TARGET_MS = 500

MODEL_FILES = {
    "vi": ("zipformer_vi_encoder.onnx", "zipformer_vi_decoder.onnx",
            "zipformer_vi_joiner.onnx", "zipformer_vi_tokens.txt"),
    "en": ("zipformer_en_encoder.onnx", "zipformer_en_decoder.onnx",
            "zipformer_en_joiner.onnx", "zipformer_en_tokens.txt"),
    "zh": ("zipformer_zh_encoder.int8.onnx", "zipformer_zh_decoder.onnx",
            "zipformer_zh_joiner.int8.onnx", "zipformer_zh_tokens.txt"),
}
VAD_MODEL = "silero_vad.onnx"

# --- VoxLingua107 ECAPA acoustic LID (VoxLingua pipeline §11–§13) ---
VOXLINGUA_MODEL = "voxlingua_lid_ecapa.onnx"
VOXLINGUA_LABELS_JSON = "voxlingua_lid_labels.json"
VOXLINGUA_N_MELS = 60
VOXLINGUA_MIN_GLOBAL_SCORE = 0.25
