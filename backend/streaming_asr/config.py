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
#
# 2026-09-16 LID-bootstrap fix (mirrors AsrState.java): measured with the
# bundled VoxLingua107 ECAPA export, a fixed 600 ms window sits below the
# argmax-flip point for both EN (~1.5 s) and VI (~1.0 s), so the gate never
# opened for EN/ZH. The pipeline now grows the window from the baseline up
# to BOOTSTRAP_MAX_MS as speech accumulates. See
# docs/voxlingua_lid_diagnosis.md.
BOOTSTRAP_MIN_MS = 400
BOOTSTRAP_LID_WINDOW_MS = 1500
BOOTSTRAP_HOP_MS = 200
# Live growing-window cap (2026-09-16 v2: 2500 → 1500). ECAPA cost scales
# with frames and the slow bootstrap path (relative + 3-hop persistence)
# commits from 600–1000 ms windows; 1500 ms is plenty live. Endpoints reuse
# the wider BOOTSTRAP_ENDPOINT_WINDOW_MS below (no realtime pressure).
BOOTSTRAP_MAX_MS = 1500
# Bootstrap window cap at VAD endpoints (mirrors AsrState).
BOOTSTRAP_ENDPOINT_WINDOW_MS = 2500
# Give-up cap: past this much speech without a commit, stop burning
# speculative-candidate decodes and keep only the hidden provisional stream
# (mirrors AsrState.BOOTSTRAP_GIVE_UP_MS — avoids CPU decode storms).
BOOTSTRAP_GIVE_UP_MS = 6000
# Cooldown between speculative-candidate passes (each pass = ≤2 decodes).
SPECULATIVE_CANDIDATE_COOLDOWN_MS = 800
BOOTSTRAP_THRESHOLD = 0.70
BOOTSTRAP_MARGIN = 0.15
# Unified bar for the speculative-candidate path: a one-shot decode used to
# commit at 0.30 while the acoustic gate demands 0.70 — every rejected LID
# fell through to a low-confidence commit, almost always Vietnamese (the
# candidate pair defaulted to [VI, EN]). Candidates now need the same
# evidence level as an acoustic commit, plus a margin over the runner-up.
CANDIDATE_COMMIT_SCORE = 0.65
CANDIDATE_MARGIN = 0.10
# Minimum supported-relative score before the provisional (hidden) decoder
# follows acoustic evidence away from VI. Below this the pipeline keeps the
# UI quiet instead of decoding on a guess.
PROVISIONAL_MIN_SUPPORTED_REL = 0.60
# Max bootstrap/referee classifies per decision phase (§6.1, mirrors
# AsrState). The budget refreshes on every commit, so a weak fallback commit
# gets its own referee passes. After the budget the runtime LID and endpoint
# verification remain (CPU bound, not correctness).
REFEREE_MAX_ATTEMPTS = 6
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
# LEGACY: the bootstrap gate no longer reads this — the 107-class argmax
# proved unreliable at the pipeline's window sizes, so the gate moved to
# VOXLINGUA_MIN_SUPPORTED_ABS_SCORE. Kept for reference (mirrors AsrState).
VOXLINGUA_MIN_GLOBAL_SCORE = 0.25
# Minimum EMA absolute posterior of the supported winner (en/vi/zh) for a
# bootstrap commit. Replaces "the 107-class argmax must be en/vi/zh": on a
# 400–600 ms window the ECAPA argmax is regularly an unrelated language
# (eu/br/cy/nn/ja/lo) while the supported-relative evidence is already
# 0.93–1.00. Measured: abs(en) ≈ 0.12 @600 ms, 0.78 @1500 ms; abs(vi) ≈
# 0.88–0.97 @600–1000 ms. The high absolute value keeps the §14 policy the
# argmax rule was written for (Japanese audio scores abs(zh) ≈ 0.05).
VOXLINGUA_MIN_SUPPORTED_ABS_SCORE = 0.45
# Foreign-audio guard for the SLOW bootstrap path (2026-09-16 v2): a STRONG
# unsupported 107-class argmax means true foreign audio — wait instead of
# forcing VI/EN/ZH. Weak/flat argmax tops (the normal short-window case on
# device mic audio, e.g. nn 0.26) do NOT block the slow path.
VOXLINGUA_FOREIGN_TOP_REJECT = 0.60
# EMA coefficient for the smoothed absolute supported posteriors (§16).
LID_EMA_ALPHA = 0.4
