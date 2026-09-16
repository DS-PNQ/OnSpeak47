/*
 * OmniVoice — Streaming ASR tuning constants + router states.
 *
 * Spec §6, §9, §11-§12, §14, §19-§20, §21 (MUST 5/6).
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Central tuning constants. Latency/RAM numbers are engineering targets. */
public final class AsrState {

    private AsrState() {}

    // --- Audio frontend (spec §7) ---
    public static final int SAMPLE_RATE = 16000;
    public static final int FRAME_MS = 20;
    public static final int FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000; // 320

    // --- ASR scheduler (spec §9) ---
    public static final int SCHEDULER_MS = 160;
    public static final int SCHEDULER_SAMPLES = SAMPLE_RATE * SCHEDULER_MS / 1000; // 2560
    public static final int SCHEDULER_FALLBACK_MS = 320;
    /** Zipformer streaming chunk baseline (chunk16; chunk32 fallback, spec §20). */
    public static final int CHUNK_SIZE = 16;
    public static final int CHUNK_SIZE_FALLBACK = 32;
    /**
     * Transducer architecture variant passed as OnlineModelConfig modelType.
     * All three bundled encoders carry model_type=zipformer2 in their own
     * metadata (verified via onnxruntime ModelMetadata, including the hynt
     * VI file) — EN 2023-06-26 and ZH-int8 2025-06-30 match the upstream
     * Kotlin demo pairings too. Do NOT set "zipformer": the v1 wrapper
     * requires an attention_dims metadata key these files don't have and
     * aborts natively at startup (no Java stack).
     */
    public static String transducerModelType(AsrLanguage lang) {
        return "zipformer2";
    }

    // --- LID scheduling (spec §11 + VoxLingua pipeline §22) ---
    // ECAPA is a referee, not a realtime decoder (ASR still ticks every
    // 160 ms). Stable → 500–800 ms; uncertain → 300–400 ms; an observed
    // switch candidate is re-polled every 200–300 ms until it passes or
    // the hysteresis persistence window expires.
    public static final int LID_INTERVAL_STABLE_MS = 600;
    public static final int LID_INTERVAL_UNCERTAIN_MS = 400;
    public static final int LID_INTERVAL_CANDIDATE_MS = 250;
    public static final int LID_WINDOW_MS = 480;

    // --- Router scoring (spec §12) ---
    public static final float W_ACOUSTIC = 0.45f;
    public static final float W_TEXT = 0.30f;
    public static final float W_CONFIDENCE = 0.15f;
    public static final float W_HISTORY = 0.10f;

    // --- Bootstrap acoustic LID (VoxLingua pipeline §5, §7-§8, §17) ---
    // Bootstrap answers "which language does this utterance START with?"
    // from AUDIO only — never from the active-ASR transcript (that path is
    // circular: VI decodes EN/ZH into Vietnamese-like text which then votes
    // VI again). Baseline engineering windows for the VoxLingua107 ECAPA
    // classifier (600 ms window / 200 ms hop); 200 ms hard-locks too early
    // (not enough acoustic evidence). Thresholds are starting points for
    // on-device tuning, not absolute optima.
    /** Minimum speech audio before the first bootstrap attempt. */
    public static final int BOOTSTRAP_MIN_MS = 400;
    /** Audio window fed to the acoustic LID classifier. */
    public static final int BOOTSTRAP_LID_WINDOW_MS = 600;
    /** Rolling hop between bootstrap attempts (temporal smoothing §16). */
    public static final int BOOTSTRAP_HOP_MS = 200;
    /** Extended window when the first attempt is uncertain (§9 cách 1). */
    public static final int BOOTSTRAP_MAX_MS = 1000;
    /** Give-up cap: past this much speech without a commit, stop burning
     *  speculative-candidate decodes (2 shadow decodes per pass) and keep
     *  only the hidden provisional stream — avoids a CPU decode storm on
     *  long UNKNOWN spans (unsupported language / noise). Endpoint recovery
     *  still gets one final chance. */
    public static final int BOOTSTRAP_GIVE_UP_MS = 3000;
    /** Cooldown between speculative-candidate passes (each pass = ≤2 decodes). */
    public static final int SPECULATIVE_CANDIDATE_COOLDOWN_MS = 800;
    /** top1 must reach this to commit a bootstrap language. */
    public static final float BOOTSTRAP_THRESHOLD = 0.70f;
    /** top1 - top2 must reach this (never force VI on max-probability). */
    public static final float BOOTSTRAP_MARGIN = 0.15f;
    /** bootstrapScore = 0.90 * acoustic + 0.10 * prior (text/history off). */
    public static final float BOOTSTRAP_W_ACOUSTIC = 0.90f;
    public static final float BOOTSTRAP_W_PRIOR = 0.10f;

    // --- Runtime LID weights (VoxLingua pipeline §10, starting point) ---
    // Kept separate from W_* below (which the current tuned pipeline + tests
    // rely on for the heuristic fallback). Once the VoxLingua acoustic scorer
    // is resident, LanguageIdEngine prefers these: acoustic is PRIMARY,
    // text/ASR-confidence are secondary, history only stabilizes.
    // ASR confidence only says the model trusts its own hypothesis, not that
    // the audio is that language — so its weight stays minimal.
    public static final float RUNTIME_W_ACOUSTIC = 0.65f;
    public static final float RUNTIME_W_TEXT = 0.20f;
    public static final float RUNTIME_W_CONFIDENCE = 0.05f;
    public static final float RUNTIME_W_HISTORY = 0.10f;

    public static final float SWITCH_THRESHOLD = 0.72f;
    public static final float SWITCH_MARGIN = 0.20f;
    /** Endpoint (inter-utterance) bar, spec §13.1: boundaries are cheap
     *  switching points — no rollback, no committed-text rewrite. */
    public static final float ENDPOINT_THRESHOLD = 0.60f;
    public static final float ENDPOINT_MARGIN = 0.10f;
    /** Hysteresis persistence: condition must hold this long (spec §12). */
    public static final long SWITCH_PERSIST_MS = 200L;
    public static final long SWITCH_PERSIST_MAX_MS = 300L;

    // --- Rollback (spec §14, MUST 6) ---
    public static final int ROLLBACK_MIN_MS = 640;
    public static final int ROLLBACK_MAX_MS = 960;
    public static final int ROLLBACK_DEFAULT_MS = 640;

    // --- Transcript policy (spec §18, OPTIONAL 8) ---
    public static final int TOKEN_STABLE_UPDATES = 2;
    /** EMA smoothing for confidences (spec OPTIONAL 7). */
    public static final float EMA_ALPHA = 0.4f;

    // --- VAD / endpointing (spec §8) ---
    public static final float VAD_SPEECH_THRESHOLD = 0.5f;
    /**
     * Trailing silence that finalizes an utterance. Raised 600 → 1000 →
     * 2000 ms: natural mid-sentence pauses must not split one sentence into
     * several separately translated fragments. A deliberate Stop flushes
     * immediately, so this only delays hands-free endpointing.
     */
    public static final int ENDPOINT_SILENCE_MS = 2000;

    /**
     * Endpoint candidate verification (spec §13.1): replay up to this much
     * recent audio through the non-active models. The text-only router cannot
     * observe other languages (single active decoder, no acoustic LID model),
     * so without this a whole utterance in another language locks to the
     * wrong model forever. Two decodes per utterance — never 3 parallel
     * decoders on the live path (spec §31).
     */
    public static final int ENDPOINT_VERIFY_MS = 6000;
    /**
     * Switch margin applied to the COMBINED endpoint score (confidence +
     * lexical fit), not confidence alone: a cross-lingual decode can be
     * nearly right ("HELLO" via the VI model) with middling confidence, and
     * a pure-confidence margin locks that in forever.
     */
    public static final float ENDPOINT_VERIFY_MARGIN = 0.05f;
    /** Weight of lexical fit inside the combined endpoint score. */
    public static final float ENDPOINT_VERIFY_LEX_W = 0.35f;
    /**
     * Single-token candidates are accepted only at/above this lexical fit
     * (e.g. a lone dictionary word like "HELLO"); otherwise MIN_TOKENS
     * applies. Genuinely ambiguous one-word blips stay put (stability bias).
     */
    public static final float ENDPOINT_VERIFY_STRONG_LEX = 0.45f;
    /** Candidate transcript must carry at least this many text units. */
    public static final int ENDPOINT_VERIFY_MIN_TOKENS = 2;

    // --- Latency targets (spec §19, engineering targets) ---
    public static final int LATENCY_P50_TARGET_MS = 350;
    public static final int LATENCY_P95_TARGET_MS = 500;

    // --- Model asset names (spec §4, §33 + VoxLingua pipeline §12) ---
    public static final String VI_ENCODER = "zipformer_vi_encoder.onnx";
    public static final String VI_DECODER = "zipformer_vi_decoder.onnx";
    public static final String VI_JOINER = "zipformer_vi_joiner.onnx";
    public static final String VI_TOKENS = "zipformer_vi_tokens.txt";
    public static final String EN_ENCODER = "zipformer_en_encoder.onnx";
    public static final String EN_DECODER = "zipformer_en_decoder.onnx";
    public static final String EN_JOINER = "zipformer_en_joiner.onnx";
    public static final String EN_TOKENS = "zipformer_en_tokens.txt";
    public static final String ZH_ENCODER = "zipformer_zh_encoder.int8.onnx";
    // NOTE: upstream's ZH int8 package ships an fp32 decoder (no int8
    // decoder published) — the asset name reflects the real file.
    public static final String ZH_DECODER = "zipformer_zh_decoder.onnx";
    public static final String ZH_JOINER = "zipformer_zh_joiner.int8.onnx";
    public static final String ZH_TOKENS = "zipformer_zh_tokens.txt";
    public static final String VAD_MODEL = "silero_vad.onnx";

    // --- VoxLingua107 ECAPA acoustic LID (VoxLingua pipeline §11–§13) ---
    // Community ONNX export (beginning-ai/...-onnx, ~85 MB on disk,
    // ~110–150 MB working set). FP32 first: validate accuracy → benchmark
    // latency/RAM → then try INT8 (VI↔EN confusion is the critical pair).
    // Frontend: 16 kHz mono, 60-bin FBank (see VoxLinguaFbankExtractor).
    // Do NOT quantize ECAPA before the FP32 parity check passes.
    public static final String VOXLINGUA_MODEL = "voxlingua_lid_ecapa.onnx";
    public static final String VOXLINGUA_LABELS_JSON = "voxlingua_lid_labels.json";
    public static final int VOXLINGUA_N_MELS = 60;
    /** Absolute global-top floor: below this the window is too flat to trust
     *  even when the supported-relative margin passes (107-way uniform ≈
     *  0.009). Starting point for tuning, not a guarantee. */
    public static final float VOXLINGUA_MIN_GLOBAL_SCORE = 0.25f;
}
