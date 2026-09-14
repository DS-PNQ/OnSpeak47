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

    // --- LID scheduling (spec §11) ---
    public static final int LID_INTERVAL_STABLE_MS = 600;
    public static final int LID_INTERVAL_UNCERTAIN_MS = 400;
    public static final int LID_WINDOW_MS = 480;

    // --- Router scoring (spec §12) ---
    public static final float W_ACOUSTIC = 0.45f;
    public static final float W_TEXT = 0.30f;
    public static final float W_CONFIDENCE = 0.15f;
    public static final float W_HISTORY = 0.10f;

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

    // --- Model asset names (spec §4, §33) ---
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
}
