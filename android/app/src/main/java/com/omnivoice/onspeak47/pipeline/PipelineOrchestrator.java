/*
 * OmniVoice — Pipeline Orchestrator
 *
 * Strings the streaming transcript through Translation → TTS.
 * ASR itself runs in asr/StreamingPipeline (Zipformer VI/EN/ZH, live
 * partials + endpoint finals). Source language is the router's active
 * language — there is no manual language-branching here.
 */

package com.omnivoice.onspeak47.pipeline;

import android.util.Log;

import java.io.File;


/**
 * Chains streaming ASR finals → Translation → TTS with no language branching.
 * Modular design: each stage has clear I/O contracts so a future
 * wearable device can swap hardware backends without redesign.
 */
public class PipelineOrchestrator {

    private static final String TAG = "PipelineOrchestrator";

    private final TranslationModule translator;
    private final TTSModule tts;

    public PipelineOrchestrator(TranslationModule translator, TTSModule tts) {
        this.translator = translator;
        this.tts = tts;
    }

    /**
     * Run Translation → TTS on a streaming FINAL transcript.
     *
     * @param transcript Streaming-committed FINAL text (already endpointed)
     * @param srcLang    Router active language ("vi", "en", "zh")
     * @param tgtLang    Target language
     * @return Pipeline result with transcript, translation, and output audio path,
     *         or null when {@code cancelled} became true between stages.
     */
    public PipelineResult processStreamingFinal(String transcript, String srcLang, String tgtLang,
                                                java.util.function.BooleanSupplier cancelled) {
        return processStreamingFinal(transcript, srcLang, tgtLang, null, cancelled);
    }

    /**
     * Overload with an explicit output directory for the synthesized WAV
     * (the Activity passes getCacheDir(); null falls back to tmp).
     */
    public PipelineResult processStreamingFinal(String transcript, String srcLang, String tgtLang,
                                                File outputDir,
                                                java.util.function.BooleanSupplier cancelled) {
        long totalStart = System.currentTimeMillis();
        logMemory("start");

        String clean = transcript == null ? "" : transcript.trim();
        if (clean.isEmpty()) {
            Log.i(TAG, "Empty streaming transcript — skipping Translation and TTS");
            long totalMs = System.currentTimeMillis() - totalStart;
            return new PipelineResult("", "", null, 0, 0, 0, totalMs, null);
        }

        if (cancelled.getAsBoolean()) return null;

        // --- Stage 1: Translation (Text → Translated Text) ---
        long t0 = System.currentTimeMillis();
        TranslationModule.TranslationResult translationResult =
                translator.translate(clean, srcLang, tgtLang);
        long translationMs = System.currentTimeMillis() - t0;
        Log.i(TAG, "Translation: \"" + translationResult.text + "\" (" + translationMs + "ms)");
        logMemory("after-Translation");

        if (cancelled.getAsBoolean()) return null;

        // --- Stage 2: TTS (Translated Text → Speech) ---
        t0 = System.currentTimeMillis();
        File dir = outputDir != null ? outputDir
                : new File(System.getProperty("java.io.tmpdir", "."));
        String outputPath = new File(dir, "translated_" + srcLang + "_to_" + tgtLang + ".wav")
                .getAbsolutePath();
        String ttsPath = tts.synthesize(translationResult.text, tgtLang, outputPath);
        long ttsMs = System.currentTimeMillis() - t0;
        String ttsError = (ttsPath == null) ? tts.getLastError() : null;
        Log.i(TAG, "TTS: " + (ttsPath != null ? "success" : "failed (" + ttsError + ")")
                + " (" + ttsMs + "ms)");
        logMemory("after-TTS");

        long totalMs = System.currentTimeMillis() - totalStart;
        Log.i(TAG, "Pipeline total: " + totalMs + "ms");

        return new PipelineResult(
                clean,
                translationResult.text,
                ttsPath,
                0, translationMs, ttsMs, totalMs,
                ttsError
        );
    }

    /** Convenience overload without cancellation. */
    public PipelineResult processStreamingFinal(String transcript, String srcLang, String tgtLang) {
        return processStreamingFinal(transcript, srcLang, tgtLang, null, () -> false);
    }

    /**
     * Per-stage resident-memory probe for on-device benchmarking
     * (adb logcat -s PipelineOrchestrator). getTotalPss includes all
     * proportional mappings — the number LMK judges the app by.
     */
    private void logMemory(String stage) {
        try {
            android.os.Debug.MemoryInfo info = new android.os.Debug.MemoryInfo();
            android.os.Debug.getMemoryInfo(info);
            Log.i(TAG, "MEM[" + stage + "] totalPss=" + (info.getTotalPss() / 1024) + "MB");
        } catch (Exception ignored) {
            // MemoryInfo is best-effort instrumentation only.
        }
    }

    /**
     * Translation-only shortcut — skips TTS.
     */
    public PipelineResult translateText(String text, String srcLang, String tgtLang) {
        long t0 = System.currentTimeMillis();
        TranslationModule.TranslationResult result = translator.translate(text, srcLang, tgtLang);
        long elapsed = System.currentTimeMillis() - t0;

        return new PipelineResult(text, result.text, null, 0, elapsed, 0, elapsed, null);
    }

    // ----------------------------------------------------------------
    // Result
    // ----------------------------------------------------------------

    public static class PipelineResult {
        public final String transcript;
        public final String translation;
        public final String audioPath;
        public final long asrMs;
        public final long translationMs;
        public final long ttsMs;
        public final long totalMs;
        /** Reason TTS failed, or null if TTS succeeded. */
        public final String ttsError;

        public PipelineResult(String transcript, String translation, String audioPath,
                              long asrMs, long translationMs, long ttsMs, long totalMs) {
            this(transcript, translation, audioPath, asrMs, translationMs, ttsMs, totalMs, null);
        }

        public PipelineResult(String transcript, String translation, String audioPath,
                              long asrMs, long translationMs, long ttsMs, long totalMs,
                              String ttsError) {
            this.transcript = transcript;
            this.translation = translation;
            this.audioPath = audioPath;
            this.asrMs = asrMs;
            this.translationMs = translationMs;
            this.ttsMs = ttsMs;
            this.totalMs = totalMs;
            this.ttsError = ttsError;
        }
    }
}
