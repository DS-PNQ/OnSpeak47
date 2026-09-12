/*
 * OmniVoice — Translation Module (Tencent Hy-MT1.5, single GGUF path)
 *
 * ALL translation traffic goes through exactly one backend: the
 * Hy-MT1.5-1.8B 1.25-bit STQ GGUF (440 MB) decoded on the mobile CPU by
 * llama.cpp + the STQ kernel (llama.cpp PR #22836) via HyMtGgufJNI.
 *
 * The former dual-profile design (LOW_RAM GGUF / HIGH_END INT4 ONNX via
 * ONNX Runtime GenAI) is gone: the INT4 path was dead weight in this
 * project — the onnxruntime-genai AAR was never bundled in app/libs, so
 * the code could never load it, while its ~1.25 GB of assets shipped in
 * every APK. RTranslator-side comparison also showed the GGUF path can
 * match its latency once the native build is optimized (see cpp/
 * CMakeLists.txt) and greedy decoding is used (see hymt_gguf_jni.cpp).
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.icu.text.BreakIterator;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Locale;

import com.omnivoice.onspeak47.util.FileUtils;

public class TranslationModule {

    private static final String TAG = "TranslationModule";

    // Single backend asset: 1.25-bit STQ GGUF (tencent/Hy-MT1.5-1.8B-1.25bit-GGUF)
    private static final String GGUF_ASSET = "Hy-MT1.5-1.8B-1.25bit.gguf";
    // Content revision of the GGUF asset ("2" = post typefix: the legacy
    // STQ1_0(42) tensor type codes remapped to 43 by optimize/
    // hymt_gguf_typefix.py so the vendored llama.cpp accepts the header).
    // That fix rewrites tensor-info header bytes IN PLACE and does not change
    // the file size, so an extracted pre-fix copy cannot be detected by
    // length — a device that copied the broken file under an older APK would
    // keep loading it forever. Bump whenever the asset's bytes change again.
    private static final String GGUF_ASSET_REV = "2";

    private static final int MAX_OUTPUT_TOKENS = 128;
    private static final int MAX_OUTPUT_TOKENS_LONG = 256;
    // Single-call threshold: ASR transcripts (~1-2 sentences, <400 chars) go
    // through ONE prefill. Only long paragraphs pay per-sentence prefills.
    // Mirrors RTranslator's join-up-to-maxLength strategy (5000 tok) instead
    // of the old split-every-sentence loop that multiplied prefill cost.
    private static final int SINGLE_CALL_CHAR_THRESHOLD = 400;

    // Obsolete dual-profile extraction (see class comment); reclaimed on boot.
    private static final String DEAD_INT4_DIR = "hymt_int4_onnx";

    private long ggufHandle = 0;

    public TranslationModule(Context context) throws Exception {
        File baseDir = context.getExternalFilesDir(null);
        if (baseDir == null) baseDir = context.getFilesDir();

        // Reclaim the ~1.25 GB INT4 extraction written by the old dual-profile
        // builds on devices that still have it on disk.
        File wasted = new File(baseDir, DEAD_INT4_DIR);
        if (wasted.isDirectory()) {
            Log.i(TAG, "Removing obsolete INT4 model extraction: " + wasted);
            deleteRecursive(wasted);
        }

        if (!assetExists(context, GGUF_ASSET)) {
            Log.e(TAG, "GGUF asset missing from APK: " + GGUF_ASSET
                    + " — translate() will return an error");
            return;
        }
        loadGguf(context, baseDir);
    }

    /**
     * Copies the GGUF asset (revision-gated) and loads it via llama.cpp.
     *
     * Threads: cores-2 (min 2). The 4-thread cap was a big.LITTLE-era
     * heuristic; on the homogeneous flagships this app targets (SM8850 /
     * Snapdragon 8 Elite: 8x Oryon, no little cores) it left 2 fast cores
     * idle and regressed throughput vs the previous cores-2 build. cores-2
     * also reserves headroom for the ASR/TTS/UI work alongside each call.
     */
    private void loadGguf(Context context, File baseDir) {
        String path = copyModel(context, baseDir, GGUF_ASSET);
        int cores = Runtime.getRuntime().availableProcessors();
        int threads = Math.max(2, cores - 2);
        // n_ctx 1024 covers prompt (<200 tok) + gen (<=256) with far less KV
        // memory-bandwidth than 2048; n_batch is set to 512 in JNI.
        ggufHandle = HyMtGgufJNI.loadModel(path, 1024, threads);
        if (ggufHandle == 0) {
            Log.e(TAG, "llama.cpp failed to load GGUF from " + path);
        } else {
            Log.i(TAG, "llama.cpp loaded GGUF successfully (threads=" + threads + ")");
        }
    }

    private static void deleteRecursive(File f) {
        File[] children = f.listFiles();
        if (children != null) {
            for (File c : children) deleteRecursive(c);
        }
        if (!f.delete()) {
            Log.w(TAG, "Could not delete " + f);
        }
    }

    // ----------------------------------------------------------------
    // Public API (unchanged contract)
    // ----------------------------------------------------------------

    public TranslationResult translate(String text, String srcLang, String tgtLang) {
        long startTime = System.currentTimeMillis();
        if (text == null || text.trim().isEmpty()) {
            return new TranslationResult("", 0);
        }
        String cleaned = correctText(text);
        // Fast path (common ASR case): ONE LLM call, ONE prefill.
        if (cleaned.length() <= SINGLE_CALL_CHAR_THRESHOLD) {
            String out = translateSentence(cleaned, srcLang, tgtLang);
            Log.i(TAG, "translate single-call chars=" + cleaned.length()
                    + " (" + (System.currentTimeMillis() - startTime) + "ms)");
            return new TranslationResult(out, System.currentTimeMillis() - startTime);
        }
        // Slow path (long paragraph): per-sentence calls.
        ArrayList<String> sentences = splitIntoSentences(cleaned, srcLang);
        StringBuilder result = new StringBuilder();
        for (String sentence : sentences) {
            if (sentence.trim().isEmpty()) continue;
            String translated = translateSentence(sentence, srcLang, tgtLang);
            if (result.length() > 0) result.append(" ");
            result.append(translated);
        }
        Log.i(TAG, "translate multi-call sentences=" + sentences.size()
                + " (" + (System.currentTimeMillis() - startTime) + "ms)");
        return new TranslationResult(result.toString(), System.currentTimeMillis() - startTime);
    }

    // ----------------------------------------------------------------
    // HY-MT prompting (decoder-only: instruction instead of lang tokens)
    // ----------------------------------------------------------------

    private String translateSentence(String text, String srcLang, String tgtLang) {
        String instruction = buildInstruction(text, srcLang, tgtLang);
        int budget = estimateMaxTokens(text);
        String raw = (ggufHandle != 0)
                ? HyMtGgufJNI.complete(ggufHandle, instruction, budget)
                : "[error: GGUF model not loaded]";
        return clean(raw);
    }

    /** Token budget by input length: short ASR -> 48-128, long -> 256 cap. */
    private static int estimateMaxTokens(String text) {
        int len = text == null ? 0 : text.length();
        if (len <= 60) return 64;
        if (len <= SINGLE_CALL_CHAR_THRESHOLD) return MAX_OUTPUT_TOKENS;
        return MAX_OUTPUT_TOKENS_LONG;
    }

    /** Port of RTranslator correctText(): terminator + whitespace collapse. */
    private static String correctText(String text) {
        String t = text.trim().replaceAll("\\s+", " ");
        if (t.length() >= 2 && Character.isLetterOrDigit(t.charAt(t.length() - 1))) {
            t += ".";
        }
        return t;
    }

    /** ZH<=>XX → Chinese instruction; XX<=>XX → English instruction (Hy-MT README). */
    private static String buildInstruction(String text, String srcLang, String tgtLang) {
        boolean zhInvolved = srcLang.startsWith("zh") || tgtLang.startsWith("zh");
        if (zhInvolved) {
            return "将以下文本翻译为" + zhName(tgtLang) + "，注意只需要输出翻译后的结果，不要额外解释：\n\n" + text;
        }
        return "Translate the following segment into " + enName(tgtLang)
                + ", without additional explanation.\n\n" + text;
    }

    private static String enName(String code) {
        switch (code) {
            case "vi": return "Vietnamese";
            case "zh_hant": return "Traditional Chinese";
            case "zh": case "zh_hans": return "Chinese";
            default: return "English";
        }
    }

    private static String zhName(String code) {
        switch (code) {
            case "vi": return "越南语";
            case "en": return "英语";
            case "zh_hant": return "繁体中文";
            default: return "中文";
        }
    }

    private static String clean(String raw) {
        if (raw == null) return "";
        int cut = raw.indexOf("<｜hy_");
        if (cut >= 0) raw = raw.substring(0, cut);
        raw = raw.replaceAll("</?(think|answer)>", "");
        return raw.trim();
    }

    private ArrayList<String> splitIntoSentences(String text, String langCode) {
        ArrayList<String> sentences = new ArrayList<>();
        Locale locale = langCode != null && langCode.equals("vi") ? new Locale("vi") : Locale.US;
        BreakIterator iterator = BreakIterator.getSentenceInstance(locale);
        iterator.setText(text);
        int start = iterator.first();
        for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
            sentences.add(text.substring(start, end).trim());
        }
        return sentences;
    }

    private boolean assetExists(Context context, String assetPath) {
        try (InputStream is = context.getAssets().open(assetPath)) {
            return true;
        } catch (IOException e) {
            // Check if it's a directory
            try {
                String[] list = context.getAssets().list(assetPath);
                return list != null && list.length > 0;
            } catch (IOException ex) {
                return false;
            }
        }
    }

    private String copyModel(Context context, File baseDir, String assetName) {
        File outFile = new File(baseDir, assetName);
        // Revision gate: the GGUF typefix was size-neutral, so drop any
        // extracted copy that predates the current asset revision and let
        // FileUtils re-copy it from the APK.
        File revFile = new File(baseDir, assetName + ".rev");
        String diskRev = readRevMarker(revFile);
        if (outFile.exists() && !GGUF_ASSET_REV.equals(diskRev)) {
            Log.i(TAG, "Asset revision changed for " + assetName + " (disk '"
                    + diskRev + "' != APK '" + GGUF_ASSET_REV + "') — recopying");
            outFile.delete();
            revFile.delete();
        }
        if (!outFile.exists()) {
            FileUtils.copyAssetToDir(context, assetName, baseDir);
            writeRevMarker(revFile);
        }
        return outFile.getAbsolutePath();
    }

    /** Sidecar "<asset>.rev" content; "" when absent or unreadable. */
    private static String readRevMarker(File revFile) {
        try {
            return new String(Files.readAllBytes(revFile.toPath()), StandardCharsets.UTF_8).trim();
        } catch (IOException e) {
            return "";
        }
    }

    /** Best-effort; failure only costs a redundant recopy on next launch. */
    private static void writeRevMarker(File revFile) {
        try {
            Files.write(revFile.toPath(), GGUF_ASSET_REV.getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            Log.w(TAG, "Could not write revision marker " + revFile, e);
        }
    }

    public void close() {
        if (ggufHandle != 0) {
            HyMtGgufJNI.freeModel(ggufHandle);
            ggufHandle = 0;
        }
    }

    public static class TranslationResult {
        public final String text;
        public final long processingMs;
        public TranslationResult(String text, long processingMs) {
            this.text = text;
            this.processingMs = processingMs;
        }
    }
}
