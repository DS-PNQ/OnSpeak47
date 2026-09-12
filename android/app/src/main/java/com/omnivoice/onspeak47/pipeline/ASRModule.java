/*
 * OmniVoice — ASR Module (Whisper Small via ONNX Runtime)
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.util.JsonReader;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.OrtSessionConfig;
import com.omnivoice.onspeak47.util.TensorUtils;


/**
 * Whisper Small ASR module using ONNX Runtime.
 */
public class ASRModule {

    private static final String TAG = "ASRModule";

    private static final String ENCODER_FILE = "whisper_encoder.onnx";
    private static final String DECODER_FILE = "whisper_decoder.onnx";
    private static final String PREPROCESS_FILE = "whisper_preprocess.onnx";
    private static final int SAMPLE_RATE = 16000;
    private static final int N_MELS = 80;
    private static final int N_FFT = 512;        
    private static final int WINDOW_SIZE = 400;  
    private static final int HOP_LENGTH = 160;
    private static final int MAX_AUDIO_FRAMES = 3000;
    private static final int MAX_TOKENS = 448;

    // Anti-hallucination: a greedy decoder that locks onto an n-gram never
    // emits EOT, so it repeats the phrase until maxTokens (the "No fucking
    // vapor." x12 bug). detectTailLoop() breaks the decode after
    // MIN_LOOP_REPEATS consecutive repeats of any 1..MAX_LOOP_NGRAM token
    // pattern and collapseTailLoop() keeps a single instance. Loops longer
    // than MAX_LOOP_NGRAM tokens fall through to the post-decode
    // compression-ratio check.
    private static final int MAX_LOOP_NGRAM = 12;
    private static final int MIN_LOOP_REPEATS = 3;
    private static final double COMPRESSION_RATIO_LIMIT = 2.4;

    private final OrtEnvironment env;
    private OrtSession encoderSession;
    private OrtSession decoderSession;
    /** Native PCM->log-mel graph (ort-extensions); null -> Java DSP fallback. */
    private OrtSession preprocessSession;
    /** True when the encoder graph accepts a dynamic frame axis (the
     *  optimize/10 export) — enables short-window mel feeding. */
    private boolean encoderDynamicFrames;

    // Whisper special token IDs
    private static final int SOT = 50258;        
    private static final int EOT = 50257;        
    private static final int TRANSCRIBE = 50359; 
    private static final int NO_TIMESTAMPS = 50363;

    // Language tokens (verified against whisper_vocab.json)
    private static final Map<String, Integer> LANGUAGE_TOKENS = new HashMap<>();
    static {
        LANGUAGE_TOKENS.put("vi", 50278);  // <|vi|> — was incorrectly 50264 (<|ko|>)
        LANGUAGE_TOKENS.put("en", 50259);  // <|en|>
        LANGUAGE_TOKENS.put("zh", 50260);  // <|zh|>
    }

    private static final String WHISPER_VOCAB_FILE = "whisper_vocab.json";
    private Map<Integer, String> vocabMap;

    // ------------------------------------------------------------------
    // GPT-2 / Whisper byte-level BPE alphabet (bytes_to_unicode()).
    //
    // Whisper's vocab.json does NOT contain literal UTF-8 text. Each raw
    // byte (0-255) is remapped to one printable Unicode "placeholder"
    // character so that byte-level BPE merges can operate on any input
    // byte using ordinary text tooling. e.g. the Vietnamese/Chinese bytes
    // of a multi-byte UTF-8 character each get their own placeholder char
    // (space -> 'Ġ', and CJK bytes -> characters like 'ãĢĤ' for "。").
    //
    // Decoding therefore requires: for every character in a decoded token
    // string, map it back to its original byte via this reverse table,
    // concatenate ALL bytes across the whole output, and only then decode
    // the byte sequence as UTF-8 — never per-token, since a single UTF-8
    // character's bytes can be split across adjacent tokens.
    // ------------------------------------------------------------------
    private static final Map<Integer, Character> BYTE_TO_UNICODE = new HashMap<>();
    private static final Map<Character, Integer> UNICODE_TO_BYTE = new HashMap<>();
    static {
        boolean[] inBaseSet = new boolean[256];
        ArrayList<Integer> bytesList = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) { bytesList.add(b); inBaseSet[b] = true; }
        for (int b = 0xA1; b <= 0xAC; b++) { bytesList.add(b); inBaseSet[b] = true; }
        for (int b = 0xAE; b <= 0xFF; b++) { bytesList.add(b); inBaseSet[b] = true; }
        ArrayList<Integer> codepointsList = new ArrayList<>(bytesList);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!inBaseSet[b]) {
                bytesList.add(b);
                codepointsList.add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bytesList.size(); i++) {
            char c = (char) (int) codepointsList.get(i);
            BYTE_TO_UNICODE.put(bytesList.get(i), c);
            UNICODE_TO_BYTE.put(c, bytesList.get(i));
        }
    }

    public ASRModule(Context context) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        encoderSession = createSessionPreferPreopt(context, ENCODER_FILE);
        decoderSession = createSessionPreferPreopt(context, DECODER_FILE);
        encoderDynamicFrames = detectDynamicFrames(encoderSession);

        // Optional native pre-processing graph: raw PCM -> log-mel via
        // ort-extensions custom ops (same approach as RTranslator's
        // Whisper initializer). Missing/broken asset -> Java DSP fallback.
        String preprocessPath = copyModelToInternalOptional(context, PREPROCESS_FILE);
        if (preprocessPath != null) {
            try {
                OrtSession.SessionOptions options = OrtSessionConfig.create(context, true, false);
                try {
                    preprocessSession = env.createSession(preprocessPath, options);
                    Log.i(TAG, "Whisper ONNX preprocessing enabled");
                } finally {
                    options.close();
                }
            } catch (OrtException e) {
                preprocessSession = null;
                Log.w(TAG, "Failed to load whisper_preprocess.onnx — Java DSP fallback: " + e.getMessage());
            }
        } else {
            Log.i(TAG, "whisper_preprocess.onnx not bundled — using Java DSP fallback");
        }

        loadVocabulary(context);
        Log.i(TAG, "ASR module initialized (Whisper Small, encoder="
                + (encoderDynamicFrames ? "dynamic-length" : "fixed 3000") + ")");
    }

    /**
     * Creates a session preferring the offline pre-optimized sibling
     * ({@code <name>.opt.onnx} from optimize/07_preoptimize.py, loaded with
     * NO_OPT) and falling back to the base asset (ALL_OPT).
     */
    private OrtSession createSessionPreferPreopt(Context context, String assetName) throws OrtException {
        String optAsset = assetName.replace(".onnx", ".opt.onnx");
        String optPath = copyModelToInternalOptional(context, optAsset);
        if (optPath != null) {
            OrtSession.SessionOptions options = OrtSessionConfig.create(context, true, true);
            try {
                Log.i(TAG, "Loading pre-optimized " + optAsset);
                return env.createSession(optPath, options);
            } finally {
                options.close();
            }
        }
        String basePath = copyModelToInternal(context, assetName);
        OrtSession.SessionOptions options = OrtSessionConfig.create(context, true, false);
        try {
            return env.createSession(basePath, options);
        } finally {
            options.close();
        }
    }

    public ASRResult transcribe(String audioPath, String language) {
        long startTime = System.currentTimeMillis();
        try {
            File audioFile = new File(audioPath);
            if (!audioFile.exists() || audioFile.length() <= 44) {
                Log.w(TAG, "Audio file is missing or empty: " + audioPath);
                return new ASRResult("[No audio recorded]", language != null ? language : "auto", 0);
            }

            float[] audioSamples = readWavPCM(audioPath);

            // RTranslator's guard: Whisper generates at most ~30 tokens per
            // second of audio; anything beyond that is an untranscribable
            // loop. Capping by real audio length avoids minutes of wasted
            // decode on short clips.
            int audioSeconds = audioSamples != null ? audioSamples.length / SAMPLE_RATE : 0;
            int maxTokens = Math.max(16, Math.min(audioSeconds * 30, MAX_TOKENS));

            // Prefer the native ONNX preprocessing graph; fall back to the
            // Java DSP path when the asset or the run is unavailable.
            OrtSession.Result preprocessResult = runPreprocess(audioSamples);

            OrtSession.Result encoderResult = null;
            String text;
            try {
                OnnxTensor melTensor;
                boolean ownMel;
                if (preprocessResult != null) {
                    melTensor = (OnnxTensor) preprocessResult.get(0);
                    ownMel = false;
                } else {
                    float[][] mel = extractMelFeatures(audioSamples);
                    melTensor = OnnxTensor.createTensor(env,
                            FloatBuffer.wrap(TensorUtils.flatten2D(mel)),
                            new long[]{1, mel.length, mel[0].length});
                    ownMel = true;
                }
                try {
                    // Dynamic-length encoder: feed only the audio's real
                    // frames — a 3 s clip skips ~90% of the 30 s window's
                    // compute (the skipped columns are constant padding).
                    OnnxTensor fed = melTensor;
                    OnnxTensor sliced = encoderDynamicFrames && audioSamples != null
                            ? sliceShortWindow(melTensor, audioSamples.length)
                            : null;
                    if (sliced != null) fed = sliced;
                    try {
                        encoderResult = runEncoder(fed);
                    } finally {
                        if (sliced != null) sliced.close();
                    }
                } finally {
                    if (ownMel) melTensor.close();
                }
                text = normalizeTranscript(runDecoder(encoderResult, language, maxTokens));
                if (isSuspectedHallucination(text)) {
                    Log.w(TAG, "Compression ratio > " + COMPRESSION_RATIO_LIMIT
                            + " — suspected repetition hallucination, discarding: " + text);
                    text = "";
                }
            } finally {
                if (encoderResult != null) encoderResult.close();
                if (preprocessResult != null) preprocessResult.close();
            }

            long elapsed = System.currentTimeMillis() - startTime;
            Log.i(TAG, "Transcription done in " + elapsed + "ms: " + text);
            return new ASRResult(text, language != null ? language : "auto", elapsed);
        } catch (Exception e) {
            Log.e(TAG, "Transcription failed", e);
            return new ASRResult("", "error", 0);
        }
    }

    /**
     * Runs the exported whisper_preprocess.onnx graph (raw 16 kHz PCM ->
     * normalized log-mel) natively via the ort-extensions custom ops, the
     * same approach as RTranslator's Whisper initializer. The graph pads to
     * the fixed 3000-frame window internally, so the PCM is passed at its
     * REAL length — no 30 s pre-padding.
     *
     * @return the open Result owning the `input_features` tensor, or null to
     *         fall back to the Java DSP path (no session / bad input / error).
     */
    private OrtSession.Result runPreprocess(float[] audioSamples) {
        if (preprocessSession == null || audioSamples == null || audioSamples.length == 0) return null;
        try {
            String inputName = preprocessSession.getInputNames().iterator().next();
            // The declared input rank decides between (N) and (1, N) — the
            // exact contract varies across onnxruntime-extensions versions.
            ai.onnxruntime.TensorInfo inputInfo =
                    (ai.onnxruntime.TensorInfo) preprocessSession.getInputInfo()
                            .get(inputName).getInfo();
            long[] declared = inputInfo.getShape();
            long[] shape = declared.length == 1
                    ? new long[]{audioSamples.length}
                    : new long[]{1, audioSamples.length};
            OnnxTensor audioTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(audioSamples), shape);
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, audioTensor);
            try {
                return preprocessSession.run(inputs);
            } finally {
                audioTensor.close();
            }
        } catch (Exception e) {
            Log.w(TAG, "ONNX preprocessing failed, using Java DSP fallback: " + e.getMessage());
            return null;
        }
    }

    /**
     * Runs the encoder and returns the OPEN Result that owns the output
     * tensor. Callers must keep it alive while decoding (the decoder consumes
     * Result-owned tensors directly) and close it when done — no Java-side
     * clone of the encoder output.
     */
    private OrtSession.Result runEncoder(OnnxTensor inputFeatures) throws OrtException {
        if (encoderSession == null) return null;
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_features", inputFeatures);
        return encoderSession.run(inputs);
    }

    /** True when the encoder's input_features frame axis is symbolic (dynamic
     *  dims read back as -1 from TensorInfo). */
    private static boolean detectDynamicFrames(OrtSession session) {
        try {
            ai.onnxruntime.TensorInfo info = (ai.onnxruntime.TensorInfo)
                    session.getInputInfo().get("input_features").getInfo();
            long[] shape = info.getShape();
            return shape.length >= 3 && shape[2] < 0;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Copies the first {@code frames} mel columns of the padded (1,80,3000)
     * mel tensor into a new (1,80,frames) tensor for the dynamic-length
     * encoder (optimize/10 export) — short clips pay only their real length
     * of encoder compute. The skipped columns are the constant clamp-padding
     * tail, so no real signal is dropped. Returns null when the tensor is
     * already short enough (nothing to slice).
     */
    private OnnxTensor sliceShortWindow(OnnxTensor mel, int audioSamplesLength) throws OrtException {
        int frames = Math.min(MAX_AUDIO_FRAMES,
                Math.max(8, audioSamplesLength / HOP_LENGTH));
        long[] shape = mel.getInfo().getShape();
        if (shape.length != 3 || shape[2] <= frames) return null;
        int nMels = (int) shape[1];
        int srcLen = (int) shape[2];
        FloatBuffer src = mel.getFloatBuffer();
        float[] out = new float[nMels * frames];
        for (int m = 0; m < nMels; m++) {
            src.position(m * srcLen);
            src.get(out, m * frames, frames);
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(out),
                new long[]{1, nMels, frames});
    }

    /**
     * Assembles one whole-sequence decode step's input map: input_ids,
     * encoder_hidden_states, use_cache_branch=false and empty past tensors —
     * a merged graph requires every declared input on each run ("Missing
     * Input" otherwise).
     */
    private void addDecoderStepInputs(Map<String, OnnxTensor> inputs, ArrayList<OnnxTensor> created,
                                      long[] tokenIds, OnnxTensor encoderOutput,
                                      boolean hasCacheBranch, ArrayList<String> pastNames) throws OrtException {
        OnnxTensor ids = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(tokenIds),
                new long[]{1, tokenIds.length});
        inputs.put("input_ids", ids);
        created.add(ids);
        inputs.put("encoder_hidden_states", encoderOutput);
        if (hasCacheBranch) {
            OnnxTensor flag = TensorUtils.booleanToTensor(env, false);
            inputs.put("use_cache_branch", flag);
            created.add(flag);
        }
        for (String n : pastNames) {
            // Whisper-small: 12 heads x 64 head_dim (batch=1, seq=0).
            OnnxTensor t = TensorUtils.createFloatTensor(env, new long[]{1, 12, 0, 64});
            created.add(t);
            inputs.put(n, t);
        }
    }

    /** Strays that leak through even with no-timestamps forced: timestamp
     *  tokens of the form <|...|> and "..." runs; then capitalizes the first
     *  letter (same normalization RTranslator applies to Whisper output). */
    private static String normalizeTranscript(String text) {
        if (text == null) return "";
        String out = text.replaceAll("<\\|[^>]*\\|> ?", "").trim();
        if (out.length() >= 2) {
            char first = out.charAt(0);
            if (Character.isLowerCase(first)) {
                out = Character.toUpperCase(first) + out.substring(1);
            }
            out = out.replace("...", "");
        }
        return out.trim();
    }

    /**
     * Longest n (1..MAX_LOOP_NGRAM) whose exact token pattern occupies the
     * tail of {@code tokens} at least MIN_LOOP_REPEATS times in a row, or 0
     * when the tail is not a loop. Longest-first matters: a phrase-level
     * loop ("No fucking vapor." x3, n=5) must not be misread as its
     * degenerate n=1 sub-pattern, so the collapse keeps the whole phrase.
     * Called after every appended token, so O(n^2) per step over at most
     * 12x2 comparisons — negligible next to a decoder step.
     */
    private static int detectTailLoop(ArrayList<Integer> tokens) {
        int len = tokens.size();
        for (int n = Math.min(MAX_LOOP_NGRAM, len / MIN_LOOP_REPEATS); n >= 1; n--) {
            int base = len - n * MIN_LOOP_REPEATS;
            boolean loop = true;
            for (int r = 1; r < MIN_LOOP_REPEATS && loop; r++) {
                for (int i = 0; i < n; i++) {
                    if (!tokens.get(base + i).equals(tokens.get(base + r * n + i))) {
                        loop = false;
                        break;
                    }
                }
            }
            if (loop) return n;
        }
        return 0;
    }

    /**
     * Trims a detected tail loop down to a single instance of its n-gram.
     * The decode was already stopped by the caller's detectTailLoop() break;
     * this only removes the redundant trailing copies before text assembly.
     */
    private static void collapseTailLoop(ArrayList<Integer> tokens) {
        int n = detectTailLoop(tokens);
        if (n > 0) {
            int keep = tokens.size() - n * (MIN_LOOP_REPEATS - 1);
            tokens.subList(keep, tokens.size()).clear();
        }
    }

    /**
     * OpenAI Whisper's compression-ratio heuristic: text dominated by a
     * repeated pattern gzips far better than natural speech, so
     * raw-bytes/gzipped-bytes above ~2.4 indicates a repetition
     * hallucination. Catches loops LONGER than MAX_LOOP_NGRAM tokens, which
     * the decode-time breaker cannot see. Short strings are skipped — the
     * gzip header overhead dominates and would inflate the ratio.
     */
    private static boolean isSuspectedHallucination(String text) {
        if (text == null || text.length() < 24) return false;
        try {
            byte[] raw = text.getBytes("UTF-8");
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            try (java.util.zip.GZIPOutputStream gz = new java.util.zip.GZIPOutputStream(bos)) {
                gz.write(raw);
            }
            double compressed = bos.size();
            return compressed > 0 && (raw.length / compressed) > COMPRESSION_RATIO_LIMIT;
        } catch (java.io.IOException e) {
            return false;  // gzip only fails on stream errors — never gate on this
        }
    }

    private String runDecoder(OrtSession.Result encoderResult, String language,
                              int maxTokens) throws OrtException {
        if (decoderSession == null || encoderResult == null) return "";
        // Owned by encoderResult — valid until the caller closes it.
        OnnxTensor encoderOutput = (OnnxTensor) encoderResult.get(0);

        int langToken = language != null && LANGUAGE_TOKENS.containsKey(language)
                ? LANGUAGE_TOKENS.get(language)
                : LANGUAGE_TOKENS.get("vi");

        ArrayList<Integer> outputTokens = new ArrayList<>();
        long[] initialTokens = new long[]{SOT, langToken, TRANSCRIBE, NO_TIMESTAMPS};

        // Detect the decoder graph's cache contract.
        boolean hasCacheBranch = false;
        ArrayList<String> decoderPastNames = new ArrayList<>();
        ArrayList<String> encoderPastNames = new ArrayList<>();
        for (String inputName : decoderSession.getInputNames()) {
            if (inputName.equals("use_cache_branch")) hasCacheBranch = true;
            else if (inputName.startsWith("past_key_values")) {
                if (inputName.contains(".encoder.")) encoderPastNames.add(inputName);
                else decoderPastNames.add(inputName);
            }
        }

        // Fast path: KV-cached decode for graphs exported by
        // optimize/09_export_whisper_decoder_kv.py (per-step cost is constant
        // instead of O(n^2)). If this runtime rejects the cached steps for any
        // reason, fall back to the whole-sequence path below instead of
        // failing the transcription.
        if (!decoderPastNames.isEmpty()) {
            try {
                decodeWithCache(encoderOutput, initialTokens, outputTokens,
                        hasCacheBranch, decoderPastNames, encoderPastNames, maxTokens);
                collapseTailLoop(outputTokens);
                return decodeTokens(outputTokens);
            } catch (Exception e) {
                Log.e(TAG, "KV-cache decode failed — falling back to whole-sequence "
                        + "decode (slower). Cause: " + e.getMessage(), e);
                outputTokens.clear();
            }
        }

        // Cache-less graph (the pre-09 asset) or cached-path failure:
        // whole-sequence decode. The 09 graph has no use_cache_branch, so its
        // else-equivalent is just "empty pasts + full sequence".
        ArrayList<String> pastNames = new ArrayList<>(decoderPastNames);
        pastNames.addAll(encoderPastNames);
        decodeNoCache(encoderOutput, initialTokens, outputTokens,
                hasCacheBranch, pastNames, maxTokens);
        collapseTailLoop(outputTokens);
        return decodeTokens(outputTokens);
    }

    /**
     * KV-cached greedy decode for the 09-export decoder graph:
     *
     * <ul>
     *   <li>Prefill: feed [SOT, lang, TRANSCRIBE, NO_TIMESTAMPS] with the full
     *       encoder output and empty past tensors. The graph returns the
     *       encoder cross-attention K/V (present.*.encoder.*) ONCE — this
     *       prefill Result must stay open for the whole decode and its encoder
     *       presents are re-fed on every step (RTranslator's
     *       cache-initializer pattern, same as TranslationModule).</li>
     *   <li>Steps: feed exactly ONE token with an EMPTY encoder_hidden_states
     *       (1&times;0&times;768) — the graph takes cross K/V from the encoder
     *       pasts. Decoder self-attention pasts cycle between steps zero-copy:
     *       each step's present.*.decoder.* is cumulative (past+new), so the
     *       previous Result's tensors are passed straight back in and the
     *       previous Result is closed only AFTER the next run consumed them.</li>
     * </ul>
     */
    private void decodeWithCache(OnnxTensor encoderOutput, long[] initialTokens,
                                 ArrayList<Integer> outputTokens,
                                 boolean hasCacheBranch,
                                 ArrayList<String> decoderPastNames,
                                 ArrayList<String> encoderPastNames,
                                 int maxTokens) throws OrtException {
        OrtSession.Result prefillResult = null;
        OrtSession.Result result = null;
        try {
            // ---- Prefill: initial prefix, full encoder output, empty pasts. ----
            Map<String, OnnxTensor> inputs = new HashMap<>();
            ArrayList<OnnxTensor> created = new ArrayList<>();
            OnnxTensor ids = OnnxTensor.createTensor(env,
                    java.nio.LongBuffer.wrap(initialTokens), new long[]{1, initialTokens.length});
            inputs.put("input_ids", ids);
            created.add(ids);
            inputs.put("encoder_hidden_states", encoderOutput);
            for (String n : decoderPastNames) {
                OnnxTensor t = TensorUtils.createFloatTensor(env, new long[]{1, 12, 0, 64});
                created.add(t);
                inputs.put(n, t);
            }
            for (String n : encoderPastNames) {
                OnnxTensor t = TensorUtils.createFloatTensor(env, new long[]{1, 12, 0, 64});
                created.add(t);
                inputs.put(n, t);
            }

            while (true) {
                OrtSession.Result newResult;
                try {
                    newResult = decoderSession.run(inputs);
                } finally {
                    for (OnnxTensor t : created) t.close();
                    created.clear();
                }
                // Free the previous step's Result only now — its decoder
                // `present` tensors were inputs to the run that just
                // completed. The prefill Result stays open until the end.
                if (result != null && result != prefillResult) result.close();
                result = newResult;
                if (prefillResult == null) prefillResult = newResult;

                int next = argmaxLastRow((OnnxTensor) result.get(0));
                if (next == EOT) break;               // done; EOT not appended
                outputTokens.add(next);
                if (detectTailLoop(outputTokens) > 0) break;  // repetition loop
                if (outputTokens.size() >= maxTokens) break;

                // ---- Next step: one token, empty encoder states, cycled pasts. ----
                inputs = new HashMap<>();
                long[] one = new long[]{next};
                OnnxTensor stepIds = OnnxTensor.createTensor(env,
                        java.nio.LongBuffer.wrap(one), new long[]{1, 1});
                inputs.put("input_ids", stepIds);
                created.add(stepIds);
                OnnxTensor emptyEhs = TensorUtils.createFloatTensor(env, new long[]{1, 0, 768});
                inputs.put("encoder_hidden_states", emptyEhs);
                created.add(emptyEhs);
                for (String n : decoderPastNames) {
                    inputs.put(n, presentTensor(result, n));
                }
                for (String n : encoderPastNames) {
                    inputs.put(n, presentTensor(prefillResult, n));
                }
            }
        } finally {
            if (result != null) result.close();
            if (prefillResult != null && prefillResult != result) prefillResult.close();
        }
    }

    /** Maps a `past_key_values.*` input name to the matching `present.*`
     *  output tensor of {@code source} (zero-copy). */
    private static OnnxTensor presentTensor(OrtSession.Result source, String pastName)
            throws OrtException {
        String presentName = pastName.replaceFirst("^past_key_values", "present");
        java.util.Optional<OnnxValue> v = source.get(presentName);
        if (!v.isPresent()) {
            throw new OrtException("decoder did not return '" + presentName + "'");
        }
        return (OnnxTensor) v.get();
    }

    /**
     * Argmax over the LAST time-step's logits row, reading the tensor's
     * FloatBuffer directly. Materializing the full [seq &#215; 51865] Java
     * array via getValue() allocates ~6 MB per step at seq 30 — this reads
     * only the final vocab-sized row.
     */
    private static int argmaxLastRow(OnnxTensor logits) throws OrtException {
        long[] shape = logits.getInfo().getShape();
        if (shape.length != 3 || shape[1] < 1) return 0;
        int vocab = (int) shape[2];
        int start = (int) ((shape[1] - 1) * vocab);
        FloatBuffer fb = logits.getFloatBuffer();
        int best = 0;
        float bestVal = -Float.MAX_VALUE;
        for (int i = 0; i < vocab; i++) {
            float v = fb.get(start + i);
            if (v > bestVal) {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }

    /**
     * Whole-sequence greedy decode: re-feeds the accumulated tokens with
     * empty past tensors each step. Used for cache-less graphs (the pre-09
     * asset) or when the cached fast path fails at runtime — correct on any
     * contract, but O(n^2). Builds inputs via addDecoderStepInputs so the
     * FULL input contract is fed.
     */
    private void decodeNoCache(OnnxTensor encoderOutput, long[] initialTokens,
                               ArrayList<Integer> outputTokens,
                               boolean hasCacheBranch, ArrayList<String> pastNames,
                               int maxTokens) throws OrtException {
        long[] currentTokens = initialTokens.clone();
        for (int i = 0; i < maxTokens; i++) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            ArrayList<OnnxTensor> created = new ArrayList<>();
            addDecoderStepInputs(inputs, created, currentTokens, encoderOutput,
                    hasCacheBranch, pastNames);
            try (OrtSession.Result result = decoderSession.run(inputs)) {
                int nextToken = argmaxLastRow((OnnxTensor) result.get(0));
                if (nextToken == EOT) break;
                outputTokens.add(nextToken);
                if (detectTailLoop(outputTokens) > 0) break;  // repetition loop
                long[] nextTokens = new long[currentTokens.length + 1];
                System.arraycopy(currentTokens, 0, nextTokens, 0, currentTokens.length);
                nextTokens[currentTokens.length] = nextToken;
                currentTokens = nextTokens;
            } finally {
                for (OnnxTensor t : created) t.close();
            }
        }
    }

    private void loadVocabulary(Context context) {
        vocabMap = new HashMap<>();
        try {
            String vocabPath = copyModelToInternal(context, WHISPER_VOCAB_FILE);
            File vocabFile = new File(vocabPath);
            if (vocabFile.exists()) {
                try (JsonReader reader = new JsonReader(new InputStreamReader(new FileInputStream(vocabFile), "UTF-8"))) {
                    reader.beginObject();
                    while (reader.hasNext()) {
                        String key = reader.nextName();
                        int id = reader.nextInt();
                        // Store the raw byte-level-BPE token string as-is;
                        // decodeTokens() reverses the byte mapping and
                        // re-decodes as UTF-8 (see BYTE_TO_UNICODE above).
                        vocabMap.put(id, key);
                    }
                    reader.endObject();
                }
                Log.i(TAG, "Loaded Whisper vocabulary: " + vocabMap.size() + " tokens");
            } else {
                Log.w(TAG, "Whisper vocab file not found");
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to load Whisper vocabulary", e);
        }
    }

    private String decodeTokens(ArrayList<Integer> tokenIds) {
        if (tokenIds.isEmpty()) return "";
        if (vocabMap == null || vocabMap.isEmpty()) return "[No vocab]";

        // Reassemble the raw byte sequence first: every character in every
        // token is a byte-level-BPE placeholder that maps back to exactly
        // one byte via UNICODE_TO_BYTE. A single UTF-8 character's bytes
        // may be split across adjacent tokens, so we must concatenate all
        // bytes and decode as UTF-8 only at the very end.
        byte[] bytes = new byte[tokenIds.size() * 16];
        int len = 0;
        for (int id : tokenIds) {
            if (id >= 50257) continue;  // skip special tokens
            String piece = vocabMap.get(id);
            if (piece == null) continue;
            for (int i = 0; i < piece.length(); i++) {
                Integer b = UNICODE_TO_BYTE.get(piece.charAt(i));
                if (b == null) continue;  // unexpected character, skip
                if (len == bytes.length) {
                    byte[] grown = new byte[bytes.length * 2];
                    System.arraycopy(bytes, 0, grown, 0, len);
                    bytes = grown;
                }
                bytes[len++] = (byte) (int) b;
            }
        }

        try {
            return new String(bytes, 0, len, "UTF-8").trim();
        } catch (java.io.UnsupportedEncodingException e) {
            return "";
        }
    }

    /**
     * Java DSP fallback when whisper_preprocess.onnx is unavailable. The
     * log-mel is computed for the audio's REAL length only; the mel matrix is
     * padded to the fixed 3000-frame window afterwards (the normalization
     * loop below fills beyond numFrames with the clamp floor). The old code
     * padded the PCM to 30 s FIRST, making a 3 s clip pay for all 3000 FFT
     * frames (~10x the necessary DSP work).
     */
    private float[][] extractMelFeatures(float[] audioSamples) {
        if (audioSamples == null || audioSamples.length == 0) return new float[N_MELS][MAX_AUDIO_FRAMES];

        float maxAbs = 0;
        for (float s : audioSamples) if (Math.abs(s) > maxAbs) maxAbs = Math.abs(s);
        Log.i(TAG, "Max audio level: " + maxAbs);

        int numFrames = audioSamples.length / HOP_LENGTH;
        if (numFrames > MAX_AUDIO_FRAMES) numFrames = MAX_AUDIO_FRAMES;

        float[][] stftMag = new float[N_FFT / 2 + 1][numFrames];
        float[] window = hannWindow(WINDOW_SIZE);

        for (int frame = 0; frame < numFrames; frame++) {
            int offset = frame * HOP_LENGTH;
            float[] real = new float[N_FFT];
            float[] imag = new float[N_FFT];
            for (int j = 0; j < WINDOW_SIZE && (offset + j) < audioSamples.length; j++) {
                real[j] = audioSamples[offset + j] * window[j];
            }
            fft(real, imag, N_FFT);
            for (int k = 0; k <= N_FFT / 2; k++) stftMag[k][frame] = real[k] * real[k] + imag[k] * imag[k];
        }

        float[][] melFilters = melFilterbank(SAMPLE_RATE, N_FFT, N_MELS);
        float[][] melSpec = new float[N_MELS][MAX_AUDIO_FRAMES];

        for (int m = 0; m < N_MELS; m++) {
            for (int frame = 0; frame < numFrames; frame++) {
                float sum = 0;
                for (int k = 0; k <= N_FFT / 2; k++) sum += melFilters[m][k] * stftMag[k][frame];
                melSpec[m][frame] = (float) Math.log10(Math.max(sum, 1e-10));
            }
        }

        float maxVal = -10.0f;
        for (int m = 0; m < N_MELS; m++) {
            for (int f = 0; f < numFrames; f++) if (melSpec[m][f] > maxVal) maxVal = melSpec[m][f];
        }

        // Whisper normalization: log-mel is clamped to [max-8, max], then
        // normalized as (val + 4.0) / 4.0  (the +4 accounts for the typical
        // log-mel floor, matching the Whisper processor's fixed offset).
        float clampMin = maxVal - 8.0f;
        for (int m = 0; m < N_MELS; m++) {
            for (int f = 0; f < MAX_AUDIO_FRAMES; f++) {
                float val = (f < numFrames) ? melSpec[m][f] : clampMin;
                val = Math.max(val, clampMin);
                melSpec[m][f] = (val + 4.0f) / 4.0f;
            }
        }
        return melSpec;
    }

    private float[] readWavPCM(String wavPath) {
        try (FileInputStream fis = new FileInputStream(wavPath)) {
            byte[] header = new byte[44];
            if (fis.read(header) < 44) return null;
            int sampleRate = ByteBuffer.wrap(header, 24, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            int bitsPerSample = ByteBuffer.wrap(header, 34, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
            int numChannels = ByteBuffer.wrap(header, 22, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
            int dataSize = (int) (new File(wavPath).length() - 44);
            if (dataSize <= 0) return null;

            byte[] audioBytes = new byte[dataSize];
            int totalRead = 0;
            while (totalRead < dataSize) {
                int n = fis.read(audioBytes, totalRead, dataSize - totalRead);
                if (n < 0) break;
                totalRead += n;
            }
            if (totalRead <= 0) return null;

            int bytesPerSample = bitsPerSample / 8;
            int numSamples = totalRead / (bytesPerSample * numChannels);
            float[] samples = new float[numSamples];
            ByteBuffer buffer = ByteBuffer.wrap(audioBytes, 0, totalRead).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < numSamples; i++) {
                if (bitsPerSample == 16) {
                    samples[i] = buffer.getShort() / 32768.0f;
                    for (int c = 1; c < numChannels; c++) if (buffer.hasRemaining()) buffer.getShort();
                }
            }
            return samples;
        } catch (IOException e) { return null; }
    }

    private float[] hannWindow(int size) {
        float[] window = new float[size];
        for (int i = 0; i < size; i++) window[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / size));
        return window;
    }

    private void fft(float[] real, float[] imag, int n) {
        if (n <= 1) return;
        int logN = Integer.numberOfTrailingZeros(Integer.highestOneBit(n));
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - logN);
            if (j > i) {
                float tempR = real[i]; real[i] = real[j]; real[j] = tempR;
                float tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
            }
        }
        for (int size = 2; size <= n; size *= 2) {
            int halfSize = size / 2;
            float angle = (float) (-2.0 * Math.PI / size);
            float wR = (float) Math.cos(angle);
            float wI = (float) Math.sin(angle);
            for (int i = 0; i < n; i += size) {
                float curR = 1.0f, curI = 0.0f;
                for (int j = 0; j < halfSize; j++) {
                    int even = i + j;
                    int odd = i + j + halfSize;
                    float tR = curR * real[odd] - curI * imag[odd];
                    float tI = curR * imag[odd] + curI * real[odd];
                    real[odd] = real[even] - tR;
                    imag[odd] = imag[even] - tI;
                    real[even] += tR;
                    imag[even] += tI;
                    float newR = curR * wR - curI * wI;
                    curI = curR * wI + curI * wR;
                    curR = newR;
                }
            }
        }
    }

    private float[][] melFilterbank(int sampleRate, int nFft, int nMels) {
        int numBins = nFft / 2 + 1;
        float[][] filters = new float[nMels][numBins];
        float melMin = hzToMel(0.0f), melMax = hzToMel(sampleRate / 2.0f);
        float[] melPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
        float[] binFreqs = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) binFreqs[i] = melToHz(melPoints[i]) * (nFft + 1) / sampleRate;
        for (int m = 0; m < nMels; m++) {
            float left = binFreqs[m], center = binFreqs[m + 1], right = binFreqs[m + 2];
            for (int k = 0; k < numBins; k++) {
                if (k >= left && k <= center && center != left) filters[m][k] = (k - left) / (center - left);
                else if (k > center && k <= right && right != center) filters[m][k] = (right - k) / (right - center);
            }
            float enorm = 2.0f / (melToHz(melPoints[m + 2]) - melToHz(melPoints[m]));
            for (int k = 0; k < numBins; k++) filters[m][k] *= enorm;
        }
        return filters;
    }

    private float hzToMel(float hz) { return 2595.0f * (float) Math.log10(1.0 + hz / 700.0); }
    private float melToHz(float mel) { return 700.0f * ((float) Math.pow(10.0, mel / 2595.0) - 1.0f); }

    private String copyModelToInternal(Context context, String assetName) {
        File outFile = new File(context.getFilesDir(), assetName);
        // Always delegate to FileUtils: it validates an existing extraction
        // against the APK asset's size and re-copies on mismatch. The old
        // `if (!outFile.exists())` guard short-circuited that check, so a
        // stale extraction from an older APK (e.g. the 352 MB FP32 encoder
        // after the asset switched back to the 98 MB INT8 one) kept being
        // loaded forever even after the fixed APK was installed.
        FileUtils.copyAssetToInternal(context, assetName);
        return outFile.getAbsolutePath();
    }

    /** Like {@link #copyModelToInternal}, but returns null when the asset is
     *  not bundled in the APK (used for the optional preprocess graph). */
    private String copyModelToInternalOptional(Context context, String assetName) {
        try {
            context.getAssets().open(assetName).close();
        } catch (IOException e) {
            return null;
        }
        return copyModelToInternal(context, assetName);
    }

    public static class ASRResult {
        public final String text;
        public final String language;
        public final long processingMs;
        public ASRResult(String text, String language, long processingMs) {
            this.text = text;
            this.language = language;
            this.processingMs = processingMs;
        }
    }
}
