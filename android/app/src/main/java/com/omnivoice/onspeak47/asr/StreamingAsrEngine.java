/*
 * OmniVoice — Streaming Zipformer engine abstraction (production ASR path).
 *
 * Spec §21 MUST 1/2: AudioRecord → streaming frames → OnlineRecognizer →
 * partial result (no full-utterance WAV staging).
 *
 * Requires the sherpa-onnx AAR (OnlineRecognizer/OnlineStream), vendored at
 * android/app/libs/sherpa-onnx-1.13.4.aar (gitignored — fetch from
 * https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-1.13.4.aar).
 * Version pinned: v1.13.4 bundles ORT 1.27.0, matching onnxruntime-android.
 * plus one encoder/decoder/joiner/tokens.txt per language
 * (spec §4: VI streaming 30M, EN streaming Zipformer, ZH streaming INT8).
 * FakeEngine remains for JVM unit tests and asset-less builds only.
 */
package com.omnivoice.onspeak47.asr;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

public abstract class StreamingAsrEngine {

    private static final String TAG = "StreamingAsrEngine";

    public static class PartialResult {
        public final String text;
        public final float confidence;
        public final boolean endpoint;

        public PartialResult(String text, float confidence, boolean endpoint) {
            this.text = text == null ? "" : text;
            this.confidence = confidence;
            this.endpoint = endpoint;
        }
    }

    /** Feed PCM floats; decode happens incrementally (spec §9). */
    public abstract void acceptAudio(float[] samples);

    public abstract void decodeAvailable();

    public abstract boolean isReadyToDecode();

    public abstract PartialResult getPartialResult();

    public abstract PartialResult getFinalResult();

    public abstract float getConfidence();

    public abstract AsrLanguage language();

    public abstract void reset();

    public abstract void close();

    public abstract boolean isFake();

    // --- Factory -----------------------------------------------------

    public interface Factory {
        StreamingAsrEngine create(AsrLanguage lang) throws Exception;
    }

    /**
     * Create the best available engine: sherpa-onnx OnlineRecognizer when the
     * AAR + model assets are present, otherwise a FakeEngine placeholder for
     * tests/asset-less builds (production builds must bundle both).
     */
    public static StreamingAsrEngine create(Context context, AsrLanguage lang,
                                            ZipformerModelManager.Assets assets) {
        StreamingAsrEngine impl = tryCreateSherpa(context, lang, assets);
        if (impl != null) return impl;
        Log.e(TAG, "sherpa-onnx unavailable for " + lang + " — FakeEngine placeholder. "
                + "Fetch models via optimize/11_fetch_streaming_zipformer.py and ensure the "
                + "sherpa-onnx AAR is bundled (see class javadoc).");
        return new FakeEngine(lang);
    }

    private static StreamingAsrEngine tryCreateSherpa(Context context, AsrLanguage lang,
                                                      ZipformerModelManager.Assets assets) {
        try {
            Class.forName("com.k2fsa.sherpa.onnx.OnlineRecognizerConfig");
            Class.forName("com.k2fsa.sherpa.onnx.OnlineModelConfig");
            Class.forName("com.k2fsa.sherpa.onnx.FeatureConfig");
            // Touch the classes; full config wiring lives in SherpaEngine below
            // and is only invoked when all assets exist on disk.
            if (assets == null || !assets.allExist()) {
                Log.w(TAG, "sherpa assets missing for " + lang + ": "
                        + (assets == null ? "null" : assets.missing()));
                return null;
            }
            return new SherpaEngine(context, lang, assets);
        } catch (ClassNotFoundException e) {
            Log.w(TAG, "sherpa-onnx AAR classes not found: " + e.getMessage());
            return null; // AAR not on the classpath.
        } catch (Exception e) {
            Log.w(TAG, "sherpa engine init failed for " + lang + ": " + e.getMessage());
            return null;
        }
    }

    // --- Fake engine (tests / no-asset placeholder) -------------------

    /** Deterministic in-memory engine for unit tests and asset-less builds. */
    public static class FakeEngine extends StreamingAsrEngine {
        /** Applied on decodeAvailable(); models re-decode after reset(). */
        public interface DecodeHook {
            PartialResult decode(long fedSamples);
        }

        private final AsrLanguage lang;
        private final StringBuilder fed = new StringBuilder();
        private long fedSamples = 0;
        private float conf = 0.5f;
        private DecodeHook decodeHook;

        public FakeEngine(AsrLanguage lang) {
            this.lang = lang;
        }

        /** Test hook: pretend the model decoded this text. */
        public void injectPartial(String text, float confidence) {
            fed.setLength(0);
            fed.append(text == null ? "" : text);
            conf = confidence;
        }

        /** Test hook: what decodeAvailable() decodes (survives reset()). */
        public void setDecodeHook(DecodeHook hook) {
            this.decodeHook = hook;
        }

        @Override public void acceptAudio(float[] samples) {
            if (samples != null) fedSamples += samples.length;
        }

        @Override public void decodeAvailable() {
            if (decodeHook != null) {
                try {
                    PartialResult r = decodeHook.decode(fedSamples);
                    if (r != null && !r.text.isEmpty()) {
                        fed.setLength(0);
                        fed.append(r.text);
                        conf = r.confidence;
                    }
                } catch (Exception ignored) {
                }
            }
        }

        @Override public boolean isReadyToDecode() {
            return fedSamples >= AsrState.SCHEDULER_SAMPLES;
        }

        @Override public PartialResult getPartialResult() {
            return new PartialResult(fed.toString(), conf, false);
        }

        @Override public PartialResult getFinalResult() {
            return new PartialResult(fed.toString(), conf, true);
        }

        @Override public float getConfidence() {
            return conf;
        }

        @Override public AsrLanguage language() {
            return lang;
        }

        @Override public void reset() {
            fed.setLength(0);
            fedSamples = 0;
        }

        @Override public void close() { }

        @Override public boolean isFake() {
            return true;
        }
    }

    // --- sherpa-onnx reflective wrapper --------------------------------

    /**
     * Thin reflective wrapper around
     * com.k2fsa.sherpa.onnx.OnlineRecognizer. Reflection keeps the compile
     * classpath free of the AAR until the team vendors it; once present this
     * path performs true incremental Zipformer decoding with state.
     *
     * Built against the Kotlin-style Java API in the vendored AAR
     * (verified against v1.13.4 classes.jar): config classes have public
     * no-arg constructors plus setters —
     * FeatureConfig.setSampleRate/setFeatureDim,
     * OnlineTransducerModelConfig.setEncoder/setDecoder/setJoiner,
     * OnlineModelConfig.setTransducer/setTokens/setNumThreads/setModelType,
     * OnlineRecognizerConfig.setFeatConfig/setModelConfig/setDecodingMethod —
     * and results come from recognizer.getResult(stream).getText().
     * Construction needs new OnlineRecognizer(assetManager, config) and
     * recognizer.createStream("") (hotwords string, empty = none).
     * Setter calls fail fast with the available-method list in the message
     * when the API drifts, instead of silently misconfiguring the model.
     */
    private static class SherpaEngine extends StreamingAsrEngine {
        private final AsrLanguage lang;
        private Object recognizer;
        private Object stream;
        private Method acceptWaveform;
        private Method decode;
        private Method isReady;
        private Method getResultText;
        private Method resetStream;
        private float lastConf = 0.5f;

        SherpaEngine(Context context, AsrLanguage lang,
                     ZipformerModelManager.Assets assets) throws Exception {
            this.lang = lang;
            ClassLoader cl = context.getClassLoader();
            Class<?> featCls = Class.forName("com.k2fsa.sherpa.onnx.FeatureConfig", true, cl);
            Class<?> transducerCls = Class.forName(
                    "com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig", true, cl);
            Class<?> modelCls = Class.forName(
                    "com.k2fsa.sherpa.onnx.OnlineModelConfig", true, cl);
            Class<?> recognizerCfg = Class.forName(
                    "com.k2fsa.sherpa.onnx.OnlineRecognizerConfig", true, cl);
            Class<?> recognizerCls = Class.forName(
                    "com.k2fsa.sherpa.onnx.OnlineRecognizer", true, cl);
            Class<?> streamCls = Class.forName(
                    "com.k2fsa.sherpa.onnx.OnlineStream", true, cl);

            Object feat = featCls.getDeclaredConstructor().newInstance();
            callSetter(feat, "setSampleRate", int.class, AsrState.SAMPLE_RATE);
            callSetter(feat, "setFeatureDim", int.class, 80);

            Object trans = transducerCls.getDeclaredConstructor().newInstance();
            callSetter(trans, "setEncoder", String.class, assets.encoderPath);
            callSetter(trans, "setDecoder", String.class, assets.decoderPath);
            callSetter(trans, "setJoiner", String.class, assets.joinerPath);

            Object model = modelCls.getDeclaredConstructor().newInstance();
            callSetter(model, "setTransducer", transducerCls, trans);
            callSetter(model, "setTokens", String.class, assets.tokensPath);
            callSetter(model, "setNumThreads", int.class, 2);
            callSetter(model, "setModelType", String.class,
                    AsrState.transducerModelType(lang));

            Object cfg = recognizerCfg.getDeclaredConstructor().newInstance();
            callSetter(cfg, "setFeatConfig", featCls, feat);
            callSetter(cfg, "setModelConfig", modelCls, model);
            callSetter(cfg, "setDecodingMethod", String.class, "greedy_search");
            // Disable sherpa's internal endpoint detector: endpointing is
            // owned by our Silero VadEngine, so the internal detector is
            // redundant work on every decode. Decoding itself is unaffected.
            try {
                callSetter(cfg, "setEnableEndpoint", boolean.class, false);
            } catch (Exception e) {
                Log.w(TAG, "setEnableEndpoint missing, internal endpoint stays on: "
                        + e.getMessage());
            }

            Constructor<?> ctor = recognizerCls.getDeclaredConstructor(
                    AssetManager.class, recognizerCfg);
            // Model paths are absolute files under getFilesDir() (copied from
            // APK assets via FileUtils). sherpa-onnx native ReadFile() aborts
            // the whole process (SIGABRT: "Read binary file ... failed") when
            // assetManager != null for an absolute path — see
            // https://github.com/k2-fsa/sherpa-onnx/issues/2562. Pass null so
            // the native side uses fopen() on the filesystem path.
            // Pre-validate first: a missing/truncated file would otherwise
            // still hit the native LOG(FATAL) which Java cannot catch.
            validateModelFiles(assets);
            recognizer = ctor.newInstance((AssetManager) null, cfg);
            Method createStream = recognizerCls.getMethod("createStream", String.class);
            stream = createStream.invoke(recognizer, "");

            acceptWaveform = streamCls.getMethod("acceptWaveform", float[].class, int.class);
            decode = recognizerCls.getMethod("decode", streamCls);
            isReady = recognizerCls.getMethod("isReady", streamCls);
            resetStream = recognizerCls.getMethod("reset", streamCls);
            // Result accessor varies by version; resolve lazily in getPartialResult.
            try {
                getResultText = recognizerCls.getMethod("getResult", streamCls);
            } catch (NoSuchMethodException e) {
                getResultText = null;
            }
            Log.i(TAG, "sherpa-onnx OnlineRecognizer ready for " + lang);
        }

        @Override public void acceptAudio(float[] samples) {
            try {
                acceptWaveform.invoke(stream, samples, AsrState.SAMPLE_RATE);
            } catch (Exception e) {
                Log.w(TAG, "acceptWaveform failed", e);
            }
        }

        @Override public void decodeAvailable() {
            try {
                decode.invoke(recognizer, stream);
            } catch (Exception e) {
                Log.w(TAG, "decode failed", e);
            }
        }

        @Override public boolean isReadyToDecode() {
            try {
                return (boolean) isReady.invoke(recognizer, stream);
            } catch (Exception e) {
                return true;
            }
        }

        @Override public PartialResult getPartialResult() {
            return readResult(false);
        }

        @Override public PartialResult getFinalResult() {
            return readResult(true);
        }

        private PartialResult readResult(boolean endpoint) {
            try {
                if (getResultText != null) {
                    Object r = getResultText.invoke(recognizer, stream);
                    String text = resultText(r);
                    // Token confidence from OnlineRecognizerResult.getYsProbs.
                    // The array may hold probabilities or log-probabilities;
                    // map both into [0,1] so the mean stays order-preserving.
                    // The transducer has no per-language confidence head; this
                    // average is what the router and endpoint verification
                    // compare.
                    if (!text.isEmpty()) {
                        lastConf = ysMean(r, lastConf);
                        logYsStats(r);
                    }
                    return new PartialResult(text, lastConf, endpoint);
                }
            } catch (Exception e) {
                Log.w(TAG, "getResult failed", e);
            }
            return new PartialResult("", lastConf, endpoint);
        }

        /**
         * Mean token score mapped into [0,1]: values in (0,1] average
         * directly, values <= 0 are treated as log-probabilities and
         * exponentiated, anything else (empty/missing/NaN/logits) falls back.
         */
        private static float ysMean(Object result, float fallback) {
            try {
                Object o = result.getClass().getMethod("getYsProbs").invoke(result);
                if (o instanceof float[]) {
                    float[] ys = (float[]) o;
                    double sum = 0;
                    int n = 0;
                    for (float v : ys) {
                        if (Float.isNaN(v) || Float.isInfinite(v) || v > 1) {
                            if (v > 1) return fallback; // logit domain — mean is meaningless
                            continue;
                        }
                        sum += v <= 0 ? Math.exp(v) : v;
                        n++;
                    }
                    if (n > 0) return (float) Math.max(0, Math.min(1, sum / n));
                }
            } catch (Exception ignored) {
            }
            return fallback;
        }

        /** One-line ysProbs shape/domain probe for device troubleshooting. */
        private static void logYsStats(Object result) {
            try {
                Object o = result.getClass().getMethod("getYsProbs").invoke(result);
                if (!(o instanceof float[])) {
                    Log.d(TAG, "ysProbs not float[]");
                    return;
                }
                float[] ys = (float[]) o;
                if (ys.length == 0) {
                    Log.d(TAG, "ysProbs empty");
                    return;
                }
                float min = ys[0];
                float max = ys[0];
                for (float v : ys) {
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
                Log.d(TAG, "ysProbs n=" + ys.length + " min=" + min + " max=" + max);
            } catch (Exception e) {
                Log.d(TAG, "ysProbs unavailable: " + e.getMessage());
            }
        }
        /** Extracts transcript text (OnlineRecognizerResult.getText()). */
        private static String resultText(Object result) {
            if (result == null) return "";
            try {
                Object text = result.getClass().getMethod("getText").invoke(result);
                return text == null ? "" : String.valueOf(text);
            } catch (Exception e) {
                Log.w(TAG, "getText missing on recognizer result, using toString()");
                return String.valueOf(result);
            }
        }

        @Override public float getConfidence() {
            return lastConf;
        }

        @Override public AsrLanguage language() {
            return lang;
        }

        @Override public void reset() {
            try {
                resetStream.invoke(recognizer, stream);
            } catch (Exception e) {
                Log.w(TAG, "reset failed", e);
            }
        }

        @Override public void close() {
            for (Object o : new Object[]{stream, recognizer}) {
                if (o == null) continue;
                try {
                    o.getClass().getMethod("release").invoke(o);
                } catch (Exception ignored) {
                }
            }
            stream = null;
            recognizer = null;
        }

        @Override public boolean isFake() {
            return false;
        }

        /** Fail fast in Java when a model file is missing/empty.
         *  The sherpa-onnx native loader calls LOG(FATAL) (process abort,
         *  uncatchable from Java) on a bad path, so this check is what keeps
         *  a corrupt install on the FakeEngine fallback instead of crashing. */
        private static void validateModelFiles(
                ZipformerModelManager.Assets assets) throws Exception {
            if (assets == null) throw new IllegalStateException("null ASR assets");
            String missing = assets.missing();
            if (!missing.isEmpty()) {
                throw new IllegalStateException("missing ASR model files: " + missing);
            }
        }

        /** Invokes a setter by reflection. */
        private static void callSetter(Object target, String method,
                                       Class<?> argType, Object arg) throws Exception {
            try {
                target.getClass().getMethod(method, argType).invoke(target, arg);
            } catch (NoSuchMethodException e) {
                throw new NoSuchMethodException(
                        target.getClass().getSimpleName() + "#" + method
                                + "(" + argType.getSimpleName() + ") missing; "
                                + "available: " + methodNames(target));
            }
        }

        private static String methodNames(Object target) {
            Class<?> c = target instanceof Class ? (Class<?>) target : target.getClass();
            StringBuilder sb = new StringBuilder();
            for (Method m : c.getMethods()) {
                if (sb.length() > 0) sb.append(',');
                sb.append(m.getName());
            }
            return sb.toString();
        }
    }
}
