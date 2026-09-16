/*
 * OmniVoice — VoxLingua107 ECAPA acoustic LID (bootstrap + runtime referee).
 *
 * Wraps speechbrain/lang-id-voxlingua107-ecapa via the community ONNX export
 * (beginning-ai/speechbrain-lang-id-voxlingua107-ecapa-onnx: model.onnx,
 * ~85 MB). Role per the VoxLingua pipeline doc §4:
 *
 *   ECAPA answers "What language is being spoken?" (acoustic classifier,
 *   107 classes, 16 kHz mono, 60-bin FBank, ECAPA-TDNN).
 *   Zipformer answers "What was said?" (streaming ASR).
 *   Router answers "When should the active ASR model change?"
 *
 * Never a streaming decoder: used as bootstrap LID (600 ms rolling window,
 * 200 ms hop) + periodic runtime referee (500–800 ms stable, 300–400 ms
 * uncertain, 200–300 ms candidate). Speculative ASR runs in parallel so the
 * 600 ms window never hard-gates the first partial (§6, §23).
 *
 * Policy (§14–§15):
 *   - 107-way softmax → global top + VI/EN/ZH absolute probs.
 *   - vi/en/zh fields are RE-normalized among the three for the 0.70/0.15
 *     gate; globalTopScore stays absolute (107-way).
 *   - Global top outside VI/EN/ZH → UNKNOWN (never force Japanese→ZH,
 *     Thai→VI), even when the supported-relative top looks confident.
 *   - Single-window noise is fused by VoxLinguaTemporalSmoother (EMA +
 *     majority vote, §16) before the router sees the scores.
 *
 * Lifecycle: loads the ONNX graph lazily off the caller thread (85 MB —
 * never block the UI/audio callback). Until the session is ready (or when
 * the asset is absent) every call returns LanguageScores.flat() so the
 * pipeline falls through to extend-window / dual-candidate instead of
 * forcing VI. Missing asset is NOT fatal — the heuristic fallback stays.
 */
package com.omnivoice.onspeak47.asr;

import android.content.Context;
import android.util.Log;

import java.nio.FloatBuffer;
import java.util.EnumMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.OrtSessionConfig;

public class VoxLinguaAcousticLidEngine implements AcousticLidEngine {

    private static final String TAG = "VoxLinguaLid";

    /** Minimum PCM for one inference (400 ms @ 16 kHz). Shorter → flat. */
    private static final int MIN_SAMPLES = AsrState.SAMPLE_RATE * 400 / 1000;

    private final VoxLinguaFbankExtractor fbank = new VoxLinguaFbankExtractor();
    private final VoxLinguaTemporalSmoother smoother = new VoxLinguaTemporalSmoother(
            VoxLinguaTemporalSmoother.DEFAULT_DEPTH, AsrState.LID_EMA_ALPHA);
    private final Object lock = new Object();

    private volatile OrtSession session;
    private volatile String featureInputName;
    private volatile String wavLensInputName;
    private volatile boolean loadAttempted = false;
    private volatile boolean ready = false;
    private volatile String loadError;
    /** Successful ECAPA inferences since load (telemetry: proves on-device
     *  the model actually judges instead of flat-fallback). */
    private volatile long inferenceCount = 0;

    /** Test/seam constructor: no model, always flat (never null). */
    public VoxLinguaAcousticLidEngine() {
        this.loadAttempted = true;
        this.loadError = "no-asset (test seam)";
    }

    public VoxLinguaAcousticLidEngine(Context context) {
        try {
            final Context app = context.getApplicationContext();
            // Heavy load off the caller thread: 85 MB graph + warmup.
            new Thread(() -> loadInBackground(app), "voxlingua-load").start();
        } catch (Exception e) {
            loadAttempted = true;
            loadError = e.getMessage();
            Log.w(TAG, "voxlingua init failed, flat fallback: " + e.getMessage());
        }
    }

    /** Returns an engine when the asset is bundled, else null (caller falls back). */
    public static VoxLinguaAcousticLidEngine createIfAvailable(Context context) {
        try {
            context.getAssets().open(AsrState.VOXLINGUA_MODEL).close();
        } catch (Exception e) {
            return null;
        }
        return new VoxLinguaAcousticLidEngine(context);
    }

    public boolean isReady() {
        return ready && session != null;
    }

    public String loadError() {
        return loadError;
    }

    @Override
    public Map<AsrLanguage, Float> classify(float[] audioWindow) {
        LanguageScores s = classifyDetailed(audioWindow);
        if (s == null) s = LanguageScores.flat();
        return s.toMap();
    }

    @Override
    public LanguageScores classifyDetailed(float[] audioWindow) {
        LanguageScores raw = inferRaw(audioWindow);
        synchronized (lock) {
            return smoother.add(raw);
        }
    }

    @Override
    public void reset() {
        synchronized (lock) {
            smoother.reset();
        }
    }

    /**
     * Last SMOOTHED detailed result (temporal-smoother aggregate; never null
     * — flat before the first inference). Satisfies
     * {@link AcousticLidEngine#smoothed()}.
     */
    @Override
    public LanguageScores smoothed() {
        synchronized (lock) {
            return smoother.smoothed();
        }
    }

    public void close() {
        synchronized (lock) {
            smoother.reset();
        }
        if (session != null) {
            try {
                session.close();
            } catch (Exception ignored) {
            }
            session = null;
        }
        ready = false;
    }

    // --- Internals ---------------------------------------------------

    private void loadInBackground(Context app) {
        try {
            String path = FileUtils.copyAssetToInternal(app, AsrState.VOXLINGUA_MODEL);
            if (path == null) {
                loadError = "asset copy returned null";
                loadAttempted = true;
                return;
            }
            OrtSession.SessionOptions options = OrtSessionConfig.create(app, false, false);
            OrtSession s;
            try {
                s = OrtEnvironment.getEnvironment().createSession(path, options);
            } finally {
                try {
                    options.close();
                } catch (Exception ignored) {
                }
            }
            // Resolve inputs by role, not by order: the community export has
            //   inputs = ['features' [1,T,60], 'wav_lens' [1]] and ORT throws
            //   when any input is missing — feeding only 'features' crashes
            //   100% of inferences (flat fallback forever). wav_lens is the
            //   relative length in batch (batch=1 → always 1.0f).
            String feat = null;
            String lens = null;
            for (String n : s.getInputNames()) {
                if (n != null && n.equalsIgnoreCase("wav_lens")) {
                    lens = n;
                } else if (feat == null) {
                    feat = n;
                }
            }
            if (feat == null) feat = s.getInputNames().iterator().next();
            synchronized (lock) {
                session = s;
                featureInputName = feat;
                wavLensInputName = lens;
                ready = true;
                loadAttempted = true;
            }
            Log.i(TAG, "voxlingua session ready: featureInput=" + feat
                    + " wavLensInput=" + lens
                    + " inputs=" + s.getInputNames()
                    + " outputs=" + s.getOutputNames());
        } catch (Exception e) {
            loadError = e.getMessage();
            loadAttempted = true;
            Log.w(TAG, "voxlingua load failed, flat fallback: " + e.getMessage());
        }
    }

    private LanguageScores inferRaw(float[] audio) {
        if (audio == null || audio.length < MIN_SAMPLES) {
            return LanguageScores.flat();
        }
        OrtSession s = session;
        if (s == null || !ready) {
            return LanguageScores.flat();
        }
        float[][] feats;
        try {
            feats = fbank.extract(audio);
        } catch (Exception e) {
            Log.w(TAG, "fbank failed: " + e.getMessage());
            return LanguageScores.flat();
        }
        if (feats.length < 10) {
            return LanguageScores.flat();
        }
        try {
            float[] probs = runSession(s, featureInputName, wavLensInputName, feats);
            if (probs == null || probs.length != VoxLinguaLabels.NUM_LANGUAGES) {
                Log.w(TAG, "unexpected voxlingua output len="
                        + (probs == null ? "null" : probs.length));
                return LanguageScores.flat();
            }
            LanguageScores out = fromLogits(probs);
            long n = ++inferenceCount;
            if (n == 1 || n % 50 == 0) {
                Log.i(TAG, "voxlingua inference #" + n + ": " + out);
            }
            return out;
        } catch (Exception e) {
            Log.w(TAG, "voxlingua inference failed (flat): " + e.getMessage());
            return LanguageScores.flat();
        }
    }

    private static LanguageScores fromLogits(float[] values) {
        if (values == null || values.length == 0) {
            return LanguageScores.flat();
        }

        // Check whether values are already softmax probabilities:
        // The ONNX model export output 'probabilities' already has softmax applied inside the graph.
        // If all values >= -1e-4 and sum is ~1.0, treat them as probabilities directly.
        // Applying softmax a second time squashes high confidence down to ~0.015!
        boolean alreadyProbabilities = false;
        double directSum = 0;
        boolean allNonNegative = true;
        for (float v : values) {
            if (v < -1e-4f) {
                allNonNegative = false;
                break;
            }
            directSum += v;
        }
        if (allNonNegative && Math.abs(directSum - 1.0) < 0.05) {
            alreadyProbabilities = true;
        }

        double[] prob = new double[values.length];
        if (alreadyProbabilities) {
            for (int i = 0; i < values.length; i++) {
                prob[i] = Math.max(0.0, values[i]);
            }
        } else {
            // 107-way softmax for raw logits (e.g. test seams).
            float max = Float.NEGATIVE_INFINITY;
            for (float v : values) if (v > max) max = v;
            double sum = 0;
            for (int i = 0; i < values.length; i++) {
                prob[i] = Math.exp(values[i] - max);
                sum += prob[i];
            }
            if (sum > 0) {
                for (int i = 0; i < prob.length; i++) prob[i] /= sum;
            }
        }

        int topIdx = 0;
        for (int i = 1; i < prob.length; i++) {
            if (prob[i] > prob[topIdx]) topIdx = i;
        }
        String topCode = VoxLinguaLabels.codeAt(topIdx);
        float topScore = (float) prob[topIdx];

        double pEn = prob[VoxLinguaLabels.IDX_EN];
        double pVi = prob[VoxLinguaLabels.IDX_VI];
        double pZh = prob[VoxLinguaLabels.IDX_ZH];
        double sSum = pEn + pVi + pZh;
        float en, vi, zh;
        if (sSum <= 1e-12) {
            en = vi = zh = 1.0f / 3;
        } else {
            // Supported-relative renormalization for the 0.70/0.15 gate;
            // the ABSOLUTE values below are carried alongside so the
            // bootstrap gate can judge the real evidence level
            // (2026-09-16 fix — see AsrState#VOXLINGUA_MIN_SUPPORTED_ABS_SCORE).
            en = (float) (pEn / sSum);
            vi = (float) (pVi / sSum);
            zh = (float) (pZh / sSum);
        }
        return new LanguageScores(vi, en, zh,
                (float) pVi, (float) pEn, (float) pZh,
                topCode, topIdx, topScore, 1);
    }

    private static float[] runSession(OrtSession s, String featName, String lensName,
                                      float[][] feats)
            throws Exception {
        int frames = feats.length;
        int mels = VoxLinguaFbankExtractor.N_MELS;
        float[] flat = new float[frames * mels];
        for (int t = 0; t < frames; t++) {
            System.arraycopy(feats[t], 0, flat, t * mels, mels);
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        // Expected shape [1, T, 60] per the community export README
        // ("SpeechBrain-compatible 60-bin filterbank features").
        // Verified on-device: inputs = ['features' [1,T,60], 'wav_lens' [1]],
        // output 'probabilities' [1,107] (softmax already applied in-graph —
        // fromLogits() must NOT softmax a second time).
        long[] shape = new long[]{1, frames, mels};
        Map<String, OnnxTensor> inputs = new java.util.HashMap<>();
        java.util.List<OnnxTensor> createdTensors = new java.util.ArrayList<>();

        try {
            String featureInput = featName;
            if (featureInput == null) {
                featureInput = "features";
                for (String name : s.getInputNames()) {
                    if (name != null && !"wav_lens".equalsIgnoreCase(name)) {
                        featureInput = name;
                        break;
                    }
                }
            }
            OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape);
            createdTensors.add(in);
            inputs.put(featureInput, in);

            // SpeechBrain ECAPA graph requires 'wav_lens' [1] = 1.0f.
            // Missing it → OrtException on every run → flat fallback forever.
            String lensKey = lensName;
            if (lensKey == null) {
                for (String name : s.getInputNames()) {
                    if (name != null && "wav_lens".equalsIgnoreCase(name)) {
                        lensKey = name;
                        break;
                    }
                }
            }
            if (lensKey != null) {
                OnnxTensor wavLens = OnnxTensor.createTensor(env,
                        FloatBuffer.wrap(new float[]{1.0f}), new long[]{1});
                createdTensors.add(wavLens);
                inputs.put(lensKey, wavLens);
            }

            try (OrtSession.Result r = s.run(inputs)) {
                // Prefer 'probabilities' when present, else the first output.
                String outName = null;
                for (String n : s.getOutputNames()) {
                    if (n != null && "probabilities".equalsIgnoreCase(n)) {
                        outName = n;
                        break;
                    }
                }
                if (outName == null) outName = s.getOutputNames().iterator().next();
                OnnxTensor out = (OnnxTensor) r.get(outName).get();
                FloatBuffer fb = out.getFloatBuffer();
                float[] values = new float[fb.remaining()];
                fb.get(values);
                return values;
            }
        } finally {
            for (OnnxTensor t : createdTensors) {
                try {
                    t.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    /** Test seam: fuse raw logits without a session (softmax + policy). */
    static LanguageScores scoresFromLogitsForTest(float[] logits) {
        return fromLogits(logits);
    }
}
