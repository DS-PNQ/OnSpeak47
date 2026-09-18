/*
 * OmniVoice — Streaming VAD (Silero VAD via ONNX Runtime, energy fallback).
 *
 * Spec §8: runs on the audio stream, emits speech_start/continue/end; used
 * for endpointing + silence suppression, NOT as the language-switch signal.
 * When silero_vad.onnx is not bundled (or fails to load) an RMS energy gate
 * with hangover keeps the pipeline functional.
 */
package com.omnivoice.onspeak47.asr;

import android.content.Context;
import android.util.Log;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.OrtSessionConfig;

public class VadEngine {

    private static final String TAG = "VadEngine";

    /** 32 ms of contiguous 16 kHz audio per native Silero inference. */
    public static final int WINDOW_SAMPLES = 512;

    /** Energy fallback: RMS above this counts as speech (~-40 dBFS). */
    private static final float ENERGY_SPEECH_RMS = 0.01f;
    private static final float ENERGY_SILENCE_RMS = 0.006f;

    public static class VadResult {
        public final boolean isSpeech;
        public final float prob;
        public final boolean speechStart;
        public final boolean speechEnd;

        VadResult(boolean isSpeech, float prob, boolean speechStart, boolean speechEnd) {
            this.isSpeech = isSpeech;
            this.prob = prob;
            this.speechStart = speechStart;
            this.speechEnd = speechEnd;
        }
    }

    private OrtSession session;
    private String inputName;
    private boolean useNative = false;

    private boolean inSpeech = false;
    private long silenceSinceMs = -1;
    private float lastProb = 0;
    private final VadWindowBuffer windows = new VadWindowBuffer(WINDOW_SAMPLES);
    private final java.util.function.ToDoubleFunction<float[]> testScorer;

    /** Native-window seam: exercises the production framing and endpoint state machine. */
    VadEngine(java.util.function.ToDoubleFunction<float[]> scorer) {
        testScorer = scorer;
    }

    // Silero VAD is stateful (h/c); kept across frames when native is active.
    private float[] vadH;
    private float[] vadC;
    private static final int VAD_STATE_DIM = 128;

    public VadEngine() {
        testScorer = null;
        // Energy fallback only (unit-test / no-asset path).
    }

    public VadEngine(Context context) {
        testScorer = null;
        try {
            String path = copyOptional(context, AsrState.VAD_MODEL);
            if (path != null) {
                OrtSession.SessionOptions options =
                        OrtSessionConfig.create(context, false, false);
                try {
                    session = ai.onnxruntime.OrtEnvironment.getEnvironment()
                            .createSession(path, options);
                } finally {
                    options.close();
                }
                inputName = session.getInputNames().iterator().next();
                vadH = new float[VAD_STATE_DIM];
                vadC = new float[VAD_STATE_DIM];
                useNative = true;
                Log.i(TAG, "Silero VAD native session loaded");
            } else {
                Log.i(TAG, "silero_vad.onnx not bundled — energy fallback");
            }
        } catch (Exception e) {
            Log.w(TAG, "VAD init failed, energy fallback: " + e.getMessage());
            closeSession();
            useNative = false;
        }
    }

    /** Process one ~20 ms frame; call at the AudioRecord frame rate. */
    public VadResult process(float[] frame) {
        return process(frame, System.currentTimeMillis());
    }

    private long processedSamples;

    VadResult process(float[] frame, long nowMs) {
        // Endpoint time follows captured samples, not worker stalls/wall-clock jumps.
        processedSamples += frame.length;
        if (!useNative && testScorer == null) {
            return updateGate(energyProb(frame), processedSamples * 1000 / AsrState.SAMPLE_RATE);
        }
        final boolean[] events = new boolean[2];
        windows.append(frame, (window, endSample) -> {
            float prob = testScorer == null ? nativeProb(window)
                    : (float) testScorer.applyAsDouble(window);
            VadResult result = updateGate(prob, endSample * 1000 / AsrState.SAMPLE_RATE);
            events[0] |= result.speechStart;
            events[1] |= result.speechEnd;
        });
        return new VadResult(inSpeech, lastProb, events[0], events[1]);
    }

    private VadResult updateGate(float prob, long nowMs) {
        lastProb = prob;
        boolean speech = prob >= AsrState.VAD_SPEECH_THRESHOLD;
        boolean start = false;
        boolean end = false;
        if (speech) {
            if (!inSpeech) {
                inSpeech = true;
                start = true;
            }
            silenceSinceMs = -1;
        } else if (inSpeech) {
            if (silenceSinceMs < 0) silenceSinceMs = nowMs;
            if (nowMs - silenceSinceMs >= AsrState.ENDPOINT_SILENCE_MS) {
                inSpeech = false;
                end = true;
                silenceSinceMs = -1;
            }
        }
        return new VadResult(inSpeech || speech, prob, start, end);
    }

    public boolean inSpeech() {
        return inSpeech;
    }

    public float lastProb() {
        return lastProb;
    }

    public boolean isNative() {
        return useNative;
    }

    public void reset() {
        inSpeech = false;
        silenceSinceMs = -1;
        lastProb = 0;
        processedSamples = 0;
        windows.reset();
        if (vadH != null) java.util.Arrays.fill(vadH, 0);
        if (vadC != null) java.util.Arrays.fill(vadC, 0);
    }

    public void close() {
        closeSession();
    }

    // --- Internals ---------------------------------------------------

    private float energyProb(float[] frame) {
        if (frame == null || frame.length == 0) return 0;
        double sum = 0;
        for (float s : frame) sum += s * s;
        float rms = (float) Math.sqrt(sum / frame.length);
        // Hysteresis band to avoid chatter at the boundary.
        float thr = inSpeech ? ENERGY_SILENCE_RMS : ENERGY_SPEECH_RMS;
        if (rms >= thr) return Math.min(1.0f, 0.55f + rms * 20.0f);
        return Math.max(0.0f, rms * 20.0f);
    }

    private float nativeProb(float[] frame) {
        try {
            float[] window = frame; // Already framed by the lossless 512-sample FIFO.
            Map<String, OnnxTensor> inputs = new HashMap<>();
            ai.onnxruntime.OrtEnvironment env = ai.onnxruntime.OrtEnvironment.getEnvironment();
            OnnxTensor audio = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(window), new long[]{1, window.length});
            inputs.put(inputName, audio);
            // Feed recurrent state when the graph declares it.
            java.util.List<OnnxTensor> created = new java.util.ArrayList<>();
            created.add(audio);
            try {
                for (String n : session.getInputNames()) {
                    if (n.equals(inputName)) continue;
                    OnnxTensor t;
                    if (n.contains("h")) {
                        t = OnnxTensor.createTensor(env, FloatBuffer.wrap(vadH),
                                new long[]{2, 1, VAD_STATE_DIM / 2});
                    } else if (n.contains("c")) {
                        t = OnnxTensor.createTensor(env, FloatBuffer.wrap(vadC),
                                new long[]{2, 1, VAD_STATE_DIM / 2});
                    } else {
                        continue;
                    }
                    inputs.put(n, t);
                    created.add(t);
                }
                try (OrtSession.Result r = session.run(inputs)) {
                    OnnxTensor out = (OnnxTensor) r.get(0);
                    FloatBuffer fb = out.getFloatBuffer();
                    float p = fb.capacity() > 0 ? fb.get(0) : 0;
                    carryState(r);
                    return Math.max(0, Math.min(1, p));
                }
            } finally {
                for (OnnxTensor t : created) t.close();
            }
        } catch (Exception e) {
            Log.w(TAG, "VAD inference failed, one-shot energy fallback: " + e.getMessage());
            return energyProb(frame);
        }
    }

    /**
     * Carry the recurrent state forward: the model returns updated
     * {@code new_h}/{@code new_c} that must seed the next frame. Without
     * this every frame runs on zeroed state and speech probabilities
     * collapse toward silence.
     */
    private void carryState(OrtSession.Result r) {
        copyState(r, "new_h", vadH);
        copyState(r, "new_c", vadC);
    }

    private static void copyState(OrtSession.Result r, String name, float[] dst) {
        if (dst == null) return;
        try {
            java.util.Optional<ai.onnxruntime.OnnxValue> v = r.get(name);
            if (v.isPresent() && v.get() instanceof OnnxTensor) {
                FloatBuffer fb = ((OnnxTensor) v.get()).getFloatBuffer();
                int n = Math.min(dst.length, fb.capacity());
                for (int i = 0; i < n; i++) dst[i] = fb.get(i);
            }
        } catch (Exception ignored) {
            // Export without recurrent outputs — stateless scoring still works.
        }
    }

    private static String copyOptional(Context context, String asset) {
        try {
            context.getAssets().open(asset).close();
        } catch (Exception e) {
            return null;
        }
        try {
            return FileUtils.copyAssetToInternal(context, asset);
        } catch (Exception e) {
            Log.w(TAG, "VAD asset copy failed: " + e.getMessage());
            return null;
        }
    }

    private void closeSession() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException ignored) {
            }
            session = null;
        }
        useNative = false;
    }
}
