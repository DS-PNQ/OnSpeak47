/*
 * OmniVoice — VoxLingua 60-bin FBank frontend (dedicated, not Zipformer's).
 *
 * SpeechBrain hyperparams.yaml for lang-id-voxlingua107-ecapa:
 *   compute_features = Fbank(n_mels=60, left_frames=0, right_frames=0,
 *                            deltas=false)
 *   mean_var_norm = InputNormalization(norm_type=sentence, std_norm=false)
 * i.e. 16 kHz mono → 60 log-mel bins → per-utterance MEAN subtraction
 * (no variance norm). The community ONNX export
 * (beginning-ai/...-onnx, model.onnx) expects these 60-bin features as
 * input — NOT raw PCM.
 *
 * SpeechBrain Fbank defaults (must re-validate parity against the Python
 * reference before grading accuracy — see optimize/12_fetch_voxlingua_lid.py
 * and backend/streaming_asr/voxlingua.py):
 *   sample_rate 16000, win_length 25 ms (400 samples), hop 10 ms
 *   (160 samples), n_fft 400 (implemented here as 512-point FFT with the
 *   frame zero-padded — magnitude bins 0..200 map identically for a 400-pt
 *   window), Hann-equivalent Hamming window, 60 HTK mel triangles
 *   spanning 0..8000 Hz, log(max(mel, 1e-10)), sentence-mean subtraction.
 *
 * If frontend parity drifts, LID accuracy collapses even with a perfect
 * ONNX graph — so this class is intentionally self-contained and unit
 * tested (frame count, bin count, determinism, mean-norm ~0) and every
 * constant stays public for the parity script to import/compare.
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

public final class VoxLinguaFbankExtractor {

    /** Must match SpeechBrain training/export reference. */
    public static final int SAMPLE_RATE = 16000;
    public static final int N_MELS = 60;
    /** 25 ms analysis window. */
    public static final int WIN_LENGTH = 400;
    /** 10 ms shift. */
    public static final int HOP_LENGTH = 160;
    /** FFT size (SpeechBrain default is 400). */
    public static final int N_FFT = 400;
    public static final int N_STFT = N_FFT / 2 + 1; // 201
    /** Mel range (Hz). */
    public static final float F_MIN = 0f;
    public static final float F_MAX = 8000f;
    /** Log floor (SpeechBrain/torchaudio default). */
    public static final float LOG_FLOOR = 1e-10f;
    public static final float TOP_DB = 80.0f;
    /** Pre-emphasis is NOT applied by SpeechBrain Fbank — keep 0.0. */
    public static final float PREEMPH = 0.0f;

    private final float[][] melFilters; // [N_MELS][N_STFT]
    private final float[] window;       // [WIN_LENGTH]
    private final float[][] cosTable;   // [N_STFT][WIN_LENGTH]
    private final float[][] sinTable;   // [N_STFT][WIN_LENGTH]

    public VoxLinguaFbankExtractor() {
        this.melFilters = buildMelFilters(N_MELS, N_FFT, SAMPLE_RATE, F_MIN, F_MAX);
        this.window = hamming(WIN_LENGTH);
        this.cosTable = new float[N_STFT][WIN_LENGTH];
        this.sinTable = new float[N_STFT][WIN_LENGTH];
        for (int k = 0; k < N_STFT; k++) {
            for (int n = 0; n < WIN_LENGTH; n++) {
                double ang = -2.0 * Math.PI * k * n / N_FFT;
                cosTable[k][n] = (float) Math.cos(ang);
                sinTable[k][n] = (float) Math.sin(ang);
            }
        }
    }

    /**
     * @param pcm16kMono mono PCM in [-1,1] @ 16 kHz.
     * @return [numFrames][60] log-mel features, sentence-mean normalized
     *         (mean subtraction only, no std — matches InputNormalization
     *         sentence/std_norm=false). Empty array when too short.
     */
    public float[][] extract(float[] pcm16kMono) {
        if (pcm16kMono == null || pcm16kMono.length < WIN_LENGTH) {
            return new float[0][];
        }

        // SpeechBrain center=True reflection padding (n_fft / 2 = 200 on each side).
        int pad = N_FFT / 2;
        int pcmLen = pcm16kMono.length;
        int paddedLen = pcmLen + 2 * pad;
        float[] padded = new float[paddedLen];
        for (int i = 0; i < paddedLen; i++) {
            if (i < pad) {
                // Reflect at start: index pad - i (bounded)
                int src = Math.min(pcmLen - 1, pad - i);
                padded[i] = pcm16kMono[src];
            } else if (i < pad + pcmLen) {
                padded[i] = pcm16kMono[i - pad];
            } else {
                // Reflect at end
                int excess = i - (pad + pcmLen);
                int src = Math.max(0, pcmLen - 2 - excess);
                padded[i] = pcm16kMono[src];
            }
        }

        int numFrames = 1 + (paddedLen - WIN_LENGTH) / HOP_LENGTH;
        float[][] feats = new float[numFrames][N_MELS];
        float[] frame = new float[WIN_LENGTH];
        float[] power = new float[N_STFT];

        float globalMaxDb = Float.NEGATIVE_INFINITY;

        for (int t = 0; t < numFrames; t++) {
            int off = t * HOP_LENGTH;
            for (int n = 0; n < WIN_LENGTH; n++) {
                frame[n] = padded[off + n] * window[n];
            }

            // Direct 201-bin real DFT with precomputed trigonometric tables.
            for (int k = 0; k < N_STFT; k++) {
                float re = 0f;
                float im = 0f;
                float[] cosK = cosTable[k];
                float[] sinK = sinTable[k];
                for (int n = 0; n < WIN_LENGTH; n++) {
                    float s = frame[n];
                    re += s * cosK[n];
                    im += s * sinK[n];
                }
                power[k] = re * re + im * im;
            }

            for (int m = 0; m < N_MELS; m++) {
                float[] filt = melFilters[m];
                double e = 0.0;
                for (int k = 0; k < N_STFT; k++) {
                    e += power[k] * filt[k];
                }
                float linearFb = (float) Math.max(e, LOG_FLOOR);
                float db = (float) (10.0 * Math.log10(linearFb));
                feats[t][m] = db;
                if (db > globalMaxDb) {
                    globalMaxDb = db;
                }
            }
        }

        // SpeechBrain _amplitude_to_DB top_db clamp: max(feats, max - 80).
        float minAllowedDb = globalMaxDb - TOP_DB;
        for (int t = 0; t < numFrames; t++) {
            for (int m = 0; m < N_MELS; m++) {
                if (feats[t][m] < minAllowedDb) {
                    feats[t][m] = minAllowedDb;
                }
            }
        }

        // Sentence mean normalization (InputNormalization norm_type=sentence, std_norm=false).
        for (int m = 0; m < N_MELS; m++) {
            double sum = 0.0;
            for (int t = 0; t < numFrames; t++) {
                sum += feats[t][m];
            }
            float mean = (float) (sum / numFrames);
            for (int t = 0; t < numFrames; t++) {
                feats[t][m] -= mean;
            }
        }

        return feats;
    }

    /** Number of FBank frames for a given sample count (with center padding). */
    public static int numFramesFor(int numSamples) {
        if (numSamples < WIN_LENGTH) return 0;
        int pad = N_FFT / 2;
        int paddedLen = numSamples + 2 * pad;
        return 1 + (paddedLen - WIN_LENGTH) / HOP_LENGTH;
    }

    // --- Internals ---------------------------------------------------

    private static float[] hamming(int n) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            w[i] = (float) (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1)));
        }
        return w;
    }

    /** SpeechBrain triangular mel filterbank [nMels][nStft]. */
    static float[][] buildMelFilters(int nMels, int nFft, int sr,
                                     float fMin, float fMax) {
        int nBins = nFft / 2 + 1; // 201
        float[][] fb = new float[nMels][nBins];
        float melMin = hzToMel(fMin);
        float melMax = hzToMel(fMax);
        float[] melPoints = new float[nMels + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = melMin + i * (melMax - melMin) / (nMels + 1);
        }
        float[] hzPoints = new float[melPoints.length];
        for (int i = 0; i < hzPoints.length; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
        }
        float[] band = new float[nMels + 1];
        for (int i = 0; i < nMels + 1; i++) {
            band[i] = hzPoints[i + 1] - hzPoints[i];
        }
        float[] fCentral = new float[nMels];
        System.arraycopy(hzPoints, 1, fCentral, 0, nMels);

        float[] allFreqs = new float[nBins];
        for (int k = 0; k < nBins; k++) {
            allFreqs[k] = (float) k * (sr / 2f) / (nBins - 1);
        }

        for (int m = 0; m < nMels; m++) {
            float fc = fCentral[m];
            float b = band[m];
            for (int k = 0; k < nBins; k++) {
                float slope = (allFreqs[k] - fc) / b;
                float tri = Math.max(0f, Math.min(slope + 1f, -slope + 1f));
                fb[m][k] = tri;
            }
        }
        return fb;
    }

    private static float hzToMel(float hz) {
        return 2595f * (float) Math.log10(1 + hz / 700f);
    }

    private static float melToHz(float mel) {
        return 700f * ((float) Math.pow(10, mel / 2595f) - 1);
    }
}
