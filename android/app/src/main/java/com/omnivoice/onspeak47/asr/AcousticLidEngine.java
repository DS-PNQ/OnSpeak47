/*
 * OmniVoice — Bootstrap acoustic language identification (fakedemo2 §16).
 *
 * Answers "which language does this utterance START with?" from PCM AUDIO
 * only — never from the active-ASR transcript. That split is the core fix:
 * the old pipeline decoded EN/ZH speech with the VI Zipformer first, then
 * fed the Vietnamese-looking hypothesis back into a text-based LID, which
 * circularly confirmed VI (§4).
 *
 * Wiring: implement this interface with the real tiny classifier (§33:
 * 16 kHz log-mel → tiny CNN, VI/EN/ZH, 1–5M params) and inject it via
 *   new LanguageIdEngine(scorer, AsrState.EMA_ALPHA)
 * or lid.setAcousticScorer(...). Until that model lands,
 * HeuristicAcousticLidEngine below is the bundled interim: it is genuinely
 * audio-derived (prosodic time-domain cues) but deliberately UNCERTAIN by
 * design — it never reaches the 0.70 bootstrap bar on its own, so every
 * utterance falls through to the extended-window / dual-candidate path (§9,
 * §32) instead of being forced into VI. That is strictly better than the
 * old flat text prior, and the pipeline treats both paths identically.
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

import java.util.EnumMap;
import java.util.Map;

public interface AcousticLidEngine {

    /**
     * Classify one audio window (typically 200–300 ms @ 16 kHz mono floats).
     *
     * @return per-language scores in [0,1]; need not sum to 1 (the caller
     *         normalizes). Must never return null. Must not touch transcripts.
     */
    Map<AsrLanguage, Float> classify(float[] audioWindow);

    // --- Interim heuristic (§33 until the trained tiny model lands) --------

    /**
     * Audio-derived heuristic LID. Uses only time-domain prosodic cues:
     * voiced-frame ratio (tonal VI/ZH lean voiced over 300 ms), sibilance
     * ratio (sustained high zero-crossing → EN-leaning frication), and short
     * energy dynamics. Biases are intentionally SMALL: the top score is
     * capped so bootstrap stays UNCERTAIN and the pipeline runs the
     * dual-candidate fallback rather than committing on weak evidence.
     * Swap for the ONNX tiny classifier without touching callers.
     */
    class HeuristicAcousticLidEngine implements AcousticLidEngine {

        /** Hard ceiling: heuristic alone must not pass BOOTSTRAP_THRESHOLD. */
        private static final float MAX_TOP = 0.58f;

        @Override
        public Map<AsrLanguage, Float> classify(float[] audioWindow) {
            Map<AsrLanguage, Float> out = new EnumMap<>(AsrLanguage.class);
            if (audioWindow == null || audioWindow.length < AsrState.SAMPLE_RATE / 10) {
                out.put(AsrLanguage.VI, 1.0f / 3);
                out.put(AsrLanguage.EN, 1.0f / 3);
                out.put(AsrLanguage.ZH, 1.0f / 3);
                return out;
            }
            // Silence carries no language information: score flat instead of
            // letting a voicing bias vote on near-zero audio (e.g. an
            // endpoint window that is mostly trailing silence).
            double energy = 0;
            for (float v : audioWindow) energy += v * v;
            energy = Math.sqrt(energy / audioWindow.length);
            if (energy < 0.008) {
                out.put(AsrLanguage.VI, 1.0f / 3);
                out.put(AsrLanguage.EN, 1.0f / 3);
                out.put(AsrLanguage.ZH, 1.0f / 3);
                return out;
            }
            double voiced = voicedRatio(audioWindow);
            double sibilant = sibilantRatio(audioWindow);
            // Tonal speech (VI/ZH) sustains voicing; EN carries more
            // frication. ZH gets half the tonal bias (no segmental evidence
            // in the time domain — CJK text stays a runtime-only cue, §19).
            double vi = 1.0 + 0.35 * (voiced - 0.5) - 0.30 * sibilant;
            double en = 1.0 - 0.25 * (voiced - 0.5) + 0.35 * sibilant;
            double zh = 1.0 + 0.18 * (voiced - 0.5) - 0.10 * sibilant;
            double sum = vi + en + zh;
            float fvi = (float) (vi / sum);
            float fen = (float) (en / sum);
            float fzh = (float) (zh / sum);
            // Cap the winner: weak audio evidence must not commit (§8).
            float top = Math.max(fvi, Math.max(fen, fzh));
            if (top > MAX_TOP) {
                float excess = top - MAX_TOP;
                if (fvi == top) {
                    fvi = MAX_TOP;
                    fen += excess / 2;
                    fzh += excess / 2;
                } else if (fen == top) {
                    fen = MAX_TOP;
                    fvi += excess / 2;
                    fzh += excess / 2;
                } else {
                    fzh = MAX_TOP;
                    fvi += excess / 2;
                    fen += excess / 2;
                }
            }
            out.put(AsrLanguage.VI, clamp01(fvi));
            out.put(AsrLanguage.EN, clamp01(fen));
            out.put(AsrLanguage.ZH, clamp01(fzh));
            return out;
        }

        /** Fraction of 20 ms frames with strong voicing (autocorrelation). */
        private static double voicedRatio(float[] s) {
            int frame = AsrState.SAMPLE_RATE * 20 / 1000;
            int voiced = 0;
            int total = 0;
            for (int start = 0; start + frame <= s.length; start += frame) {
                total++;
                if (isVoiced(s, start, frame)) voiced++;
            }
            return total == 0 ? 0.5 : (double) voiced / total;
        }

        private static boolean isVoiced(float[] s, int off, int n) {
            double e0 = 0;
            for (int i = 0; i < n; i++) e0 += s[off + i] * s[off + i];
            if (e0 < 1e-6) return false;
            // Normalized autocorrelation at lags 2.5–12.5 ms (80–400 Hz F0).
            double best = 0;
            for (int lag = 40; lag <= 200; lag += 4) {
                if (off + lag + n > s.length) break;
                double e1 = 0;
                double corr = 0;
                for (int i = 0; i < n - lag; i++) {
                    corr += s[off + i] * s[off + i + lag];
                    e1 += s[off + i + lag] * s[off + i + lag];
                }
                double denom = Math.sqrt(e0 * e1);
                if (denom > 0) best = Math.max(best, corr / denom);
            }
            return best > 0.45;
        }

        /** Fraction of frames dominated by zero-crossings (frication). */
        private static double sibilantRatio(float[] s) {
            int frame = AsrState.SAMPLE_RATE * 20 / 1000;
            int sib = 0;
            int total = 0;
            for (int start = 0; start + frame <= s.length; start += frame) {
                total++;
                int zc = 0;
                for (int i = 1; i < frame; i++) {
                    if ((s[start + i] >= 0) != (s[start + i - 1] >= 0)) zc++;
                }
                double zcr = (double) zc / frame;
                double rms = 0;
                for (int i = 0; i < frame; i++) rms += s[start + i] * s[start + i];
                rms = Math.sqrt(rms / frame);
                if (zcr > 0.28 && rms > 0.015) sib++;
            }
            return total == 0 ? 0 : (double) sib / total;
        }

        private static float clamp01(float v) {
            return Math.max(0, Math.min(1, v));
        }
    }

    /** Adapter: expose any AcousticLidEngine as a LanguageIdEngine scorer. */
    class Adapter implements LanguageIdEngine.AcousticScorer {
        private final AcousticLidEngine engine;

        public Adapter(AcousticLidEngine engine) {
            this.engine = engine;
        }

        @Override
        public Map<AsrLanguage, Float> score(float[] audioWindow) {
            if (engine == null) return null;
            return engine.classify(audioWindow);
        }
    }
}
