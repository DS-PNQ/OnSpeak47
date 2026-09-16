/*
 * OmniVoice — VoxLingua temporal smoothing (§16 of the VoxLingua pipeline).
 *
 * Never decide from a single ECAPA inference. The pipeline feeds rolling
 * windows (600 ms window, 200 ms hop) and this smoother fuses the last N
 * results with EMA + persistence before the router gate sees them:
 *
 *   t=400: VI .43 EN .49 ZH .08  → uncertain
 *   t=600: VI .22 EN .72 ZH .06  → leaning EN
 *   t=800: VI .12 EN .82 ZH .06  → EN locked
 *
 * EMA (alpha 0.5) reacts within ~2 hops while rejecting single-window
 * flaps; the router's 0.70 + 0.15 gate still applies on the SMOOTHED
 * scores, so no threshold is bypassed here.
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

import java.util.ArrayDeque;
import java.util.Deque;

public final class VoxLinguaTemporalSmoother {

    /** Rolling depth: 3 hops × 200 ms = 600 ms of history. */
    public static final int DEFAULT_DEPTH = 3;
    /** EMA weight for the newest window. */
    public static final float DEFAULT_ALPHA = 0.5f;

    private final int depth;
    private final float alpha;
    private final Deque<LanguageScores> history = new ArrayDeque<>();
    private LanguageScores ema;

    public VoxLinguaTemporalSmoother() {
        this(DEFAULT_DEPTH, DEFAULT_ALPHA);
    }

    public VoxLinguaTemporalSmoother(int depth, float alpha) {
        this.depth = Math.max(1, depth);
        this.alpha = Math.min(1f, Math.max(0f, alpha));
    }

    /** Feed one raw window result; returns the smoothed aggregate. */
    public synchronized LanguageScores add(LanguageScores raw) {
        if (raw == null) raw = LanguageScores.flat();
        history.addLast(raw);
        while (history.size() > depth) history.removeFirst();
        if (ema == null) {
            ema = raw;
        } else {
            float vi = alpha * raw.vi + (1 - alpha) * ema.vi;
            float en = alpha * raw.en + (1 - alpha) * ema.en;
            float zh = alpha * raw.zh + (1 - alpha) * ema.zh;
            // Global top follows the newest confident window; when the new
            // window is flat/uncertain keep the previous top so one silence
            // hop does not wipe the language.
            String gTop = ema.globalTopLanguage;
            int gIdx = ema.globalTopIndex;
            float gScore = ema.globalTopScore;
            if (raw.numWindows > 0 && raw.globalTopScore >= ema.globalTopScore) {
                gTop = raw.globalTopLanguage;
                gIdx = raw.globalTopIndex;
                gScore = raw.globalTopScore;
            }
            ema = new LanguageScores(vi, en, zh, gTop, gIdx, gScore,
                    ema.numWindows + 1);
        }
        // Majority-vote stabilizer: when the last `depth` windows agree on
        // the supported top, snap the EMA toward that language so a clean
        // 3-window run locks without waiting for full EMA convergence.
        if (history.size() >= depth) {
            AsrLanguage vote = majorityTop();
            if (vote != null && vote == ema.topSupported() && ema.isSupportedTop()) {
                float boost = 0.05f;
                float vi = ema.vi, en = ema.en, zh = ema.zh;
                if (vote == AsrLanguage.VI) vi = Math.min(1f, vi + boost);
                else if (vote == AsrLanguage.EN) en = Math.min(1f, en + boost);
                else zh = Math.min(1f, zh + boost);
                float sum = vi + en + zh;
                ema = new LanguageScores(vi / sum, en / sum, zh / sum,
                        ema.globalTopLanguage, ema.globalTopIndex,
                        ema.globalTopScore, ema.numWindows);
            }
        }
        return ema;
    }

    public synchronized LanguageScores smoothed() {
        return ema == null ? LanguageScores.flat() : ema;
    }

    public synchronized int size() {
        return history.size();
    }

    public synchronized void reset() {
        history.clear();
        ema = null;
    }

    private AsrLanguage majorityTop() {
        int vi = 0, en = 0, zh = 0;
        for (LanguageScores s : history) {
            switch (s.topSupported()) {
                case VI: vi++; break;
                case EN: en++; break;
                default: zh++; break;
            }
        }
        int need = depth / 2 + 1;
        if (vi >= need) return AsrLanguage.VI;
        if (en >= need) return AsrLanguage.EN;
        if (zh >= need) return AsrLanguage.ZH;
        return null;
    }
}
