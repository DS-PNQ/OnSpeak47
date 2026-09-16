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
 * EMA reacts within ~2 hops while rejecting single-window flaps; the
 * router's 0.70 + 0.15 gate still applies on the SMOOTHED scores, so no
 * threshold is bypassed here.
 *
 * 2026-09-16 fix: the global-top label FOLLOWS THE LATEST window. The old
 * "keep the highest score ever seen" rule latched one noisy window (e.g.
 * `lo 0.474` at 400 ms) and locked the whole utterance to UND. The bootstrap
 * gate no longer reads the argmax label (it reads absolute mass), so a
 * single noisy window can no longer jam the pipeline. Absolute supported
 * posteriors are EMA-smoothed alongside the relative ones.
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

import java.util.ArrayDeque;
import java.util.Deque;

public final class VoxLinguaTemporalSmoother {

    /** Rolling depth: 3 hops × 200 ms = 600 ms of history. */
    public static final int DEFAULT_DEPTH = 3;
    /** EMA weight for the newest window (unit-test default; the engine wires
     *  AsrState.LID_EMA_ALPHA explicitly — mirrors backend/voxlingua.py). */
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
            float a = alpha;
            float vi = a * raw.vi + (1 - a) * ema.vi;
            float en = a * raw.en + (1 - a) * ema.en;
            float zh = a * raw.zh + (1 - a) * ema.zh;
            // Absolute posteriors are EMA'd as well: the bootstrap gate reads
            // the smoothed absolute evidence instead of one noisy window
            // (2026-09-16 fix).
            float viAbs = a * raw.viAbs + (1 - a) * ema.viAbs;
            float enAbs = a * raw.enAbs + (1 - a) * ema.enAbs;
            float zhAbs = a * raw.zhAbs + (1 - a) * ema.zhAbs;
            // The global top simply follows the newest window. It used to be
            // pinned to the highest globalTopScore ever seen, which meant one
            // junk window (0.6 × "lo") locked isSupportedTop()==false for the
            // rest of the utterance and the gate could never reopen.
            ema = new LanguageScores(vi, en, zh, viAbs, enAbs, zhAbs,
                    raw.globalTopLanguage, raw.globalTopIndex,
                    raw.globalTopScore, ema.numWindows + 1);
        }
        // Majority-vote stabilizer: when the last `depth` windows agree on
        // the supported top, snap the EMA toward that language so a clean
        // 3-window run locks without waiting for full EMA convergence.
        // (No isSupportedTop() requirement any more — the argmax is telemetry
        // only since the gate moved to absolute evidence.)
        if (history.size() >= depth) {
            AsrLanguage vote = majorityTop();
            if (vote != null && vote == ema.topSupported()) {
                float boost = 0.05f;
                float vi = ema.vi, en = ema.en, zh = ema.zh;
                if (vote == AsrLanguage.VI) vi = Math.min(1f, vi + boost);
                else if (vote == AsrLanguage.EN) en = Math.min(1f, en + boost);
                else zh = Math.min(1f, zh + boost);
                float sum = vi + en + zh;
                ema = new LanguageScores(vi / sum, en / sum, zh / sum,
                        ema.viAbs, ema.enAbs, ema.zhAbs,
                        ema.globalTopLanguage, ema.globalTopIndex,
                        ema.globalTopScore, ema.numWindows);
            }
        }
        // Stamp unanimity for the SLOW bootstrap gate (2026-09-16 v2): a full
        // depth of raw windows agreeing on the supported top. The boost above
        // never changes history, so this stays exact.
        ema.unanimous = history.size() >= depth && allAgree();
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

    /** True when every buffered raw window shares one supported top. */
    private boolean allAgree() {
        AsrLanguage first = null;
        for (LanguageScores s : history) {
            AsrLanguage t = s.topSupported();
            if (first == null) first = t;
            else if (t != first) return false;
        }
        return first != null;
    }
}
