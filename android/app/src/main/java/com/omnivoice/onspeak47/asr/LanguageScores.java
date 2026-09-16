/*
 * OmniVoice — VoxLingua bootstrap/runtime score carrier.
 *
 * Replaces the raw Map<AsrLanguage, Float> for the VoxLingua path (§31 of
 * the VoxLingua pipeline doc): the router needs more than VI/EN/ZH
 * probabilities — it needs the GLOBAL 107-class winner to implement the
 * "don't force unsupported languages into VI/EN/ZH" policy (§14).
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

import java.util.EnumMap;
import java.util.Map;

public final class LanguageScores {

    /** Supported-language probabilities (need not sum to 1; caller normalizes). */
    public final float vi;
    public final float en;
    public final float zh;

    /** Global 107-class winner (ISO code, e.g. "en", "ja", "th"). */
    public final String globalTopLanguage;
    /** Global winner index in VoxLinguaLabels.CODES, -1 when unknown. */
    public final int globalTopIndex;
    /** Global winner probability after 107-way softmax, in [0,1]. */
    public final float globalTopScore;

    /** Number of raw frames/windows fused into this result (smoothing depth). */
    public final int numWindows;

    public LanguageScores(float vi, float en, float zh,
                          String globalTopLanguage, int globalTopIndex,
                          float globalTopScore, int numWindows) {
        this.vi = vi;
        this.en = en;
        this.zh = zh;
        this.globalTopLanguage = globalTopLanguage == null ? "unk" : globalTopLanguage;
        this.globalTopIndex = globalTopIndex;
        this.globalTopScore = globalTopScore;
        this.numWindows = numWindows;
    }

    /** Flat uncertain prior (no evidence yet). */
    public static LanguageScores flat() {
        return new LanguageScores(1.0f / 3, 1.0f / 3, 1.0f / 3,
                "unk", -1, 1.0f / 107, 0);
    }

    /** True when the global winner is one of VI/EN/ZH. */
    public boolean isSupportedTop() {
        return VoxLinguaLabels.isSupported(globalTopLanguage);
    }

    /** Best supported language by VI/EN/ZH probability. */
    public AsrLanguage topSupported() {
        if (vi >= en && vi >= zh) return AsrLanguage.VI;
        if (en >= vi && en >= zh) return AsrLanguage.EN;
        return AsrLanguage.ZH;
    }

    /** Top supported probability. */
    public float topSupportedScore() {
        switch (topSupported()) {
            case VI: return vi;
            case EN: return en;
            default: return zh;
        }
    }

    /** Second-best supported probability (for the margin gate). */
    public float secondSupportedScore() {
        AsrLanguage top = topSupported();
        if (top == AsrLanguage.VI) return Math.max(en, zh);
        if (top == AsrLanguage.EN) return Math.max(vi, zh);
        return Math.max(vi, en);
    }

    /**
     * Bootstrap gate (§15): top >= 0.70 AND top - second >= 0.15 AND the
     * global winner is supported (otherwise UNKNOWN even when the supported
     * top looks confident — e.g. Japanese audio scoring ZH 0.75 by relative
     * margin must not commit ZH).
     */
    public boolean isBootstrapConfident() {
        if (!isSupportedTop()) return false;
        float top = topSupportedScore();
        float second = secondSupportedScore();
        return top >= AsrState.BOOTSTRAP_THRESHOLD
                && (top - second) >= AsrState.BOOTSTRAP_MARGIN;
    }

    /** VI/EN/ZH view for the legacy Map-based router/LID paths. */
    public Map<AsrLanguage, Float> toMap() {
        Map<AsrLanguage, Float> m = new EnumMap<>(AsrLanguage.class);
        m.put(AsrLanguage.VI, vi);
        m.put(AsrLanguage.EN, en);
        m.put(AsrLanguage.ZH, zh);
        return m;
    }

    @Override
    public String toString() {
        return "LanguageScores{vi=" + vi + " en=" + en + " zh=" + zh
                + " globalTop=" + globalTopLanguage + "(" + globalTopIndex + ")"
                + " globalScore=" + globalTopScore + " n=" + numWindows + "}";
    }
}
