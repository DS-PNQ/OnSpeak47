/*
 * OmniVoice — VoxLingua bootstrap/runtime score carrier.
 *
 * Replaces the raw Map<AsrLanguage, Float> for the VoxLingua path (§31 of
 * the VoxLingua pipeline doc): the router needs more than VI/EN/ZH
 * probabilities — it needs absolute evidence plus the global 107-class
 * winner for the "don't force unsupported languages into VI/EN/ZH" policy
 * (§14).
 *
 * vi/en/zh are SUPPORTED-RELATIVE posteriors (renormalized among the three,
 * sum ~= 1) for the 0.70/0.15 relative gate. viAbs/enAbs/zhAbs are ABSOLUTE
 * 107-way posteriors for the absolute-evidence gate: on short windows the
 * 107-class argmax is regularly an unrelated language while the relative
 * evidence is already decisive, so the gate reads absolute mass, not the
 * argmax label (2026-09-16 LID-bootstrap fix, see
 * docs/voxlingua_lid_diagnosis.md).
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

    /** Absolute 107-way posteriors for the supported languages. */
    public final float viAbs;
    public final float enAbs;
    public final float zhAbs;

    /** Global 107-class winner (ISO code, e.g. "en", "ja", "th"). */
    public final String globalTopLanguage;
    /** Global winner index in VoxLinguaLabels.CODES, -1 when unknown. */
    public final int globalTopIndex;
    /** Global winner probability after 107-way softmax, in [0,1]. */
    public final float globalTopScore;

    /** Number of raw frames/windows fused into this result (smoothing depth). */
    public final int numWindows;

    /**
     * Smoother-stamped flag (2026-09-16 v2): true when the smoother's full
     * depth of raw windows unanimously agrees on the supported top. Set by
     * {@link VoxLinguaTemporalSmoother#add} only — never by the engine. A
     * plain field (not constructor state) so all existing call sites keep
     * compiling; it only ever transitions false → true inside the smoother.
     */
    public boolean unanimous = false;

    public LanguageScores(float vi, float en, float zh,
                          float viAbs, float enAbs, float zhAbs,
                          String globalTopLanguage, int globalTopIndex,
                          float globalTopScore, int numWindows) {
        this.vi = vi;
        this.en = en;
        this.zh = zh;
        this.viAbs = viAbs;
        this.enAbs = enAbs;
        this.zhAbs = zhAbs;
        this.globalTopLanguage = globalTopLanguage == null ? "unk" : globalTopLanguage;
        this.globalTopIndex = globalTopIndex;
        this.globalTopScore = globalTopScore;
        this.numWindows = numWindows;
    }

    /**
     * Legacy constructor (tests/seams without absolute mass): absolute
     * values are estimated from the global top — the supported winner keeps
     * the global score, the rest share the remainder proportionally. Real
     * engine output must use the full constructor.
     */
    public LanguageScores(float vi, float en, float zh,
                          String globalTopLanguage, int globalTopIndex,
                          float globalTopScore, int numWindows) {
        this(vi, en, zh,
                estimateAbs(AsrLanguage.VI, vi, en, zh,
                        globalTopLanguage, globalTopScore),
                estimateAbs(AsrLanguage.EN, vi, en, zh,
                        globalTopLanguage, globalTopScore),
                estimateAbs(AsrLanguage.ZH, vi, en, zh,
                        globalTopLanguage, globalTopScore),
                globalTopLanguage, globalTopIndex, globalTopScore, numWindows);
    }

    private static float estimateAbs(AsrLanguage lang, float vi, float en, float zh,
                                     String globalTop, float globalScore) {
        String code = lang == AsrLanguage.VI ? "vi"
                : lang == AsrLanguage.EN ? "en" : "zh";
        float rel = lang == AsrLanguage.VI ? vi : lang == AsrLanguage.EN ? en : zh;
        if (code.equals(globalTop)) return globalScore;
        return rel * (1f - globalScore);
    }

    /** Flat uncertain prior (no evidence yet). */
    public static LanguageScores flat() {
        return new LanguageScores(1.0f / 3, 1.0f / 3, 1.0f / 3,
                1.0f / 107, 1.0f / 107, 1.0f / 107,
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

    /** Top supported ABSOLUTE posterior. */
    public float topSupportedAbs() {
        switch (topSupported()) {
            case VI: return viAbs;
            case EN: return enAbs;
            default: return zhAbs;
        }
    }

    /** Second-best supported probability (for the margin gate). */
    public float secondSupportedScore() {
        AsrLanguage top = topSupported();
        if (top == AsrLanguage.VI) return Math.max(en, zh);
        if (top == AsrLanguage.EN) return Math.max(vi, zh);
        return Math.max(vi, en);
    }

    /** Shared relative part: top {@code >=} 0.70 AND top - second {@code >=} 0.15. */
    private boolean relativeGate() {
        float top = topSupportedScore();
        float second = secondSupportedScore();
        return top >= AsrState.BOOTSTRAP_THRESHOLD
                && (top - second) >= AsrState.BOOTSTRAP_MARGIN;
    }

    /**
     * FAST bootstrap gate (§15, 2026-09-16 fix).
     *
     * Relative gate PLUS absolute posterior mass (replaces the old
     * "107-class argmax must be en/vi/zh" rule). Commits clean audio
     * quickly; on noisy device-mic audio the absolute mass stays tiny and
     * the SLOW path below takes over instead.
     */
    public boolean isBootstrapConfident() {
        if (!relativeGate()) return false;
        return topSupportedAbs() >= AsrState.VOXLINGUA_MIN_SUPPORTED_ABS_SCORE;
    }

    /**
     * SLOW bootstrap gate (2026-09-16 v2, field-log fix).
     *
     * Device-mic audio yields flat 107-way distributions (supported abs
     * ~= 0.01) with a CORRECT relative ranking — the absolute bar never
     * opens there. The slow path trusts the relative gate once a full
     * smoother depth UNANIMOUSLY agrees (single-window flips, which happen
     * in both directions, cannot commit alone). Residue of the §14 policy:
     * a STRONG unsupported argmax still blocks (true foreign audio waits);
     * weak/flat argmax tops do not.
     */
    public boolean isSlowBootstrapConfident() {
        if (!unanimous) return false;
        if (!relativeGate()) return false;
        if (!isSupportedTop()
                && globalTopScore >= AsrState.VOXLINGUA_FOREIGN_TOP_REJECT) {
            return false;
        }
        return true;
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
                + " viAbs=" + viAbs + " enAbs=" + enAbs + " zhAbs=" + zhAbs
                + " globalTop=" + globalTopLanguage + "(" + globalTopIndex + ")"
                + " globalScore=" + globalTopScore + " n=" + numWindows + "}";
    }
}
