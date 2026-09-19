/*
 * OmniVoice — ASR model identity (the unit the router switches between).
 *
 * Mixed EN/ZH plan: the pipeline loads TWO streaming transducer models, not
 * three.
 *
 *   VI     → zipformer_vi_* (hynt 30M)
 *   EN_ZH  → zipformer_en_zh_mixed_* (k2fsa-zipformer-chinese-english-mixed)
 *
 * {@link AsrLanguage} stays the *language* label — VoxLingua107 scores en and
 * zh separately and the UI names the heard language — while AsrModelType is
 * the unit the model pool keys on and the unit the router switches between.
 *
 * Because EN and ZH map to ONE model type:
 *   - a LID flip en↔zh mid-utterance is not a model switch: no rollback, no
 *     shadow decode, no stream reset — the bilingual BPE vocabulary handles
 *     code switching inside the model;
 *   - the resident pool holds 2 native sessions instead of 3, even though EN
 *     and ZH previously shared the same files (they were loaded twice).
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Streaming ASR model identity: one entry per loaded native model. */
public enum AsrModelType {
    /** Vietnamese streaming Zipformer. */
    VI("vi", AsrLanguage.VI),
    /** Shared bilingual EN+ZH model; handles EN↔ZH code switching internally. */
    EN_ZH("en_zh", AsrLanguage.EN),
    /** Undefined / unsupported — never a model slot. */
    UND("und", AsrLanguage.UND);

    /** Stable id for logs/telemetry ("vi", "en_zh", "und"). */
    public final String code;
    /**
     * Canonical language label of this bucket, used where a concrete
     * {@link AsrLanguage} is still required (engine construction, asset
     * resolution). EN_ZH → EN; the EN/ZH distinction survives only as a
     * label, never as a second model.
     */
    public final AsrLanguage canonical;

    AsrModelType(String code, AsrLanguage canonical) {
        this.code = code;
        this.canonical = canonical;
    }

    /** Model bucket that serves {@code lang}; null/UND → {@link #UND}. */
    public static AsrModelType of(AsrLanguage lang) {
        if (lang == null) return UND;
        switch (lang) {
            case VI: return VI;
            case EN:
            case ZH: return EN_ZH;
            default: return UND;
        }
    }

    /** True when this bucket is the model that decodes {@code lang}. */
    public boolean covers(AsrLanguage lang) {
        return of(lang) == this;
    }

    /** Parse a stable id; unknown/null → {@link #UND}. */
    public static AsrModelType fromCode(String code) {
        if (code == null) return UND;
        switch (code) {
            case "vi": return VI;
            case "en_zh": return EN_ZH;
            default: return UND;
        }
    }
}