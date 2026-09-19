/*
 * OmniVoice — Streaming ASR shared state & tuning constants.
 *
 * Spec: realtime_multilingual_asr_vi_en_zh_android.md §12, §19-§20, §23.
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Languages handled by the streaming router. */
public enum AsrLanguage {
    VI("vi"),
    EN("en"),
    ZH("zh"),
    UND("und");

    public final String code;

    AsrLanguage(String code) {
        this.code = code;
    }

    public static AsrLanguage fromCode(String code) {
        if (code == null) return UND;
        switch (code) {
            case "vi": return VI;
            case "en": return EN;
            case "zh": return ZH;
            default: return UND;
        }
    }

    /**
     * Model bucket that decodes this language (mixed EN/ZH plan): EN and ZH
     * share one bilingual model, so both map to {@link AsrModelType#EN_ZH}.
     * Routing decisions, rollback and the resident model pool all key on this
     * — the language label is only used for display and LID math.
     */
    public AsrModelType modelType() {
        return AsrModelType.of(this);
    }

    /** Canonical language label for a model bucket (for engine construction). */
    public static AsrLanguage canonicalOf(AsrModelType type) {
        return type == null ? UND : type.canonical;
    }
}
