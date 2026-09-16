/*
 * OmniVoice — VoxLingua107 label table.
 *
 * Ordered mapping for speechbrain/lang-id-voxlingua107-ecapa (107 classes)
 * and its community ONNX export
 * (beginning-ai/speechbrain-lang-id-voxlingua107-ecapa-onnx: model.onnx +
 * labels.json). Order below == labels.json order == classifier output order.
 *
 * App only decodes VI/EN/ZH, but the policy (§14 of the VoxLingua pipeline
 * doc) must see the GLOBAL top-1 first: if the global winner is outside
 * VI/EN/ZH the utterance is UNKNOWN/unsupported instead of being forced
 * into the nearest of the three (e.g. Japanese → ZH, Thai → VI).
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

public final class VoxLinguaLabels {

    private VoxLinguaLabels() {}

    /** Number of VoxLingua107 classes. */
    public static final int NUM_LANGUAGES = 107;

    /** Classifier indices for the three supported languages. */
    public static final int IDX_EN = 20;
    public static final int IDX_VI = 102;
    public static final int IDX_ZH = 106;

    /** ISO codes in classifier order (labels.json). */
    public static final String[] CODES = {
            "ab", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn",
            "bo", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el",
            "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl",
            "gn", "gu", "gv", "ha", "haw", "hi", "hr", "ht", "hu", "hy",
            "ia", "id", "is", "it", "iw", "ja", "jw", "ka", "kk", "km",
            "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi",
            "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
            "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sco",
            "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv",
            "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk",
            "ur", "uz", "vi", "war", "yi", "yo", "zh"
    };

    static {
        if (CODES.length != NUM_LANGUAGES) {
            throw new IllegalStateException(
                    "VoxLingua label table must hold 107 entries, got " + CODES.length);
        }
    }

    /** ISO code for a classifier index, or "unk" when out of range. */
    public static String codeAt(int index) {
        if (index < 0 || index >= CODES.length) return "unk";
        return CODES[index];
    }

    /** Classifier index for an ISO code, or -1 when unknown. */
    public static int indexOf(String code) {
        if (code == null) return -1;
        for (int i = 0; i < CODES.length; i++) {
            if (CODES[i].equals(code)) return i;
        }
        return -1;
    }

    /** True when the ISO code is one of the three decoded languages. */
    public static boolean isSupported(String code) {
        return "en".equals(code) || "vi".equals(code) || "zh".equals(code);
    }

    /** Map a supported ISO code to the app enum, else UND. */
    public static AsrLanguage toAsrLanguage(String code) {
        if ("vi".equals(code)) return AsrLanguage.VI;
        if ("en".equals(code)) return AsrLanguage.EN;
        if ("zh".equals(code)) return AsrLanguage.ZH;
        return AsrLanguage.UND;
    }
}
