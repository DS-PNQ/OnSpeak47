/*
 * OmniVoice — Async language identification (stage-1 cheap signals).
 *
 * Spec §10-§12, §16-§17, OPTIONAL 3/6:
 *   score(lang) = 0.45*acoustic + 0.30*text + 0.15*confidence + 0.10*history
 * Stage 1 here = text evidence (CJK ratio + VI/EN lexicons) + an injectable
 * acoustic score hook for the real acoustic LID model (stage 2, OPTIONAL 3).
 * Runs off the ASR thread. EMA smoothing (OPTIONAL 7) feeds the
 * telemetry view only — the router gates on fused raw scores so switches
 * don't lag by seconds (see LidResult scores vs smoothedScores()).
 * Pure-Java (no android.* imports).
 */
package com.omnivoice.onspeak47.asr;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class LanguageIdEngine {

    /** LID output for one window. */
    public static class LidResult {
        public final AsrLanguage language;
        public final float confidence;
        public final Map<AsrLanguage, Float> scores;

        public LidResult(AsrLanguage language, float confidence,
                         Map<AsrLanguage, Float> scores) {
            this.language = language;
            this.confidence = confidence;
            this.scores = scores;
        }
    }

    /** Injectable acoustic scorer (real model = stage 2, spec OPTIONAL 3). */
    public interface AcousticScorer {
        /** Per-language acoustic probabilities, need not sum to 1. */
        Map<AsrLanguage, Float> score(float[] audioWindow);
    }

    // --- Secondary lexical evidence (spec §17) — never the sole signal. ---
    private static final Set<String> EN_WORDS = new HashSet<>(Arrays.asList(
            "meeting", "project", "system", "beautiful", "discussion",
            "report", "team", "final", "tomorrow", "today", "please",
            "thanks", "hello", "discuss", "schedule", "deadline", "email"));
    private static final Set<String> VI_WORDS = new HashSet<>(Arrays.asList(
            "hôm", "nay", "trời", "rất", "đẹp", "mình", "tôi", "đang",
            "không", "của", "và", "là", "một", "người", "với", "để",
            "hôm nay", "xin", "chào", "cảm", "ơn", "gửi"));

    /** High-frequency English function words (word-boundary matched). */
    private static final Set<String> EN_COMMON = new HashSet<>(Arrays.asList(
            "you", "the", "and", "are", "was", "were", "have", "has", "will",
            "would", "what", "when", "this", "that", "with", "from", "hi"));

    private AcousticScorer acousticScorer;
    private final Map<AsrLanguage, Float> smoothed = new EnumMap<>(AsrLanguage.class);
    private final Map<AsrLanguage, Float> history = new EnumMap<>(AsrLanguage.class);
    private final float emaAlpha;

    public LanguageIdEngine() {
        this(null, AsrState.EMA_ALPHA);
    }

    public LanguageIdEngine(AcousticScorer acousticScorer, float emaAlpha) {
        this.acousticScorer = acousticScorer;
        this.emaAlpha = emaAlpha;
        for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
            smoothed.put(l, 1.0f / 3);
            history.put(l, 1.0f / 3);
        }
    }

    public void setAcousticScorer(AcousticScorer scorer) {
        this.acousticScorer = scorer;
    }

    /**
     * Classify one window. Never blocks ASR — call from the LID worker.
     *
     * @param audioWindow  recent PCM (e.g. last 480 ms from the ring buffer)
     * @param partialText  current active-ASR partial transcript
     * @param tokenConf    active-ASR token confidence in [0,1]
     * @param active       currently active language (for the history term)
     */
    public synchronized LidResult classify(float[] audioWindow, String partialText,
                                           float tokenConf, AsrLanguage active) {
        Map<AsrLanguage, Float> acoustic = acousticScores(audioWindow, partialText);
        Map<AsrLanguage, Float> textEv = textEvidence(partialText);

        Map<AsrLanguage, Float> combined = new EnumMap<>(AsrLanguage.class);
        for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
            float a = acoustic.getOrDefault(l, 1.0f / 3);
            float t = textEv.getOrDefault(l, 1.0f / 3);
            float h = history.getOrDefault(l, 1.0f / 3);
            float s = AsrState.W_ACOUSTIC * a
                    + AsrState.W_TEXT * t
                    + AsrState.W_CONFIDENCE * tokenConfWeight(l, active, tokenConf)
                    + AsrState.W_HISTORY * h;
            combined.put(l, s);
        }
        normalize(combined);

        // EMA (spec OPTIONAL 7) smooths only the telemetry view: the router's
        // hysteresis gate (threshold + margin + 200 ms persistence, spec §12)
        // already rejects single-window flaps, so gating on the EMA as well
        // would double-damp and push switch latency into seconds.
        for (AsrLanguage l : combined.keySet()) {
            float prev = smoothed.getOrDefault(l, 1.0f / 3);
            smoothed.put(l, emaAlpha * combined.get(l) + (1 - emaAlpha) * prev);
        }

        AsrLanguage best = AsrLanguage.VI;
        float bestScore = -1;
        for (Map.Entry<AsrLanguage, Float> e : combined.entrySet()) {
            if (e.getValue() > bestScore) {
                bestScore = e.getValue();
                best = e.getKey();
            }
        }
        // Language-history prior drifts toward the winner (spec §12 term).
        for (AsrLanguage l : history.keySet()) {
            float prev = history.get(l);
            history.put(l, 0.9f * prev + 0.1f * (l == best ? 1.0f : 0.0f));
        }
        normalize(history);

        return new LidResult(best, bestScore, new EnumMap<>(combined));
    }

    /** EMA-stability view for UI/telemetry (not the router gate). */
    public synchronized Map<AsrLanguage, Float> smoothedScores() {
        return new EnumMap<>(smoothed);
    }

    private float tokenConfWeight(AsrLanguage lang, AsrLanguage active, float conf) {
        // Token confidence supports the *active* hypothesis; alternatives get
        // the complement. Keeps the 0.15 weight meaningful without a per-lang
        // confidence model.
        if (lang == active) return clamp01(conf);
        return clamp01(1.0f - conf) / 2.0f;
    }

    private Map<AsrLanguage, Float> acousticScores(float[] audio, String partialText) {
        if (acousticScorer != null && audio != null && audio.length > 0) {
            try {
                Map<AsrLanguage, Float> s = acousticScorer.score(audio);
                if (s != null && !s.isEmpty()) {
                    Map<AsrLanguage, Float> out = new EnumMap<>(AsrLanguage.class);
                    for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
                        out.put(l, clamp01(s.getOrDefault(l, 1.0f / 3)));
                    }
                    normalize(out);
                    return out;
                }
            } catch (Exception ignored) {
                // Fall through to the text-derived prior.
            }
        }
        // No acoustic model: weak prior from CJK density (spec §16). This is
        // intentionally flat for VI vs EN (both Latin script, spec §17) so the
        // text + history terms decide until stage-2 LID lands.
        float cjk = cjkRatio(partialText);
        Map<AsrLanguage, Float> out = new EnumMap<>(AsrLanguage.class);
        out.put(AsrLanguage.ZH, 0.15f + 0.7f * cjk);
        out.put(AsrLanguage.VI, 0.425f - 0.35f * cjk);
        out.put(AsrLanguage.EN, 0.425f - 0.35f * cjk);
        normalize(out);
        return out;
    }

    /** Cheap text evidence: CJK density (spec §16) + VI/EN lexicons (§17). */
    private Map<AsrLanguage, Float> textEvidence(String text) {
        Map<AsrLanguage, Float> out = new EnumMap<>(AsrLanguage.class);
        out.put(AsrLanguage.VI, 1.0f / 3);
        out.put(AsrLanguage.EN, 1.0f / 3);
        out.put(AsrLanguage.ZH, 1.0f / 3);
        if (text == null || text.trim().isEmpty()) return out;

        float cjk = cjkRatio(text);
        String lower = text.toLowerCase();
        int enHits = 0;
        for (String w : EN_WORDS) {
            if (lower.contains(w)) enHits++;
        }
        int viHits = 0;
        for (String w : VI_WORDS) {
            if (lower.contains(w)) viHits++;
        }
        float vi = 0.34f + 0.12f * Math.min(viHits, 3) - 0.10f * Math.min(enHits, 3);
        float en = 0.33f + 0.12f * Math.min(enHits, 3) - 0.10f * Math.min(viHits, 3);
        float zh = 0.33f + 0.60f * cjk - 0.05f * (Math.min(viHits + enHits, 3));
        out.put(AsrLanguage.VI, vi);
        out.put(AsrLanguage.EN, en);
        out.put(AsrLanguage.ZH, zh);
        // Clamp + normalize.
        for (AsrLanguage l : out.keySet()) out.put(l, Math.max(0.01f, out.get(l)));
        normalize(out);
        return out;
    }

    /**
     * Lexical fit of a transcript to one language in [0,1], for endpoint
     * verification re-ranking (Option A). Word-boundary matched (unlike the
     * substring scan in textEvidence) so short fragments don't false-hit.
     */
    public static float lexicalFit(String text, AsrLanguage lang) {
        if (text == null || text.trim().isEmpty() || lang == null) return 0f;
        Set<String> words = new HashSet<>();
        for (String w : text.toLowerCase(java.util.Locale.US).split("\\s+")) {
            if (!w.isEmpty()) words.add(w);
        }
        int enHits = 0;
        for (String w : EN_WORDS) {
            if (!w.contains(" ") && words.contains(w)) enHits++;
        }
        for (String w : EN_COMMON) {
            if (words.contains(w)) enHits++;
        }
        int viHits = 0;
        for (String w : VI_WORDS) {
            if (!w.contains(" ") && words.contains(w)) viHits++;
        }
        float cjk = cjkRatio(text);
        switch (lang) {
            case EN:
                return clamp01(0.20f + 0.30f * Math.min(enHits, 3)
                        - 0.15f * Math.min(viHits, 3));
            case VI:
                return clamp01(0.20f + 0.30f * Math.min(viHits, 3)
                        - 0.15f * Math.min(enHits, 3));
            case ZH:
                return clamp01(0.10f + 0.85f * cjk);
            default:
                return 0f;
        }
    }

    /** Fraction of CJK Unified Ideographs in the text (spec §16). */
    public static float cjkRatio(String text) {
        if (text == null || text.isEmpty()) return 0;
        int cjk = 0;
        int total = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isWhitespace(c)) continue;
            total++;
            if (isCjk(c)) {
                cjk++;
            }
        }
        return total == 0 ? 0 : (float) cjk / total;
    }

    /** Number of CJK Unified Ideograph characters. */
    public static int cjkCount(String text) {
        if (text == null || text.isEmpty()) return 0;
        int n = 0;
        for (int i = 0; i < text.length(); i++) {
            if (isCjk(text.charAt(i))) n++;
        }
        return n;
    }

    /**
     * Length gate for transcripts: whitespace tokens + CJK chars/2.
     * Chinese has no word spaces, so plain token counting would reject any
     * fluent ZH hypothesis (e.g. "然后我们开始" is one 'token').
     */
    public static int textUnits(String text) {
        if (text == null || text.trim().isEmpty()) return 0;
        return text.trim().split("\\s+").length + cjkCount(text) / 2;
    }

    private static boolean isCjk(char c) {
        return (c >= 0x4E00 && c <= 0x9FFF)
                || (c >= 0x3400 && c <= 0x4DBF)
                || (c >= 0x20000 && c <= 0x2A6DF);
    }

    private static void normalize(Map<AsrLanguage, Float> m) {
        float sum = 0;
        for (float v : m.values()) sum += v;
        if (sum <= 0) {
            for (AsrLanguage l : m.keySet()) m.put(l, 1.0f / m.size());
            return;
        }
        for (AsrLanguage l : m.keySet()) m.put(l, m.get(l) / sum);
    }

    private static float clamp01(float v) {
        return Math.max(0, Math.min(1, v));
    }

    /** Adaptive LID interval (spec OPTIONAL 6). */
    public static int lidIntervalMs(boolean uncertain) {
        return uncertain ? AsrState.LID_INTERVAL_UNCERTAIN_MS : AsrState.LID_INTERVAL_STABLE_MS;
    }
}
