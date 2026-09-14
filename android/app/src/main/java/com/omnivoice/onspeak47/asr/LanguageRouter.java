/*
 * OmniVoice — VI/EN/ZH language router with hysteresis + shadow verification.
 *
 * Spec §10-§15, §21 MUST 5/6:
 *   - Never hard-switch on a single LID prediction.
 *   - candidate_score > 0.72 AND > active + 0.20 AND persists 200–300 ms.
 *   - Intra-utterance switch = rollback 640–960 ms + candidate shadow decode.
 *   - Inter-utterance switch (endpoint) = direct switch, no rollback.
 * Pure-Java (no android.* imports).
 */
package com.omnivoice.onspeak47.asr;

import java.util.Map;

public class LanguageRouter {

    /** Outcome of feeding one LID result into the router. */
    public enum Decision {
        /** Stay on the active language. */
        HOLD,
        /** Candidate observed but hysteresis not yet satisfied. */
        OBSERVE,
        /** Hysteresis satisfied — caller should run candidate verification. */
        START_ROLLBACK,
        /** Verification passed — active language changed. */
        COMMITTED,
        /** Verification failed — candidate discarded. */
        DISCARDED
    }

    /** Candidate verification outcome (shadow decode, spec §15). */
    public static class Verification {
        public final boolean supportsSwitch;
        public final String tokens;
        public final float confidence;

        public Verification(boolean supportsSwitch, String tokens, float confidence) {
            this.supportsSwitch = supportsSwitch;
            this.tokens = tokens == null ? "" : tokens;
            this.confidence = confidence;
        }
    }

    /** Shadow-decodes the rollback window on the candidate model. */
    public interface CandidateVerifier {
        Verification verify(AsrLanguage candidate, float[] rollbackAudio);
    }

    /** Time source (injectable for deterministic tests). */
    public interface Clock {
        long now();
    }

    private AsrLanguage active = AsrLanguage.UND;
    private float activeConfidence = 1.0f / 3;
    private AsrLanguage candidate;
    private long candidateSinceMs = -1;
    private RouterState state = RouterState.ACTIVE;

    private final float switchThreshold;
    private final float switchMargin;
    private final long persistMs;
    private final long discardCooldownMs;
    private final Clock clock;

    private CandidateVerifier verifier;
    private long lastSwitchMs = -1;
    private int switchCount = 0;
    private long lastDiscardMs = Long.MIN_VALUE / 2;
    private AsrLanguage lastDiscarded;

    public LanguageRouter() {
        this(AsrState.SWITCH_THRESHOLD, AsrState.SWITCH_MARGIN,
                AsrState.SWITCH_PERSIST_MS, System::currentTimeMillis);
    }

    public LanguageRouter(float switchThreshold, float switchMargin,
                          long persistMs, Clock clock) {
        this(switchThreshold, switchMargin, persistMs, clock, 1500L);
    }

    public LanguageRouter(float switchThreshold, float switchMargin,
                          long persistMs, Clock clock, long discardCooldownMs) {
        this.switchThreshold = switchThreshold;
        this.switchMargin = switchMargin;
        this.persistMs = persistMs;
        this.clock = clock;
        this.discardCooldownMs = discardCooldownMs;
    }

    public void setCandidateVerifier(CandidateVerifier verifier) {
        this.verifier = verifier;
    }

    public synchronized void setActive(AsrLanguage lang, float confidence) {
        this.active = lang == null ? AsrLanguage.UND : lang;
        this.activeConfidence = clamp01(confidence);
        clearCandidate();
        state = active == AsrLanguage.UND ? RouterState.UNKNOWN : RouterState.ACTIVE;
    }

    /**
     * fakedemo2 §22: bootstrap vs runtime paths are split. While no language
     * is committed the router holds UNKNOWN — runtime hysteresis must not
     * fire on bootstrap scores and vice versa.
     */
    public synchronized boolean isBootstrapping() {
        return active == AsrLanguage.UND;
    }

    /**
     * Commit the bootstrap decision (§22): first confident language for this
     * utterance. Never called with UND — use {@link #reset} to go back to
     * UNKNOWN between utterances.
     */
    public synchronized void commitBootstrap(AsrLanguage lang, float confidence) {
        if (lang == null || lang == AsrLanguage.UND) return;
        active = lang;
        activeConfidence = clamp01(confidence);
        clearCandidate();
        state = RouterState.ACTIVE;
    }

    /**
     * Bootstrap gate (§22, §42): confident only when top1 >= 0.70 AND
     * top1 - top2 >= 0.15. Anything else stays UNKNOWN — the pipeline then
     * extends the window (§9 cách 1) or runs ≤2 speculative candidates (§32),
     * never forcing VI on a bare max-probability.
     *
     * @return the committed language, or UND when still uncertain.
     */
    public synchronized AsrLanguage onBootstrapLidResult(LanguageIdEngine.LidResult lid) {
        if (lid == null || lid.language == null || lid.language == AsrLanguage.UND) {
            state = RouterState.BOOTSTRAPPING;
            return AsrLanguage.UND;
        }
        if (active != AsrLanguage.UND) return active; // already bootstrapped
        float top1 = lid.confidence;
        float top2 = 0f;
        if (lid.scores != null) {
            for (Map.Entry<AsrLanguage, Float> e : lid.scores.entrySet()) {
                if (e.getKey() == null || e.getKey() == lid.language
                        || e.getKey() == AsrLanguage.UND || e.getValue() == null) continue;
                top2 = Math.max(top2, e.getValue());
            }
        }
        if (top1 >= AsrState.BOOTSTRAP_THRESHOLD
                && (top1 - top2) >= AsrState.BOOTSTRAP_MARGIN) {
            commitBootstrap(lid.language, top1);
            return active;
        }
        state = RouterState.BOOTSTRAPPING;
        return AsrLanguage.UND;
    }

    public synchronized AsrLanguage active() {
        return active;
    }

    public synchronized RouterState state() {
        return state;
    }

    public synchronized AsrLanguage candidate() {
        return candidate;
    }

    /** EMA update of the active hypothesis from ASR token confidence. */
    public synchronized void onPartialConfidence(float tokenConf) {
        activeConfidence = AsrState.EMA_ALPHA * clamp01(tokenConf)
                + (1 - AsrState.EMA_ALPHA) * activeConfidence;
    }

    public synchronized float activeConfidence() {
        return activeConfidence;
    }

    /**
     * Feed one async LID result (spec §25 pseudocode).
     * Pure decision step — the caller performs the actual rollback decode and
     * then calls {@link #onVerification} with the outcome.
     */
    public synchronized Decision onLidResult(LanguageIdEngine.LidResult lid) {
        if (lid == null || lid.language == null) return Decision.HOLD;
        // Bootstrap owns the UND phase (§23): runtime hysteresis must not
        // vote while no active model exists. The pipeline routes bootstrap
        // scores through onBootstrapLidResult() instead.
        if (active == AsrLanguage.UND) return Decision.HOLD;
        AsrLanguage cand = lid.language;
        if (cand == active || cand == AsrLanguage.UND) {
            clearCandidate();
            state = RouterState.ACTIVE;
            return Decision.HOLD;
        }
        if (lid.confidence < switchThreshold) {
            clearCandidate();
            if (state == RouterState.CANDIDATE_SWITCH) state = RouterState.ACTIVE;
            return Decision.HOLD;
        }
        if (lid.confidence < activeConfidence + switchMargin) {
            return state == RouterState.CANDIDATE_SWITCH ? Decision.OBSERVE : Decision.HOLD;
        }
        long now = clock.now();
        if (candidate == null || candidate != cand) {
            // Cooldown after a failed verification: don't re-run a costly
            // shadow decode for the same rejected candidate immediately.
            if (cand == lastDiscarded && now - lastDiscardMs < discardCooldownMs) {
                return Decision.HOLD;
            }
            candidate = cand;
            candidateSinceMs = now;
            state = RouterState.CANDIDATE_SWITCH;
            return Decision.OBSERVE;
        }
        if (now - candidateSinceMs < persistMs) {
            state = RouterState.CANDIDATE_SWITCH;
            return Decision.OBSERVE;
        }
        state = RouterState.ROLLBACK;
        return Decision.START_ROLLBACK;
    }

    /**
     * Run shadow verification synchronously (convenience for the pipeline:
     * START_ROLLBACK -> verify -> COMMITTED/DISCARDED).
     */
    public synchronized Decision verifyAndCommit(float[] rollbackAudio) {
        if (state != RouterState.ROLLBACK || candidate == null) return Decision.HOLD;
        state = RouterState.VERIFY;
        Verification v = verifier == null
                ? new Verification(false, "", 0)
                : verifier.verify(candidate, rollbackAudio);
        return onVerification(v != null && v.supportsSwitch, v == null ? "" : v.tokens,
                v == null ? 0 : v.confidence);
    }

    /** Apply a verification outcome (for async candidate decodes). */
    public synchronized Decision onVerification(boolean supportsSwitch, String tokens, float conf) {
        if (state != RouterState.VERIFY && state != RouterState.ROLLBACK) return Decision.HOLD;
        if (supportsSwitch && candidate != null) {
            active = candidate;
            activeConfidence = clamp01(conf);
            lastSwitchMs = clock.now();
            switchCount++;
            state = RouterState.COMMIT;
            Decision d = Decision.COMMITTED;
            clearCandidate();
            state = RouterState.ACTIVE;
            return d;
        }
        lastDiscarded = candidate;
        lastDiscardMs = clock.now();
        clearCandidate();
        state = RouterState.ACTIVE;
        return Decision.DISCARDED;
    }

    /**
     * Inter-utterance switch at a VAD endpoint (spec §13.1): no rollback, the
     * next utterance simply starts on the LID winner when confident. The bar
     * is lower than the intra-utterance gate on purpose — a boundary switch
     * rewrites nothing and spends no rollback decode.
     */
    public synchronized boolean onEndpoint(Map<AsrLanguage, Float> scores) {
        if (scores == null || scores.isEmpty()) return false;
        AsrLanguage best = active;
        float bestScore = -1;
        for (Map.Entry<AsrLanguage, Float> e : scores.entrySet()) {
            if (e.getKey() == AsrLanguage.UND) continue;
            if (e.getValue() != null && e.getValue() > bestScore) {
                bestScore = e.getValue();
                best = e.getKey();
            }
        }
        Float activeScore = scores.get(active);
        float activeScoreV = activeScore == null ? 0 : activeScore;
        if (best != active && bestScore >= AsrState.ENDPOINT_THRESHOLD
                && bestScore > activeScoreV + AsrState.ENDPOINT_MARGIN) {
            active = best;
            activeConfidence = clamp01(bestScore);
            lastSwitchMs = clock.now();
            switchCount++;
            clearCandidate();
            state = RouterState.ACTIVE;
            return true;
        }
        clearCandidate();
        return false;
    }

    /**
     * Record a switch decided by endpoint candidate verification (the caller
     * already compared the candidate re-decode against the active
     * hypothesis). Commits and counts it like any other switch so telemetry
     * stays in one place.
     */
    public synchronized void commitEndpointSwitch(AsrLanguage lang, float confidence) {
        active = lang == null ? AsrLanguage.UND : lang;
        activeConfidence = clamp01(confidence);
        lastSwitchMs = clock.now();
        switchCount++;
        clearCandidate();
        state = RouterState.ACTIVE;
    }

    public synchronized void reset() {
        active = AsrLanguage.UND;
        activeConfidence = 1.0f / 3;
        clearCandidate();
        state = RouterState.UNKNOWN;
        switchCount = 0;
        lastSwitchMs = -1;
        lastDiscarded = null;
        lastDiscardMs = Long.MIN_VALUE / 2;
    }

    public synchronized int switchCount() {
        return switchCount;
    }

    public synchronized long lastSwitchMs() {
        return lastSwitchMs;
    }

    private void clearCandidate() {
        candidate = null;
        candidateSinceMs = -1;
    }

    private static float clamp01(float v) {
        return Math.max(0, Math.min(1, v));
    }
}
