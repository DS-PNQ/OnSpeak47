/*
 * OmniVoice — Rollback window manager for intra-utterance language switches.
 *
 * Spec §14-§15: on a suspected switch keep only the last 640–960 ms of audio
 * (never re-decode the whole utterance); the candidate model verifies on that
 * window (shadow decode) before anything is committed.
 * Pure-Java (no android.* imports).
 */
package com.omnivoice.onspeak47.asr;

public class RollbackManager {

    private int windowMs = AsrState.ROLLBACK_DEFAULT_MS;
    private long rollbackStartMs = -1;
    private float[] rollbackAudio;
    private AsrLanguage candidate;

    public RollbackManager() {}

    public RollbackManager(int windowMs) {
        setWindowMs(windowMs);
    }

    public void setWindowMs(int ms) {
        this.windowMs = Math.max(AsrState.ROLLBACK_MIN_MS,
                Math.min(AsrState.ROLLBACK_MAX_MS, ms));
    }

    public int windowMs() {
        return windowMs;
    }

    /** Capture the rollback window from the ring buffer (spec §14). */
    public float[] startRollback(AudioRingBuffer ring, AsrLanguage candidateLang) {
        return startRollback(ring, candidateLang, System.currentTimeMillis());
    }

    public float[] startRollback(AudioRingBuffer ring, AsrLanguage candidateLang, long nowMs) {
        this.candidate = candidateLang;
        this.rollbackStartMs = nowMs;
        this.rollbackAudio = ring.lastMs(windowMs);
        return rollbackAudio;
    }

    public float[] rollbackAudio() {
        return rollbackAudio;
    }

    public AsrLanguage candidate() {
        return candidate;
    }

    public long rollbackStartMs() {
        return rollbackStartMs;
    }

    public int rollbackSamples() {
        return rollbackAudio == null ? 0 : rollbackAudio.length;
    }

    /** Expected sample count for the configured window (16 kHz). */
    public int expectedSamples() {
        return (int) ((long) AsrState.SAMPLE_RATE * windowMs / 1000);
    }

    public void clear() {
        rollbackAudio = null;
        candidate = null;
        rollbackStartMs = -1;
    }

    public boolean hasPending() {
        return rollbackAudio != null && candidate != null;
    }
}
