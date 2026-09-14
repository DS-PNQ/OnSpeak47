/*
 * OmniVoice — Transcript stability tiers (spec §18).
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Transcript stability tiers (spec §18). */
public enum TranscriptTier {
    SPECULATIVE,
    STABLE,
    FINAL
}
