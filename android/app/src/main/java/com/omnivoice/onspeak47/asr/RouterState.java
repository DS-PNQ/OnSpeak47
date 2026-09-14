/*
 * OmniVoice — Router state machine states (spec §21 MUST 5).
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Router state machine states (spec §21 MUST 5). */
public enum RouterState {
    ACTIVE,
    CANDIDATE_SWITCH,
    ROLLBACK,
    VERIFY,
    COMMIT
}
