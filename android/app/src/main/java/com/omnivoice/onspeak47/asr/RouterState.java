/*
 * OmniVoice — Router state machine states (spec §21 MUST 5).
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Router state machine states (spec §21 MUST 5 + fakedemo2 §6). */
public enum RouterState {
    /** No language decided yet — utterance has not bootstrapped (§6 UNKNOWN). */
    UNKNOWN,
    /** Bootstrap acoustic LID in flight (200–300 ms window, §7). */
    BOOTSTRAPPING,
    ACTIVE,
    CANDIDATE_SWITCH,
    ROLLBACK,
    VERIFY,
    COMMIT
}
