/*
 * OmniVoice — Partial transcript policy (SPECULATIVE / STABLE / FINAL).
 *
 * Spec §18 + OPTIONAL 8: a token becomes STABLE only after surviving
 * TOKEN_STABLE_UPDATES consecutive incremental updates. Rollback drops the
 * speculative tail without touching committed text.
 * Pure-Java (no android.* imports).
 */
package com.omnivoice.onspeak47.asr;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PartialTranscriptManager {

    private final StringBuilder committed = new StringBuilder();
    private String speculative = "";
    /** Consecutive-update survival counter per speculative token. */
    private final Map<String, Integer> survival = new HashMap<>();
    private String lastSpeculative = "";
    private boolean finalized = false;

    public synchronized void updateSpeculative(String partial) {
        if (finalized) return;
        String next = partial == null ? "" : partial.trim();
        // Transducer partials re-emit the whole hypothesis including already
        // committed words — strip that prefix or it would be committed twice.
        String committedStr = committed.toString();
        if (!committedStr.isEmpty() && next.startsWith(committedStr)) {
            next = next.substring(committedStr.length()).trim();
        }
        if (next.equals(lastSpeculative)) {
            // Same surface form again: age all current tokens.
            for (Map.Entry<String, Integer> e : survival.entrySet()) {
                e.setValue(e.getValue() + 1);
            }
        } else {
            // New update: keep counters for tokens still present, reset others.
            Map<String, Integer> kept = new HashMap<>();
            for (String tok : next.split("\\s+")) {
                if (tok.isEmpty()) continue;
                kept.put(tok, survival.getOrDefault(tok, 0) + 1);
            }
            survival.clear();
            survival.putAll(kept);
            speculative = next;
            lastSpeculative = next;
        }
        // Promote tokens surviving N updates into the committed prefix when
        // they form a stable head of the speculative string.
        promoteStableHead();
    }

    private void promoteStableHead() {
        if (speculative.isEmpty()) return;
        String[] toks = speculative.split("\\s+");
        int stableCount = 0;
        for (String tok : toks) {
            if (survival.getOrDefault(tok, 0) >= AsrState.TOKEN_STABLE_UPDATES) stableCount++;
            else break;
        }
        if (stableCount == 0) return;
        StringBuilder head = new StringBuilder();
        for (int i = 0; i < stableCount; i++) {
            if (head.length() > 0) head.append(' ');
            head.append(toks[i]);
        }
        String headStr = head.toString();
        String committedStr = committed.toString();
        if (headStr.startsWith(committedStr)) {
            String delta = headStr.substring(committedStr.length()).trim();
            if (!delta.isEmpty()) {
                if (committed.length() > 0) committed.append(' ');
                committed.append(delta);
                // Remove promoted tokens from speculation + survival.
                String rest;
                if (speculative.length() > headStr.length()) {
                    rest = speculative.substring(headStr.length()).trim();
                } else {
                    rest = "";
                }
                speculative = rest;
                lastSpeculative = rest;
                for (String tok : delta.split("\\s+")) survival.remove(tok);
            }
        }
    }

    /** Drop the speculative tail (router rejected / rollback path). */
    public synchronized void rollbackSpeculative() {
        speculative = "";
        lastSpeculative = "";
        survival.clear();
    }

    /** Commit verified tokens (e.g. after candidate verification). */
    public synchronized void commit(String tokens) {
        if (tokens == null || tokens.trim().isEmpty()) return;
        String t = tokens.trim();
        // Avoid double-committing an already-committed prefix.
        String c = committed.toString();
        if (t.startsWith(c) && !c.isEmpty()) {
            t = t.substring(c.length()).trim();
            if (t.isEmpty()) return;
        }
        if (committed.length() > 0) committed.append(' ');
        committed.append(t);
        // Committed text is no longer speculative.
        if (speculative.startsWith(t)) {
            speculative = speculative.substring(t.length()).trim();
            lastSpeculative = speculative;
        } else {
            rollbackSpeculative();
        }
    }

    public synchronized void finalizeTranscript(String finalText) {
        committed.setLength(0);
        if (finalText != null) committed.append(finalText.trim());
        rollbackSpeculative();
        finalized = true;
    }

    public synchronized void reset() {
        committed.setLength(0);
        rollbackSpeculative();
        finalized = false;
    }

    public synchronized String getCommitted() {
        return committed.toString();
    }

    public synchronized String getSpeculative() {
        return speculative;
    }

    /** UI display string: stable part + speculative tail. */
    public synchronized String getDisplay() {
        String c = committed.toString();
        if (speculative.isEmpty()) return c;
        return c.isEmpty() ? speculative : c + " " + speculative;
    }

    public synchronized TranscriptTier tier() {
        if (finalized) return TranscriptTier.FINAL;
        if (!speculative.isEmpty()) return TranscriptTier.SPECULATIVE;
        return TranscriptTier.STABLE;
    }

    public synchronized List<String> speculativeTokens() {
        if (speculative.isEmpty()) return Collections.emptyList();
        List<String> out = new ArrayList<>();
        for (String t : speculative.split("\\s+")) {
            if (!t.isEmpty()) out.add(t);
        }
        return out;
    }
}
