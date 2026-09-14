/*
 * OmniVoice — ASR telemetry: event timestamps + latency percentiles.
 *
 * Spec §26-§28: record P50/P90/P95/P99, RTF, dropped frames; per-event
 * timestamps allow partial_latency and switch_latency computation.
 * Pure-Java (no android.* imports).
 */
package com.omnivoice.onspeak47.asr;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class AsrMetrics {

    // Event keys (spec §28).
    public static final String AUDIO_RECEIVED = "timestamp_audio_received";
    public static final String VAD = "timestamp_vad";
    public static final String ASR_START = "timestamp_asr_start";
    public static final String ASR_END = "timestamp_asr_end";
    public static final String PARTIAL_VISIBLE = "timestamp_partial_visible";
    public static final String LID_START = "timestamp_lid_start";
    public static final String LID_END = "timestamp_lid_end";
    public static final String SWITCH_DETECTED = "timestamp_switch_detected";
    public static final String ROLLBACK_START = "timestamp_rollback_start";
    public static final String ROLLBACK_END = "timestamp_rollback_end";
    public static final String COMMIT = "timestamp_commit";

    private final Map<String, Long> events = new LinkedHashMap<>();
    private final List<Long> partialLatencies = new ArrayList<>();
    private final List<Long> switchLatencies = new ArrayList<>();
    private final List<Double> rtfSamples = new ArrayList<>();
    private long droppedFrames = 0;
    private long utterances = 0;

    public synchronized void mark(String event) {
        events.put(event, System.currentTimeMillis());
    }

    public synchronized void mark(String event, long atMs) {
        events.put(event, atMs);
    }

    public synchronized Long get(String event) {
        return events.get(event);
    }

    /** partial_latency = partial_visible - speech_audio_time (spec §28). */
    public synchronized void addPartialLatency(long ms) {
        partialLatencies.add(ms);
    }

    public synchronized void addSwitchLatency(long ms) {
        switchLatencies.add(ms);
    }

    public synchronized void addRtf(double rtf) {
        rtfSamples.add(rtf);
    }

    public synchronized void addDroppedFrames(long n) {
        droppedFrames += n;
    }

    public synchronized void addUtterance() {
        utterances++;
    }

    public synchronized long droppedFrames() {
        return droppedFrames;
    }

    // --- Percentiles -------------------------------------------------

    public static double percentile(List<Long> sortedAsc, double p) {
        if (sortedAsc == null || sortedAsc.isEmpty()) return Double.NaN;
        ArrayList<Long> s = new ArrayList<>(sortedAsc);
        Collections.sort(s);
        double rank = p / 100.0 * (s.size() - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) return s.get(lo);
        return s.get(lo) + (rank - lo) * (s.get(hi) - s.get(lo));
    }

    public synchronized double partialP50() { return percentile(partialLatencies, 50); }
    public synchronized double partialP90() { return percentile(partialLatencies, 90); }
    public synchronized double partialP95() { return percentile(partialLatencies, 95); }
    public synchronized double partialP99() { return percentile(partialLatencies, 99); }

    public synchronized double meanRtf() {
        if (rtfSamples.isEmpty()) return Double.NaN;
        double sum = 0;
        for (double v : rtfSamples) sum += v;
        return sum / rtfSamples.size();
    }

    /** Targets from spec §19: P50 < 350 ms, P95 < 500 ms. */
    public synchronized boolean meetsLatencyTargets() {
        double p50 = partialP50();
        double p95 = partialP95();
        if (Double.isNaN(p50) || Double.isNaN(p95)) return false;
        return p50 < AsrState.LATENCY_P50_TARGET_MS && p95 < AsrState.LATENCY_P95_TARGET_MS;
    }

    public synchronized String summaryJson() {
        return "{"
                + "\"partial_p50\":" + fmt(partialP50())
                + ",\"partial_p90\":" + fmt(partialP90())
                + ",\"partial_p95\":" + fmt(partialP95())
                + ",\"partial_p99\":" + fmt(partialP99())
                + ",\"mean_rtf\":" + fmt(meanRtf())
                + ",\"dropped_frames\":" + droppedFrames
                + ",\"utterances\":" + utterances
                + ",\"meets_targets\":" + meetsLatencyTargets()
                + "}";
    }

    /** NaN (no samples yet) is not valid JSON — emit -1 instead. */
    private static String fmt(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return "-1";
        return Double.toString(v);
    }

    public synchronized void reset() {
        events.clear();
        partialLatencies.clear();
        switchLatencies.clear();
        rtfSamples.clear();
        droppedFrames = 0;
        utterances = 0;
    }

    public synchronized int partialCount() {
        return partialLatencies.size();
    }
}
