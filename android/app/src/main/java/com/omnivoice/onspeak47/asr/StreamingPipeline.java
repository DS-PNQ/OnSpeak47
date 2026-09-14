/*
 * OmniVoice — Streaming pipeline glue (AudioRecord → VAD → Zipformer →
 * partial transcript → async LID → router → rollback → stable transcript).
 *
 * Implements spec §24 (frame pseudocode) + §25 (router pseudocode):
 *   - 20 ms frontend frames, 160 ms ASR scheduler (320 ms fallback).
 *   - LID every 400–600 ms on a background worker, never blocking ASR.
 *   - 1 active ASR normally; candidate ASR only inside the rollback window.
 */
package com.omnivoice.onspeak47.asr;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class StreamingPipeline implements AudioCapture.FrameListener {

    private static final String TAG = "StreamingPipeline";

    /** UI / downstream callbacks (always posted on the main thread). */
    public interface Listener {
        void onPartial(String stable, String speculative, String activeLang);
        void onFinal(String text, String activeLang);
        void onLanguageSwitch(String fromLang, String toLang);
        void onMetrics(String summaryJson);
    }

    private final Context appContext;
    private final AudioRingBuffer ring = AudioRingBuffer.withCapacityMs(30_000);
    private final VadEngine vad;
    private final ZipformerModelManager models;
    private final LanguageIdEngine lid;
    private final LanguageRouter router;
    private final RollbackManager rollback = new RollbackManager();
    private final PartialTranscriptManager transcripts = new PartialTranscriptManager();
    private final AsrMetrics metrics = new AsrMetrics();

    private final ExecutorService lidExecutor = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final List<Listener> listeners = new ArrayList<>();

    private AudioCapture capture;
    private StreamingAsrEngine activeAsr;
    private AsrLanguage activeLang = AsrLanguage.VI;

    private final float[] schedulerBuf = new float[AsrState.SCHEDULER_SAMPLES];
    private int schedulerFill = 0;
    private int schedulerTarget = AsrState.SCHEDULER_SAMPLES;

    private long lastLidMs = 0;
    private long utteranceStartMs = -1;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private volatile boolean adaptiveChunk = false;

    // Debug telemetry: frame/speech/decode counters + last partial snapshot,
    // logged every ~5 s while running (spec §28 troubleshooting).
    private long dbgFrames = 0;
    private long dbgSpeechFrames = 0;
    private int dbgDecodes = 0;
    private String dbgLastPartial = "";
    private float dbgLastConf = 0f;
    private float dbgLastVadProb = 0f;
    private boolean dbgFirstFrameLogged = false;
    private double dbgSpeechRmsSum = 0;
    private float dbgSpeechMaxProb = 0f;

    public StreamingPipeline(Context context, VadEngine vad, ZipformerModelManager models,
                             LanguageIdEngine lid, LanguageRouter router) {
        this.appContext = context.getApplicationContext();
        this.vad = vad;
        this.models = models;
        this.lid = lid;
        this.router = router;
        this.router.setCandidateVerifier(this::verifyCandidate);
        setActiveLanguage(AsrLanguage.VI, 1.0f / 3);
    }

    /** Convenience constructor with default engines. */
    public StreamingPipeline(Context context) {
        this(context, new VadEngine(context), new ZipformerModelManager(context),
                new LanguageIdEngine(), new LanguageRouter());
    }

    // --- Lifecycle ----------------------------------------------------

    public void addListener(Listener l) {
        synchronized (listeners) {
            listeners.add(l);
        }
    }

    public void removeListener(Listener l) {
        synchronized (listeners) {
            listeners.remove(l);
        }
    }

    public void start() {
        if (running.getAndSet(true)) return;
        models.preloadForDevice();
        setActiveLanguage(activeLang, 1.0f / 3);
        // Engines may be shared across sessions (see StreamingController):
        // drop any stale decoder state from the previous session.
        if (activeAsr != null) {
            try {
                activeAsr.reset();
            } catch (Exception e) {
                Log.w(TAG, "active engine reset failed", e);
            }
        }
        dbgFirstFrameLogged = false;
        capture = new AudioCapture(this);
        if (!capture.start()) {
            running.set(false);
            Log.e(TAG, "AudioCapture failed to start");
            return;
        }
        metrics.mark(AsrMetrics.AUDIO_RECEIVED);
        Log.i(TAG, "streaming pipeline started, active=" + activeLang
                + " endpoint=" + AsrState.ENDPOINT_SILENCE_MS + "ms"
                + " verifyMargin=" + AsrState.ENDPOINT_VERIFY_MARGIN
                + " sherpaEndpoint=off");
    }

    public void stop() {
        running.set(false);
        if (capture != null) {
            capture.stop();
            capture = null;
        }
        // Flush a final transcript for the trailing utterance (residue
        // included, so a deliberate Stop never drops tail words).
        flushScheduler();
        if (activeAsr != null) {
            StreamingAsrEngine.PartialResult fin = activeAsr.getFinalResult();
            transcripts.finalizeTranscript(mergeDisplay(fin.text));
            postFinal(transcripts.getCommitted(), activeLang.code);
        }
        metrics.addUtterance();
        postMetrics();
    }

    public void shutdown() {
        shutdown(true);
    }

    /**
     * @param closeModels false when the model manager is shared with the
     *                    caller (e.g. preloaded at app startup) and must stay
     *                    open after this pipeline goes away.
     */
    public void shutdown(boolean closeModels) {
        stop();
        lidExecutor.shutdownNow();
        if (closeModels) models.close();
        vad.close();
    }

    public AsrLanguage activeLanguage() {
        return activeLang;
    }

    public AsrMetrics metrics() {
        return metrics;
    }

    /** Test/seam hook: drive the pipeline without the microphone. */
    public void feedFrameForTest(short[] pcm16) {
        onFrame(pcm16, 0);
    }

    // --- AudioCapture.FrameListener (spec §24) --------------------------

    @Override
    public void onFrame(short[] pcm16, long frameIndex) {
        if (!running.get() && capture != null) return; // live mode gate
        float[] frame = toFloat(pcm16);
        ring.append(frame);
        long nowMs = System.currentTimeMillis();
        metrics.mark(AsrMetrics.AUDIO_RECEIVED, nowMs);

        VadEngine.VadResult v = vad.process(frame, nowMs);
        metrics.mark(AsrMetrics.VAD, System.currentTimeMillis());
        dbgFrames++;
        dbgLastVadProb = v.prob;
        float frameRms = rms(frame);
        if (!dbgFirstFrameLogged) {
            dbgFirstFrameLogged = true;
            Log.i(TAG, "first audio frame: n=" + frame.length
                    + " rms=" + String.format(java.util.Locale.US, "%.4f", frameRms)
                    + " vadNative=" + vad.isNative());
        }
        if (v.speechEnd) {
            onEndpoint();
            return;
        }
        if (!v.isSpeech) {
            if (dbgFrames % 100 == 0) logDbg();
            return; // silence suppression (spec §8)
        }
        dbgSpeechFrames++;
        dbgSpeechRmsSum += frameRms;
        if (v.prob > dbgSpeechMaxProb) dbgSpeechMaxProb = v.prob;
        if (utteranceStartMs < 0) utteranceStartMs = nowMs;

        // 160 ms scheduler accumulation.
        int copied = 0;
        while (copied < frame.length) {
            int room = schedulerTarget - schedulerFill;
            int n = Math.min(room, frame.length - copied);
            System.arraycopy(frame, copied, schedulerBuf, schedulerFill, n);
            schedulerFill += n;
            copied += n;
            if (schedulerFill >= schedulerTarget) {
                float[] chunk = new float[schedulerTarget];
                System.arraycopy(schedulerBuf, 0, chunk, 0, schedulerTarget);
                schedulerFill = 0;
                // Latency is measured from the START of the audio window, not
                // from when its last frame arrived (else ~160 ms goes missing).
                decodeChunk(chunk, nowMs - (long) schedulerTarget * 1000
                        / AsrState.SAMPLE_RATE);
            }
        }

        // Async LID at 400–600 ms cadence (spec §11), never blocking ASR.
        int lidInterval = LanguageIdEngine.lidIntervalMs(
                router.state() != RouterState.ACTIVE);
        if (nowMs - lastLidMs >= lidInterval) {
            lastLidMs = nowMs;
            final float[] window = ring.lastMs(AsrState.LID_WINDOW_MS);
            // Text evidence covers the same recent span as the 480 ms audio
            // window (last ~6 tokens). Scoring the whole committed transcript
            // would pin the evidence to the utterance's first language and
            // blind the router to intra-utterance switches.
            final String partial = recentText(6);
            final float conf = activeAsr == null ? 0.33f : activeAsr.getConfidence();
            final AsrLanguage active = activeLang;
            metrics.mark(AsrMetrics.LID_START);
            lidExecutor.submit(() -> {
                long t0 = System.currentTimeMillis();
                LanguageIdEngine.LidResult r = lid.classify(window, partial, conf, active);
                metrics.mark(AsrMetrics.LID_END);
                metrics.addRtf((System.currentTimeMillis() - t0) / (double) AsrState.LID_WINDOW_MS);
                onLidResult(r);
            });
        }

        router.onPartialConfidence(activeAsr == null ? 0.33f : activeAsr.getConfidence());
    }

    @Override
    public void onError(int code) {
        metrics.addDroppedFrames(1);
    }

    // --- Internals -----------------------------------------------------

    private void decodeChunk(float[] chunk, long audioTimeMs) {
        if (activeAsr == null) return;
        long t0 = System.currentTimeMillis();
        metrics.mark(AsrMetrics.ASR_START, t0);
        activeAsr.acceptAudio(chunk);
        // NEVER decode without the readiness gate: sherpa aborts the whole
        // process on an under-buffered decode (GetFrames OOB is fatal, not
        // an exception). See flushScheduler.
        if (activeAsr.isReadyToDecode()) {
            activeAsr.decodeAvailable();
        }
        StreamingAsrEngine.PartialResult p = activeAsr.getPartialResult();
        metrics.mark(AsrMetrics.ASR_END);
        transcripts.updateSpeculative(p.text);
        metrics.mark(AsrMetrics.PARTIAL_VISIBLE);
        metrics.addPartialLatency(System.currentTimeMillis() - audioTimeMs);
        double inferMs = System.currentTimeMillis() - t0;
        metrics.addRtf(inferMs / schedulerTarget * AsrState.SAMPLE_RATE / 1000.0);
        dbgDecodes++;
        dbgLastPartial = p.text == null ? "" : p.text;
        dbgLastConf = p.confidence;
        Log.d(TAG, "decode lang=" + activeLang.code + " conf=" + p.confidence
                + " text=" + truncate(dbgLastPartial, 80));
        if (dbgFrames % 100 == 0) logDbg();
        postPartial(transcripts.getCommitted(), transcripts.getSpeculative(), activeLang.code);
    }

    private void onLidResult(LanguageIdEngine.LidResult r) {
        LanguageRouter.Decision d = router.onLidResult(r);
        if (d == LanguageRouter.Decision.START_ROLLBACK) {
            metrics.mark(AsrMetrics.SWITCH_DETECTED);
            metrics.mark(AsrMetrics.ROLLBACK_START);
            float[] window = rollback.startRollback(ring, r.language);
            // Candidate shadow decode (spec §15): active keeps its state; the
            // candidate only sees the 640–960 ms window.
            LanguageRouter.Decision out = router.verifyAndCommit(window);
            metrics.mark(AsrMetrics.ROLLBACK_END);
            if (out == LanguageRouter.Decision.COMMITTED) {
                onCommitted(r.language, window);
            } else {
                transcripts.rollbackSpeculative();
            }
            metrics.mark(AsrMetrics.COMMIT);
        }
    }

    /** Shadow decode of the rollback window on the candidate model. */
    private LanguageRouter.Verification verifyCandidate(AsrLanguage candidate,
                                                        float[] rollbackAudio) {
        try {
            StreamingAsrEngine cand = models.get(candidate);
            // Fresh stream state for the window (never the active stream),
            // paced like live input.
            cand.reset();
            int step = AsrState.SCHEDULER_SAMPLES;
            for (int off = 0; off < rollbackAudio.length; off += step) {
                int n = Math.min(step, rollbackAudio.length - off);
                float[] slice = new float[n];
                System.arraycopy(rollbackAudio, off, slice, 0, n);
                cand.acceptAudio(slice);
                if (cand.isReadyToDecode()) cand.decodeAvailable();
            }
            StreamingAsrEngine.PartialResult r = cand.getPartialResult();
            String tokens = r.text == null ? "" : r.text.trim();
            // Heuristic gate: accept when the candidate actually decoded
            // content with reasonable confidence. A real deployment compares
            // per-model scores; this keeps the state machine honest without
            // over-switching on silence.
            boolean ok = !tokens.isEmpty() && r.confidence >= 0.35f;
            return new LanguageRouter.Verification(ok, tokens, r.confidence);
        } catch (Exception e) {
            Log.w(TAG, "candidate verify failed", e);
            return new LanguageRouter.Verification(false, "", 0);
        }
    }

    /**
     * Fold leftover scheduler audio into the stream, then pad trailing
     * silence until a full chunk is decodable (bounded) and decode once.
     * Every decode stays behind isReadyToDecode: sherpa ABORTS the process
     * on an under-buffered decode (its GetFrames has no clamp — a warning
     * followed by exit, not an exception), so an unconditional "flush
     * decode" is a startup-crash-grade bug, not an optimization.
     * The silence padding doubles as the right-context held-out tail words
     * need to emit, so they join THIS utterance instead of leaking (as
     * undecoded residue) into the next one.
     */
    private void flushScheduler() {
        if (activeAsr == null) {
            schedulerFill = 0;
            return;
        }
        if (schedulerFill > 0) {
            float[] tail = new float[schedulerFill];
            System.arraycopy(schedulerBuf, 0, tail, 0, schedulerFill);
            activeAsr.acceptAudio(tail);
        }
        schedulerFill = 0;
        int padded = 0;
        int cap = 960 * AsrState.SAMPLE_RATE / 1000;
        while (!activeAsr.isReadyToDecode() && padded < cap) {
            activeAsr.acceptAudio(new float[AsrState.SCHEDULER_SAMPLES]);
            padded += AsrState.SCHEDULER_SAMPLES;
        }
        if (activeAsr.isReadyToDecode()) {
            activeAsr.decodeAvailable();
            StreamingAsrEngine.PartialResult p = activeAsr.getPartialResult();
            transcripts.updateSpeculative(p.text);
            postPartial(transcripts.getCommitted(), transcripts.getSpeculative(),
                    activeLang.code);
        }
    }

    private void onCommitted(AsrLanguage next, float[] window) {
        AsrLanguage prev = activeLang;
        transcripts.rollbackSpeculative();
        StreamingAsrEngine cand = models.get(next);
        StreamingAsrEngine.PartialResult r = cand.getPartialResult();
        transcripts.commit(r.text);
        setActiveLanguage(next, r.confidence);
        // Re-feed the window into the new active stream so its incremental
        // state continues from the switch point (spec §14).
        activeAsr.acceptAudio(window);
        postSwitch(prev.code, next.code);
        postPartial(transcripts.getCommitted(), transcripts.getSpeculative(), next.code);
    }

    private void onEndpoint() {
        if (activeAsr == null) return;
        Log.d(TAG, "endpoint reached, active=" + activeLang.code);
        // Flush residue first so tail words join THIS utterance instead of
        // leaking into the next one (see flushScheduler).
        flushScheduler();
        StreamingAsrEngine.PartialResult fin = activeAsr.getFinalResult();
        String baseText = mergeDisplay(fin.text);
        // Endpoint candidate verification (spec §13.1): replay the utterance
        // through the non-active models. The text-only router cannot observe
        // other languages (single active decoder, no acoustic LID model), so
        // without this a whole utterance in another language would lock to
        // the wrong model forever. Two decodes per utterance — never 3
        // parallel decoders on the live path (spec §31).
        EndpointVerdict verdict = verifyEndpointLanguage(baseText, fin.confidence);
        if (verdict != null && verdict.lang != activeLang) {
            AsrLanguage prev = activeLang;
            if (!verdict.text.isEmpty()) baseText = verdict.text;
            setActiveLanguage(verdict.lang, verdict.confidence);
            router.commitEndpointSwitch(verdict.lang, verdict.confidence);
            postSwitch(prev.code, verdict.lang.code);
            Log.i(TAG, "endpoint verification switched " + prev.code
                    + " → " + verdict.lang.code);
        } else {
            // Confirmed-active or inconclusive: the text-router endpoint bar
            // stays as backstop for lexical cases the re-decode missed
            // (e.g. an unavailable candidate asset).
            if (verdict != null) router.setActive(activeLang, activeAsr.getConfidence());
            float[] window = ring.lastMs(AsrState.LID_WINDOW_MS);
            LanguageIdEngine.LidResult r = lid.classify(window, transcripts.getCommitted(),
                    activeAsr.getConfidence(), activeLang);
            if (router.onEndpoint(r.scores)) {
                setActiveLanguage(router.active(), r.confidence);
            }
        }
        transcripts.finalizeTranscript(baseText);
        postFinal(transcripts.getCommitted(), activeLang.code);
        metrics.addUtterance();
        // Fresh utterance state.
        transcripts.reset();
        activeAsr.reset();
        utteranceStartMs = -1;
        postMetrics();
    }

    /** Decided language for the finished utterance, or null if inconclusive. */
    private static class EndpointVerdict {
        AsrLanguage lang;
        String text;
        float confidence;
    }

    /**
     * Replay the utterance audio (up to ENDPOINT_VERIFY_MS) through each
     * non-active model and compare hypotheses on a COMBINED score
     * (confidence + lexical fit). Pure-confidence margins lock in
     * cross-lingual near-misses ("HELLO" via the VI model at 0.60), so the
     * lexical term breaks those ties. Returns the winner when it beats the
     * active hypothesis by ENDPOINT_VERIFY_MARGIN, the active language when
     * its hypothesis is solid and unbeaten, or null when nothing is
     * decidable.
     */
    private EndpointVerdict verifyEndpointLanguage(String baseText, float baseConf) {
        long nowMs = System.currentTimeMillis();
        long spanMs = utteranceStartMs < 0 ? AsrState.ENDPOINT_VERIFY_MS
                : Math.min(Math.max(nowMs - utteranceStartMs, 500),
                AsrState.ENDPOINT_VERIFY_MS);
        float[] audio;
        try {
            audio = ring.lastMs((int) spanMs);
        } catch (Exception e) {
            Log.w(TAG, "endpoint verify: no audio window", e);
            return null;
        }
        if (audio == null || audio.length < AsrState.SAMPLE_RATE / 2) return null;

        String cleanBase = baseText == null ? "" : baseText.trim();
        float confW = 1f - AsrState.ENDPOINT_VERIFY_LEX_W;
        float activeScore = confW * baseConf
                + AsrState.ENDPOINT_VERIFY_LEX_W
                * LanguageIdEngine.lexicalFit(cleanBase, activeLang);
        AsrLanguage best = null;
        float bestScore = 0;
        String bestText = "";
        float bestConf = 0;
        for (AsrLanguage lang : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
            if (lang == activeLang) continue;
            EndpointVerdict cand = scoreCandidate(lang, audio);
            if (cand == null || cand.text.isEmpty()) continue;
            float lexCand = LanguageIdEngine.lexicalFit(cand.text, lang);
            int minUnits = lexCand >= AsrState.ENDPOINT_VERIFY_STRONG_LEX
                    ? 1 : AsrState.ENDPOINT_VERIFY_MIN_TOKENS;
            if (LanguageIdEngine.textUnits(cand.text) < minUnits) continue;
            float candScore = confW * cand.confidence
                    + AsrState.ENDPOINT_VERIFY_LEX_W * lexCand;
            if (candScore >= activeScore + AsrState.ENDPOINT_VERIFY_MARGIN
                    && candScore > bestScore) {
                best = lang;
                bestScore = candScore;
                bestText = cand.text;
                bestConf = cand.confidence;
            }
        }
        if (best != null) {
            EndpointVerdict v = new EndpointVerdict();
            v.lang = best;
            v.text = bestText;
            v.confidence = bestConf;
            return v;
        }
        // Active unbeaten: confirmed only when its own hypothesis is solid,
        // otherwise inconclusive (let the text router have a say).
        if (LanguageIdEngine.textUnits(cleanBase) >= 2 && baseConf >= 0.5f) {
            EndpointVerdict v = new EndpointVerdict();
            v.lang = activeLang;
            v.text = cleanBase;
            v.confidence = baseConf;
            return v;
        }
        return null;
    }

    /**
     * Decode audio on a non-active engine without disturbing the active
     * stream. Feeds 160 ms slices exactly like live pacing so the hypothesis
     * covers the whole window (a bulk feed + capped drain would only decode
     * its head and truncate switched transcripts).
     */
    private EndpointVerdict scoreCandidate(AsrLanguage lang, float[] audio) {
        try {
            StreamingAsrEngine cand = models.get(lang);
            cand.reset();
            int step = AsrState.SCHEDULER_SAMPLES;
            for (int off = 0; off < audio.length; off += step) {
                int n = Math.min(step, audio.length - off);
                float[] slice = new float[n];
                System.arraycopy(audio, off, slice, 0, n);
                cand.acceptAudio(slice);
                if (cand.isReadyToDecode()) cand.decodeAvailable();
            }
            for (int i = 0; i < 4 && cand.isReadyToDecode(); i++) {
                cand.decodeAvailable();
            }
            StreamingAsrEngine.PartialResult r = cand.getPartialResult();
            EndpointVerdict v = new EndpointVerdict();
            v.lang = lang;
            v.text = r.text == null ? "" : r.text.trim();
            v.confidence = r.confidence;
            return v;
        } catch (Exception e) {
            Log.w(TAG, "endpoint verify failed for " + lang, e);
            return null;
        }
    }

    /** Last {@code maxTokens} of the display string (LID text window). */
    private String recentText(int maxTokens) {
        String display = transcripts.getDisplay();
        if (display == null || display.isEmpty()) return "";
        String[] toks = display.split("\\s+");
        if (toks.length <= maxTokens) return display;
        StringBuilder sb = new StringBuilder();
        for (int i = toks.length - maxTokens; i < toks.length; i++) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(toks[i]);
        }
        return sb.toString();
    }

    private String mergeDisplay(String fresh) {
        String committed = transcripts.getCommitted();
        if (fresh == null || fresh.trim().isEmpty()) return committed;
        String f = fresh.trim();
        if (!committed.isEmpty() && f.startsWith(committed)) return f;
        return committed.isEmpty() ? f : committed + " " + f;
    }

    private void setActiveLanguage(AsrLanguage lang, float conf) {
        activeLang = lang;
        activeAsr = models.get(lang);
        router.setActive(lang, conf);
    }

    /** OPTIONAL 5: widen the scheduler under high uncertainty. */
    public void setAdaptiveChunk(boolean highUncertainty) {
        adaptiveChunk = highUncertainty;
        schedulerTarget = highUncertainty
                ? AsrState.SAMPLE_RATE * AsrState.SCHEDULER_FALLBACK_MS / 1000
                : AsrState.SCHEDULER_SAMPLES;
    }

    private static float[] toFloat(short[] pcm) {
        float[] out = new float[pcm.length];
        for (int i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768.0f;
        return out;
    }

    private static float rms(float[] s) {
        if (s == null || s.length == 0) return 0;
        double sum = 0;
        for (float v : s) sum += v * v;
        return (float) Math.sqrt(sum / s.length);
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "…";
    }

    private void logDbg() {
        double rmsAvg = dbgSpeechFrames == 0 ? 0 : dbgSpeechRmsSum / dbgSpeechFrames;
        Log.i(TAG, "audioDBG frames=" + dbgFrames + " speech=" + dbgSpeechFrames
                + " decodes=" + dbgDecodes + " vadProb=" + dbgLastVadProb
                + " speechRmsAvg=" + String.format(java.util.Locale.US, "%.4f", rmsAvg)
                + " speechMaxVad=" + dbgSpeechMaxProb
                + " conf=" + dbgLastConf + " lastPartial=" + truncate(dbgLastPartial, 60));
        dbgSpeechRmsSum = 0;
        dbgSpeechMaxProb = 0f;
    }

    // --- Listener fan-out (main thread) ---------------------------------

    private void postPartial(String stable, String spec, String lang) {
        main.post(() -> {
            synchronized (listeners) {
                for (Listener l : listeners) l.onPartial(stable, spec, lang);
            }
        });
    }

    private void postFinal(String text, String lang) {
        main.post(() -> {
            synchronized (listeners) {
                for (Listener l : listeners) l.onFinal(text, lang);
            }
        });
    }

    private void postSwitch(String from, String to) {
        main.post(() -> {
            synchronized (listeners) {
                for (Listener l : listeners) l.onLanguageSwitch(from, to);
            }
        });
    }

    private void postMetrics() {
        final String json = metrics.summaryJson();
        main.post(() -> {
            synchronized (listeners) {
                for (Listener l : listeners) l.onMetrics(json);
            }
        });
    }
}
