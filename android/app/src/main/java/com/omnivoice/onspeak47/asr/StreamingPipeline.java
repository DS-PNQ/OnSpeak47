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
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

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
    // Volatile: written on the LID worker (activate) / audio thread, read on
    // both. fakedemo2 §6, §12: never start an utterance as VI — start UNKNOWN
    // and let bootstrap acoustic LID pick the first model. activeAsr stays
    // null until then (never models.get(UND)).
    private volatile StreamingAsrEngine activeAsr;
    private volatile AsrLanguage activeLang = AsrLanguage.UND;

    private final float[] schedulerBuf = new float[AsrState.SCHEDULER_SAMPLES];
    private int schedulerFill = 0;
    private int schedulerTarget = AsrState.SCHEDULER_SAMPLES;

    private long lastLidMs = 0;
    private long utteranceStartMs = -1;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private volatile boolean adaptiveChunk = false;

    // --- Bootstrap state (fakedemo2 §7, §24, §27) ---
    /** Monotonic utterance id: every async LID result carries the id it was
     *  requested for and is dropped when it no longer matches (§27). */
    private final AtomicLong utteranceSeq = new AtomicLong(0);
    /** Speech audio (ms) accumulated since the current utterance started. */
    private volatile long utteranceSpeechMs = 0;
    /** Ring write position at the utterance's first speech frame (§15). */
    private volatile long utteranceStartSample = -1;
    /** True while one bootstrap classify task is in flight (single-thread
     *  executor, but the flag also gates re-submit from the audio thread). */
    private volatile boolean bootstrapPending = false;
    /** utteranceSpeechMs at the last bootstrap attempt (extend-window gate). */
    private long lastBootstrapSpeechMs = -1;
    /** utteranceSpeechMs at the last dual-candidate run (decode-storm gate:
     *  a candidate pass costs 2 decodes — rerun at most every 800 ms). */
    private volatile long lastCandidateRunSpeechMs = -1;
    /** Last speculative pair tried (rotation covers the 3rd language). */
    private volatile AsrLanguage[] lastCandidatePair = null;
    /**
     * Confidence of the decision that activated the current language. While
     * it stays below BOOTSTRAP_THRESHOLD the acoustic referee keeps running
     * (§6.1 Referee Mode) so a weak early commit can still be corrected; a
     * solid commit turns the referee off and only the cheap 480 ms runtime
     * LID remains.
     */
    private volatile float bootstrapCommitConfidence = 0f;
    /** Bootstrap/referee classifies submitted this utterance (§6.1, CPU bound). */
    private volatile int refereeAttempts = 0;
    /** utteranceSpeechMs at the last referee submit (speech-gated cadence). */
    private volatile long lastRefereeSpeechMs = -1;
    /** Latest bootstrap LidResult for this utterance (candidate retries reuse
     *  its ranking after the ECAPA budget is spent — never a fresh guess). */
    private volatile LanguageIdEngine.LidResult lastBootstrapResult = null;

    /**
     * Serializes ALL streaming-engine calls (acceptAudio / decodeAvailable /
     * getPartialResult / reset). OnlineStream is not thread-safe, but the
     * audio thread (live decode + provisional) and the LID worker (bootstrap
     * replay, candidate shadow decodes, endpoint verification) share the
     * resident engines — unsynchronized access crashed the process in
     * GetFrames (fatal native abort, uncatchable). Always a LEAF lock: never
     * acquire `this` or the router monitor while holding it.
     */
    private final Object engineLock = new Object();

    // --- Provisional speculative decode (Option A hotfix) ---
    // While UNKNOWN the pipeline ALSO decodes continuously so partials flow
    // after ~160 ms like the old system. It starts on the always-resident VI
    // engine, but uncertain LID evidence steers it (2026-09-16 §6.2): a
    // persistent lean to a resident engine moves the stream there, a lean to
    // a non-resident engine quiets the UI instead of showing VI-decoded
    // guesses for foreign audio. Provisional output is SPECULATIVE-only: it
    // lives in provisionalSpeculative (never the shared transcript manager),
    // is adopted without replay when bootstrap confirms the same language on
    // a complete stream, and is discarded without a trace otherwise (§11:
    // nothing committed before the language is confirmed).
    private volatile StreamingAsrEngine provisionalAsr;
    private volatile AsrLanguage provisionalLang = AsrLanguage.VI;
    private volatile String provisionalSpeculative = "";
    private volatile float provisionalConf = 0.33f;
    /** Set while foreign-leaning evidence hides VI-guess text (decode runs on). */
    private volatile boolean provisionalQuiet = false;
    /** False after a mid-utterance engine switch: the stream missed the head
     *  audio, so a commit to this language must replay (no adopt). */
    private volatile boolean provisionalComplete = true;
    /** Worker→audio handoff for a provisional engine switch (never lazy-load
     *  on the worker: the audio thread resolves it). */
    private volatile AsrLanguage pendingProvisionalLang = null;
    /** Top language of the previous uncertain result (switch needs 2 hops). */
    private volatile AsrLanguage lastUncertainTop = null;
    /** Last provisional text already posted to UI (dedup: the transducer
     *  re-emits the same hypothesis every 160 ms — re-posting it only
     *  churns the UI without new information). The decode itself still
     *  runs (adopt fast-path needs the live stream state); only the UI
     *  post is gated. */
    private volatile String lastPostedProvisional = "";

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
        // Start UNKNOWN (§12.1): no active model until bootstrap acoustic LID
        // commits a language. Do NOT setActiveLanguage(VI) here.
        this.activeLang = AsrLanguage.UND;
        this.activeAsr = null;
        this.router.reset();
    }

    /** Convenience constructor with default engines. */
    public StreamingPipeline(Context context) {
        this(context, new VadEngine(context), new ZipformerModelManager(context),
                defaultLidEngine(context), new LanguageRouter());
    }

    /**
     * Default LID engine: VoxLingua107 ECAPA when its ONNX asset is bundled,
     * otherwise the interim audio-derived heuristic (§16).
     *
     * VoxLingua loads lazily off-thread (~85 MB) and reports flat/uncertain
     * until ready, so construction never blocks and the pipeline falls
     * through to extend-window / dual-candidate in the meantime. The old
     * {@code new LanguageIdEngine()} with a null scorer fell back to a
     * transcript heuristic — the circular VI lock (§3–§4) — and must not
     * come back.
     */
    public static LanguageIdEngine defaultLidEngine(Context context) {
        LanguageIdEngine lid = new LanguageIdEngine();
        VoxLinguaAcousticLidEngine vox = null;
        try {
            vox = VoxLinguaAcousticLidEngine.createIfAvailable(context);
        } catch (Exception e) {
            Log.w(TAG, "voxlingua probe failed, heuristic fallback: " + e.getMessage());
        }
        if (vox != null) {
            lid.setAcousticLidEngine(vox);
            Log.i(TAG, "LID: VoxLingua107 ECAPA (loading in background)");
        } else {
            lid.setAcousticLidEngine(new AcousticLidEngine.HeuristicAcousticLidEngine());
            Log.i(TAG, "LID: heuristic fallback (voxlingua asset absent)");
        }
        return lid;
    }

    /** Legacy no-context path (unit tests): heuristic fallback. */
    public static LanguageIdEngine defaultLidEngine() {
        LanguageIdEngine lid = new LanguageIdEngine();
        lid.setAcousticLidEngine(new AcousticLidEngine.HeuristicAcousticLidEngine());
        return lid;
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
        // Fresh session starts UNKNOWN (§12.1, §40 Phase 1): ASR must not
        // run while activeLang == UND; bootstrap picks the model per
        // utterance. Stale decoder state from a previous session is dropped
        // by resetting the resident engines lazily on activate.
        activeLang = AsrLanguage.UND;
        activeAsr = null;
        router.reset();
        transcripts.reset();
        try {
            lid.reset();
        } catch (Exception ignored) {
        }
        schedulerFill = 0;
        utteranceStartMs = -1;
        lastLidMs = 0;
        utteranceSpeechMs = 0;
        lastBootstrapSpeechMs = -1;
        bootstrapCommitConfidence = 0f;
        refereeAttempts = 0;
        lastRefereeSpeechMs = -1;
        lastBootstrapResult = null;
        lastCandidateRunSpeechMs = -1;
        lastCandidatePair = null;
        bootstrapPending = false;
        utteranceSeq.incrementAndGet();
        resetProvisional();
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
        if (activeAsr != null) {
            flushScheduler();
            final StreamingAsrEngine eng = activeAsr;
            final StreamingAsrEngine.PartialResult fin;
            synchronized (engineLock) {
                fin = eng.getFinalResult();
            }
            transcripts.finalizeTranscript(mergeDisplay(fin.text));
            postFinal(transcripts.getCommitted(), activeLang.code);
            metrics.addUtterance();
        } else if (utteranceStartMs >= 0 && utteranceSpeechMs > 0) {
            // Stopped mid-bootstrap: recover via the endpoint path (it folds
            // the scheduler residue into the provisional engine itself, so
            // flushScheduler must NOT run first here). Counts + resets itself.
            finishUnbootstrappedEndpoint();
            return;
        } else {
            metrics.addUtterance();
        }
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
        if (utteranceStartMs < 0) {
            utteranceStartMs = nowMs;
            // The frame was already appended above: rewind to its first
            // sample so replay starts at the true speech onset.
            utteranceStartSample = ring.totalWritten() - frame.length;
        }
        utteranceSpeechMs += AsrState.FRAME_MS;

        // --- Bootstrap phase (Option A): provisional speculative decode keeps
        // partials flowing after ~160 ms while bootstrap acoustic LID decides
        // the language in parallel. Provisional text is SPECULATIVE-only and
        // is adopted (same language on a complete stream, no replay) or
        // discarded (other language) on commit — never committed itself (§11).
        if (activeLang == AsrLanguage.UND) {
            applyPendingProvisionalSwitch();
            tryBootstrapLid(nowMs);
            accumulateScheduler(frame, nowMs, true);
            return;
        }

        // 160 ms scheduler accumulation on the active stream.
        accumulateScheduler(frame, nowMs, false);

        // Referee mode (§6.1): after a WEAK bootstrap commit the acoustic LID
        // keeps re-evaluating the utterance (tryBootstrapLid owns the cadence
        // and the per-utterance attempt cap) instead of going silent. A
        // confident disagreement is handed to the runtime hysteresis/rollback
        // gate (see refereeOnActive) — never a hard switch. Solid commits skip
        // this, so ECAPA costs nothing then.
        if (bootstrapCommitConfidence < AsrState.BOOTSTRAP_THRESHOLD) {
            tryBootstrapLid(nowMs);
        }

        // Async ECAPA referee (§22), never blocking ASR: stable 600 ms,
        // uncertain 400 ms, observed candidate 250 ms. ASR still ticks every
        // 160 ms — LID only votes.
        int lidInterval = LanguageIdEngine.lidIntervalMs(
                router.state() != RouterState.ACTIVE,
                router.candidate() != null);
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
            final long uid = utteranceSeq.get();
            metrics.mark(AsrMetrics.LID_START);
            lidExecutor.submit(() -> {
                long t0 = System.currentTimeMillis();
                LanguageIdEngine.LidResult r = lid.classify(window, partial, conf, active);
                metrics.mark(AsrMetrics.LID_END);
                metrics.addRtf((System.currentTimeMillis() - t0) / (double) AsrState.LID_WINDOW_MS);
                onLidResult(r, uid);
            });
        }

        router.onPartialConfidence(activeAsr == null ? 0.33f : activeAsr.getConfidence());
    }

    @Override
    public void onError(int code) {
        metrics.addDroppedFrames(1);
    }

    // --- Internals -----------------------------------------------------

    /**
     * Shared 160 ms scheduler accumulation. While UNKNOWN ({@code provisional}
     * true) chunks go to the provisional decoder; otherwise to the active
     * stream. The residue stays in schedulerBuf in both cases, so adopting
     * the provisional engine keeps a continuous stream with no duplication.
     */
    private void accumulateScheduler(float[] frame, long nowMs, boolean provisional) {
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
                long audioTimeMs = nowMs - (long) schedulerTarget * 1000
                        / AsrState.SAMPLE_RATE;
                if (provisional) decodeProvisional(chunk, audioTimeMs);
                else decodeChunk(chunk, audioTimeMs);
            }
        }
    }

    /**
     * Provisional speculative decode (Option A): continuous partials while
     * UNKNOWN. The decode MUST keep running (the adopt fast-path needs the
     * live stream state), but UI posts are speculative-only (lang=UND, never
     * committed), de-duplicated, and gated by {@link #provisionalQuiet}:
     * foreign-leaning evidence hides VI-guess text instead of rendering it.
     * Text is held outside the shared transcript manager so a later commit
     * to another language discards it without a trace.
     *
     * 2026-09-16 §6.2: the stream starts on the always-resident VI engine
     * but uncertain LID evidence steers it (see
     * {@link #applyProvisionalPolicy}) — never a hard-coded VI lock.
     * All engine calls are serialized on {@link #engineLock} (GetFrames race).
     */
    private void decodeProvisional(float[] chunk, long audioTimeMs) {
        if (provisionalAsr == null) {
            try {
                provisionalAsr = models.get(AsrLanguage.VI);
                synchronized (engineLock) {
                    provisionalAsr.reset();
                }
                provisionalLang = AsrLanguage.VI;
                provisionalComplete = true;
            } catch (Exception e) {
                Log.w(TAG, "provisional engine unavailable", e);
                return;
            }
        }
        final StreamingAsrEngine eng = provisionalAsr;
        final StreamingAsrEngine.PartialResult p;
        long t0 = System.currentTimeMillis();
        metrics.mark(AsrMetrics.ASR_START, t0);
        synchronized (engineLock) {
            eng.acceptAudio(chunk);
            // Readiness gate applies here too (sherpa aborts on under-buffered
            // decode — fatal, not an exception).
            if (eng.isReadyToDecode()) {
                eng.decodeAvailable();
            }
            p = eng.getPartialResult();
        }
        metrics.mark(AsrMetrics.ASR_END);
        metrics.mark(AsrMetrics.PARTIAL_VISIBLE);
        metrics.addPartialLatency(System.currentTimeMillis() - audioTimeMs);
        double inferMs = System.currentTimeMillis() - t0;
        metrics.addRtf(inferMs / schedulerTarget * AsrState.SAMPLE_RATE / 1000.0);
        dbgDecodes++;
        dbgLastPartial = p.text == null ? "" : p.text;
        dbgLastConf = p.confidence;
        provisionalSpeculative = dbgLastPartial;
        provisionalConf = p.confidence;
        // The decode keeps running even while quiet (adopt needs the live
        // stream); only the UI post is gated.
        if (!dbgLastPartial.isEmpty() && !dbgLastPartial.equals(lastPostedProvisional)
                && !provisionalQuiet) {
            lastPostedProvisional = dbgLastPartial;
            Log.d(TAG, "provisional lang=" + provisionalLang.code + " conf=" + p.confidence
                    + " text=" + truncate(dbgLastPartial, 80));
            postPartial("", dbgLastPartial, AsrLanguage.UND.code);
        }
        if (dbgFrames % 100 == 0) logDbg();
    }

    /**
     * Steer the hidden provisional decoder from uncertain LID evidence
     * (LID worker thread; 2026-09-16 §6.2, mirrors the Python reference).
     *
     * When the fused evidence persistently (2 agreeing hops) leans to another
     * language strongly enough, the provisional stream follows it — so the
     * adopt fast-path works for EN/ZH too and live partials decode in the
     * right language. When that engine is not resident the UI stays quiet
     * instead of showing VI-decoded garbage for foreign audio (the field-log
     * symptom: English rendered as Vietnamese text). Weak evidence keeps the
     * current provisional audible (avoids silence on noisy starts). Never
     * lazy-loads: the audio thread resolves the handoff.
     */
    private void applyProvisionalPolicy(LanguageIdEngine.LidResult r) {
        if (r == null || r.scores == null || r.scores.isEmpty()) return;
        AsrLanguage top = null;
        float topScore = -1f;
        for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
            Float v = r.scores.get(l);
            float f = v == null ? 0f : v;
            if (f > topScore) {
                topScore = f;
                top = l;
            }
        }
        if (top == null) return;
        boolean persistent = (top == lastUncertainTop);
        lastUncertainTop = top;
        if (top == provisionalLang) {
            provisionalQuiet = false;
            return;
        }
        if (topScore < AsrState.PROVISIONAL_MIN_SUPPORTED_REL || !persistent) {
            return;
        }
        if (models.isResident(top)) {
            pendingProvisionalLang = top;
            provisionalQuiet = false;
        } else {
            provisionalQuiet = true;
        }
    }

    /**
     * Apply a worker-requested provisional engine switch (audio thread).
     * The new stream starts empty: {@link #provisionalComplete} is cleared so
     * a later commit to this language replays instead of adopting a headless
     * stream. Pointer swap is under the pipeline monitor (mutually exclusive
     * with the worker-side adopt); engine calls stay on {@link #engineLock}.
     */
    private void applyPendingProvisionalSwitch() {
        AsrLanguage pending = pendingProvisionalLang;
        if (pending == null || pending == provisionalLang
                || activeLang != AsrLanguage.UND) {
            return;
        }
        pendingProvisionalLang = null;
        synchronized (this) {
            if (pending == provisionalLang || activeLang != AsrLanguage.UND) return;
            try {
                StreamingAsrEngine eng = models.get(pending);
                synchronized (engineLock) {
                    eng.reset();
                }
                provisionalAsr = eng;
                provisionalLang = pending;
                provisionalSpeculative = "";
                provisionalConf = 0.33f;
                lastPostedProvisional = "";
                provisionalQuiet = false;
                provisionalComplete = false;
                Log.i(TAG, "provisional decoder follows evidence: " + pending.code);
            } catch (Exception e) {
                Log.w(TAG, "provisional switch failed for " + pending, e);
            }
        }
    }

    private void decodeChunk(float[] chunk, long audioTimeMs) {
        if (activeAsr == null) return;
        final StreamingAsrEngine eng = activeAsr;
        final StreamingAsrEngine.PartialResult p;
        long t0 = System.currentTimeMillis();
        metrics.mark(AsrMetrics.ASR_START, t0);
        // Serialized on engineLock: the LID worker may shadow-decode a
        // candidate (or replay the bootstrap winner) concurrently.
        synchronized (engineLock) {
            eng.acceptAudio(chunk);
            // NEVER decode without the readiness gate: sherpa aborts the whole
            // process on an under-buffered decode (GetFrames OOB is fatal, not
            // an exception). See flushScheduler.
            if (eng.isReadyToDecode()) {
                eng.decodeAvailable();
            }
            p = eng.getPartialResult();
        }
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

    // --- Bootstrap acoustic LID (§7–§9, §14, §24, §27) --------------------

    /**
     * Submit one async bootstrap/referee classify. Runs on the audio thread;
     * the classify itself runs on lidExecutor, never in the audio callback
     * (§26). Mirrors the Python reference.
     *
     * Bootstrap phase (no language yet): hop-gated submits with a GROWING
     * window (baseline → MAX), capped per utterance (CPU bound). Referee
     * phase (§6.1, weak commit only): speech-gated re-evaluation so a weak
     * early decision can still be corrected via the runtime gate. All pacing
     * checks run BEFORE the single-flight flag is set, so the flag can never
     * get stuck by an early return.
     */
    private void tryBootstrapLid(long nowMs) {
        if (bootstrapPending) return;
        final boolean bootstrapping = activeLang == AsrLanguage.UND;
        if (bootstrapping) {
            if (utteranceSpeechMs < AsrState.BOOTSTRAP_MIN_MS) return;
            // Rolling hop gate (§5, §16): one attempt per 200 ms of fresh audio
            // (window / hop → temporal smoother fuses 3 hops).
            if (lastBootstrapSpeechMs >= 0
                    && utteranceSpeechMs - lastBootstrapSpeechMs < AsrState.BOOTSTRAP_HOP_MS) {
                return;
            }
            if (refereeAttempts >= AsrState.REFEREE_MAX_ATTEMPTS) {
                // Budget spent without a commit: no more ECAPA — keep
                // retrying candidates on the last ranking (cooldown-gated,
                // worker thread; mirrors the Python reference).
                retryCandidatesFromLast();
                return;
            }
            lastBootstrapSpeechMs = utteranceSpeechMs;
        } else {
            // Referee pass (§6.1): only after a WEAK commit (the caller
            // checks bootstrapCommitConfidence), paced by speech and bounded
            // per utterance. nowMs is unused here; cadence is speech-gated so
            // paused speech burns no ECAPA.
            if (bootstrapCommitConfidence >= AsrState.BOOTSTRAP_THRESHOLD) return;
            if (refereeAttempts >= AsrState.REFEREE_MAX_ATTEMPTS) return;
            if (utteranceSpeechMs - lastRefereeSpeechMs
                    < AsrState.SPECULATIVE_CANDIDATE_COOLDOWN_MS) {
                return;
            }
            lastRefereeSpeechMs = utteranceSpeechMs;
        }
        bootstrapPending = true;
        refereeAttempts++;
        // Dynamic window: grow with the utterance up to BOOTSTRAP_MAX_MS. The
        // fixed 600 ms window never showed ECAPA enough context for reliable
        // evidence (see docs/voxlingua_lid_diagnosis.md).
        final int windowMs = Math.min(AsrState.BOOTSTRAP_MAX_MS,
                Math.max(AsrState.BOOTSTRAP_LID_WINDOW_MS, (int) utteranceSpeechMs));
        // Onset-anchored (2026-09-16 v2): lastMs dilutes a growing window
        // with trailing/pre-speech silence — score the utterance audio
        // itself, oldest-first, like the commit replay does. The referee
        // below keeps lastMs (it judges CURRENT audio, not the onset).
        final float[] window = bootstrapping
                ? utteranceAudio(windowMs)
                : ring.lastMs(windowMs);
        final long uid = utteranceSeq.get();
        final long speechMs = utteranceSpeechMs;
        metrics.mark(AsrMetrics.LID_START);
        lidExecutor.submit(() -> {
            LanguageIdEngine.LidResult r;
            try {
                long t0 = System.currentTimeMillis();
                r = lid.classifyBootstrap(window);
                metrics.mark(AsrMetrics.LID_END);
                metrics.addRtf((System.currentTimeMillis() - t0)
                        / (double) windowMs);
            } catch (Exception e) {
                Log.w(TAG, "bootstrap classify failed", e);
                bootstrapPending = false;
                return;
            }
            onBootstrapLidResult(r, uid, speechMs);
        });
    }

    /**
     * Bootstrap result handler (LID worker thread). Confident → activate one
     * model + replay; uncertain → extend once, then ≤2 speculative
     * candidates (§9, §32). Stale results for a finished utterance are
     * dropped (§27).
     */
    private void onBootstrapLidResult(LanguageIdEngine.LidResult r, long uid,
                                      long speechMsAtRequest) {
        try {
            if (uid != utteranceSeq.get()) return; // stale: utterance ended
            final boolean alreadyActive;
            synchronized (this) {
                alreadyActive = activeLang != AsrLanguage.UND;
            }
            if (alreadyActive) {
                // §6.1 Referee Mode: a language is already active, so this is a
                // referee verdict, not a bootstrap one. Never a hard switch —
                // hand it to the runtime hysteresis/rollback gate (the same
                // path the 480 ms runtime LID uses). Deliberately OUTSIDE the
                // monitor: that gate can run a shadow decode.
                refereeOnActive(r, uid);
                return;
            }
            AsrLanguage decided = router.onBootstrapLidResult(r);
            if (decided != AsrLanguage.UND) {
                activateBootstrap(decided, r.confidence, uid);
                return;
            }
            // Uncertain: steer the provisional stream from this evidence, then
            // either extend (§9 cách 1) or run dual-candidates (§32).
            lastBootstrapResult = r;
            applyProvisionalPolicy(r);
            boolean mayExtend;
            synchronized (this) {
                mayExtend = utteranceSpeechMs < AsrState.BOOTSTRAP_MAX_MS
                        && refereeAttempts < AsrState.REFEREE_MAX_ATTEMPTS
                        && uid == utteranceSeq.get()
                        && activeLang == AsrLanguage.UND;
            }
            if (mayExtend) {
                bootstrapPending = false; // allow one more windowed attempt
                return;
            }
            // Cooldown: a candidate pass costs 2 shadow decodes — rerun at
            // most every SPECULATIVE_CANDIDATE_COOLDOWN_MS of additional audio,
            // not every LID tick.
            long sinceCandidates = speechMsAtRequest - lastCandidateRunSpeechMs;
            if (lastCandidateRunSpeechMs >= 0
                    && sinceCandidates < AsrState.SPECULATIVE_CANDIDATE_COOLDOWN_MS) {
                bootstrapPending = false;
                return;
            }
            // Give-up: past BOOTSTRAP_GIVE_UP_MS without a commit (unsupported
            // language / noise), stop burning candidate decodes. The hidden
            // provisional stream keeps flowing and the endpoint recovery still
            // gets one final chance — this only stops the CPU decode storm.
            if (speechMsAtRequest > AsrState.BOOTSTRAP_GIVE_UP_MS) {
                Log.i(TAG, "bootstrap give-up after " + speechMsAtRequest
                        + "ms, keeping hidden provisional only");
                bootstrapPending = false;
                return;
            }
            lastCandidateRunSpeechMs = speechMsAtRequest;
            runSpeculativeCandidates(r, uid);
        } catch (Exception e) {
            Log.w(TAG, "bootstrap result handling failed", e);
            if (activeLang == AsrLanguage.UND) bootstrapPending = false;
        } finally {
            // Safety net: a stale utterance must never leave the flag set
            // (the next utterance would refuse to bootstrap).
            if (bootstrapPending && uid != utteranceSeq.get()) {
                bootstrapPending = false;
            }
        }
    }

    /**
     * Activate the bootstrap winner (§13, §15, §25) — adopt or rollback:
     * <ul>
     *   <li>Winner == provisional language on a COMPLETE stream → <b>adopt</b>
     *   the live stream as-is: no reset, no replay, zero duplication (the
     *   scheduler residue in schedulerBuf continues into the same
     *   engine).</li>
     *   <li>Otherwise → <b>rollback</b>: discard the provisional text without
     *   a trace, reset the winner and REPLAY the buffered utterance audio
     *   into it, so the first words are decoded — not dropped. In particular
     *   a mid-utterance provisional engine switch leaves a headless stream,
     *   which must never be adopted.</li>
     * </ul>
     * Engine calls run on {@link #engineLock} (GetFrames race); the method
     * itself stays under the pipeline monitor so the audio-thread switch
     * ({@link #applyPendingProvisionalSwitch}) cannot interleave.
     */
    private synchronized void activateBootstrap(AsrLanguage lang, float confidence,
                                                long uid) {
        if (uid != utteranceSeq.get()) {
            bootstrapPending = false;
            return;
        }
        if (activeLang != AsrLanguage.UND) {
            bootstrapPending = false;
            return;
        }
        if (lang == null || lang == AsrLanguage.UND) {
            bootstrapPending = false;
            return;
        }
        if (lang == provisionalLang && provisionalAsr != null && provisionalComplete) {
            // Adopt: the live stream already holds the full utterance state.
            activeAsr = provisionalAsr;
            provisionalAsr = null;
            activeLang = lang;
            bootstrapCommitConfidence = confidence;
            refereeAttempts = 0; // fresh referee budget for a weak commit
            router.commitBootstrap(lang, confidence);
            // Seed the transcript lifecycle with the speculative text shown
            // so far (it firms up via STABLE like any other partial).
            transcripts.updateSpeculative(provisionalSpeculative);
            postPartial(transcripts.getCommitted(), transcripts.getSpeculative(),
                    lang.code);
            Log.i(TAG, "bootstrap adopted provisional " + lang.code + " conf=" + confidence);
            provisionalSpeculative = "";
            lastPostedProvisional = "";
            provisionalQuiet = false;
            bootstrapPending = false;
            return;
        }
        // Rollback: provisional text belonged to the wrong language (or a
        // headless switched stream) — drop it (it never entered the
        // transcript manager) and rebuild on the winner.
        provisionalSpeculative = "";
        lastPostedProvisional = "";
        provisionalAsr = null;
        provisionalQuiet = false;
        schedulerFill = 0; // replay below re-covers this audio from the ring
        setActiveLanguage(lang, confidence);
        float[] buffered;
        try {
            buffered = utteranceAudio(30_000);
        } catch (Exception e) {
            Log.w(TAG, "bootstrap replay: no audio window", e);
            buffered = new float[0];
        }
        if (buffered.length > 0 && activeAsr != null) {
            final StreamingAsrEngine target = activeAsr;
            final int step = schedulerTarget;
            final StreamingAsrEngine.PartialResult p;
            synchronized (engineLock) {
                for (int off = 0; off < buffered.length; off += step) {
                    int n = Math.min(step, buffered.length - off);
                    float[] slice = new float[n];
                    System.arraycopy(buffered, off, slice, 0, n);
                    target.acceptAudio(slice);
                    if (target.isReadyToDecode()) target.decodeAvailable();
                }
                p = target.getPartialResult();
            }
            // Still SPECULATIVE (§11): bootstrap chose the language, but the
            // transcript firms up only with more audio.
            transcripts.updateSpeculative(p.text);
            metrics.mark(AsrMetrics.PARTIAL_VISIBLE);
            postPartial(transcripts.getCommitted(), transcripts.getSpeculative(),
                    lang.code);
            Log.i(TAG, "bootstrap committed " + lang.code + " conf=" + confidence
                    + " replayMs=" + (buffered.length * 1000 / AsrState.SAMPLE_RATE));
        }
        bootstrapPending = false;
    }

    /**
     * Candidate retry after the ECAPA budget is spent (audio thread): reuses
     * the last bootstrap ranking on the cooldown cadence instead of burning
     * more ECAPA. The slot is claimed pre-submit so the 20 ms audio callback
     * cannot flood the single-thread worker; the task itself rechecks
     * utterance/staleness before decoding. Mirrors the Python reference
     * (which retries candidates every cooldown until give-up).
     */
    private void retryCandidatesFromLast() {
        if (activeLang != AsrLanguage.UND) return;
        final LanguageIdEngine.LidResult r = lastBootstrapResult;
        if (r == null) return;
        final long speechMs = utteranceSpeechMs;
        if (speechMs > AsrState.BOOTSTRAP_GIVE_UP_MS) return;
        if (lastCandidateRunSpeechMs >= 0
                && speechMs - lastCandidateRunSpeechMs
                < AsrState.SPECULATIVE_CANDIDATE_COOLDOWN_MS) {
            return;
        }
        lastCandidateRunSpeechMs = speechMs;
        final long uid = utteranceSeq.get();
        lidExecutor.submit(() -> runSpeculativeCandidates(r, uid));
    }

    /**
     * Uncertain-bootstrap fallback (§32): decode a short window on at most
     * the top-2 candidate models (1–2 chunks each, one-shot — never 3 models
     * continuously, §10) and commit the winner by confidence + lexical fit.
     * If ZH is among the top-2 the pair is EN+ZH/ZH-led per §32; nothing is
     * committed when neither candidate decodes real content (short blips
     * stay speculative, §35).
     */
    private void runSpeculativeCandidates(LanguageIdEngine.LidResult r, long uid) {
        float[] window;
        try {
            // Onset-anchored like the bootstrap window (no silence dilution).
            window = utteranceAudio(AsrState.BOOTSTRAP_MAX_MS);
        } catch (Exception e) {
            Log.w(TAG, "speculative candidates: no audio", e);
            return;
        }
        // Too-short window → decode would be garbage + burn 2 engines.
        if (window == null || window.length < AsrState.SAMPLE_RATE * 400 / 1000) {
            bootstrapPending = false;
            return;
        }
        runSpeculativeCandidatesOn(r, uid, window, chooseCandidatePair(
                r == null ? null : r.scores));
    }

    /**
     * Rotating candidate pair (≤2 decodes per pass, §10): normally the
     * acoustic top-2; when the previous pass was inconclusive, the leftover
     * 3rd language pairs with the top-1 so all three get tried within ~1.6 s
     * without ever running 3 models at once.
     */
    private AsrLanguage[] chooseCandidatePair(Map<AsrLanguage, Float> scores) {
        AsrLanguage[] top = topTwo(scores);
        if (top[0] == null) return top; // defensive: topTwo never returns nulls
        AsrLanguage[] last = lastCandidatePair;
        if (last == null) return top;
        AsrLanguage leftover = null;
        for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
            if (l != last[0] && l != last[1]) {
                leftover = l;
                break;
            }
        }
        if (leftover == null || leftover == top[0]) return top;
        return new AsrLanguage[]{top[0], leftover};
    }

    /** Languages tried by neither member of {@code pair} (0–1 entries). */
    private static AsrLanguage[] leftoversOf(AsrLanguage[] pair) {
        java.util.List<AsrLanguage> out = new java.util.ArrayList<>();
        if (pair != null) {
            for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
                if (l != pair[0] && l != pair[1]) out.add(l);
            }
        }
        return out.toArray(new AsrLanguage[0]);
    }

    /** One-shot decode of {@code window} on at most the top-2 models. */
    private void runSpeculativeCandidatesOn(LanguageIdEngine.LidResult r, long uid,
                                            float[] window) {
        runSpeculativeCandidatesOn(r, uid, window, topTwo(r == null ? null : r.scores));
    }

    /** One-shot decode of {@code window} on the given pair (max 2). */
    private void runSpeculativeCandidatesOn(LanguageIdEngine.LidResult r, long uid,
                                            float[] window, AsrLanguage[] pair) {
        try {
            if (uid != utteranceSeq.get()) return;
            synchronized (this) {
                if (activeLang != AsrLanguage.UND) return;
            }
            if (window == null || window.length == 0) return;
            lastCandidatePair = pair;
            AsrLanguage winner = null;
            float winnerScore = -1;
            float secondScore = -1;
            String winnerText = "";
            float winnerConf = 0;
            for (AsrLanguage cand : pair) {
                if (cand == null || cand == AsrLanguage.UND) continue;
                CandidateScore s = scoreSpeculativeCandidate(cand, window);
                if (s == null) continue;
                float combined = 0.65f * s.confidence
                        + 0.35f * LanguageIdEngine.lexicalFit(s.text, cand);
                if (s.text.isEmpty()) continue;
                if (combined > winnerScore) {
                    secondScore = winnerScore;
                    winnerScore = combined;
                    winner = cand;
                    winnerText = s.text;
                    winnerConf = s.confidence;
                } else if (combined > secondScore) {
                    secondScore = combined;
                }
            }
            // Unified bar (§6.3, mirrors the Python reference): the candidate
            // path used to commit at 0.30, far below the 0.70 acoustic gate —
            // a rejected LID therefore always produced a low-confidence
            // commit, and the [VI, EN] tie-break made it Vietnamese. Now the
            // winner needs acoustic-commit-level evidence plus a margin over
            // the runner-up (no coin-flip commits); confidence stays capped
            // below 0.70 so the referee can still correct a fallback commit.
            if (winner != null && !winnerText.isEmpty()
                    && winnerScore >= AsrState.CANDIDATE_COMMIT_SCORE
                    && (winnerScore - secondScore) >= AsrState.CANDIDATE_MARGIN) {
                Log.i(TAG, "speculative candidates picked " + winner.code
                        + " score=" + winnerScore);
                activateBootstrap(winner, Math.min(0.69f, winnerScore), uid);
            } else {
                Log.i(TAG, "speculative candidates inconclusive, staying UNKNOWN");
            }
        } finally {
            bootstrapPending = false;
        }
    }

    /** Top-2 languages from a bootstrap score map (UND excluded).
     *
     * 2026-09-16 §6.3: pure score order with a fixed scan order that does NOT
     * lead with VI ([EN, ZH, VI]) — the old [VI, EN, ZH] scan returned
     * [VI, EN] on every flat/uncertain map, and with the 0.30 bar the
     * fallback commit was almost always Vietnamese. Rotation
     * ({@link #chooseCandidatePair}) still covers the third language on the
     * next pass. Mirrors the Python reference exactly (stable sort over
     * [en, zh, vi]).
     */
    private static AsrLanguage[] topTwo(Map<AsrLanguage, Float> scores) {
        // No ranking at all (null map): same non-VI-first default as a flat
        // map below — never a VI-first guess.
        if (scores == null) return new AsrLanguage[]{AsrLanguage.EN, AsrLanguage.ZH};
        AsrLanguage first = null;
        AsrLanguage second = null;
        float s1 = Float.NEGATIVE_INFINITY;
        float s2 = Float.NEGATIVE_INFINITY;
        for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.EN, AsrLanguage.ZH, AsrLanguage.VI}) {
            Float v = scores.get(l);
            float f = v == null ? 0f : v;
            if (f > s1) {
                s2 = s1;
                second = first;
                s1 = f;
                first = l;
            } else if (f > s2) {
                s2 = f;
                second = l;
            }
        }
        return new AsrLanguage[]{first, second};
    }

    private static class CandidateScore {
        String text = "";
        float confidence = 0;
    }

    /**
     * One-shot decode of the bootstrap window on a candidate (never the
     * active stream). The provisional engine is read WITHOUT reset — wiping
     * it would destroy the live speculative stream it is still decoding.
     * Serialized on {@link #engineLock} (GetFrames race: the audio thread may
     * be decoding the provisional engine concurrently).
     */
    private CandidateScore scoreSpeculativeCandidate(AsrLanguage cand, float[] window) {
        try {
            StreamingAsrEngine e = models.get(cand);
            boolean isLiveProvisional = (e == provisionalAsr);
            final StreamingAsrEngine.PartialResult p;
            synchronized (engineLock) {
                if (!isLiveProvisional) {
                    e.reset();
                    int step = AsrState.SCHEDULER_SAMPLES;
                    for (int off = 0; off < window.length; off += step) {
                        int n = Math.min(step, window.length - off);
                        float[] slice = new float[n];
                        System.arraycopy(window, off, slice, 0, n);
                        e.acceptAudio(slice);
                        if (e.isReadyToDecode()) e.decodeAvailable();
                    }
                }
                p = e.getPartialResult();
            }
            CandidateScore s = new CandidateScore();
            s.text = p.text == null ? "" : p.text.trim();
            s.confidence = p.confidence;
            return s;
        } catch (Exception ex) {
            Log.w(TAG, "speculative decode failed for " + cand, ex);
            return null;
        }
    }

    /**
     * §6.1 Referee Mode verdict while a language is already active. The
     * acoustic referee never switches anything by itself: an agreement just
     * firms up the current decision (and then stops the referee), a
     * disagreement is forwarded to the runtime gate (threshold + margin +
     * persistence + shadow verification), which is what keeps a single noisy
     * long window from rewriting committed text.
     */
    private void refereeOnActive(LanguageIdEngine.LidResult r, long uid) {
        if (r == null || r.language == null) return;
        final AsrLanguage active = activeLang;
        if (r.language == AsrLanguage.UND || r.language == active) {
            if (r.confidence >= AsrState.BOOTSTRAP_THRESHOLD) {
                bootstrapCommitConfidence = r.confidence;
                Log.i(TAG, "referee confirms " + active.code + " conf=" + r.confidence);
            } else {
                Log.d(TAG, "referee agrees weakly, staying " + active.code);
            }
            return;
        }
        Log.i(TAG, "referee disagrees: " + r.language.code + " conf=" + r.confidence
                + " vs active " + active.code + " → runtime gate");
        onLidResult(r, uid);
    }

    private void onLidResult(LanguageIdEngine.LidResult r, long uid) {
        if (uid != utteranceSeq.get()) return; // stale runtime LID (§27)
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

    /**
     * Shadow decode of the rollback window on the candidate model.
     * Serialized on {@link #engineLock}: the audio thread keeps decoding the
     * active stream while this runs (GetFrames race).
     */
    private LanguageRouter.Verification verifyCandidate(AsrLanguage candidate,
                                                        float[] rollbackAudio) {
        try {
            StreamingAsrEngine cand = models.get(candidate);
            // Fresh stream state for the window (never the active stream),
            // paced like live input.
            final StreamingAsrEngine.PartialResult r;
            synchronized (engineLock) {
                cand.reset();
                int step = AsrState.SCHEDULER_SAMPLES;
                for (int off = 0; off < rollbackAudio.length; off += step) {
                    int n = Math.min(step, rollbackAudio.length - off);
                    float[] slice = new float[n];
                    System.arraycopy(rollbackAudio, off, slice, 0, n);
                    cand.acceptAudio(slice);
                    if (cand.isReadyToDecode()) cand.decodeAvailable();
                }
                r = cand.getPartialResult();
            }
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
        StreamingAsrEngine.PartialResult p = flushInto(activeAsr);
        if (p != null) {
            transcripts.updateSpeculative(p.text);
            postPartial(transcripts.getCommitted(), transcripts.getSpeculative(),
                    activeLang.code);
        }
    }

    /**
     * Fold scheduler residue into {@code engine}, pad trailing silence until
     * decodable (bounded) and decode once. Returns the fresh partial, or null
     * when nothing became decodable. Shared by the active flush and the
     * provisional endpoint flush.
     */
    private StreamingAsrEngine.PartialResult flushInto(StreamingAsrEngine engine) {
        if (engine == null) {
            schedulerFill = 0;
            return null;
        }
        // All engine calls serialized on engineLock (GetFrames race: the audio
        // thread may be decoding another engine concurrently).
        synchronized (engineLock) {
            if (schedulerFill > 0) {
                float[] tail = new float[schedulerFill];
                System.arraycopy(schedulerBuf, 0, tail, 0, schedulerFill);
                engine.acceptAudio(tail);
            }
            schedulerFill = 0;
            int padded = 0;
            int cap = 960 * AsrState.SAMPLE_RATE / 1000;
            while (!engine.isReadyToDecode() && padded < cap) {
                engine.acceptAudio(new float[AsrState.SCHEDULER_SAMPLES]);
                padded += AsrState.SCHEDULER_SAMPLES;
            }
            if (!engine.isReadyToDecode()) return null;
            engine.decodeAvailable();
            return engine.getPartialResult();
        }
    }

    private void onCommitted(AsrLanguage next, float[] window) {
        AsrLanguage prev = activeLang;
        transcripts.rollbackSpeculative();
        StreamingAsrEngine cand = models.get(next);
        final StreamingAsrEngine.PartialResult r;
        synchronized (engineLock) {
            r = cand.getPartialResult();
        }
        transcripts.commit(r.text);
        setActiveLanguage(next, r.confidence);
        // Re-feed the window into the new active stream so its incremental
        // state continues from the switch point (spec §14).
        synchronized (engineLock) {
            if (activeAsr != null) activeAsr.acceptAudio(window);
        }
        postSwitch(prev.code, next.code);
        postPartial(transcripts.getCommitted(), transcripts.getSpeculative(), next.code);
    }

    private void onEndpoint() {
        // Short utterance that never bootstrapped (§35): try one synchronous
        // bootstrap on the endpoint thread (no realtime pressure here), then
        // finalize whatever it yields. Never force a language — a blip with
        // no confident evidence finalizes empty on UND.
        if (activeAsr == null || activeLang == AsrLanguage.UND) {
            finishUnbootstrappedEndpoint();
            return;
        }
        Log.d(TAG, "endpoint reached, active=" + activeLang.code);
        // Flush residue first so tail words join THIS utterance instead of
        // leaking into the next one (see flushScheduler).
        flushScheduler();
        final StreamingAsrEngine eng = activeAsr;
        final StreamingAsrEngine.PartialResult fin;
        synchronized (engineLock) {
            fin = eng.getFinalResult();
        }
        String baseText = mergeDisplay(fin.text);
        // Endpoint candidate verification (spec §13.1, kept as backstop in
        // fakedemo2 Phase 5): replay the utterance through the non-active
        // models. Bootstrap usually commits the right model already, but a
        // mid-utterance language change that the runtime gate missed still
        // gets one correction chance here. Two decodes per utterance — never
        // 3 parallel decoders on the live path (spec §31).
        EndpointVerdict verdict = verifyEndpointLanguage(baseText, fin.confidence,
                activeLang);
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
        // Fresh utterance state: the NEXT utterance starts UNKNOWN again
        // (§45 rule 1–2). Resident engines stay cached in the model manager
        // (§29 — switch = pointer change, never load/unload); only the
        // active pointer is cleared.
        resetUtteranceState();
        postMetrics();
    }

    /** Utterance audio from speech onset, oldest-first, capped at maxMs. */
    private float[] utteranceAudio(int maxMs) {
        int maxSamples = (int) ((long) AsrState.SAMPLE_RATE * maxMs / 1000);
        if (utteranceStartSample >= 0) {
            return ring.since(utteranceStartSample, maxSamples);
        }
        return ring.lastMs(maxMs);
    }

    /**
     * Endpoint arrived before any language was committed (§35). Recovery in
     * order: (1) sync audio-only bootstrap on the utterance onset; (2) one
     * speculative pair pass + the leftover language over the utterance audio;
     * (3) endpoint-verification backstop with the flushed provisional
     * hypothesis as base. Never forces a language — a blip with no evidence
     * finalizes empty on UND.
     */
    private void finishUnbootstrappedEndpoint() {
        long uid = utteranceSeq.get();
        try {
            // Flush the provisional residue first so the short utterance is
            // fully decoded before it is judged (mirrors flushScheduler).
            if (provisionalAsr != null) {
                StreamingAsrEngine.PartialResult flushed = flushInto(provisionalAsr);
                if (flushed != null && flushed.text != null
                        && !flushed.text.trim().isEmpty()) {
                    provisionalSpeculative = flushed.text.trim();
                    provisionalConf = flushed.confidence;
                }
            }
            // Read from the utterance start: at an endpoint lastMs would
            // return mostly the trailing silence (§35 short utterances).
            // Bootstrap replays up to the ENDPOINT window (wider than the live
            // cap — no realtime pressure here; mirrors the Python reference).
            float[] spoken = utteranceAudio(AsrState.ENDPOINT_VERIFY_MS);
            float[] window = spoken.length > AsrState.SAMPLE_RATE
                    * AsrState.BOOTSTRAP_ENDPOINT_WINDOW_MS / 1000
                    ? java.util.Arrays.copyOfRange(spoken, 0,
                            AsrState.SAMPLE_RATE * AsrState.BOOTSTRAP_ENDPOINT_WINDOW_MS / 1000)
                    : spoken;
            LanguageIdEngine.LidResult r = lid.classifyBootstrap(window);
            AsrLanguage decided = router.onBootstrapLidResult(r);
            if (decided != AsrLanguage.UND) {
                activateBootstrap(decided, r.confidence, uid);
                decided = activeLang;
            } else {
                // Second chance over the utterance audio itself: the top pair,
                // then the leftover language if still undecided.
                AsrLanguage[] pair = chooseCandidatePair(
                        r == null ? null : r.scores);
                // Defensive null guard (topTwo never returns nulls — flat maps
                // fall back to [EN, ZH] with VI covered by the leftover pass).
                if (pair[0] != null) {
                    runSpeculativeCandidatesOn(r, uid, spoken, pair);
                    decided = activeLang;
                    if (decided == AsrLanguage.UND) {
                        AsrLanguage[] rest = leftoversOf(pair);
                        if (rest.length > 0) {
                            runSpeculativeCandidatesOn(r, uid, spoken,
                                    new AsrLanguage[]{rest[0], AsrLanguage.UND});
                            decided = activeLang;
                        }
                    }
                }
            }
            if (decided == AsrLanguage.UND) {
                // Final backstop: only adopt if acoustic LID had decisive confidence
                // (>= 0.60), never defaulting to VI without evidence.
                if (r != null && r.language != AsrLanguage.UND && r.confidence >= 0.60f) {
                    activateBootstrap(r.language, r.confidence, uid);
                    decided = activeLang;
                }
            }
            if (decided != AsrLanguage.UND && activeAsr != null) {
                flushScheduler();
                final StreamingAsrEngine eng = activeAsr;
                final StreamingAsrEngine.PartialResult fin;
                synchronized (engineLock) {
                    fin = eng.getFinalResult();
                }
                String text = mergeDisplay(fin.text);
                transcripts.finalizeTranscript(text);
                postFinal(transcripts.getCommitted(), activeLang.code);
                metrics.addUtterance();
            } else {
                // Inconclusive short blip: finalize empty, stay UNKNOWN.
                transcripts.finalizeTranscript("");
                postFinal("", AsrLanguage.UND.code);
                metrics.addUtterance();
            }
        } catch (Exception e) {
            Log.w(TAG, "unbootstrapped endpoint failed", e);
            try {
                transcripts.finalizeTranscript("");
                postFinal("", AsrLanguage.UND.code);
            } catch (Exception ignored) {
            }
        } finally {
            resetUtteranceState();
            postMetrics();
        }
    }

    /** Clear per-utterance state; next utterance bootstraps from UNKNOWN. */
    private void resetUtteranceState() {
        transcripts.reset();
        if (activeAsr != null) {
            final StreamingAsrEngine eng = activeAsr;
            try {
                synchronized (engineLock) {
                    eng.reset();
                }
            } catch (Exception e) {
                Log.w(TAG, "engine reset failed", e);
            }
        }
        activeAsr = null;
        activeLang = AsrLanguage.UND;
        router.reset();
        try {
            // VoxLingua temporal smoother holds the previous utterance's
            // language — clear it so the next bootstrap starts flat (§16).
            lid.reset();
        } catch (Exception ignored) {
        }
        schedulerFill = 0;
        utteranceStartMs = -1;
        utteranceStartSample = -1;
        utteranceSpeechMs = 0;
        lastBootstrapSpeechMs = -1;
        bootstrapCommitConfidence = 0f;
        refereeAttempts = 0;
        lastRefereeSpeechMs = -1;
        lastBootstrapResult = null;
        lastCandidateRunSpeechMs = -1;
        lastCandidatePair = null;
        bootstrapPending = false;
        resetProvisional();
        utteranceSeq.incrementAndGet(); // invalidate in-flight LID results
    }

    /**
     * (Re)point the provisional decoder at the always-resident VI engine with
     * fresh state (§6.2: the START point is VI, but uncertain LID evidence
     * steers it from there — never a hard-coded VI lock). Safe to call when
     * the same object is also the (now cleared) active engine — reset is
     * idempotent. Mirrors the Python reference.
     */
    private void resetProvisional() {
        provisionalLang = AsrLanguage.VI;
        provisionalSpeculative = "";
        lastPostedProvisional = "";
        provisionalConf = 0.33f;
        provisionalQuiet = false;
        provisionalComplete = true;
        pendingProvisionalLang = null;
        lastUncertainTop = null;
        try {
            provisionalAsr = models.get(AsrLanguage.VI);
            synchronized (engineLock) {
                provisionalAsr.reset();
            }
        } catch (Exception e) {
            Log.w(TAG, "provisional reset failed", e);
            provisionalAsr = null;
        }
    }

    /** Decided language for the finished utterance, or null if inconclusive. */
    private static class EndpointVerdict {
        AsrLanguage lang;
        String text;
        float confidence;
    }

    /**
     * Replay the utterance audio (up to ENDPOINT_VERIFY_MS) through each
     * non-base model and compare hypotheses on a COMBINED score
     * (confidence + lexical fit). Pure-confidence margins lock in
     * cross-lingual near-misses ("HELLO" via the VI model at 0.60), so the
     * lexical term breaks those ties. Returns the winner when it beats the
     * base hypothesis by ENDPOINT_VERIFY_MARGIN, the base language when its
     * hypothesis is solid and unbeaten, or null when nothing is decidable.
     *
     * @param baseLang the hypothesis owner to compare against (active model,
     *                 or the provisional model on the unbootstrapped path).
     */
    private EndpointVerdict verifyEndpointLanguage(String baseText, float baseConf,
                                                   AsrLanguage baseLang) {
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
        float baseScore = confW * baseConf
                + AsrState.ENDPOINT_VERIFY_LEX_W
                * LanguageIdEngine.lexicalFit(cleanBase, baseLang);
        AsrLanguage best = null;
        float bestScore = 0;
        String bestText = "";
        float bestConf = 0;
        for (AsrLanguage lang : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
            if (lang == baseLang) continue;
            EndpointVerdict cand = scoreCandidate(lang, audio);
            if (cand == null || cand.text.isEmpty()) continue;
            float lexCand = LanguageIdEngine.lexicalFit(cand.text, lang);
            int minUnits = lexCand >= AsrState.ENDPOINT_VERIFY_STRONG_LEX
                    ? 1 : AsrState.ENDPOINT_VERIFY_MIN_TOKENS;
            if (LanguageIdEngine.textUnits(cand.text) < minUnits) continue;
            float candScore = confW * cand.confidence
                    + AsrState.ENDPOINT_VERIFY_LEX_W * lexCand;
            if (candScore >= baseScore + AsrState.ENDPOINT_VERIFY_MARGIN
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
        // Base unbeaten: confirmed only when its own hypothesis is solid —
        // fluent multi-unit text, or a single dictionary-strong token like a
        // lone "HELLO"/"một" (§35 short utterances) — otherwise inconclusive.
        if (baseConf >= 0.5f
                && (LanguageIdEngine.textUnits(cleanBase) >= 2
                || (LanguageIdEngine.textUnits(cleanBase) >= 1
                && LanguageIdEngine.lexicalFit(cleanBase, baseLang)
                >= AsrState.ENDPOINT_VERIFY_STRONG_LEX))) {
            EndpointVerdict v = new EndpointVerdict();
            v.lang = baseLang;
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
            final StreamingAsrEngine.PartialResult r;
            synchronized (engineLock) {
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
                r = cand.getPartialResult();
            }
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
        // §6.1: remember how solid this decision was. The acoustic referee
        // keeps re-evaluating the utterance until the decision reaches
        // BOOTSTRAP_THRESHOLD (see onFrame / refereeOnActive), with a fresh
        // per-commit attempt budget.
        bootstrapCommitConfidence = conf;
        refereeAttempts = 0;
        // §13: never models.get(UND) — UND means "no active model yet".
        if (lang == null || lang == AsrLanguage.UND) {
            activeLang = AsrLanguage.UND;
            activeAsr = null;
            router.setActive(AsrLanguage.UND, conf);
            return;
        }
        activeLang = lang;
        activeAsr = models.get(lang);
        try {
            final StreamingAsrEngine eng = activeAsr;
            synchronized (engineLock) {
                eng.reset();
            }
        } catch (Exception e) {
            Log.w(TAG, "engine reset failed for " + lang, e);
        }
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
