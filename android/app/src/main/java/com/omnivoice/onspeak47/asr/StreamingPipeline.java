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

    // --- Provisional speculative decode (Option A hotfix) ---
    // While UNKNOWN the pipeline ALSO decodes continuously on one provisional
    // model (VI, always resident) so partials flow after ~160 ms like the old
    // system. Provisional output is SPECULATIVE-only: it lives in
    // provisionalSpeculative (never the shared transcript manager), is
    // adopted without replay when bootstrap confirms the same language, and
    // is discarded without a trace otherwise (§11: nothing committed before
    // the language is confirmed).
    private volatile StreamingAsrEngine provisionalAsr;
    private volatile AsrLanguage provisionalLang = AsrLanguage.VI;
    private volatile String provisionalSpeculative = "";
    private volatile float provisionalConf = 0.33f;

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
                defaultLidEngine(), new LanguageRouter());
    }

    /**
     * Default LID engine wired to the interim audio-derived heuristic (§16).
     * Replaces the old {@code new LanguageIdEngine()} whose null scorer fell
     * back to a transcript heuristic — the circular VI lock (§3–§4). Inject a
     * trained tiny classifier via the 5-arg constructor to upgrade.
     */
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
        schedulerFill = 0;
        utteranceStartMs = -1;
        lastLidMs = 0;
        utteranceSpeechMs = 0;
        lastBootstrapSpeechMs = -1;
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
            StreamingAsrEngine.PartialResult fin = activeAsr.getFinalResult();
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
        // is adopted (same language, no replay) or discarded (other language)
        // on commit — never committed itself (§11).
        if (activeLang == AsrLanguage.UND) {
            tryBootstrapLid(nowMs);
            accumulateScheduler(frame, nowMs, true);
            return;
        }

        // 160 ms scheduler accumulation on the active stream.
        accumulateScheduler(frame, nowMs, false);

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
     * Provisional speculative decode (Option A): continuous partials on the
     * VI engine while UNKNOWN. Emits immediately for live UI feedback; the
     * text is held outside the shared transcript manager so a later commit
     * to another language discards it without a trace.
     */
    private void decodeProvisional(float[] chunk, long audioTimeMs) {
        if (provisionalAsr == null) {
            try {
                provisionalAsr = models.get(AsrLanguage.VI);
                provisionalAsr.reset();
            } catch (Exception e) {
                Log.w(TAG, "provisional engine unavailable", e);
                return;
            }
        }
        long t0 = System.currentTimeMillis();
        metrics.mark(AsrMetrics.ASR_START, t0);
        provisionalAsr.acceptAudio(chunk);
        // Readiness gate applies here too (sherpa aborts on under-buffered
        // decode — fatal, not an exception).
        if (provisionalAsr.isReadyToDecode()) {
            provisionalAsr.decodeAvailable();
        }
        StreamingAsrEngine.PartialResult p = provisionalAsr.getPartialResult();
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
        if (!dbgLastPartial.isEmpty()) {
            Log.d(TAG, "provisional lang=" + provisionalLang.code + " conf=" + p.confidence
                    + " text=" + truncate(dbgLastPartial, 80));
            postPartial("", dbgLastPartial, provisionalLang.code);
        }
        if (dbgFrames % 100 == 0) logDbg();
    }

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

    // --- Bootstrap acoustic LID (§7–§9, §14, §24, §27) --------------------

    /**
     * Submit one async bootstrap classify when enough speech audio has
     * accumulated. Runs on the audio thread; the classify itself runs on
     * lidExecutor, never in the audio callback (§26).
     */
    private void tryBootstrapLid(long nowMs) {
        if (activeLang != AsrLanguage.UND || bootstrapPending) return;
        if (utteranceSpeechMs < AsrState.BOOTSTRAP_MIN_MS) return;
        // Extend-window gate (§9 cách 1): retry only as fresh audio arrives,
        // so a stuck-uncertain utterance doesn't spin the executor.
        if (lastBootstrapSpeechMs >= 0
                && utteranceSpeechMs - lastBootstrapSpeechMs < AsrState.SCHEDULER_MS
                && utteranceSpeechMs >= AsrState.BOOTSTRAP_MAX_MS) {
            return;
        }
        bootstrapPending = true;
        lastBootstrapSpeechMs = utteranceSpeechMs;
        final float[] window = ring.lastMs(AsrState.BOOTSTRAP_LID_WINDOW_MS);
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
                        / (double) AsrState.BOOTSTRAP_LID_WINDOW_MS);
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
            synchronized (this) {
                if (activeLang != AsrLanguage.UND) return; // won race (§27)
            }
            AsrLanguage decided = router.onBootstrapLidResult(r);
            if (decided != AsrLanguage.UND) {
                activateBootstrap(decided, r.confidence, uid);
                return;
            }
            // Uncertain: one extension (§9 cách 1), then dual-candidate (§32).
            boolean mayExtend;
            synchronized (this) {
                mayExtend = utteranceSpeechMs < AsrState.BOOTSTRAP_MAX_MS
                        && uid == utteranceSeq.get()
                        && activeLang == AsrLanguage.UND;
            }
            if (mayExtend) {
                bootstrapPending = false; // allow one more windowed attempt
                return;
            }
            // Cooldown: a candidate pass costs 2 shadow decodes — rerun at
            // most every 800 ms of additional audio, not every LID tick.
            long sinceCandidates = speechMsAtRequest - lastCandidateRunSpeechMs;
            if (lastCandidateRunSpeechMs >= 0 && sinceCandidates < 800) {
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
     *   <li>Winner == provisional language → <b>adopt</b> the provisional
     *   stream as-is: no reset, no replay, zero duplication (the scheduler
     *   residue in schedulerBuf continues into the same engine).</li>
     *   <li>Winner differs → <b>rollback</b>: discard the provisional text
     *   without a trace, reset the winner and REPLAY the buffered utterance
     *   audio into it, so the first 200–300 ms are decoded — not dropped.</li>
     * </ul>
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
        if (lang == provisionalLang && provisionalAsr != null) {
            // Adopt: the live stream already holds the full utterance state.
            activeAsr = provisionalAsr;
            provisionalAsr = null;
            activeLang = lang;
            router.commitBootstrap(lang, confidence);
            // Seed the transcript lifecycle with the speculative text shown
            // so far (it firms up via STABLE like any other partial).
            transcripts.updateSpeculative(provisionalSpeculative);
            postPartial(transcripts.getCommitted(), transcripts.getSpeculative(),
                    lang.code);
            Log.i(TAG, "bootstrap adopted provisional " + lang.code + " conf=" + confidence);
            provisionalSpeculative = "";
            bootstrapPending = false;
            return;
        }
        // Rollback: provisional text belonged to the wrong language — drop it
        // (it never entered the transcript manager) and rebuild on the winner.
        provisionalSpeculative = "";
        provisionalAsr = null;
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
            int step = schedulerTarget;
            for (int off = 0; off < buffered.length; off += step) {
                int n = Math.min(step, buffered.length - off);
                float[] slice = new float[n];
                System.arraycopy(buffered, off, slice, 0, n);
                activeAsr.acceptAudio(slice);
                if (activeAsr.isReadyToDecode()) activeAsr.decodeAvailable();
            }
            StreamingAsrEngine.PartialResult p = activeAsr.getPartialResult();
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
            window = ring.lastMs(AsrState.BOOTSTRAP_MAX_MS);
        } catch (Exception e) {
            Log.w(TAG, "speculative candidates: no audio", e);
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
            String winnerText = "";
            float winnerConf = 0;
            for (AsrLanguage cand : pair) {
                if (cand == null || cand == AsrLanguage.UND) continue;
                CandidateScore s = scoreSpeculativeCandidate(cand, window);
                if (s == null) continue;
                float combined = 0.65f * s.confidence
                        + 0.35f * LanguageIdEngine.lexicalFit(s.text, cand);
                if (!s.text.isEmpty() && combined > winnerScore) {
                    winnerScore = combined;
                    winner = cand;
                    winnerText = s.text;
                    winnerConf = s.confidence;
                }
            }
            // Minimum bar: a lone weak decode must not force a language.
            if (winner != null && !winnerText.isEmpty() && winnerScore >= 0.30f) {
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

    /** Top-2 languages from a bootstrap score map (UND excluded). */
    private static AsrLanguage[] topTwo(Map<AsrLanguage, Float> scores) {
        AsrLanguage first = AsrLanguage.VI;
        AsrLanguage second = AsrLanguage.EN;
        float s1 = -1;
        float s2 = -1;
        if (scores != null) {
            for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
                Float v = scores.get(l);
                float f = v == null ? 0 : v;
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
     */
    private CandidateScore scoreSpeculativeCandidate(AsrLanguage cand, float[] window) {
        try {
            StreamingAsrEngine e = models.get(cand);
            boolean isLiveProvisional = (e == provisionalAsr);
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
            StreamingAsrEngine.PartialResult p = e.getPartialResult();
            CandidateScore s = new CandidateScore();
            s.text = p.text == null ? "" : p.text.trim();
            s.confidence = p.confidence;
            return s;
        } catch (Exception ex) {
            Log.w(TAG, "speculative decode failed for " + cand, ex);
            return null;
        }
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
        StreamingAsrEngine.PartialResult fin = activeAsr.getFinalResult();
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
            float[] spoken = utteranceAudio(AsrState.ENDPOINT_VERIFY_MS);
            float[] window = spoken.length > AsrState.SAMPLE_RATE
                    * AsrState.BOOTSTRAP_LID_WINDOW_MS / 1000
                    ? java.util.Arrays.copyOfRange(spoken, 0,
                            AsrState.SAMPLE_RATE * AsrState.BOOTSTRAP_LID_WINDOW_MS / 1000)
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
            if (decided == AsrLanguage.UND) {
                // Final backstop: full-utterance verification with the
                // provisional hypothesis as base (old-system behavior).
                String base = provisionalSpeculative == null
                        ? "" : provisionalSpeculative.trim();
                EndpointVerdict verdict = verifyEndpointLanguage(
                        base, provisionalConf, provisionalLang);
                if (verdict != null) {
                    if (verdict.lang != provisionalLang) {
                        activateBootstrap(verdict.lang, verdict.confidence, uid);
                    } else {
                        adoptProvisional(verdict.confidence, uid);
                    }
                    decided = activeLang;
                }
            }
            if (decided != AsrLanguage.UND && activeAsr != null) {
                flushScheduler();
                StreamingAsrEngine.PartialResult fin = activeAsr.getFinalResult();
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

    /**
     * Adopt the provisional stream as the committed active model (used when
     * endpoint verification confirms the provisional hypothesis). No reset,
     * no replay — the engine already holds the full utterance state.
     */
    private synchronized void adoptProvisional(float confidence, long uid) {
        if (uid != utteranceSeq.get() || activeLang != AsrLanguage.UND
                || provisionalAsr == null) {
            return;
        }
        activeAsr = provisionalAsr;
        provisionalAsr = null;
        activeLang = provisionalLang;
        router.commitBootstrap(activeLang, confidence);
        transcripts.updateSpeculative(provisionalSpeculative);
        provisionalSpeculative = "";
        Log.i(TAG, "endpoint adopted provisional " + activeLang.code
                + " conf=" + confidence);
    }

    /** Clear per-utterance state; next utterance bootstraps from UNKNOWN. */
    private void resetUtteranceState() {
        transcripts.reset();
        if (activeAsr != null) {
            try {
                activeAsr.reset();
            } catch (Exception e) {
                Log.w(TAG, "engine reset failed", e);
            }
        }
        activeAsr = null;
        activeLang = AsrLanguage.UND;
        router.reset();
        schedulerFill = 0;
        utteranceStartMs = -1;
        utteranceStartSample = -1;
        utteranceSpeechMs = 0;
        lastBootstrapSpeechMs = -1;
        lastCandidateRunSpeechMs = -1;
        lastCandidatePair = null;
        bootstrapPending = false;
        resetProvisional();
        utteranceSeq.incrementAndGet(); // invalidate in-flight LID results
    }

    /**
     * (Re)point the provisional decoder at the always-resident VI engine
     * with fresh state. Safe to call when the same object is also the (now
     * cleared) active engine — reset is idempotent.
     */
    private void resetProvisional() {
        provisionalLang = AsrLanguage.VI;
        provisionalSpeculative = "";
        provisionalConf = 0.33f;
        try {
            provisionalAsr = models.get(AsrLanguage.VI);
            provisionalAsr.reset();
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
            activeAsr.reset();
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
