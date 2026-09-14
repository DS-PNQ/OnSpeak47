/*
 * OmniVoice — JVM unit tests for the pure-Java streaming ASR core.
 *
 * Mirrors tests_local/test_07_streaming_asr.py. These classes use no
 * android.* APIs, so they run as plain local unit tests
 * (./gradlew :app:testDebugUnitTest).
 */
package com.omnivoice.onspeak47.asr;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Test;

public class StreamingAsrUnitTest {

    // --- MUST 3: ring buffer + rollback window ---------------------------

    @Test
    public void ringBuffer_lastWindowSizes() {
        AudioRingBuffer ring = AudioRingBuffer.withCapacityMs(30_000);
        float[] oneSec = new float[AsrState.SAMPLE_RATE];
        for (int i = 0; i < oneSec.length; i++) oneSec[i] = 0.1f;
        ring.append(oneSec);
        assertEquals(AsrState.SAMPLE_RATE, ring.available());
        assertEquals(AsrState.SAMPLE_RATE * 640 / 1000, ring.lastMs(640).length);
        assertEquals(AsrState.SAMPLE_RATE * 960 / 1000, ring.lastMs(960).length);
        assertEquals(0, ring.droppedCount());
    }

    @Test
    public void ringBuffer_overwriteCountsDropped() {
        AudioRingBuffer ring = new AudioRingBuffer(1000);
        ring.append(new float[1500]);
        assertEquals(1000, ring.available());
        assertEquals(500, ring.droppedCount());
    }

    @Test
    public void ringBuffer_sinceReplaysFromUtteranceStart() {
        AudioRingBuffer ring = AudioRingBuffer.withCapacityMs(30_000);
        ring.append(new float[1600]); // silence prefix
        long start = ring.totalWritten();
        float[] speech = new float[3200];
        for (int i = 0; i < speech.length; i++) speech[i] = 0.2f;
        ring.append(speech);
        ring.append(new float[32000]); // trailing silence (endpoint delay)
        float[] replay = ring.since(start, AsrState.SAMPLE_RATE * 6);
        assertEquals(3200 + 32000, replay.length);
        assertEquals(0.2f, replay[0], 1e-6f);
    }

    @Test
    public void rollback_defaultsTo640ms() {
        AudioRingBuffer ring = AudioRingBuffer.withCapacityMs(30_000);
        float[] twoSec = new float[AsrState.SAMPLE_RATE * 2];
        ring.append(twoSec);
        RollbackManager rb = new RollbackManager();
        assertEquals(640, rb.windowMs());
        float[] audio = rb.startRollback(ring, AsrLanguage.EN);
        assertEquals(AsrState.SAMPLE_RATE * 640 / 1000, audio.length);
        assertTrue(rb.hasPending());
    }

    // --- Router: hysteresis + state machine -------------------------------

    private static LanguageRouter testRouter(AtomicLong now) {
        return new LanguageRouter(AsrState.SWITCH_THRESHOLD, AsrState.SWITCH_MARGIN,
                AsrState.SWITCH_PERSIST_MS, now::get);
    }

    @Test
    public void router_startsUnknownNeverVi() {
        // fakedemo2 §6: never ACTIVE+VI at start — UNKNOWN until bootstrap.
        LanguageRouter router = testRouter(new AtomicLong(0));
        assertEquals(AsrLanguage.UND, router.active());
        assertTrue(router.isBootstrapping());
        assertEquals(RouterState.UNKNOWN, router.state());
        router.reset();
        assertEquals(AsrLanguage.UND, router.active());
    }

    @Test
    public void router_runtimeGateHoldsWhileBootstrapping() {
        LanguageRouter router = testRouter(new AtomicLong(0));
        // Even a 0.95 candidate must not fire runtime hysteresis pre-bootstrap.
        assertEquals(LanguageRouter.Decision.HOLD,
                router.onLidResult(lid(AsrLanguage.EN, 0.95f)));
    }

    @Test
    public void router_bootstrapGateConfidentVsUncertain() {
        LanguageRouter router = testRouter(new AtomicLong(0));
        // Confident VI (0.82, margin 0.69) commits.
        Map<AsrLanguage, Float> confident = new EnumMap<>(AsrLanguage.class);
        confident.put(AsrLanguage.VI, 0.82f);
        confident.put(AsrLanguage.EN, 0.13f);
        confident.put(AsrLanguage.ZH, 0.05f);
        assertEquals(AsrLanguage.VI, router.onBootstrapLidResult(
                new LanguageIdEngine.LidResult(AsrLanguage.VI, 0.82f, confident)));
        assertEquals(AsrLanguage.VI, router.active());
        assertTrue(!router.isBootstrapping());

        // VI 0.46 / EN 0.43 must NOT force VI on max-probability (§8).
        router.reset();
        Map<AsrLanguage, Float> tie = new EnumMap<>(AsrLanguage.class);
        tie.put(AsrLanguage.VI, 0.46f);
        tie.put(AsrLanguage.EN, 0.43f);
        tie.put(AsrLanguage.ZH, 0.11f);
        assertEquals(AsrLanguage.UND, router.onBootstrapLidResult(
                new LanguageIdEngine.LidResult(AsrLanguage.VI, 0.46f, tie)));
        assertEquals(AsrLanguage.UND, router.active());
    }

    @Test
    public void router_commitBootstrapRejectsUnd() {
        LanguageRouter router = testRouter(new AtomicLong(0));
        router.commitBootstrap(AsrLanguage.UND, 0.9f);
        assertEquals(AsrLanguage.UND, router.active());
        router.commitBootstrap(AsrLanguage.EN, 0.8f);
        assertEquals(AsrLanguage.EN, router.active());
    }

    private static LanguageIdEngine.LidResult lid(AsrLanguage lang, float conf) {
        Map<AsrLanguage, Float> scores = new EnumMap<>(AsrLanguage.class);
        scores.put(AsrLanguage.VI, 0.1f);
        scores.put(AsrLanguage.EN, 0.1f);
        scores.put(AsrLanguage.ZH, 0.1f);
        scores.put(lang, conf);
        return new LanguageIdEngine.LidResult(lang, conf, scores);
    }

    @Test
    public void router_rejectsWeakCandidate() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = testRouter(now);
        router.setActive(AsrLanguage.VI, 0.5f);
        assertEquals(LanguageRouter.Decision.HOLD,
                router.onLidResult(lid(AsrLanguage.EN, 0.5f)));
    }

    @Test
    public void router_persistenceThenCommit() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = testRouter(now);
        router.setActive(AsrLanguage.VI, 0.3f);
        LanguageIdEngine.LidResult strong = lid(AsrLanguage.EN, 0.85f);
        assertEquals(LanguageRouter.Decision.OBSERVE, router.onLidResult(strong));
        now.set(AsrState.SWITCH_PERSIST_MS - 50);
        assertEquals(LanguageRouter.Decision.OBSERVE, router.onLidResult(strong));
        now.set(AsrState.SWITCH_PERSIST_MS + 100);
        assertEquals(LanguageRouter.Decision.START_ROLLBACK, router.onLidResult(strong));
        router.setCandidateVerifier((c, audio) ->
                new LanguageRouter.Verification(true, "is very beautiful", 0.8f));
        assertEquals(LanguageRouter.Decision.COMMITTED,
                router.verifyAndCommit(new float[100]));
        assertEquals(AsrLanguage.EN, router.active());
        assertEquals(1, router.switchCount());
    }

    @Test
    public void router_discardsFailedVerificationWithCooldown() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = testRouter(now);
        router.setActive(AsrLanguage.VI, 0.3f);
        LanguageIdEngine.LidResult strong = lid(AsrLanguage.EN, 0.9f);
        router.onLidResult(strong);
        now.set(AsrState.SWITCH_PERSIST_MS + 10);
        assertEquals(LanguageRouter.Decision.START_ROLLBACK, router.onLidResult(strong));
        router.setCandidateVerifier((c, audio) ->
                new LanguageRouter.Verification(false, "", 0.1f));
        assertEquals(LanguageRouter.Decision.DISCARDED,
                router.verifyAndCommit(new float[100]));
        assertEquals(AsrLanguage.VI, router.active());
        // Same rejected candidate is held during cooldown (no decode storm).
        assertEquals(LanguageRouter.Decision.HOLD, router.onLidResult(strong));
    }

    @Test
    public void router_endpointSwitchNeedsNoRollback() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = testRouter(now);
        router.setActive(AsrLanguage.VI, 0.4f);
        Map<AsrLanguage, Float> scores = new EnumMap<>(AsrLanguage.class);
        scores.put(AsrLanguage.VI, 0.1f);
        scores.put(AsrLanguage.EN, 0.8f);
        scores.put(AsrLanguage.ZH, 0.1f);
        assertTrue(router.onEndpoint(scores));
        assertEquals(AsrLanguage.EN, router.active());
    }

    @Test
    public void router_endpointBarLowerThanIntraGate() {
        // 0.65 never passes the intra-utterance 0.72 + persistence gate, but
        // at a VAD boundary switching is cheap (spec §13.1) so it commits.
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = testRouter(now);
        router.setActive(AsrLanguage.VI, 0.4f);
        assertEquals(LanguageRouter.Decision.HOLD,
                router.onLidResult(lid(AsrLanguage.EN, 0.65f)));
        Map<AsrLanguage, Float> scores = new EnumMap<>(AsrLanguage.class);
        scores.put(AsrLanguage.VI, 0.25f);
        scores.put(AsrLanguage.EN, 0.65f);
        scores.put(AsrLanguage.ZH, 0.1f);
        assertTrue(router.onEndpoint(scores));
        assertEquals(AsrLanguage.EN, router.active());
    }

    // --- Partial tiers -----------------------------------------------------

    @Test
    public void partial_stabilizesAfterTwoUpdates() {
        PartialTranscriptManager pm = new PartialTranscriptManager();
        pm.updateSpeculative("xin");
        assertEquals(TranscriptTier.SPECULATIVE, pm.tier());
        pm.updateSpeculative("xin");
        pm.updateSpeculative("xin chào");
        pm.updateSpeculative("xin chào");
        assertTrue(pm.getCommitted().contains("xin"));
        assertTrue(pm.getDisplay().startsWith("xin"));
    }

    @Test
    public void partial_stripsCommittedPrefix() {
        PartialTranscriptManager pm = new PartialTranscriptManager();
        pm.updateSpeculative("hôm nay tôi có");
        pm.updateSpeculative("hôm nay tôi có");
        assertEquals("hôm nay tôi có", pm.getCommitted());
        // Transducer partials re-emit the hypothesis: must not duplicate.
        pm.updateSpeculative("hôm nay tôi có");
        pm.updateSpeculative("hôm nay tôi có meeting");
        pm.updateSpeculative("hôm nay tôi có meeting");
        assertEquals("hôm nay tôi có", pm.getCommitted());
        assertFalse(pm.getDisplay().contains("có hôm"));
    }

    @Test
    public void partial_rollbackAndFinalize() {
        PartialTranscriptManager pm = new PartialTranscriptManager();
        pm.updateSpeculative("hôm nay trời");
        pm.rollbackSpeculative();
        assertEquals("", pm.getSpeculative());
        pm.commit("is very beautiful");
        assertEquals("is very beautiful", pm.getCommitted());
        pm.finalizeTranscript("Hôm nay trời is very beautiful");
        assertEquals(TranscriptTier.FINAL, pm.tier());
    }

    // --- Metrics ------------------------------------------------------------

    @Test
    public void metrics_percentilesAndTargets() {
        AsrMetrics m = new AsrMetrics();
        m.addPartialLatency(200);
        m.addPartialLatency(250);
        m.addPartialLatency(300);
        m.addPartialLatency(350);
        m.addPartialLatency(400);
        assertEquals(300.0, m.partialP50(), 1e-9);
        assertTrue(m.meetsLatencyTargets());
        AsrMetrics bad = new AsrMetrics();
        bad.addPartialLatency(600);
        bad.addPartialLatency(700);
        assertFalse(bad.meetsLatencyTargets());
    }

    // --- LID -----------------------------------------------------------------

    @Test
    public void lid_zhBoostedByCjk() {
        LanguageIdEngine lid = new LanguageIdEngine();
        LanguageIdEngine.LidResult r =
                lid.classify(new float[100], "你好我今天去开会", 0.5f, AsrLanguage.VI);
        assertEquals(AsrLanguage.ZH, r.language);
        assertTrue(r.scores.get(AsrLanguage.ZH) > 0.5f);
    }

    @Test
    public void lid_intervalAdaptive() {
        assertEquals(600, LanguageIdEngine.lidIntervalMs(false));
        assertEquals(400, LanguageIdEngine.lidIntervalMs(true));
    }

    // --- Bootstrap acoustic LID (fakedemo2 §16–§17) -------------------------

    @Test
    public void bootstrap_usesAudioNotTranscript() {
        // EN acoustic evidence wins even though no transcript exists yet —
        // bootstrap never consults text (§17).
        Map<AsrLanguage, Float> en = new EnumMap<>(AsrLanguage.class);
        en.put(AsrLanguage.VI, 0.05f);
        en.put(AsrLanguage.EN, 0.90f);
        en.put(AsrLanguage.ZH, 0.05f);
        LanguageIdEngine lid = new LanguageIdEngine((audio) -> en, AsrState.EMA_ALPHA);
        LanguageIdEngine.LidResult r = lid.classifyBootstrap(new float[4800]);
        assertEquals(AsrLanguage.EN, r.language);
        assertTrue(r.confidence >= AsrState.BOOTSTRAP_THRESHOLD);
    }

    @Test
    public void bootstrap_withoutScorerStaysFlatNeverViLocked() {
        // No acoustic model: flat prior → router stays UNCERTAIN (extend /
        // dual-candidate), never forced VI (§8, §19).
        LanguageIdEngine lid = new LanguageIdEngine();
        LanguageIdEngine.LidResult r = lid.classifyBootstrap(new float[4800]);
        assertEquals(1.0f / 3, r.scores.get(AsrLanguage.VI), 1e-6f);
        assertEquals(1.0f / 3, r.scores.get(AsrLanguage.EN), 1e-6f);
        LanguageRouter router = testRouter(new AtomicLong(0));
        assertEquals(AsrLanguage.UND, router.onBootstrapLidResult(r));
    }

    @Test
    public void heuristicLid_isAudioDerivedAndNeverConfident() {
        AcousticLidEngine h = new AcousticLidEngine.HeuristicAcousticLidEngine();
        // Silence → flat.
        Map<AsrLanguage, Float> silence = h.classify(new float[4800]);
        assertEquals(1.0f / 3, silence.get(AsrLanguage.VI), 1e-6f);
        // Voiced-like signal → leans tonal but stays below the commit bar.
        float[] voiced = new float[4800];
        for (int i = 0; i < voiced.length; i++) {
            voiced[i] = (float) (0.3 * Math.sin(2 * Math.PI * 120 * i / 16000));
        }
        Map<AsrLanguage, Float> s = h.classify(voiced);
        float top = Math.max(s.get(AsrLanguage.VI),
                Math.max(s.get(AsrLanguage.EN), s.get(AsrLanguage.ZH)));
        assertTrue(top <= 0.58f + 1e-6f);
        assertTrue(top >= 1.0f / 3 - 1e-6f);
        // Empty input → flat, never null.
        assertTrue(h.classify(new float[0]) != null);
        assertTrue(h.classify(null) != null);
    }
}
