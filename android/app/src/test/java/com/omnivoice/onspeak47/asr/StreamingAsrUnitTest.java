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
        LanguageIdEngine lid = new LanguageIdEngine((AcousticLidEngine) (audio) -> en, AsrState.EMA_ALPHA);
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

    // --- VoxLingua107 ECAPA LID tests (pipeline §4, §11–§16) ----------------

    @Test
    public void voxLinguaLabels_indicesAndMappings() {
        assertEquals(107, VoxLinguaLabels.NUM_LANGUAGES);
        assertEquals(20, VoxLinguaLabels.IDX_EN);
        assertEquals(102, VoxLinguaLabels.IDX_VI);
        assertEquals(106, VoxLinguaLabels.IDX_ZH);
        assertEquals("en", VoxLinguaLabels.codeAt(20));
        assertEquals("vi", VoxLinguaLabels.codeAt(102));
        assertEquals("zh", VoxLinguaLabels.codeAt(106));
        assertEquals(20, VoxLinguaLabels.indexOf("en"));
        assertEquals(102, VoxLinguaLabels.indexOf("vi"));
        assertEquals(106, VoxLinguaLabels.indexOf("zh"));

        assertTrue(VoxLinguaLabels.isSupported("vi"));
        assertTrue(VoxLinguaLabels.isSupported("en"));
        assertTrue(VoxLinguaLabels.isSupported("zh"));
        assertFalse(VoxLinguaLabels.isSupported("ja"));
        assertFalse(VoxLinguaLabels.isSupported("th"));
        assertFalse(VoxLinguaLabels.isSupported("unk"));

        assertEquals(AsrLanguage.VI, VoxLinguaLabels.toAsrLanguage("vi"));
        assertEquals(AsrLanguage.EN, VoxLinguaLabels.toAsrLanguage("en"));
        assertEquals(AsrLanguage.ZH, VoxLinguaLabels.toAsrLanguage("zh"));
        assertEquals(AsrLanguage.UND, VoxLinguaLabels.toAsrLanguage("ja"));
        assertEquals(AsrLanguage.UND, VoxLinguaLabels.toAsrLanguage(null));
    }

    @Test
    public void voxLinguaScores_bootstrapGateAndUnsupported() {
        LanguageScores flat = LanguageScores.flat();
        assertFalse(flat.isSupportedTop());
        assertFalse(flat.isBootstrapConfident());

        // Confident VI with supported global winner.
        LanguageScores confidentVi = new LanguageScores(0.82f, 0.13f, 0.05f,
                "vi", 102, 0.75f, 1);
        assertTrue(confidentVi.isSupportedTop());
        assertEquals(AsrLanguage.VI, confidentVi.topSupported());
        assertTrue(confidentVi.isBootstrapConfident());

        // Supported but margin too narrow (< 0.15).
        LanguageScores narrow = new LanguageScores(0.46f, 0.43f, 0.11f,
                "vi", 102, 0.40f, 1);
        assertTrue(narrow.isSupportedTop());
        assertFalse(narrow.isBootstrapConfident());

        // Unsupported global winner (e.g. Japanese): relative score looks high
        // among VI/EN/ZH, but global winner is NOT supported -> must NOT commit.
        LanguageScores jaWinner = new LanguageScores(0.82f, 0.13f, 0.05f,
                "ja", 45, 0.85f, 1);
        assertFalse(jaWinner.isSupportedTop());
        assertFalse(jaWinner.isBootstrapConfident());
    }

    @Test
    public void voxLinguaFbank_dimensionsAndMeanNorm() {
        VoxLinguaFbankExtractor extractor = new VoxLinguaFbankExtractor();
        assertEquals(0, extractor.extract(null).length);
        assertEquals(0, extractor.extract(new float[300]).length);

        // 1 second of audio @ 16 kHz.
        float[] audio = new float[16000];
        for (int i = 0; i < audio.length; i++) {
            audio[i] = (float) (0.2 * Math.sin(2 * Math.PI * 300 * i / 16000));
        }
        float[][] feats = extractor.extract(audio);
        int expectedFrames = VoxLinguaFbankExtractor.numFramesFor(16000);
        assertEquals(expectedFrames, feats.length);
        assertTrue(feats.length > 0);
        assertEquals(VoxLinguaFbankExtractor.N_MELS, feats[0].length);

        // Verify sentence-mean normalization: per-bin mean over time must be ~0.
        for (int m = 0; m < VoxLinguaFbankExtractor.N_MELS; m++) {
            double sum = 0;
            for (float[] frame : feats) {
                sum += frame[m];
            }
            double mean = sum / feats.length;
            assertEquals(0.0, mean, 1e-4);
        }
    }

    @Test
    public void voxLinguaTemporalSmoother_emaAndMajorityVote() {
        VoxLinguaTemporalSmoother smoother = new VoxLinguaTemporalSmoother();
        assertEquals(0, smoother.size());

        // Window 1: uncertain leaning EN.
        LanguageScores s1 = smoother.add(new LanguageScores(0.43f, 0.49f, 0.08f,
                "en", 20, 0.40f, 1));
        assertEquals(1, smoother.size());
        assertEquals(AsrLanguage.EN, s1.topSupported());

        // Window 2: strong EN.
        LanguageScores s2 = smoother.add(new LanguageScores(0.22f, 0.72f, 0.06f,
                "en", 20, 0.65f, 1));
        assertEquals(2, smoother.size());

        // Window 3: dominant EN (3 consecutive windows agreement -> majority vote boost).
        LanguageScores s3 = smoother.add(new LanguageScores(0.12f, 0.82f, 0.06f,
                "en", 20, 0.75f, 1));
        assertEquals(3, smoother.size());
        assertTrue(s3.isBootstrapConfident());
        assertEquals(AsrLanguage.EN, s3.topSupported());

        smoother.reset();
        assertEquals(0, smoother.size());
        assertFalse(smoother.smoothed().isSupportedTop());
    }

    @Test
    public void voxLinguaLidEngine_scoresFromLogits() {
        float[] logits = new float[VoxLinguaLabels.NUM_LANGUAGES];

        // 1. VI dominant.
        logits[VoxLinguaLabels.IDX_VI] = 10.0f;
        LanguageScores viScores = VoxLinguaAcousticLidEngine.scoresFromLogitsForTest(logits);
        assertEquals("vi", viScores.globalTopLanguage);
        assertTrue(viScores.isSupportedTop());
        assertEquals(AsrLanguage.VI, viScores.topSupported());
        assertTrue(viScores.isBootstrapConfident());

        // 2. EN dominant.
        logits[VoxLinguaLabels.IDX_VI] = 0f;
        logits[VoxLinguaLabels.IDX_EN] = 10.0f;
        LanguageScores enScores = VoxLinguaAcousticLidEngine.scoresFromLogitsForTest(logits);
        assertEquals("en", enScores.globalTopLanguage);
        assertTrue(enScores.isSupportedTop());
        assertEquals(AsrLanguage.EN, enScores.topSupported());
        assertTrue(enScores.isBootstrapConfident());

        // 3. ZH dominant.
        logits[VoxLinguaLabels.IDX_EN] = 0f;
        logits[VoxLinguaLabels.IDX_ZH] = 10.0f;
        LanguageScores zhScores = VoxLinguaAcousticLidEngine.scoresFromLogitsForTest(logits);
        assertEquals("zh", zhScores.globalTopLanguage);
        assertTrue(zhScores.isSupportedTop());
        assertEquals(AsrLanguage.ZH, zhScores.topSupported());
        assertTrue(zhScores.isBootstrapConfident());

        // 4. Unsupported global winner (Japanese, index 45 "ja").
        logits[VoxLinguaLabels.IDX_ZH] = 0f;
        int jaIdx = VoxLinguaLabels.indexOf("ja");
        assertTrue(jaIdx >= 0);
        logits[jaIdx] = 10.0f;
        LanguageScores jaScores = VoxLinguaAcousticLidEngine.scoresFromLogitsForTest(logits);
        assertEquals("ja", jaScores.globalTopLanguage);
        assertFalse(jaScores.isSupportedTop());
        assertFalse(jaScores.isBootstrapConfident());
    }

    @Test
    public void languageIdEngine_voxLinguaDetailedIntegration() {
        // Mock detailed engine returning unsupported Japanese top.
        AcousticLidEngine jaEngine = new AcousticLidEngine() {
            @Override
            public Map<AsrLanguage, Float> classify(float[] audioWindow) {
                return classifyDetailed(audioWindow).toMap();
            }

            @Override
            public LanguageScores classifyDetailed(float[] audioWindow) {
                return new LanguageScores(0.80f, 0.15f, 0.05f, "ja", 45, 0.85f, 1);
            }
        };
        LanguageIdEngine lidJa = new LanguageIdEngine(jaEngine, AsrState.EMA_ALPHA);
        LanguageIdEngine.LidResult resJa = lidJa.classifyBootstrap(new float[4800]);
        // Unsupported global top must map to UND and flat scores.
        assertEquals(AsrLanguage.UND, resJa.language);
        assertEquals(1.0f / 3, resJa.scores.get(AsrLanguage.VI), 1e-5f);

        // Mock detailed engine returning confident EN.
        AcousticLidEngine enEngine = new AcousticLidEngine() {
            @Override
            public Map<AsrLanguage, Float> classify(float[] audioWindow) {
                return classifyDetailed(audioWindow).toMap();
            }

            @Override
            public LanguageScores classifyDetailed(float[] audioWindow) {
                return new LanguageScores(0.05f, 0.90f, 0.05f, "en", 20, 0.88f, 1);
            }
        };
        LanguageIdEngine lidEn = new LanguageIdEngine(enEngine, AsrState.EMA_ALPHA);
        LanguageIdEngine.LidResult resEn = lidEn.classifyBootstrap(new float[4800]);
        assertEquals(AsrLanguage.EN, resEn.language);
        assertTrue(resEn.confidence >= AsrState.BOOTSTRAP_THRESHOLD);

        // Three-level interval.
        assertEquals(AsrState.LID_INTERVAL_CANDIDATE_MS,
                LanguageIdEngine.lidIntervalMs(false, true));
        assertEquals(AsrState.LID_INTERVAL_UNCERTAIN_MS,
                LanguageIdEngine.lidIntervalMs(true, false));
        assertEquals(AsrState.LID_INTERVAL_STABLE_MS,
                LanguageIdEngine.lidIntervalMs(false, false));
    }
}
