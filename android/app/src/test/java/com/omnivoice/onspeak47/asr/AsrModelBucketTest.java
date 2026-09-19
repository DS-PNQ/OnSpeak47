/*
 * OmniVoice — JVM tests for the mixed EN/ZH model bucket (routing plan §2–§5).
 *
 * Pins the behaviour the shared bilingual model depends on:
 *   - EN and ZH are ONE routing unit (AsrModelType.EN_ZH), VI is the other;
 *   - a LID flip between en and zh never triggers a model switch, a rollback
 *     window or a shadow decode (the "code-switched utterance" case);
 *   - VI ↔ EN_ZH switching still works exactly as before.
 *
 * Pure-Java (no android.* APIs), so these run as local unit tests
 * (./gradlew :app:testDebugUnitTest).
 */
package com.omnivoice.onspeak47.asr;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Test;

public class AsrModelBucketTest {

    // --- AsrModelType mapping ---------------------------------------------

    @Test
    public void modelType_enAndZhShareOneBucket() {
        assertEquals(AsrModelType.EN_ZH, AsrLanguage.EN.modelType());
        assertEquals(AsrModelType.EN_ZH, AsrLanguage.ZH.modelType());
        assertEquals(AsrModelType.VI, AsrLanguage.VI.modelType());
        assertEquals(AsrModelType.UND, AsrLanguage.UND.modelType());
        assertEquals(AsrModelType.UND, AsrModelType.of(null));
    }

    @Test
    public void modelType_coversAndCanonicalLabel() {
        assertTrue(AsrModelType.EN_ZH.covers(AsrLanguage.EN));
        assertTrue(AsrModelType.EN_ZH.covers(AsrLanguage.ZH));
        assertFalse(AsrModelType.EN_ZH.covers(AsrLanguage.VI));
        assertTrue(AsrModelType.VI.covers(AsrLanguage.VI));
        // EN is the canonical label of the shared bucket (engine/asset lookup).
        assertEquals(AsrLanguage.EN, AsrLanguage.canonicalOf(AsrModelType.EN_ZH));
        assertEquals(AsrLanguage.VI, AsrLanguage.canonicalOf(AsrModelType.VI));
        // Stable ids round-trip for telemetry.
        assertEquals(AsrModelType.EN_ZH, AsrModelType.fromCode("en_zh"));
        assertEquals(AsrModelType.VI, AsrModelType.fromCode("vi"));
        assertEquals(AsrModelType.UND, AsrModelType.fromCode("klingon"));
        assertEquals(AsrModelType.UND, AsrModelType.fromCode(null));
    }

    @Test
    public void modelType_isPerAssetSet_neverGlobal() {
        // A wrong model_type aborts natively (no Java exception), so pin it:
        // the shared bilingual export is zipformer v1, the VI export is v2.
        assertEquals("zipformer", AsrState.transducerModelType(AsrLanguage.EN));
        assertEquals("zipformer", AsrState.transducerModelType(AsrLanguage.ZH));
        assertEquals("zipformer2", AsrState.transducerModelType(AsrLanguage.VI));
    }

    // --- LanguageScores: non-VI bucket ------------------------------------

    @Test
    public void scores_nonViScoreIsMaxOfEnZh() {
        LanguageScores s = new LanguageScores(0.10f, 0.55f, 0.35f,
                0.01f, 0.05f, 0.03f, "en", 20, 0.30f, 1);
        assertEquals(0.55f, s.getNonViScore(), 1e-6f);
        assertEquals(AsrModelType.EN_ZH, s.getTopModelType());

        LanguageScores viTop = new LanguageScores(0.80f, 0.12f, 0.08f,
                0.50f, 0.02f, 0.01f, "vi", 102, 0.90f, 1);
        assertEquals(AsrModelType.VI, viTop.getTopModelType());
        assertTrue(viTop.getNonViScore() < viTop.vi);
    }

    @Test
    public void scores_zhTopStillRoutesToSharedBucket() {
        // Chinese-only scores: the bucket is EN_ZH (the same model decodes it),
        // while the language label stays ZH for display.
        LanguageScores zh = new LanguageScores(0.05f, 0.25f, 0.70f,
                0.01f, 0.03f, 0.40f, "zh", 106, 0.80f, 1);
        assertEquals(AsrModelType.EN_ZH, zh.getTopModelType());
        assertEquals(AsrLanguage.ZH, zh.topSupported());
        assertEquals(0.70f, zh.getNonViScore(), 1e-6f);
    }

    // --- Router: bucket guard (the point of the whole change) --------------

    private static LanguageRouter router(AtomicLong now) {
        return new LanguageRouter(AsrState.SWITCH_THRESHOLD, AsrState.SWITCH_MARGIN,
                AsrState.SWITCH_PERSIST_MS, now::get);
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
    public void router_neverSwitchesBetweenEnAndZh() {
        // Spec §4: the model is the same, so this must NEVER reach the
        // rollback/verify path no matter how confident or how long it persists.
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = router(now);
        router.setActive(AsrLanguage.EN, 0.4f);

        LanguageIdEngine.LidResult zhIsTop = lid(AsrLanguage.ZH, 0.95f);
        assertEquals(LanguageRouter.Decision.HOLD, router.onLidResult(zhIsTop));
        now.set(AsrState.SWITCH_PERSIST_MS * 4);
        assertEquals(LanguageRouter.Decision.HOLD, router.onLidResult(zhIsTop));
        now.set(AsrState.SWITCH_PERSIST_MS * 10);
        assertEquals(LanguageRouter.Decision.HOLD, router.onLidResult(zhIsTop));
        assertEquals(0, router.switchCount());
        assertEquals(AsrLanguage.EN, router.active());

        // …and symmetrically ZH → EN.
        router.setActive(AsrLanguage.ZH, 0.4f);
        LanguageIdEngine.LidResult enIsTop = lid(AsrLanguage.EN, 0.95f);
        now.set(AsrState.SWITCH_PERSIST_MS * 20);
        assertEquals(LanguageRouter.Decision.HOLD, router.onLidResult(enIsTop));
        assertEquals(0, router.switchCount());
        assertEquals(AsrLanguage.ZH, router.active());
    }

    @Test
    public void router_codeSwitchedUtteranceKeepsOneModel() {
        // "I will go to 北京": VoxLingua oscillates between en and zh while VI
        // stays low. None of these may look like a switch candidate.
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = router(now);
        router.setActive(AsrLanguage.EN, 0.75f);
        AsrLanguage[] flips = {AsrLanguage.ZH, AsrLanguage.EN, AsrLanguage.ZH, AsrLanguage.EN};
        for (int i = 0; i < flips.length; i++) {
            now.set(AsrState.SWITCH_PERSIST_MS * (i + 1));
            assertEquals(LanguageRouter.Decision.HOLD,
                    router.onLidResult(lid(flips[i], 0.9f)));
        }
        assertEquals(0, router.switchCount());
        assertEquals(AsrLanguage.EN, router.active());
    }

    @Test
    public void router_stillSwitchesViToSharedBucket() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = router(now);
        router.setActive(AsrLanguage.VI, 0.3f);
        LanguageIdEngine.LidResult strong = lid(AsrLanguage.ZH, 0.85f);
        assertEquals(LanguageRouter.Decision.OBSERVE, router.onLidResult(strong));
        now.set(AsrState.SWITCH_PERSIST_MS + 100);
        assertEquals(LanguageRouter.Decision.START_ROLLBACK, router.onLidResult(strong));
        router.setCandidateVerifier((c, audio) -> {
            // The candidate reaches the verifier as a real language label.
            assertEquals(AsrLanguage.ZH, c);
            return new LanguageRouter.Verification(true, "你好", 0.8f);
        });
        assertEquals(LanguageRouter.Decision.COMMITTED, router.verifyAndCommit(new float[100]));
        assertEquals(AsrLanguage.ZH, router.active());
        assertEquals(1, router.switchCount());
    }

    @Test
    public void router_stillSwitchesSharedBucketToVi() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = router(now);
        router.setActive(AsrLanguage.ZH, 0.2f);
        LanguageIdEngine.LidResult strongVi = lid(AsrLanguage.VI, 0.9f);
        assertEquals(LanguageRouter.Decision.OBSERVE, router.onLidResult(strongVi));
        now.set(AsrState.SWITCH_PERSIST_MS + 50);
        assertEquals(LanguageRouter.Decision.START_ROLLBACK, router.onLidResult(strongVi));
        assertNotEquals(AsrLanguage.UND, router.candidate());
        assertEquals(AsrLanguage.VI, router.candidate());
    }

    @Test
    public void router_endpointIgnoresLabelFlipInsideBucket() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = router(now);
        router.setActive(AsrLanguage.EN, 0.5f);

        Map<AsrLanguage, Float> zhLead = new EnumMap<>(AsrLanguage.class);
        zhLead.put(AsrLanguage.VI, 0.10f);
        zhLead.put(AsrLanguage.EN, 0.25f);
        zhLead.put(AsrLanguage.ZH, 0.65f);
        assertFalse(router.onEndpoint(zhLead));
        assertEquals(AsrLanguage.EN, router.active());
        assertEquals(0, router.switchCount());
    }

    @Test
    public void router_endpointStillSwitchesAcrossBuckets() {
        AtomicLong now = new AtomicLong(0);
        LanguageRouter router = router(now);
        router.setActive(AsrLanguage.VI, 0.4f);

        Map<AsrLanguage, Float> enLead = new EnumMap<>(AsrLanguage.class);
        enLead.put(AsrLanguage.VI, 0.25f);
        enLead.put(AsrLanguage.EN, 0.65f);
        enLead.put(AsrLanguage.ZH, 0.10f);
        assertTrue(router.onEndpoint(enLead));
        assertEquals(AsrLanguage.EN, router.active());
        assertEquals(1, router.switchCount());

        // Same for a ZH-led map: VI ↔ EN_ZH is the only guard the router has.
        router.setActive(AsrLanguage.VI, 0.4f);
        Map<AsrLanguage, Float> zhLead = new EnumMap<>(AsrLanguage.class);
        zhLead.put(AsrLanguage.VI, 0.20f);
        zhLead.put(AsrLanguage.EN, 0.15f);
        zhLead.put(AsrLanguage.ZH, 0.65f);
        assertTrue(router.onEndpoint(zhLead));
        assertEquals(AsrLanguage.ZH, router.active());
    }
}