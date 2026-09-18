package com.omnivoice.onspeak47.asr;

import org.junit.Test;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.Assert.*;

/** Runs the real framing + gate with deterministic window probabilities (not native inference). */
public class VadEndpointTest {
    private static float[] frame(float value) {
        float[] pcm = new float[320];
        Arrays.fill(pcm, value);
        return pcm;
    }

    private static VadEngine engine() {
        return new VadEngine(window -> {
            for (float v : window) if (v != 0f) return 0.9;
            return 0.1;
        });
    }

    @Test
    public void incompleteWindowDoesNotRunInferenceOrEmitEvents() {
        AtomicInteger calls = new AtomicInteger();
        VadEngine vad = new VadEngine(window -> {
            assertEquals(512, window.length);
            calls.incrementAndGet();
            return 0.9;
        });
        VadEngine.VadResult first = vad.process(frame(0.2f), 0);
        assertFalse(first.speechStart);
        assertFalse(first.speechEnd);
        assertEquals(0, calls.get());
        assertTrue(vad.process(frame(0.2f), 20).speechStart);
        assertEquals(1, calls.get());
        VadEngine.VadResult third = vad.process(frame(0.2f), 40);
        assertTrue(third.isSpeech);
        assertFalse(third.speechStart);
        assertEquals(1, calls.get());
    }

    @Test
    public void threePhrasesWith200msPausesStayOneUtterance() {
        VadEngine vad = engine();
        int starts = 0;
        int ends = 0;
        for (int phrase = 0; phrase < 3; phrase++) {
            for (int i = 0; i < 50; i++) {
                VadEngine.VadResult r = vad.process(frame(0.2f), Long.MAX_VALUE);
                if (r.speechStart) starts++;
                if (r.speechEnd) ends++;
            }
            for (int i = 0; i < 10; i++) {
                VadEngine.VadResult r = vad.process(frame(0f), 0);
                assertTrue(r.isSpeech);
                assertFalse(r.speechEnd);
            }
        }
        assertEquals(1, starts);
        assertEquals(0, ends);
        for (int i = 0; i < 120; i++) {
            if (vad.process(frame(0f), 0).speechEnd) ends++;
        }
        assertEquals(1, ends);
        assertFalse(vad.inSpeech());
    }

    @Test
    public void endpointRequiresAudioSilenceNotWallClockGap() {
        VadEngine vad = engine();
        for (int i = 0; i < 8; i++) vad.process(frame(0.2f), 0); // exact FIFO boundary
        for (int i = 1; i <= 100; i++) {
            VadEngine.VadResult r = vad.process(frame(0f), Long.MAX_VALUE);
            assertFalse("Endpoint before 2000ms below threshold at frame " + i, r.speechEnd);
            assertTrue(r.isSpeech);
        }
        int ends = 0;
        for (int i = 0; i < 6; i++) {
            if (vad.process(frame(0f), -1000).speechEnd) ends++;
        }
        assertEquals(1, ends); // 32ms window quantization + callback delivery
        assertFalse(vad.inSpeech());
    }

    @Test
    public void resetClearsFifoProbabilityAndEndpointState() {
        VadEngine vad = engine();
        vad.process(frame(0.2f), 0);
        vad.process(frame(0.2f), 20);
        assertTrue(vad.inSpeech());
        vad.reset();
        assertFalse(vad.inSpeech());
        assertEquals(0f, vad.lastProb(), 0f);
        VadEngine.VadResult r = vad.process(frame(0f), Long.MAX_VALUE);
        assertFalse(r.isSpeech);
        assertFalse(r.speechStart);
        assertFalse(r.speechEnd);
        assertEquals(0f, r.prob, 0f);
        assertFalse(vad.process(frame(0f), 0).isSpeech);
    }

    @Test
    public void realPcmWithInsertedPausesUsesAudioTime() throws java.io.IOException {
        float[] source = VadWindowBufferTest.realAudio();
        assertTrue(source.length >= 10560);
        float[] phrase = Arrays.copyOfRange(source, 8000, 10560);
        VadEngine vad = engine(); // Deterministic scorer, NOT a Silero accuracy test.
        for (int repeat = 0; repeat < 3; repeat++) {
            for (int off = 0; off < phrase.length; off += 320) {
                assertFalse(vad.process(Arrays.copyOfRange(phrase, off, off + 320), 0).speechEnd);
            }
            assertTrue(vad.inSpeech());
            for (int i = 0; i < 10; i++) { // 200ms of inserted PCM silence
                assertFalse(vad.process(frame(0f), Long.MAX_VALUE).speechEnd);
            }
        }
        // An additional 1800ms completes 2000ms silence; allow window quantization.
        for (int i = 0; i < 90; i++) {
            assertFalse(vad.process(frame(0f), -1).speechEnd);
        }
        int ends = 0;
        for (int i = 0; i < 6; i++) {
            if (vad.process(frame(0f), -1).speechEnd) ends++;
        }
        assertEquals(1, ends);
        assertFalse(vad.inSpeech());
    }

}
