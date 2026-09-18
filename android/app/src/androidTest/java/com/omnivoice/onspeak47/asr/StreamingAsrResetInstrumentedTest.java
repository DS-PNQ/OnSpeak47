package com.omnivoice.onspeak47.asr;

import android.content.Context;
import android.util.Log;

import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static org.junit.Assert.*;

/** Real Android/JNI regression probe; never substitutes FakeEngine. */
public class StreamingAsrResetInstrumentedTest {
    @Test
    public void sameFramesAfterResetProduceSameTranscript() throws Exception {
        Context target = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context tests = InstrumentationRegistry.getInstrumentation().getContext();
        float[] frames = loadFrames(tests);
        ZipformerModelManager models = new ZipformerModelManager(target);
        try {
            StreamingAsrEngine engine = models.get(AsrLanguage.EN);
            assertFalse("EN native engine must load; FakeEngine is not a passing test", engine.isFake());
            String before = feed(engine, frames);
            assertFalse("Baseline EN transcript must not be empty", before.trim().isEmpty());
            engine.reset();
            assertEquals("Reset must clear visible text", "", engine.getPartialResult().text);
            String after = feed(engine, frames);
            Log.i("AsrResetTest", "frames=" + frames.length / AsrState.SCHEDULER_SAMPLES
                    + " before='" + before + "' after='" + after + "'");
            assertEquals("Identical PCM after reset must reproduce the baseline", before, after);
        } finally {
            models.close();
        }
    }

    private static String feed(StreamingAsrEngine engine, float[] frames) {
        for (int offset = 0; offset < frames.length; offset += AsrState.SCHEDULER_SAMPLES) {
            engine.acceptAudio(Arrays.copyOfRange(frames, offset,
                    offset + AsrState.SCHEDULER_SAMPLES));
            // Match the production scheduler: one readiness-gated decode per tick.
            if (engine.isReadyToDecode()) engine.decodeAvailable();
        }
        return engine.getPartialResult().text;
    }

    private static float[] loadFrames(Context context) throws Exception {
        byte[] wav;
        try (InputStream in = context.getAssets().open("parity_en.wav");
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int n;
            while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
            wav = out.toByteArray();
        }
        assertTrue("Truncated WAV", wav.length >= 12);
        assertEquals("RIFF", tag(wav, 0));
        assertEquals("WAVE", tag(wav, 8));
        ByteBuffer bytes = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        boolean formatOk = false;
        int dataOffset = -1;
        int dataLength = 0;
        for (int offset = 12; offset <= wav.length - 8;) {
            String tag = tag(wav, offset);
            int length = bytes.getInt(offset + 4);
            assertTrue("Invalid WAV chunk", length >= 0 && length <= wav.length - offset - 8);
            int body = offset + 8;
            if ("fmt ".equals(tag)) {
                assertTrue(length >= 16);
                assertEquals(1, bytes.getShort(body)); // PCM
                assertEquals(1, bytes.getShort(body + 2)); // mono
                assertEquals(AsrState.SAMPLE_RATE, bytes.getInt(body + 4));
                assertEquals(16, bytes.getShort(body + 14));
                formatOk = true;
            } else if ("data".equals(tag)) {
                dataOffset = body;
                dataLength = length;
            }
            offset = body + length + (length & 1);
        }
        assertTrue("Expected PCM format and nonempty data", formatOk && dataOffset >= 0 && dataLength > 0);
        assertEquals(0, dataLength % 2);
        int samples = dataLength / 2;
        // A fixed two-second silence tail flushes the utterance in BOTH passes.
        // The entire sequence is aligned to scheduler frames, with no timing dependencies.
        int step = AsrState.SCHEDULER_SAMPLES;
        int count = (samples + 2 * AsrState.SAMPLE_RATE + step - 1) / step;
        float[] result = new float[count * step];
        for (int i = 0; i < samples; i++) result[i] = bytes.getShort(dataOffset + i * 2) / 32768.0f;
        return result;
    }

    private static String tag(byte[] bytes, int offset) {
        return new String(bytes, offset, 4, StandardCharsets.US_ASCII);
    }
}
