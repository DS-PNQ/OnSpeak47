package com.omnivoice.onspeak47.asr;

import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/** Tests production FIFO with PCM from the repository's real 16 kHz WAV fixture. */
public class VadWindowBufferTest {
    static float[] realAudio() throws IOException {
        Path path = Path.of(System.getProperty("onspeak.testAudio"));
        ByteBuffer wav = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0x46464952, wav.getInt()); // RIFF
        wav.getInt();
        assertEquals(0x45564157, wav.getInt()); // WAVE
        boolean formatChecked = false;
        while (wav.remaining() >= 8) {
            int id = wav.getInt();
            int size = wav.getInt();
            int start = wav.position();
            assertTrue(size >= 0 && size <= wav.remaining());
            if (id == 0x20746d66) { // fmt
                assertTrue(size >= 16);
                assertEquals(1, wav.getShort()); // PCM
                assertEquals(1, wav.getShort()); // mono
                assertEquals(16000, wav.getInt());
                wav.getInt();
                wav.getShort();
                assertEquals(16, wav.getShort());
                formatChecked = true;
            } else if (id == 0x61746164) { // data
                assertTrue(formatChecked);
                float[] samples = new float[size / 2];
                for (int i = 0; i < samples.length; i++) samples[i] = wav.getShort() / 32768f;
                return samples;
            }
            wav.position(start + size + (size & 1));
        }
        throw new IOException("Missing PCM data chunk: " + path);
    }

    @Test
    public void exactly512RealSamplesAreEmittedLosslessly() throws IOException {
        float[] pcm = realAudio();
        assertTrue(pcm.length >= 8512);
        float[] input = Arrays.copyOfRange(pcm, 8000, 8512);
        boolean nonzero = false;
        for (float v : input) nonzero |= v != 0;
        assertTrue("Fixture must contain actual audio, not just silence", nonzero);
        List<float[]> output = new ArrayList<>();
        List<Long> ends = new ArrayList<>();
        new VadWindowBuffer(512).append(input, (window, end) -> {
            output.add(window);
            ends.add(end);
        });
        assertEquals(1, output.size());
        assertEquals(512, output.get(0).length);
        assertArrayEquals(input, output.get(0), 0f);
        assertEquals(Long.valueOf(512), ends.get(0));
        assertNotSame(input, output.get(0));
    }

    @Test
    public void captureFramesPreserveOrderWithoutLossOrDuplication() throws IOException {
        float[] pcm = Arrays.copyOf(realAudio(), 2560); // 8 capture frames = 5 VAD windows
        VadWindowBuffer fifo = new VadWindowBuffer(512);
        List<float[]> output = new ArrayList<>();
        for (int off = 0; off < pcm.length; off += 320) {
            fifo.append(Arrays.copyOfRange(pcm, off, off + 320), (window, end) -> {
                output.add(window);
                assertEquals(output.size() * 512L, end);
            });
        }
        assertEquals(5, output.size());
        for (int i = 0; i < output.size(); i++) {
            assertArrayEquals(Arrays.copyOfRange(pcm, i * 512, (i + 1) * 512), output.get(i), 0f);
        }
    }

    @Test
    public void resetDiscardsPartialWindowAndRestartsSampleClock() {
        VadWindowBuffer fifo = new VadWindowBuffer(512);
        fifo.append(new float[320], (window, end) -> fail("Incomplete window emitted"));
        fifo.reset();
        float[] fresh = new float[512];
        Arrays.fill(fresh, 0.25f);
        List<float[]> output = new ArrayList<>();
        fifo.append(fresh, (window, end) -> {
            output.add(window);
            assertEquals(512L, end);
        });
        assertEquals(1, output.size());
        assertArrayEquals(fresh, output.get(0), 0f);
    }
}
