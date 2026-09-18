package com.omnivoice.onspeak47.asr;

import java.util.Arrays;

/** Lossless fixed-size PCM framing. Timestamps are sample offsets, never wall time. */
final class VadWindowBuffer {
    interface Consumer {
        void accept(float[] window, long endSample);
    }

    private final float[] pending;
    private int fill;
    private long samples;

    VadWindowBuffer(int size) {
        if (size <= 0) throw new IllegalArgumentException("size must be positive");
        pending = new float[size];
    }

    void append(float[] audio, Consumer consumer) {
        int offset = 0;
        while (offset < audio.length) {
            int count = Math.min(pending.length - fill, audio.length - offset);
            System.arraycopy(audio, offset, pending, fill, count);
            fill += count;
            samples += count;
            offset += count;
            if (fill == pending.length) {
                fill = 0;
                consumer.accept(Arrays.copyOf(pending, pending.length), samples);
            }
        }
    }

    void reset() {
        fill = 0;
        samples = 0;
        Arrays.fill(pending, 0f);
    }
}
