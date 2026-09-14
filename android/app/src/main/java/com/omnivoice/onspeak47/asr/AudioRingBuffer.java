/*
 * OmniVoice — Audio ring buffer for streaming ASR.
 *
 * Spec §21 MUST 3 + §14: keeps recent audio for rollback / candidate
 * verification without re-reading a file. Thread-safe, lock-free-ish via a
 * single intrinsic lock. Pure-Java (no android.* imports).
 */
package com.omnivoice.onspeak47.asr;

/**
 * Fixed-capacity circular buffer of mono 16 kHz float samples.
 * Oldest samples are overwritten once full (no blocking).
 */
public class AudioRingBuffer {

    private final float[] buf;
    private final int capacity;
    private long writePos = 0; // total samples ever appended
    private long droppedSamples = 0;

    /** @param capacitySamples e.g. 30 s * 16000 = 480000. */
    public AudioRingBuffer(int capacitySamples) {
        if (capacitySamples <= 0) throw new IllegalArgumentException("capacity must be > 0");
        this.capacity = capacitySamples;
        this.buf = new float[capacitySamples];
    }

    public static AudioRingBuffer withCapacityMs(int ms) {
        return new AudioRingBuffer((int) ((long) AsrState.SAMPLE_RATE * ms / 1000));
    }

    public synchronized void append(float[] samples) {
        append(samples, 0, samples.length);
    }

    public synchronized void append(float[] samples, int offset, int length) {
        for (int i = 0; i < length; i++) {
            buf[(int) (writePos % capacity)] = samples[offset + i];
            writePos++;
        }
        // Total samples overwritten at least once: useful "how much history
        // was lost" signal for rollback-window sizing.
        droppedSamples = Math.max(0, writePos - capacity);
    }

    /** Append 16-bit PCM, normalized to [-1, 1]. */
    public synchronized void appendPcm16(short[] pcm) {
        float[] f = new float[pcm.length];
        for (int i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768.0f;
        append(f);
    }

    /** Number of samples currently retained (<= capacity). */
    public synchronized int available() {
        return (int) Math.min(writePos, (long) capacity);
    }

    public synchronized long totalWritten() {
        return writePos;
    }

    public synchronized long droppedCount() {
        return droppedSamples;
    }

    /** Copy of the most recent {@code nSamples} (or fewer if not yet filled). */
    public synchronized float[] last(int nSamples) {
        int avail = available();
        int n = Math.min(nSamples, avail);
        float[] out = new float[n];
        long start = writePos - n;
        for (int i = 0; i < n; i++) {
            out[i] = buf[(int) ((start + i) % capacity)];
        }
        return out;
    }

    /** Copy of the most recent {@code windowMs} of audio. */
    public float[] lastMs(int windowMs) {
        return last((int) ((long) AsrState.SAMPLE_RATE * windowMs / 1000));
    }

    /**
     * Samples appended since {@code startPos} (see {@link #totalWritten}),
     * oldest-first, capped at {@code maxSamples}. Used to replay the current
     * utterance from its speech start (§15, §25) — {@link #lastMs} would
     * return trailing silence instead when the request happens at an
     * endpoint after seconds of silence.
     */
    public synchronized float[] since(long startPos, int maxSamples) {
        long end = writePos;
        long start = Math.max(startPos, end - capacity);
        start = Math.max(start, end - Math.max(0, maxSamples));
        int n = (int) Math.max(0, end - start);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = buf[(int) ((start + i) % capacity)];
        }
        return out;
    }

    public synchronized void clear() {
        writePos = 0;
        droppedSamples = 0;
    }

    public int capacity() {
        return capacity;
    }
}
