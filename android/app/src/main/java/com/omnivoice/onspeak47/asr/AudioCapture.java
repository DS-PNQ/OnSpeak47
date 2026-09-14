/*
 * OmniVoice — Streaming audio capture (16 kHz mono, 20 ms frames).
 *
 * Spec §7: AudioRecord → 20 ms PCM frames → RingBuffer. No WAV staging —
 * frames flow straight into the streaming pipeline (spec §21 MUST 1).
 */
package com.omnivoice.onspeak47.asr;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Process;
import android.util.Log;

public class AudioCapture {

    private static final String TAG = "AudioCapture";

    public interface FrameListener {
        /** Called on the capture thread for every 20 ms frame. */
        void onFrame(short[] pcm16, long frameIndex);
        default void onError(int code) {}
    }

    private final int sampleRate;
    private final FrameListener listener;
    private AudioRecord record;
    private Thread thread;
    private volatile boolean running = false;
    private long frameIndex = 0;
    private long droppedFrames = 0;

    public AudioCapture(FrameListener listener) {
        this(AsrState.SAMPLE_RATE, listener);
    }

    public AudioCapture(int sampleRate, FrameListener listener) {
        this.sampleRate = sampleRate;
        this.listener = listener;
    }

    /** Start the high-priority capture thread. False when init fails. */
    public boolean start() {
        if (running) return true;
        int minBuf = AudioRecord.getMinBufferSize(sampleRate,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (minBuf <= 0) minBuf = sampleRate * 2;
        int bufSize = Math.max(minBuf * 2, sampleRate * 2);
        try {
            record = new AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize);
        } catch (SecurityException e) {
            Log.e(TAG, "RECORD_AUDIO not granted", e);
            return false;
        }
        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord init failed");
            record.release();
            record = null;
            return false;
        }
        int frameSamples = sampleRate * AsrState.FRAME_MS / 1000;
        running = true;
        frameIndex = 0;
        thread = new Thread(() -> loop(frameSamples), "AsrAudioCapture");
        thread.start();
        return true;
    }

    private void loop(int frameSamples) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
        try {
            record.startRecording();
        } catch (IllegalStateException e) {
            Log.e(TAG, "startRecording failed", e);
            running = false;
            return;
        }
        short[] frame = new short[frameSamples];
        while (running) {
            int read = record.read(frame, 0, frameSamples);
            if (read < 0) {
                droppedFrames++;
                listener.onError(read);
                continue;
            }
            if (read != frameSamples) {
                // Partial read: pad with silence so downstream timing holds.
                for (int i = read; i < frameSamples; i++) frame[i] = 0;
                droppedFrames++;
            }
            short[] copy = frame.clone();
            try {
                listener.onFrame(copy, frameIndex++);
            } catch (Exception e) {
                Log.w(TAG, "frame listener threw", e);
            }
        }
    }

    public void stop() {
        running = false;
        if (thread != null) {
            try {
                thread.join(2000);
            } catch (InterruptedException ignored) {
            }
            thread = null;
        }
        if (record != null) {
            try {
                record.stop();
            } catch (IllegalStateException ignored) {
            }
            record.release();
            record = null;
        }
    }

    public boolean isRunning() {
        return running;
    }

    public long droppedFrames() {
        return droppedFrames;
    }
}
