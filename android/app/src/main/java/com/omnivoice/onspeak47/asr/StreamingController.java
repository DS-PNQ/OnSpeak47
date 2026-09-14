/*
 * OmniVoice — StreamingController: lifecycle owner for the production
 * realtime VI/EN/ZH pipeline (spec §23-§25).
 *
 * Streaming is the only ASR path: start() opens the microphone for live
 * partials + endpoint finals. Missing assets fail fast with false so the
 * UI can surface "models missing" instead of silently degrading.
 *
 * Heavy resources are shared across sessions: the caller should pass the
 * app-preloaded ZipformerModelManager (loading 3 recognizers costs seconds
 * and hundreds of MB — never redo it per tap). The VAD engine is owned here
 * and reused (reset per session). Only caller-owned resources are closed on
 * shutdown; app-shared models are left open.
 */
package com.omnivoice.onspeak47.asr;

import android.content.Context;
import android.util.Log;

import java.io.File;

public class StreamingController {

    private static final String TAG = "StreamingController";

    private StreamingPipeline pipeline;
    private VadEngine sharedVad;
    private boolean live = false;
    private boolean ownsModels = true;

    /** True when at least one language's streaming assets are on device. */
    public static boolean isAvailable(Context context) {
        // Check extraction dir first (fast path after first launch), then APK assets.
        String[] probes = {
                AsrState.VI_ENCODER, AsrState.EN_ENCODER, AsrState.ZH_ENCODER,
                AsrState.VI_TOKENS, AsrState.EN_TOKENS, AsrState.ZH_TOKENS,
        };
        for (String asset : probes) {
            File extracted = new File(context.getFilesDir(), asset);
            if (extracted.exists() && extracted.length() > 0) return true;
            try {
                context.getAssets().open(asset).close();
                return true;
            } catch (Exception ignored) {
            }
        }
        return false;
    }

    /**
     * Start live streaming transcription. Returns false when assets are
     * missing, the mic is busy, or init throws. Must be called on the main
     * thread (the pipeline posts UI callbacks through the main Looper).
     */
    public synchronized boolean start(Context context, StreamingPipeline.Listener listener) {
        return start(context, null, listener);
    }

    /**
     * Start with a shared (preloaded) model manager, e.g. from
     * OmniVoiceApp.getStreamingAsrModels(). Null falls back to a
     * session-local manager (slower first start, closed on shutdown).
     */
    public synchronized boolean start(Context context, ZipformerModelManager models,
                                      StreamingPipeline.Listener listener) {
        if (live) return true;
        try {
            if (!isAvailable(context)) {
                Log.e(TAG, "streaming assets absent — fetch via optimize/11_fetch_streaming_zipformer.py");
                return false;
            }
            Context app = context.getApplicationContext();
            if (sharedVad == null) sharedVad = new VadEngine(app);
            else sharedVad.reset();
            ZipformerModelManager mm = models != null ? models : new ZipformerModelManager(app);
            ownsModels = (models == null);
            pipeline = new StreamingPipeline(app, sharedVad, mm,
                    new LanguageIdEngine(), new LanguageRouter());
            if (listener != null) pipeline.addListener(listener);
            pipeline.start();
            live = true;
            Log.i(TAG, "streaming live, active=" + pipeline.activeLanguage()
                    + " sharedModels=" + !ownsModels);
            return true;
        } catch (Exception e) {
            Log.e(TAG, "streaming start failed", e);
            pipeline = null;
            live = false;
            return false;
        }
    }

    /** Stop capture and flush the trailing final transcript. */
    public synchronized void stop() {
        if (!live || pipeline == null) return;
        try {
            pipeline.stop();
        } catch (Exception e) {
            Log.w(TAG, "streaming stop failed", e);
        } finally {
            live = false;
        }
    }

    /** Release session resources (call from Activity.onDestroy). */
    public synchronized void shutdown() {
        stop();
        if (pipeline != null) {
            try {
                pipeline.shutdown(ownsModels);
            } catch (Exception e) {
                Log.w(TAG, "streaming shutdown failed", e);
            } finally {
                pipeline = null;
            }
        }
        // sharedVad is closed with the pipeline above; drop the reference so
        // a future session rebuilds it cleanly.
        sharedVad = null;
    }

    public synchronized boolean isLive() {
        return live;
    }

    public synchronized String activeLanguageCode() {
        return pipeline == null ? "vi" : pipeline.activeLanguage().code;
    }
}
