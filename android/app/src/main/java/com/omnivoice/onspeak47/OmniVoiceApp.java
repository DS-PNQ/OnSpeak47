/*
 * OmniVoice — On-Device Speech Translation
 * Application class.
 */

package com.omnivoice.onspeak47;

import android.app.Application;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.omnivoice.onspeak47.asr.ZipformerModelManager;
import com.omnivoice.onspeak47.pipeline.PipelineOrchestrator;
import com.omnivoice.onspeak47.pipeline.TTSModule;
import com.omnivoice.onspeak47.pipeline.TranslationModule;


/**
 * Global Application class — manages the lifecycle of the on-device models:
 * streaming Zipformer ASR (VI/EN/ZH), HyMT translation (GGUF) and MMS-TTS.
 */
public class OmniVoiceApp extends Application {

    private static final String TAG = "OmniVoiceApp";

    @Nullable private TranslationModule translationModule;
    @Nullable private TTSModule ttsModule;
    @Nullable private ZipformerModelManager streamingAsrModels;
    @Nullable private PipelineOrchestrator orchestrator;

    private Handler mainHandler;

    // ----------------------------------------------------------------
    // Lifecycle
    // ----------------------------------------------------------------

    @Override
    public void onCreate() {
        super.onCreate();
        mainHandler = new Handler(Looper.getMainLooper());
        Log.i(TAG, "OmniVoice application created");
    }

    // ----------------------------------------------------------------
    // Module initializers (lazy, called from LoadingActivity)
    // ----------------------------------------------------------------

    /**
     * Preload streaming Zipformer models per the device RAM bucket
     * (VI+EN+ZH on >=8GB, VI+EN on 6GB, VI on smaller devices).
     */
    public void initializeStreamingAsr(@NonNull InitListener listener) {
        if (streamingAsrModels != null) {
            listener.onInitialized();
            return;
        }
        new Thread(() -> {
            try {
                streamingAsrModels = new ZipformerModelManager(this);
                streamingAsrModels.preloadForDevice();
                mainHandler.post(listener::onInitialized);
            } catch (Exception e) {
                Log.e(TAG, "Streaming ASR init failed", e);
                mainHandler.post(() -> listener.onError("Streaming ASR init failed: " + e.getMessage()));
            }
        }).start();
    }

    /**
     * Initialize the Translation (HyMT GGUF) module.
     */
    public void initializeTranslation(@NonNull InitListener listener) {
        if (translationModule != null) {
            listener.onInitialized();
            return;
        }
        new Thread(() -> {
            try {
                translationModule = new TranslationModule(this);
                mainHandler.post(listener::onInitialized);
            } catch (Exception e) {
                Log.e(TAG, "Translation init failed", e);
                mainHandler.post(() -> listener.onError("Translation initialization failed: " + e.getMessage()));
            }
        }).start();
    }

    /**
     * Initialize the TTS (MMS-TTS / Android TTS fallback) module.
     */
    public void initializeTTS(@NonNull InitListener listener) {
        if (ttsModule != null) {
            listener.onInitialized();
            return;
        }
        new Thread(() -> {
            try {
                ttsModule = new TTSModule(this);
                mainHandler.post(listener::onInitialized);
            } catch (Exception e) {
                Log.e(TAG, "TTS init failed", e);
                mainHandler.post(() -> listener.onError("TTS initialization failed: " + e.getMessage()));
            }
        }).start();
    }

    /**
     * Initialize streaming ASR + translation + TTS in parallel and create
     * the pipeline orchestrator. This significantly reduces startup time
     * compared to sequential loading.
     */
    public void initializeAll(@NonNull InitListener listener) {
        new Thread(() -> {
            java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(3);
            final String[] errors = new String[3];

            // Preload streaming ASR in parallel
            new Thread(() -> {
                try {
                    streamingAsrModels = new ZipformerModelManager(OmniVoiceApp.this);
                    streamingAsrModels.preloadForDevice();
                    Log.i(TAG, "Streaming ASR models loaded");
                } catch (Exception e) {
                    Log.e(TAG, "Streaming ASR init failed", e);
                    errors[0] = "Streaming ASR init failed: " + e.getMessage();
                } finally {
                    latch.countDown();
                }
            }).start();

            // Load Translation in parallel
            new Thread(() -> {
                try {
                    translationModule = new TranslationModule(OmniVoiceApp.this);
                    Log.i(TAG, "Translation module loaded");
                } catch (Exception e) {
                    Log.e(TAG, "Translation init failed", e);
                    errors[1] = "Translation initialization failed: " + e.getMessage();
                } finally {
                    latch.countDown();
                }
            }).start();

            // Load TTS in parallel
            new Thread(() -> {
                try {
                    ttsModule = new TTSModule(OmniVoiceApp.this);
                    Log.i(TAG, "TTS module loaded");
                } catch (Exception e) {
                    Log.e(TAG, "TTS init failed", e);
                    errors[2] = "TTS initialization failed: " + e.getMessage();
                } finally {
                    latch.countDown();
                }
            }).start();

            try {
                latch.await(); // Wait for all three to complete
            } catch (InterruptedException e) {
                Log.e(TAG, "Model loading interrupted", e);
                mainHandler.post(() -> listener.onError("Model loading interrupted"));
                return;
            }

            // Check for errors
            for (String error : errors) {
                if (error != null) {
                    mainHandler.post(() -> listener.onError(error));
                    return;
                }
            }

            // All loaded successfully — create orchestrator
            orchestrator = new PipelineOrchestrator(translationModule, ttsModule);
            mainHandler.post(listener::onInitialized);
        }).start();
    }

    /**
     * Create orchestrator instance if modules are initialized.
     */
    public void createOrchestrator() {
        if (translationModule != null && ttsModule != null) {
            orchestrator = new PipelineOrchestrator(translationModule, ttsModule);
        }
    }

    // ----------------------------------------------------------------
    // Getters
    // ----------------------------------------------------------------

    @Nullable public ZipformerModelManager getStreamingAsrModels() { return streamingAsrModels; }
    @Nullable public TranslationModule getTranslationModule() { return translationModule; }
    @Nullable public TTSModule getTTSModule() { return ttsModule; }
    @Nullable public PipelineOrchestrator getOrchestrator() { return orchestrator; }

    // ----------------------------------------------------------------
    // Listener interface
    // ----------------------------------------------------------------

    public interface InitListener {
        void onInitialized();
        void onError(String message);
    }
}
