/*
 * OmniVoice — Main Translation Activity
 *
 * Streaming-first UI: toggle Start/Stop. While live, the Zipformer
 * pipeline (AudioCapture → VAD → streaming ASR → router) renders partials
 * continuously; each endpoint FINAL auto-translates → TTS. Source language
 * is auto-detected by the router (VI/EN/ZH) — the UI only selects the
 * translation target.
 */

package com.omnivoice.onspeak47;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.omnivoice.onspeak47.audio.AudioPlayer;
import com.omnivoice.onspeak47.asr.StreamingController;
import com.omnivoice.onspeak47.asr.StreamingPipeline;
import com.omnivoice.onspeak47.pipeline.PipelineOrchestrator;
import com.omnivoice.onspeak47.util.LanguageConfig;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * Streaming translation screen with Start/Stop toggle.
 */
public class TranslationActivity extends AppCompatActivity {

    private static final String TAG = "TranslationActivity";
    private static final int PERMISSION_REQUEST_AUDIO = 100;

    // UI elements
    private Button talkButton;
    private TextView transcriptView;
    private TextView translationView;
    private TextView timingView;
    private Spinner tgtLangSpinner;

    // Pipeline (Translation → TTS; ASR lives in StreamingPipeline)
    private PipelineOrchestrator orchestrator;
    private AudioPlayer player;

    // Streaming session
    private final StreamingController streaming = new StreamingController();
    private boolean isStreaming = false;
    private String detectedLang = "vi";
    // Accidental double-taps produced 10 ms sessions; ignore toggles faster
    // than this.
    private long lastToggleMs = 0;
    private static final long TOGGLE_DEBOUNCE_MS = 400;

    // State
    private String tgtLang = "en";

    // Single-thread pipeline executor: Translation/TTS are CPU-bound, so two
    // concurrent requests spike RAM and thrash caches for no throughput gain.
    // Rapid finals are latest-wins: a stale request exits at the next stage
    // boundary via its generation counter.
    private ExecutorService pipelineExecutor;
    private final AtomicInteger requestGeneration = new AtomicInteger();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_translation);

        // Get pipeline from Application
        OmniVoiceApp app = (OmniVoiceApp) getApplication();
        orchestrator = app.getOrchestrator();
        player = new AudioPlayer();
        pipelineExecutor = Executors.newSingleThreadExecutor();

        // Bind UI
        talkButton = findViewById(R.id.btn_talk);
        transcriptView = findViewById(R.id.txt_transcript);
        translationView = findViewById(R.id.txt_translation);
        timingView = findViewById(R.id.txt_timing);
        tgtLangSpinner = findViewById(R.id.spinner_tgt_lang);

        // Source spinner is gone in the streaming UI (router auto-detects);
        // hide it if the layout still contains it.
        View srcSpinner = findViewById(R.id.spinner_src_lang);
        if (srcSpinner != null) srcSpinner.setVisibility(View.GONE);

        setupTargetSpinner();
        setupToggleButton();
        updateToggleUi();
        requestAudioPermission();
    }

    // ----------------------------------------------------------------
    // Target language selection (source is auto-detected)
    // ----------------------------------------------------------------

    private void setupTargetSpinner() {
        String[] languages = LanguageConfig.getDisplayNames();

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_dropdown_item, languages
        );

        tgtLangSpinner.setAdapter(adapter);
        tgtLangSpinner.setSelection(LanguageConfig.indexOf("en"));

        tgtLangSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                tgtLang = LanguageConfig.getCodeAtIndex(pos);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    // ----------------------------------------------------------------
    // Start/Stop toggle
    // ----------------------------------------------------------------

    private void setupToggleButton() {
        talkButton.setOnClickListener(v -> {
            long now = SystemClock.uptimeMillis();
            if (now - lastToggleMs < TOGGLE_DEBOUNCE_MS) return;
            lastToggleMs = now;
            if (isStreaming) {
                stopStreaming();
            } else {
                startStreaming();
            }
        });
    }

    private void startStreaming() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestAudioPermission();
            return;
        }
        if (isStreaming) return;
        if (orchestrator == null) {
            OmniVoiceApp app = (OmniVoiceApp) getApplication();
            orchestrator = app.getOrchestrator();
        }
        if (orchestrator == null) {
            transcriptView.setText("Error: pipeline not ready");
            return;
        }

        transcriptView.setText("Đang nghe…");
        translationView.setText("");
        timingView.setText("");
        detectedLang = "vi";

        boolean ok;
        try {
            OmniVoiceApp app = (OmniVoiceApp) getApplication();
            ok = streaming.start(this, app.getStreamingAsrModels(),
                    new StreamingPipeline.Listener() {
                @Override
                public void onPartial(String stable, String speculative, String lang) {
                    final String text = (stable + " " + speculative).trim();
                    if (!text.isEmpty()) {
                        detectedLang = lang;
                        runOnUiThread(() ->
                                transcriptView.setText("[" + lang + "] " + text + "…"));
                    }
                }

                @Override
                public void onFinal(String text, String lang) {
                    if (text == null || text.trim().isEmpty()) return;
                    detectedLang = lang;
                    final String finalText = text;
                    final String finalLang = lang;
                    runOnUiThread(() -> transcriptView.setText(finalText));
                    translateFinal(finalText, finalLang);
                }

                @Override
                public void onLanguageSwitch(String fromLang, String toLang) {
                    detectedLang = toLang;
                    Log.i(TAG, "streaming language switch: " + fromLang + " → " + toLang);
                }

                @Override
                public void onMetrics(String summaryJson) {
                    Log.i(TAG, "streaming metrics: " + summaryJson);
                }
            });
        } catch (Exception e) {
            Log.w(TAG, "streaming start failed", e);
            ok = false;
        }

        if (!ok) {
            transcriptView.setText("Error: streaming ASR unavailable (missing models?)");
            return;
        }
        isStreaming = true;
        updateToggleUi();
        Log.i(TAG, "streaming started");
    }

    private void stopStreaming() {
        if (!isStreaming) return;
        isStreaming = false;
        try {
            streaming.stop();
        } catch (Exception e) {
            Log.w(TAG, "streaming stop failed", e);
        }
        updateToggleUi();
        Log.i(TAG, "streaming stopped");
    }

    private void updateToggleUi() {
        talkButton.setText(isStreaming
                ? getString(R.string.btn_stop)
                : getString(R.string.btn_start));
    }

    /** Translate one endpoint FINAL in the background (latest-wins). */
    private void translateFinal(String text, String srcLang) {
        if (orchestrator == null) return;
        final int generation = requestGeneration.incrementAndGet();
        final File cacheDir = getCacheDir();
        pipelineExecutor.execute(() -> {
            try {
                PipelineOrchestrator.PipelineResult result =
                        orchestrator.processStreamingFinal(text, srcLang, tgtLang, cacheDir,
                                () -> generation != requestGeneration.get());

                if (result == null                 // superseded between stages
                        || generation != requestGeneration.get()) {
                    Log.i(TAG, "Pipeline request #" + generation + " superseded — dropped");
                    return;
                }

                runOnUiThread(() -> {
                    translationView.setText(result.translation);
                    timingView.setText(String.format(
                            "Translation: %dms | TTS: %dms | Total: %dms",
                            result.translationMs, result.ttsMs, result.totalMs
                    ));
                    if (result.ttsError != null) {
                        android.widget.Toast.makeText(
                                TranslationActivity.this,
                                "TTS failed: " + result.ttsError,
                                android.widget.Toast.LENGTH_LONG).show();
                    }
                });

                if (result.audioPath != null) {
                    player.play(result.audioPath);
                }

            } catch (Exception e) {
                Log.e(TAG, "Pipeline error", e);
                runOnUiThread(() -> translationView.setText("Error: " + e.getMessage()));
            }
        });
    }

    // ----------------------------------------------------------------
    // Permissions
    // ----------------------------------------------------------------

    private void requestAudioPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    PERMISSION_REQUEST_AUDIO
            );
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "Audio permission granted");
            } else {
                Log.w(TAG, "Audio permission denied — streaming won't work");
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            streaming.shutdown();
        } catch (Exception e) {
            Log.w(TAG, "streaming shutdown failed", e);
        }
        if (player != null) player.release();
        if (pipelineExecutor != null) {
            pipelineExecutor.shutdownNow();
            pipelineExecutor = null;
        }
    }
}
