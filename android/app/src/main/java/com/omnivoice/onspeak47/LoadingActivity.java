/*
 * OmniVoice — Splash / Model Loading Activity
 */

package com.omnivoice.onspeak47;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;


/**
 * Loading screen shown at launch while the on-device models
 * (streaming Zipformer ASR, HyMT translation, MMS-TTS) are initialized.
 */
public class LoadingActivity extends AppCompatActivity {

    private static final String TAG = "LoadingActivity";

    private ProgressBar progressBar;
    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_loading);

        progressBar = findViewById(R.id.loading_progress);
        statusText = findViewById(R.id.loading_status);

        startModelLoading();
    }

    private void startModelLoading() {
        OmniVoiceApp app = (OmniVoiceApp) getApplication();

        updateStatus("Loading streaming ASR (Zipformer VI/EN/ZH)...", 0);

        app.initializeStreamingAsr(new OmniVoiceApp.InitListener() {
            @Override
            public void onInitialized() {
                updateStatus("Loading Translation model (HyMT-1.5)...", 33);

                app.initializeTranslation(new OmniVoiceApp.InitListener() {
                    @Override
                    public void onInitialized() {
                        updateStatus("Loading TTS model (MMS-TTS)...", 66);

                        app.initializeTTS(new OmniVoiceApp.InitListener() {
                            @Override
                            public void onInitialized() {
                                app.createOrchestrator();
                                updateStatus("Ready!", 100);
                                navigateToTranslation();
                            }

                            @Override
                            public void onError(String message) {
                                showError(message);
                            }
                        });
                    }

                    @Override
                    public void onError(String message) {
                        showError(message);
                    }
                });
            }

            @Override
            public void onError(String message) {
                showError(message);
            }
        });
    }

    private void updateStatus(String message, int progress) {
        Log.i(TAG, message);
        statusText.setText(message);
        progressBar.setProgress(progress);
    }

    private void showError(String message) {
        Log.e(TAG, "Loading error: " + message);
        statusText.setText("Error: " + message);
    }

    private void navigateToTranslation() {
        Intent intent = new Intent(this, TranslationActivity.class);
        startActivity(intent);
        finish();
    }
}
