/*
 * OmniVoice — TTS Module
 *
 * Primary backend: MMS-TTS (VITS) via ONNX Runtime.
 * Fallback backend: Android system TextToSpeech (used when an MMS-TTS ONNX
 * model for a language isn't bundled in assets).
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.TensorUtils;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.extensions.OrtxPackage;


/**
 * Text-to-Speech module with dual backend:
 * <ol>
 *   <li><b>MMS-TTS ONNX</b> — higher quality, loaded per-language from assets
 *       (exported by {@code optimize/export_mms_tts.py}).</li>
 *   <li><b>Android system TTS</b> — fallback for languages without a bundled
 *       ONNX model.</li>
 * </ol>
 *
 * <p>Follows the ONNX-Runtime + NNAPI pattern used by {@link TranslationModule},
 * and writes PCM WAV files compatible with
 * {@link com.omnivoice.onspeak47.audio.AudioPlayer}.</p>
 */
public class TTSModule {

    private static final String TAG = "TTSModule";
    private static final String[] SUPPORTED = {"vi", "en", "zh"};

    private final Context context;

    // ---- Android system TTS (fallback) --------------------------------
    /**
     * The TextToSpeech object is created on the main thread — the setup every
     * stock TTS engine (Google, Samsung, Pico, iFlytek …) is developed and
     * tested against, and the most reliable across OEM ROMs. Creating it on a
     * worker/HandlerThread or on a Looper-less thread delays or even misses
     * the onInit callback on some engines. Nothing ever blocks the main thread
     * though: only worker threads wait on the latch.
     */
    private final Handler ttsMainHandler = new Handler(Looper.getMainLooper());

    /** Written on the main thread, read on pipeline threads. */
    private volatile TextToSpeech androidTTS;
    private volatile boolean systemTtsReady = false;
    /**
     * Most recent {@link TextToSpeech.OnInitListener} result:
     * {@link TextToSpeech#SUCCESS}, {@link TextToSpeech#ERROR} — or
     * {@link #SYSTEM_TTS_PENDING} while a bind is still in flight.
     * {@code TextToSpeech.ERROR == -1}, so "pending" must use a distinct
     * sentinel value: conflating them used to hide a real engine ERROR behind
     * a bogus "did not finish binding" message (and skipped the retry).
     */
    private volatile int systemTtsInitStatus = SYSTEM_TTS_PENDING;
    /**
     * Latch that the current {@link TextToSpeech.OnInitListener} counts down
     * exactly once (success or failure). Swapped on each (re)bind by
     * {@link #initSystemTts()} / the rebind branch of
     * {@link #awaitSystemTtsReady(long)}.
     */
    private volatile CountDownLatch systemTtsInitLatch = new CountDownLatch(1);
    /** Guards TTS (re)bind and instance swap. */
    private final Object ttsBindLock = new Object();
    /** Whether the single allowed lazy rebind has already been attempted. */
    private boolean rebindAttempted = false;
    private volatile boolean ttsReleased = false;

    /**
     * TTS engine package we explicitly bind to (resolved once by
     * {@link #chooseEnginePackage}). The framework's "default engine" setting
     * is empty on many phones even when Google TTS is installed — that is the
     * root cause of the "engine null / not ready" failure, so we drive the
     * engine selection ourselves.
     */
    private volatile String requestedEnginePackage = null;
    /** One-shot flag: only auto-open the voice-data installer once per process. */
    private boolean ttsHelpOffered = false;

    /** Well-known TTS engine packages, checked directly as a fallback. */
    private static final String[] KNOWN_TTS_ENGINES = {
            "com.google.android.tts", // Google Speech Services / Google TTS
            "com.samsung.SMT",        // Samsung TTS
            "com.iflytek.tts",        // iFlytek Speech (common on Chinese phones)
            "com.svox.pico",          // AOSP Pico TTS (default on many emulators/ROMs)
    };

    /** Sentinel meaning "TextToSpeech bind still in flight" (ERROR == -1). */
    private static final int SYSTEM_TTS_PENDING = Integer.MIN_VALUE;

    /**
     * How long a caller waits for the system TTS engine to finish binding.
     * Generous on purpose: the engine can legitimately take a few seconds at
     * cold start, and we never destroy an in-flight bind (see
     * {@link #awaitSystemTtsReady(long)}).
     */
    private static final long SYSTEM_TTS_INIT_TIMEOUT_MS = 10_000L;

    /** Human-readable reason for the most recent synthesis failure (surfaced in the UI). */
    private volatile String lastError = null;

    /** Reason the last {@link #synthesize} call failed, or null if it succeeded. */
    public String getLastError() { return lastError; }

    // ---- MMS-TTS ONNX (primary) ---------------------------------------
    private final Map<String, OrtSession> ortSessions = new HashMap<>();
    private final Map<String, MmsTtsTokenizer> tokenizers = new HashMap<>();
    private final Map<String, Integer> sampleRates = new HashMap<>();
    /** Bundled language -> copied ONNX model path; sessions are created lazily. */
    private final Map<String, String> modelPaths = new HashMap<>();
    private OrtEnvironment env;
    private File baseDir;

    /** VITS default sample rate; overridden by each model's config.json. */
    private static final int DEFAULT_SAMPLE_RATE = 16000;

    // Language locale mapping for the system-TTS fallback
    private static final Map<String, Locale> LOCALES = new HashMap<>();
    static {
        LOCALES.put("vi", new Locale("vi"));
        LOCALES.put("en", Locale.ENGLISH);
        LOCALES.put("zh", Locale.CHINESE);
        LOCALES.put("zh_hans", Locale.SIMPLIFIED_CHINESE);
        LOCALES.put("zh_hant", Locale.TRADITIONAL_CHINESE);
    }

    /**
     * Initialize the TTS module.
     *
     * Sets up the Android system TTS and probes for bundled MMS-TTS ONNX
     * assets for all supported languages.
     */
    public TTSModule(Context context) {
        this.context = context;
        initSystemTts();
        try {
            initMmsOnnx();
        } catch (Exception e) {
            Log.e(TAG, "MMS-TTS ONNX init failed — system TTS will be used", e);
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Synthesize text to a WAV file.
     *
     * @param text       Text to synthesize
     * @param language   Language code ("vi", "en", "zh")
     * @param outputPath Path for the output WAV file
     * @return Path to the generated audio file, or null on failure
     */
    public String synthesize(String text, String language, String outputPath) {
        long startTime = System.currentTimeMillis();
        String lang = normalizeLang(language);
        lastError = null;

        // Primary path: MMS-TTS ONNX
        if (hasOnnxModel(lang) && runOnnxSynthesis(text, lang, outputPath)) {
            Log.i(TAG, "TTS synthesis (MMS-TTS ONNX) done in "
                    + (System.currentTimeMillis() - startTime) + "ms");
            return outputPath;
        }

        // Fallback: Android system TTS
        if (synthesizeWithSystemTts(text, lang, outputPath)) {
            Log.i(TAG, "TTS synthesis (system TTS fallback) done in "
                    + (System.currentTimeMillis() - startTime) + "ms");
            return outputPath;
        }

        if (lastError == null) {
            lastError = "All TTS backends failed for [" + lang + "] (no ONNX model, system TTS failed)";
        }
        Log.e(TAG, "TTS synthesis failed for language: " + lang + " — " + lastError);
        return null;
    }

    /**
     * Speak text immediately (without saving to file).
     *
     * Uses system TTS directly for instant feedback.
     */
    public void speak(String text, String language) {
        ensureSystemTtsStarted();
        if (!awaitSystemTtsReady(SYSTEM_TTS_INIT_TIMEOUT_MS)) return; // lastError set
        if (androidTTS == null) return;
        Locale locale = resolveSystemLocale(normalizeLang(language));
        if (locale == null) return; // lastError set; nothing speakable installed
        androidTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null, "omnivoice_speak");
    }

    /** Whether a bundled MMS-TTS ONNX model is available for the language. */
    public boolean hasOnnxSupport(String language) {
        return hasOnnxModel(normalizeLang(language));
    }

    /**
     * Release TTS resources.
     */
    public void release() {
        synchronized (ttsBindLock) {
            ttsReleased = true;
            if (androidTTS != null) {
                try { androidTTS.stop(); } catch (Exception ignored) {}
                try { androidTTS.shutdown(); } catch (Exception ignored) {}
                androidTTS = null;
            }
        }
        for (OrtSession s : ortSessions.values()) {
            try { s.close(); } catch (Exception ignored) {}
        }
        ortSessions.clear();
        tokenizers.clear();
        sampleRates.clear();
    }

    // ------------------------------------------------------------------
    // System TTS (fallback backend)
    // ------------------------------------------------------------------

    /**
     * (Re)start the Android system TTS engine bind.
     *
     * The engine is created on the main thread (see {@link #ttsMainHandler}),
     * which is the canonical, most OEM-compatible setup. Fire-and-forget:
     * nobody blocks in here; the listener records the reported status and
     * counts down the latch so {@link #awaitSystemTtsReady(long)} can wait for
     * the outcome of *this* bind from a worker thread.
     */
    private void initSystemTts() {
        synchronized (ttsBindLock) {
            if (ttsReleased) return;
            systemTtsInitStatus = SYSTEM_TTS_PENDING;
            systemTtsReady = false;
            systemTtsInitLatch = new CountDownLatch(1);
            final CountDownLatch latch = systemTtsInitLatch;
            ttsMainHandler.post(() -> createSystemTts(latch));
        }
    }

    /** Lazily start TTS init if no bind is currently in flight. */
    private void ensureSystemTtsStarted() {
        synchronized (ttsBindLock) {
            if (androidTTS == null && systemTtsInitLatch.getCount() == 0) {
                initSystemTts();
            }
        }
    }

    /**
     * Construct the {@link TextToSpeech} instance (on the main thread)
     * and register an onInit that records the engine's reported status and
     * counts down {@code latch}. A fresh latch is passed on every rebind so
     * callers of {@link #awaitSystemTtsReady(long)} observe *this* bind's
     * outcome. The instance is only kept if no other instance was bound in the
     * meantime and the module hasn't been released.
     */
    private void createSystemTts(final CountDownLatch latch) {
        String engine = requestedEnginePackage;
        if (engine == null) {
            engine = chooseEnginePackage();
            requestedEnginePackage = engine;
            Log.i(TAG, "System TTS: explicit engine="
                    + (engine != null ? engine : "none — falling back to framework default"));
        }
        final TextToSpeech[] instanceHolder = new TextToSpeech[1];
        TextToSpeech.OnInitListener onInit = status -> {
            TextToSpeech instance = instanceHolder[0];
            // onInit can fire INLINE during the constructor (before
            // instanceHolder[0] is populated) when the engine fails to
            // bind immediately — then instance is null and androidTTS is
            // still null too, so null == null records the ERROR correctly.
            // A callback from a *stale* instance (already replaced by a
            // rebind) must NOT clobber the new bind's state.
            if (androidTTS == instance) {
                systemTtsInitStatus = status;
                if (status == TextToSpeech.SUCCESS) {
                    systemTtsReady = true;
                    // Log which engine handled it — Chinese support is highly
                    // engine-dependent (Google TTS vs Samsung vs Pico), and this
                    // is the first thing to check when zh synthesis is silent.
                    Log.i(TAG, "Android system TTS initialized, engine="
                            + safeEngineName());
                } else {
                    systemTtsReady = false;
                    Log.e(TAG, "Android system TTS init failed, status=" + status);
                }
            } else if (instance != null) {
                Log.w(TAG, "Ignoring onInit from a discarded system TTS instance");
            }
            latch.countDown();
        };
        try {
            instanceHolder[0] = (engine != null)
                    ? new TextToSpeech(context.getApplicationContext(), onInit, engine)
                    : new TextToSpeech(context.getApplicationContext(), onInit);
        } catch (Throwable t) {
            // Never let a TextToSpeech construction failure kill the setup or
            // leave the latch un-counted — report it as a hard init error so
            // awaitSystemTtsReady() can retry once and give an accurate reason.
            Log.e(TAG, "TextToSpeech constructor failed (engine=" + engine + ")", t);
            systemTtsInitStatus = TextToSpeech.ERROR;
            systemTtsReady = false;
            latch.countDown();
            return;
        }
        TextToSpeech instance = instanceHolder[0];
        synchronized (ttsBindLock) {
            if (ttsReleased || androidTTS != null) {
                // Don't leak an instance created after release(), or a second
                // instance racing an existing bind — shut the extra one down.
                try { instance.shutdown(); } catch (Exception ignored) {}
            } else {
                androidTTS = instance;
            }
        }
    }

    /**
     * Wait (on the calling thread) until the system TTS engine reports ready
     * or {@code timeoutMs} elapses.
     *
     * <p>Deliberately waits ONCE without touching the instance. Destroying and
     * recreating a TextToSpeech whose bind has merely been slow — which the
     * old code did on every timeout — restarted the whole bind dance, so a
     * slowly-connecting engine never got a chance and every zh call failed
     * with the persistent "System TTS engine not ready" error. Only an
     * explicit hard error from {@code onInit} triggers the single allowed
     * rebind (some OEM engines fail the first transient bind and succeed on
     * retry). Nothing here ever blocks the engine's own callback thread.
     *
     * @return true if {@code onInit} reported SUCCESS in time, false otherwise
     */
    private boolean awaitSystemTtsReady(long timeoutMs) {
        if (systemTtsReady && androidTTS != null) return true;

        long deadline = SystemClock.uptimeMillis() + timeoutMs;
        CountDownLatch pending = systemTtsInitLatch;

        // Wait once, and only once, for the async bind. The engine can
        // legitimately need several seconds at cold start — especially on a
        // low-end device or right after the three heavy ONNX models finish
        // loading in parallel and the CPU is still saturated.
        waitForLatch(pending, deadline);

        if (systemTtsReady && androidTTS != null) return true;

        boolean reportedError = systemTtsInitStatus != SYSTEM_TTS_PENDING
                && systemTtsInitStatus != TextToSpeech.SUCCESS;

        if (reportedError) {
            CountDownLatch rebound = maybeRebind();
            if (rebound != null) {
                waitForLatch(rebound, deadline + SYSTEM_TTS_INIT_TIMEOUT_MS);
                if (systemTtsReady && androidTTS != null) return true;
                // Rebinding also failed — fall through so the diagnostic below
                // (the first bind's reported error) is always surfaced.
                reportedError = true;
            }
        }

        if (systemTtsReady && androidTTS != null) return true;

        String hint;
        if (!isAnyTtsEngineInstalled()) {
            hint = " — no text-to-speech engine service is installed or enabled"
                    + " on this device (install Google Speech Services / Google TTS)";
        } else if (!hasDefaultTtsEngine()) {
            hint = " — no default TTS engine is selected; app auto-selected "
                    + (requestedEnginePackage != null ? requestedEnginePackage : "an engine")
                    + " but it did not activate: open Settings → Text-to-speech →"
                    + " select the engine, then download Chinese (普通话) voice data";
        } else {
            hint = " — the TTS engine did not activate; open Settings →"
                    + " Text-to-speech to check voice data";
        }
        Log.e(TAG, "System TTS not ready: status=" + systemTtsInitStatus
                + " defaultEngine="
                + (androidTTS != null ? androidTTS.getDefaultEngine() : "n/a")
                + " requested=" + requestedEnginePackage
                + " installedEngines=[" + describeEngines() + "]");
        lastError = "System TTS engine not ready (engine " + safeEngineName()
                + (reportedError
                    ? " reported init status " + systemTtsInitStatus
                    : " did not finish binding within " + timeoutMs + "ms")
                + hint + ")";
        return false;
    }

    /**
     * Perform the single allowed in-place rebind after a hard init error.
     * Returns the fresh latch to wait on, or null when no rebind should or may
     * happen (already attempted, released, or no init looper available).
     */
    private CountDownLatch maybeRebind() {
        synchronized (ttsBindLock) {
            if (ttsReleased || rebindAttempted) {
                return null;
            }
            rebindAttempted = true;
            Log.i(TAG, "Retrying system TTS engine bind (onInit status="
                    + systemTtsInitStatus + ")");
            if (androidTTS != null) {
                try { androidTTS.stop(); } catch (Exception ignored) {}
                try { androidTTS.shutdown(); } catch (Exception ignored) {}
                androidTTS = null;
            }
            systemTtsInitStatus = SYSTEM_TTS_PENDING;
            systemTtsReady = false;
            systemTtsInitLatch = new CountDownLatch(1);
            final CountDownLatch rebound = systemTtsInitLatch;
            ttsMainHandler.post(() -> createSystemTts(rebound));
            return rebound;
        }
    }

    /** Await {@code latch} until {@code deadlineUptimeMs}; never throws. */
    private void waitForLatch(CountDownLatch latch, long deadlineUptimeMs) {
        long remaining = deadlineUptimeMs - SystemClock.uptimeMillis();
        if (remaining <= 0) {
            Log.w(TAG, "System TTS init wait skipped — deadline already reached");
            return;
        }
        try {
            if (!latch.await(remaining, TimeUnit.MILLISECONDS)) {
                Log.w(TAG, "System TTS init still pending after " + remaining + "ms");
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "System TTS init wait interrupted", e);
            Thread.currentThread().interrupt();
        }
    }

    /** Comma-joined packages of every installed TTS engine service (for logs). */
    private String describeEngines() {
        List<String> engines = discoveredEngines();
        return engines.isEmpty() ? "none" : String.join(", ", engines);
    }

    /**
     * Pick the TTS engine package to bind to explicitly, rather than relying on
     * the framework's "default engine" setting — which is empty on phones where
     * an engine is installed but never selected (the exact symptom reported for
     * the Chinese TTS failure). Prefers Google's engine for the best Chinese
     * voice support.
     */
    private String chooseEnginePackage() {
        List<String> engines = discoveredEngines();
        if (engines.isEmpty()) return null;
        for (String pkg : engines) {
            if ("com.google.android.tts".equals(pkg)) return pkg;
        }
        return engines.get(0);
    }

    /**
     * Every TTS engine visible to this app. Returns the union of:
     * 1. the installed services matching {@code TTS_SERVICE} (requires the
     *    manifest {@code <queries>} declaration on Android 11+ — added), and
     * 2. well-known engine packages present on the device.
     */
    private List<String> discoveredEngines() {
        List<String> out = new ArrayList<>();
        try {
            android.content.pm.PackageManager pm = context.getPackageManager();
            // MATCH_ALL is essential on API 33+: a freshly-installed TTS engine
            // (Google TTS / Samsung TTS) is often present but not yet enabled,
            // and flag 0 omits disabled/inactive components entirely — which
            // made an installed engine look "not installed". MATCH_ALL (value
            // 0x00020000 covering disabled components) returns them so we can
            // still bind explicitly. The <queries> TTS_SERVICE entry in the
            // manifest is what makes these services visible at all on API 30+.
            int flags = android.os.Build.VERSION.SDK_INT >= 33
                    ? android.content.pm.PackageManager.MATCH_ALL
                    : android.content.pm.PackageManager.MATCH_DISABLED_COMPONENTS;
            List<android.content.pm.ResolveInfo> services = pm.queryIntentServices(
                    new Intent(TextToSpeech.Engine.INTENT_ACTION_TTS_SERVICE), flags);
            if (services != null) {
                for (android.content.pm.ResolveInfo ri : services) {
                    if (ri.serviceInfo != null) {
                        String pkg = ri.serviceInfo.packageName;
                        if (!out.contains(pkg)) out.add(pkg);
                    }
                }
            }
        } catch (Exception ignored) {}
        try {
            android.content.pm.PackageManager pm = context.getPackageManager();
            for (String pkg : KNOWN_TTS_ENGINES) {
                if (out.contains(pkg)) continue;
                try {
                    pm.getPackageInfo(pkg, 0);
                    out.add(pkg);
                } catch (android.content.pm.PackageManager.NameNotFoundException ignored) {}
            }
        } catch (Exception ignored) {}
        return out;
    }

    /**
     * Auto-open the system TTS voice-data installer ONCE per process when a
     * synthesis failed to activate the engine — that screen is where the user
     * selects a TTS engine and downloads the Chinese voice data.
     */
    private void offerTtsDataInstallOnce() {
        if (ttsHelpOffered) return;
        ttsHelpOffered = true;
        if (!isAnyTtsEngineInstalled()) return; // no engine → installer can't help
        Log.i(TAG, "Auto-opening system TTS voice-data installer for Chinese setup");
        try {
            ttsMainHandler.post(this::openTtsDataInstall);
        } catch (Exception e) {
            Log.e(TAG, "Could not post TTS data installer launch", e);
        }
    }

    /**
     * True if at least one TTS engine *service* is installed and enabled on the
     * device. Queries the PackageManager directly (flags 0, NOT
     * MATCH_DEFAULT_ONLY) so an installed-but-not-set-as-default engine such as
     * com.google.android.tts is still counted as present — the TextToSpeech
     * getEngines() API's default-only filtering is what wrongly reported "no
     * engine installed" before. Requires the manifest {@code <queries>} entry
     * for the TTS service on Android 11+ (added in AndroidManifest.xml).
     */
    private boolean isAnyTtsEngineInstalled() {
        return !discoveredEngines().isEmpty();
    }

    /** True if the device currently has a TTS engine configured as default. */
    private boolean hasDefaultTtsEngine() {
        try {
            String engine = androidTTS != null ? androidTTS.getDefaultEngine() : null;
            return engine != null && !engine.isEmpty();
        } catch (Exception e) {
            return true; // can't tell — don't fabricate a "set default" hint
        }
    }

    private boolean synthesizeWithSystemTts(String text, String language, String outputPath) {
        // Make sure a bind is at least in flight before we start waiting on it.
        ensureSystemTtsStarted();

        // Give the (async) init a generously long window rather than failing
        // immediately — the engine can legitimately take a few seconds to bind
        // at cold start (and right after the three heavy ONNX models finish
        // loading in parallel, the CPU is still saturated). The old code's
        // short timeout combined with an aggressive shutdown()/recreate-on-
        // timeout raced the slow bind and produced the persistent "System TTS
        // engine not ready" error for every zh call; now we wait once and
        // never destroy an instance whose bind is merely slow. All diagnostics
        // (engine name, init status, missing-engine) are set inside await.
        if (!awaitSystemTtsReady(SYSTEM_TTS_INIT_TIMEOUT_MS)) {
            // lastError set inside awaitSystemTtsReady — open the system
            // voice-data/engine setup screen once so the user can remedy it.
            offerTtsDataInstallOnce();
            return false;
        }
        if (androidTTS == null) {
            lastError = "System TTS engine not ready (null instance)";
            return false;
        }

        // For zh we must NOT silently continue after setLanguage() reports
        // missing data — the old path logged a warning then produced a 0-byte
        // WAV (the "Chinese TTS silently fails" symptom). Fail loudly so
        // getLastError() tells the UI/user exactly what voice data to install.
        Locale locale = resolveSystemLocale(language);
        if (locale == null) {
            return false; // lastError set inside resolveSystemLocale
        }

        File outputFile = new File(outputPath);
        outputFile.getParentFile().mkdirs();

        CountDownLatch latch = new CountDownLatch(1);
        final boolean[] success = {false};
        androidTTS.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override public void onStart(String id) {}
            @Override public void onDone(String id) { success[0] = true; latch.countDown(); }
            @Override public void onError(String id) {
                lastError = "System TTS utterance error for [" + language + "]";
                latch.countDown();
            }
        });

        if (androidTTS.synthesizeToFile(text, null, outputFile, "omnivoice_tts_" + language)
                != TextToSpeech.SUCCESS) {
            lastError = "System TTS synthesizeToFile rejected the request for [" + language + "]";
            return false;
        }
        boolean completed = false;
        try { completed = latch.await(30, TimeUnit.SECONDS); }
        catch (InterruptedException e) {
            Log.e(TAG, "System TTS interrupted", e);
            Thread.currentThread().interrupt();
        }
        if (!completed) {
            lastError = "System TTS timed out synthesizing [" + language + "]";
            return false;
        }
        if (!success[0]) {
            lastError = "System TTS utterance error for [" + language + "]";
            return false;
        }
        if (outputFile.length() <= 44) { // WAV header only = no audio frames
            lastError = "System TTS produced empty audio for [" + language
                    + "] (engine likely has no Chinese voice installed)";
            return false;
        }
        return true;
    }

    /**
     * Pick a {@link Locale} the installed engine can speak for {@code language},
     * trying reasonable variants. For Chinese we prefer Simplified (zh-CN) then
     * Traditional (zh-TW) then the raw "zh" locale, and set a precise
     * {@link #lastError} if none is available.
     *
     * @return a usable locale, or null (with {@link #lastError} set)
     */
    private Locale resolveSystemLocale(String language) {
        List<Locale> candidates = new ArrayList<>();
        if (language.startsWith("zh")) {
            candidates.add(Locale.SIMPLIFIED_CHINESE);   // zh_CN
            candidates.add(Locale.TRADITIONAL_CHINESE);  // zh_TW
            candidates.add(Locale.CHINESE);              // zh
        } else {
            candidates.add(LOCALES.getOrDefault(language, Locale.ENGLISH));
        }

        int lastResult = TextToSpeech.LANG_NOT_SUPPORTED;
        for (Locale loc : candidates) {
            int r = androidTTS.setLanguage(loc);
            if (r >= TextToSpeech.LANG_AVAILABLE) {
                if (!loc.equals(candidates.get(0))) {
                    Log.i(TAG, "System TTS [" + language + "] using fallback locale "
                            + loc + " (primary unavailable, result=" + lastResult + ")");
                }
                return loc;
            }
            lastResult = r;
            Log.w(TAG, "System TTS locale " + loc + " unavailable (result=" + r + ")");
        }

        boolean isChinese = language.startsWith("zh");
        lastError = isChinese
                ? "No TTS voice data installed for Chinese on this device "
                  + "(engine " + safeEngineName() + " reports "
                  + langStatusName(lastResult) + "). Install a Chinese voice or call "
                  + "TTSModule.openTtsDataInstall()."
                : "System TTS language not available for [" + language + "]: "
                  + langStatusName(lastResult) + " (engine " + safeEngineName() + ")";
        Log.e(TAG, lastError);
        return null;
    }

    private static String langStatusName(int r) {
        switch (r) {
            case TextToSpeech.LANG_MISSING_DATA:  return "LANG_MISSING_DATA";
            case TextToSpeech.LANG_NOT_SUPPORTED: return "LANG_NOT_SUPPORTED";
            default:                              return "status(" + r + ")";
        }
    }

    private String safeEngineName() {
        try {
            String def = androidTTS != null ? androidTTS.getDefaultEngine() : null;
            if (def != null && !def.isEmpty()) return def;
            if (requestedEnginePackage != null) return requestedEnginePackage;
            List<TextToSpeech.EngineInfo> engines = androidTTS != null
                    ? androidTTS.getEngines() : null;
            if (engines != null && !engines.isEmpty()) return engines.get(0).name;
            return "none";
        } catch (Exception e) {
            return "unknown";
        }
    }

    /**
     * Launch the system TTS voice-data installer so the user can download a
     * Chinese voice. Wire this to a Settings button and call it when
     * {@link #getLastError()} mentions a missing Chinese voice.
     */
    public void openTtsDataInstall() {
        Intent intent = new Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        context.getApplicationContext().startActivity(intent);
    }

    // ------------------------------------------------------------------
    // MMS-TTS ONNX backend
    // ------------------------------------------------------------------

    private void initMmsOnnx() throws OrtException {
        env = OrtEnvironment.getEnvironment();

        baseDir = context.getExternalFilesDir(null);
        if (baseDir == null) baseDir = context.getFilesDir();

        // Probe + copy assets for every supported language whose ONNX model
        // is bundled, and load the cheap tokenizer/config eagerly. The VITS
        // sessions (~114 MB each) are created LAZILY on first synthesis (see
        // getOrtSession) with at most one resident at a time — on a ≤4 GB
        // wearable this keeps ~114-229 MB of RAM free until TTS is used.
        for (String lang : SUPPORTED) {
            String onnxName = "mms_tts_" + lang + ".onnx";
            String vocabName = "mms_tts_" + lang + "_vocab.json";
            try {
                // Only copy if present in assets
                context.getAssets().open(onnxName).close();
                context.getAssets().open(vocabName).close();

                String onnxPath = FileUtils.copyAssetToDir(context, onnxName, baseDir);
                String vocabPath = FileUtils.copyAssetToDir(context, vocabName, baseDir);

                modelPaths.put(lang, onnxPath);
                tokenizers.put(lang, new MmsTtsTokenizer(vocabPath));
                sampleRates.put(lang, loadSampleRate(lang));
                Log.i(TAG, "Bundled MMS-TTS for [" + lang + "] @ "
                        + sampleRates.get(lang) + "Hz (session lazy)");
            } catch (IOException e) {
                Log.i(TAG, "No bundled MMS-TTS asset for [" + lang + "] (" + onnxName + ")");
            }
        }
    }

    /**
     * Returns the VITS session for a language, creating it on first use.
     * LRU-1: switching languages closes the previous session, so at most one
     * ~114 MB model is resident at a time (wearable RAM budget).
     */
    private synchronized OrtSession getOrtSession(String language) {
        OrtSession session = ortSessions.get(language);
        if (session != null) return session;

        String onnxPath = modelPaths.get(language);
        if (onnxPath == null) return null;

        for (Map.Entry<String, OrtSession> e : ortSessions.entrySet()) {
            try { e.getValue().close(); } catch (Exception ignored) {}
            Log.i(TAG, "TTS evicted session [" + e.getKey() + "] (LRU-1, ~114 MB freed)");
        }
        ortSessions.clear();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            try {
                opts.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
            } catch (OrtException e) { /* extensions optional */ }
            // NOTE: NNAPI is intentionally NOT enabled for MMS-TTS (VITS).
            // The VITS decoder graph is built from ops
            // NNAPI does not support: 1D Conv (NNAPI is 2D-only),
            // RandomNormalLike (the noise sampling node), 1D Resize/Upsample
            // and dynamic-shape Scatter/CumSum duration expansion. Registering
            // NNAPI makes ORT partition the graph at createSession and then
            // throw inside session.run() on the dynamic-sequence Conv1d /
            // RandomNormalLike boundary — which the catch in runOnnxSynthesis
            // swallows, producing the silent "TTS: 0ms" failure. VITS runs
            // fast enough on the CPU EP for these short utterances.
            session = env.createSession(onnxPath, opts);
            ortSessions.put(language, session);
            Log.i(TAG, "TTS [" + language + "] session created (CPU EP, NNAPI off — "
                    + "unsupported VITS ops); model inputs: " + session.getInputNames());
        } catch (OrtException e) {
            session = null;
            Log.e(TAG, "Failed to load MMS-TTS ONNX for [" + language
                    + "] — system TTS fallback will be used", e);
        } finally {
            opts.close();
        }
        return session;
    }

    private boolean hasOnnxModel(String language) {
        return modelPaths.containsKey(language) && tokenizers.containsKey(language);
    }

    /**
     * Run VITS inference with ONNX Runtime.
     *
     * Expected graph I/O (Optimum `ORTModelForTextToWaveform` export):
     *   input  "input_ids"   : int64 [batch, sequence]
     *   output "waveform"    : float32 [batch, 1, time]
     */
    private boolean runOnnxSynthesis(String text, String language, String outputPath) {
        OrtSession session = getOrtSession(language);   // lazy, LRU-1
        MmsTtsTokenizer tokenizer = tokenizers.get(language);
        if (session == null || tokenizer == null) return false;

        try {
            List<String> chunks = chunkText(text, MAX_CHARS);
            List<float[]> waveforms = new ArrayList<>();
            for (String chunk : chunks) {
                float[] w = synthesizeOnnxChunk(chunk, language, session, tokenizer);
                if (w == null) return false; // error already logged
                if (w.length > 0) waveforms.add(w);
            }
            if (waveforms.isEmpty()) {
                lastError = "ONNX synthesis produced no audio";
                Log.e(TAG, "ONNX synthesis produced no audio for: " + text);
                return false;
            }

            int total = 0;
            for (float[] w : waveforms) total += w.length;
            float[] waveform = new float[total];
            int pos = 0;
            for (float[] w : waveforms) {
                System.arraycopy(w, 0, waveform, pos, w.length);
                pos += w.length;
            }

            int sampleRate = sampleRates.getOrDefault(language, DEFAULT_SAMPLE_RATE);
            writeWav(outputPath, waveform, sampleRate);
            Log.i(TAG, "ONNX synthesis OK [" + language + "]: " + chunks.size()
                    + " chunk(s), " + waveform.length + " samples @ " + sampleRate + "Hz");
            return true;
        } catch (Exception e) {
            lastError = "ONNX inference failed [" + language + "]: "
                    + e.getClass().getSimpleName() + ": " + e.getMessage();
            Log.e(TAG, "ONNX synthesis failed for [" + language + "]: " + lastError, e);
            return false;
        }
    }

    /**
     * Synthesize one text chunk through ONNX. Returns null on failure,
     * or an empty float array when the tokenizer produced no tokens
     * (punctuation-only chunk) so other chunks still synthesize.
     */
    private float[] synthesizeOnnxChunk(String text, String language,
                                        OrtSession session, MmsTtsTokenizer tokenizer)
            throws OrtException {
        int[] ids = tokenizer.encode(text, language);
        if (ids.length == 0) {
            Log.w(TAG, "Tokenizer produced no tokens for: " + text);
            return new float[0];
        }

        OnnxTensor input = null;
        OrtSession.Result result = null;
        try {
            // Prefer "input_ids" (Optimum MMS-TTS export), but fall back to the
            // model's actual first input name — a wrong hardcoded name throws
            // instantly inside run() and shows up as a silent "TTS: 0ms".
            java.util.Set<String> inputNames = session.getInputNames();
            String inputName = inputNames.contains("input_ids")
                    ? "input_ids"
                    : inputNames.iterator().next();

            Map<String, OnnxTensor> feeds = new HashMap<>();
            input = TensorUtils.intArrayToTensor(env, ids);
            feeds.put(inputName, input);

            // Some MMS-TTS ONNX exports declare extra inputs besides
            // "input_ids" (e.g. "attention_mask" from the HF Optimum export,
            // or explicit VITS hyper-params such as "speaker_id" /
            // "noise_scale"). ORT throws "Missing Input: <name>" inside run()
            // when any declared input isn't provided — that failure happens
            // before real work, producing the "TTS: 0-1ms" symptom. Feed
            // sensible defaults for every declared input the model asks for.
            List<OnnxTensor> auxTensors = new ArrayList<>();
            for (String name : inputNames) {
                if (name.equals(inputName) || feeds.containsKey(name)) continue;
                OnnxTensor t = createDefaultInputTensor(name, ids.length);
                if (t == null) {
                    Log.e(TAG, "Unsupported extra ONNX input '" + name
                            + "' — cannot synthesize [" + language + "]");
                    return null;
                }
                auxTensors.add(t);
                feeds.put(name, t);
                Log.i(TAG, "Feeding default for ONNX input '" + name + "'");
            }

            result = session.run(feeds);

            // Close the auxiliary default tensors once inference has consumed them.
            for (OnnxTensor t : auxTensors) {
                try { t.close(); } catch (Exception ignored) {}
            }

            OnnxValue waveformVal = null;
            for (String name : session.getOutputNames()) {
                String ln = name.toLowerCase();
                if (ln.contains("waveform") || ln.contains("audio") || ln.contains("wav")) {
                    waveformVal = result.get(name).get();
                    break;
                }
            }
            if (waveformVal == null) {
                waveformVal = result.get(0);
            }
            if (waveformVal == null) {
                Log.e(TAG, "ONNX output is missing for [" + language + "]");
                return null;
            }

            float[] waveform = flattenWaveform(waveformVal);
            if (waveform.length == 0) {
                Log.w(TAG, "ONNX returned empty waveform for: " + text);
            }
            return trimSilence(waveform);
        } finally {
            if (result != null) try { result.close(); } catch (Exception ignored) {}
            if (input != null) try { input.close(); } catch (Exception ignored) {}
        }
    }

    /**
     * Build a default tensor for a non-token ONNX input declared by the
     * MMS-TTS model. Returns null when the input's element type is
     * unsupported by this app.
     *
     * <p>Defaults follow the HF MMS/Optimum conventions: an all-ones
     * [1, seqLen] attention mask, single-speaker id 0, and VITS
     * noise/length scales used by the exported MMS models.</p>
     */
    private OnnxTensor createDefaultInputTensor(String name, int seqLen)
            throws OrtException {
        TensorInfo info = getTensorInfo(name);
        if (info == null) return null;

        float[] fvals = null;
        long[] shape;

        String lower = name.toLowerCase(Locale.ROOT);

        switch (info.type) {
            case FLOAT:
                if (lower.contains("noise_scale_duration") || lower.equals("duration_noise_scale")) {
                    fvals = new float[]{0.667f};
                    shape = new long[]{1};
                } else if (lower.contains("noise_scale")) {
                    fvals = new float[]{0.6f};
                    shape = new long[]{1};
                } else if (lower.contains("length_scale") || lower.contains("speaking_rate")) {
                    fvals = new float[]{1.0f};
                    shape = new long[]{1};
                } else if (lower.contains("attention_mask") || lower.contains("mask")) {
                    fvals = allOnesFloat(seqLen);
                    shape = new long[]{1, seqLen};
                } else if (lower.contains("speaker")) {
                    fvals = new float[]{0f};
                    shape = new long[]{1};
                } else {
                    fvals = new float[]{1.0f};
                    shape = new long[]{1};
                }
                break;

            case DOUBLE:
                if (lower.contains("noise_scale_duration") || lower.equals("duration_noise_scale")) {
                    double[] d = new double[]{0.667};
                    return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1});
                } else if (lower.contains("noise_scale")) {
                    double[] d = new double[]{0.6};
                    return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1});
                } else if (lower.contains("attention_mask") || lower.contains("mask")) {
                    double[] d = new double[seqLen];
                    java.util.Arrays.fill(d, 1.0);
                    return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1, seqLen});
                }
                double[] d = new double[]{1.0};
                return OnnxTensor.createTensor(env, java.nio.DoubleBuffer.wrap(d), new long[]{1});

            case INT64:
            case INT32:
            case INT16:
            case INT8:
            case UINT8:
                long[] lvals;
                if (lower.contains("attention_mask") || lower.contains("mask")) {
                    lvals = allOnesLong(seqLen);
                    shape = new long[]{1, seqLen};
                } else if (lower.contains("speaker")) {
                    lvals = new long[]{0L};
                    shape = new long[]{1};
                } else {
                    lvals = new long[]{1L};
                    shape = new long[]{1};
                }

                if (info.type == OnnxJavaType.INT64) {
                    return OnnxTensor.createTensor(env, LongBuffer.wrap(lvals), shape);
                } else if (info.type == OnnxJavaType.INT32) {
                    int[] iv = toInt(lvals);
                    return OnnxTensor.createTensor(env, java.nio.IntBuffer.wrap(iv), shape);
                } else if (info.type == OnnxJavaType.INT16) {
                    short[] sv = toShort(lvals);
                    return OnnxTensor.createTensor(env, java.nio.ShortBuffer.wrap(sv), shape);
                } else { // INT8 / UINT8
                    byte[] bv = toByte(lvals);
                    return OnnxTensor.createTensor(env, ByteBuffer.wrap(bv), shape);
                }

            case BOOL:
                return OnnxTensor.createTensor(env, new boolean[]{false});

            default:
                return null;
        }

        if (fvals != null) {
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(fvals), shape);
        }
        return null;
    }

    private static float[] allOnesFloat(int n) {
        float[] a = new float[n];
        Arrays.fill(a, 1.0f);
        return a;
    }

    private static long[] allOnesLong(int n) {
        long[] a = new long[n];
        Arrays.fill(a, 1L);
        return a;
    }

    private static int[] toInt(long[] src) {
        int[] out = new int[src.length];
        for (int i = 0; i < src.length; i++) out[i] = (int) src[i];
        return out;
    }

    private static short[] toShort(long[] src) {
        short[] out = new short[src.length];
        for (int i = 0; i < src.length; i++) out[i] = (short) src[i];
        return out;
    }

    private static byte[] toByte(long[] src) {
        byte[] out = new byte[src.length];
        for (int i = 0; i < src.length; i++) out[i] = (byte) src[i];
        return out;
    }

    /** Fetch TensorInfo for the named input from the active session. */
    private TensorInfo getTensorInfo(String inputName) {
        for (OrtSession session : ortSessions.values()) {
            try {
                Object infoObj = session.getInputInfo().get(inputName).getInfo();
                if (infoObj instanceof TensorInfo) return (TensorInfo) infoObj;
            } catch (Exception ignored) {}
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Waveform → WAV helpers
    // ------------------------------------------------------------------

    /**
     * Flatten an OnnxValue (any of [time], [1,time], [1,1,time]) to float[].
     */
    private float[] flattenWaveform(OnnxValue value) throws OrtException {
        Object raw = value.getValue();
        if (raw instanceof float[]) {
            return (float[]) raw;
        } else if (raw instanceof float[][]) {
            return ((float[][]) raw)[0];
        } else if (raw instanceof float[][][]) {
            return ((float[][][]) raw)[0][0];
        }
        return new float[0];
    }

    /**
     * Trim near-silent samples from both ends of a waveform so concatenated
     * chunks don't carry long leading/trailing pauses.
     */
    private static float[] trimSilence(float[] w) {
        if (w == null || w.length == 0) return w;
        final float threshold = 5e-4f;
        final int margin = 400; // keep a short pad to avoid clipped phonemes
        int start = 0;
        while (start < w.length && Math.abs(w[start]) < threshold) start++;
        int end = w.length - 1;
        while (end > start && Math.abs(w[end]) < threshold) end--;
        if (start >= end) return new float[0];
        start = Math.max(0, start - margin);
        end = Math.min(w.length - 1, end + margin);
        return Arrays.copyOfRange(w, start, end + 1);
    }

    /**
     * Split text into chunks of at most {@code maxChars} characters,
     * preferring sentence boundaries (". " / "! " / "? " / newlines) so a
     * chunk never starts mid-word when a sentence split is available.
     */
    private static List<String> chunkText(String text, int maxChars) {
        List<String> out = new ArrayList<>();
        if (text == null) return out;
        String remaining = text.trim();
        while (remaining.length() > maxChars) {
            int cut = -1;
            for (int i = maxChars; i > maxChars / 2; i--) {
                char c = remaining.charAt(i - 1);
                if (c == '.' || c == '!' || c == '?' || c == '\n') { cut = i; break; }
            }
            if (cut < 0) {
                cut = remaining.lastIndexOf(' ', maxChars);
                if (cut < maxChars / 2) cut = maxChars;
            }
            String chunk = remaining.substring(0, cut).trim();
            if (!chunk.isEmpty()) out.add(chunk);
            remaining = remaining.substring(cut).trim();
        }
        if (!remaining.isEmpty()) out.add(remaining);
        if (out.isEmpty()) out.add(text.trim());
        return out;
    }

    /**
     * Read the model's sampling rate from the sibling
     * {@code mms_tts_<lang>_config.json} asset (falls back to
     * {@link #DEFAULT_SAMPLE_RATE}).
     */
    private int loadSampleRate(String lang) {
        String name = "mms_tts_" + lang + "_config.json";
        try (java.io.InputStream in = context.getAssets().open(name)) {
            byte[] data = new byte[in.available()];
            int n = in.read(data);
            String json = n > 0 ? new String(data, 0, n, "UTF-8") : new String(data, "UTF-8");
            JSONObject root = new JSONObject(json);
            int sr = root.optInt("sampling_rate", DEFAULT_SAMPLE_RATE);
            if (sr > 0) return sr;
        } catch (Exception e) {
            Log.d(TAG, "No/invalid " + name + " — using default sample rate");
        }
        return DEFAULT_SAMPLE_RATE;
    }

    /**
     * Maximum character count per ONNX inference call. VITS degrades / OOMs
     * on very long inputs, so long translations are chunked sentence-wise
     * and concatenated.
     */
    private static final int MAX_CHARS = 220;

    /**
     * Write a mono 16-bit PCM WAV file.
     */
    private void writeWav(String path, float[] samples, int sampleRate) throws IOException {
        File f = new File(path);
        f.getParentFile().mkdirs();

        int dataSize = samples.length * 2;
        ByteBuffer buf = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN);

        // RIFF header
        buf.put("RIFF".getBytes());
        buf.putInt(36 + dataSize);          // total chunk size
        buf.put("WAVE".getBytes());

        // fmt sub-chunk
        buf.put("fmt ".getBytes());
        buf.putInt(16);                     // PCM sub-chunk size
        buf.putShort((short) 1);            // PCM format
        buf.putShort((short) 1);            // mono
        buf.putInt(sampleRate);
        buf.putInt(sampleRate * 2);         // byte rate
        buf.putShort((short) 2);            // block align
        buf.putShort((short) 16);           // bits per sample

        // data sub-chunk
        buf.put("data".getBytes());
        buf.putInt(dataSize);
        for (float s : samples) {
            float clamped = Math.max(-1.0f, Math.min(1.0f, s));
            buf.putShort((short) (clamped * 32767f));
        }

        try (RandomAccessFile raf = new RandomAccessFile(f, "rw")) {
            raf.setLength(0);
            raf.write(buf.array());
        }
    }

    /** Normalize language codes like "zh_hans" / "zh_hant" to "zh". */
    private static String normalizeLang(String language) {
        if (language == null) return "vi";
        if (language.startsWith("zh")) return "zh";
        if (language.equals("vi") || language.equals("en")) return language;
        return language;
    }
}


