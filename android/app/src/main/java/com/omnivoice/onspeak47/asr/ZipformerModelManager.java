/*
 * OmniVoice — Zipformer model lifecycle (preload / lazy-load per RAM bucket).
 *
 * Spec §5 + §21 MUST 7: persistent model instances, never load/unload per
 * language switch. Buckets: >=8 GB → VI+EN+ZH resident; 6 GB → VI+EN
 * resident + ZH lazy/mmap; 4 GB → VI resident, EN/ZH lazy. Under memory
 * pressure only active + candidate stay resident.
 */
package com.omnivoice.onspeak47.asr;

import android.app.ActivityManager;
import android.content.Context;
import android.util.Log;

import java.io.File;
import java.util.EnumMap;
import java.util.Map;

import com.omnivoice.onspeak47.util.FileUtils;

public class ZipformerModelManager {

    private static final String TAG = "ZipformerModelManager";

    /** Resolved on-disk asset paths for one language. */
    public static class Assets {
        public final AsrLanguage lang;
        public final String encoderPath;
        public final String decoderPath;
        public final String joinerPath;
        public final String tokensPath;

        Assets(AsrLanguage lang, String encoderPath, String decoderPath,
               String joinerPath, String tokensPath) {
            this.lang = lang;
            this.encoderPath = encoderPath;
            this.decoderPath = decoderPath;
            this.joinerPath = joinerPath;
            this.tokensPath = tokensPath;
        }

        public boolean allExist() {
            return exists(encoderPath) && exists(decoderPath)
                    && exists(joinerPath) && exists(tokensPath);
        }

        /** Human-readable list of absent files for diagnostics. */
        public String missing() {
            StringBuilder sb = new StringBuilder();
            if (!exists(encoderPath)) sb.append("encoder=").append(encoderPath).append(' ');
            if (!exists(decoderPath)) sb.append("decoder=").append(decoderPath).append(' ');
            if (!exists(joinerPath)) sb.append("joiner=").append(joinerPath).append(' ');
            if (!exists(tokensPath)) sb.append("tokens=").append(tokensPath).append(' ');
            return sb.toString().trim();
        }

        private static boolean exists(String p) {
            return p != null && new File(p).exists() && new File(p).length() > 0;
        }
    }

    private final Context appContext;
    private final StreamingAsrEngine.Factory factory;
    private final Map<AsrLanguage, StreamingAsrEngine> resident =
            new EnumMap<>(AsrLanguage.class);
    private final Map<AsrLanguage, Assets> assets = new EnumMap<>(AsrLanguage.class);

    public ZipformerModelManager(Context context) {
        this(context, null);
    }

    public ZipformerModelManager(Context context, StreamingAsrEngine.Factory factory) {
        this.appContext = context.getApplicationContext();
        this.factory = factory != null ? factory
                : lang -> StreamingAsrEngine.create(appContext, lang, assetsFor(lang));
    }

    /** Preload per the device RAM bucket (spec §5). */
    public synchronized void preloadForDevice() {
        long totalMem = totalMemBytes();
        if (totalMem >= 8L * 1024 * 1024 * 1024) {
            ensure(AsrLanguage.VI);
            ensure(AsrLanguage.EN);
            ensure(AsrLanguage.ZH);
            Log.i(TAG, ">=8GB: VI+EN+ZH resident");
        } else if (totalMem >= 6L * 1024 * 1024 * 1024) {
            ensure(AsrLanguage.VI);
            ensure(AsrLanguage.EN);
            Log.i(TAG, "6GB: VI+EN resident, ZH lazy");
        } else {
            ensure(AsrLanguage.VI);
            Log.i(TAG, "<6GB: VI resident, EN/ZH lazy");
        }
    }

    /** Get (creating + caching on first use) the engine for a language. */
    public synchronized StreamingAsrEngine get(AsrLanguage lang) {
        // §13: UND is "no language yet", never a model slot. Fail fast so a
        // caller bug can't silently decode on a null-asset engine.
        if (lang == null || lang == AsrLanguage.UND) {
            throw new IllegalArgumentException("no engine for UND (bootstrap first)");
        }
        return ensure(lang);
    }

    public synchronized boolean isResident(AsrLanguage lang) {
        return resident.containsKey(lang);
    }

    /** Drop everything except active + candidate under memory pressure. */
    public synchronized void trimTo(AsrLanguage active, AsrLanguage candidate) {
        for (AsrLanguage l : new AsrLanguage[]{AsrLanguage.VI, AsrLanguage.EN, AsrLanguage.ZH}) {
            if (l != active && l != candidate && resident.containsKey(l)) {
                try {
                    resident.get(l).close();
                } catch (Exception ignored) {
                }
                resident.remove(l);
                Log.i(TAG, "trimmed " + l + " under memory pressure");
            }
        }
    }

    public synchronized void close() {
        for (StreamingAsrEngine e : resident.values()) {
            try {
                e.close();
            } catch (Exception ignored) {
            }
        }
        resident.clear();
    }

    // --- Internals ---------------------------------------------------

    private StreamingAsrEngine ensure(AsrLanguage lang) {
        StreamingAsrEngine e = resident.get(lang);
        if (e != null) return e;
        try {
            e = factory.create(lang);
            resident.put(lang, e);
            return e;
        } catch (Exception ex) {
            Log.e(TAG, "engine create failed for " + lang, ex);
            StreamingAsrEngine fake = new StreamingAsrEngine.FakeEngine(lang);
            resident.put(lang, fake);
            return fake;
        }
    }

    private Assets assetsFor(AsrLanguage lang) {
        Assets a = assets.get(lang);
        if (a != null) return a;
        switch (lang) {
            case VI:
                a = resolve(lang, AsrState.VI_ENCODER, AsrState.VI_DECODER,
                        AsrState.VI_JOINER, AsrState.VI_TOKENS);
                break;
            case EN:
                a = resolve(lang, AsrState.EN_ENCODER, AsrState.EN_DECODER,
                        AsrState.EN_JOINER, AsrState.EN_TOKENS);
                break;
            case ZH:
                a = resolve(lang, AsrState.ZH_ENCODER, AsrState.ZH_DECODER,
                        AsrState.ZH_JOINER, AsrState.ZH_TOKENS);
                break;
            default:
                a = new Assets(lang, null, null, null, null);
                break;
        }
        assets.put(lang, a);
        return a;
    }

    private Assets resolve(AsrLanguage lang, String enc, String dec, String join, String tok) {
        return new Assets(lang,
                copyOptional(enc), copyOptional(dec), copyOptional(join), copyOptional(tok));
    }

    private String copyOptional(String asset) {
        try {
            appContext.getAssets().open(asset).close();
        } catch (Exception e) {
            return new File(appContext.getFilesDir(), asset).getAbsolutePath(); // lazy path
        }
        try {
            return FileUtils.copyAssetToInternal(appContext, asset);
        } catch (Exception e) {
            Log.w(TAG, "asset copy failed: " + asset);
            return null;
        }
    }

    private long totalMemBytes() {
        try {
            ActivityManager am = (ActivityManager) appContext.getSystemService(Context.ACTIVITY_SERVICE);
            if (am == null) return 0;
            ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
            am.getMemoryInfo(mi);
            return mi.totalMem;
        } catch (Exception e) {
            return 0;
        }
    }
}
