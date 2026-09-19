/*
 * OmniVoice — Zipformer model lifecycle (preload / lazy-load per RAM bucket).
 *
 * Spec §5 + §21 MUST 7: persistent model instances, never load/unload per
 * language switch.
 *
 * Mixed EN/ZH plan §5.1: the pool holds TWO native sessions, keyed by
 * {@link AsrModelType} — VI and the shared EN/ZH bilingual model. EN and ZH
 * used to be separate slots that loaded the SAME files into two separate
 * recognizers, so "VI+EN+ZH resident" actually allocated three sessions to
 * cover two models. Buckets: >=6 GB → VI + EN_ZH resident; below → VI
 * resident, EN_ZH lazy. Under memory pressure only the active + candidate
 * buckets stay resident.
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
    private final Map<AsrModelType, StreamingAsrEngine> resident =
            new EnumMap<>(AsrModelType.class);
    private final Map<AsrModelType, Assets> assets = new EnumMap<>(AsrModelType.class);

    public ZipformerModelManager(Context context) {
        this(context, null);
    }

    public ZipformerModelManager(Context context, StreamingAsrEngine.Factory factory) {
        this.appContext = context.getApplicationContext();
        // The factory still speaks AsrLanguage (engine construction needs a
        // concrete language for FakeEngine/logs); the bucket decides identity.
        this.factory = factory != null ? factory
                : lang -> StreamingAsrEngine.create(appContext, lang,
                        assetsFor(AsrModelType.of(lang)));
    }

    /** Preload per the device RAM bucket (spec §5, mixed EN/ZH plan §5.1). */
    public synchronized void preloadForDevice() {
        long totalMem = totalMemBytes();
        // Two models, not three: EN and ZH share the bilingual encoder, so
        // 6 GB devices can now keep BOTH buckets resident (the old bucket
        // table kept EN resident and left ZH lazy only because ZH was a
        // second 161 MB session over different files).
        if (totalMem >= 6L * 1024 * 1024 * 1024) {
            ensure(AsrModelType.VI);
            ensure(AsrModelType.EN_ZH);
            Log.i(TAG, ">=6GB: VI + EN_ZH (shared bilingual) resident");
        } else {
            ensure(AsrModelType.VI);
            Log.i(TAG, "<6GB: VI resident, EN_ZH lazy");
        }
    }

    /**
     * Get (creating + caching on first use) the engine for a language. EN and
     * ZH resolve to the SAME engine instance (one bilingual model).
     */
    public synchronized StreamingAsrEngine get(AsrLanguage lang) {
        // §13: UND is "no language yet", never a model slot. Fail fast so a
        // caller bug can't silently decode on a null-asset engine.
        if (lang == null || lang == AsrLanguage.UND) {
            throw new IllegalArgumentException("no engine for UND (bootstrap first)");
        }
        return ensure(AsrModelType.of(lang));
    }

    /** Get the engine for a model bucket (VI / EN_ZH). */
    public synchronized StreamingAsrEngine get(AsrModelType type) {
        if (type == null || type == AsrModelType.UND) {
            throw new IllegalArgumentException("no engine for UND (bootstrap first)");
        }
        return ensure(type);
    }

    public synchronized boolean isResident(AsrLanguage lang) {
        return lang != null && resident.containsKey(AsrModelType.of(lang));
    }

    public synchronized boolean isResident(AsrModelType type) {
        return type != null && resident.containsKey(type);
    }

    /** Drop everything except active + candidate buckets under memory pressure. */
    public synchronized void trimTo(AsrLanguage active, AsrLanguage candidate) {
        trimTo(AsrModelType.of(active), AsrModelType.of(candidate));
    }

    /** Drop everything except active + candidate buckets under memory pressure. */
    public synchronized void trimTo(AsrModelType active, AsrModelType candidate) {
        for (AsrModelType t : new AsrModelType[]{AsrModelType.VI, AsrModelType.EN_ZH}) {
            if (t != active && t != candidate && resident.containsKey(t)) {
                try {
                    resident.get(t).close();
                } catch (Exception ignored) {
                }
                resident.remove(t);
                Log.i(TAG, "trimmed " + t.code + " under memory pressure");
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

    private StreamingAsrEngine ensure(AsrModelType type) {
        StreamingAsrEngine e = resident.get(type);
        if (e != null) return e;
        // The bucket's canonical label is what the engine is constructed with
        // (EN_ZH → EN). language() is informational only; identity is the
        // bucket, so EN and ZH can never allocate two sessions again.
        AsrLanguage lang = AsrLanguage.canonicalOf(type);
        try {
            e = factory.create(lang);
            resident.put(type, e);
            return e;
        } catch (Exception ex) {
            Log.e(TAG, "engine create failed for " + type.code, ex);
            StreamingAsrEngine fake = new StreamingAsrEngine.FakeEngine(lang);
            resident.put(type, fake);
            return fake;
        }
    }

    private Assets assetsFor(AsrModelType type) {
        Assets a = assets.get(type);
        if (a != null) return a;
        AsrLanguage lang = AsrLanguage.canonicalOf(type);
        switch (type) {
            case VI:
                a = resolve(lang, AsrState.VI_ENCODER, AsrState.VI_DECODER,
                        AsrState.VI_JOINER, AsrState.VI_TOKENS);
                break;
            case EN_ZH:
                // ONE bilingual model serves both labels (mixed EN/ZH plan).
                a = resolve(lang, AsrState.EN_ZH_ENCODER, AsrState.EN_ZH_DECODER,
                        AsrState.EN_ZH_JOINER, AsrState.EN_ZH_TOKENS);
                break;
            default:
                a = new Assets(lang, null, null, null, null);
                break;
        }
        assets.put(type, a);
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
