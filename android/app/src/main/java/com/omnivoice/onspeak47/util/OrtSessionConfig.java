/*
 * OmniVoice — Mobile-tuned ONNX Runtime session options
 */

package com.omnivoice.onspeak47.util;

import android.app.ActivityManager;
import android.content.Context;
import android.util.Log;

import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.extensions.OrtxPackage;

/**
 * Shared factory for OrtSession.SessionOptions tuned for on-device
 * inference, following RTranslator's production configuration:
 *
 * <ul>
 *   <li>Low-RAM devices (&le; 7 GB, RTranslator's cutoff): the CPU arena
 *       allocator and memory-pattern planner are disabled. Both pre-allocate
 *       from worst-case graph shapes and can inflate resident RAM
 *       substantially; disabling them trades a little speed for much lower
 *       memory. High-RAM devices keep both enabled.</li>
 *   <li>NNAPI is off by default: with dynamic-int8 quantized models the NNAPI
 *       EP frequently falls back to CPU per-node and adds conversion
 *       overhead; RTranslator ships CPU-only. Flip {@link #USE_NNAPI} to
 *       A/B test on device.</li>
 *   <li>Graph optimization stays at ALL_OPT because the bundled .onnx files
 *       are not pre-optimized offline. (To move optimization offline: save an
 *       optimized model via Python's SessionOptions.optimized_model_filepath,
 *       ship that file, and switch to NO_OPT here — faster session creation
 *       and less runtime RAM.)</li>
 * </ul>
 */
public final class OrtSessionConfig {

    private static final String TAG = "OrtSessionConfig";

    /** RTranslator's cutoff between "low memory" and normal session config. */
    private static final long LOW_RAM_THRESHOLD_BYTES = 7L * 1024 * 1024 * 1024;

    /**
     * Enable the NNAPI execution provider for ASR/translation sessions.
     * Read the class documentation before flipping — with the current
     * dynamic-int8 models this usually measures slower than pure CPU.
     */
    public static final boolean USE_NNAPI = false;

    /**
     * Enable the XNNPACK execution provider (experimental, Phase 3.1):
     * accelerates fp32 Conv/MatMul workloads. A/B on device with the
     * PipelineOrchestrator timing/MEM[...] logs before making this true
     * by default.
     */
    public static final boolean USE_XNNPACK = false;

    private OrtSessionConfig() {}

    /**
     * Creates session options with the ort-extensions custom-op library
     * optionally registered (needed by graphs using Mel/BPE/tensor ops).
     * The caller owns the returned options and should close them after
     * creating its sessions.
     */
    public static OrtSession.SessionOptions create(Context context, boolean registerOrtx) throws OrtException {
        return create(context, registerOrtx, false);
    }

    /**
     * @param preOptimized true when loading a graph already optimized offline
     *                     by optimize/07_preoptimize.py (*.opt.onnx) — runtime
     *                     re-optimization is then skipped (NO_OPT), removing
     *                     the multi-second createSession pass and its peak-RAM
     *                     spike on device.
     */
    public static OrtSession.SessionOptions create(Context context, boolean registerOrtx,
                                                   boolean preOptimized) throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        if (registerOrtx) {
            try {
                options.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
            } catch (OrtException e) {
                Log.e(TAG, "Extensions library not found", e);
            }
        }
        boolean lowRam = isLowRamDevice(context);
        options.setCPUArenaAllocator(!lowRam);
        options.setMemoryPatternOptimization(!lowRam);
        options.setOptimizationLevel(preOptimized
                ? OrtSession.SessionOptions.OptLevel.NO_OPT
                : OrtSession.SessionOptions.OptLevel.ALL_OPT);
        if (USE_NNAPI) {
            try {
                options.addNnapi();
                Log.i(TAG, "NNAPI execution provider enabled");
            } catch (OrtException e) {
                Log.w(TAG, "NNAPI not available, using CPU fallback", e);
            }
        }
        if (USE_XNNPACK) {
            try {
                Map<String, String> xnnpackOpts = new HashMap<>();
                xnnpackOpts.put("intra_op_num_threads",
                        String.valueOf(Math.max(2, Runtime.getRuntime().availableProcessors() / 2)));
                options.addXnnpack(xnnpackOpts);
                Log.i(TAG, "XNNPACK execution provider enabled");
            } catch (OrtException e) {
                Log.w(TAG, "XNNPACK not available, using CPU fallback", e);
            }
        }
        Log.i(TAG, "Session options: arena=" + !lowRam + ", memPattern=" + !lowRam
                + ", nnapi=" + USE_NNAPI + ", optLevel="
                + (preOptimized ? "NO_OPT (pre-optimized graph)" : "ALL_OPT")
                + " (lowRamDevice=" + lowRam + ")");
        return options;
    }

    private static boolean isLowRamDevice(Context context) {
        try {
            ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            if (am == null) return true;
            ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
            am.getMemoryInfo(info);
            return info.totalMem <= LOW_RAM_THRESHOLD_BYTES;
        } catch (Exception e) {
            return true;   // assume constrained when unknown
        }
    }
}
