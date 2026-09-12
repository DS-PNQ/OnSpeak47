/*
 * OmniVoice — Hy-MT 1.25-bit GGUF JNI Bridge (llama.cpp)
 */

package com.omnivoice.onspeak47.pipeline;

public final class HyMtGgufJNI {

    static {
        try {
            System.loadLibrary("hymt_gguf");
        } catch (UnsatisfiedLinkError e) {
            // Logged when running in tests or if native lib not present
        }
    }

    private HyMtGgufJNI() {}

    /**
     * Load a GGUF model from the specified file path.
     *
     * @param modelPath Absolute path to the .gguf file.
     * @param nCtx Context window size (e.g. 2048).
     * @param nThreads Number of CPU threads to use for decoding.
     * @return Native handle pointer (>0 on success, 0 on failure).
     */
    public static native long loadModel(String modelPath, int nCtx, int nThreads);

    /**
     * Complete an instruction prompt using the loaded GGUF model.
     *
     * @param handle Native handle from loadModel.
     * @param prompt Instruction text.
     * @param maxTokens Maximum new tokens to generate.
     * @return Generated translation string.
     */
    public static native String complete(long handle, String prompt, int maxTokens);

    /**
     * Release and free resources associated with the loaded model.
     *
     * @param handle Native handle from loadModel.
     */
    public static native void freeModel(long handle);
}
