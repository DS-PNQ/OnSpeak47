// OmniVoice — Hy-MT 1.25-bit GGUF JNI bridge (llama.cpp, greedy decoding)
//
// Decode path tuned from the RTranslator comparison:
//  - exact Hy-MT chat prefix by token id: the GGUF chat_template starts with
//    <｜hy_begin▁of▁sentence｜> (bos, id 120000) which the gpt2-pre tokenizer
//    (pre = hunyuan-dense) does NOT auto-insert — it must be prepended
//    manually, exactly like RTranslator's Tokenizer does for HY_MT;
//  - greedy (argmax) sampling: MT output is deterministic, sampling adds RNG
//    cost and translation regressions;
//  - generation stops on the real end-of-assistant token
//    <｜hy_place▁holder▁no▁2｜> (eos_token_id 120020) — matches RTranslator;
//  - per-call timing logs so tokens/s can be compared against RTranslator.

#include <jni.h>
#include <android/log.h>
#include <chrono>
#include <string>
#include <vector>
#include "llama.h"

#define TAG "HyMtGgufJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct HyMtModelContext {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_threads = 4;
};

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_omnivoice_onspeak47_pipeline_HyMtGgufJNI_loadModel(
    JNIEnv* env, jclass clazz, jstring jModelPath, jint nCtx, jint nThreads) {
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    if (!modelPath) return 0;

    LOGI("Loading GGUF model from: %s", modelPath);

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(modelPath, model_params);
    env->ReleaseStringUTFChars(jModelPath, modelPath);

    if (!model) {
        LOGE("Failed to load llama model from file");
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    // Translation prompts are short (<200 tok) + gen <= 128 tok: 1024 covers
    // everything with far less KV memory-bandwidth than 2048. Java passes
    // 1024 (see TranslationModule); the fallback here matches it.
    ctx_params.n_ctx = nCtx > 0 ? nCtx : 1024;
    ctx_params.n_threads = nThreads > 0 ? nThreads : 4;
    ctx_params.n_threads_batch = ctx_params.n_threads;
    // Single large prefill batch: the whole HY-MT prompt decodes in one
    // llama_decode() call instead of being chunked.
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create llama context");
        llama_model_free(model);
        return 0;
    }

    auto* holder = new HyMtModelContext();
    holder->model = model;
    holder->ctx = ctx;
    holder->vocab = llama_model_get_vocab(model);
    holder->n_threads = ctx_params.n_threads;

    LOGI("Hy-MT GGUF model loaded successfully (handle=%p, threads=%d)", (void*) holder, holder->n_threads);
    return reinterpret_cast<jlong>(holder);
}

JNIEXPORT jstring JNICALL
Java_com_omnivoice_onspeak47_pipeline_HyMtGgufJNI_complete(
    JNIEnv* env, jclass clazz, jlong handle, jstring jPrompt, jint maxTokens) {
    if (handle == 0) {
        return env->NewStringUTF("[error: model not loaded]");
    }
    auto* holder = reinterpret_cast<HyMtModelContext*>(handle);
    const char* promptCStr = env->GetStringUTFChars(jPrompt, nullptr);
    if (!promptCStr) return env->NewStringUTF("");

    // Chat-format body; the special markers are parsed into their single
    // control tokens by llama_tokenize(parse_special=true) below.
    std::string prompt = "<｜hy_User｜>" + std::string(promptCStr) + "<｜hy_Assistant｜>";
    env->ReleaseStringUTFChars(jPrompt, promptCStr);

    const llama_vocab* vocab = holder->vocab;

    // Tokenize WITHOUT auto-added specials (add_special=false): we control
    // the BOS token by hand below. First pass sizes the token buffer.
    const int32_t prompt_len = (int32_t) prompt.length();
    int n_body = -llama_tokenize(vocab, prompt.c_str(), prompt_len, nullptr, 0, false, true);
    if (n_body < 0) {
        LOGE("Failed to tokenize prompt");
        return env->NewStringUTF("[error: tokenization failed]");
    }

    std::vector<llama_token> tokens;
    tokens.reserve(n_body + 1);
    // The Hy-MT chat_template starts with <｜hy_begin▁of▁sentence｜>; prepend
    // the model's BOS token id explicitly (this tokenizer never adds it).
    llama_token bos = llama_vocab_bos(vocab);
    if (bos != LLAMA_TOKEN_NULL) tokens.push_back(bos);
    const size_t n_head = tokens.size();
    tokens.resize(n_head + n_body);
    if (llama_tokenize(vocab, prompt.c_str(), prompt_len, tokens.data() + n_head, n_body, false, true) < 0) {
        LOGE("Failed to tokenize prompt (pass 2)");
        return env->NewStringUTF("[error: tokenization failed]");
    }

    llama_memory_clear(llama_get_memory(holder->ctx), true);

    const auto t_prefill0 = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(holder->ctx, batch) != 0) {
        LOGE("llama_decode failed on prompt batch");
        return env->NewStringUTF("[error: decode failed]");
    }
    const double prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_prefill0).count();

    // Greedy decode — deterministic and cheaper than top-k/top-p/temp chains.
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // The Hy-MT end-of-assistant token: the GGUF eos_token_id (120020,
    // "<｜hy_place▁holder▁no▁2｜>"), the same stop token RTranslator uses.
    llama_token eos = llama_vocab_eos(vocab);

    std::string response = "";
    int cur_tokens = 0;
    // Clamp runaway requests: ASR transcripts never need >256 tok; Java
    // already budgets 48-128 by input length (see TranslationModule).
    int max_gen = maxTokens > 0 ? maxTokens : 128;
    if (max_gen > 256) max_gen = 256;

    // Anti-loop guard (mirrors RTranslator's j > 3-8x input_len early-stop):
    // break when one token repeats consecutively (model stuck) instead of
    // burning the full budget. 10 is conservative — real translations never
    // emit the same subword 10x in a row.
    llama_token last_token = LLAMA_TOKEN_NULL;
    int repeat_run = 0;

    const auto t_gen0 = std::chrono::steady_clock::now();
    while (cur_tokens < max_gen) {
        llama_token new_token_id = llama_sampler_sample(smpl, holder->ctx, -1);
        llama_sampler_accept(smpl, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)
                || (eos != LLAMA_TOKEN_NULL && new_token_id == eos)) {
            break;
        }

        if (new_token_id == last_token) {
            if (++repeat_run >= 10) break;
        } else {
            last_token = new_token_id;
            repeat_run = 0;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            response.append(buf, n);
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(holder->ctx, batch) != 0) {
            LOGE("llama_decode failed during generation step");
            break;
        }
        cur_tokens++;
    }
    llama_sampler_free(smpl);

    const double gen_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_gen0).count();
    LOGI("decode: prompt=%d tok (prefill %.0f ms) + %d tok (gen %.0f ms, %.1f tok/s) [threads=%d]",
         (int) tokens.size(), prefill_ms, cur_tokens, gen_ms,
         gen_ms > 0.0 ? cur_tokens * 1000.0 / gen_ms : 0.0, holder->n_threads);

    return env->NewStringUTF(response.c_str());
}

JNIEXPORT void JNICALL
Java_com_omnivoice_onspeak47_pipeline_HyMtGgufJNI_freeModel(
    JNIEnv* env, jclass clazz, jlong handle) {
    if (handle == 0) return;
    auto* holder = reinterpret_cast<HyMtModelContext*>(handle);
    if (holder->ctx) {
        llama_free(holder->ctx);
    }
    if (holder->model) {
        llama_model_free(holder->model);
    }
    delete holder;
    LOGI("Hy-MT GGUF model freed");
}

} // extern "C"
