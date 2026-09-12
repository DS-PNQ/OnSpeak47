"""Hy-MT1.5 translation wrapper — replaces facebook/nllb-200-distilled-600M.

HY-MT1.5 is a *decoder-only* translation LLM, structurally unrelated to
NLLB's encoder/decoder seq2seq graph:

* there are no forced-BOS FLORES-200 language tokens and no SentencePiece
  BPE vocab — the target language is steered by a natural-language
  instruction prompt (Tencent-Hunyuan/Hy-MT prompt templates);
* autoregressive greedy decoding (beamSize=1, like RTranslator) replaces
  beam-search over encoder hidden states;
* two on-device profiles are supported (project pipeline diagram):

  - low_ram  : 1.25-bit STQ GGUF (Sherry, 440 MB) from
               tencent/Hy-MT1.5-1.8B-1.25bit-GGUF, decoded on the mobile
               CPU by llama.cpp + STQ kernel (llama.cpp PR #22836);
               fits phones with 4-6 GB RAM;
  - high_end : INT4 ONNX graph executed by ONNX Runtime GenAI
               (QNN EP on Snapdragon, CPU EP elsewhere).

The public API (translate / translate_batch / translate_file /
translate_pivot / preprocess_chinese) is unchanged from NLLBTranslator, so
the orchestrator and the local tests keep working as-is.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

MODEL_REPO = "tencent/Hy-MT1.5-1.8B-1.25bit-GGUF"
GGUF_FILENAME = "Hy-MT1.5-1.8B-1.25bit.gguf"

GGUF_PATH = Path(os.environ.get("HYMT_GGUF_PATH", f"onnx_models/{GGUF_FILENAME}"))
INT4_ONNX_DIR = Path(os.environ.get("HYMT_INT4_ONNX_DIR", "onnx_models/hymt_int4_onnx"))

# Same simple language keys the rest of the pipeline uses.
LANG_CODES = {
    "en": "en",
    "vi": "vi",
    "zh_hans": "zh",
    "zh_hant": "zh-Hant",
    "zh": "zh",
}

# HY-MT prompt templates (Tencent-Hunyuan/Hy-MT README):
#   ZH<=>XX : Chinese instruction, Chinese language names
#   XX<=>XX : English instruction, English language names
LANG_NAMES = {
    "en": ("English", "英语"),
    "vi": ("Vietnamese", "越南语"),
    "zh": ("Chinese", "中文"),
    "zh_hans": ("Chinese", "中文"),
    "zh_hant": ("Traditional Chinese", "繁体中文"),
}

PROMPT_ZH = "将以下文本翻译为{tgt}，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"
PROMPT_XX = (
    "Translate the following segment into {tgt}, without additional explanation.\n\n{src}"
)

# Greedy decoding (matches RTranslator beamSize=1 + on-device JNI path).
# MT output is deterministic: sampling (top_k/top_p/temp/repeat) only adds
# RNG cost, longer runaway outputs and translation regressions.
# The HY-MT report values (top_k=20/top_p=0.6/temp=0.7/repeat=1.05) are kept
# below for reference but are NOT used by either backend path.
GEN_TOP_K = 20
GEN_TOP_P = 0.6
GEN_TEMP = 0.7
GEN_REPEAT_PENALTY = 1.05
# ASR transcripts are 1-2 sentences: 128 covers them; long paragraphs fall
# back to 256 via _budget(). Never burn 256 tok on a short "xin chào".
MAX_NEW_TOKENS = 128
MAX_NEW_TOKENS_LONG = 256
# Translation prompts are short (<200 tok): 1024 KV window is plenty and far
# cheaper than 2048 on both llama-cpp-python and ORT-GenAI.
N_CTX = 1024


def _budget(text: str) -> int:
    """Token budget by input length (mirrors TranslationModule.java)."""
    if len(text) < 200:
        return MAX_NEW_TOKENS
    return MAX_NEW_TOKENS_LONG


def correct_text(text: str) -> str:
    """Port of RTranslator correctText(): terminator + whitespace collapse."""
    t = text.strip()
    if len(t) >= 2 and t[-1].isalnum():
        t += "."
    return re.sub(r"\s+", " ", t)


def build_instruction(text: str, src_lang: str, tgt_lang: str) -> str:
    """Render the HY-MT instruction prompt for one segment (no chat framing)."""
    if LANG_CODES[src_lang] == "zh" or LANG_CODES[tgt_lang] == "zh":
        return PROMPT_ZH.format(tgt=LANG_NAMES[tgt_lang][1], src=text)
    return PROMPT_XX.format(tgt=LANG_NAMES[tgt_lang][0], src=text)


def render_chat_prompt(instruction: str) -> str:
    """Wrap an instruction in Hy-MT's chat framing.

    Mirrors the chat template embedded in the GGUF metadata / tokenizer
    (special tokens <｜hy_User｜> / <｜hy_Assistant｜>).  The llama.cpp
    backend applies the embedded template itself, so only the ORT-GenAI
    path uses this helper.
    """
    return f"<｜hy_User｜>{instruction}<｜hy_Assistant｜>"


def _clean(raw: str) -> str:
    """Strip chat markers / stray reasoning tags from a completion."""
    raw = raw.split("<｜hy_")[0]
    raw = re.sub(r"</?(think|answer)>", "", raw)
    return raw.strip()


class HyMTTranslator:
    """Dual-profile Hy-MT1.5 translator (drop-in replacement for NLLBTranslator)."""

    def __init__(
        self,
        profile: str = "auto",
        gguf_path: str | Path = GGUF_PATH,
        onnx_dir: str | Path = INT4_ONNX_DIR,
    ):
        self.gguf_path = Path(gguf_path)
        self.onnx_dir = Path(onnx_dir)
        forced = os.environ.get("HYMT_PROFILE", profile)
        if forced not in ("auto", "low_ram", "high_end"):
            raise ValueError(f"unknown Hy-MT profile: {forced}")
        if forced == "auto":
            forced = "high_end" if self.onnx_dir.exists() else "low_ram"
        self.profile = forced
        self._backend = None

    # ------------------------------------------------------------------
    # Backend loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        if self.profile == "low_ram":
            from llama_cpp import Llama  # STQ kernel build required for 1.25-bit

            self._backend = Llama(
                model_path=str(self.gguf_path),
                n_ctx=N_CTX,
                n_gpu_layers=0,  # STQ kernel is CPU-only
                n_threads=os.cpu_count() or 4,
                verbose=False,
            )
        else:
            import onnxruntime_genai as og

            model = og.Model(str(self.onnx_dir))
            self._backend = (og, model, og.Tokenizer(model))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, instruction: str) -> str:
        self._ensure_loaded()
        budget = _budget(instruction)
        if self.profile == "low_ram":
            # Greedy: temp=0/top_k=1/top_p=1/repeat=1 — deterministic MT,
            # no RNG/top-p sort cost per token, stops at EOS quickly.
            out = self._backend.create_chat_completion(
                messages=[{"role": "user", "content": instruction}],
                max_tokens=budget,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                repeat_penalty=1.0,
            )
            return _clean(out["choices"][0]["message"]["content"])

        og, model, tokenizer = self._backend
        stream = tokenizer.create_stream()
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=512,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            repetition_penalty=1.0,
        )
        generator = og.Generator(model, params)
        generator.append_tokens(tokenizer.encode(render_chat_prompt(instruction)))
        chunks: list[str] = []
        while not generator.is_done():
            generator.generate_next_token()
            chunks.append(stream.decode(generator.get_next_tokens()[0]))
        return _clean("".join(chunks))

    # ------------------------------------------------------------------
    # Public API (same contract as the old NLLBTranslator)
    # ------------------------------------------------------------------

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text or not text.strip():
            return ""
        # Single call per transcript (no per-sentence prefill loop — the
        # Java path handles long-paragraph splitting, see TranslationModule).
        return self._generate(build_instruction(correct_text(text), src_lang, tgt_lang))

    def translate_batch(
        self,
        texts: Sequence[str],
        src_lang: str,
        tgt_lang: str,
        *,
        batch_size: int = 16,  # kept for API compatibility; decoder-only => serial
        max_length: int = MAX_NEW_TOKENS,
    ) -> list[str]:
        return [self.translate(t, src_lang, tgt_lang) for t in texts]

    def translate_file(
        self,
        src_path: str | Path,
        tgt_path: str | Path,
        src_lang: str,
        tgt_lang: str,
        *,
        max_lines: int | None = None,
        batch_size: int = 16,
    ) -> list[str]:
        src_path = Path(src_path)
        tgt_path = Path(tgt_path)
        with open(src_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if max_lines is not None:
            lines = lines[:max_lines]
        hypotheses = self.translate_batch(lines, src_lang, tgt_lang)
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tgt_path, "w", encoding="utf-8") as f:
            for h in hypotheses:
                f.write(h + "\n")
        return hypotheses

    def translate_pivot(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        pivot_lang: str = "en",
    ) -> str:
        intermediate = self.translate(text, src_lang, pivot_lang)
        return self.translate(intermediate, pivot_lang, tgt_lang)

    @staticmethod
    def preprocess_chinese(text: str) -> str:
        """Collapse whitespace between CJK chars (kept from the NLLB module)."""
        text = re.sub(
            r"(?<=[\u4e00-\u9fff\u3400-\u4dbf])\s+(?=[\u4e00-\u9fff\u3400-\u4dbf])",
            "",
            text,
        )
        return text.strip()
