# OmniVoice — Generate the PyTorch reference for the ONNX parity gate
# (tests_local/test_06_onnx_parity.py).
#
# Runs the DESKTOP PyTorch backends (backend/) with GREEDY decoding that
# mirrors the on-device scheme (KV-cached greedy in ASRModule /
# TranslationModule), so test_06 measures ONNX-vs-PyTorch divergence instead
# of greedy-vs-beam differences. Also synthesizes the Whisper ASR fixtures
# (MMS-TTS wavs) when possible.
#
# Usage (desktop python with torch/transformers installed):
#   python tests_local/gen_parity_reference.py
#
# Output:
#   tests_local/output/parity_reference.json
#   tests_local/data/audio_samples/parity_{en,vi}.wav (+ .txt references)
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("parity_ref")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "tests_local/data"
SAMPLES = DATA / "audio_samples"
OUT = ROOT / "tests_local/output/parity_reference.json"

# Same directions as tests_local/test_02_translation.py.
DIRECTIONS = [
    ("vi", "en"), ("en", "vi"),
    ("vi", "zh_hans"), ("zh_hans", "vi"),
    ("vi", "zh_hant"), ("zh_hant", "vi"),
]

# Short sentences for the ASR fixtures (kept short on purpose — MMS-TTS
# synthesis and Whisper transcription both degrade on long utterances).
ASR_SAMPLES = {"en": "I would like to renew my citizen ID card.",
               "vi": "Tôi muốn gia hạn căn cước công dân."}


def sacrebleu_score(hyps: list[str], refs: list[str], tgt: str) -> float:
    import sacrebleu
    tok = "zh" if tgt.startswith("zh") else "13a"
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok).score


def gen_hymt(sentences: list[dict]) -> dict:
    from backend.translation_hymt import HyMTTranslator

    t = HyMTTranslator()
    bleus: dict[str, float] = {}
    hyp_store: dict[str, list[str]] = {}
    for src, tgt in DIRECTIONS:
        texts = [s[src] for s in sentences]
        refs = [s[tgt] for s in sentences]
        hyps = t.translate_batch(texts, src, tgt)
        bleus[f"{src}->{tgt}"] = sacrebleu_score(hyps, refs, tgt)
        hyp_store[f"{src}->{tgt}"] = hyps
        log.info(f"Hy-MT {src}->{tgt}: BLEU={bleus[f'{src}->{tgt}']:.2f}")
    return {"bleu": bleus, "hypotheses": hyp_store}


def gen_asr() -> dict:
    """Synthesize ASR fixtures with MMS-TTS and transcribe with PyTorch Whisper."""
    from backend.asr_whisper import WhisperASR
    from backend.tts_mms import MMSTTS

    tts = MMSTTS()
    asr = WhisperASR()
    wers: dict[str, float] = {}
    transcripts: dict[str, str] = {}
    SAMPLES.mkdir(parents=True, exist_ok=True)
    import jiwer
    for lang, text in ASR_SAMPLES.items():
        wav = SAMPLES / f"parity_{lang}.wav"
        if not wav.exists():
            tts.synthesize(text, lang, wav, sample_rate=16_000)
            log.info(f"Synthesized fixture {wav.name}")
        (wav.with_suffix(".txt")).write_text(text, encoding="utf-8")
        result = asr.transcribe(str(wav), language=lang)
        transcripts[lang] = result.text
        wers[lang] = float(jiwer.wer(text, result.text))
        log.info(f"Whisper {lang}: WER={wers[lang]:.3f} transcript={result.text!r}")
    return {"wer": wers, "transcripts": transcripts}


def main() -> None:
    sentences = json.loads((DATA / "parallel_sentences.json").read_text(encoding="utf-8"))
    log.info(f"Loaded {len(sentences)} parallel sentences")

    reference: dict = {"meta": {"date": str(date.today()), "decode": "greedy"}}

    log.info("Generating Hy-MT reference (greedy)…")
    reference["hymt"] = gen_hymt(sentences)

    try:
        log.info("Generating Whisper ASR reference + fixtures…")
        reference["asr"] = gen_asr()
    except Exception as e:  # ASR fixtures are optional for the gate
        log.warning(f"ASR reference skipped ({type(e).__name__}: {e})")
        reference["asr"] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reference, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"Reference written -> {OUT}")


if __name__ == "__main__":
    main()
