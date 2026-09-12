# Strings models together with NO language branching
#
# Every input goes through the same three stages regardless of direction:
#   [audio] → Whisper (ASR) → NLLB (Translation) → MMS-TTS (Synthesis) → [audio]

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.asr_whisper import WhisperASR, ASRResult
from backend.translation_hymt import HyMTTranslator
from backend.tts_mms import MMSTTS


@dataclass
class PipelineResult:
    """Immutable result of a full pipeline pass."""
    transcript: str
    src_language: str
    translation: str
    tgt_language: str
    audio_path: str | None
    timings: dict = field(default_factory=dict)


class OmniVoicePipeline:
    """Three-stage on-device speech translation pipeline.

    Architecture is intentionally modular — each stage has clear input/output
    contracts so a future wearable/lanyard device can swap in a different
    hardware backend without redesigning the pipeline.

    The pipeline applies **no** language-routing logic: every input goes
    through ASR → Translation → TTS regardless of direction.
    """

    def __init__(
        self,
        asr: WhisperASR | None = None,
        translator: HyMTTranslator | None = None,
        tts: MMSTTS | None = None,
    ):
        self.asr = asr or WhisperASR()
        self.translator = translator or HyMTTranslator()
        self.tts = tts or MMSTTS()

    # ------------------------------------------------------------------
    # Full pipeline: audio → audio
    # ------------------------------------------------------------------

    def process(
        self,
        audio_path: str,
        src_lang: str,
        tgt_lang: str,
        *,
        output_dir: str | Path = "output",
    ) -> PipelineResult:
        """Run the complete ASR → Translation → TTS pipeline.

        Parameters
        ----------
        audio_path : str
            Input audio file (any format librosa supports).
        src_lang : str
            Source language code (``"vi"``, ``"en"``, ``"zh"``).
        tgt_lang : str
            Target language code.
        output_dir : str | Path
            Directory for the synthesized output WAV.

        Returns
        -------
        PipelineResult
        """
        timings: dict = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: ASR ---
        t0 = time.perf_counter()
        asr_result: ASRResult = self.asr.transcribe(audio_path, language=src_lang)
        timings["asr_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- Stage 2: Translation ---
        t0 = time.perf_counter()
        translated = self.translator.translate(
            asr_result.text, src_lang, tgt_lang
        )
        timings["translation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- Stage 3: TTS ---
        t0 = time.perf_counter()
        out_wav = output_dir / f"translated_{src_lang}_to_{tgt_lang}.wav"
        self.tts.synthesize(translated, tgt_lang, out_wav)
        timings["tts_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        timings["total_ms"] = round(
            timings["asr_ms"] + timings["translation_ms"] + timings["tts_ms"], 1
        )

        return PipelineResult(
            transcript=asr_result.text,
            src_language=asr_result.language,
            translation=translated,
            tgt_language=tgt_lang,
            audio_path=str(out_wav.resolve()),
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Text-only pipeline (no ASR / no TTS) for quick evaluation
    # ------------------------------------------------------------------

    def translate_text(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> PipelineResult:
        """Translation-only shortcut — skips ASR and TTS.

        Useful for batch quality evaluation where audio is not involved.
        """
        t0 = time.perf_counter()
        translated = self.translator.translate(text, src_lang, tgt_lang)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        return PipelineResult(
            transcript=text,
            src_language=src_lang,
            translation=translated,
            tgt_language=tgt_lang,
            audio_path=None,
            timings={"translation_ms": elapsed, "total_ms": elapsed},
        )
