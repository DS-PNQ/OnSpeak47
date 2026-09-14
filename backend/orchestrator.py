# Streaming-first pipeline: Zipformer streaming ASR → HyMT → MMS-TTS.
#
# No Whisper path remains. Audio enters as 20 ms frames via
# StreamingPipeline (backend/streaming_asr/pipeline.py); the source language
# is the router's active language (auto VI/EN/ZH), never a manual hint.
# Every input goes through ASR → Translation → TTS regardless of direction.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from backend.streaming_asr.pipeline import StreamingPipeline
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
    """Streaming-transcript → Translation → TTS pipeline.

    Architecture is intentionally modular — each stage has clear input/output
    contracts so a future wearable/lanyard device can swap in a different
    hardware backend without redesigning the pipeline.

    The pipeline applies **no** manual language-routing logic: the source
    language always comes from the streaming router's active language.
    """

    def __init__(
        self,
        translator: HyMTTranslator | None = None,
        tts: MMSTTS | None = None,
        # Deprecated: accepted for backwards compat with old tests/callers,
        # ignored — ASR is always the streaming Zipformer stack now.
        asr=None,
        **kwargs,
    ):
        self.translator = translator or HyMTTranslator()
        self.tts = tts or MMSTTS()

    # ------------------------------------------------------------------
    # Streaming final transcript → translation → TTS (production path)
    # ------------------------------------------------------------------

    def process_final_transcript(
        self,
        transcript: str,
        src_lang: str,
        tgt_lang: str,
        *,
        output_dir: str | Path = "output",
    ) -> PipelineResult:
        """Translate a streaming FINAL transcript and synthesize output audio.

        Parameters
        ----------
        transcript : str
            Committed FINAL text from StreamingPipeline (already endpointed).
        src_lang : str
            Router active language (``"vi"``, ``"en"``, ``"zh"``).
        tgt_lang : str
            Target language code.
        output_dir : str | Path
            Directory for the synthesized output WAV.
        """
        timings: dict = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        transcript = (transcript or "").strip()
        if not transcript:
            return PipelineResult(
                transcript="",
                src_language=src_lang,
                translation="",
                tgt_language=tgt_lang,
                audio_path=None,
                timings={"total_ms": 0.0},
            )

        # --- Stage 1: Translation ---
        t0 = time.perf_counter()
        translated = self.translator.translate(transcript, src_lang, tgt_lang)
        timings["translation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- Stage 2: TTS ---
        t0 = time.perf_counter()
        out_wav = output_dir / f"translated_{src_lang}_to_{tgt_lang}.wav"
        self.tts.synthesize(translated, tgt_lang, out_wav)
        timings["tts_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        timings["total_ms"] = round(
            timings["translation_ms"] + timings["tts_ms"], 1
        )

        return PipelineResult(
            transcript=transcript,
            src_language=src_lang,
            translation=translated,
            tgt_language=tgt_lang,
            audio_path=str(out_wav.resolve()),
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Text-only pipeline (no ASR / no TTS input) for quick evaluation
    # ------------------------------------------------------------------

    def translate_text(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> PipelineResult:
        """Translation-only shortcut — skips ASR and TTS synthesis."""
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


class StreamingOmniVoicePipeline:
    """Live 20 ms frames → partials/finals → translate+TTS on endpoint.

    Thin glue over StreamingPipeline for local tests and backend parity with
    android/.../asr/StreamingPipeline.java. The caller feeds 20 ms frames;
    every FINAL event is translated + synthesized via OmniVoicePipeline.
    """

    def __init__(
        self,
        translator: HyMTTranslator | None = None,
        tts: MMSTTS | None = None,
        tgt_lang: str = "en",
        output_dir: str | Path = "output",
        on_partial: Callable[[str, str, str], None] | None = None,
        on_result: Callable[[PipelineResult], None] | None = None,
        **streaming_kwargs,
    ):
        self.pipe = OmniVoicePipeline(translator=translator, tts=tts)
        self.stream = StreamingPipeline(**streaming_kwargs)
        self.tgt_lang = tgt_lang
        self.output_dir = Path(output_dir)
        self.on_partial = on_partial
        self.on_result = on_result
        self.results: list[PipelineResult] = []
        self._seen_finals = 0

    @property
    def active_lang(self) -> str:
        return self.stream.active_lang

    @property
    def partials(self):
        return [e for e in self.stream.events if e[0] == "partial"]

    def on_frame(self, frame, now_ms: int | None = None) -> PipelineResult | None:
        """Feed one 20 ms frame; returns a PipelineResult when a FINAL fires."""
        self.stream.on_frame(frame, now_ms)

        if self.on_partial is not None:
            partials = [e for e in self.stream.events if e[0] == "partial"]
            if partials:
                _, committed, speculative, lang = partials[-1]
                self.on_partial(committed, speculative, lang)

        finals = [e for e in self.stream.events if e[0] == "final"]
        if len(finals) <= self._seen_finals:
            return None
        self._seen_finals = len(finals)
        text, lang = finals[-1][1], finals[-1][2]
        result = self.pipe.process_final_transcript(
            text, lang, self.tgt_lang, output_dir=self.output_dir
        )
        self.results.append(result)
        if self.on_result is not None:
            self.on_result(result)
        return result
