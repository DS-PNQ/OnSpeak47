# End-to-end pipeline tests (streaming-first).
#
# Verifies OmniVoicePipeline on streaming FINAL transcripts:
# Translation → TTS. ASR accuracy itself is covered by
# tests_local/test_07_streaming_asr.py (no models/network needed).

from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class TestPipeline:
    """Streaming-final → Translation → TTS integration tests."""

    def test_text_only_pipeline_vi_en(self, pipeline):
        """Text-only shortcut: vi → en translation."""
        result = pipeline.translate_text(
            "Tôi muốn gia hạn căn cước công dân.", "vi", "en"
        )
        assert result.transcript == "Tôi muốn gia hạn căn cước công dân."
        assert result.src_language == "vi"
        assert result.tgt_language == "en"
        assert len(result.translation) > 0
        assert result.timings["total_ms"] > 0
        print(f"\n  vi→en: {result.translation}")

    def test_text_only_pipeline_vi_zh(self, pipeline):
        """Text-only shortcut: vi → zh translation."""
        result = pipeline.translate_text(
            "Vui lòng mang theo bản gốc và bản sao của giấy khai sinh.",
            "vi",
            "zh_hans",
        )
        assert result.tgt_language == "zh_hans"
        assert len(result.translation) > 0
        print(f"\n  vi→zh: {result.translation}")

    def test_text_only_pipeline_zh_vi(self, pipeline):
        """Text-only shortcut: zh → vi translation."""
        result = pipeline.translate_text(
            "请携带出生证明原件和复印件。", "zh_hans", "vi"
        )
        assert result.tgt_language == "vi"
        assert len(result.translation) > 0
        print(f"\n  zh→vi: {result.translation}")

    def test_text_only_pipeline_en_vi(self, pipeline):
        """Text-only shortcut: en → vi translation."""
        result = pipeline.translate_text(
            "Please bring both the original and a copy of your birth certificate.",
            "en",
            "vi",
        )
        assert result.tgt_language == "vi"
        assert len(result.translation) > 0
        print(f"\n  en→vi: {result.translation}")

    def test_streaming_final_to_translation_tts(self, pipeline, tmp_path):
        """Streaming FINAL (router active_lang) → Translation → TTS.

        Simulates what StreamingOmniVoicePipeline does on endpoint: a
        committed Zipformer transcript with its detected language.
        """
        result = pipeline.process_final_transcript(
            "xin chào mọi người",
            src_lang="vi",
            tgt_lang="en",
            output_dir=str(tmp_path / "output"),
        )

        assert result.transcript == "xin chào mọi người"
        assert result.src_language == "vi"
        assert isinstance(result.translation, str) and len(result.translation) > 0
        assert result.audio_path is not None
        assert Path(result.audio_path).exists()
        assert result.timings["total_ms"] > 0

        print(f"\n  Timings: {result.timings}")

    def test_streaming_empty_final_skips_stages(self, pipeline, tmp_path):
        """Empty FINAL (silence endpoint) skips Translation and TTS."""
        result = pipeline.process_final_transcript(
            "   ", src_lang="vi", tgt_lang="en",
            output_dir=str(tmp_path / "output"),
        )
        assert result.transcript == ""
        assert result.translation == ""
        assert result.audio_path is None

    def test_pipeline_result_timings(self, pipeline):
        """Verify timings dict has expected keys."""
        result = pipeline.translate_text("Hello world.", "en", "vi")
        assert "translation_ms" in result.timings
        assert "total_ms" in result.timings
        assert result.timings["total_ms"] >= result.timings["translation_ms"]
