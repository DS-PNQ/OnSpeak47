# Large-corpus VI↔ZH BLEU evaluation
#
# Uses the 1000-pair test corpus from ``test vi-zh/`` to score Hy-MT quality
# on the VI↔CN pair — the single biggest technical risk flagged in
# pipeline_overview.md (Risk #1).

import math
import json
from pathlib import Path

import pytest
import sacrebleu

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Maximum number of test pairs to evaluate (controls runtime).
MAX_PAIRS = 200


class TestViZhCorpus:
    """BLEU evaluation on the large VI↔ZH test corpus.

    Evaluates both directions:
      - vi → zh_hans  (Vietnamese to Simplified Chinese)
      - zh_hans → vi  (Simplified Chinese to Vietnamese)

    Compares direct translation quality against the pivot (vi→en→zh) path
    to quantify the Risk #1 gap.
    """

    # ------------------------------------------------------------------
    # Direct translation
    # ------------------------------------------------------------------

    def test_vi_to_zh_direct(self, translator, vizh_test_data, results_recorder):
        """Evaluate vi → zh_hans BLEU on the large test corpus."""
        pairs = vizh_test_data[:MAX_PAIRS]
        sources = [p["vi"] for p in pairs]
        references = [p["zh_hans"] for p in pairs]

        hypotheses = translator.translate_batch(sources, "vi", "zh_hans", batch_size=8)

        assert len(hypotheses) == len(references)
        assert all(isinstance(h, str) and len(h) > 0 for h in hypotheses)

        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
        assert math.isfinite(bleu.score)

        results_recorder("vi->zh_hans (corpus)", len(pairs), bleu.score, "zh")

        # Dump detailed results
        self._dump_results("vi_to_zh_corpus", pairs, "vi", "zh_hans", hypotheses)

        print(f"\n  vi→zh BLEU (n={len(pairs)}): {bleu.score:.2f}")

    def test_zh_to_vi_direct(self, translator, vizh_test_data, results_recorder):
        """Evaluate zh_hans → vi BLEU on the large test corpus."""
        pairs = vizh_test_data[:MAX_PAIRS]
        sources = [p["zh_hans"] for p in pairs]
        references = [p["vi"] for p in pairs]

        hypotheses = translator.translate_batch(sources, "zh_hans", "vi", batch_size=8)

        assert len(hypotheses) == len(references)
        assert all(isinstance(h, str) and len(h) > 0 for h in hypotheses)

        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a")
        assert math.isfinite(bleu.score)

        results_recorder("zh_hans->vi (corpus)", len(pairs), bleu.score, "13a")

        self._dump_results("zh_to_vi_corpus", pairs, "zh_hans", "vi", hypotheses)

        print(f"\n  zh→vi BLEU (n={len(pairs)}): {bleu.score:.2f}")

    # ------------------------------------------------------------------
    # Pivot translation (vi → en → zh) for Risk #1 comparison
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_vi_to_zh_pivot(self, translator, vizh_test_data, results_recorder):
        """Evaluate vi → en → zh_hans pivot path.

        This is ~2× slower (two translation hops) but may yield higher
        quality for the VN↔CN pair where direct weights are weaker.
        """
        pairs = vizh_test_data[:50]  # Fewer pairs since it's 2× cost
        references = [p["zh_hans"] for p in pairs]

        hypotheses = [
            translator.translate_pivot(p["vi"], "vi", "zh_hans", pivot_lang="en")
            for p in pairs
        ]

        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
        assert math.isfinite(bleu.score)

        results_recorder("vi->en->zh_hans (pivot)", len(pairs), bleu.score, "zh")

        print(f"\n  vi→en→zh PIVOT BLEU (n={len(pairs)}): {bleu.score:.2f}")

    # ------------------------------------------------------------------
    # Chinese pre-processing impact
    # ------------------------------------------------------------------

    def test_zh_preprocessing_impact(self, translator, vizh_test_data):
        """Test whether Chinese pre-processing changes output quality.

        Compares BLEU with and without ``preprocess_chinese()`` on zh inputs.
        """
        from backend.translation_hymt import HyMTTranslator

        pairs = vizh_test_data[:50]
        references = [p["vi"] for p in pairs]

        # Without preprocessing
        raw_sources = [p["zh_hans"] for p in pairs]
        hyp_raw = translator.translate_batch(raw_sources, "zh_hans", "vi", batch_size=8)
        bleu_raw = sacrebleu.corpus_bleu(hyp_raw, [references], tokenize="13a")

        # With preprocessing
        processed_sources = [HyMTTranslator.preprocess_chinese(s) for s in raw_sources]
        hyp_processed = translator.translate_batch(processed_sources, "zh_hans", "vi", batch_size=8)
        bleu_processed = sacrebleu.corpus_bleu(hyp_processed, [references], tokenize="13a")

        print(f"\n  zh→vi BLEU (raw):        {bleu_raw.score:.2f}")
        print(f"  zh→vi BLEU (preprocessed): {bleu_processed.score:.2f}")
        print(f"  Delta: {bleu_processed.score - bleu_raw.score:+.2f}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dump_results(
        name: str,
        pairs: list[dict],
        src_key: str,
        tgt_key: str,
        hypotheses: list[str],
    ):
        OUTPUT_DIR.mkdir(exist_ok=True)
        records = [
            {
                "src": p[src_key],
                "hypothesis": h,
                "reference": p[tgt_key],
            }
            for p, h in zip(pairs, hypotheses)
        ]
        path = OUTPUT_DIR / f"translations_{name}.json"
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
