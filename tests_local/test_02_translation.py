# Quality tests for Hy-MT1.5 (crucial for VN<->CN weights)

import math

import pytest
import sacrebleu

# (direction_name, src_key, tgt_key, requires_hant)
DIRECTIONS = [
    ("vi->en", "vi", "en", False),
    ("en->vi", "en", "vi", False),
    ("vi->zh_hans", "vi", "zh_hans", False),
    ("zh_hans->vi", "zh_hans", "vi", False),
    ("vi->zh_hant", "vi", "zh_hant", True),
    ("zh_hant->vi", "zh_hant", "vi", True),
]


@pytest.mark.parametrize("direction,src_key,tgt_key,requires_hant", DIRECTIONS, ids=[d[0] for d in DIRECTIONS])
def test_translation_quality(
    translator, parallel_data, results_recorder, translations_dumper, direction, src_key, tgt_key, requires_hant
):
    if requires_hant and not all("zh_hant" in row for row in parallel_data):
        pytest.skip("no zh_hant data available in parallel_sentences.json")

    hypotheses = [translator.translate(row[src_key], src_key, tgt_key) for row in parallel_data]
    references = [row[tgt_key] for row in parallel_data]

    assert len(hypotheses) == len(references)
    assert all(isinstance(h, str) and h.strip() for h in hypotheses)

    translations_dumper(direction, parallel_data, src_key, tgt_key, hypotheses)

    # BLEU tokenizes only the hypothesis/reference text, i.e. the target language.
    tokenize = "zh" if tgt_key.startswith("zh") else "13a"
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)

    assert math.isfinite(bleu.score)
    assert bleu.score >= 0

    results_recorder(direction, len(parallel_data), bleu.score, tokenize)
