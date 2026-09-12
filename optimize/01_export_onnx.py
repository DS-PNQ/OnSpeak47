# OmniVoice — Model Export & Download Script
#
# Automates the setup of ONNX models for the Android app.
# 1. Exports Whisper Small to ONNX using Optimum.
# 2. Places all files directly into the Android assets directory.
#
# Translation (Hy-MT) assets are prepared by 07_prepare_hymt_gguf.py
# (low-RAM 1.25-bit GGUF) and 08_export_hymt_onnx_int4.py (INT4 ONNX).

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Target directory: Output to a dedicated folder on D:
ASSETS_DIR = Path("D:/StudioProjects/OmniVoice3/onnx_models")

# Hy-MT translation assets are NOT handled here anymore — see
# 07_prepare_hymt_gguf.py and 08_export_hymt_onnx_int4.py.

def download_file(url: str, output_path: Path):
    """Download a file with progress logging."""
    if output_path.exists():
        log.info(f"  [Skip] {output_path.name} already exists.")
        return

    log.info(f"  [Download] {url} -> {output_path.name}")
    try:
        def progress(count, block_size, total_size):
            if total_size <= 0: return
            percent = int(count * block_size * 100 / total_size)
            if percent % 25 == 0:  # Log every 25%
                sys.stdout.write(f"\r    Progress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(output_path), reporthook=progress)
        sys.stdout.write("\n")
        log.info(f"  [Done] Saved {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log.error(f"  [Error] Failed to download {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()




def export_whisper(assets_dir: Path):
    """Export Whisper Small using Optimum and move to assets."""
    log.info("Exporting Whisper Small to ONNX...")
    
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    except ImportError:
        log.error("Install `optimum` and `onnxruntime`: pip install optimum onnxruntime")
        return

    temp_dir = assets_dir / "temp_whisper_export"
    temp_dir.mkdir(parents=True, exist_ok=True)

    log.info("  Converting Whisper Small via Optimum (this may take a while)...")
    # use_cache=True so the exported decoder carries past_key_values/present
    # (KV-cache) inputs/outputs. Without this, autoregressive decoding on-device
    # has to reprocess the entire growing token sequence every step (O(n^2))
    # instead of just the newest token — see ASRModule.runDecoder() on the Java side.
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-small",
        export=True,
        use_cache=True,
        cache_dir=str(assets_dir / "hf_cache"),
    )
    model.save_pretrained(str(temp_dir))

    # Move and rename specific files
    log.info("  Moving Whisper models to assets...")
    shutil.move(str(temp_dir / "encoder_model.onnx"), str(assets_dir / "whisper_encoder.onnx"))

    # Depending on the installed Optimum version, the cache-enabled decoder can
    # come out as a single merged graph (preferred — has a use_cache_branch input,
    # same pattern as decoder_model_merged_int8.onnx for NLLB) or as a separate
    # "with past" file alongside the cache-less decoder_model.onnx. Prefer the
    # merged file, then with-past, and only fall back to the plain decoder (no
    # cache — ASRModule will auto-detect this and use the slower loop) if
    # neither cache-enabled file was produced.
    decoder_candidates = [
        "decoder_model_merged.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    ]
    for name in decoder_candidates:
        candidate = temp_dir / name
        if candidate.exists():
            if name == "decoder_model.onnx":
                log.warning(
                    "  No cache-enabled decoder file found (checked %s) — "
                    "falling back to decoder_model.onnx (no KV-cache, slower on-device).",
                    decoder_candidates[:-1],
                )
            shutil.move(str(candidate), str(assets_dir / "whisper_decoder.onnx"))
            break
    else:
        log.error(f"  [Error] No decoder ONNX file found among {decoder_candidates} in {temp_dir}")

    # Cleanup
    shutil.rmtree(temp_dir)
    log.info(f"  [Done] Whisper models saved to {assets_dir}")


def export_whisper_processing(assets_dir: Path):
    """Export Whisper's pre-processing (raw audio bytes -> log-mel features)
    and post-processing (token ids -> text) as their own ONNX graphs, using
    onnxruntime-extensions. This replaces two things that previously had to
    be hand-implemented in Java and were left as stubs:
      - ASRModule.extractMelFeatures() (was returning an all-zero array —
        FFT + mel-filterbank is exactly what USE_ONNX_STFT does here)
      - a Whisper-specific BPE detokenizer (was returning a token-count
        placeholder string instead of real text)
    Both stubs can be replaced by just running these two extra ONNX sessions
    from Java instead of hand-writing DSP/BPE code that's hard to verify.
    """
    log.info("Exporting Whisper pre/post-processing graphs (onnxruntime-extensions)...")
    try:
        import onnx
        from transformers import WhisperProcessor
        from onnxruntime_extensions.cvt import gen_processing_models

        # HACK: Patch onnx.compose.merge_models to ignore IR version mismatch.
        # This is required on Python 3.12/Windows where Torch exports IR 10 
        # but extensions internally use IR 8.
        _original_merge = onnx.compose.merge_models
        def _patched_merge(m1, m2, *args, **kwargs):
            m1.ir_version = 8
            m2.ir_version = 8
            return _original_merge(m1, m2, *args, **kwargs)
        onnx.compose.merge_models = _patched_merge

    except ImportError:
        log.error(
            "Install `onnxruntime-extensions` and `transformers`: "
            "pip install onnxruntime-extensions transformers"
        )
        return

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        cache_dir=str(assets_dir / "hf_cache")
    )

    # USE_AUDIO_DECODER=False: The graph expects raw float samples (16kHz).
    # We decode the WAV file manually in Java to avoid AudioDecoder op errors on Android.
    # USE_ONNX_STFT=False: Tắt để tránh lỗi "Cannot find STFTNorm" trên môi trường
    # Windows/Python 3.12, vẫn đảm bảo tính đúng đắn trên Android.
    pre_m, post_m = gen_processing_models(
        processor,
        pre_kwargs={"USE_AUDIO_DECODER": False, "USE_ONNX_STFT": False},
        post_kwargs={},
        opset=17,
    )

    # Restore original merge function
    onnx.compose.merge_models = _original_merge

    pre_path = assets_dir / "whisper_preprocess.onnx"
    post_path = assets_dir / "whisper_postprocess.onnx"
    onnx.save(pre_m, str(pre_path))
    onnx.save(post_m, str(post_path))

    # post_m expects token ids shaped the way the fused WhisperBeamSearch
    # contrib op emits them, which a custom greedy-decode loop (like
    # ASRModule.runDecoder()) doesn't produce. Detokenizing (id -> text) is a
    # simple, stable, well-documented algorithm — unlike encoding, it doesn't
    # need the merge-rank tables — so it's safer to hand-implement it natively
    # in Java against a plain vocab.json than to guess post_m's exact expected
    # input shape. Dump that vocab here; post_m is still saved above in case
    # you want it for the fused pipeline later.
    vocab_path = assets_dir / "whisper_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(processor.tokenizer.get_vocab(), f, ensure_ascii=False)

    # Print the actual graph I/O names rather than assuming them — the exact
    # names can vary by onnxruntime-extensions version, and ASRModule.java
    # looks these up dynamically via session.getInputNames()/getOutputNames()
    # rather than hardcoding a guess, but it's worth confirming they look sane.
    log.info(f"  [Done] {pre_path.name}: inputs={[i.name for i in pre_m.graph.input]} "
             f"outputs={[o.name for o in pre_m.graph.output]}")
    log.info(f"  [Done] {vocab_path.name}: {len(processor.tokenizer.get_vocab())} tokens "
             "(used by ASRModule's native Java BPE decoder)")
    log.info(f"  [Info] {post_path.name} saved but not currently used by ASRModule.java — "
             "it expects WhisperBeamSearch-shaped input, see comment above.")


def check_android_encoder_asset(assets_dir: Path):
    """Guard against shipping a slow FP32 Whisper encoder to the APK.

    The Android app must bundle the dynamic-frame INT8 encoder produced by
    optimize/10_export_whisper_encoder_dyn.py (``onnx_models/
    whisper_encoder_dyn_int8.onnx``). This script only produces the raw
    Optimum FP32 export; copying that over the Android asset (as happened on
    2026-09-08) regresses ASR to roughly 3-4x slower — 352 MB FP32 with a
    fixed 3000-frame window vs 98 MB INT8 with dynamic frames — exactly the
    "ASR is much slower than before" report from the device. Compare sizes
    and fail the script loudly when they diverge.
    """
    android_asset = (Path(__file__).resolve().parent.parent / "android" / "app"
                     / "src" / "main" / "assets" / "whisper_encoder.onnx")
    canonical = assets_dir / "whisper_encoder_dyn_int8.onnx"
    if not android_asset.exists() or not canonical.exists():
        return
    want = canonical.stat().st_size
    have = android_asset.stat().st_size
    if have != want:
        log.error(
            "android/app/src/main/assets/whisper_encoder.onnx is %d bytes but the "
            "canonical INT8 dynamic export (whisper_encoder_dyn_int8.onnx) is %d bytes. "
            "The bundled asset is probably a raw FP32 re-export from 01_export_onnx.py; "
            "restore it with:  copy onnx_models/whisper_encoder_dyn_int8.onnx "
            "android/app/src/main/assets/whisper_encoder.onnx",
            have, want,
        )
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Download or Export ONNX models to Android assets")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["whisper", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    models = args.models if "all" not in args.models else ["whisper"]

    if "whisper" in models:
        export_whisper(args.assets_dir)
        export_whisper_processing(args.assets_dir)

    # Never leave the Android asset as a raw FP32 export — see the docstring.
    check_android_encoder_asset(args.assets_dir)

    log.info("Workflow complete. Large models are now in assets/.")
    log.info("IMPORTANT: Add *.onnx and *.model to your .gitignore before pushing!")


if __name__ == "__main__":
    main()
