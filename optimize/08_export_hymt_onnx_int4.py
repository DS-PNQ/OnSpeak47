# OmniVoice — Hy-MT HIGH-END profile: INT4 ONNX for ONNX Runtime GenAI
#
# Builds tencent/HY-MT1.5-1.8B (arch HunYuanDenseV1ForCausalLM — "HunYuan
# Dense V1" is supported by the ORT GenAI model builder) into an INT4 ONNX
# model folder that ORT GenAI can run on CPU or on the QNN EP.
#
# pip install onnxruntime-genai torch transformers
# python optimize/08_export_hymt_onnx_int4.py --ep cpu     # CPU EP
# python optimize/08_export_hymt_onnx_int4.py --ep qnn     # Snapdragon QNN EP
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL = "tencent/HY-MT1.5-1.8B"

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT_DIR / "android" / "app" / "src" / "main" / "assets" / "hymt_int4_onnx"
DEFAULT_CACHE_DIR = ROOT_DIR / "onnx_models" / "hf_cache"
DEFAULT_BACKEND_COPY = ROOT_DIR / "onnx_models" / "hymt_int4_onnx"


def build(ep: str, out: Path, cache: Path, hf_token: str | None = None) -> bool:
    token_val = hf_token or os.environ.get("HF_TOKEN") or "false"
    cmd = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", BASE_MODEL,
        "-o", str(out),
        "-p", "int4",
        "-e", ep,
        "-c", str(cache),
        "--extra_options",
        f"hf_token={token_val}",
        "hf_remote=true",
    ]
    log.info("  Running: " + " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log.error("  ORT GenAI model builder failed — try the Olive fallback:")
        log.error(f"    olive auto-opt -m {BASE_MODEL} -o {out} -p int4 "
                  f"--device {'qnn' if ep == 'qnn' else 'cpu'} --use_ort_genai")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Export Hy-MT1.5 INT4 ONNX (ORT GenAI)")
    ap.add_argument("--ep", choices=["cpu", "qnn"], default="cpu",
                    help="execution provider baked into the model (QNN or CPU)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--backend-copy", type=Path, default=DEFAULT_BACKEND_COPY,
                    help="also copy the folder for the Python high_end profile")
    ap.add_argument("--hf-token", type=str, default=None,
                    help="Hugging Face authentication token (optional)")
    args = ap.parse_args()

    if not build(args.ep, args.out, args.cache, args.hf_token):
        sys.exit(1)

    if args.backend_copy and args.backend_copy != args.out:
        import shutil
        if args.backend_copy.exists():
            shutil.rmtree(args.backend_copy)
        shutil.copytree(args.out, args.backend_copy)
        log.info(f"  [Done] copied to {args.backend_copy} for the Python backend")

    log.info("  [Done] INT4 ONNX model folder ready (genai_config.json + weights).")
    log.info("  The Android app picks this up automatically as the HIGH_END profile;")
    log.info("  devices with <= 6 GB RAM use the 1.25-bit GGUF instead (see 07_).")


if __name__ == "__main__":
    main()
