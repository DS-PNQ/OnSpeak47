# OmniVoice — Offline ONNX Runtime graph pre-optimization (Phase 2.1).
#
# Runs ORT's full graph-optimization pass OFFLINE and installs the result
# next to the original asset as <name>.opt.onnx, moving the original to
# onnx_models/preopt_backup/ (the .opt file REPLACES the original in the APK
# — shipping both would double the package size).
#
# The app loads *.opt.onnx with OptLevel.NO_OPT, which:
#   - removes the multi-second on-device optimization pass at createSession
#   - flattens the peak-RAM spike during session init (the "bad allocation"
#     class reproduced on low-RAM targets)
#
# IMPORTANT: run with the SAME onnxruntime version the app ships (currently
# 1.27.0, in lockstep with the sherpa-onnx AAR) — optimized graphs are not
# portable across ORT versions:
#   <venv>/Scripts/python optimize/07_preoptimize.py
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

EXPECTED_ORT = "1.22.0"

ASSETS = Path(__file__).resolve().parent.parent / "android/app/src/main/assets"
BACKUP = Path(__file__).resolve().parent.parent / "onnx_models/preopt_backup"

MODELS = [
    # Zipformer streaming models (sherpa-onnx) are already inference-ready;
    # offline pre-opt is NOT applied to them — pass explicit --models if a
    # future asset needs it.
]


def preoptimize(name: str, check_only: bool) -> None:
    import onnxruntime as ort

    src = ASSETS / name
    opt_name = name.replace(".onnx", ".opt.onnx")
    opt_path = ASSETS / opt_name

    if not src.exists():
        if opt_path.exists():
            log.info(f"[Skip] {name} not present (already pre-optimized: {opt_name})")
            return
        log.error(f"[Error] {src} not found")
        return

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(opt_path)
    log.info(f"Optimizing {name} ({src.stat().st_size/1e6:.1f} MB) -> {opt_name}")
    ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])

    # The optimized file must load cleanly by itself at NO_OPT (exactly how
    # the app will consume it).
    verify = ort.SessionOptions()
    verify.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(opt_path), verify, providers=["CPUExecutionProvider"])
    log.info(f"  verified: {len(sess.get_inputs())} inputs, "
             f"{len(sess.get_outputs())} outputs, "
             f"{opt_path.stat().st_size/1e6:.1f} MB")

    if check_only:
        opt_path.unlink()
        return

    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(BACKUP / name))
    log.info(f"  original moved to {BACKUP / name}")


def main() -> None:
    import onnxruntime as ort

    if ort.__version__ != EXPECTED_ORT:
        log.warning(f"onnxruntime {ort.__version__} != app version {EXPECTED_ORT} — "
                    "optimized graphs may not load on device. Aborting; rerun in a "
                    "venv pinned to the app version.")
        sys.exit(2)

    parser = argparse.ArgumentParser(description="Offline ORT graph pre-optimization")
    parser.add_argument("--check-only", action="store_true",
                        help="Produce and verify the optimized file but keep the original")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    for name in args.models:
        preoptimize(name, args.check_only)


if __name__ == "__main__":
    main()
