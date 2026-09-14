# Optional: AIMET/generic quantization logic
#
# Two paths:
#   1. Qualcomm AIMET (for Snapdragon NPU — primary path per pipeline_overview.md)
#   2. Generic PyTorch INT8 (for non-Qualcomm ARM chips — fallback path)
#
# Skip on the first attempt if time-constrained; revisit if profiling numbers look bad.

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ======================================================================
# Path 1: Generic ONNX Runtime INT8 quantization
# ======================================================================

def quantize_onnx_int8(onnx_path: Path, output_path: Path):
    """Apply static INT8 quantization to an ONNX model.

    Uses ONNX Runtime's built-in quantization toolkit.

    Parameters
    ----------
    onnx_path : Path
        Input .onnx file.
    output_path : Path
        Output quantized .onnx file.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        log.error("Install onnxruntime: pip install onnxruntime")
        return

    log.info(f"Quantizing {onnx_path.name} → INT8 (dynamic)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    original_mb = onnx_path.stat().st_size / 1e6
    quantized_mb = output_path.stat().st_size / 1e6
    reduction = (1 - quantized_mb / original_mb) * 100

    log.info(f"  {onnx_path.name}: {original_mb:.1f} MB → {quantized_mb:.1f} MB "
             f"({reduction:.1f}% reduction)")

    return {
        "model": onnx_path.name,
        "original_mb": round(original_mb, 1),
        "quantized_mb": round(quantized_mb, 1),
        "reduction_pct": round(reduction, 1),
    }


def quantize_all_onnx(model_dir: Path, output_dir: Path):
    """Quantize all .onnx files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for onnx_file in sorted(model_dir.rglob("*.onnx")):
        relative = onnx_file.relative_to(model_dir)
        out_path = output_dir / relative.with_suffix(".quant.onnx")
        result = quantize_onnx_int8(onnx_file, out_path)
        if result:
            results.append(result)

    if results:
        _print_summary(results)

    return results


# ======================================================================
# Path 2: Qualcomm AIMET quantization (for Snapdragon NPU)
# ======================================================================

def quantize_aimet(model_path: Path, output_path: Path, calibration_data=None):
    """Apply AIMET quantization for Qualcomm Snapdragon NPU.

    AIMET (AI Model Efficiency Toolkit) provides quantization-aware training
    and post-training quantization optimized for Qualcomm hardware.

    NOTE: AIMET requires a Linux environment and specific dependencies.
    This function provides the framework — actual AIMET installation is
    documented at https://quic.github.io/aimet-pages/

    Parameters
    ----------
    model_path : Path
        PyTorch model checkpoint or ONNX file.
    output_path : Path
        Output path for the AIMET-quantized model.
    calibration_data
        Representative data for calibration (e.g., sample sentences).
    """
    log.info("AIMET quantization path — Qualcomm Snapdragon NPU optimized")
    log.info("=" * 60)

    try:
        # AIMET is only available on Linux with specific CUDA/PyTorch versions
        import aimet_torch
        from aimet_torch.quantsim import QuantizationSimModel
        from aimet_common.defs import QuantScheme

        log.info("AIMET is available — proceeding with quantization")

        # This is the AIMET quantization workflow:
        # 1. Load the PyTorch model
        # 2. Create a QuantizationSimModel
        # 3. Run calibration with representative data
        # 4. Export the quantized model

        # [Implementation depends on actual model architecture]
        log.warning("AIMET quantization requires model-specific calibration setup.")
        log.warning("See AIMET documentation for model-specific guides.")

    except ImportError:
        log.warning("AIMET not installed — this is expected on Windows.")
        log.warning("AIMET requires Linux. Use generic INT8 quantization instead,")
        log.warning("or set up a Linux environment for AIMET.")
        log.info("")
        log.info("Alternative: Use generic ONNX Runtime INT8 quantization:")
        log.info("  python 03_quantize_aimet.py --method onnx_int8")


# ======================================================================
# Helpers
# ======================================================================

def _print_summary(results: list[dict]):
    """Print a summary table of quantization results."""
    log.info("")
    log.info("Quantization Summary:")
    log.info(f"  {'Model':<40} {'Original':>10} {'Quantized':>10} {'Reduction':>10}")
    log.info("  " + "-" * 72)
    for r in results:
        log.info(f"  {r['model']:<40} {r['original_mb']:>8.1f} MB {r['quantized_mb']:>8.1f} MB "
                 f"{r['reduction_pct']:>8.1f}%")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Quantize pipeline models")
    parser.add_argument(
        "--method",
        choices=["onnx_int8", "aimet"],
        default="onnx_int8",
        help="Quantization method",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "onnx_models",
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "onnx_models_quantized",
    )
    args = parser.parse_args()

    if args.method == "onnx_int8":
        quantize_all_onnx(args.model_dir, args.output_dir)
    elif args.method == "aimet":
        quantize_aimet(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
