# Scripts to trigger compile/profile jobs on Qualcomm AI Hub (QAH)
#
# Streaming-first pipeline:
#   - BYOM (Bring Your Own Model) for Zipformer streaming ASR and MMS-TTS
#   - After profile job: ALWAYS check execution provider (NPU vs CPU)
#   - Never report latency as "NPU-accelerated" without confirming
#   - CPU INT8 is the baseline; QNN/NNAPI only after benchmark wins

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ======================================================================
# QAH API helpers
# ======================================================================

def check_qah_installed():
    """Check if qai_hub is installed and authenticated."""
    try:
        import qai_hub
        log.info(f"qai_hub version: {qai_hub.__version__}")
        return True
    except ImportError:
        log.error("Qualcomm AI Hub SDK not installed.")
        log.error("Install with: pip install qai-hub")
        log.error("Then authenticate: qai-hub configure --token <YOUR_TOKEN>")
        return False


# ======================================================================
# Submit compile job
# ======================================================================

def submit_compile_job(
    onnx_path: Path,
    device_name: str = "Samsung Galaxy S24 (Family)",
    model_name: str = "custom_model",
):
    """Submit a compile job to Qualcomm AI Hub.

    Parameters
    ----------
    onnx_path : Path
        Path to the .onnx model file.
    device_name : str
        Target Snapdragon device from QAH catalog.
    model_name : str
        Human-readable name for the job.
    """
    if not check_qah_installed():
        return None

    import qai_hub as hub

    log.info(f"Submitting compile job: {model_name}")
    log.info(f"  Model: {onnx_path}")
    log.info(f"  Target device: {device_name}")

    try:
        compile_job = hub.submit_compile_job(
            model=str(onnx_path),
            device=hub.Device(device_name),
            name=model_name,
        )
        log.info(f"  Job ID: {compile_job.job_id}")
        log.info(f"  Status URL: {compile_job.url}")
        return compile_job

    except Exception as e:
        log.error(f"  Compile job failed: {e}")
        return None


# ======================================================================
# Submit profile job
# ======================================================================

def submit_profile_job(
    compiled_model_or_path,
    device_name: str = "Samsung Galaxy S24 (Family)",
    model_name: str = "custom_model",
):
    """Submit a profile job to get real latency/memory numbers.

    CRITICAL (from pipeline_overview.md): After profiling, ALWAYS check
    the execution provider.  If it says CPU when NPU was expected, the
    model needs quantization and/or --onnx_execution_providers=qnn.
    """
    if not check_qah_installed():
        return None

    import qai_hub as hub

    log.info(f"Submitting profile job: {model_name}")
    log.info(f"  Target device: {device_name}")

    try:
        profile_job = hub.submit_profile_job(
            model=compiled_model_or_path,
            device=hub.Device(device_name),
        )
        log.info(f"  Job ID: {profile_job.job_id}")
        log.info(f"  Status URL: {profile_job.url}")
        return profile_job

    except Exception as e:
        log.error(f"  Profile job failed: {e}")
        return None


def check_execution_provider(profile_result) -> str:
    """Check whether a profile result ran on NPU or CPU.

    Returns 'NPU', 'CPU', or 'UNKNOWN'.

    WARNING from pipeline_overview.md:
      Compiling a model does not guarantee NPU execution. Some devices
      silently fall back to CPU unless QNN EP is explicitly enabled.
    """
    log.info("Checking execution provider...")

    try:
        details = profile_result.get_details()
        provider = details.get("execution_provider", "UNKNOWN")

        if "cpu" in provider.lower():
            log.warning("⚠️  Model is running on CPU, NOT NPU!")
            log.warning("  → The model likely needs quantization (INT8)")
            log.warning("  → Or add --onnx_execution_providers=qnn")
            log.warning("  → Do NOT report this latency as 'NPU-accelerated'")
        elif "qnn" in provider.lower() or "npu" in provider.lower():
            log.info("✓ Model confirmed running on NPU (QNN EP)")
        else:
            log.warning(f"  Unknown execution provider: {provider}")

        return provider

    except Exception as e:
        log.error(f"  Could not determine execution provider: {e}")
        return "UNKNOWN"


# ======================================================================
# Full pipeline submission
# ======================================================================

def submit_all_models(onnx_dir: Path, device_name: str):
    """Submit compile + profile jobs for all pipeline models.

    Expected structure in onnx_dir:
      - zipformer_vi/*.onnx (streaming Zipformer VI)
      - zipformer_en/*.onnx (streaming Zipformer EN)
      - zipformer_zh/*.onnx (streaming Zipformer ZH INT8)
      - mms_tts/*/*.onnx
    """
    if not check_qah_installed():
        return

    models_to_submit = [
        # (name, path pattern)
        ("Zipformer VI Encoder", "zipformer_vi/*encoder*.onnx"),
        ("Zipformer EN Encoder", "zipformer_en/*encoder*.onnx"),
        ("Zipformer ZH Encoder", "zipformer_zh/*encoder*.onnx"),
        ("MMS-TTS Vietnamese", "mms_tts/vie/*.onnx"),
        ("MMS-TTS English", "mms_tts/eng/*.onnx"),
    ]

    results = []
    for name, pattern in models_to_submit:
        matches = list(onnx_dir.glob(pattern))
        if not matches:
            log.warning(f"  No files matching {pattern} — skipping {name}")
            continue

        for onnx_path in matches:
            log.info(f"\n{'=' * 60}")
            log.info(f"Processing: {name} ({onnx_path.name})")
            log.info(f"{'=' * 60}")

            compile_job = submit_compile_job(onnx_path, device_name, name)
            if compile_job:
                results.append({
                    "model": name,
                    "file": str(onnx_path),
                    "compile_job_id": compile_job.job_id,
                    "compile_url": compile_job.url,
                })

    # Save results
    if results:
        results_path = onnx_dir / "qah_submission_results.json"
        results_path.write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        log.info(f"\nResults saved to {results_path}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Submit models to Qualcomm AI Hub for compile/profile"
    )
    parser.add_argument(
        "--action",
        choices=["compile", "profile", "all", "check"],
        default="check",
        help="Action to perform",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "onnx_models",
    )
    parser.add_argument(
        "--device",
        default="Samsung Galaxy S24 (Family)",
        help="Target Snapdragon device",
    )
    parser.add_argument("--model", type=Path, help="Specific ONNX model to submit")
    args = parser.parse_args()

    if args.action == "check":
        if check_qah_installed():
            log.info("✓ Qualcomm AI Hub SDK is ready")
        return

    if args.action == "compile" and args.model:
        submit_compile_job(args.model, args.device)
    elif args.action == "all":
        submit_all_models(args.onnx_dir, args.device)
    else:
        log.info("Use --action compile --model <path> or --action all")


if __name__ == "__main__":
    main()
