# OmniVoice — Hy-MT HIGH-END profile: INT4 ONNX for ONNX Runtime GenAI
#
# Builds tencent/HY-MT1.5-1.8B (arch HunYuanDenseV1ForCausalLM — "HunYuan
# Dense V1" is supported by the ORT GenAI model builder) into an INT4 ONNX
# model folder that ORT GenAI can run on CPU or on the QNN EP.
#
# Builder sourcing: PyPI's onnxruntime-genai (<= 0.11.4) predates Hunyuan
# support (upstream PR #2144/#2189, first released in 0.14.0 — and upstream
# stopped publishing Python wheels to PyPI after 0.11.x, shipping only
# native libs + AARs on GitHub releases). So this script prefers the
# installed wheel's builder when it knows HunYuanDenseV1, and otherwise
# fetches the pinned upstream source tree (pure-Python export path, no
# native build needed) and runs its builder instead.
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
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL = "tencent/HY-MT1.5-1.8B"

# Upstream tag whose model builder knows HunYuanDenseV1 (PR #2144/#2189).
OGA_BUILDER_TAG = "v0.15.2"
OGA_BUILDER_ZIP_URL = (
    f"https://github.com/microsoft/onnxruntime-genai/archive/refs/tags/{OGA_BUILDER_TAG}.zip"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT_DIR / "android" / "app" / "src" / "main" / "assets" / "hymt_int4_onnx"
DEFAULT_CACHE_DIR = ROOT_DIR / "onnx_models" / "hf_cache"
DEFAULT_BACKEND_COPY = ROOT_DIR / "onnx_models" / "hymt_int4_onnx"
# Pinned upstream builder source (under gitignored onnx_models/, extracted once).
BUILDER_SRC_DIR = ROOT_DIR / "onnx_models" / "_oga_builder" / f"onnxruntime-genai-{OGA_BUILDER_TAG[1:]}"

# Export holds BF16 weights in torch (~1x), an FP32 ONNX proto (~2x) and the
# quantizer's working copies (~2x): budget ~5x the safetensors footprint.
# HY-MT1.5-1.8B ships BF16 (4.1 GB) → ~20 GB RAM. A 16 GB box OOMs
# deterministically inside onnx_ir serialization, so fail fast instead.
EXPORT_RAM_FACTOR = 5


def physical_memory() -> tuple[int, int]:
    """(total, free) physical RAM bytes; (0, 0) when undetectable."""
    import platform

    if platform.system() == "Windows":
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
        return 0, 0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        free = os.sysconf("SC_AVPHYS_PAGES") * page
        return total, free
    except Exception:
        return 0, 0


def check_export_ram(cache: Path) -> bool:
    """Fail fast when the box cannot hold the FP32-intermediate export."""
    snaps = cache / "models--tencent--HY-MT1.5-1.8B" / "snapshots"
    weight_bytes = 0
    if snaps.exists():
        for f in snaps.rglob("*.safetensors"):
            try:
                weight_bytes += f.stat().st_size
            except OSError:
                pass
    if weight_bytes == 0:
        # Weights not cached yet — estimate from the known BF16 footprint.
        weight_bytes = 4_077_072_784
    required = weight_bytes * EXPORT_RAM_FACTOR
    total, free = physical_memory()
    log.info(f"  [RAM] weights={weight_bytes / 1e9:.1f} GB → need ~{required / 1e9:.0f} GB; "
             f"machine total={total / 1e9:.1f} GB free={free / 1e9:.1f} GB")
    if total and total < required:
        log.error(f"  [RAM] Export needs ~{required / 1e9:.0f} GB but the machine has "
                  f"{total / 1e9:.1f} GB — it will OOM in serialization. "
                  "Run this script on a 32 GB+ machine, then copy "
                  "hymt_int4_onnx/ back. The GGUF path (07_) is unaffected "
                  "and remains the production translation backend.")
        return False
    if free and free < required:
        log.warning("  [RAM] Free RAM below estimate — close other apps and retry; "
                    "OOM inside onnx_ir serialization means the box is too small.")
    return True


def wheel_builder_supports_hunyuan() -> bool:
    """True when the installed onnxruntime_genai wheel can build HY-MT1.5."""
    try:
        import onnxruntime_genai.models.builder as b

        return "HunYuanDenseV1ForCausalLM" in Path(b.__file__).read_text(encoding="utf-8")
    except Exception:
        return False


def ensure_source_builder() -> Path:
    """Fetch + extract the pinned upstream builder tree; return its models/ dir."""
    models_dir = BUILDER_SRC_DIR / "src" / "python" / "py" / "models"
    if (models_dir / "builder.py").exists() and (models_dir / "builders" / "hunyuan.py").exists():
        return models_dir
    BUILDER_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = BUILDER_SRC_DIR.parent / f"oga-{OGA_BUILDER_TAG}.zip"
    if not zip_path.exists():
        log.info(f"  [Builder] downloading upstream model builder ({OGA_BUILDER_TAG})...")
        urllib.request.urlretrieve(OGA_BUILDER_ZIP_URL, zip_path)
    log.info(f"  [Builder] extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(BUILDER_SRC_DIR.parent)
    if not (models_dir / "builder.py").exists():
        raise RuntimeError(f"upstream tree layout changed — {models_dir} missing builder.py")
    return models_dir


def build(ep: str, out: Path, cache: Path, hf_token: str | None = None) -> bool:
    token_val = hf_token or os.environ.get("HF_TOKEN") or "false"
    if wheel_builder_supports_hunyuan():
        log.info("  [Builder] installed onnxruntime_genai wheel supports HunYuanDenseV1")
        cmd = [sys.executable, "-m", "onnxruntime_genai.models.builder"]
        cwd: Path | None = None
    else:
        log.info("  [Builder] wheel predates Hunyuan support — using pinned upstream source tree")
        models_dir = ensure_source_builder()
        # Run via the compat launcher: v0.15.2's builder expects a newer
        # onnxruntime quantizer API (`bits=`) than PyPI ships; the launcher
        # shims it for our INT4-only export (see _oga_hunyuan_compat.py).
        compat = ROOT_DIR / "optimize" / "_oga_hunyuan_compat.py"
        cmd = [sys.executable, str(compat), str(models_dir)]
        cwd = models_dir  # `from builders import ...` resolves from here
    cmd += [
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
    proc = subprocess.run(cmd, cwd=cwd)
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

    if not check_export_ram(args.cache):
        sys.exit(2)

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
