# OmniVoice — Hy-MT LOW-RAM profile asset preparation
#
# Downloads the Sherry 1.25-bit STQ GGUF (440 MB) from
# tencent/Hy-MT1.5-1.8B-1.25bit-GGUF and places it into the Android assets
# dir (and onnx_models/ for the Python backend).
#
# NOTE: this GGUF only decodes with llama.cpp built with the STQ kernel
# (llama.cpp PR #22836).  On Android that build is wired up automatically by
# android/app/src/main/cpp/CMakeLists.txt; on desktop, build llama-cpp-python
# against a STQ-enabled llama.cpp.
#
# The upstream file ALSO carries legacy tensor-type codes: its quantizer wrote
# the 1.25-bit tensors as code 42 (STQ1_0 in its old ggml enum), which the
# vendored llama.cpp numbers 43 (42 = Q2_0 there).  As downloaded, the header
# offset chain does not line up and gguf_init_from_file() fails ("GGUF model
# not loaded").  After download/copy this script remaps the codes in place via
# hymt_gguf_typefix.py — idempotent and proof-gated, so already-fixed (or
# genuinely Q2_0) files are left untouched.
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from hymt_gguf_typefix import GgufParseError, patch_file as patch_gguf_types

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO = "tencent/Hy-MT1.5-1.8B-1.25bit-GGUF"
GGUF_NAME = "Hy-MT1.5-1.8B-1.25bit.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/main/{GGUF_NAME}"
EXPECTED_SIZE = 461_860_704  # bytes, per the HF repo listing

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_DIR = ROOT_DIR / "android" / "app" / "src" / "main" / "assets"
DEFAULT_BACKEND_DIR = ROOT_DIR / "onnx_models"


def verify(path: Path) -> bool:
    if not path.exists() or path.stat().st_size != EXPECTED_SIZE:
        return False
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"GGUF"
    except Exception:
        return False


def ensure_current_types(path: Path) -> None:
    """Remap legacy STQ1_0 tensor-type codes in place (idempotent, proof-gated).

    See hymt_gguf_typefix.py for the full analysis.  Exits the script when the
    file cannot be proven to be the legacy-encoding Hy-MT file — we never
    leave an ambiguous model in the assets/backend dirs.
    """
    try:
        patched, msg = patch_gguf_types(path)
    except (GgufParseError, OSError) as e:
        log.error(f"  [Typefix] {path}: {e}")
        sys.exit(1)
    log.info(f"  [Typefix] {path}: {msg}")


def download_via_hf_hub(dst: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        log.info(f"  [HF Hub] Downloading {GGUF_NAME} from {REPO}...")
        downloaded_path = hf_hub_download(
            repo_id=REPO,
            filename=GGUF_NAME,
            resume_download=True,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_path, dst)
        return verify(dst)
    except Exception as e:
        log.warning(f"  [HF Hub] Download failed ({e}), falling back to resumable stream download...")
        return False


def download_resumable(url: str, dst: Path, max_retries: int = 10) -> bool:
    try:
        import requests
    except ImportError:
        log.warning("  [Deps] 'requests' not installed — stdlib urllib fallback "
                    "(pip install requests huggingface_hub for the robust path)")
        return _download_resumable_urllib(url, dst, max_retries)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = tmp.stat().st_size if tmp.exists() else 0
            if downloaded >= EXPECTED_SIZE:
                tmp.replace(dst)  # os.replace: atomic overwrite, Windows-safe
                return verify(dst)

            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                log.info(f"  [Resume] Attempt {attempt}/{max_retries}: resuming from {downloaded / 1e6:.1f} MB...")
            else:
                log.info(f"  [Download] Attempt {attempt}/{max_retries}: {url}")

            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    log.error(f"  [Error] HTTP {r.status_code}")
                    time.sleep(2)
                    continue

                mode = "ab" if downloaded > 0 and r.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0

                chunk_size = 1024 * 1024  # 1 MB
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pct = int(downloaded * 100 / EXPECTED_SIZE)
                            sys.stdout.write(f"\r    Progress: {pct}% ({downloaded / 1e6:.1f}/{EXPECTED_SIZE / 1e6:.1f} MB)")
                            sys.stdout.flush()

            sys.stdout.write("\n")
            if tmp.stat().st_size == EXPECTED_SIZE:
                tmp.replace(dst)  # os.replace: atomic overwrite, Windows-safe
                return verify(dst)

        except (requests.RequestException, IOError) as e:
            sys.stdout.write("\n")
            log.warning(f"  [Warning] Connection interrupted: {e}. Retrying in 3s...")
            time.sleep(3)

    return False


def _download_resumable_urllib(url: str, dst: Path, max_retries: int = 10) -> bool:
    """Stdlib fallback when `requests` is unavailable: single-stream HTTP(S)
    with Range resume and the same .tmp + size-verify contract."""
    import urllib.error
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = tmp.stat().st_size if tmp.exists() else 0
            if downloaded >= EXPECTED_SIZE:
                tmp.replace(dst)
                return verify(dst)

            req = urllib.request.Request(url)
            if downloaded > 0:
                req.add_header("Range", f"bytes={downloaded}-")
                log.info(f"  [Resume] Attempt {attempt}/{max_retries}: resuming from {downloaded / 1e6:.1f} MB...")
            else:
                log.info(f"  [Download] Attempt {attempt}/{max_retries}: {url}")

            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                if status not in (200, 206):
                    log.error(f"  [Error] HTTP {status}")
                    time.sleep(2)
                    continue

                mode = "ab" if downloaded > 0 and status == 206 else "wb"
                if mode == "wb":
                    downloaded = 0
                with open(tmp, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

            sys.stdout.write(f"\r    Progress: {int(downloaded * 100 / EXPECTED_SIZE)}% "
                             f"({downloaded / 1e6:.1f}/{EXPECTED_SIZE / 1e6:.1f} MB)\n")
            sys.stdout.flush()
            if tmp.stat().st_size == EXPECTED_SIZE:
                tmp.replace(dst)
                return verify(dst)
            log.warning(f"  [Warning] Incomplete ({tmp.stat().st_size}/{EXPECTED_SIZE} bytes). Retrying in 3s...")
            time.sleep(3)
        except Exception as e:
            log.warning(f"  [Warning] Connection interrupted: {e}. Retrying in 3s...")
            time.sleep(3)

    return False


def download_file(dst: Path) -> bool:
    if verify(dst):
        log.info(f"  [Skip] {dst.name} already present and verified ({EXPECTED_SIZE / 1e6:.0f} MB)")
        return True

    # Try HF Hub first (handles multi-connections, resume, cache automatically)
    if download_via_hf_hub(dst):
        return True

    # Fallback to direct HTTP with Range resume
    return download_resumable(URL, dst)


def main():
    ap = argparse.ArgumentParser(description="Fetch the Hy-MT 1.25-bit STQ GGUF")
    ap.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    ap.add_argument("--backend-dir", type=Path, default=DEFAULT_BACKEND_DIR)
    args = ap.parse_args()

    target = args.assets_dir / GGUF_NAME
    log.info(f"Target assets path: {target}")
    if not download_file(target):
        log.error("Failed to download or verify GGUF model file.")
        sys.exit(1)

    # Fresh downloads (and pre-typefix copies) still carry legacy type codes.
    ensure_current_types(target)

    backend = args.backend_dir / GGUF_NAME
    backend.parent.mkdir(parents=True, exist_ok=True)
    if not verify(backend):
        shutil.copy2(target, backend)
    ensure_current_types(backend)

    log.info(f"  [Done] GGUF verified and available at:")
    log.info(f"    - Assets:  {target}")
    log.info(f"    - Backend: {backend}")
    log.info("  Next: build the Android app (llama.cpp STQ kernel is fetched by CMake)")


if __name__ == "__main__":
    main()
