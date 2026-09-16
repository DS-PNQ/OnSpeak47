#!/usr/bin/env python3
"""Fetch VoxLingua107 ECAPA acoustic LID model and labels.

Pipeline §4, §11–§13, §42.
Downloads community ONNX export:
  beginning-ai/speechbrain-lang-id-voxlingua107-ecapa-onnx
    - model.onnx (~85 MB) -> onnx_models/voxlingua_lid_ecapa.onnx
    - labels.json         -> onnx_models/voxlingua_lid_labels.json

Usage:
    python optimize/12_fetch_voxlingua_lid.py --check
    python optimize/12_fetch_voxlingua_lid.py --download
    python optimize/12_fetch_voxlingua_lid.py --download --stage-assets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "onnx_models"
ASSETS_DIR = ROOT / "android" / "app" / "src" / "main" / "assets"

HF_REPO_URL = (
    "https://huggingface.co/beginning-ai/speechbrain-lang-id-voxlingua107-ecapa-onnx/resolve/main"
)

FILES = {
    "voxlingua_lid_ecapa.onnx": f"{HF_REPO_URL}/model.onnx",
    "voxlingua_lid_labels.json": f"{HF_REPO_URL}/labels.json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url}\n     -> {dest}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "OnSpeak47-Fetch/1.0"},
    )
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    print(f"        sha256={_sha256(dest)[:16]}… size={dest.stat().st_size / 1e6:.1f} MB")
    return dest


def fetch_assets(stage_assets: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest = OUT_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {filename} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            _download(url, dest)

        if stage_assets:
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            asset_dest = ASSETS_DIR / filename
            shutil.copyfile(dest, asset_dest)
            print(f"[stage] -> {asset_dest}")


def check_assets() -> bool:
    print(f"=== Checking VoxLingua Assets in {OUT_DIR} ===")
    all_ok = True
    for filename in FILES:
        p = OUT_DIR / filename
        if p.exists() and p.stat().st_size > 0:
            print(f"  [OK] {filename}: {p.stat().st_size / 1e6:.2f} MB")
        else:
            print(f"  [MISSING] {filename}")
            all_ok = False

    model_path = OUT_DIR / "voxlingua_lid_ecapa.onnx"
    if model_path.exists():
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(model_path))
            inp = sess.get_inputs()[0]
            out = sess.get_outputs()[0]
            print(f"  [ONNX Graph] input: {inp.name} shape={inp.shape} dtype={inp.type}")
            print(f"  [ONNX Graph] output: {out.name} shape={out.shape} dtype={out.type}")
        except Exception as e:
            print(f"  [WARN] onnxruntime inspect failed: {e}")

    labels_path = OUT_DIR / "voxlingua_lid_labels.json"
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            print(f"  [Labels] count: {len(labels)} (expected 107)")
            if isinstance(labels, list) and len(labels) == 107:
                print(f"  [Labels] EN idx={labels.index('en')} VI idx={labels.index('vi')} ZH idx={labels.index('zh')}")
        except Exception as e:
            print(f"  [WARN] labels inspect failed: {e}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Fetch VoxLingua107 ECAPA acoustic LID assets")
    parser.add_argument("--download", action="store_true", help="Download missing models/labels")
    parser.add_argument("--stage-assets", action="store_true", help="Copy into android assets directory")
    parser.add_argument("--check", action="store_true", help="Check local asset status")
    args = parser.parse_args()

    if not (args.download or args.stage_assets or args.check):
        parser.print_help()
        sys.exit(0)

    if args.download or args.stage_assets:
        fetch_assets(stage_assets=args.stage_assets)

    if args.check:
        ok = check_assets()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
