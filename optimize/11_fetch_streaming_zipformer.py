#!/usr/bin/env python3
"""Fetch streaming Zipformer assets for the realtime VI/EN/ZH pipeline.

Spec §4 + §33. Downloads (gitignored) model files into
``onnx_models/streaming/`` and optionally stages them into
``android/app/src/main/assets/``.

Sources (re-check at implementation time — upstream repacks happen):
  EN+ZH: csukuangfj/k2fsa-zipformer-chinese-english-mixed (HuggingFace) — ONE
         bilingual model for both language slots (was: separate EN 2023-06-26
         and ZH int8 2025-06-30 tarball releases)
  VI : hynt/Zipformer-30M-RNNT-Streaming-6000h (HuggingFace, streaming)
  VAD: silero_vad.onnx (sherpa-onnx release asset)

Usage:
    python optimize/11_fetch_streaming_zipformer.py --list
    python optimize/11_fetch_streaming_zipformer.py --lang mixed --stage-assets
    python optimize/11_fetch_streaming_zipformer.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "onnx_models" / "streaming"
ASSETS_DIR = ROOT / "android" / "app" / "src" / "main" / "assets"

SHERPA_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# EN + ZH are served by ONE bilingual model. Member names verified against
# the upstream repo (huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed).
# The export ships int8 encoder/joiner + fp32 decoder (no int8 decoder is
# published), so the local asset names keep the real precision suffix.
MIXED_HF_REPO = "csukuangfj/k2fsa-zipformer-chinese-english-mixed"
MIXED_MEMBERS = {
    "zipformer_en_zh_mixed_encoder.int8.onnx": "exp/encoder-epoch-99-avg-1.int8.onnx",
    "zipformer_en_zh_mixed_decoder.onnx": "exp/decoder-epoch-99-avg-1.onnx",
    "zipformer_en_zh_mixed_joiner.int8.onnx": "exp/joiner-epoch-99-avg-1.int8.onnx",
    "zipformer_en_zh_mixed_tokens.txt": "data/lang_char_bpe/tokens.txt",
}

# VI chunk preference: chunk-16 is the latency baseline (spec §9/§20);
# chunk-32/64 are fallbacks. The HF repo ships all three.
VI_CHUNK_PREFERENCE = ("chunk-16", "chunk-32", "chunk-64")

# Vietnamese streaming model lives on HuggingFace (no fixed tarball name —
# resolved at fetch time via huggingface_hub when available).
VI_HF_REPO = "hynt/Zipformer-30M-RNNT-Streaming-6000h"
VI_WANTED_SUFFIXES = (".onnx", ".onnx_data", "tokens.txt", ".model", ".json")

VAD_URL = f"{SHERPA_RELEASE}/silero_vad.onnx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url}\n     -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    print(f"        sha256={_sha256(dest)[:16]}… size={dest.stat().st_size / 1e6:.1f} MB")
    return dest


def _have_all(names: list[str]) -> bool:
    return all((OUT_DIR / n).exists() and (OUT_DIR / n).stat().st_size > 0 for n in names)


def fetch_mixed() -> list[Path]:
    """Fetch the shared EN+ZH bilingual model from HuggingFace."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if _have_all(list(MIXED_MEMBERS)):
        print(f"[skip] mixed: all {len(MIXED_MEMBERS)} assets already present")
        return [OUT_DIR / n for n in MIXED_MEMBERS]
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print("[error] huggingface_hub is required for the EN/ZH mixed model:\n"
              "        pip install huggingface_hub\n"
              f"        repo: {MIXED_HF_REPO}", file=sys.stderr)
        return []
    got = []
    for asset_name, repo_path in MIXED_MEMBERS.items():
        dest = OUT_DIR / asset_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] mixed: {asset_name} already present")
            got.append(dest)
            continue
        try:
            src = Path(hf_hub_download(repo_id=MIXED_HF_REPO, filename=repo_path))
        except Exception as exc:  # network / 404 / gated repo
            print(f"[warn] {repo_path} not fetched ({exc}); skipping {asset_name}",
                  file=sys.stderr)
            continue
        shutil.copy2(src, dest)
        print(f"[pick] {asset_name} <- {repo_path} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        got.append(dest)
    return got


def fetch_vi() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print("[error] huggingface_hub is required for the VI model:\n"
              "        pip install huggingface_hub\n"
              f"        repo: {VI_HF_REPO}", file=sys.stderr)
        return []
    snap = Path(snapshot_download(VI_HF_REPO))
    got = []
    for cand in sorted(snap.rglob("*")):
        if cand.is_file() and cand.name.endswith(VI_WANTED_SUFFIXES):
            dest = OUT_DIR / cand.name
            if not dest.exists():
                shutil.copy2(cand, dest)
            got.append(dest)
    # Normalize to the asset names AsrState/ZipformerModelManager expect,
    # preferring the chunk-16 variant (latency baseline, spec §9).
    by_kind: dict[str, list[Path]] = {"encoder": [], "decoder": [], "joiner": []}
    tokens_src: Path | None = None
    for p in got:
        n = p.name.lower()
        if p.suffix != ".onnx":
            if n == "tokens.txt":
                tokens_src = p
            continue
        for kind in by_kind:
            if n.startswith(kind):
                by_kind[kind].append(p)

    def _pick(cands: list[Path]) -> Path | None:
        for chunk in VI_CHUNK_PREFERENCE:
            for p in cands:
                if chunk in p.name.lower():
                    return p
        return cands[0] if cands else None

    final = []
    for kind, asset in (("encoder", "zipformer_vi_encoder.onnx"),
                        ("decoder", "zipformer_vi_decoder.onnx"),
                        ("joiner", "zipformer_vi_joiner.onnx")):
        src = _pick(by_kind[kind])
        if src is None:
            print(f"[warn] no VI {kind} .onnx in {VI_HF_REPO}", file=sys.stderr)
            continue
        dst = OUT_DIR / asset
        shutil.copy2(src, dst)
        print(f"[pick] {asset} <- {src.name}")
        final.append(dst)
    if tokens_src is not None:
        dst = OUT_DIR / "zipformer_vi_tokens.txt"
        shutil.copy2(tokens_src, dst)
        final.append(dst)
    else:
        # The hynt repo ships bpe.model but no tokens.txt — derive it in the
        # sherpa transducer convention (<blk>/<sos/eos>/<unk> + spm pieces).
        dst = OUT_DIR / "zipformer_vi_tokens.txt"
        if _tokens_from_bpe_model(OUT_DIR / "bpe.model", dst):
            final.append(dst)
    # Drop the raw chunk-variant downloads — the canonical copies above are
    # byte-identical. Keep bpe.model/config.json for provenance.
    for p in OUT_DIR.glob("*-chunk-*-left-*.onnx"):
        if p.name.startswith(("encoder", "decoder", "joiner")):
            p.unlink()
            print(f"[clean] removed raw download {p.name}")
    return final or got


def _tokens_from_bpe_model(bpe_path: Path, dst: Path) -> bool:
    try:
        import sentencepiece as spm
    except ImportError:
        print("[error] sentencepiece is required to derive VI tokens.txt:\n"
              "        pip install sentencepiece", file=sys.stderr)
        return False
    if not bpe_path.exists():
        print(f"[warn] {bpe_path.name} missing — cannot derive tokens.txt", file=sys.stderr)
        return False
    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_path))
    with open(dst, "w", encoding="utf-8") as f:
        f.write("<blk> 0\n<sos/eos> 1\n<unk> 2\n")
        for i in range(3, sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    print(f"[gen] {dst.name} <- {bpe_path.name} ({sp.get_piece_size()} pieces)")
    return True


def fetch_vad() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / "silero_vad.onnx"
    if dst.exists() and dst.stat().st_size > 0:
        print("[skip] vad: silero_vad.onnx already present")
        return dst
    return _download(VAD_URL, dst)


def stage_to_assets(files: list[Path]) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f and f.exists():
            shutil.copy2(f, ASSETS_DIR / f.name)
            print(f"[stage] {f.name} -> assets/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["vi", "en", "zh", "mixed", "vad"],
                    help="fetch one asset set (en/zh both map to the mixed model)")
    ap.add_argument("--all", action="store_true", help="fetch vi+mixed+vad")
    ap.add_argument("--list", action="store_true", help="print expected asset names")
    ap.add_argument("--stage-assets", action="store_true", help="copy into android assets/")
    args = ap.parse_args()

    if args.list:
        print("EN+ZH (shared mixed model, "
              f"{MIXED_HF_REPO}):", list(MIXED_MEMBERS))
        print("VI: zipformer_vi_{encoder,decoder,joiner}.onnx + zipformer_vi_tokens.txt"
              f"  (from {VI_HF_REPO})")
        print("VAD: silero_vad.onnx")
        return 0

    if args.all or args.lang is None:
        targets = ["vi", "mixed", "vad"]
        print("[info] fetching vi + mixed (EN/ZH) + vad")
    else:
        targets = [args.lang]
    fetched: list[Path] = []
    mixed_done = False
    for t in targets:
        if t == "vi":
            fetched.extend(fetch_vi())
        elif t == "vad":
            fetched.append(fetch_vad())
        else:
            # en / zh / mixed are all the same bilingual export — fetch once.
            if mixed_done:
                print(f"[skip] {t}: same files as 'mixed'")
                continue
            fetched.extend(fetch_mixed())
            mixed_done = True
    print(f"[done] {len(fetched)} files in {OUT_DIR}")
    if args.stage_assets:
        stage_to_assets(fetched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
