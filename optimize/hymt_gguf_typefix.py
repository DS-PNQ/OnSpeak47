#!/usr/bin/env python3
"""OmniVoice — Hy-MT GGUF legacy tensor-type remap (the "GGUF model not loaded" fix).

The Sherry-quantized ``Hy-MT1.5-1.8B-1.25bit.gguf``
(``tencent/Hy-MT1.5-1.8B-1.25bit-GGUF``) was written by a quantizer built
against a *legacy* ggml enum in which ``STQ1_0 = 42``.  The llama.cpp snapshot
vendored for Android (PR #22836 — see
``android/app/src/main/cpp/third_party/README.md``) numbers
``Q1_0 = 41, Q2_0 = 42, STQ1_0 = 43``, so as downloaded the 224 1.25-bit
tensors are read as Q2_0 (18 B / 64 el) instead of STQ1_0 (42 B / 256 el).
Every one of those tensors then has the wrong byte size, the accumulated
data-section offsets stop matching (first divergence: blk.0.attn_k_norm),
``gguf_init_from_file()`` returns NULL, and ``HyMtGgufJNI.loadModel()``
returns handle 0 => ``[error: GGUF model not loaded]``.

The fix is a size-neutral, in-place header rewrite — nothing in the ~457 MB
tensor-data region moves:

  * every tensor-info ``type`` field 42 -> 43  (GGML_TYPE_STQ1_0)
  * KV ``general.file_type``     41 -> 42  (LLAMA_FTYPE_MOSTLY_STQ1_0)

The rewrite is *proof-gated*: it only runs when the file's own offset chain
is consistent under the STQ1_0 interpretation (42 B / 256 el) and inconsistent
under the as-read Q2_0 one.  A genuine current-enum Q2_0 GGUF (whose tensors
also carry code 42) is detected as self-consistent and left untouched.  The
tool is idempotent — an already-patched file is a no-op.

Usage:
    python hymt_gguf_typefix.py model.gguf [model2.gguf ...]   # patch
    python hymt_gguf_typefix.py --check model.gguf              # dry run

Exit codes: 0 = patched / nothing to do, 2 = parse/refusal error.
Importable: ``needs_patch(path)``, ``patch_file(path, dry_run=False)``.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

MAGIC = b"GGUF"

# ggml tensor-type table (CURRENT llama.cpp numbering, matches the vendored
# PR #22836 snapshot): code -> (name, blck_size, type_size in bytes)
GGML_TYPES = {
    0:  ("F32", 1, 4),
    1:  ("F16", 1, 2),
    2:  ("Q4_0", 32, 18),
    3:  ("Q4_1", 32, 20),
    6:  ("Q5_0", 32, 22),
    7:  ("Q5_1", 32, 24),
    8:  ("Q8_0", 32, 34),
    9:  ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 42),
    17: ("IQ2_XS", 256, 52),
    18: ("IQ3_XXS", 256, 96),
    19: ("IQ1_S", 256, 56),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 32, 18),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 62),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 85),
    35: ("TQ2_0", 256, 128),
    39: ("MXFP4", 32, 16),
    40: ("NVFP4", 16, 8),
    41: ("Q1_0", 256, 42),
    42: ("Q2_0", 64, 18),
    43: ("STQ1_0", 256, 42),
}

LEGACY_TYPE_CODE = 42   # STQ1_0 in the quantizer's legacy ggml enum
CURRENT_TYPE_CODE = 43  # GGML_TYPE_STQ1_0 in the vendored llama.cpp (PR #22836)

# general.file_type values (llama.h): the legacy quantizer's
# LLAMA_FTYPE_MOSTLY_STQ1_0 is 41; the vendored snapshot numbers it 42.
LEGACY_FTYPE = 41
CURRENT_FTYPE = 42

DEFAULT_ALIGNMENT = 32

class GgufParseError(Exception):
    pass


def _read_exact(f, n, what):
    b = f.read(n)
    if len(b) != n:
        raise GgufParseError("truncated %s" % what)
    return b


def _skip_value(f, vt):
    """Skip one GGUF KV value of the given type (bytes only, no decode)."""
    if vt in (0, 1):          # u8 / i8
        _read_exact(f, 1, "kv value")
    elif vt in (2, 3):         # u16 / i16
        _read_exact(f, 2, "kv value")
    elif vt in (4, 5, 6):      # u32 / i32 / f32
        _read_exact(f, 4, "kv value")
    elif vt == 7:              # bool (1 byte!)
        _read_exact(f, 1, "kv value")
    elif vt == 8:              # string
        (sl,) = struct.unpack("<Q", _read_exact(f, 8, "string length"))
        _read_exact(f, sl, "string value")
    elif vt == 9:              # array
        (et,) = struct.unpack("<I", _read_exact(f, 4, "array element type"))
        (cnt,) = struct.unpack("<Q", _read_exact(f, 8, "array length"))
        if et == 8:
            for _ in range(cnt):
                (sl,) = struct.unpack("<Q", _read_exact(f, 8, "string length"))
                _read_exact(f, sl, "string element")
        elif et in (0, 1, 7):
            _read_exact(f, cnt, "array payload")
        elif et in (2, 3):
            _read_exact(f, 2 * cnt, "array payload")
        elif et in (4, 5, 6):
            _read_exact(f, 4 * cnt, "array payload")
        elif et in (10, 11, 12):
            _read_exact(f, 8 * cnt, "array payload")
        else:
            raise GgufParseError("bad array element type %d" % et)
    elif vt in (10, 11, 12):   # u64 / i64 / f64
        _read_exact(f, 8, "kv value")
    else:
        raise GgufParseError("bad value type %d" % vt)


def parse_header(f):
    """Parse magic/version/KVs/tensor-infos. Returns a dict with:
       version, n_tensors, kv (key names), tensors (list of dicts),
       alignment, file_type (None or (payload_offset, value)), data_start.
    KV entries other than general.file_type / general.alignment are skipped
    without decoding; only their presence is recorded."""
    if _read_exact(f, 4, "magic") != MAGIC:
        raise GgufParseError("not a GGUF file")
    (ver,) = struct.unpack("<I", _read_exact(f, 4, "version"))
    if ver < 1 or ver > 3:
        raise GgufParseError("unsupported GGUF version %d" % ver)
    n_tensors, n_kv = struct.unpack("<QQ", _read_exact(f, 16, "header counts"))

    kv = []
    file_type = None    # (payload_offset, value)
    alignment = None
    for _ in range(n_kv):
        (kl,) = struct.unpack("<Q", _read_exact(f, 8, "key length"))
        key = _read_exact(f, kl, "key").decode("utf-8", "replace")
        (vt,) = struct.unpack("<I", _read_exact(f, 4, "kv type"))
        if vt == 4 and key in ("general.file_type", "general.alignment"):
            off = f.tell()
            (val,) = struct.unpack("<I", _read_exact(f, 4, key))
            if key == "general.file_type":
                file_type = (off, val)
            else:
                alignment = val
        else:
            _skip_value(f, vt)
        kv.append(key)

    tensors = []
    for _ in range(n_tensors):
        (nl,) = struct.unpack("<Q", _read_exact(f, 8, "tensor name length"))
        name = _read_exact(f, nl, "tensor name").decode("utf-8", "replace")
        (nd,) = struct.unpack("<I", _read_exact(f, 4, "dim count"))
        if nd < 1 or nd > 4:
            raise GgufParseError("bad dim count %d for %s" % (nd, name))
        dims = struct.unpack("<%dQ" % nd, _read_exact(f, 8 * nd, "dims"))
        type_field_off = f.tell()
        (tc,) = struct.unpack("<I", _read_exact(f, 4, "tensor type"))
        (off,) = struct.unpack("<Q", _read_exact(f, 8, "tensor offset"))
        tensors.append({
            "name": name, "dims": dims, "type": tc,
            "offset": off, "type_field_off": type_field_off,
        })

    align = alignment or DEFAULT_ALIGNMENT
    if align == 0 or (align & (align - 1)) != 0:
        raise GgufParseError("alignment %d is not a power of 2" % align)

    # gguf.cpp seeks PAD(tell, alignment) to find the data section start
    data_start = (f.tell() + align - 1) // align * align

    return {
        "version": ver, "n_tensors": n_tensors, "kv": kv,
        "tensors": tensors, "alignment": align,
        "file_type": file_type, "data_start": data_start,
    }

def _tensor_nbytes(t, entry):
    """Byte size of a tensor's data, as ggml_nbytes() computes it."""
    _, blck, tsize = entry
    nbytes = ((t["dims"][0] + blck - 1) // blck) * tsize
    for d in t["dims"][1:]:
        nbytes *= d
    return nbytes


def chain_check(tensors, align):
    """Simulate gguf_init_from_file()'s data-section accumulation with the
    CURRENT enum. Returns (ok, total_data_bytes, first_failure_or_None)."""
    expected = 0
    for t in tensors:
        entry = GGML_TYPES.get(t["type"])
        if entry is None:
            return False, None, "%s: unknown tensor type %d" % (t["name"], t["type"])
        nbytes = _tensor_nbytes(t, entry)
        if t["offset"] != expected:
            return False, None, "%s: offset %d != expected %d" % (
                t["name"], t["offset"], expected)
        expected += (nbytes + align - 1) // align * align
    return True, expected, None


def analyze(path):
    """Classify a GGUF under the current llama.cpp enum.

    Returns (state, detail) with state in:
      "current" — offset chain consistent as-read: the file already loads
                  under the vendored enum (patched, or never legacy). No-op.
      "legacy"  — chain fails as-read but is consistent when the code-42
                  tensors are read as STQ1_0: needs the 42->43 remap.
      "broken"  — neither interpretation self-consistent: refuse to touch.
    """
    with open(path, "rb") as f:
        hdr = parse_header(f)
    tensors = hdr["tensors"]
    n42 = sum(1 for t in tensors if t["type"] == LEGACY_TYPE_CODE)

    ok_cur, _, err_cur = chain_check(tensors, hdr["alignment"])
    if ok_cur:
        return "current", "offset chain consistent as-read (%d code-42 tensors)" % n42

    remapped = [dict(t) for t in tensors]
    for t in remapped:
        if t["type"] == LEGACY_TYPE_CODE:
            t["type"] = CURRENT_TYPE_CODE
    ok_leg, total_leg, err_leg = chain_check(remapped, hdr["alignment"])
    if ok_leg:
        fsize = os.path.getsize(path)
        end = hdr["data_start"] + total_leg
        if end > fsize:
            return "broken", "STQ1_0 layout consistent but data (%d B) exceeds file size (%d B)" % (end, fsize)
        return "legacy", (
            "%d legacy STQ1_0(42) tensors; as-read Q2_0 view fails at: %s"
            % (n42, err_cur)
        )
    return "broken", "as-read: %s | as-STQ1_0: %s" % (err_cur, err_leg)

def needs_patch(path):
    """True when the GGUF carries the legacy STQ1_0 type codes."""
    state, _ = analyze(path)
    return state == "legacy"


def patch_file(path, dry_run=False):
    """Remap legacy tensor types 42->43 (+ file_type 41->42) in place.

    Returns (patched: bool, message: str). Raises GgufParseError when the
    file is broken or the interpretation cannot be proven. Never touches a
    file that already loads under the current enum.
    """
    path = Path(path)
    state, detail = analyze(path)
    if state == "current":
        return False, "nothing to do (%s)" % detail
    if state != "legacy":
        raise GgufParseError("refusing to patch: %s" % detail)

    with open(path, "rb") as f:
        hdr = parse_header(f)
    n_patch = sum(1 for t in hdr["tensors"] if t["type"] == LEGACY_TYPE_CODE)
    ft = hdr["file_type"]

    if dry_run:
        return True, "would remap %d tensor type codes 42->43%s" % (
            n_patch, " and file_type 41->42" if ft and ft[1] == LEGACY_FTYPE else "")

    with open(path, "r+b") as f:
        for t in hdr["tensors"]:
            if t["type"] == LEGACY_TYPE_CODE:
                f.seek(t["type_field_off"])
                f.write(struct.pack("<I", CURRENT_TYPE_CODE))
        if ft is not None and ft[1] == LEGACY_FTYPE:
            f.seek(ft[0])
            f.write(struct.pack("<I", CURRENT_FTYPE))

    # Post-patch verification: the header must now parse cleanly as-is.
    state2, detail2 = analyze(path)
    if state2 != "current":
        raise GgufParseError("post-patch verification failed (%s): %s" % (state2, detail2))
    return True, "remapped %d tensor type codes 42->43 (STQ1_0)%s — verified" % (
        n_patch, " + file_type 41->42" if ft and ft[1] == LEGACY_FTYPE else "")


def main(argv):
    args = list(argv[1:])
    dry = False
    if args and args[0] == "--check":
        dry = True
        args = args[1:]
    if not args:
        sys.stderr.write(__doc__)
        return 2

    rc = 0
    for p in args:
        try:
            patched, msg = patch_file(p, dry_run=dry)
            print("%s: %s%s" % (p, "[dry-run] " if dry and patched else "", msg))
        except (GgufParseError, OSError) as e:
            print("%s: ERROR %s" % (p, e))
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))



