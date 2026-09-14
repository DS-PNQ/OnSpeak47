# OmniVoice — launcher for the pinned upstream ORT GenAI model builder.
#
# The v0.15.2 builder source (fetched by 08_export_hymt_onnx_int4.py) tracks
# onnxruntime main and passes `bits=` to MatMulNBitsQuantizer (mixed
# int4/int8 support, upstream #2275), but the newest PyPI onnxruntime
# (1.23.2) predates that parameter. For our INT4-only export the old
# quantizer is already 4-bit, so this launcher installs a narrow compat
# wrapper (asserts bits==4, forwards everything else) and then runs the
# upstream builder in-process. If a future onnxruntime grows the `bits`
# parameter, the shim disables itself automatically.
from __future__ import annotations

import inspect
import runpy
import sys
from pathlib import Path


def _install_matmul_nbits_compat() -> None:
    try:
        import onnxruntime.quantization.matmul_nbits_quantizer as mod
    except Exception:
        return
    try:
        params = inspect.signature(mod.MatMulNBitsQuantizer.__init__).parameters
    except Exception:
        return
    if "bits" in params:
        return  # onnxruntime already supports the new API — no shim needed

    _orig = mod.MatMulNBitsQuantizer

    class CompatMatMulNBitsQuantizer(_orig):  # type: ignore[valid-type,misc]
        def __init__(self, *args, bits: int = 4, **kwargs):
            if bits != 4:
                raise RuntimeError(
                    f"INT4-only compat shim cannot satisfy bits={bits} — "
                    "upgrade onnxruntime to a version with native `bits` support"
                )
            super().__init__(*args, **kwargs)

    mod.MatMulNBitsQuantizer = CompatMatMulNBitsQuantizer


def main() -> None:
    # argv: <models_dir> <builder args...>
    models_dir = Path(sys.argv[1])
    sys.argv = ["builder.py", *sys.argv[2:]]
    sys.path.insert(0, str(models_dir))  # `from builders import ...`
    _install_matmul_nbits_compat()
    runpy.run_path(str(models_dir / "builder.py"), run_name="__main__")


if __name__ == "__main__":
    main()
