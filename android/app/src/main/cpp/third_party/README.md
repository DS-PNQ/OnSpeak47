# third_party/

Vendored native dependencies for `android/app/src/main/cpp/CMakeLists.txt`.

## llama.cpp (STQ kernel, PR #22836)

`llama.cpp/` holds a pinned snapshot of **PR #22836**
([ggml-cpu: add STQ1_0 ternary quantization with ARM NEON vec_dot kernel](https://github.com/ggml-org/llama.cpp/pull/22836),
head = `sjl623:STQ_0` @ `1e411d8f5a1e23525fa3265dfb4bd76265465397`).

The Hy-MT 1.25-bit GGUF (`tencent/Hy-MT1.5-1.8B-1.25bit-GGUF`) only decodes
with the STQ1_0 kernel. **The PR is NOT merged into master**, so do not
`git checkout master` here — the GGUF would fail to load at runtime with
"unknown tensor type / unsupported quantization".

**Header type-code remap (2026-09-10).** The Sherry quantizer wrote the file
against a legacy ggml enum where `STQ1_0 = 42`; this llama.cpp snapshot
numbers `Q1_0 = 41, Q2_0 = 42, STQ1_0 = 43`, so as downloaded the 224
1.25-bit tensors are misread as Q2_0 (wrong byte sizes, offset chain breaks,
`gguf_init_from_file` returns NULL → `[error: GGUF model not loaded]`).
The vendored assets GGUF is already fixed in place; to fix a fresh download
run `optimize/hymt_gguf_typefix.py` (also auto-invoked by
`optimize/07_prepare_hymt_gguf.py`). The rewrite is size-neutral, so the
Android side gates the extracted copy with an asset-revision marker
(`GGUF_ASSET_REV` in `TranslationModule.java`) to force a re-copy on devices
that previously extracted the broken file.

### Why vendored?

`CMakeLists.txt` used to `FetchContent`-clone llama.cpp from GitHub during
every build. On networks where `github.com` DNS/HTTPS is unreliable the
configure step dies with `Could not resolve host: github.com`. The vendored
copy makes the Android build fully offline.

### Refreshing / re-downloading

Run `download_llama_cpp.ps1` — downloads the PR-head tarball via the
`gh-proxy.com` mirror (fast) with fallbacks to `ghfast.top`, `ghproxy.net`
and `codeload.github.com` (codeload is usually reachable even when
`github.com` git/HTTPS is flaky, but tends to cut connections
mid-transfer, so it is the last resort). Every candidate tarball is
verified with `tar -tzf` before extraction. Alternatively, restore
`llama.cpp/` from git history.
