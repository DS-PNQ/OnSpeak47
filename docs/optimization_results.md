# OmniVoice — Runtime & RAM Optimization (target: wearable/edge ≤ 4 GB)

> **RETIRED (migrate/zipformer-only):** Whisper/NLLB optimization log. Kept
> for history — production ASR is now streaming Zipformer (see
> `docs/streaming_asr.md`). Do not follow the Whisper export/quantize
> commands below; those scripts were deleted.

Ngày: 2026-08-17 · Branch: `fakedemo4` · Mọi thay đổi model đều đã qua parity gate
(`tests_local/test_06_onnx_parity.py`, chạy bằng venv `onnxruntime==1.22.0` khớp app).

## Kết quả chính

| Hạng mục | Trước | Sau | Ghi chú |
|---|---|---|---|
| Whisper encoder | 352.7 MB fp32 | **98.5 MB** int8 (MatMul-only) + pre-opt | gate: en parity tuyệt đối; xem "điều kiện" dưới |
| Whisper decoder | 195.4 MB int8 | 195.1 MB (`.opt.onnx`) | pre-opt, kích thước giữ nguyên |
| NLLB encoder | 415.3 MB int8 | 415.3 MB (`.opt.onnx`) | pre-opt |
| NLLB decoder | 467.0 MB int8 (slim) | giữ nguyên | pre-opt bị LOẠI (inline làm phình 467→991 MB) |
| MMS-TTS sessions | 2 × ~114 MB load eager | **lazy + LRU-1** (≤1 session) | −114…−229 MB RAM thường trú |
| Peak RAM lúc init | ALL_OPT chạy trên device | **NO_OPT** với file pre-opt | tránh lớp lỗi "bad allocation" |
| Pipeline requests | bare Thread mỗi lần, không cancel | executor 1-thread + latest-wins | chống spike RAM khi request đè nhau |
| Whisper decode | (asset mới là cache-less) | whole-sequence + token budget 30/giây | duy nhất đường đúng cho graph hiện tại |
| NLLB decode | O(n²) re-feed toàn bộ | **KV-cache zero-copy** (encoder-KV giữ từ prefill) | fix "[error]" mọi ngôn ngữ |
| Quality gate | không có | parity gate tự động so baseline | chạy trước mỗi lần đổi asset |

Tổng RAM weights thường trú: **~1.66 GB → ~1.29 GB** (tối thiểu ~1.18 GB khi TTS chưa dùng).

## Parity baseline (số liệu gate)

ONNX hiện tại vs PyTorch greedy (40 câu / BLEU; 2 mẫu ASR / WER) — chi tiết:
`tests_local/baselines/onnx_parity_baseline.json`, kết quả mỗi lần chạy:
`tests_local/output/onnx_parity_results.json`.

- vi↔en, vi↔zh_hant: Δ BLEU −0.05…+0.45 (chặt)
- zh_hans: Δ −2.9…−3.1 — đặc điểm đã có của decoder int8 (không phải regression)
- ASR en: onnx WER 0.333 vs torch 0.222; ASR vi: fixture MMS-synthétic — cả fp32
  lẫn int8 đều tệ trên 2/4 mẫu (xem dưới)

## Quy trình tái lập (thứ tự bắt buộc)

```powershell
# venv phải pin onnxruntime==1.22.0 (khớp app) — script tự kiểm tra version
<venv>\Scripts\python optimize\06_quantize_whisper.py        # decoder int8 (nếu re-export)
<venv>\Scripts\python optimize\05_slim_decoder.py            # NLLB decoder (nếu re-download)
<venv>\Scripts\python optimize\08_quantize_whisper_encoder.py  # encoder int8 MatMul-only
<venv>\Scripts\python optimize\07_preoptimize.py             # *.opt.onnx (3 model, đã loại NLLB dec)
<venv>\Scripts\python -m pytest tests_local\test_06_onnx_parity.py -v   # GATE — phải pass
```

Java tự ưu tiên load `*.opt.onnx` (NO_OPT) và tự fallback về file gốc (ALL_OPT).

## Điều kiện cần xác minh trên device (trước demo/thi)

1. **Encoder int8 với giọng VIỆT THẬT** (micro, không phải TTS): fixtures MMS
   synthetic không kết luận được — 2/4 mẫu loop ở CẢ fp32 lẫn int8. Rollback
   1 lệnh nếu cần:
   `copy onnx_models\whisper_encoder_fp32.onnx android\app\src\main\assets\whisper_encoder.onnx`
   (rồi xóa `whisper_encoder.opt.onnx` và chạy lại gate).
2. Đo PSS + timing qua logcat:
   `adb logcat -s PipelineOrchestrator OrtSessionConfig TranslationModule ASRModule`
   — các dòng `MEM[stage] totalPss=…MB` và stage timing dùng để so A/B
   (XNNPACK: bật `OrtSessionConfig.USE_XNNPACK = true` rồi đo lại).
3. Log `Loading pre-optimized *.opt.onnx` phải hiện cho encoder/decoder
   Whisper + NLLB encoder (nếu không: file .opt thiếu trong APK).

## Quyết định kỹ thuật đáng chú ý (và lý do)

- **Whisper decode = whole-sequence**: asset `whisper_decoder.onnx` hiện là bản
  export **cache-less** (không past inputs, không use_cache_branch — bản export
  mới của optimum không sinh biến thể merged). KV-cache không khả thi; token
  budget 30 token/giây giấy giữ chi phí O(n²) có giới hạn.
- **NLLB decode = KV-cache zero-copy, encoder-KV giữ từ prefill**: then-branch
  của graph int8 xuất encoder-presents hỏng shape `(0,16,1,64)` — phải re-feed
  presents của prefill mỗi bước (pattern cache-initializer của RTranslator).
- **Pre-opt bị loại cho NLLB decoder**: optimization (kể cả BASIC) inline hai
  nhánh If làm file phình gấp đôi.
- **ConvInteger**: quantize full encoder sinh node ConvInteger không có kernel
  trên ORT 1.22 → recipe chỉ quantize MatMul (vẫn giữ ~99% khối lượng trọng số).

## Các file mới/sửa chính

- `util/OrtSessionConfig.java` — session options theo RAM, flag NNAPI/XNNPACK,
  NO_OPT cho file pre-opt
- `pipeline/ASRModule.java` — mel native (whisper_preprocess.onnx) + fallback
  DSP đã bỏ pad 30s, whole-seq decode, token budget, chuẩn hóa transcript
- `pipeline/TranslationModule.java` — KV-cache zero-copy + fallback, terminator,
  runaway guard, load `.opt.onnx`
- `pipeline/TTSModule.java` — lazy session + LRU-1
- `pipeline/PipelineOrchestrator.java` — cancel-between-stages, PSS logging
- `TranslationActivity.java` — single-thread executor + latest-wins
- `tests_local/gen_parity_reference.py`, `test_06_onnx_parity.py`,
  `verify_preprocess_onnx.py`, `diagnose_whisper_paths.py` — bộ gate/chẩn đoán
- `optimize/07_preoptimize.py`, `optimize/08_quantize_whisper_encoder.py`

## ASR streaming hóa — KV-cache decoder (2026-08-17, session 2)

Mục tiêu: ASR nhanh như streaming model. Hai nút thắt cũ: (1) decoder bản export
cache-less bắt buộc decode whole-sequence O(n²) — mỗi token sinh ra phải re-feed
toàn bộ chuỗi; (2) mỗi bước decode materialize toàn bộ logits [seq × 51865]
sang mảng Java (~6 MB/step).

### Decoder KV-cache mới (`optimize/09_export_whisper_decoder_kv.py`)

- Tự export từ checkpoint `openai/whisper-small` (HF cache local) bằng
  `torch.onnx.export`: **một graph uniform, không If/use_cache_branch** — prefill
  (past rỗng, L=0) và từng step (L>0) cùng một đường concat, nên pre-opt
  (07) không bị phình file như NLLB decoder (196.3 → 196.1 MB).
- Contract: vào `input_ids`, `encoder_hidden_states` (**E=0 ở các step** — cross
  K/V lấy từ encoder pasts), 48 tensor `past_key_values.{i}.decoder/encoder.*`;
  ra `logits` + `present.{i}.decoder.*` **cộng dồn** (zero-copy cycle giữa các
  step) + `present.{i}.encoder.*` **chỉ phần chiếu mới** (prefill fill 1500,
  step trả 0 — app giữ prefill Result mở và re-feed, pattern cache-initializer
  giống TranslationModule/NLLB).
- Cross-attention tính dạng split (`q@past_ek^T ++ q@k_new^T`; out =
  `p_past@ev + p_new@v_new`) nên một step **không bao giờ copy khối encoder KV
  ~55 MB** — chỉ concat score nhỏ.
- Validated: fp32 logits khớp eager HF < 1e-3 ở CẢ prefill lẫn step có cache;
  quantize int8 theo đúng recipe 06 (per-channel QInt8).
- Gate trên ort==1.22.0 (app runtime): `test_06 -k whisper` **PASS 2/2**;
  WER cải thiện: en 0.333 → **0.222** (đúng bằng torch ref), vi 0.75 → **0.625**.
- Benchmark decode (desktop, ort 1.22, fixture ~3 s): **en 7.0 s → 1.2 s (5.8×),
  vi 4.2 s → 0.6 s (6.7×)**. Trên device tỉ lệ giữ nguyên (tỉ lệ compute),
  tuyệt đối chậm hơn desktop.

### Java (`ASRModule.java`)

- `decodeWithCache`: greedy KV zero-copy — prefill [SOT, lang, TRANSCRIBE,
  NO_TIMESTAMPS] → mỗi step 1 token + `encoder_hidden_states` rỗng (1×0×768);
  Result trước chỉ đóng SAU khi run kế consumed presents; fallback whole-sequence
  nếu runtime từ chối (asset cũ cache-less vẫn chạy đúng).
- `argmaxLastRow`: đọc đúng hàng logits cuối qua FloatBuffer — bỏ materialize
  mảng Java `[seq][51865]` mỗi step (áp cho cả fallback path).

### Tái lập / rollback (decoder)

```powershell
python optimize/09_export_whisper_decoder_kv.py        # export + validate + install int8
.venv-ort122\Scripts\python optimize/07_preoptimize.py --models whisper_decoder.onnx
.venv-ort122\Scripts\python -m pytest tests_local\test_06_onnx_parity.py -k whisper -v
```

Rollback 1 lệnh về graph cũ (cache-less):
`copy onnx_models\preopt_backup\whisper_decoder.opt.cacheless.onnx android\app\src\main\assets\whisper_decoder.opt.onnx`
(xóa `whisper_decoder.onnx` nếu còn). Note: `*.onnx` không track trong git —
asset phải regenerate bằng script khi checkout máy khác.

### Encoder dynamic-length (`optimize/10_export_whisper_encoder_dyn.py`) — ĐÃ INSTALL

- Tái export encoder từ cùng checkpoint với **frame axis động** (bỏ check cứng
  3000 của HF bằng wrapper gọi thẳng conv1/conv2 + slice bảng vị trí +
  layer stock). Quantize đúng recipe 08 (MatMul-only int8, per-channel) →
  **98.5 MB**, pre-opt `.opt.onnx` 98.4 MB (1 input/1 output, load NO_OPT OK).
- Validate fp32: frames=3000 khớp eager HF Δ 4.2e-04; frames 512/97 shape/finite OK
  (eager HF không nhận input ≠ 3000 nên không có reference số ở độ dài ngắn).
- **Java** (`ASRModule`): detect frame axis động lúc init (`TensorInfo` dim < 0);
  `sliceShortWindow` copy đúng cột mel thật (bỏ đuôi pad hằng số) → clip 3 s
  chỉ trả ~300 frame thay vì 3000 (~10× ít compute encoder). Graph cũ fixed-3000
  vẫn chạy đúng (không slice).
- Gate whisper trên asset cuối: **PASS 2/2**; transcript en giờ **khớp tuyệt đối
  PyTorch reference** ("I would like to renew my citizen-nid card.", WER 0.222 =
  torch); vi 0.625 (fixture MMS-synthetic vốn tệ — same pattern as fp32).
- Harness `whisper_transcribe` mirror cả slice short-window.

### Benchmark tổng ASR (desktop, ort 1.22.0, fixture ~3 s)

| Hạng mục | Cũ (fixed enc + whole-seq) | Mới (dynamic enc + KV) | Tốc độ |
|---|---|---|---|
| en encoder | 1865 ms | 173 ms | 10.8× |
| en decode | 3650 ms | 361 ms | 10.1× |
| **en tổng** | **5515 ms** | **533 ms** | **10.3×** |
| vi encoder | 1623 ms | 215 ms | 7.6× |
| vi decode | 3536 ms | 354 ms | 10.0× |
| **vi tổng** | **5160 ms** | **570 ms** | **9.1×** |

Trên device tỉ lệ giữ nguyên (cùng graph cùng EP), tuyệt đối chậm hơn desktop.

### Tái lập / rollback (encoder)

```powershell
python optimize/10_export_whisper_encoder_dyn.py   # export + validate + install int8
.venv-ort122\Scripts\python optimize\07_preoptimize.py --models whisper_encoder.onnx
.venv-ort122\Scripts\python -m pytest tests_local\test_06_onnx_parity.py -k whisper -v
```

Rollback về fixed-3000:
`copy onnx_models\preopt_backup\whisper_encoder.opt.onnx android\app\src\main\assets\whisper_encoder.opt.onnx`

### Trạng thái gate

Full gate trên asset cuối (encoder dynamic + decoder KV, ort 1.22.0):
**8/8 PASS** (6 hướng NLLB + whisper en/vi) — `tests_local/output/onnx_parity_results.json`.
