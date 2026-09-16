# Chẩn đoán: bootstrap LID luôn kết luận "tiếng Việt" (VoxLingua107 ECAPA)

Ngày: 2026-09-16 · Branch: `fakedemo2` · **Trạng thái: ĐÃ SỬA — Fix 1+2+3 landed (Java + Python mirror + tests)**
Log nguồn: phiên chạy thật `com.omnivoice.onspeak47`, 16:14:45 – 16:15:25.

> Cập nhật sau chẩn đoán: toàn bộ 3 hướng sửa bên dưới đã được triển khai —
> cửa sổ LID tăng dần, gate absolute-mass (0.45) thay argmax-ghim,
> provisional evidence-steered + referee mode + candidate bar 0.65/0.10,
> serialize engine access (GetFrames race). Chi tiết xem `docs/streaming_asr.md`
> ("2026-09-16" notes). Cần retest trên máy thật + validate FBank/ONNX parity
> trước khi đánh giá accuracy.
>
> ## V2 — Field log 22:10–22:11 (APK đã chứa fix v1): abs bar 0.45 quá cao
>
> Log device cho thấy inference VoxLingua trả `en-rel 0.77–0.87` (ranking ĐÚNG
> cho audio tiếng Anh) nhưng `enAbs chỉ 0.010–0.022` — thấp hơn 20–40 lần so
> với bar 0.45 vốn tune trên 5 sample sạch (0.12–0.98). Hệ quả: gate không bao
> giờ mở trên audio mic thật (xa, ồn, từ ngắn + silence) → UNKNOWN triền miên
> → candidate one-shot decode rỗng → final rỗng. Log còn cho thấy single-window
> rel cũng có thể sai (`adopted provisional vi conf=0.79` trên audio EN sau ~4s),
> nên không thể chỉ hạ bar — cần persistence.
>
> Fix v2 (trong tree):
>
> - Gate 2 tầng: FAST (abs ≥ 0.45, audio sạch) + SLOW (relative 0.70/0.15 trên
>   EMA + unanimity 3 hops + foreign-guard: argmax-unsupported ≥ 0.60 mới block).
> - Confidence bootstrap = relative top (bỏ fusion-compression 0.90/0.10 làm
>   router re-reject kết quả near-bar).
> - Window onset-anchored (hết silence-dilution của lastMs); live MAX 2500→1500
>   (ECAPA rẻ hơn, slow path commit từ window 600–1000 ms), endpoint giữ 2500.
> - Validate trên model + sample thật: EN commit 800 ms, VI 1000 ms (rolling
>   onset-anchored, smoother thật).
>
> Số đo E2E trong log (`Pipeline total: 2661 ms = Translation 737 ms + TTS
> 1787 ms`, endpoint silence 2000 ms) cho thấy sau khi ASR commit sớm trở lại,
> dư địa latency lớn nhất nằm ở MT/TTS + endpoint — xem phần trả lời latency.

## TL;DR

Không phải model LID mới bị lỗi. Model ONNX + FBank frontend + bảng nhãn đều **đúng**
(đo lại bên dưới: câu EN 3 s → `en 0.9945`, câu VI 3 s → `vi 1.000`).

Ba lỗi phối hợp tạo ra "mặc định tiếng Việt":

1. **Cửa sổ LID luôn là 600 ms** — `tryBootstrapLid()` gọi
   `ring.lastMs(BOOTSTRAP_LID_WINDOW_MS)` và không bao giờ mở rộng, dù
   `BOOTSTRAP_MAX_MS` được ghi chú là "extended window". Ở 600 ms, argmax 107 lớp của
   ECAPA gần như luôn là một ngôn ngữ khác (`eu`/`br`/`cy`/`gn`/`ja`/`nn`/`lo`), dù
   phân bố trong 3 ngôn ngữ hỗ trợ vẫn rất rõ (`en` 0.93–1.00).
2. **Gate dùng sai đại lượng**: đòi argmax toàn cục phải là `vi/en/zh` **và**
   `globalTopScore ≥ 0.25`; xác suất tuyệt đối của vi/en/zh bị bỏ ngay trong
   `fromLogits()`; thêm nữa smoother giữ `globalTop` theo kiểu "điểm cao nhất từng
   thấy" nên **một cửa sổ rác điểm cao sẽ khoá gate cho hết utterance**.
3. **Đường dự phòng nghiêng về VI**: engine provisional hard-code là VI, tie-break
   `topTwo()` trả `[VI, EN]` khi điểm bằng nhau, candidate chỉ cần ≥ 0.30 là commit
   (so với 0.70 của EN/ZH). Log cho thấy `adopted provisional vi conf=0.32391822`.

Kết quả: tiếng Anh/Trung không bao giờ qua được bootstrap; tiếng Việt qua bình
thường (hoặc qua đường candidate 0.30) → ứng dụng "mặc định" nhận là VI.

## Phương pháp kiểm chứng

Mọi số liệu dưới đây chạy **trực tiếp trên asset của app**, không phải mô phỏng:

| Thành phần | Giá trị |
|---|---|
| Model | `android/app/src/main/assets/voxlingua_lid_ecapa.onnx` (85.4 MB) |
| Nhãn | `android/app/src/main/assets/voxlingua_lid_labels.json` (107 lớp, list `{language, label}`) |
| Frontend | `backend/streaming_asr/voxlingua.py::VoxLinguaFbankExtractor` (cùng công thức với `VoxLinguaFbankExtractor.java`) |
| Runtime | `.venv-ort122` — onnxruntime 1.22.0, numpy 2.2.6 (khớp app) |
| Audio | `tests_local/data/audio_samples/parity_en.wav` (3.04 s), `parity_vi.wav` (3.38 s), `parity_vi2/3/4.wav` |

Input graph: `features [1,T,60]`, `wav_lens [1] = 1.0`; output `probabilities [1,107]`
(softmax đã nằm trong graph, không softmax lần hai). Gate được mô phỏng đúng như Android:

```
isSupportedTop(argmax) AND argmaxScore >= VOXLINGUA_MIN_GLOBAL_SCORE (0.25)
AND relTop >= BOOTSTRAP_THRESHOLD (0.70) AND (relTop - relSecond) >= BOOTSTRAP_MARGIN (0.15)
```

## Bước 1 — Model / FBank / nhãn có đúng không?

| Sample | Cả câu | top-1 (107 lớp) |
|---|---|---|
| `parity_en.wav` | 3.04 s | `en 0.9945`, `cy 0.0036`, `sco 0.0012` |
| `parity_vi.wav` | 3.38 s | `vi 1.000` |

→ Frontend + graph + bảng nhãn **đúng**; index 20 = `en`, 102 = `vi`, 106 = `zh`,
69 = `nn`, 55 = `lo` khớp `labels.json`. Vấn đề nằm ở phía pipeline Android.

## Bước 2 — Cùng model đó, ở độ dài cửa sổ mà app đang dùng

### Cắt từ đầu câu (mô phỏng `ring.lastMs(600)` tại thời điểm bootstrap)

`parity_en.wav` — tiếng Anh:

| Cửa sổ | absTop (107 lớp) | relTop (3 ngôn ngữ) | Gate hiện tại |
|---|---|---|---|
| 400 ms | `eu 0.476` | `en 0.933` | UND (bị từ chối) |
| 600 ms | `br 0.282` | `en 0.997` | UND (bị từ chối) |
| 800 ms | `br 0.414` | `en 0.992` | UND (bị từ chối) |
| 1000 ms | `cy 0.490` | `en 1.000` | UND (bị từ chối) |
| 1500 ms | `en 0.780` | `en 1.000` | **COMMIT-EN** |
| 2000 ms | `en 0.566` | `en 1.000` | COMMIT-EN |
| 3000 ms | `en 0.994` | `en 1.000` | COMMIT-EN |

`parity_vi.wav` — tiếng Việt (đối chứng):

| Cửa sổ | absTop (107 lớp) | relTop (3 ngôn ngữ) | Gate hiện tại |
|---|---|---|---|
| 400 ms | `nn 0.085` | `en 0.889` | UND (bị từ chối) |
| 600 ms | `ja 0.631` | `vi 0.993` | UND (bị từ chối) |
| 800 ms | `gn 0.268` | `vi 0.977` | UND (bị từ chối) |
| 1000 ms | `vi 0.975` | `vi 0.999` | **COMMIT-VI** |
| 1500 ms | `vi 0.964` | `vi 1.000` | COMMIT-VI |
| 3000 ms | `vi 1.000` | `vi 1.000` | COMMIT-VI |

**Đọc bảng này:** tiếng Anh cần ~1.5 s mới có argmax `en`; tiếng Việt cần ~1.0 s.
Cửa sổ 600 ms nằm **dưới ngưỡng đó cho cả hai** → gate đóng với EN/ZH, và mở khi VI
tình cờ đạt argmax `vi`. Đó chính là cơ chế "app luôn nhận là tiếng Việt".

Các sample VI khác (xác nhận lại cùng quy luật):

| Sample | 400 ms | 600 ms | 800 ms | 1000 ms+ |
|---|---|---|---|---|
| `parity_vi2.wav` | `mg 0.154`, rel `en 0.843` → UND | `vi 0.880` → COMMIT-VI | `vi 0.817` → COMMIT-VI | COMMIT-VI |
| `parity_vi3.wav` | `nn 0.224`, rel `en 0.712` → UND | `th 0.917`, rel `vi 0.886` → UND | `th 0.444` → UND | `vi 0.998` @1500 ms → COMMIT-VI |
| `parity_vi4.wav` | `sd 0.175`, rel `en 0.952` → UND | `la 0.733`, rel `vi 0.552` → UND | `vi 0.970` → COMMIT-VI | COMMIT-VI |

### Cửa sổ rolling 600 ms / hop 200 ms (đúng như pipeline chạy thật)

`parity_en.wav` — 12 cửa sổ, chỉ 4 cửa sổ qua được gate:

| Cửa sổ | absTop | relTop | Gate |
|---|---|---|---|
| 0.0–0.6 s | `br 0.282` | `en 0.997` | UND |
| 0.2–0.8 s | `br 0.473` | `en 0.999` | UND |
| 0.4–1.0 s | `en 0.465` | `en 1.000` | COMMIT-EN |
| 0.6–1.2 s | `en 0.982` | `en 0.999` | COMMIT-EN |
| 0.8–1.4 s | `my 0.905` | `en 0.623` | UND |
| 1.0–1.6 s | `pt 0.435` | `en 0.998` | UND |
| 1.2–1.8 s | `ps 0.568` | **`vi 0.956`** | UND |
| 1.4–2.0 s | `af 0.890` | `en 0.999` | UND |
| 1.6–2.2 s | `es 0.477` | `en 0.998` | UND |
| 1.8–2.4 s | `is 0.581` | `en 1.000` | UND |
| 2.0–2.6 s | `sq 0.662` | `en 0.999` | UND |
| 2.2–2.8 s | `en 0.694` | `en 0.997` | COMMIT-EN |
| 2.4–3.0 s | `en 0.710` | `en 1.000` | COMMIT-EN |

`parity_vi.wav` — cửa sổ rolling vừa **đóng gate sai** vừa **mở gate sai**:

| Cửa sổ | absTop | relTop | Gate |
|---|---|---|---|
| 0.6–1.2 s | `sw 1.000` | `vi 0.998` | UND |
| 1.4–2.0 s | `zh 0.545` | `zh 0.986` | **COMMIT-ZH (sai!)** |
| 1.6–2.2 s | `vi 0.979` | `vi 0.998` | COMMIT-VI |
| 1.8–2.4 s | `jw 0.967` | `vi 0.993` | UND |

→ Ở 600 ms, quyết định LID **sai cả hai chiều**, kể cả khi chỉ dùng margin tương đối.
Một cửa sổ 600 ms của câu EN còn cho `relTop = vi 0.956`, và một cửa sổ của câu VI cho
`relTop = zh 0.986` (đủ để commit ZH sai).

## Bước 3 — Đối chiếu log của bạn với kết quả tái hiện

| Dòng log | Giải thích |
|---|---|
| `voxlingua inference #1: LanguageScores{vi=0.07991563 en=0.9200461 zh=3.828411E-5 globalTop=lo(55) globalScore=0.4743643}` | EN ở ~400 ms: argmax `lo 0.474` (tái hiện: `eu 0.476`) → gate đóng vì argmax không thuộc vi/en/zh |
| `voxlingua inference #1: LanguageScores{vi=0.12566283 en=0.82173944 zh=0.052597716 globalTop=nn(69) globalScore=0.2339771}` | Tái hiện đúng cả lớp `nn` ở 400 ms (`nn 0.085`, rel `en 0.889`) |
| `speculative candidates inconclusive, staying UNKNOWN` (lặp) | Sau khi gate trả `undFlat()`, điểm 3 ngôn ngữ = 1/3 → candidate không có bằng chứng nào để chọn |
| `provisional lang=vi conf=0.29351592 text=O` / `conf=0.47141263 text=HELLO` | Partial hiển thị do **engine VI** sinh ra (`models.get(AsrLanguage.VI)`) |
| `speculative candidates picked vi score=0.32391822` → `bootstrap adopted provisional vi conf=0.32391822` | Đường candidate commit VI với 0.32 ≪ 0.70 |
| `decode lang=en conf=0.14918469 text=` (lặp >30 lần) rồi `endpoint verification switched en → vi` | Utterance tiếng Việt bị LID nhận thành `en 0.787` → engine EN decode rỗng → endpoint lật lại VI |
| `Playing: translated_en_to_vi.wav` / `translated_vi_to_vi.wav` | Hệ quả: source được nhận là `en` hoặc chính `vi` → có lần dịch VI→VI, TTS đọc tiếng Việt |

## Nguyên nhân gốc

### 1. Cửa sổ LID không bao giờ được mở rộng

`android/app/src/main/java/com/omnivoice/onspeak47/asr/StreamingPipeline.java`

```java
483  private void tryBootstrapLid(long nowMs) {
484      if (activeLang != AsrLanguage.UND || bootstrapPending) return;   // ← chỉ 1 lần/utterance
485      if (utteranceSpeechMs < AsrState.BOOTSTRAP_MIN_MS) return;
...
494      bootstrapPending = true;
495      lastBootstrapSpeechMs = utteranceSpeechMs;
496      final float[] window = ring.lastMs(AsrState.BOOTSTRAP_LID_WINDOW_MS);  // ← luôn 600 ms
```

`onBootstrapLidResult()` (dòng 536-545) chỉ cho phép **thử thêm ở cùng 600 ms**;
`BOOTSTRAP_MAX_MS` thực chất chỉ là mốc thời gian (`mayExtend`) và là độ dài cửa sổ cho
**ASR** một-shot ở `runSpeculativeCandidates()` (dòng 668), **không phải** cho LID.
Đường endpoint cũng cắt còn 600 ms (dòng 1022-1027). Kết luận: ECAPA **chưa bao giờ
được xem quá 600 ms audio**.

Các hằng số liên quan (`android/.../asr/AsrState.java`):

| Hằng số | Dòng | Giá trị |
|---|---|---|
| `BOOTSTRAP_MIN_MS` | 64 | 400 |
| `BOOTSTRAP_LID_WINDOW_MS` | 66 | 600 |
| `BOOTSTRAP_HOP_MS` | 68 | 200 |
| `BOOTSTRAP_MAX_MS` | 70 | 1000 (ghi chú: "Extended window khi uncertain") |
| `BOOTSTRAP_GIVE_UP_MS` | 76 | 3000 |
| `BOOTSTRAP_THRESHOLD` / `BOOTSTRAP_MARGIN` | 80 / 82 | 0.70 / 0.15 |
| `VOXLINGUA_MIN_GLOBAL_SCORE` | 189 | 0.25 |
| `SPECULATIVE_CANDIDATE_COOLDOWN_MS` | 78 | 800 |

### 2. Gate + smoother dùng argmax thay vì phân bố

`android/.../asr/LanguageIdEngine.java` — `classifyBootstrap()` (dòng 209-234):

```java
211  LanguageScores detailed = lidEngine.classifyDetailed(audioWindow);
212  if (detailed != null && detailed.numWindows > 0) {
213      if (!detailed.isSupportedTop()) { return undFlat(); }               // argmax phải là vi/en/zh
216      if (detailed.globalTopScore < AsrState.VOXLINGUA_MIN_GLOBAL_SCORE)  // điểm CỦA ARGMAX
217              { return undFlat(); }
```

`android/.../asr/VoxLinguaAcousticLidEngine.java` — `fromLogits()` (dòng 287-301) chỉ
giữ vi/en/zh **đã renormalize trong 3 lớp**, **bỏ xác suất tuyệt đối**; vì vậy không
caller nào có thể kiểm tra "độ trội tuyệt đối" của ngôn ngữ hỗ trợ.
(`LanguageScores.isBootstrapConfident()`, dòng 86-92, dùng cùng điều kiện.)

`android/.../asr/VoxLinguaTemporalSmoother.java` — `add()` (dòng 61-65) giữ `globalTop`
theo kiểu **điểm cao nhất từng thấy**:

```java
61  if (raw.numWindows > 0 && raw.globalTopScore >= ema.globalTopScore) {
62      gTop = raw.globalTopLanguage;
63      gIdx = raw.globalTopIndex;
64      gScore = raw.globalTopScore;
```

Trong log thật, cửa sổ đầu cho `lo 0.474`; các cửa sổ sau thấp hơn (`nn 0.234`) nên
`globalTop` **giữ nguyên `lo`** → `isSupportedTop()` luôn `false` suốt utterance. Đây là
lý do LID không bao giờ tự sửa: nó không chỉ "không chắc", mà bị **khoá** ở trạng thái UND.

### 3. Đường dự phòng mặc định nghiêng VI

| Vị trí (`StreamingPipeline.java`) | Nội dung |
|---|---|
| dòng 96, 1141 | `provisionalLang = AsrLanguage.VI` hard-code |
| dòng 414 | `provisionalAsr = models.get(AsrLanguage.VI)` — provisional luôn là engine VI |
| dòng 761-782 | `topTwo()` khởi tạo `first = VI`; với 3 điểm bằng nhau (map flat sau khi LID bị từ chối) tie-break trả `[VI, EN]` |
| dòng 748-751 | `winnerScore >= 0.30f` là đủ để commit, với `Math.min(0.69f, winnerScore)` |
| dòng 484 | Đã commit → không bao giờ gọi LID lại trong utterance (log: 1 inference/utterance) |

Nói cách khác: **ngưỡng hiệu dụng của tiếng Việt là 0.30, còn ngưỡng dành cho EN/ZH là
một điều kiện gần như không thể đạt ở cửa sổ 600 ms.**

### Chuỗi sự kiện khiến log kết thúc bằng "tiếng Việt"

1. User nói tiếng Anh; ECAPA trên 600 ms trả `en 0.92` (tương đối) nhưng argmax `lo` →
   `classifyBootstrap()` → `undFlat()` → `speculative candidates inconclusive`.
2. Provisional (engine VI) vẫn chạy và hiển thị `HELLO`, `A`, `O`, `ĐI HỌC`.
3. Candidate pass lấy cặp `[VI, EN]` (tie-break), VI thắng với
   `0.65*conf + 0.35*lexicalFit = 0.32` → `activateBootstrap(VI, 0.32)`.
4. Từ đây ECAPA im lặng; chỉ còn endpoint verify (một-shot + lexical fit) có thể lật, và
   trong log có lần nó lật **EN → VI**.
5. Kết quả: `Translation: "Xin chào."`, `Playing: translated_vi_to_vi.wav`.

## Hướng sửa đề xuất (chưa áp dụng)

### Fix 1 — Mở rộng cửa sổ LID thật (ưu tiên, rủi ro thấp, không đổi policy)

- `AsrState.java`: `BOOTSTRAP_LID_WINDOW_MS` 600 → 1500; `BOOTSTRAP_MAX_MS` 1000 → 2500;
  `BOOTSTRAP_GIVE_UP_MS` 3000 → 6000.
- `StreamingPipeline.tryBootstrapLid()`: thay cửa sổ cố định bằng cửa sổ tăng dần, ví dụ
  `ring.lastMs(Math.min(AsrState.BOOTSTRAP_MAX_MS,
  Math.max(AsrState.BOOTSTRAP_LID_WINDOW_MS, (int) utteranceSpeechMs)))`.
- Theo số đo: EN đạt `en 0.780` ở 1500 ms và VI đạt `vi 0.975` ở 1000 ms → **gate hiện tại
  pass mà không cần sửa ngưỡng**. Cần đo lại trên máy thật (audio thiết bị khác sample repo).
- Đánh đổi: partial "đúng ngôn ngữ" đầu tiên muộn hơn (~1.5 s); trong lúc chờ vẫn có
  provisional stream — nên xử lý Fix 3 để stream đó không "trông như tiếng Việt".

### Fix 2 — Bỏ kiểu "argmax ghim" (đúng bản chất, cần sửa test)

- `LanguageScores`: thêm `viAbs/enAbs/zhAbs` (xác suất tuyệt đối) do `fromLogits()` điền vào.
- `VoxLinguaTemporalSmoother`: bỏ logic "giữ điểm cao nhất"; suy ra argmax/độ trội từ **phân
  bố đã làm mượt** (dùng abs đã EMA).
- Gate: chấp nhận khi `argmax == topSupported` **hoặc** `topSupportedAbs >= K * argmaxScore`
  (K ≈ 0.5), vẫn giữ margin 0.70/0.15. Vẫn đúng policy §14: audio `ja` cho `ja 0.80` vs
  `zh 0.05` → `0.05 < 0.5*0.80` nên không bị ép sang ZH; nhưng bằng chứng `en 0.99` tương
  đối ở cửa sổ ngắn sẽ không còn bị vứt bỏ.

### Fix 3 — Bỏ "mặc định VI"

- Provisional engine chọn theo candidate hiện tại (hoặc luân phiên) thay vì hard-code VI.
- `topTwo()`: tie-break xác định nhưng **không** ưu tiên VI khi điểm bằng nhau.
- Candidate: yêu cầu winner ≥ 0.69 (bằng mức cap) hoặc chỉ commit ở endpoint; thêm điều kiện
  winner phải vượt candidate thứ hai một margin.
- Cho ECAPA chạy lại như trọng tài định kỳ (ví dụ mỗi 800 ms) khi đã commit với
  confidence < 0.70, thay vì dừng hẳn ở `tryBootstrapLid()` dòng 484.

## Ảnh hưởng tới test khi sửa gate

| File | Nội dung cần xem lại |
|---|---|
| `android/app/src/test/java/com/omnivoice/onspeak47/asr/StreamingAsrUnitTest.java:490-534` | `languageIdEngine_voxLinguaDetailedIntegration` — đang assert "unsupported global top → UND và flat scores" |
| `tests_local/test_07_streaming_asr.py:761` | `test_voxlingua_scores_bootstrap_gate` (`is_supported_top` / `is_bootstrap_confident`) |
| `backend/streaming_asr/voxlingua.py:101-134` và `backend/streaming_asr/lid.py:269` | Bản Python tham chiếu có cùng gate — phải sửa song song để parity Java/Python giữ nguyên |

## Phụ lục — Lệnh tái hiện

```powershell
# PowerShell: pipe script trực tiếp vào python của venv (không tạo file)
cd d:\StudioProjects\OnSpeak47
@'
import sys, wave, numpy as np, onnxruntime as ort
sys.path.insert(0, r"d:\StudioProjects\OnSpeak47")
from backend.streaming_asr.voxlingua import VoxLinguaFbankExtractor, VoxLinguaLabels
fb = VoxLinguaFbankExtractor()
sess = ort.InferenceSession(r"android\app\src\main\assets\voxlingua_lid_ecapa.onnx")
w = wave.open(r"tests_local\data\audio_samples\parity_en.wav", "rb")
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
for ms in (400, 600, 800, 1000, 1500, 3000):
    f = fb.extract(pcm[:16000 * ms // 1000])
    p = np.maximum(sess.run(None, {"features": f[None].astype(np.float32),
                                   "wav_lens": np.array([1.0], np.float32)})[0].squeeze(), 0)
    top = int(np.argmax(p))
    sup = {c: float(p[i]) for c, i in (("en", 20), ("vi", 102), ("zh", 106))}
    s = sum(sup.values())
    print(ms, "absTop =", VoxLinguaLabels.code_at(top), round(float(p[top]), 3),
          "rel =", {c: round(v / s, 3) for c, v in sup.items()})
'@ | .venv-ort122\Scripts\python.exe -
```

Kết quả kỳ vọng (khớp bảng ở Bước 2):

```
400  absTop = eu 0.476 rel = {'en': 0.933, 'vi': 0.011, 'zh': 0.056}
600  absTop = br 0.282 rel = {'en': 0.997, 'vi': 0.0,   'zh': 0.003}
1500 absTop = en 0.78  rel = {'en': 1.0,   'vi': 0.0,   'zh': 0.0}
```

## Ghi chú

- Tài liệu này **không kèm thay đổi code**; mọi phát hiện dựa trên code tại HEAD `eb511fc` và
  asset model thật trong `android/app/src/main/assets/`.
- `__pycache__` sinh ra trong `backend/` khi chạy script kiểm chứng nằm trong `.gitignore` nên
  không ảnh hưởng repo.

