# Omni Voice — Architecture Overview

> **RETIRED (migrate/zipformer-only):** this document describes the original
> Whisper + NLLB prototype. Production is now streaming Zipformer ASR
> (VI/EN/ZH auto-router) → HyMT-1.5 GGUF → MMS-TTS — see `README.md` and
> `docs/streaming_asr.md`. Kept for history only.

## Pipeline Design (retired)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Omni Voice Pipeline                         │
│                                                                 │
│  ┌──────────┐    ┌───────────────┐    ┌──────────┐             │
│  │  Whisper  │    │   NLLB-200    │    │ MMS-TTS  │             │
│  │  Small    │───▶│ Distilled     │───▶│          │             │
│  │  (ASR)    │    │ 600M (NMT)   │    │ (Synth)  │             │
│  └──────────┘    └───────────────┘    └──────────┘             │
│      ▲                                     │                    │
│      │                                     ▼                    │
│  [Microphone]                        [Speaker]                  │
│                                                                 │
│  No language branching — same path for all directions           │
└─────────────────────────────────────────────────────────────────┘
```

## Model Details

### Stage 1: ASR — Whisper Small
- **Model**: `openai/whisper-small`
- **Parameters**: 244M
- **Input**: 16kHz mono audio (log-mel spectrogram)
- **Output**: Transcribed text + detected language
- **Languages**: vi, en, zh (auto-detected)
- **QAH Status**: Natively in catalog

### Stage 2: Translation — NLLB-200 Distilled 600M
- **Model**: `facebook/nllb-200-distilled-600M`
- **Parameters**: 600M (before pruning)
- **Architecture**: Encoder-decoder with KV-cache
- **ONNX files**: 4 (encoder, decoder, cache_init, embed_and_lm_head)
- **Tokenizer**: SentencePiece BPE
- **Languages**: vie_Latn, eng_Latn, zho_Hans, zho_Hant
- **QAH Status**: BYOM (Bring Your Own Model)

### Stage 3: TTS — MMS-TTS
- **Model**: `facebook/mms-tts-{vie,eng,zho}` (VITS architecture)
- **Output**: 16kHz 16-bit PCM WAV
- **Fallback**: Android system TTS (for languages where MMS-TTS is unavailable)
- **QAH Status**: BYOM

## On-Device Architecture (Android)

```
┌─────────────────────────────────────────────────────────────┐
│                    Android Application                       │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               Pipeline Orchestrator                  │    │
│  │  (chains ASR → Translation → TTS sequentially)       │    │
│  └────────┬──────────────┬──────────────┬──────────────┘    │
│           │              │              │                    │
│  ┌────────▼──────┐ ┌────▼──────────┐ ┌─▼──────────────┐   │
│  │  ASRModule    │ │ Translation   │ │  TTSModule      │   │
│  │  (Whisper)    │ │ Module (NLLB) │ │  (System TTS)   │   │
│  └───────┬───────┘ └──────┬────────┘ └────────┬────────┘   │
│          │                │                    │             │
│  ┌───────▼────────────────▼────────────────────▼───────┐   │
│  │              ONNX Runtime (android)                  │   │
│  │         onnxruntime-android:1.19.0                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ AudioRecorder│  │  AudioPlayer │  │  SentencePiece  │   │
│  │  (16kHz WAV) │  │  (MediaPlayer│  │  JNI Tokenizer  │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Strategy

### Path 1 (Primary): Qualcomm AI Hub
1. Export PyTorch → ONNX (`optimize/01_export_onnx.py`)
2. Prune NLLB vocabulary to VN/EN/CN (`optimize/02_prune_vocab.py`)
3. Quantize with AIMET or generic INT8 (`optimize/03_quantize_aimet.py`)
4. Upload to QAH Workbench (`optimize/04_qah_submit.py`)
5. Submit compile/profile jobs for target Snapdragon device
6. **ALWAYS verify execution provider** (NPU vs CPU fallback)

### Path 2 (Fallback): Non-Qualcomm
- ONNX Runtime + Android NNAPI (cross-platform by design)
- Generic INT8 quantization (PyTorch native / ONNX Runtime)
- Works on MediaTek, Exynos, Google Tensor

## VI↔ZH Quality (Risk #1)

The VN↔CN translation quality is the single biggest technical risk.
Mitigation strategies:

1. **Vocabulary pruning** — remove 197 unused languages from NLLB embeddings
2. **Pivot translation** — vi→en→zh path as quality comparison baseline
3. **Large corpus evaluation** — 1000-pair BLEU test from `test vi-zh/`
4. **Chinese preprocessing** — consistent whitespace handling for CJK text
5. **Fine-tuning potential** — 300K parallel pairs in `train vi-zh/`

## Data Assets

| Dataset | Location | Size | Purpose |
|---------|----------|------|---------|
| Parallel sentences | `tests_local/data/parallel_sentences.json` | 40 pairs | Quick BLEU scoring |
| Test VI↔ZH | `test vi-zh/train2022.{vi,zh}` | 1000 pairs | Large-corpus BLEU |
| Train VI↔ZH | `train vi-zh/train2022.{vi,zh}` | ~300K pairs | Fine-tuning potential |
