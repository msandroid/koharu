# Koharu on iOS and Android

This fork specializes Koharu for phones. Everything runs on the device: detection,
OCR, and inpainting through ONNX Runtime, and translation through either a small
local GGUF model (llama.cpp) or a hosted provider.

## Why the desktop stack does not fit

| Desktop piece | Why it cannot ship on a phone | Mobile replacement |
| --- | --- | --- |
| libtorch vision models (`koharu-ml`) | No iOS/Android libtorch builds; hundreds of MB per platform | ONNX Runtime with Core ML (iOS) and NNAPI/XNNPACK (Android) |
| RF-DETR Seg 2XL detector | 1152 px 2XL model, too heavy for phones | Comic text detector (`comic-text-detector-onnx`, 95 MB) |
| PaddleOCR-VL default OCR | Vision-language model through llama.cpp | Manga OCR (`manga-ocr-onnx`) |
| Runtime downloads from GitHub releases | Apps must bundle native libraries | Libraries bundled in the app package |
| Tauri with the CEF runtime | CEF is desktop-only | Tauri v2 mobile (system WebView) |

## Phase 1 (done): torch-free models in the existing pipeline

The pipeline gained three model choices that never touch libtorch:

| Stage | Choice | Weights |
| --- | --- | --- |
| Detection | `comic-text-detector-onnx` | `mayocream/comic-text-detector-onnx` |
| OCR | `manga-ocr-onnx` | `mayocream/manga-ocr-onnx` (+ config from `mayocream/manga-ocr`) |
| Inpainting | `lama-onnx` | `mayocream/lama-manga-onnx` |

They live in `koharu_ml::onnx` and share preprocessing, decoding, beam scoring, and
IOPaint crop orchestration with the Torch ports, so only the forward pass differs.
Comic text detector blocks are expressed as layout text detections, so the
detection stage's typography inference, scene writing, and masks are reused.

ONNX Runtime is loaded dynamically (`ort` with `load-dynamic`). Point
`ORT_DYLIB_PATH` at `libonnxruntime` or call `koharu_ml::onnx::init` with the
bundled path.

```bash
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
cargo run -p koharu-pipeline --bin run -- --cpu \
  -i page.png -o page.en.png \
  --detection comic-text-detector-onnx --ocr manga-ocr-onnx --inpainting lama-onnx \
  --llm gemma4-e2b-it
```

Measured on a 4-core x86 Linux container, CPU only, debug build, 768×1086 page:

| Stage | Torch desktop models | ONNX mobile models |
| --- | --- | --- |
| Detection | 5.3 s (RF-DETR) | 36.4 s (CTD) |
| OCR | 7.9 s | 8.3 s |
| Inpainting | 6.0 s | 20.7 s |
| Translation (gemma4-e2b-it) | 61.3 s | 69.0 s |
| Process wall time incl. model loading | 4 m 54 s | 1 m 57 s |

Detection is dominated by ONNX Runtime's x86 CPU `ConvTranspose` kernel (about 25 s
across the U-Net decoder's four 4×4 stride-2 layers). Core ML and XNNPACK implement
transposed convolution natively, so device numbers have to be measured in phase 2.
Rewriting those layers as sub-pixel convolutions is the fallback if phones without a
capable NNAPI driver stay slow.

Known differences from the desktop defaults:

- The comic text detector finds no bubbles or panels, so text is not linked to a
  balloon shape and layout falls back to the text region.
- Very tall webtoon strips are letterboxed whole; the Torch port's rearranged
  inference for strips is not ported yet.
- Page furniture such as copyright lines is detected as text.

## Phase 2: mobile packaging

1. **Native libraries per target.** Build or fetch ONNX Runtime (iOS xcframework,
   Android AAR `jni/arm64-v8a/libonnxruntime.so`) and llama.cpp for
   `aarch64-apple-ios` and `aarch64-linux-android`; teach `koharu-runtime` to resolve
   bundled libraries instead of downloading them.
2. **Runtime discovery.** Skip the Torch and diffusion features on mobile so
   `koharu_ml::init` never tries to download them; split the llama.cpp backend and
   `llm` module out of `koharu-ml` so translation does not pull in Torch bindings.
3. **Tauri v2 mobile shell.** A mobile app crate that drops `tauri-runtime-cef`,
   desktop-only plugins (updater, window state, single instance), and exposes the
   pipeline commands. Generate the Xcode and Android Studio projects with
   `tauri ios init` / `tauri android init`.
4. **Model delivery.** Download the ONNX set (about 760 MB at fp32) on first launch
   with progress; evaluate the int8/fp16 Manga OCR exports to cut it roughly in half.

## Phase 3: mobile UI

A phone-first frontend: pick or capture pages, run the full pipeline per page,
review and edit translations in a sheet, and export or share the result. The
desktop canvas editor stays desktop-only.
