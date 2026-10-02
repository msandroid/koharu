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

| Stage | Choice | Weights | Download |
| --- | --- | --- | --- |
| Detection (text) | `comic-text-detector-onnx` | `mayocream/comic-text-detector-onnx` | 95 MB |
| Detection (bubbles) | part of `comic-text-detector-onnx` | `ogkalu/comic-text-and-bubble-detector` (`detector-v4-s_int8.onnx`) | 11 MB |
| OCR | `manga-ocr-onnx` | `onnx-community/manga-ocr-base-ONNX` int8 (+ config from `mayocream/manga-ocr`) | 117 MB |
| Inpainting | `lama-onnx` | `Liiesl/lama-manga-onnx-quant` weight-only FP16 | 109 MB |

About 332 MB in total, against 760 MB for the FP32 exports. On this page's balloon
crops the int8 Manga OCR reads the same text as FP32, and the FP16 LaMa differs from
FP32 by 0.01/255 on average inside the mask. The int8 LaMa (max error 167/255) and
the FP16 Manga OCR decoder (rejected by ONNX Runtime's type checker) were not used.

They live in `koharu_ml::onnx` and share preprocessing, decoding, beam scoring, and
IOPaint crop orchestration with the Torch ports, so only the forward pass differs.
Comic text detector blocks and bubble boxes are expressed as layout detections, so
the detection stage's typography inference, dialogue linking, scene writing, and
masks are reused:

- **Text** keeps only line groups confirmed by the detector's YOLOv5 block head,
  which drops page furniture such as copyright lines. Each block's mask is the
  refined text mask grown by 2 px inside the block, covering glyph halos.
- **Bubbles** get a mask from a flood fill of the balloon interior bounded by the
  outline and a slightly grown inscribed ellipse. Uniform balloons are then
  flat-filled instead of inpainted, as on desktop; LaMa leaves faint glyph ghosts
  when given a whole balloon column.
- **Tall strips** use the Torch port's rearranged inference, now shared with the
  ONNX detector.

ONNX Runtime is loaded dynamically (`ort` with `load-dynamic`). Point
`ORT_DYLIB_PATH` at `libonnxruntime` or call `koharu_ml::onnx::init` with the
bundled path. Sessions flush subnormal floats to zero: the comic text detector's
activations decay into subnormals and ran about 11x slower without it, with
bit-identical outputs either way.

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
| Detection | 5.3 s (RF-DETR) | 3.1 s (CTD + bubbles) |
| OCR | 7.9 s | 2.7 s |
| Inpainting | 6.0 s | 11.5 s |
| Translation (gemma4-e2b-it) | 61.3 s | 59.5 s |
| Process wall time incl. model loading | 4 m 54 s | 1 m 11 s |

Known differences from the desktop defaults:

- Panels are not detected.
- Text outside balloons that the YOLOv5 head misses is not translated.

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
4. **Model delivery.** Download the ONNX set (about 332 MB) on first launch with
   progress, and work around `hf-hub` failing on Xet-backed large files behind proxies.

## Phase 3: mobile UI

A phone-first frontend: pick or capture pages, run the full pipeline per page,
review and edit translations in a sheet, and export or share the result. The
desktop canvas editor stays desktop-only.
