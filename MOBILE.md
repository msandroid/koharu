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
| Tauri with the CEF runtime | CEF is desktop-only | Tauri mobile on the wry runtime (system WebView) |

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

## Phase 2 (done): mobile packaging

The app lives in `crates/koharu-mobile` (Tauri, system WebView) with its frontend
in `packages/mobile`.

**Runtime discovery.** On iOS and Android `koharu_ml::init` asks only for the
llama.cpp feature, so Torch and diffusion are never discovered or downloaded.
llama.cpp resolves to a `bundled` package that ships with the app and is never
installed. iOS gets a Metal device like Apple Silicon Macs.

**Native libraries.** The generated loaders look for bundled libraries where each
platform puts them: `Frameworks/<name>.framework/<name>` on iOS, and the app's
native library directory (by soname) on Android. A library that fails to load is
skipped as long as another one provides the symbols, so one iOS `llama.framework`
carrying llama, ggml, and mtmd serves both `llama` and `mtmd`.

| Target | llama.cpp (b11317) | ONNX Runtime 1.30.0 | Built by |
| --- | --- | --- | --- |
| Android arm64 | `libllama.so`, `libmtmd.so`, `libggml*.so` (NDK, shared) | `libonnxruntime.so` from the Maven AAR | `scripts/mobile/android-native.sh` |
| iOS arm64 | `llama.xcframework` (static libs relinked into one dylib, Metal embedded) | `onnxruntime.xcframework` (CocoaPods archive, relinked as a dylib) | `scripts/mobile/ios-native.sh` |

`koharu_ml::onnx::init` loads ONNX Runtime eagerly, so a missing library is an
error the app shows instead of a panic on first inference. ONNX Runtime's
telemetry is switched off with `ORT_DISABLE_TELEMETRY` before anything starts.

**App directories and secrets.** The engine points the package store at the app's
cache directory and `koharu-config` at its data directory; nothing is written
outside the sandbox. API keys go to the Keychain (iOS, data-protection keychain)
or the Android keystore through `keyring-core` stores.

**Engine and commands.** `koharu_mobile_lib::engine::Engine` owns one pipeline with
the ONNX models and exposes `status`, `prepare` (downloads the ONNX set with
progress events), `models`, `languages`, `save_settings`, and `translate`
(encoded page in, PNG out, as raw IPC bodies). Pipeline stages and downloads are
emitted as `koharu://event`.

**UI.** One screen: download the models once, choose a page, translate with live
stage chips (Find text, Read text, Translate, Clean up), then share (Web Share
API) or save the result. Settings choose the target language, the translator
(on-device GGUF model and size, or a provider with an API key).

Verified on Linux (desktop build of the same crate, WebKitGTK): the real window
was driven through settings, saving, picking a page, and translating;
the translated page came back in 63 s (debug build, CPU). Translating without
the runtime library shows the load error.

```bash
bun run --filter @koharu/mobile build
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
cargo run -p koharu-mobile --features tauri/custom-protocol
# Headless engine check without a window:
cargo run -p koharu-mobile --example translate_page -- page.jpg page.en.png cache data
```

**CI.** `.github/workflows/mobile.yml` builds the native libraries, generates the
Android Studio and Xcode projects (`tauri android init`, `tauri ios init`), and
uploads an arm64 APK signed with a throwaway key and an unsigned iOS app (zip as
`.ipa`). Signing for devices and stores needs the owner's keys.

Open items:

- Android was built end to end with the same steps as CI (NDK 27, release,
  arm64): the signed APK is 94 MB and holds the app library plus llama.cpp and
  ONNX Runtime. It has not been run on a phone yet; there is no emulator here
  (no KVM). The first real build found two problems, both fixed: the generated
  loaders needed `libc` on Android, and the release APK kept about 90 MB of
  debug symbols.
- iOS has not been built: Xcode only runs on macOS. Run the Mobile workflow on
  GitHub Actions to verify it.
- The `koharu-ml` crate still compiles the Torch bindings on mobile (they are
  loaded dynamically and never used there); splitting the llama.cpp and `llm`
  modules out would shorten mobile builds.
- `hf-hub` fails on Xet-backed files behind some HTTPS proxies; phones on normal
  networks are not affected.

## Phase 3: mobile UI

A phone-first frontend: pick or capture pages, run the full pipeline per page,
review and edit translations in a sheet, and export or share the result. The
desktop canvas editor stays desktop-only.
