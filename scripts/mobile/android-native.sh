#!/usr/bin/env bash
# Builds the native libraries the Android app loads at runtime and copies
# them into the Tauri Android project's jniLibs.
#
# Usage: scripts/mobile/android-native.sh <jniLibs/<abi> directory>
# Requires ANDROID_NDK_HOME, cmake and ninja. ANDROID_ABI picks the ABI
# (arm64-v8a for phones, x86_64 for the emulator).
set -euo pipefail

OUT=${1:?usage: android-native.sh <jniLibs/<abi> directory>}
ANDROID_ABI=${ANDROID_ABI:-arm64-v8a}
LLAMA_RELEASE=${LLAMA_RELEASE:-b11317}
ORT_VERSION=${ORT_VERSION:-1.30.0}
ANDROID_API=${ANDROID_API:-28}
WORK=${WORK:-$(mktemp -d)}
: "${ANDROID_NDK_HOME:?ANDROID_NDK_HOME must point at the Android NDK}"

mkdir -p "$OUT"

# llama.cpp as shared libraries; the app loads libllama.so and libmtmd.so by
# soname and they pull in the ggml libraries next to them.
if [ ! -d "$WORK/llama.cpp" ]; then
  git clone --depth 1 --branch "$LLAMA_RELEASE" https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
fi
cmake -S "$WORK/llama.cpp" -B "$WORK/llama-android-$ANDROID_ABI" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM="android-$ANDROID_API" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_CCACHE=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build "$WORK/llama-android-$ANDROID_ABI" --target llama mtmd
find "$WORK/llama-android-$ANDROID_ABI" -name '*.so' -exec cp {} "$OUT/" \;
# Release builds keep symbols the app never needs (about 65 MB).
STRIP=$(find "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" -name llvm-strip | head -1)
for library in "$OUT"/libllama.so "$OUT"/libmtmd.so "$OUT"/libggml*.so; do
  "$STRIP" --strip-unneeded "$library"
done

# ONNX Runtime from the official Android package; the library is self-contained.
curl -sSfL -o "$WORK/onnxruntime.aar" \
  "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VERSION/onnxruntime-android-$ORT_VERSION.aar"
unzip -o -j "$WORK/onnxruntime.aar" "jni/$ANDROID_ABI/libonnxruntime.so" -d "$OUT"

ls -la "$OUT"
for library in libllama.so libmtmd.so libonnxruntime.so; do
  test -f "$OUT/$library" || { echo "missing $library" >&2; exit 1; }
done
