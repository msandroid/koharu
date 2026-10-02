#!/usr/bin/env bash
# Builds the dynamic frameworks the iOS app loads at runtime:
#   llama.xcframework        llama.cpp, ggml and mtmd in one dylib
#   onnxruntime.xcframework  ONNX Runtime (relinked as a dylib when the
#                            official package ships a static framework)
#
# Usage: scripts/mobile/ios-native.sh <output directory>
# Runs on macOS with Xcode, cmake and ninja.
set -euo pipefail

OUT=${1:?usage: ios-native.sh <output directory>}
LLAMA_RELEASE=${LLAMA_RELEASE:-b11317}
ORT_VERSION=${ORT_VERSION:-1.30.0}
IOS_MIN=${IOS_MIN:-16.0}
WORK=${WORK:-$(mktemp -d)}
SDK=$(xcrun --sdk iphoneos --show-sdk-path)

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# Wraps a dylib in a device framework and then an xcframework, which is what
# Tauri's iOS project embeds under Frameworks/.
make_framework() {
  local name=$1 dylib=$2 bundle_id=$3
  local framework="$WORK/frameworks/$name.framework"
  rm -rf "$framework" "$OUT/$name.xcframework"
  mkdir -p "$framework"
  cp "$dylib" "$framework/$name"
  install_name_tool -id "@rpath/$name.framework/$name" "$framework/$name"
  cat > "$framework/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>$name</string>
  <key>CFBundleIdentifier</key><string>$bundle_id</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
  <key>MinimumOSVersion</key><string>$IOS_MIN</string>
</dict>
</plist>
PLIST
  xcodebuild -create-xcframework -framework "$framework" -output "$OUT/$name.xcframework"
}

link_dylib() {
  local output=$1
  shift
  xcrun --sdk iphoneos clang++ -dynamiclib -arch arm64 -isysroot "$SDK" \
    -miphoneos-version-min="$IOS_MIN" -o "$output" "$@"
}

# llama.cpp: static libraries linked into one dylib, so the app's loader finds
# the llama, ggml and mtmd symbols in llama.framework.
if [ ! -d "$WORK/llama.cpp" ]; then
  git clone --depth 1 --branch "$LLAMA_RELEASE" https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
fi
cmake -S "$WORK/llama.cpp" -B "$WORK/llama-ios" -G Ninja \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_BLAS=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build "$WORK/llama-ios" --target llama mtmd
LLAMA_ARCHIVES=()
while IFS= read -r archive; do LLAMA_ARCHIVES+=("$archive"); done \
  < <(find "$WORK/llama-ios" -name 'lib*.a' \( -name 'libllama.a' -o -name 'libggml*.a' -o -name 'libmtmd.a' \))
link_dylib "$WORK/libllama.dylib" -Wl,-all_load "${LLAMA_ARCHIVES[@]}" \
  -framework Foundation -framework Metal -framework MetalKit -framework Accelerate -lc++
make_framework llama "$WORK/libllama.dylib" rs.koharu.llama

# ONNX Runtime from the official CocoaPods archive.
curl -sSfL -o "$WORK/onnxruntime-c.zip" \
  "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-$ORT_VERSION.zip"
rm -rf "$WORK/onnxruntime-c" && unzip -q "$WORK/onnxruntime-c.zip" -d "$WORK/onnxruntime-c"
ORT_BINARY=$(find "$WORK/onnxruntime-c" -path '*ios-arm64/onnxruntime.framework/onnxruntime' | head -1)
test -n "$ORT_BINARY" || { echo "no ios-arm64 onnxruntime.framework in the pod archive" >&2; exit 1; }
if file "$ORT_BINARY" | grep -q 'dynamically linked shared library'; then
  cp "$ORT_BINARY" "$WORK/libonnxruntime.dylib"
else
  link_dylib "$WORK/libonnxruntime.dylib" -Wl,-all_load "$ORT_BINARY" \
    -framework Foundation -framework CoreML -lc++
fi
make_framework onnxruntime "$WORK/libonnxruntime.dylib" rs.koharu.onnxruntime

nm -gU "$OUT/llama.xcframework/ios-arm64/llama.framework/llama" | grep -q ' _mtmd_init_from_file'
nm -gU "$OUT/onnxruntime.xcframework/ios-arm64/onnxruntime.framework/onnxruntime" | grep -q ' _OrtGetApiBase'
ls -la "$OUT"
