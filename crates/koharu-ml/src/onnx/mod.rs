//! ONNX Runtime ports of the lightweight models used by mobile builds.
//!
//! These models share preprocessing, decoding, and orchestration with their Torch
//! counterparts so both backends produce the same structured results. Only the
//! network forward pass differs: it runs through ONNX Runtime, which mobile
//! platforms accelerate with Core ML (iOS) or NNAPI/XNNPACK (Android).
//!
//! ONNX Runtime is loaded dynamically. Applications either call [`init`] with the
//! bundled library path or set `ORT_DYLIB_PATH` before the first session loads.

pub mod comic_bubble_detector;
pub mod comic_text_detector;
pub mod lama;
pub mod manga_ocr;

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, anyhow};
use ort::{
    ep::ExecutionProviderDispatch,
    session::{Session, builder::GraphOptimizationLevel},
};

pub use self::{
    comic_bubble_detector::{ComicBubbleDetection, ComicBubbleDetectorOnnx},
    comic_text_detector::ComicTextDetectorOnnx,
    lama::LaMaOnnx,
    manga_ocr::MangaOcrOnnx,
};

static LIBRARY: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Every artifact the ONNX models resolve, so apps can download them up front.
#[must_use]
pub fn files() -> Vec<koharu_runtime::HuggingFaceFile<'static>> {
    [
        comic_text_detector::FILES,
        comic_bubble_detector::FILES,
        manga_ocr::FILES,
        lama::FILES,
    ]
    .concat()
}

/// Turns off ONNX Runtime's telemetry for this process.
///
/// Builds since 1.2x report usage from POSIX platforms, including Android and
/// iOS, unless `ORT_DISABLE_TELEMETRY` is set before the runtime starts; the
/// environment's telemetry switch does not cover that path. Translation runs
/// on the device so that pages and usage stay there.
///
/// # Safety
///
/// Call before any other thread exists, as with [`std::env::set_var`].
pub unsafe fn disable_telemetry() {
    // SAFETY: the caller guarantees that no other thread reads the environment.
    unsafe { std::env::set_var("ORT_DISABLE_TELEMETRY", "1") };
}

/// Selects the ONNX Runtime shared library and configures the process-wide
/// environment used by every later session.
///
/// `None` keeps ONNX Runtime's own lookup: `ORT_DYLIB_PATH`, then the platform
/// library name on the loader path. Only the first call has an effect.
/// The environment's telemetry switch is turned off as well; see
/// [`disable_telemetry`] for the part it does not cover.
pub fn init(library: Option<&Path>) -> Result<()> {
    let library = LIBRARY.get_or_init(|| library.map(Path::to_path_buf));
    // Load the library now: ONNX Runtime's lazy lookup panics on first use
    // instead of returning an error the app could show.
    let path = library.clone().unwrap_or_else(default_library);
    ort::init_from(&path)
        .with_context(|| format!("failed to load ONNX Runtime from {}", path.display()))?
        .with_name("koharu")
        .with_telemetry(false)
        .commit();
    Ok(())
}

/// ONNX Runtime's own lookup when no library is given.
fn default_library() -> PathBuf {
    match std::env::var_os("ORT_DYLIB_PATH") {
        Some(path) if !path.is_empty() => path.into(),
        _ if cfg!(target_os = "windows") => "onnxruntime.dll".into(),
        _ if cfg!(any(target_os = "macos", target_os = "ios")) => "libonnxruntime.dylib".into(),
        _ => "libonnxruntime.so".into(),
    }
}

/// Execution providers in preference order for the current platform.
///
/// ONNX Runtime falls back to the next provider, and finally the CPU, for any
/// operator or platform that a provider cannot serve.
fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    let mut providers = Vec::new();
    if cfg!(any(target_os = "ios", target_os = "macos")) {
        providers.push(
            ort::ep::CoreML::default()
                .with_subgraphs(true)
                .with_static_input_shapes(true)
                .build(),
        );
    }
    if cfg!(target_os = "android") {
        providers.push(ort::ep::NNAPI::default().with_fp16(true).build());
    }
    if cfg!(any(target_os = "ios", target_os = "android")) {
        providers.push(ort::ep::XNNPACK::default().build());
    }
    providers
}

pub(crate) fn session(model: &Path) -> Result<Session> {
    init(None)?;
    // Builder errors carry the builder for recovery, which is not `Send`.
    let message = |error: ort::Error<_>| anyhow!("failed to configure ONNX Runtime: {error}");
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(message)?
        .with_intra_threads(threads)
        .map_err(message)?
        // Idle pool threads would otherwise busy-wait between runs, starving the
        // llama.cpp translator that shares the CPU and draining phone batteries.
        .with_intra_op_spinning(false)
        .map_err(message)?
        // The comic text detector's activations decay into subnormal floats, which
        // made its CPU run about 11x slower; flushing them leaves outputs unchanged.
        .with_flush_to_zero()
        .map_err(message)?
        .with_execution_providers(execution_providers())
        .map_err(message)?
        .commit_from_file(model)
        .with_context(|| format!("failed to load ONNX model {}", model.display()))
}
