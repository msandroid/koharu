//! ONNX Runtime ports of the lightweight models used by mobile builds.
//!
//! These models share preprocessing, decoding, and orchestration with their Torch
//! counterparts so both backends produce the same structured results. Only the
//! network forward pass differs: it runs through ONNX Runtime, which mobile
//! platforms accelerate with Core ML (iOS) or NNAPI/XNNPACK (Android).
//!
//! ONNX Runtime is loaded dynamically. Applications either call [`init`] with the
//! bundled library path or set `ORT_DYLIB_PATH` before the first session loads.

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
    comic_text_detector::ComicTextDetectorOnnx, lama::LaMaOnnx, manga_ocr::MangaOcrOnnx,
};

static LIBRARY: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Selects the ONNX Runtime shared library used by every later session.
///
/// `None` keeps ONNX Runtime's own lookup: `ORT_DYLIB_PATH`, then the platform
/// library name on the loader path. Only the first call has an effect.
pub fn init(library: Option<&Path>) -> Result<()> {
    let library = LIBRARY.get_or_init(|| library.map(Path::to_path_buf));
    if let Some(path) = library {
        ort::init_from(path)
            .with_context(|| format!("failed to load ONNX Runtime from {}", path.display()))?
            .with_name("koharu")
            .commit();
    }
    Ok(())
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
        .with_execution_providers(execution_providers())
        .map_err(message)?
        .commit_from_file(model)
        .with_context(|| format!("failed to load ONNX model {}", model.display()))
}
