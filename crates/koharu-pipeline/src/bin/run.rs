use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    DetectionModel, Flux2KleinConfig, InpaintingModel, KoharuLayoutRFDetrSeg2XLConfig, OcrModel,
    Operation, Pipeline, PipelineConfig, Progress, Request, RoremMixedConfig, Scope,
    TranslationConfig, import_image, render_image,
};
use koharu_rasterizer::Rasterizer;
use koharu_renderer::Renderer;
use koharu_scene::Session;
use koharu_translator::{GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig};

#[derive(Debug, Parser)]
#[command(version, about = "Run Koharu's complete in-process pipeline")]
struct Arguments {
    #[arg(short, long, value_name = "INPUT")]
    input: PathBuf,

    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    #[arg(long, value_enum, default_value = "koharu-layout-rfdetr-seg-2xl")]
    detection: DetectionChoice,

    #[arg(long, value_enum, default_value = "paddleocr-vl-1.6")]
    ocr: OcrChoice,

    #[arg(long, value_enum, default_value = "lama")]
    inpainting: InpaintingChoice,

    #[arg(long, default_value = "en-US")]
    target_language: Language,

    #[arg(long)]
    translation_instructions: Option<String>,

    #[arg(long, default_value = "gemma4-12b-it")]
    llm: String,

    #[arg(long)]
    cpu: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DetectionChoice {
    #[value(name = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL,
    #[value(name = "comic-text-detector-onnx")]
    ComicTextDetectorOnnx,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OcrChoice {
    #[value(name = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[value(name = "manga-ocr")]
    MangaOcr,
    #[value(name = "manga-ocr-onnx")]
    MangaOcrOnnx,
    #[value(name = "baberu-ocr")]
    BaberuOcr,
    #[value(name = "hayai-ocr")]
    HayaiOcr,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InpaintingChoice {
    #[value(name = "lama")]
    LaMa,
    #[value(name = "lama-onnx")]
    LaMaOnnx,
    #[value(name = "aot-inpainting")]
    AotInpainting,
    #[value(name = "flux2-klein")]
    Flux2Klein,
    #[value(name = "rorem-mixed")]
    RoremMixed,
}

impl Arguments {
    fn pipeline_config(&self) -> PipelineConfig {
        PipelineConfig {
            detection: match self.detection {
                DetectionChoice::KoharuLayoutRFDetrSeg2XL => {
                    DetectionModel::KoharuLayoutRFDetrSeg2XL(
                        KoharuLayoutRFDetrSeg2XLConfig::default(),
                    )
                }
                DetectionChoice::ComicTextDetectorOnnx => DetectionModel::ComicTextDetectorOnnx {},
            },
            ocr: match self.ocr {
                OcrChoice::PaddleOcrVl1_6 => OcrModel::PaddleOcrVl1_6,
                OcrChoice::MangaOcr => OcrModel::MangaOcr,
                OcrChoice::MangaOcrOnnx => OcrModel::MangaOcrOnnx,
                OcrChoice::BaberuOcr => OcrModel::BaberuOcr,
                OcrChoice::HayaiOcr => OcrModel::HayaiOcr,
            },
            translation: TranslationConfig {
                model: ModelSelection {
                    provider: Provider::Local,
                    model: Some(self.llm.clone()),
                    quantization: None,
                    vision: true,
                    reasoning: true,
                },
                generation: GenerationConfig::default(),
                target_language: self.target_language,
                instructions: self.translation_instructions.clone(),
            },
            inpainting: match self.inpainting {
                InpaintingChoice::LaMa => InpaintingModel::LaMa {},
                InpaintingChoice::LaMaOnnx => InpaintingModel::LaMaOnnx {},
                InpaintingChoice::AotInpainting => InpaintingModel::AotInpainting {},
                InpaintingChoice::Flux2Klein => {
                    InpaintingModel::Flux2Klein(Flux2KleinConfig::default())
                }
                InpaintingChoice::RoremMixed => {
                    InpaintingModel::RoremMixed(RoremMixedConfig::default())
                }
            },
            processor: Default::default(),
        }
    }
}

fn main() -> Result<()> {
    // SAFETY: no other thread exists yet.
    unsafe { koharu_ml::onnx::disable_telemetry() };
    tokio::runtime::Runtime::new()?.block_on(run(Arguments::parse()))
}

async fn run(arguments: Arguments) -> Result<()> {
    initialize_with_retry().await;

    let source = fs::read(&arguments.input)
        .with_context(|| format!("failed to read {}", arguments.input.display()))?;
    let name = arguments
        .input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("input");
    let mut session = Session::memory().await?;
    let page = import_image(&mut session, name, source).await?;

    let device = koharu_ml::device(arguments.cpu);
    let pipeline = Pipeline::from_config(
        Config::memory(arguments.pipeline_config()),
        Config::memory(ProvidersConfig::default()),
        device,
    )?;
    let snapshot = session.snapshot();
    let report = pipeline
        .execute(
            snapshot,
            Request {
                operation: Operation::Full,
                scope: Scope::Pages(vec![page]),
                progress: Some(Arc::new(|event| {
                    if let Progress::Finished { stage, elapsed, .. } = event {
                        eprintln!("{stage} finished in {:.2}s", elapsed.as_secs_f64());
                    }
                })),
                ..Request::default()
            },
            &mut session,
        )
        .await?;
    eprintln!("pipeline finished in {:.2}s", report.elapsed.as_secs_f64());

    let renderer = Renderer::new()?;
    let rasterizer = Rasterizer::new()?;
    let render_started = Instant::now();
    let image = render_image(&renderer, &rasterizer, &session.snapshot(), page).await?;
    let render_elapsed = render_started.elapsed();
    image
        .save(&arguments.output)
        .with_context(|| format!("failed to write {}", arguments.output.display()))?;
    eprintln!(
        "rendered {} in {:.2}s",
        arguments.output.display(),
        render_elapsed.as_secs_f64()
    );
    Ok(())
}

async fn initialize_with_retry() {
    let mut delay = Duration::from_secs(1);
    let mut attempt = 0_u64;
    loop {
        attempt += 1;
        match koharu_ml::init().await {
            Ok(()) => return,
            Err(error) => {
                let jitter = Duration::from_millis((attempt.wrapping_mul(137)) % 251);
                let wait = delay + jitter;
                eprintln!(
                    "runtime initialization attempt {attempt} failed: {error}; retrying in {:.1}s",
                    wait.as_secs_f64()
                );
                tokio::time::sleep(wait).await;
                delay = delay.saturating_mul(2).min(Duration::from_secs(30));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_flags_select_models() {
        let arguments = Arguments::try_parse_from([
            "run",
            "--input",
            "input.png",
            "--output",
            "output.png",
            "--ocr",
            "manga-ocr",
            "--inpainting",
            "flux2-klein",
        ])
        .unwrap();
        assert!(matches!(
            arguments.pipeline_config().ocr,
            OcrModel::MangaOcr
        ));
        assert!(matches!(
            arguments.pipeline_config().inpainting,
            InpaintingModel::Flux2Klein(_)
        ));
    }
}
