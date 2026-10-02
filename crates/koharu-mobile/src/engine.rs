//! The on-device translation engine behind the mobile app's commands.
//!
//! It owns one pipeline configured with the ONNX vision models, keeps the
//! user's translation settings in the app's data directory, and forwards
//! pipeline and download progress to the UI.

use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result};
use koharu_config::Config;
use koharu_pipeline::{
    DetectionModel, InpaintingModel, OcrModel, Operation, Pipeline, PipelineConfig, Progress,
    Request, Scope, TranslationConfig, import_image, render_image,
};
use koharu_rasterizer::Rasterizer;
use koharu_renderer::Renderer;
use koharu_runtime::{Store, download};
use koharu_scene::Session;
use koharu_translator::{GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

const SETTINGS_FILE: &str = "settings.json";

/// Translation choices the user controls.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub target_language: Language,
    pub model: ModelSelection,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target_language: Language::English,
            // The 2B Gemma 4 model translates Japanese well enough to start with
            // and fits in phone memory. Vision is off so the multimodal
            // projector is not downloaded.
            model: ModelSelection {
                provider: Provider::Local,
                model: Some("gemma4-e2b-it".to_owned()),
                quantization: Some("Q4_K_XL".to_owned()),
                vision: false,
                reasoning: false,
            },
        }
    }
}

/// Progress reported to the UI while preparing models or translating.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Stage {
        stage: String,
        model: Option<String>,
        state: StageState,
    },
    Download {
        name: String,
        completed: u64,
        total: u64,
    },
    DownloadFailed {
        name: String,
        error: String,
    },
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    Loading,
    Running,
    Finished,
    Skipped,
}

pub type EventSink = Arc<dyn Fn(Event) + Send + Sync>;

/// Where the app may write; sandboxed mobile apps own no other directories.
pub struct Directories {
    pub cache: PathBuf,
    pub data: PathBuf,
    /// ONNX Runtime bundled with the app, when it is not on the loader path.
    pub onnx_runtime: Option<PathBuf>,
}

pub struct Engine {
    pipeline: Pipeline,
    pipeline_config: Config<PipelineConfig>,
    renderer: Renderer,
    rasterizer: Rasterizer,
    settings: std::sync::Mutex<Settings>,
    settings_path: PathBuf,
    events: EventSink,
    translating: Mutex<()>,
}

impl Engine {
    /// Points every store at the app's directories, then loads native runtimes.
    pub async fn start(directories: Directories, events: EventSink) -> Result<Self> {
        Store::configure(directories.cache.join("packages"))?;
        koharu_config::configure(directories.data.join("config.toml"))?;
        koharu_ml::onnx::init(directories.onnx_runtime.as_deref())?;
        koharu_ml::init().await?;
        forward_downloads(events.clone());

        let settings_path = directories.data.join(SETTINGS_FILE);
        let settings = read_settings(&settings_path)?;
        let pipeline_config = Config::memory(pipeline_config(&settings));
        let pipeline = Pipeline::from_config(
            pipeline_config.clone(),
            Config::memory(ProvidersConfig::default()),
            koharu_ml::device(false),
        )?;
        Ok(Self {
            pipeline,
            pipeline_config,
            renderer: Renderer::new()?,
            rasterizer: Rasterizer::new()?,
            settings: std::sync::Mutex::new(settings),
            settings_path,
            events,
            translating: Mutex::new(()),
        })
    }

    #[must_use]
    pub fn settings(&self) -> Settings {
        self.settings
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Persists `settings` and stores an API key for its provider when given.
    pub fn save_settings(&self, settings: Settings, api_key: Option<String>) -> Result<()> {
        if let Some(api_key) = api_key.filter(|key| !key.trim().is_empty()) {
            koharu_secrets::set(
                &settings.model.provider.to_string(),
                &koharu_secrets::SecretString::from(api_key.trim().to_owned()),
            )?;
        }
        write_settings(&self.settings_path, &settings)?;
        self.pipeline_config.write()?.translation = translation_config(&settings);
        *self
            .settings
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = settings;
        Ok(())
    }

    /// Whether every vision model is already downloaded.
    #[must_use]
    pub fn models_ready(&self) -> bool {
        koharu_ml::onnx::files()
            .into_iter()
            .all(|file| file.exists())
    }

    /// Downloads the vision models; progress arrives as download events.
    pub async fn prepare(&self) -> Result<()> {
        for file in koharu_ml::onnx::files() {
            file.resolve().await?;
        }
        Ok(())
    }

    /// Translates one encoded page image and returns the result as PNG.
    pub async fn translate(&self, image: Vec<u8>) -> Result<Vec<u8>> {
        let _translating = self.translating.lock().await;
        let mut session = Session::memory().await?;
        let page = import_image(&mut session, "page", image).await?;
        let events = self.events.clone();
        self.pipeline
            .execute(
                session.snapshot(),
                Request {
                    operation: Operation::Full,
                    scope: Scope::Pages(vec![page]),
                    progress: Some(Arc::new(move |progress| {
                        if let Some(event) = stage_event(progress) {
                            events(event);
                        }
                    })),
                    ..Request::default()
                },
                &mut session,
            )
            .await?;
        let rendered =
            render_image(&self.renderer, &self.rasterizer, &session.snapshot(), page).await?;
        let mut png = Vec::new();
        rendered
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .context("failed to encode the translated page")?;
        Ok(png)
    }
}

fn pipeline_config(settings: &Settings) -> PipelineConfig {
    PipelineConfig {
        detection: DetectionModel::ComicTextDetectorOnnx {},
        ocr: OcrModel::MangaOcrOnnx,
        translation: translation_config(settings),
        inpainting: InpaintingModel::LaMaOnnx {},
        processor: Default::default(),
    }
}

fn translation_config(settings: &Settings) -> TranslationConfig {
    TranslationConfig {
        model: settings.model.clone(),
        generation: GenerationConfig::default(),
        target_language: settings.target_language,
        instructions: None,
    }
}

fn read_settings(path: &Path) -> Result<Settings> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn write_settings(path: &Path, settings: &Settings) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(settings)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn stage_event(progress: Progress) -> Option<Event> {
    let (stage, model, state) = match progress {
        Progress::Started { .. } => return None,
        Progress::Loading { stage, model, .. } => (stage, Some(model), StageState::Loading),
        Progress::Running { stage, model, .. } => (stage, Some(model), StageState::Running),
        Progress::Finished { stage, model, .. } => (stage, Some(model), StageState::Finished),
        Progress::Skipped { stage, .. } => (stage, None, StageState::Skipped),
    };
    Some(Event::Stage {
        stage: stage.to_string(),
        model,
        state,
    })
}

fn forward_downloads(events: EventSink) {
    let mut downloads = download::subscribe();
    tokio::spawn(async move {
        loop {
            let event = match downloads.recv().await {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
            };
            match event {
                download::Event::Progress {
                    name,
                    completed,
                    total,
                    ..
                } => events(Event::Download {
                    name,
                    completed,
                    total,
                }),
                download::Event::Failed { name, error, .. } => {
                    events(Event::DownloadFailed { name, error });
                }
                download::Event::Started { .. } | download::Event::Finished { .. } => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_default_when_missing() {
        let directory = std::env::temp_dir().join(format!("koharu-mobile-{}", std::process::id()));
        let path = directory.join(SETTINGS_FILE);
        assert_eq!(read_settings(&path).unwrap(), Settings::default());

        let settings = Settings {
            target_language: Language::German,
            model: ModelSelection {
                provider: Provider::Claude,
                model: Some("claude-sonnet".to_owned()),
                ..ModelSelection::default()
            },
        };
        write_settings(&path, &settings).unwrap();
        assert_eq!(read_settings(&path).unwrap(), settings);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn the_pipeline_uses_the_onnx_models() {
        let config = pipeline_config(&Settings::default());
        assert_eq!(config.detection, DetectionModel::ComicTextDetectorOnnx {});
        assert_eq!(config.ocr, OcrModel::MangaOcrOnnx);
        assert_eq!(config.inpainting, InpaintingModel::LaMaOnnx {});
        assert_eq!(config.translation.target_language, Language::English);
    }
}
