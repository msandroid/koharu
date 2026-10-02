//! Koharu's mobile app: a Tauri v2 shell around the on-device [`engine`].

mod engine;

use std::{path::PathBuf, sync::Arc};

use tauri::{
    AppHandle, Emitter, Manager, State,
    ipc::{InvokeBody, Request, Response},
};
use tokio::sync::OnceCell;

pub use engine::{Directories, Engine, Event, Settings};

const EVENT: &str = "koharu://event";

/// Starts the engine on first use so the window appears immediately.
struct EngineCell {
    engine: OnceCell<Engine>,
    app: AppHandle,
}

impl EngineCell {
    async fn get(&self) -> Result<&Engine, String> {
        self.engine
            .get_or_try_init(|| async {
                let app = self.app.clone();
                let sink: engine::EventSink = Arc::new(move |event: Event| {
                    if let Err(error) = app.emit(EVENT, event) {
                        tracing::warn!(%error, "failed to deliver an engine event");
                    }
                });
                Engine::start(directories(&self.app)?, sink).await
            })
            .await
            .map_err(|error| format!("{error:#}"))
    }
}

fn directories(app: &AppHandle) -> anyhow::Result<Directories> {
    let paths = app.path();
    Ok(Directories {
        cache: paths.app_cache_dir()?,
        data: paths.app_data_dir()?,
        onnx_runtime: bundled_onnx_runtime(),
    })
}

/// The ONNX Runtime library an app bundle carries, if the loader cannot find it.
///
/// Android's linker resolves `libonnxruntime.so` from the APK by name and
/// desktop development builds use `ORT_DYLIB_PATH`; iOS embeds a framework.
fn bundled_onnx_runtime() -> Option<PathBuf> {
    if !cfg!(target_os = "ios") {
        return None;
    }
    let executable = std::env::current_exe().ok()?;
    Some(
        executable
            .parent()?
            .join("Frameworks")
            .join("onnxruntime.framework")
            .join("onnxruntime"),
    )
}

#[derive(serde::Serialize)]
struct Status {
    models_ready: bool,
    settings: Settings,
}

#[tauri::command]
async fn status(cell: State<'_, EngineCell>) -> Result<Status, String> {
    let engine = cell.get().await?;
    Ok(Status {
        models_ready: engine.models_ready(),
        settings: engine.settings(),
    })
}

#[tauri::command]
async fn prepare(cell: State<'_, EngineCell>) -> Result<(), String> {
    cell.get()
        .await?
        .prepare()
        .await
        .map_err(|error| format!("{error:#}"))
}

#[tauri::command]
async fn models() -> Result<Vec<koharu_translator::Model>, String> {
    koharu_translator::Translator::models()
        .await
        .map_err(|error| format!("{error:#}"))
}

#[derive(serde::Serialize)]
struct LanguageOption {
    tag: koharu_translator::Language,
    name: String,
}

#[tauri::command]
fn languages() -> Vec<LanguageOption> {
    use strum::VariantArray as _;
    koharu_translator::Language::VARIANTS
        .iter()
        .map(|language| LanguageOption {
            tag: *language,
            name: language.to_string(),
        })
        .collect()
}

#[tauri::command]
async fn save_settings(
    cell: State<'_, EngineCell>,
    settings: Settings,
    api_key: Option<String>,
) -> Result<(), String> {
    cell.get()
        .await?
        .save_settings(settings, api_key)
        .map_err(|error| format!("{error:#}"))
}

/// Takes the encoded page as the raw request body and returns PNG bytes, so
/// images cross the bridge without JSON encoding.
#[tauri::command]
async fn translate(cell: State<'_, EngineCell>, request: Request<'_>) -> Result<Response, String> {
    let InvokeBody::Raw(image) = request.body() else {
        return Err("translate expects the page image as a raw body".to_owned());
    };
    let png = cell
        .get()
        .await?
        .translate(image.clone())
        .await
        .map_err(|error| format!("{error:#}"))?;
    Ok(Response::new(png))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // SAFETY: the app's entry point runs before Tauri or Tokio start threads.
    unsafe { koharu_ml::onnx::disable_telemetry() };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();
    tauri::Builder::default()
        .runtime(tauri_runtime_wry::Wry::default())
        .setup(|app| {
            app.manage(EngineCell {
                engine: OnceCell::new(),
                app: app.handle().clone(),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            status,
            prepare,
            models,
            languages,
            save_settings,
            translate
        ])
        .run(tauri::generate_context!())
        .expect("failed to run the Koharu mobile app");
}
