//! Runs the mobile engine on one page without a window, for development.
//!
//! cargo run -p koharu-mobile --example translate_page -- <input> <output.png> <cache-dir> <data-dir>

use std::sync::Arc;

use anyhow::{Context as _, Result};
use koharu_mobile_lib::{Directories, Engine, Event};

fn main() -> Result<()> {
    // SAFETY: no other thread exists yet.
    unsafe { koharu_ml::onnx::disable_telemetry() };
    tokio::runtime::Runtime::new()?.block_on(translate())
}

async fn translate() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let mut next = |name: &str| arguments.next().with_context(|| format!("missing {name}"));
    let (input, output) = (next("input")?, next("output")?);
    let (cache, data) = (next("cache directory")?, next("data directory")?);

    let engine = Engine::start(
        Directories {
            cache: cache.into(),
            data: data.into(),
            onnx_runtime: None,
        },
        Arc::new(|event: Event| eprintln!("{}", serde_json::to_string(&event).unwrap())),
    )
    .await?;
    eprintln!("models ready: {}", engine.models_ready());
    engine.prepare().await?;
    let png = engine.translate(std::fs::read(&input)?).await?;
    std::fs::write(&output, png)?;
    eprintln!("wrote {output}");
    Ok(())
}
