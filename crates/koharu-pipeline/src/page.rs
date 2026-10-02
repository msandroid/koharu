//! Single-image entry points shared by the command-line runner and mobile app.

use std::{collections::BTreeMap, sync::Arc};

use anyhow::{Context as _, Result};
use image::RgbaImage;
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, EntityId, PageDraft, Session, Snapshot,
};

/// Adds an encoded image to the end of `session` as a page with a `source` asset.
pub async fn import_image(session: &mut Session, name: &str, bytes: Vec<u8>) -> Result<EntityId> {
    let decoded = image::load_from_memory(&bytes).context("failed to decode the page image")?;
    let media_type = image::guess_format(&bytes)
        .map(|format| format.to_mime_type())
        .unwrap_or("image/png");
    let mut page = None;
    let patch = session.snapshot().patch(|edit| {
        let id = edit.add_page(
            PageDraft::new(
                name,
                f64::from(decoded.width()),
                f64::from(decoded.height()),
            ),
            At::End,
        )?;
        edit.set_asset(
            id,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes),
                media_type,
                AssetMetadata {
                    width: Some(decoded.width()),
                    height: Some(decoded.height()),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        page = Some(id);
        Ok(())
    })?;
    session.commit(patch).await?;
    page.context("the page edit did not assign an ID")
}

/// Renders `page` with its translated text and cleanup layers composited.
pub async fn render_image(
    renderer: &Renderer,
    rasterizer: &Rasterizer,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<RgbaImage> {
    let frame = renderer.render(snapshot, page).await?;
    let raster = rasterizer.rasterize(&frame.raster_frame()?, RasterOptions::default())?;
    Ok(raster.image)
}
