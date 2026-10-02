//! Comic text detector through the ONNX export of the BallonsTranslator network.
//!
//! The export fuses the YOLOv5 block detector, U-Net text segmenter, and DBNet
//! line head behind one `images` input of fixed size. Its `seg` output matches
//! the Torch port's `Output::mask` and the first `det` channel matches the shrink
//! map in `Output::line_maps`, so decoding is shared with that port.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use image::{DynamicImage, GrayImage};
use ort::{session::Session, value::Tensor};

use crate::comic_text_detector::{
    TextBlock,
    processor::{decode_maps, letterbox},
};

crate::model_repository!("mayocream/comic-text-detector-onnx" @ "a5d67ec772adef819ef5b0e7aa701fcf4c8bf74a" {
    WEIGHTS = "comic-text-detector.onnx",
});

/// The export was traced at 1024×1024; the Torch port letterboxes to 1280.
const INPUT_SIZE: u32 = 1024;

#[derive(Debug)]
pub struct ComicTextDetectorOnnx {
    session: Mutex<Session>,
}

impl ComicTextDetectorOnnx {
    pub async fn load() -> Result<Self> {
        let path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve comic-text-detector ONNX weights")?;
        Ok(Self {
            session: Mutex::new(super::session(&path)?),
        })
    }

    /// Returns the refined text mask at the source resolution and the grouped
    /// text blocks, like `ComicTextDetector::inference`.
    ///
    /// The Torch port additionally rearranges very tall strips before inference;
    /// this port always letterboxes the whole page.
    pub fn inference(&self, image: &DynamicImage) -> Result<(GrayImage, Vec<TextBlock>)> {
        let (letterboxed, dimensions) = letterbox(image, INPUT_SIZE)?;
        let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mut pixels = vec![0.0f32; 3 * plane];
        for (index, pixel) in letterboxed.pixels().enumerate() {
            for channel in 0..3 {
                pixels[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
            }
        }
        let input = Tensor::from_array((
            [1usize, 3, INPUT_SIZE as usize, INPUT_SIZE as usize],
            pixels,
        ))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("comic text detector session lock is poisoned"))?;
        let outputs = session.run(ort::inputs!["images" => input])?;
        let (seg_shape, seg) = outputs["seg"].try_extract_tensor::<f32>()?;
        let (det_shape, det) = outputs["det"].try_extract_tensor::<f32>()?;
        let size = INPUT_SIZE as i64;
        ensure!(
            seg_shape.as_ref() == [1, 1, size, size] && det_shape.as_ref() == [1, 2, size, size],
            "unexpected comic text detector output shapes {seg_shape:?} and {det_shape:?}"
        );

        let [_, _, width, height] = dimensions;
        let mask = crop_plane(seg, width, height);
        let shrink = crop_plane(&det[..plane], width, height);
        decode_maps(&mask, &shrink, dimensions, image)
    }
}

/// Copies the top-left `width`×`height` content area out of a square map.
fn crop_plane(plane: &[f32], width: u32, height: u32) -> Vec<f32> {
    let stride = INPUT_SIZE as usize;
    plane
        .chunks_exact(stride)
        .take(height as usize)
        .flat_map(|row| &row[..width as usize])
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_plane_keeps_the_letterboxed_content_area() {
        let size = INPUT_SIZE as usize;
        let plane = (0..size * size)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        let cropped = crop_plane(&plane, 3, 2);

        let second_row = size as f32;
        assert_eq!(
            cropped,
            [
                0.0,
                1.0,
                2.0,
                second_row,
                second_row + 1.0,
                second_row + 2.0
            ]
        );
    }
}
