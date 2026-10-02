//! Speech bubble detection through the RT-DETRv2 comic text and bubble detector.
//!
//! The ONNX export embeds RT-DETR's postprocessing, so it takes the resized image
//! with the original `(width, height)` and returns the top 300 queries as labels,
//! source-resolution `xyxy` boxes, and scores. Preprocessing follows the
//! checkpoint's `RTDetrImageProcessor`: a plain 640×640 bilinear resize and
//! rescaling to `[0, 1]` without normalization.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, RgbImage};
use ort::{session::Session, value::Tensor};

crate::model_repository!("ogkalu/comic-text-and-bubble-detector" @ "16e8a622f91fabc6b5b65c96d32d1183f8843546" {
    WEIGHTS = "detector-v4-s_int8.onnx",
});

pub(super) const FILES: &[koharu_runtime::HuggingFaceFile<'static>] = &[WEIGHTS];

const INPUT_SIZE: u32 = 640;
const LABELS: [&str; 3] = ["bubble", "text_bubble", "text_free"];

#[derive(Debug, Clone, PartialEq)]
pub struct ComicBubbleDetection {
    /// One of `bubble`, `text_bubble` (text inside a bubble), or `text_free`.
    pub label: &'static str,
    pub score: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug)]
pub struct ComicBubbleDetectorOnnx {
    session: Mutex<Session>,
}

impl ComicBubbleDetectorOnnx {
    pub async fn load() -> Result<Self> {
        let path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve comic bubble detector ONNX weights")?;
        Ok(Self {
            session: Mutex::new(super::session(&path)?),
        })
    }

    /// Returns detections scoring above `threshold`, highest score first.
    pub fn inference(
        &self,
        image: &DynamicImage,
        threshold: f32,
    ) -> Result<Vec<ComicBubbleDetection>> {
        let (width, height) = image.dimensions();
        ensure!(width > 0 && height > 0, "empty image");
        let mut resized = RgbImage::new(INPUT_SIZE, INPUT_SIZE);
        Resizer::new()
            .resize(
                &image.to_rgb8(),
                &mut resized,
                &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
            )
            .map_err(|error| anyhow!("failed to resize bubble detector input: {error}"))?;
        let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mut pixels = vec![0.0f32; 3 * plane];
        for (index, pixel) in resized.pixels().enumerate() {
            for channel in 0..3 {
                pixels[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
            }
        }
        let size = INPUT_SIZE as usize;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("bubble detector session lock is poisoned"))?;
        let outputs = session.run(ort::inputs![
            "images" => Tensor::from_array(([1usize, 3, size, size], pixels))?,
            "orig_target_sizes" => Tensor::from_array(([1usize, 2], vec![i64::from(width), i64::from(height)]))?,
        ])?;
        let (_, labels) = outputs["labels"].try_extract_tensor::<i64>()?;
        let (_, boxes) = outputs["boxes"].try_extract_tensor::<f32>()?;
        let (_, scores) = outputs["scores"].try_extract_tensor::<f32>()?;
        ensure!(
            boxes.len() == labels.len() * 4 && scores.len() == labels.len(),
            "bubble detector outputs disagree on the number of queries"
        );

        let mut detections = labels
            .iter()
            .zip(scores)
            .zip(boxes.chunks_exact(4))
            .filter(|((_, score), _)| **score > threshold)
            .filter_map(|((label, score), bbox)| {
                let label = *LABELS.get(usize::try_from(*label).ok()?)?;
                Some(ComicBubbleDetection {
                    label,
                    score: *score,
                    bbox: [
                        bbox[0].clamp(0.0, width as f32),
                        bbox[1].clamp(0.0, height as f32),
                        bbox[2].clamp(0.0, width as f32),
                        bbox[3].clamp(0.0, height as f32),
                    ],
                })
            })
            .collect::<Vec<_>>();
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(detections)
    }
}
