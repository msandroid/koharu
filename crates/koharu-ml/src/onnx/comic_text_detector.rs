//! Comic text detector through the ONNX export of the BallonsTranslator network.
//!
//! The export fuses the YOLOv5 block detector, U-Net text segmenter, and DBNet
//! line head behind one `images` input of fixed size. Its `seg` output matches
//! the Torch port's `Output::mask` and the first `det` channel matches the shrink
//! map in `Output::line_maps`, so decoding is shared with that port.
//!
//! Unlike the Torch port, which groups DBNet lines on their own, this port keeps
//! only line groups that the YOLOv5 block head also detects. Page furniture such
//! as copyright lines produces DBNet lines but no text block, and would otherwise
//! be recognized and translated as dialogue.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use ort::{session::Session, value::Tensor};

use crate::comic_text_detector::{
    TextBlock,
    processor::{decode_maps, letterbox, rearranged_detection},
};

crate::model_repository!("mayocream/comic-text-detector-onnx" @ "a5d67ec772adef819ef5b0e7aa701fcf4c8bf74a" {
    WEIGHTS = "comic-text-detector.onnx",
});

/// The export was traced at 1024×1024; the Torch port letterboxes to 1280.
const INPUT_SIZE: u32 = 1024;
/// BallonsTranslator's block confidence and NMS defaults; upstream decodes the
/// block head but then groups lines without it (`group_output([], lines, ...)`):
/// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/textdetector/ctd/inference.py#L238-L257
const BLOCK_CONFIDENCE_THRESHOLD: f32 = 0.4;
const BLOCK_NMS_THRESHOLD: f32 = 0.35;
/// Share of a line group's area that a detected block must cover to confirm it.
const BLOCK_COVERAGE_THRESHOLD: f32 = 0.5;

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
    pub fn inference(&self, image: &DynamicImage) -> Result<(GrayImage, Vec<TextBlock>)> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("comic text detector session lock is poisoned"))?;
        // Tall strips are split into square composites. The block head is not
        // consulted there, matching upstream, whose rearranged path only
        // produces maps.
        if let Some(detection) = rearranged_detection(image, INPUT_SIZE, |composites| {
            composites
                .iter()
                .map(|composite| Ok(forward(&mut session, composite)?.maps))
                .collect()
        })? {
            return Ok(detection);
        }

        let (letterboxed, dimensions) = letterbox(image, INPUT_SIZE)?;
        let Forward { blocks, maps } = forward(&mut session, &letterboxed)?;
        let [original_width, _, width, height] = dimensions;
        let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mask = crop_plane(&maps[..plane], width, height);
        let shrink = crop_plane(&maps[plane..2 * plane], width, height);
        let (mask, grouped) = decode_maps(&mask, &shrink, dimensions, image)?;
        let detected = detected_blocks(&blocks, width as f32 / original_width as f32);
        let grouped = grouped
            .into_iter()
            .filter(|block| confirmed(block, &detected))
            .collect();
        Ok((mask, grouped))
    }
}

struct Forward {
    /// YOLOv5 rows `[cx, cy, w, h, objectness, class...]` in input pixels.
    blocks: Vec<f32>,
    /// Text mask, DBNet shrink map, and DBNet threshold map, one plane each.
    maps: Vec<f32>,
}

fn forward(session: &mut Session, image: &RgbImage) -> Result<Forward> {
    let size = INPUT_SIZE as usize;
    let plane = size * size;
    ensure!(
        image.dimensions() == (INPUT_SIZE, INPUT_SIZE),
        "comic text detector input must be {INPUT_SIZE}x{INPUT_SIZE}"
    );
    let mut pixels = vec![0.0f32; 3 * plane];
    for (index, pixel) in image.pixels().enumerate() {
        for channel in 0..3 {
            pixels[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
        }
    }
    let input = Tensor::from_array(([1usize, 3, size, size], pixels))?;
    let outputs = session.run(ort::inputs!["images" => input])?;
    let (blk_shape, blk) = outputs["blk"].try_extract_tensor::<f32>()?;
    let (seg_shape, seg) = outputs["seg"].try_extract_tensor::<f32>()?;
    let (det_shape, det) = outputs["det"].try_extract_tensor::<f32>()?;
    let side = INPUT_SIZE as i64;
    ensure!(
        blk_shape.len() == 3
            && blk_shape[0] == 1
            && blk_shape[2] == 7
            && seg_shape.as_ref() == [1, 1, side, side]
            && det_shape.as_ref() == [1, 2, side, side],
        "unexpected comic text detector output shapes {blk_shape:?}, {seg_shape:?}, {det_shape:?}"
    );
    Ok(Forward {
        blocks: blk.to_vec(),
        maps: [seg, det].concat(),
    })
}

/// Decodes the YOLOv5 head rows `[cx, cy, w, h, objectness, class...]` into
/// source-resolution boxes after confidence filtering and greedy NMS.
fn detected_blocks(rows: &[f32], scale: f32) -> Vec<[f32; 4]> {
    let mut candidates = rows
        .chunks_exact(7)
        .filter_map(|row| {
            let class = row[5].max(row[6]);
            let score = row[4] * class;
            (score > BLOCK_CONFIDENCE_THRESHOLD).then(|| {
                let [cx, cy, w, h] = [row[0], row[1], row[2], row[3]].map(|value| value / scale);
                (
                    score,
                    [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                )
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut kept: Vec<[f32; 4]> = Vec::new();
    for (_, candidate) in candidates {
        if kept
            .iter()
            .all(|existing| iou(*existing, candidate) <= BLOCK_NMS_THRESHOLD)
        {
            kept.push(candidate);
        }
    }
    kept
}

fn confirmed(block: &TextBlock, detected: &[[f32; 4]]) -> bool {
    let bbox = block.xyxy.map(|value| value as f32);
    let area = area(bbox).max(f32::EPSILON);
    detected
        .iter()
        .any(|candidate| intersection(bbox, *candidate) / area >= BLOCK_COVERAGE_THRESHOLD)
}

fn area([left, top, right, bottom]: [f32; 4]) -> f32 {
    (right - left).max(0.0) * (bottom - top).max(0.0)
}

fn intersection(a: [f32; 4], b: [f32; 4]) -> f32 {
    area([
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ])
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let shared = intersection(a, b);
    shared / (area(a) + area(b) - shared).max(f32::EPSILON)
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
    #[test]
    fn line_groups_need_a_detected_block() {
        // Two overlapping confident rows and one below the confidence threshold,
        // in letterboxed pixels at half the source resolution.
        let rows = [
            [60.0, 60.0, 40.0, 40.0, 0.9, 0.0, 1.0],
            [61.0, 61.0, 40.0, 40.0, 0.8, 0.0, 1.0],
            [300.0, 300.0, 20.0, 20.0, 0.3, 1.0, 0.0],
        ]
        .concat();
        let detected = detected_blocks(&rows, 0.5);
        assert_eq!(
            detected,
            [[80.0, 80.0, 160.0, 160.0]],
            "NMS keeps the best row"
        );

        let block = |xyxy| TextBlock {
            xyxy,
            lines: Vec::new(),
            language: "unknown".to_owned(),
            vertical: true,
            angle: 0,
            detected_font_size: 12.0,
        };
        assert!(confirmed(&block([90, 90, 150, 150]), &detected));
        assert!(!confirmed(&block([140, 140, 220, 220]), &detected));
        assert!(!confirmed(&block([580, 580, 620, 620]), &detected));
    }
}
