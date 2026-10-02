//! LaMa manga inpainting through its fixed-resolution ONNX export.
//!
//! IOPaint's crop orchestration is shared with the Torch port. The export only
//! accepts 512×512 inputs, so each crop is scaled down to fit when larger and
//! padded with the symmetric reflection that the Torch port uses for its
//! modulo-8 padding. Unmasked pixels are restored afterwards, matching
//! `sd_keep_unmasked_area`.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use ort::{session::Session, value::Tensor};

use crate::lama::{
    InpaintRequest,
    processor::{orchestrate, resize_gray, resize_rgb, symmetric_index},
};

crate::model_repository!("mayocream/lama-manga-onnx" @ "b55497aadbfcb9740e1ed16f008268d71b4f3f79" {
    WEIGHTS = "lama-manga.onnx",
});

const INPUT_SIZE: u32 = 512;

#[derive(Debug)]
pub struct LaMaOnnx {
    session: Mutex<Session>,
}

impl LaMaOnnx {
    pub async fn load() -> Result<Self> {
        let path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve LaMa ONNX weights")?;
        Ok(Self {
            session: Mutex::new(super::session(&path)?),
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        orchestrate(image, mask, config, |image, mask| {
            let inpainted = self.forward(image, mask)?;
            Ok(if config.sd_keep_unmasked_area {
                keep_unmasked_area(&inpainted, image, mask)
            } else {
                inpainted
            })
        })
    }

    fn forward(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        ensure!(width > 0 && height > 0, "image dimensions must be non-zero");
        let scale = (f64::from(INPUT_SIZE) / f64::from(width.max(height))).min(1.0);
        let scaled_width = ((f64::from(width) * scale).round() as u32).clamp(1, INPUT_SIZE);
        let scaled_height = ((f64::from(height) * scale).round() as u32).clamp(1, INPUT_SIZE);
        let (scaled_image, scaled_mask) = if scale < 1.0 {
            (
                resize_rgb(image, scaled_width, scaled_height)?,
                resize_gray(mask, scaled_width, scaled_height)?,
            )
        } else {
            (image.clone(), mask.clone())
        };

        let size = INPUT_SIZE as usize;
        let plane = size * size;
        let mut pixels = vec![0.0f32; 3 * plane];
        let mut holes = vec![0.0f32; plane];
        for y in 0..INPUT_SIZE {
            let source_y = symmetric_index(y, scaled_height);
            for x in 0..INPUT_SIZE {
                let source_x = symmetric_index(x, scaled_width);
                let index = y as usize * size + x as usize;
                let pixel = scaled_image.get_pixel(source_x, source_y);
                for channel in 0..3 {
                    pixels[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
                }
                if scaled_mask.get_pixel(source_x, source_y)[0] > 0 {
                    holes[index] = 1.0;
                }
            }
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("LaMa session lock is poisoned"))?;
        let outputs = session.run(ort::inputs![
            "image" => Tensor::from_array(([1usize, 3, size, size], pixels))?,
            "mask" => Tensor::from_array(([1usize, 1, size, size], holes))?,
        ])?;
        let (shape, output) = outputs["output"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, 3, INPUT_SIZE as i64, INPUT_SIZE as i64],
            "unexpected LaMa output shape {shape:?}"
        );
        let inpainted = RgbImage::from_fn(scaled_width, scaled_height, |x, y| {
            let index = y as usize * size + x as usize;
            image::Rgb(std::array::from_fn(|channel| {
                (output[channel * plane + index].clamp(0.0, 1.0) * 255.0) as u8
            }))
        });
        if scale < 1.0 {
            resize_rgb(&inpainted, width, height)
        } else {
            Ok(inpainted)
        }
    }
}

/// Blends the network output back over the source with the mask as alpha.
fn keep_unmasked_area(inpainted: &RgbImage, source: &RgbImage, mask: &GrayImage) -> RgbImage {
    RgbImage::from_fn(source.width(), source.height(), |x, y| {
        let alpha = f32::from(mask.get_pixel(x, y)[0]) / 255.0;
        let output = inpainted.get_pixel(x, y);
        let original = source.get_pixel(x, y);
        image::Rgb(std::array::from_fn(|channel| {
            (f32::from(output[channel]) * alpha + f32::from(original[channel]) * (1.0 - alpha))
                as u8
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmasked_pixels_keep_the_source_colors() {
        let source = RgbImage::from_pixel(2, 1, image::Rgb([10, 20, 30]));
        let inpainted = RgbImage::from_pixel(2, 1, image::Rgb([200, 200, 200]));
        let mask = GrayImage::from_raw(2, 1, vec![0, 255]).unwrap();

        let output = keep_unmasked_area(&inpainted, &source, &mask);

        assert_eq!(output.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(output.get_pixel(1, 0).0, [200, 200, 200]);
    }
}
