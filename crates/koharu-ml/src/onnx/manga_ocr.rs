//! Manga OCR through the Optimum ONNX export of the ViT/BERT encoder-decoder.
//!
//! The weights are the int8 dynamic quantization published by onnx-community,
//! a quarter of the FP32 download; on manga balloon crops it reads the same text.
//!
//! Configuration, image processing, tokenization, and beam scoring come from the
//! Torch port and the original checkpoint. The export has no past-key-value
//! inputs, so every beam step re-runs the decoder over the full prefix; Manga OCR
//! lines are short enough that this stays cheap.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use image::DynamicImage;
use ort::{session::Session, value::Tensor};

use crate::manga_ocr::{
    ViTImageProcessor,
    config::MangaOcrConfig,
    model::{BeamHypotheses, banned_ngram_tokens},
    processor::Tokenizer,
};

crate::model_repository!("mayocream/manga-ocr" @ "4380edba990b959c508752350955350c1c80c31c" {
    CONFIG = "config.json",
    PROCESSOR = "preprocessor_config.json",
    VOCABULARY = "vocab.txt",
});

crate::model_repository!("onnx-community/manga-ocr-base-ONNX" @ "f9023406bb2f6b17df67bc4a327c56ecd20611f0" {
    ENCODER = "onnx/encoder_model_int8.onnx",
    DECODER = "onnx/decoder_model_int8.onnx",
});

pub(super) const FILES: &[koharu_runtime::HuggingFaceFile<'static>] =
    &[CONFIG, PROCESSOR, VOCABULARY, ENCODER, DECODER];

#[derive(Debug)]
pub struct MangaOcrOnnx {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    config: MangaOcrConfig,
    processor: ViTImageProcessor,
    tokenizer: Tokenizer,
}

impl MangaOcrOnnx {
    pub async fn load() -> Result<Self> {
        let config_path = CONFIG
            .resolve()
            .await
            .context("failed to resolve Manga OCR config")?;
        let processor_path = PROCESSOR
            .resolve()
            .await
            .context("failed to resolve Manga OCR image processor")?;
        let vocabulary_path = VOCABULARY
            .resolve()
            .await
            .context("failed to resolve Manga OCR vocabulary")?;
        let encoder_path = ENCODER
            .resolve()
            .await
            .context("failed to resolve Manga OCR ONNX encoder")?;
        let decoder_path = DECODER
            .resolve()
            .await
            .context("failed to resolve Manga OCR ONNX decoder")?;

        let config = MangaOcrConfig::from_file(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let processor = ViTImageProcessor::from_file(&processor_path)
            .with_context(|| format!("failed to read {}", processor_path.display()))?;
        let tokenizer = Tokenizer::from_file(&vocabulary_path)
            .with_context(|| format!("failed to read {}", vocabulary_path.display()))?;
        ensure!(
            tokenizer.len() == config.decoder.vocab_size as usize,
            "Manga OCR vocabulary has {} entries but the decoder has {} outputs",
            tokenizer.len(),
            config.decoder.vocab_size
        );
        ensure!(config.num_beams > 1, "Manga OCR requires beam search");

        Ok(Self {
            encoder: Mutex::new(super::session(&encoder_path)?),
            decoder: Mutex::new(super::session(&decoder_path)?),
            config,
            processor,
            tokenizer,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<String> {
        let ([width, height], pixels) = self.processor.preprocess_host(image)?;
        let pixel_values =
            Tensor::from_array(([1usize, 3, height as usize, width as usize], pixels))?;
        let (hidden_shape, hidden) = {
            let mut encoder = self
                .encoder
                .lock()
                .map_err(|_| anyhow!("Manga OCR encoder lock is poisoned"))?;
            let outputs = encoder.run(ort::inputs!["pixel_values" => pixel_values])?;
            let (shape, values) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
            (shape.to_vec(), values.to_vec())
        };
        ensure!(
            hidden_shape.len() == 3 && hidden_shape[0] == 1,
            "unexpected Manga OCR encoder output shape {hidden_shape:?}"
        );
        let token_ids =
            self.beam_search(&hidden, hidden_shape[1] as usize, hidden_shape[2] as usize)?;
        self.tokenizer.decode(&token_ids)
    }

    /// Mirrors `manga_ocr::model::Model::beam_search` on host buffers.
    fn beam_search(&self, hidden: &[f32], sequence: usize, width: usize) -> Result<Vec<i64>> {
        let config = &self.config;
        let num_beams = config.num_beams;
        let vocab_size = config.decoder.vocab_size as usize;
        let encoder_hidden_states = hidden.repeat(num_beams);

        let mut sequences = vec![vec![config.decoder_start_token_id]; num_beams];
        let mut beam_scores = vec![-1.0e9f32; num_beams];
        beam_scores[0] = 0.0;
        let mut hypotheses =
            BeamHypotheses::new(num_beams, config.length_penalty, config.early_stopping);
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| anyhow!("Manga OCR decoder lock is poisoned"))?;

        while sequences[0].len() < config.max_length {
            let current_length = sequences[0].len();
            let input_ids = Tensor::from_array(([num_beams, current_length], sequences.concat()))?;
            let states =
                Tensor::from_array(([num_beams, sequence, width], encoder_hidden_states.clone()))?;
            let outputs = decoder.run(ort::inputs![
                "input_ids" => input_ids,
                "encoder_hidden_states" => states,
            ])?;
            let (logits_shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            ensure!(
                logits_shape.as_ref()
                    == [num_beams as i64, current_length as i64, vocab_size as i64],
                "unexpected Manga OCR decoder output shape {logits_shape:?}"
            );

            let mut scores = Vec::with_capacity(num_beams * vocab_size);
            for (beam, sequence) in sequences.iter().enumerate() {
                let start = (beam * current_length + current_length - 1) * vocab_size;
                let mut row = log_softmax(&logits[start..start + vocab_size]);
                for token in banned_ngram_tokens(sequence, config.no_repeat_ngram_size) {
                    row[token as usize] = f32::NEG_INFINITY;
                }
                scores.extend(row.into_iter().map(|score| score + beam_scores[beam]));
            }
            let mut candidates = (0..scores.len()).collect::<Vec<_>>();
            let take = 2 * num_beams;
            candidates.select_nth_unstable_by(take - 1, |&a, &b| scores[b].total_cmp(&scores[a]));
            candidates.truncate(take);
            candidates.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

            let mut next_sequences = Vec::with_capacity(num_beams);
            let mut next_scores = Vec::with_capacity(num_beams);
            for (rank, flat_index) in candidates.into_iter().enumerate() {
                let score = scores[flat_index];
                let beam_index = flat_index / vocab_size;
                let token_id = (flat_index % vocab_size) as i64;
                if token_id == config.eos_token_id {
                    if rank < num_beams {
                        hypotheses.add(sequences[beam_index].clone(), f64::from(score));
                    }
                    continue;
                }
                let mut next = sequences[beam_index].clone();
                next.push(token_id);
                next_sequences.push(next);
                next_scores.push(score);
                if next_sequences.len() == num_beams {
                    break;
                }
            }
            ensure!(
                next_sequences.len() == num_beams,
                "Manga OCR beam search could not fill the next beam"
            );
            sequences = next_sequences;
            beam_scores = next_scores;

            if hypotheses.is_done(f64::from(beam_scores[0]), current_length) {
                break;
            }
        }

        if !hypotheses.done {
            for (sequence, score) in sequences.into_iter().zip(beam_scores) {
                hypotheses.add(sequence, f64::from(score));
            }
        }
        let mut best = hypotheses.best()?;
        if best.len() < config.max_length {
            best.push(config.eos_token_id);
        }
        Ok(best)
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|value| (value - max).exp()).sum::<f32>();
    let log_sum = sum.ln();
    logits.iter().map(|value| value - max - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_normalizes_without_overflow() {
        let scores = log_softmax(&[1000.0, 1000.0]);

        assert!(
            scores
                .iter()
                .all(|score| (score - (0.5f32).ln()).abs() < 1e-6)
        );
    }
}
