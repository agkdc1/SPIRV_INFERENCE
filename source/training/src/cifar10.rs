use anyhow::{bail, Context, Result};
use std::path::Path;

const CIFAR10_URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const TRAIN_BATCHES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];
const TEST_BATCH: &str = "test_batch.bin";
const RECORD_SIZE: usize = 3073; // 1 label + 3072 pixels
const IMG_C: usize = 3;
const IMG_H: usize = 32;
const IMG_W: usize = 32;
const IMG_PIXELS: usize = IMG_C * IMG_H * IMG_W;

// CIFAR-10 per-channel normalization constants
const MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];
const STD: [f32; 3] = [0.2470, 0.2435, 0.2616];

pub struct Cifar10Data {
    /// Train images: [N, C, H, W] in CHW order, normalized
    pub train_images: Vec<f32>, // 50000 * 3072
    pub train_labels: Vec<u8>,  // 50000
    /// Test images: same format
    pub test_images: Vec<f32>,  // 10000 * 3072
    pub test_labels: Vec<u8>,   // 10000
}

pub fn load(data_dir: &Path) -> Result<Cifar10Data> {
    let cifar_dir = data_dir.join("cifar-10-batches-bin");

    // Download and extract if missing
    if !cifar_dir.exists() {
        let tar_path = data_dir.join("cifar-10-binary.tar.gz");
        if !tar_path.exists() {
            println!("[cifar10] downloading {}", CIFAR10_URL);
            let status = std::process::Command::new("curl")
                .args(["-sL", "-o", tar_path.to_str().unwrap(), CIFAR10_URL])
                .status()
                .context("curl")?;
            if !status.success() {
                bail!("curl failed for CIFAR-10");
            }
        }
        println!("[cifar10] extracting...");
        let status = std::process::Command::new("tar")
            .args(["xzf", tar_path.to_str().unwrap(), "-C", data_dir.to_str().unwrap()])
            .status()
            .context("tar")?;
        if !status.success() {
            bail!("tar extraction failed");
        }
    }

    // Parse training data
    let mut train_images = Vec::with_capacity(50000 * IMG_PIXELS);
    let mut train_labels = Vec::with_capacity(50000);
    for batch_file in &TRAIN_BATCHES {
        let path = cifar_dir.join(batch_file);
        parse_batch(&path, &mut train_images, &mut train_labels)?;
    }

    // Parse test data
    let mut test_images = Vec::with_capacity(10000 * IMG_PIXELS);
    let mut test_labels = Vec::with_capacity(10000);
    parse_batch(&cifar_dir.join(TEST_BATCH), &mut test_images, &mut test_labels)?;

    println!(
        "[cifar10] loaded: {} train, {} test",
        train_labels.len(),
        test_labels.len()
    );

    Ok(Cifar10Data {
        train_images,
        train_labels,
        test_images,
        test_labels,
    })
}

fn parse_batch(path: &Path, images: &mut Vec<f32>, labels: &mut Vec<u8>) -> Result<()> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let num_records = data.len() / RECORD_SIZE;
    for i in 0..num_records {
        let offset = i * RECORD_SIZE;
        let label = data[offset];
        labels.push(label);

        // CIFAR-10 binary: 1024 R + 1024 G + 1024 B (planar)
        // Output: CHW order, normalized per channel
        let pixel_start = offset + 1;
        for c in 0..IMG_C {
            let chan_start = pixel_start + c * IMG_H * IMG_W;
            for p in 0..IMG_H * IMG_W {
                let raw = data[chan_start + p] as f32 / 255.0;
                let normalized = (raw - MEAN[c]) / STD[c];
                images.push(normalized);
            }
        }
    }
    Ok(())
}

/// Random horizontal flip (CPU-side augmentation)
/// Flips image in-place with 50% probability
pub fn random_hflip(img: &mut [f32], rng: &mut impl rand::Rng) {
    if rng.gen::<bool>() {
        // img is [C, H, W] with C=3, H=32, W=32
        for c in 0..IMG_C {
            for h in 0..IMG_H {
                for w in 0..IMG_W / 2 {
                    let left = c * IMG_H * IMG_W + h * IMG_W + w;
                    let right = c * IMG_H * IMG_W + h * IMG_W + (IMG_W - 1 - w);
                    img.swap(left, right);
                }
            }
        }
    }
}

/// Create one-hot encoded label batch
pub fn one_hot(labels: &[u8], num_classes: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; labels.len() * num_classes];
    for (i, &l) in labels.iter().enumerate() {
        out[i * num_classes + l as usize] = 1.0;
    }
    out
}
