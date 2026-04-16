use anyhow::{bail, Context, Result};
use std::path::Path;

const MNIST_BASE: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";

const FILES: [&str; 4] = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
];

pub struct MnistData {
    pub train_images: Vec<f32>,  // 60000 x 784, normalized /255
    pub train_labels: Vec<u8>,   // 60000
    pub test_images: Vec<f32>,   // 10000 x 784
    pub test_labels: Vec<u8>,    // 10000
}

pub fn load(data_dir: &Path) -> Result<MnistData> {
    std::fs::create_dir_all(data_dir)?;
    // Download if missing
    for fname in &FILES {
        let gz_path = data_dir.join(fname);
        let raw_name = fname.trim_end_matches(".gz");
        let raw_path = data_dir.join(raw_name);
        if !raw_path.exists() {
            if !gz_path.exists() {
                let url = format!("{}/{}", MNIST_BASE, fname);
                println!("[mnist] downloading {}", url);
                let status = std::process::Command::new("curl")
                    .args(["-sL", "-o", gz_path.to_str().unwrap(), &url])
                    .status()
                    .context("curl")?;
                if !status.success() {
                    bail!("curl failed for {}", url);
                }
            }
            println!("[mnist] decompressing {}", fname);
            let status = std::process::Command::new("gunzip")
                .args(["-f", gz_path.to_str().unwrap()])
                .status()
                .context("gunzip")?;
            if !status.success() {
                bail!("gunzip failed for {}", fname);
            }
        }
    }

    let train_img = parse_images(&data_dir.join("train-images-idx3-ubyte"))?;
    let train_lbl = parse_labels(&data_dir.join("train-labels-idx1-ubyte"))?;
    let test_img = parse_images(&data_dir.join("t10k-images-idx3-ubyte"))?;
    let test_lbl = parse_labels(&data_dir.join("t10k-labels-idx1-ubyte"))?;

    println!(
        "[mnist] loaded: {} train, {} test",
        train_lbl.len(),
        test_lbl.len()
    );

    Ok(MnistData {
        train_images: train_img,
        train_labels: train_lbl,
        test_images: test_img,
        test_labels: test_lbl,
    })
}

fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn parse_images(path: &Path) -> Result<Vec<f32>> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let magic = read_u32_be(&data, 0);
    if magic != 2051 {
        bail!("bad image magic: {}", magic);
    }
    let count = read_u32_be(&data, 4) as usize;
    let rows = read_u32_be(&data, 8) as usize;
    let cols = read_u32_be(&data, 12) as usize;
    let pixels = rows * cols;
    let mut images = Vec::with_capacity(count * pixels);
    for i in 0..count {
        let offset = 16 + i * pixels;
        for j in 0..pixels {
            images.push(data[offset + j] as f32 / 255.0);
        }
    }
    Ok(images)
}

fn parse_labels(path: &Path) -> Result<Vec<u8>> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let magic = read_u32_be(&data, 0);
    if magic != 2049 {
        bail!("bad label magic: {}", magic);
    }
    let count = read_u32_be(&data, 4) as usize;
    Ok(data[8..8 + count].to_vec())
}

/// Create one-hot encoded label batch
pub fn one_hot(labels: &[u8], num_classes: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; labels.len() * num_classes];
    for (i, &l) in labels.iter().enumerate() {
        out[i * num_classes + l as usize] = 1.0;
    }
    out
}
