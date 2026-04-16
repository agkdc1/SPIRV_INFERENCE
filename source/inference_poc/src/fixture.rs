use anyhow::{Context, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub const INPUT: usize = 64;
pub const CONV_WEIGHT: usize = 12;
pub const BN_PARAM: usize = 16;
pub const CONV_OUT: usize = 24;
pub const FC_WEIGHT: usize = 240;
pub const FC_BIAS: usize = 10;
pub const LOGITS: usize = 10;
pub const EPSILON: f32 = 1.0e-5;

#[derive(Clone)]
pub struct Fixture {
    pub input: Vec<f32>,
    pub conv_weight: Vec<f32>,
    pub bn_param: Vec<f32>,
    pub fc_param: Vec<f32>,
    pub conv_output: Vec<f32>,
    pub batchnorm_output: Vec<f32>,
    pub relu_output: Vec<f32>,
    pub logits: Vec<f32>,
    pub softmax_output: Vec<f32>,
}

pub fn generate() -> Fixture {
    let input = (0..INPUT)
        .map(|i| (((i * 17) % 101) as f32 - 50.0) * 0.015625)
        .collect::<Vec<_>>();
    let conv_weight = (0..CONV_WEIGHT)
        .map(|i| (((i * 11) % 31) as f32 - 15.0) * 0.03125)
        .collect::<Vec<_>>();
    let mut bn_param = Vec::with_capacity(BN_PARAM);
    for c in 0..4 {
        bn_param.push(0.85 + c as f32 * 0.05);
        bn_param.push(-0.10 + c as f32 * 0.04);
        bn_param.push(-0.05 + c as f32 * 0.025);
        bn_param.push(0.70 + c as f32 * 0.08);
    }
    let fc_weight = (0..FC_WEIGHT)
        .map(|i| (((i * 7) % 43) as f32 - 21.0) * 0.0078125)
        .collect::<Vec<_>>();
    let fc_bias = (0..FC_BIAS)
        .map(|i| (i as f32 - 4.5) * 0.015625)
        .collect::<Vec<_>>();
    let mut fc_param = fc_weight;
    fc_param.extend_from_slice(&fc_bias);

    let conv_output = conv1d(&input, &conv_weight);
    let batchnorm_output = batchnorm(&conv_output, &bn_param);
    let relu_output = conv_output
        .iter()
        .zip(batchnorm_output.iter())
        .map(|(_, x)| if *x > 0.0 { *x } else { 0.0 })
        .collect::<Vec<_>>();
    let logits = fc(&relu_output, &fc_param);
    let softmax_output = softmax(&logits);

    Fixture {
        input,
        conv_weight,
        bn_param,
        fc_param,
        conv_output,
        batchnorm_output,
        relu_output,
        logits,
        softmax_output,
    }
}

pub fn write_fixture_dir(dir: &Path) -> Result<serde_json::Value> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let f = generate();
    let files = [
        ("input", "input.raw.f32", &f.input[..], vec![8, 8]),
        (
            "conv_weight",
            "conv_weight.raw.f32",
            &f.conv_weight[..],
            vec![3, 1, 4],
        ),
        (
            "batchnorm_param",
            "batchnorm_param.raw.f32",
            &f.bn_param[..],
            vec![4, 4],
        ),
        ("fc_param", "fc_param.raw.f32", &f.fc_param[..], vec![250]),
        (
            "conv_output_ref",
            "conv_output_ref.raw.f32",
            &f.conv_output[..],
            vec![6, 4],
        ),
        (
            "batchnorm_output_ref",
            "batchnorm_output_ref.raw.f32",
            &f.batchnorm_output[..],
            vec![6, 4],
        ),
        (
            "relu_output_ref",
            "relu_output_ref.raw.f32",
            &f.relu_output[..],
            vec![6, 4],
        ),
        ("logits_ref", "logits_ref.raw.f32", &f.logits[..], vec![10]),
        (
            "softmax_output_ref",
            "softmax_output_ref.raw.f32",
            &f.softmax_output[..],
            vec![10],
        ),
    ];
    let mut manifest_files = serde_json::Map::new();
    for (name, rel, values, shape) in files {
        let path = dir.join(rel);
        let bytes = f32_bytes(values);
        fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
        manifest_files.insert(
            name.to_string(),
            json!({
                "path": rel,
                "sha256": sha256(bytes),
                "elements": values.len(),
                "shape": shape,
            }),
        );
    }
    let manifest = json!({
        "format": "raw little-endian float32",
        "epsilon": EPSILON,
        "graph": ["conv1d", "batchnorm", "relu", "fc", "softmax"],
        "files": manifest_files,
        "notes": {
            "input": "8x8 tensor; Conv1D consumes the first column as a length-8 one-channel signal",
            "conv1d": "valid kernel width 3, 4 output channels, output shape 6x4",
            "fc": "24 features by 10 classes plus 10 bias values"
        }
    });
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(manifest)
}

fn conv1d(input: &[f32], weight: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; CONV_OUT];
    for r in 0..6 {
        for oc in 0..4 {
            let mut acc = 0.0_f32;
            for k in 0..3 {
                acc += input[(r + k) * 8] * weight[k * 4 + oc];
            }
            out[r * 4 + oc] = acc;
        }
    }
    out
}

fn batchnorm(input: &[f32], param: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; CONV_OUT];
    for i in 0..CONV_OUT {
        let c = i % 4;
        let scale = param[c * 4];
        let bias = param[c * 4 + 1];
        let mean = param[c * 4 + 2];
        let var = param[c * 4 + 3];
        out[i] = (input[i] - mean) / (var + EPSILON).sqrt() * scale + bias;
    }
    out
}

fn fc(input: &[f32], param: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; LOGITS];
    for cls in 0..LOGITS {
        let mut acc = param[FC_WEIGHT + cls];
        for i in 0..CONV_OUT {
            acc += input[i] * param[i * LOGITS + cls];
        }
        out[cls] = acc;
    }
    out
}

fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = input.iter().map(|x| (*x - max).exp()).collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|x| x / sum).collect()
}

pub fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}
