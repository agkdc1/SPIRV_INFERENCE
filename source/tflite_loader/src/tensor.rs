use sha2::{Digest, Sha256};

pub fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

pub fn read_f32_bytes(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!(
            "f32 byte input length {} is not divisible by four",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

pub fn sha256(bytes: impl AsRef<[u8]>) -> String {
    let mut h = Sha256::new();
    h.update(bytes.as_ref());
    format!("{:x}", h.finalize())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Compare {
    pub mismatch_count: usize,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
}

pub fn compare(actual: &[f32], expected: &[f32], epsilon: f32) -> Compare {
    let mut out = Compare {
        mismatch_count: 0,
        max_abs_error: 0.0,
        max_rel_error: 0.0,
    };
    for (a, e) in actual.iter().zip(expected.iter()) {
        let abs = (*a - *e).abs();
        let rel = abs / e.abs().max(1.0e-30);
        out.max_abs_error = out.max_abs_error.max(abs);
        out.max_rel_error = out.max_rel_error.max(rel);
        if abs > epsilon {
            out.mismatch_count += 1;
        }
    }
    if actual.len() != expected.len() {
        out.mismatch_count += actual.len().abs_diff(expected.len());
    }
    out
}
