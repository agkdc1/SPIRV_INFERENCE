use crate::{
    kernels,
    model::{BuiltinOp, ModelInfo},
    scheduler,
    tensor::{compare, f32_bytes, sha256},
    vulkan::{groups, run_kernel, KernelRun},
};
use anyhow::{Context, Result};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn run_kernel_self_test(
    kernel_dir: &Path,
    device: &str,
    dump_dir: &Path,
) -> Result<serde_json::Value> {
    fs::create_dir_all(dump_dir)?;
    let input = vec![0.25_f32, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0, 4.0];
    let unused = vec![0.0_f32; input.len()];
    let rhs = vec![2.0_f32; input.len()];
    let cases: Vec<(&str, &str, Vec<i32>, Vec<f32>, Vec<f32>)> = vec![
        (
            "abs",
            "abs.spv",
            vec![input.len() as i32],
            unused.clone(),
            input.iter().copied().map(f32::abs).collect::<Vec<_>>(),
        ),
        (
            "exp",
            "exp.spv",
            vec![input.len() as i32],
            unused.clone(),
            input.iter().copied().map(f32::exp).collect::<Vec<_>>(),
        ),
        (
            "log",
            "log.spv",
            vec![input.len() as i32],
            unused.clone(),
            input.iter().copied().map(f32::ln).collect::<Vec<_>>(),
        ),
        (
            "neg",
            "neg.spv",
            vec![input.len() as i32],
            unused.clone(),
            input.iter().copied().map(|x| -x).collect::<Vec<_>>(),
        ),
        (
            "sqrt",
            "sqrt.spv",
            vec![input.len() as i32],
            unused.clone(),
            input.iter().copied().map(f32::sqrt).collect::<Vec<_>>(),
        ),
        (
            "rsqrt",
            "rsqrt.spv",
            vec![input.len() as i32],
            unused.clone(),
            input
                .iter()
                .copied()
                .map(|x| 1.0 / x.sqrt())
                .collect::<Vec<_>>(),
        ),
        (
            "sub",
            "sub.spv",
            vec![input.len() as i32],
            rhs.clone(),
            input
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        ),
        (
            "mul",
            "mul.spv",
            vec![input.len() as i32],
            rhs.clone(),
            input
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        ),
        (
            "div",
            "div.spv",
            vec![input.len() as i32],
            rhs.clone(),
            input
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a / b)
                .collect::<Vec<_>>(),
        ),
        (
            "pow",
            "pow.spv",
            vec![input.len() as i32],
            rhs.clone(),
            input
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a.powf(*b))
                .collect::<Vec<_>>(),
        ),
        (
            "min",
            "min.spv",
            vec![input.len() as i32],
            rhs.clone(),
            input
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a.min(*b))
                .collect::<Vec<_>>(),
        ),
        (
            "max",
            "max.spv",
            vec![input.len() as i32],
            rhs.clone(),
            input
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a.max(*b))
                .collect::<Vec<_>>(),
        ),
        (
            "leaky_relu",
            "leaky_relu.spv",
            vec![input.len() as i32, 0.1_f32.to_bits() as i32],
            unused.clone(),
            input
                .iter()
                .copied()
                .map(|x| x.max(0.1 * x))
                .collect::<Vec<_>>(),
        ),
        (
            "sigmoid",
            "sigmoid.spv",
            vec![input.len() as i32],
            unused.clone(),
            input
                .iter()
                .copied()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect::<Vec<_>>(),
        ),
        (
            "tanh",
            "tanh.spv",
            vec![input.len() as i32],
            unused.clone(),
            input
                .iter()
                .copied()
                .map(|x| {
                    let ep = x.exp();
                    let en = (-x).exp();
                    (ep - en) / (ep + en)
                })
                .collect::<Vec<_>>(),
        ),
    ];
    let mut mismatch_count = 0;
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    let mut case_reports = Vec::new();
    let mut device_report = json!(null);
    for (name, spv_name, push, input1, expected) in cases {
        let path = kernels::resolve_spv(kernel_dir, spv_name);
        let spv = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let out = run_kernel(
            device,
            &KernelRun {
                label: name,
                spv: &spv,
                input0: &input,
                input1: &input1,
                output_elements: input.len(),
                dispatch_x: groups(input.len(), 64),
                push: &push,
            },
        )?;
        fs::write(
            dump_dir.join(format!("{name}.raw.f32")),
            f32_bytes(&out.values),
        )?;
        let c = compare(&out.values, &expected, 1.0e-5);
        mismatch_count += c.mismatch_count;
        max_abs_error = max_abs_error.max(c.max_abs_error);
        max_rel_error = max_rel_error.max(c.max_rel_error);
        device_report = json!({
            "selector": device,
            "name": out.device_name,
            "vendor_id": format!("0x{:04x}", out.vendor_id),
            "device_id": format!("0x{:04x}", out.device_id)
        });
        case_reports.push(json!({
            "kernel": name,
            "spv": path,
            "elapsed_ms": out.elapsed_ms,
            "output_sha256": sha256(f32_bytes(&out.values)),
            "reference_sha256": sha256(f32_bytes(&expected)),
            "mismatch_count": c.mismatch_count,
            "max_abs_error": c.max_abs_error,
            "max_rel_error": c.max_rel_error
        }));
    }
    Ok(json!({
        "status": if mismatch_count == 0 { "pass" } else { "fail" },
        "device": device_report,
        "graph_execution_kind": "generic_tflite_loader_kernel_self_test_vulkan_spirv",
        "supported_kernel_registry": kernels::TFLITE_KERNELS,
        "cases": case_reports,
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "epsilon": 1.0e-5
    }))
}

pub fn run_model(
    model: &ModelInfo,
    kernel_dir: &Path,
    device: &str,
    dump_dir: &Path,
    input_f32: Option<&Path>,
) -> Result<serde_json::Value> {
    fs::create_dir_all(dump_dir)?;
    if model.operators.len() != 1 {
        return scheduler::run_graph(model, kernel_dir, device, dump_dir, input_f32);
    }
    let op = &model.operators[0];
    let output_tensor = op
        .outputs
        .first()
        .and_then(|idx| model.tensors.get(*idx as usize))
        .context("single-op model output tensor missing")?;
    let elements = output_tensor.shape.iter().try_fold(1_usize, |acc, dim| {
        if *dim <= 0 {
            anyhow::bail!("dynamic or invalid output dimension {dim}");
        }
        Ok(acc * *dim as usize)
    })?;
    let (input, input_source) = load_input(elements, input_f32)?;
    let unused = vec![0.0_f32; 1];
    let (kernel, push, expected) = match &op.builtin {
        BuiltinOp::Logistic => (
            "sigmoid.spv",
            vec![elements as i32],
            input
                .iter()
                .copied()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect::<Vec<_>>(),
        ),
        BuiltinOp::Tanh => (
            "tanh.spv",
            vec![elements as i32],
            input
                .iter()
                .copied()
                .map(|x| {
                    let ep = x.exp();
                    let en = (-x).exp();
                    (ep - en) / (ep + en)
                })
                .collect::<Vec<_>>(),
        ),
        BuiltinOp::LeakyRelu => (
            "leaky_relu.spv",
            vec![elements as i32, 0.1_f32.to_bits() as i32],
            input
                .iter()
                .copied()
                .map(|x| x.max(0.1 * x))
                .collect::<Vec<_>>(),
        ),
        other => anyhow::bail!("single-op model execution does not support {:?}", other),
    };
    let path = kernels::resolve_spv(kernel_dir, kernel);
    let spv = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let out = run_kernel(
        device,
        &KernelRun {
            label: "model_single_op",
            spv: &spv,
            input0: &input,
            input1: &unused,
            output_elements: elements,
            dispatch_x: groups(elements, 64),
            push: &push,
        },
    )?;
    fs::write(dump_dir.join("input.raw.f32"), f32_bytes(&input))?;
    fs::write(dump_dir.join("output.raw.f32"), f32_bytes(&out.values))?;
    fs::write(dump_dir.join("reference.raw.f32"), f32_bytes(&expected))?;
    let c = compare(&out.values, &expected, 1.0e-5);
    Ok(serde_json::json!({
        "status": if c.mismatch_count == 0 { "pass" } else { "fail" },
        "execution_kind": "model_driven_single_unary_tflite_op_vulkan_spirv",
        "op": format!("{:?}", op.builtin),
        "spv": path,
        "elements": elements,
        "device": {
            "selector": device,
            "name": out.device_name,
            "vendor_id": format!("0x{:04x}", out.vendor_id),
            "device_id": format!("0x{:04x}", out.device_id)
        },
        "elapsed_ms": out.elapsed_ms,
        "epsilon": 1.0e-5,
        "mismatch_count": c.mismatch_count,
        "max_abs_error": c.max_abs_error,
        "max_rel_error": c.max_rel_error,
        "input_sha256": sha256(f32_bytes(&input)),
        "input_source": input_source,
        "output_sha256": sha256(f32_bytes(&out.values)),
        "reference_sha256": sha256(f32_bytes(&expected)),
    }))
}

fn load_input(elements: usize, input_f32: Option<&Path>) -> Result<(Vec<f32>, serde_json::Value)> {
    if let Some(path) = input_f32 {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if bytes.len() != elements * 4 {
            anyhow::bail!(
                "{} has {} bytes, expected {}",
                path.display(),
                bytes.len(),
                elements * 4
            );
        }
        let values = crate::tensor::read_f32_bytes(&bytes)
            .with_context(|| format!("reading {}", path.display()))?;
        return Ok((
            values,
            json!({
                "kind": "cli_input_f32",
                "path": path,
                "bytes": bytes.len(),
                "sha256": sha256(&bytes)
            }),
        ));
    }
    if let Ok(path) = std::env::var("TFLITE_LOADER_INPUT_F32") {
        let path = PathBuf::from(path);
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        if bytes.len() != elements * 4 {
            anyhow::bail!(
                "{} has {} bytes, expected {}",
                path.display(),
                bytes.len(),
                elements * 4
            );
        }
        let values = crate::tensor::read_f32_bytes(&bytes)
            .with_context(|| format!("reading {}", path.display()))?;
        return Ok((
            values,
            json!({
                "kind": "env_input_f32",
                "env": "TFLITE_LOADER_INPUT_F32",
                "path": path,
                "bytes": bytes.len(),
                "sha256": sha256(&bytes)
            }),
        ));
    }
    let values = (0..elements)
        .map(|i| ((i as f32) - (elements as f32 / 2.0)) * 0.25)
        .collect::<Vec<_>>();
    Ok((
        values.clone(),
        json!({
            "kind": "synthetic_default",
            "elements": elements,
            "sha256": sha256(f32_bytes(&values))
        }),
    ))
}
