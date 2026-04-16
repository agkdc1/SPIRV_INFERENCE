use crate::ir::{HloModule, HloOp};
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use tflite_loader::{
    tensor::{compare, f32_bytes, read_f32_bytes, sha256},
    vulkan::{groups, run_kernel, KernelRun},
};

pub fn supported_ops() -> BTreeSet<&'static str> {
    [
        "add",
        "abs",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "reshape",
        "transpose",
        "broadcast_in_dim",
        "broadcast",
        "concatenate",
        "convolution",
        "dot",
        "dot_general",
        "reduce",
        "reduce-window",
        "exponential",
        "log",
        "sqrt",
        "rsqrt",
        "negate",
        "tanh",
        "logistic",
        "compare",
        "select",
        "convert",
        "slice",
        "dynamic-slice",
        "dynamic-update-slice",
        "pad",
        "iota",
        "gather",
        "scatter",
        "floor",
        "ceil",
        "round-nearest-even",
        "sine",
        "cosine",
        "remainder",
        "power",
        "constant",
        "parameter",
        "tuple",
        "get-tuple-element",
    ]
    .into_iter()
    .collect()
}

pub fn op_histogram(module: &HloModule) -> BTreeMap<String, usize> {
    let mut hist = BTreeMap::new();
    for op in &module.ops {
        *hist.entry(op.opcode.clone()).or_insert(0) += 1;
    }
    hist
}

pub fn validate_supported(module: &HloModule) -> Result<()> {
    let supported = supported_ops();
    let unsupported: Vec<_> = module
        .ops
        .iter()
        .map(|o| o.opcode.as_str())
        .filter(|op| !supported.contains(op))
        .collect();
    if !unsupported.is_empty() {
        bail!("unsupported StableHLO ops: {}", unsupported.join(", "));
    }
    Ok(())
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LoweredKernel {
    pub opcode: String,
    pub spv_name: String,
    pub arity: usize,
    pub output_elements: usize,
}

#[derive(Debug, Clone)]
struct KernelMapping {
    spv_name: &'static str,
    arity: usize,
}

pub fn executable_ops(module: &HloModule) -> Vec<&HloOp> {
    module
        .ops
        .iter()
        .filter(|op| kernel_mapping(&op.opcode).is_some())
        .collect()
}

pub fn lower_single_primitive(module: &HloModule, output_elements: usize) -> Result<LoweredKernel> {
    validate_supported(module)?;
    let executable = executable_ops(module);
    let primitive = executable
        .iter()
        .copied()
        .filter(|op| !matches!(op.opcode.as_str(), "reshape" | "convert"))
        .collect::<Vec<_>>();
    let selected = if primitive.len() == 1 {
        primitive[0]
    } else if executable.len() == 1 {
        executable[0]
    } else {
        bail!(
            "single-primitive HLO runner requires exactly one executable non-copy primitive, found {} executable ops and {} non-copy primitives",
            executable.len(),
            primitive.len()
        );
    };
    let op = selected;
    let mapping = kernel_mapping(&op.opcode).expect("filtered by executable_ops");
    Ok(LoweredKernel {
        opcode: op.opcode.clone(),
        spv_name: mapping.spv_name.to_string(),
        arity: mapping.arity,
        output_elements,
    })
}

pub fn run_single_primitive(
    module: &HloModule,
    kernel_dir: &Path,
    device: &str,
    input0_path: &Path,
    input1_path: Option<&Path>,
    output_path: Option<&Path>,
    expected_path: Option<&Path>,
    epsilon: f32,
) -> Result<serde_json::Value> {
    let input0_bytes = std::fs::read(input0_path)?;
    let input0 = read_f32_bytes(&input0_bytes)?;
    let input1 = if let Some(path) = input1_path {
        read_f32_bytes(&std::fs::read(path)?)?
    } else {
        vec![0.0_f32; 1]
    };
    let elements = input0.len();
    let lowered = lower_single_primitive(module, elements)?;
    if lowered.arity == 2 && input1.len() != elements {
        bail!(
            "binary HLO primitive {} requires equal input lengths, got {} and {}",
            lowered.opcode,
            elements,
            input1.len()
        );
    }
    let spv_path = kernel_dir
        .join(kernel_subdir(&lowered.spv_name))
        .join("spv")
        .join(&lowered.spv_name);
    let spv = std::fs::read(&spv_path)?;
    let second = if lowered.arity == 2 {
        input1.as_slice()
    } else {
        &[0.0_f32]
    };
    let out = run_kernel(
        device,
        &KernelRun {
            label: &format!("xla_hlo_{}", lowered.opcode),
            spv: &spv,
            input0: &input0,
            input1: second,
            output_elements: elements,
            dispatch_x: groups(elements, 128),
            push: &[elements as i32],
        },
    )?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, f32_bytes(&out.values))?;
    }
    let parity = if let Some(path) = expected_path {
        let expected = read_f32_bytes(&std::fs::read(path)?)?;
        let cmp = compare(&out.values, &expected, epsilon);
        serde_json::json!({
            "status": if cmp.mismatch_count == 0 { "pass" } else { "fail" },
            "epsilon": epsilon,
            "mismatch_count": cmp.mismatch_count,
            "max_abs_error": cmp.max_abs_error,
            "max_rel_error": cmp.max_rel_error,
            "expected_sha256": sha256(std::fs::read(path)?)
        })
    } else {
        serde_json::json!({"status":"not_requested"})
    };
    Ok(serde_json::json!({
        "status": if parity["status"] == "fail" { "fail" } else { "pass" },
        "lowered_kernel": lowered,
        "spv_path": spv_path,
        "input0": {"path": input0_path, "elements": input0.len(), "sha256": sha256(input0_bytes)},
        "input1": input1_path.map(|p| serde_json::json!({"path": p, "elements": input1.len(), "sha256": sha256(std::fs::read(p).unwrap_or_default())})),
        "output": {
            "path": output_path,
            "elements": out.values.len(),
            "sha256": sha256(f32_bytes(&out.values))
        },
        "device": {
            "name": out.device_name,
            "vendor_id": out.vendor_id,
            "device_id": out.device_id,
            "elapsed_ms": out.elapsed_ms
        },
        "parity": parity
    }))
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphStep {
    pub result: String,
    pub opcode: String,
    pub execution: String,
    pub elements: usize,
    pub spv_name: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DispatchRecord {
    result: String,
    opcode: String,
    spv_name: String,
    elements: usize,
    elapsed_ms: f64,
    device_name: String,
    vendor_id: u32,
    device_id: u32,
}

#[derive(Debug, Clone)]
struct Tensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Tensor {
    fn new(shape: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        let expected = shape_elements(&shape).unwrap_or(values.len());
        if expected != values.len() {
            bail!(
                "tensor shape {:?} has {} elements but data has {}",
                shape,
                expected,
                values.len()
            );
        }
        Ok(Self { shape, values })
    }
}

pub fn run_graph(
    module: &HloModule,
    kernel_dir: &Path,
    device: &str,
    input_paths: &[&Path],
    output_path: Option<&Path>,
    expected_path: Option<&Path>,
    epsilon: f32,
) -> Result<serde_json::Value> {
    validate_supported(module)?;
    let mut values: BTreeMap<String, Tensor> = BTreeMap::new();
    let mut steps = Vec::new();
    let mut dispatches = Vec::new();
    let mut parameter_index = 0usize;
    let mut last_result: Option<String> = None;
    let device_info = probe_vulkan_device(kernel_dir, device).ok();

    for op in &module.ops {
        let Some(result) = op.result.clone() else {
            continue;
        };
        let output_shape = if op.shape.is_empty() {
            op.result_type
                .as_deref()
                .and_then(|ty| type_shape(Some(ty)))
                .unwrap_or_else(Vec::new)
        } else {
            op.shape.clone()
        };
        let output_elements = shape_elements(&output_shape)
            .or_else(|| {
                op.operands
                    .first()
                    .and_then(|name| values.get(name))
                    .map(|t| t.values.len())
            })
            .unwrap_or(1);

        match op.opcode.as_str() {
            "parameter" => {
                let index = parameter_number(&op.raw).unwrap_or(parameter_index);
                let path = input_paths.get(index).with_context(|| {
                    format!("missing --inputs-f32 entry for parameter({index})")
                })?;
                let input = read_f32_bytes(&std::fs::read(path)?)?;
                let tensor = Tensor::new(output_shape.clone(), input.clone())?;
                values.insert(result.clone(), tensor);
                parameter_index = parameter_index.max(index + 1);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: format!("loaded parameter({index}) from {}", path.display()),
                    elements: input.len(),
                    spv_name: None,
                });
            }
            "constant" => {
                let scalar = constant_scalar(&op.raw)
                    .with_context(|| format!("parsing scalar constant from {}", op.raw))?;
                let data = vec![scalar; output_elements.max(1)];
                values.insert(result.clone(), Tensor::new(output_shape.clone(), data)?);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "materialized scalar constant".to_string(),
                    elements: output_elements,
                    spv_name: None,
                });
            }
            "broadcast" | "broadcast_in_dim" => {
                let src = operand_tensor(&values, op, 0)?;
                let dims = parse_usize_list(op.attributes.get("dimensions"));
                let data = broadcast_tensor(src, &output_shape, &dims)?;
                values.insert(result.clone(), data);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "cpu_shaped_broadcast".to_string(),
                    elements: output_elements,
                    spv_name: None,
                });
            }
            "reshape" | "convert" => {
                let src = operand_tensor(&values, op, 0)?.clone();
                let data = Tensor::new(output_shape.clone(), src.values)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "metadata_reshape".to_string(),
                    elements,
                    spv_name: None,
                });
            }
            "tuple" | "get-tuple-element" => {
                let src = operand_tensor(&values, op, 0)?.clone();
                let elements = src.values.len();
                values.insert(result.clone(), src);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "tuple_alias".to_string(),
                    elements,
                    spv_name: None,
                });
            }
            "transpose" => {
                let src = operand_tensor(&values, op, 0)?;
                let dims = parse_usize_list(op.attributes.get("dimensions"));
                let data = transpose_tensor(src, &output_shape, &dims)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "cpu_transpose".to_string(),
                    elements,
                    spv_name: None,
                });
            }
            "pad" => {
                let src = operand_tensor(&values, op, 0)?;
                let pad_value = operand_tensor(&values, op, 1)?
                    .values
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let padding = parse_padding(op.attributes.get("padding"), src.shape.len())?;
                let (data, dispatch) =
                    vulkan_pad(kernel_dir, device, &result, src, &output_shape, &padding, pad_value)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                dispatches.push(dispatch);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "vulkan_pad".to_string(),
                    elements,
                    spv_name: Some("pad.spv".to_string()),
                });
            }
            "convolution" => {
                let lhs = operand_tensor(&values, op, 0)?;
                let rhs = operand_tensor(&values, op, 1)?;
                let (data, dispatch) =
                    vulkan_convolution(kernel_dir, device, &result, lhs, rhs, &output_shape, op)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                dispatches.push(dispatch);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "vulkan_convolution_nhwc_hwio".to_string(),
                    elements,
                    spv_name: Some("hlo_conv_nhwc_hwio.spv".to_string()),
                });
            }
            "reduce-window" => {
                let src = operand_tensor(&values, op, 0)?;
                let init = operand_tensor(&values, op, 1)?
                    .values
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let (data, dispatch) = vulkan_reduce_window(
                    kernel_dir,
                    device,
                    &result,
                    src,
                    &output_shape,
                    op,
                    init,
                )?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                dispatches.push(dispatch);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "vulkan_reduce_window".to_string(),
                    elements,
                    spv_name: Some("hlo_reduce_window_rank4.spv".to_string()),
                });
            }
            "reduce" => {
                let src = operand_tensor(&values, op, 0)?;
                let init = operand_tensor(&values, op, 1)?
                    .values
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let dims = parse_usize_list(op.attributes.get("dimensions"));
                if src.shape.len() > 4 {
                    // CPU fallback for rank > 4 (e.g. GroupNorm rank-5 reduce)
                    let data = reduce_tensor(src, &output_shape, &dims, op, init)?;
                    let elements = data.values.len();
                    values.insert(result.clone(), data);
                    steps.push(GraphStep {
                        result: result.clone(),
                        opcode: op.opcode.clone(),
                        execution: "cpu_reduce_rank5".to_string(),
                        elements,
                        spv_name: None,
                    });
                } else {
                    let (data, dispatch) = vulkan_reduce(
                        kernel_dir,
                        device,
                        &result,
                        src,
                        &output_shape,
                        &dims,
                        op,
                        init,
                    )?;
                    let elements = data.values.len();
                    values.insert(result.clone(), data);
                    dispatches.push(dispatch);
                    steps.push(GraphStep {
                        result: result.clone(),
                        opcode: op.opcode.clone(),
                        execution: "vulkan_reduce".to_string(),
                        elements,
                        spv_name: Some("hlo_reduce_rank4.spv".to_string()),
                    });
                }
            }
            "dot" | "dot_general" => {
                let lhs = operand_tensor(&values, op, 0)?;
                let rhs = operand_tensor(&values, op, 1)?;
                let (data, dispatch) =
                    vulkan_dot_general(kernel_dir, device, &result, lhs, rhs, &output_shape, op)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                dispatches.push(dispatch);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "vulkan_dot_general".to_string(),
                    elements,
                    spv_name: Some("hlo_dot_batched.spv".to_string()),
                });
            }
            "compare" => {
                let lhs = operand_tensor(&values, op, 0)?;
                let rhs = operand_tensor(&values, op, 1)?;
                let data = binary_tensor(lhs, rhs, &output_shape, |a, b| {
                    let pass = match op.attributes.get("direction").map(String::as_str) {
                        Some("LT") => a < b,
                        Some("LE") => a <= b,
                        Some("EQ") => a == b,
                        Some("NE") => a != b,
                        Some("GE") => a >= b,
                        _ => a > b,
                    };
                    if pass {
                        1.0
                    } else {
                        0.0
                    }
                })?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "cpu_compare_predicate_as_f32".to_string(),
                    elements,
                    spv_name: None,
                });
            }
            "select" => {
                let pred = operand_tensor(&values, op, 0)?;
                let on_true = operand_tensor(&values, op, 1)?;
                let on_false = operand_tensor(&values, op, 2)?;
                let data = select_tensor(pred, on_true, on_false, &output_shape)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "cpu_select".to_string(),
                    elements,
                    spv_name: None,
                });
            }
            "concatenate" => {
                let dim = parse_usize_list(op.attributes.get("dimensions"))
                    .first()
                    .copied()
                    .unwrap_or(0);
                let inputs: Vec<Tensor> = (0..op.operands.len())
                    .map(|i| operand_tensor(&values, op, i).cloned())
                    .collect::<Result<Vec<_>>>()?;
                let data = cpu_concatenate(&inputs, &output_shape, dim)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "cpu_concatenate".to_string(),
                    elements,
                    spv_name: None,
                });
            }
            "slice" => {
                let src = operand_tensor(&values, op, 0)?;
                let (starts, limits, strides_s) = parse_slice_spec(&op.raw)?;
                let data = cpu_slice(src, &output_shape, &starts, &limits, &strides_s)?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "cpu_slice".to_string(),
                    elements,
                    spv_name: None,
                });
            }
            _ => {
                let mapping = kernel_mapping(&op.opcode).with_context(|| {
                    format!("no SPIR-V lowering for executable HLO op {}", op.opcode)
                })?;
                let (data, dispatch) = vulkan_elementwise(
                    kernel_dir,
                    device,
                    &result,
                    &values,
                    op,
                    &output_shape,
                    &mapping,
                )
                .with_context(|| format!("lowering HLO op {} through SPIR-V", op.opcode))?;
                let elements = data.values.len();
                values.insert(result.clone(), data);
                dispatches.push(dispatch);
                steps.push(GraphStep {
                    result: result.clone(),
                    opcode: op.opcode.clone(),
                    execution: "vulkan_elementwise".to_string(),
                    elements,
                    spv_name: Some(mapping.spv_name.to_string()),
                });
            }
        }
        // NaN tracing: detect first op producing NaN
        if let Some(tensor) = values.get(&result) {
            let nan_count = tensor.values.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                let first_nan = tensor.values.iter().position(|v| v.is_nan()).unwrap_or(0);
                eprintln!(
                    "NaN TRACE: {} ({}) produced {}/{} NaN values (first at index {}), shape={:?}",
                    result, op.opcode, nan_count, tensor.values.len(), first_nan, tensor.shape
                );
            }
        }
        last_result = Some(result);
    }

    let final_name = last_result.context("HLO module did not produce a result")?;
    let final_values = values
        .get(&final_name)
        .with_context(|| format!("final result {final_name} not materialized"))?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, f32_bytes(&final_values.values))?;
    }
    let parity = if let Some(path) = expected_path {
        let expected_bytes = std::fs::read(path)?;
        let expected = read_f32_bytes(&expected_bytes)?;
        let cmp = compare(&final_values.values, &expected, epsilon);
        serde_json::json!({
            "status": if cmp.mismatch_count == 0 { "pass" } else { "fail" },
            "epsilon": epsilon,
            "mismatch_count": cmp.mismatch_count,
            "max_abs_error": cmp.max_abs_error,
            "max_rel_error": cmp.max_rel_error,
            "expected_sha256": sha256(expected_bytes)
        })
    } else {
        serde_json::json!({"status":"not_requested"})
    };
    Ok(serde_json::json!({
        "status": if parity["status"] == "fail" { "fail" } else { "pass" },
        "mode": "shaped_hlo_graph",
        "final_result": final_name,
        "steps": steps,
        "dispatch_count": dispatches.len(),
        "dispatches": dispatches,
        "device": device_info,
        "output": {
            "path": output_path,
            "elements": final_values.values.len(),
            "shape": final_values.shape,
            "sha256": sha256(f32_bytes(&final_values.values))
        },
        "parity": parity
    }))
}

fn spv_bytes(kernel_dir: &Path, subdir: &str, spv_name: &str) -> Result<Vec<u8>> {
    let path = kernel_dir.join(subdir).join("spv").join(spv_name);
    std::fs::read(&path).with_context(|| format!("reading {}", path.display()))
}

fn dispatch_record(
    result: &str,
    opcode: &str,
    spv_name: &str,
    elements: usize,
    out: &tflite_loader::vulkan::KernelOutput,
) -> DispatchRecord {
    DispatchRecord {
        result: result.to_string(),
        opcode: opcode.to_string(),
        spv_name: spv_name.to_string(),
        elements,
        elapsed_ms: out.elapsed_ms,
        device_name: out.device_name.clone(),
        vendor_id: out.vendor_id,
        device_id: out.device_id,
    }
}

fn vulkan_elementwise(
    kernel_dir: &Path,
    device: &str,
    result: &str,
    values: &BTreeMap<String, Tensor>,
    op: &HloOp,
    output_shape: &[usize],
    mapping: &KernelMapping,
) -> Result<(Tensor, DispatchRecord)> {
    let lhs = operand_tensor(values, op, 0)?;
    let out_elements = shape_elements(output_shape).unwrap_or(lhs.values.len());
    let lhs_b = if lhs.values.len() == out_elements {
        lhs.clone()
    } else {
        broadcast_tensor(
            lhs,
            output_shape,
            &infer_broadcast_dims(&lhs.shape, output_shape),
        )?
    };
    let rhs_b;
    let rhs_values = if mapping.arity == 2 {
        let rhs = operand_tensor(values, op, 1)?;
        rhs_b = if rhs.values.len() == out_elements {
            rhs.clone()
        } else {
            broadcast_tensor(
                rhs,
                output_shape,
                &infer_broadcast_dims(&rhs.shape, output_shape),
            )?
        };
        rhs_b.values.as_slice()
    } else {
        &[0.0_f32][..]
    };
    if lhs_b.values.len() != out_elements {
        bail!(
            "SPIR-V elementwise lhs length {} does not match output {} for {}",
            lhs_b.values.len(),
            out_elements,
            op.opcode
        );
    }
    if mapping.arity == 2 && rhs_values.len() != out_elements {
        bail!(
            "SPIR-V elementwise rhs length {} does not match output {} for {}",
            rhs_values.len(),
            out_elements,
            op.opcode
        );
    }
    let spv = spv_bytes(
        kernel_dir,
        kernel_subdir(mapping.spv_name),
        mapping.spv_name,
    )?;
    let out = run_kernel(
        device,
        &KernelRun {
            label: &format!("xla_hlo_{}", op.opcode),
            spv: &spv,
            input0: &lhs_b.values,
            input1: rhs_values,
            output_elements: out_elements,
            dispatch_x: groups(out_elements, 128),
            push: &[out_elements as i32],
        },
    )?;
    let tensor = Tensor::new(output_shape.to_vec(), out.values.clone())?;
    let record = dispatch_record(result, &op.opcode, mapping.spv_name, out_elements, &out);
    Ok((tensor, record))
}

fn vulkan_convolution(
    kernel_dir: &Path,
    device: &str,
    result: &str,
    lhs: &Tensor,
    rhs: &Tensor,
    output_shape: &[usize],
    op: &HloOp,
) -> Result<(Tensor, DispatchRecord)> {
    if lhs.shape.len() != 4 || rhs.shape.len() != 4 || output_shape.len() != 4 {
        bail!(
            "convolution supports rank-4 NHWC/HWIO only, got {:?} {:?} -> {:?}",
            lhs.shape,
            rhs.shape,
            output_shape
        );
    }
    if op.attributes.get("dim_labels").map(String::as_str) != Some("b01f_01io->b01f") {
        bail!(
            "unsupported convolution dim_labels {:?}",
            op.attributes.get("dim_labels")
        );
    }
    let window = op.attributes.get("window");
    let stride = {
        let s = parse_window(window, "stride");
        if s.is_empty() {
            vec![1, 1]
        } else {
            s
        }
    };
    let pad = parse_window_padding(window);
    let (n, ih, iw, ic) = (lhs.shape[0], lhs.shape[1], lhs.shape[2], lhs.shape[3]);
    let (kh, kw, ric, oc) = (rhs.shape[0], rhs.shape[1], rhs.shape[2], rhs.shape[3]);
    let (on, oh, ow, ooc) = (
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    );
    if n != on || ic != ric || oc != ooc {
        bail!(
            "convolution shape mismatch {:?} {:?} -> {:?}",
            lhs.shape,
            rhs.shape,
            output_shape
        );
    }
    let (pad_top, pad_left) = (
        pad.first().map(|p| p.0).unwrap_or(0),
        pad.get(1).map(|p| p.0).unwrap_or(0),
    );
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let spv_name = "hlo_conv_nhwc_hwio.spv";
    let spv = spv_bytes(kernel_dir, "hlo_conv_nhwc_hwio", spv_name)?;
    let push = [
        n as i32,
        ih as i32,
        iw as i32,
        ic as i32,
        kh as i32,
        kw as i32,
        oc as i32,
        oh as i32,
        ow as i32,
        stride[0] as i32,
        stride[1] as i32,
        pad_top as i32,
        pad_left as i32,
        out_elements as i32,
    ];
    let out = run_kernel(
        device,
        &KernelRun {
            label: "xla_hlo_convolution",
            spv: &spv,
            input0: &lhs.values,
            input1: &rhs.values,
            output_elements: out_elements,
            dispatch_x: groups(out_elements, 64),
            push: &push,
        },
    )?;
    let tensor = Tensor::new(output_shape.to_vec(), out.values.clone())?;
    let record = dispatch_record(result, "convolution", spv_name, out_elements, &out);
    Ok((tensor, record))
}

fn vulkan_reduce_window(
    kernel_dir: &Path,
    device: &str,
    result: &str,
    src: &Tensor,
    output_shape: &[usize],
    op: &HloOp,
    init: f32,
) -> Result<(Tensor, DispatchRecord)> {
    if src.shape.len() != 4 || output_shape.len() != 4 {
        bail!(
            "SPIR-V reduce-window supports rank-4 NHWC tensors only, got {:?} -> {:?}",
            src.shape,
            output_shape
        );
    }
    let window = op.attributes.get("window");
    let size = {
        let parsed = parse_window(window, "size");
        if parsed.is_empty() {
            vec![1; 4]
        } else {
            parsed
        }
    };
    let stride = {
        let parsed = parse_window(window, "stride");
        if parsed.is_empty() {
            vec![1; 4]
        } else {
            parsed
        }
    };
    if size.len() != 4 || stride.len() != 4 || size[0] != 1 || size[3] != 1 {
        bail!(
            "SPIR-V reduce-window supports NHWC spatial windows only, size {:?} stride {:?}",
            size,
            stride
        );
    }
    let pad = parse_window_padding(window);
    let (pad_top, pad_left) = (
        pad.get(1).map(|p| p.0).unwrap_or(0),
        pad.get(2).map(|p| p.0).unwrap_or(0),
    );
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let kind = if reduce_kind(op, init) == "max" { 1 } else { 0 };
    let spv_name = "hlo_reduce_window_rank4.spv";
    let spv = spv_bytes(kernel_dir, "hlo_reduce_window_rank4", spv_name)?;
    let push = [
        src.shape[0] as i32,
        src.shape[1] as i32,
        src.shape[2] as i32,
        src.shape[3] as i32,
        output_shape[1] as i32,
        output_shape[2] as i32,
        size[1] as i32,
        size[2] as i32,
        stride[1] as i32,
        stride[2] as i32,
        pad_top as i32,
        pad_left as i32,
        kind,
        out_elements as i32,
    ];
    let out = run_kernel(
        device,
        &KernelRun {
            label: "xla_hlo_reduce_window",
            spv: &spv,
            input0: &src.values,
            input1: &[0.0],
            output_elements: out_elements,
            dispatch_x: groups(out_elements, 64),
            push: &push,
        },
    )?;
    let tensor = Tensor::new(output_shape.to_vec(), out.values.clone())?;
    let record = dispatch_record(result, "reduce-window", spv_name, out_elements, &out);
    Ok((tensor, record))
}

fn vulkan_reduce(
    kernel_dir: &Path,
    device: &str,
    result: &str,
    src: &Tensor,
    output_shape: &[usize],
    dims: &[usize],
    op: &HloOp,
    init: f32,
) -> Result<(Tensor, DispatchRecord)> {
    if src.shape.len() > 4 {
        bail!("SPIR-V reduce supports rank <= 4, got {:?}", src.shape);
    }
    let mut src_dims = [1usize; 4];
    for (idx, dim) in src.shape.iter().enumerate() {
        src_dims[idx] = *dim;
    }
    let mut reduce_flags = [0i32; 4];
    for dim in dims {
        if *dim >= src.shape.len() {
            bail!("reduce dimension {} outside shape {:?}", dim, src.shape);
        }
        reduce_flags[*dim] = 1;
    }
    let mut out_dims_by_axis = [1usize; 4];
    let mut out_cursor = 0usize;
    for axis in 0..src.shape.len() {
        if reduce_flags[axis] == 0 {
            out_dims_by_axis[axis] = *output_shape.get(out_cursor).unwrap_or(&1);
            out_cursor += 1;
        }
    }
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let kind = if reduce_kind(op, init) == "max" { 1 } else { 0 };
    let spv_name = "hlo_reduce_rank4.spv";
    let spv = spv_bytes(kernel_dir, "hlo_reduce_rank4", spv_name)?;
    let push = [
        src.shape.len() as i32,
        src_dims[0] as i32,
        src_dims[1] as i32,
        src_dims[2] as i32,
        src_dims[3] as i32,
        out_dims_by_axis[0] as i32,
        out_dims_by_axis[1] as i32,
        out_dims_by_axis[2] as i32,
        out_dims_by_axis[3] as i32,
        reduce_flags[0],
        reduce_flags[1],
        reduce_flags[2],
        reduce_flags[3],
        kind,
        out_elements as i32,
    ];
    let out = run_kernel(
        device,
        &KernelRun {
            label: "xla_hlo_reduce",
            spv: &spv,
            input0: &src.values,
            input1: &[0.0],
            output_elements: out_elements,
            dispatch_x: groups(out_elements, 64),
            push: &push,
        },
    )?;
    let tensor = Tensor::new(output_shape.to_vec(), out.values.clone())?;
    let record = dispatch_record(result, "reduce", spv_name, out_elements, &out);
    Ok((tensor, record))
}

fn vulkan_dot_general(
    kernel_dir: &Path,
    device: &str,
    result: &str,
    lhs: &Tensor,
    rhs: &Tensor,
    output_shape: &[usize],
    op: &HloOp,
) -> Result<(Tensor, DispatchRecord)> {
    let lhs_contract = parse_usize_list(op.attributes.get("lhs_contracting_dims"));
    let rhs_contract = parse_usize_list(op.attributes.get("rhs_contracting_dims"));
    let lhs_batch = parse_usize_list(op.attributes.get("lhs_batch_dims"));
    let rhs_batch = parse_usize_list(op.attributes.get("rhs_batch_dims"));
    if lhs_contract.len() != 1 || rhs_contract.len() != 1 {
        bail!(
            "dot_general requires exactly one contracting dim per side: {}",
            op.raw
        );
    }
    let lc = lhs_contract[0];
    let rc = rhs_contract[0];
    let k = lhs.shape[lc];
    if k != rhs.shape[rc] {
        bail!("dot contracting dim mismatch {} vs {}", k, rhs.shape[rc]);
    }
    let batch_size: usize = lhs_batch.iter().map(|&d| lhs.shape[d]).product::<usize>().max(1);
    let lhs_free: Vec<usize> = (0..lhs.shape.len())
        .filter(|d| !lhs_batch.contains(d) && !lhs_contract.contains(d))
        .collect();
    let m: usize = lhs_free.iter().map(|&d| lhs.shape[d]).product::<usize>().max(1);
    let rhs_free: Vec<usize> = (0..rhs.shape.len())
        .filter(|d| !rhs_batch.contains(d) && !rhs_contract.contains(d))
        .collect();
    let n: usize = rhs_free.iter().map(|&d| rhs.shape[d]).product::<usize>().max(1);
    // Transpose lhs to [batch, free, contract] = [batch_size, M, K]
    let lhs_order: Vec<usize> = lhs_batch
        .iter()
        .chain(lhs_free.iter())
        .chain(lhs_contract.iter())
        .copied()
        .collect();
    let lhs_flat = transpose_flat(&lhs.values, &lhs.shape, &lhs_order);
    // Transpose rhs to [batch, contract, free] = [batch_size, K, N]
    let rhs_order: Vec<usize> = rhs_batch
        .iter()
        .chain(rhs_contract.iter())
        .chain(rhs_free.iter())
        .copied()
        .collect();
    let rhs_flat = transpose_flat(&rhs.values, &rhs.shape, &rhs_order);
    let out_elements = batch_size * m * n;
    let spv_name = "hlo_dot_batched.spv";
    let spv = spv_bytes(kernel_dir, "hlo_dot_batched", spv_name)?;
    let push = [
        batch_size as i32,
        m as i32,
        k as i32,
        n as i32,
        out_elements as i32,
    ];
    let out = run_kernel(
        device,
        &KernelRun {
            label: "xla_hlo_dot_batched",
            spv: &spv,
            input0: &lhs_flat,
            input1: &rhs_flat,
            output_elements: out_elements,
            dispatch_x: groups(out_elements, 64),
            push: &push,
        },
    )?;
    let tensor = Tensor::new(output_shape.to_vec(), out.values.clone())?;
    let record = dispatch_record(result, &op.opcode, spv_name, out_elements, &out);
    Ok((tensor, record))
}

fn vulkan_pad(
    kernel_dir: &Path,
    device: &str,
    result: &str,
    src: &Tensor,
    output_shape: &[usize],
    padding: &[(isize, isize)],
    pad_value: f32,
) -> Result<(Tensor, DispatchRecord)> {
    let rank = src.shape.len();
    if rank > 4 {
        bail!("SPIR-V pad supports rank <= 4, got {:?}", src.shape);
    }
    let mut id = [1i32; 4];
    let mut od = [1i32; 4];
    let mut pl = [0i32; 4];
    let offset = 4usize.saturating_sub(rank);
    for i in 0..rank {
        id[offset + i] = src.shape[i] as i32;
        od[offset + i] = output_shape[i] as i32;
        pl[offset + i] = padding[i].0 as i32;
    }
    for i in 0..offset {
        od[i] = 1;
    }
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let spv_name = "pad.spv";
    let spv = spv_bytes(kernel_dir, "pad", spv_name)?;
    let push = [
        od[0], od[1], od[2], od[3],
        id[0], id[1], id[2], id[3],
        pl[0], pl[1], pl[2], pl[3],
        out_elements as i32,
    ];
    let out = run_kernel(
        device,
        &KernelRun {
            label: "xla_hlo_pad",
            spv: &spv,
            input0: &src.values,
            input1: &[pad_value],
            output_elements: out_elements,
            dispatch_x: groups(out_elements, 128),
            push: &push,
        },
    )?;
    let tensor = Tensor::new(output_shape.to_vec(), out.values.clone())?;
    let record = dispatch_record(result, "pad", spv_name, out_elements, &out);
    Ok((tensor, record))
}

fn transpose_flat(values: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    if dims.iter().enumerate().all(|(i, &d)| i == d) {
        return values.to_vec();
    }
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let out_elements: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_elements];
    for (out_idx, slot) in out.iter_mut().enumerate() {
        let out_coord = unravel(out_idx, &out_shape);
        let mut src_coord = vec![0usize; shape.len()];
        for (out_axis, &src_axis) in dims.iter().enumerate() {
            src_coord[src_axis] = out_coord[out_axis];
        }
        *slot = values[ravel(&src_coord, shape)];
    }
    out
}

fn kernel_mapping(opcode: &str) -> Option<KernelMapping> {
    let mapping = match opcode {
        "add" => KernelMapping {
            spv_name: "add.spv",
            arity: 2,
        },
        "subtract" => KernelMapping {
            spv_name: "sub.spv",
            arity: 2,
        },
        "multiply" => KernelMapping {
            spv_name: "mul.spv",
            arity: 2,
        },
        "divide" => KernelMapping {
            spv_name: "div.spv",
            arity: 2,
        },
        "maximum" => KernelMapping {
            spv_name: "max.spv",
            arity: 2,
        },
        "minimum" => KernelMapping {
            spv_name: "min.spv",
            arity: 2,
        },
        "abs" => KernelMapping {
            spv_name: "abs.spv",
            arity: 1,
        },
        "exponential" => KernelMapping {
            spv_name: "exp.spv",
            arity: 1,
        },
        "log" => KernelMapping {
            spv_name: "log.spv",
            arity: 1,
        },
        "sqrt" => KernelMapping {
            spv_name: "sqrt.spv",
            arity: 1,
        },
        "rsqrt" => KernelMapping {
            spv_name: "rsqrt.spv",
            arity: 1,
        },
        "negate" => KernelMapping {
            spv_name: "neg.spv",
            arity: 1,
        },
        "tanh" => KernelMapping {
            spv_name: "tanh.spv",
            arity: 1,
        },
        "logistic" => KernelMapping {
            spv_name: "sigmoid.spv",
            arity: 1,
        },
        "reshape" | "convert" => KernelMapping {
            spv_name: "reshape.spv",
            arity: 1,
        },
        "power" => KernelMapping {
            spv_name: "pow.spv",
            arity: 2,
        },
        "sine" => KernelMapping {
            spv_name: "sine.spv",
            arity: 1,
        },
        _ => return None,
    };
    Some(mapping)
}

fn kernel_subdir(spv_name: &str) -> &str {
    spv_name.strip_suffix(".spv").unwrap_or(spv_name)
}

fn operand_tensor<'a>(
    values: &'a BTreeMap<String, Tensor>,
    op: &HloOp,
    index: usize,
) -> Result<&'a Tensor> {
    let name = op
        .operands
        .get(index)
        .with_context(|| format!("op {} missing operand {index}: {}", op.opcode, op.raw))?;
    values
        .get(name)
        .with_context(|| format!("operand {name} for op {} was not materialized", op.opcode))
}

fn parameter_number(raw: &str) -> Option<usize> {
    let start = raw.find("parameter(")? + "parameter(".len();
    let end = raw[start..].find(')')? + start;
    raw[start..end].parse().ok()
}

fn constant_scalar(raw: &str) -> Option<f32> {
    let start = raw.find("constant(")? + "constant(".len();
    let end = raw[start..].find(')')? + start;
    match raw[start..end].trim() {
        "inf" | "+inf" => Some(f32::INFINITY),
        "-inf" => Some(f32::NEG_INFINITY),
        "-0" => Some(-0.0),
        value => value.parse().ok(),
    }
}

fn type_shape(ty: Option<&str>) -> Option<Vec<usize>> {
    let ty = ty?;
    let start = ty.find('[')? + 1;
    let end = ty[start..].find(']')? + start;
    let dims = &ty[start..end];
    if dims.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(
        dims.split(',')
            .filter_map(|dim| dim.trim().parse::<usize>().ok())
            .collect(),
    )
}

fn shape_elements(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(1);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1; shape.len()];
    if shape.len() > 1 {
        for i in (0..shape.len() - 1).rev() {
            out[i] = out[i + 1] * shape[i + 1];
        }
    }
    out
}

fn unravel(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let st = strides(shape);
    let mut coord = Vec::with_capacity(shape.len());
    for stride in st {
        coord.push(index / stride);
        index %= stride;
    }
    coord
}

fn ravel(coord: &[usize], shape: &[usize]) -> usize {
    coord
        .iter()
        .zip(strides(shape))
        .map(|(idx, stride)| idx * stride)
        .sum()
}

fn parse_usize_list(value: Option<&String>) -> Vec<usize> {
    value
        .map(|v| {
            v.split(|c: char| !(c.is_ascii_digit()))
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_window(value: Option<&String>, key: &str) -> Vec<usize> {
    let Some(value) = value else {
        return Vec::new();
    };
    let marker = format!("{key}=");
    let Some(start) = value.find(&marker).map(|idx| idx + marker.len()) else {
        return Vec::new();
    };
    let rest = &value[start..];
    let end = rest.find(' ').unwrap_or(rest.len());
    rest[..end]
        .split('x')
        .filter_map(|s| s.parse::<usize>().ok())
        .collect()
}

fn parse_padding(value: Option<&String>, rank: usize) -> Result<Vec<(isize, isize)>> {
    let Some(value) = value else {
        return Ok(vec![(0, 0); rank]);
    };
    let pads = value
        .split('x')
        .map(|part| {
            let mut it = part.split('_').filter_map(|s| s.parse::<isize>().ok());
            Ok((it.next().unwrap_or(0), it.next().unwrap_or(0)))
        })
        .collect::<Result<Vec<_>>>()?;
    if pads.len() != rank {
        bail!("padding rank mismatch: got {:?} for rank {}", pads, rank);
    }
    Ok(pads)
}

fn broadcast_tensor(src: &Tensor, output_shape: &[usize], dims: &[usize]) -> Result<Tensor> {
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    if src.values.len() == out_elements && src.shape == output_shape {
        return Tensor::new(output_shape.to_vec(), src.values.clone());
    }
    if src.values.len() == 1 {
        return Tensor::new(output_shape.to_vec(), vec![src.values[0]; out_elements]);
    }
    if dims.len() != src.shape.len() {
        bail!(
            "broadcast dimensions {:?} do not match source shape {:?}",
            dims,
            src.shape
        );
    }
    let mut out = Vec::with_capacity(out_elements);
    for idx in 0..out_elements {
        let out_coord = unravel(idx, output_shape);
        let src_coord = dims
            .iter()
            .enumerate()
            .map(|(src_axis, out_axis)| {
                if src.shape[src_axis] == 1 {
                    0
                } else {
                    out_coord[*out_axis]
                }
            })
            .collect::<Vec<_>>();
        out.push(src.values[ravel(&src_coord, &src.shape)]);
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn transpose_tensor(src: &Tensor, output_shape: &[usize], dims: &[usize]) -> Result<Tensor> {
    if dims.is_empty() || dims.iter().enumerate().all(|(i, dim)| i == *dim) {
        return Tensor::new(output_shape.to_vec(), src.values.clone());
    }
    if dims.len() != output_shape.len() || dims.len() != src.shape.len() {
        bail!(
            "transpose dims {:?} incompatible with {:?} -> {:?}",
            dims,
            src.shape,
            output_shape
        );
    }
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let mut out = vec![0.0; out_elements];
    for (out_idx, slot) in out.iter_mut().enumerate() {
        let out_coord = unravel(out_idx, output_shape);
        let mut src_coord = vec![0; dims.len()];
        for (out_axis, src_axis) in dims.iter().enumerate() {
            src_coord[*src_axis] = out_coord[out_axis];
        }
        *slot = src.values[ravel(&src_coord, &src.shape)];
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn pad_tensor(
    src: &Tensor,
    output_shape: &[usize],
    padding: &[(isize, isize)],
    pad_value: f32,
) -> Result<Tensor> {
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let mut out = vec![pad_value; out_elements];
    for (src_idx, value) in src.values.iter().enumerate() {
        let src_coord = unravel(src_idx, &src.shape);
        let out_coord = src_coord
            .iter()
            .enumerate()
            .map(|(axis, idx)| (*idx as isize + padding[axis].0) as usize)
            .collect::<Vec<_>>();
        if out_coord
            .iter()
            .zip(output_shape)
            .all(|(idx, dim)| *idx < *dim)
        {
            let out_idx = ravel(&out_coord, output_shape);
            out[out_idx] = *value;
        }
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn cpu_elementwise(
    values: &BTreeMap<String, Tensor>,
    op: &HloOp,
    output_shape: &[usize],
) -> Result<Tensor> {
    let lhs = operand_tensor(values, op, 0)?;
    let rhs_storage;
    let out = match op.opcode.as_str() {
        "add" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, |a, b| a + b)?
        }
        "subtract" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, |a, b| a - b)?
        }
        "multiply" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, |a, b| a * b)?
        }
        "divide" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, |a, b| a / b)?
        }
        "maximum" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, f32::max)?
        }
        "minimum" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, f32::min)?
        }
        "power" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, f32::powf)?
        }
        "remainder" => {
            rhs_storage = operand_tensor(values, op, 1)?;
            binary_tensor(lhs, rhs_storage, output_shape, |a, b| a % b)?
        }
        "abs" => unary_tensor(lhs, output_shape, f32::abs)?,
        "exponential" => unary_tensor(lhs, output_shape, f32::exp)?,
        "log" => unary_tensor(lhs, output_shape, f32::ln)?,
        "sqrt" => unary_tensor(lhs, output_shape, f32::sqrt)?,
        "rsqrt" => unary_tensor(lhs, output_shape, |a| 1.0 / a.sqrt())?,
        "negate" => unary_tensor(lhs, output_shape, |a| -a)?,
        "tanh" => unary_tensor(lhs, output_shape, f32::tanh)?,
        "logistic" => unary_tensor(lhs, output_shape, |a| 1.0 / (1.0 + (-a).exp()))?,
        "floor" => unary_tensor(lhs, output_shape, f32::floor)?,
        "ceil" => unary_tensor(lhs, output_shape, f32::ceil)?,
        "round-nearest-even" => unary_tensor(lhs, output_shape, f32::round)?,
        "sine" => unary_tensor(lhs, output_shape, f32::sin)?,
        "cosine" => unary_tensor(lhs, output_shape, f32::cos)?,
        "slice" | "dynamic-slice" | "dynamic-update-slice" | "iota" | "gather" | "scatter" => {
            bail!(
                "{} is accepted by validation but not needed by current fixtures",
                op.opcode
            )
        }
        _ => bail!("no CPU elementwise lowering for {}", op.opcode),
    };
    Ok(out)
}

fn unary_tensor(src: &Tensor, output_shape: &[usize], f: impl Fn(f32) -> f32) -> Result<Tensor> {
    if src.values.len() != shape_elements(output_shape).unwrap_or(src.values.len()) {
        bail!("unary shape mismatch {:?} -> {:?}", src.shape, output_shape);
    }
    Tensor::new(
        output_shape.to_vec(),
        src.values.iter().copied().map(f).collect(),
    )
}

fn binary_tensor(
    lhs: &Tensor,
    rhs: &Tensor,
    output_shape: &[usize],
    f: impl Fn(f32, f32) -> f32,
) -> Result<Tensor> {
    let out_elements = shape_elements(output_shape).unwrap_or(lhs.values.len());
    let lhs_b = if lhs.values.len() == out_elements {
        lhs.clone()
    } else {
        broadcast_tensor(
            lhs,
            output_shape,
            &infer_broadcast_dims(&lhs.shape, output_shape),
        )?
    };
    let rhs_b = if rhs.values.len() == out_elements {
        rhs.clone()
    } else {
        broadcast_tensor(
            rhs,
            output_shape,
            &infer_broadcast_dims(&rhs.shape, output_shape),
        )?
    };
    if lhs_b.values.len() != rhs_b.values.len() {
        bail!(
            "binary op length mismatch {} vs {} for output {:?}",
            lhs_b.values.len(),
            rhs_b.values.len(),
            output_shape
        );
    }
    Tensor::new(
        output_shape.to_vec(),
        lhs_b
            .values
            .iter()
            .copied()
            .zip(rhs_b.values.iter().copied())
            .map(|(a, b)| f(a, b))
            .collect(),
    )
}

fn infer_broadcast_dims(src_shape: &[usize], output_shape: &[usize]) -> Vec<usize> {
    if src_shape.is_empty() {
        return Vec::new();
    }
    let offset = output_shape.len().saturating_sub(src_shape.len());
    (0..src_shape.len()).map(|axis| axis + offset).collect()
}

fn select_tensor(
    pred: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
    output_shape: &[usize],
) -> Result<Tensor> {
    let out_elements = shape_elements(output_shape).unwrap_or(on_true.values.len());
    let pred_b = if pred.values.len() == out_elements {
        pred.clone()
    } else {
        broadcast_tensor(
            pred,
            output_shape,
            &infer_broadcast_dims(&pred.shape, output_shape),
        )?
    };
    let true_b = if on_true.values.len() == out_elements {
        on_true.clone()
    } else {
        broadcast_tensor(
            on_true,
            output_shape,
            &infer_broadcast_dims(&on_true.shape, output_shape),
        )?
    };
    let false_b = if on_false.values.len() == out_elements {
        on_false.clone()
    } else {
        broadcast_tensor(
            on_false,
            output_shape,
            &infer_broadcast_dims(&on_false.shape, output_shape),
        )?
    };
    Tensor::new(
        output_shape.to_vec(),
        (0..out_elements)
            .map(|i| {
                if pred_b.values[i] != 0.0 {
                    true_b.values[i]
                } else {
                    false_b.values[i]
                }
            })
            .collect(),
    )
}

fn conv_nhwc_hwio(
    lhs: &Tensor,
    rhs: &Tensor,
    output_shape: &[usize],
    op: &HloOp,
) -> Result<Tensor> {
    if lhs.shape.len() != 4 || rhs.shape.len() != 4 || output_shape.len() != 4 {
        bail!(
            "convolution supports rank-4 NHWC/HWIO only, got {:?} {:?} -> {:?}",
            lhs.shape,
            rhs.shape,
            output_shape
        );
    }
    if op.attributes.get("dim_labels").map(String::as_str) != Some("b01f_01io->b01f") {
        bail!(
            "unsupported convolution dim_labels {:?}",
            op.attributes.get("dim_labels")
        );
    }
    let window = op.attributes.get("window");
    let stride = {
        let s = parse_window(window, "stride");
        if s.is_empty() {
            vec![1, 1]
        } else {
            s
        }
    };
    let pad = parse_window_padding(window);
    let (n, ih, iw, ic) = (lhs.shape[0], lhs.shape[1], lhs.shape[2], lhs.shape[3]);
    let (kh, kw, ric, oc) = (rhs.shape[0], rhs.shape[1], rhs.shape[2], rhs.shape[3]);
    let (on, oh, ow, ooc) = (
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    );
    if n != on || ic != ric || oc != ooc {
        bail!(
            "convolution shape mismatch {:?} {:?} -> {:?}",
            lhs.shape,
            rhs.shape,
            output_shape
        );
    }
    let (pad_top, pad_left) = (
        pad.first().map(|p| p.0).unwrap_or(0),
        pad.get(1).map(|p| p.0).unwrap_or(0),
    );
    let mut out = vec![0.0; shape_elements(output_shape).unwrap_or(0)];
    for b in 0..n {
        for y in 0..oh {
            for x in 0..ow {
                for out_c in 0..oc {
                    let mut acc = 0.0_f32;
                    for ky in 0..kh {
                        let in_y = y as isize * stride[0] as isize + ky as isize - pad_top;
                        if in_y < 0 || in_y >= ih as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let in_x = x as isize * stride[1] as isize + kx as isize - pad_left;
                            if in_x < 0 || in_x >= iw as isize {
                                continue;
                            }
                            for in_c in 0..ic {
                                let li =
                                    ((b * ih + in_y as usize) * iw + in_x as usize) * ic + in_c;
                                let ri = ((ky * kw + kx) * ic + in_c) * oc + out_c;
                                acc += lhs.values[li] * rhs.values[ri];
                            }
                        }
                    }
                    out[((b * oh + y) * ow + x) * oc + out_c] = acc;
                }
            }
        }
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn parse_window_padding(window: Option<&String>) -> Vec<(isize, isize)> {
    let Some(window) = window else {
        return vec![(0, 0), (0, 0)];
    };
    let Some(start) = window.find("pad=").map(|idx| idx + 4) else {
        return vec![(0, 0), (0, 0)];
    };
    let rest = &window[start..];
    let end = rest.find(' ').unwrap_or(rest.len());
    rest[..end]
        .split('x')
        .map(|part| {
            let mut it = part.split('_').filter_map(|s| s.parse::<isize>().ok());
            (it.next().unwrap_or(0), it.next().unwrap_or(0))
        })
        .collect()
}

fn reduce_kind(op: &HloOp, init: f32) -> &'static str {
    if op.raw.contains("to_apply=%max") || init.is_infinite() && init.is_sign_negative() {
        "max"
    } else {
        "sum"
    }
}

fn reduce_tensor(
    src: &Tensor,
    output_shape: &[usize],
    dims: &[usize],
    op: &HloOp,
    init: f32,
) -> Result<Tensor> {
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let mut out = vec![init; out_elements];
    let kind = reduce_kind(op, init);
    for (src_idx, value) in src.values.iter().enumerate() {
        let src_coord = unravel(src_idx, &src.shape);
        let out_coord = src_coord
            .iter()
            .enumerate()
            .filter(|(axis, _)| !dims.contains(axis))
            .map(|(_, idx)| *idx)
            .collect::<Vec<_>>();
        let out_idx = if output_shape.is_empty() {
            0
        } else {
            ravel(&out_coord, output_shape)
        };
        match kind {
            "max" => out[out_idx] = out[out_idx].max(*value),
            _ => out[out_idx] += *value,
        }
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn reduce_window(src: &Tensor, output_shape: &[usize], op: &HloOp, init: f32) -> Result<Tensor> {
    if src.shape.len() != output_shape.len() {
        bail!(
            "reduce-window rank mismatch {:?} -> {:?}",
            src.shape,
            output_shape
        );
    }
    let rank = src.shape.len();
    let window = op.attributes.get("window");
    let size = {
        let parsed = parse_window(window, "size");
        if parsed.is_empty() {
            vec![1; rank]
        } else {
            parsed
        }
    };
    let stride = {
        let parsed = parse_window(window, "stride");
        if parsed.is_empty() {
            vec![1; rank]
        } else {
            parsed
        }
    };
    if size.len() != rank || stride.len() != rank {
        bail!(
            "reduce-window size/stride rank mismatch {:?} {:?} rank {}",
            size,
            stride,
            rank
        );
    }
    let kind = reduce_kind(op, init);
    let mut out = vec![init; shape_elements(output_shape).unwrap_or(1)];
    for (out_idx, slot) in out.iter_mut().enumerate() {
        let out_coord = unravel(out_idx, output_shape);
        let mut acc = init;
        visit_window(&size, |win_coord| {
            let src_coord = (0..rank)
                .map(|axis| out_coord[axis] * stride[axis] + win_coord[axis])
                .collect::<Vec<_>>();
            if src_coord
                .iter()
                .zip(&src.shape)
                .all(|(idx, dim)| *idx < *dim)
            {
                let value = src.values[ravel(&src_coord, &src.shape)];
                match kind {
                    "max" => acc = acc.max(value),
                    _ => acc += value,
                }
            }
        });
        *slot = acc;
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn visit_window(size: &[usize], mut f: impl FnMut(&[usize])) {
    let total = shape_elements(size).unwrap_or(1);
    for idx in 0..total {
        let coord = unravel(idx, size);
        f(&coord);
    }
}

fn dot_general(lhs: &Tensor, rhs: &Tensor, output_shape: &[usize], op: &HloOp) -> Result<Tensor> {
    let lhs_contract = parse_usize_list(op.attributes.get("lhs_contracting_dims"));
    let rhs_contract = parse_usize_list(op.attributes.get("rhs_contracting_dims"));
    let lhs_batch = parse_usize_list(op.attributes.get("lhs_batch_dims"));
    let rhs_batch = parse_usize_list(op.attributes.get("rhs_batch_dims"));
    if lhs_contract.len() != 1 || rhs_contract.len() != 1 || lhs_batch.len() != rhs_batch.len() {
        bail!("unsupported dot dimension_numbers in {}", op.raw);
    }
    let lc = lhs_contract[0];
    let rc = rhs_contract[0];
    let k = lhs.shape[lc];
    if k != rhs.shape[rc] {
        bail!("dot contracting dim mismatch {} vs {}", k, rhs.shape[rc]);
    }
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let mut out = vec![0.0; out_elements];
    for (out_idx, slot) in out.iter_mut().enumerate() {
        let out_coord = unravel(out_idx, output_shape);
        let mut lhs_coord = vec![0; lhs.shape.len()];
        let mut rhs_coord = vec![0; rhs.shape.len()];
        let mut cursor = 0usize;
        for (lb, rb) in lhs_batch.iter().zip(&rhs_batch) {
            lhs_coord[*lb] = out_coord[cursor];
            rhs_coord[*rb] = out_coord[cursor];
            cursor += 1;
        }
        for axis in 0..lhs.shape.len() {
            if axis != lc && !lhs_batch.contains(&axis) {
                lhs_coord[axis] = out_coord[cursor];
                cursor += 1;
            }
        }
        for axis in 0..rhs.shape.len() {
            if axis != rc && !rhs_batch.contains(&axis) {
                rhs_coord[axis] = out_coord[cursor];
                cursor += 1;
            }
        }
        let mut acc = 0.0_f32;
        for kk in 0..k {
            lhs_coord[lc] = kk;
            rhs_coord[rc] = kk;
            acc += lhs.values[ravel(&lhs_coord, &lhs.shape)]
                * rhs.values[ravel(&rhs_coord, &rhs.shape)];
        }
        *slot = acc;
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn cpu_concatenate(inputs: &[Tensor], output_shape: &[usize], dim: usize) -> Result<Tensor> {
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let mut out = vec![0.0f32; out_elements];
    let mut offset_in_dim = 0usize;
    for input in inputs {
        for (src_idx, &value) in input.values.iter().enumerate() {
            let src_coord = unravel(src_idx, &input.shape);
            let mut out_coord = src_coord.clone();
            out_coord[dim] += offset_in_dim;
            if out_coord.iter().zip(output_shape).all(|(c, d)| *c < *d) {
                out[ravel(&out_coord, output_shape)] = value;
            }
        }
        offset_in_dim += input.shape[dim];
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn parse_slice_spec(raw: &str) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let slice_marker = "slice={";
    let Some(start) = raw.find(slice_marker).map(|i| i + slice_marker.len()) else {
        bail!("cannot find slice={{}} spec in: {}", raw);
    };
    let Some(end) = raw[start..].find('}').map(|i| i + start) else {
        bail!("cannot find closing '}}' in slice spec: {}", raw);
    };
    let spec = &raw[start..end];
    let mut starts = Vec::new();
    let mut limits = Vec::new();
    let mut strides_out = Vec::new();
    for bracket in spec.split(']') {
        let trimmed = bracket.trim().trim_start_matches(',').trim().trim_start_matches('[');
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(':').collect();
        if parts.len() >= 2 {
            starts.push(parts[0].trim().parse::<usize>().unwrap_or(0));
            limits.push(parts[1].trim().parse::<usize>().unwrap_or(0));
            strides_out.push(if parts.len() >= 3 {
                parts[2].trim().parse::<usize>().unwrap_or(1)
            } else {
                1
            });
        }
    }
    if starts.is_empty() {
        bail!("empty slice spec in: {}", raw);
    }
    Ok((starts, limits, strides_out))
}

fn cpu_slice(
    src: &Tensor,
    output_shape: &[usize],
    starts: &[usize],
    _limits: &[usize],
    strides_s: &[usize],
) -> Result<Tensor> {
    let out_elements = shape_elements(output_shape).unwrap_or(1);
    let mut out = vec![0.0f32; out_elements];
    for (out_idx, slot) in out.iter_mut().enumerate() {
        let out_coord = unravel(out_idx, output_shape);
        let src_coord: Vec<usize> = out_coord
            .iter()
            .enumerate()
            .map(|(axis, &c)| starts[axis] + c * strides_s[axis])
            .collect();
        if src_coord.iter().zip(&src.shape).all(|(c, d)| *c < *d) {
            *slot = src.values[ravel(&src_coord, &src.shape)];
        }
    }
    Tensor::new(output_shape.to_vec(), out)
}

fn probe_vulkan_device(kernel_dir: &Path, device: &str) -> Result<serde_json::Value> {
    let spv_path = kernel_dir.join("add").join("spv").join("add.spv");
    let spv =
        std::fs::read(&spv_path).with_context(|| format!("reading {}", spv_path.display()))?;
    let out = run_kernel(
        device,
        &KernelRun {
            label: "xla_hlo_device_probe_add",
            spv: &spv,
            input0: &[1.0],
            input1: &[2.0],
            output_elements: 1,
            dispatch_x: 1,
            push: &[1],
        },
    )?;
    Ok(serde_json::json!({
        "name": out.device_name,
        "vendor_id": out.vendor_id,
        "device_id": out.device_id,
        "probe_elapsed_ms": out.elapsed_ms,
        "probe_output": out.values
    }))
}
