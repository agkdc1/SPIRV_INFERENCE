use crate::{
    kernels,
    model::{BuiltinOp, ModelInfo, TensorInfo, TensorType},
    op_decoder::{self, Activation, DecodedOptions, Padding},
    tensor::{compare, f32_bytes, read_f32_bytes, sha256},
    vulkan::{groups, run_kernel, KernelRun},
};
use anyhow::{bail, Context, Result};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
struct TensorValue {
    values: Vec<f32>,
    layout: Layout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Layout {
    Flat,
    Nchw,
}

pub fn run_graph(
    model: &ModelInfo,
    kernel_dir: &Path,
    device: &str,
    dump_dir: &Path,
    input_f32: Option<&Path>,
) -> Result<serde_json::Value> {
    fs::create_dir_all(dump_dir)?;
    let order = topological_order(model)?;
    let mut tensors: BTreeMap<usize, TensorValue> = BTreeMap::new();
    let mut op_reports = Vec::new();
    let unused = vec![0.0_f32; 1];
    let mut input_reports = Vec::new();

    load_constants(model, &mut tensors)?;
    if input_f32.is_some() && model.inputs.len() != 1 {
        bail!(
            "--input-f32 currently supports exactly one model input, got {}",
            model.inputs.len()
        );
    }
    for &input_idx in &model.inputs {
        let idx = valid_tensor_index(input_idx)?;
        let tensor = model
            .tensors
            .get(idx)
            .with_context(|| format!("model input tensor {idx} missing"))?;
        if let std::collections::btree_map::Entry::Vacant(entry) = tensors.entry(idx) {
            let (values, source) = load_input(tensor, input_f32)?;
            let layout = if tensor.shape.len() == 4 {
                Layout::Nchw
            } else {
                Layout::Flat
            };
            input_reports.push(json!({
                "tensor": idx,
                "shape": tensor.shape,
                "elements": values.len(),
                "layout": layout,
                "source": source
            }));
            entry.insert(TensorValue { values, layout });
        }
    }

    let mut device_report = json!(null);
    for &op_index in &order {
        let op = &model.operators[op_index];
        let options = op_decoder::decode_options(model, op)
            .with_context(|| format!("decoding options for operator {}", op.index))?;
        let started_tensor_count = tensors.len();
        let mut dispatches = Vec::new();
        match op.builtin {
            BuiltinOp::Abs
            | BuiltinOp::Exp
            | BuiltinOp::Log
            | BuiltinOp::Logistic
            | BuiltinOp::Neg
            | BuiltinOp::Rsqrt
            | BuiltinOp::Sqrt
            | BuiltinOp::Tanh
            | BuiltinOp::Relu6
            | BuiltinOp::LeakyRelu => {
                let input_idx = single_input(op.inputs.as_slice(), op.index)?;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                let elements = element_count(model.tensor(output_idx)?)?;
                let (spv, push) = match (&op.builtin, &options) {
                    (BuiltinOp::Abs, _) => ("abs.spv", vec![elements as i32]),
                    (BuiltinOp::Exp, _) => ("exp.spv", vec![elements as i32]),
                    (BuiltinOp::Log, _) => ("log.spv", vec![elements as i32]),
                    (BuiltinOp::Logistic, _) => ("sigmoid.spv", vec![elements as i32]),
                    (BuiltinOp::Neg, _) => ("neg.spv", vec![elements as i32]),
                    (BuiltinOp::Rsqrt, _) => ("rsqrt.spv", vec![elements as i32]),
                    (BuiltinOp::Sqrt, _) => ("sqrt.spv", vec![elements as i32]),
                    (BuiltinOp::Tanh, _) => ("tanh.spv", vec![elements as i32]),
                    (BuiltinOp::Relu6, _) => ("relu6.spv", vec![elements as i32]),
                    (BuiltinOp::LeakyRelu, DecodedOptions::LeakyRelu { alpha }) => (
                        "leaky_relu.spv",
                        vec![elements as i32, alpha.to_bits() as i32],
                    ),
                    (BuiltinOp::LeakyRelu, _) => (
                        "leaky_relu.spv",
                        vec![elements as i32, 0.2_f32.to_bits() as i32],
                    ),
                    _ => unreachable!(),
                };
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    spv,
                    &input.values,
                    &unused,
                    elements,
                    &push,
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(spv, out.elapsed_ms, elements));
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout: input.layout,
                    },
                );
            }
            BuiltinOp::Add
            | BuiltinOp::Sub
            | BuiltinOp::Mul
            | BuiltinOp::Div
            | BuiltinOp::Pow
            | BuiltinOp::Minimum
            | BuiltinOp::Maximum => {
                op_decoder::require_no_fused_activation(&options)?;
                let (a_idx, b_idx) = two_inputs(op.inputs.as_slice(), op.index)?;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let a = tensors
                    .get(&a_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, a_idx
                        )
                    })?
                    .clone();
                let b = tensors
                    .get(&b_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, b_idx
                        )
                    })?
                    .clone();
                if a.values.len() != b.values.len() {
                    bail!(
                        "operator {} {:?} requires equal-sized tensors, got {} and {}",
                        op.index,
                        op.builtin,
                        a.values.len(),
                        b.values.len()
                    );
                }
                let elements = element_count(model.tensor(output_idx)?)?;
                let spv = match op.builtin {
                    BuiltinOp::Add => "add.spv",
                    BuiltinOp::Sub => "sub.spv",
                    BuiltinOp::Mul => "mul.spv",
                    BuiltinOp::Div => "div.spv",
                    BuiltinOp::Pow => "pow.spv",
                    BuiltinOp::Minimum => "min.spv",
                    BuiltinOp::Maximum => "max.spv",
                    _ => unreachable!(),
                };
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    spv,
                    &a.values,
                    &b.values,
                    elements,
                    &[elements as i32],
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(spv, out.elapsed_ms, elements));
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout: a.layout,
                    },
                );
            }
            BuiltinOp::Reshape | BuiltinOp::Squeeze | BuiltinOp::ExpandDims | BuiltinOp::Cast => {
                let input_idx = op
                    .inputs
                    .first()
                    .copied()
                    .filter(|idx| *idx >= 0)
                    .with_context(|| format!("operator {} Reshape missing data input", op.index))?
                    as usize;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                if op.builtin == BuiltinOp::Cast {
                    let in_ty = &model.tensor(input_idx)?.tensor_type;
                    let out_ty = &model.tensor(output_idx)?.tensor_type;
                    if *in_ty != TensorType::Float32 || *out_ty != TensorType::Float32 {
                        bail!(
                            "operator {} Cast only supports Float32->Float32, got {:?}->{:?}",
                            op.index,
                            in_ty,
                            out_ty
                        );
                    }
                }
                let elements = element_count(model.tensor(output_idx)?)?;
                if input.values.len() != elements {
                    bail!(
                        "operator {} {:?} element count changed from {} to {}",
                        op.index,
                        op.builtin,
                        input.values.len(),
                        elements
                    );
                }
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "reshape.spv",
                    &input.values,
                    &unused,
                    elements,
                    &[elements as i32],
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report("reshape.spv", out.elapsed_ms, elements));
                let output_layout = if model.tensor(output_idx)?.shape.len() == 4 {
                    Layout::Nchw
                } else {
                    Layout::Flat
                };
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout: output_layout,
                    },
                );
            }
            BuiltinOp::Softmax => {
                let input_idx = single_input(op.inputs.as_slice(), op.index)?;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                let classes = element_count(model.tensor(output_idx)?)?;
                let (spv, local) = if classes == 10 {
                    ("softmax10.spv", 10)
                } else {
                    ("softmax1000.spv", 256)
                };
                let push = if spv == "softmax10.spv" {
                    Vec::new()
                } else {
                    vec![classes as i32]
                };
                let out = dispatch_with_local(
                    kernel_dir,
                    device,
                    &op_label(op),
                    spv,
                    &input.values,
                    &unused,
                    classes,
                    &push,
                    local,
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(spv, out.elapsed_ms, classes));
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout: Layout::Flat,
                    },
                );
            }
            BuiltinOp::Conv2d => {
                let (padding, stride_h, stride_w, dilation_h, dilation_w, activation) =
                    match &options {
                        DecodedOptions::Conv2d {
                            padding,
                            stride_h,
                            stride_w,
                            dilation_h_factor,
                            dilation_w_factor,
                            fused_activation,
                        } => (
                            padding.clone(),
                            *stride_h,
                            *stride_w,
                            *dilation_h_factor,
                            *dilation_w_factor,
                            fused_activation.clone(),
                        ),
                        _ => (Padding::Valid, 1, 1, 1, 1, Activation::None),
                    };
                if dilation_h != 1 || dilation_w != 1 {
                    bail!(
                        "operator {} Conv2D dilation {dilation_h}x{dilation_w} is unsupported",
                        op.index
                    );
                }
                let (input_idx, weight_idx, bias_idx) =
                    three_inputs(op.inputs.as_slice(), op.index)?;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                if input.layout != Layout::Nchw {
                    bail!(
                        "operator {} Conv2D requires NCHW activation layout",
                        op.index
                    );
                }
                let weight = pack_conv_weight(model, weight_idx, bias_idx)?;
                let in_shape = nchw_from_nhwc(&model.tensor(input_idx)?.shape)?;
                let out_shape = nchw_from_nhwc(&model.tensor(output_idx)?.shape)?;
                let w_shape = &model.tensor(weight_idx)?.shape;
                let total = element_count(model.tensor(output_idx)?)?;
                let (pad_top, pad_left) = same_padding_offsets(
                    &padding,
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    (w_shape[1] - 1) * dilation_h + 1,
                    (w_shape[2] - 1) * dilation_w + 1,
                    stride_h,
                    stride_w,
                );
                let push = vec![
                    in_shape[0],
                    in_shape[1],
                    in_shape[2],
                    in_shape[3],
                    out_shape[1],
                    out_shape[2],
                    out_shape[3],
                    w_shape[1],
                    w_shape[2],
                    stride_h,
                    stride_w,
                    same_flag(&padding),
                    pad_top,
                    pad_left,
                    total as i32,
                ];
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "conv2d.spv",
                    &input.values,
                    &weight,
                    total,
                    &push,
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report("conv2d.spv", out.elapsed_ms, total));
                let values = apply_activation(
                    kernel_dir,
                    device,
                    &op_label(op),
                    activation,
                    out.values,
                    &unused,
                    &mut dispatches,
                    &mut device_report,
                )?;
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values,
                        layout: Layout::Nchw,
                    },
                );
            }
            BuiltinOp::DepthwiseConv2d => {
                let (
                    padding,
                    stride_h,
                    stride_w,
                    depth_multiplier,
                    dilation_h,
                    dilation_w,
                    activation,
                ) = match &options {
                    DecodedOptions::DepthwiseConv2d {
                        padding,
                        stride_h,
                        stride_w,
                        depth_multiplier,
                        dilation_h_factor,
                        dilation_w_factor,
                        fused_activation,
                    } => (
                        padding.clone(),
                        *stride_h,
                        *stride_w,
                        *depth_multiplier,
                        *dilation_h_factor,
                        *dilation_w_factor,
                        fused_activation.clone(),
                    ),
                    _ => (Padding::Valid, 1, 1, 1, 1, 1, Activation::None),
                };
                if depth_multiplier != 1 {
                    bail!(
                        "operator {} DepthwiseConv2D requires depth_multiplier=1, got {depth_multiplier}",
                        op.index
                    );
                }
                let (input_idx, weight_idx, bias_idx) =
                    three_inputs(op.inputs.as_slice(), op.index)?;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                if input.layout != Layout::Nchw {
                    bail!(
                        "operator {} DepthwiseConv2D requires NCHW activation layout",
                        op.index
                    );
                }
                let weight = pack_depthwise_weight(model, weight_idx, bias_idx)?;
                let in_shape = nchw_from_nhwc(&model.tensor(input_idx)?.shape)?;
                let out_shape = nchw_from_nhwc(&model.tensor(output_idx)?.shape)?;
                let w_shape = &model.tensor(weight_idx)?.shape;
                let total = element_count(model.tensor(output_idx)?)?;
                let (pad_top, pad_left) = same_padding_offsets(
                    &padding,
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    (w_shape[1] - 1) * dilation_h + 1,
                    (w_shape[2] - 1) * dilation_w + 1,
                    stride_h,
                    stride_w,
                );
                let push = vec![
                    in_shape[0],
                    in_shape[1],
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    w_shape[1],
                    w_shape[2],
                    stride_h,
                    stride_w,
                    same_flag(&padding),
                    pad_top,
                    pad_left,
                    dilation_h,
                    dilation_w,
                    total as i32,
                ];
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "depthwise_conv2d.spv",
                    &input.values,
                    &weight,
                    total,
                    &push,
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(
                    "depthwise_conv2d.spv",
                    out.elapsed_ms,
                    total,
                ));
                let values = apply_activation(
                    kernel_dir,
                    device,
                    &op_label(op),
                    activation,
                    out.values,
                    &unused,
                    &mut dispatches,
                    &mut device_report,
                )?;
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values,
                        layout: Layout::Nchw,
                    },
                );
            }
            BuiltinOp::AveragePool2d => {
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input_idx = single_input(op.inputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                let in_shape = nchw_from_nhwc(&model.tensor(input_idx)?.shape)?;
                let out_elements = element_count(model.tensor(output_idx)?)?;
                if out_elements != (in_shape[0] * in_shape[1]) as usize {
                    bail!(
                        "operator {} AveragePool2D only supports global pool to N*C output",
                        op.index
                    );
                }
                let push = vec![in_shape[0], in_shape[1], in_shape[2], in_shape[3]];
                let sum = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "global_avg_pool_sum.spv",
                    &input.values,
                    &unused,
                    out_elements,
                    &push,
                )?;
                dispatches.push(dispatch_report(
                    "global_avg_pool_sum.spv",
                    sum.elapsed_ms,
                    out_elements,
                ));
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "global_avg_pool_finalize.spv",
                    &sum.values,
                    &unused,
                    out_elements,
                    &push,
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(
                    "global_avg_pool_finalize.spv",
                    out.elapsed_ms,
                    out_elements,
                ));
                let output_layout = if model.tensor(output_idx)?.shape.len() == 4 {
                    Layout::Nchw
                } else {
                    Layout::Flat
                };
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout: output_layout,
                    },
                );
            }
            BuiltinOp::MaxPool2d => {
                let (padding, stride_h, stride_w, filter_h, filter_w, activation) = match &options {
                    DecodedOptions::Pool2d {
                        padding,
                        stride_h,
                        stride_w,
                        filter_h,
                        filter_w,
                        fused_activation,
                    } => (
                        padding.clone(),
                        *stride_h,
                        *stride_w,
                        *filter_h,
                        *filter_w,
                        fused_activation.clone(),
                    ),
                    _ => (Padding::Valid, 1, 1, 1, 1, Activation::None),
                };
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input_idx = single_input(op.inputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                if input.layout != Layout::Nchw {
                    bail!(
                        "operator {} MaxPool2D requires NCHW activation layout",
                        op.index
                    );
                }
                let in_shape = nchw_from_nhwc(&model.tensor(input_idx)?.shape)?;
                let out_shape = nchw_from_nhwc(&model.tensor(output_idx)?.shape)?;
                let total = element_count(model.tensor(output_idx)?)?;
                let (pad_top, pad_left) = same_padding_offsets(
                    &padding,
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    filter_h,
                    filter_w,
                    stride_h,
                    stride_w,
                );
                let push = vec![
                    in_shape[0],
                    in_shape[1],
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    filter_h,
                    filter_w,
                    stride_h,
                    stride_w,
                    same_flag(&padding),
                    pad_top,
                    pad_left,
                    total as i32,
                ];
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "max_pool2d.spv",
                    &input.values,
                    &unused,
                    total,
                    &push,
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report("max_pool2d.spv", out.elapsed_ms, total));
                let values = apply_activation(
                    kernel_dir,
                    device,
                    &op_label(op),
                    activation,
                    out.values,
                    &unused,
                    &mut dispatches,
                    &mut device_report,
                )?;
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values,
                        layout: Layout::Nchw,
                    },
                );
            }
            BuiltinOp::Concatenation => {
                op_decoder::require_no_fused_activation(&options)?;
                let real_inputs = op
                    .inputs
                    .iter()
                    .copied()
                    .filter(|i| *i >= 0)
                    .collect::<Vec<_>>();
                if real_inputs.len() != 2 {
                    bail!(
                        "operator {} Concatenation currently supports exactly two inputs, got {:?}",
                        op.index,
                        op.inputs
                    );
                }
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let a_idx = real_inputs[0] as usize;
                let b_idx = real_inputs[1] as usize;
                let a = tensors
                    .get(&a_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, a_idx
                        )
                    })?
                    .clone();
                let b = tensors
                    .get(&b_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, b_idx
                        )
                    })?
                    .clone();
                let output_elements = element_count(model.tensor(output_idx)?)?;
                if a.values.len() + b.values.len() != output_elements {
                    bail!(
                        "operator {} Concatenation input elements {}+{} do not match output {}",
                        op.index,
                        a.values.len(),
                        b.values.len(),
                        output_elements
                    );
                }
                let layout = if a.layout == b.layout {
                    a.layout
                } else {
                    Layout::Flat
                };
                if layout == Layout::Nchw {
                    let out_shape = nchw_from_nhwc(&model.tensor(output_idx)?.shape)?;
                    if out_shape[0] != 1 {
                        bail!(
                            "operator {} Concatenation NCHW path currently supports batch=1, got {}",
                            op.index,
                            out_shape[0]
                        );
                    }
                }
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "concat.spv",
                    &a.values,
                    &b.values,
                    output_elements,
                    &[a.values.len() as i32, output_elements as i32],
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(
                    "concat.spv",
                    out.elapsed_ms,
                    output_elements,
                ));
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout,
                    },
                );
            }
            BuiltinOp::L2Normalization => {
                let input_idx = single_input(op.inputs.as_slice(), op.index)?;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                let elements = element_count(model.tensor(output_idx)?)?;
                if input.values.len() != elements {
                    bail!(
                        "operator {} L2Normalization input/output element count mismatch {} vs {}",
                        op.index,
                        input.values.len(),
                        elements
                    );
                }
                let norm = input.values.iter().map(|v| v * v).sum::<f32>().sqrt();
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "l2norm.spv",
                    &input.values,
                    &unused,
                    elements,
                    &[elements as i32, norm.to_bits() as i32],
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report("l2norm.spv", out.elapsed_ms, elements));
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout: input.layout,
                    },
                );
            }
            BuiltinOp::ResizeBilinear => {
                let input_idx = op
                    .inputs
                    .first()
                    .copied()
                    .filter(|idx| *idx >= 0)
                    .with_context(|| {
                        format!("operator {} ResizeBilinear missing data input", op.index)
                    })? as usize;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                if input.layout != Layout::Nchw {
                    bail!(
                        "operator {} ResizeBilinear requires NCHW activation layout",
                        op.index
                    );
                }
                let (align_corners, half_pixel_centers) = match &options {
                    DecodedOptions::Resize {
                        align_corners,
                        half_pixel_centers,
                    } => (*align_corners, *half_pixel_centers),
                    _ => (false, false),
                };
                let in_shape = nchw_from_nhwc(&model.tensor(input_idx)?.shape)?;
                let out_shape = nchw_from_nhwc(&model.tensor(output_idx)?.shape)?;
                if in_shape[0] != out_shape[0] || in_shape[1] != out_shape[1] {
                    bail!(
                        "operator {} ResizeBilinear requires unchanged N/C, got {:?} -> {:?}",
                        op.index,
                        in_shape,
                        out_shape
                    );
                }
                let total = element_count(model.tensor(output_idx)?)?;
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "resize_bilinear.spv",
                    &input.values,
                    &unused,
                    total,
                    &[
                        in_shape[0],
                        in_shape[1],
                        in_shape[2],
                        in_shape[3],
                        out_shape[2],
                        out_shape[3],
                        i32::from(align_corners),
                        i32::from(half_pixel_centers),
                        total as i32,
                    ],
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(
                    "resize_bilinear.spv",
                    out.elapsed_ms,
                    total,
                ));
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values: out.values,
                        layout: Layout::Nchw,
                    },
                );
            }
            BuiltinOp::FullyConnected => {
                let activation = match &options {
                    DecodedOptions::FullyConnected { fused_activation } => fused_activation.clone(),
                    _ => Activation::None,
                };
                let (input_idx, weight_idx, bias_idx) =
                    three_inputs(op.inputs.as_slice(), op.index)?;
                let output_idx = single_output(op.outputs.as_slice(), op.index)?;
                let input = tensors
                    .get(&input_idx)
                    .with_context(|| {
                        format!(
                            "operator {} input tensor {} not materialized",
                            op.index, input_idx
                        )
                    })?
                    .clone();
                let output_tensor = model.tensor(output_idx)?;
                let total = element_count(output_tensor)?;
                let out_shape = &output_tensor.shape;
                if out_shape.is_empty() {
                    bail!("operator {} FullyConnected output rank is zero", op.index);
                }
                let output_size = *out_shape.last().with_context(|| {
                    format!("operator {} FullyConnected output shape missing", op.index)
                })? as usize;
                if output_size == 0 || total % output_size != 0 {
                    bail!(
                        "operator {} FullyConnected output shape {:?} is unsupported",
                        op.index,
                        out_shape
                    );
                }
                let batches = total / output_size;
                let input_size = input.values.len() / batches;
                if input.values.len() != batches * input_size {
                    bail!(
                        "operator {} FullyConnected input elements {} are not divisible by batch count {}",
                        op.index,
                        input.values.len(),
                        batches
                    );
                }
                let weight = pack_fully_connected_weight(
                    model,
                    weight_idx,
                    bias_idx,
                    input_size,
                    output_size,
                )?;
                let out = dispatch(
                    kernel_dir,
                    device,
                    &op_label(op),
                    "fully_connected.spv",
                    &input.values,
                    &weight,
                    total,
                    &[
                        batches as i32,
                        input_size as i32,
                        output_size as i32,
                        total as i32,
                    ],
                )?;
                device_report = out_device(&out);
                dispatches.push(dispatch_report(
                    "fully_connected.spv",
                    out.elapsed_ms,
                    total,
                ));
                let values = apply_activation(
                    kernel_dir,
                    device,
                    &op_label(op),
                    activation,
                    out.values,
                    &unused,
                    &mut dispatches,
                    &mut device_report,
                )?;
                tensors.insert(
                    output_idx,
                    TensorValue {
                        values,
                        layout: Layout::Flat,
                    },
                );
            }
            ref other => bail!(
                "operator {} {:?} is recognized by the parser but has no scheduler lowering",
                op.index,
                other
            ),
        }
        let mut op_outputs = Vec::new();
        for &raw_idx in &op.outputs {
            if raw_idx < 0 {
                continue;
            }
            let idx = raw_idx as usize;
            if let Some(value) = tensors.get(&idx) {
                let path = dump_dir.join(format!("op{:03}_tensor_{idx}.raw.f32", op.index));
                fs::write(&path, f32_bytes(&value.values))?;
                op_outputs.push(json!({
                    "tensor": idx,
                    "path": path,
                    "elements": value.values.len(),
                    "layout": value.layout,
                    "sha256": sha256(f32_bytes(&value.values)),
                    "shape": model.tensor(idx)?.shape,
                }));
            }
        }
        op_reports.push(json!({
            "index": op.index,
            "op": format!("{:?}", op.builtin),
            "options": options,
            "inputs": op.inputs,
            "outputs": op.outputs,
            "output_tensors": op_outputs,
            "dispatches": dispatches,
            "materialized_tensors_before": started_tensor_count,
            "materialized_tensors_after": tensors.len()
        }));
    }

    let mut output_reports = Vec::new();
    let mut primary_output = Vec::new();
    for &out_idx_raw in &model.outputs {
        let out_idx = valid_tensor_index(out_idx_raw)?;
        let value = tensors
            .get(&out_idx)
            .with_context(|| format!("model output tensor {out_idx} not materialized"))?;
        let path = dump_dir.join(format!("output_tensor_{out_idx}.raw.f32"));
        fs::write(&path, f32_bytes(&value.values))?;
        if primary_output.is_empty() {
            primary_output = value.values.clone();
            fs::write(dump_dir.join("output.raw.f32"), f32_bytes(&value.values))?;
        }
        output_reports.push(json!({
            "tensor": out_idx,
            "path": path,
            "elements": value.values.len(),
            "layout": value.layout,
            "sha256": sha256(f32_bytes(&value.values))
        }));
    }
    let reference = load_reference_output(&model.path, primary_output.len())?;
    let comparison = reference
        .as_ref()
        .map(|r| compare(&primary_output, r, 1.0e-4));
    Ok(json!({
        "status": if comparison.as_ref().is_some_and(|c| c.mismatch_count != 0) { "fail" } else { "pass" },
        "execution_kind": "generic_tflite_scheduler_multi_op_vulkan_spirv",
        "device": device_report,
        "operator_count": model.operators.len(),
        "inputs": input_reports,
        "topological_order": order,
        "ops": op_reports,
        "outputs": output_reports,
        "top1": top1(&primary_output),
        "top5": topk(&primary_output, 5),
        "epsilon": 1.0e-4,
        "reference": reference.as_ref().map(|r| json!({"sha256": sha256(f32_bytes(r)), "elements": r.len()})),
        "comparison": comparison,
        "liveness": liveness_report(model)
    }))
}

trait TensorLookup {
    fn tensor(&self, idx: usize) -> Result<&TensorInfo>;
}

impl TensorLookup for ModelInfo {
    fn tensor(&self, idx: usize) -> Result<&TensorInfo> {
        self.tensors
            .get(idx)
            .with_context(|| format!("tensor {idx} out of range"))
    }
}

fn dispatch(
    kernel_dir: &Path,
    device: &str,
    label: &str,
    spv_name: &str,
    input0: &[f32],
    input1: &[f32],
    output_elements: usize,
    push: &[i32],
) -> Result<crate::vulkan::KernelOutput> {
    dispatch_with_local(
        kernel_dir,
        device,
        label,
        spv_name,
        input0,
        input1,
        output_elements,
        push,
        64,
    )
}

fn dispatch_with_local(
    kernel_dir: &Path,
    device: &str,
    label: &str,
    spv_name: &str,
    input0: &[f32],
    input1: &[f32],
    output_elements: usize,
    push: &[i32],
    local: usize,
) -> Result<crate::vulkan::KernelOutput> {
    let path = kernels::resolve_spv(kernel_dir, spv_name);
    let spv = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    run_kernel(
        device,
        &KernelRun {
            label,
            spv: &spv,
            input0,
            input1,
            output_elements,
            dispatch_x: groups(output_elements, local),
            push,
        },
    )
}

fn apply_activation(
    kernel_dir: &Path,
    device: &str,
    label: &str,
    activation: Activation,
    values: Vec<f32>,
    unused: &[f32],
    dispatches: &mut Vec<serde_json::Value>,
    device_report: &mut serde_json::Value,
) -> Result<Vec<f32>> {
    match activation {
        Activation::None => Ok(values),
        Activation::Relu => {
            let total = values.len();
            let out = dispatch_with_local(
                kernel_dir,
                device,
                label,
                "relu.spv",
                &values,
                unused,
                total,
                &[total as i32],
                128,
            )?;
            *device_report = out_device(&out);
            dispatches.push(dispatch_report("relu.spv", out.elapsed_ms, total));
            Ok(out.values)
        }
        Activation::Relu6 => {
            let total = values.len();
            let out = dispatch_with_local(
                kernel_dir,
                device,
                label,
                "relu6.spv",
                &values,
                unused,
                total,
                &[total as i32],
                128,
            )?;
            *device_report = out_device(&out);
            dispatches.push(dispatch_report("relu6.spv", out.elapsed_ms, total));
            Ok(out.values)
        }
        other => bail!(
            "fused activation {:?} has no scheduler post-op lowering",
            other
        ),
    }
}

fn load_constants(model: &ModelInfo, tensors: &mut BTreeMap<usize, TensorValue>) -> Result<()> {
    for tensor in &model.tensors {
        if model.inputs.contains(&(tensor.index as i32)) || tensor.buffer == 0 {
            continue;
        }
        let Some(buffer) = model.buffers.get(tensor.buffer as usize) else {
            continue;
        };
        if buffer.data.is_empty() || tensor.tensor_type != TensorType::Float32 {
            continue;
        }
        tensors.insert(
            tensor.index,
            TensorValue {
                values: read_f32_bytes(&buffer.data)
                    .with_context(|| format!("reading constant tensor {}", tensor.index))?,
                layout: Layout::Flat,
            },
        );
    }
    Ok(())
}

fn load_input(
    tensor: &TensorInfo,
    input_f32: Option<&Path>,
) -> Result<(Vec<f32>, serde_json::Value)> {
    let elements = element_count(tensor)?;
    if let Some(path) = input_f32 {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if bytes.len() != elements * 4 {
            bail!(
                "{} has {} bytes, expected {} for tensor {}",
                path.display(),
                bytes.len(),
                elements * 4,
                tensor.index
            );
        }
        let values =
            read_f32_bytes(&bytes).with_context(|| format!("reading {}", path.display()))?;
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
            bail!(
                "{} has {} bytes, expected {} for tensor {}",
                path.display(),
                bytes.len(),
                elements * 4,
                tensor.index
            );
        }
        let values =
            read_f32_bytes(&bytes).with_context(|| format!("reading {}", path.display()))?;
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
    for path in [
        "mobilenet/fixtures/input.raw.f32",
        "mobilenet/fixtures/imagenet_input.bin",
        "mobilenet/fixtures/input.bin",
    ] {
        if let Ok(bytes) = fs::read(path) {
            if bytes.len() == elements * 4 {
                let values = read_f32_bytes(&bytes).with_context(|| format!("reading {path}"))?;
                return Ok((
                    values,
                    json!({
                        "kind": "fixture_auto_discovery",
                        "path": path,
                        "bytes": bytes.len(),
                        "sha256": sha256(&bytes)
                    }),
                ));
            }
        }
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

fn load_reference_output(model_path: &str, elements: usize) -> Result<Option<Vec<f32>>> {
    if !model_path.to_ascii_lowercase().contains("mobilenet") {
        return Ok(None);
    }
    for path in [
        "mobilenet/fixtures/softmax.raw.f32",
        "mobilenet/fixtures/softmax.bin",
    ] {
        if let Ok(bytes) = fs::read(path) {
            if bytes.len() == elements * 4 {
                return Ok(Some(
                    read_f32_bytes(&bytes).with_context(|| format!("reading {path}"))?,
                ));
            }
        }
    }
    Ok(None)
}

fn pack_conv_weight(model: &ModelInfo, weight_idx: usize, bias_idx: usize) -> Result<Vec<f32>> {
    let w_tensor = model.tensor(weight_idx)?;
    let shape = &w_tensor.shape;
    if shape.len() != 4 {
        bail!(
            "Conv2D weight tensor {weight_idx} rank {:?} is not 4",
            shape
        );
    }
    let oc = shape[0] as usize;
    let kh = shape[1] as usize;
    let kw = shape[2] as usize;
    let ic = shape[3] as usize;
    let src = const_f32(model, weight_idx)?;
    let bias = const_f32(model, bias_idx)?;
    if bias.len() != oc {
        bail!(
            "Conv2D bias tensor {bias_idx} has {} elements, expected {oc}",
            bias.len()
        );
    }
    let mut out = vec![0.0; oc * ic * kh * kw + oc];
    for o in 0..oc {
        for i in 0..ic {
            for y in 0..kh {
                for x in 0..kw {
                    let src_idx = ((o * kh + y) * kw + x) * ic + i;
                    let dst_idx = ((o * ic + i) * kh + y) * kw + x;
                    out[dst_idx] = src[src_idx];
                }
            }
        }
        out[oc * ic * kh * kw + o] = bias[o];
    }
    Ok(out)
}

fn pack_depthwise_weight(
    model: &ModelInfo,
    weight_idx: usize,
    bias_idx: usize,
) -> Result<Vec<f32>> {
    let w_tensor = model.tensor(weight_idx)?;
    let shape = &w_tensor.shape;
    if shape.len() != 4 || shape[0] != 1 {
        bail!(
            "DepthwiseConv2D weight tensor {weight_idx} shape {:?} is unsupported",
            shape
        );
    }
    let kh = shape[1] as usize;
    let kw = shape[2] as usize;
    let c = shape[3] as usize;
    let src = const_f32(model, weight_idx)?;
    let bias = const_f32(model, bias_idx)?;
    if bias.len() != c {
        bail!(
            "DepthwiseConv2D bias tensor {bias_idx} has {} elements, expected {c}",
            bias.len()
        );
    }
    let mut out = vec![0.0; c * kh * kw + c];
    for ch in 0..c {
        for y in 0..kh {
            for x in 0..kw {
                let src_idx = (y * kw + x) * c + ch;
                let dst_idx = (ch * kh + y) * kw + x;
                out[dst_idx] = src[src_idx];
            }
        }
        out[c * kh * kw + ch] = bias[ch];
    }
    Ok(out)
}

fn pack_fully_connected_weight(
    model: &ModelInfo,
    weight_idx: usize,
    bias_idx: usize,
    input_size: usize,
    output_size: usize,
) -> Result<Vec<f32>> {
    let w_tensor = model.tensor(weight_idx)?;
    let shape = &w_tensor.shape;
    if shape.len() != 2 || shape[0] as usize != output_size || shape[1] as usize != input_size {
        bail!(
            "FullyConnected weight tensor {weight_idx} shape {:?} does not match output_size={output_size}, input_size={input_size}",
            shape
        );
    }
    let src = const_f32(model, weight_idx)?;
    let bias = const_f32(model, bias_idx)?;
    if src.len() != output_size * input_size {
        bail!(
            "FullyConnected weight tensor {weight_idx} has {} elements, expected {}",
            src.len(),
            output_size * input_size
        );
    }
    if bias.len() != output_size {
        bail!(
            "FullyConnected bias tensor {bias_idx} has {} elements, expected {output_size}",
            bias.len()
        );
    }
    let mut out = vec![0.0; output_size * input_size + output_size];
    out[..src.len()].copy_from_slice(&src);
    out[src.len()..].copy_from_slice(&bias);
    Ok(out)
}

fn const_f32(model: &ModelInfo, tensor_idx: usize) -> Result<Vec<f32>> {
    let tensor = model.tensor(tensor_idx)?;
    if tensor.tensor_type != TensorType::Float32 {
        bail!(
            "tensor {tensor_idx} is {:?}, expected Float32",
            tensor.tensor_type
        );
    }
    let buffer = model
        .buffers
        .get(tensor.buffer as usize)
        .with_context(|| format!("tensor {tensor_idx} buffer {} out of range", tensor.buffer))?;
    read_f32_bytes(&buffer.data)
        .with_context(|| format!("reading tensor {tensor_idx} buffer {}", tensor.buffer))
}

fn topological_order(model: &ModelInfo) -> Result<Vec<usize>> {
    let mut producer = BTreeMap::new();
    for op in &model.operators {
        for &out in &op.outputs {
            if out >= 0 {
                if let Some(prev) = producer.insert(out, op.index) {
                    bail!(
                        "tensor {out} has multiple producers: {prev} and {}",
                        op.index
                    );
                }
            }
        }
    }
    let mut done = BTreeSet::new();
    let mut order = Vec::new();
    while order.len() < model.operators.len() {
        let mut progressed = false;
        for op in &model.operators {
            if done.contains(&op.index) {
                continue;
            }
            let ready = op.inputs.iter().all(|&input| {
                input < 0
                    || model.inputs.contains(&input)
                    || producer
                        .get(&input)
                        .is_none_or(|producer_idx| done.contains(producer_idx))
            });
            if ready {
                done.insert(op.index);
                order.push(op.index);
                progressed = true;
            }
        }
        if !progressed {
            bail!("operator graph contains a cycle or unresolved dependency");
        }
    }
    Ok(order)
}

fn liveness_report(model: &ModelInfo) -> serde_json::Value {
    let naive_bytes: usize = model
        .tensors
        .iter()
        .filter(|t| t.tensor_type == TensorType::Float32)
        .filter_map(|t| element_count(t).ok())
        .map(|n| n * 4)
        .sum();
    json!({
        "policy": "materialize_constants_and_outputs_with_fail_closed_static_shapes",
        "naive_float32_bytes": naive_bytes,
        "pooled_peak_bytes": naive_bytes,
        "reuse_count": 0
    })
}

fn element_count(tensor: &TensorInfo) -> Result<usize> {
    tensor.shape.iter().try_fold(1_usize, |acc, dim| {
        if *dim <= 0 {
            bail!(
                "tensor {} has dynamic or invalid dimension {dim}",
                tensor.index
            );
        }
        Ok(acc * *dim as usize)
    })
}

fn valid_tensor_index(raw: i32) -> Result<usize> {
    if raw < 0 {
        bail!("optional tensor index {raw} is not valid in this context");
    }
    Ok(raw as usize)
}

fn single_input(inputs: &[i32], op_index: usize) -> Result<usize> {
    let real = inputs
        .iter()
        .copied()
        .filter(|i| *i >= 0)
        .collect::<Vec<_>>();
    if real.len() != 1 {
        bail!("operator {op_index} expected one input, got {:?}", inputs);
    }
    Ok(real[0] as usize)
}

fn two_inputs(inputs: &[i32], op_index: usize) -> Result<(usize, usize)> {
    let real = inputs
        .iter()
        .copied()
        .filter(|i| *i >= 0)
        .collect::<Vec<_>>();
    if real.len() != 2 {
        bail!("operator {op_index} expected two inputs, got {:?}", inputs);
    }
    Ok((real[0] as usize, real[1] as usize))
}

fn three_inputs(inputs: &[i32], op_index: usize) -> Result<(usize, usize, usize)> {
    let real = inputs
        .iter()
        .copied()
        .filter(|i| *i >= 0)
        .collect::<Vec<_>>();
    if real.len() < 3 {
        bail!(
            "operator {op_index} expected input, weight, bias tensors, got {:?}",
            inputs
        );
    }
    Ok((real[0] as usize, real[1] as usize, real[2] as usize))
}

fn single_output(outputs: &[i32], op_index: usize) -> Result<usize> {
    let real = outputs
        .iter()
        .copied()
        .filter(|i| *i >= 0)
        .collect::<Vec<_>>();
    if real.len() != 1 {
        bail!("operator {op_index} expected one output, got {:?}", outputs);
    }
    Ok(real[0] as usize)
}

fn nchw_from_nhwc(shape: &[i32]) -> Result<[i32; 4]> {
    if shape.len() != 4 {
        bail!("expected rank-4 NHWC tensor, got {:?}", shape);
    }
    Ok([shape[0], shape[3], shape[1], shape[2]])
}

fn same_flag(padding: &Padding) -> i32 {
    if *padding == Padding::Same {
        1
    } else {
        0
    }
}

fn same_padding_offsets(
    padding: &Padding,
    ih: i32,
    iw: i32,
    oh: i32,
    ow: i32,
    kh: i32,
    kw: i32,
    stride_h: i32,
    stride_w: i32,
) -> (i32, i32) {
    if *padding != Padding::Same {
        return (0, 0);
    }
    let total_h = ((oh - 1) * stride_h + kh - ih).max(0);
    let total_w = ((ow - 1) * stride_w + kw - iw).max(0);
    (total_h / 2, total_w / 2)
}

fn op_label(op: &crate::model::OperatorInfo) -> String {
    format!("op{}_ {:?}", op.index, op.builtin).replace(' ', "")
}

fn dispatch_report(spv: &str, elapsed_ms: f64, output_elements: usize) -> serde_json::Value {
    json!({
        "spv": spv,
        "elapsed_ms": elapsed_ms,
        "output_elements": output_elements
    })
}

fn out_device(out: &crate::vulkan::KernelOutput) -> serde_json::Value {
    json!({
        "selector_result": "matched",
        "name": out.device_name,
        "vendor_id": format!("0x{:04x}", out.vendor_id),
        "device_id": format!("0x{:04x}", out.device_id)
    })
}

fn top1(values: &[f32]) -> serde_json::Value {
    let offset = imagenet_class_offset(values);
    let (model_class, confidence) = values
        .iter()
        .copied()
        .enumerate()
        .skip(offset)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    json!({
        "class": model_class.saturating_sub(offset),
        "model_class": model_class,
        "confidence": confidence,
        "class_index_basis": if offset == 1 { "imagenet_1000_without_tflite_background" } else { "model_output_index" }
    })
}

fn topk(values: &[f32], k: usize) -> serde_json::Value {
    let offset = imagenet_class_offset(values);
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    json!(indexed
        .into_iter()
        .filter(|(model_class, _)| *model_class >= offset)
        .take(k)
        .map(|(model_class, confidence)| json!({
            "class": model_class.saturating_sub(offset),
            "model_class": model_class,
            "confidence": confidence,
            "class_index_basis": if offset == 1 { "imagenet_1000_without_tflite_background" } else { "model_output_index" }
        }))
        .collect::<Vec<_>>())
}

fn imagenet_class_offset(values: &[f32]) -> usize {
    if values.len() == 1001 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BufferInfo, OperatorInfo};

    #[test]
    fn topological_order_preserves_ready_model_order() {
        let model = ModelInfo {
            path: "synthetic".into(),
            bytes: 0,
            raw: Vec::new(),
            version: 3,
            description: None,
            tensors: vec![tensor(0), tensor(1), tensor(2)],
            inputs: vec![0],
            outputs: vec![2],
            operators: vec![op(0, vec![0], vec![1]), op(1, vec![1], vec![2])],
            buffers: vec![BufferInfo {
                index: 0,
                data_offset: None,
                data_len: 0,
                data_sha256: sha256([]),
                data: Vec::new(),
            }],
            op_histogram: BTreeMap::new(),
            unsupported_ops: Vec::new(),
            quantized_tensors: Vec::new(),
        };
        assert_eq!(topological_order(&model).unwrap(), vec![0, 1]);
    }

    fn tensor(index: usize) -> TensorInfo {
        TensorInfo {
            index,
            name: format!("t{index}"),
            shape: vec![1],
            tensor_type: TensorType::Float32,
            buffer: 0,
        }
    }

    fn op(index: usize, inputs: Vec<i32>, outputs: Vec<i32>) -> OperatorInfo {
        OperatorInfo {
            index,
            opcode_index: 0,
            builtin: BuiltinOp::Logistic,
            inputs,
            outputs,
            builtin_options_type: None,
            builtin_options_table_pos: None,
        }
    }
}
