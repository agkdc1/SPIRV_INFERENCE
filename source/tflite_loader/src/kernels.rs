use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, serde::Serialize)]
pub struct KernelSpec {
    pub name: &'static str,
    pub spv: &'static str,
    pub bindings: usize,
    pub push_constants: &'static [&'static str],
}

pub const TFLITE_KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "add",
        spv: "add.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "sub",
        spv: "sub.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "mul",
        spv: "mul.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "div",
        spv: "div.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "pow",
        spv: "pow.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "min",
        spv: "min.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "max",
        spv: "max.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "exp",
        spv: "exp.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "neg",
        spv: "neg.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "abs",
        spv: "abs.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "sqrt",
        spv: "sqrt.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "rsqrt",
        spv: "rsqrt.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "log",
        spv: "log.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "leaky_relu",
        spv: "leaky_relu.spv",
        bindings: 3,
        push_constants: &["total", "alpha_bits"],
    },
    KernelSpec {
        name: "logistic",
        spv: "sigmoid.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "sigmoid",
        spv: "sigmoid.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "tanh",
        spv: "tanh.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "relu6",
        spv: "relu6.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "relu",
        spv: "relu.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "reshape",
        spv: "reshape.spv",
        bindings: 3,
        push_constants: &["total"],
    },
    KernelSpec {
        name: "concat",
        spv: "concat.spv",
        bindings: 3,
        push_constants: &["a_total", "total"],
    },
    KernelSpec {
        name: "transpose",
        spv: "transpose.spv",
        bindings: 3,
        push_constants: &["n", "c", "h", "w", "direction"],
    },
    KernelSpec {
        name: "conv2d",
        spv: "conv2d.spv",
        bindings: 3,
        push_constants: &[
            "n", "ic", "ih", "iw", "oc", "oh", "ow", "kh", "kw", "stride_h", "stride_w", "same",
            "pad_top", "pad_left", "total",
        ],
    },
    KernelSpec {
        name: "depthwise_conv2d",
        spv: "depthwise_conv2d.spv",
        bindings: 3,
        push_constants: &[
            "n",
            "c",
            "ih",
            "iw",
            "oh",
            "ow",
            "kh",
            "kw",
            "stride_h",
            "stride_w",
            "same",
            "pad_top",
            "pad_left",
            "dilation_h",
            "dilation_w",
            "total",
        ],
    },
    KernelSpec {
        name: "global_avg_pool_sum",
        spv: "global_avg_pool_sum.spv",
        bindings: 3,
        push_constants: &["n", "c", "h", "w"],
    },
    KernelSpec {
        name: "max_pool2d",
        spv: "max_pool2d.spv",
        bindings: 3,
        push_constants: &[
            "n", "c", "ih", "iw", "oh", "ow", "kh", "kw", "stride_h", "stride_w", "same",
            "pad_top", "pad_left", "total",
        ],
    },
    KernelSpec {
        name: "l2norm",
        spv: "l2norm.spv",
        bindings: 3,
        push_constants: &["total", "norm_bits"],
    },
    KernelSpec {
        name: "resize_bilinear",
        spv: "resize_bilinear.spv",
        bindings: 3,
        push_constants: &[
            "n",
            "c",
            "ih",
            "iw",
            "oh",
            "ow",
            "align_corners",
            "half_pixel_centers",
            "total",
        ],
    },
    KernelSpec {
        name: "global_avg_pool_finalize",
        spv: "global_avg_pool_finalize.spv",
        bindings: 3,
        push_constants: &["n", "c", "h", "w"],
    },
    KernelSpec {
        name: "fully_connected",
        spv: "fully_connected.spv",
        bindings: 3,
        push_constants: &["batches", "input_size", "output_size", "total"],
    },
    KernelSpec {
        name: "softmax10",
        spv: "softmax10.spv",
        bindings: 3,
        push_constants: &["classes"],
    },
    KernelSpec {
        name: "softmax1000",
        spv: "softmax1000.spv",
        bindings: 3,
        push_constants: &["classes"],
    },
];

pub fn registry() -> BTreeMap<&'static str, &'static KernelSpec> {
    TFLITE_KERNELS.iter().map(|k| (k.name, k)).collect()
}

pub fn resolve_spv(kernel_dir: &Path, spv: &str) -> PathBuf {
    let direct = kernel_dir.join(spv);
    if direct.exists() {
        return direct;
    }
    let stem = spv.strip_suffix(".spv").unwrap_or(spv);
    match stem {
        "global_avg_pool_sum" | "global_avg_pool_finalize" => {
            kernel_dir.join("global_avg_pool").join("spv").join(spv)
        }
        "softmax10" => kernel_dir.join("softmax").join("spv").join(spv),
        other => kernel_dir.join(other).join("spv").join(spv),
    }
}
