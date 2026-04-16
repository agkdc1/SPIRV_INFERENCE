use crate::model::{BuiltinOp, ModelInfo, OperatorInfo, Table};
use anyhow::{bail, Result};

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub enum Padding {
    Same,
    Valid,
    Unknown(i8),
}

impl Padding {
    fn from_i8(v: i8) -> Self {
        match v {
            0 => Self::Same,
            1 => Self::Valid,
            x => Self::Unknown(x),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub enum Activation {
    None,
    Relu,
    ReluN1To1,
    Relu6,
    Tanh,
    SignBit,
    Unknown(i8),
}

impl Activation {
    fn from_i8(v: i8) -> Self {
        match v {
            0 => Self::None,
            1 => Self::Relu,
            2 => Self::ReluN1To1,
            3 => Self::Relu6,
            4 => Self::Tanh,
            5 => Self::SignBit,
            x => Self::Unknown(x),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecodedOptions {
    Conv2d {
        padding: Padding,
        stride_w: i32,
        stride_h: i32,
        fused_activation: Activation,
        dilation_w_factor: i32,
        dilation_h_factor: i32,
    },
    DepthwiseConv2d {
        padding: Padding,
        stride_w: i32,
        stride_h: i32,
        depth_multiplier: i32,
        fused_activation: Activation,
        dilation_w_factor: i32,
        dilation_h_factor: i32,
    },
    Pool2d {
        padding: Padding,
        stride_w: i32,
        stride_h: i32,
        filter_w: i32,
        filter_h: i32,
        fused_activation: Activation,
    },
    Elementwise {
        fused_activation: Activation,
    },
    Concatenation {
        axis: i32,
        fused_activation: Activation,
    },
    FullyConnected {
        fused_activation: Activation,
    },
    Softmax {
        beta: f32,
    },
    Reshape {
        new_shape: Vec<i32>,
    },
    LeakyRelu {
        alpha: f32,
    },
    Squeeze {
        squeeze_dims: Vec<i32>,
    },
    Split {
        num_splits: i32,
    },
    Gather {
        axis: i32,
        batch_dims: i32,
    },
    Pack {
        values_count: i32,
        axis: i32,
    },
    Unpack {
        num: i32,
        axis: i32,
    },
    Resize {
        align_corners: bool,
        half_pixel_centers: bool,
    },
    Reducer {
        keep_dims: bool,
    },
    ArgMax,
    TopK,
    Cast,
    Empty,
}

pub fn decode_options(model: &ModelInfo, op: &OperatorInfo) -> Result<DecodedOptions> {
    let Some(pos) = op.builtin_options_table_pos else {
        return Ok(DecodedOptions::Empty);
    };
    let table = Table {
        bytes: &model.raw,
        pos,
    };
    let activation =
        |field| -> Result<Activation> { Ok(Activation::from_i8(table.i8(field)?.unwrap_or(0))) };
    Ok(match op.builtin {
        BuiltinOp::Conv2d => DecodedOptions::Conv2d {
            padding: Padding::from_i8(table.i8(0)?.unwrap_or(0)),
            stride_w: table.i32(1)?.unwrap_or(1),
            stride_h: table.i32(2)?.unwrap_or(1),
            fused_activation: activation(3)?,
            dilation_w_factor: table.i32(4)?.unwrap_or(1),
            dilation_h_factor: table.i32(5)?.unwrap_or(1),
        },
        BuiltinOp::DepthwiseConv2d => DecodedOptions::DepthwiseConv2d {
            padding: Padding::from_i8(table.i8(0)?.unwrap_or(0)),
            stride_w: table.i32(1)?.unwrap_or(1),
            stride_h: table.i32(2)?.unwrap_or(1),
            depth_multiplier: table.i32(3)?.unwrap_or(1),
            fused_activation: activation(4)?,
            dilation_w_factor: table.i32(5)?.unwrap_or(1),
            dilation_h_factor: table.i32(6)?.unwrap_or(1),
        },
        BuiltinOp::AveragePool2d | BuiltinOp::MaxPool2d => DecodedOptions::Pool2d {
            padding: Padding::from_i8(table.i8(0)?.unwrap_or(0)),
            stride_w: table.i32(1)?.unwrap_or(1),
            stride_h: table.i32(2)?.unwrap_or(1),
            filter_w: table.i32(3)?.unwrap_or(1),
            filter_h: table.i32(4)?.unwrap_or(1),
            fused_activation: activation(5)?,
        },
        BuiltinOp::Add | BuiltinOp::Mul | BuiltinOp::Sub | BuiltinOp::Div => {
            DecodedOptions::Elementwise {
                fused_activation: activation(0)?,
            }
        }
        BuiltinOp::Concatenation => DecodedOptions::Concatenation {
            axis: table.i32(0)?.unwrap_or(0),
            fused_activation: activation(1)?,
        },
        BuiltinOp::FullyConnected => DecodedOptions::FullyConnected {
            fused_activation: activation(0)?,
        },
        BuiltinOp::Softmax => DecodedOptions::Softmax {
            beta: table.f32(0)?.unwrap_or(1.0),
        },
        BuiltinOp::Reshape => DecodedOptions::Reshape {
            new_shape: table.vector_i32(0)?.unwrap_or_default(),
        },
        BuiltinOp::LeakyRelu => DecodedOptions::LeakyRelu {
            alpha: table.f32(0)?.unwrap_or(0.2),
        },
        BuiltinOp::Squeeze => DecodedOptions::Squeeze {
            squeeze_dims: table.vector_i32(0)?.unwrap_or_default(),
        },
        BuiltinOp::Split => DecodedOptions::Split {
            num_splits: table.i32(0)?.unwrap_or(0),
        },
        BuiltinOp::Gather => DecodedOptions::Gather {
            axis: table.i32(0)?.unwrap_or(0),
            batch_dims: table.i32(1)?.unwrap_or(0),
        },
        BuiltinOp::Pack => DecodedOptions::Pack {
            values_count: table.i32(0)?.unwrap_or(0),
            axis: table.i32(1)?.unwrap_or(0),
        },
        BuiltinOp::Unpack => DecodedOptions::Unpack {
            num: table.i32(0)?.unwrap_or(0),
            axis: table.i32(1)?.unwrap_or(0),
        },
        BuiltinOp::ResizeBilinear | BuiltinOp::ResizeNearestNeighbor => DecodedOptions::Resize {
            align_corners: table.bool(0)?.unwrap_or(false),
            half_pixel_centers: table.bool(1)?.unwrap_or(false),
        },
        BuiltinOp::Mean | BuiltinOp::Sum | BuiltinOp::ReduceMax => DecodedOptions::Reducer {
            keep_dims: table.bool(0)?.unwrap_or(false),
        },
        BuiltinOp::ArgMax => DecodedOptions::ArgMax,
        BuiltinOp::TopKV2 => DecodedOptions::TopK,
        BuiltinOp::Cast => DecodedOptions::Cast,
        _ => DecodedOptions::Empty,
    })
}

pub fn require_no_fused_activation(options: &DecodedOptions) -> Result<()> {
    let activation = match options {
        DecodedOptions::Conv2d {
            fused_activation, ..
        }
        | DecodedOptions::DepthwiseConv2d {
            fused_activation, ..
        }
        | DecodedOptions::Pool2d {
            fused_activation, ..
        }
        | DecodedOptions::Concatenation {
            fused_activation, ..
        }
        | DecodedOptions::FullyConnected { fused_activation }
        | DecodedOptions::Elementwise {
            fused_activation, ..
        } => fused_activation,
        _ => return Ok(()),
    };
    if *activation != Activation::None {
        bail!(
            "fused activation {:?} requires a post-op kernel path",
            activation
        );
    }
    Ok(())
}
