use anyhow::{bail, Context, Result};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
pub enum TensorType {
    Float32,
    Int32,
    UInt8,
    Int64,
    String,
    Bool,
    Int16,
    Complex64,
    Int8,
    Float16,
    Float64,
    Unknown(i8),
}

impl TensorType {
    fn from_i8(v: i8) -> Self {
        match v {
            0 => Self::Float32,
            2 => Self::Int32,
            3 => Self::UInt8,
            4 => Self::Int64,
            5 => Self::String,
            6 => Self::Bool,
            7 => Self::Int16,
            8 => Self::Complex64,
            9 => Self::Int8,
            10 => Self::Float16,
            11 => Self::Float64,
            x => Self::Unknown(x),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum BuiltinOp {
    Abs,
    Add,
    ArgMax,
    AveragePool2d,
    BatchMatmul,
    Cast,
    Concatenation,
    Conv2d,
    DepthwiseConv2d,
    Div,
    EmbeddingLookup,
    Exp,
    ExpandDims,
    FullyConnected,
    Gather,
    Gelu,
    L2Normalization,
    LeakyRelu,
    Logistic,
    Log,
    Maximum,
    MaxPool2d,
    Mean,
    Minimum,
    Mul,
    Neg,
    Pad,
    Pack,
    Pow,
    ReduceMax,
    Relu6,
    Reshape,
    ResizeBilinear,
    ResizeNearestNeighbor,
    Rsqrt,
    Slice,
    Softmax,
    Split,
    Sqrt,
    Squeeze,
    Sub,
    Sum,
    Tanh,
    Tile,
    TopKV2,
    Transpose,
    TransposeConv,
    Unpack,
    Custom(String),
    Unknown(i32),
}

impl BuiltinOp {
    fn from_code(code: i32, custom: Option<String>) -> Self {
        match code {
            0 => Self::Add,
            1 => Self::AveragePool2d,
            2 => Self::Concatenation,
            3 => Self::Conv2d,
            4 => Self::DepthwiseConv2d,
            7 => Self::EmbeddingLookup,
            9 => Self::FullyConnected,
            11 => Self::L2Normalization,
            14 => Self::Logistic,
            17 => Self::MaxPool2d,
            18 => Self::Mul,
            23 => Self::ResizeBilinear,
            21 => Self::Relu6,
            22 => Self::Reshape,
            25 => Self::Softmax,
            28 => Self::Tanh,
            32_768 => Self::Custom(custom.unwrap_or_else(|| "CUSTOM".to_string())),
            32 => Self::Custom(custom.unwrap_or_else(|| "CUSTOM".to_string())),
            34 => Self::Pad,
            36 => Self::Gather,
            39 => Self::Transpose,
            40 => Self::Mean,
            41 => Self::Sub,
            42 => Self::Div,
            43 => Self::Squeeze,
            47 => Self::Exp,
            48 => Self::TopKV2,
            49 => Self::Split,
            53 => Self::Cast,
            55 => Self::Maximum,
            56 => Self::ArgMax,
            57 => Self::Minimum,
            59 => Self::Neg,
            65 => Self::Slice,
            67 => Self::TransposeConv,
            69 => Self::Tile,
            70 => Self::ExpandDims,
            73 => Self::Log,
            74 => Self::Sum,
            75 => Self::Sqrt,
            76 => Self::Rsqrt,
            78 => Self::Pow,
            82 => Self::ReduceMax,
            83 => Self::Pack,
            88 => Self::Unpack,
            97 => Self::ResizeNearestNeighbor,
            98 => Self::LeakyRelu,
            101 => Self::Abs,
            126 => Self::BatchMatmul,
            150 => Self::Gelu,
            x => Self::Unknown(x),
        }
    }
}

impl BuiltinOp {
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            Self::Abs
                | Self::Add
                | Self::ArgMax
                | Self::AveragePool2d
                | Self::BatchMatmul
                | Self::Cast
                | Self::Concatenation
                | Self::Conv2d
                | Self::DepthwiseConv2d
                | Self::Div
                | Self::EmbeddingLookup
                | Self::Exp
                | Self::ExpandDims
                | Self::FullyConnected
                | Self::Gather
                | Self::Gelu
                | Self::L2Normalization
                | Self::LeakyRelu
                | Self::Logistic
                | Self::Log
                | Self::Maximum
                | Self::MaxPool2d
                | Self::Mean
                | Self::Minimum
                | Self::Mul
                | Self::Neg
                | Self::Pad
                | Self::Pack
                | Self::Pow
                | Self::ReduceMax
                | Self::Relu6
                | Self::Reshape
                | Self::ResizeBilinear
                | Self::ResizeNearestNeighbor
                | Self::Rsqrt
                | Self::Slice
                | Self::Softmax
                | Self::Split
                | Self::Sqrt
                | Self::Squeeze
                | Self::Sub
                | Self::Sum
                | Self::Tanh
                | Self::Tile
                | Self::TopKV2
                | Self::Transpose
                | Self::TransposeConv
                | Self::Unpack
        )
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorInfo {
    pub index: usize,
    pub name: String,
    pub shape: Vec<i32>,
    pub tensor_type: TensorType,
    pub buffer: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OperatorInfo {
    pub index: usize,
    pub opcode_index: u32,
    pub builtin: BuiltinOp,
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
    pub builtin_options_type: Option<i8>,
    pub builtin_options_table_pos: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BufferInfo {
    pub index: usize,
    pub data_offset: Option<usize>,
    pub data_len: usize,
    pub data_sha256: String,
    #[serde(skip_serializing)]
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub path: String,
    pub bytes: usize,
    #[serde(skip_serializing)]
    pub raw: Vec<u8>,
    pub version: i32,
    pub description: Option<String>,
    pub tensors: Vec<TensorInfo>,
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
    pub operators: Vec<OperatorInfo>,
    pub buffers: Vec<BufferInfo>,
    pub op_histogram: BTreeMap<String, usize>,
    pub unsupported_ops: Vec<String>,
    pub quantized_tensors: Vec<String>,
}

pub fn load_model(path: &Path) -> Result<ModelInfo> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse_model(path, &bytes)
}

pub fn parse_model(path: &Path, bytes: &[u8]) -> Result<ModelInfo> {
    if bytes.len() < 8 {
        bail!("{} is too small to be a TFLite FlatBuffer", path.display());
    }
    if &bytes[4..8] != b"TFL3" {
        bail!("{} missing TFL3 FlatBuffer identifier", path.display());
    }
    let root = u32_at(bytes, 0)? as usize;
    let model = Table { bytes, pos: root };
    let version = model.i32(0)?.unwrap_or_default();
    let description = model.string(3)?;
    let opcodes = model.vector_tables(1)?;
    let subgraphs = model.vector_tables(2)?;
    let buffers = model
        .vector_tables(4)?
        .iter()
        .enumerate()
        .map(|(index, b)| {
            let data = b.vector_u8_with_offset(0)?;
            Ok(BufferInfo {
                index,
                data_offset: data.as_ref().map(|(offset, _)| *offset),
                data_len: data.as_ref().map(|(_, bytes)| bytes.len()).unwrap_or(0),
                data_sha256: crate::tensor::sha256(
                    data.as_ref()
                        .map(|(_, bytes)| bytes.as_slice())
                        .unwrap_or(&[]),
                ),
                data: data.map(|(_, bytes)| bytes).unwrap_or_default(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let subgraph = subgraphs.first().context("model has no subgraphs")?;
    let tensors = subgraph
        .vector_tables(0)?
        .iter()
        .enumerate()
        .map(|(index, t)| {
            Ok(TensorInfo {
                index,
                shape: t.vector_i32(0)?.unwrap_or_default(),
                tensor_type: TensorType::from_i8(t.i8(1)?.unwrap_or_default()),
                buffer: t.u32(2)?.unwrap_or_default(),
                name: t.string(3)?.unwrap_or_else(|| format!("tensor_{index}")),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let inputs = subgraph.vector_i32(1)?.unwrap_or_default();
    let outputs = subgraph.vector_i32(2)?.unwrap_or_default();
    let operators = subgraph
        .vector_tables(3)?
        .iter()
        .enumerate()
        .map(|(index, op)| {
            let opcode_index = op.u32(0)?.unwrap_or_default();
            let opcode = opcodes
                .get(opcode_index as usize)
                .context("opcode index out of range")?;
            let new_code = opcode.i32(3)?;
            let old_code = opcode.i8(0)?.map(i32::from);
            let custom = opcode.string(1)?;
            let builtin = BuiltinOp::from_code(new_code.or(old_code).unwrap_or(0), custom);
            Ok(OperatorInfo {
                index,
                opcode_index,
                builtin,
                inputs: op.vector_i32(1)?.unwrap_or_default(),
                outputs: op.vector_i32(2)?.unwrap_or_default(),
                builtin_options_type: op.i8(3)?,
                builtin_options_table_pos: op.table_from_field(4)?.map(|t| t.pos),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut op_histogram = BTreeMap::new();
    let mut unsupported_ops = Vec::new();
    for op in &operators {
        let name = format!("{:?}", op.builtin);
        *op_histogram.entry(name.clone()).or_insert(0) += 1;
        if !op.builtin.is_supported() {
            unsupported_ops.push(format!("operator {}: {name}", op.index));
        }
    }
    let quantized_tensors = tensors
        .iter()
        .filter(|t| t.tensor_type != TensorType::Float32 && t.tensor_type != TensorType::Int32)
        .map(|t| format!("{}:{:?}", t.name, t.tensor_type))
        .collect();
    Ok(ModelInfo {
        path: path.display().to_string(),
        bytes: bytes.len(),
        raw: bytes.to_vec(),
        version,
        description,
        tensors,
        inputs,
        outputs,
        operators,
        buffers,
        op_histogram,
        unsupported_ops,
        quantized_tensors,
    })
}

#[derive(Clone, Copy)]
pub struct Table<'a> {
    pub bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> Table<'a> {
    pub fn field(&self, id: u16) -> Result<Option<usize>> {
        let vtoff = i32_at(self.bytes, self.pos)? as isize;
        let vt = self
            .pos
            .checked_add_signed(-vtoff)
            .context("invalid vtable offset")?;
        let vtsize = u16_at(self.bytes, vt)? as usize;
        let slot = 4 + id as usize * 2;
        if slot + 2 > vtsize {
            return Ok(None);
        }
        let off = u16_at(self.bytes, vt + slot)? as usize;
        Ok((off != 0).then_some(self.pos + off))
    }

    pub fn i8(&self, id: u16) -> Result<Option<i8>> {
        self.field(id)?
            .map(|p| {
                self.bytes
                    .get(p)
                    .copied()
                    .map(|v| v as i8)
                    .context("i8 field out of range")
            })
            .transpose()
    }

    pub fn i32(&self, id: u16) -> Result<Option<i32>> {
        self.field(id)?.map(|p| i32_at(self.bytes, p)).transpose()
    }

    pub fn f32(&self, id: u16) -> Result<Option<f32>> {
        self.field(id)?.map(|p| f32_at(self.bytes, p)).transpose()
    }

    pub fn bool(&self, id: u16) -> Result<Option<bool>> {
        self.field(id)?
            .map(|p| {
                self.bytes
                    .get(p)
                    .copied()
                    .map(|v| v != 0)
                    .context("bool field out of range")
            })
            .transpose()
    }

    pub fn u32(&self, id: u16) -> Result<Option<u32>> {
        self.field(id)?.map(|p| u32_at(self.bytes, p)).transpose()
    }

    pub fn string(&self, id: u16) -> Result<Option<String>> {
        let Some(p) = self.field(id)? else {
            return Ok(None);
        };
        let start = p + u32_at(self.bytes, p)? as usize;
        let len = u32_at(self.bytes, start)? as usize;
        let raw = self
            .bytes
            .get(start + 4..start + 4 + len)
            .context("string out of range")?;
        Ok(Some(String::from_utf8_lossy(raw).into_owned()))
    }

    pub fn vector_i32(&self, id: u16) -> Result<Option<Vec<i32>>> {
        let Some(p) = self.field(id)? else {
            return Ok(None);
        };
        let start = p + u32_at(self.bytes, p)? as usize;
        let len = u32_at(self.bytes, start)? as usize;
        (0..len)
            .map(|i| i32_at(self.bytes, start + 4 + i * 4))
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    pub fn vector_u8(&self, id: u16) -> Result<Option<Vec<u8>>> {
        Ok(self.vector_u8_with_offset(id)?.map(|(_, bytes)| bytes))
    }

    pub fn vector_u8_with_offset(&self, id: u16) -> Result<Option<(usize, Vec<u8>)>> {
        let Some(p) = self.field(id)? else {
            return Ok(None);
        };
        let start = p + u32_at(self.bytes, p)? as usize;
        let len = u32_at(self.bytes, start)? as usize;
        let data_start = start + 4;
        let raw = self
            .bytes
            .get(data_start..data_start + len)
            .context("u8 vector out of range")?;
        Ok(Some((data_start, raw.to_vec())))
    }

    pub fn table_from_field(&self, id: u16) -> Result<Option<Table<'a>>> {
        let Some(p) = self.field(id)? else {
            return Ok(None);
        };
        let off = u32_at(self.bytes, p)? as usize;
        Ok(Some(Table {
            bytes: self.bytes,
            pos: p + off,
        }))
    }

    pub fn vector_tables(&self, id: u16) -> Result<Vec<Table<'a>>> {
        let Some(p) = self.field(id)? else {
            return Ok(Vec::new());
        };
        let start = p + u32_at(self.bytes, p)? as usize;
        let len = u32_at(self.bytes, start)? as usize;
        (0..len)
            .map(|i| {
                let elem = start + 4 + i * 4;
                let off = u32_at(self.bytes, elem)? as usize;
                Ok(Table {
                    bytes: self.bytes,
                    pos: elem + off,
                })
            })
            .collect()
    }
}

fn u16_at(bytes: &[u8], pos: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(pos..pos + 2)
            .context("u16 out of range")?
            .try_into()?,
    ))
}

fn f32_at(bytes: &[u8], pos: usize) -> Result<f32> {
    Ok(f32::from_le_bytes(
        bytes
            .get(pos..pos + 4)
            .context("f32 out of range")?
            .try_into()?,
    ))
}

fn i32_at(bytes: &[u8], pos: usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        bytes
            .get(pos..pos + 4)
            .context("i32 out of range")?
            .try_into()?,
    ))
}

fn u32_at(bytes: &[u8], pos: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(pos..pos + 4)
            .context("u32 out of range")?
            .try_into()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_tflite() {
        let err = parse_model(Path::new("bad.tflite"), b"not-flat").unwrap_err();
        assert!(err.to_string().contains("missing TFL3") || err.to_string().contains("too small"));
    }
}
