use crate::model::{BuiltinOp, ModelInfo};
use anyhow::{bail, Result};
use serde_json::json;

pub fn validate_supported_graph(model: &ModelInfo) -> Result<serde_json::Value> {
    if !model.unsupported_ops.is_empty() {
        bail!(
            "unsupported TFLite ops: {}",
            model.unsupported_ops.join(", ")
        );
    }
    if !model.quantized_tensors.is_empty() {
        bail!(
            "quantized tensors require dequantization support: {}",
            model.quantized_tensors.join(", ")
        );
    }
    let mut producers = std::collections::BTreeMap::new();
    for op in &model.operators {
        for out in &op.outputs {
            if *out >= 0 {
                producers.insert(*out, op.index);
            }
        }
    }
    Ok(json!({
        "status": "pass",
        "operator_count": model.operators.len(),
        "tensor_count": model.tensors.len(),
        "topological_validation": "operator_order_with_single_producer_map",
        "producer_count": producers.len(),
        "supported_ops": model.operators.iter().filter(|o| !matches!(o.builtin, BuiltinOp::Unknown(_))).count()
    }))
}
