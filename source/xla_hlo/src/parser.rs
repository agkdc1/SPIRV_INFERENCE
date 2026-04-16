use crate::ir::{HloModule, HloOp};
use anyhow::{Context, Result};
use std::collections::BTreeMap;

pub fn parse_stablehlo_text(input: &str) -> Result<HloModule> {
    let mut module = HloModule {
        name: "module".to_string(),
        ops: Vec::new(),
    };
    let is_xla_hlo = input
        .lines()
        .any(|line| line.trim().starts_with("HloModule "));
    let mut in_entry = !is_xla_hlo;

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("HloModule ") {
            module.name = trimmed
                .split_whitespace()
                .nth(1)
                .unwrap_or("module")
                .trim_end_matches(',')
                .to_string();
        }
        if trimmed.starts_with("module @") {
            module.name = trimmed
                .split_whitespace()
                .nth(1)
                .unwrap_or("@module")
                .trim_start_matches('@')
                .to_string();
        }
        if is_xla_hlo && trimmed.starts_with("ENTRY ") {
            in_entry = true;
            continue;
        }
        if is_xla_hlo && in_entry && trimmed == "}" {
            in_entry = false;
            continue;
        }
        if !in_entry {
            continue;
        }
        if trimmed.contains("stablehlo.") {
            module.ops.push(parse_stablehlo_op(trimmed)?);
            continue;
        }

        if let Some(op) = parse_xla_hlo_op(trimmed) {
            module.ops.push(op);
        }
    }

    Ok(module)
}

fn parse_stablehlo_op(trimmed: &str) -> Result<HloOp> {
    let (lhs, rhs) = trimmed.split_once('=').unwrap_or(("", trimmed));
    let result = lhs.trim().starts_with('%').then(|| lhs.trim().to_string());
    let opcode = trimmed
        .split("stablehlo.")
        .nth(1)
        .context("stablehlo opcode marker")?
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .next()
        .unwrap_or_default()
        .to_string();
    let result_type = trimmed
        .rsplit(" -> ")
        .next()
        .filter(|s| *s != trimmed)
        .map(str::to_string);
    let (element_type, shape) = parse_type(result_type.as_deref());
    Ok(HloOp {
        result,
        opcode,
        operands: percent_tokens(rhs),
        result_type,
        element_type,
        shape,
        attributes: parse_attributes(trimmed),
        raw: trimmed.to_string(),
    })
}

fn parse_xla_hlo_op(trimmed: &str) -> Option<HloOp> {
    let (lhs, rhs) = trimmed.split_once('=')?;
    let (opcode, before_paren) = xla_opcode_and_type(rhs)?;
    if opcode.is_empty() || opcode.contains('[') || opcode.contains('{') {
        return None;
    }
    let result = lhs
        .trim()
        .trim_start_matches("ROOT ")
        .split_whitespace()
        .find(|tok| tok.starts_with('%'))
        .map(str::to_string);
    let result_type = before_paren.split_whitespace().next().map(str::to_string);
    let (element_type, shape) = parse_type(result_type.as_deref());
    Some(HloOp {
        result,
        opcode,
        operands: percent_tokens(rhs),
        result_type,
        element_type,
        shape,
        attributes: parse_attributes(trimmed),
        raw: trimmed.to_string(),
    })
}

fn xla_opcode_and_type(rhs: &str) -> Option<(String, &str)> {
    const OPS: &[&str] = &[
        "get-tuple-element",
        "reduce-window",
        "dynamic-slice",
        "dynamic-update-slice",
        "broadcast_in_dim",
        "dot_general",
        "parameter",
        "constant",
        "broadcast",
        "convolution",
        "transpose",
        "subtract",
        "multiply",
        "maximum",
        "minimum",
        "exponential",
        "reshape",
        "convert",
        "divide",
        "negate",
        "rsqrt",
        "sqrt",
        "tanh",
        "logistic",
        "compare",
        "select",
        "slice",
        "reduce",
        "tuple",
        "add",
        "dot",
        "abs",
        "log",
        "pad",
        "floor",
        "ceil",
        "round-nearest-even",
        "sine",
        "cosine",
        "remainder",
        "power",
        "iota",
        "gather",
        "scatter",
    ];
    for op in OPS {
        if let Some(idx) = rhs.find(&format!("{op}(")) {
            return Some(((*op).to_string(), rhs[..idx].trim()));
        }
    }
    let open_paren = rhs.find('(')?;
    let before_paren = rhs[..open_paren].trim();
    Some((
        before_paren.split_whitespace().last()?.to_string(),
        before_paren,
    ))
}

fn percent_tokens(line: &str) -> Vec<String> {
    line.split_whitespace()
        .filter(|tok| tok.starts_with('%'))
        .map(|tok| {
            tok.trim_matches(|c: char| c == ',' || c == ')' || c == '(')
                .to_string()
        })
        .collect()
}

fn parse_type(ty: Option<&str>) -> (Option<String>, Vec<usize>) {
    let Some(ty) = ty else {
        return (None, Vec::new());
    };
    let trimmed = ty.trim();
    let element_type = trimmed
        .split('[')
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('('))
        .map(str::to_string);
    let Some(start) = trimmed.find('[') else {
        return (element_type, Vec::new());
    };
    let Some(end_rel) = trimmed[start + 1..].find(']') else {
        return (element_type, Vec::new());
    };
    let dims = &trimmed[start + 1..start + 1 + end_rel];
    let shape = dims
        .split(',')
        .filter_map(|dim| dim.trim().parse::<usize>().ok())
        .collect();
    (element_type, shape)
}

fn parse_attributes(line: &str) -> BTreeMap<String, String> {
    let mut attrs = BTreeMap::new();
    for key in [
        "window",
        "dim_labels",
        "dimensions",
        "padding",
        "lhs_batch_dims",
        "rhs_batch_dims",
        "lhs_contracting_dims",
        "rhs_contracting_dims",
        "feature_group_count",
        "batch_group_count",
        "to_apply",
        "direction",
        "index",
    ] {
        if let Some(value) = parse_attr_value(line, key) {
            attrs.insert(key.to_string(), value);
        }
    }
    if line.contains("metadata={") {
        if let Some(op_type) = parse_quoted_attr(line, "op_type") {
            attrs.insert("metadata.op_type".to_string(), op_type);
        }
        if let Some(op_name) = parse_quoted_attr(line, "op_name") {
            attrs.insert("metadata.op_name".to_string(), op_name);
        }
    }
    attrs
}

fn parse_attr_value(line: &str, key: &str) -> Option<String> {
    let marker = format!("{key}=");
    let start = line.find(&marker)? + marker.len();
    let rest = &line[start..];
    if let Some(stripped) = rest.strip_prefix('{') {
        let end = stripped.find('}')?;
        return Some(stripped[..end].to_string());
    }
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    let end = rest
        .find(',')
        .or_else(|| rest.find(' '))
        .unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn parse_quoted_attr(line: &str, key: &str) -> Option<String> {
    let marker = format!("{key}=\"");
    let start = line.find(&marker)? + marker.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}
