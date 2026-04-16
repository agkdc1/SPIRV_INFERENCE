#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct HloModule {
    pub name: String,
    pub ops: Vec<HloOp>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct HloOp {
    pub result: Option<String>,
    pub opcode: String,
    pub operands: Vec<String>,
    pub result_type: Option<String>,
    pub element_type: Option<String>,
    pub shape: Vec<usize>,
    pub attributes: std::collections::BTreeMap<String, String>,
    pub raw: String,
}
