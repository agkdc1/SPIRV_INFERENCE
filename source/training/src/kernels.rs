use crate::vulkan_ctx::{Buf, Pipe, VulkanTrainer};
use anyhow::Result;

pub struct Pipelines {
    pub matmul: Pipe,
    pub matmul_tn: Pipe,
    pub matmul_nt: Pipe,
    pub relu_fwd: Pipe,
    pub relu_bwd: Pipe,
    pub bias_add: Pipe,
    pub softmax: Pipe,
    pub ce_loss: Pipe,
    pub ce_bwd: Pipe,
    pub reduce_rows: Pipe,
    pub adam: Pipe,
}

impl Pipelines {
    pub fn load(gpu: &mut VulkanTrainer) -> Result<Self> {
        Ok(Self {
            matmul:      gpu.load_pipeline(include_bytes!("../shaders/spv/matmul.spv"), 16)?,
            matmul_tn:   gpu.load_pipeline(include_bytes!("../shaders/spv/matmul_tn.spv"), 16)?,
            matmul_nt:   gpu.load_pipeline(include_bytes!("../shaders/spv/matmul_nt.spv"), 16)?,
            relu_fwd:    gpu.load_pipeline(include_bytes!("../shaders/spv/relu_forward.spv"), 4)?,
            relu_bwd:    gpu.load_pipeline(include_bytes!("../shaders/spv/relu_backward.spv"), 4)?,
            bias_add:    gpu.load_pipeline(include_bytes!("../shaders/spv/bias_add.spv"), 8)?,
            softmax:     gpu.load_pipeline(include_bytes!("../shaders/spv/batched_softmax10.spv"), 4)?,
            ce_loss:     gpu.load_pipeline(include_bytes!("../shaders/spv/cross_entropy_loss.spv"), 8)?,
            ce_bwd:      gpu.load_pipeline(include_bytes!("../shaders/spv/softmax_ce_backward.spv"), 8)?,
            reduce_rows: gpu.load_pipeline(include_bytes!("../shaders/spv/reduce_sum_rows.spv"), 8)?,
            adam:        gpu.load_pipeline(include_bytes!("../shaders/spv/adam_step.spv"), 28)?,
        })
    }
}

fn div_ceil(a: usize, b: usize) -> u32 {
    ((a + b - 1) / b) as u32
}

// push bytes helper
fn push_i(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Standard matmul: C[m,n] = A[m,k] @ B[k,n]
pub fn matmul(gpu: &VulkanTrainer, p: &Pipelines, a: Buf, b: Buf, c: Buf, m: i32, k: i32, n: i32) -> Result<()> {
    let total = m * n;
    let push = push_i(&[m, k, n, total]);
    gpu.dispatch(p.matmul, [a, b, c], &push, div_ceil(total as usize, 64))
}

/// Transpose-left matmul: C[m,n] = A^T[m,k] @ B[k,n]  (A stored as [k,m])
pub fn matmul_tn(gpu: &VulkanTrainer, p: &Pipelines, a: Buf, b: Buf, c: Buf, m: i32, k: i32, n: i32) -> Result<()> {
    let total = m * n;
    let push = push_i(&[m, k, n, total]);
    gpu.dispatch(p.matmul_tn, [a, b, c], &push, div_ceil(total as usize, 64))
}

/// Transpose-right matmul: C[m,n] = A[m,k] @ B^T[n,k]  (B stored as [n,k])
pub fn matmul_nt(gpu: &VulkanTrainer, p: &Pipelines, a: Buf, b: Buf, c: Buf, m: i32, k: i32, n: i32) -> Result<()> {
    let total = m * n;
    let push = push_i(&[m, k, n, total]);
    gpu.dispatch(p.matmul_nt, [a, b, c], &push, div_ceil(total as usize, 64))
}

/// ReLU forward: out[i] = max(0, in[i])
pub fn relu_forward(gpu: &VulkanTrainer, p: &Pipelines, input: Buf, dummy: Buf, output: Buf, total: i32) -> Result<()> {
    let push = total.to_le_bytes().to_vec();
    gpu.dispatch(p.relu_fwd, [input, dummy, output], &push, div_ceil(total as usize, 128))
}

/// ReLU backward: grad_in[i] = grad_out[i] * (act_in[i] > 0)
pub fn relu_backward(gpu: &VulkanTrainer, p: &Pipelines, grad_out: Buf, act_in: Buf, grad_in: Buf, total: i32) -> Result<()> {
    let push = total.to_le_bytes().to_vec();
    gpu.dispatch(p.relu_bwd, [grad_out, act_in, grad_in], &push, div_ceil(total as usize, 128))
}

/// Bias add: out[i] = input[i] + bias[i % cols]
pub fn bias_add(gpu: &VulkanTrainer, p: &Pipelines, input: Buf, bias: Buf, output: Buf, total: i32, cols: i32) -> Result<()> {
    let push = push_i(&[total, cols]);
    gpu.dispatch(p.bias_add, [input, bias, output], &push, div_ceil(total as usize, 128))
}

/// Batched softmax for 10-class output
pub fn batched_softmax10(gpu: &VulkanTrainer, p: &Pipelines, input: Buf, dummy: Buf, output: Buf, batch_size: i32) -> Result<()> {
    let push = batch_size.to_le_bytes().to_vec();
    gpu.dispatch(p.softmax, [input, dummy, output], &push, batch_size as u32)
}

/// Cross-entropy loss (single scalar output)
pub fn cross_entropy_loss(gpu: &VulkanTrainer, p: &Pipelines, sm: Buf, target: Buf, loss: Buf, batch_size: i32, num_classes: i32) -> Result<()> {
    let push = push_i(&[batch_size, num_classes]);
    gpu.dispatch(p.ce_loss, [sm, target, loss], &push, 1)
}

/// Softmax + cross-entropy backward: grad[i] = (sm[i] - target[i]) / batch_size
pub fn softmax_ce_backward(gpu: &VulkanTrainer, p: &Pipelines, sm: Buf, target: Buf, grad: Buf, total: i32, batch_size: i32) -> Result<()> {
    let push = push_i(&[total, batch_size]);
    gpu.dispatch(p.ce_bwd, [sm, target, grad], &push, div_ceil(total as usize, 128))
}

/// Reduce sum over rows: out[col] = sum_r input[r*cols + col]
pub fn reduce_sum_rows(gpu: &VulkanTrainer, p: &Pipelines, input: Buf, dummy: Buf, output: Buf, rows: i32, cols: i32) -> Result<()> {
    let push = push_i(&[rows, cols]);
    gpu.dispatch(p.reduce_rows, [input, dummy, output], &push, div_ceil(cols as usize, 128))
}

/// Adam optimizer step
pub fn adam_step(
    gpu: &VulkanTrainer, p: &Pipelines,
    params: Buf, grads: Buf, mv: Buf,
    total: i32, lr: f32, beta1: f32, beta2: f32, eps: f32, beta1_t: f32, beta2_t: f32,
) -> Result<()> {
    let mut push = Vec::with_capacity(28);
    push.extend_from_slice(&total.to_le_bytes());
    push.extend_from_slice(&lr.to_le_bytes());
    push.extend_from_slice(&beta1.to_le_bytes());
    push.extend_from_slice(&beta2.to_le_bytes());
    push.extend_from_slice(&eps.to_le_bytes());
    push.extend_from_slice(&beta1_t.to_le_bytes());
    push.extend_from_slice(&beta2_t.to_le_bytes());
    gpu.dispatch(p.adam, [params, grads, mv], &push, div_ceil(total as usize, 128))
}
