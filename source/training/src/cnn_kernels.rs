use crate::vulkan_ctx::{Buf, Pipe, VulkanTrainer};
use anyhow::Result;

pub struct CnnPipelines {
    // Reused from MLP (11)
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
    // New CNN ops (17)
    pub conv2d_fwd: Pipe,
    pub conv_bias_add: Pipe,
    pub conv2d_bwd_data: Pipe,
    pub conv2d_bwd_weight: Pipe,
    pub bn_mean: Pipe,
    pub bn_var: Pipe,
    pub bn_pack: Pipe,
    pub bn_forward: Pipe,
    pub bn_xhat: Pipe,
    pub bn_dgamma_dbeta: Pipe,
    pub bn_dx_p1: Pipe,
    pub bn_dx_p2: Pipe,
    pub bn_dx_p3: Pipe,
    pub maxpool_fwd: Pipe,
    pub maxpool_bwd: Pipe,
    pub reduce_nhw: Pipe,
    pub zero_fill: Pipe,
}

impl CnnPipelines {
    pub fn load(gpu: &mut VulkanTrainer) -> Result<Self> {
        Ok(Self {
            // Reused MLP shaders
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
            // New CNN shaders
            conv2d_fwd:       gpu.load_pipeline(include_bytes!("../shaders/spv/conv2d_forward.spv"), 48)?,
            conv_bias_add:    gpu.load_pipeline(include_bytes!("../shaders/spv/conv_bias_add.spv"), 12)?,
            conv2d_bwd_data:  gpu.load_pipeline(include_bytes!("../shaders/spv/conv2d_backward_data.spv"), 48)?,
            conv2d_bwd_weight:gpu.load_pipeline(include_bytes!("../shaders/spv/conv2d_backward_weight.spv"), 48)?,
            bn_mean:          gpu.load_pipeline(include_bytes!("../shaders/spv/bn_mean.spv"), 16)?,
            bn_var:           gpu.load_pipeline(include_bytes!("../shaders/spv/bn_var.spv"), 16)?,
            bn_pack:          gpu.load_pipeline(include_bytes!("../shaders/spv/bn_pack_params.spv"), 4)?,
            bn_forward:       gpu.load_pipeline(include_bytes!("../shaders/spv/bn_forward.spv"), 20)?,
            bn_xhat:          gpu.load_pipeline(include_bytes!("../shaders/spv/bn_xhat.spv"), 20)?,
            bn_dgamma_dbeta:  gpu.load_pipeline(include_bytes!("../shaders/spv/bn_dgamma_dbeta.spv"), 16)?,
            bn_dx_p1:         gpu.load_pipeline(include_bytes!("../shaders/spv/bn_dx_part1.spv"), 16)?,
            bn_dx_p2:         gpu.load_pipeline(include_bytes!("../shaders/spv/bn_dx_part2.spv"), 8)?,
            bn_dx_p3:         gpu.load_pipeline(include_bytes!("../shaders/spv/bn_dx_part3.spv"), 24)?,
            maxpool_fwd:      gpu.load_pipeline(include_bytes!("../shaders/spv/maxpool2d_forward_mask.spv"), 44)?,
            maxpool_bwd:      gpu.load_pipeline(include_bytes!("../shaders/spv/maxpool2d_backward.spv"), 4)?,
            reduce_nhw:       gpu.load_pipeline(include_bytes!("../shaders/spv/reduce_sum_nhw.spv"), 16)?,
            zero_fill:        gpu.load_pipeline(include_bytes!("../shaders/spv/zero_fill.spv"), 4)?,
        })
    }
}

fn div_ceil(a: usize, b: usize) -> u32 {
    ((a + b - 1) / b) as u32
}

fn push_i(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn push_mixed(ints: &[i32], floats: &[f32]) -> Vec<u8> {
    let mut v: Vec<u8> = ints.iter().flat_map(|i| i.to_le_bytes()).collect();
    v.extend(floats.iter().flat_map(|f| f.to_le_bytes()));
    v
}

// ===== Conv2D =====

pub fn conv2d_forward(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, weight: Buf, output: Buf,
    n: i32, ic: i32, ih: i32, iw: i32,
    oc: i32, oh: i32, ow: i32,
    kh: i32, kw: i32, stride: i32, pad: i32,
) -> Result<()> {
    let total = n * oc * oh * ow;
    let push = push_i(&[n, ic, ih, iw, oc, oh, ow, kh, kw, stride, pad, total]);
    gpu.dispatch(p.conv2d_fwd, [input, weight, output], &push, div_ceil(total as usize, 64))
}

pub fn conv_bias_add(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, bias: Buf, output: Buf,
    total: i32, channels: i32, hw: i32,
) -> Result<()> {
    let push = push_i(&[total, channels, hw]);
    gpu.dispatch(p.conv_bias_add, [input, bias, output], &push, div_ceil(total as usize, 128))
}

pub fn conv2d_backward_data(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    dy: Buf, weight: Buf, dx: Buf,
    n: i32, ic: i32, ih: i32, iw: i32,
    oc: i32, oh: i32, ow: i32,
    kh: i32, kw: i32, stride: i32, pad: i32,
) -> Result<()> {
    let total = n * ic * ih * iw;
    let push = push_i(&[n, ic, ih, iw, oc, oh, ow, kh, kw, stride, pad, total]);
    gpu.dispatch(p.conv2d_bwd_data, [dy, weight, dx], &push, div_ceil(total as usize, 64))
}

pub fn conv2d_backward_weight(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, dy: Buf, dw: Buf,
    n: i32, ic: i32, ih: i32, iw: i32,
    oc: i32, oh: i32, ow: i32,
    kh: i32, kw: i32, stride: i32, pad: i32,
) -> Result<()> {
    let total = oc * ic * kh * kw;
    let push = push_i(&[n, ic, ih, iw, oc, oh, ow, kh, kw, stride, pad, total]);
    gpu.dispatch(p.conv2d_bwd_weight, [input, dy, dw], &push, div_ceil(total as usize, 64))
}

// ===== Batch Normalization =====

pub fn bn_compute_mean(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, dummy: Buf, stats: Buf,
    n: i32, c: i32, h: i32, w: i32,
) -> Result<()> {
    let push = push_i(&[n, c, h, w]);
    gpu.dispatch(p.bn_mean, [input, dummy, stats], &push, c as u32)
}

pub fn bn_compute_var(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, stats: Buf, stats_out: Buf,
    n: i32, c: i32, h: i32, w: i32,
) -> Result<()> {
    let push = push_i(&[n, c, h, w]);
    gpu.dispatch(p.bn_var, [input, stats, stats_out], &push, c as u32)
}

pub fn bn_pack_params(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    gamma_beta: Buf, stats: Buf, bnparams: Buf,
    c: i32,
) -> Result<()> {
    let push = push_i(&[c]);
    gpu.dispatch(p.bn_pack, [gamma_beta, stats, bnparams], &push, div_ceil(c as usize, 64))
}

pub fn bn_forward_apply(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, bnparams: Buf, output: Buf,
    n: i32, c: i32, h: i32, w: i32, eps: f32,
) -> Result<()> {
    let total = n * c * h * w;
    let push = push_mixed(&[n, c, h, w], &[eps]);
    gpu.dispatch(p.bn_forward, [input, bnparams, output], &push, div_ceil(total as usize, 128))
}

pub fn bn_compute_xhat(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, bnparams: Buf, xhat: Buf,
    n: i32, c: i32, h: i32, w: i32, eps: f32,
) -> Result<()> {
    let total = n * c * h * w;
    let push = push_mixed(&[n, c, h, w], &[eps]);
    gpu.dispatch(p.bn_xhat, [input, bnparams, xhat], &push, div_ceil(total as usize, 128))
}

pub fn bn_compute_dgamma_dbeta(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    dy: Buf, xhat: Buf, dparams: Buf,
    n: i32, c: i32, h: i32, w: i32,
) -> Result<()> {
    let push = push_i(&[n, c, h, w]);
    gpu.dispatch(p.bn_dgamma_dbeta, [dy, xhat, dparams], &push, c as u32)
}

pub fn bn_dx_part1(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    xhat: Buf, dparams: Buf, scratch: Buf,
    n: i32, c: i32, h: i32, w: i32,
) -> Result<()> {
    let total = n * c * h * w;
    let push = push_i(&[n, c, h, w]);
    gpu.dispatch(p.bn_dx_p1, [xhat, dparams, scratch], &push, div_ceil(total as usize, 128))
}

pub fn bn_dx_part2(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    dy: Buf, scratch: Buf, prescale: Buf,
    total: i32, m: i32,
) -> Result<()> {
    let push = push_i(&[total, m]);
    gpu.dispatch(p.bn_dx_p2, [dy, scratch, prescale], &push, div_ceil(total as usize, 128))
}

pub fn bn_dx_part3(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    prescale: Buf, bnparams: Buf, dx: Buf,
    n: i32, c: i32, h: i32, w: i32, eps: f32, m: i32,
) -> Result<()> {
    let total = n * c * h * w;
    let push = push_mixed(&[n, c, h, w], &[eps]);
    let mut push = push;
    push.extend_from_slice(&m.to_le_bytes());
    gpu.dispatch(p.bn_dx_p3, [prescale, bnparams, dx], &push, div_ceil(total as usize, 128))
}

// ===== MaxPool =====

pub fn maxpool2d_forward(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, mask: Buf, output: Buf,
    n: i32, c: i32, ih: i32, iw: i32,
    oh: i32, ow: i32, kh: i32, kw: i32,
    stride: i32, pad: i32,
) -> Result<()> {
    let total = n * c * oh * ow;
    let push = push_i(&[n, c, ih, iw, oh, ow, kh, kw, stride, pad, total]);
    gpu.dispatch(p.maxpool_fwd, [input, mask, output], &push, div_ceil(total as usize, 128))
}

pub fn maxpool2d_backward(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    dy: Buf, mask: Buf, dx: Buf,
    total: i32,
) -> Result<()> {
    let push = push_i(&[total]);
    gpu.dispatch(p.maxpool_bwd, [dy, mask, dx], &push, div_ceil(total as usize, 128))
}

// ===== Utility =====

pub fn zero_fill(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    dummy: Buf, output: Buf,
    total: i32,
) -> Result<()> {
    let push = push_i(&[total]);
    gpu.dispatch(p.zero_fill, [dummy, dummy, output], &push, div_ceil(total as usize, 128))
}

pub fn reduce_sum_nhw(
    gpu: &VulkanTrainer, p: &CnnPipelines,
    input: Buf, dummy: Buf, output: Buf,
    n: i32, c: i32, h: i32, w: i32,
) -> Result<()> {
    let push = push_i(&[n, c, h, w]);
    gpu.dispatch(p.reduce_nhw, [input, dummy, output], &push, c as u32)
}

// ===== Reused MLP kernels =====

pub fn matmul(gpu: &VulkanTrainer, p: &CnnPipelines, a: Buf, b: Buf, c: Buf, m: i32, k: i32, n: i32) -> Result<()> {
    let total = m * n;
    let push = push_i(&[m, k, n, total]);
    gpu.dispatch(p.matmul, [a, b, c], &push, div_ceil(total as usize, 64))
}

pub fn matmul_tn(gpu: &VulkanTrainer, p: &CnnPipelines, a: Buf, b: Buf, c: Buf, m: i32, k: i32, n: i32) -> Result<()> {
    let total = m * n;
    let push = push_i(&[m, k, n, total]);
    gpu.dispatch(p.matmul_tn, [a, b, c], &push, div_ceil(total as usize, 64))
}

pub fn relu_forward(gpu: &VulkanTrainer, p: &CnnPipelines, input: Buf, dummy: Buf, output: Buf, total: i32) -> Result<()> {
    let push = total.to_le_bytes().to_vec();
    gpu.dispatch(p.relu_fwd, [input, dummy, output], &push, div_ceil(total as usize, 128))
}

pub fn relu_backward(gpu: &VulkanTrainer, p: &CnnPipelines, grad_out: Buf, act_in: Buf, grad_in: Buf, total: i32) -> Result<()> {
    let push = total.to_le_bytes().to_vec();
    gpu.dispatch(p.relu_bwd, [grad_out, act_in, grad_in], &push, div_ceil(total as usize, 128))
}

pub fn bias_add(gpu: &VulkanTrainer, p: &CnnPipelines, input: Buf, bias: Buf, output: Buf, total: i32, cols: i32) -> Result<()> {
    let push = push_i(&[total, cols]);
    gpu.dispatch(p.bias_add, [input, bias, output], &push, div_ceil(total as usize, 128))
}

pub fn batched_softmax10(gpu: &VulkanTrainer, p: &CnnPipelines, input: Buf, dummy: Buf, output: Buf, batch_size: i32) -> Result<()> {
    let push = batch_size.to_le_bytes().to_vec();
    gpu.dispatch(p.softmax, [input, dummy, output], &push, batch_size as u32)
}

pub fn cross_entropy_loss(gpu: &VulkanTrainer, p: &CnnPipelines, sm: Buf, target: Buf, loss: Buf, batch_size: i32, num_classes: i32) -> Result<()> {
    let push = push_i(&[batch_size, num_classes]);
    gpu.dispatch(p.ce_loss, [sm, target, loss], &push, 1)
}

pub fn softmax_ce_backward(gpu: &VulkanTrainer, p: &CnnPipelines, sm: Buf, target: Buf, grad: Buf, total: i32, batch_size: i32) -> Result<()> {
    let push = push_i(&[total, batch_size]);
    gpu.dispatch(p.ce_bwd, [sm, target, grad], &push, div_ceil(total as usize, 128))
}

pub fn reduce_sum_rows(gpu: &VulkanTrainer, p: &CnnPipelines, input: Buf, dummy: Buf, output: Buf, rows: i32, cols: i32) -> Result<()> {
    let push = push_i(&[rows, cols]);
    gpu.dispatch(p.reduce_rows, [input, dummy, output], &push, div_ceil(cols as usize, 128))
}

pub fn adam_step(
    gpu: &VulkanTrainer, p: &CnnPipelines,
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
