mod vulkan_ctx;
mod cnn_kernels;
mod cifar10;
mod report;

use anyhow::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::PathBuf;
use std::time::Instant;

// ResNet-18 for CIFAR-10 (32x32 input, no initial 7x7 conv/maxpool)
// conv1(3->64,3x3,s=1,p=1)->BN->ReLU
// layer1: 2 BasicBlocks (64->64)
// layer2: 2 BasicBlocks (64->128, first block stride=2, with 1x1 downsample)
// layer3: 2 BasicBlocks (128->256, first block stride=2, with 1x1 downsample)
// layer4: 2 BasicBlocks (256->512, first block stride=2, with 1x1 downsample)
// GlobalAvgPool -> FC(512,10) -> Softmax -> CE

const BATCH: usize = 64;
const EPOCHS: usize = 20;
const LR: f32 = 0.001;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const BN_EPS: f32 = 1e-5;
const CLASSES: usize = 10;

const IN_C: usize = 3;
const IN_H: usize = 32;
const IN_W: usize = 32;

use vulkan_ctx::Buf;

/// Extended pipelines with 3 new shaders for ResNet
struct ResNetPipelines {
    cnn: cnn_kernels::CnnPipelines,
    element_add: vulkan_ctx::Pipe,
    global_avg_pool: vulkan_ctx::Pipe,
    global_avg_pool_bwd: vulkan_ctx::Pipe,
}

impl ResNetPipelines {
    fn load(gpu: &mut vulkan_ctx::VulkanTrainer) -> Result<Self> {
        let cnn = cnn_kernels::CnnPipelines::load(gpu)?;
        let element_add = gpu.load_pipeline(
            include_bytes!("../shaders/spv/element_add.spv"), 4)?;
        let global_avg_pool = gpu.load_pipeline(
            include_bytes!("../shaders/spv/global_avg_pool.spv"), 16)?;
        let global_avg_pool_bwd = gpu.load_pipeline(
            include_bytes!("../shaders/spv/global_avg_pool_backward.spv"), 20)?;
        Ok(Self { cnn, element_add, global_avg_pool, global_avg_pool_bwd })
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

// New dispatch wrappers for the 3 new shaders

fn element_add(gpu: &vulkan_ctx::VulkanTrainer, p: &ResNetPipelines,
    a: Buf, b: Buf, out: Buf, total: i32) -> Result<()> {
    let push = total.to_le_bytes().to_vec();
    gpu.dispatch(p.element_add, [a, b, out], &push, div_ceil(total as usize, 128))
}

fn global_avg_pool(gpu: &vulkan_ctx::VulkanTrainer, p: &ResNetPipelines,
    input: Buf, dummy: Buf, output: Buf, n: i32, c: i32, h: i32, w: i32) -> Result<()> {
    let push = push_i(&[n, c, h, w]);
    let total_nc = (n * c) as u32;
    gpu.dispatch(p.global_avg_pool, [input, dummy, output], &push, div_ceil(total_nc as usize, 64))
}

fn global_avg_pool_backward(gpu: &vulkan_ctx::VulkanTrainer, p: &ResNetPipelines,
    dy: Buf, dummy: Buf, dx: Buf, n: i32, c: i32, h: i32, w: i32) -> Result<()> {
    let total = n * c * h * w;
    let push = push_i(&[n, c, h, w, total]);
    gpu.dispatch(p.global_avg_pool_bwd, [dy, dummy, dx], &push, div_ceil(total as usize, 128))
}

/// Buffers for one BN layer
struct BnBufs {
    gb: Buf,        // gamma+beta [2*C]
    stats: Buf,     // mean+var [2*C]
    params: Buf,    // packed [4*C]
    out: Buf,       // [N,C,H,W]
    xhat: Buf,      // [N,C,H,W]
    // backward
    d_out: Buf,     // [N,C,H,W]
    d_params: Buf,  // dgamma+dbeta [2*C]
    d_scratch: Buf, // [N,C,H,W]
    d_prescale: Buf,// [N,C,H,W]
    d_x: Buf,       // [N,C,H,W] - dx output
    // adam
    mv_gb: Buf,     // [4*C]
}

/// Buffers for one conv layer
struct ConvBufs {
    w: Buf,         // [OC, IC, KH, KW]
    b: Buf,         // [OC]
    out: Buf,       // [N, OC, OH, OW] pre-bias
    bias_out: Buf,  // [N, OC, OH, OW] post-bias (BN input)
    // backward
    d_w: Buf,       // weight grad
    d_b: Buf,       // bias grad
    // adam
    mv_w: Buf,      // [2 * OC*IC*KH*KW]
    mv_b: Buf,      // [2 * OC]
}

/// Buffers for a BasicBlock
struct BasicBlockBufs {
    conv1: ConvBufs,
    bn1: BnBufs,
    relu1_out: Buf,
    conv2: ConvBufs,
    bn2: BnBufs,
    // relu2 output (after residual add)
    relu2_out: Buf,
    // residual add output (before relu2)
    add_out: Buf,
    // downsample (optional: only if stride != 1 or in_c != out_c)
    has_downsample: bool,
    ds_conv: Option<ConvBufs>,
    ds_bn: Option<BnBufs>,
    // backward scratch
    d_relu2: Buf,       // grad into relu2
    d_add_a: Buf,       // grad branch A (main path before add)
    d_relu1: Buf,       // grad into relu1
    d_ds_input: Buf,    // grad through downsample path (or identity grad)
}

fn alloc_conv(gpu: &mut vulkan_ctx::VulkanTrainer, oc: usize, ic: usize, k: usize, batch: usize, oh: usize, ow: usize) -> Result<ConvBufs> {
    let wt_size = oc * ic * k * k;
    Ok(ConvBufs {
        w: gpu.alloc(wt_size)?,
        b: gpu.alloc(oc)?,
        out: gpu.alloc(batch * oc * oh * ow)?,
        bias_out: gpu.alloc(batch * oc * oh * ow)?,
        d_w: gpu.alloc(wt_size)?,
        d_b: gpu.alloc(oc)?,
        mv_w: gpu.alloc(2 * wt_size)?,
        mv_b: gpu.alloc(2 * oc)?,
    })
}

fn alloc_bn(gpu: &mut vulkan_ctx::VulkanTrainer, c: usize, batch: usize, h: usize, w: usize) -> Result<BnBufs> {
    let spatial = batch * c * h * w;
    Ok(BnBufs {
        gb: gpu.alloc(2 * c)?,
        stats: gpu.alloc(2 * c)?,
        params: gpu.alloc(4 * c)?,
        out: gpu.alloc(spatial)?,
        xhat: gpu.alloc(spatial)?,
        d_out: gpu.alloc(spatial)?,
        d_params: gpu.alloc(2 * c)?,
        d_scratch: gpu.alloc(spatial)?,
        d_prescale: gpu.alloc(spatial)?,
        d_x: gpu.alloc(spatial)?,
        mv_gb: gpu.alloc(4 * c)?,
    })
}

fn alloc_block(gpu: &mut vulkan_ctx::VulkanTrainer, ic: usize, oc: usize, stride: usize,
    batch: usize, ih: usize, iw: usize) -> Result<BasicBlockBufs> {
    let oh = ih / stride;
    let ow = iw / stride;
    // conv1: ic->oc, 3x3, stride, pad=1
    let conv1 = alloc_conv(gpu, oc, ic, 3, batch, oh, ow)?;
    let bn1 = alloc_bn(gpu, oc, batch, oh, ow)?;
    let relu1_out = gpu.alloc(batch * oc * oh * ow)?;
    // conv2: oc->oc, 3x3, stride=1, pad=1
    let conv2 = alloc_conv(gpu, oc, oc, 3, batch, oh, ow)?;
    let bn2 = alloc_bn(gpu, oc, batch, oh, ow)?;
    let relu2_out = gpu.alloc(batch * oc * oh * ow)?;
    let add_out = gpu.alloc(batch * oc * oh * ow)?;

    let has_downsample = stride != 1 || ic != oc;
    let (ds_conv, ds_bn) = if has_downsample {
        let dc = alloc_conv(gpu, oc, ic, 1, batch, oh, ow)?;
        let db = alloc_bn(gpu, oc, batch, oh, ow)?;
        (Some(dc), Some(db))
    } else {
        (None, None)
    };

    let spatial = batch * oc * oh * ow;
    let d_relu2 = gpu.alloc(spatial)?;
    let d_add_a = gpu.alloc(spatial)?;
    let d_relu1 = gpu.alloc(spatial)?;
    let d_ds_input = gpu.alloc(if has_downsample { batch * ic * ih * iw } else { spatial })?;

    Ok(BasicBlockBufs {
        conv1, bn1, relu1_out, conv2, bn2, relu2_out, add_out,
        has_downsample, ds_conv, ds_bn,
        d_relu2, d_add_a, d_relu1, d_ds_input,
    })
}

fn init_conv_weights(conv: &ConvBufs, gpu: &vulkan_ctx::VulkanTrainer,
    ic: usize, oc: usize, k: usize, rng: &mut impl rand::Rng) -> Result<()> {
    let fan_in = ic * k * k;
    let w = kaiming_init(fan_in, oc, rng);
    let b = vec![0.0_f32; oc];
    gpu.upload(conv.w, &w)?;
    gpu.upload(conv.b, &b)?;
    Ok(())
}

fn init_bn_weights(bn: &BnBufs, gpu: &vulkan_ctx::VulkanTrainer, c: usize) -> Result<()> {
    let mut gb = vec![1.0_f32; c]; // gamma=1
    gb.extend(vec![0.0_f32; c]);   // beta=0
    gpu.upload(bn.gb, &gb)?;
    Ok(())
}

fn init_block_weights(block: &BasicBlockBufs, gpu: &vulkan_ctx::VulkanTrainer,
    ic: usize, oc: usize, rng: &mut impl rand::Rng) -> Result<()> {
    init_conv_weights(&block.conv1, gpu, ic, oc, 3, rng)?;
    init_bn_weights(&block.bn1, gpu, oc)?;
    init_conv_weights(&block.conv2, gpu, oc, oc, 3, rng)?;
    init_bn_weights(&block.bn2, gpu, oc)?;
    if let Some(ref dc) = block.ds_conv {
        init_conv_weights(dc, gpu, ic, oc, 1, rng)?;
    }
    if let Some(ref db) = block.ds_bn {
        init_bn_weights(db, gpu, oc)?;
    }
    Ok(())
}

// ======= Forward through one conv->bias->BN sequence =======
fn fwd_conv_bn(
    gpu: &vulkan_ctx::VulkanTrainer, p: &cnn_kernels::CnnPipelines,
    input: Buf, conv: &ConvBufs, bn: &BnBufs,
    n: i32, ic: i32, ih: i32, iw: i32,
    oc: i32, oh: i32, ow: i32,
    kh: i32, kw: i32, stride: i32, pad: i32,
) -> Result<()> {
    cnn_kernels::conv2d_forward(gpu, p, input, conv.w, conv.out,
        n, ic, ih, iw, oc, oh, ow, kh, kw, stride, pad)?;
    let total = n * oc * oh * ow;
    let hw = oh * ow;
    cnn_kernels::conv_bias_add(gpu, p, conv.out, conv.b, conv.bias_out, total, oc, hw)?;
    // BN
    let dummy = Buf(0); // will be replaced
    cnn_kernels::bn_compute_mean(gpu, p, conv.bias_out, dummy, bn.stats, n, oc, oh, ow)?;
    cnn_kernels::bn_compute_var(gpu, p, conv.bias_out, bn.stats, bn.stats, n, oc, oh, ow)?;
    cnn_kernels::bn_pack_params(gpu, p, bn.gb, bn.stats, bn.params, oc)?;
    cnn_kernels::bn_forward_apply(gpu, p, conv.bias_out, bn.params, bn.out, n, oc, oh, ow, BN_EPS)?;
    cnn_kernels::bn_compute_xhat(gpu, p, conv.bias_out, bn.params, bn.xhat, n, oc, oh, ow, BN_EPS)?;
    Ok(())
}

// ======= Backward through BN->conv =======
fn bwd_bn_conv(
    gpu: &vulkan_ctx::VulkanTrainer, p: &cnn_kernels::CnnPipelines,
    grad_in: Buf, // gradient coming in (dL/d(bn.out))
    input_to_conv: Buf, // what was fed into this conv (for weight grad)
    conv: &ConvBufs, bn: &BnBufs,
    n: i32, ic: i32, ih: i32, iw: i32,
    oc: i32, oh: i32, ow: i32,
    kh: i32, kw: i32, stride: i32, pad: i32,
    compute_dx: bool, // whether to compute dX (input grad)
    dx_out: Buf, // where to put dX
) -> Result<()> {
    let total_out = n * oc * oh * ow;
    let m = n * oh * ow;

    // BN backward: grad_in -> bn.d_x (which is dL/d(conv.bias_out))
    cnn_kernels::bn_compute_dgamma_dbeta(gpu, p, grad_in, bn.xhat, bn.d_params,
        n, oc, oh, ow)?;
    cnn_kernels::bn_dx_part1(gpu, p, bn.xhat, bn.d_params, bn.d_scratch,
        n, oc, oh, ow)?;
    cnn_kernels::bn_dx_part2(gpu, p, grad_in, bn.d_scratch, bn.d_prescale,
        total_out, m)?;
    cnn_kernels::bn_dx_part3(gpu, p, bn.d_prescale, bn.params, bn.d_x,
        n, oc, oh, ow, BN_EPS, m)?;

    // Conv bias backward
    cnn_kernels::reduce_sum_nhw(gpu, p, bn.d_x, Buf(0), conv.d_b,
        n, oc, oh, ow)?;

    // Conv weight backward
    cnn_kernels::conv2d_backward_weight(gpu, p, input_to_conv, bn.d_x, conv.d_w,
        n, ic, ih, iw, oc, oh, ow, kh, kw, stride, pad)?;

    // Conv input backward (optional)
    if compute_dx {
        cnn_kernels::conv2d_backward_data(gpu, p, bn.d_x, conv.w, dx_out,
            n, ic, ih, iw, oc, oh, ow, kh, kw, stride, pad)?;
    }

    Ok(())
}

// ======= Forward through a BasicBlock =======
fn fwd_block(
    gpu: &vulkan_ctx::VulkanTrainer, p: &ResNetPipelines,
    input: Buf, block: &BasicBlockBufs,
    n: i32, ic: i32, ih: i32, iw: i32,
    oc: i32, stride: i32,
) -> Result<()> {
    let oh = ih / stride;
    let ow = iw / stride;

    // Main path: conv1(stride)->BN1->ReLU->conv2(s=1)->BN2
    fwd_conv_bn(gpu, &p.cnn, input, &block.conv1, &block.bn1,
        n, ic, ih, iw, oc, oh, ow, 3, 3, stride, 1)?;
    let spatial = (n * oc * oh * ow) as i32;
    cnn_kernels::relu_forward(gpu, &p.cnn, block.bn1.out, Buf(0), block.relu1_out, spatial)?;
    fwd_conv_bn(gpu, &p.cnn, block.relu1_out, &block.conv2, &block.bn2,
        n, oc, oh, ow, oc, oh, ow, 3, 3, 1, 1)?;

    // Residual path
    if block.has_downsample {
        let dc = block.ds_conv.as_ref().unwrap();
        let db = block.ds_bn.as_ref().unwrap();
        fwd_conv_bn(gpu, &p.cnn, input, dc, db,
            n, ic, ih, iw, oc, oh, ow, 1, 1, stride, 0)?;
        // add_out = bn2.out + ds_bn.out
        element_add(gpu, p, block.bn2.out, db.out, block.add_out, spatial)?;
    } else {
        // add_out = bn2.out + input (identity shortcut)
        element_add(gpu, p, block.bn2.out, input, block.add_out, spatial)?;
    }

    // ReLU after add
    cnn_kernels::relu_forward(gpu, &p.cnn, block.add_out, Buf(0), block.relu2_out, spatial)?;

    Ok(())
}

// ======= Backward through a BasicBlock =======
fn bwd_block(
    gpu: &vulkan_ctx::VulkanTrainer, p: &ResNetPipelines,
    grad_out: Buf, // dL/d(relu2_out)
    input: Buf,    // what was fed into this block
    block: &BasicBlockBufs,
    n: i32, ic: i32, ih: i32, iw: i32,
    oc: i32, stride: i32,
    dx_out: Buf,   // where to put dL/d(input)
) -> Result<()> {
    let oh = ih / stride;
    let ow = iw / stride;
    let spatial = (n * oc * oh * ow) as i32;

    // ReLU2 backward: grad through relu2
    cnn_kernels::relu_backward(gpu, &p.cnn, grad_out, block.add_out, block.d_relu2, spatial)?;

    // d_relu2 goes to BOTH branches of the residual add
    // Branch A (main path): d_relu2 -> BN2 -> conv2 -> ReLU1 -> BN1 -> conv1
    // Branch B (shortcut): d_relu2 -> downsample backward OR identity

    // Branch A: BN2 backward + conv2 backward
    bwd_bn_conv(gpu, &p.cnn, block.d_relu2, block.relu1_out,
        &block.conv2, &block.bn2,
        n, oc, oh, ow, oc, oh, ow, 3, 3, 1, 1,
        true, block.d_relu1)?;

    // ReLU1 backward
    cnn_kernels::relu_backward(gpu, &p.cnn, block.d_relu1, block.bn1.out, block.d_add_a,
        spatial)?;

    // BN1 backward + conv1 backward
    bwd_bn_conv(gpu, &p.cnn, block.d_add_a, input,
        &block.conv1, &block.bn1,
        n, ic, ih, iw, oc, oh, ow, 3, 3, stride, 1,
        true, dx_out)?;

    // Branch B: shortcut gradient
    if block.has_downsample {
        let dc = block.ds_conv.as_ref().unwrap();
        let db = block.ds_bn.as_ref().unwrap();
        bwd_bn_conv(gpu, &p.cnn, block.d_relu2, input,
            dc, db,
            n, ic, ih, iw, oc, oh, ow, 1, 1, stride, 0,
            true, block.d_ds_input)?;
        // dx_out += d_ds_input (accumulate both branches)
        let in_total = (n * ic * ih * iw) as i32;
        element_add(gpu, p, dx_out, block.d_ds_input, dx_out, in_total)?;
    } else {
        // Identity shortcut: dx_out += d_relu2
        // dx_out already has conv1 backward grad, add the shortcut grad
        element_add(gpu, p, dx_out, block.d_relu2, dx_out, spatial)?;
    }

    Ok(())
}

fn adam_update_conv(gpu: &vulkan_ctx::VulkanTrainer, p: &cnn_kernels::CnnPipelines,
    conv: &ConvBufs, wt_size: usize, oc: usize,
    lr: f32, beta1: f32, beta2: f32, eps: f32, b1t: f32, b2t: f32) -> Result<()> {
    cnn_kernels::adam_step(gpu, p, conv.w, conv.d_w, conv.mv_w,
        wt_size as i32, lr, beta1, beta2, eps, b1t, b2t)?;
    cnn_kernels::adam_step(gpu, p, conv.b, conv.d_b, conv.mv_b,
        oc as i32, lr, beta1, beta2, eps, b1t, b2t)?;
    Ok(())
}

fn adam_update_bn(gpu: &vulkan_ctx::VulkanTrainer, p: &cnn_kernels::CnnPipelines,
    bn: &BnBufs, c: usize,
    lr: f32, beta1: f32, beta2: f32, eps: f32, b1t: f32, b2t: f32) -> Result<()> {
    cnn_kernels::adam_step(gpu, p, bn.gb, bn.d_params, bn.mv_gb,
        (2 * c) as i32, lr, beta1, beta2, eps, b1t, b2t)?;
    Ok(())
}

fn adam_update_block(gpu: &vulkan_ctx::VulkanTrainer, p: &cnn_kernels::CnnPipelines,
    block: &BasicBlockBufs, ic: usize, oc: usize, k: usize,
    lr: f32, beta1: f32, beta2: f32, eps: f32, b1t: f32, b2t: f32) -> Result<()> {
    adam_update_conv(gpu, p, &block.conv1, oc * ic * k * k, oc, lr, beta1, beta2, eps, b1t, b2t)?;
    adam_update_bn(gpu, p, &block.bn1, oc, lr, beta1, beta2, eps, b1t, b2t)?;
    adam_update_conv(gpu, p, &block.conv2, oc * oc * k * k, oc, lr, beta1, beta2, eps, b1t, b2t)?;
    adam_update_bn(gpu, p, &block.bn2, oc, lr, beta1, beta2, eps, b1t, b2t)?;
    if block.has_downsample {
        let dc = block.ds_conv.as_ref().unwrap();
        let db = block.ds_bn.as_ref().unwrap();
        adam_update_conv(gpu, p, dc, oc * ic * 1 * 1, oc, lr, beta1, beta2, eps, b1t, b2t)?;
        adam_update_bn(gpu, p, db, oc, lr, beta1, beta2, eps, b1t, b2t)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let start = Instant::now();
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");

    // 1. Load CIFAR-10
    let cifar = cifar10::load(&data_dir)?;

    // 2. Init Vulkan + pipelines
    let mut gpu = vulkan_ctx::VulkanTrainer::new("nvidia")?;
    let pipes = ResNetPipelines::load(&mut gpu)?;
    println!("[init] 31 pipelines loaded (28 CNN + 3 ResNet)");

    // 3. Allocate buffers
    let buf_input = gpu.alloc(BATCH * IN_C * IN_H * IN_W)?;
    let buf_labels = gpu.alloc(BATCH * CLASSES)?;
    let buf_dummy = gpu.alloc(1)?;

    // Conv1: 3->64, 3x3, s=1, p=1, out=[N,64,32,32]
    let conv1 = alloc_conv(&mut gpu, 64, IN_C, 3, BATCH, 32, 32)?;
    let bn1 = alloc_bn(&mut gpu, 64, BATCH, 32, 32)?;
    let buf_relu1 = gpu.alloc(BATCH * 64 * 32 * 32)?;

    // Layer1: 2 blocks, 64->64, stride=1, spatial=32x32
    let l1b0 = alloc_block(&mut gpu, 64, 64, 1, BATCH, 32, 32)?;
    let l1b1 = alloc_block(&mut gpu, 64, 64, 1, BATCH, 32, 32)?;

    // Layer2: 2 blocks, 64->128, first stride=2, spatial 32->16
    let l2b0 = alloc_block(&mut gpu, 64, 128, 2, BATCH, 32, 32)?;
    let l2b1 = alloc_block(&mut gpu, 128, 128, 1, BATCH, 16, 16)?;

    // Layer3: 2 blocks, 128->256, first stride=2, spatial 16->8
    let l3b0 = alloc_block(&mut gpu, 128, 256, 2, BATCH, 16, 16)?;
    let l3b1 = alloc_block(&mut gpu, 256, 256, 1, BATCH, 8, 8)?;

    // Layer4: 2 blocks, 256->512, first stride=2, spatial 8->4
    let l4b0 = alloc_block(&mut gpu, 256, 512, 2, BATCH, 8, 8)?;
    let l4b1 = alloc_block(&mut gpu, 512, 512, 1, BATCH, 4, 4)?;

    // Global avg pool: [N,512,4,4] -> [N,512]
    let buf_pool_out = gpu.alloc(BATCH * 512)?;

    // FC: 512->10
    let buf_fc_w = gpu.alloc(512 * CLASSES)?;
    let buf_fc_b = gpu.alloc(CLASSES)?;
    let buf_fc_out = gpu.alloc(BATCH * CLASSES)?;
    let buf_fc_z = gpu.alloc(BATCH * CLASSES)?;
    let buf_sm = gpu.alloc(BATCH * CLASSES)?;
    let buf_loss = gpu.alloc(1)?;

    // FC backward
    let buf_dfc_z = gpu.alloc(BATCH * CLASSES)?;
    let buf_dfc_w = gpu.alloc(512 * CLASSES)?;
    let buf_dfc_b = gpu.alloc(CLASSES)?;
    let buf_dpool = gpu.alloc(BATCH * 512)?; // grad into pool_out

    // Global avg pool backward: [N,512] -> [N,512,4,4]
    let buf_dl4 = gpu.alloc(BATCH * 512 * 4 * 4)?; // grad into layer4 output

    // Inter-layer backward scratch buffers
    let buf_dl3 = gpu.alloc(BATCH * 256 * 8 * 8)?;
    let buf_dl2 = gpu.alloc(BATCH * 128 * 16 * 16)?;
    let buf_dl1 = gpu.alloc(BATCH * 64 * 32 * 32)?;
    let buf_drelu1 = gpu.alloc(BATCH * 64 * 32 * 32)?;
    // We need scratch for inter-block gradients too
    let buf_dl4_mid = gpu.alloc(BATCH * 512 * 4 * 4)?; // between l4b1 and l4b0
    let buf_dl3_mid = gpu.alloc(BATCH * 256 * 8 * 8)?;
    let buf_dl2_mid = gpu.alloc(BATCH * 128 * 16 * 16)?;
    let buf_dl1_mid = gpu.alloc(BATCH * 64 * 32 * 32)?;

    // FC adam state
    let buf_mv_fc_w = gpu.alloc(2 * 512 * CLASSES)?;
    let buf_mv_fc_b = gpu.alloc(2 * CLASSES)?;

    println!("[init] GPU buffers allocated");

    // 4. Initialize weights
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    init_conv_weights(&conv1, &gpu, IN_C, 64, 3, &mut rng)?;
    init_bn_weights(&bn1, &gpu, 64)?;

    init_block_weights(&l1b0, &gpu, 64, 64, &mut rng)?;
    init_block_weights(&l1b1, &gpu, 64, 64, &mut rng)?;
    init_block_weights(&l2b0, &gpu, 64, 128, &mut rng)?;
    init_block_weights(&l2b1, &gpu, 128, 128, &mut rng)?;
    init_block_weights(&l3b0, &gpu, 128, 256, &mut rng)?;
    init_block_weights(&l3b1, &gpu, 256, 256, &mut rng)?;
    init_block_weights(&l4b0, &gpu, 256, 512, &mut rng)?;
    init_block_weights(&l4b1, &gpu, 512, 512, &mut rng)?;

    // FC weights
    let fc_w = kaiming_init(512, CLASSES, &mut rng);
    let fc_b = vec![0.0_f32; CLASSES];
    gpu.upload(buf_fc_w, &fc_w)?;
    gpu.upload(buf_fc_b, &fc_b)?;

    println!("[init] weights initialized (Kaiming He, seed=42)");
    println!("[arch] ResNet-18 for CIFAR-10 (32x32, no initial 7x7/maxpool)");
    println!("[arch] conv1(3->64)->BN->ReLU -> 4 layers x 2 BasicBlocks -> GAP -> FC(512,10)");

    // Count params
    let total_params = count_params(IN_C, 64, 3)  // conv1
        + 2 * 64  // bn1
        + count_block_params(64, 64)   // l1b0
        + count_block_params(64, 64)   // l1b1
        + count_block_params_ds(64, 128)  // l2b0 (with downsample)
        + count_block_params(128, 128) // l2b1
        + count_block_params_ds(128, 256) // l3b0
        + count_block_params(256, 256) // l3b1
        + count_block_params_ds(256, 512) // l4b0
        + count_block_params(512, 512) // l4b1
        + 512 * CLASSES + CLASSES;     // FC
    println!("[params] total trainable: {}", total_params);

    // 5. Training loop
    let num_train = cifar.train_labels.len();
    let batches_per_epoch = num_train / BATCH;
    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut epoch_results = Vec::new();
    let mut adam_t: u32 = 0;
    let img_size = IN_C * IN_H * IN_W;

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        indices.shuffle(&mut rng);
        let mut total_loss = 0.0_f64;
        let mut loss_count = 0;

        for batch_idx in 0..batches_per_epoch {
            adam_t += 1;
            let beta1_t = BETA1.powi(adam_t as i32);
            let beta2_t = BETA2.powi(adam_t as i32);

            // Prepare batch
            let mut batch_img = vec![0.0_f32; BATCH * img_size];
            let mut batch_lbl = vec![0_u8; BATCH];
            for i in 0..BATCH {
                let idx = indices[batch_idx * BATCH + i];
                let src = &cifar.train_images[idx * img_size..(idx + 1) * img_size];
                batch_img[i * img_size..(i + 1) * img_size].copy_from_slice(src);
                cifar10::random_hflip(&mut batch_img[i * img_size..(i + 1) * img_size], &mut rng);
                batch_lbl[i] = cifar.train_labels[idx];
            }
            let batch_onehot = cifar10::one_hot(&batch_lbl, CLASSES);
            gpu.upload(buf_input, &batch_img)?;
            gpu.upload(buf_labels, &batch_onehot)?;

            let b = BATCH as i32;

            // ===== FORWARD =====
            // Conv1: [N,3,32,32] -> [N,64,32,32]
            fwd_conv_bn(&gpu, &pipes.cnn, buf_input, &conv1, &bn1,
                b, IN_C as i32, 32, 32, 64, 32, 32, 3, 3, 1, 1)?;
            cnn_kernels::relu_forward(&gpu, &pipes.cnn, bn1.out, buf_dummy, buf_relu1,
                (BATCH * 64 * 32 * 32) as i32)?;

            // Layer1
            fwd_block(&gpu, &pipes, buf_relu1, &l1b0, b, 64, 32, 32, 64, 1)?;
            fwd_block(&gpu, &pipes, l1b0.relu2_out, &l1b1, b, 64, 32, 32, 64, 1)?;

            // Layer2
            fwd_block(&gpu, &pipes, l1b1.relu2_out, &l2b0, b, 64, 32, 32, 128, 2)?;
            fwd_block(&gpu, &pipes, l2b0.relu2_out, &l2b1, b, 128, 16, 16, 128, 1)?;

            // Layer3
            fwd_block(&gpu, &pipes, l2b1.relu2_out, &l3b0, b, 128, 16, 16, 256, 2)?;
            fwd_block(&gpu, &pipes, l3b0.relu2_out, &l3b1, b, 256, 8, 8, 256, 1)?;

            // Layer4
            fwd_block(&gpu, &pipes, l3b1.relu2_out, &l4b0, b, 256, 8, 8, 512, 2)?;
            fwd_block(&gpu, &pipes, l4b0.relu2_out, &l4b1, b, 512, 4, 4, 512, 1)?;

            // Global avg pool: [N,512,4,4] -> [N,512]
            global_avg_pool(&gpu, &pipes, l4b1.relu2_out, buf_dummy, buf_pool_out,
                b, 512, 4, 4)?;

            // FC: [N,512] @ [512,10] + bias
            cnn_kernels::matmul(&gpu, &pipes.cnn, buf_pool_out, buf_fc_w, buf_fc_out,
                b, 512, CLASSES as i32)?;
            cnn_kernels::bias_add(&gpu, &pipes.cnn, buf_fc_out, buf_fc_b, buf_fc_z,
                (BATCH * CLASSES) as i32, CLASSES as i32)?;

            // Softmax + CE loss
            cnn_kernels::batched_softmax10(&gpu, &pipes.cnn, buf_fc_z, buf_dummy, buf_sm, b)?;
            cnn_kernels::cross_entropy_loss(&gpu, &pipes.cnn, buf_sm, buf_labels, buf_loss,
                b, CLASSES as i32)?;

            // Read loss
            let loss_val = gpu.download(buf_loss, 1)?[0];
            total_loss += loss_val as f64;
            loss_count += 1;

            if batch_idx % 100 == 0 {
                println!("  epoch {}/{} batch {}/{} loss={:.4}",
                    epoch + 1, EPOCHS, batch_idx, batches_per_epoch, loss_val);
            }

            // ===== BACKWARD =====
            // FC backward
            cnn_kernels::softmax_ce_backward(&gpu, &pipes.cnn, buf_sm, buf_labels, buf_dfc_z,
                (BATCH * CLASSES) as i32, b)?;
            cnn_kernels::matmul_tn(&gpu, &pipes.cnn, buf_pool_out, buf_dfc_z, buf_dfc_w,
                512, b, CLASSES as i32)?;
            cnn_kernels::reduce_sum_rows(&gpu, &pipes.cnn, buf_dfc_z, buf_dummy, buf_dfc_b,
                b, CLASSES as i32)?;
            // dpool = dfc_z @ fc_w^T ... wait, matmul_nt not loaded in CnnPipelines
            // Actually we need: dpool = dfc_z[N,10] @ fc_w[512,10]^T = dfc_z @ fc_w^T
            // This is matmul_nt: A[M,K] @ B[N,K]^T = C[M,N] where M=batch, K=10, N=512
            // But CnnPipelines doesn't have matmul_nt exposed... let me check
            // Actually cnn_kernels::matmul does A[M,K] @ B[K,N] = C[M,N]
            // We need [N_batch, 10] @ [10, 512] = [N_batch, 512]
            // But fc_w is [512, 10], so fc_w^T is [10, 512]
            // Use matmul: dfc_z[batch, 10] @ fc_w_T[10, 512]
            // Hmm, we don't have fc_w transposed. Let's use matmul_nt which does A @ B^T
            // matmul in cnn_kernels does A[M,K] @ B[K,N] = C[M,N]
            // We want: dpool[batch, 512] = dfc_z[batch, 10] @ fc_w[512, 10]^T
            // This requires matmul_nt: A[M,K] @ B[N,K]^T => C[M,N]
            // where M=batch, K=10, N=512
            // matmul_nt is available but not wrapped in cnn_kernels... it IS in CnnPipelines
            // Let me use a direct dispatch
            {
                let m = b;
                let k = CLASSES as i32;
                let nn = 512_i32;
                let total = m * nn;
                let push = push_i(&[m, k, nn, total]);
                gpu.dispatch(pipes.cnn.matmul_nt, [buf_dfc_z, buf_fc_w, buf_dpool], &push,
                    div_ceil(total as usize, 64))?;
            }

            // Global avg pool backward: [N,512] -> [N,512,4,4]
            global_avg_pool_backward(&gpu, &pipes, buf_dpool, buf_dummy, buf_dl4,
                b, 512, 4, 4)?;

            // Layer4 backward
            bwd_block(&gpu, &pipes, buf_dl4, l4b0.relu2_out, &l4b1,
                b, 512, 4, 4, 512, 1, buf_dl4_mid)?;
            bwd_block(&gpu, &pipes, buf_dl4_mid, l3b1.relu2_out, &l4b0,
                b, 256, 8, 8, 512, 2, buf_dl3)?;

            // Layer3 backward
            bwd_block(&gpu, &pipes, buf_dl3, l3b0.relu2_out, &l3b1,
                b, 256, 8, 8, 256, 1, buf_dl3_mid)?;
            bwd_block(&gpu, &pipes, buf_dl3_mid, l2b1.relu2_out, &l3b0,
                b, 128, 16, 16, 256, 2, buf_dl2)?;

            // Layer2 backward
            bwd_block(&gpu, &pipes, buf_dl2, l2b0.relu2_out, &l2b1,
                b, 128, 16, 16, 128, 1, buf_dl2_mid)?;
            bwd_block(&gpu, &pipes, buf_dl2_mid, l1b1.relu2_out, &l2b0,
                b, 64, 32, 32, 128, 2, buf_dl1)?;

            // Layer1 backward
            bwd_block(&gpu, &pipes, buf_dl1, l1b0.relu2_out, &l1b1,
                b, 64, 32, 32, 64, 1, buf_dl1_mid)?;
            bwd_block(&gpu, &pipes, buf_dl1_mid, buf_relu1, &l1b0,
                b, 64, 32, 32, 64, 1, buf_drelu1)?;

            // Conv1 backward (BN1 + conv1, no need for dX into input)
            cnn_kernels::relu_backward(&gpu, &pipes.cnn, buf_drelu1, bn1.out, bn1.d_out,
                (BATCH * 64 * 32 * 32) as i32)?;
            bwd_bn_conv(&gpu, &pipes.cnn, bn1.d_out, buf_input,
                &conv1, &bn1,
                b, IN_C as i32, 32, 32, 64, 32, 32, 3, 3, 1, 1,
                false, buf_dummy)?;

            // ===== ADAM UPDATES =====
            let (b1t, b2t) = (beta1_t, beta2_t);

            // Conv1 + BN1
            adam_update_conv(&gpu, &pipes.cnn, &conv1, IN_C * 3 * 3 * 64, 64,
                LR, BETA1, BETA2, EPS, b1t, b2t)?;
            adam_update_bn(&gpu, &pipes.cnn, &bn1, 64,
                LR, BETA1, BETA2, EPS, b1t, b2t)?;

            // Layer1
            adam_update_block(&gpu, &pipes.cnn, &l1b0, 64, 64, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;
            adam_update_block(&gpu, &pipes.cnn, &l1b1, 64, 64, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;

            // Layer2
            adam_update_block(&gpu, &pipes.cnn, &l2b0, 64, 128, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;
            adam_update_block(&gpu, &pipes.cnn, &l2b1, 128, 128, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;

            // Layer3
            adam_update_block(&gpu, &pipes.cnn, &l3b0, 128, 256, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;
            adam_update_block(&gpu, &pipes.cnn, &l3b1, 256, 256, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;

            // Layer4
            adam_update_block(&gpu, &pipes.cnn, &l4b0, 256, 512, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;
            adam_update_block(&gpu, &pipes.cnn, &l4b1, 512, 512, 3, LR, BETA1, BETA2, EPS, b1t, b2t)?;

            // FC
            cnn_kernels::adam_step(&gpu, &pipes.cnn, buf_fc_w, buf_dfc_w, buf_mv_fc_w,
                (512 * CLASSES) as i32, LR, BETA1, BETA2, EPS, b1t, b2t)?;
            cnn_kernels::adam_step(&gpu, &pipes.cnn, buf_fc_b, buf_dfc_b, buf_mv_fc_b,
                CLASSES as i32, LR, BETA1, BETA2, EPS, b1t, b2t)?;
        }

        let avg_loss = (total_loss / loss_count as f64) as f32;

        // === TEST ACCURACY ===
        let test_acc = evaluate_resnet(&gpu, &pipes,
            &cifar.test_images, &cifar.test_labels,
            buf_input, buf_dummy,
            &conv1, &bn1, buf_relu1,
            &l1b0, &l1b1, &l2b0, &l2b1,
            &l3b0, &l3b1, &l4b0, &l4b1,
            buf_pool_out,
            buf_fc_w, buf_fc_b, buf_fc_out, buf_fc_z, buf_sm,
        )?;

        let epoch_secs = epoch_start.elapsed().as_secs_f64();
        println!("=== Epoch {}/{}: loss={:.4}, test_acc={:.2}%, time={:.1}s ===",
            epoch + 1, EPOCHS, avg_loss, test_acc * 100.0, epoch_secs);

        epoch_results.push(report::EpochRecord {
            epoch: epoch + 1,
            train_loss: avg_loss,
            test_accuracy: test_acc,
        });
    }

    let total_secs = start.elapsed().as_secs_f64();
    let final_acc = epoch_results.last().map(|e| e.test_accuracy).unwrap_or(0.0);
    let final_loss = epoch_results.last().map(|e| e.train_loss).unwrap_or(0.0);

    println!("\n========================================");
    println!("RESNET-18 TRAINING COMPLETE");
    println!("Final test accuracy: {:.2}%", final_acc * 100.0);
    println!("Final train loss:    {:.4}", final_loss);
    println!("Total time:          {:.1}s", total_secs);
    println!("Device:              {}", gpu.device_name);
    println!("CUDA used:           false");
    println!("========================================\n");

    // Write report
    let spv_kernels: Vec<String> = vec![
        "matmul.spv", "matmul_tn.spv", "matmul_nt.spv",
        "relu_forward.spv", "relu_backward.spv", "bias_add.spv",
        "batched_softmax10.spv", "cross_entropy_loss.spv", "softmax_ce_backward.spv",
        "reduce_sum_rows.spv", "adam_step.spv",
        "conv2d_forward.spv", "conv_bias_add.spv",
        "conv2d_backward_data.spv", "conv2d_backward_weight.spv",
        "bn_mean.spv", "bn_var.spv", "bn_pack_params.spv",
        "bn_forward.spv", "bn_xhat.spv", "bn_dgamma_dbeta.spv",
        "bn_dx_part1.spv", "bn_dx_part2.spv", "bn_dx_part3.spv",
        "maxpool2d_forward_mask.spv", "maxpool2d_backward.spv",
        "reduce_sum_nhw.spv", "zero_fill.spv",
        "element_add.spv", "global_avg_pool.spv", "global_avg_pool_backward.spv",
    ].into_iter().map(String::from).collect();

    let rpt = report::TrainingReport {
        model: "ResNet-18 for CIFAR-10 (32x32 input, no 7x7 conv)".into(),
        architecture: "conv1(3,64,3x3,s=1,p=1)->BN->ReLU->[BasicBlock(64)]x2->[BasicBlock(128,s=2)]x2->[BasicBlock(256,s=2)]x2->[BasicBlock(512,s=2)]x2->GAP->FC(512,10)->Softmax+CE".into(),
        optimizer: format!("Adam(lr={}, beta1={}, beta2={}, eps={})", LR, BETA1, BETA2, EPS),
        learning_rate: LR,
        batch_size: BATCH,
        epochs: EPOCHS,
        device_name: gpu.device_name.clone(),
        vendor_id: gpu.vendor_id,
        device_id: gpu.device_id,
        runtime: "Vulkan 1.1 compute via ash 0.38 (SPIR-V shaders)".into(),
        cuda_used: false,
        epoch_results,
        final_test_accuracy: final_acc,
        final_train_loss: final_loss,
        total_training_time_secs: total_secs,
        spv_kernels,
    };

    let json_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("RESNET_TRAINING_RESULT.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&rpt)?)?;
    println!("[report] {}", json_path.display());

    Ok(())
}

fn evaluate_resnet(
    gpu: &vulkan_ctx::VulkanTrainer, p: &ResNetPipelines,
    images: &[f32], labels: &[u8],
    buf_input: Buf, buf_dummy: Buf,
    conv1: &ConvBufs, bn1: &BnBufs, buf_relu1: Buf,
    l1b0: &BasicBlockBufs, l1b1: &BasicBlockBufs,
    l2b0: &BasicBlockBufs, l2b1: &BasicBlockBufs,
    l3b0: &BasicBlockBufs, l3b1: &BasicBlockBufs,
    l4b0: &BasicBlockBufs, l4b1: &BasicBlockBufs,
    buf_pool_out: Buf,
    buf_fc_w: Buf, buf_fc_b: Buf,
    buf_fc_out: Buf, buf_fc_z: Buf, buf_sm: Buf,
) -> Result<f32> {
    let n = labels.len();
    let num_batches = n / BATCH;
    let mut correct = 0_usize;
    let mut total = 0_usize;
    let img_size = IN_C * IN_H * IN_W;
    let b = BATCH as i32;

    for batch in 0..num_batches {
        let start = batch * BATCH;
        let batch_img = &images[start * img_size..(start + BATCH) * img_size];
        gpu.upload(buf_input, batch_img)?;

        // Forward pass (no loss computation needed)
        fwd_conv_bn(gpu, &p.cnn, buf_input, conv1, bn1,
            b, IN_C as i32, 32, 32, 64, 32, 32, 3, 3, 1, 1)?;
        cnn_kernels::relu_forward(gpu, &p.cnn, bn1.out, buf_dummy, buf_relu1,
            (BATCH * 64 * 32 * 32) as i32)?;

        fwd_block(gpu, p, buf_relu1, l1b0, b, 64, 32, 32, 64, 1)?;
        fwd_block(gpu, p, l1b0.relu2_out, l1b1, b, 64, 32, 32, 64, 1)?;

        fwd_block(gpu, p, l1b1.relu2_out, l2b0, b, 64, 32, 32, 128, 2)?;
        fwd_block(gpu, p, l2b0.relu2_out, l2b1, b, 128, 16, 16, 128, 1)?;

        fwd_block(gpu, p, l2b1.relu2_out, l3b0, b, 128, 16, 16, 256, 2)?;
        fwd_block(gpu, p, l3b0.relu2_out, l3b1, b, 256, 8, 8, 256, 1)?;

        fwd_block(gpu, p, l3b1.relu2_out, l4b0, b, 256, 8, 8, 512, 2)?;
        fwd_block(gpu, p, l4b0.relu2_out, l4b1, b, 512, 4, 4, 512, 1)?;

        global_avg_pool(gpu, p, l4b1.relu2_out, buf_dummy, buf_pool_out,
            b, 512, 4, 4)?;

        cnn_kernels::matmul(gpu, &p.cnn, buf_pool_out, buf_fc_w, buf_fc_out,
            b, 512, CLASSES as i32)?;
        cnn_kernels::bias_add(gpu, &p.cnn, buf_fc_out, buf_fc_b, buf_fc_z,
            (BATCH * CLASSES) as i32, CLASSES as i32)?;
        cnn_kernels::batched_softmax10(gpu, &p.cnn, buf_fc_z, buf_dummy, buf_sm, b)?;

        let sm = gpu.download(buf_sm, BATCH * CLASSES)?;
        for i in 0..BATCH {
            let pred = sm[i * CLASSES..(i + 1) * CLASSES]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred == labels[start + i] as usize {
                correct += 1;
            }
            total += 1;
        }
    }

    Ok(correct as f32 / total as f32)
}

fn kaiming_init(fan_in: usize, fan_out: usize, rng: &mut impl rand::Rng) -> Vec<f32> {
    let std = (2.0 / fan_in as f64).sqrt() as f32;
    (0..fan_in * fan_out)
        .map(|_| {
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z * std
        })
        .collect()
}

fn count_params(ic: usize, oc: usize, k: usize) -> usize {
    oc * ic * k * k + oc // weight + bias
}

fn count_block_params(ic: usize, oc: usize) -> usize {
    count_params(ic, oc, 3) + 2 * oc  // conv1 + bn1
    + count_params(oc, oc, 3) + 2 * oc  // conv2 + bn2
}

fn count_block_params_ds(ic: usize, oc: usize) -> usize {
    count_block_params(ic, oc)
    + count_params(ic, oc, 1) + 2 * oc  // downsample conv + bn
}
