mod vulkan_ctx;
mod cnn_kernels;
mod cifar10;
mod report;

use anyhow::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::PathBuf;
use std::time::Instant;

// Architecture: Conv2D(3->32,3x3,pad=1)->BN->ReLU->MaxPool(2)
//            -> Conv2D(32->64,3x3,pad=1)->BN->ReLU->MaxPool(2)
//            -> Flatten -> FC(4096->10) -> Softmax -> CE

const BATCH: usize = 64;
const EPOCHS: usize = 10;
const LR: f32 = 0.001;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const BN_EPS: f32 = 1e-5;
const CLASSES: usize = 10;

// Input: [N, 3, 32, 32]
const IN_C: usize = 3;
const IN_H: usize = 32;
const IN_W: usize = 32;

// Conv1: 3->32, 3x3, pad=1, stride=1 => [N, 32, 32, 32]
const C1_OC: usize = 32;
const C1_K: usize = 3;
const C1_PAD: usize = 1;
const C1_OH: usize = 32;
const C1_OW: usize = 32;

// Pool1: 2x2, stride=2 => [N, 32, 16, 16]
const P1_OH: usize = 16;
const P1_OW: usize = 16;

// Conv2: 32->64, 3x3, pad=1, stride=1 => [N, 64, 16, 16]
const C2_OC: usize = 64;
const C2_K: usize = 3;
const C2_PAD: usize = 1;
const C2_OH: usize = 16;
const C2_OW: usize = 16;

// Pool2: 2x2, stride=2 => [N, 64, 8, 8]
const P2_OH: usize = 8;
const P2_OW: usize = 8;

// FC: 64*8*8=4096 -> 10
const FC_IN: usize = C2_OC * P2_OH * P2_OW; // 4096

fn main() -> Result<()> {
    let start = Instant::now();
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");

    // 1. Load CIFAR-10
    let cifar = cifar10::load(&data_dir)?;

    // 2. Init Vulkan
    let mut gpu = vulkan_ctx::VulkanTrainer::new("nvidia")?;
    let pipes = cnn_kernels::CnnPipelines::load(&mut gpu)?;
    println!("[init] 28 pipelines loaded");

    // 3. Allocate GPU buffers
    // Input/labels
    let buf_input  = gpu.alloc(BATCH * IN_C * IN_H * IN_W)?;
    let buf_labels = gpu.alloc(BATCH * CLASSES)?;
    let buf_dummy  = gpu.alloc(1)?;

    // Conv1 weights: [32, 3, 3, 3] = 864
    let buf_conv1_w = gpu.alloc(C1_OC * IN_C * C1_K * C1_K)?;
    let buf_conv1_b = gpu.alloc(C1_OC)?;
    // Conv1 forward activations
    let buf_conv1_out  = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;  // pre-bias
    let buf_conv1_bias = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;  // post-bias (BN input)

    // BN1: gamma/beta packed [gamma(32), beta(32)], stats [mean(32), var(32)], bnparams [mean,var,gamma,beta = 4*32]
    let buf_bn1_gb     = gpu.alloc(2 * C1_OC)?;        // gamma+beta
    let buf_bn1_stats  = gpu.alloc(2 * C1_OC)?;        // mean+var
    let buf_bn1_params = gpu.alloc(4 * C1_OC)?;        // packed params
    let buf_bn1_out    = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;
    let buf_bn1_xhat   = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;

    // ReLU1
    let buf_relu1_out = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;

    // Pool1: [N, 32, 16, 16]
    let buf_pool1_out  = gpu.alloc(BATCH * C1_OC * P1_OH * P1_OW)?;
    let buf_pool1_mask = gpu.alloc(BATCH * C1_OC * P1_OH * P1_OW)?;

    // Conv2 weights: [64, 32, 3, 3] = 18432
    let buf_conv2_w = gpu.alloc(C2_OC * C1_OC * C2_K * C2_K)?;
    let buf_conv2_b = gpu.alloc(C2_OC)?;
    // Conv2 forward activations
    let buf_conv2_out  = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;
    let buf_conv2_bias = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;

    // BN2
    let buf_bn2_gb     = gpu.alloc(2 * C2_OC)?;
    let buf_bn2_stats  = gpu.alloc(2 * C2_OC)?;
    let buf_bn2_params = gpu.alloc(4 * C2_OC)?;
    let buf_bn2_out    = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;
    let buf_bn2_xhat   = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;

    // ReLU2
    let buf_relu2_out = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;

    // Pool2: [N, 64, 8, 8]
    let buf_pool2_out  = gpu.alloc(BATCH * C2_OC * P2_OH * P2_OW)?;
    let buf_pool2_mask = gpu.alloc(BATCH * C2_OC * P2_OH * P2_OW)?;

    // FC: flatten pool2 -> [N, 4096] @ W[4096, 10] + b[10]
    let buf_fc_w   = gpu.alloc(FC_IN * CLASSES)?;
    let buf_fc_b   = gpu.alloc(CLASSES)?;
    let buf_fc_out = gpu.alloc(BATCH * CLASSES)?;  // pre-bias scratch
    let buf_fc_z   = gpu.alloc(BATCH * CLASSES)?;  // post-bias
    let buf_sm     = gpu.alloc(BATCH * CLASSES)?;
    let buf_loss   = gpu.alloc(1)?;

    // Backward buffers
    let buf_dfc_z  = gpu.alloc(BATCH * CLASSES)?;
    let buf_dfc_w  = gpu.alloc(FC_IN * CLASSES)?;
    let buf_dfc_b  = gpu.alloc(CLASSES)?;
    let buf_dpool2 = gpu.alloc(BATCH * C2_OC * P2_OH * P2_OW)?;  // grad from FC

    // Pool2 backward -> [N, 64, 16, 16]
    let buf_drelu2 = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;

    // BN2 backward
    let buf_dbn2_out    = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;
    let buf_dbn2_params = gpu.alloc(2 * C2_OC)?;    // dgamma + dbeta
    let buf_dbn2_scratch= gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;
    let buf_dbn2_prescale= gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;
    let buf_dconv2_bias = gpu.alloc(BATCH * C2_OC * C2_OH * C2_OW)?;

    // Conv2 backward
    let buf_dconv2_w = gpu.alloc(C2_OC * C1_OC * C2_K * C2_K)?;
    let buf_dconv2_b = gpu.alloc(C2_OC)?;
    let buf_dpool1   = gpu.alloc(BATCH * C1_OC * P1_OH * P1_OW)?;

    // Pool1 backward -> [N, 32, 32, 32]
    let buf_drelu1 = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;

    // BN1 backward
    let buf_dbn1_out     = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;
    let buf_dbn1_params  = gpu.alloc(2 * C1_OC)?;
    let buf_dbn1_scratch = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;
    let buf_dbn1_prescale= gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;
    let buf_dconv1_bias  = gpu.alloc(BATCH * C1_OC * C1_OH * C1_OW)?;

    // Conv1 backward
    let buf_dconv1_w = gpu.alloc(C1_OC * IN_C * C1_K * C1_K)?;
    let buf_dconv1_b = gpu.alloc(C1_OC)?;

    // Adam state (m+v packed)
    let buf_mv_conv1_w = gpu.alloc(2 * C1_OC * IN_C * C1_K * C1_K)?;
    let buf_mv_conv1_b = gpu.alloc(2 * C1_OC)?;
    let buf_mv_bn1_gb  = gpu.alloc(2 * 2 * C1_OC)?;
    let buf_mv_conv2_w = gpu.alloc(2 * C2_OC * C1_OC * C2_K * C2_K)?;
    let buf_mv_conv2_b = gpu.alloc(2 * C2_OC)?;
    let buf_mv_bn2_gb  = gpu.alloc(2 * 2 * C2_OC)?;
    let buf_mv_fc_w    = gpu.alloc(2 * FC_IN * CLASSES)?;
    let buf_mv_fc_b    = gpu.alloc(2 * CLASSES)?;

    let num_bufs = 56; // approximate
    println!("[init] ~{} GPU buffers allocated", num_bufs);

    // 4. Initialize weights
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Conv1: Kaiming He for [OC, IC, KH, KW]
    let conv1_w = kaiming_init(IN_C * C1_K * C1_K, C1_OC, &mut rng);
    let conv1_b = vec![0.0_f32; C1_OC];
    gpu.upload(buf_conv1_w, &conv1_w)?;
    gpu.upload(buf_conv1_b, &conv1_b)?;

    // BN1: gamma=1, beta=0
    let mut bn1_gb = vec![1.0_f32; C1_OC];
    bn1_gb.extend(vec![0.0_f32; C1_OC]);
    gpu.upload(buf_bn1_gb, &bn1_gb)?;

    // Conv2: Kaiming He
    let conv2_w = kaiming_init(C1_OC * C2_K * C2_K, C2_OC, &mut rng);
    let conv2_b = vec![0.0_f32; C2_OC];
    gpu.upload(buf_conv2_w, &conv2_w)?;
    gpu.upload(buf_conv2_b, &conv2_b)?;

    // BN2: gamma=1, beta=0
    let mut bn2_gb = vec![1.0_f32; C2_OC];
    bn2_gb.extend(vec![0.0_f32; C2_OC]);
    gpu.upload(buf_bn2_gb, &bn2_gb)?;

    // FC: Kaiming He
    let fc_w = kaiming_init(FC_IN, CLASSES, &mut rng);
    let fc_b = vec![0.0_f32; CLASSES];
    gpu.upload(buf_fc_w, &fc_w)?;
    gpu.upload(buf_fc_b, &fc_b)?;

    println!("[init] weights uploaded (Kaiming He, seed=42)");
    println!("[arch] Conv2D(3->32,3x3,p=1)->BN->ReLU->MaxPool(2)->Conv2D(32->64,3x3,p=1)->BN->ReLU->MaxPool(2)->FC(4096->10)");
    println!("[params] conv1_w={}, conv1_b={}, bn1={}, conv2_w={}, conv2_b={}, bn2={}, fc_w={}, fc_b={}",
        conv1_w.len(), conv1_b.len(), 2*C1_OC, conv2_w.len(), conv2_b.len(), 2*C2_OC, fc_w.len(), fc_b.len());
    let total_params = conv1_w.len() + conv1_b.len() + 2*C1_OC + conv2_w.len() + conv2_b.len() + 2*C2_OC + fc_w.len() + fc_b.len();
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

            // Prepare batch with augmentation
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

            // ===== FORWARD =====
            forward_pass(&gpu, &pipes,
                buf_input, buf_dummy,
                buf_conv1_w, buf_conv1_b, buf_conv1_out, buf_conv1_bias,
                buf_bn1_gb, buf_bn1_stats, buf_bn1_params, buf_bn1_out, buf_bn1_xhat,
                buf_relu1_out, buf_pool1_out, buf_pool1_mask,
                buf_conv2_w, buf_conv2_b, buf_conv2_out, buf_conv2_bias,
                buf_bn2_gb, buf_bn2_stats, buf_bn2_params, buf_bn2_out, buf_bn2_xhat,
                buf_relu2_out, buf_pool2_out, buf_pool2_mask,
                buf_fc_w, buf_fc_b, buf_fc_out, buf_fc_z, buf_sm,
                buf_loss, buf_labels,
            )?;

            // Read loss
            let loss_val = gpu.download(buf_loss, 1)?[0];
            total_loss += loss_val as f64;
            loss_count += 1;

            if batch_idx % 100 == 0 {
                println!(
                    "  epoch {}/{} batch {}/{} loss={:.4}",
                    epoch + 1, EPOCHS, batch_idx, batches_per_epoch, loss_val
                );
            }

            // ===== BACKWARD =====
            backward_pass(&gpu, &pipes,
                buf_input, buf_dummy,
                buf_conv1_w, buf_conv1_bias,
                buf_bn1_params, buf_bn1_xhat, buf_bn1_out,
                buf_relu1_out, buf_pool1_out, buf_pool1_mask,
                buf_conv2_w, buf_conv2_bias,
                buf_bn2_params, buf_bn2_xhat, buf_bn2_out,
                buf_relu2_out, buf_pool2_out, buf_pool2_mask,
                buf_fc_w, buf_sm, buf_labels,
                // grad buffers
                buf_dfc_z, buf_dfc_w, buf_dfc_b, buf_dpool2,
                buf_drelu2, buf_dbn2_out, buf_dbn2_params, buf_dbn2_scratch, buf_dbn2_prescale,
                buf_dconv2_bias, buf_dconv2_w, buf_dconv2_b, buf_dpool1,
                buf_drelu1, buf_dbn1_out, buf_dbn1_params, buf_dbn1_scratch, buf_dbn1_prescale,
                buf_dconv1_bias, buf_dconv1_w, buf_dconv1_b,
            )?;

            // ===== ADAM =====
            let btp = (beta1_t, beta2_t);
            // Conv1
            cnn_kernels::adam_step(&gpu, &pipes, buf_conv1_w, buf_dconv1_w, buf_mv_conv1_w,
                (C1_OC * IN_C * C1_K * C1_K) as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
            cnn_kernels::adam_step(&gpu, &pipes, buf_conv1_b, buf_dconv1_b, buf_mv_conv1_b,
                C1_OC as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
            // BN1 gamma/beta
            cnn_kernels::adam_step(&gpu, &pipes, buf_bn1_gb, buf_dbn1_params, buf_mv_bn1_gb,
                (2 * C1_OC) as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
            // Conv2
            cnn_kernels::adam_step(&gpu, &pipes, buf_conv2_w, buf_dconv2_w, buf_mv_conv2_w,
                (C2_OC * C1_OC * C2_K * C2_K) as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
            cnn_kernels::adam_step(&gpu, &pipes, buf_conv2_b, buf_dconv2_b, buf_mv_conv2_b,
                C2_OC as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
            // BN2 gamma/beta
            cnn_kernels::adam_step(&gpu, &pipes, buf_bn2_gb, buf_dbn2_params, buf_mv_bn2_gb,
                (2 * C2_OC) as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
            // FC
            cnn_kernels::adam_step(&gpu, &pipes, buf_fc_w, buf_dfc_w, buf_mv_fc_w,
                (FC_IN * CLASSES) as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
            cnn_kernels::adam_step(&gpu, &pipes, buf_fc_b, buf_dfc_b, buf_mv_fc_b,
                CLASSES as i32, LR, BETA1, BETA2, EPS, btp.0, btp.1)?;
        }

        let avg_loss = (total_loss / loss_count as f64) as f32;

        // === TEST ACCURACY ===
        let test_acc = evaluate(&gpu, &pipes, &cifar.test_images, &cifar.test_labels,
            buf_input, buf_dummy,
            buf_conv1_w, buf_conv1_b, buf_conv1_out, buf_conv1_bias,
            buf_bn1_gb, buf_bn1_stats, buf_bn1_params, buf_bn1_out, buf_bn1_xhat,
            buf_relu1_out, buf_pool1_out, buf_pool1_mask,
            buf_conv2_w, buf_conv2_b, buf_conv2_out, buf_conv2_bias,
            buf_bn2_gb, buf_bn2_stats, buf_bn2_params, buf_bn2_out, buf_bn2_xhat,
            buf_relu2_out, buf_pool2_out, buf_pool2_mask,
            buf_fc_w, buf_fc_b, buf_fc_out, buf_fc_z, buf_sm,
        )?;

        let epoch_secs = epoch_start.elapsed().as_secs_f64();
        println!(
            "=== Epoch {}/{}: loss={:.4}, test_acc={:.2}%, time={:.1}s ===",
            epoch + 1, EPOCHS, avg_loss, test_acc * 100.0, epoch_secs
        );

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
    println!("CNN TRAINING COMPLETE");
    println!("Final test accuracy: {:.2}%", final_acc * 100.0);
    println!("Final train loss:    {:.4}", final_loss);
    println!("Total time:          {:.1}s", total_secs);
    println!("Device:              {}", gpu.device_name);
    println!("CUDA used:           false");
    println!("========================================\n");

    // 6. Write report
    let rpt = report::TrainingReport {
        model: "CNN: Conv2D(3->32)->BN->ReLU->Pool->Conv2D(32->64)->BN->ReLU->Pool->FC(4096->10)".into(),
        architecture: "Conv2D(3,32,3x3,pad=1)->BN(32)->ReLU->MaxPool(2)->Conv2D(32,64,3x3,pad=1)->BN(64)->ReLU->MaxPool(2)->FC(4096,10)->Softmax+CE".into(),
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
        spv_kernels: vec![
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
        ].into_iter().map(String::from).collect(),
    };

    let json_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("CNN_TRAINING_RESULT.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&rpt)?)?;
    println!("[report] {}", json_path.display());

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_pass(
    gpu: &vulkan_ctx::VulkanTrainer,
    p: &cnn_kernels::CnnPipelines,
    buf_input: vulkan_ctx::Buf, buf_dummy: vulkan_ctx::Buf,
    // Conv1
    buf_conv1_w: vulkan_ctx::Buf, buf_conv1_b: vulkan_ctx::Buf,
    buf_conv1_out: vulkan_ctx::Buf, buf_conv1_bias: vulkan_ctx::Buf,
    // BN1
    buf_bn1_gb: vulkan_ctx::Buf, buf_bn1_stats: vulkan_ctx::Buf,
    buf_bn1_params: vulkan_ctx::Buf, buf_bn1_out: vulkan_ctx::Buf,
    buf_bn1_xhat: vulkan_ctx::Buf,
    // ReLU1 + Pool1
    buf_relu1_out: vulkan_ctx::Buf,
    buf_pool1_out: vulkan_ctx::Buf, buf_pool1_mask: vulkan_ctx::Buf,
    // Conv2
    buf_conv2_w: vulkan_ctx::Buf, buf_conv2_b: vulkan_ctx::Buf,
    buf_conv2_out: vulkan_ctx::Buf, buf_conv2_bias: vulkan_ctx::Buf,
    // BN2
    buf_bn2_gb: vulkan_ctx::Buf, buf_bn2_stats: vulkan_ctx::Buf,
    buf_bn2_params: vulkan_ctx::Buf, buf_bn2_out: vulkan_ctx::Buf,
    buf_bn2_xhat: vulkan_ctx::Buf,
    // ReLU2 + Pool2
    buf_relu2_out: vulkan_ctx::Buf,
    buf_pool2_out: vulkan_ctx::Buf, buf_pool2_mask: vulkan_ctx::Buf,
    // FC + softmax + loss
    buf_fc_w: vulkan_ctx::Buf, buf_fc_b: vulkan_ctx::Buf,
    buf_fc_out: vulkan_ctx::Buf, buf_fc_z: vulkan_ctx::Buf,
    buf_sm: vulkan_ctx::Buf, buf_loss: vulkan_ctx::Buf,
    buf_labels: vulkan_ctx::Buf,
) -> Result<()> {
    let b = BATCH as i32;

    // Conv1: input[N,3,32,32] * w[32,3,3,3] -> [N,32,32,32]
    cnn_kernels::conv2d_forward(gpu, p, buf_input, buf_conv1_w, buf_conv1_out,
        b, IN_C as i32, IN_H as i32, IN_W as i32,
        C1_OC as i32, C1_OH as i32, C1_OW as i32,
        C1_K as i32, C1_K as i32, 1, C1_PAD as i32)?;
    // + bias
    cnn_kernels::conv_bias_add(gpu, p, buf_conv1_out, buf_conv1_b, buf_conv1_bias,
        (BATCH * C1_OC * C1_OH * C1_OW) as i32, C1_OC as i32, (C1_OH * C1_OW) as i32)?;

    // BN1
    cnn_kernels::bn_compute_mean(gpu, p, buf_conv1_bias, buf_dummy, buf_bn1_stats,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32)?;
    cnn_kernels::bn_compute_var(gpu, p, buf_conv1_bias, buf_bn1_stats, buf_bn1_stats,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32)?;
    cnn_kernels::bn_pack_params(gpu, p, buf_bn1_gb, buf_bn1_stats, buf_bn1_params,
        C1_OC as i32)?;
    cnn_kernels::bn_forward_apply(gpu, p, buf_conv1_bias, buf_bn1_params, buf_bn1_out,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32, BN_EPS)?;
    // Save xhat for backward
    cnn_kernels::bn_compute_xhat(gpu, p, buf_conv1_bias, buf_bn1_params, buf_bn1_xhat,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32, BN_EPS)?;

    // ReLU1
    cnn_kernels::relu_forward(gpu, p, buf_bn1_out, buf_dummy, buf_relu1_out,
        (BATCH * C1_OC * C1_OH * C1_OW) as i32)?;

    // MaxPool1: [N,32,32,32] -> [N,32,16,16]
    cnn_kernels::maxpool2d_forward(gpu, p, buf_relu1_out, buf_pool1_mask, buf_pool1_out,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32,
        P1_OH as i32, P1_OW as i32, 2, 2, 2, 0)?;

    // Conv2: [N,32,16,16] * w[64,32,3,3] -> [N,64,16,16]
    cnn_kernels::conv2d_forward(gpu, p, buf_pool1_out, buf_conv2_w, buf_conv2_out,
        b, C1_OC as i32, P1_OH as i32, P1_OW as i32,
        C2_OC as i32, C2_OH as i32, C2_OW as i32,
        C2_K as i32, C2_K as i32, 1, C2_PAD as i32)?;
    cnn_kernels::conv_bias_add(gpu, p, buf_conv2_out, buf_conv2_b, buf_conv2_bias,
        (BATCH * C2_OC * C2_OH * C2_OW) as i32, C2_OC as i32, (C2_OH * C2_OW) as i32)?;

    // BN2
    cnn_kernels::bn_compute_mean(gpu, p, buf_conv2_bias, buf_dummy, buf_bn2_stats,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32)?;
    cnn_kernels::bn_compute_var(gpu, p, buf_conv2_bias, buf_bn2_stats, buf_bn2_stats,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32)?;
    cnn_kernels::bn_pack_params(gpu, p, buf_bn2_gb, buf_bn2_stats, buf_bn2_params,
        C2_OC as i32)?;
    cnn_kernels::bn_forward_apply(gpu, p, buf_conv2_bias, buf_bn2_params, buf_bn2_out,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32, BN_EPS)?;
    cnn_kernels::bn_compute_xhat(gpu, p, buf_conv2_bias, buf_bn2_params, buf_bn2_xhat,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32, BN_EPS)?;

    // ReLU2
    cnn_kernels::relu_forward(gpu, p, buf_bn2_out, buf_dummy, buf_relu2_out,
        (BATCH * C2_OC * C2_OH * C2_OW) as i32)?;

    // MaxPool2: [N,64,16,16] -> [N,64,8,8]
    cnn_kernels::maxpool2d_forward(gpu, p, buf_relu2_out, buf_pool2_mask, buf_pool2_out,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32,
        P2_OH as i32, P2_OW as i32, 2, 2, 2, 0)?;

    // FC: pool2_out is [N, 64*8*8=4096], treat as flat
    // z = pool2_out @ fc_w + fc_b
    cnn_kernels::matmul(gpu, p, buf_pool2_out, buf_fc_w, buf_fc_out,
        b, FC_IN as i32, CLASSES as i32)?;
    cnn_kernels::bias_add(gpu, p, buf_fc_out, buf_fc_b, buf_fc_z,
        (BATCH * CLASSES) as i32, CLASSES as i32)?;

    // Softmax + CE loss
    cnn_kernels::batched_softmax10(gpu, p, buf_fc_z, buf_dummy, buf_sm, b)?;
    cnn_kernels::cross_entropy_loss(gpu, p, buf_sm, buf_labels, buf_loss, b, CLASSES as i32)?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backward_pass(
    gpu: &vulkan_ctx::VulkanTrainer,
    p: &cnn_kernels::CnnPipelines,
    buf_input: vulkan_ctx::Buf, buf_dummy: vulkan_ctx::Buf,
    // Conv1 saved
    buf_conv1_w: vulkan_ctx::Buf, buf_conv1_bias: vulkan_ctx::Buf,
    buf_bn1_params: vulkan_ctx::Buf, buf_bn1_xhat: vulkan_ctx::Buf,
    _buf_bn1_out: vulkan_ctx::Buf,
    buf_relu1_out: vulkan_ctx::Buf,
    buf_pool1_out: vulkan_ctx::Buf, buf_pool1_mask: vulkan_ctx::Buf,
    // Conv2 saved
    buf_conv2_w: vulkan_ctx::Buf, buf_conv2_bias: vulkan_ctx::Buf,
    buf_bn2_params: vulkan_ctx::Buf, buf_bn2_xhat: vulkan_ctx::Buf,
    _buf_bn2_out: vulkan_ctx::Buf,
    buf_relu2_out: vulkan_ctx::Buf,
    _buf_pool2_out: vulkan_ctx::Buf, buf_pool2_mask: vulkan_ctx::Buf,
    // FC saved
    buf_fc_w: vulkan_ctx::Buf,
    buf_sm: vulkan_ctx::Buf, buf_labels: vulkan_ctx::Buf,
    // Grad buffers
    buf_dfc_z: vulkan_ctx::Buf, buf_dfc_w: vulkan_ctx::Buf,
    buf_dfc_b: vulkan_ctx::Buf, buf_dpool2: vulkan_ctx::Buf,
    buf_drelu2: vulkan_ctx::Buf,
    buf_dbn2_out: vulkan_ctx::Buf, buf_dbn2_params: vulkan_ctx::Buf,
    buf_dbn2_scratch: vulkan_ctx::Buf, buf_dbn2_prescale: vulkan_ctx::Buf,
    buf_dconv2_bias: vulkan_ctx::Buf,
    buf_dconv2_w: vulkan_ctx::Buf, buf_dconv2_b: vulkan_ctx::Buf,
    buf_dpool1: vulkan_ctx::Buf,
    buf_drelu1: vulkan_ctx::Buf,
    buf_dbn1_out: vulkan_ctx::Buf, buf_dbn1_params: vulkan_ctx::Buf,
    buf_dbn1_scratch: vulkan_ctx::Buf, buf_dbn1_prescale: vulkan_ctx::Buf,
    buf_dconv1_bias: vulkan_ctx::Buf,
    buf_dconv1_w: vulkan_ctx::Buf, buf_dconv1_b: vulkan_ctx::Buf,
) -> Result<()> {
    let b = BATCH as i32;

    // === FC backward ===
    // dfc_z = (sm - labels) / batch_size
    cnn_kernels::softmax_ce_backward(gpu, p, buf_sm, buf_labels, buf_dfc_z,
        (BATCH * CLASSES) as i32, b)?;
    // dfc_w = pool2_out^T @ dfc_z  (pool2_out is buf_pool2_out but we reuse the flat view)
    // We need pool2_out here — it's _buf_pool2_out
    cnn_kernels::matmul_tn(gpu, p, _buf_pool2_out, buf_dfc_z, buf_dfc_w,
        FC_IN as i32, b, CLASSES as i32)?;
    // dfc_b = sum_rows(dfc_z)
    cnn_kernels::reduce_sum_rows(gpu, p, buf_dfc_z, buf_dummy, buf_dfc_b,
        b, CLASSES as i32)?;
    // dpool2 = dfc_z @ fc_w^T  [N, 4096]
    cnn_kernels::matmul(gpu, p, buf_dfc_z, buf_fc_w, buf_dpool2,
        b, CLASSES as i32, FC_IN as i32)?;

    // === Pool2 backward: scatter [N,64,8,8] -> [N,64,16,16] ===
    let pool2_out_size = (BATCH * C2_OC * C2_OH * C2_OW) as i32;
    cnn_kernels::zero_fill(gpu, p, buf_dummy, buf_drelu2, pool2_out_size)?;
    cnn_kernels::maxpool2d_backward(gpu, p, buf_dpool2, buf_pool2_mask, buf_drelu2,
        (BATCH * C2_OC * P2_OH * P2_OW) as i32)?;

    // === ReLU2 backward ===
    cnn_kernels::relu_backward(gpu, p, buf_drelu2, buf_relu2_out, buf_dbn2_out,
        pool2_out_size)?;

    // === BN2 backward (3-pass decomposition) ===
    let m2 = (BATCH * C2_OH * C2_OW) as i32;
    cnn_kernels::bn_compute_dgamma_dbeta(gpu, p, buf_dbn2_out, buf_bn2_xhat, buf_dbn2_params,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32)?;
    cnn_kernels::bn_dx_part1(gpu, p, buf_bn2_xhat, buf_dbn2_params, buf_dbn2_scratch,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32)?;
    cnn_kernels::bn_dx_part2(gpu, p, buf_dbn2_out, buf_dbn2_scratch, buf_dbn2_prescale,
        pool2_out_size, m2)?;
    cnn_kernels::bn_dx_part3(gpu, p, buf_dbn2_prescale, buf_bn2_params, buf_dconv2_bias,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32, BN_EPS, m2)?;

    // === Conv2 backward ===
    // dconv2_b = reduce over N,H,W
    cnn_kernels::reduce_sum_nhw(gpu, p, buf_dconv2_bias, buf_dummy, buf_dconv2_b,
        b, C2_OC as i32, C2_OH as i32, C2_OW as i32)?;
    // dconv2_w
    cnn_kernels::conv2d_backward_weight(gpu, p, buf_pool1_out, buf_dconv2_bias, buf_dconv2_w,
        b, C1_OC as i32, P1_OH as i32, P1_OW as i32,
        C2_OC as i32, C2_OH as i32, C2_OW as i32,
        C2_K as i32, C2_K as i32, 1, C2_PAD as i32)?;
    // dpool1 = conv2d_backward_data
    cnn_kernels::conv2d_backward_data(gpu, p, buf_dconv2_bias, buf_conv2_w, buf_dpool1,
        b, C1_OC as i32, P1_OH as i32, P1_OW as i32,
        C2_OC as i32, C2_OH as i32, C2_OW as i32,
        C2_K as i32, C2_K as i32, 1, C2_PAD as i32)?;

    // === Pool1 backward: scatter [N,32,16,16] -> [N,32,32,32] ===
    let pool1_in_size = (BATCH * C1_OC * C1_OH * C1_OW) as i32;
    cnn_kernels::zero_fill(gpu, p, buf_dummy, buf_drelu1, pool1_in_size)?;
    cnn_kernels::maxpool2d_backward(gpu, p, buf_dpool1, buf_pool1_mask, buf_drelu1,
        (BATCH * C1_OC * P1_OH * P1_OW) as i32)?;

    // === ReLU1 backward ===
    cnn_kernels::relu_backward(gpu, p, buf_drelu1, buf_relu1_out, buf_dbn1_out,
        pool1_in_size)?;

    // === BN1 backward ===
    let m1 = (BATCH * C1_OH * C1_OW) as i32;
    cnn_kernels::bn_compute_dgamma_dbeta(gpu, p, buf_dbn1_out, buf_bn1_xhat, buf_dbn1_params,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32)?;
    cnn_kernels::bn_dx_part1(gpu, p, buf_bn1_xhat, buf_dbn1_params, buf_dbn1_scratch,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32)?;
    cnn_kernels::bn_dx_part2(gpu, p, buf_dbn1_out, buf_dbn1_scratch, buf_dbn1_prescale,
        pool1_in_size, m1)?;
    cnn_kernels::bn_dx_part3(gpu, p, buf_dbn1_prescale, buf_bn1_params, buf_dconv1_bias,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32, BN_EPS, m1)?;

    // === Conv1 backward ===
    cnn_kernels::reduce_sum_nhw(gpu, p, buf_dconv1_bias, buf_dummy, buf_dconv1_b,
        b, C1_OC as i32, C1_OH as i32, C1_OW as i32)?;
    cnn_kernels::conv2d_backward_weight(gpu, p, buf_input, buf_dconv1_bias, buf_dconv1_w,
        b, IN_C as i32, IN_H as i32, IN_W as i32,
        C1_OC as i32, C1_OH as i32, C1_OW as i32,
        C1_K as i32, C1_K as i32, 1, C1_PAD as i32)?;
    // We don't need dX for input layer, skip conv1 backward_data

    Ok(())
}

fn evaluate(
    gpu: &vulkan_ctx::VulkanTrainer,
    p: &cnn_kernels::CnnPipelines,
    images: &[f32],
    labels: &[u8],
    buf_input: vulkan_ctx::Buf, buf_dummy: vulkan_ctx::Buf,
    buf_conv1_w: vulkan_ctx::Buf, buf_conv1_b: vulkan_ctx::Buf,
    buf_conv1_out: vulkan_ctx::Buf, buf_conv1_bias: vulkan_ctx::Buf,
    buf_bn1_gb: vulkan_ctx::Buf, buf_bn1_stats: vulkan_ctx::Buf,
    buf_bn1_params: vulkan_ctx::Buf, buf_bn1_out: vulkan_ctx::Buf,
    _buf_bn1_xhat: vulkan_ctx::Buf,
    buf_relu1_out: vulkan_ctx::Buf,
    buf_pool1_out: vulkan_ctx::Buf, buf_pool1_mask: vulkan_ctx::Buf,
    buf_conv2_w: vulkan_ctx::Buf, buf_conv2_b: vulkan_ctx::Buf,
    buf_conv2_out: vulkan_ctx::Buf, buf_conv2_bias: vulkan_ctx::Buf,
    buf_bn2_gb: vulkan_ctx::Buf, buf_bn2_stats: vulkan_ctx::Buf,
    buf_bn2_params: vulkan_ctx::Buf, buf_bn2_out: vulkan_ctx::Buf,
    _buf_bn2_xhat: vulkan_ctx::Buf,
    buf_relu2_out: vulkan_ctx::Buf,
    buf_pool2_out: vulkan_ctx::Buf, buf_pool2_mask: vulkan_ctx::Buf,
    buf_fc_w: vulkan_ctx::Buf, buf_fc_b: vulkan_ctx::Buf,
    buf_fc_out: vulkan_ctx::Buf, buf_fc_z: vulkan_ctx::Buf,
    buf_sm: vulkan_ctx::Buf,
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

        // Forward only (no loss needed)
        cnn_kernels::conv2d_forward(gpu, p, buf_input, buf_conv1_w, buf_conv1_out,
            b, IN_C as i32, IN_H as i32, IN_W as i32,
            C1_OC as i32, C1_OH as i32, C1_OW as i32,
            C1_K as i32, C1_K as i32, 1, C1_PAD as i32)?;
        cnn_kernels::conv_bias_add(gpu, p, buf_conv1_out, buf_conv1_b, buf_conv1_bias,
            (BATCH * C1_OC * C1_OH * C1_OW) as i32, C1_OC as i32, (C1_OH * C1_OW) as i32)?;
        cnn_kernels::bn_compute_mean(gpu, p, buf_conv1_bias, buf_dummy, buf_bn1_stats,
            b, C1_OC as i32, C1_OH as i32, C1_OW as i32)?;
        cnn_kernels::bn_compute_var(gpu, p, buf_conv1_bias, buf_bn1_stats, buf_bn1_stats,
            b, C1_OC as i32, C1_OH as i32, C1_OW as i32)?;
        cnn_kernels::bn_pack_params(gpu, p, buf_bn1_gb, buf_bn1_stats, buf_bn1_params,
            C1_OC as i32)?;
        cnn_kernels::bn_forward_apply(gpu, p, buf_conv1_bias, buf_bn1_params, buf_bn1_out,
            b, C1_OC as i32, C1_OH as i32, C1_OW as i32, BN_EPS)?;
        cnn_kernels::relu_forward(gpu, p, buf_bn1_out, buf_dummy, buf_relu1_out,
            (BATCH * C1_OC * C1_OH * C1_OW) as i32)?;
        cnn_kernels::maxpool2d_forward(gpu, p, buf_relu1_out, buf_pool1_mask, buf_pool1_out,
            b, C1_OC as i32, C1_OH as i32, C1_OW as i32,
            P1_OH as i32, P1_OW as i32, 2, 2, 2, 0)?;

        cnn_kernels::conv2d_forward(gpu, p, buf_pool1_out, buf_conv2_w, buf_conv2_out,
            b, C1_OC as i32, P1_OH as i32, P1_OW as i32,
            C2_OC as i32, C2_OH as i32, C2_OW as i32,
            C2_K as i32, C2_K as i32, 1, C2_PAD as i32)?;
        cnn_kernels::conv_bias_add(gpu, p, buf_conv2_out, buf_conv2_b, buf_conv2_bias,
            (BATCH * C2_OC * C2_OH * C2_OW) as i32, C2_OC as i32, (C2_OH * C2_OW) as i32)?;
        cnn_kernels::bn_compute_mean(gpu, p, buf_conv2_bias, buf_dummy, buf_bn2_stats,
            b, C2_OC as i32, C2_OH as i32, C2_OW as i32)?;
        cnn_kernels::bn_compute_var(gpu, p, buf_conv2_bias, buf_bn2_stats, buf_bn2_stats,
            b, C2_OC as i32, C2_OH as i32, C2_OW as i32)?;
        cnn_kernels::bn_pack_params(gpu, p, buf_bn2_gb, buf_bn2_stats, buf_bn2_params,
            C2_OC as i32)?;
        cnn_kernels::bn_forward_apply(gpu, p, buf_conv2_bias, buf_bn2_params, buf_bn2_out,
            b, C2_OC as i32, C2_OH as i32, C2_OW as i32, BN_EPS)?;
        cnn_kernels::relu_forward(gpu, p, buf_bn2_out, buf_dummy, buf_relu2_out,
            (BATCH * C2_OC * C2_OH * C2_OW) as i32)?;
        cnn_kernels::maxpool2d_forward(gpu, p, buf_relu2_out, buf_pool2_mask, buf_pool2_out,
            b, C2_OC as i32, C2_OH as i32, C2_OW as i32,
            P2_OH as i32, P2_OW as i32, 2, 2, 2, 0)?;

        cnn_kernels::matmul(gpu, p, buf_pool2_out, buf_fc_w, buf_fc_out,
            b, FC_IN as i32, CLASSES as i32)?;
        cnn_kernels::bias_add(gpu, p, buf_fc_out, buf_fc_b, buf_fc_z,
            (BATCH * CLASSES) as i32, CLASSES as i32)?;
        cnn_kernels::batched_softmax10(gpu, p, buf_fc_z, buf_dummy, buf_sm, b)?;

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
