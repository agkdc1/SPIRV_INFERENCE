mod vulkan_ctx;
mod kernels;
mod mnist;
mod report;

use anyhow::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::PathBuf;
use std::time::Instant;

const BATCH: usize = 64;
const EPOCHS: usize = 5;
const LR: f32 = 0.001;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const INPUT: usize = 784;
const HIDDEN: usize = 128;
const CLASSES: usize = 10;

fn main() -> Result<()> {
    let start = Instant::now();
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");

    // 1. Load MNIST
    let mnist = mnist::load(&data_dir)?;

    // 2. Init Vulkan
    let mut gpu = vulkan_ctx::VulkanTrainer::new("nvidia")?;
    let pipes = kernels::Pipelines::load(&mut gpu)?;
    println!("[init] 11 pipelines loaded");

    // 3. Allocate GPU buffers
    let buf_images = gpu.alloc(BATCH * INPUT)?;    // batch input
    let buf_labels = gpu.alloc(BATCH * CLASSES)?;   // one-hot labels
    let buf_w1     = gpu.alloc(INPUT * HIDDEN)?;    // weights layer 1
    let buf_b1     = gpu.alloc(HIDDEN)?;
    let buf_w2     = gpu.alloc(HIDDEN * CLASSES)?;   // weights layer 2
    let buf_b2     = gpu.alloc(CLASSES)?;
    let buf_temp   = gpu.alloc(BATCH * INPUT)?;     // scratch (max size needed)
    let buf_z1     = gpu.alloc(BATCH * HIDDEN)?;    // pre-relu activations
    let buf_a1     = gpu.alloc(BATCH * HIDDEN)?;    // post-relu activations
    let buf_z2     = gpu.alloc(BATCH * CLASSES)?;   // pre-softmax
    let buf_sm     = gpu.alloc(BATCH * CLASSES)?;   // softmax output
    let buf_loss   = gpu.alloc(1)?;                  // scalar loss
    let buf_dz2    = gpu.alloc(BATCH * CLASSES)?;   // grad pre-softmax
    let buf_dw2    = gpu.alloc(HIDDEN * CLASSES)?;
    let buf_db2    = gpu.alloc(CLASSES)?;
    let buf_da1    = gpu.alloc(BATCH * HIDDEN)?;
    let buf_dz1    = gpu.alloc(BATCH * HIDDEN)?;
    let buf_dw1    = gpu.alloc(INPUT * HIDDEN)?;
    let buf_db1    = gpu.alloc(HIDDEN)?;
    let buf_dummy  = gpu.alloc(1)?;                  // unused binding
    // Adam state: m and v packed [m..., v...]
    let buf_mv_w1  = gpu.alloc(2 * INPUT * HIDDEN)?;
    let buf_mv_b1  = gpu.alloc(2 * HIDDEN)?;
    let buf_mv_w2  = gpu.alloc(2 * HIDDEN * CLASSES)?;
    let buf_mv_b2  = gpu.alloc(2 * CLASSES)?;
    println!("[init] {} GPU buffers allocated", 24);

    // 4. Initialize weights (Kaiming He)
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let w1 = kaiming_init(INPUT, HIDDEN, &mut rng);
    let b1 = vec![0.0_f32; HIDDEN];
    let w2 = kaiming_init(HIDDEN, CLASSES, &mut rng);
    let b2 = vec![0.0_f32; CLASSES];
    gpu.upload(buf_w1, &w1)?;
    gpu.upload(buf_b1, &b1)?;
    gpu.upload(buf_w2, &w2)?;
    gpu.upload(buf_b2, &b2)?;
    println!("[init] weights uploaded (Kaiming He, seed=42)");

    // 5. Training loop
    let num_train = mnist.train_labels.len();
    let batches_per_epoch = num_train / BATCH;
    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut epoch_results = Vec::new();
    let mut adam_t: u32 = 0;

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
            let mut batch_img = vec![0.0_f32; BATCH * INPUT];
            let mut batch_lbl = vec![0_u8; BATCH];
            for i in 0..BATCH {
                let idx = indices[batch_idx * BATCH + i];
                let src = &mnist.train_images[idx * INPUT..(idx + 1) * INPUT];
                batch_img[i * INPUT..(i + 1) * INPUT].copy_from_slice(src);
                batch_lbl[i] = mnist.train_labels[idx];
            }
            let batch_onehot = mnist::one_hot(&batch_lbl, CLASSES);
            gpu.upload(buf_images, &batch_img)?;
            gpu.upload(buf_labels, &batch_onehot)?;

            // === FORWARD ===
            // z1_raw = images @ W1
            kernels::matmul(&gpu, &pipes, buf_images, buf_w1, buf_temp, BATCH as i32, INPUT as i32, HIDDEN as i32)?;
            // z1 = z1_raw + b1
            kernels::bias_add(&gpu, &pipes, buf_temp, buf_b1, buf_z1, (BATCH * HIDDEN) as i32, HIDDEN as i32)?;
            // a1 = relu(z1)
            kernels::relu_forward(&gpu, &pipes, buf_z1, buf_dummy, buf_a1, (BATCH * HIDDEN) as i32)?;
            // z2_raw = a1 @ W2
            kernels::matmul(&gpu, &pipes, buf_a1, buf_w2, buf_temp, BATCH as i32, HIDDEN as i32, CLASSES as i32)?;
            // z2 = z2_raw + b2
            kernels::bias_add(&gpu, &pipes, buf_temp, buf_b2, buf_z2, (BATCH * CLASSES) as i32, CLASSES as i32)?;
            // sm = softmax(z2)
            kernels::batched_softmax10(&gpu, &pipes, buf_z2, buf_dummy, buf_sm, BATCH as i32)?;
            // loss = cross_entropy(sm, labels)
            kernels::cross_entropy_loss(&gpu, &pipes, buf_sm, buf_labels, buf_loss, BATCH as i32, CLASSES as i32)?;

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

            // === BACKWARD ===
            // dz2 = (sm - labels) / batch_size
            kernels::softmax_ce_backward(&gpu, &pipes, buf_sm, buf_labels, buf_dz2, (BATCH * CLASSES) as i32, BATCH as i32)?;
            // dW2 = a1^T @ dz2
            kernels::matmul_tn(&gpu, &pipes, buf_a1, buf_dz2, buf_dw2, HIDDEN as i32, BATCH as i32, CLASSES as i32)?;
            // db2 = sum_rows(dz2)
            kernels::reduce_sum_rows(&gpu, &pipes, buf_dz2, buf_dummy, buf_db2, BATCH as i32, CLASSES as i32)?;
            // da1 = dz2 @ W2^T
            kernels::matmul_nt(&gpu, &pipes, buf_dz2, buf_w2, buf_da1, BATCH as i32, CLASSES as i32, HIDDEN as i32)?;
            // dz1 = da1 * relu'(z1)
            kernels::relu_backward(&gpu, &pipes, buf_da1, buf_z1, buf_dz1, (BATCH * HIDDEN) as i32)?;
            // dW1 = images^T @ dz1
            kernels::matmul_tn(&gpu, &pipes, buf_images, buf_dz1, buf_dw1, INPUT as i32, BATCH as i32, HIDDEN as i32)?;
            // db1 = sum_rows(dz1)
            kernels::reduce_sum_rows(&gpu, &pipes, buf_dz1, buf_dummy, buf_db1, BATCH as i32, HIDDEN as i32)?;

            // === ADAM ===
            kernels::adam_step(&gpu, &pipes, buf_w1, buf_dw1, buf_mv_w1, (INPUT * HIDDEN) as i32, LR, BETA1, BETA2, EPS, beta1_t, beta2_t)?;
            kernels::adam_step(&gpu, &pipes, buf_b1, buf_db1, buf_mv_b1, HIDDEN as i32, LR, BETA1, BETA2, EPS, beta1_t, beta2_t)?;
            kernels::adam_step(&gpu, &pipes, buf_w2, buf_dw2, buf_mv_w2, (HIDDEN * CLASSES) as i32, LR, BETA1, BETA2, EPS, beta1_t, beta2_t)?;
            kernels::adam_step(&gpu, &pipes, buf_b2, buf_db2, buf_mv_b2, CLASSES as i32, LR, BETA1, BETA2, EPS, beta1_t, beta2_t)?;
        }

        let avg_loss = (total_loss / loss_count as f64) as f32;

        // === TEST ACCURACY ===
        let test_acc = evaluate(&gpu, &pipes, &mnist.test_images, &mnist.test_labels,
            buf_images, buf_temp, buf_z1, buf_a1, buf_z2, buf_sm,
            buf_w1, buf_b1, buf_w2, buf_b2, buf_dummy)?;

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
    println!("TRAINING COMPLETE");
    println!("Final test accuracy: {:.2}%", final_acc * 100.0);
    println!("Final train loss:    {:.4}", final_loss);
    println!("Total time:          {:.1}s", total_secs);
    println!("Device:              {}", gpu.device_name);
    println!("CUDA used:           false");
    println!("========================================\n");

    // 6. Write report
    let rpt = report::TrainingReport {
        model: "MLP 784->128->10".into(),
        architecture: "Linear(784,128) -> ReLU -> Linear(128,10) -> Softmax+CE".into(),
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
            "matmul.spv".into(),
            "matmul_tn.spv".into(),
            "matmul_nt.spv".into(),
            "relu_forward.spv".into(),
            "relu_backward.spv".into(),
            "bias_add.spv".into(),
            "batched_softmax10.spv".into(),
            "cross_entropy_loss.spv".into(),
            "softmax_ce_backward.spv".into(),
            "reduce_sum_rows.spv".into(),
            "adam_step.spv".into(),
        ],
    };

    let json_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("TRAINING_RESULT.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&rpt)?)?;
    println!("[report] {}", json_path.display());

    Ok(())
}

fn kaiming_init(fan_in: usize, fan_out: usize, rng: &mut impl rand::Rng) -> Vec<f32> {
    let std = (2.0 / fan_in as f64).sqrt() as f32;
    (0..fan_in * fan_out)
        .map(|_| {
            // Box-Muller for normal distribution
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z * std
        })
        .collect()
}

fn evaluate(
    gpu: &vulkan_ctx::VulkanTrainer,
    pipes: &kernels::Pipelines,
    images: &[f32],
    labels: &[u8],
    buf_img: vulkan_ctx::Buf,
    buf_temp: vulkan_ctx::Buf,
    buf_z1: vulkan_ctx::Buf,
    buf_a1: vulkan_ctx::Buf,
    buf_z2: vulkan_ctx::Buf,
    buf_sm: vulkan_ctx::Buf,
    buf_w1: vulkan_ctx::Buf,
    buf_b1: vulkan_ctx::Buf,
    buf_w2: vulkan_ctx::Buf,
    buf_b2: vulkan_ctx::Buf,
    buf_dummy: vulkan_ctx::Buf,
) -> Result<f32> {
    let n = labels.len();
    let num_batches = n / BATCH;
    let mut correct = 0_usize;
    let mut total = 0_usize;

    for b in 0..num_batches {
        let start = b * BATCH;
        let batch_img = &images[start * INPUT..(start + BATCH) * INPUT];
        gpu.upload(buf_img, batch_img)?;

        // Forward only
        kernels::matmul(gpu, pipes, buf_img, buf_w1, buf_temp, BATCH as i32, INPUT as i32, HIDDEN as i32)?;
        kernels::bias_add(gpu, pipes, buf_temp, buf_b1, buf_z1, (BATCH * HIDDEN) as i32, HIDDEN as i32)?;
        kernels::relu_forward(gpu, pipes, buf_z1, buf_dummy, buf_a1, (BATCH * HIDDEN) as i32)?;
        kernels::matmul(gpu, pipes, buf_a1, buf_w2, buf_temp, BATCH as i32, HIDDEN as i32, CLASSES as i32)?;
        kernels::bias_add(gpu, pipes, buf_temp, buf_b2, buf_z2, (BATCH * CLASSES) as i32, CLASSES as i32)?;
        kernels::batched_softmax10(gpu, pipes, buf_z2, buf_dummy, buf_sm, BATCH as i32)?;

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
