use serde::Serialize;

#[derive(Serialize)]
pub struct EpochRecord {
    pub epoch: usize,
    pub train_loss: f32,
    pub test_accuracy: f32,
}

#[derive(Serialize)]
pub struct TrainingReport {
    pub model: String,
    pub architecture: String,
    pub optimizer: String,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub runtime: String,
    pub cuda_used: bool,
    pub epoch_results: Vec<EpochRecord>,
    pub final_test_accuracy: f32,
    pub final_train_loss: f32,
    pub total_training_time_secs: f64,
    pub spv_kernels: Vec<String>,
}
