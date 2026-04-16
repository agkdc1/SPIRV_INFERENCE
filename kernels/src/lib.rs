pub const MOBILENET_KERNELS: &[&str] = &[
    "conv2d",
    "depthwise_conv2d",
    "relu6",
    "global_avg_pool_sum",
    "global_avg_pool_finalize",
    "transpose",
    "reshape",
    "add",
];

pub const TFLITE_KERNELS: &[&str] = &[
    "conv2d",
    "depthwise_conv2d",
    "relu6",
    "global_avg_pool_sum",
    "global_avg_pool_finalize",
    "transpose",
    "reshape",
    "add",
    "leaky_relu",
    "sigmoid",
    "tanh",
];
