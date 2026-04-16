#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

static constexpr int INPUT = 64;
static constexpr int CONV_WEIGHT = 12;
static constexpr int BN_PARAM = 16;
static constexpr int CONV_OUT = 24;
static constexpr int FC_PARAM = 250;
static constexpr int LOGITS = 10;

__global__ void conv1d_kernel(const float *input, const float *weight,
                              float *output) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= CONV_OUT)
    return;
  int row = i / 4;
  int oc = i % 4;
  float acc = 0.0f;
  for (int k = 0; k < 3; ++k) {
    acc += input[(row + k) * 8] * weight[k * 4 + oc];
  }
  output[i] = acc;
}

__global__ void batchnorm_kernel(const float *input, const float *param,
                                 float *output) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= CONV_OUT)
    return;
  int c = i % 4;
  float scale = param[c * 4 + 0];
  float bias = param[c * 4 + 1];
  float mean = param[c * 4 + 2];
  float var = param[c * 4 + 3];
  output[i] = (input[i] - mean) * rsqrtf(var + 0.00001f) * scale + bias;
}

__global__ void relu_kernel(const float *input, float *output) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < CONV_OUT)
    output[i] = fmaxf(input[i], 0.0f);
}

__global__ void fc_kernel(const float *input, const float *param,
                          float *output) {
  int cls = threadIdx.x;
  if (cls >= LOGITS)
    return;
  float acc = param[240 + cls];
  for (int i = 0; i < CONV_OUT; ++i) {
    acc += input[i] * param[i * LOGITS + cls];
  }
  output[cls] = acc;
}

__global__ void softmax_kernel(const float *input, float *output) {
  int i = threadIdx.x;
  if (i >= LOGITS)
    return;
  float m = input[0];
  for (int j = 1; j < LOGITS; ++j)
    m = fmaxf(m, input[j]);
  float sum = 0.0f;
  for (int j = 0; j < LOGITS; ++j)
    sum += expf(input[j] - m);
  output[i] = expf(input[i] - m) / sum;
}

static std::vector<float> read_f32(const std::string &path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "failed to open %s\n", path.c_str());
    std::exit(3);
  }
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  if (f.gcount() != static_cast<std::streamsize>(n * sizeof(float))) {
    std::fprintf(stderr, "short read from %s\n", path.c_str());
    std::exit(3);
  }
  return v;
}

static void write_f32(const std::string &path, const std::vector<float> &v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "failed to write %s\n", path.c_str());
    std::exit(4);
  }
  f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
}

static std::string arg_value(int argc, char **argv, const char *name,
                             const char *fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == name)
      return argv[i + 1];
  }
  return fallback;
}

int main(int argc, char **argv) {
  std::string manifest = arg_value(argc, argv, "--fixtures",
                                   "inference_poc/fixtures/manifest.json");
  std::string out_report =
      arg_value(argc, argv, "--out", "inference_poc/out/nvidia_cuda/report.json");
  std::string fixture_dir = manifest.substr(0, manifest.find_last_of('/'));
  std::string out_dir = out_report.substr(0, out_report.find_last_of('/'));
  std::string mkdir_cmd = "mkdir -p '" + out_dir + "'";
  if (std::system(mkdir_cmd.c_str()) != 0)
    return 5;

  auto input = read_f32(fixture_dir + "/input.raw.f32", INPUT);
  auto conv_weight = read_f32(fixture_dir + "/conv_weight.raw.f32", CONV_WEIGHT);
  auto bn_param = read_f32(fixture_dir + "/batchnorm_param.raw.f32", BN_PARAM);
  auto fc_param = read_f32(fixture_dir + "/fc_param.raw.f32", FC_PARAM);

  float *d_input = nullptr, *d_conv_weight = nullptr, *d_bn_param = nullptr,
        *d_fc_param = nullptr, *d_conv = nullptr, *d_bn = nullptr,
        *d_relu = nullptr, *d_logits = nullptr, *d_softmax = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, INPUT * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_conv_weight, CONV_WEIGHT * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_bn_param, BN_PARAM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_fc_param, FC_PARAM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_conv, CONV_OUT * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_bn, CONV_OUT * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_relu, CONV_OUT * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_logits, LOGITS * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_softmax, LOGITS * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_input, input.data(), INPUT * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_conv_weight, conv_weight.data(),
                        CONV_WEIGHT * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_bn_param, bn_param.data(), BN_PARAM * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_fc_param, fc_param.data(), FC_PARAM * sizeof(float),
                        cudaMemcpyHostToDevice));

  conv1d_kernel<<<1, 32>>>(d_input, d_conv_weight, d_conv);
  batchnorm_kernel<<<1, 32>>>(d_conv, d_bn_param, d_bn);
  relu_kernel<<<1, 32>>>(d_bn, d_relu);
  fc_kernel<<<1, 10>>>(d_relu, d_fc_param, d_logits);
  softmax_kernel<<<1, 10>>>(d_logits, d_softmax);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> conv(CONV_OUT), bn(CONV_OUT), relu(CONV_OUT),
      logits(LOGITS), softmax(LOGITS);
  CHECK_CUDA(cudaMemcpy(conv.data(), d_conv, CONV_OUT * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(bn.data(), d_bn, CONV_OUT * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(relu.data(), d_relu, CONV_OUT * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(logits.data(), d_logits, LOGITS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(softmax.data(), d_softmax, LOGITS * sizeof(float),
                        cudaMemcpyDeviceToHost));

  write_f32(out_dir + "/conv1d_output.raw.f32", conv);
  write_f32(out_dir + "/batchnorm_output.raw.f32", bn);
  write_f32(out_dir + "/relu_output.raw.f32", relu);
  write_f32(out_dir + "/logits_output.raw.f32", logits);
  write_f32(out_dir + "/softmax_output.raw.f32", softmax);
  write_f32(out_dir + "/output.raw.f32", softmax);

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::ofstream report(out_report);
  report << "{\n"
         << "  \"status\": \"pass\",\n"
         << "  \"baseline_kind\": \"cuda_custom_kernels\",\n"
         << "  \"device\": {\"name\": \"" << prop.name
         << "\", \"compute_capability\": \"" << prop.major << "." << prop.minor
         << "\"},\n"
         << "  \"graph\": [\"conv1d\", \"batchnorm\", \"relu\", \"fc\", "
            "\"softmax\"],\n"
         << "  \"output_path\": \"" << out_dir << "/output.raw.f32\",\n"
         << "  \"non_claims\": {\"uses_cublas\": false, \"uses_cudnn\": false, "
            "\"uses_tensorrt\": false, \"claims_tensorflow_runtime\": false, "
            "\"claims_full_tensorflow_gpu_support\": false}\n"
         << "}\n";

  cudaFree(d_softmax);
  cudaFree(d_logits);
  cudaFree(d_relu);
  cudaFree(d_bn);
  cudaFree(d_conv);
  cudaFree(d_fc_param);
  cudaFree(d_bn_param);
  cudaFree(d_conv_weight);
  cudaFree(d_input);
  return 0;
}
