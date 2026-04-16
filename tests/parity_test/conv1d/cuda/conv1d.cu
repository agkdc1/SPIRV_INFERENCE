#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void conv1d_kernel(const float* x, const float* w, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 512) return;
    int row = idx / 32;
    int oc = idx - row * 32;
    float acc = 0.0f;
    for (int kk = 0; kk < 3; ++kk) {
        int src_row = row + kk - 1;
        if (src_row < 0 || src_row >= 16) continue;
        for (int ic = 0; ic < 64; ++ic) {
            acc += x[src_row * 64 + ic] * w[(kk * 64 + ic) * 32 + oc];
        }
    }
    y[idx] = acc;
}

static uint64_t fnv1a64(const unsigned char* data, size_t len) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; ++i) { h ^= (uint64_t)data[i]; h *= 1099511628211ull; }
    return h;
}

static int read_f32(const char* path, float* data, size_t n) {
    FILE* f = fopen(path, "rb"); if (!f) return 1;
    size_t got = fread(data, sizeof(float), n, f); fclose(f); return got == n ? 0 : 1;
}

static int write_f32(const char* path, const float* data, size_t n) {
    FILE* f = fopen(path, "wb"); if (!f) return 1;
    size_t wrote = fwrite(data, sizeof(float), n, f); fclose(f); return wrote == n ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const int xn = 16 * 64, wn = 3 * 64 * 32, yn = 16 * 32;
    char in_path[4096], w_path[4096], out_path[4096], json_path[4096];
    snprintf(in_path, sizeof(in_path), "%s/conv1d_input.raw.f32", argv[1]);
    snprintf(w_path, sizeof(w_path), "%s/conv1d_filter.raw.f32", argv[1]);
    snprintf(out_path, sizeof(out_path), "%s/output.raw.f32", argv[2]);
    snprintf(json_path, sizeof(json_path), "%s/summary.json", argv[2]);
    float *h_x = (float*)malloc(xn * sizeof(float)), *h_w = (float*)malloc(wn * sizeof(float)), *h_y = (float*)calloc(yn, sizeof(float));
    if (!h_x || !h_w || !h_y || read_f32(in_path, h_x, xn) || read_f32(w_path, h_w, wn)) return 3;
    float *d_x = NULL, *d_w = NULL, *d_y = NULL;
    cudaMalloc(&d_x, xn * sizeof(float)); cudaMalloc(&d_w, wn * sizeof(float)); cudaMalloc(&d_y, yn * sizeof(float));
    cudaMemcpy(d_x, h_x, xn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, wn * sizeof(float), cudaMemcpyHostToDevice);
    conv1d_kernel<<<4, 128>>>(d_x, d_w, d_y);
    cudaError_t status = cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, yn * sizeof(float), cudaMemcpyDeviceToHost);
    if (write_f32(out_path, h_y, yn)) return 4;
    FILE* jf = fopen(json_path, "w");
    fprintf(jf, "{\"kernel\":\"conv1d\",\"status\":\"%s\",\"cuda_status\":\"%s\",\"elements\":%d,\"output_fnv1a64\":\"%016llx\"}\n",
            status == cudaSuccess ? "pass" : "fail", cudaGetErrorString(status), yn,
            (unsigned long long)fnv1a64((const unsigned char*)h_y, yn * sizeof(float)));
    fclose(jf);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_y); free(h_x); free(h_w); free(h_y);
    return status == cudaSuccess ? 0 : 1;
}
