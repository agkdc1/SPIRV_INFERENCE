#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void batchnorm_kernel(const float* x, const float* p, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = ((x[i] - p[0]) / sqrtf(p[1] + p[4])) * p[2] + p[3];
    }
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
    const int n = 256, pn = 5;
    char in_path[4096], p_path[4096], out_path[4096], json_path[4096];
    snprintf(in_path, sizeof(in_path), "%s/batchnorm_input.raw.f32", argv[1]);
    snprintf(p_path, sizeof(p_path), "%s/batchnorm_params.raw.f32", argv[1]);
    snprintf(out_path, sizeof(out_path), "%s/output.raw.f32", argv[2]);
    snprintf(json_path, sizeof(json_path), "%s/summary.json", argv[2]);
    float *h_x = (float*)malloc(n * sizeof(float)), *h_p = (float*)malloc(pn * sizeof(float)), *h_y = (float*)calloc(n, sizeof(float));
    if (!h_x || !h_p || !h_y || read_f32(in_path, h_x, n) || read_f32(p_path, h_p, pn)) return 3;
    float *d_x = NULL, *d_p = NULL, *d_y = NULL;
    cudaMalloc(&d_x, n * sizeof(float)); cudaMalloc(&d_p, pn * sizeof(float)); cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p, pn * sizeof(float), cudaMemcpyHostToDevice);
    batchnorm_kernel<<<1, 256>>>(d_x, d_p, d_y, n);
    cudaError_t status = cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (write_f32(out_path, h_y, n)) return 4;
    FILE* jf = fopen(json_path, "w");
    fprintf(jf, "{\"kernel\":\"batchnorm\",\"status\":\"%s\",\"cuda_status\":\"%s\",\"elements\":%d,\"output_fnv1a64\":\"%016llx\"}\n",
            status == cudaSuccess ? "pass" : "fail", cudaGetErrorString(status), n,
            (unsigned long long)fnv1a64((const unsigned char*)h_y, n * sizeof(float)));
    fclose(jf);
    cudaFree(d_x); cudaFree(d_p); cudaFree(d_y); free(h_x); free(h_p); free(h_y);
    return status == cudaSuccess ? 0 : 1;
}
