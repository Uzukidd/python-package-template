#include <torch/extension.h>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have same shape");

    auto c = torch::zeros_like(a);
    
    int threads = 256;
    int blocks = (a.numel() + threads - 1) / threads;
    
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        a.numel());
    
    return c;
}

