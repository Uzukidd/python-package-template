#include <torch/extension.h>

__device__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have same shape");

    auto c = torch::zeros_like(a);
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    int threads = 256;
    int blocks = (a.numel() + threads - 1) / threads;
    
    vector_add_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        a.numel());
    
    return c;
}

