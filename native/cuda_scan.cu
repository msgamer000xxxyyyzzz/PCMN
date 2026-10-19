#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// Parallel across batch * feature. Each individual time-axis recurrence is
// sequential. This is NOT a work-efficient parallel-prefix kernel.
__global__ void pcmn_diagonal_scan(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    const float* __restrict__ h0,
                                    float* __restrict__ out,
                                    int64_t total, int64_t seq, int64_t dim) {
    int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    const int64_t batch = id / dim;
    const int64_t feature = id % dim;
    float state = h0 ? h0[id] : 0.f;
    for (int64_t t = 0; t < seq; t++) {
        int64_t offset = (batch * seq + t) * dim + feature;
        state = a[offset] * state + b[offset];
        out[offset] = state;
    }
}

__global__ void pcmn_diagonal_scan_backward(const float* __restrict__ a,
                                             const float* __restrict__ h,
                                             const float* __restrict__ initial,
                                             const float* __restrict__ gout,
                                             float* __restrict__ da,
                                             float* __restrict__ db,
                                             float* __restrict__ dh0,
                                             int64_t total, int64_t seq, int64_t dim) {
    const int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    const int64_t batch = id / dim;
    const int64_t feature = id % dim;
    float adj = 0.f;
    for (int64_t t = seq; t-- > 0;) {
        const int64_t off = (batch * seq + t) * dim + feature;
        // dL/dh[t] includes future through transition at t+1.
        adj = gout[off] + (t + 1 < seq ? a[off + dim] * adj : 0.f);
        const float previous = t == 0 ? (initial ? initial[id] : 0.f) : h[off - dim];
        da[off] = adj * previous;
        db[off] = adj;
    }
    if (dh0) dh0[id] = a[(batch*seq)*dim + feature] * adj;
}

static void validate(torch::Tensor a, torch::Tensor b, torch::Tensor initial) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(a.device() == b.device(), "Inputs must be on same GPU");
    TORCH_CHECK(a.dim() == 3 && a.size(1) > 0 && b.sizes() == a.sizes(), "[B,T,D] required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "Only FP32 supported");
    TORCH_CHECK(initial.numel() == 0 || (initial.dim() == 2 &&
        initial.size(0) == a.size(0) && initial.size(1) == a.size(2) &&
        initial.scalar_type() == torch::kFloat32 && initial.is_cuda() &&
        initial.device() == a.device()), "Initial state must be empty or CUDA [B,D] FP32");
}

torch::Tensor pcmn_scan_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor initial) {
    validate(a,b,initial);
    auto A = a.contiguous(), B = b.contiguous(), H = initial.contiguous();
    auto out = torch::empty_like(B);
    const int64_t n = a.size(0) * a.size(2);
    constexpr int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    pcmn_diagonal_scan<<<(n+threads-1)/threads, threads, 0, stream.stream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), H.numel() ? H.data_ptr<float>() : nullptr,
        out.data_ptr<float>(), n, a.size(1), a.size(2));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA scan forward launch failed");
    return out;
}

std::vector<torch::Tensor> pcmn_scan_backward_cuda(torch::Tensor a, torch::Tensor output,
                                        torch::Tensor initial, torch::Tensor grad_output) {
    validate(a, grad_output, initial);
    TORCH_CHECK(output.sizes()==a.sizes() && output.is_cuda() && output.device()==a.device() &&
                output.scalar_type()==torch::kFloat32, "Invalid saved state");
    auto A=a.contiguous(), H=output.contiguous(), G=grad_output.contiguous(), I=initial.contiguous();
    auto ga=torch::empty_like(A), gb=torch::empty_like(G);
    auto gi=initial.numel() ? torch::empty_like(initial) : torch::empty_like(initial);
    const int64_t n=a.size(0)*a.size(2);
    constexpr int threads=256;
    auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
    pcmn_diagonal_scan_backward<<<(n+threads-1)/threads,threads,0,stream.stream()>>>(
        A.data_ptr<float>(), H.data_ptr<float>(), I.numel()?I.data_ptr<float>():nullptr,
        G.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(),
        gi.numel()?gi.data_ptr<float>():nullptr,n,a.size(1),a.size(2));
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"CUDA scan backward launch failed");
    return {ga,gb,gi};
}
