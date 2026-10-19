// Experimental FP32 extension: GPU forward and custom backward.
// Compilation and performance MUST be validated on each target GPU.
#include <torch/extension.h>
#include <vector>
torch::Tensor pcmn_scan_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor initial);
std::vector<torch::Tensor> pcmn_scan_backward_cuda(torch::Tensor a, torch::Tensor output,
                                        torch::Tensor initial, torch::Tensor grad_output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan", &pcmn_scan_cuda, "PCMN CUDA diagonal scan (FP32)");
    m.def("scan_backward", &pcmn_scan_backward_cuda, "PCMN CUDA analytic gradient (FP32)");
}
