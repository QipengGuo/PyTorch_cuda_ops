#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor bmv_cuda_forward(
	const at::Tensor& A,
	const at::Tensor& B);

at::Tensor bmv_forward(
	const at::Tensor& A,
	const at::Tensor& B) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	return bmv_cuda_forward(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmv_forward", &bmv_forward, "Batched Mv forward");
}
