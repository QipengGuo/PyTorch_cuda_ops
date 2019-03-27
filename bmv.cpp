#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor outer_cuda_forward(
	const at::Tensor& A,
	const at::Tensor& B);

at::Tensor outer_forward(
	const at::Tensor& A,
	const at::Tensor& B) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	return outer_cuda_forward(A, B);
}
at::Tensor bmv4_cuda_forward1(
	const at::Tensor& A,
	const at::Tensor& B0,
	const at::Tensor& B1,
	const at::Tensor& B2,
	const at::Tensor& B3);

at::Tensor bmv4_forward1(
	const at::Tensor& A,
	const at::Tensor& B0,
	const at::Tensor& B1,
	const at::Tensor& B2,
	const at::Tensor& B3) {
	CHECK_INPUT(A);
	CHECK_INPUT(B0);
	CHECK_INPUT(B1);
	CHECK_INPUT(B2);
	CHECK_INPUT(B3);
	return bmv4_cuda_forward1(A, B0, B1, B2, B3);
}

at::Tensor bmv4_cuda_forward2(
	const at::Tensor& A,
	const at::Tensor& B0,
	const at::Tensor& B1,
	const at::Tensor& B2,
	const at::Tensor& B3);

at::Tensor bmv4_forward2(
	const at::Tensor& A,
	const at::Tensor& B0,
	const at::Tensor& B1,
	const at::Tensor& B2,
	const at::Tensor& B3) {
	CHECK_INPUT(A);
	CHECK_INPUT(B0);
	CHECK_INPUT(B1);
	CHECK_INPUT(B2);
	CHECK_INPUT(B3);
	return bmv4_cuda_forward2(A, B0, B1, B2, B3);
}

at::Tensor bmv_cuda_forward2(
	const at::Tensor& A,
	const at::Tensor& B);

at::Tensor bmv_forward2(
	const at::Tensor& A,
	const at::Tensor& B) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	return bmv_cuda_forward2(A, B);
}

at::Tensor bmv_cuda_forward1(
	const at::Tensor& A,
	const at::Tensor& B);

at::Tensor bmv_forward1(
	const at::Tensor& A,
	const at::Tensor& B) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	return bmv_cuda_forward1(A, B);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmv_forward1", &bmv_forward1, "Batched Mv forward1");
    m.def("bmv_forward2", &bmv_forward2, "Batched Mv forward2");
    m.def("bmv4_forward1", &bmv4_forward1, "Batched Mv forward1 for 4 inputs");
    m.def("bmv4_forward2", &bmv4_forward2, "Batched Mv forward2 for 4 inputs");
    m.def("outer_forward", &outer_forward, "Outer Product forward");
}
