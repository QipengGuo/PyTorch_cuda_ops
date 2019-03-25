#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Type.h>
#include <c10/util/Exception.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

/*
 * CUDA kernel of batched matrix multiplication:
 * (b, n, m) * (b, m, p)
 */

/*
template <typename scalar_t>
__global__ void bmm_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int b, const int n, const int m, const int p) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    for (int x = tx; x < n; x += blockDim.x) {
        for (int y = ty; y < p; y += blockDim.y) {
            scalar_t sum = 0;
            for (int k = 0; k < m; ++k) {
                sum +=  A[((i * n) + x) * m + k] * B[((i * m) + k) * p + y];
            }
            C[((i * n) + x) * p + y] = sum;
        } 
    }
}
*/

// (b, n, m) (b, m) = (b,n,p) b->block, n->tx, p->ty
template <typename scalar_t>
__global__ void bmv_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int n, const int m, const int p) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    scalar_t sum = 0;
    for (int x=ty; x<m; x += blockDim.y) {
	    sum += A[(i*n+tx)*m+x] * B[i*m+x];
    }
    C[((i*n)+tx)*blockDim.y+ty] = sum;
}

template <typename scalar_t>
__global__ void reduce(const scalar_t* __restrict__ A, scalar_t* __restrict__ B, const int n, const int p) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    scalar_t sum = 0;
    for (int x=0; x<p; x++){
	    sum += A[(i*n+tx)*p+x];
    }
    B[i*n+tx] = sum;
}

} // End of namespace

at::Tensor bmv_cuda_forward(
    const at::Tensor& A,
    const at::Tensor& B) {
    // A: (b, n, m), B: (b, m)
    const auto b = A.size(0);
    const auto n = A.size(1);
    const auto m = A.size(2);
    assert(m == B.size(1));

    auto y = at::zeros({b, n, 4}, A.options());
    auto C = at::zeros({b, n}, A.options());

    const dim3 threads(n, 4);
    const dim3 blocks(b);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmv_cuda_forward1", ([&] {
        bmv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            y.data<scalar_t>(),
            n, m, 4);
    }));
    THCudaCheck(cudaGetLastError());

    const dim3 threads1(n);
    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmv_cuda_forward2", ([&] {
        reduce<scalar_t><<<blocks, threads1, 0, stream>>>(
            y.data<scalar_t>(),
            C.data<scalar_t>(),
            n, 4);
    }));
    THCudaCheck(cudaGetLastError());
    return C;
}
