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


// (b, n, m) (b, m) where n>>m ,b-> block, tx->n
template <typename scalar_t>
__global__ void bmv_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int n, const int m) {
//    __shared__ scalar_t la[5];
    int i = blockIdx.x;
    int tx = threadIdx.x;
//    if (tx<m) la[tx] = B[i*m+tx];
//    __syncthreads();

    int i1 = i*n;
    for (int x=tx; x<n; x+= blockDim.x) {
//        scalar_t sum = 0;
        int i2 = (i1+x)*m;
        int i3 = i*m;
//        for (int y=0; y<m; y++) {
//            sum += A[(i*n+x)*m+y] * la[y];
//            sum += A[i2+y] * B[i3+y];
//        }
        C[i1+x] = A[i2] * B[i3] + A[i2+1] * B[i3+1] + A[i2+2] * B[i3+2] + A[i2+3] * B[i3+3] + A[i2+4] * B[i3+4];
//        C[i1+x] = sum;
    }
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

    auto y = at::zeros({b, n}, A.options());

    const dim3 threads(n);
    const dim3 blocks(b);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmv_cuda_forward1", ([&] {
        bmv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            y.data<scalar_t>(),
            n, m);
    }));
    THCudaCheck(cudaGetLastError());

    return y;
}
