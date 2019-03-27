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
// (b, m) * [m, (b, n)] = (b, n) b,ty->block, tx->n, m==4
template <typename scalar_t>
__global__ void bmv4_kernel2(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B0, const scalar_t* __restrict__ B1, const scalar_t* __restrict__ B2, const scalar_t* __restrict__ B3, scalar_t* __restrict__ C, const int b, const int n, const int m) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int j=i*blockDim.y+ty; j<b; j+= gridDim.x * blockDim.y) {
        for (int x=tx; x<n; x+= blockDim.x) {
            C[j*n+x] = A[j*m] * B0[j*n+x] + A[j*m+1] * B1[j*n+x] + A[j*m+2] * B2[j*n+x] + A[j*m+3] * B3[j*n+x];
        }
    }
}

// (b, m) * [n, (b, m)] = (b, n) b,ty->block, tx->n, n==4
template <typename scalar_t>
__global__ void bmv4_kernel1(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B0, const scalar_t* __restrict__ B1, const scalar_t* __restrict__ B2, const scalar_t* __restrict__ B3, scalar_t* __restrict__ C, const int b, const int n, const int m) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int j=i*blockDim.y+ty; j<b; j+= gridDim.x * blockDim.y) {
        scalar_t sum = 0;
        if (tx==0) {
            for (int y=0; y<m; y++) {
                sum += B0[j*m+y] * A[j*m+y];
            }
        }
        if (tx==1) {
            for (int y=0; y<m; y++) {
                sum += B1[j*m+y] * A[j*m+y];
            }
        }
        if (tx==2) {
            for (int y=0; y<m; y++) {
                sum += B2[j*m+y] * A[j*m+y];
            }
        }
        if (tx==3) {
            for (int y=0; y<m; y++) {
                sum += B3[j*m+y] * A[j*m+y];
            }
        }
        C[j*n+tx] = sum;
    }
}

// (b, n, 1) * (b, 1, m) = (b, n, m) n>m
template <typename scalar_t>
__global__ void outer_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int b, const int n, const int m) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int j=i; j<b; j+= gridDim.x) {
        for (int x=tx; x<n; x+= blockDim.x){
            for (int y=ty; y<m; y+= blockDim.y) {
                C[(j*n+x)*m+y] = A[j*n+x] * B[j*m+y];
            }
        }
    }
}

// (b, m) * (b, m, n) = (b, n) b,ty-> block, tx->n
template <typename scalar_t>
__global__ void bmv_kernel1(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int b, const int n, const int m) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int j=i*blockDim.y+ty; j<b; j+= gridDim.x * blockDim.y) {
        for (int x=tx; x<n; x+= blockDim.x) {
            scalar_t sum = 0;
            for (int y=0; y<m; y++) {
                if (y<m) sum += B[(j*m+y)*n+x] * A[j*m+y];
            }
            C[j*n+x] = sum;
        }
    }
}

// (b, m) * (b, n, m) = (b, n) b,ty-> block, tx->n
template <typename scalar_t>
__global__ void bmv_kernel2(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int b, const int n, const int m) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int j=i*blockDim.y+ty; j<b; j+= gridDim.x * blockDim.y) {
        int i1 = j*n;
        int i3 = j*m;
        for (int x=tx; x<n; x+= blockDim.x) {
            scalar_t sum = 0;
            int i2 = (i1+x)*m;
            for (int y=0; y<m; y++) {
                if (y<m) sum += B[i2+y] * A[i3+y];
            }
            C[i1+x] = sum;
        }
    }
}

} // End of namespace

at::Tensor outer_cuda_forward(
    const at::Tensor& A,
    const at::Tensor& B) {
    // A: (b, n), B: (b, m)
    const auto b = B.size(0);
    const auto n = A.size(1);
    const auto m = B.size(1);

    auto y = at::zeros({b, n, m}, A.options());

    //const dim3 threads((n<(1024/m-1))?n:(1024/m-1) , m);
    const dim3 threads(8, m);
//    const dim3 blocks(b);
    const dim3 blocks(2048);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "outer_cuda_forward", ([&] {
        outer_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            y.data<scalar_t>(),
            b, n, m);
    }));
    THCudaCheck(cudaGetLastError());

    return y;
}

at::Tensor bmv4_cuda_forward2(
    const at::Tensor& A,
    const at::Tensor& B0,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& B3) {
    // A: (b, m), B: [m, (b, n)]
    const auto b = A.size(0);
    const auto n = B0.size(1);
    const auto m = 4;

    auto y = at::zeros({b, n}, A.options());

    const dim3 threads(n, (n>128)?1:8);
//    const dim3 blocks(b);
    const dim3 blocks(2048);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmv4_cuda_forward2", ([&] {
        bmv4_kernel2<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B0.data<scalar_t>(),
            B1.data<scalar_t>(),
            B2.data<scalar_t>(),
            B3.data<scalar_t>(),
            y.data<scalar_t>(),
            b, n, m);
    }));
    THCudaCheck(cudaGetLastError());

    return y;
}

at::Tensor bmv4_cuda_forward1(
    const at::Tensor& A,
    const at::Tensor& B0,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& B3) {
    // A: (b, m), B: [n, (b, m)]
    const auto b = A.size(0);
    const auto n = 4;
    const auto m = A.size(1);

    auto y = at::zeros({b, n}, A.options());

    const dim3 threads(n, (n>128)?1:8);
//    const dim3 blocks(b);
    const dim3 blocks(2048);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmv4_cuda_forward1", ([&] {
        bmv4_kernel1<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B0.data<scalar_t>(),
            B1.data<scalar_t>(),
            B2.data<scalar_t>(),
            B3.data<scalar_t>(),
            y.data<scalar_t>(),
            b, n, m);
    }));
    THCudaCheck(cudaGetLastError());

    return y;
}

at::Tensor bmv_cuda_forward2(
    const at::Tensor& A,
    const at::Tensor& B) {
    // A: (b, m), B: (b, n, m)
    const auto b = B.size(0);
    const auto n = B.size(1);
    const auto m = B.size(2);
    assert(m == A.size(1));

    auto y = at::zeros({b, n}, A.options());

    const dim3 threads(n, (n>128)?1:8);
//    const dim3 blocks(b);
    const dim3 blocks(2048);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmv_cuda_forward2", ([&] {
        bmv_kernel2<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            y.data<scalar_t>(),
            b, n, m);
    }));
    THCudaCheck(cudaGetLastError());

    return y;
}

at::Tensor bmv_cuda_forward1(
    const at::Tensor& A,
    const at::Tensor& B) {
    // A: (b, m), B: (b, m, n)
    const auto b = B.size(0);
    const auto m = B.size(1);
    const auto n = B.size(2);
    assert(m == A.size(1));

    auto y = at::zeros({b, n}, A.options());

    const dim3 threads(n, (n>128)?1:8);
//    const dim3 blocks(b);
    const dim3 blocks(2048);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmv_cuda_forward1", ([&] {
        bmv_kernel1<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            y.data<scalar_t>(),
            b, n, m);
    }));
    THCudaCheck(cudaGetLastError());

    return y;
}
