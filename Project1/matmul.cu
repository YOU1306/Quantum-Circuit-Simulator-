#include "common.h"
#include <cassert>

__global__ void matMulComplexKernel(Complex* A, Complex* B, Complex* C, int n) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < (unsigned int)n && col < (unsigned int)n) {
        Complex sum = makeComplex(0.0, 0.0);
        for (int k = 0; k < n; ++k) {
            sum = addComplex(sum, mulComplex(A[row * n + k], B[k * n + col]));
        }
        C[row * n + col] = sum;
    }
}

extern "C" void runMatMul(Complex* d_A, Complex* d_B, Complex* d_C, int n) {
    assert(d_A != nullptr && d_B != nullptr && d_C != nullptr && n > 0);
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    matMulComplexKernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
}
