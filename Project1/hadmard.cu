#include "common.h"
#include <cmath>
#include <cassert>

__global__ void hadamardKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int size = 1U << nQubits;
    if (idx >= size / 2) return;

    unsigned int pairDistance = 1U << targetQubit;
    unsigned int blockSize = pairDistance * 2U;
    unsigned int blockStart = (idx / pairDistance) * blockSize;

    unsigned int i0 = blockStart + (idx % pairDistance);
    unsigned int i1 = i0 + pairDistance;

    Complex a = state[i0];
    Complex b = state[i1];

    double invSqrt2 = 1.0 / sqrt(2.0);
    state[i0] = scaleComplex(addComplex(a, b), invSqrt2);
    state[i1] = scaleComplex(subComplex(a, b), invSqrt2);
}

extern "C" void runHadamard(Complex* d_state, int nQubits, int targetQubit) {
    assert(d_state != nullptr && targetQubit < nQubits);
    const int threadsPerBlock = 256;
    unsigned int needed = 1U << (nQubits - 1);
    int blocks = (int)((needed + threadsPerBlock - 1) / threadsPerBlock);
    hadamardKernel<<<blocks, threadsPerBlock>>>(d_state, nQubits, targetQubit);
    cudaDeviceSynchronize();
}
