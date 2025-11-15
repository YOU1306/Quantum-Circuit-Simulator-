#include "common.h"
#include <cstdint>
#include <iostream>
#include <cassert>

__global__ void singleQubitKernel(Complex* state, int nQubits, int targetQubit, Complex u00, Complex u01, Complex u10, Complex u11) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long size = 1ULL << nQubits;
    unsigned long long half = size >> 1;
    if (idx >= half) return;

    unsigned long long pairDistance = 1ULL << targetQubit;
    unsigned long long blockSize = pairDistance << 1;
    unsigned long long blockStart = (idx / pairDistance) * blockSize;

    unsigned long long i0 = blockStart + (idx % pairDistance);
    unsigned long long i1 = i0 + pairDistance;

    Complex a = state[i0];
    Complex b = state[i1];

    state[i0] = addComplex(mulComplex(u00, a), mulComplex(u01, b));
    state[i1] = addComplex(mulComplex(u10, a), mulComplex(u11, b));
}

extern "C" void runSingleQubit(Complex* d_state, int nQubits, int targetQubit,
                               Complex u00, Complex u01, Complex u10, Complex u11) {
    assert(d_state != nullptr);
    assert(targetQubit < nQubits && nQubits < 64);
    const int threadsPerBlock = 256;
    unsigned long long needed = 1ULL << (nQubits - 1);
    unsigned long long blocksULL = (needed + threadsPerBlock - 1) / threadsPerBlock;
    int blocks = (int)blocksULL;
    singleQubitKernel<<<blocks, threadsPerBlock>>>(d_state, nQubits, targetQubit, u00, u01, u10, u11);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "runSingleQubit launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
}
