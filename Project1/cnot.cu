#include "common.h"
#include <cstdint>
#include <iostream>
#include <cassert>

__global__ void cnotKernel(Complex* state, int nQubits, int controlQubit, int targetQubit) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long size = 1ULL << nQubits;
    if (idx >= size) return;

    unsigned long long controlMask = 1ULL << controlQubit;
    unsigned long long targetMask  = 1ULL << targetQubit;

    if ((idx & controlMask) && ((idx & targetMask) == 0ULL)) {
        unsigned long long partner = idx | targetMask;
        Complex tmp = state[idx];
        state[idx] = state[partner];
        state[partner] = tmp;
    }
}

extern "C" void runCNOT(Complex* d_state, int nQubits, int controlQubit, int targetQubit) {
    assert(d_state != nullptr);
    assert(controlQubit < nQubits && targetQubit < nQubits);
    const int threadsPerBlock = 256;
    unsigned long long needed = 1ULL << nQubits;
    unsigned long long blocksULL = (needed + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksULL == 0) return;
    if (blocksULL > 0x7fffffffULL) {
        std::cerr << "runCNOT: too many blocks (nQubits too large)." << std::endl;
        return;
    }
    int blocks = (int)blocksULL;
    cnotKernel<<<blocks, threadsPerBlock>>>(d_state, nQubits, controlQubit, targetQubit);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "runCNOT launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
}
