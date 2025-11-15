#include "common.h"
#include <cassert>

__global__ void pauliXKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int size = 1U << nQubits;
    if (idx >= size) return;

    unsigned int mask = 1U << targetQubit;
    unsigned int partner = idx ^ mask;

    if (idx < partner) {
        Complex tmp = state[idx];
        state[idx] = state[partner];
        state[partner] = tmp;
    }
}

extern "C" void runPauliX(Complex* d_state, int nQubits, int targetQubit) {
    assert(d_state != nullptr && targetQubit < nQubits);
    const int threadsPerBlock = 256;
    unsigned int needed = 1U << nQubits;
    int blocks = (int)((needed + threadsPerBlock - 1) / threadsPerBlock);
    pauliXKernel<<<blocks, threadsPerBlock>>>(d_state, nQubits, targetQubit);
    cudaDeviceSynchronize();
}
