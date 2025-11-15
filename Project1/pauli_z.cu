#include "common.h"
#include <cassert>

__global__ void pauliZKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int size = 1U << nQubits;
    if (idx >= size) return;

    unsigned int mask = 1U << targetQubit;
    if ((idx & mask) != 0U) {
        state[idx].x = -state[idx].x;
        state[idx].y = -state[idx].y;
    }
}

extern "C" void runPauliZ(Complex* d_state, int nQubits, int targetQubit) {
    assert(d_state != nullptr && targetQubit < nQubits);
    const int threadsPerBlock = 256;
    unsigned int needed = 1U << nQubits;
    int blocks = (int)((needed + threadsPerBlock - 1) / threadsPerBlock);
    pauliZKernel<<<blocks, threadsPerBlock>>>(d_state, nQubits, targetQubit);
    cudaDeviceSynchronize();
}
