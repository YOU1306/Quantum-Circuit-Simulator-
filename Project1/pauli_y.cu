#include "common.h"
#include <cassert>

__global__ void pauliYKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int size = 1U << nQubits;
    if (idx >= size) return;

    unsigned int mask = 1U << targetQubit;
    unsigned int partner = idx ^ mask;

    if (idx < partner) {
        Complex a = state[idx];     // target=0
        Complex b = state[partner]; // target=1

        // -i*b = b.y - i*b.x
        Complex new_i0 = makeComplex(b.y, -b.x);
        // i*a = -a.y + i*a.x
        Complex new_i1 = makeComplex(-a.y, a.x);

        state[idx] = new_i0;
        state[partner] = new_i1;
    }
}

extern "C" void runPauliY(Complex* d_state, int nQubits, int targetQubit) {
    assert(d_state != nullptr && targetQubit < nQubits);
    const int threadsPerBlock = 256;
    unsigned int needed = 1U << nQubits;
    int blocks = (int)((needed + threadsPerBlock - 1) / threadsPerBlock);
    pauliYKernel<<<blocks, threadsPerBlock>>>(d_state, nQubits, targetQubit);
    cudaDeviceSynchronize();
}
