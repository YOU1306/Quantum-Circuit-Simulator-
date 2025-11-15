#include "common.h"
#include <vector>
#include <iostream>
#include <cassert>

extern "C" void computeProbabilities(Complex* d_state, int nQubits, double* h_probs) {
    assert(d_state != nullptr);
    size_t size = 1ULL << nQubits;
    std::vector<Complex> h_state(size);
    cudaError_t err = cudaMemcpy(h_state.data(), d_state, size * sizeof(Complex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "computeProbabilities cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    for (size_t i = 0; i < size; ++i) {
        double re = h_state[i].x;
        double im = h_state[i].y;
        h_probs[i] = re*re + im*im;
    }
}
