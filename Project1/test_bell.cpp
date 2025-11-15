// test_bell.cpp
#include "common.h"
#include <vector>
#include <iostream>
#include <cmath>

extern "C" void runSingleQubit(Complex* d_state, int nQubits, int targetQubit,
    Complex u00, Complex u01, Complex u10, Complex u11);
extern "C" void runCNOT(Complex* d_state, int nQubits, int controlQubit, int targetQubit);
extern "C" void computeProbabilities(Complex* d_state, int nQubits, double* h_probs);

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1);} } while(0)

int main() {
    const int nQubits = 2;
    const size_t numStates = 1ULL << nQubits;

    // initialize |00>
    std::vector<Complex> h_state(numStates, makeComplex(0.0, 0.0));
    h_state[0] = makeComplex(1.0, 0.0);

    Complex* d_state = nullptr;
    CUDA_CHECK(cudaMalloc(&d_state, numStates * sizeof(Complex)));
    CUDA_CHECK(cudaMemcpy(d_state, h_state.data(), numStates * sizeof(Complex), cudaMemcpyHostToDevice));

    // Apply H on qubit 0 using runSingleQubit
    double invS = 1.0 / std::sqrt(2.0);
    Complex u00 = makeComplex(invS, 0.0), u01 = makeComplex(invS, 0.0),
        u10 = makeComplex(invS, 0.0), u11 = makeComplex(-invS, 0.0);

    runSingleQubit(d_state, nQubits, 0, u00, u01, u10, u11);

    // CNOT 0 -> 1
    runCNOT(d_state, nQubits, 0, 1);

    // compute probs
    double probs[4] = { 0 };
    computeProbabilities(d_state, nQubits, probs);

    std::cout << "Bell state probabilities (expected ~0.5 at |00> and |11>):\n";
    for (size_t i = 0; i < numStates; ++i) {
        std::cout << "|" << i << "> : " << probs[i] << "\n";
    }

    CUDA_CHECK(cudaFree(d_state));
    return 0;
}
