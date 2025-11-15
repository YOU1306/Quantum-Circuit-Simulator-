#include "common.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

extern "C" void runHadamard(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runPauliX(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runPauliY(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runPauliZ(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runMatMul(Complex* d_A, Complex* d_B, Complex* d_C, int n);

extern "C" void runSingleQubit(Complex* d_state, int nQubits, int targetQubit,
    Complex u00, Complex u01, Complex u10, Complex u11);
extern "C" void runCNOT(Complex* d_state, int nQubits, int controlQubit, int targetQubit);

extern "C" void computeProbabilities(Complex* d_state, int nQubits, double* h_probs);


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


static void printStateHost(const std::vector<Complex>& state, size_t maxPrint = 32, double eps = 1e-12) {
    size_t sz = state.size();
    for (size_t i = 0; i < std::min(sz, maxPrint); ++i) {
        double rx = (std::fabs(state[i].x) < eps) ? 0.0 : state[i].x;
        double ry = (std::fabs(state[i].y) < eps) ? 0.0 : state[i].y;
        double prob = rx * rx + ry * ry;
        std::cout << "|" << i << "> = (" << rx << "," << ry << "i)  prob=" << prob << "\n";
    }
    if (sz > maxPrint) {
        std::cout << "... (" << (sz - maxPrint) << " more states not shown)\n";
    }
}

int main() {
  
    {
        const int nQubits = 9;
        const size_t numStates = 1ULL << nQubits;

        std::vector<Complex> h_state(numStates, makeComplex(0.0, 0.0));
        h_state[0] = makeComplex(1.0, 0.0);

        Complex* d_state = nullptr;
        CUDA_CHECK(cudaMalloc(&d_state, numStates * sizeof(Complex)));
        CUDA_CHECK(cudaMemcpy(d_state, h_state.data(), numStates * sizeof(Complex), cudaMemcpyHostToDevice));

        std::cout << "Applying Hadamard to all " << nQubits << " qubits to create uniform superposition...\n";
        for (int q = 0; q < nQubits; ++q) {
            runHadamard(d_state, nQubits, q);
        }

        CUDA_CHECK(cudaMemcpy(h_state.data(), d_state, numStates * sizeof(Complex), cudaMemcpyDeviceToHost));

        std::cout << "\n--- Quantum State: first 64 amplitudes (uniform superposition) ---\n";
        printStateHost(h_state, 64);

        double sum = 0.0;
        for (const auto& c : h_state) sum += c.x * c.x + c.y * c.y;
        std::cout << "\nNormalization sum = " << sum << "\n";

        double expected_amp = 1.0 / std::sqrt((double)numStates);
        std::cout << "Expected amplitude (each state) = " << expected_amp
            << " (probability per state = " << expected_amp * expected_amp << ")\n";

        CUDA_CHECK(cudaFree(d_state));
    }

    
    {
        int n = 2;
        Complex h_A[4] = { makeComplex(1,0), makeComplex(2,0),
                           makeComplex(3,0), makeComplex(4,0) };
        Complex h_B[4] = { makeComplex(5,0), makeComplex(6,0),
                           makeComplex(7,0), makeComplex(8,0) };
        Complex h_C[4] = { makeComplex(0,0), makeComplex(0,0), makeComplex(0,0), makeComplex(0,0) };

        Complex* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, 4 * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&d_B, 4 * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&d_C, 4 * sizeof(Complex)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, 4 * sizeof(Complex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, 4 * sizeof(Complex), cudaMemcpyHostToDevice));
       
        runMatMul(d_A, d_B, d_C, n);

        CUDA_CHECK(cudaMemcpy(h_C, d_C, 4 * sizeof(Complex), cudaMemcpyDeviceToHost));
        std::cout << "\n--- Matrix Multiplication Result (2x2) ---\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << "(" << h_C[i * n + j].x << "," << h_C[i * n + j].y << "i) ";
            }
            std::cout << "\n";
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    
    {
        const int nQ = 2;
        const size_t N = 1ULL << nQ;
        std::vector<Complex> h_small(N, makeComplex(0.0, 0.0));
        h_small[0] = makeComplex(1.0, 0.0);

        Complex* d_small = nullptr;
        CUDA_CHECK(cudaMalloc(&d_small, N * sizeof(Complex)));
        CUDA_CHECK(cudaMemcpy(d_small, h_small.data(), N * sizeof(Complex), cudaMemcpyHostToDevice));

        double invS = 1.0 / std::sqrt(2.0);
        Complex u00 = makeComplex(invS, 0.0), u01 = makeComplex(invS, 0.0),
            u10 = makeComplex(invS, 0.0), u11 = makeComplex(-invS, 0.0);

        std::cout << "\n--- Bell test: runSingleQubit(H on q0) then CNOT(0->1) ---\n";
        runSingleQubit(d_small, nQ, 0, u00, u01, u10, u11);
        runCNOT(d_small, nQ, 0, 1);

        double probs[4] = { 0.0,0.0,0.0,0.0 };
        computeProbabilities(d_small, nQ, probs);

        std::cout << "Bell state probabilities (expected ~0.5 at |00> and |11>):\n";
        for (size_t i = 0; i < N; i++) std::cout << "|" << i << "> : " << probs[i] << "\n";

        CUDA_CHECK(cudaFree(d_small));
    }

   
    {
        
        const int nQ = 1;
        const size_t N = 1ULL << nQ;
        std::vector<Complex> h_one(N);
        Complex* d_one = nullptr;
        CUDA_CHECK(cudaMalloc(&d_one, N * sizeof(Complex)));

        
        h_one.assign(N, makeComplex(0.0, 0.0));
        h_one[0] = makeComplex(1.0, 0.0);
        CUDA_CHECK(cudaMemcpy(d_one, h_one.data(), N * sizeof(Complex), cudaMemcpyHostToDevice));
        runPauliX(d_one, nQ, 0);
        CUDA_CHECK(cudaMemcpy(h_one.data(), d_one, N * sizeof(Complex), cudaMemcpyDeviceToHost));
        std::cout << "\nPauli-X applied to |0> -> amplitudes:\n";
        printStateHost(h_one, 4);

        
        h_one.assign(N, makeComplex(0.0, 0.0));
        h_one[0] = makeComplex(1.0, 0.0);
        CUDA_CHECK(cudaMemcpy(d_one, h_one.data(), N * sizeof(Complex), cudaMemcpyHostToDevice));
        runPauliY(d_one, nQ, 0);
        CUDA_CHECK(cudaMemcpy(h_one.data(), d_one, N * sizeof(Complex), cudaMemcpyDeviceToHost));
        std::cout << "\nPauli-Y applied to |0> -> amplitudes:\n";
        printStateHost(h_one, 4);

        CUDA_CHECK(cudaFree(d_one));
    }

    
    {
        const int nQ = 2;
        const size_t N = 1ULL << nQ;
        std::vector<Complex> h_two(N, makeComplex(0.0, 0.0));
        h_two[1] = makeComplex(1.0, 0.0); // |01>

        Complex* d_two = nullptr;
        CUDA_CHECK(cudaMalloc(&d_two, N * sizeof(Complex)));
        CUDA_CHECK(cudaMemcpy(d_two, h_two.data(), N * sizeof(Complex), cudaMemcpyHostToDevice));

        std::cout << "\nPauli-Z applied to target qubit 0 on state |01>:\n";
        runPauliZ(d_two, nQ, 0);

        CUDA_CHECK(cudaMemcpy(h_two.data(), d_two, N * sizeof(Complex), cudaMemcpyDeviceToHost));
        printStateHost(h_two, 8);

        CUDA_CHECK(cudaFree(d_two));
    }

    std::cout << "\nAll tests complete.\n";
    return 0;
}
