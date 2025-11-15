// sim_wrapper.cpp
#include "sim_wrapper.h"   // defines SimComplex
#include "common.h"        // defines device Complex (with __host__ __device__ helpers)
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <iostream>

// Forward declarations (implemented in .cu files)
extern "C" void runHadamard(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runPauliX(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runPauliY(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runPauliZ(Complex* d_state, int nQubits, int targetQubit);
extern "C" void runMatMul(Complex* d_A, Complex* d_B, Complex* d_C, int n);
extern "C" void runSingleQubit(Complex* d_state, int nQubits, int targetQubit,
    Complex u00, Complex u01, Complex u10, Complex u11);
extern "C" void runCNOT(Complex* d_state, int nQubits, int controlQubit, int targetQubit);
extern "C" void computeProbabilities(Complex* d_state, int nQubits, double* h_probs);

static std::mutex g_cuda_mutex;
static const int MAX_QUBITS_SAFE = 28; // conservative default

/* Helper */
static bool check_qubit_count(int nQubits) {
    if (nQubits <= 0) return false;
    if (nQubits > 62) return false;
    if (nQubits > MAX_QUBITS_SAFE) return false;
    return true;
}

extern "C" SIMAPI void* sim_init_state(int nQubits) {
    if (!check_qubit_count(nQubits)) return nullptr;
    uint64_t numStates = 1ULL << nQubits;
    size_t bytes = (size_t)numStates * sizeof(Complex);

    // Prevent huge allocation; adjust if you know GPU memory
    const size_t HARD_LIMIT = (size_t)12ULL * 1024 * 1024 * 1024;
    if (bytes > HARD_LIMIT) return nullptr;

    Complex* d_state = nullptr;
    cudaError_t err = cudaMalloc(&d_state, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    // init |0...0>
    try {
        // create host buffer of SimComplex then copy as bytes
        std::vector<SimComplex> h_state(numStates);
        for (uint64_t i = 0; i < numStates; ++i) { h_state[i].x = 0.0; h_state[i].y = 0.0; }
        h_state[0].x = 1.0; h_state[0].y = 0.0;

        err = cudaMemcpy(d_state, h_state.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy init failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_state);
            return nullptr;
        }
    }
    catch (...) {
        cudaFree(d_state);
        return nullptr;
    }
    return reinterpret_cast<void*>(d_state);
}

extern "C" SIMAPI void sim_hadamard(void* statePtr, int nQubits, int targetQubit) {
    if (!statePtr || !check_qubit_count(nQubits)) return;
    Complex* d_state = reinterpret_cast<Complex*>(statePtr);
    std::lock_guard<std::mutex> lg(g_cuda_mutex);
    runHadamard(d_state, nQubits, targetQubit);
    cudaDeviceSynchronize();
}

extern "C" SIMAPI void sim_single_qubit(void* statePtr, int nQubits, int targetQubit,
    double u00_re, double u00_im,
    double u01_re, double u01_im,
    double u10_re, double u10_im,
    double u11_re, double u11_im) {
    if (!statePtr || !check_qubit_count(nQubits)) return;
    Complex* d_state = reinterpret_cast<Complex*>(statePtr);
    Complex u00 = makeComplex(u00_re, u00_im);
    Complex u01 = makeComplex(u01_re, u01_im);
    Complex u10 = makeComplex(u10_re, u10_im);
    Complex u11 = makeComplex(u11_re, u11_im);
    std::lock_guard<std::mutex> lg(g_cuda_mutex);
    runSingleQubit(d_state, nQubits, targetQubit, u00, u01, u10, u11);
    cudaDeviceSynchronize();
}

extern "C" SIMAPI void sim_cnot(void* statePtr, int nQubits, int control, int target) {
    if (!statePtr || !check_qubit_count(nQubits)) return;
    Complex* d_state = reinterpret_cast<Complex*>(statePtr);
    std::lock_guard<std::mutex> lg(g_cuda_mutex);
    runCNOT(d_state, nQubits, control, target);
    cudaDeviceSynchronize();
}

extern "C" SIMAPI int sim_compute_probabilities(void* statePtr, int nQubits, double* out_probs) {
    if (!statePtr || !out_probs) return -1;
    if (!check_qubit_count(nQubits)) return -2;
    Complex* d_state = reinterpret_cast<Complex*>(statePtr);
    std::lock_guard<std::mutex> lg(g_cuda_mutex);
    try {
        computeProbabilities(d_state, nQubits, out_probs);
    }
    catch (...) {
        return -3;
    }
    return 0;
}

extern "C" SIMAPI int sim_get_state(void* statePtr, int nQubits, SimComplex* out_amplitudes) {
    if (!statePtr || !out_amplitudes) return -1;
    if (!check_qubit_count(nQubits)) return -2;
    Complex* d_state = reinterpret_cast<Complex*>(statePtr);
    uint64_t N = 1ULL << nQubits;
    size_t bytes = (size_t)N * sizeof(Complex);

    // direct device->host copy into SimComplex buffer (same layout)
    cudaError_t err = cudaMemcpy(out_amplitudes, d_state, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "sim_get_state cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        return -3;
    }
    return 0;
}

extern "C" SIMAPI int sim_matmul(const SimComplex* A_host, const SimComplex* B_host, SimComplex* C_host, int n) {
    if (!A_host || !B_host || !C_host) return -1;
    if (n <= 0) return -2;
    size_t bytes = (size_t)n * n * sizeof(Complex);

    Complex* d_A = nullptr;
    Complex* d_B = nullptr;
    Complex* d_C = nullptr;

    cudaError_t err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) return -3;
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) { cudaFree(d_A); return -4; }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return -5; }

    // host arrays are SimComplex (same layout) — copy directly
    err = cudaMemcpy(d_A, A_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -6; }
    err = cudaMemcpy(d_B, B_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -7; }

    {
        std::lock_guard<std::mutex> lg(g_cuda_mutex);
        runMatMul(d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
    }

    err = cudaMemcpy(C_host, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    if (err != cudaSuccess) return -8;
    return 0;
}

extern "C" SIMAPI void sim_free_state(void* statePtr) {
    if (!statePtr) return;
    Complex* d_state = reinterpret_cast<Complex*>(statePtr);
    cudaFree(d_state);
}
