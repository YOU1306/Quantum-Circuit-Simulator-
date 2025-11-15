// gates.cu
#include "common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ---------------------------
// Helper macros for CUDA error
// ---------------------------
#define CUDA_KERNEL_CHECK() \
    do { cudaError_t err = cudaGetLastError(); \
         if (err != cudaSuccess) { \
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err)); \
            exit(-1); \
         } \
    } while(0)

// ----------------------------------
// Hadamard gate kernel for one qubit
// ----------------------------------
__global__ void hadamardKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = 1 << targetQubit;
    unsigned int half = 1 << (targetQubit);

    if (idx & mask) return; // each pair handled once

    unsigned int partner = idx | mask;

    Complex a = state[idx];
    Complex b = state[partner];

    double invS = 1.0 / sqrt(2.0);

    state[idx]     = Complex((a.x + b.x) * invS, (a.y + b.y) * invS);
    state[partner] = Complex((a.x - b.x) * invS, (a.y - b.y) * invS);
}

extern "C" void runHadamard(Complex* d_state, int nQubits, int targetQubit) {
    size_t numStates = 1ULL << nQubits;
    int threads = 256;
    int blocks = (numStates / 2 + threads - 1) / threads;
    hadamardKernel<<<blocks, threads>>>(d_state, nQubits, targetQubit);
    CUDA_KERNEL_CHECK();
}

// ---------------------------
// Pauli-X (NOT) gate
// ---------------------------
__global__ void pauliXKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = 1 << targetQubit;

    if (!(idx & mask)) {
        unsigned int partner = idx | mask;
        Complex temp = state[idx];
        state[idx] = state[partner];
        state[partner] = temp;
    }
}

extern "C" void runPauliX(Complex* d_state, int nQubits, int targetQubit) {
    size_t numStates = 1ULL << nQubits;
    int threads = 256;
    int blocks = (numStates / 2 + threads - 1) / threads;
    pauliXKernel<<<blocks, threads>>>(d_state, nQubits, targetQubit);
    CUDA_KERNEL_CHECK();
}

// ---------------------------
// Pauli-Y gate
// ---------------------------
__global__ void pauliYKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = 1 << targetQubit;

    if (!(idx & mask)) {
        unsigned int partner = idx | mask;
        Complex a = state[idx];
        Complex b = state[partner];

        state[idx] = Complex(-b.y, b.x);  // i * |1> -> -y + xi
        state[partner] = Complex(a.y, -a.x); // -i * |0>
    }
}

extern "C" void runPauliY(Complex* d_state, int nQubits, int targetQubit) {
    size_t numStates = 1ULL << nQubits;
    int threads = 256;
    int blocks = (numStates / 2 + threads - 1) / threads;
    pauliYKernel<<<blocks, threads>>>(d_state, nQubits, targetQubit);
    CUDA_KERNEL_CHECK();
}

// ---------------------------
// Pauli-Z gate
// ---------------------------
__global__ void pauliZKernel(Complex* state, int nQubits, int targetQubit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = 1 << targetQubit;

    if (idx & mask) {
        state[idx].x *= -1.0;
        state[idx].y *= -1.0;
    }
}

extern "C" void runPauliZ(Complex* d_state, int nQubits, int targetQubit) {
    size_t numStates = 1ULL << nQubits;
    int threads = 256;
    int blocks = (numStates + threads - 1) / threads;
    pauliZKernel<<<blocks, threads>>>(d_state, nQubits, targetQubit);
    CUDA_KERNEL_CHECK();
}

// ---------------------------
// CNOT gate
// ---------------------------
__global__ void cnotKernel(Complex* state, int nQubits, int control, int target) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = 1 << target;
    unsigned int controlMask = 1 << control;

    if ((idx & controlMask) && !(idx & mask)) {
        unsigned int partner = idx | mask;
        Complex temp = state[idx];
        state[idx] = state[partner];
        state[partner] = temp;
    }
}

extern "C" void runCNOT(Complex* d_state, int nQubits, int controlQubit, int targetQubit) {
    size_t numStates = 1ULL << nQubits;
    int threads = 256;
    int blocks = (numStates / 2 + threads - 1) / threads;
    cnotKernel<<<blocks, threads>>>(d_state, nQubits, controlQubit, targetQubit);
    CUDA_KERNEL_CHECK();
}

// ---------------------------
// Generic single qubit gate
// ---------------------------
__global__ void singleQubitKernel(Complex* state, int nQubits, int targetQubit,
                                  Complex u00, Complex u01, Complex u10, Complex u11) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = 1 << targetQubit;

    if (!(idx & mask)) {
        unsigned int partner = idx | mask;
        Complex a = state[idx];
        Complex b = state[partner];

        state[idx]     = a * u00 + b * u01;
        state[partner] = a * u10 + b * u11;
    }
}

extern "C" void runSingleQubit(Complex* d_state, int nQubits, int targetQubit,
                               Complex u00, Complex u01, Complex u10, Complex u11) {
    size_t numStates = 1ULL << nQubits;
    int threads = 256;
    int blocks = (numStates / 2 + threads - 1) / threads;
    singleQubitKernel<<<blocks, threads>>>(d_state, nQubits, targetQubit, u00, u01, u10, u11);
    CUDA_KERNEL_CHECK();
}

// ---------------------------
// Compute probabilities
// ---------------------------
__global__ void computeProbKernel(Complex* state, int nQubits, double* probs) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int N = 1ULL << nQubits;
    if (idx < N) {
        probs[idx] = state[idx].mag2();
    }
}

extern "C" void computeProbabilities(Complex* d_state, int nQubits, double* h_probs) {
    size_t N = 1ULL << nQubits;
    double* d_probs = nullptr;
    cudaMalloc(&d_probs, N * sizeof(double));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    computeProbKernel<<<blocks, threads>>>(d_state, nQubits, d_probs);
    CUDA_KERNEL_CHECK();

    cudaMemcpy(h_probs, d_probs, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_probs);
}

// ---------------------------
// 2x2 Matrix multiplication
// ---------------------------
__global__ void matMulKernel(Complex* A, Complex* B, Complex* C, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    Complex sum = makeComplex(0.0, 0.0);
    for (int k = 0; k < n; ++k) {
        sum = sum + A[row*n + k] * B[k*n + col];
    }
    C[row*n + col] = sum;
}

extern "C" void runMatMul(Complex* d_A, Complex* d_B, Complex* d_C, int n) {
    dim3 threads(n, n);
    dim3 blocks(1, 1);
    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_KERNEL_CHECK();
}
