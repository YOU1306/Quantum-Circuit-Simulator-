// sim_wrapper.h  (host-friendly public header — DOES NOT include common.h/cuda_runtime.h)
#pragma once
#include <cstddef>

#ifdef SIMWRAPPER_EXPORTS
#define SIMAPI __declspec(dllexport)
#else
#define SIMAPI __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // Host-visible complex type (two doubles). Memory layout matches device Complex (real, imag).
    typedef struct {
        double x; // real
        double y; // imag
    } SimComplex;

    // Initialize simulator state for nQubits. Returns device pointer (opaque).
    // Returns nullptr on failure (bad nQubits or cudaMalloc fail).
    SIMAPI void* sim_init_state(int nQubits);

    // Apply hadamard to target qubit
    SIMAPI void sim_hadamard(void* d_state, int nQubits, int targetQubit);

    // Generic single-qubit unitary (4 complex numbers as real,imag pairs)
    SIMAPI void sim_single_qubit(void* d_state, int nQubits, int targetQubit,
        double u00_re, double u00_im,
        double u01_re, double u01_im,
        double u10_re, double u10_im,
        double u11_re, double u11_im);

    // CNOT
    SIMAPI void sim_cnot(void* d_state, int nQubits, int control, int target);

    // Compute probabilities; out_probs must point to host array of length (1<<nQubits).
    // Returns 0 on success, non-zero on failure.
    SIMAPI int sim_compute_probabilities(void* d_state, int nQubits, double* out_probs);

    // Copy full complex amplitudes from device to host. out_amplitudes must point to an array of length (1<<nQubits).
    // Returns 0 on success, non-zero on failure.
    SIMAPI int sim_get_state(void* d_state, int nQubits, SimComplex* out_amplitudes);

    // Multiply two host matrices A and B (size n x n) and write result to C (host).
    // Helper copies A,B to device, calls runMatMul, copies C back.
    // Returns 0 on success.
    SIMAPI int sim_matmul(const SimComplex* A, const SimComplex* B, SimComplex* C, int n);

    // Free device state
    SIMAPI void sim_free_state(void* d_state);

#ifdef __cplusplus
}
#endif
