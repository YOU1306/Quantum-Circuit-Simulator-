// test_client.cpp
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <limits>
#include "sim_wrapper.h"   // public header only (defines SimComplex, no CUDA headers)

// Helper: convert integer to binary string of length n (MSB..LSB)
static std::string toBinary(uint64_t value, int n) {
    std::string s;
    s.reserve(n);
    for (int i = n - 1; i >= 0; --i) s.push_back(((value >> i) & 1ULL) ? '1' : '0');
    return s;
}

static void printMenu() {
    std::cout << "\n===== QuantumCore Interactive Simulator =====\n";
    std::cout << "1. Apply Hadamard gate\n";
    std::cout << "2. Apply Pauli-X gate\n";
    std::cout << "3. Apply Pauli-Y gate\n";
    std::cout << "4. Apply Pauli-Z gate\n";
    std::cout << "5. Apply CNOT gate\n";
    std::cout << "6. Apply custom 2x2 matrix gate\n";
    std::cout << "7. Compute & print probabilities\n";
    std::cout << "8. Print first K amplitudes (complex) [requires sim_get_state]\n";
    std::cout << "9. Reset quantum state\n";
    std::cout << "0. Exit\n";
    std::cout << "=============================================\n";
    std::cout << "Enter your choice: ";
}

static void printProbabilities(void* state, int nQubits, int maxPrint = -1) {
    if (!state) { std::cout << "No state allocated.\n"; return; }
    uint64_t N = 1ULL << nQubits;
    std::vector<double> probs;
    try { probs.resize(N); }
    catch (...) { std::cout << "Host memory error\n"; return; }

    int rc = sim_compute_probabilities(state, nQubits, probs.data());
    if (rc != 0) { std::cout << "sim_compute_probabilities failed: " << rc << "\n"; return; }

    std::cout << "\n--- Measurement Probabilities ---\n";
    uint64_t toPrint = (maxPrint <= 0) ? N : std::min<uint64_t>((uint64_t)maxPrint, N);
    for (uint64_t i = 0; i < toPrint; ++i) {
        std::cout << "|" << toBinary(i, nQubits) << "> : "
            << std::fixed << std::setprecision(8) << probs[i] << "\n";
    }
    if (toPrint < N) std::cout << "... (" << (N - toPrint) << " more states not shown)\n";
}

static void printAmplitudes_if_supported(void* state, int nQubits, int maxPrint = 64) {
    if (!state) { std::cout << "No state allocated.\n"; return; }
    uint64_t N = 1ULL << nQubits;
    std::vector<SimComplex> amps;
    try { amps.resize(N); }
    catch (...) { std::cout << "Not enough host memory to retrieve amplitudes.\n"; return; }

    int rc = sim_get_state(state, nQubits, amps.data());
    if (rc != 0) {
        std::cout << "sim_get_state returned error: " << rc << ".\n";
        std::cout << "If this function is not implemented in the DLL, update sim_wrapper and rebuild QuantumCore.\n";
        return;
    }

    std::cout << "\n--- Complex Amplitudes (first " << std::min<uint64_t>(maxPrint, N) << ") ---\n";
    uint64_t toPrint = std::min<uint64_t>((uint64_t)maxPrint, N);
    for (uint64_t i = 0; i < toPrint; ++i) {
        double re = amps[i].x;
        double im = amps[i].y;
        double prob = re * re + im * im;
        std::cout << "|" << toBinary(i, nQubits) << "> = (" << std::fixed << std::setprecision(8)
            << re << (im >= 0 ? "+" : "") << im << "i)  prob=" << prob << "\n";
    }
    if (toPrint < N) std::cout << "... (" << (N - toPrint) << " more states not shown)\n";
}

int main() {
    std::cout << "=== QuantumCore Interactive Simulator ===\n";

    int nQubits = 0;
    while (true) {
        std::cout << "Enter number of qubits (1.." << 28 << "): ";
        if (!(std::cin >> nQubits)) { std::cout << "Invalid input. Exiting.\n"; return 1; }
        if (nQubits >= 1 && nQubits <= 28) break;
        std::cout << "Please enter a value between 1 and 28.\n";
    }

    void* state = sim_init_state(nQubits);
    if (!state) { std::cout << "Failed to initialize CUDA quantum state. Exiting.\n"; return 1; }
    std::cout << "Initialized " << nQubits << " qubits.\n";

    int choice = -1;
    while (true) {
        printMenu();
        if (!(std::cin >> choice)) { std::cout << "Input error. Exiting.\n"; break; }
        if (choice == 0) break;

        switch (choice) {
        case 1: {
            int q; std::cout << "Apply Hadamard to qubit index (0 = LSB ... " << (nQubits - 1) << " = MSB): ";
            std::cin >> q;
            if (q < 0 || q >= nQubits) { std::cout << "Invalid qubit index.\n"; break; }
            sim_hadamard(state, nQubits, q);
            std::cout << "Hadamard applied to qubit " << q << ".\n"; break;
        }
        case 2: {
            int q; std::cout << "Apply Pauli-X to qubit index: "; std::cin >> q;
            if (q < 0 || q >= nQubits) { std::cout << "Invalid qubit index.\n"; break; }
            sim_single_qubit(state, nQubits, q, 0, 0, 1, 0, 1, 0, 0, 0);
            std::cout << "Pauli-X applied to qubit " << q << ".\n"; break;
        }
        case 3: {
            int q; std::cout << "Apply Pauli-Y to qubit index: "; std::cin >> q;
            if (q < 0 || q >= nQubits) { std::cout << "Invalid qubit index.\n"; break; }
            sim_single_qubit(state, nQubits, q, 0, 0, 0, -1, 0, 1, 0, 0);
            std::cout << "Pauli-Y applied to qubit " << q << ".\n"; break;
        }
        case 4: {
            int q; std::cout << "Apply Pauli-Z to qubit index: "; std::cin >> q;
            if (q < 0 || q >= nQubits) { std::cout << "Invalid qubit index.\n"; break; }
            sim_single_qubit(state, nQubits, q, 1, 0, 0, 0, 0, 0, -1, 0);
            std::cout << "Pauli-Z applied to qubit " << q << ".\n"; break;
        }
        case 5: {
            int control, target;
            std::cout << "Control qubit index: "; std::cin >> control;
            std::cout << "Target qubit index: "; std::cin >> target;
            if (control < 0 || control >= nQubits || target < 0 || target >= nQubits) { std::cout << "Invalid qubit index.\n"; break; }
            if (control == target) { std::cout << "Control and target must differ.\n"; break; }
            sim_cnot(state, nQubits, control, target);
            std::cout << "CNOT applied (control=" << control << ", target=" << target << ").\n"; break;
        }
        case 6: {
            int q; std::cout << "Apply custom 2x2 gate to qubit index: "; std::cin >> q;
            if (q < 0 || q >= nQubits) { std::cout << "Invalid qubit index.\n"; break; }
            std::cout << "Enter 8 numbers (real imag) row-major for 2x2 matrix:\n";
            double vals[8];
            for (int i = 0; i < 8; ++i) { while (!(std::cin >> vals[i])) { std::cin.clear(); std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); std::cout << "Invalid number, try again: "; } }
            sim_single_qubit(state, nQubits, q, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
            std::cout << "Custom 2x2 gate applied to qubit " << q << ".\n"; break;
        }
        case 7: {
            int maxPrint; std::cout << "How many states to print? (enter 0 to print all): "; std::cin >> maxPrint;
            if (maxPrint == 0) maxPrint = -1; printProbabilities(state, nQubits, maxPrint); break;
        }
        case 8: {
            int k; std::cout << "How many amplitudes to print (default 64): "; std::cin >> k; if (k <= 0) k = 64;
            printAmplitudes_if_supported(state, nQubits, k); break;
        }
        case 9: {
            sim_free_state(state);
            state = sim_init_state(nQubits);
            if (!state) { std::cout << "Failed to reinitialize state.\n"; return 1; }
            std::cout << "State reset to |0...0>.\n"; break;
        }
        default: std::cout << "Invalid choice.\n"; break;
        }
    }

    sim_free_state(state);
    std::cout << "Exiting simulator. Goodbye!\n";
    return 0;
}
