#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

struct Complex {
    double x, y;
    __host__ __device__ Complex() : x(0.0), y(0.0) {}
    __host__ __device__ Complex(double _x, double _y) : x(_x), y(_y) {}
    __host__ __device__ double mag2() const { return x * x + y * y; }
    __host__ __device__ Complex operator+(const Complex& b) const { return Complex(x + b.x, y + b.y); }
    __host__ __device__ Complex operator-(const Complex& b) const { return Complex(x - b.x, y - b.y); }
    __host__ __device__ Complex operator*(const Complex& b) const { return Complex(x * b.x - y * b.y, x * b.y + y * b.x); }
    __host__ __device__ Complex operator*(double s) const { return Complex(x * s, y * s); }
    __host__ __device__ Complex& operator+=(const Complex& b) { x += b.x; y += b.y; return *this; }
};

inline __host__ __device__ Complex makeComplex(double x, double y) { return Complex(x, y); }
inline __host__ __device__ Complex addComplex(const Complex& a, const Complex& b) { return a + b; }
inline __host__ __device__ Complex subComplex(const Complex& a, const Complex& b) { return a - b; }
inline __host__ __device__ Complex mulComplex(const Complex& a, const Complex& b) { return a * b; }
inline __host__ __device__ Complex scaleComplex(const Complex& a, double s) { return a * s; }
