// Copyright (c) 2026 PHYDRA, Inc. All rights reserved.
#include "quda_kernels.hpp"
#include <cuda_runtime.h>

namespace phx::cuda::quda_kernels {
namespace {
__global__ void any_unowned(const bool* mask, int64_t size, int* invalid) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    if (!mask[i]) atomicExch(invalid, 1);
}

template <class Real>
__global__ void decode_momentum(const Real* packed, Real* out, int64_t sites) {
  for (int64_t site = blockIdx.x * blockDim.x + threadIdx.x; site < sites; site += blockDim.x * gridDim.x) {
    const Real* p = packed + site * 10;
    Real* matrix = out + site * 18;
    for (int i = 0; i < 18; ++i) matrix[i] = 0;
    const int row[3] = {0, 0, 1};
    const int col[3] = {1, 2, 2};
    for (int k = 0; k < 3; ++k) {
      matrix[2 * (3 * row[k] + col[k])] = p[2 * k];
      matrix[2 * (3 * row[k] + col[k]) + 1] = p[2 * k + 1];
      matrix[2 * (3 * col[k] + row[k])] = -p[2 * k];
      matrix[2 * (3 * col[k] + row[k]) + 1] = p[2 * k + 1];
    }
    for (int color = 0; color < 3; ++color) matrix[2 * (4 * color) + 1] = p[6 + color];
  }
}
}  // namespace

void launch_any_unowned(const bool* mask, int64_t size, int* invalid) {
  any_unowned<<<128, 256>>>(mask, size, invalid);
}

void launch_decode_momentum(const float* packed, float* out, int64_t sites) {
  decode_momentum<<<128, 256>>>(packed, out, sites);
}

void launch_decode_momentum(const double* packed, double* out, int64_t sites) {
  decode_momentum<<<128, 256>>>(packed, out, sites);
}
}  // namespace phx::cuda::quda_kernels
