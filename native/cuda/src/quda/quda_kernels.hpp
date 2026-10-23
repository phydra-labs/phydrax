// Copyright (c) 2026 PHYDRA, Inc. All rights reserved.
#pragma once
#include <cstdint>

namespace phx::cuda::quda_kernels {
// The caller owns serialization, blocking readback, error checks, and barriers.
void launch_any_unowned(const bool* mask, int64_t size, int* invalid);
void launch_decode_momentum(const float* packed, float* out, int64_t sites);
void launch_decode_momentum(const double* packed, double* out, int64_t sites);
}  // namespace phx::cuda::quda_kernels
