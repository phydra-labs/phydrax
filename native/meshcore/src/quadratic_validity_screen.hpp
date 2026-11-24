// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Rigorous binary64 screen in front of the exact quadratic tetrahedron
// validity proof (S2 design B.8 step 3).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace phx::mc {

struct QuadraticDerivativeTerm {
  double weight;
  std::uint8_t coordinate_offset;
};

// A (3, 4, 10) stencil, packed once for a bank of cells. Terms keep their
// original node order, including the order of each coordinate sum.
class QuadraticDerivativeStencil {
 public:
  explicit QuadraticDerivativeStencil(const std::int32_t* weights) noexcept {
    std::uint8_t count = 0;
    offsets_[0] = 0;
    for (int entry = 0; entry < 12; ++entry) {
      for (int node = 0; node < 10; ++node) {
        const auto weight = weights[10 * entry + node];
        if (weight != 0) {
          terms_[count++] = {static_cast<double>(weight), static_cast<std::uint8_t>(3 * node)};
        }
      }
      offsets_[entry + 1] = count;
    }
  }

  std::span<const QuadraticDerivativeTerm> terms(int column, int power) const noexcept {
    const auto entry = static_cast<std::size_t>(4 * column + power);
    const auto start = offsets_[entry];
    return {terms_.data() + start, static_cast<std::size_t>(offsets_[entry + 1] - start)};
  }

 private:
  std::array<QuadraticDerivativeTerm, 120> terms_;
  std::array<std::uint8_t, 13> offsets_;
};

// True only if the exact path of phx_mc_quadratic_tetrahedron_validity decides
// the cell (30 node scalars, (3,4,10) derivative weights) VALID at its root;
// false means "not decided here", never "invalid". Requires finite nodes and
// round-to-nearest with gradual underflow (the entry's GradualNearestScope).
bool quadratic_validity_screen(const double* nodes, const QuadraticDerivativeStencil& stencil,
                               double relative_floor) noexcept;

}  // namespace phx::mc
