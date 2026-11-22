//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact carriers of a floating-point goal in a tetrahedral source.
//
// A metric edit moves a vertex to a point of the exact source that a binary64
// carrier can stand for: an exact combination of source vertices with dyadic
// weights. This kernel locates a goal in the source (in the first cell that
// holds it, or by orthogonal projection on a given edge or triangle), rounds
// its barycentric weights to the dyadic grids 2^-bits and returns the
// combinations whose rounded image is closer to the goal than the current
// point. Everything but the final distance comparison is integer arithmetic,
// so the result is one for every platform and equals the reference
// definition bit for bit.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "dyadic.hpp"

struct phx_mc_metric_source;

namespace phx::mc::metric {

// One term of a stencil: source point row and its binary64 weight.
struct StencilTerm {
  std::int32_t row = 0;
  double weight = 0.0;
};

// One carrier: the exact point sum_i weights[i] / 2^bits * X(rows[i]) over
// `size` distinct source points, in ascending row order, with positive
// integer weights summing to 2^bits, its correctly rounded binary64 image,
// and whether that image is the exact point itself.
struct Carrier {
  std::array<std::int64_t, 4> rows{};
  std::array<std::int64_t, 4> weights{};
  int size = 0;
  int bits = 0;
  std::array<double, 3> point{};
  bool exact = false;
};

// The order in which the dyadic resolutions of one goal are tried. `first` is
// the coarsest resolution that can resolve the move, at least 1 and at most 53.
enum class CarrierOrder : int {
  kCoarseFirst = 0,     // first, ..., 53
  kFineFirst = 1,       // 53, ..., first
  kCoarseThenFine = 2,  // first, ... up to the first carrier, then 53, ..., first
};

// An edge or a triangle of the source to project on, and whether a carrier
// on it may be the rounding of an exact point that is not binary64 (a face
// declared as an exact plane).
struct CarrierSimplex {
  std::array<std::int64_t, 3> rows{};
  int size = 0;
  bool declared = false;
  // Prepared by the source: coordinates preserved by orthogonal projection
  // onto an axis-aligned hull. Zero leaves location entirely to the proof.
  std::uint8_t projected_axes = 0;
};

class CarrierSource {
 public:
  // Exact point p is numerators[3 p + axis] / denominator with a positive
  // denominator; `points` holds the binary64 roundings (three per point) and
  // `cells` four point rows per cell, the cells in the order they are
  // searched. Throws std::invalid_argument for a row outside the points or a
  // denominator that is not positive.
  CarrierSource(std::vector<double> points, std::vector<Dyadic> numerators, Dyadic denominator,
                std::vector<std::int64_t> cells);

  std::int64_t point_count() const noexcept {
    return static_cast<std::int64_t>(points_.size() / 3);
  }
  // The binary64 rounding of exact point `row` (three values).
  const double* point(std::int64_t row) const noexcept {
    return points_.data() + 3 * static_cast<std::size_t>(row);
  }
  std::int64_t cell_count() const noexcept {
    return static_cast<std::int64_t>(cells_.size() / 4);
  }

  // Identify a coordinate-subspace hull with exact source numerators.
  CarrierSimplex prepare_simplex(CarrierSimplex simplex) const;

  bool matches_roundings(std::span<const double> points, std::span<const std::int64_t> cells) const;
  // Correctly rounded image of one coordinate of a dyadic stencil.
  double stencil_coordinate(std::span<const StencilTerm> stencil, int axis,
                            bool require_exact, bool& admitted) const;

  // Carriers of `target` closer to it than `current`, at most `limit`, in the
  // order of `order`.
  //
  // Location. With no simplex: the first cell whose closed binary64 box holds
  // the target and in which its exact barycentric weights are all
  // nonnegative, among all cells or, with `cell` not negative, that cell
  // alone; the goal is the target. With simplices: the first edge or
  // triangle with independent edges on which the orthogonal projection of the
  // target has nonnegative weights; the goal is the correctly rounded exact
  // projection. No location, a goal at `current` or a support of zero extent
  // gives no carrier.
  //
  // Resolutions. At 2^-bits the partial sums of the weights, in located
  // order, are rounded to the grid (half to even), made monotone and capped
  // at one; a weight is the difference of consecutive rounded sums and the
  // last takes the rest, so the weights are nonnegative and sum to one.
  // `first` is floor(log2(span / |goal - current|)) clamped to [1, 53], span
  // the largest coordinate extent of the support's binary64 points.
  //
  // A carrier is returned when its rounded image is closer to the goal than
  // `current` is (squared binary64 distances, summed x, y, z) and, on a
  // simplex, when the image is exact or the simplex is declared.
  std::vector<Carrier> carriers(const double* target, const double* current,
                                const std::vector<CarrierSimplex>& simplices, std::int64_t cell,
                                CarrierOrder order, std::int64_t limit) const;

 private:
  std::vector<double> points_;
  std::vector<Dyadic> numerators_;
  Dyadic denominator_;
  std::vector<std::int64_t> cells_;
  std::vector<double> lower_;  // closed binary64 box of each cell
  std::vector<double> upper_;
  // A uniform bucket grid over the cell boxes: bucket b lists, ascending, the
  // cells whose box meets it (CSR), so a location reads one bucket instead
  // of every cell and still finds the first holding cell in search order.
  std::array<double, 3> grid_origin_{};
  std::array<double, 3> grid_scale_{};
  std::array<std::int64_t, 3> grid_size_{};
  std::vector<std::int64_t> grid_offsets_;
  std::vector<std::int64_t> grid_cells_;
  // The numerators and then the denominator as 64-bit limbs, integer i in
  // limbs [offsets[i], offsets[i + 1]), when each fits the inline integers
  // of the search; a query that leaves their capacity runs on the unbounded
  // ones above.
  bool inline_ = false;
  std::vector<std::uint64_t> inline_limbs_;
  std::vector<std::uint32_t> inline_offsets_;
  std::vector<std::uint8_t> inline_negative_;
};

const CarrierSource& metric_carrier_source(const phx_mc_metric_source& source) noexcept;

}  // namespace phx::mc::metric
