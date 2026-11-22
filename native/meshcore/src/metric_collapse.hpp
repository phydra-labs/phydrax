//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Positions of the survivor of an edge collapse in a metric tetrahedral mesh.
//
// A collapse merges a removed vertex into a survivor. The survivor may stay
// or move to an exact carrier of the source (metric_carrier.hpp). A position
// is admitted when it strictly decreases the band violations of the measured
// edges, keeps every cell of the survivor positively oriented (exact
// predicate) above a determinant floor and gives every cell a positive
// quality. Lengths, violations and qualities are deterministic kernel values
// (metric_float.hpp), carriers are exact and the orientation is exact: the
// verdicts and the moves are the same on every platform.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "metric_carrier.hpp"

namespace phx::mc::metric {

// The star of a collapse. Row 0 of `points` (three values each) and
// `metrics` (nine each) is the survivor; a position replaces that row.
struct CollapseStar {
  std::int64_t point_count = 0;
  const double* points = nullptr;
  const double* metrics = nullptr;
  // Measured spokes of the survivor after the collapse: the star row, not 0,
  // of each spoke's other end.
  std::int64_t spoke_count = 0;
  const std::int64_t* spokes = nullptr;
  // A nonzero entry puts the neighbor first. Empty uses survivor first.
  std::span<const std::uint8_t> neighbor_first;
  // Metric lengths of the measured edges the collapse removes.
  std::int64_t before_count = 0;
  const double* before_lengths = nullptr;
  // Cells of the survivor after the collapse, four star rows each.
  std::int64_t cell_count = 0;
  const std::int64_t* cells = nullptr;
  // A cell counts as oriented only when its relative determinant
  // (metric_float.hpp) exceeds this floor.
  double determinant_floor = 0.0;
};

enum class PositionVerdict : int {
  kAdmitted = 0,
  kCoincident = 1,     // the position is the other end of a spoke
  kNotDecreasing = 2,  // the band violations do not strictly decrease
  kNotOriented = 3,    // a cell is not positive or is under the floor
  kNoQuality = 4,      // a cell quality is not positive and finite
  kUnmeasured = 5,     // a length is not a positive finite number
  kOutsideDomain = 6,  // a cell coordinate is outside the exact predicate domain
};

// Evaluation of positions of one star. The violations of a set of lengths
// are its nonzero band quanta, largest first, above the band and then below
// it; one set is smaller than another when its long violations are
// lexicographically smaller (a proper prefix is smaller), or equal with
// smaller short violations.
class CollapseObjective {
 public:
  // The star must outlive the objective. May throw std::bad_alloc.
  explicit CollapseObjective(const CollapseStar& star);

  // Verdict of the survivor at `point` with tensor `metric`, decided in the
  // order of PositionVerdict's failures: coincidence; lengths of the spokes
  // (written to `lengths`, spoke_count values, for every verdict but
  // kCoincident); a length before or after that has no finite quantum; the
  // violations; the predicate domain of every cell coordinate; orientation
  // and floor; quality. `quality` is the smallest cell quality of an
  // admitted position.
  PositionVerdict evaluate(const double* point, const double* metric, double* lengths,
                           double& quality);

  // Whether the last admitted position leaves a spoke above the band.
  bool long_violation() const noexcept { return !long_.empty(); }

 private:
  const CollapseStar& star_;
  bool before_measured_ = true;
  std::vector<double> before_long_;
  std::vector<double> before_short_;
  std::vector<double> long_;
  std::vector<double> short_;
};

// One admitted move of a survivor: goal `goal` (0, 1 or 2), the carrier, its
// interpolated tensor and the smallest cell quality at it.
struct SurvivorMove {
  int goal = 0;
  Carrier carrier;
  std::array<double, 9> metric{};
  double quality = 0.0;
};

enum class SearchEnd : int {
  kExhausted = 0,       // every carrier of every goal was evaluated
  kInBand = 1,          // the last move leaves no spoke above the band
  kGoalNotFinite = 2,   // a goal is not a finite point; earlier moves stand
  kUnmeasured = 3,      // the last entry is a carrier with verdict kUnmeasured
  kOutsideDomain = 4,   // the last entry is a carrier with verdict kOutsideDomain
};

// Moves of the survivor of `star` (at least the survivor and the removed
// vertex), in the order they are found. The three goals of
// metric_goal.hpp's survivor_goals are tried in order, each projected onto
// the hull of the first simplex when there is one. The carriers of a goal
// are those of CarrierSource::carriers toward the goal from the survivor,
// coarse first: located in the source cells without simplices, on the
// simplices otherwise; all of them with `every_resolution`, the coarsest
// only without. A carrier at the survivor or at the removed vertex is
// skipped. Its tensor is the log-Euclidean mean of the source tensors
// (`source_metrics`, with their logarithms, nine values per source point)
// over its support in ascending row order; a carrier without one is skipped.
// Every other carrier is evaluated as a position of the star, with its spoke
// lengths appended to `lengths` when it is recorded: an admitted one is a
// move, and the search ends at the first move that leaves no spoke above the
// band, at a goal that is not finite, or at a carrier whose evaluation has no
// verdict here (recorded last, for the caller to report). May throw
// std::bad_alloc.
SearchEnd survivor_search(const CarrierSource& source, const double* source_metrics,
                          const double* source_logarithms, const CollapseStar& star,
                          CarrierChartView chart, bool every_resolution,
                          std::vector<SurvivorMove>& moves, std::vector<double>& lengths);

// The nonzero band quanta (metric_float.hpp) of `count` lengths, largest
// first, above the band in `above` and below it in `below`. False when a
// length has no finite quantum: it was not measured.
bool band_violations(std::int64_t count, const double* lengths, std::vector<double>& above,
                     std::vector<double>& below);

}  // namespace phx::mc::metric
