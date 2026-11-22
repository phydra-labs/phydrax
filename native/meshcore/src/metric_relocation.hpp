//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Candidate positions of a relocated vertex in a metric tetrahedral mesh.
//
// A vertex is pulled toward unit metric length along its edges. Each pull p
// gives the goals x + 2^-s p for s = 0, 1, ... until the goal is x itself
// (s <= 5 unless the search is deep); a goal is projected onto the hull of
// the vertex's source carrier when it has one, and a goal seen before is
// passed over. A goal is screened in binary64 and, when admitted, replaced
// by exact carriers of the source (metric_carrier.hpp); a carrier is a
// candidate when its evaluation admits it:
//   * no neighbor of the vertex lies at the position;
//   * the band violations of the measured spokes do not increase (and
//     strictly decrease when the search is band-only);
//   * every cell of the star has a positive determinant with a relative
//     determinant above the floor (metric_float.hpp);
//   * when the violations are unchanged, the smallest cell quality exceeds
//     the one before by more than the improvement.
// In a constant field the screen of a goal is this evaluation at the goal
// with the vertex's own tensor; in a graded field it is the determinant
// test alone. The screen only spares work: every candidate is an exact
// carrier evaluated in full.
//
// The search is a sequence: next() returns the candidates of the next goal
// that has any, in the order of its carriers, and never depends on how often
// it was called before. Goals, lengths, violations and qualities are
// deterministic kernel values and carriers are exact, so the sequence is the
// same on every platform.
#pragma once

#include <array>
#include <cstdint>
#include <set>
#include <vector>

#include "metric_carrier.hpp"

namespace phx::mc::metric {

// The star of a relocation. Row 0 of `points` (three values each) and
// `metrics` (nine each) is the moving vertex; rows 1.. are its neighbors,
// spoke j running from row 0 to row j + 1.
struct RelocationStar {
  std::int64_t point_count = 0;
  const double* points = nullptr;
  const double* metrics = nullptr;
  // Per spoke: whether its length is measured, and its metric length now.
  const std::int8_t* measured = nullptr;
  const double* before_lengths = nullptr;
  // Smallest cell quality now.
  double before_quality = 0.0;
  // Cells of the star, four star rows each.
  std::int64_t cell_count = 0;
  const std::int64_t* cells = nullptr;
  double determinant_floor = 0.0;
  double improvement = 0.0;
};

struct RelocationMode {
  // Goals at every step and the coarsest and the finest carrier of a goal.
  bool deep = false;
  // The finest carrier of every goal first; then, for the same goals in
  // order, every remaining resolution. Takes precedence over `deep` for the
  // carriers of a goal.
  bool refine = false;
  // Unchanged violations do not admit a position.
  bool band_only = false;
  // The field is one tensor: goals are screened by the full evaluation.
  bool constant_field = false;
};

// One candidate: the carrier, its interpolated tensor and the smallest cell
// quality of the star with the vertex at it.
struct RelocationCandidate {
  Carrier carrier;
  std::array<double, 9> metric{};
  double quality = 0.0;
};

enum class RelocationEnd : int {
  kGroup = 0,      // the candidates of one goal, at least one
  kExhausted = 1,  // no goal is left
  // The search cannot go on; the caller reports the entry it is given.
  kUnmeasuredGoal = 2,     // a goal whose screen met a spoke without a finite quantum
  kUnmeasuredCarrier = 3,  // a carrier whose evaluation met one
  kNoTensor = 4,           // a carrier whose tensor could not be interpolated
};

class RelocationSearch {
 public:
  // Copies the star, the pulls (three values each), the frame (`origin`,
  // three values, and `projector`, nine, or both null). The chart, source,
  // and source tensors with their logarithms (nine values per source point)
  // must outlive the search. An empty chart locates in the source cells.
  // Throws std::invalid_argument when a measured spoke of the
  // star has no finite quantum now; may throw std::bad_alloc.
  RelocationSearch(const CarrierSource& source, const double* source_metrics,
                   const double* source_logarithms, const RelocationStar& star,
                   std::int64_t pull_count, const double* pulls, const double* origin,
                   const double* projector, CarrierChartView chart, RelocationMode mode);

  std::int64_t spoke_count() const noexcept { return spoke_count_; }

  // The next group. For kGroup, `group` holds the candidates and `lengths`
  // the spoke lengths of each (spoke_count values per candidate). For the
  // three failures `group` holds one entry, the goal or the carrier that
  // cannot be decided (a goal has no support: size 0, with the vertex's
  // tensor), and `lengths` its spoke lengths when they were measured. May
  // throw std::bad_alloc.
  RelocationEnd next(std::vector<RelocationCandidate>& group, std::vector<double>& lengths);

 private:
  enum class Verdict { kAdmitted, kRejected, kUnmeasured };

  const double* point(std::int64_t row) const noexcept {
    return points_.data() + 3 * static_cast<std::size_t>(row);
  }
  bool raw_goal(double* goal);
  bool floor_oriented(const double* position) const noexcept;
  Verdict evaluate(const double* position, const double* metric, double* lengths,
                   double& quality);

  const CarrierSource& source_;
  const double* source_metrics_;
  const double* source_logarithms_;
  std::int64_t point_count_ = 0;
  std::int64_t spoke_count_ = 0;
  std::int64_t cell_count_ = 0;
  std::vector<double> points_;
  std::vector<double> metrics_;
  std::vector<std::int8_t> measured_;
  std::vector<std::int64_t> cells_;
  double before_quality_ = 0.0;
  double determinant_floor_ = 0.0;
  double improvement_ = 0.0;
  std::vector<double> before_long_;
  std::vector<double> before_short_;
  std::vector<double> pulls_;
  bool framed_ = false;
  std::array<double, 3> origin_{};
  std::array<double, 9> projector_{};
  CarrierChartView chart_;
  RelocationMode mode_;

  // The cursor: the pull and step of the next raw goal, the goals seen, the
  // stencils examined, and for a refining search the admitted goals and the
  // position of their second pass.
  std::int64_t pull_ = 0;
  std::int64_t step_ = 0;
  std::set<std::array<std::uint64_t, 3>> seen_;
  std::set<std::vector<std::int64_t>> examined_;
  std::vector<double> admitted_goals_;
  bool second_pass_ = false;
  std::size_t revisit_ = 0;
  // Scratch of one evaluation.
  std::vector<double> scratch_;
  std::vector<double> long_;
  std::vector<double> short_;
  std::vector<double> selected_;
};

}  // namespace phx::mc::metric
