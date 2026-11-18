//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Positions of the survivor of an edge collapse; see metric_collapse.hpp.
#include "metric_collapse.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>

#include "deterministic_float.hpp"
#include "metric_float.hpp"
#include "metric_goal.hpp"
#include "predicates.hpp"

namespace phx::mc::metric {
namespace {

bool finite(double value) noexcept {
  return (value < 0.0 ? -value : value) <= std::numeric_limits<double>::max();
}

bool same_point(const double* a, const double* b) noexcept {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

// The nonzero band quanta of `count` lengths, largest first, above the band
// in `above` and below it in `below`. False when a length has no finite
// quantum: it was not measured.
bool violations(std::int64_t count, const double* lengths, std::vector<double>& above,
                std::vector<double>& below) {
  above.clear();
  below.clear();
  bool measured = true;
  for (std::int64_t row = 0; row < count; ++row) {
    double over = 0.0;
    double under = 0.0;
    band_quanta(lengths[row], over, under);
    if (!finite(over) || !finite(under)) {
      measured = false;
      continue;
    }
    if (over != 0.0) above.push_back(over);
    if (under != 0.0) below.push_back(under);
  }
  std::sort(above.begin(), above.end(), std::greater<double>());
  std::sort(below.begin(), below.end(), std::greater<double>());
  return measured;
}

bool smaller(const std::vector<double>& a, const std::vector<double>& b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

}  // namespace

CollapseObjective::CollapseObjective(const CollapseStar& star) : star_(star) {
  before_measured_ =
      violations(star.before_count, star.before_lengths, before_long_, before_short_);
  const std::size_t spokes = static_cast<std::size_t>(star.spoke_count);
  long_.reserve(spokes);
  short_.reserve(spokes);
}

PositionVerdict CollapseObjective::evaluate(const double* point, const double* metric,
                                            double* lengths, double& quality) {
  const CollapseStar& star = star_;
  for (std::int64_t spoke = 0; spoke < star.spoke_count; ++spoke) {
    if (same_point(point, star.points + 3 * star.spokes[spoke])) {
      return PositionVerdict::kCoincident;
    }
  }
  for (std::int64_t spoke = 0; spoke < star.spoke_count; ++spoke) {
    const std::int64_t other = star.spokes[spoke];
    lengths[spoke] = length(point, star.points + 3 * other, metric, star.metrics + 9 * other);
  }
  if (!before_measured_) return PositionVerdict::kUnmeasured;
  if (!violations(star.spoke_count, lengths, long_, short_)) {
    return PositionVerdict::kUnmeasured;
  }
  if (!(smaller(long_, before_long_) ||
        (long_ == before_long_ && smaller(short_, before_short_)))) {
    return PositionVerdict::kNotDecreasing;
  }

  const auto corner = [&](std::int64_t row) { return row == 0 ? point : star.points + 3 * row; };
  for (std::int64_t entry = 0; entry < 4 * star.cell_count; ++entry) {
    const double* values = corner(star.cells[entry]);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!finite(values[axis]) || !coordinate_in_domain(values[axis])) {
        return PositionVerdict::kOutsideDomain;
      }
    }
  }
  for (std::int64_t cell = 0; cell < star.cell_count; ++cell) {
    const std::int64_t* rows = star.cells + 4 * cell;
    if (orient3d(corner(rows[0]), corner(rows[1]), corner(rows[2]), corner(rows[3])) != 1) {
      return PositionVerdict::kNotOriented;
    }
  }
  double corners[12];
  const auto gather = [&](std::int64_t cell) {
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const double* values = corner(star.cells[4 * cell + static_cast<std::int64_t>(slot)]);
      for (std::size_t axis = 0; axis < 3; ++axis) corners[3 * slot + axis] = values[axis];
    }
  };
  for (std::int64_t cell = 0; cell < star.cell_count; ++cell) {
    gather(cell);
    double determinant = 0.0;
    double relative = 0.0;
    relative_determinant(corners, determinant, relative);
    if (!(relative > star.determinant_floor)) return PositionVerdict::kNotOriented;
  }
  double tensors[36];
  double smallest = 0.0;
  for (std::int64_t cell = 0; cell < star.cell_count; ++cell) {
    gather(cell);
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const std::int64_t row = star.cells[4 * cell + static_cast<std::int64_t>(slot)];
      const double* tensor = row == 0 ? metric : star.metrics + 9 * row;
      for (std::size_t index = 0; index < 9; ++index) tensors[9 * slot + index] = tensor[index];
    }
    const double value = tetra_quality(corners, tensors);
    if (!finite(value) || !(value > 0.0)) return PositionVerdict::kNoQuality;
    if (cell == 0 || value < smallest) smallest = value;
  }
  quality = smallest;
  return PositionVerdict::kAdmitted;
}

SearchEnd survivor_search(const CarrierSource& source, const double* source_metrics,
                          const double* source_logarithms, const CollapseStar& star,
                          const std::vector<CarrierSimplex>& simplices, bool every_resolution,
                          std::vector<SurvivorMove>& moves, std::vector<double>& lengths) {
  const double* survivor = star.points;
  const double* removed = star.points + 3;
  int corner_count = 0;
  double corners[9];
  if (!simplices.empty()) {
    // The simplices of one carrier are coplanar or collinear: the first
    // spans their hull.
    const CarrierSimplex& first = simplices.front();
    corner_count = first.size;
    for (int slot = 0; slot < first.size; ++slot) {
      const double* values = source.point(first.rows[static_cast<std::size_t>(slot)]);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        corners[3 * static_cast<std::size_t>(slot) + axis] = values[axis];
      }
    }
  }
  double goals[9];
  bool valid[3];
  survivor_goals(star.point_count, star.points, star.metrics, corner_count, corners, goals,
                 valid);

  CollapseObjective objective(star);
  const std::size_t spokes = static_cast<std::size_t>(star.spoke_count);
  std::vector<double> measured(spokes);
  // Three goals at the 53 resolutions of a carrier.
  const std::int64_t limit = every_resolution ? 53 : 1;
  for (int goal = 0; goal < 3; ++goal) {
    if (!valid[goal]) return SearchEnd::kGoalNotFinite;
    const std::vector<Carrier> carriers = source.carriers(
        goals + 3 * goal, survivor, simplices, CarrierOrder::kCoarseFirst, limit);
    for (const Carrier& carrier : carriers) {
      const double* point = carrier.point.data();
      if (same_point(point, survivor) || same_point(point, removed)) continue;
      double weights[4];
      double tensors[36];
      double logarithms[36];
      const double scale = static_cast<double>(std::int64_t{1} << carrier.bits);
      for (int slot = 0; slot < carrier.size; ++slot) {
        const std::size_t index = static_cast<std::size_t>(slot);
        // An integer of at most 53 bits over a power of two: exact.
        weights[index] = static_cast<double>(carrier.weights[index]) / scale;
        const std::int64_t row = carrier.rows[index];
        for (std::size_t entry = 0; entry < 9; ++entry) {
          tensors[9 * index + entry] = source_metrics[9 * row + static_cast<std::int64_t>(entry)];
          logarithms[9 * index + entry] =
              source_logarithms[9 * row + static_cast<std::int64_t>(entry)];
        }
      }
      SurvivorMove move;
      move.goal = goal;
      move.carrier = carrier;
      if (!interpolate(carrier.size, weights, tensors, logarithms, move.metric.data())) continue;
      const PositionVerdict verdict =
          objective.evaluate(point, move.metric.data(), measured.data(), move.quality);
      SearchEnd end = SearchEnd::kExhausted;
      switch (verdict) {
        case PositionVerdict::kAdmitted:
          if (!objective.long_violation()) end = SearchEnd::kInBand;
          break;
        case PositionVerdict::kUnmeasured:
          end = SearchEnd::kUnmeasured;
          break;
        case PositionVerdict::kOutsideDomain:
          end = SearchEnd::kOutsideDomain;
          break;
        case PositionVerdict::kCoincident:
        case PositionVerdict::kNotDecreasing:
        case PositionVerdict::kNotOriented:
        case PositionVerdict::kNoQuality:
          continue;
      }
      moves.push_back(move);
      lengths.insert(lengths.end(), measured.begin(), measured.end());
      if (end != SearchEnd::kExhausted) return end;
    }
  }
  return SearchEnd::kExhausted;
}

}  // namespace phx::mc::metric
