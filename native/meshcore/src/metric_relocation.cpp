//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Candidate positions of a relocated vertex; see metric_relocation.hpp.
#include "metric_relocation.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "deterministic_float.hpp"
#include "metric_collapse.hpp"
#include "metric_float.hpp"
#include "metric_goal.hpp"

namespace phx::mc::metric {
namespace {

// Seeds of a search that is not deep: steps 0 to 5.
constexpr std::int64_t kSeedSteps = 5;
// A finite pull scaled by 2^-s is zero long before this step, so every goal
// sequence has ended; a pull that is not finite ends here as well.
constexpr std::int64_t kLastStep = 2200;
// The 53 resolutions of a carrier and one more, the bound the host states.
constexpr std::int64_t kAllCarriers = 54;
constexpr std::uint8_t kMeasured = 1;
constexpr std::uint8_t kNeighborFirst = 2;

bool same_point(const double* a, const double* b) noexcept {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

bool smaller(const std::vector<double>& a, const std::vector<double>& b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

}  // namespace

RelocationSearch::RelocationSearch(const CarrierSource& source, const double* source_metrics,
                                   const double* source_logarithms, const RelocationStar& star,
                                   std::int64_t pull_count, const double* pulls,
                                   const double* origin, const double* projector,
                                   CarrierChartView chart, RelocationMode mode)
    : source_(source),
      source_metrics_(source_metrics),
      source_logarithms_(source_logarithms),
      point_count_(star.point_count),
      spoke_count_(star.point_count - 1),
      cell_count_(star.cell_count),
      points_(star.points, star.points + 3 * star.point_count),
      metrics_(star.metrics, star.metrics + 9 * star.point_count),
      spoke_flags_(static_cast<std::size_t>(spoke_count_)),
      cells_(star.cells, star.cells + 4 * star.cell_count),
      before_quality_(star.before_quality),
      determinant_floor_(star.determinant_floor),
      improvement_(star.improvement),
      pulls_(pulls, pulls + 3 * pull_count),
      chart_(chart),
      mode_(mode) {
  if (origin != nullptr && projector != nullptr) {
    framed_ = true;
    std::copy(origin, origin + 3, origin_.begin());
    std::copy(projector, projector + 9, projector_.begin());
  }
  const std::size_t spokes = static_cast<std::size_t>(spoke_count_);
  if (!star.neighbor_first.empty() && star.neighbor_first.size() != spokes) {
    throw std::invalid_argument("Relocation measurement directions must match the spokes.");
  }
  scratch_.resize(spokes);
  selected_.reserve(spokes);
  for (std::size_t spoke = 0; spoke < spokes; ++spoke) {
    if (star.measured[spoke] != 0) {
      spoke_flags_[spoke] |= kMeasured;
      selected_.push_back(star.before_lengths[spoke]);
    }
    if (!star.neighbor_first.empty() && star.neighbor_first[spoke] != 0) {
      spoke_flags_[spoke] |= kNeighborFirst;
    }
  }
  if (!band_violations(static_cast<std::int64_t>(selected_.size()), selected_.data(),
                       before_long_, before_short_)) {
    throw std::invalid_argument("A measured spoke of a relocation star has no finite length.");
  }
}

// The next goal of the pulls before projection: x + 2^-step pull for the
// current pull while it differs from x.
bool RelocationSearch::raw_goal(double* goal) {
  const std::int64_t pull_count = static_cast<std::int64_t>(pulls_.size() / 3);
  const std::int64_t last = mode_.deep ? kLastStep : kSeedSteps;
  const double* vertex = point(0);
  while (pull_ < pull_count) {
    if (step_ <= last) {
      const double* pull = pulls_.data() + 3 * static_cast<std::size_t>(pull_);
      bool reached = true;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        // ldexp is exact, or correctly rounded where the result is subnormal.
        goal[axis] = vertex[axis] + std::ldexp(pull[axis], -static_cast<int>(step_));
        reached = reached && goal[axis] == vertex[axis];
      }
      if (!reached) {
        if (pull_ == 0 && mode_.try_first_pull_once) {
          ++pull_;
        } else {
          ++step_;
        }
        return true;
      }
    }
    ++pull_;
    step_ = 0;
  }
  return false;
}

bool RelocationSearch::floor_oriented(const double* position) const noexcept {
  double corners[12];
  for (std::int64_t cell = 0; cell < cell_count_; ++cell) {
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const std::int64_t row = cells_[4 * static_cast<std::size_t>(cell) + slot];
      const double* values = row == 0 ? position : point(row);
      for (std::size_t axis = 0; axis < 3; ++axis) corners[3 * slot + axis] = values[axis];
    }
    double determinant = 0.0;
    double relative = 0.0;
    relative_determinant(corners, determinant, relative);
    if (!(determinant > 0.0) || !(relative > determinant_floor_)) return false;
  }
  return true;
}

RelocationSearch::Verdict RelocationSearch::evaluate(const double* position,
                                                     const double* metric, double* lengths,
                                                     double& quality) {
  for (std::int64_t row = 1; row < point_count_; ++row) {
    if (same_point(position, point(row))) return Verdict::kRejected;
  }
  selected_.clear();
  for (std::int64_t spoke = 0; spoke < spoke_count_; ++spoke) {
    const std::size_t index = static_cast<std::size_t>(spoke);
    const double* neighbor = point(spoke + 1);
    const double* tensor = metrics_.data() + 9 * (index + 1);
    lengths[index] = (spoke_flags_[index] & kNeighborFirst) != 0
                         ? length(neighbor, position, tensor, metric)
                         : length(position, neighbor, metric, tensor);
    if ((spoke_flags_[index] & kMeasured) != 0) selected_.push_back(lengths[index]);
  }
  if (!band_violations(static_cast<std::int64_t>(selected_.size()), selected_.data(), long_,
                       short_)) {
    return Verdict::kUnmeasured;
  }
  const bool shape = mode_.shape_lower > 0.0;
  if (shape && (!long_.empty() || !short_.empty() ||
                (mode_.shape_lower > kLowerLength &&
                 std::ranges::any_of(selected_, [&](double value) {
                   return value < mode_.shape_lower;
                 })))) {
    return Verdict::kRejected;
  }
  const bool equal = long_ == before_long_ && short_ == before_short_;
  const bool decreased = smaller(long_, before_long_) ||
                         (long_ == before_long_ && smaller(short_, before_short_));
  if (equal ? mode_.band_only : !decreased) return Verdict::kRejected;
  if (!floor_oriented(position)) return Verdict::kRejected;
  if (!equal && !mode_.report_quality && !shape) return Verdict::kAdmitted;

  // Prepare only on the first candidate that reaches the shape test.
  if (quality_logs_.empty()) {
    quality_logs_.reserve(static_cast<std::size_t>(point_count_));
    quality_logs_.push_back(0.0);
    for (std::int64_t row = 1; row < point_count_; ++row) {
      quality_logs_.push_back(log_determinant(metrics_.data() + 9 * row));
    }
  }
  quality_logs_.front() = log_determinant(metric);

  const bool mean_shape = shape && !mode_.minimum_shape;
  double corners[12];
  double tensors[36];
  double logs[4];
  double measure = 0.0;
  bool undefined = false;
  for (std::int64_t cell = 0; cell < cell_count_; ++cell) {
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const std::int64_t row = cells_[4 * static_cast<std::size_t>(cell) + slot];
      const double* values = row == 0 ? position : point(row);
      const double* tensor =
          row == 0 ? metric : metrics_.data() + 9 * static_cast<std::size_t>(row);
      for (std::size_t axis = 0; axis < 3; ++axis) corners[3 * slot + axis] = values[axis];
      for (std::size_t entry = 0; entry < 9; ++entry) tensors[9 * slot + entry] = tensor[entry];
      logs[slot] = quality_logs_[static_cast<std::size_t>(row)];
    }
    const double value = tetra_quality(corners, tensors, logs);
    if (shape && !std::isfinite(value)) return Verdict::kRejected;
    if (mean_shape) {
      measure += value;
    } else {
      if (value != value) undefined = true;
      if (cell == 0 || value < measure) measure = value;
    }
  }
  if (mean_shape) measure /= static_cast<double>(cell_count_);
  quality = undefined ? std::numeric_limits<double>::quiet_NaN() : measure;
  if (equal && !(quality > before_quality_ + improvement_)) return Verdict::kRejected;
  return Verdict::kAdmitted;
}

RelocationEnd RelocationSearch::next(std::vector<RelocationCandidate>& group,
                                     std::vector<double>& lengths) {
  const double* vertex = point(0);
  const auto report = [&](const RelocationCandidate& entry, bool measured) {
    group.assign(1, entry);
    lengths.clear();
    if (measured) lengths.assign(scratch_.begin(), scratch_.end());
  };
  for (;;) {
    group.clear();
    lengths.clear();
    double goal[3];
    bool remaining = false;
    if (!second_pass_) {
      double raw[3];
      if (!raw_goal(raw)) {
        if (!mode_.refine) return RelocationEnd::kExhausted;
        second_pass_ = true;
        continue;
      }
      if (framed_) {
        project_goals(1, raw, origin_.data(), projector_.data(), goal);
      } else {
        std::copy(raw, raw + 3, goal);
      }
      const std::array<std::uint64_t, 3> key = {std::bit_cast<std::uint64_t>(goal[0]),
                                                std::bit_cast<std::uint64_t>(goal[1]),
                                                std::bit_cast<std::uint64_t>(goal[2])};
      if (!seen_.insert(key).second) continue;
      if (mode_.constant_field) {
        double quality = 0.0;
        const Verdict screened = evaluate(goal, metrics_.data(), scratch_.data(), quality);
        if (screened == Verdict::kUnmeasured) {
          RelocationCandidate entry;
          std::copy(goal, goal + 3, entry.carrier.point.begin());
          std::copy(metrics_.begin(), metrics_.begin() + 9, entry.metric.begin());
          report(entry, true);
          return RelocationEnd::kUnmeasuredGoal;
        }
        if (screened != Verdict::kAdmitted) continue;
      } else if (!floor_oriented(goal)) {
        continue;
      }
      if (mode_.refine) admitted_goals_.insert(admitted_goals_.end(), goal, goal + 3);
    } else {
      if (3 * revisit_ >= admitted_goals_.size()) return RelocationEnd::kExhausted;
      std::copy(admitted_goals_.begin() + static_cast<std::ptrdiff_t>(3 * revisit_),
                admitted_goals_.begin() + static_cast<std::ptrdiff_t>(3 * revisit_ + 3), goal);
      ++revisit_;
      remaining = true;
    }

    CarrierOrder order = CarrierOrder::kCoarseFirst;
    std::int64_t limit = 1;
    std::size_t first = 0;
    if (mode_.refine) {
      order = CarrierOrder::kFineFirst;
      if (remaining) {
        limit = kAllCarriers;
        first = 1;
      }
    } else if (mode_.deep) {
      order = CarrierOrder::kCoarseThenFine;
      limit = 2;
    }
    const std::vector<Carrier> carriers =
        source_.carriers(goal, vertex, chart_, -1, order, limit);
    for (std::size_t index = first; index < carriers.size(); ++index) {
      const Carrier& carrier = carriers[index];
      const double* position = carrier.point.data();
      bool occupied = same_point(position, vertex);
      for (std::int64_t row = 1; row < point_count_ && !occupied; ++row) {
        occupied = same_point(position, point(row));
      }
      if (occupied) continue;
      // A common reduced denominator identifies the same fractions as
      // reducing each term separately. Positive weights distinguish active
      // rows from zero padding, even when source row zero is used.
      std::uint64_t combined = 0;
      const auto count = static_cast<std::size_t>(carrier.size);
      for (std::size_t row = 0; row < count; ++row) {
        combined |= static_cast<std::uint64_t>(carrier.weights[row]);
      }
      const int shift = std::countr_zero(combined);
      std::array<std::int64_t, 9> stencil{};
      stencil[0] = carrier.bits - shift;
      for (std::size_t row = 0; row < count; ++row) {
        stencil[1 + 2 * row] = carrier.rows[row];
        stencil[2 + 2 * row] = carrier.weights[row] >> shift;
      }
      if (!examined_.insert(stencil).second) continue;

      RelocationCandidate candidate;
      candidate.carrier = carrier;
      double weights[4];
      double tensors[36];
      double logarithms[36];
      const double scale = static_cast<double>(std::int64_t{1} << carrier.bits);
      for (int slot = 0; slot < carrier.size; ++slot) {
        const std::size_t at = static_cast<std::size_t>(slot);
        // An integer of at most 53 bits over a power of two: exact.
        weights[at] = static_cast<double>(carrier.weights[at]) / scale;
        const std::int64_t row = carrier.rows[at];
        for (std::size_t entry = 0; entry < 9; ++entry) {
          tensors[9 * at + entry] = source_metrics_[9 * row + static_cast<std::int64_t>(entry)];
          logarithms[9 * at + entry] =
              source_logarithms_[9 * row + static_cast<std::int64_t>(entry)];
        }
      }
      if (!interpolate(carrier.size, weights, tensors, logarithms, candidate.metric.data())) {
        report(candidate, false);
        return RelocationEnd::kNoTensor;
      }
      const Verdict verdict =
          evaluate(position, candidate.metric.data(), scratch_.data(), candidate.quality);
      if (verdict == Verdict::kUnmeasured) {
        report(candidate, true);
        return RelocationEnd::kUnmeasuredCarrier;
      }
      if (verdict != Verdict::kAdmitted) continue;
      group.push_back(candidate);
      lengths.insert(lengths.end(), scratch_.begin(), scratch_.end());
    }
    if (!group.empty()) return RelocationEnd::kGroup;
  }
}

}  // namespace phx::mc::metric
