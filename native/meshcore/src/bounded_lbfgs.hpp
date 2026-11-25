// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <utility>

namespace phx::mc::detail {

enum class BoundedLbfgsStatus { kConverged, kStalled, kIterationLimit, kInvalidObjective };

struct BoundedLbfgsResult {
  BoundedLbfgsStatus status;
  std::int32_t steps;  // Accepted steps, not callback evaluations.
  double value;        // Last accepted merit, or NaN if the initial state was refused.
};

class BoundedLbfgsHistory {
 public:
  static constexpr std::size_t kCapacity = 4;

  BoundedLbfgsHistory(std::span<double> storage, std::size_t width) {
    for (auto& pair : pairs_) {
      pair.step = storage.first(width);
      pair.change = storage.subspan(width, width);
      storage = storage.subspan(2 * width);
    }
  }

  void direction(std::span<const double> gradient, double initial_inverse_scale,
                 std::span<double> result) const {
    std::copy(gradient.begin(), gradient.end(), result.begin());
    std::array<double, kCapacity> alpha;
    for (std::size_t k = 0; k < count_; ++k) {
      const auto& pair = newest(k);
      alpha[k] = pair.inverse_curvature * dot(pair.step, result);
      for (std::size_t i = 0; i < result.size(); ++i) result[i] -= alpha[k] * pair.change[i];
    }
    const double scale = count_ == 0 ? initial_inverse_scale : newest(0).inverse_scale;
    for (double& component : result) component *= scale;
    for (std::size_t k = count_; k > 0; --k) {
      const auto& pair = newest(k - 1);
      const double weight = alpha[k - 1] - pair.inverse_curvature * dot(pair.change, result);
      for (std::size_t i = 0; i < result.size(); ++i) result[i] += weight * pair.step[i];
    }
    for (double& component : result) component = -component;
  }

  void update(std::span<const double> position, std::span<const double> trial,
              std::span<const double> gradient, std::span<const double> trial_gradient) {
    double curvature = 0.0;
    double change_sq = 0.0;
    for (std::size_t i = 0; i < position.size(); ++i) {
      const double change = trial_gradient[i] - gradient[i];
      curvature += (trial[i] - position[i]) * change;
      change_sq += change * change;
    }
    // Use the native search's absolute curvature floor; reject degenerate scales
    // before overwriting an older pair.
    if (!(curvature > 1e-18) || !std::isfinite(curvature) || !std::isfinite(change_sq)) return;
    const double scale = curvature / change_sq;
    if (!(scale > 0.0) || !std::isfinite(scale)) return;
    auto& pair = pairs_[next_];
    for (std::size_t i = 0; i < position.size(); ++i) {
      pair.step[i] = trial[i] - position[i];
      pair.change[i] = trial_gradient[i] - gradient[i];
    }
    pair.inverse_curvature = 1.0 / curvature;
    pair.inverse_scale = scale;
    next_ = (next_ + 1) % kCapacity;
    count_ = std::min(count_ + 1, kCapacity);
  }

 private:
  struct Pair {
    std::span<double> step;
    std::span<double> change;
    double inverse_curvature;
    double inverse_scale;
  };

  static double dot(std::span<const double> a, std::span<const double> b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  }

  const Pair& newest(std::size_t age) const {
    return pairs_[(next_ + kCapacity - 1 - age) % kCapacity];
  }

  std::array<Pair, kCapacity> pairs_;
  std::size_t next_ = 0;
  std::size_t count_ = 0;
};

namespace bounded_lbfgs_impl {

inline constexpr double kInvalidValue = std::numeric_limits<double>::quiet_NaN();
inline constexpr std::size_t kCoreBuffers = 4 + 2 * BoundedLbfgsHistory::kCapacity;

template <bool Projected, class BoundsAt, class Objective, class Stop>
BoundedLbfgsResult solve(std::span<double> position, BoundsAt bounds_at, double initial_inverse_scale,
                         std::int32_t maximum_steps, double projected_tolerance,
                         std::span<double> work, Objective&& objective, Stop&& stop) {
  using Status = BoundedLbfgsStatus;
  constexpr double kMinimumFraction = Projected ? 0x1p-15 : 0x1p-7;
  const auto finite = [](double x) { return std::isfinite(x); };
  const std::size_t n = position.size();
  auto gradient = work.first(n);
  auto trial_gradient = work.subspan(n, n);
  const auto trial = work.subspan(2 * n, n);
  const auto direction = work.subspan(3 * n, n);
  BoundedLbfgsHistory history(work.subspan(4 * n), n);
  const auto evaluate = [&](std::span<const double> x, std::span<double> grad) {
    const double value = objective(x, grad);
    return finite(value) && std::all_of(grad.begin(), grad.end(), finite) ? value : kInvalidValue;
  };

  double value = evaluate(position, gradient);
  if (!finite(value)) return {Status::kInvalidObjective, 0, kInvalidValue};
  if (stop(value)) return {Status::kConverged, 0, value};
  for (std::int32_t steps = 0;; ++steps) {
    std::span<const double> direction_gradient = gradient;
    if constexpr (Projected) {
      const auto projected = work.subspan(kCoreBuffers * n, n);
      double norm = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const auto [lo, hi] = bounds_at(i);
        projected[i] = std::clamp(gradient[i], position[i] - hi, position[i] - lo);
        norm = std::max(norm, std::abs(projected[i]));
      }
      if (norm <= projected_tolerance) return {Status::kConverged, steps, value};
      direction_gradient = projected;
    } else {
      bool can_descend = false;
      for (std::size_t i = 0; i < n && !can_descend; ++i) {
        const auto [lo, hi] = bounds_at(i);
        can_descend = (gradient[i] > 0.0 && position[i] > lo) ||
                      (gradient[i] < 0.0 && position[i] < hi);
      }
      if (!can_descend) return {Status::kConverged, steps, value};
    }
    if (steps == maximum_steps) return {Status::kIterationLimit, steps, value};

    history.direction(direction_gradient, initial_inverse_scale, direction);
    double next_value = kInvalidValue;
    bool invalid_trial = false;
    bool finite_trial = false;
    const auto line_search = [&]() {
      if (!std::all_of(direction.begin(), direction.end(), finite)) return false;
      for (double fraction = 1.0; fraction >= kMinimumFraction; fraction *= 0.5) {
        double slope = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          const auto [lo, hi] = bounds_at(i);
          trial[i] = std::clamp(position[i] + fraction * direction[i], lo, hi);
          slope += gradient[i] * (trial[i] - position[i]);
        }
        if (!finite(slope) || !(slope < 0.0)) continue;
        next_value = evaluate(trial, trial_gradient);
        if (!finite(next_value)) {
          invalid_trial = true;
          continue;
        }
        finite_trial = true;
        // Projection changes the step, so use its actual directional derivative.
        if (next_value < value && next_value <= value + 1e-4 * slope) return true;
      }
      return false;
    };
    bool accepted = line_search();
    if constexpr (Projected) {
      if (!accepted) {
        for (std::size_t i = 0; i < n; ++i) {
          direction[i] = -initial_inverse_scale * direction_gradient[i];
        }
        accepted = line_search();
      }
    }
    if (!accepted) {
      if constexpr (Projected) {
        // The scientific stop and subsequent stages consume objective scratch.
        // Even an invalid rejected trial can have overwritten part of it.
        if ((invalid_trial || finite_trial) && !finite(evaluate(position, gradient))) {
          return {Status::kInvalidObjective, steps, value};
        }
        return {Status::kStalled, steps, value};
      } else {
        return {invalid_trial && !finite_trial ? Status::kInvalidObjective : Status::kStalled,
                steps, value};
      }
    }
    const bool satisfied = stop(next_value);
    if (steps + 1 < maximum_steps && !satisfied) {
      history.update(position, trial, gradient, trial_gradient);
    }
    std::copy(trial.begin(), trial.end(), position.begin());
    std::swap(gradient, trial_gradient);
    value = next_value;
    if (satisfied) return {Status::kConverged, steps + 1, value};
  }
}

}  // namespace bounded_lbfgs_impl

// The objective writes every gradient component. Its scratch may describe a
// rejected trial on return; only position and the result record are committed.
// value_tolerance is an absolute target merit, not a step-size threshold.
template <class Objective>
BoundedLbfgsResult bounded_lbfgs(std::span<double> position, double component_bound,
                                 double initial_inverse_scale, std::int32_t maximum_steps,
                                 double value_tolerance, Objective&& objective) {
  using namespace bounded_lbfgs_impl;
  const auto finite = [](double x) { return std::isfinite(x); };
  if (!(component_bound > 0.0) || !finite(component_bound) || !(initial_inverse_scale > 0.0) ||
      !finite(initial_inverse_scale) || maximum_steps < 0 || !(value_tolerance >= 0.0) ||
      !finite(value_tolerance) ||
      position.size() > std::numeric_limits<std::size_t>::max() / sizeof(double) / kCoreBuffers ||
      !std::all_of(position.begin(), position.end(), [component_bound](double x) {
        return x >= -component_bound && x <= component_bound;
      })) {
    return {BoundedLbfgsStatus::kInvalidObjective, 0, kInvalidValue};
  }

  const std::size_t count = kCoreBuffers * position.size();
  const auto storage = std::make_unique_for_overwrite<double[]>(count);
  const auto bounds_at = [=](std::size_t) { return std::array{-component_bound, component_bound}; };
  return solve<false>(position, bounds_at, initial_inverse_scale, maximum_steps, 0.0,
                      {storage.get(), count}, std::forward<Objective>(objective),
                      [=](double value) { return value <= value_tolerance; });
}

// Asymmetric intervals, projected stationarity and a projected-gradient retry.
// stop() reads objective scratch at the initial point and after each accepted
// trial. A stalled solve restores that scratch to the last accepted position.
// Reusable work holds (5 + 2 * BoundedLbfgsHistory::kCapacity) doubles per DOF.
template <class Interval, class Objective, class Stop>
BoundedLbfgsResult projected_bounded_lbfgs(std::span<double> position, std::span<const Interval> bounds,
                                           double initial_inverse_scale, std::int32_t maximum_steps,
                                           double projected_tolerance, std::span<double> work,
                                           Objective&& objective, Stop&& stop) {
  using namespace bounded_lbfgs_impl;
  if (bounds.size() != position.size() || position.size() > work.size() / (kCoreBuffers + 1) ||
      !(initial_inverse_scale > 0.0) || !std::isfinite(initial_inverse_scale) || maximum_steps < 0 ||
      !(projected_tolerance >= 0.0) || !std::isfinite(projected_tolerance)) {
    return {BoundedLbfgsStatus::kInvalidObjective, 0, kInvalidValue};
  }
  for (std::size_t i = 0; i < bounds.size(); ++i) {
    const auto [lo, hi] = bounds[i];
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo <= position[i] && position[i] <= hi)) {
      return {BoundedLbfgsStatus::kInvalidObjective, 0, kInvalidValue};
    }
  }

  return solve<true>(position, [=](std::size_t i) { return bounds[i]; }, initial_inverse_scale,
                     maximum_steps, projected_tolerance, work, std::forward<Objective>(objective),
                     [&](double) { return stop(); });
}

}  // namespace phx::mc::detail
