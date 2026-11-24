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

// The objective writes every gradient component. Its scratch may describe a
// rejected trial on return; only position and the result record are committed.
// value_tolerance is an absolute target merit, not a step-size threshold.
template <class Objective>
BoundedLbfgsResult bounded_lbfgs(std::span<double> position, double component_bound,
                                 double initial_inverse_scale, std::int32_t maximum_steps,
                                 double value_tolerance, Objective&& objective) {
  using Status = BoundedLbfgsStatus;
  constexpr double kInvalidValue = std::numeric_limits<double>::quiet_NaN();
  constexpr std::size_t kBufferCount = 4 + 2 * BoundedLbfgsHistory::kCapacity;
  const auto finite = [](double x) { return std::isfinite(x); };
  if (!(component_bound > 0.0) || !finite(component_bound) || !(initial_inverse_scale > 0.0) ||
      !finite(initial_inverse_scale) || maximum_steps < 0 || !(value_tolerance >= 0.0) ||
      !finite(value_tolerance) ||
      position.size() > std::numeric_limits<std::size_t>::max() / sizeof(double) / kBufferCount ||
      !std::all_of(position.begin(), position.end(), [component_bound](double x) {
        return x >= -component_bound && x <= component_bound;
      })) {
    return {Status::kInvalidObjective, 0, kInvalidValue};
  }

  const std::size_t n = position.size();
  auto storage = std::make_unique_for_overwrite<double[]>(kBufferCount * n);
  const std::span<double> work{storage.get(), kBufferCount * n};
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
  for (std::int32_t steps = 0;; ++steps) {
    if (value <= value_tolerance) return {Status::kConverged, steps, value};
    bool can_descend = false;
    for (std::size_t i = 0; i < n && !can_descend; ++i) {
      can_descend = (gradient[i] > 0.0 && position[i] > -component_bound) ||
                    (gradient[i] < 0.0 && position[i] < component_bound);
    }
    if (!can_descend) return {Status::kConverged, steps, value};
    if (steps == maximum_steps) return {Status::kIterationLimit, steps, value};

    history.direction(gradient, initial_inverse_scale, direction);
    double next_value = kInvalidValue;
    bool invalid_trial = false;
    bool finite_trial = false;
    bool accepted = false;
    if (std::all_of(direction.begin(), direction.end(), finite)) {
      for (double fraction = 1.0; fraction >= 1.0 / 128.0; fraction *= 0.5) {
        double slope = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          trial[i] = std::clamp(position[i] + fraction * direction[i],
                                -component_bound, component_bound);
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
        if (next_value < value && next_value <= value + 1e-4 * slope) {
          accepted = true;
          break;
        }
      }
    }
    if (!accepted) {
      return {invalid_trial && !finite_trial ? Status::kInvalidObjective : Status::kStalled,
              steps, value};
    }
    if (steps + 1 < maximum_steps && next_value > value_tolerance) {
      history.update(position, trial, gradient, trial_gradient);
    }
    std::copy(trial.begin(), trial.end(), position.begin());
    std::swap(gradient, trial_gradient);
    value = next_value;
  }
}

}  // namespace phx::mc::detail
