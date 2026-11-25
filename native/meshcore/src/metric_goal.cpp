//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Floating-point goals of metric mesh edits; see metric_goal.hpp. Every sum
// below is written in the order it is evaluated.
#include "metric_goal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "bounded_lbfgs.hpp"
#include "deterministic_float.hpp"
#include "metric_float.hpp"

namespace phx::mc::metric {
namespace {

double magnitude(double value) noexcept { return value < 0.0 ? -value : value; }
bool finite(double value) noexcept {
  return magnitude(value) <= std::numeric_limits<double>::max();
}
bool finite3(const double* values) noexcept {
  return finite(values[0]) && finite(values[1]) && finite(values[2]);
}

double dot3(const double* a, const double* b) noexcept {
  return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

double squared_distance(const double* a, const double* b) noexcept {
  const double x = a[0] - b[0];
  const double y = a[1] - b[1];
  const double z = a[2] - b[2];
  return (x * x + y * y) + z * z;
}

// Smallest enclosing ball of mapped points by Welzl's recursion on an
// explicit stack. `center` and `radius` (squared, negative for the empty
// ball) are the ball of the last evaluated boundary set.
class EnclosingBall {
 public:
  EnclosingBall(const std::vector<double>& mapped, double slack)
      : mapped_(mapped), slack_(slack) {}

  bool run(double* center) {
    struct Frame {
      std::int64_t count;
      int boundary_size;
      std::int64_t boundary[4];
      bool returning;
    };
    const std::int64_t total = static_cast<std::int64_t>(mapped_.size() / 3);
    std::vector<Frame> stack;
    stack.push_back({total, 0, {0, 0, 0, 0}, false});
    while (!stack.empty()) {
      const Frame frame = stack.back();
      stack.pop_back();
      if (frame.returning) {
        const double distance = squared_distance(point(frame.count - 1), center_);
        if (radius_ < 0.0 || distance > radius_ + slack_) {
          Frame child = frame;
          child.count = frame.count - 1;
          child.boundary[child.boundary_size] = frame.count - 1;
          child.boundary_size += 1;
          child.returning = false;
          stack.push_back(child);
        }
      } else if (frame.count == 0 || frame.boundary_size == 4) {
        if (!boundary_ball(frame.boundary, frame.boundary_size)) return false;
      } else {
        Frame after = frame;
        after.returning = true;
        stack.push_back(after);
        Frame without = frame;
        without.count = frame.count - 1;
        stack.push_back(without);
      }
    }
    for (std::size_t axis = 0; axis < 3; ++axis) center[axis] = center_[axis];
    return true;
  }

 private:
  const double* point(std::int64_t index) const noexcept {
    return mapped_.data() + 3 * static_cast<std::size_t>(index);
  }

  // Circumcenter of the selected points in their affine hull: the first point
  // plus the combination x of the edges d_j from it with G x = diag(G) / 2,
  // G the matrix of their scalar products.
  bool circumcenter(const std::int64_t* selected, int size, double* center) const noexcept {
    const double* origin = point(selected[0]);
    const int order = size - 1;
    double directions[3][3];
    for (int row = 0; row < order; ++row) {
      const double* end = point(selected[row + 1]);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        directions[row][axis] = end[axis] - origin[axis];
      }
    }
    double gram[9];
    double right[3];
    for (int row = 0; row < order; ++row) {
      for (int column = row; column < order; ++column) {
        gram[order * row + column] = dot3(directions[row], directions[column]);
      }
      right[row] = dot3(directions[row], directions[row]) * 0.5;
    }
    double coefficients[3] = {0.0, 0.0, 0.0};
    if (deterministic::symmetric_pseudo_solve(order, gram, right, coefficients) < 0) return false;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      double offset = coefficients[0] * directions[0][axis];
      for (int row = 1; row < order; ++row) {
        offset = offset + coefficients[row] * directions[row][axis];
      }
      center[axis] = origin[axis] + offset;
    }
    return finite3(center);
  }

  // The smallest ball, over the subsets of the boundary set by size and then
  // in lexicographic order, of the circumscribed ball of the subset that holds
  // the whole set. Subsets of every size also cover coincident, collinear and
  // coplanar boundary points.
  bool boundary_ball(const std::int64_t* boundary, int size) noexcept {
    if (size == 0) {
      center_[0] = center_[1] = center_[2] = 0.0;
      radius_ = -1.0;
      return true;
    }
    double best_center[3] = {point(boundary[0])[0], point(boundary[0])[1],
                             point(boundary[0])[2]};
    double best_radius = std::numeric_limits<double>::infinity();
    for (int subset = 1; subset <= size; ++subset) {
      int chosen[4] = {0, 1, 2, 3};
      while (true) {
        std::int64_t selected[4];
        for (int index = 0; index < subset; ++index) selected[index] = boundary[chosen[index]];
        double center[3];
        if (subset == 1) {
          for (std::size_t axis = 0; axis < 3; ++axis) center[axis] = point(selected[0])[axis];
        } else if (!circumcenter(selected, subset, center)) {
          return false;
        }
        double radius = 0.0;
        for (int index = 0; index < size; ++index) {
          const double distance = squared_distance(point(boundary[index]), center);
          if (index == 0 || distance > radius) radius = distance;
        }
        if (radius < best_radius) {
          best_radius = radius;
          for (std::size_t axis = 0; axis < 3; ++axis) best_center[axis] = center[axis];
        }
        // The next combination of `subset` positions out of `size`.
        int position = subset - 1;
        while (position >= 0 && chosen[position] == size - subset + position) --position;
        if (position < 0) break;
        chosen[position] += 1;
        for (int index = position + 1; index < subset; ++index) {
          chosen[index] = chosen[index - 1] + 1;
        }
      }
    }
    for (std::size_t axis = 0; axis < 3; ++axis) center_[axis] = best_center[axis];
    radius_ = best_radius;
    return true;
  }

  const std::vector<double>& mapped_;
  double slack_;
  double center_[3] = {0.0, 0.0, 0.0};
  double radius_ = -1.0;
};

}  // namespace

double midpoint_parameter(const double* p, const double* q, const double* metric_p,
                          const double* metric_q) noexcept {
  const double delta[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
  const double first = deterministic::quadratic_form3(delta, metric_p);
  const double second = deterministic::quadratic_form3(delta, metric_q);
  if (!finite(first + second) || !(first > 0.0 && second > 0.0)) return 0.5;
  const double ratio = std::sqrt(second / first);
  if (!(ratio > 0.0 && finite(ratio))) return 0.5;
  if (ratio == 1.0) return 0.5;
  const double excess = ratio - 1.0;
  return deterministic::log1p(0.5 * excess) / deterministic::log1p(excess);
}

bool enclosing_center(std::int64_t count, const double* points, const double* metric,
                      double* center) {
  if (count < 1) return false;
  double factor[9];
  if (!deterministic::cholesky3(metric, factor)) return false;
  // Welzl's recursion has poor worst-case order on structured point sets. A
  // fixed permutation (Fisher-Yates from Knuth's 64-bit linear congruential
  // generator, integer arithmetic only) keeps the result reproducible while
  // avoiding the systematic geometric order of mesh vertices.
  std::vector<std::int64_t> order(static_cast<std::size_t>(count));
  for (std::int64_t index = 0; index < count; ++index) {
    order[static_cast<std::size_t>(index)] = index;
  }
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (std::int64_t last = count - 1; last > 0; --last) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::int64_t other =
        static_cast<std::int64_t>((state >> 33) % static_cast<std::uint64_t>(last + 1));
    const std::int64_t swap = order[static_cast<std::size_t>(last)];
    order[static_cast<std::size_t>(last)] = order[static_cast<std::size_t>(other)];
    order[static_cast<std::size_t>(other)] = swap;
  }
  std::vector<double> mapped(3 * static_cast<std::size_t>(count));
  double scale = std::numeric_limits<double>::min();
  for (std::int64_t index = 0; index < count; ++index) {
    double* image = mapped.data() + 3 * static_cast<std::size_t>(index);
    deterministic::multiply_upper3(factor, points + 3 * order[static_cast<std::size_t>(index)],
                                   image);
    if (!finite3(image)) return false;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (magnitude(image[axis]) > scale) scale = magnitude(image[axis]);
    }
  }
  const double slack = 0x1p-47 * (scale * scale);
  if (!finite(slack)) return false;
  double mapped_center[3];
  if (!EnclosingBall(mapped, slack).run(mapped_center)) return false;
  double result[3];
  deterministic::solve_upper3(factor, mapped_center, result);
  if (!finite3(result)) return false;
  for (std::size_t axis = 0; axis < 3; ++axis) center[axis] = result[axis];
  return true;
}

bool hull_projection(int corner_count, const double* corners, const double* goal,
                     double* projected) noexcept {
  const int order = corner_count - 1;
  double directions[2][3];
  double offset[3];
  for (std::size_t axis = 0; axis < 3; ++axis) offset[axis] = goal[axis] - corners[axis];
  for (int row = 0; row < order; ++row) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      directions[row][axis] = corners[3 * (row + 1) + axis] - corners[axis];
    }
  }
  // The solve reads the upper triangle of the leading order x order block
  // only; the zeros define the remaining entries for any order.
  double gram[4] = {0.0, 0.0, 0.0, 0.0};
  double right[2] = {0.0, 0.0};
  for (int row = 0; row < order; ++row) {
    for (int column = row; column < order; ++column) {
      gram[order * row + column] = dot3(directions[row], directions[column]);
    }
    right[row] = dot3(directions[row], offset);
  }
  double coefficients[2] = {0.0, 0.0};
  if (deterministic::symmetric_pseudo_solve(order, gram, right, coefficients) < 0) return false;
  double result[3];
  for (std::size_t axis = 0; axis < 3; ++axis) {
    double step = coefficients[0] * directions[0][axis];
    for (int row = 1; row < order; ++row) step = step + coefficients[row] * directions[row][axis];
    result[axis] = corners[axis] + step;
  }
  if (!finite3(result)) return false;
  for (std::size_t axis = 0; axis < 3; ++axis) projected[axis] = result[axis];
  return true;
}

void survivor_goals(std::int64_t count, const double* points, const double* metrics,
                    int corner_count, const double* corners, double* goals, bool* valid) {
  const double* survivor = points;
  const double* removed = points + 3;
  const double parameter = midpoint_parameter(survivor, removed, metrics, metrics + 9);
  const double fractions[2] = {parameter, 0.5 * parameter};
  for (std::size_t goal = 0; goal < 2; ++goal) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      goals[3 * goal + axis] =
          survivor[axis] + fractions[goal] * (removed[axis] - survivor[axis]);
    }
    valid[goal] = finite3(goals + 3 * goal);
  }
  const double weights[2] = {0.5, 0.5};
  double mean[9];
  valid[2] = count > 2 && interpolate(2, weights, metrics, nullptr, mean) &&
             enclosing_center(count - 2, points + 6, mean, goals + 6);
  for (std::size_t goal = 0; goal < 3; ++goal) {
    if (valid[goal] && corner_count > 0) {
      valid[goal] = hull_projection(corner_count, corners, goals + 3 * goal, goals + 3 * goal);
    }
    if (!valid[goal]) {
      for (std::size_t axis = 0; axis < 3; ++axis) goals[3 * goal + axis] = 0.0;
    }
  }
}

int hull_frame(int corner_count, const double* corners, double* basis,
               double* projector) noexcept {
  const int order = corner_count - 1;
  double directions[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int row = 0; row < order; ++row) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      directions[3 * static_cast<std::size_t>(row) + axis] =
          corners[3 * static_cast<std::size_t>(row + 1) + axis] - corners[axis];
    }
  }
  const int rank = deterministic::orthonormal_basis3(order, directions, basis);
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      double sum = 0.0;
      for (int vector = 0; vector < rank; ++vector) {
        const double product = basis[3 * static_cast<std::size_t>(vector) + row] *
                               basis[3 * static_cast<std::size_t>(vector) + column];
        sum = vector == 0 ? product : sum + product;
      }
      projector[3 * row + column] = sum;
    }
  }
  return rank;
}

void project_goals(std::int64_t count, const double* goals, const double* origin,
                   const double* projector, double* projected) noexcept {
  for (std::int64_t row = 0; row < count; ++row) {
    const double offset[3] = {goals[3 * row] - origin[0], goals[3 * row + 1] - origin[1],
                              goals[3 * row + 2] - origin[2]};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      projected[3 * row + static_cast<std::int64_t>(axis)] =
          origin[axis] + dot3(projector + 3 * axis, offset);
    }
  }
}

std::int64_t relocation_pulls(std::int64_t count, const double* points, const double* metric,
                              const double* lengths, const std::int8_t* measured, bool deep,
                              bool interior, int basis_count, const double* basis,
                              const double* projector, const double* jacobian,
                              std::int64_t capacity, double* pulls) {
  const std::size_t size = static_cast<std::size_t>(count);
  std::int64_t written = 0;
  bool overflow = false;
  const auto emit = [&](const double* pull) {
    if (!finite3(pull)) return;
    if (written >= capacity) {
      overflow = true;
      return;
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      pulls[3 * written + static_cast<std::int64_t>(axis)] = pull[axis];
    }
    ++written;
  };

  // Spokes, spring forces and their mean, in neighbor order.
  std::vector<double> spokes(3 * size);
  std::vector<double> forces(3 * size);
  std::vector<std::size_t> outside;
  double mean[3] = {0.0, 0.0, 0.0};
  for (std::size_t spoke = 0; spoke < size; ++spoke) {
    const double factor = 1.0 - 1.0 / lengths[spoke];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double component = points[3 * (spoke + 1) + axis] - points[axis];
      const double force = factor * component;
      spokes[3 * spoke + axis] = component;
      forces[3 * spoke + axis] = force;
      mean[axis] = spoke == 0 ? force : mean[axis] + force;
    }
    if (measured[spoke] != 0 && (lengths[spoke] < kLowerLength || lengths[spoke] > kUpperLength)) {
      outside.push_back(spoke);
    }
  }
  for (double& value : mean) value = value / static_cast<double>(count);
  const bool searching = deep && !outside.empty();

  if (searching && jacobian != nullptr && basis_count > 0) {
    // Normal equations of the step x along the basis that moves the measured
    // lengths by their residuals to the band: (J J^T) x = J r, summed over
    // the measured spokes in order.
    const std::size_t order = static_cast<std::size_t>(basis_count);
    double normal[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double right[3] = {0.0, 0.0, 0.0};
    bool first = true;
    for (std::size_t spoke = 0; spoke < size; ++spoke) {
      if (measured[spoke] == 0) continue;
      const double length = lengths[spoke];
      const double target =
          length < kLowerLength ? kLowerLength : (length > kUpperLength ? kUpperLength : length);
      const double residual = target - length;
      for (std::size_t row = 0; row < order; ++row) {
        const double derivative = jacobian[row * size + spoke];
        const double product = derivative * residual;
        right[row] = first ? product : right[row] + product;
        for (std::size_t column = row; column < order; ++column) {
          const double entry = derivative * jacobian[column * size + spoke];
          normal[order * row + column] = first ? entry : normal[order * row + column] + entry;
        }
      }
      first = false;
    }
    double step[3] = {0.0, 0.0, 0.0};
    if (!first && deterministic::symmetric_pseudo_solve(basis_count, normal, right, step) >= 0) {
      double joint[3];
      for (std::size_t axis = 0; axis < 3; ++axis) {
        double sum = step[0] * basis[axis];
        for (std::size_t row = 1; row < order; ++row) sum = sum + step[row] * basis[3 * row + axis];
        joint[axis] = sum;
      }
      if (joint[0] != 0.0 || joint[1] != 0.0 || joint[2] != 0.0) emit(joint);
    }
    for (const std::size_t spoke : outside) {
      double direction[3];
      for (std::size_t axis = 0; axis < 3; ++axis) {
        double sum = jacobian[spoke] * basis[axis];
        for (std::size_t row = 1; row < order; ++row) {
          sum = sum + jacobian[row * size + spoke] * basis[3 * row + axis];
        }
        direction[axis] = sum;
      }
      const double norm = std::sqrt(dot3(direction, direction));
      if (!(norm > 0.0 && finite(norm))) continue;
      const double* force = forces.data() + 3 * spoke;
      const double scale = std::sqrt(dot3(force, force)) / norm;
      const double sign = lengths[spoke] < kLowerLength ? 1.0 : -1.0;
      const double pull[3] = {sign * (direction[0] * scale), sign * (direction[1] * scale),
                              sign * (direction[2] * scale)};
      emit(pull);
    }
  }

  if (!deep) emit(mean);
  for (const std::size_t spoke : outside) emit(forces.data() + 3 * spoke);
  if (deep) emit(mean);

  if (searching) {
    // Band normals n_j = (p_j - p_0)^T M of the measured spokes, nearest band
    // edge first (ties in spoke order): a normal points toward its neighbor,
    // so motion along it shortens the spoke.
    const auto row_times = [](const double* row, const double* matrix, double* product) {
      const double result[3] = {
          (row[0] * matrix[0] + row[1] * matrix[3]) + row[2] * matrix[6],
          (row[0] * matrix[1] + row[1] * matrix[4]) + row[2] * matrix[7],
          (row[0] * matrix[2] + row[1] * matrix[5]) + row[2] * matrix[8]};
      for (std::size_t axis = 0; axis < 3; ++axis) product[axis] = result[axis];
    };
    std::vector<std::size_t> active;
    std::vector<double> margins(size, std::numeric_limits<double>::infinity());
    for (std::size_t spoke = 0; spoke < size; ++spoke) {
      if (measured[spoke] == 0) continue;
      const double low = magnitude(lengths[spoke] - kLowerLength);
      const double high = magnitude(lengths[spoke] - kUpperLength);
      const double margin = low < high ? low : high;
      if (margin == margin) margins[spoke] = margin;
      active.push_back(spoke);
    }
    std::stable_sort(active.begin(), active.end(), [&](std::size_t first, std::size_t second) {
      return margins[first] < margins[second];
    });
    std::vector<double> normals(3 * active.size());
    for (std::size_t index = 0; index < active.size(); ++index) {
      double* normal = normals.data() + 3 * index;
      row_times(spokes.data() + 3 * active[index], metric, normal);
      if (projector != nullptr) row_times(normal, projector, normal);
    }
    for (const std::size_t spoke : outside) {
      double direction[3] = {forces[3 * spoke], forces[3 * spoke + 1], forces[3 * spoke + 2]};
      if (projector != nullptr) row_times(direction, projector, direction);
      for (std::size_t index = 0; index < active.size(); ++index) {
        const double* normal = normals.data() + 3 * index;
        const double squared = dot3(normal, normal);
        const double component = dot3(direction, normal);
        const double length = lengths[active[index]];
        const bool crosses =
            (length < 1.0 && component > 0.0) || (length > 1.0 && component < 0.0);
        if (!(crosses && squared > 0.0)) continue;
        const double share = component / squared;
        const double tangent[3] = {direction[0] - share * normal[0],
                                   direction[1] - share * normal[1],
                                   direction[2] - share * normal[2]};
        emit(tangent);
      }
      if (!interior) continue;
      for (std::size_t left = 0; left < active.size(); ++left) {
        for (std::size_t right = left + 1; right < active.size(); ++right) {
          const double* a = normals.data() + 3 * left;
          const double* b = normals.data() + 3 * right;
          const double crossed[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                     a[0] * b[1] - a[1] * b[0]};
          const double squared = dot3(crossed, crossed);
          if (!(squared > 0.0)) continue;
          const double share = dot3(direction, crossed) / squared;
          const double tangent[3] = {share * crossed[0], share * crossed[1],
                                     share * crossed[2]};
          emit(tangent);
        }
      }
    }
  }
  return overflow ? -1 : written;
}

namespace {

// Evaluate the residual and, only out of band, its unscaled length
// direction in the two frozen endpoint tensors.
bool frozen_tensor_residual(const std::array<double, 9>& a,
                            const std::array<double, 9>& b,
                            const std::array<double, 3>& delta, double scale,
                            double lower, double upper, double& residual,
                            std::array<double, 3>& direction) noexcept {
  const auto apply = [&](const std::array<double, 9>& tensor) -> std::array<double, 3> {
    return {dot3(tensor.data(), delta.data()), dot3(tensor.data() + 3, delta.data()),
            dot3(tensor.data() + 6, delta.data())};
  };
  const auto ma = apply(a);
  const auto mb = apply(b);
  const double f = std::sqrt(dot3(delta.data(), ma.data()));
  const double g = std::sqrt(dot3(delta.data(), mb.data()));
  if (!(f > 0.0 && g > 0.0) || !finite(f) || !finite(g)) return false;

  const double hi = std::max(f, g);
  const double lo = std::min(f, g);
  const double excess = (hi - lo) / lo;
  const bool nearly_equal = excess <= 1e-6;
  double log_ratio = 0.0;
  if (!nearly_equal) {
    log_ratio = finite(excess) ? deterministic::log1p(excess)
                               : deterministic::log(hi) - deterministic::log(lo);
  }
  const double norm = nearly_equal ? 0.5 * (f + g) : (hi - lo) / log_ratio;
  const double length = scale * norm;
  if (!(length > 0.0) || !finite(length)) return false;
  residual = length - std::clamp(length, lower, upper);
  if (residual == 0.0) return true;

  // Differentiate the logarithmic quotient, then chain through the
  // endpoint directional norms.
  const double d_hi = nearly_equal ? 0.5 : (1.0 - norm / hi) / log_ratio;
  const double d_lo = nearly_equal ? 0.5 : (norm / lo - 1.0) / log_ratio;
  const double d_f = f >= g ? d_hi : d_lo;
  const double d_g = f >= g ? d_lo : d_hi;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    direction[axis] = d_f * (ma[axis] / f) + d_g * (mb[axis] / g);
  }
  return finite3(direction.data());
}

}  // namespace

bool propose_shape_cage(std::span<const ShapeCagePoint> points,
                        std::span<const std::array<std::size_t, 4>> cells,
                        std::span<const ShapeCageEdge> edges, double radius,
                        double minimum_length, double target_shape,
                        std::span<double> displacement, bool improve_mean,
                        std::span<const std::array<double, 9>> frozen_tensors) {
  constexpr double kInset = 0.001;
  const double lower = minimum_length + kInset;
  constexpr double kUpper = kUpperLength - kInset;
  const bool tensor_band = !frozen_tensors.empty();
  if (displacement.empty() || cells.empty() || !(radius > 0.0) || !finite(radius) ||
      !(minimum_length > 0.0 && lower <= kUpper) ||
      (tensor_band && frozen_tensors.size() != points.size()) ||
      !(target_shape > 0.0 || (tensor_band && target_shape == 0.0)) || !finite(target_shape)) {
    return false;
  }
  for (const ShapeCagePoint& p : points) {
    if (!finite3(p.point.data())) return false;
    for (const std::int32_t index : p.variables) {
      if (index < -1 || (index >= 0 && static_cast<std::size_t>(index) >= displacement.size())) {
        return false;
      }
    }
  }
  for (const auto& cell : cells) {
    for (const std::size_t row : cell) {
      if (row >= points.size()) return false;
    }
  }
  for (const ShapeCageEdge& edge : edges) {
    if (edge.rows[0] >= points.size() || edge.rows[1] >= points.size() ||
        !(edge.scale > 0.0) || !finite(edge.scale)) {
      return false;
    }
  }

  using Point = std::array<double, 3>;
  std::vector<Point> trial(points.size());
  std::vector<double> position(displacement.size(), 0.0);
  const double mean_weight = improve_mean ? 0.025 / static_cast<double>(cells.size()) : 0.0;
  const auto difference = [](const Point& a, const Point& b) -> Point {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  };
  const auto cross = [](const Point& a, const Point& b) -> Point {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  };
  constexpr double kInvalid = std::numeric_limits<double>::quiet_NaN();
  const auto merit = [&](std::span<const double> x, std::span<double> gradient) {
    for (std::size_t row = 0; row < points.size(); ++row) {
      const ShapeCagePoint& p = points[row];
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::int32_t index = p.variables[axis];
        trial[row][axis] =
            index < 0 ? p.point[axis] : p.point[axis] + radius * x[static_cast<std::size_t>(index)];
      }
      if (!finite3(trial[row].data())) return kInvalid;
    }
    std::fill(gradient.begin(), gradient.end(), 0.0);
    const auto add_gradient = [&](std::size_t row, const Point& direction, double weight) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::int32_t index = points[row].variables[axis];
        if (index >= 0) {
          gradient[static_cast<std::size_t>(index)] += (radius * weight) * direction[axis];
        }
      }
    };

    double value = 0.0;
    constexpr double kEdgeWeight = 16.0;
    for (const ShapeCageEdge& edge : edges) {
      const auto [a, b] = edge.rows;
      const Point delta = difference(trial[a], trial[b]);
      if (tensor_band) {
        double residual = 0.0;
        Point direction;
        if (!frozen_tensor_residual(frozen_tensors[a], frozen_tensors[b], delta, edge.scale,
                                    lower, kUpper, residual, direction)) {
          return kInvalid;
        }
        if (residual == 0.0) continue;
        value += kEdgeWeight * residual * residual;
        const double weight = (2.0 * kEdgeWeight * residual) * edge.scale;
        add_gradient(a, direction, weight);
        add_gradient(b, direction, -weight);
        continue;
      }
      const double length = edge.scale * std::sqrt(dot3(delta.data(), delta.data()));
      if (!(length > 0.0) || !finite(length)) return kInvalid;
      const double residual = length - std::clamp(length, lower, kUpper);
      if (residual == 0.0) continue;
      value += kEdgeWeight * residual * residual;
      const double weight = (2.0 * kEdgeWeight * residual) * edge.scale * edge.scale / length;
      add_gradient(a, delta, weight);
      add_gradient(b, delta, -weight);
    }
    for (const auto& cell : cells) {
      const Point& a = trial[cell[0]];
      const Point ab = difference(trial[cell[1]], a);
      const Point ac = difference(trial[cell[2]], a);
      const Point ad = difference(trial[cell[3]], a);
      const Point first_gradient = cross(ac, ad);
      const double volume = dot3(ab.data(), first_gradient.data());  // Six times signed volume.
      if (!(volume > 0.0) || !finite(volume)) return kInvalid;
      if (target_shape == 0.0 && !improve_mean) continue;

      double squared = 0.0;
      for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
          squared += squared_distance(trial[cell[i]].data(), trial[cell[j]].data());
        }
      }
      if (!(squared > 0.0) || !finite(squared)) return kInvalid;
      const double shape = 12.0 * deterministic::cube_root(volume * volume * 0.25) / squared;
      if (!(shape > 0.0) || !finite(shape)) return kInvalid;
      const double residual = std::min(0.0, shape - target_shape);
      if (residual == 0.0 && mean_weight == 0.0) continue;
      value += residual * residual + mean_weight * (1.0 - shape);

      // q is proportional to volume^(2/3) / squared. Differentiate both factors.
      std::array<Point, 4> d_volume{Point{}, first_gradient, cross(ad, ab), cross(ab, ac)};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        d_volume[0][axis] = -((d_volume[1][axis] + d_volume[2][axis]) + d_volume[3][axis]);
      }
      for (std::size_t i = 0; i < 4; ++i) {
        Point d_squared{};
        for (std::size_t j = 0; j < 4; ++j) {
          if (i == j) continue;
          for (std::size_t axis = 0; axis < 3; ++axis) {
            d_squared[axis] += 2.0 * (trial[cell[i]][axis] - trial[cell[j]][axis]);
          }
        }
        Point d_shape;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          d_shape[axis] =
              shape * (2.0 * d_volume[i][axis] / (3.0 * volume) - d_squared[axis] / squared);
        }
        add_gradient(cell[i], d_shape, 2.0 * residual - mean_weight);
      }
    }
    return value;
  };
  const auto result = detail::bounded_lbfgs(position, 1.0, 0.15, 96, 1e-9, merit);
  // Each accepted step strictly decreases finite merit. A later stall or
  // invalid trial does not invalidate the last accepted position.
  if (result.steps == 0 || !finite(result.value)) return false;
  for (double& component : position) component *= radius;
  if (std::none_of(position.begin(), position.end(), [](double x) { return x != 0.0; })) {
    return false;
  }
  std::copy(position.begin(), position.end(), displacement.begin());
  return true;
}

}  // namespace phx::mc::metric
