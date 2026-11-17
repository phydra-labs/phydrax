//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Floating-point goals of metric mesh edits; see metric_goal.hpp. Every sum
// below is written in the order it is evaluated.
#include "metric_goal.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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
  double gram[4];
  double right[2];
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

}  // namespace phx::mc::metric
