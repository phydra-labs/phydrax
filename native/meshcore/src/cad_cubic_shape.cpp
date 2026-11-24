// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_cubic_shape.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "cad_curving.hpp"
#include "geometry3.hpp"

namespace phx::mc {
namespace {

using Point = std::array<double, 3>;
using Barycentric = std::array<double, 4>;
using Exponents = std::array<int, 4>;
using Map = std::array<Point, 20>;
using Matrix = std::array<Point, 3>;
using Anchors = std::array<Matrix, 10>;
constexpr int kEdges[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

constexpr auto node_exponents() {
  std::array<Exponents, 20> result{};
  for (int i = 0; i < 4; ++i) result[i][i] = 3;
  for (int i = 0; i < 6; ++i) {
    const int a = kEdges[i][0], b = kEdges[i][1];
    result[4 + 2 * i][a] = result[5 + 2 * i][b] = 2;
    result[4 + 2 * i][b] = result[5 + 2 * i][a] = 1;
  }
  for (int face = 0; face < 4; ++face) {
    result[16 + face].fill(1);
    result[16 + face][face] = 0;
  }
  return result;
}
constexpr auto kExponents = node_exponents();

std::array<double, 2> factor(int degree, double site) {
  const double z = 3 * site;
  switch (degree) {
    case 0: return {1, 0};
    case 1: return {z, 3};
    case 2: return {z * (z - 1) / 2, 1.5 * (2 * z - 1)};
    default: return {z * (z - 1) * (z - 2) / 6, (3 * z * (z - 2) + 2) / 2};
  }
}

Point gradient(const Exponents& node, const Barycentric& site) {
  Barycentric value{}, derivative{};
  for (int i = 0; i < 4; ++i) {
    const auto term = factor(node[i], site[i]);
    value[i] = term[0];
    derivative[i] = term[1];
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) if (j != i) derivative[i] *= value[j];
  }
  return {derivative[1] - derivative[0], derivative[2] - derivative[0],
          derivative[3] - derivative[0]};
}

// J is quadratic. Use the same ten anchor matrices for all 84 samples.
struct Stencil {
  std::array<Map, 10> derivatives{};
  std::array<std::array<double, 10>, 84> weights{};
  Stencil() {
    for (int anchor = 0; anchor < 10; ++anchor) {
      Barycentric site{};
      if (anchor < 4) {
        site[anchor] = 1.0;
      } else {
        site[kEdges[anchor - 4][0]] = site[kEdges[anchor - 4][1]] = 0.5;
      }
      for (int node = 1; node < 20; ++node) derivatives[anchor][node] = gradient(kExponents[node], site);
    }
    std::size_t sample = 0;
    for (int a = 0; a <= 6; ++a) {
      for (int b = 0; b <= 6 - a; ++b) {
        for (int c = 0; c <= 6 - a - b; ++c) {
          const Barycentric site{(6 - a - b - c) / 6.0, a / 6.0, b / 6.0, c / 6.0};
          auto& row = weights[sample++];
          for (int i = 0; i < 4; ++i) row[i] = site[i] * (2 * site[i] - 1);
          for (int i = 0; i < 6; ++i) row[4 + i] = 4 * site[kEdges[i][0]] * site[kEdges[i][1]];
        }
      }
    }
  }
};

void accumulate(Matrix& matrix, const Point& point, const Point& derivative) {
  for (int column = 0; column < 3; ++column) {
    if (derivative[column] == 0) continue;
    for (int axis = 0; axis < 3; ++axis) matrix[column][axis] += derivative[column] * point[axis];
  }
}

double determinant(const Matrix& j) {
  Point cross{};
  geometry::cross(j[1].data(), j[2].data(), cross.data());
  return geometry::dot(j[0].data(), cross.data());
}

struct Shape {
  double scaled, ratio;
};

// A sufficient numerical screen, not evidence of exact map validity. Each
// degree-three basis derivative is bounded in magnitude by 54 on the simplex.
// Bound the column perturbation from the affine map by the sum of the nodal
// residual norms, then apply multilinearity and Hadamard to its determinant.
Shape affine_shape_bound(const Map& points) {
  const Matrix affine{points[1], points[2], points[3]};
  const double a = geometry::length(affine[0]), b = geometry::length(affine[1]);
  const double c = geometry::length(affine[2]);
  double residual = 0.0;
  for (int node = 4; node < 20; ++node) {
    Point offset = points[node];
    for (int axis = 0; axis < 3; ++axis) {
      offset[axis] -= (kExponents[node][1] * affine[0][axis] +
                      kExponents[node][2] * affine[1][axis] +
                      kExponents[node][3] * affine[2][axis]) / 3.0;
    }
    residual += geometry::length(offset);
  }
  // Margin covers evaluation of the numerical lattice near a threshold.
  const double e = 54.0 * residual + 0x1p-38 * (a + b + c);
  const double d = determinant(affine);
  const double error = e * (a * b + b * c + a * c + e * (a + b + c + e));
  if (!(d > error)) return {};
  return {std::sqrt(2.0) * ((d - error) / ((a + e) * (b + e) * (c + e))),
          (d - error) / (d + error)};
}

Shape sample_cell(const Map& points, const Stencil& stencil) {
  Anchors anchors{};
  for (std::size_t a = 0; a < anchors.size(); ++a) {
    for (int node = 1; node < 20; ++node) {
      accumulate(anchors[a], points[node], stencil.derivatives[a][node]);
    }
  }
  double scaled = std::numeric_limits<double>::infinity();
  double lower = scaled, upper = 0.0;
  for (const auto& weights : stencil.weights) {
    Matrix map{};
    for (std::size_t a = 0; a < weights.size(); ++a) {
      if (weights[a] == 0.0) continue;
      for (int column = 0; column < 3; ++column) {
        for (int axis = 0; axis < 3; ++axis) map[column][axis] += weights[a] * anchors[a][column][axis];
      }
    }
    const double d = determinant(map);
    const double norm = geometry::length(map[0]) * geometry::length(map[1]) * geometry::length(map[2]);
    if (!(d > 0.0 && norm > 0.0 && std::isfinite(d) && std::isfinite(norm))) return {};
    scaled = std::min(scaled, std::sqrt(2.0) * (d / norm));
    lower = std::min(lower, d);
    upper = std::max(upper, d);
  }
  return {scaled, lower / upper};
}

}  // namespace

CadCubicShape sample_cad_cubic_shape(const CadQuadraticInput& input, const CadCubicMesh& mesh,
                                    std::span<const std::int64_t> cells) {
  static const Stencil stencil;
  CadCubicShape result;
  const std::size_t count = cells.empty() ? mesh.cell_nodes.size() : cells.size();
  for (std::size_t row = 0; row < count; ++row) {
    const std::size_t cell = cells.empty() ? row : static_cast<std::size_t>(cells[row]);
    Map points{};
    double scale = 0.0;
    const double* origin = input.points + 3 * static_cast<std::int64_t>(input.tetrahedra[4 * cell]);
    for (int node = 1; node < 20; ++node) {
      const double* source = node < 4
          ? input.points + 3 * static_cast<std::int64_t>(input.tetrahedra[4 * cell + node])
          : mesh.nodes[static_cast<std::size_t>(mesh.cell_nodes[cell][node - 4])].data();
      for (int axis = 0; axis < 3; ++axis) {
        const double delta = source[axis] - origin[axis];
        points[node][axis] = delta;
        scale = std::max(scale, std::isfinite(delta) ? std::abs(delta)
                                                      : std::numeric_limits<double>::infinity());
      }
    }
    Shape shape{};
    if (scale > 0.0 && std::isfinite(scale)) {
      // Local normalization keeps determinants and column norms in range on
      // graded CAD models; the global minimum requested h need not do so.
      const double inverse = 1.0 / scale;
      for (int node = 1; node < 20; ++node) {
        for (double& value : points[node])
          value = std::isfinite(inverse) ? value * inverse : value / scale;
      }
      shape = affine_shape_bound(points);
      if (!(shape.scaled >= input.sampled_minimum_scaled_jacobian &&
            shape.ratio >= input.sampled_minimum_determinant_ratio)) {
        shape = sample_cell(points, stencil);
      }
    }
    const bool missed_scaled = shape.scaled < input.sampled_minimum_scaled_jacobian;
    const bool missed_ratio = shape.ratio < input.sampled_minimum_determinant_ratio;
    result.unmet[0] += missed_scaled;
    result.unmet[1] += missed_ratio;
    if (missed_scaled || missed_ratio) result.failing_cells.push_back(static_cast<std::int64_t>(cell));
    result.minimum_scaled = std::min(result.minimum_scaled, shape.scaled);
    result.minimum_ratio = std::min(result.minimum_ratio, shape.ratio);
  }
  return result;
}

}  // namespace phx::mc
