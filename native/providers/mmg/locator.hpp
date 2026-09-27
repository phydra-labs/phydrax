// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Deterministic P1 transfer of vertex fields from a source simplicial mesh to
// arbitrary target points (Mmg does not interpolate user fields itself).
//
// Every target point is assigned the closest point of the source complex: the
// simplex minimizing Euclidean distance, ties broken by the lowest simplex
// index. Points inside a simplex (distance zero) are "located" and use their
// exact barycentric coordinates; points outside the source (boundary
// approximation within the Hausdorff tolerance, or off a curved surface) are
// "projected" onto the closest point with clamped barycentric coordinates.
// Candidates come from a uniform bucket grid searched in Chebyshev rings until
// no unvisited bucket can hold a closer simplex, so the search is exact and
// bounded by the grid.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace phydrax::mmg {

struct Closest {
  double distance = std::numeric_limits<double>::infinity();
  std::int64_t simplex = -1;
  std::array<double, 4> weights{0.0, 0.0, 0.0, 0.0};
};

struct TransferEvidence {
  std::int64_t located = 0;
  std::int64_t projected = 0;
  double maximum_projection_distance = 0.0;
  double tolerance = 0.0;
};

namespace detail {

using Point = std::array<double, 3>;

inline Point sub(Point const& a, Point const& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline double dot(Point const& a, Point const& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Point cross(Point const& a, Point const& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double distance(Point const& a, Point const& b) {
  Point const d = sub(a, b);
  return std::sqrt(dot(d, d));
}

// Closest point of triangle abc to p (Ericson, Real-Time Collision Detection
// 5.1.5); weights are the barycentric coordinates of that closest point.
inline double closest_on_triangle(Point const& p, Point const& a, Point const& b, Point const& c,
                                  std::array<double, 3>& weights) {
  Point const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  double const d1 = dot(ab, ap), d2 = dot(ac, ap);
  auto finish = [&](double u, double v, double w) {
    weights = {u, v, w};
    Point const q = {u * a[0] + v * b[0] + w * c[0], u * a[1] + v * b[1] + w * c[1],
                     u * a[2] + v * b[2] + w * c[2]};
    return distance(p, q);
  };
  if (d1 <= 0.0 && d2 <= 0.0) return finish(1.0, 0.0, 0.0);
  Point const bp = sub(p, b);
  double const d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return finish(0.0, 1.0, 0.0);
  double const vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    double const v = d1 / (d1 - d3);
    return finish(1.0 - v, v, 0.0);
  }
  Point const cp = sub(p, c);
  double const d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return finish(0.0, 0.0, 1.0);
  double const vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    double const w = d2 / (d2 - d6);
    return finish(1.0 - w, 0.0, w);
  }
  double const va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    double const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return finish(0.0, 1.0 - w, w);
  }
  double const denominator = 1.0 / (va + vb + vc);
  double const v = vb * denominator, w = vc * denominator;
  return finish(1.0 - v - w, v, w);
}

// Barycentric coordinates of p in a non-degenerate simplex embedded in its own
// dimension (triangle in the plane or tetrahedron in space).
inline bool barycentric(Point const& p, Point const* corners, int arity, std::array<double, 4>& weights) {
  if (arity == 3) {
    double const x1 = corners[1][0] - corners[0][0], y1 = corners[1][1] - corners[0][1];
    double const x2 = corners[2][0] - corners[0][0], y2 = corners[2][1] - corners[0][1];
    double const determinant = x1 * y2 - x2 * y1;
    if (determinant == 0.0) return false;
    double const px = p[0] - corners[0][0], py = p[1] - corners[0][1];
    double const v = (px * y2 - x2 * py) / determinant;
    double const w = (x1 * py - px * y1) / determinant;
    weights = {1.0 - v - w, v, w, 0.0};
    return true;
  }
  Point const e1 = sub(corners[1], corners[0]), e2 = sub(corners[2], corners[0]),
              e3 = sub(corners[3], corners[0]), q = sub(p, corners[0]);
  double const determinant = dot(e1, cross(e2, e3));
  if (determinant == 0.0) return false;
  double const u = dot(q, cross(e2, e3)) / determinant;
  double const v = dot(e1, cross(q, e3)) / determinant;
  double const w = dot(e1, cross(e2, q)) / determinant;
  weights = {1.0 - u - v - w, u, v, w};
  return true;
}

}  // namespace detail

class Locator {
 public:
  // points: count x dimension (2 or 3); simplices: count x arity, 0-based.
  // arity 3 with dimension 3 is a surface; otherwise simplices are full-dimensional.
  Locator(std::vector<double> const& points, int dimension, std::vector<std::int64_t> const& simplices,
          int arity)
      : points_(points), dimension_(dimension), simplices_(simplices), arity_(arity) {
    if ((dimension != 2 && dimension != 3) || (arity != 3 && arity != 4) || (arity == 4 && dimension != 3))
      throw std::invalid_argument("Unsupported locator simplex kind");
    count_ = simplices.size() / static_cast<std::size_t>(arity);
    if (count_ == 0 || points.empty()) throw std::invalid_argument("Locator requires a source mesh");
    std::size_t const vertex_count = points.size() / static_cast<std::size_t>(dimension);
    for (double& value : low_) value = 0.0;
    for (double& value : high_) value = 0.0;
    for (int axis = 0; axis < dimension; ++axis) {
      low_[axis] = std::numeric_limits<double>::infinity();
      high_[axis] = -std::numeric_limits<double>::infinity();
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex)
      for (int axis = 0; axis < dimension; ++axis) {
        low_[axis] = std::min(low_[axis], points[vertex * dimension + axis]);
        high_[axis] = std::max(high_[axis], points[vertex * dimension + axis]);
      }
    double diagonal_squared = 0.0, volume = 1.0, largest = 0.0;
    for (int axis = 0; axis < dimension; ++axis) {
      double const extent = high_[axis] - low_[axis];
      diagonal_squared += extent * extent;
      largest = std::max(largest, extent);
    }
    diagonal_ = std::sqrt(diagonal_squared);
    if (!(diagonal_ > 0.0) || !std::isfinite(diagonal_))
      throw std::invalid_argument("Locator source mesh has no finite extent");
    // About one simplex per bucket; coarsen until the grid stays O(simplices).
    for (int axis = 0; axis < dimension; ++axis)
      volume *= std::max(high_[axis] - low_[axis], largest * 1.0e-3);
    double spacing = std::pow(volume / static_cast<double>(count_), 1.0 / dimension);
    std::uint64_t total = 0;
    while (true) {
      total = 1;
      for (int axis = 0; axis < 3; ++axis) {
        cells_[axis] = axis < dimension ? static_cast<std::int64_t>(std::max(
                                              1.0, std::ceil((high_[axis] - low_[axis]) / spacing)))
                                        : 1;
        total *= static_cast<std::uint64_t>(cells_[axis]);
      }
      if (total <= 8 * static_cast<std::uint64_t>(count_) + 64) break;
      spacing *= 1.25;
    }
    spacing_ = spacing;
    offsets_.assign(total + 1, 0);
    std::vector<std::array<std::int64_t, 6>> ranges(count_);
    for (std::size_t simplex = 0; simplex < count_; ++simplex) {
      auto& range = ranges[simplex];
      for (int axis = 0; axis < 3; ++axis) {
        if (axis >= dimension) {
          range[axis] = range[axis + 3] = 0;
          continue;
        }
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (int corner = 0; corner < arity; ++corner) {
          double const value = coordinate(simplices[simplex * arity + corner], axis);
          lo = std::min(lo, value);
          hi = std::max(hi, value);
        }
        range[axis] = index(lo, axis);
        range[axis + 3] = index(hi, axis);
      }
      for_each_bucket(range, [&](std::uint64_t bucket) { ++offsets_[bucket + 1]; });
    }
    for (std::size_t bucket = 0; bucket < total; ++bucket) offsets_[bucket + 1] += offsets_[bucket];
    items_.assign(offsets_[total], 0);
    std::vector<std::uint64_t> cursor(offsets_.begin(), offsets_.end() - 1);
    // Ascending simplex order inside each bucket keeps ties deterministic.
    for (std::size_t simplex = 0; simplex < count_; ++simplex)
      for_each_bucket(ranges[simplex], [&](std::uint64_t bucket) {
        items_[cursor[bucket]++] = static_cast<std::int64_t>(simplex);
      });
    stamp_.assign(count_, 0);
  }

  double diagonal() const { return diagonal_; }

  Closest closest(detail::Point const& p) {
    ++generation_;
    Closest best;
    std::array<std::int64_t, 3> center{};
    for (int axis = 0; axis < 3; ++axis) center[axis] = axis < dimension_ ? index(p[axis], axis) : 0;
    std::int64_t const rings = std::max({cells_[0], cells_[1], cells_[2]});
    for (std::int64_t ring = 0; ring <= rings; ++ring) {
      visit_ring(center, ring, [&](std::uint64_t bucket) {
        for (std::uint64_t item = offsets_[bucket]; item < offsets_[bucket + 1]; ++item) {
          std::int64_t const simplex = items_[item];
          if (stamp_[simplex] == generation_) continue;
          stamp_[simplex] = generation_;
          Closest candidate = evaluate(p, simplex);
          if (candidate.distance < best.distance ||
              (candidate.distance == best.distance && candidate.simplex < best.simplex))
            best = candidate;
        }
      });
      // Buckets beyond this ring lie at least ring * spacing away from p.
      if (best.simplex >= 0 && best.distance <= static_cast<double>(ring) * spacing_) break;
    }
    return best;
  }

 private:
  double coordinate(std::int64_t vertex, int axis) const {
    return points_[static_cast<std::size_t>(vertex) * dimension_ + axis];
  }

  detail::Point point(std::int64_t vertex) const {
    detail::Point value{0.0, 0.0, 0.0};
    for (int axis = 0; axis < dimension_; ++axis) value[axis] = coordinate(vertex, axis);
    return value;
  }

  std::int64_t index(double value, int axis) const {
    double const scaled = std::floor((value - low_[axis]) / spacing_);
    if (!(scaled > 0.0)) return 0;
    return std::min(static_cast<std::int64_t>(scaled), cells_[axis] - 1);
  }

  std::uint64_t bucket(std::int64_t i, std::int64_t j, std::int64_t k) const {
    return static_cast<std::uint64_t>((k * cells_[1] + j) * cells_[0] + i);
  }

  template <class Visit>
  void for_each_bucket(std::array<std::int64_t, 6> const& range, Visit&& visit) const {
    for (std::int64_t k = range[2]; k <= range[5]; ++k)
      for (std::int64_t j = range[1]; j <= range[4]; ++j)
        for (std::int64_t i = range[0]; i <= range[3]; ++i) visit(bucket(i, j, k));
  }

  template <class Visit>
  void visit_ring(std::array<std::int64_t, 3> const& center, std::int64_t ring, Visit&& visit) const {
    std::array<std::int64_t, 3> lo{}, hi{};
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = std::max<std::int64_t>(0, center[axis] - ring);
      hi[axis] = std::min<std::int64_t>(cells_[axis] - 1, center[axis] + ring);
    }
    for (std::int64_t k = lo[2]; k <= hi[2]; ++k)
      for (std::int64_t j = lo[1]; j <= hi[1]; ++j)
        for (std::int64_t i = lo[0]; i <= hi[0]; ++i) {
          std::int64_t const chebyshev = std::max(
              {std::abs(i - center[0]), std::abs(j - center[1]), std::abs(k - center[2])});
          if (chebyshev == ring) visit(bucket(i, j, k));
        }
  }

  Closest evaluate(detail::Point const& p, std::int64_t simplex) const {
    Closest result;
    result.simplex = simplex;
    detail::Point corners[4];
    for (int corner = 0; corner < arity_; ++corner)
      corners[corner] = point(simplices_[static_cast<std::size_t>(simplex) * arity_ + corner]);
    if (!(arity_ == 3 && dimension_ == 3)) {
      std::array<double, 4> weights{};
      if (detail::barycentric(p, corners, arity_, weights) &&
          *std::min_element(weights.begin(), weights.begin() + arity_) >= 0.0) {
        result.distance = 0.0;
        result.weights = weights;
        return result;
      }
    }
    // Outside (or a surface triangle): closest point over the boundary faces.
    static constexpr int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    static constexpr int triangle[3] = {0, 1, 2};
    int const face_count = arity_ == 4 ? 4 : 1;
    for (int face = 0; face < face_count; ++face) {
      int const* order = arity_ == 4 ? faces[face] : triangle;
      std::array<double, 3> weights{};
      double const d = detail::closest_on_triangle(p, corners[order[0]], corners[order[1]],
                                                   corners[order[2]], weights);
      if (d < result.distance) {
        result.distance = d;
        result.weights = {0.0, 0.0, 0.0, 0.0};
        for (int corner = 0; corner < 3; ++corner) result.weights[order[corner]] = weights[corner];
      }
    }
    return result;
  }

  std::vector<double> const& points_;
  int dimension_;
  std::vector<std::int64_t> const& simplices_;
  int arity_;
  std::size_t count_ = 0;
  double low_[3], high_[3];
  double diagonal_ = 0.0, spacing_ = 1.0;
  std::int64_t cells_[3] = {1, 1, 1};
  std::vector<std::uint64_t> offsets_;
  std::vector<std::int64_t> items_;
  std::vector<std::uint64_t> stamp_;
  std::uint64_t generation_ = 0;
};

// Interpolates `fields` (source_count x width) at target points; a target is
// located when its distance to the source is within `relative_tolerance`
// times the source bounding-box diagonal. Evidence counts only targets whose
// `counted` flag is set (every target when `counted` is empty), so ranks that
// share interface vertices report each vertex once.
inline std::vector<double> transfer(std::vector<double> const& source_points, int dimension,
                                    std::vector<std::int64_t> const& simplices, int arity,
                                    std::vector<double> const& fields, int width,
                                    std::vector<double> const& targets, double relative_tolerance,
                                    std::vector<std::uint8_t> const& counted,
                                    TransferEvidence& evidence) {
  Locator locator(source_points, dimension, simplices, arity);
  evidence = TransferEvidence{};
  evidence.tolerance = relative_tolerance * locator.diagonal();
  std::size_t const count = targets.size() / static_cast<std::size_t>(dimension);
  if (!counted.empty() && counted.size() != count)
    throw std::invalid_argument("Transfer evidence mask must match the targets");
  std::vector<double> values(count * static_cast<std::size_t>(width), 0.0);
  for (std::size_t target = 0; target < count; ++target) {
    detail::Point p{0.0, 0.0, 0.0};
    for (int axis = 0; axis < dimension; ++axis) p[axis] = targets[target * dimension + axis];
    Closest const found = locator.closest(p);
    if (counted.empty() || counted[target]) {
      if (found.distance <= evidence.tolerance) {
        ++evidence.located;
      } else {
        ++evidence.projected;
        evidence.maximum_projection_distance =
            std::max(evidence.maximum_projection_distance, found.distance);
      }
    }
    for (int corner = 0; corner < arity; ++corner) {
      double const weight = found.weights[corner];
      if (weight == 0.0) continue;
      std::size_t const vertex =
          static_cast<std::size_t>(simplices[static_cast<std::size_t>(found.simplex) * arity + corner]);
      for (int component = 0; component < width; ++component)
        values[target * width + component] += weight * fields[vertex * width + component];
    }
  }
  return values;
}

}  // namespace phydrax::mmg
