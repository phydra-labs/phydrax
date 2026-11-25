// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_scalar_box.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "deterministic_float.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "mesh.hpp"
#include "refine3d.hpp"

namespace phx::mc::metric {
namespace {

using Face = std::array<std::int32_t, 3>;
using Segment = std::array<std::int32_t, 2>;

// Accumulate the displacement from the receiving site rather than its
// absolute target: a translation of the box should not swamp the move.
struct Moment {
  double weight = 0.0;
  BoxPoint sum{};

  void add(double w, double scale, const BoxPoint& center, const BoxPoint& offset) {
    weight += w;
    for (int axis = 0; axis < 3; ++axis) {
      sum[axis] += ((center[axis] - offset[axis]) * scale) * w;
    }
  }
};

bool positive_finite(double x) { return x > 0.0 && std::isfinite(x); }

// Translate to the first site and scale by a local power of two before
// solving. In this chart the largest corner offset has magnitude in [1/2,1).
template <std::size_t N>
double condition(const std::vector<BoxPoint>& points, const std::int32_t* ids,
                 std::array<BoxPoint, N>& offset) {
  const auto& origin = points[ids[0]];
  offset[0] = {};
  double extent = 0.0;
  for (std::size_t k = 1; k < N; ++k) {
    offset[k] = geometry::difference(points[ids[k]], origin);
    for (const double d : offset[k]) extent = std::max(extent, std::abs(d));
  }
  if (!positive_finite(extent)) return 0.0;
  int exponent;
  std::frexp(extent, &exponent);
  const double inverse = std::ldexp(1.0, -exponent);
  for (std::size_t k = 1; k < N; ++k) {
    for (auto& d : offset[k]) d *= inverse;
  }
  return std::ldexp(1.0, exponent);
}

// Physical D-measure times rate^(D+2). Form the dimensionless metric
// measure first instead of taking a large or small power of the raw rate.
template <int D>
double weighted_measure(double measure, double scale, double rate) {
  const double metric_scale = scale * rate;
  for (int k = 0; k < D; ++k) measure *= metric_scale;
  return (measure * rate) * rate;
}

bool volume_moments(const ScalarBoxSites& sites, const std::int32_t* ids,
                    const std::vector<unsigned>& walls, std::vector<Moment>& moments) {
  std::array<BoxPoint, 4> offset;
  const double scale = condition(sites.points, ids, offset);
  if (!(scale > 0.0)) return false;
  const auto cross = geometry::cross(offset[2], offset[3]);
  const double volume6 = geometry::dot(offset[1].data(), cross.data());
  if (!(volume6 > 0.0)) return false;
  const double mean = 0.25 * ((sites.samples[ids[0]].log_rate + sites.samples[ids[1]].log_rate) +
                              (sites.samples[ids[2]].log_rate + sites.samples[ids[3]].log_rate));
  const double h = deterministic::exp(-mean);
  if (!positive_finite(h)) return false;
  const double weight = weighted_measure<3>(volume6, scale, 1.0 / h);
  if (!positive_finite(weight)) return false;
  const double* p[4] = {offset[0].data(), offset[1].data(), offset[2].data(), offset[3].data()};
  geometry::TetrahedronShape shape;
  if (!geometry::tetrahedron_shape(p, shape)) return false;
  BoxPoint center = shape.circumcenter;
  const double limit = h / scale;
  for (int axis = 0; axis < 3; ++axis) {
    const double centroid = 0.25 * (offset[1][axis] + offset[2][axis] + offset[3][axis]);
    center[axis] = centroid + std::clamp(center[axis] - centroid, -limit, limit);
  }
  for (int k = 0; k < 4; ++k) {
    if (walls[ids[k]] == 0) moments[ids[k]].add(weight, scale, center, offset[k]);
  }
  return true;
}

bool face_moments(const ScalarBoxSites& sites, const Face& face, unsigned wall,
                  const std::vector<unsigned>& walls, std::vector<Moment>& moments) {
  const int normal = std::countr_zero(wall) / 2;
  const int i = (normal + 1) % 3, j = (normal + 2) % 3;
  std::array<BoxPoint, 3> offset;
  const double scale = condition(sites.points, face.data(), offset);
  if (!(scale > 0.0)) return false;
  const auto& a = offset[1];
  const auto& b = offset[2];
  const double det = a[i] * b[j] - a[j] * b[i];
  if (det == 0.0) return false;
  const double aa = 0.5 * geometry::dot(a.data(), a.data());
  const double bb = 0.5 * geometry::dot(b.data(), b.data());
  BoxPoint center{};
  center[i] = (aa * b[j] - a[j] * bb) / det;
  center[j] = (a[i] * bb - aa * b[i]) / det;
  if (!std::isfinite(center[i]) || !std::isfinite(center[j])) return false;
  const double mean = (sites.samples[face[0]].log_rate + sites.samples[face[1]].log_rate +
                       sites.samples[face[2]].log_rate) / 3.0;
  const double weight = weighted_measure<2>(std::abs(det), scale, deterministic::exp(mean));
  if (!positive_finite(weight)) return false;
  for (std::size_t k = 0; k < face.size(); ++k) {
    if (walls[face[k]] == wall) moments[face[k]].add(weight, scale, center, offset[k]);
  }
  return true;
}

bool seam_moments(const ScalarBoxSites& sites, const Segment& edge,
                  const std::vector<unsigned>& walls, std::vector<Moment>& moments) {
  const auto [a, b] = edge;
  const unsigned seam = walls[a] & walls[b];
  int axis = 0;
  while ((seam & (3u << (2 * axis))) != 0) ++axis;
  const double delta = sites.points[b][axis] - sites.points[a][axis];
  const double mean = 0.5 * (sites.samples[a].log_rate + sites.samples[b].log_rate);
  const double weight = weighted_measure<1>(1.0, std::abs(delta), deterministic::exp(mean));
  if (!positive_finite(weight)) return false;
  const double shift = (0.5 * delta) * weight;
  if (walls[a] == seam) {
    moments[a].weight += weight;
    moments[a].sum[axis] += shift;
  }
  if (walls[b] == seam) {
    moments[b].weight += weight;
    moments[b].sum[axis] -= shift;
  }
  return true;
}

}  // namespace

bool relax_scalar_box(const ScalarBoxField& field, ScalarBoxSites& sites, int sweeps) {
  const std::size_t count = sites.points.size();
  if (sweeps <= 0 || count < 4 || count > kMaxScalarSites || sites.cells.size() > kMaxScalarCells) {
    return false;
  }
  const GradualNearestScope mode;
  if (!mode.established()) return false;
  BoxPoint lower, upper;
  for (int axis = 0; axis < 3; ++axis) {
    const double lo = field.low[axis], hi = field.high[axis];
    if (!coordinate_in_domain(lo) || !coordinate_in_domain(hi) || !(lo < hi)) return false;
    const double margin = 1.e-8 * (hi - lo);
    lower[axis] = std::max(lo + margin, std::nextafter(lo, hi));
    upper[axis] = std::min(hi - margin, std::nextafter(hi, lo));
  }

  std::vector<double> coordinates(3 * count);
  std::vector<unsigned> walls(count, 0);
  sites.samples.resize(count);
  for (std::size_t v = 0; v < count; ++v) {
    const auto& p = sites.points[v];
    for (int axis = 0; axis < 3; ++axis) {
      const double lo = field.low[axis], hi = field.high[axis];
      if (!(p[axis] >= lo && p[axis] <= hi)) return false;
      if (p[axis] == lo) walls[v] |= 1u << (2 * axis);
      if (p[axis] == hi) walls[v] |= 2u << (2 * axis);
    }
    if (!field.sample(p, sites.samples[v])) return false;
    std::copy(p.begin(), p.end(), coordinates.data() + 3 * v);
  }

  std::vector<Moment> moments(count);
  std::vector<Face> faces;
  std::vector<Segment> seams;
  constexpr int kFace[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
  constexpr int kEdge[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    phx_mc_mesh* raw = nullptr;
    const auto status = triangulate_3d(static_cast<std::int64_t>(count), coordinates.data(),
                                       nullptr, false, true, &raw, 1, kMaxScalarCells);
    const std::unique_ptr<phx_mc_mesh> mesh(raw);
    if (!scalar_topology_accepted(status)) return false;
    const auto cell_count = static_cast<std::size_t>(mesh->cell_count());
    if (cell_count == 0 || cell_count > kMaxScalarCells) return false;
    for (std::size_t v = 0; v < count; ++v) {
      if (mesh->vertex_map[v] != static_cast<std::int32_t>(v)) return false;
    }

    std::fill(moments.begin(), moments.end(), Moment{});
    faces.clear();
    seams.clear();
    for (std::size_t row = 0; row < cell_count; ++row) {
      const auto* ids = mesh->cells.data() + 4 * row;
      const unsigned t[4] = {walls[ids[0]], walls[ids[1]], walls[ids[2]], walls[ids[3]]};
      if ((t[0] == 0 || t[1] == 0 || t[2] == 0 || t[3] == 0) &&
          !volume_moments(sites, ids, walls, moments)) return false;
      // A triple in a single supporting box plane is a hull facet. Find
      // just these faces, without sorting the entire volume incidence.
      for (const auto& f : kFace) {
        if ((t[f[0]] & t[f[1]] & t[f[2]]) == 0) continue;
        Face face{ids[f[0]], ids[f[1]], ids[f[2]]};
        std::sort(face.begin(), face.end());
        faces.push_back(face);
      }
    }
    std::sort(faces.begin(), faces.end());
    for (const auto& face : faces) {
      const unsigned wall = walls[face[0]] & walls[face[1]] & walls[face[2]];
      if ((walls[face[0]] == wall || walls[face[1]] == wall || walls[face[2]] == wall) &&
          !face_moments(sites, face, wall, walls, moments)) return false;
      for (const auto& e : kEdge) {
        const auto a = face[e[0]], b = face[e[1]];
        const unsigned seam = walls[a] & walls[b];
        if (std::popcount(seam) == 2 && (walls[a] == seam || walls[b] == seam)) {
          seams.push_back({a, b});
        }
      }
    }
    std::sort(seams.begin(), seams.end());
    seams.erase(std::unique(seams.begin(), seams.end()), seams.end());
    for (const auto& edge : seams) {
      if (!seam_moments(sites, edge, walls, moments)) return false;
    }

    const bool last = sweep + 1 == sweeps;
    for (std::size_t v = 0; v < count; ++v) {
      if (std::popcount(walls[v]) == 3) continue;
      auto& point = sites.points[v];
      auto& sample = sites.samples[v];
      const auto& moment = moments[v];
      const double limit = 0.25 * deterministic::exp(-sample.log_rate);
      if (!positive_finite(limit) || !positive_finite(moment.weight)) return false;
      for (int axis = 0; axis < 3; ++axis) {
        if ((walls[v] & (3u << (2 * axis))) != 0) continue;
        const double delta = moment.sum[axis] / moment.weight;
        if (!std::isfinite(delta)) return false;
        const double old = point[axis];
        // An already free site that starts within the relative margin must
        // not be pushed farther than its clipped step just to reach it.
        point[axis] = std::clamp(old + std::clamp(delta, -limit, limit),
                                 std::min(old, lower[axis]), std::max(old, upper[axis]));
      }
      if (!field.sample(point, sample)) return false;
      if (!last) std::copy(point.begin(), point.end(), coordinates.data() + 3 * v);
    }
    if (last) {
      // Keep the positive pre-move cell order. The continuous correction and
      // publication certificate, not this proposal, own post-move orientation.
      sites.cells.resize(cell_count);
      for (std::size_t row = 0; row < cell_count; ++row) {
        std::copy_n(mesh->cells.data() + 4 * row, 4, sites.cells[row].begin());
      }
    }
  }
  return true;
}

}  // namespace phx::mc::metric
