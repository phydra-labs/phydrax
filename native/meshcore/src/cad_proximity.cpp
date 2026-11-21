// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_proximity.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "phydrax_meshcore.h"

namespace phx::mc::cad {
namespace {

double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Distance from p to the triangle (a, b, c) (Ericson, Real-Time Collision
// Detection 5.1.5: the closest point by its Voronoi region).
double triangle_distance(const double* p, const double* t) {
  const double* a = t;
  const double* b = t + 3;
  const double* c = t + 6;
  double ab[3], ac[3], ap[3], q[3];
  for (int k = 0; k < 3; ++k) {
    ab[k] = b[k] - a[k];
    ac[k] = c[k] - a[k];
    ap[k] = p[k] - a[k];
  }
  const double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  const auto at = [&](const double* x) {
    double e = 0;
    for (int k = 0; k < 3; ++k) e += (p[k] - x[k]) * (p[k] - x[k]);
    return std::sqrt(e);
  };
  if (d1 <= 0 && d2 <= 0) return at(a);
  double bp[3];
  for (int k = 0; k < 3; ++k) bp[k] = p[k] - b[k];
  const double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
  if (d3 >= 0 && d4 <= d3) return at(b);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const double v = d1 / (d1 - d3);
    for (int k = 0; k < 3; ++k) q[k] = a[k] + v * ab[k];
    return at(q);
  }
  double cp[3];
  for (int k = 0; k < 3; ++k) cp[k] = p[k] - c[k];
  const double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
  if (d6 >= 0 && d5 <= d6) return at(c);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const double w = d2 / (d2 - d6);
    for (int k = 0; k < 3; ++k) q[k] = a[k] + w * ac[k];
    return at(q);
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    for (int k = 0; k < 3; ++k) q[k] = b[k] + w * (c[k] - b[k]);
    return at(q);
  }
  const double denominator = 1.0 / (va + vb + vc);
  const double v = vb * denominator, w = vc * denominator;
  for (int k = 0; k < 3; ++k) q[k] = a[k] + v * ab[k] + w * ac[k];
  return at(q);
}

}  // namespace

int32_t ProximityField::Set::build_node(int32_t begin, int32_t end) {
  const auto index = static_cast<int32_t>(nodes.size());
  nodes.emplace_back();
  Node node;
  node.begin = begin;
  node.end = end;
  for (int k = 0; k < 3; ++k) {
    node.low[k] = std::numeric_limits<double>::infinity();
    node.high[k] = -std::numeric_limits<double>::infinity();
  }
  for (int32_t i = begin; i < end; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        node.low[k] = std::min(node.low[k], triangles[order[i]][3 * j + k]);
        node.high[k] = std::max(node.high[k], triangles[order[i]][3 * j + k]);
      }
  if (end - begin > 4) {
    int axis = 0;
    for (int k = 1; k < 3; ++k)
      if (node.high[k] - node.low[k] > node.high[axis] - node.low[axis]) axis = k;
    const int32_t middle = begin + (end - begin) / 2;
    const auto centre = [&](int32_t row) {
      return triangles[row][axis] + triangles[row][3 + axis] + triangles[row][6 + axis];
    };
    std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                     [&](int32_t x, int32_t y) {
                       return centre(x) < centre(y) || (centre(x) == centre(y) && x < y);
                     });
    node.left = build_node(begin, middle);
    node.right = build_node(middle, end);
  }
  nodes[static_cast<std::size_t>(index)] = node;
  return index;
}

void ProximityField::Set::nearest(int32_t index, const double* p, double& best,
                                  int32_t& row) const {
  const Node& node = nodes[static_cast<std::size_t>(index)];
  double box2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double delta = std::max({node.low[k] - p[k], 0.0, p[k] - node.high[k]});
    box2 += delta * delta;
  }
  if (box2 > best * best) return;
  if (node.left >= 0) {
    nearest(node.left, p, best, row);
    nearest(node.right, p, best, row);
    return;
  }
  for (int32_t i = node.begin; i < node.end; ++i) {
    const double distance = triangle_distance(p, triangles[order[i]].data());
    if (distance < best || (distance == best && order[i] < row)) {
      best = distance;
      row = order[i];
    }
  }
}

double proxy_deviation(double circumradius, double kappa, double tolerance) {
  const double kr = kappa * circumradius;
  if (!(kr < 1.0)) return tolerance;
  return std::min(tolerance, kappa * circumradius * circumradius / (1.0 + std::sqrt(1.0 - kr * kr)));
}

int32_t ProximityField::build(const std::vector<ProximityControl>& controls,
                              const std::vector<std::array<double, 3>>& points,
                              const std::vector<std::array<int32_t, 3>>& triangles,
                              const std::vector<int32_t>& triangle_faces,
                              const std::vector<double>& deviations, int64_t face_count) {
  controls_.clear();
  face_rows_.assign(static_cast<std::size_t>(face_count), {});
  for (const ProximityControl& control : controls) {
    if (control.elements_per_gap <= 0 || !(control.minimum_size >= 0.0) ||
        !(control.maximum_size >= 0.0) || control.source_faces.empty() ||
        control.target_faces.empty())
      return PHX_MC_INVALID_INPUT;
    const auto row = static_cast<int32_t>(controls_.size());
    std::vector<int8_t> side(static_cast<std::size_t>(face_count), -1);
    for (int s = 0; s < 2; ++s)
      for (int32_t face : s ? control.target_faces : control.source_faces) {
        if (face < 0 || face >= face_count) return PHX_MC_INVALID_INPUT;
        auto& mark = side[static_cast<std::size_t>(face)];
        if (mark == s) continue;
        if (mark >= 0) return PHX_MC_INVALID_INPUT;  // overlapping scopes
        mark = static_cast<int8_t>(s);
        face_rows_[static_cast<std::size_t>(face)].push_back({row, s});
      }
    Entry entry{control, {}, {}};
    for (std::size_t t = 0; t < triangles.size(); ++t) {
      const int8_t s = side[static_cast<std::size_t>(triangle_faces[t])];
      if (s < 0) continue;
      Set& set = s ? entry.target : entry.source;
      std::array<double, 9> corners;
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) corners[3 * j + k] = points[triangles[t][j]][k];
      double u[3], v[3], n[3];
      for (int k = 0; k < 3; ++k) {
        u[k] = corners[3 + k] - corners[k];
        v[k] = corners[6 + k] - corners[k];
      }
      n[0] = u[1] * v[2] - u[2] * v[1];
      n[1] = u[2] * v[0] - u[0] * v[2];
      n[2] = u[0] * v[1] - u[1] * v[0];
      set.triangles.push_back(corners);
      set.normals.push_back({n[0], n[1], n[2]});
      set.tolerances.push_back(deviations[t]);
    }
    for (Set* set : {&entry.source, &entry.target}) {
      if (set->triangles.empty()) return PHX_MC_INVALID_INPUT;
      set->order.resize(set->triangles.size());
      for (std::size_t i = 0; i < set->order.size(); ++i)
        set->order[i] = static_cast<int32_t>(i);
      set->build_node(0, static_cast<int32_t>(set->order.size()));
    }
    controls_.push_back(std::move(entry));
  }
  return PHX_MC_OK;
}

double ProximityField::size(int32_t face, const double* p) const {
  double result = std::numeric_limits<double>::infinity();
  if (face < 0 || static_cast<std::size_t>(face) >= face_rows_.size()) return result;
  for (const auto& [row, side] : face_rows_[static_cast<std::size_t>(face)]) {
    const Entry& entry = controls_[static_cast<std::size_t>(row)];
    const Set& own = side ? entry.target : entry.source;
    const Set& other = side ? entry.source : entry.target;
    double distance = std::numeric_limits<double>::infinity();
    int32_t nearest = -1;
    other.nearest(0, p, distance, nearest);
    double gap = std::max(0.0, distance - other.tolerances[static_cast<std::size_t>(nearest)]);
    if (entry.control.opposite_normals_only) {
      double mine = std::numeric_limits<double>::infinity();
      int32_t own_row = -1;
      own.nearest(0, p, mine, own_row);
      if (!(dot3(own.normals[static_cast<std::size_t>(own_row)].data(),
                 other.normals[static_cast<std::size_t>(nearest)].data()) < 0.0))
        gap = std::numeric_limits<double>::infinity();
    }
    double size = gap / entry.control.elements_per_gap;
    if (entry.control.minimum_size > 0.0) size = std::max(size, entry.control.minimum_size);
    if (entry.control.maximum_size > 0.0) size = std::min(size, entry.control.maximum_size);
    result = std::min(result, size);
  }
  return result;
}

}  // namespace phx::mc::cad
