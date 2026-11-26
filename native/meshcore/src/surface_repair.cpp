//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Native surface mesh repair kernels:
//  * displacement-bounded vertex welding, T-junction stitching, exact
//    collinearity detection, and link-condition sliver/cap cleanup;
//  * pinch-safe boundary loop extraction, Liepa O(k^3) minimum-dihedral
//    dynamic programming hole filling, optional refinement & harmonic fairing,
//    and exact illegal-contact embedding verification;
//  * manifold-BFS consistent orientation, signed-volume + nested-cavity
//    winding parity, generalized-winding open-sheet orientation, and debris
//    pruning.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "surface_adapt.hpp"
#include "surface_certify.hpp"
#include "topology_keys.hpp"

namespace phx::mc {
namespace {

using remesh::Point;
using remesh::SurfaceResult;
using remesh::Triangle;

constexpr double kPi = 3.14159265358979323846;

Point sub3(const Point& a, const Point& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Point add3(const Point& a, const Point& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Point scale3(const Point& a, double s) {
  return {a[0] * s, a[1] * s, a[2] * s};
}

double dot3(const Point& a, const Point& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point cross3(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double norm2_3(const Point& a) { return dot3(a, a); }
double norm3(const Point& a) { return std::sqrt(norm2_3(a)); }

double dist2_3(const Point& a, const Point& b) { return norm2_3(sub3(a, b)); }
double dist3(const Point& a, const Point& b) { return std::sqrt(dist2_3(a, b)); }

Point tri_cross(const Point& a, const Point& b, const Point& c) {
  return cross3(sub3(b, a), sub3(c, a));
}

double tri_area(const Point& a, const Point& b, const Point& c) {
  return 0.5 * norm3(tri_cross(a, b, c));
}

bool unit_normal(const Point& a, const Point& b, const Point& c, Point& out) {
  const Point cr = tri_cross(a, b, c);
  const double len = norm3(cr);
  if (len <= 0.0 || !std::isfinite(len)) {
    out = {0.0, 0.0, 0.0};
    return false;
  }
  out = {cr[0] / len, cr[1] / len, cr[2] / len};
  return true;
}

double dihedral_angle(const Point& n1, const Point& n2) {
  const double d = std::clamp(dot3(n1, n2), -1.0, 1.0);
  return std::acos(d);
}

bool exact_collinear_3d(const Point& a, const Point& b, const Point& c) {
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3;
    const int v = (axis + 2) % 3;
    const double aa[2] = {a[static_cast<std::size_t>(u)], a[static_cast<std::size_t>(v)]};
    const double bb[2] = {b[static_cast<std::size_t>(u)], b[static_cast<std::size_t>(v)]};
    const double cc[2] = {c[static_cast<std::size_t>(u)], c[static_cast<std::size_t>(v)]};
    if (phx::mc::orient2d(aa, bb, cc) != 0) return false;
  }
  return true;
}

bool validate_input_mesh(int64_t vertex_count, const double* vertices, int64_t triangle_count,
                         const int64_t* triangles, std::vector<Point>& points,
                         std::vector<Triangle>& faces, int32_t& status) {
  if (vertex_count < 0 || triangle_count < 0 || (vertex_count > 0 && vertices == nullptr) ||
      (triangle_count > 0 && triangles == nullptr) ||
      !addressable(vertex_count, 3, sizeof(double)) ||
      !addressable(triangle_count, 3, sizeof(int64_t))) {
    status = PHX_MC_INVALID_ARGUMENT;
    return false;
  }
  for (int64_t i = 0; i < 3 * vertex_count; ++i) {
    if (!std::isfinite(vertices[i])) {
      status = PHX_MC_NONFINITE_INPUT;
      return false;
    }
  }
  for (int64_t i = 0; i < 3 * triangle_count; ++i) {
    if (triangles[i] < 0 || triangles[i] >= vertex_count) {
      status = PHX_MC_INVALID_INPUT;
      return false;
    }
  }
  points.resize(static_cast<std::size_t>(vertex_count));
  for (int64_t v = 0; v < vertex_count; ++v) {
    points[static_cast<std::size_t>(v)] = {vertices[3 * v], vertices[3 * v + 1],
                                           vertices[3 * v + 2]};
  }
  faces.resize(static_cast<std::size_t>(triangle_count));
  for (int64_t t = 0; t < triangle_count; ++t) {
    faces[static_cast<std::size_t>(t)] = {triangles[3 * t], triangles[3 * t + 1],
                                          triangles[3 * t + 2]};
  }
  status = PHX_MC_OK;
  return true;
}

void compact_mesh(std::vector<Point>& points, std::vector<Triangle>& faces) {
  std::vector<int64_t> remap(points.size(), -1);
  std::vector<Point> kept;
  kept.reserve(points.size());
  for (Triangle& tri : faces) {
    for (int k = 0; k < 3; ++k) {
      const auto v = static_cast<std::size_t>(tri[static_cast<std::size_t>(k)]);
      if (remap[v] < 0) {
        remap[v] = static_cast<int64_t>(kept.size());
        kept.push_back(points[v]);
      }
      tri[static_cast<std::size_t>(k)] = remap[v];
    }
  }
  points = std::move(kept);
}

// -------------------------------------------------- Stage 0/1/2: weld & clean

struct GridCell {
  int64_t x;
  int64_t y;
  int64_t z;
  bool operator==(const GridCell& other) const = default;
};

struct GridCellHash {
  std::size_t operator()(const GridCell& c) const noexcept {
    const auto h1 = static_cast<std::uint64_t>(c.x) * 73856093ULL;
    const auto h2 = static_cast<std::uint64_t>(c.y) * 19349663ULL;
    const auto h3 = static_cast<std::uint64_t>(c.z) * 83492791ULL;
    return static_cast<std::size_t>(h1 ^ h2 ^ h3);
  }
};

struct EdgeKey {
  int64_t u;
  int64_t v;
  bool operator<(const EdgeKey& other) const {
    return u != other.u ? u < other.u : v < other.v;
  }
  bool operator==(const EdgeKey& other) const = default;
};

EdgeKey make_edge(int64_t a, int64_t b) { return {std::min(a, b), std::max(a, b)}; }

struct WeldCandidate {
  double dist2;
  int64_t u;
  int64_t v;
};

void bitwise_weld_vertices(std::vector<Point>& points, std::vector<Triangle>& faces,
                           std::vector<Point>& original_points, int64_t& welded_count,
                           int64_t& dropped_degenerate) {
  const std::size_t n = points.size();
  if (n == 0) return;
  std::vector<int64_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    const Point& pa = points[static_cast<std::size_t>(a)];
    const Point& pb = points[static_cast<std::size_t>(b)];
    for (int d = 0; d < 3; ++d) {
      if (pa[static_cast<std::size_t>(d)] != pb[static_cast<std::size_t>(d)]) {
        return pa[static_cast<std::size_t>(d)] < pb[static_cast<std::size_t>(d)];
      }
    }
    return a < b;
  });
  std::vector<int64_t> remap(n, -1);
  std::vector<Point> kept;
  kept.reserve(n);
  for (std::size_t i = 0; i < n;) {
    const int64_t rep = order[i];
    const Point& pref = points[static_cast<std::size_t>(rep)];
    const int64_t new_id = static_cast<int64_t>(kept.size());
    kept.push_back(pref);
    remap[static_cast<std::size_t>(rep)] = new_id;
    std::size_t j = i + 1;
    while (j < n) {
      const Point& pq = points[static_cast<std::size_t>(order[j])];
      if (pq[0] == pref[0] && pq[1] == pref[1] && pq[2] == pref[2]) {
        remap[static_cast<std::size_t>(order[j])] = new_id;
        ++welded_count;
        ++j;
      } else {
        break;
      }
    }
    i = j;
  }
  points = std::move(kept);
  original_points = points;
  std::vector<Triangle> valid;
  valid.reserve(faces.size());
  for (const Triangle& tri : faces) {
    const Triangle mapped{remap[static_cast<std::size_t>(tri[0])],
                          remap[static_cast<std::size_t>(tri[1])],
                          remap[static_cast<std::size_t>(tri[2])]};
    if (mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[2] == mapped[0]) {
      ++dropped_degenerate;
      continue;
    }
    valid.push_back(mapped);
  }
  faces = std::move(valid);
}

void tolerance_weld_vertices(std::vector<Point>& points, std::vector<Triangle>& faces,
                             double tolerance, bool preserve_embedding, int64_t& welded_count,
                             int64_t& rejected_fold_welds, int64_t& dropped_degenerate,
                             double& max_displacement) {
  if (tolerance <= 0.0 || points.size() <= 1) return;
  const std::size_t v_count = points.size();
  const double tol2 = tolerance * tolerance;
  // Choose the coordinate axis with the largest bounding span for 1D sweep-and-prune.
  double lo[3] = {points[0][0], points[0][1], points[0][2]};
  double hi[3] = {points[0][0], points[0][1], points[0][2]};
  for (std::size_t v = 1; v < v_count; ++v) {
    for (int d = 0; d < 3; ++d) {
      const double c = points[v][static_cast<std::size_t>(d)];
      if (c < lo[d]) lo[d] = c;
      if (c > hi[d]) hi[d] = c;
    }
  }
  int a0 = 0;
  if (hi[1] - lo[1] > hi[a0] - lo[a0]) a0 = 1;
  if (hi[2] - lo[2] > hi[a0] - lo[a0]) a0 = 2;
  const auto s0 = static_cast<std::size_t>(a0);
  const auto s1 = static_cast<std::size_t>((a0 + 1) % 3);
  const auto s2 = static_cast<std::size_t>((a0 + 2) % 3);

  std::vector<int64_t> order(v_count);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    const double ca = points[static_cast<std::size_t>(a)][s0];
    const double cb = points[static_cast<std::size_t>(b)][s0];
    return ca != cb ? ca < cb : a < b;
  });

  std::vector<WeldCandidate> candidates;
  candidates.reserve(v_count);
  for (std::size_t i = 0; i < v_count; ++i) {
    const int64_t u = order[i];
    const Point& pu = points[static_cast<std::size_t>(u)];
    for (std::size_t j = i + 1; j < v_count; ++j) {
      const int64_t v = order[j];
      const Point& pv = points[static_cast<std::size_t>(v)];
      const double d0 = pv[s0] - pu[s0];
      if (d0 > tolerance) break;
      const double d1 = pv[s1] - pu[s1];
      if (std::fabs(d1) > tolerance) continue;
      const double d2c = pv[s2] - pu[s2];
      if (std::fabs(d2c) > tolerance) continue;
      const double d2 = d0 * d0 + d1 * d1 + d2c * d2c;
      if (d2 <= tol2) {
        candidates.push_back({d2, std::min(u, v), std::max(u, v)});
      }
    }
  }
  if (candidates.empty()) return;
  std::sort(candidates.begin(), candidates.end(),
            [](const WeldCandidate& a, const WeldCandidate& b) {
              if (a.dist2 != b.dist2) return a.dist2 < b.dist2;
              if (a.u != b.u) return a.u < b.u;
              return a.v < b.v;
            });

  // Flat CSR vertex -> incident faces when fold protection is enabled.
  std::vector<std::size_t> vf_offsets;
  std::vector<std::size_t> vf_faces;
  if (preserve_embedding) {
    vf_offsets.assign(v_count + 1, 0);
    for (const Triangle& tri : faces) {
      ++vf_offsets[static_cast<std::size_t>(tri[0]) + 1];
      ++vf_offsets[static_cast<std::size_t>(tri[1]) + 1];
      ++vf_offsets[static_cast<std::size_t>(tri[2]) + 1];
    }
    for (std::size_t v = 0; v < v_count; ++v) vf_offsets[v + 1] += vf_offsets[v];
    vf_faces.resize(vf_offsets[v_count]);
    std::vector<std::size_t> cursor(vf_offsets.begin(), vf_offsets.end() - 1);
    for (std::size_t t = 0; t < faces.size(); ++t) {
      for (int k = 0; k < 3; ++k) {
        const auto v = static_cast<std::size_t>(faces[t][static_cast<std::size_t>(k)]);
        vf_faces[cursor[v]++] = t;
      }
    }
  }

  // Flat intrusive singly-linked cluster lists (zero per-vertex heap allocations).
  std::vector<int64_t> parent(v_count);
  std::iota(parent.begin(), parent.end(), 0);
  std::vector<int64_t> next_member(v_count, -1);
  std::vector<int64_t> tail_member(v_count);
  std::iota(tail_member.begin(), tail_member.end(), 0);

  for (const WeldCandidate& cand : candidates) {
    const int64_t ra = topology::find_root(parent, cand.u);
    const int64_t rb = topology::find_root(parent, cand.v);
    if (ra == rb) continue;
    const int64_t keep = std::min(ra, rb);
    const int64_t drop = std::max(ra, rb);
    const Point& target = points[static_cast<std::size_t>(keep)];
    // Enforce strict max_vertex_displacement <= tolerance for EVERY cluster member.
    double worst = 0.0;
    bool exceeds = false;
    for (int64_t member = drop; member >= 0;
         member = next_member[static_cast<std::size_t>(member)]) {
      const double d = dist3(points[static_cast<std::size_t>(member)], target);
      if (d > tolerance) {
        exceeds = true;
        break;
      }
      worst = std::max(worst, d);
    }
    if (exceeds) continue;
    if (preserve_embedding) {
      bool folds = false;
      for (int64_t member = drop; member >= 0 && !folds;
           member = next_member[static_cast<std::size_t>(member)]) {
        const auto mv = static_cast<std::size_t>(member);
        for (std::size_t idx = vf_offsets[mv]; idx < vf_offsets[mv + 1]; ++idx) {
          const Triangle& tri = faces[vf_faces[idx]];
          const int64_t cur[3] = {topology::find_root(parent, tri[0]),
                                  topology::find_root(parent, tri[1]),
                                  topology::find_root(parent, tri[2])};
          if (cur[0] == cur[1] || cur[1] == cur[2] || cur[2] == cur[0]) continue;
          const int64_t nxt[3] = {cur[0] == drop ? keep : cur[0], cur[1] == drop ? keep : cur[1],
                                  cur[2] == drop ? keep : cur[2]};
          if (nxt[0] == nxt[1] || nxt[1] == nxt[2] || nxt[2] == nxt[0]) continue;
          Point n_old{}, n_new{};
          if (unit_normal(points[static_cast<std::size_t>(cur[0])],
                          points[static_cast<std::size_t>(cur[1])],
                          points[static_cast<std::size_t>(cur[2])], n_old)) {
            if (!unit_normal(points[static_cast<std::size_t>(nxt[0])],
                             points[static_cast<std::size_t>(nxt[1])],
                             points[static_cast<std::size_t>(nxt[2])], n_new) ||
                dot3(n_old, n_new) <= 0.0) {
              folds = true;
              break;
            }
          }
        }
      }
      if (folds) {
        ++rejected_fold_welds;
        continue;
      }
    }
    parent[static_cast<std::size_t>(drop)] = keep;
    next_member[static_cast<std::size_t>(tail_member[static_cast<std::size_t>(keep)])] = drop;
    tail_member[static_cast<std::size_t>(keep)] = tail_member[static_cast<std::size_t>(drop)];
    ++welded_count;
    max_displacement = std::max(max_displacement, worst);
  }

  std::vector<Triangle> updated;
  updated.reserve(faces.size());
  for (const Triangle& tri : faces) {
    const Triangle mapped{topology::find_root(parent, tri[0]),
                          topology::find_root(parent, tri[1]),
                          topology::find_root(parent, tri[2])};
    if (mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[2] == mapped[0]) {
      ++dropped_degenerate;
      continue;
    }
    updated.push_back(mapped);
  }
  faces = std::move(updated);
}

void split_t_junctions(const std::vector<Point>& points, std::vector<Triangle>& faces,
                       double tolerance, int64_t& split_count) {
  if (tolerance <= 0.0 || faces.empty()) return;
  const double tol2 = tolerance * tolerance;
  // Identify boundary vertices (vertices on any degree-1 edge).
  std::map<EdgeKey, std::vector<std::size_t>> edge_faces;
  for (std::size_t t = 0; t < faces.size(); ++t) {
    for (int k = 0; k < 3; ++k) {
      edge_faces[make_edge(faces[t][static_cast<std::size_t>(k)],
                           faces[t][static_cast<std::size_t>((k + 1) % 3)])]
          .push_back(t);
    }
  }
  std::vector<std::uint8_t> is_boundary_vertex(points.size(), 0);
  for (const auto& [edge, list] : edge_faces) {
    if (list.size() == 1) {
      is_boundary_vertex[static_cast<std::size_t>(edge.u)] = 1;
      is_boundary_vertex[static_cast<std::size_t>(edge.v)] = 1;
    }
  }
  std::vector<int64_t> boundary_verts;
  for (std::size_t v = 0; v < points.size(); ++v) {
    if (is_boundary_vertex[v]) boundary_verts.push_back(static_cast<int64_t>(v));
  }
  if (boundary_verts.empty()) return;

  // Spatial hash over boundary vertices for fast lookup around every edge AABB.
  const double cell_h = std::max(tolerance, 1.0e-12);
  const double inv_h = 1.0 / cell_h;
  std::unordered_map<GridCell, std::vector<int64_t>, GridCellHash> v_grid;
  v_grid.reserve(boundary_verts.size());
  for (const int64_t v : boundary_verts) {
    const Point& p = points[static_cast<std::size_t>(v)];
    const GridCell cell{static_cast<int64_t>(std::floor(p[0] * inv_h)),
                        static_cast<int64_t>(std::floor(p[1] * inv_h)),
                        static_cast<int64_t>(std::floor(p[2] * inv_h))};
    v_grid[cell].push_back(v);
  }

  struct EdgeSplitPoint {
    double t;
    int64_t vertex;
  };
  std::map<EdgeKey, std::vector<EdgeSplitPoint>> splits_by_edge;
  for (const auto& [edge, list] : edge_faces) {
    const Point& pa = points[static_cast<std::size_t>(edge.u)];
    const Point& pb = points[static_cast<std::size_t>(edge.v)];
    const Point ab = sub3(pb, pa);
    const double len2 = norm2_3(ab);
    if (len2 <= 0.0) continue;
    const double lo[3] = {std::min(pa[0], pb[0]) - tolerance, std::min(pa[1], pb[1]) - tolerance,
                          std::min(pa[2], pb[2]) - tolerance};
    const double hi[3] = {std::max(pa[0], pb[0]) + tolerance, std::max(pa[1], pb[1]) + tolerance,
                          std::max(pa[2], pb[2]) + tolerance};
    // Guard against huge edges spanning millions of tiny tolerance cells.
    const int64_t c0[3] = {static_cast<int64_t>(std::floor(lo[0] * inv_h)),
                           static_cast<int64_t>(std::floor(lo[1] * inv_h)),
                           static_cast<int64_t>(std::floor(lo[2] * inv_h))};
    const int64_t c1[3] = {static_cast<int64_t>(std::floor(hi[0] * inv_h)),
                           static_cast<int64_t>(std::floor(hi[1] * inv_h)),
                           static_cast<int64_t>(std::floor(hi[2] * inv_h))};
    const int64_t span = (c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1) * (c1[2] - c0[2] + 1);
    std::vector<EdgeSplitPoint> found;
    const auto test_vertex = [&](int64_t v) {
      if (v == edge.u || v == edge.v) return;
      const Point& pv = points[static_cast<std::size_t>(v)];
      if (pv[0] < lo[0] || pv[0] > hi[0] || pv[1] < lo[1] || pv[1] > hi[1] || pv[2] < lo[2] ||
          pv[2] > hi[2]) {
        return;
      }
      const double t = dot3(sub3(pv, pa), ab) / len2;
      if (t <= 1.0e-6 || t >= 1.0 - 1.0e-6) return;
      const Point proj = add3(pa, scale3(ab, t));
      if (dist2_3(pv, proj) <= tol2) {
        found.push_back({t, v});
      }
    };
    if (span > 512 || static_cast<std::size_t>(span) > boundary_verts.size()) {
      for (const int64_t v : boundary_verts) test_vertex(v);
    } else {
      for (int64_t ix = c0[0]; ix <= c1[0]; ++ix) {
        for (int64_t iy = c0[1]; iy <= c1[1]; ++iy) {
          for (int64_t iz = c0[2]; iz <= c1[2]; ++iz) {
            const auto it = v_grid.find({ix, iy, iz});
            if (it == v_grid.end()) continue;
            for (const int64_t v : it->second) test_vertex(v);
          }
        }
      }
    }
    if (!found.empty()) {
      std::sort(found.begin(), found.end(), [](const EdgeSplitPoint& a, const EdgeSplitPoint& b) {
        return a.t != b.t ? a.t < b.t : a.vertex < b.vertex;
      });
      splits_by_edge.emplace(edge, std::move(found));
    }
  }
  if (splits_by_edge.empty()) return;

  // Iteratively split any triangle whose edge carries split points.
  std::vector<Triangle> current = std::move(faces);
  for (int pass = 0; pass < 3; ++pass) {
    bool any = false;
    std::vector<Triangle> next;
    next.reserve(current.size() + splits_by_edge.size());
    for (const Triangle& tri : current) {
      int split_corner = -1;
      for (int k = 0; k < 3; ++k) {
        const EdgeKey key =
            make_edge(tri[static_cast<std::size_t>(k)], tri[static_cast<std::size_t>((k + 1) % 3)]);
        if (splits_by_edge.find(key) != splits_by_edge.end()) {
          split_corner = k;
          break;
        }
      }
      if (split_corner < 0) {
        next.push_back(tri);
        continue;
      }
      const int64_t a = tri[static_cast<std::size_t>(split_corner)];
      const int64_t b = tri[static_cast<std::size_t>((split_corner + 1) % 3)];
      const int64_t c = tri[static_cast<std::size_t>((split_corner + 2) % 3)];
      const EdgeKey key = make_edge(a, b);
      const auto& pts = splits_by_edge.at(key);
      std::vector<int64_t> chain;
      chain.reserve(pts.size() + 2);
      chain.push_back(a);
      if (a < b) {
        for (const auto& sp : pts) {
          if (sp.vertex != c) chain.push_back(sp.vertex);
        }
      } else {
        for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
          if (it->vertex != c) chain.push_back(it->vertex);
        }
      }
      chain.push_back(b);
      if (chain.size() <= 2) {
        next.push_back(tri);
        continue;
      }
      any = true;
      for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
        next.push_back({chain[i], chain[i + 1], c});
      }
      split_count += static_cast<int64_t>(chain.size() - 2);
    }
    current = std::move(next);
    if (!any) break;
  }
  faces = std::move(current);
}

void drop_exact_duplicate_faces(std::vector<Triangle>& faces, int64_t& dropped_duplicates) {
  if (faces.empty()) return;
  struct FaceKey {
    Triangle sorted;
    std::size_t index;
    int8_t parity;
  };
  std::vector<FaceKey> keys;
  keys.reserve(faces.size());
  for (std::size_t i = 0; i < faces.size(); ++i) {
    Triangle t = faces[i];
    int8_t parity = 1;
    if (t[0] > t[1]) { std::swap(t[0], t[1]); parity = static_cast<int8_t>(-parity); }
    if (t[1] > t[2]) { std::swap(t[1], t[2]); parity = static_cast<int8_t>(-parity); }
    if (t[0] > t[1]) { std::swap(t[0], t[1]); parity = static_cast<int8_t>(-parity); }
    keys.push_back({t, i, parity});
  }
  std::sort(keys.begin(), keys.end(), [](const FaceKey& a, const FaceKey& b) {
    if (a.sorted != b.sorted) return a.sorted < b.sorted;
    return a.index < b.index;
  });
  std::vector<std::uint8_t> keep(faces.size(), 0);
  for (std::size_t i = 0; i < keys.size();) {
    std::size_t j = i + 1;
    int64_t net = keys[i].parity;
    while (j < keys.size() && keys[j].sorted == keys[i].sorted) {
      net += keys[j].parity;
      ++j;
    }
    const std::size_t group = j - i;
    if (net == 0) {
      dropped_duplicates += static_cast<int64_t>(group);
    } else {
      const int8_t target_parity = net > 0 ? 1 : -1;
      for (std::size_t k = i; k < j; ++k) {
        if (keys[k].parity == target_parity) {
          keep[keys[k].index] = 1;
          break;
        }
      }
      dropped_duplicates += static_cast<int64_t>(group - 1);
    }
    i = j;
  }
  std::vector<Triangle> filtered;
  filtered.reserve(faces.size());
  for (std::size_t i = 0; i < faces.size(); ++i) {
    if (keep[i]) filtered.push_back(faces[i]);
  }
  faces = std::move(filtered);
}

void clean_slivers_and_degenerates(std::vector<Point>& points, std::vector<Triangle>& faces,
                                   double min_edge_length, double min_triangle_area,
                                   bool drop_degenerate, int64_t& collapsed_edges,
                                   int64_t& flipped_caps, int64_t& dropped_degenerate,
                                   double& max_displacement) {
  if (!drop_degenerate && min_edge_length <= 0.0 && min_triangle_area <= 0.0) return;
  const double min_edge2 = min_edge_length * min_edge_length;

  // Fast O(F) screen: skip topology map construction when no triangle needs cleanup.
  bool any_candidate = false;
  for (const Triangle& tri : faces) {
    if (tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0]) {
      any_candidate = true;
      break;
    }
    const Point& p0 = points[static_cast<std::size_t>(tri[0])];
    const Point& p1 = points[static_cast<std::size_t>(tri[1])];
    const Point& p2 = points[static_cast<std::size_t>(tri[2])];
    if (exact_collinear_3d(p0, p1, p2) ||
        (min_triangle_area > 0.0 && tri_area(p0, p1, p2) < min_triangle_area) ||
        (min_edge_length > 0.0 &&
         (dist2_3(p0, p1) <= min_edge2 || dist2_3(p1, p2) <= min_edge2 ||
          dist2_3(p2, p0) <= min_edge2))) {
      any_candidate = true;
      break;
    }
  }
  if (!any_candidate) return;

  // Bounded topology-preserving passes: flip obtuse caps, link-condition collapse needles.
  for (int pass = 0; pass < 4; ++pass) {
    bool changed = false;
    std::map<EdgeKey, std::vector<std::size_t>> edge_tris;
    std::vector<std::vector<std::size_t>> vert_tris(points.size());
    for (std::size_t t = 0; t < faces.size(); ++t) {
      for (int k = 0; k < 3; ++k) {
        const int64_t a = faces[t][static_cast<std::size_t>(k)];
        const int64_t b = faces[t][static_cast<std::size_t>((k + 1) % 3)];
        edge_tris[make_edge(a, b)].push_back(t);
        vert_tris[static_cast<std::size_t>(a)].push_back(t);
      }
    }
    std::vector<std::uint8_t> alive(faces.size(), 1);

    // Pass A: try flipping the longest edge of obtuse cap slivers.
    for (std::size_t t = 0; t < faces.size(); ++t) {
      if (!alive[t]) continue;
      const Triangle& tri = faces[t];
      const Point& p0 = points[static_cast<std::size_t>(tri[0])];
      const Point& p1 = points[static_cast<std::size_t>(tri[1])];
      const Point& p2 = points[static_cast<std::size_t>(tri[2])];
      const bool degen = exact_collinear_3d(p0, p1, p2) ||
                         (min_triangle_area > 0.0 && tri_area(p0, p1, p2) < min_triangle_area);
      if (!degen) continue;
      const double e2[3] = {dist2_3(p0, p1), dist2_3(p1, p2), dist2_3(p2, p0)};
      int longest = 0;
      if (e2[1] > e2[longest]) longest = 1;
      if (e2[2] > e2[longest]) longest = 2;
      int shortest = 0;
      if (e2[1] < e2[shortest]) shortest = 1;
      if (e2[2] < e2[shortest]) shortest = 2;
      // Only treat as a cap when the shortest edge is not itself tiny.
      if (min_edge2 > 0.0 && e2[shortest] <= min_edge2) continue;
      const int64_t u = tri[static_cast<std::size_t>(longest)];
      const int64_t v = tri[static_cast<std::size_t>((longest + 1) % 3)];
      const int64_t w = tri[static_cast<std::size_t>((longest + 2) % 3)];
      const auto it = edge_tris.find(make_edge(u, v));
      if (it == edge_tris.end() || it->second.size() != 2) continue;
      const std::size_t other = (it->second[0] == t) ? it->second[1] : it->second[0];
      if (!alive[other]) continue;
      const Triangle& ot = faces[other];
      int64_t x = -1;
      bool opposite_dir = false;
      for (int k = 0; k < 3; ++k) {
        if (ot[static_cast<std::size_t>(k)] == v &&
            ot[static_cast<std::size_t>((k + 1) % 3)] == u) {
          x = ot[static_cast<std::size_t>((k + 2) % 3)];
          opposite_dir = true;
        }
      }
      if (!opposite_dir || x < 0 || x == w) continue;
      if (edge_tris.find(make_edge(w, x)) != edge_tris.end()) continue;
      Point n_ref{}, n1{}, n2{};
      if (!unit_normal(points[static_cast<std::size_t>(ot[0])],
                       points[static_cast<std::size_t>(ot[1])],
                       points[static_cast<std::size_t>(ot[2])], n_ref)) {
        continue;
      }
      const Triangle t1{u, x, w};
      const Triangle t2{x, v, w};
      if (exact_collinear_3d(points[static_cast<std::size_t>(t1[0])],
                             points[static_cast<std::size_t>(t1[1])],
                             points[static_cast<std::size_t>(t1[2])]) ||
          exact_collinear_3d(points[static_cast<std::size_t>(t2[0])],
                             points[static_cast<std::size_t>(t2[1])],
                             points[static_cast<std::size_t>(t2[2])])) {
        continue;
      }
      if (!unit_normal(points[static_cast<std::size_t>(t1[0])],
                       points[static_cast<std::size_t>(t1[1])],
                       points[static_cast<std::size_t>(t1[2])], n1) ||
          !unit_normal(points[static_cast<std::size_t>(t2[0])],
                       points[static_cast<std::size_t>(t2[1])],
                       points[static_cast<std::size_t>(t2[2])], n2)) {
        continue;
      }
      if (dot3(n_ref, n1) <= 0.1 || dot3(n_ref, n2) <= 0.1) continue;
      faces[t] = t1;
      faces[other] = t2;
      ++flipped_caps;
      changed = true;
      break;
    }
    if (changed) continue;

    // Pass B: link-condition collapse of short edges.
    if (min_edge_length > 0.0) {
      for (const auto& [edge, tris] : edge_tris) {
        if (tris.empty() || tris.size() > 2) continue;
        const double d2 =
            dist2_3(points[static_cast<std::size_t>(edge.u)],
                    points[static_cast<std::size_t>(edge.v)]);
        if (d2 > min_edge2) continue;
        // Collect opposite vertices of incident faces on edge (u, v).
        std::set<int64_t> edge_opp;
        bool valid_incident = true;
        for (const std::size_t f : tris) {
          if (!alive[f]) {
            valid_incident = false;
            break;
          }
          for (int k = 0; k < 3; ++k) {
            const int64_t cv = faces[f][static_cast<std::size_t>(k)];
            if (cv != edge.u && cv != edge.v) edge_opp.insert(cv);
          }
        }
        if (!valid_incident) continue;
        // Link condition: N(u) intersect N(v) == edge_opp.
        std::set<int64_t> nu, nv;
        for (const std::size_t f : vert_tris[static_cast<std::size_t>(edge.u)]) {
          if (!alive[f]) continue;
          for (int k = 0; k < 3; ++k) {
            const int64_t cv = faces[f][static_cast<std::size_t>(k)];
            if (cv != edge.u) nu.insert(cv);
          }
        }
        for (const std::size_t f : vert_tris[static_cast<std::size_t>(edge.v)]) {
          if (!alive[f]) continue;
          for (int k = 0; k < 3; ++k) {
            const int64_t cv = faces[f][static_cast<std::size_t>(k)];
            if (cv != edge.v) nv.insert(cv);
          }
        }
        std::set<int64_t> common;
        for (const int64_t x : nu) {
          if (nv.find(x) != nv.end()) common.insert(x);
        }
        if (common != edge_opp) continue;
        // Collapse v -> u only if surviving stars around v keep positive normal alignment.
        bool flips = false;
        for (const std::size_t f : vert_tris[static_cast<std::size_t>(edge.v)]) {
          if (!alive[f]) continue;
          const Triangle& tri = faces[f];
          if (tri[0] == edge.u || tri[1] == edge.u || tri[2] == edge.u) continue;
          Triangle repl = tri;
          for (int k = 0; k < 3; ++k) {
            if (repl[static_cast<std::size_t>(k)] == edge.v) repl[static_cast<std::size_t>(k)] = edge.u;
          }
          Point n_old{}, n_new{};
          if (unit_normal(points[static_cast<std::size_t>(tri[0])],
                          points[static_cast<std::size_t>(tri[1])],
                          points[static_cast<std::size_t>(tri[2])], n_old)) {
            if (!unit_normal(points[static_cast<std::size_t>(repl[0])],
                             points[static_cast<std::size_t>(repl[1])],
                             points[static_cast<std::size_t>(repl[2])], n_new) ||
                dot3(n_old, n_new) <= 0.0) {
              flips = true;
              break;
            }
          }
        }
        if (flips) continue;
        for (const std::size_t f : tris) alive[f] = 0;
        for (const std::size_t f : vert_tris[static_cast<std::size_t>(edge.v)]) {
          if (!alive[f]) continue;
          for (int k = 0; k < 3; ++k) {
            if (faces[f][static_cast<std::size_t>(k)] == edge.v) {
              faces[f][static_cast<std::size_t>(k)] = edge.u;
            }
          }
        }
        max_displacement = std::max(max_displacement, std::sqrt(d2));
        ++collapsed_edges;
        changed = true;
        break;
      }
      if (changed) {
        std::vector<Triangle> compacted;
        compacted.reserve(faces.size());
        for (std::size_t t = 0; t < faces.size(); ++t) {
          if (!alive[t]) continue;
          const Triangle& tri = faces[t];
          if (tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0]) continue;
          compacted.push_back(tri);
        }
        faces = std::move(compacted);
        continue;
      }
    }
    break;
  }

  // Final pass: drop remaining exact collinear or sub-threshold-area triangles.
  std::vector<Triangle> kept;
  kept.reserve(faces.size());
  for (const Triangle& tri : faces) {
    if (tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0]) {
      ++dropped_degenerate;
      continue;
    }
    const Point& p0 = points[static_cast<std::size_t>(tri[0])];
    const Point& p1 = points[static_cast<std::size_t>(tri[1])];
    const Point& p2 = points[static_cast<std::size_t>(tri[2])];
    const bool bad = (drop_degenerate && exact_collinear_3d(p0, p1, p2)) ||
                     (min_triangle_area > 0.0 && tri_area(p0, p1, p2) < min_triangle_area);
    if (bad) {
      ++dropped_degenerate;
      continue;
    }
    kept.push_back(tri);
  }
  faces = std::move(kept);
}

// ------------------------------------------------ Stage 4: Liepa hole filling

struct HoleEdge {
  int64_t from;
  int64_t to;
  Point rim_normal;
  bool has_normal;
  bool used;
};

struct SimpleHoleLoop {
  std::vector<int64_t> vertices;
  std::vector<Point> rim_normals;
  std::vector<std::uint8_t> has_rim_normal;
};

std::vector<SimpleHoleLoop> extract_boundary_loops(const std::vector<Point>& points,
                                                   const std::vector<Triangle>& faces,
                                                   int64_t& total_boundary_edges) {
  struct DirEdgeRec {
    int64_t u;
    int64_t v;
    std::size_t face;
    int8_t dir;
  };
  const std::size_t f_count = faces.size();
  std::vector<DirEdgeRec> edges;
  edges.reserve(3 * f_count);
  for (std::size_t t = 0; t < f_count; ++t) {
    for (int k = 0; k < 3; ++k) {
      const int64_t a = faces[t][static_cast<std::size_t>(k)];
      const int64_t b = faces[t][static_cast<std::size_t>((k + 1) % 3)];
      edges.push_back({std::min(a, b), std::max(a, b), t, static_cast<int8_t>(a < b ? 1 : -1)});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const DirEdgeRec& x, const DirEdgeRec& y) {
    return x.u != y.u ? x.u < y.u : x.v < y.v;
  });
  std::vector<HoleEdge> hole_edges;
  for (std::size_t i = 0; i < edges.size();) {
    std::size_t j = i + 1;
    while (j < edges.size() && edges[j].u == edges[i].u && edges[j].v == edges[i].v) ++j;
    if (j - i == 1) {
      const DirEdgeRec& rec = edges[i];
      const int64_t orig_a = rec.dir > 0 ? rec.u : rec.v;
      const int64_t orig_b = rec.dir > 0 ? rec.v : rec.u;
      const std::size_t f = rec.face;
      Point n{};
      const bool ok = unit_normal(points[static_cast<std::size_t>(faces[f][0])],
                                  points[static_cast<std::size_t>(faces[f][1])],
                                  points[static_cast<std::size_t>(faces[f][2])], n);
      // Traverse hole rim in reverse direction (orig_b -> orig_a) to match mesh orientation.
      hole_edges.push_back({orig_b, orig_a, n, ok, false});
    }
    i = j;
  }
  total_boundary_edges = static_cast<int64_t>(hole_edges.size());
  if (hole_edges.empty()) return {};
  std::sort(hole_edges.begin(), hole_edges.end(), [](const HoleEdge& a, const HoleEdge& b) {
    return a.from != b.from ? a.from < b.from : a.to < b.to;
  });

  std::map<int64_t, std::vector<std::size_t>> outgoing;
  for (std::size_t i = 0; i < hole_edges.size(); ++i) {
    outgoing[hole_edges[i].from].push_back(i);
  }
  for (auto& [v, list] : outgoing) {
    std::sort(list.begin(), list.end(), [&](std::size_t a, std::size_t b) {
      return hole_edges[a].to < hole_edges[b].to;
    });
  }

  std::vector<SimpleHoleLoop> loops;
  for (std::size_t start_edge = 0; start_edge < hole_edges.size(); ++start_edge) {
    if (hole_edges[start_edge].used) continue;
    std::vector<std::size_t> trail;
    std::map<int64_t, std::size_t> pos_in_trail;
    std::size_t cur = start_edge;
    while (true) {
      if (hole_edges[cur].used) break;
      hole_edges[cur].used = true;
      pos_in_trail[hole_edges[cur].from] = trail.size();
      trail.push_back(cur);
      const int64_t v = hole_edges[cur].to;
      const auto visited = pos_in_trail.find(v);
      if (visited != pos_in_trail.end()) {
        // Peel off exact simple cycle [loop_start .. end).
        const std::size_t loop_start = visited->second;
        SimpleHoleLoop loop;
        for (std::size_t idx = loop_start; idx < trail.size(); ++idx) {
          const HoleEdge& e = hole_edges[trail[idx]];
          loop.vertices.push_back(e.from);
          loop.rim_normals.push_back(e.rim_normal);
          loop.has_rim_normal.push_back(e.has_normal ? 1 : 0);
        }
        if (loop.vertices.size() >= 3) loops.push_back(std::move(loop));
        for (std::size_t idx = loop_start; idx < trail.size(); ++idx) {
          pos_in_trail.erase(hole_edges[trail[idx]].from);
        }
        trail.resize(loop_start);
        if (trail.empty()) break;
        // Continue walking from the pinch vertex v if more unused outgoing edges exist.
        const auto out_it = outgoing.find(v);
        if (out_it == outgoing.end()) break;
        std::size_t next_e = hole_edges.size();
        for (const std::size_t cand : out_it->second) {
          if (!hole_edges[cand].used) {
            next_e = cand;
            break;
          }
        }
        if (next_e == hole_edges.size()) break;
        cur = next_e;
        continue;
      }
      const auto out_it = outgoing.find(v);
      if (out_it == outgoing.end()) break;
      // Pick best unused outgoing edge at vertex v (smallest turning angle at pinch vertices).
      std::size_t next_e = hole_edges.size();
      double best_score = std::numeric_limits<double>::infinity();
      const Point in_dir =
          sub3(points[static_cast<std::size_t>(v)],
               points[static_cast<std::size_t>(hole_edges[cur].from)]);
      const double in_len = norm3(in_dir);
      for (const std::size_t cand : out_it->second) {
        if (hole_edges[cand].used) continue;
        if (out_it->second.size() == 1) {
          next_e = cand;
          break;
        }
        const Point out_dir =
            sub3(points[static_cast<std::size_t>(hole_edges[cand].to)],
                 points[static_cast<std::size_t>(v)]);
        const double out_len = norm3(out_dir);
        double score = 0.0;
        if (in_len > 0.0 && out_len > 0.0) {
          const double cos_turn = std::clamp(dot3(in_dir, out_dir) / (in_len * out_len), -1.0, 1.0);
          score = std::acos(cos_turn);
        }
        if (hole_edges[cur].has_normal && hole_edges[cand].has_normal) {
          score += dihedral_angle(hole_edges[cur].rim_normal, hole_edges[cand].rim_normal);
        }
        if (score < best_score) {
          best_score = score;
          next_e = cand;
        }
      }
      if (next_e == hole_edges.size()) break;
      cur = next_e;
    }
  }
  return loops;
}

double loop_diameter(const std::vector<Point>& points, const std::vector<int64_t>& verts) {
  double max_d2 = 0.0;
  for (std::size_t i = 0; i < verts.size(); ++i) {
    const Point& pi = points[static_cast<std::size_t>(verts[i])];
    for (std::size_t j = i + 1; j < verts.size(); ++j) {
      max_d2 = std::max(max_d2, dist2_3(pi, points[static_cast<std::size_t>(verts[j])]));
    }
  }
  return std::sqrt(max_d2);
}

void triangulate_loop_liepa(const std::vector<Point>& points, const SimpleHoleLoop& loop,
                            std::vector<Triangle>& cap_tris) {
  const std::size_t k = loop.vertices.size();
  if (k < 3) return;
  if (k == 3) {
    cap_tris.push_back({loop.vertices[0], loop.vertices[1], loop.vertices[2]});
    return;
  }
  // Greedy min-dihedral ear clipping fallback for very large loops (> 512 vertices).
  if (k > 512) {
    std::vector<int64_t> poly = loop.vertices;
    std::vector<Point> normals = loop.rim_normals;
    std::vector<std::uint8_t> has_n = loop.has_rim_normal;
    while (poly.size() > 3) {
      const std::size_t m = poly.size();
      std::size_t best_i = 0;
      double best_ang = std::numeric_limits<double>::infinity();
      double best_a = std::numeric_limits<double>::infinity();
      Point best_n{};
      bool best_ok = false;
      for (std::size_t i = 0; i < m; ++i) {
        const std::size_t ip = (i + m - 1) % m;
        const std::size_t in = (i + 1) % m;
        const Point& pa = points[static_cast<std::size_t>(poly[ip])];
        const Point& pb = points[static_cast<std::size_t>(poly[i])];
        const Point& pc = points[static_cast<std::size_t>(poly[in])];
        Point n{};
        const bool ok = unit_normal(pa, pb, pc, n);
        double ang = ok ? 0.0 : (2.0 * kPi);
        if (ok) {
          if (has_n[ip]) ang = std::max(ang, dihedral_angle(n, normals[ip]));
          if (has_n[i]) ang = std::max(ang, dihedral_angle(n, normals[i]));
        }
        const double area = tri_area(pa, pb, pc);
        if (ang + 1.0e-9 < best_ang || (std::fabs(ang - best_ang) <= 1.0e-9 && area < best_a)) {
          best_ang = ang;
          best_a = area;
          best_i = i;
          best_n = n;
          best_ok = ok;
        }
      }
      const std::size_t ip = (best_i + m - 1) % m;
      const std::size_t in = (best_i + 1) % m;
      cap_tris.push_back({poly[ip], poly[best_i], poly[in]});
      normals[ip] = best_n;
      has_n[ip] = best_ok ? 1 : 0;
      poly.erase(poly.begin() + static_cast<std::ptrdiff_t>(best_i));
      normals.erase(normals.begin() + static_cast<std::ptrdiff_t>(best_i));
      has_n.erase(has_n.begin() + static_cast<std::ptrdiff_t>(best_i));
    }
    cap_tris.push_back({poly[0], poly[1], poly[2]});
    return;
  }

  // Liepa (2003) O(k^3) exact dynamic programming minimizing (max_dihedral, total_area).
  const std::size_t table_size = k * k;
  std::vector<double> best_angle(table_size, 0.0);
  std::vector<double> best_area(table_size, 0.0);
  std::vector<std::size_t> split(table_size, 0);
  std::vector<Point> chord_normal(table_size, {0.0, 0.0, 0.0});
  std::vector<std::uint8_t> has_chord_normal(table_size, 0);
  const auto idx = [k](std::size_t i, std::size_t j) { return i * k + j; };

  for (std::size_t span = 2; span < k; ++span) {
    for (std::size_t i = 0; i + span < k; ++i) {
      const std::size_t j = i + span;
      double opt_ang = std::numeric_limits<double>::infinity();
      double opt_area = std::numeric_limits<double>::infinity();
      std::size_t opt_m = i + 1;
      Point opt_n{};
      bool opt_ok = false;
      const Point& pi = points[static_cast<std::size_t>(loop.vertices[i])];
      const Point& pj = points[static_cast<std::size_t>(loop.vertices[j])];
      for (std::size_t m = i + 1; m < j; ++m) {
        const Point& pm = points[static_cast<std::size_t>(loop.vertices[m])];
        Point n{};
        const bool ok = unit_normal(pi, pm, pj, n);
        double ang = std::max(best_angle[idx(i, m)], best_angle[idx(m, j)]);
        if (!ok) {
          ang = 2.0 * kPi;
        } else {
          if (m == i + 1) {
            if (loop.has_rim_normal[i]) ang = std::max(ang, dihedral_angle(n, loop.rim_normals[i]));
          } else if (has_chord_normal[idx(i, m)]) {
            ang = std::max(ang, dihedral_angle(n, chord_normal[idx(i, m)]));
          }
          if (j == m + 1) {
            if (loop.has_rim_normal[m]) ang = std::max(ang, dihedral_angle(n, loop.rim_normals[m]));
          } else if (has_chord_normal[idx(m, j)]) {
            ang = std::max(ang, dihedral_angle(n, chord_normal[idx(m, j)]));
          }
          if (i == 0 && j == k - 1 && loop.has_rim_normal[k - 1]) {
            ang = std::max(ang, dihedral_angle(n, loop.rim_normals[k - 1]));
          }
        }
        const double area = best_area[idx(i, m)] + best_area[idx(m, j)] + tri_area(pi, pm, pj);
        if (ang + 1.0e-9 < opt_ang || (std::fabs(ang - opt_ang) <= 1.0e-9 && area < opt_area)) {
          opt_ang = ang;
          opt_area = area;
          opt_m = m;
          opt_n = n;
          opt_ok = ok;
        }
      }
      best_angle[idx(i, j)] = opt_ang;
      best_area[idx(i, j)] = opt_area;
      split[idx(i, j)] = opt_m;
      chord_normal[idx(i, j)] = opt_n;
      has_chord_normal[idx(i, j)] = opt_ok ? 1 : 0;
    }
  }

  std::vector<std::pair<std::size_t, std::size_t>> stack;
  stack.push_back({0, k - 1});
  while (!stack.empty()) {
    const auto [i, j] = stack.back();
    stack.pop_back();
    if (j <= i + 1) continue;
    const std::size_t m = split[idx(i, j)];
    cap_tris.push_back({loop.vertices[i], loop.vertices[m], loop.vertices[j]});
    if (m > i + 1) stack.push_back({i, m});
    if (j > m + 1) stack.push_back({m, j});
  }
}

void refine_and_fair_cap(std::vector<Point>& points, const SimpleHoleLoop& loop,
                         std::vector<Triangle>& cap_tris, bool refine, bool fair) {
  if ((!refine && !fair) || cap_tris.empty()) return;
  std::set<int64_t> rim_verts(loop.vertices.begin(), loop.vertices.end());
  std::set<EdgeKey> rim_edges;
  double rim_len_sum = 0.0;
  for (std::size_t i = 0; i < loop.vertices.size(); ++i) {
    const int64_t a = loop.vertices[i];
    const int64_t b = loop.vertices[(i + 1) % loop.vertices.size()];
    rim_edges.insert(make_edge(a, b));
    rim_len_sum += dist3(points[static_cast<std::size_t>(a)], points[static_cast<std::size_t>(b)]);
  }
  const double target_h = rim_len_sum / static_cast<double>(loop.vertices.size());
  const double split_thresh = (4.0 / 3.0) * target_h;

  // Subdivide long interior cap edges (and centroid-split oversized triangles when fair && refine).
  if (refine && target_h > 0.0) {
    for (int iter = 0; iter < 6; ++iter) {
      std::map<EdgeKey, std::vector<std::size_t>> edge_tris;
      for (std::size_t t = 0; t < cap_tris.size(); ++t) {
        for (int k = 0; k < 3; ++k) {
          edge_tris[make_edge(cap_tris[t][static_cast<std::size_t>(k)],
                              cap_tris[t][static_cast<std::size_t>((k + 1) % 3)])]
              .push_back(t);
        }
      }
      EdgeKey longest_edge{-1, -1};
      double longest_len = split_thresh;
      for (const auto& [edge, tris] : edge_tris) {
        if (rim_edges.find(edge) != rim_edges.end() || tris.size() != 2) continue;
        const double len = dist3(points[static_cast<std::size_t>(edge.u)],
                                 points[static_cast<std::size_t>(edge.v)]);
        if (len > longest_len) {
          longest_len = len;
          longest_edge = edge;
        }
      }
      if (longest_edge.u < 0) break;
      const Point mid = scale3(add3(points[static_cast<std::size_t>(longest_edge.u)],
                                    points[static_cast<std::size_t>(longest_edge.v)]),
                               0.5);
      const int64_t mid_id = static_cast<int64_t>(points.size());
      points.push_back(mid);
      const auto& tris = edge_tris.at(longest_edge);
      std::vector<Triangle> added;
      for (const std::size_t f : tris) {
        const Triangle orig = cap_tris[f];
        for (int k = 0; k < 3; ++k) {
          const int64_t a = orig[static_cast<std::size_t>(k)];
          const int64_t b = orig[static_cast<std::size_t>((k + 1) % 3)];
          const int64_t c = orig[static_cast<std::size_t>((k + 2) % 3)];
          if (make_edge(a, b) == longest_edge) {
            cap_tris[f] = {a, mid_id, c};
            added.push_back({mid_id, b, c});
            break;
          }
        }
      }
      cap_tris.insert(cap_tris.end(), added.begin(), added.end());
      if (cap_tris.size() > 4096) break;
    }
  }

  if (fair) {
    // Collect strictly interior cap vertices (never move rim vertices).
    std::map<int64_t, std::set<int64_t>> adj;
    for (const Triangle& tri : cap_tris) {
      for (int k = 0; k < 3; ++k) {
        const int64_t a = tri[static_cast<std::size_t>(k)];
        const int64_t b = tri[static_cast<std::size_t>((k + 1) % 3)];
        if (rim_verts.find(a) == rim_verts.end()) adj[a].insert(b);
        if (rim_verts.find(b) == rim_verts.end()) adj[b].insert(a);
      }
    }
    if (!adj.empty()) {
      for (int iter = 0; iter < 12; ++iter) {
        std::vector<std::pair<int64_t, Point>> next_pos;
        next_pos.reserve(adj.size());
        for (const auto& [v, neighbors] : adj) {
          if (neighbors.empty()) continue;
          Point avg{0.0, 0.0, 0.0};
          for (const int64_t nb : neighbors) {
            avg = add3(avg, points[static_cast<std::size_t>(nb)]);
          }
          avg = scale3(avg, 1.0 / static_cast<double>(neighbors.size()));
          const Point cur = points[static_cast<std::size_t>(v)];
          next_pos.push_back({v, add3(scale3(cur, 0.5), scale3(avg, 0.5))});
        }
        for (const auto& [v, p] : next_pos) {
          points[static_cast<std::size_t>(v)] = p;
        }
      }
    }
  }
}

// ----------------------------------------- Stage 5b/6a: orient & prune debris

double solid_angle_point(const Point& p, const Point& a, const Point& b, const Point& c) {
  const Point ra = sub3(a, p);
  const Point rb = sub3(b, p);
  const Point rc = sub3(c, p);
  const double la = norm3(ra), lb = norm3(rb), lc = norm3(rc);
  if (la == 0.0 || lb == 0.0 || lc == 0.0) return 0.0;
  const double det = dot3(ra, cross3(rb, rc));
  const double denom = la * lb * lc + dot3(ra, rb) * lc + dot3(rb, rc) * la + dot3(rc, ra) * lb;
  return 2.0 * std::atan2(det, denom);
}

double component_winding_at(const Point& p, const std::vector<Point>& points,
                            const std::vector<Triangle>& faces,
                            const std::vector<std::size_t>& comp_faces) {
  double sum = 0.0;
  for (const std::size_t f : comp_faces) {
    const Triangle& t = faces[f];
    sum += solid_angle_point(p, points[static_cast<std::size_t>(t[0])],
                             points[static_cast<std::size_t>(t[1])],
                             points[static_cast<std::size_t>(t[2])]);
  }
  return sum / (4.0 * kPi);
}

struct CompBounds {
  Point lo;
  Point hi;
};

CompBounds compute_comp_bounds(const std::vector<Point>& points,
                               const std::vector<Triangle>& faces,
                               const std::vector<std::size_t>& comp_faces) {
  CompBounds b{{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()},
               {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()}};
  for (const std::size_t f : comp_faces) {
    for (int k = 0; k < 3; ++k) {
      const Point& p = points[static_cast<std::size_t>(faces[f][static_cast<std::size_t>(k)])];
      for (int d = 0; d < 3; ++d) {
        b.lo[static_cast<std::size_t>(d)] =
            std::min(b.lo[static_cast<std::size_t>(d)], p[static_cast<std::size_t>(d)]);
        b.hi[static_cast<std::size_t>(d)] =
            std::max(b.hi[static_cast<std::size_t>(d)], p[static_cast<std::size_t>(d)]);
      }
    }
  }
  return b;
}

bool point_in_bounds(const Point& p, const CompBounds& b) {
  return p[0] >= b.lo[0] && p[0] <= b.hi[0] && p[1] >= b.lo[1] && p[1] <= b.hi[1] &&
         p[2] >= b.lo[2] && p[2] <= b.hi[2];
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_surface_weld_and_clean(const phx_mc_surface_clean_request* request,
                                      phx_mc_surface_clean_report* report, void** out_surface) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (request == nullptr || report == nullptr || out_surface == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *report = {};
    *out_surface = nullptr;
    if (!std::isfinite(request->weld_tolerance) || request->weld_tolerance < 0.0 ||
        !std::isfinite(request->t_junction_tolerance) || request->t_junction_tolerance < 0.0 ||
        !std::isfinite(request->min_edge_length) || request->min_edge_length < 0.0 ||
        !std::isfinite(request->min_triangle_area) || request->min_triangle_area < 0.0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    auto result = std::make_unique<phx::mc::remesh::SurfaceResult>();
    int32_t status = PHX_MC_OK;
    if (!phx::mc::validate_input_mesh(request->vertex_count, request->vertices,
                                      request->triangle_count, request->triangles, result->points,
                                      result->triangles, status)) {
      return status;
    }
    std::vector<phx::mc::remesh::Point> original;
    phx::mc::bitwise_weld_vertices(result->points, result->triangles, original,
                                   report->welded_vertices, report->dropped_degenerate_faces);
    if (request->drop_degenerate != 0) {
      // Remove exact zero-area collinear slivers before tolerance welding can perturb
      // one of their corners onto a nearby valid vertex and create a nonzero needle.
      std::vector<phx::mc::remesh::Triangle> noncollinear;
      noncollinear.reserve(result->triangles.size());
      for (const auto& tri : result->triangles) {
        if (phx::mc::exact_collinear_3d(result->points[static_cast<std::size_t>(tri[0])],
                                        result->points[static_cast<std::size_t>(tri[1])],
                                        result->points[static_cast<std::size_t>(tri[2])])) {
          ++report->dropped_degenerate_faces;
          continue;
        }
        noncollinear.push_back(tri);
      }
      result->triangles = std::move(noncollinear);
      phx::mc::compact_mesh(result->points, result->triangles);
    }
    phx::mc::tolerance_weld_vertices(
        result->points, result->triangles, request->weld_tolerance,
        request->preserve_embedding != 0, report->welded_vertices, report->rejected_fold_welds,
        report->dropped_degenerate_faces, report->max_vertex_displacement);
    phx::mc::compact_mesh(result->points, result->triangles);
    phx::mc::split_t_junctions(result->points, result->triangles, request->t_junction_tolerance,
                               report->t_junction_splits);
    phx::mc::drop_exact_duplicate_faces(result->triangles, report->dropped_duplicate_faces);
    phx::mc::clean_slivers_and_degenerates(
        result->points, result->triangles, request->min_edge_length, request->min_triangle_area,
        request->drop_degenerate != 0, report->collapsed_edges, report->flipped_caps,
        report->dropped_degenerate_faces, report->max_vertex_displacement);
    phx::mc::compact_mesh(result->points, result->triangles);
    *out_surface = result.release();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_surface_fill_holes(const phx_mc_surface_hole_fill_request* request,
                                  phx_mc_surface_hole_fill_report* report, void** out_surface) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (request == nullptr || report == nullptr || out_surface == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *report = {};
    *out_surface = nullptr;
    if (request->max_hole_edges < 0 || !std::isfinite(request->max_hole_diameter) ||
        request->max_hole_diameter < 0.0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    auto result = std::make_unique<phx::mc::remesh::SurfaceResult>();
    int32_t status = PHX_MC_OK;
    if (!phx::mc::validate_input_mesh(request->vertex_count, request->vertices,
                                      request->triangle_count, request->triangles, result->points,
                                      result->triangles, status)) {
      return status;
    }
    int64_t initial_boundary_edges = 0;
    const std::vector<phx::mc::SimpleHoleLoop> loops =
        phx::mc::extract_boundary_loops(result->points, result->triangles, initial_boundary_edges);
    const std::size_t original_face_count = result->triangles.size();
    for (const phx::mc::SimpleHoleLoop& loop : loops) {
      const auto edge_count = static_cast<int64_t>(loop.vertices.size());
      const double diam = phx::mc::loop_diameter(result->points, loop.vertices);
      if ((request->max_hole_edges > 0 && edge_count > request->max_hole_edges) ||
          (request->max_hole_diameter > 0.0 && diam > request->max_hole_diameter)) {
        ++report->skipped_large_holes;
        continue;
      }
      std::vector<phx::mc::remesh::Triangle> cap;
      phx::mc::triangulate_loop_liepa(result->points, loop, cap);
      phx::mc::refine_and_fair_cap(result->points, loop, cap, request->refine_caps != 0,
                                   request->fair_caps != 0);
      for (const auto& tri : cap) {
        report->filled_hole_area +=
            phx::mc::tri_area(result->points[static_cast<std::size_t>(tri[0])],
                              result->points[static_cast<std::size_t>(tri[1])],
                              result->points[static_cast<std::size_t>(tri[2])]);
        result->triangles.push_back(tri);
      }
      report->added_cap_triangles += static_cast<int64_t>(cap.size());
      ++report->filled_hole_count;
    }
    if (request->check_embedding != 0 && report->added_cap_triangles > 0) {
      std::vector<std::uint8_t> changed(result->triangles.size(), 0);
      for (std::size_t f = original_face_count; f < result->triangles.size(); ++f) changed[f] = 1;
      std::vector<std::uint8_t> refused;
      phx::mc::remesh::refuse_illegal_contacts(result->points, result->triangles, &changed,
                                               refused);
      for (std::size_t f = original_face_count; f < refused.size(); ++f) {
        if (refused[f] != 0) ++report->colliding_cap_triangles;
      }
    }
    if (report->added_cap_triangles > 0) {
      int64_t remaining = 0;
      (void)phx::mc::extract_boundary_loops(result->points, result->triangles, remaining);
      report->remaining_boundary_edges = remaining;
    } else {
      report->remaining_boundary_edges = initial_boundary_edges;
    }
    phx::mc::compact_mesh(result->points, result->triangles);
    *out_surface = result.release();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_surface_orient_and_prune(const phx_mc_surface_orient_request* request,
                                        phx_mc_surface_orient_report* report, void** out_surface) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (request == nullptr || report == nullptr || out_surface == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *report = {};
    *out_surface = nullptr;
    if (request->min_component_faces < 0 ||
        !std::isfinite(request->min_component_area_fraction) ||
        request->min_component_area_fraction < 0.0 ||
        request->min_component_area_fraction > 1.0 ||
        !std::isfinite(request->min_component_volume_fraction) ||
        request->min_component_volume_fraction < 0.0 ||
        request->min_component_volume_fraction > 1.0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    auto result = std::make_unique<phx::mc::remesh::SurfaceResult>();
    int32_t status = PHX_MC_OK;
    if (!phx::mc::validate_input_mesh(request->vertex_count, request->vertices,
                                      request->triangle_count, request->triangles, result->points,
                                      result->triangles, status)) {
      return status;
    }
    const std::size_t n = result->triangles.size();
    if (n == 0) {
      *out_surface = result.release();
      return PHX_MC_OK;
    }

    // Flat sorted edge pass: builds components (DSU), closure flags, and
    // fixed-capacity manifold adjacency without any std::map or per-face vectors.
    struct OrientEdge {
      int64_t u;
      int64_t v;
      std::size_t face;
      int8_t dir;
    };
    std::vector<OrientEdge> edges;
    edges.reserve(3 * n);
    for (std::size_t t = 0; t < n; ++t) {
      for (int k = 0; k < 3; ++k) {
        const int64_t a = result->triangles[t][static_cast<std::size_t>(k)];
        const int64_t b = result->triangles[t][static_cast<std::size_t>((k + 1) % 3)];
        edges.push_back({std::min(a, b), std::max(a, b), t, static_cast<int8_t>(a < b ? 1 : -1)});
      }
    }
    std::sort(edges.begin(), edges.end(), [](const OrientEdge& x, const OrientEdge& y) {
      if (x.u != y.u) return x.u < y.u;
      if (x.v != y.v) return x.v < y.v;
      return x.face < y.face;
    });

    std::vector<int64_t> comp_parent(n);
    std::iota(comp_parent.begin(), comp_parent.end(), 0);
    const auto unite_comp = [&](std::size_t a, std::size_t b) {
      const int64_t ra = phx::mc::topology::find_root(comp_parent, static_cast<int64_t>(a));
      const int64_t rb = phx::mc::topology::find_root(comp_parent, static_cast<int64_t>(b));
      if (ra != rb) comp_parent[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
    };

    struct ManifoldNbrs {
      std::size_t nb[3];
      int8_t rel[3];
      std::uint8_t count = 0;
    };
    std::vector<ManifoldNbrs> man_adj;
    if (request->orient_shells != 0) man_adj.resize(n);
    std::vector<std::uint8_t> face_open_or_nm(n, 0);

    for (std::size_t i = 0; i < edges.size();) {
      std::size_t j = i + 1;
      while (j < edges.size() && edges[j].u == edges[i].u && edges[j].v == edges[i].v) ++j;
      const std::size_t f0 = edges[i].face;
      for (std::size_t k = i + 1; k < j; ++k) unite_comp(f0, edges[k].face);
      if (j - i != 2) {
        face_open_or_nm[f0] = 1;
      } else if (request->orient_shells != 0 && edges[i + 1].face != f0) {
        const std::size_t f1 = edges[i + 1].face;
        const int8_t rel = (edges[i].dir == edges[i + 1].dir) ? -1 : 1;
        if (man_adj[f0].count < 3) {
          man_adj[f0].nb[man_adj[f0].count] = f1;
          man_adj[f0].rel[man_adj[f0].count] = rel;
          ++man_adj[f0].count;
        }
        if (man_adj[f1].count < 3) {
          man_adj[f1].nb[man_adj[f1].count] = f0;
          man_adj[f1].rel[man_adj[f1].count] = rel;
          ++man_adj[f1].count;
        }
      }
      i = j;
    }

    // Group faces into components in deterministic ascending lowest-face order.
    std::vector<int64_t> root_to_comp(n, -1);
    std::vector<std::vector<std::size_t>> comps;
    std::vector<std::uint8_t> comp_closed;
    for (std::size_t t = 0; t < n; ++t) {
      const auto root = static_cast<std::size_t>(
          phx::mc::topology::find_root(comp_parent, static_cast<int64_t>(t)));
      if (root_to_comp[root] < 0) {
        root_to_comp[root] = static_cast<int64_t>(comps.size());
        comps.emplace_back();
        comp_closed.push_back(1);
      }
      const auto cid = static_cast<std::size_t>(root_to_comp[root]);
      comps[cid].push_back(t);
      if (face_open_or_nm[t]) comp_closed[cid] = 0;
    }
    report->component_count = static_cast<int64_t>(comps.size());

    const std::size_t m = comps.size();
    std::vector<double> comp_area(m, 0.0);
    std::vector<double> comp_volume(m, 0.0);
    std::vector<int8_t> face_flip(n, 1);

    if (request->orient_shells != 0) {
      std::vector<std::uint8_t> seen(n, 0);
      std::vector<std::size_t> bfs_queue;
      bfs_queue.reserve(n);
      for (std::size_t c = 0; c < m; ++c) {
        for (const std::size_t seed : comps[c]) {
          if (seen[seed]) continue;
          seen[seed] = 1;
          face_flip[seed] = 1;
          bfs_queue.clear();
          bfs_queue.push_back(seed);
          for (std::size_t head = 0; head < bfs_queue.size(); ++head) {
            const std::size_t cur = bfs_queue[head];
            const ManifoldNbrs& nbrs = man_adj[cur];
            for (std::uint8_t idx = 0; idx < nbrs.count; ++idx) {
              const std::size_t nb = nbrs.nb[idx];
              if (!seen[nb]) {
                seen[nb] = 1;
                face_flip[nb] = static_cast<int8_t>(face_flip[cur] * nbrs.rel[idx]);
                bfs_queue.push_back(nb);
              }
            }
          }
        }
      }
      for (std::size_t t = 0; t < n; ++t) {
        if (face_flip[t] < 0) std::swap(result->triangles[t][1], result->triangles[t][2]);
      }
    }

    // Compute per-component area and signed volume directly.
    for (std::size_t c = 0; c < m; ++c) {
      double area = 0.0;
      double vol6 = 0.0;
      for (const std::size_t f : comps[c]) {
        const auto& t = result->triangles[f];
        const auto& p0 = result->points[static_cast<std::size_t>(t[0])];
        const auto& p1 = result->points[static_cast<std::size_t>(t[1])];
        const auto& p2 = result->points[static_cast<std::size_t>(t[2])];
        area += phx::mc::tri_area(p0, p1, p2);
        vol6 += phx::mc::dot3(p0, phx::mc::cross3(p1, p2));
      }
      comp_area[c] = area;
      comp_volume[c] = vol6 / 6.0;
    }

    if (request->orient_shells != 0) {
      // First orient every closed component outward (positive volume).
      for (std::size_t c = 0; c < m; ++c) {
        if (comp_closed[c] && comp_volume[c] < 0.0) {
          for (const std::size_t f : comps[c]) {
            std::swap(result->triangles[f][1], result->triangles[f][2]);
            face_flip[f] = static_cast<int8_t>(-face_flip[f]);
          }
          comp_volume[c] = -comp_volume[c];
        }
      }
      // Compute bounding boxes and nested containment parity for closed components.
      std::vector<phx::mc::CompBounds> bounds(m);
      for (std::size_t c = 0; c < m; ++c) {
        bounds[c] = phx::mc::compute_comp_bounds(result->points, result->triangles, comps[c]);
      }
      std::vector<int> nesting_depth(m, 0);
      for (std::size_t c = 0; c < m; ++c) {
        if (!comp_closed[c] || comps[c].empty()) continue;
        const auto& t = result->triangles[comps[c].front()];
        const phx::mc::remesh::Point probe = phx::mc::scale3(
            phx::mc::add3(
                phx::mc::add3(result->points[static_cast<std::size_t>(t[0])],
                              result->points[static_cast<std::size_t>(t[1])]),
                result->points[static_cast<std::size_t>(t[2])]),
            1.0 / 3.0);
        int depth = 0;
        for (std::size_t other = 0; other < m; ++other) {
          if (other == c || !comp_closed[other]) continue;
          if (!phx::mc::point_in_bounds(probe, bounds[other])) continue;
          const double w = phx::mc::component_winding_at(probe, result->points, result->triangles,
                                                         comps[other]);
          if (std::fabs(w) > 0.5) ++depth;
        }
        nesting_depth[c] = depth;
      }
      for (std::size_t c = 0; c < m; ++c) {
        if (comp_closed[c] && (nesting_depth[c] & 1) != 0) {
          ++report->cavity_component_count;
          for (const std::size_t f : comps[c]) {
            std::swap(result->triangles[f][1], result->triangles[f][2]);
            face_flip[f] = static_cast<int8_t>(-face_flip[f]);
          }
          comp_volume[c] = -comp_volume[c];
        } else if (!comp_closed[c]) {
          // Open component: use Van Oosterom-Strackee generalized winding asymmetry.
          const double diag = phx::mc::dist3(bounds[c].lo, bounds[c].hi);
          const double eps = std::max(1.0e-5 * diag, 1.0e-10);
          std::vector<std::size_t> samples = comps[c];
          if (samples.size() > 64) {
            std::sort(samples.begin(), samples.end(), [&](std::size_t a, std::size_t b) {
              const auto& ta = result->triangles[a];
              const auto& tb = result->triangles[b];
              const double aa = phx::mc::tri_area(result->points[static_cast<std::size_t>(ta[0])],
                                                  result->points[static_cast<std::size_t>(ta[1])],
                                                  result->points[static_cast<std::size_t>(ta[2])]);
              const double ab = phx::mc::tri_area(result->points[static_cast<std::size_t>(tb[0])],
                                                  result->points[static_cast<std::size_t>(tb[1])],
                                                  result->points[static_cast<std::size_t>(tb[2])]);
              return aa != ab ? aa > ab : a < b;
            });
            samples.resize(64);
          }
          double score = 0.0;
          for (const std::size_t f : samples) {
            const auto& t = result->triangles[f];
            const auto& p0 = result->points[static_cast<std::size_t>(t[0])];
            const auto& p1 = result->points[static_cast<std::size_t>(t[1])];
            const auto& p2 = result->points[static_cast<std::size_t>(t[2])];
            phx::mc::remesh::Point nrm{};
            if (!phx::mc::unit_normal(p0, p1, p2, nrm)) continue;
            const phx::mc::remesh::Point cen =
                phx::mc::scale3(phx::mc::add3(phx::mc::add3(p0, p1), p2), 1.0 / 3.0);
            const double w_out = phx::mc::component_winding_at(
                phx::mc::add3(cen, phx::mc::scale3(nrm, eps)), result->points, result->triangles,
                comps[c]);
            const double w_in = phx::mc::component_winding_at(
                phx::mc::sub3(cen, phx::mc::scale3(nrm, eps)), result->points, result->triangles,
                comps[c]);
            // Outward normal should have lower exterior winding (w_out < w_in) and |w_out| < |w_in|.
            score += (std::fabs(w_in) - std::fabs(w_out));
          }
          if (score < 0.0) {
            for (const std::size_t f : comps[c]) {
              std::swap(result->triangles[f][1], result->triangles[f][2]);
              face_flip[f] = static_cast<int8_t>(-face_flip[f]);
            }
          }
        }
      }
      for (std::size_t t = 0; t < n; ++t) {
        if (face_flip[t] < 0) ++report->flipped_face_count;
      }
    }

    if (request->prune_small != 0) {
      double total_area = 0.0;
      double total_abs_vol = 0.0;
      for (std::size_t c = 0; c < m; ++c) {
        total_area += comp_area[c];
        if (comp_closed[c]) total_abs_vol += std::fabs(comp_volume[c]);
      }
      std::vector<std::uint8_t> keep_face(n, 1);
      for (std::size_t c = 0; c < m; ++c) {
        const bool too_few = request->min_component_faces > 0 &&
                             static_cast<int64_t>(comps[c].size()) < request->min_component_faces;
        const bool too_small_area =
            request->min_component_area_fraction > 0.0 && total_area > 0.0 &&
            (comp_area[c] / total_area) < request->min_component_area_fraction;
        const bool too_small_vol =
            request->min_component_volume_fraction > 0.0 && comp_closed[c] &&
            total_abs_vol > 0.0 &&
            (std::fabs(comp_volume[c]) / total_abs_vol) < request->min_component_volume_fraction;
        if (too_few || too_small_area || too_small_vol) {
          ++report->pruned_component_count;
          report->pruned_face_count += static_cast<int64_t>(comps[c].size());
          for (const std::size_t f : comps[c]) keep_face[f] = 0;
        }
      }
      std::vector<phx::mc::remesh::Triangle> kept;
      kept.reserve(n);
      for (std::size_t t = 0; t < n; ++t) {
        if (keep_face[t]) kept.push_back(result->triangles[t]);
      }
      result->triangles = std::move(kept);
    }

    phx::mc::compact_mesh(result->points, result->triangles);
    *out_surface = result.release();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_surface_closure_audit(int64_t vertex_count, const double* vertices,
                                     int64_t triangle_count, const int64_t* triangles,
                                     phx_mc_surface_closure_report* report) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (report == nullptr || vertex_count < 0 || triangle_count < 0 ||
        (vertex_count > 0 && vertices == nullptr) || (triangle_count > 0 && triangles == nullptr) ||
        !phx::mc::addressable(vertex_count, 3, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 3, sizeof(int64_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *report = {};
    for (int64_t i = 0; i < 3 * vertex_count; ++i) {
      if (!std::isfinite(vertices[i])) return PHX_MC_NONFINITE_INPUT;
    }
    for (int64_t i = 0; i < 3 * triangle_count; ++i) {
      if (triangles[i] < 0 || triangles[i] >= vertex_count) return PHX_MC_INVALID_INPUT;
    }
    if (triangle_count == 0) return PHX_MC_OK;

    const auto n = static_cast<std::size_t>(triangle_count);
    report->triangle_count = triangle_count;

    double vol6 = 0.0;
    struct ClosureEdge {
      int64_t u;
      int64_t v;
      int64_t face;
      int64_t corner;
      int8_t forward;
    };
    std::vector<ClosureEdge> edges;
    edges.reserve(3 * n);
    for (std::size_t t = 0; t < n; ++t) {
      const int64_t i0 = triangles[3 * t];
      const int64_t i1 = triangles[3 * t + 1];
      const int64_t i2 = triangles[3 * t + 2];
      const double* p0 = vertices + 3 * i0;
      const double* p1 = vertices + 3 * i1;
      const double* p2 = vertices + 3 * i2;
      const double cx = p1[1] * p2[2] - p1[2] * p2[1];
      const double cy = p1[2] * p2[0] - p1[0] * p2[2];
      const double cz = p1[0] * p2[1] - p1[1] * p2[0];
      vol6 += p0[0] * cx + p0[1] * cy + p0[2] * cz;
      const int64_t tri[3] = {i0, i1, i2};
      for (int k = 0; k < 3; ++k) {
        const int64_t a = tri[k];
        const int64_t b = tri[(k + 1) % 3];
        edges.push_back({std::min(a, b), std::max(a, b), static_cast<int64_t>(t),
                         static_cast<int64_t>(3 * t + static_cast<std::size_t>(k)),
                         static_cast<int8_t>(a < b ? 1 : -1)});
      }
    }
    report->signed_volume = vol6 / 6.0;

    std::sort(edges.begin(), edges.end(), [](const ClosureEdge& x, const ClosureEdge& y) {
      if (x.u != y.u) return x.u < y.u;
      if (x.v != y.v) return x.v < y.v;
      return x.corner < y.corner;
    });

    std::vector<int64_t> face_parent(n);
    std::iota(face_parent.begin(), face_parent.end(), 0);
    std::vector<int64_t> corner_parent(3 * n);
    std::iota(corner_parent.begin(), corner_parent.end(), 0);
    const auto unite = [](std::vector<int64_t>& parent, int64_t a, int64_t b) {
      const int64_t ra = phx::mc::topology::find_root(parent, a);
      const int64_t rb = phx::mc::topology::find_root(parent, b);
      if (ra != rb) parent[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
    };
    const auto next_c = [](int64_t c) { return (c % 3 == 2) ? (c - 2) : (c + 1); };

    int64_t boundary_edges = 0;
    int64_t nonmanifold_edges = 0;
    for (std::size_t i = 0; i < edges.size();) {
      std::size_t j = i + 1;
      while (j < edges.size() && edges[j].u == edges[i].u && edges[j].v == edges[i].v) ++j;
      const int64_t first_face = edges[i].face;
      for (std::size_t k = i + 1; k < j; ++k) unite(face_parent, first_face, edges[k].face);
      const std::size_t deg = j - i;
      if (deg == 1) {
        ++boundary_edges;
      } else if (deg == 2) {
        if (edges[i].forward == edges[i + 1].forward) {
          ++nonmanifold_edges;
        } else {
          const int64_t c0 = edges[i].corner;
          const int64_t c0n = next_c(c0);
          const int64_t c1 = edges[i + 1].corner;
          const int64_t c1n = next_c(c1);
          unite(corner_parent, c0, c1n);
          unite(corner_parent, c0n, c1);
        }
      } else {
        ++nonmanifold_edges;
      }
      i = j;
    }

    int64_t components = 0;
    for (std::size_t t = 0; t < n; ++t) {
      if (phx::mc::topology::find_root(face_parent, static_cast<int64_t>(t)) ==
          static_cast<int64_t>(t)) {
        ++components;
      }
    }

    std::vector<int64_t> first_root(static_cast<std::size_t>(vertex_count), -1);
    int64_t nonmanifold_vertices = 0;
    for (std::size_t t = 0; t < n; ++t) {
      for (int k = 0; k < 3; ++k) {
        const auto v = static_cast<std::size_t>(triangles[3 * t + static_cast<std::size_t>(k)]);
        const int64_t r = phx::mc::topology::find_root(
            corner_parent, static_cast<int64_t>(3 * t + static_cast<std::size_t>(k)));
        if (first_root[v] == -1) {
          first_root[v] = r;
        } else if (first_root[v] >= 0 && first_root[v] != r) {
          ++nonmanifold_vertices;
          first_root[v] = -2;
        }
      }
    }

    report->component_count = components;
    report->boundary_edges = boundary_edges;
    report->nonmanifold_edges = nonmanifold_edges;
    report->nonmanifold_vertices = nonmanifold_vertices;
    return PHX_MC_OK;
  });
}

}  // extern "C"
