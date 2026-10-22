//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// PLC validation and recovery: volume and facet coverage oracles, fixed
// boundary preservation and failure evidence.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <vector>

#include "bounded_memory.hpp"
#include "check.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "plc3d.hpp"
#include "predicates.hpp"
#include "tet_mesh_fixtures.hpp"

namespace {

using phx::mc::BoundaryPolicy;
using phx::mc::PlcInput;
using phx::mc::PlcRecovery;
using phx::mc::test::face_area;
using phx::mc::test::tet_volume;

// Frozen exhaustive protection arithmetic: this is intentionally the original
// primitive order/kernel, not an ideal-distance oracle for the new broad phase.
double protection_distance(const double* a, const double* b) {
  return std::sqrt(phx::mc::geometry::squared_distance(a, b));
}

double protection_segment(const double* p, const double* a, const double* b) {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double length = phx::mc::geometry::dot(ab, ab);
  const double t = length > 0.0
                       ? std::clamp(phx::mc::geometry::dot(ap, ab) / length, 0.0, 1.0)
                       : 0.0;
  const double q[3] = {a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]};
  return protection_distance(p, q);
}

double protection_triangle(const double* p, const double* a, const double* b,
                           const double* c) {
  using phx::mc::geometry::dot;
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return protection_distance(p, a);
  const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return protection_distance(p, b);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return protection_segment(p, a, b);
  const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return protection_distance(p, c);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return protection_segment(p, a, c);
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) return protection_segment(p, b, c);
  const double denominator = va + vb + vc;
  if (!(denominator > 0.0)) {
    return std::min({protection_segment(p, a, b), protection_segment(p, b, c),
                     protection_segment(p, c, a)});
  }
  const double v = vb / denominator, w = vc / denominator;
  const double q[3] = {a[0] + ab[0] * v + ac[0] * w,
                       a[1] + ab[1] * v + ac[1] * w,
                       a[2] + ab[2] * v + ac[2] * w};
  return protection_distance(p, q);
}

std::vector<double> exhaustive_protection(const std::vector<double>& points,
                                          const std::vector<int32_t>& edges,
                                          const std::vector<int32_t>& triangles) {
  using phx::mc::geometry::dot;
  const std::size_t count = points.size() / 3;
  std::vector<std::vector<int32_t>> incident(count);
  for (std::size_t e = 0; e < edges.size(); e += 2) {
    incident[edges[e]].push_back(edges[e + 1]);
    incident[edges[e + 1]].push_back(edges[e]);
  }
  std::vector<double> result(count, 0.0);
  for (std::size_t v = 0; v < count; ++v) {
    const double* p = points.data() + 3 * v;
    bool acute = false;
    const auto& around = incident[v];
    for (std::size_t i = 0; i < around.size() && !acute; ++i) {
      const double* a = points.data() + 3 * around[i];
      const double u[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]};
      for (std::size_t j = i + 1; j < around.size() && !acute; ++j) {
        const double* b = points.data() + 3 * around[j];
        const double w[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
        acute = dot(u, w) > 0.0;
      }
    }
    if (!acute) continue;
    double reach = std::numeric_limits<double>::infinity();
    for (int32_t other : around) {
      reach = std::min(reach, protection_distance(p, points.data() + 3 * other) / 3.0);
    }
    for (std::size_t u = 0; u < count; ++u) {
      if (u != v) reach = std::min(reach, 0.5 * protection_distance(p, points.data() + 3 * u));
    }
    for (std::size_t e = 0; e < edges.size(); e += 2) {
      if (static_cast<std::size_t>(edges[e]) != v && static_cast<std::size_t>(edges[e + 1]) != v) {
        reach = std::min(reach, 0.5 * protection_segment(
            p, points.data() + 3 * edges[e], points.data() + 3 * edges[e + 1]));
      }
    }
    for (std::size_t t = 0; t < triangles.size(); t += 3) {
      const int32_t a = triangles[t], b = triangles[t + 1], c = triangles[t + 2];
      if (static_cast<std::size_t>(a) != v && static_cast<std::size_t>(b) != v &&
          static_cast<std::size_t>(c) != v) {
        reach = std::min(reach, 0.5 * protection_triangle(
            p, points.data() + 3 * a, points.data() + 3 * b, points.data() + 3 * c));
      }
    }
    result[v] = reach;
  }
  return result;
}

void check_protection_equivalence(const std::vector<double>& points,
                                  const std::vector<int32_t>& edges,
                                  const std::vector<int32_t>& triangles) {
  const auto expected = exhaustive_protection(points, edges, triangles);
  int64_t work = 0;
  const auto actual = phx::mc::protection_radii(points.data(), points.size() / 3, edges,
                                               triangles, work, 100000000);
  PHX_CHECK(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    PHX_CHECK(std::bit_cast<uint64_t>(actual[i]) == std::bit_cast<uint64_t>(expected[i]));
  }
  int64_t admitted_work = 0;
  const auto admitted = phx::mc::protection_radii(points.data(), points.size() / 3, edges,
                                                 triangles, admitted_work, work);
  PHX_CHECK(admitted.size() == expected.size());
  PHX_CHECK(admitted_work == work);
  if (work > 0) {
    int64_t refused_work = 0;
    const auto refused = phx::mc::protection_radii(points.data(), points.size() / 3, edges,
                                                  triangles, refused_work, work - 1);
    PHX_CHECK(refused.empty());
    PHX_CHECK(refused_work == work - 1);
  }
}

void test_protection_index_matches_actual_exhaustive_kernels() {
  std::vector<double> points;
  std::vector<int32_t> edges, triangles;
  for (int32_t i = 0; i < 128; ++i) {
    points.insert(points.end(), {((37 * i) % 127) / 127.0,
                                 ((53 * i) % 131) / 131.0,
                                 ((71 * i) % 137) / 137.0});
    edges.insert(edges.end(), {i, (i + 1) % 128, i, (i + 7) % 128});
    triangles.insert(triangles.end(), {i, (i + 1) % 128, (i + 7) % 128});
  }
  check_protection_equivalence(points, edges, triangles);
  for (int exponent : {-540, -300, 300, 600}) {
    auto scaled = points;
    for (double& value : scaled) value = std::ldexp(value, exponent);
    check_protection_equivalence(scaled, edges, triangles);
  }
  // Rounded a + (b - a) is zero, outside the ideal segment [1, 1e16].
  // The actual segment kernel therefore reduces vertex 2's radius to 1/2.
  const std::vector<double> rounded{1e16, 0, 0, 1, 0, 0, 0, 1, 0,
                                    100, 1, 0, 100, 2, 0, 0, 100, 0};
  const std::vector<int32_t> rounded_edges{0, 1, 2, 3, 2, 4};
  check_protection_equivalence(rounded, rounded_edges, {});
  const auto reference = exhaustive_protection(rounded, rounded_edges, {});
  PHX_CHECK(reference[2] == 0.5);
  // Degenerate triangles and collapsed/zero/subnormal coordinate directions
  // continue to use the unchanged Voronoi kernel and its NaN min semantics.
  const double tiny = std::numeric_limits<double>::denorm_min();
  check_protection_equivalence({0, 0, 0, 1, 0, 0, 2, tiny, 0, 3, 0, -0.0},
                               {0, 1, 0, 2, 1, 2, 2, 3}, {0, 1, 2, 1, 2, 3});
}

struct Plc {
  std::vector<double> points;
  std::vector<int64_t> offsets{0};
  std::vector<int32_t> loops;
  std::vector<int32_t> polygon_facets;
  std::vector<int32_t> facet_regions;
  std::vector<int32_t> segments;

  int32_t facet(int32_t positive, int32_t negative) {
    facet_regions.push_back(positive);
    facet_regions.push_back(negative);
    return static_cast<int32_t>(facet_regions.size() / 2 - 1);
  }
  void polygon(std::vector<int32_t> loop, int32_t facet_id) {
    loops.insert(loops.end(), loop.begin(), loop.end());
    offsets.push_back(static_cast<int64_t>(loops.size()));
    polygon_facets.push_back(facet_id);
  }
  int32_t run(BoundaryPolicy policy, PlcRecovery& out, int64_t work = int64_t{1} << 40) const {
    PlcInput in;
    in.point_count = static_cast<int64_t>(points.size() / 3);
    in.points = points.data();
    in.polygon_count = static_cast<int64_t>(polygon_facets.size());
    in.polygon_offsets = offsets.data();
    in.polygon_vertices = loops.data();
    in.polygon_facets = polygon_facets.data();
    in.facet_count = static_cast<int64_t>(facet_regions.size() / 2);
    in.facet_regions = facet_regions.data();
    in.segment_count = static_cast<int64_t>(segments.size() / 2);
    in.segments = segments.data();
    in.policy = policy;
    in.max_vertices = 100000;
    in.max_tetrahedra = 1000000;
    in.work_limit = work;
    return phx::mc::recover_plc(in, out);
  }
};

// Unit cube, vertex i = (i & 1, i >> 1 & 1, i >> 2 & 1), outward loops.
Plc cube(double scale = 1.0) {
  Plc plc;
  for (int i = 0; i < 8; ++i) {
    plc.points.insert(plc.points.end(),
                      {scale * (i & 1), scale * ((i >> 1) & 1), scale * ((i >> 2) & 1)});
  }
  const int32_t facet = plc.facet(-1, 0);
  for (const auto& loop : std::vector<std::vector<int32_t>>{
           {0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}}) {
    plc.polygon(loop, facet);
  }
  return plc;
}


// Region measures of published cells, each exactly positively oriented (the
// floating volume of a valid sliver between nearly coplanar facets may round
// to zero or below).
std::map<int32_t, double> region_volumes(const PlcRecovery& r) {
  std::map<int32_t, double> volumes;
  for (std::size_t t = 0; t < r.tet_regions.size(); ++t) {
    const double* p = r.points.data();
    const int32_t* v = r.tets.data() + 4 * t;
    PHX_CHECK(phx::mc::orient3d(p + 3 * v[0], p + 3 * v[1], p + 3 * v[2], p + 3 * v[3]) > 0);
    volumes[r.tet_regions[t]] += tet_volume(r, t);
  }
  return volumes;
}

struct TriangleSurface {
  std::span<const double> points;
  std::span<const int32_t> faces;
};

// Divergence-theorem contribution of one outward source triangle.
double source_triangle_volume(const double* a, const double* b, const double* c) {
  return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
          a[2] * (b[0] * c[1] - b[1] * c[0])) /
         6.0;
}

// Independent oriented incidence oracle, reconstructed from published cells
// rather than generator cavity state. Every exterior face occurs once and
// has declared facet ancestry; each interior face occurs twice oppositely.
void check_cover_incidence(const Plc& plc, const PlcRecovery& r) {
  using Key = std::array<int32_t, 3>;
  std::map<Key, std::pair<int, int>> incidence;
  constexpr int slots[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  for (std::size_t t = 0; t < r.tets.size() / 4; ++t) {
    for (const auto& slot : slots) {
      Key face{r.tets[4 * t + slot[0]], r.tets[4 * t + slot[1]], r.tets[4 * t + slot[2]]};
      const int inversions = (face[0] > face[1]) + (face[0] > face[2]) + (face[1] > face[2]);
      std::sort(face.begin(), face.end());
      auto& entry = incidence[face];
      ++entry.first;
      entry.second += inversions % 2 ? -1 : 1;
    }
  }
  std::map<Key, int32_t> constraints;
  for (std::size_t f = 0; f < r.faces.size() / 3; ++f) {
    Key face{r.faces[3 * f], r.faces[3 * f + 1], r.faces[3 * f + 2]};
    std::sort(face.begin(), face.end());
    PHX_CHECK(constraints.emplace(face, r.face_sources[f]).second);
  }
  for (const auto& [face, count] : incidence) {
    PHX_CHECK(count.first == 1 || count.first == 2);
    if (count.first == 2) {
      PHX_CHECK(count.second == 0);
    } else {
      PHX_CHECK(constraints.count(face) == 1);
    }
  }
  for (const auto& [face, source] : constraints) {
    const bool boundary = plc.facet_regions[2 * source] < 0 ||
                          plc.facet_regions[2 * source + 1] < 0;
    PHX_CHECK(incidence.at(face).first == (boundary ? 1 : 2));
  }
}

void test_reflex_l_prism() {
  Plc plc;
  const double xy[12] = {0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2};
  for (int z = 0; z < 2; ++z) {
    for (int i = 0; i < 6; ++i) {
      plc.points.insert(plc.points.end(), {xy[2 * i], xy[2 * i + 1], static_cast<double>(z)});
    }
  }
  const int32_t facet = plc.facet(-1, 0);
  plc.polygon({5, 4, 3, 2, 1, 0}, facet);
  plc.polygon({6, 7, 8, 9, 10, 11}, facet);
  for (int i = 0; i < 6; ++i) {
    const int j = (i + 1) % 6;
    plc.polygon({i, j, j + 6, i + 6}, facet);
  }
  PlcRecovery r;
  // The planar CDT and recovery fit within 20,000 work units; the independent
  // 500-unit request below must refuse without exceeding its hard bound.
  const int32_t status = plc.run(BoundaryPolicy::kConforming, r, 20000);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) {
    return;
  }
  PHX_CHECK_NEAR(region_volumes(r).at(0), 3.0, 1e-13);
  double area = 0.0;
  for (std::size_t f = 0; f < r.face_sources.size(); ++f) {
    area += face_area(r, f);
  }
  PHX_CHECK_NEAR(area, 14.0, 1e-13);
  check_cover_incidence(plc, r);
  PlcRecovery depleted;
  PHX_CHECK(plc.run(BoundaryPolicy::kConforming, depleted, 500) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(depleted.failure.reason == phx::mc::kPlcWorkBudget);
  PHX_CHECK(depleted.counters.work_units <= 500);
}

// Extruded dumbbell whose blocks meet through a 0.2 x 0.02 x 1 channel. Its
// rectangular walls are exactly cocircular, so each polygon CDT diagonal is
// a tie; ties that cross the tetrahedralization's own wall diagonals would
// otherwise need an interior kernel point inside the channel, whose cells
// span its 0.02 width.
void test_thin_channel_prism(BoundaryPolicy policy) {
  const double xy[24] = {0.0, 0.0,  0.4, 0.0,  0.4, 0.19, 0.6, 0.19, 0.6, 0.0,  1.0, 0.0,
                         1.0, 0.4,  0.6, 0.4,  0.6, 0.21, 0.4, 0.21, 0.4, 0.4,  0.0, 0.4};
  Plc plc;
  for (int z = 0; z < 2; ++z) {
    for (int i = 0; i < 12; ++i) {
      plc.points.insert(plc.points.end(), {xy[2 * i], xy[2 * i + 1], static_cast<double>(z)});
    }
  }
  const int32_t facet = plc.facet(-1, 0);
  plc.polygon({11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, facet);
  plc.polygon({12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}, facet);
  for (int i = 0; i < 12; ++i) {
    const int j = (i + 1) % 12;
    plc.polygon({i, j, j + 12, i + 12}, facet);
  }
  PlcRecovery r;
  const int32_t status = plc.run(policy, r, 640000);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) {
    return;
  }
  PHX_CHECK(r.counters.interior_steiner_points == 0);
  PHX_CHECK(r.counters.facet_steiner_points == 0);
  PHX_CHECK(r.counters.segment_steiner_points == 0);
  PHX_CHECK(r.points.size() == plc.points.size());
  PHX_CHECK_NEAR(region_volumes(r).at(0), 0.324, 1e-14);
  double area = 0.0;
  for (std::size_t f = 0; f < r.face_sources.size(); ++f) {
    area += face_area(r, f);
  }
  PHX_CHECK_NEAR(area, 2.0 * 0.324 + 3.56, 1e-13);
  // The published input triangulation still tiles every polygon exactly once.
  std::vector<double> polygon_area(plc.polygon_facets.size(), 0.0);
  for (std::size_t t = 0; t < r.input_polygons.size(); ++t) {
    const double* p = plc.points.data();
    const int32_t* v = r.input_triangles.data() + 3 * t;
    double u[3], w[3];
    for (int k = 0; k < 3; ++k) {
      u[k] = p[3 * v[1] + k] - p[3 * v[0] + k];
      w[k] = p[3 * v[2] + k] - p[3 * v[0] + k];
    }
    const double n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                         u[0] * w[1] - u[1] * w[0]};
    polygon_area[static_cast<std::size_t>(r.input_polygons[t])] +=
        0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  }
  PHX_CHECK_NEAR(polygon_area[0], 0.324, 1e-14);
  PHX_CHECK_NEAR(polygon_area[1], 0.324, 1e-14);
  check_cover_incidence(plc, r);
}

void test_cube(BoundaryPolicy policy) {
  PlcRecovery r;
  PHX_CHECK(cube().run(policy, r) == PHX_MC_OK);
  const auto volumes = region_volumes(r);
  PHX_CHECK(volumes.size() == 1);
  PHX_CHECK_NEAR(volumes.at(0), 1.0, 1e-14);
  double area = 0.0;
  for (std::size_t f = 0; f < r.face_sources.size(); ++f) {
    area += face_area(r, f);
  }
  PHX_CHECK_NEAR(area, 6.0, 1e-14);
  PHX_CHECK(r.plc_edges.size() == 2 * 12);
  PHX_CHECK(r.counters.segment_steiner_points == 0);
}

// Two unit cubes sharing the interface x = 1 (regions 0 and 1).
void test_two_materials() {
  Plc plc;
  for (int i = 0; i < 12; ++i) {
    plc.points.insert(plc.points.end(), {static_cast<double>(i % 3), static_cast<double>((i / 3) % 2),
                                         static_cast<double>(i / 6)});
  }
  // Vertex (x, y, z) = x + 3 y + 6 z.
  const int32_t left = plc.facet(-1, 0);
  const int32_t right = plc.facet(-1, 1);
  const int32_t interface = plc.facet(1, 0);
  for (int32_t x = 0; x < 2; ++x) {
    const int32_t f = x == 0 ? left : right;
    const int32_t o = x;
    plc.polygon({o, o + 3, o + 4, o + 1}, f);            // z = 0
    plc.polygon({o + 6, o + 7, o + 10, o + 9}, f);       // z = 1
    plc.polygon({o, o + 1, o + 7, o + 6}, f);            // y = 0
    plc.polygon({o + 3, o + 9, o + 10, o + 4}, f);       // y = 1
  }
  plc.polygon({0, 6, 9, 3}, left);         // x = 0
  plc.polygon({2, 5, 11, 8}, right);       // x = 2
  plc.polygon({1, 4, 10, 7}, interface);   // x = 1, normal +x toward region 1
  PlcRecovery r;
  PHX_CHECK(plc.run(BoundaryPolicy::kFixed, r) == PHX_MC_OK);
  const auto volumes = region_volumes(r);
  PHX_CHECK(volumes.size() == 2);
  PHX_CHECK_NEAR(volumes.at(0), 1.0, 1e-14);
  PHX_CHECK_NEAR(volumes.at(1), 1.0, 1e-14);
  double interface_area = 0.0;
  for (std::size_t f = 0; f < r.face_sources.size(); ++f) {
    interface_area += r.face_sources[f] == interface ? face_area(r, f) : 0.0;
  }
  PHX_CHECK_NEAR(interface_area, 1.0, 1e-14);
}

Plc triple_junction() {
  Plc plc;
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        plc.points.insert(plc.points.end(),
                          {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)});
      }
    }
  }
  const auto vertex = [](int x, int y, int z) { return x + 3 * y + 9 * z; };
  const int exterior[3] = {plc.facet(-1, 0), plc.facet(-1, 1), plc.facet(-1, 2)};
  const int junction[3] = {plc.facet(1, 0), plc.facet(2, 0), plc.facet(2, 1)};
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
      const int region = y == 1 ? 2 : x;
      const int facet = exterior[region];
      plc.polygon({vertex(x, y, 0), vertex(x, y + 1, 0), vertex(x + 1, y + 1, 0),
                   vertex(x + 1, y, 0)}, facet);
      plc.polygon({vertex(x, y, 1), vertex(x + 1, y, 1), vertex(x + 1, y + 1, 1),
                   vertex(x, y + 1, 1)}, facet);
      if (x == 0) {
        plc.polygon({vertex(0, y, 0), vertex(0, y, 1), vertex(0, y + 1, 1),
                     vertex(0, y + 1, 0)}, facet);
      } else {
        plc.polygon({vertex(2, y, 0), vertex(2, y + 1, 0), vertex(2, y + 1, 1),
                     vertex(2, y, 1)}, facet);
      }
      if (y == 0) {
        plc.polygon({vertex(x, 0, 0), vertex(x + 1, 0, 0), vertex(x + 1, 0, 1),
                     vertex(x, 0, 1)}, facet);
      } else {
        plc.polygon({vertex(x, 2, 0), vertex(x, 2, 1), vertex(x + 1, 2, 1),
                     vertex(x + 1, 2, 0)}, facet);
      }
    }
  }
  plc.polygon({vertex(1, 0, 0), vertex(1, 1, 0), vertex(1, 1, 1), vertex(1, 0, 1)}, junction[0]);
  for (int x = 0; x < 2; ++x) {
    plc.polygon({vertex(x, 1, 0), vertex(x, 1, 1), vertex(x + 1, 1, 1),
                 vertex(x + 1, 1, 0)}, junction[x + 1]);
  }
  return plc;
}

void test_material_triple_junction() {
  const Plc plc = triple_junction();
  PlcRecovery r;
  PHX_CHECK(plc.run(BoundaryPolicy::kFixed, r, 20000) == PHX_MC_OK);
  const auto volumes = region_volumes(r);
  PHX_CHECK_NEAR(volumes.at(0), 1.0, 1e-13);
  PHX_CHECK_NEAR(volumes.at(1), 1.0, 1e-13);
  PHX_CHECK_NEAR(volumes.at(2), 2.0, 1e-13);
  check_cover_incidence(plc, r);
}

// Twisted triangular prism whose side diagonals are reflex (Schönhardt).
Plc schonhardt() {
  Plc plc;
  plc.points = {1.0,  0.0,    0.0, -0.5, 0.875, 0.0, -0.5, -0.875, 0.0,
                0.875, 0.5, 1.0, -0.875, 0.5,  1.0, 0.0,  -1.0,  1.0};
  const int32_t facet = plc.facet(-1, 0);
  plc.polygon({0, 2, 1}, facet);
  plc.polygon({3, 4, 5}, facet);
  // Side quad (a_i, a_j, b_j, b_i) split along the diagonal that is not an
  // edge of the convex hull.
  for (int32_t i = 0; i < 3; ++i) {
    const int32_t j = (i + 1) % 3;
    const int32_t a = i, b = j, c = j + 3, d = i + 3;
    const double* p = plc.points.data();
    bool hull = true;
    int sign = 0;
    for (int32_t k = 0; k < 6; ++k) {
      if (k == a || k == b || k == c) {
        continue;
      }
      const int s = phx::mc::orient3d(p + 3 * a, p + 3 * b, p + 3 * c, p + 3 * k);
      hull = hull && (sign == 0 || s == sign);
      sign = sign == 0 ? s : sign;
    }
    if (hull) {
      plc.polygon({a, b, d}, facet);
      plc.polygon({b, c, d}, facet);
    } else {
      plc.polygon({a, b, c}, facet);
      plc.polygon({a, c, d}, facet);
    }
  }
  return plc;
}

void test_schonhardt() {
  const Plc plc = schonhardt();
  PlcRecovery conforming;
  PHX_CHECK(plc.run(BoundaryPolicy::kConforming, conforming) == PHX_MC_OK);
  const auto volumes = region_volumes(conforming);
  // Volume of the closed triangle mesh by the divergence theorem.
  // This oracle is independent of cavity/refinement ancestry.
  double reference = 0.0;
  for (std::size_t p = 0; p + 1 < plc.offsets.size(); ++p) {
    const int32_t* loop = plc.loops.data() + plc.offsets[p];
    const double* a = plc.points.data() + 3 * loop[0];
    const double* b = plc.points.data() + 3 * loop[1];
    const double* c = plc.points.data() + 3 * loop[2];
    reference += source_triangle_volume(a, b, c);
  }
  PHX_CHECK_NEAR(volumes.at(0), reference, 1e-13);
  PlcRecovery fixed;
  PHX_CHECK(plc.run(BoundaryPolicy::kFixed, fixed) == PHX_MC_OK);
  PHX_CHECK_NEAR(region_volumes(fixed).at(0), reference, 1e-13);
  PHX_CHECK(fixed.counters.segment_steiner_points == 0);
  PHX_CHECK(fixed.counters.facet_steiner_points == 0);
  PHX_CHECK(std::equal(plc.points.begin(), plc.points.end(), fixed.points.begin()));
  std::map<std::array<int32_t, 3>, int> expected, actual;
  for (std::size_t p = 0; p + 1 < plc.offsets.size(); ++p) {
    std::array<int32_t, 3> face{plc.loops[plc.offsets[p]], plc.loops[plc.offsets[p] + 1],
                                plc.loops[plc.offsets[p] + 2]};
    std::sort(face.begin(), face.end());
    ++expected[face];
  }
  for (std::size_t f = 0; f < fixed.face_sources.size(); ++f) {
    std::array<int32_t, 3> face{fixed.faces[3 * f], fixed.faces[3 * f + 1],
                                fixed.faces[3 * f + 2]};
    std::sort(face.begin(), face.end());
    ++actual[face];
  }
  PHX_CHECK(actual == expected);
  check_cover_incidence(plc, fixed);
  check_cover_incidence(plc, conforming);
}

// Freudenthal categorical reconstruction of a two-label 2 x 1 x 1 image with
// unit voxels at the integers (labels "left" and "right"): 276 single-triangle
// facets, many coplanar with their neighbors, with coordinates in twelfths.
// Most of its PLC edges have no exactly representable split point, so both
// policies must recover them by reconnection, those between coplanar facets
// from both sides of their common planar component.
constexpr int8_t kCategoricalTwelfths[] = {
    -2, -4, -6, -3, -6, -6, -6, -6, -6, 0, -6, -6, -3, -3, -6, 0, 0, -6, 0, -3, -6, -2, -6, -4,
    -3, -6, -3, 0, -6, 0, 0, -6, -3, -4, -2, -6, -6, -3, -6, -6, 0, -6, -3, 0, -6, -6, -2, -4, -6,
    -3, -3, -6, 0, 0, -6, 0, -3, -4, -6, -2, -6, -6, -3, -6, -6, 0, -3, -6, 0, -6, -4, -2, -6, -3,
    0, -2, -4, 2, -3, -3, 3, 0, 0, 6, 0, -3, 3, -4, -2, 2, -3, 0, 3, -2, 2, -4, -3, 3, -3, 0, 6,
    0, 0, 3, -3, -4, 2, -2, -3, 3, 0, -2, 4, 2, -3, 3, 3, 0, 6, 6, 0, 6, 3, -2, 2, 4, 0, 3, 6, 10,
    -4, -6, 9, -6, -6, 6, -6, -6, 12, -6, -6, 9, -3, -6, 12, 0, -6, 12, -3, -6, 10, -6, -4, 9, -6,
    -3, 12, -6, 0, 12, -6, -3, 8, -2, -6, 6, -3, -6, 6, 0, -6, 9, 0, -6, 4, -2, -4, 4, -4, -4, 6,
    0, 0, 4, 0, -4, 8, -6, -2, 6, -6, -3, 6, -6, 0, 9, -6, 0, 4, -4, -2, 4, -4, 0, 10, -4, 2, 9,
    -3, 3, 12, 0, 6, 12, -3, 3, 6, -2, 2, 3, -3, 3, 6, 0, 6, 8, 0, 4, 2, -2, 4, 3, 0, 6, 10, 2,
    -4, 9, 3, -3, 12, 6, 0, 12, 3, -3, 6, 2, -2, 3, 3, -3, 6, 6, 0, 8, 4, 0, 2, 4, -2, 3, 6, 0, 8,
    4, 2, 8, 4, 4, 6, 6, 6, 12, 6, 6, 6, 6, 3, 12, 6, 3, 8, 2, 4, 6, 3, 6, 12, 3, 6, 4, 6, 2, 3,
    6, 3, 2, 6, 4, 3, 6, 6, 4, 2, 6, 3, 3, 6, 2, 4, 6, 14, -2, -4, 15, -3, -3, 18, 0, 0, 15, 0,
    -3, 14, -4, -2, 15, -3, 0, 16, -2, 2, 15, -3, 3, 18, 0, 6, 18, 0, 3, 14, -2, 4, 15, 0, 6, 16,
    2, -2, 15, 3, -3, 18, 6, 0, 18, 3, 0, 14, 4, -2, 15, 6, 0, 18, 4, 2, 18, 3, 3, 18, 6, 6, 18,
    6, 3, 18, 2, 4, 18, 3, 6, 16, 6, 2, 15, 6, 3, 14, 6, 4, 15, 6, 6, 16, 2, 6, 15, 3, 6, 14, 4,
    6};

// Triangle vertices and declared sides: 0 = (void, left), 1 = (void, right),
// 2 = (right, left) for the (positive, negative) facet regions.
constexpr int16_t kCategoricalTriangles[] = {
    2, 0, 1, 0, 3, 1, 0, 0, 2, 4, 0, 0, 5, 0, 4, 0, 3, 0, 6, 0, 5, 6, 0, 0, 2, 1, 7, 0, 3, 7, 1,
    0, 2, 7, 8, 0, 9, 8, 7, 0, 3, 10, 7, 0, 9, 7, 10, 0, 2, 12, 11, 0, 13, 11, 12, 0, 2, 11, 4, 0,
    5, 4, 11, 0, 13, 14, 11, 0, 5, 11, 14, 0, 2, 15, 12, 0, 13, 12, 15, 0, 2, 16, 15, 0, 17, 15,
    16, 0, 13, 15, 18, 0, 17, 18, 15, 0, 2, 19, 20, 0, 21, 20, 19, 0, 2, 8, 19, 0, 9, 19, 8, 0,
    21, 19, 22, 0, 9, 22, 19, 0, 2, 20, 23, 0, 21, 23, 20, 0, 2, 23, 16, 0, 17, 16, 23, 0, 21, 24,
    23, 0, 17, 23, 24, 0, 21, 22, 25, 0, 9, 25, 22, 0, 21, 25, 26, 0, 27, 26, 25, 0, 9, 28, 25, 0,
    27, 25, 28, 0, 21, 29, 24, 0, 17, 24, 29, 0, 21, 26, 29, 0, 27, 29, 26, 0, 17, 29, 30, 0, 27,
    30, 29, 0, 13, 31, 14, 0, 5, 14, 31, 0, 13, 32, 31, 0, 33, 31, 32, 0, 5, 31, 34, 0, 33, 34,
    31, 0, 13, 18, 35, 0, 17, 35, 18, 0, 13, 35, 32, 0, 33, 32, 35, 0, 17, 36, 35, 0, 33, 35, 36,
    0, 17, 37, 36, 0, 33, 36, 37, 0, 17, 38, 37, 0, 39, 37, 38, 0, 33, 37, 40, 0, 39, 40, 37, 0,
    17, 30, 41, 0, 27, 41, 30, 0, 17, 41, 38, 0, 39, 38, 41, 0, 27, 42, 41, 0, 39, 41, 42, 0, 45,
    43, 44, 1, 46, 44, 43, 1, 45, 47, 43, 1, 48, 43, 47, 1, 46, 43, 49, 1, 48, 49, 43, 1, 45, 44,
    50, 1, 46, 50, 44, 1, 45, 50, 51, 1, 52, 51, 50, 1, 46, 53, 50, 1, 52, 50, 53, 1, 45, 55, 54,
    1, 56, 54, 55, 1, 45, 54, 47, 1, 48, 47, 54, 1, 56, 57, 54, 1, 48, 54, 57, 1, 3, 6, 58, 0, 5,
    58, 6, 0, 45, 58, 55, 1, 56, 55, 58, 1, 3, 58, 59, 0, 45, 59, 58, 1, 60, 59, 58, 2, 5, 61, 58,
    0, 56, 58, 61, 1, 60, 58, 61, 2, 45, 62, 63, 1, 64, 63, 62, 1, 45, 51, 62, 1, 52, 62, 51, 1,
    64, 62, 65, 1, 52, 65, 62, 1, 3, 66, 10, 0, 9, 10, 66, 0, 45, 63, 66, 1, 64, 66, 63, 1, 3, 59,
    66, 0, 45, 66, 59, 1, 60, 66, 59, 2, 9, 66, 67, 0, 64, 67, 66, 1, 60, 67, 66, 2, 64, 65, 68,
    1, 52, 68, 65, 1, 64, 68, 69, 1, 70, 69, 68, 1, 52, 71, 68, 1, 70, 68, 71, 1, 9, 67, 72, 0,
    64, 72, 67, 1, 60, 72, 67, 2, 9, 72, 73, 0, 74, 73, 72, 0, 64, 69, 72, 1, 70, 72, 69, 1, 60,
    75, 72, 2, 74, 72, 75, 0, 70, 75, 72, 1, 9, 76, 28, 0, 27, 28, 76, 0, 9, 73, 76, 0, 74, 76,
    73, 0, 27, 76, 77, 0, 74, 77, 76, 0, 56, 78, 57, 1, 48, 57, 78, 1, 56, 79, 78, 1, 80, 78, 79,
    1, 48, 78, 81, 1, 80, 81, 78, 1, 5, 82, 61, 0, 56, 61, 82, 1, 60, 61, 82, 2, 5, 83, 82, 0, 84,
    82, 83, 0, 56, 82, 79, 1, 80, 79, 82, 1, 60, 82, 85, 2, 84, 85, 82, 0, 80, 82, 85, 1, 5, 34,
    86, 0, 33, 86, 34, 0, 5, 86, 83, 0, 84, 83, 86, 0, 33, 87, 86, 0, 84, 86, 87, 0, 60, 85, 88,
    2, 84, 88, 85, 0, 80, 85, 88, 1, 60, 88, 89, 2, 90, 89, 88, 0, 91, 88, 89, 1, 84, 92, 88, 0,
    90, 88, 92, 0, 80, 88, 93, 1, 91, 93, 88, 1, 60, 94, 75, 2, 74, 75, 94, 0, 70, 94, 75, 1, 60,
    89, 94, 2, 90, 94, 89, 0, 91, 89, 94, 1, 74, 94, 95, 0, 90, 95, 94, 0, 70, 96, 94, 1, 91, 94,
    96, 1, 33, 97, 87, 0, 84, 87, 97, 0, 33, 98, 97, 0, 90, 97, 98, 0, 84, 97, 92, 0, 90, 92, 97,
    0, 33, 40, 99, 0, 39, 99, 40, 0, 33, 99, 98, 0, 90, 98, 99, 0, 39, 100, 99, 0, 90, 99, 100, 0,
    27, 77, 101, 0, 74, 101, 77, 0, 27, 101, 102, 0, 90, 102, 101, 0, 74, 95, 101, 0, 90, 101, 95,
    0, 27, 103, 42, 0, 39, 42, 103, 0, 27, 102, 103, 0, 90, 103, 102, 0, 39, 103, 100, 0, 90, 100,
    103, 0, 46, 49, 104, 1, 48, 104, 49, 1, 46, 104, 105, 1, 106, 105, 104, 1, 48, 107, 104, 1,
    106, 104, 107, 1, 46, 108, 53, 1, 52, 53, 108, 1, 46, 105, 108, 1, 106, 108, 105, 1, 52, 108,
    109, 1, 106, 109, 108, 1, 52, 109, 110, 1, 106, 110, 109, 1, 52, 110, 111, 1, 112, 111, 110,
    1, 106, 113, 110, 1, 112, 110, 113, 1, 52, 114, 71, 1, 70, 71, 114, 1, 52, 111, 114, 1, 112,
    114, 111, 1, 70, 114, 115, 1, 112, 115, 114, 1, 48, 116, 107, 1, 106, 107, 116, 1, 48, 117,
    116, 1, 118, 116, 117, 1, 106, 116, 119, 1, 118, 119, 116, 1, 48, 81, 120, 1, 80, 120, 81, 1,
    48, 120, 117, 1, 118, 117, 120, 1, 80, 121, 120, 1, 118, 120, 121, 1, 106, 119, 122, 1, 118,
    122, 119, 1, 106, 122, 123, 1, 124, 123, 122, 1, 118, 125, 122, 1, 124, 122, 125, 1, 106, 126,
    113, 1, 112, 113, 126, 1, 106, 123, 126, 1, 124, 126, 123, 1, 112, 126, 127, 1, 124, 127, 126,
    1, 80, 128, 121, 1, 118, 121, 128, 1, 80, 129, 128, 1, 124, 128, 129, 1, 118, 128, 125, 1,
    124, 125, 128, 1, 80, 93, 130, 1, 91, 130, 93, 1, 80, 130, 129, 1, 124, 129, 130, 1, 91, 131,
    130, 1, 124, 130, 131, 1, 70, 115, 132, 1, 112, 132, 115, 1, 70, 132, 133, 1, 124, 133, 132,
    1, 112, 127, 132, 1, 124, 132, 127, 1, 70, 134, 96, 1, 91, 96, 134, 1, 70, 133, 134, 1, 124,
    134, 133, 1, 91, 134, 131, 1, 124, 131, 134, 1};

Plc categorical_interface() {
  Plc plc;
  for (const int8_t twelfths : kCategoricalTwelfths) {
    plc.points.push_back(twelfths / 12.0);
  }
  constexpr int32_t kSides[3][2] = {{-1, 0}, {-1, 1}, {1, 0}};
  for (std::size_t t = 0; t < std::size(kCategoricalTriangles); t += 4) {
    const int32_t* sides = kSides[kCategoricalTriangles[t + 3]];
    plc.polygon({kCategoricalTriangles[t], kCategoricalTriangles[t + 1],
                 kCategoricalTriangles[t + 2]},
                plc.facet(sides[0], sides[1]));
  }
  return plc;
}

void test_categorical_interface(BoundaryPolicy policy) {
  const Plc plc = categorical_interface();
  // Region measures by the divergence theorem over the declared facets: a
  // facet bounds its negative region outward and its positive region inward.
  std::map<int32_t, double> reference;
  std::vector<double> facet_area;
  for (std::size_t p = 0; p + 1 < plc.offsets.size(); ++p) {
    const int32_t* loop = plc.loops.data() + plc.offsets[p];
    const double* a = plc.points.data() + 3 * loop[0];
    const double* b = plc.points.data() + 3 * loop[1];
    const double* c = plc.points.data() + 3 * loop[2];
    const double volume = source_triangle_volume(a, b, c);
    reference[plc.facet_regions[2 * p + 1]] += volume;
    if (plc.facet_regions[2 * p] >= 0) {
      reference[plc.facet_regions[2 * p]] -= volume;
    }
    facet_area.push_back(face_area(TriangleSurface{plc.points, {loop, 3}}, 0));
  }
  PlcRecovery r;
  const int32_t status = plc.run(policy, r);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) {
    return;
  }
  const auto volumes = region_volumes(r);
  PHX_CHECK(volumes.size() == 2);
  PHX_CHECK_NEAR(volumes.at(0), reference.at(0), 1e-13);
  PHX_CHECK_NEAR(volumes.at(1), reference.at(1), 1e-13);
  check_cover_incidence(plc, r);
  // Every facet keeps its whole area through the ancestry of its subfacets.
  std::vector<double> covered(facet_area.size(), 0.0);
  for (std::size_t f = 0; f < r.face_sources.size(); ++f) {
    covered[static_cast<std::size_t>(r.face_sources[f])] += face_area(r, f);
  }
  for (std::size_t facet = 0; facet < facet_area.size(); ++facet) {
    PHX_CHECK_NEAR(covered[facet], facet_area[facet], 1e-14);
  }
  if (policy == BoundaryPolicy::kFixed) {
    PHX_CHECK(r.counters.segment_steiner_points == 0);
    PHX_CHECK(r.counters.facet_steiner_points == 0);
    PHX_CHECK(r.face_sources.size() == facet_area.size());
  }
}

void test_failures() {
  Plc open = cube();
  open.offsets.pop_back();
  open.loops.resize(static_cast<std::size_t>(open.offsets.back()));
  open.polygon_facets.pop_back();
  PlcRecovery r;
  PHX_CHECK(open.run(BoundaryPolicy::kConforming, r) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(r.failure.reason == phx::mc::kPlcOpenBoundary);
  PHX_CHECK(r.failure.first_kind == phx::mc::kPlcRegion && r.failure.first == 0);

  // A second cube shifted into the first: facets cross.
  Plc crossing = cube();
  const Plc other = cube();
  for (std::size_t k = 0; k < other.points.size(); ++k) {
    crossing.points.push_back(other.points[k] + 0.5);
  }
  const int32_t facet = crossing.facet(-1, 1);
  for (std::size_t p = 0; p + 1 < other.offsets.size(); ++p) {
    std::vector<int32_t> loop;
    for (int64_t k = other.offsets[p]; k < other.offsets[p + 1]; ++k) {
      loop.push_back(other.loops[static_cast<std::size_t>(k)] + 8);
    }
    crossing.polygon(loop, facet);
  }
  PlcRecovery c;
  PHX_CHECK(crossing.run(BoundaryPolicy::kConforming, c) == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK(c.failure.reason == phx::mc::kPlcIntersecting);

  PlcRecovery budget;
  PHX_CHECK(schonhardt().run(BoundaryPolicy::kConforming, budget, 40) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(budget.failure.reason == phx::mc::kPlcWorkBudget);

  // Reversed orientation of the whole cube: region 0 would be unbounded.
  Plc inverted = cube();
  inverted.facet_regions = {0, -1};
  PlcRecovery leak;
  PHX_CHECK(inverted.run(BoundaryPolicy::kConforming, leak) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(leak.failure.reason == phx::mc::kPlcRegionLeak);
}

// A cavity limit too small for a recovery edit is a resource refusal with the
// tetrahedron-budget reason, never an internal error.
void test_cavity_limit_refusal() {
  struct Case {
    Plc plc;
    uint64_t cavity_limit;
  };
  const Case cases[] = {{schonhardt(), 11}, {categorical_interface(), 64}};
  for (const Case& c : cases) {
    phx::mc::NativeExecutionScope scope(std::numeric_limits<uint64_t>::max(),
                                        std::numeric_limits<uint64_t>::max(), c.cavity_limit,
                                        std::numeric_limits<std::size_t>::max(),
                                        std::numeric_limits<double>::infinity());
    PlcRecovery r;
    PHX_CHECK(c.plc.run(BoundaryPolicy::kFixed, r) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(r.failure.reason == phx::mc::kPlcTetrahedronBudget);
    PHX_CHECK(r.failure.first_kind == phx::mc::kPlcEdge);
    PHX_CHECK(scope.status() == PHX_MC_CAPACITY_EXCEEDED);
  }
}

phx_mc_plc3d* recover_handle(const Plc& plc) {
  phx_mc_plc3d* handle = nullptr;
  phx_mc_plc3d_recover_request request{};
  request.point_count = static_cast<int64_t>(plc.points.size() / 3);
  request.points = plc.points.data();
  request.polygon_count = static_cast<int64_t>(plc.polygon_facets.size());
  request.polygon_offsets = plc.offsets.data();
  request.polygon_vertices = plc.loops.data();
  request.polygon_facets = plc.polygon_facets.data();
  request.facet_count = static_cast<int64_t>(plc.facet_regions.size() / 2);
  request.facet_regions = plc.facet_regions.data();
  request.segment_count = static_cast<int64_t>(plc.segments.size() / 2);
  request.segments = plc.segments.data();
  request.boundary_policy = PHX_MC_PLC3D_FIXED;
  request.max_vertices = 100000;
  request.max_tetrahedra = 1000000;
  request.work_limit = int64_t{1} << 40;
  request.maximum_scratch_bytes = std::numeric_limits<uint64_t>::max();
  request.result = &handle;
  const int32_t status = phx_mc_plc3d_recover(&request);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  if (status != PHX_MC_OK) {
    phx_mc_plc3d_free(handle);
    return nullptr;
  }
  return handle;
}

void test_getters_reject_null_handles() {
  std::array<int64_t, PHX_MC_PLC3D_COUNTERS> output;
  output.fill(-7);
  double refusal[PHX_MC_PLC3D_SOURCE_REFUSAL_VALUES] = {-7.0, -7.0};
  PHX_CHECK(phx_mc_plc3d_sizes(nullptr, output.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_counters(nullptr, output.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_failure(nullptr, output.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_export(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_source_witnesses(nullptr, nullptr, nullptr, nullptr, nullptr,
                                        refusal) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(std::all_of(output.begin(), output.end(), [](int64_t v) { return v == -7; }));
  PHX_CHECK(refusal[0] == -7.0 && refusal[1] == -7.0);
}

void test_diagnostic_getter_outputs() {
  phx_mc_plc3d* handle = recover_handle(cube());
  if (handle == nullptr) {
    return;
  }
  PHX_CHECK(phx_mc_plc3d_sizes(handle, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_counters(handle, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_failure(handle, nullptr) == PHX_MC_INVALID_ARGUMENT);
  int64_t sizes[PHX_MC_PLC3D_SIZES];
  int64_t counters[PHX_MC_PLC3D_COUNTERS];
  int64_t failure[PHX_MC_PLC3D_FAILURE];
  PHX_CHECK(phx_mc_plc3d_sizes(handle, sizes) == PHX_MC_OK);
  PHX_CHECK(phx_mc_plc3d_counters(handle, counters) == PHX_MC_OK);
  PHX_CHECK(phx_mc_plc3d_failure(handle, failure) == PHX_MC_OK);
  PHX_CHECK(sizes[PHX_MC_PLC3D_SIZE_POINTS] == 8);
  PHX_CHECK(sizes[PHX_MC_PLC3D_SIZE_TETRAHEDRA] > 0);
  PHX_CHECK(counters[PHX_MC_PLC3D_WORK_UNITS] > 0);
  PHX_CHECK(failure[PHX_MC_PLC3D_FAILURE_REASON] == phx::mc::kPlcNoFailure);
  phx_mc_plc3d_free(handle);
}

void test_export_checks_all_buffers_before_copying() {
  const Plc plc = cube();
  phx_mc_plc3d* handle = recover_handle(plc);
  if (handle == nullptr) {
    return;
  }
  int64_t sizes[PHX_MC_PLC3D_SIZES];
  const int32_t size_status = phx_mc_plc3d_sizes(handle, sizes);
  PHX_CHECK(size_status == PHX_MC_OK);
  if (size_status != PHX_MC_OK) {
    phx_mc_plc3d_free(handle);
    return;
  }
  PlcRecovery out;
  const auto points = static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_POINTS]);
  const auto tets = static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_TETRAHEDRA]);
  const auto faces = static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_FACES]);
  const auto segments = static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_SEGMENTS]);
  const auto edges = static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_PLC_EDGES]);
  const auto triangles = static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_INPUT_TRIANGLES]);
  out.points.assign(3 * points, -7.0);
  out.tets.assign(4 * tets, -7);
  out.tet_regions.assign(tets, -7);
  out.faces.assign(3 * faces, -7);
  out.face_sources.assign(faces, -7);
  out.segments.assign(2 * segments, -7);
  out.segment_sources.assign(segments, -7);
  out.protection.assign(points, -7.0);
  out.plc_edges.assign(2 * edges, -7);
  out.vertex_dimension.assign(points, -7);
  out.input_triangles.assign(3 * triangles, -7);
  out.input_polygons.assign(triangles, -7);
  std::array<void*, 12> buffers{
      out.points.data(), out.tets.data(), out.tet_regions.data(), out.faces.data(),
      out.face_sources.data(), out.segments.data(), out.segment_sources.data(),
      out.protection.data(), out.plc_edges.data(), out.vertex_dimension.data(),
      out.input_triangles.data(), out.input_polygons.data()};
  constexpr const char* names[] = {
      "missing points", "missing tets", "missing tet_regions", "missing faces",
      "missing face_sources", "missing segments", "missing segment_sources",
      "missing protection", "missing plc_edges", "missing vertex_dimension",
      "missing input_triangles", "missing input_polygons"};
  const auto export_buffers = [&]() {
    return phx_mc_plc3d_export(
        handle, static_cast<double*>(buffers[0]), static_cast<int32_t*>(buffers[1]),
        static_cast<int32_t*>(buffers[2]), static_cast<int32_t*>(buffers[3]),
        static_cast<int32_t*>(buffers[4]), static_cast<int32_t*>(buffers[5]),
        static_cast<int32_t*>(buffers[6]), static_cast<double*>(buffers[7]),
        static_cast<int32_t*>(buffers[8]), static_cast<int8_t*>(buffers[9]),
        static_cast<int32_t*>(buffers[10]), static_cast<int32_t*>(buffers[11]));
  };
  for (std::size_t k = 0; k < buffers.size(); ++k) {
    void* buffer = buffers[k];
    buffers[k] = nullptr;
    if (export_buffers() != PHX_MC_INVALID_ARGUMENT) {
      phx::mc::test::record_failure(__FILE__, __LINE__, names[k]);
    }
    buffers[k] = buffer;
  }
  const auto untouched = [](const auto& values) {
    return std::all_of(values.begin(), values.end(), [](auto value) { return value == -7; });
  };
  PHX_CHECK(untouched(out.points));
  PHX_CHECK(untouched(out.tets));
  PHX_CHECK(untouched(out.tet_regions));
  PHX_CHECK(untouched(out.faces));
  PHX_CHECK(untouched(out.face_sources));
  PHX_CHECK(untouched(out.segments));
  PHX_CHECK(untouched(out.segment_sources));
  PHX_CHECK(untouched(out.protection));
  PHX_CHECK(untouched(out.plc_edges));
  PHX_CHECK(untouched(out.vertex_dimension));
  PHX_CHECK(untouched(out.input_triangles));
  PHX_CHECK(untouched(out.input_polygons));
  PHX_CHECK(export_buffers() == PHX_MC_OK);
  PHX_CHECK_NEAR(region_volumes(out).at(0), 1.0, 1e-14);
  check_cover_incidence(plc, out);
  phx_mc_plc3d_free(handle);
}

void test_witness_output_contract() {
  phx_mc_plc3d* handle = recover_handle(cube());
  if (handle == nullptr) {
    return;
  }
  int64_t sizes[PHX_MC_PLC3D_SIZES];
  const int32_t size_status = phx_mc_plc3d_sizes(handle, sizes);
  PHX_CHECK(size_status == PHX_MC_OK);
  if (size_status != PHX_MC_OK) {
    phx_mc_plc3d_free(handle);
    return;
  }
  const auto points = static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_POINTS]);
  std::vector<int8_t> strata(points, -7);
  std::vector<int32_t> entities(points, -7);
  std::vector<double> parameters(2 * points, -7.0);
  std::vector<double> deviations(points, -7.0);
  double refusal[PHX_MC_PLC3D_SOURCE_REFUSAL_VALUES] = {-7.0, -7.0};
  PHX_CHECK(phx_mc_plc3d_source_witnesses(handle, nullptr, entities.data(), parameters.data(),
                                        deviations.data(), refusal) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_source_witnesses(handle, strata.data(), nullptr, parameters.data(),
                                        deviations.data(), refusal) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_source_witnesses(handle, strata.data(), entities.data(), nullptr,
                                        deviations.data(), refusal) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_source_witnesses(handle, strata.data(), entities.data(), parameters.data(),
                                        nullptr, refusal) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(std::all_of(strata.begin(), strata.end(), [](int8_t v) { return v == -7; }));
  PHX_CHECK(std::all_of(entities.begin(), entities.end(), [](int32_t v) { return v == -7; }));
  PHX_CHECK(std::all_of(parameters.begin(), parameters.end(), [](double v) { return v == -7.0; }));
  PHX_CHECK(std::all_of(deviations.begin(), deviations.end(), [](double v) { return v == -7.0; }));
  PHX_CHECK(refusal[0] == -7.0 && refusal[1] == -7.0);
  PHX_CHECK(phx_mc_plc3d_source_witnesses(handle, strata.data(), entities.data(), parameters.data(),
                                        deviations.data(), nullptr) == PHX_MC_OK);
  PHX_CHECK(phx_mc_plc3d_source_witnesses(handle, strata.data(), entities.data(), parameters.data(),
                                        deviations.data(), refusal) == PHX_MC_OK);
  PHX_CHECK(refusal[PHX_MC_PLC3D_SOURCE_REFUSAL_CERTIFIED_DEVIATION] == 0.0);
  PHX_CHECK(refusal[PHX_MC_PLC3D_SOURCE_REFUSAL_DECLARED_BOUND] == 0.0);
  phx_mc_plc3d_free(handle);
}

void test_source_export_allows_empty_banks() {
  const Plc plc = cube();
  phx_mc_plc3d* handle = nullptr;
  const int32_t source_status = phx_mc_plc3d_source_constraints(
      static_cast<int64_t>(plc.points.size() / 3), plc.points.data(),
      static_cast<int64_t>(plc.polygon_facets.size()), plc.offsets.data(),
      plc.loops.data(), plc.polygon_facets.data(),
      static_cast<int64_t>(plc.facet_regions.size() / 2), plc.facet_regions.data(),
      0, nullptr, int64_t{1} << 40, std::numeric_limits<uint64_t>::max(), 0, &handle);
  PHX_CHECK(source_status == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  if (source_status != PHX_MC_OK || handle == nullptr) {
    phx_mc_plc3d_free(handle);
    return;
  }
  int64_t sizes[PHX_MC_PLC3D_SIZES];
  const int32_t size_status = phx_mc_plc3d_sizes(handle, sizes);
  PHX_CHECK(size_status == PHX_MC_OK);
  if (size_status != PHX_MC_OK) {
    phx_mc_plc3d_free(handle);
    return;
  }
  std::vector<double> points(3 * static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_POINTS]));
  std::vector<int32_t> edges(2 * static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_PLC_EDGES]));
  std::vector<int32_t> triangles(
      3 * static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_INPUT_TRIANGLES]));
  std::vector<int32_t> polygons(
      static_cast<std::size_t>(sizes[PHX_MC_PLC3D_SIZE_INPUT_TRIANGLES]));
  PHX_CHECK(phx_mc_plc3d_export(handle, points.data(), nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, edges.data(), nullptr,
                              triangles.data(), polygons.data()) == PHX_MC_OK);
  PHX_CHECK(points == plc.points);
  PHX_CHECK(phx_mc_plc3d_source_witnesses(handle, nullptr, nullptr, nullptr, nullptr,
                                        nullptr) == PHX_MC_OK);
  phx_mc_plc3d_free(handle);
}

void test_planar_recovery_debits_global_work_once() {
  const Plc plc = cube();
  PlcRecovery baseline;
  PHX_CHECK(plc.run(BoundaryPolicy::kFixed, baseline) == PHX_MC_OK);
  const uint64_t required = static_cast<uint64_t>(baseline.counters.work_units);
  PHX_CHECK(required > 1);
  {
    phx::mc::NativeExecutionScope scope(
        required, std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(), 1 << 24, INFINITY);
    PlcRecovery result;
    PHX_CHECK(plc.run(BoundaryPolicy::kFixed, result,
                      static_cast<int64_t>(required)) == PHX_MC_OK);
    PHX_CHECK(result.counters.work_units == baseline.counters.work_units);
    const auto work = scope.evidence();
    PHX_CHECK(work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == required);
    PHX_CHECK(work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 0);
    PHX_CHECK(work[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
    PHX_CHECK(result.points == baseline.points && result.tets == baseline.tets);
  }
  {
    phx::mc::NativeExecutionScope scope(
        required - 1, std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(), 1 << 24, INFINITY);
    PlcRecovery result;
    PHX_CHECK(plc.run(BoundaryPolicy::kFixed, result,
                      static_cast<int64_t>(required)) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(result.failure.reason == phx::mc::kPlcWorkBudget);
    const auto work = scope.evidence();
    PHX_CHECK(work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] ==
              static_cast<uint64_t>(result.counters.work_units));
    PHX_CHECK(work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] <= required - 1);
    PHX_CHECK(work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 0);
    PHX_CHECK(work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
  }
}

}  // namespace

int main() {
  test_cube(BoundaryPolicy::kFixed);
  test_cube(BoundaryPolicy::kConforming);
  test_thin_channel_prism(BoundaryPolicy::kConforming);
  test_thin_channel_prism(BoundaryPolicy::kFixed);
  test_two_materials();
  test_material_triple_junction();
  test_schonhardt();
  test_reflex_l_prism();
  test_categorical_interface(BoundaryPolicy::kConforming);
  test_categorical_interface(BoundaryPolicy::kFixed);
  test_failures();
  test_cavity_limit_refusal();
  test_getters_reject_null_handles();
  test_diagnostic_getter_outputs();
  test_export_checks_all_buffers_before_copying();
  test_witness_output_contract();
  test_source_export_allows_empty_banks();
  test_planar_recovery_debits_global_work_once();
  test_protection_index_matches_actual_exhaustive_kernels();
  return phx::mc::test::finish("test_plc3d");
}
