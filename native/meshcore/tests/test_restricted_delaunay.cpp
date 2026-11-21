//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Outward bounds of exact tetrahedron circumcenters, and restricted Delaunay
// meshing of implicit solids checked against their analytic distances.
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {
struct CenterBounds {
  int32_t status = -1;
  std::array<double, 6> bounds{};
  std::array<int32_t, 1> item_status{};

  bool operator==(const CenterBounds&) const = default;
};

CenterBounds center_bounds(const std::array<double, 12>& points) {
  constexpr int32_t tets[] = {0, 1, 2, 3};
  CenterBounds result;
  result.status = phx_mc_restricted_centers_3d(4, points.data(), 1, tets, result.bounds.data(),
                                               result.item_status.data());
  return result;
}

// The unit corner tetrahedron has circumcenter (1/2, 1/2, 1/2); the symmetric
// tetrahedron is centered exactly at the origin. Both give their
// round-to-nearest results in every caller mode; a subnormal coordinate lies
// outside the exact coordinate domain.
void circumcenter_bounds() {
  constexpr std::array<double, 12> corner{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr std::array<double, 12> symmetric{1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1};
  constexpr std::array<double, 12> subnormal{0, 0, 0, 1, 0, 0, 0, 1, 0, 0x1p-1060, 0, 1};
  const CenterBounds corner_reference = center_bounds(corner);
  PHX_CHECK(corner_reference.status == PHX_MC_OK);
  PHX_CHECK(corner_reference.item_status[0] == PHX_MC_OK);
  const CenterBounds symmetric_reference = center_bounds(symmetric);
  PHX_CHECK(symmetric_reference.status == PHX_MC_OK);
  PHX_CHECK(symmetric_reference.item_status[0] == PHX_MC_OK);
  for (int axis = 0; axis < 3; ++axis) {
    PHX_CHECK(corner_reference.bounds[axis] <= 0.5);
    PHX_CHECK(corner_reference.bounds[axis + 3] >= 0.5);
    PHX_CHECK(symmetric_reference.bounds[axis] <= 0.0);
    PHX_CHECK(symmetric_reference.bounds[axis + 3] >= 0.0);
  }
  PHX_CHECK(center_bounds(subnormal).status == PHX_MC_RANGE_ERROR);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    PHX_CHECK(center_bounds(corner) == corner_reference);
    PHX_CHECK(center_bounds(symmetric) == symmetric_reference);
    PHX_CHECK(center_bounds(subnormal).status == PHX_MC_RANGE_ERROR);
  });
}

// A point program of f: the inputs x, y, z are workspace entries 0 to 2;
// seeds is the workspace template (lower and upper alike) and parameters
// the reduce-sum rounding factors (n eps for n terms, as the lowering emits).
struct FieldProgram {
  std::vector<int64_t> rows;
  std::vector<double> seeds;
  std::vector<double> parameters;
  int64_t output = 0;
  std::vector<int64_t> guards{};  // entries whose lower bound must be >= 0
};

constexpr int64_t kAdd = 2;
constexpr int64_t kSubtract = 3;
constexpr int64_t kReduceSum = 7;
constexpr int64_t kSqrt = 10;
constexpr int64_t kSquare = 12;

// |p| - radius: entries 3 radius, 4-6 squares, 7 their sum, 8 its root, 9 f.
FieldProgram sphere_program(double radius) {
  return {{kSquare, 4, 3, 0, -1, -1, -1, 0, kReduceSum, 7, 1, 4, -1, -1, 0, 3,
           kSqrt, 8, 1, 7, -1, -1, -1, 0, kSubtract, 9, 1, 8, -1, 3, -1, 0},
          {0.0, 0.0, 0.0, radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {3.0 * 0x1p-52},
          9};
}

// The ring torus about z, sqrt((sqrt(x^2 + y^2) - major)^2 + z^2) - minor:
// entries 3 major, 4 minor, 5-6 x^2 and y^2, 7 their sum, 8 its root, 9 the
// radial offset, 10 its square, 11 z^2, 12 the sum, 13 its root, 14 f.
FieldProgram torus_program(double major, double minor) {
  return {{kSquare, 5, 2, 0, -1, -1, -1, 0, kReduceSum, 7, 1, 5, -1, -1, 0, 2,
           kSqrt, 8, 1, 7, -1, -1, -1, 0, kSubtract, 9, 1, 8, -1, 3, -1, 0,
           kSquare, 10, 1, 9, -1, -1, -1, 0, kSquare, 11, 1, 2, -1, -1, -1, 0,
           kAdd, 12, 1, 10, -1, 11, -1, 0, kSqrt, 13, 1, 12, -1, -1, -1, 0,
           kSubtract, 14, 1, 13, -1, 4, -1, 0},
          {0.0, 0.0, 0.0, major, minor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {2.0 * 0x1p-52},
          14};
}

struct Solid {
  FieldProgram program;
  std::array<double, 3> lower, upper;
  int64_t components, euler;
  double (*distance)(const double*);
};

double sphere_distance(const double* p) { return std::hypot(p[0], p[1], p[2]) - 0.371; }
double torus_distance(const double* p) {
  return std::hypot(std::hypot(p[0], p[1]) - 0.742, p[2]) - 0.371;
}

struct Volume {
  int32_t status = -1;
  std::vector<double> points;
  std::vector<int32_t> cells, faces;
  std::array<int64_t, PHX_MC_RESTRICTED_REPORT_SIZE> report{};

  bool operator==(const Volume&) const = default;
};

// The ledger criteria at tolerance `tolerance` of a solid of reach 0.371:
// the facet edge of a certified deviation 0.95 * tolerance, Boissonnat and
// Oudot's 30 degrees, no cell size, radius-edge 1.98, 15.5 degrees.
std::array<double, PHX_MC_RESTRICTED_CRITERIA_SIZE> ledger_criteria(double tolerance) {
  const double aim = 0.95 * tolerance;
  const double edge = std::sqrt(9.0 * aim * aim + 6.0 * aim * 0.371) - 3.0 * aim;
  return {edge, tolerance, 30.0, std::numeric_limits<double>::infinity(), 1.98, 15.5, 0x1p-21, 0.8};
}

Volume restricted_volume(const Solid& solid, const std::array<double, PHX_MC_RESTRICTED_CRITERIA_SIZE>& criteria) {
  const int64_t topology[2] = {solid.components, solid.euler};
  const int64_t no_index = 0;
  const FieldProgram& program = solid.program;
  phx_mc_restricted_volume* handle = nullptr;
  Volume result;
  result.status = phx_mc_implicit_restricted_delaunay(
      program.rows.data(), static_cast<int64_t>(program.rows.size() / 8), &no_index, 0,
      program.parameters.data(), static_cast<int64_t>(program.parameters.size()), program.seeds.data(),
      program.seeds.data(), static_cast<int64_t>(program.seeds.size()), program.output,
      program.guards.data(), static_cast<int64_t>(program.guards.size()), solid.lower.data(),
      solid.upper.data(), criteria.data(), topology, 0, nullptr, &handle);
  if (handle == nullptr) return result;
  int64_t sizes[3] = {};
  PHX_CHECK(phx_mc_restricted_volume_sizes(handle, sizes, result.report.data()) == PHX_MC_OK);
  result.points.resize(static_cast<std::size_t>(3 * sizes[0]));
  result.cells.resize(static_cast<std::size_t>(4 * sizes[1]));
  result.faces.resize(static_cast<std::size_t>(3 * sizes[2]));
  PHX_CHECK(phx_mc_restricted_volume_copy(handle, result.points.data(), result.cells.data(),
                                          result.faces.data()) == PHX_MC_OK);
  phx_mc_restricted_volume_free(handle);
  return result;
}

int32_t find_root(std::vector<int32_t>& parent, int32_t x) {
  while (parent[static_cast<std::size_t>(x)] != x) x = parent[static_cast<std::size_t>(x)];
  return x;
}

// The boundary is the set of unshared outward cell faces, and a closed
// orientable surface (every directed edge once, its reverse once) with the
// solid's component count and Euler characteristic.
void check_boundary(const Volume& volume, const Solid& solid) {
  constexpr int kOutward[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  std::map<std::array<int32_t, 3>, std::vector<std::array<int32_t, 3>>> cell_faces;
  for (std::size_t c = 0; c < volume.cells.size() / 4; ++c)
    for (const auto& face : kOutward) {
      std::array<int32_t, 3> oriented{};
      for (std::size_t r = 0; r < 3; ++r) oriented[r] = volume.cells[4 * c + static_cast<std::size_t>(face[r])];
      std::array<int32_t, 3> key = oriented;
      std::sort(key.begin(), key.end());
      cell_faces[key].push_back(oriented);
    }
  std::map<std::array<int32_t, 3>, std::array<int32_t, 3>> unshared;
  for (const auto& [key, faces] : cell_faces) {
    PHX_CHECK(faces.size() <= 2);
    if (faces.size() == 1) unshared[key] = faces[0];
  }
  const std::size_t face_count = volume.faces.size() / 3;
  PHX_CHECK(face_count == unshared.size());
  std::map<std::array<int32_t, 2>, int32_t> directed;
  for (std::size_t f = 0; f < face_count; ++f) {
    const int32_t* v = volume.faces.data() + 3 * f;
    std::array<int32_t, 3> key{v[0], v[1], v[2]};
    std::sort(key.begin(), key.end());
    const auto found = unshared.find(key);
    PHX_CHECK(found != unshared.end());
    // The same cyclic orientation as the cell's outward face.
    if (found != unshared.end()) {
      const auto& o = found->second;
      PHX_CHECK((o[0] == v[0] && o[1] == v[1]) || (o[0] == v[1] && o[1] == v[2]) ||
                (o[0] == v[2] && o[1] == v[0]));
    }
    for (int r = 0; r < 3; ++r) {
      const bool fresh = directed.insert({{v[r], v[(r + 1) % 3]}, static_cast<int32_t>(f)}).second;
      PHX_CHECK(fresh);
    }
  }
  std::vector<int32_t> parent(face_count);
  std::iota(parent.begin(), parent.end(), 0);
  for (const auto& [edge, face] : directed) {
    const auto reverse = directed.find({edge[1], edge[0]});
    PHX_CHECK(reverse != directed.end());
    if (reverse == directed.end()) continue;
    const int32_t a = find_root(parent, face), b = find_root(parent, reverse->second);
    parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
  }
  int64_t components = 0;
  for (std::size_t f = 0; f < face_count; ++f) components += find_root(parent, static_cast<int32_t>(f)) == static_cast<int32_t>(f);
  std::vector<int32_t> vertices(volume.faces);
  std::sort(vertices.begin(), vertices.end());
  vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
  const auto euler = static_cast<int64_t>(vertices.size()) - static_cast<int64_t>(directed.size() / 2) +
                     static_cast<int64_t>(face_count);
  PHX_CHECK(components == solid.components);
  PHX_CHECK(euler == solid.euler);
}

double triangle_distance(const double* p, const double* a, const double* b, const double* c);

// Every cell positive with its smallest dihedral angle at least `dihedral`
// degrees; the boundary within `tolerance` of the zero set both ways: the
// analytic distance at sample points of every boundary triangle, and the
// distance from analytic surface samples to the nearest boundary triangle.
void check_shape_and_distance(const Volume& volume, const Solid& solid, double tolerance,
                              double dihedral, bool torus) {
  const auto at = [&](int32_t v) { return volume.points.data() + 3 * static_cast<std::size_t>(v); };
  double smallest = 180.0;
  for (std::size_t c = 0; c < volume.cells.size() / 4; ++c) {
    const int32_t* v = volume.cells.data() + 4 * c;
    PHX_CHECK(phx::mc::orient3d(at(v[0]), at(v[1]), at(v[2]), at(v[3])) > 0);
    constexpr int kOutward[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    std::array<std::array<double, 3>, 4> normals{};
    for (std::size_t f = 0; f < 4; ++f) {
      const double* p0 = at(v[kOutward[f][0]]);
      const double* p1 = at(v[kOutward[f][1]]);
      const double* p2 = at(v[kOutward[f][2]]);
      const double u[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
      const double w[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      normals[f] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
      const double length = std::hypot(normals[f][0], normals[f][1], normals[f][2]);
      for (double& x : normals[f]) x /= length;
    }
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j) {
        const double cosine = -(normals[i][0] * normals[j][0] + normals[i][1] * normals[j][1] +
                                normals[i][2] * normals[j][2]);
        smallest = std::min(smallest, std::acos(std::clamp(cosine, -1.0, 1.0)) * (180.0 / std::numbers::pi));
      }
  }
  PHX_CHECK(smallest >= dihedral);
  const std::size_t face_count = volume.faces.size() / 3;
  double farthest = 0.0;
  for (std::size_t f = 0; f < face_count; ++f) {
    const double* p[3] = {at(volume.faces[3 * f]), at(volume.faces[3 * f + 1]), at(volume.faces[3 * f + 2])};
    for (int i = 0; i <= 8; ++i)
      for (int j = 0; i + j <= 8; ++j) {
        const double s = i / 8.0, t = j / 8.0;
        double q[3];
        for (int d = 0; d < 3; ++d) q[d] = p[0][d] + s * (p[1][d] - p[0][d]) + t * (p[2][d] - p[0][d]);
        farthest = std::max(farthest, std::abs(solid.distance(q)));
      }
  }
  PHX_CHECK(farthest <= tolerance);
  // The surface samples: a 60 x 60 parameter grid.
  double uncovered = 0.0;
  for (int i = 0; i < 60; ++i)
    for (int j = 0; j < 60; ++j) {
      const double u = 2.0 * std::numbers::pi * (i + 0.5) / 60.0, w = std::numbers::pi * (j + 0.5) / 60.0;
      double q[3];
      if (torus) {
        const double ring = 0.742 + 0.371 * std::cos(2.0 * w);
        q[0] = ring * std::cos(u);
        q[1] = ring * std::sin(u);
        q[2] = 0.371 * std::sin(2.0 * w);
      } else {
        q[0] = 0.371 * std::sin(w) * std::cos(u);
        q[1] = 0.371 * std::sin(w) * std::sin(u);
        q[2] = 0.371 * std::cos(w);
      }
      double nearest = std::numeric_limits<double>::infinity();
      for (std::size_t f = 0; f < face_count; ++f)
        nearest = std::min(nearest, triangle_distance(q, at(volume.faces[3 * f]), at(volume.faces[3 * f + 1]),
                                                      at(volume.faces[3 * f + 2])));
      uncovered = std::max(uncovered, nearest);
    }
  PHX_CHECK(uncovered <= tolerance);
}

// Distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
double triangle_distance(const double* p, const double* a, const double* b, const double* c) {
  const auto sub = [](const double* x, const double* y) {
    return std::array<double, 3>{x[0] - y[0], x[1] - y[1], x[2] - y[2]};
  };
  const auto dot = [](const std::array<double, 3>& x, const std::array<double, 3>& y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  };
  const auto ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  std::array<double, 3> closest{};
  const auto point = [&](double s, double t) {
    for (std::size_t d = 0; d < 3; ++d) closest[d] = a[d] + s * ab[d] + t * ac[d];
  };
  const auto bp = sub(p, b), cp = sub(p, c);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp), d5 = dot(ab, cp), d6 = dot(ac, cp);
  const double va = d3 * d6 - d5 * d4, vb = d5 * d2 - d1 * d6, vc = d1 * d4 - d3 * d2;
  if (d1 <= 0.0 && d2 <= 0.0) point(0.0, 0.0);
  else if (d3 >= 0.0 && d4 <= d3) point(1.0, 0.0);
  else if (d6 >= 0.0 && d5 <= d6) point(0.0, 1.0);
  else if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) point(d1 / (d1 - d3), 0.0);
  else if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) point(0.0, d2 / (d2 - d6));
  else if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    point(1.0 - t, t);
  } else {
    const double denominator = 1.0 / (va + vb + vc);
    point(vb * denominator, vc * denominator);
  }
  const std::array<double, 3> gap{p[0] - closest[0], p[1] - closest[1], p[2] - closest[2]};
  return std::sqrt(dot(gap, gap));
}

// The ledger sphere (r = 0.371, tolerance 0.01) and ring torus (0.742 and
// 0.371, tolerance 0.02): closed manifold boundaries of the right topology
// within the tolerance of the analytic surface both ways, cells of at least
// 15 degrees, a clean report, and bitwise equal meshes from two runs.
void restricted_meshing() {
  const Solid sphere{sphere_program(0.371), {-0.371, -0.371, -0.371}, {0.371, 0.371, 0.371}, 1, 2,
                     sphere_distance};
  const Solid torus{torus_program(0.742, 0.371), {-1.113, -1.113, -0.371}, {1.113, 1.113, 0.371}, 1, 0,
                    torus_distance};
  for (const auto& [solid, tolerance, is_torus] :
       {std::tuple{&sphere, 0.01, false}, std::tuple{&torus, 0.02, true}}) {
    const auto criteria = ledger_criteria(tolerance);
    const Volume volume = restricted_volume(*solid, criteria);
    PHX_CHECK(volume.status == PHX_MC_OK);
    if (volume.status != PHX_MC_OK) continue;
    PHX_CHECK(volume.report[2] == 1);  // topology matched
    PHX_CHECK(volume.report[3] == 0);  // every box root covered
    PHX_CHECK(volume.report[4] == 0);  // no refused facet
    PHX_CHECK(volume.report[5] == 0);  // no unmet cell
    check_boundary(volume, *solid);
    check_shape_and_distance(volume, *solid, tolerance, 15.0, is_torus);
    PHX_CHECK(restricted_volume(*solid, criteria) == volume);
  }
}

// Arguments are validated before any work: a facet angle above 30 degrees
// (no termination guarantee) or a missing output is refused.
void restricted_meshing_arguments() {
  const Solid sphere{sphere_program(0.371), {-0.371, -0.371, -0.371}, {0.371, 0.371, 0.371}, 1, 2,
                     sphere_distance};
  auto criteria = ledger_criteria(0.01);
  criteria[2] = 31.0;
  PHX_CHECK(restricted_volume(sphere, criteria).status == PHX_MC_INVALID_ARGUMENT);
  const int64_t topology[2] = {1, 2};
  const int64_t no_index = 0;
  const auto valid = ledger_criteria(0.01);
  PHX_CHECK(phx_mc_implicit_restricted_delaunay(
                sphere.program.rows.data(), 4, &no_index, 0, sphere.program.parameters.data(), 1,
                sphere.program.seeds.data(), sphere.program.seeds.data(), 10, 9, nullptr, 0,
                sphere.lower.data(), sphere.upper.data(), valid.data(), topology, 0, nullptr,
                nullptr) == PHX_MC_INVALID_ARGUMENT);
  // A guard outside the workspace is refused.
  const int64_t outside = 10;
  phx_mc_restricted_volume* handle = nullptr;
  PHX_CHECK(phx_mc_implicit_restricted_delaunay(
                sphere.program.rows.data(), 4, &no_index, 0, sphere.program.parameters.data(), 1,
                sphere.program.seeds.data(), sphere.program.seeds.data(), 10, 9, &outside, 1,
                sphere.lower.data(), sphere.upper.data(), valid.data(), topology, 0, nullptr,
                &handle) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(handle == nullptr);
}

// A real-power field with zeros on two spheres: q = (|p|^2 - 0.01)^0.5 and
// f = (q - 0.2)(q - 0.4), the product of its two radial factors, zero at
// |p|^2 = 0.05 and 0.17. f is undefined inside |p| = 0.1, where the power's
// interval is the placeholder [0, 0] with its guard entry failed (f there
// [0.08, 0.08], no zero): only its guard keeps the coarse boxes holding both
// that ball and the inner sphere. Passed the guard, the seed cover finds both
// spheres and the restricted surface has their two components.
void guarded_real_power_seeds_both_spheres() {
  constexpr int64_t kMultiply = 5, kPower = 27;
  // 0-2 x, 3 0.01, 4 0.5, 5 0.2, 6 0.4, 7-9 squares, 10 |p|^2, 11 the base,
  // 12 q and 13 its guard, 14-15 the factors, 16 f.
  FieldProgram program{
      {kSquare, 7, 3, 0, -1, -1, -1, 0, kReduceSum, 10, 1, 7, -1, -1, 0, 3,
       kSubtract, 11, 1, 10, -1, 3, -1, 0, kPower, 12, 1, 11, -1, 4, -1, 0,
       kSubtract, 14, 1, 12, -1, 5, -1, 0, kSubtract, 15, 1, 12, -1, 6, -1, 0,
       kMultiply, 16, 1, 14, -1, 15, -1, 0},
      {0.0, 0.0, 0.0, 0.01, 0.5, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {3.0 * 0x1p-52},
      16,
      {13}};
  const Solid shell{program, {-0.42, -0.42, -0.42}, {0.42, 0.42, 0.42}, 2, 4,
                    [](const double* p) {
                      const double r = std::hypot(p[0], p[1], p[2]);
                      return std::max(std::sqrt(0.05) - r, r - std::sqrt(0.17));
                    }};
  const Volume volume = restricted_volume(shell, ledger_criteria(0.01));
  PHX_CHECK(volume.status == PHX_MC_OK);
  if (volume.status != PHX_MC_OK) return;
  PHX_CHECK(volume.report[2] == 1);  // topology matched
  PHX_CHECK(volume.report[3] == 0);  // every box root covered
  check_boundary(volume, shell);
  // Without the guard the seed cover drops those boxes: the surface misses a
  // sphere and its topology does not match.
  Solid unguarded = shell;
  unguarded.program.guards.clear();
  const Volume dropped = restricted_volume(unguarded, ledger_criteria(0.01));
  PHX_CHECK(dropped.status != PHX_MC_OK || dropped.report[2] != 1);
}
}  // namespace

int main() {
  circumcenter_bounds();
  restricted_meshing();
  restricted_meshing_arguments();
  guarded_real_power_seeds_both_spheres();
  return phx::mc::test::finish("test_restricted_delaunay");
}
