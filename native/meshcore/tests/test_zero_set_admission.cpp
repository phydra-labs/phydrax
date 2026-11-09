// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// kZeroSetBand admission soundness (sphere and torus fields against their exact
// distance functions), interval-Newton projection, and the zero-set remesh C entry.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <csignal>
#include <sys/resource.h>
#endif

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "scratch_faces.hpp"
#include "zero_set_admission.hpp"

namespace {
using phx::mc::remesh::IntervalBoxProgram;
using phx::mc::remesh::Point;
using phx::mc::remesh::VertexEvidence;
using phx::mc::remesh::ZeroSetBandAdmission;

constexpr int64_t kAdd = 2;
constexpr int64_t kSubtract = 3;
constexpr int64_t kMultiply = 5;
constexpr int64_t kReduceSum = 7;
constexpr int64_t kSqrt = 10;
constexpr int64_t kReciprocal = 11;
constexpr int64_t kSquare = 12;
constexpr int64_t kCompare = 13;
constexpr int64_t kSelect = 14;
constexpr int64_t kMaximum = 15;
constexpr double kGamma = 1.0e-15;  // >= 3u / (1 - 3u) for the short reductions
constexpr double kPi = 3.141592653589793;

// A hand-lowered program in the native layout: box at [0, 3), seeds after it.
struct Program {
  std::vector<int64_t> rows;
  std::vector<int64_t> indices;
  std::vector<double> parameters{kGamma};
  std::vector<double> lower, upper;
  std::vector<int64_t> outputs;

  void row(int64_t opcode, int64_t out, int64_t size, int64_t a, int64_t a_index, int64_t b,
           int64_t b_index, int64_t aux) {
    rows.insert(rows.end(), {opcode, out, size, a, a_index, b, b_index, aux});
  }
  phx_mc_interval_box_program view() const {
    return {rows.data(),       static_cast<int64_t>(rows.size() / 8),
            indices.data(),    static_cast<int64_t>(indices.size()),
            parameters.data(), static_cast<int64_t>(parameters.size()),
            lower.data(),      upper.data(),
            static_cast<int64_t>(lower.size()),
            outputs.data(),    static_cast<int64_t>(outputs.size()),
            nullptr,           0};
  }
  bool bind(IntervalBoxProgram& program) const {
    const auto v = view();
    return program.bind(v.instructions, v.instruction_count, v.indices, v.index_count,
                        v.parameters, v.parameter_count, v.workspace_lower, v.workspace_upper,
                        v.workspace_size, v.outputs, v.output_count, v.guards,
                        v.guard_count) == PHX_MC_OK;
  }
};

Program seeded(std::initializer_list<double> seeds, int64_t size) {
  Program program;
  program.lower.assign(static_cast<size_t>(size), 0.0);
  size_t at = 3;
  for (double seed : seeds) program.lower[at++] = seed;
  program.upper = program.lower;
  return program;
}

// f = x^2 + y^2 + z^2 - 1 and grad f = 2 (x, y, z).
Program sphere_value() {
  Program p = seeded({1.0}, 9);
  p.row(kSquare, 4, 3, 0, -1, -1, -1, 0);
  p.row(kReduceSum, 7, 1, 4, -1, -1, 0, 3);
  p.row(kSubtract, 8, 1, 7, -1, 3, -1, 0);
  p.outputs = {8};
  return p;
}
Program sphere_gradient() {
  Program p = seeded({2.0}, 7);
  p.indices = {0, 0, 0};
  p.row(kMultiply, 4, 3, 0, -1, 3, 0, 0);
  p.outputs = {4, 5, 6};
  return p;
}

// f = (s - R)^2 + z^2 - r^2 with s = |(x, y)|, R = 1, r = 0.4;
// grad f = (2 (s - R) x / s, 2 (s - R) y / s, 2 z).
constexpr double kMajor = 1.0, kMinor = 0.4;
Program torus_value() {
  Program p = seeded({kMajor, kMinor * kMinor}, 14);
  p.row(kSquare, 5, 2, 0, -1, -1, -1, 0);
  p.row(kReduceSum, 7, 1, 5, -1, -1, 0, 2);
  p.row(kSqrt, 8, 1, 7, -1, -1, -1, 0);
  p.row(kSubtract, 9, 1, 8, -1, 3, -1, 0);
  p.row(kSquare, 10, 1, 9, -1, -1, -1, 0);
  p.row(kSquare, 11, 1, 2, -1, -1, -1, 0);
  p.row(kAdd, 12, 1, 10, -1, 11, -1, 0);
  p.row(kSubtract, 13, 1, 12, -1, 4, -1, 0);
  p.outputs = {13};
  return p;
}
Program torus_gradient() {
  Program p = seeded({kMajor, 2.0}, 16);
  p.indices = {0, 0};
  p.row(kSquare, 5, 2, 0, -1, -1, -1, 0);
  p.row(kReduceSum, 7, 1, 5, -1, -1, 0, 2);
  p.row(kSqrt, 8, 1, 7, -1, -1, -1, 0);
  p.row(kSubtract, 9, 1, 8, -1, 3, -1, 0);
  p.row(kReciprocal, 10, 1, 8, -1, -1, -1, 0);
  p.row(kMultiply, 11, 1, 9, -1, 10, -1, 0);
  p.row(kMultiply, 12, 1, 11, -1, 4, -1, 0);
  p.row(kMultiply, 13, 2, 0, -1, 12, 0, 0);
  p.row(kMultiply, 15, 1, 2, -1, 4, -1, 0);
  p.outputs = {13, 14, 15};
  return p;
}

double sphere_distance(const Point& p) {
  return std::fabs(std::hypot(p[0], p[1], p[2]) - 1.0);
}
double torus_distance(const Point& p) {
  return std::fabs(std::hypot(std::hypot(p[0], p[1]) - kMajor, p[2]) - kMinor);
}
Point sphere_point(double theta, double phi) {
  return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
}
Point torus_point(double u, double v) {
  return {(kMajor + kMinor * std::cos(v)) * std::cos(u),
          (kMajor + kMinor * std::cos(v)) * std::sin(u), kMinor * std::sin(v)};
}
Point outward_torus(double u, double v) {
  return {std::cos(v) * std::cos(u), std::cos(v) * std::sin(u), std::sin(v)};
}

Point combine(const Point& a, const Point& b, const Point& c, double s, double t) {
  Point x{};
  for (int d = 0; d < 3; ++d) x[d] = (1 - s - t) * a[d] + s * b[d] + t * c[d];
  return x;
}

std::array<VertexEvidence, 3> evidence_of(const ZeroSetBandAdmission& admission, const Point& a,
                                          const Point& b, const Point& c) {
  return {admission.evaluate(a), admission.evaluate(b), admission.evaluate(c)};
}

// Random small triangles near a surface: every admitted triangle lies within
// the tolerance of the exact zero set at dense samples, and within its own
// certified distance bound; refusals must also occur (the test is not vacuous).
template <class Sample, class Normal, class Distance>
void admission_soundness(const IntervalBoxProgram& value, const IntervalBoxProgram& gradient,
                         Sample sample, Normal normal, Distance distance, unsigned seed) {
  constexpr double tolerance = 0.03;
  const ZeroSetBandAdmission admission(value, gradient, tolerance, 1);
  std::mt19937_64 random(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int admitted = 0, refused = 0, reversed_admitted = 0;
  for (int trial = 0; trial < 1500; ++trial) {
    const double u = 2 * kPi * unit(random), v = 2 * kPi * unit(random);
    const Point base = sample(u, v);
    const Point n = normal(u, v);
    const double lift = 0.08 * (2 * unit(random) - 1);
    const double size = 0.01 + 0.15 * unit(random);
    // Tangent frame at the base point.
    Point e1 = std::fabs(n[2]) < 0.9 ? Point{-n[1], n[0], 0} : Point{0, -n[2], n[1]};
    const double e1n = std::hypot(e1[0], e1[1], e1[2]);
    for (double& c : e1) c /= e1n;
    const Point e2{n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2],
                   n[0] * e1[1] - n[1] * e1[0]};
    std::array<Point, 3> corners{};
    for (int k = 0; k < 3; ++k) {
      const double angle = 2 * kPi * k / 3 + 0.6 * unit(random);
      const double tilt = 0.3 * (2 * unit(random) - 1);
      for (int d = 0; d < 3; ++d)
        corners[k][d] = base[d] + lift * n[d] +
                        size * (std::cos(angle) * e1[d] + std::sin(angle) * e2[d] + tilt * n[d]);
    }
    const bool flip = unit(random) < 0.2;
    if (flip) std::swap(corners[1], corners[2]);
    const auto& [a, b, c] = corners;
    const auto evidence = evidence_of(admission, a, b, c);
    if (!admission.admits(a, b, c, evidence)) {
      ++refused;
      continue;
    }
    ++admitted;
    if (flip) ++reversed_admitted;
    const double bound = admission.distance_bound(a, b, c, evidence);
    PHX_CHECK(bound <= tolerance);
    for (int i = 0; i <= 12; ++i)
      for (int j = 0; i + j <= 12; ++j) {
        const double sampled = distance(combine(a, b, c, i / 12.0, j / 12.0));
        PHX_CHECK(sampled <= bound * (1 + 1e-9) + 1e-14);
      }
  }
  PHX_CHECK(admitted > 100);
  PHX_CHECK(refused > 100);
  PHX_CHECK(reversed_admitted == 0);  // orientation-reversed triangles are never admitted
}

void sphere_cases() {
  IntervalBoxProgram value, gradient;
  PHX_CHECK(sphere_value().bind(value));
  PHX_CHECK(sphere_gradient().bind(gradient));
  const ZeroSetBandAdmission admission(value, gradient, 0.02, 1);
  // A small outward-facing triangle on the sphere near the north pole.
  const Point a = sphere_point(0.05, 0.0), b = sphere_point(0.05, 2.1),
              c = sphere_point(0.05, 4.2);
  PHX_CHECK(admission.admits(a, b, c, evidence_of(admission, a, b, c)));
  PHX_CHECK(admission.transversal_sign(a, b, c) == 1);
  // Reversed orientation and a parallel copy lifted outside the band refuse.
  PHX_CHECK(!admission.admits(a, c, b, evidence_of(admission, a, c, b)));
  const Point la{a[0], a[1], a[2] + 0.3}, lb{b[0], b[1], b[2] + 0.3}, lc{c[0], c[1], c[2] + 0.3};
  PHX_CHECK(!admission.admits(la, lb, lc, evidence_of(admission, la, lb, lc)));
  // A triangle through the centre has no certified gradient lower bound.
  const Point o1{-0.1, -0.1, 0.0}, o2{0.1, -0.1, 0.0}, o3{0.0, 0.1, 0.0};
  PHX_CHECK(admission.distance_bound(o1, o2, o3, evidence_of(admission, o1, o2, o3)) ==
            std::numeric_limits<double>::infinity());
  // Interval Newton lands on the sphere from either side.
  for (const double radius : {0.8, 1.15}) {
    const Point start{radius * 0.48, radius * 0.6, radius * 0.64};
    const Point projected = admission.project(start);
    PHX_CHECK(sphere_distance(projected) <= 1e-12);
  }
  admission_soundness(
      value, gradient, [](double u, double v) { return sphere_point(0.5 * v, u); },
      [](double u, double v) { return sphere_point(0.5 * v, u); }, sphere_distance, 7u);
}

void torus_cases() {
  IntervalBoxProgram value, gradient;
  PHX_CHECK(torus_value().bind(value));
  PHX_CHECK(torus_gradient().bind(gradient));
  const ZeroSetBandAdmission admission(value, gradient, 0.02, 1);
  for (const double v : {0.0, 1.5, kPi, 4.0}) {
    const Point start = torus_point(0.7, v);
    const Point n = outward_torus(0.7, v);
    const Point off{start[0] + 0.07 * n[0], start[1] + 0.07 * n[1], start[2] + 0.07 * n[2]};
    PHX_CHECK(torus_distance(admission.project(off)) <= 1e-12);
  }
  admission_soundness(value, gradient, torus_point, outward_torus, torus_distance, 11u);
}

void near_band_distance_cases() {
  IntervalBoxProgram value, gradient;
  PHX_CHECK(sphere_value().bind(value));
  PHX_CHECK(sphere_gradient().bind(gradient));
  constexpr double tolerance = 0.02;
  const ZeroSetBandAdmission admission(value, gradient, tolerance, 1);
  const Point a{0.0, 0.0, 1.0199}, b{0.001, 0.0, 1.0199}, c{0.0, 0.001, 1.0199};
  // The entire triangle has 1 < z <= |x| <= max(vertex norms) < 1+tolerance.
  // The tolerance-wide gradient-flow ratio nevertheless exceeds tolerance,
  // even at a single vertex: subdivision alone cannot close that gap.
  const double independent_upper = std::max({sphere_distance(a), sphere_distance(b),
                                             sphere_distance(c)});
  PHX_CHECK(independent_upper < tolerance);
  const double bound = admission.distance_bound(a, b, c, evidence_of(admission, a, b, c));
  PHX_CHECK(independent_upper <= bound && bound <= tolerance);
  PHX_CHECK(admission.admits(a, b, c, evidence_of(admission, a, b, c)));
  PHX_CHECK(!admission.admits(a, c, b, evidence_of(admission, a, c, b)));
  const double outside_z =
      std::nextafter(1.0 + tolerance, std::numeric_limits<double>::infinity());
  const Point outside_a{0.0, 0.0, outside_z}, outside_b{0.001, 0.0, outside_z},
              outside_c{0.0, 0.001, outside_z};
  PHX_CHECK(!admission.admits(outside_a, outside_b, outside_c,
                             evidence_of(admission, outside_a, outside_b, outside_c)));
}

void whole_piece_bracketing_cases() {
  // f=z-x*y*(1-x-y): all three corners lie on Z, but the triangle's center
  // has no zero in its tolerance ball. Corner-only brackets are insufficient.
  Program field = seeded({1.0}, 9);
  field.row(kAdd, 4, 1, 0, -1, 1, -1, 0);
  field.row(kSubtract, 5, 1, 3, -1, 4, -1, 0);
  field.row(kMultiply, 6, 1, 0, -1, 1, -1, 0);
  field.row(kMultiply, 7, 1, 6, -1, 5, -1, 0);
  field.row(kSubtract, 8, 1, 2, -1, 7, -1, 0);
  field.outputs = {8};
  Program derivative = seeded({1.0, 2.0, -1.0}, 16);
  derivative.row(kMultiply, 6, 1, 0, -1, 4, -1, 0);
  derivative.row(kSubtract, 7, 1, 3, -1, 6, -1, 0);
  derivative.row(kSubtract, 8, 1, 7, -1, 1, -1, 0);
  derivative.row(kMultiply, 9, 1, 1, -1, 8, -1, 0);
  derivative.row(kMultiply, 10, 1, 9, -1, 5, -1, 0);
  derivative.row(kMultiply, 11, 1, 1, -1, 4, -1, 0);
  derivative.row(kSubtract, 12, 1, 3, -1, 0, -1, 0);
  derivative.row(kSubtract, 13, 1, 12, -1, 11, -1, 0);
  derivative.row(kMultiply, 14, 1, 0, -1, 13, -1, 0);
  derivative.row(kMultiply, 15, 1, 14, -1, 5, -1, 0);
  derivative.outputs = {10, 15, 3};
  IntervalBoxProgram value, gradient;
  PHX_CHECK(field.bind(value));
  PHX_CHECK(derivative.bind(gradient));
  constexpr double tolerance = 0.01;
  const ZeroSetBandAdmission admission(value, gradient, tolerance, 1);
  const Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{0.0, 1.0, 0.0};
  // In the center's enclosing radius-0.01 box, x,y>0.32 and
  // 1-x-y>0.31, hence f < 0.01-0.32*0.32*0.31 < 0 everywhere.
  PHX_CHECK(admission.transversal_sign(a, b, c) == 1);
  PHX_CHECK(!admission.admits(a, b, c, evidence_of(admission, a, b, c)));
}

struct Surface {
  std::vector<double> vertices;
  std::vector<int64_t> triangles;
};

// UV sphere: slivers at both poles (minimum angle ~ 4.5 degrees).
Surface uv_sphere(int rings, int segments) {
  Surface s;
  auto add = [&](const Point& p) {
    s.vertices.insert(s.vertices.end(), p.begin(), p.end());
    return static_cast<int64_t>(s.vertices.size() / 3 - 1);
  };
  const int64_t north = add({0, 0, 1});
  std::vector<std::vector<int64_t>> ring(static_cast<size_t>(rings - 1));
  for (int k = 1; k < rings; ++k)
    for (int j = 0; j < segments; ++j)
      ring[static_cast<size_t>(k - 1)].push_back(
          add(sphere_point(kPi * k / rings, 2 * kPi * j / segments)));
  const int64_t south = add({0, 0, -1});
  auto tri = [&](int64_t a, int64_t b, int64_t c) { s.triangles.insert(s.triangles.end(), {a, b, c}); };
  for (int j = 0; j < segments; ++j) {
    const size_t n = static_cast<size_t>((j + 1) % segments), m = static_cast<size_t>(j);
    tri(north, ring.front()[m], ring.front()[n]);
    for (size_t k = 0; k + 1 < ring.size(); ++k) {
      tri(ring[k][m], ring[k + 1][m], ring[k + 1][n]);
      tri(ring[k][m], ring[k + 1][n], ring[k][n]);
    }
    tri(south, ring.back()[n], ring.back()[m]);
  }
  return s;
}

double minimum_angle(const std::vector<double>& v, const std::vector<int64_t>& t) {
  double result = 180;
  for (size_t f = 0; f < t.size(); f += 3)
    for (int k = 0; k < 3; ++k) {
      const double* p = &v[3 * static_cast<size_t>(t[f + static_cast<size_t>(k)])];
      const double* q = &v[3 * static_cast<size_t>(t[f + static_cast<size_t>((k + 1) % 3)])];
      const double* r = &v[3 * static_cast<size_t>(t[f + static_cast<size_t>((k + 2) % 3)])];
      const double u[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
      const double w[3] = {r[0] - p[0], r[1] - p[1], r[2] - p[2]};
      const double cosine = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) /
                            (std::hypot(u[0], u[1], u[2]) * std::hypot(w[0], w[1], w[2]));
      result = std::min(result, std::acos(std::clamp(cosine, -1.0, 1.0)) * 180 / kPi);
    }
  return result;
}

double signed_volume(const std::vector<double>& v, const std::vector<int64_t>& t) {
  double volume = 0;
  for (size_t f = 0; f < t.size(); f += 3) {
    const double* a = &v[3 * static_cast<size_t>(t[f])];
    const double* b = &v[3 * static_cast<size_t>(t[f + 1])];
    const double* c = &v[3 * static_cast<size_t>(t[f + 2])];
    volume += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
              a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return volume / 6;
}

void remesh_sphere_carrier() {
  const Surface input = uv_sphere(8, 40);
  PHX_CHECK(signed_volume(input.vertices, input.triangles) > 0);
  const double before = minimum_angle(input.vertices, input.triangles);
  const Program value = sphere_value(), gradient = sphere_gradient();
  void* handle = nullptr;
  std::array<int64_t, 8> counters{};
  double distance = -1;
  constexpr double tolerance = 0.05;
  phx_mc_zero_set_remesh_request request{
      input.vertices.data(), static_cast<int64_t>(input.vertices.size() / 3),
      input.triangles.data(), static_cast<int64_t>(input.triangles.size() / 3),
      value.view(),          gradient.view(),
      tolerance,             tolerance,
      0.3,                   30.0,
      &handle,               counters.data(),
      &distance};
  PHX_CHECK(phx_mc_zero_set_remesh(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  PHX_CHECK(counters[5] == 1);
  PHX_CHECK(distance >= 0 && distance <= tolerance);
  std::array<int64_t, 2> sizes{};
  PHX_CHECK(phx_mc_zero_set_remesh_sizes(handle, sizes.data()) == PHX_MC_OK);
  std::vector<double> vertices(static_cast<size_t>(3 * sizes[0]));
  std::vector<int64_t> triangles(static_cast<size_t>(3 * sizes[1]));
  PHX_CHECK(phx_mc_zero_set_remesh_arrays(handle, sizes[0] - 1, sizes[1], vertices.data(),
                                          triangles.data()) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx_mc_zero_set_remesh_arrays(handle, sizes[0], sizes[1], vertices.data(),
                                          triangles.data()) == PHX_MC_OK);
  phx_mc_zero_set_remesh_free(handle);
  // The compact borrowed-input ABI must produce exactly the same edits,
  // certificates and ordering as the ordinary int64 entry.
  std::vector<int32_t> compact(input.triangles.begin(), input.triangles.end());
  const std::vector<int32_t> original_compact = compact;
  void* compact_handle = nullptr;
  std::array<int64_t, 8> compact_counters{};
  double compact_distance = -1;
  auto compact_request = request;
  compact_request.triangles = reinterpret_cast<const int64_t*>(compact.data());
  compact_request.result = &compact_handle;
  compact_request.counters = compact_counters.data();
  compact_request.distance = &compact_distance;
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&compact_request, nullptr, 0, nullptr, 0) == PHX_MC_OK);
  PHX_CHECK(compact_handle != nullptr);
  if (compact_handle != nullptr) {
    std::array<int64_t, 2> compact_sizes{};
    PHX_CHECK(phx_mc_zero_set_remesh_sizes(compact_handle, compact_sizes.data()) == PHX_MC_OK);
    PHX_CHECK(compact_sizes == sizes);
    std::vector<double> compact_vertices(static_cast<size_t>(3 * compact_sizes[0]));
    std::vector<int64_t> compact_triangles(static_cast<size_t>(3 * compact_sizes[1]));
    PHX_CHECK(phx_mc_zero_set_remesh_arrays(compact_handle, compact_sizes[0], compact_sizes[1],
                                            compact_vertices.data(), compact_triangles.data()) ==
              PHX_MC_OK);
    PHX_CHECK(compact_vertices == vertices);
    PHX_CHECK(compact_triangles == triangles);
    PHX_CHECK(compact_counters == counters);
    PHX_CHECK(compact_distance == distance);
    PHX_CHECK(compact == original_compact);
    phx_mc_zero_set_remesh_free(compact_handle);
  }
  // Admission-limited coarsening is opt-in: it keeps the certified band, may
  // only remove vertices relative to the ordinary passes, and the flag is a
  // strict boolean.
  void* coarse_handle = nullptr;
  std::array<int64_t, 8> coarse_counters{};
  double coarse_distance = -1;
  auto coarse_request = request;
  coarse_request.result = &coarse_handle;
  coarse_request.counters = coarse_counters.data();
  coarse_request.distance = &coarse_distance;
  coarse_request.admission_limited_coarsening = 1;
  PHX_CHECK(phx_mc_zero_set_remesh(&coarse_request) == PHX_MC_OK);
  PHX_CHECK(coarse_handle != nullptr);
  if (coarse_handle != nullptr) {
    std::array<int64_t, 2> coarse_sizes{};
    PHX_CHECK(phx_mc_zero_set_remesh_sizes(coarse_handle, coarse_sizes.data()) == PHX_MC_OK);
    PHX_CHECK(coarse_sizes[0] <= sizes[0]);
    PHX_CHECK(coarse_distance >= 0 && coarse_distance <= tolerance);
    PHX_CHECK(coarse_counters[5] == 1);
    phx_mc_zero_set_remesh_free(coarse_handle);
  }
  coarse_request.admission_limited_coarsening = 2;
  coarse_handle = nullptr;
  PHX_CHECK(phx_mc_zero_set_remesh(&coarse_request) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(coarse_handle == nullptr);
  PHX_CHECK(counters[1] > 0);
  compact[0] = -1;
  compact_handle = nullptr;
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&compact_request, nullptr, 0, nullptr, 0) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(compact_handle == nullptr);
  compact = original_compact;
  compact_request.triangles = reinterpret_cast<const int64_t*>(compact.data());
  std::swap(compact[1], compact[2]);
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&compact_request, nullptr, 0, nullptr, 0) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(compact_handle == nullptr);
  compact = original_compact;
  compact_request.triangles = reinterpret_cast<const int64_t*>(compact.data());
  auto nonfinite_vertices = input.vertices;
  nonfinite_vertices[0] = std::numeric_limits<double>::infinity();
  compact_request.vertices = nonfinite_vertices.data();
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&compact_request, nullptr, 0, nullptr, 0) == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(compact_handle == nullptr);
  // Closed oriented 2-manifold: every directed edge appears once, reversed once.
  std::map<std::pair<int64_t, int64_t>, int> directed;
  for (size_t f = 0; f < triangles.size(); f += 3)
    for (size_t k = 0; k < 3; ++k) ++directed[{triangles[f + k], triangles[f + (k + 1) % 3]}];
  bool manifold = true;
  for (const auto& [edge, count] : directed)
    manifold = manifold && count == 1 && directed.count({edge.second, edge.first}) == 1;
  PHX_CHECK(manifold);
  std::vector<double> edge_lengths;
  for (const auto& item : directed) {
    const auto& edge = item.first;
    if (edge.first >= edge.second) continue;
    const size_t a = static_cast<size_t>(edge.first) * 3;
    const size_t b = static_cast<size_t>(edge.second) * 3;
    edge_lengths.push_back(std::hypot(vertices[a] - vertices[b],
                                      vertices[a + 1] - vertices[b + 1],
                                      vertices[a + 2] - vertices[b + 2]));
  }
  std::sort(edge_lengths.begin(), edge_lengths.end());
  const auto quantile = [&](double fraction) {
    const double position = fraction * static_cast<double>(edge_lengths.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, edge_lengths.size() - 1);
    return edge_lengths[lower] + (position - static_cast<double>(lower)) *
                                    (edge_lengths[upper] - edge_lengths[lower]);
  };
  PHX_CHECK(!edge_lengths.empty());
  if (!edge_lengths.empty()) {
    PHX_CHECK(quantile(0.5) >= 0.15);
    PHX_CHECK(quantile(0.95) >= 0.15 && quantile(0.95) <= 0.45);
  }
  PHX_CHECK(counters[3] > 0);
  PHX_CHECK(signed_volume(vertices, triangles) > 0);
  for (size_t v = 0; v < vertices.size(); v += 3)
    PHX_CHECK(sphere_distance({vertices[v], vertices[v + 1], vertices[v + 2]}) <= distance);
  const double after = minimum_angle(vertices, triangles);
  std::printf("sphere carrier: min angle %.2f -> %.2f deg, %lld -> %lld triangles, "
              "unmet %lld, distance %.3g\n",
              before, after, static_cast<long long>(input.triangles.size() / 3),
              static_cast<long long>(sizes[1]), static_cast<long long>(counters[4]), distance);
  PHX_CHECK(before < 10.0);
  PHX_CHECK(after > 20.0);
  // Program/request validation.
  request.value = gradient.view();  // three outputs where one is declared
  PHX_CHECK(phx_mc_zero_set_remesh(&request) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_zero_set_remesh(nullptr) == PHX_MC_INVALID_ARGUMENT);
}
void remesh_edit_parity(int rings, int segments, double tolerance, double edge,
                        double angle, bool require_split) {
  const Surface input = uv_sphere(rings, segments);
  const auto original_vertices = input.vertices;
  const auto original_faces = input.triangles;
  const Program value = sphere_value(), gradient = sphere_gradient();
  std::vector<int32_t> compact(input.triangles.begin(), input.triangles.end());
  const auto original = compact;
  std::vector<int32_t> offsets(input.vertices.size() / 3 + 1, 0);
  for (int32_t vertex : compact) ++offsets[static_cast<size_t>(vertex) + 1];
  for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
  std::vector<int32_t> cursor = offsets;
  std::vector<int32_t> incidence(compact.size());
  for (size_t face = 0; face < compact.size() / 3; ++face)
    for (size_t corner = 0; corner < 3; ++corner) {
      const size_t vertex = static_cast<size_t>(compact[3 * face + corner]);
      incidence[static_cast<size_t>(cursor[vertex]++)] = static_cast<int32_t>(face);
    }
  const auto original_offsets = offsets;
  const auto original_incidence = incidence;
  const int64_t offset_count = static_cast<int64_t>(offsets.size());
  const int64_t incidence_count = static_cast<int64_t>(incidence.size());
  void* wide_handle = nullptr;
  void* compact_handle = nullptr;
  std::array<int64_t, 8> wide_counts{}, compact_counts{};
  double wide_distance = -1, compact_distance = -1;
  phx_mc_zero_set_remesh_request request{
      input.vertices.data(), static_cast<int64_t>(input.vertices.size() / 3),
      input.triangles.data(), static_cast<int64_t>(input.triangles.size() / 3),
      value.view(), gradient.view(), tolerance, tolerance, edge, angle,
      &wide_handle, wide_counts.data(), &wide_distance};
  PHX_CHECK(phx_mc_zero_set_remesh(&request) == PHX_MC_OK);
  request.triangles = reinterpret_cast<const int64_t*>(compact.data());
  request.result = &compact_handle;
  request.counters = compact_counts.data();
  request.distance = &compact_distance;
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&request, offsets.data(), offset_count,
                                       incidence.data(), incidence_count) == PHX_MC_OK);
  PHX_CHECK(wide_handle != nullptr && compact_handle != nullptr);
  if (wide_handle != nullptr && compact_handle != nullptr) {
    std::array<int64_t, 2> a{}, b{};
    PHX_CHECK(phx_mc_zero_set_remesh_sizes(wide_handle, a.data()) == PHX_MC_OK);
    PHX_CHECK(phx_mc_zero_set_remesh_sizes(compact_handle, b.data()) == PHX_MC_OK);
    PHX_CHECK(a == b);
    std::vector<double> av(static_cast<size_t>(3 * a[0])), bv(static_cast<size_t>(3 * b[0]));
    std::vector<int64_t> af(static_cast<size_t>(3 * a[1])), bf(static_cast<size_t>(3 * b[1]));
    PHX_CHECK(phx_mc_zero_set_remesh_arrays(wide_handle, a[0], a[1], av.data(), af.data()) ==
              PHX_MC_OK);
    PHX_CHECK(phx_mc_zero_set_remesh_arrays(compact_handle, b[0], b[1], bv.data(), bf.data()) ==
              PHX_MC_OK);
    PHX_CHECK(av == bv && af == bf);
    PHX_CHECK(wide_counts == compact_counts && wide_distance == compact_distance);
    PHX_CHECK(require_split ? wide_counts[0] > 0 : wide_counts[1] > 0);
    PHX_CHECK(compact == original && input.vertices == original_vertices &&
              input.triangles == original_faces && offsets == original_offsets &&
              incidence == original_incidence);
    PHX_CHECK(signed_volume(av, af) > 0);
    for (size_t i = 0; i < av.size(); i += 3)
      PHX_CHECK(sphere_distance({av[i], av[i + 1], av[i + 2]}) <= wide_distance);
    if (!require_split) {
      bool moved = false;
      for (size_t i = 0; i < av.size(); i += 3) {
        bool original_vertex = false;
        for (size_t j = 0; j < input.vertices.size(); j += 3)
          if (av[i] == input.vertices[j] && av[i + 1] == input.vertices[j + 1] &&
              av[i + 2] == input.vertices[j + 2]) original_vertex = true;
        moved = moved || !original_vertex;
      }
      PHX_CHECK(moved);
      PHX_CHECK(af.size() < input.triangles.size());
    }
  }
  if (wide_handle != nullptr) phx_mc_zero_set_remesh_free(wide_handle);
  if (compact_handle != nullptr) phx_mc_zero_set_remesh_free(compact_handle);
  compact_handle = nullptr;
  compact_counts.fill(-17);
  compact_distance = -17;
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&request, offsets.data(), offset_count - 1,
                                       incidence.data(), incidence_count) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(compact_handle == nullptr && compact_counts[0] == -17 && compact_distance == -17);
  offsets[0] = 1;
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&request, offsets.data(), offset_count,
                                       incidence.data(), incidence_count) == PHX_MC_INVALID_INPUT);
  offsets = original_offsets;
  incidence[0] = -1;
  PHX_CHECK(phx_mc_zero_set_remesh_i32(&request, offsets.data(), offset_count,
                                       incidence.data(), incidence_count) == PHX_MC_INVALID_INPUT);
  incidence = original_incidence;
  const size_t vertex = 0;
  if (offsets[vertex + 1] - offsets[vertex] >= 2) {
    incidence[static_cast<size_t>(offsets[vertex] + 1)] = incidence[static_cast<size_t>(offsets[vertex])];
    PHX_CHECK(phx_mc_zero_set_remesh_i32(&request, offsets.data(), offset_count,
                                         incidence.data(), incidence_count) == PHX_MC_INVALID_INPUT);
  }
  incidence = original_incidence;
  bool membership_checked = false;
  for (size_t owner = 0; owner + 1 < offsets.size() && !membership_checked; ++owner) {
    const int32_t begin = offsets[owner], end = offsets[owner + 1];
    for (int32_t slot = begin; slot < end && !membership_checked; ++slot) {
      const int32_t lo = slot == begin ? 0 : incidence[static_cast<size_t>(slot - 1)] + 1;
      const int32_t hi = slot + 1 == end ? static_cast<int32_t>(compact.size() / 3)
                                         : incidence[static_cast<size_t>(slot + 1)];
      for (int32_t face = lo; face < hi; ++face) {
        if (face == incidence[static_cast<size_t>(slot)]) continue;
        const size_t at = 3 * static_cast<size_t>(face);
        if (compact[at] == static_cast<int32_t>(owner) ||
            compact[at + 1] == static_cast<int32_t>(owner) ||
            compact[at + 2] == static_cast<int32_t>(owner)) continue;
        incidence[static_cast<size_t>(slot)] = face;
        PHX_CHECK(phx_mc_zero_set_remesh_i32(&request, offsets.data(), offset_count,
                                             incidence.data(), incidence_count) == PHX_MC_INVALID_INPUT);
        membership_checked = true;
        break;
      }
    }
  }
  PHX_CHECK(membership_checked);
  PHX_CHECK(compact_handle == nullptr && compact_counts[0] == -17 && compact_distance == -17);
}


// Lens: intersection of the unit balls centered at (0, 0, -+kLensShift),
// f = max(f1, f2) with f1 = |x - c1|^2 - 1, f2 = |x - c2|^2 - 1. The caps meet on
// the circle z = 0 of radius 0.6 with a 106 degree normal turn (a CSG crease).
constexpr double kLensShift = 0.8;
Program lens_value() {
  Program p = seeded({kLensShift, 1.0}, 17);
  p.row(kAdd, 5, 1, 2, -1, 3, -1, 0);       // z + shift
  p.row(kSubtract, 6, 1, 2, -1, 3, -1, 0);  // z - shift
  p.row(kSquare, 7, 2, 0, -1, -1, -1, 0);
  p.row(kReduceSum, 9, 1, 7, -1, -1, 0, 2);
  p.row(kSquare, 10, 1, 5, -1, -1, -1, 0);
  p.row(kSquare, 11, 1, 6, -1, -1, -1, 0);
  p.row(kAdd, 12, 1, 9, -1, 10, -1, 0);
  p.row(kAdd, 13, 1, 9, -1, 11, -1, 0);
  p.row(kSubtract, 14, 1, 12, -1, 4, -1, 0);  // f1
  p.row(kSubtract, 15, 1, 13, -1, 4, -1, 0);  // f2
  p.row(kMaximum, 16, 1, 14, -1, 15, -1, 0);
  p.outputs = {16};
  return p;
}
// grad f = grad f1 where f1 > f2 (z > 0), else grad f2; the hull of both on an
// undecided box.
Program lens_gradient() {
  Program p = seeded({kLensShift, 2.0, 0.0}, 20);
  p.indices = {0, 0, 0};
  p.row(kAdd, 6, 1, 2, -1, 3, -1, 0);
  p.row(kSubtract, 7, 1, 2, -1, 3, -1, 0);
  p.row(kMultiply, 8, 2, 0, -1, 4, 0, 0);    // case 0: grad f2
  p.row(kMultiply, 10, 1, 7, -1, 4, -1, 0);
  p.row(kMultiply, 11, 2, 0, -1, 4, 0, 0);   // case 1: grad f1
  p.row(kMultiply, 13, 1, 6, -1, 4, -1, 0);
  p.row(kCompare, 14, 3, 2, 0, 5, 0, 0);     // z > 0
  p.row(kSelect, 17, 3, 14, -1, 8, -1, 3);
  p.outputs = {17, 18, 19};
  return p;
}
Point lens_point(int ring, int rings, double phi) {
  // Rings 1..rings run down the upper cap to the crease, then up the lower cap.
  const double crease = std::acos(kLensShift);
  const bool upper = ring <= rings;
  const double theta = crease * (upper ? ring : 2 * rings - ring) / rings;
  const double z = std::cos(theta) - kLensShift;
  return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), upper ? z : -z};
}
Surface lens_surface(int rings, int segments) {
  Surface s;
  auto add = [&](const Point& p) {
    s.vertices.insert(s.vertices.end(), p.begin(), p.end());
    return static_cast<int64_t>(s.vertices.size() / 3 - 1);
  };
  const int64_t north = add({0, 0, 1 - kLensShift});
  std::vector<std::vector<int64_t>> ring(static_cast<size_t>(2 * rings - 1));
  for (int k = 1; k < 2 * rings; ++k)
    for (int j = 0; j < segments; ++j)
      ring[static_cast<size_t>(k - 1)].push_back(add(lens_point(k, rings, 2 * kPi * j / segments)));
  const int64_t south = add({0, 0, kLensShift - 1});
  auto tri = [&](int64_t a, int64_t b, int64_t c) { s.triangles.insert(s.triangles.end(), {a, b, c}); };
  for (int j = 0; j < segments; ++j) {
    const size_t n = static_cast<size_t>((j + 1) % segments), m = static_cast<size_t>(j);
    tri(north, ring.front()[m], ring.front()[n]);
    for (size_t k = 0; k + 1 < ring.size(); ++k) {
      tri(ring[k][m], ring[k + 1][m], ring[k + 1][n]);
      tri(ring[k][m], ring[k + 1][n], ring[k][n]);
    }
    tri(south, ring.back()[n], ring.back()[m]);
  }
  return s;
}

// Largest angle (degrees) between the normals of two triangles sharing an edge.
double maximum_turn(const std::vector<double>& v, const std::vector<int64_t>& t) {
  auto normal = [&](size_t f) {
    const double* a = &v[3 * static_cast<size_t>(t[f])];
    const double* b = &v[3 * static_cast<size_t>(t[f + 1])];
    const double* c = &v[3 * static_cast<size_t>(t[f + 2])];
    const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    Point n{u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
    const double length = std::hypot(n[0], n[1], n[2]);
    for (double& x : n) x /= length;
    return n;
  };
  std::map<std::pair<int64_t, int64_t>, std::vector<size_t>> edges;
  for (size_t f = 0; f < t.size(); f += 3)
    for (size_t k = 0; k < 3; ++k) {
      const int64_t a = t[f + k], b = t[f + (k + 1) % 3];
      edges[{std::min(a, b), std::max(a, b)}].push_back(f);
    }
  double turn = 0;
  for (const auto& [edge, faces] : edges) {
    if (faces.size() != 2) continue;
    const Point n = normal(faces[0]), m = normal(faces[1]);
    const double cosine = std::clamp(n[0] * m[0] + n[1] * m[1] + n[2] * m[2], -1.0, 1.0);
    turn = std::max(turn, std::acos(cosine) * 180 / kPi);
  }
  return turn;
}

// The afba level-0 sphere carrier (radius 0.371) scaled to the unit sphere: a
// graded rosette of 29.1-29.4 degree triangles with edges down to 0.0016
// around (-0.018, 0.738, -0.674), which refuses every single and relaxed
// collapse under the preserve admission.
const std::vector<double> kRosettePoints{
    0x1.eeda6ae2d21c0p-4, 0x1.3250dd65e43d6p-1, 0x1.a3b00cacfb909p-1,
    -0x1.1539ba00cd504p-3, 0x1.28e732f4b620fp-2, 0x1.effd5df9a051fp-1,
    0x1.1a1f1704eebbdp-2, 0x1.8c7c41da36572p-3, 0x1.ea0beccb18aa1p-1,
    0x1.04371bd452169p-1, 0x1.f43bcd08d2226p-2, 0x1.7c7eda30a5fd3p-1,
    0x1.eee818a03aa26p-1, 0x1.3948d3559920ap-4, 0x1.3b3a453f17ed2p-2,
    0x1.65bd6bb620306p-1, 0x1.d274fbdf966adp-4, 0x1.74bc80912193fp-1,
    0x1.94961f2d94024p-1, -0x1.25f71de0f3e0cp-2, 0x1.243605e728a0ap-1,
    -0x1.58ae4facd1ae1p-1, -0x1.35de981d47246p-1, -0x1.067eb178479f7p-1,
    -0x1.d2ebac70dec87p-2, -0x1.c449b23d2875fp-1, -0x1.b333bba8ebb8fp-3,
    -0x1.9aa59ca85d442p-1, -0x1.3efc5bc2f02dep-1, -0x1.858b26c3dddbfp-6,
    -0x1.fec32face981dp-1, -0x1.a96b6cc0af127p-3, 0x1.57019b7ce1807p-3,
    -0x1.63fbe33476659p-1, -0x1.240f5676c4fa6p-1, 0x1.e45da359f5054p-2,
    -0x1.e87d7b2cc9214p-1, -0x1.5f4c4212d333cp-3, -0x1.5c05057b75b94p-2,
    -0x1.9bb1ce742cf48p-1, -0x1.c1c2d1aa0d3d4p-4, 0x1.3db776fae2b83p-1,
    -0x1.e463a2eda8133p-1, 0x1.a293598d3c9fep-3, 0x1.4ed63f7eb314ap-2,
    -0x1.76e9f932c9bd7p-1, 0x1.782673623c1ccp-2, 0x1.40ac7e8c23b75p-1,
    -0x1.0c49e0ff5f7ebp-1, 0x1.7b3177608bff0p-4, 0x1.c3f12395360a4p-1,
    -0x1.5efa3d1f7f214p-2, 0x1.289bd88cd786bp-1, 0x1.8cad6d6d155b4p-1,
    -0x1.0b5aca318d79bp-4, 0x1.b1cf6a23946a8p-1, 0x1.226eba51f8332p-1,
    -0x1.d76fdb1bcd301p-2, 0x1.9ce816013265fp-1, 0x1.c7dcbaee794c1p-2,
    -0x1.9717944071b25p-2, -0x1.d59b0bee86c68p-1, 0x1.8b71401e2c755p-3,
    0x1.877bd2b9a9512p-2, 0x1.c1ec43500e0d2p-1, 0x1.a2e85cd842e22p-2,
    0x1.e3d53cb95eaa9p-1, -0x1.9f33b82ce78b1p-2, 0x1.5fbb622e4bdd5p-3,
    -0x1.c0c3a62a60e98p-2, -0x1.95f3f4440889bp-2, 0x1.a72547328598dp-1,
    -0x1.21224031d36b8p-2, -0x1.8f16ebc17bc2bp-1, 0x1.2dbb50677f9cap-1,
    0x1.1151c5b0c39a0p-3, -0x1.e97c56b38993cp-1, 0x1.51d6d6063a479p-2,
    0x1.33424e9ccaa0ap-4, -0x1.4cd46d259462ep-1, 0x1.922d5bbd5dc3bp-1,
    0x1.2a80155ca0a14p-1, -0x1.5a1fb3fe15c17p-1, 0x1.f703c785b15bcp-2,
    -0x1.2a2dce78bed9dp-4, -0x1.22f300bbbc5c8p-3, 0x1.010e6bc07c6d7p+0,
    0x1.be7e68880fd60p-2, -0x1.351f538ba2173p-2, 0x1.bd38b7d38b247p-1,
    0x1.6511c9f42c247p-5, -0x1.01f6b8f080fa6p+0, -0x1.44affc2f2d413p-3,
    0x1.344e15cee2adap-1, -0x1.a4dc439e91b85p-1, -0x1.744958f89911ap-7,
    -0x1.5dfe212f829c0p-1, -0x1.55aea0e320c9dp-3, -0x1.7f65e977a3085p-1,
    -0x1.a41641b3288c3p-1, 0x1.471c09abe96d3p-2, -0x1.1ca30af5a24f5p-1,
    -0x1.b6a3b7181eb52p-2, 0x1.30d1c0928a491p-1, -0x1.6d2df8c77a297p-1,
    -0x1.09f6f03cf6ac7p-1, 0x1.a26e20453ec5ep-3, -0x1.b81d817bc57ccp-1,
    -0x1.7dd7aed8cb54dp-3, 0x1.7374de04c0621p-1, -0x1.586669a0d0b99p-1,
    -0x1.e112520b5c729p-5, 0x1.417b59422c6b1p-1, -0x1.914f0eb71437fp-1,
    -0x1.14e8b0cf0884fp-4, 0x1.7024be3a40682p-1, -0x1.62728985e2fa9p-1,
    -0x1.e37497eee1e07p-5, 0x1.87c968f526c11p-1, -0x1.48e7da2a93fa4p-1,
    -0x1.1e7a06578f006p-5, 0x1.7b720e78c8beep-1, -0x1.57496a43f9a60p-1,
    -0x1.4ff0f5b5e1ca1p-6, 0x1.7c288b984c37fp-1, -0x1.56cd6167cbe72p-1,
    -0x1.7a8ba54082a3fp-6, 0x1.79ca2f7b1a8f2p-1, -0x1.595e2a2b8fec0p-1,
    -0x1.4616ac7bd699ep-6, 0x1.7abe0c56886d4p-1, -0x1.5860f7d3d4065p-1,
    -0x1.7b733a0aee504p-6, 0x1.745bb57536ac2p-1, -0x1.5f1d57e812aafp-1,
    -0x1.cd7c68bedcc0dp-7, 0x1.78a4b2ae9f2a9p-1, -0x1.5ac9228cf7931p-1,
    -0x1.196d2e08cc41bp-6, 0x1.7a1c0015c2671p-1, -0x1.591b5b621b3e7p-1,
    -0x1.dcc4084b0f46ep-7, 0x1.7a18505c16649p-1, -0x1.59274f7947650p-1,
    -0x1.6ddb1ad6c7c71p-7, 0x1.7b0588d4ee328p-1, -0x1.582bdf17000f3p-1,
    -0x1.18e172d25be0dp-6, 0x1.7aacb3bd5ffe2p-1, -0x1.587e37302d7bap-1,
    -0x1.1f1ed9e9e9d5bp-6, 0x1.812ab7483756bp-1, -0x1.51484d4eb1673p-1,
    -0x1.96ca428c19d60p-2, 0x1.b7ee1b1a4093ep-1, -0x1.962458a80720bp-2,
    -0x1.f9c433610cb75p-1, 0x1.2c711184bd531p-2, -0x1.88c2762fb18dfp-4,
    -0x1.7535e7b93af33p-1, 0x1.614d84c139e75p-1, -0x1.03fd01b7644f5p-2,
    -0x1.933157b31e546p-1, 0x1.49d29dea4762bp-1, 0x1.8e531c6b42da1p-3,
    -0x1.d6ffe9531fab5p-2, 0x1.d683575467c06p-1, 0x1.38d603a4f55e8p-8,
    -0x1.52f3dc02fc5e8p-2, -0x1.08a4e56bc60d9p-1, -0x1.ab05dda445b51p-1,
    -0x1.0adc4d05b99f7p-3, -0x1.aa5d9559e90a3p-1, -0x1.3132240740c40p-1,
    -0x1.df03365ef0029p-3, -0x1.5fd7ae5d19a62p-3, -0x1.fe57ea82b1eb8p-1,
    0x1.08aff7e7e1d67p-1, -0x1.7ed6317851ec2p-1, -0x1.0324032cf0721p-1,
    0x1.bbb773d6992bbp-1, -0x1.e71bc2b32f214p-2, -0x1.07fa1c476f699p-2,
    -0x1.3bde7ac19454bp-3, 0x1.5d12ee23f9477p-2, -0x1.e889a7d475336p-1,
    0x1.c2462ca517c2ep-3, 0x1.b6e31a4d2c469p-4, -0x1.003f79aea7b1ep+0,
    0x1.0737baa1a23b5p-2, -0x1.b64404c08465ap-2, -0x1.cce057712edaap-1,
    -0x1.3278b5811ee38p-7, 0x1.677a33e7b8ab2p-1, -0x1.6e100028dfe34p-1,
    -0x1.aae47cc5181d6p-12, 0x1.77a8e678cb6ecp-1, -0x1.5bcaa804c8d40p-1,
    0x1.cc7e47085c51ep-7, 0x1.7f4152df2dee8p-1, -0x1.53c29719c157ep-1,
    0x1.5514ccd043626p-5, 0x1.7766e6047b359p-1, -0x1.5b7486e53c7a5p-1,
    0x1.67185b39f91e8p-7, 0x1.92cb1c6b2dd11p-1, -0x1.3c733d4d25fadp-1,
    0x1.35b782e21ece9p-4, 0x1.86eae5fab735bp-1, -0x1.47ede0040dcb2p-1,
    0x1.255986670b4d8p-3, 0x1.9dafafac3df0fp-1, -0x1.2cc20289cf6bfp-1,
    0x1.e09ee90a7fcc7p-4, 0x1.6b34a9c1ca917p-1, -0x1.67126f30a0320p-1,
    0x1.03dcccc8b916ep-3, 0x1.2106a2082a38fp-1, -0x1.a86d82892f41dp-1,
    -0x1.215da30f53b98p-4, 0x1.b663bef97d458p-1, -0x1.0e1f5008e8a5bp-1,
    -0x1.44bc84f605a98p-4, 0x1.f94d1dc0508fdp-1, -0x1.1a6e2aa72b64ap-2,
    0x1.2f71f9f20ba02p-1, 0x1.483c747cf8a91p-7, -0x1.b5da3ae38bb9dp-1,
    0x1.92491fdd0eb43p-1, -0x1.ee1564b519c7bp-3, -0x1.3981a2d50ebcep-1,
    0x1.e5bd5882747e0p-2, 0x1.82aa2403130c3p-2, -0x1.a8b5303ae0f4ep-1,
    0x1.9b0bea8db080cp-1, 0x1.6c77c8d6eb7dfp-2, -0x1.1bbb89267ca71p-1,
    0x1.99fc66b5a9e84p-2, 0x1.6ed2af77f6d3cp-1, -0x1.3c6f22c83565ep-1,
    0x1.60f6f85cbbdbbp-1, 0x1.69b4888837a53p-1, -0x1.2e1a711f02a63p-2,
    -0x1.11bacb87e81b1p-4, 0x1.05a49b1139b46p+0, 0x1.66f79a4dc4832p-3,
    0x1.6158522ee03a9p-2, 0x1.ed0218167a92fp-1, -0x1.6ca7986b488edp-3,
    0x1.05fd2ae034fbdp+0, -0x1.50323c2acbcb7p-6, -0x1.9ff869c30b015p-3,
    0x1.e263e3cc86963p-1, 0x1.afbd1108acd3cp-2, -0x1.7220e24775550p-4,
    0x1.9c956a31e7721p-1, 0x1.0487406aca354p-1, 0x1.b0f400e9f564fp-2,
    0x1.5f064c3baf1afp-1, 0x1.867efa1af3b5fp-1, 0x1.3a4b84e909539p-4,
};
const std::vector<int64_t> kRosetteTriangles{
    0, 1, 2, 3, 0, 2, 4, 5, 6, 5, 3, 2, 7, 8, 9, 10, 9, 11,
    7, 9, 12, 10, 12, 9, 11, 13, 10, 10, 13, 14, 15, 14, 13, 13, 16, 15,
    1, 0, 17, 0, 18, 17, 1, 17, 16, 15, 16, 17, 17, 18, 19, 17, 19, 15,
    8, 20, 9, 11, 9, 20, 0, 21, 18, 0, 3, 21, 4, 6, 22, 23, 13, 11,
    23, 16, 13, 11, 20, 24, 23, 11, 24, 20, 25, 24, 25, 26, 24, 23, 24, 26,
    25, 27, 26, 2, 1, 28, 28, 1, 16, 23, 28, 16, 23, 26, 28, 6, 5, 29,
    29, 5, 2, 6, 29, 27, 27, 29, 26, 29, 2, 28, 29, 28, 26, 6, 27, 22,
    20, 8, 30, 20, 30, 25, 30, 31, 25, 25, 31, 27, 31, 22, 27, 12, 32, 7,
    33, 32, 12, 34, 35, 33, 32, 33, 35, 34, 36, 37, 36, 38, 37, 36, 39, 38,
    40, 41, 42, 38, 39, 40, 42, 41, 43, 40, 42, 44, 44, 38, 40, 44, 42, 45,
    42, 43, 46, 42, 46, 45, 45, 47, 48, 45, 46, 47, 49, 41, 48, 43, 41, 49,
    43, 49, 46, 49, 48, 47, 46, 49, 47, 50, 48, 41, 40, 50, 41, 40, 39, 50,
    36, 34, 51, 12, 10, 52, 52, 10, 14, 33, 12, 52, 34, 33, 53, 34, 53, 51,
    33, 52, 53, 15, 54, 14, 19, 54, 15, 52, 14, 54, 54, 53, 52, 51, 53, 55,
    55, 54, 19, 54, 55, 53, 32, 56, 7, 8, 7, 57, 30, 8, 57, 7, 56, 57,
    58, 32, 35, 58, 56, 32, 30, 59, 31, 30, 57, 59, 31, 60, 22, 59, 60, 31,
    61, 35, 34, 61, 34, 37, 58, 35, 61, 58, 61, 62, 56, 63, 57, 56, 58, 63,
    63, 59, 57, 63, 58, 62, 38, 64, 37, 38, 44, 64, 45, 48, 65, 44, 45, 65,
    64, 44, 65, 66, 48, 50, 48, 66, 65, 65, 67, 64, 66, 67, 65, 39, 68, 50,
    50, 68, 66, 66, 69, 67, 70, 69, 68, 66, 68, 69, 71, 64, 67, 70, 71, 69,
    67, 69, 71, 61, 37, 72, 62, 61, 72, 37, 64, 72, 72, 64, 71, 36, 73, 39,
    36, 51, 73, 68, 73, 70, 73, 68, 39, 51, 55, 74, 74, 70, 73, 73, 51, 74,
    75, 63, 62, 76, 60, 59, 76, 59, 63, 75, 76, 63, 62, 72, 77, 75, 62, 77,
    78, 76, 75, 78, 75, 77, 71, 70, 79, 72, 71, 79, 77, 72, 79, 78, 77, 79,
    80, 78, 79, 19, 18, 81, 21, 81, 18, 55, 19, 81, 55, 81, 74, 82, 70, 74,
    82, 79, 70, 79, 82, 80, 81, 21, 82, 81, 82, 74, 4, 22, 83, 83, 22, 60,
    76, 83, 60, 76, 78, 83, 78, 80, 84, 83, 84, 4, 83, 78, 84, 85, 5, 4,
    5, 85, 3, 85, 21, 3, 84, 85, 4, 21, 86, 82, 86, 80, 82, 80, 86, 84,
    86, 21, 85, 85, 84, 86,
};

// Admission-limited coarsening of that carrier: the absorbing collapse merges
// the rosette's core, so no triangle stays below full score and no edge of the
// rosette core survives, within the same certified band and orientation.
void remesh_coarsening_absorbs_rosette() {
  PHX_CHECK(signed_volume(kRosettePoints, kRosetteTriangles) > 0);
  PHX_CHECK(minimum_angle(kRosettePoints, kRosetteTriangles) < 29.2);
  const Program value = sphere_value(), gradient = sphere_gradient();
  constexpr double tolerance = 0.02 / 0.371, edge = 0.5;
  void* handle = nullptr;
  std::array<int64_t, 8> counters{};
  double distance = -1;
  phx_mc_zero_set_remesh_request request{
      kRosettePoints.data(), static_cast<int64_t>(kRosettePoints.size() / 3),
      kRosetteTriangles.data(), static_cast<int64_t>(kRosetteTriangles.size() / 3),
      value.view(),          gradient.view(),
      tolerance,             tolerance,
      edge,                  30.0,
      &handle,               counters.data(),
      &distance,             1};
  PHX_CHECK(phx_mc_zero_set_remesh(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  std::array<int64_t, 2> sizes{};
  PHX_CHECK(phx_mc_zero_set_remesh_sizes(handle, sizes.data()) == PHX_MC_OK);
  std::vector<double> vertices(static_cast<size_t>(3 * sizes[0]));
  std::vector<int64_t> triangles(static_cast<size_t>(3 * sizes[1]));
  PHX_CHECK(phx_mc_zero_set_remesh_arrays(handle, sizes[0], sizes[1], vertices.data(),
                                          triangles.data()) == PHX_MC_OK);
  phx_mc_zero_set_remesh_free(handle);
  PHX_CHECK(counters[4] == 0);  // no unmet triangle
  PHX_CHECK(counters[5] == 1);
  PHX_CHECK(distance >= 0 && distance <= tolerance);
  PHX_CHECK(signed_volume(vertices, triangles) > 0);
  PHX_CHECK(minimum_angle(vertices, triangles) >= 30.0);
  double shortest = std::numeric_limits<double>::infinity();
  for (size_t f = 0; f < triangles.size(); f += 3)
    for (int k = 0; k < 3; ++k) {
      const double* p = &vertices[3 * static_cast<size_t>(triangles[f + static_cast<size_t>(k)])];
      const double* q =
          &vertices[3 * static_cast<size_t>(triangles[f + static_cast<size_t>((k + 1) % 3)])];
      shortest = std::min(shortest, std::hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]));
    }
  // The rosette core's edges reach 0.0016 (0.003 of the requested length);
  // no edge of that scale survives.
  PHX_CHECK(shortest > 0.2 * edge);
  PHX_CHECK(sizes[0] < static_cast<int64_t>(kRosettePoints.size() / 3));
}


// The crease of a CSG lens is a turn its source admits: the fold factor leaves
// it at full score, so remeshing keeps the crease and leaves nothing unmet
// there (a fixed 90 degree turn limit scored every crease triangle unmet).
void remesh_lens_crease() {
  const Surface input = lens_surface(4, 24);
  PHX_CHECK(signed_volume(input.vertices, input.triangles) > 0);
  PHX_CHECK(maximum_turn(input.vertices, input.triangles) > 100);
  const Program value = lens_value(), gradient = lens_gradient();
  void* handle = nullptr;
  std::array<int64_t, 8> counters{};
  double distance = -1;
  constexpr double tolerance = 0.02;
  phx_mc_zero_set_remesh_request request{
      input.vertices.data(), static_cast<int64_t>(input.vertices.size() / 3),
      input.triangles.data(), static_cast<int64_t>(input.triangles.size() / 3),
      value.view(),          gradient.view(),
      tolerance,             tolerance,
      0.2,                   30.0,
      &handle,               counters.data(),
      &distance};
  PHX_CHECK(phx_mc_zero_set_remesh(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  std::array<int64_t, 2> sizes{};
  PHX_CHECK(phx_mc_zero_set_remesh_sizes(handle, sizes.data()) == PHX_MC_OK);
  std::vector<double> vertices(static_cast<size_t>(3 * sizes[0]));
  std::vector<int64_t> triangles(static_cast<size_t>(3 * sizes[1]));
  PHX_CHECK(phx_mc_zero_set_remesh_arrays(handle, sizes[0], sizes[1], vertices.data(),
                                          triangles.data()) == PHX_MC_OK);
  phx_mc_zero_set_remesh_free(handle);
  const double turn = maximum_turn(vertices, triangles);
  std::printf("lens crease: %lld -> %lld triangles, max turn %.1f deg, unmet %lld, "
              "distance %.3g, counters %lld %lld %lld %lld\n",
              static_cast<long long>(input.triangles.size() / 3),
              static_cast<long long>(sizes[1]), turn, static_cast<long long>(counters[4]),
              distance, static_cast<long long>(counters[0]), static_cast<long long>(counters[1]),
              static_cast<long long>(counters[2]), static_cast<long long>(counters[3]));
  PHX_CHECK(distance >= 0 && distance <= tolerance);
  PHX_CHECK(turn > 100);
  PHX_CHECK(counters[4] == 0);
}
void scratch_faces_eviction() {
  phx::mc::remesh::ScratchFaces scratch;
  constexpr int32_t count = 1030 * 1024 + 37;
  for (int32_t i = 0; i < count; ++i) scratch.push({i, -i, i ^ 0x12345});
  PHX_CHECK(scratch.size() == static_cast<size_t>(count));
  for (int32_t i = count - 1; i >= 0; --i) {
    const std::array<int32_t, 3> expected{i, -i, i ^ 0x12345};
    PHX_CHECK(scratch.get(static_cast<size_t>(i)) == expected);
  }
  scratch.clear();
  PHX_CHECK(scratch.size() == 0);
  scratch.push({7, 11, 13});
  const std::array<int32_t, 3> expected{7, 11, 13};
  PHX_CHECK(scratch.get(0) == expected);
  phx::mc::remesh::ScratchRows<VertexEvidence, 2> evidence;
  for (int i = 0; i < 3 * 1024 + 1; ++i) evidence.push({double(i), -double(i)});
  evidence.set(0, {0.125, 0.25});
  evidence.set(2048, {3.5, 4.5});
  PHX_CHECK(evidence.get(0).lower == 0.125 && evidence.get(0).upper == 0.25);
  PHX_CHECK(evidence.get(2048).lower == 3.5 && evidence.get(2048).upper == 4.5);
  evidence.pop();
  evidence.push({7.5, 8.5});
  PHX_CHECK(evidence.get(3072).lower == 7.5 && evidence.get(3072).upper == 8.5);
  {
    phx::mc::remesh::ScratchRows<Point, 2> points;
    for (int i = 0; i < 3 * 1024 + 1; ++i) points.push({double(i), -double(i), 0.5});
    const Point replacement{1.25, -2.5, 3.75};
    points.set(0, replacement);
    PHX_CHECK(points.get(0) == replacement);
    PHX_CHECK(points.get(3072) == (Point{3072., -3072., 0.5}));
  }
#if !defined(_WIN32)
  std::FILE* probe_before = std::tmpfile();
  PHX_CHECK(probe_before != nullptr);
  const int expected_fd = probe_before == nullptr ? -1 : fileno(probe_before);
  if (probe_before != nullptr) std::fclose(probe_before);
  {
    phx::mc::remesh::ScratchRows<Point, 2> temporary;
    temporary.push({1., 2., 3.});
  }
  std::FILE* probe_after = std::tmpfile();
  PHX_CHECK(probe_after != nullptr);
  if (probe_after != nullptr) {
    PHX_CHECK(fileno(probe_after) == expected_fd);
    std::fclose(probe_after);
  }
  struct rlimit old_limit{};
  PHX_CHECK(getrlimit(RLIMIT_FSIZE, &old_limit) == 0);
  struct rlimit limited = old_limit;
  limited.rlim_cur = 0;
  const auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
  PHX_CHECK(setrlimit(RLIMIT_FSIZE, &limited) == 0);
  bool failed = false;
  try {
    phx::mc::remesh::ScratchFaces full;
    for (int32_t i = 0; i < count; ++i) full.push({i, i, i});
  } catch (const std::system_error&) {
    failed = true;
  }
  bool points_failed = false;
  const Point immutable{1., 2., 3.};
  try {
    phx::mc::remesh::ScratchRows<Point, 2> full_points;
    for (int i = 0; i < 3 * 1024 + 1; ++i) full_points.push(immutable);
  } catch (const std::system_error&) {
    points_failed = true;
  }
  PHX_CHECK(points_failed);
  PHX_CHECK(immutable == (Point{1., 2., 3.}));
  PHX_CHECK(setrlimit(RLIMIT_FSIZE, &old_limit) == 0);
  std::signal(SIGXFSZ, old_signal);
  PHX_CHECK(failed);
#endif
}

}  // namespace

int main() {
  sphere_cases();
  torus_cases();
  near_band_distance_cases();
  whole_piece_bracketing_cases();
  remesh_sphere_carrier();
  remesh_coarsening_absorbs_rosette();
  remesh_lens_crease();
  remesh_edit_parity(5, 12, 0.08, 0.22, 25.0, true);
  remesh_edit_parity(8, 40, 0.2, 1.0, 25.0, false);
  scratch_faces_eviction();
  return phx::mc::test::finish("test_zero_set_admission");
}
