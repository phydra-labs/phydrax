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
  remesh_sphere_carrier();
  remesh_lens_crease();
  remesh_edit_parity(5, 12, 0.08, 0.22, 25.0, true);
  remesh_edit_parity(8, 40, 0.2, 1.0, 25.0, false);
  scratch_faces_eviction();
  return phx::mc::test::finish("test_zero_set_admission");
}
