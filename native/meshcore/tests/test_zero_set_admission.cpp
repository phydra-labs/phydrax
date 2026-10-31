// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// kZeroSetBand admission soundness (sphere and torus fields against their exact
// distance functions), interval-Newton projection, and the zero-set remesh C entry.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
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
  // Closed oriented 2-manifold: every directed edge appears once, reversed once.
  std::map<std::pair<int64_t, int64_t>, int> directed;
  for (size_t f = 0; f < triangles.size(); f += 3)
    for (size_t k = 0; k < 3; ++k) ++directed[{triangles[f + k], triangles[f + (k + 1) % 3]}];
  bool manifold = true;
  for (const auto& [edge, count] : directed)
    manifold = manifold && count == 1 && directed.count({edge.second, edge.first}) == 1;
  PHX_CHECK(manifold);
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
}  // namespace

int main() {
  sphere_cases();
  torus_cases();
  remesh_sphere_carrier();
  return phx::mc::test::finish("test_zero_set_admission");
}
