//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Bit vectors of the metric edit goals. The inputs are produced here from
// integers only (small integer matrices and coordinates scaled by powers of
// two), so they are the same binary64 values on every platform.
// metric_goal_vectors.inc holds each input with the output bits the kernels
// produced when the file was written; this test requires the same bits on
// every platform. `--emit` prints that file.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bit_vectors.hpp"
#include "check.hpp"
#include "floating_point_mode.hpp"
#include "metric_goal.hpp"
#include "phydrax_meshcore.h"

namespace {

using phx::mc::test::append_bits;
using phx::mc::test::BitRow;
using phx::mc::test::bits_of;
using phx::mc::test::IntegerGenerator;
using phx::mc::test::kInfinityBits;
using phx::mc::test::kNaNBits;
using phx::mc::test::power_of_two;
using phx::mc::test::value_of;

#define PHX_MG_MIDPOINT(...) {"PHX_MG_MIDPOINT", {__VA_ARGS__}},
#define PHX_MG_CENTER(...) {"PHX_MG_CENTER", {__VA_ARGS__}},
#define PHX_MG_HULL(...) {"PHX_MG_HULL", {__VA_ARGS__}},
#define PHX_MG_SURVIVOR(...) {"PHX_MG_SURVIVOR", {__VA_ARGS__}},
const std::vector<BitRow> kRecorded = {
#include "metric_goal_vectors.inc"
};

using Metric = std::array<double, 9>;
using Point = std::array<double, 3>;

constexpr Metric kIdentity = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

// (A^T A + I) 2^e for an integer matrix A with entries in [-4, 4].
Metric generated_metric(IntegerGenerator& generator) {
  std::int64_t a[9];
  for (std::int64_t& entry : a) entry = generator.integer(-4, 4);
  const double scale = power_of_two(static_cast<int>(generator.integer(-6, 6)));
  Metric metric{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::int64_t sum = i == j ? 1 : 0;
      for (int k = 0; k < 3; ++k) sum += a[3 * k + i] * a[3 * k + j];
      metric[static_cast<std::size_t>(3 * i + j)] = static_cast<double>(sum) * scale;
    }
  }
  return metric;
}

// Coordinates k / 16 for an integer k in [-64, 64].
Point generated_point(IntegerGenerator& generator) {
  Point point{};
  for (double& value : point) value = static_cast<double>(generator.integer(-64, 64)) * 0x1p-4;
  return point;
}

void midpoint_rows(std::vector<BitRow>& rows) {
  struct Segment {
    Point first;
    Point second;
    Metric first_metric;
    Metric second_metric;
  };
  const Metric quadruple = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
  const Metric indefinite = {1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const Metric huge = {0x1p+1000, 0.0, 0.0, 0.0, 0x1p+1000, 0.0, 0.0, 0.0, 0x1p+1000};
  const Metric tiny = {0x1p-1000, 0.0, 0.0, 0.0, 0x1p-1000, 0.0, 0.0, 0.0, 0x1p-1000};
  Metric with_nan = kIdentity;
  with_nan[4] = value_of(kNaNBits);
  std::vector<Segment> segments = {
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, kIdentity, kIdentity},
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, kIdentity, quadruple},
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, quadruple, kIdentity},
      {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, kIdentity, quadruple},
      // No metric midpoint: a form that is negative, not a number or zero,
      // and a ratio of the forms beyond the binary64 range.
      {{0.0, 0.0, 0.0}, {1.0, -1.0, 0.0}, indefinite, kIdentity},
      {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, kIdentity, with_nan},
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, tiny, huge},
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, huge, tiny},
      {{0.0, 0.0, 0.0}, {0x1p+600, 0.0, 0.0}, kIdentity, quadruple}};
  IntegerGenerator generator{0x2ffd72dbd01adfb7ULL};
  for (int row = 0; row < 40; ++row) {
    segments.push_back({generated_point(generator), generated_point(generator),
                        generated_metric(generator), generated_metric(generator)});
  }
  const std::size_t count = segments.size();
  std::vector<double> first, second, first_metrics, second_metrics;
  for (const Segment& segment : segments) {
    first.insert(first.end(), segment.first.begin(), segment.first.end());
    second.insert(second.end(), segment.second.begin(), segment.second.end());
    first_metrics.insert(first_metrics.end(), segment.first_metric.begin(),
                         segment.first_metric.end());
    second_metrics.insert(second_metrics.end(), segment.second_metric.begin(),
                          segment.second_metric.end());
  }
  std::vector<double> parameters(count), points(3 * count);
  PHX_CHECK(phx_mc_metric_midpoints(static_cast<int64_t>(count), first.data(), second.data(),
                                    first_metrics.data(), second_metrics.data(),
                                    parameters.data(), points.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < count; ++row) {
    BitRow record{"PHX_MG_MIDPOINT", {}};
    append_bits(record, first.data() + 3 * row, 3);
    append_bits(record, second.data() + 3 * row, 3);
    append_bits(record, first_metrics.data() + 9 * row, 9);
    append_bits(record, second_metrics.data() + 9 * row, 9);
    record.fields.push_back(bits_of(parameters[row]));
    append_bits(record, points.data() + 3 * row, 3);
    rows.push_back(record);
    // A row does not depend on the rows evaluated with it.
    double parameter = -1.0;
    double point[3];
    PHX_CHECK(phx_mc_metric_midpoints(1, first.data() + 3 * row, second.data() + 3 * row,
                                      first_metrics.data() + 9 * row,
                                      second_metrics.data() + 9 * row, &parameter,
                                      point) == PHX_MC_OK);
    PHX_CHECK(bits_of(parameter) == bits_of(parameters[row]));
    for (std::size_t axis = 0; axis < 3; ++axis) {
      PHX_CHECK(bits_of(point[axis]) == bits_of(points[3 * row + axis]));
    }
  }
}

void center_rows(std::vector<BitRow>& rows) {
  struct Cloud {
    Metric metric;
    std::vector<double> points;
  };
  const Metric anisotropic = {4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25};
  const Metric indefinite = {1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<Cloud> clouds = {
      {kIdentity, {1.0, 2.0, 3.0}},
      {kIdentity, {0.0, 0.0, 0.0, 2.0, 0.0, 0.0}},
      // Coincident, collinear and coplanar points.
      {kIdentity, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
      {kIdentity, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0}},
      {kIdentity, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0}},
      // The corners of the unit cube: eight points on one sphere.
      {kIdentity, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                   0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
      {anisotropic, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}},
      // Refused: no factor, a point that is not finite, a squared coordinate
      // beyond the binary64 range, no point.
      {indefinite, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}},
      {kIdentity, {0.0, 0.0, 0.0, value_of(kInfinityBits), 0.0, 0.0}},
      {kIdentity, {0.0, 0.0, 0.0, 0x1p+600, 0.0, 0.0}},
      {kIdentity, {}}};
  IntegerGenerator generator{0x6a09e667bb67ae85ULL};
  for (int row = 0; row < 36; ++row) {
    Cloud cloud{generated_metric(generator), {}};
    const std::int64_t count = generator.integer(2, 14);
    for (std::int64_t index = 0; index < count; ++index) {
      const Point point = generated_point(generator);
      cloud.points.insert(cloud.points.end(), point.begin(), point.end());
    }
    clouds.push_back(cloud);
  }
  for (const Cloud& cloud : clouds) {
    const std::int64_t count = static_cast<std::int64_t>(cloud.points.size() / 3);
    double center[3] = {0.0, 0.0, 0.0};
    const bool valid =
        phx::mc::metric::enclosing_center(count, cloud.points.data(), cloud.metric.data(), center);
    BitRow record{"PHX_MG_CENTER", {static_cast<std::uint64_t>(count)}};
    append_bits(record, cloud.metric.data(), 9);
    append_bits(record, cloud.points.data(), cloud.points.size());
    record.fields.push_back(valid ? 1U : 0U);
    append_bits(record, center, 3);
    rows.push_back(record);
  }
}

void hull_rows(std::vector<BitRow>& rows) {
  struct Hull {
    int corner_count;
    std::array<double, 9> corners;
    Point goal;
  };
  std::vector<Hull> hulls = {
      {2, {0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 5.0, -3.0}},
      {3, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, {0.25, 0.5, 7.0}},
      {3, {1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0}, {1.0, 1.0, 1.0}},
      // Dependent edges: a repeated corner, collinear corners, one point.
      {2, {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0}, {4.0, 4.0, 4.0}},
      {3, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0}, {1.0, 0.0, 5.0}},
      {3, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}},
      // Refused: values that are not finite.
      {3, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, {value_of(kNaNBits), 0.0, 0.0}},
      {2, {0.0, 0.0, 0.0, 0x1p+600, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
  IntegerGenerator generator{0x3c6ef372a54ff53aULL};
  for (int row = 0; row < 32; ++row) {
    Hull hull{2 + row % 2, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {}};
    for (int corner = 0; corner < hull.corner_count; ++corner) {
      const Point point = generated_point(generator);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        hull.corners[3 * static_cast<std::size_t>(corner) + axis] = point[axis];
      }
    }
    hull.goal = generated_point(generator);
    hulls.push_back(hull);
  }
  for (const Hull& hull : hulls) {
    double projected[3] = {0.0, 0.0, 0.0};
    const bool valid = phx::mc::metric::hull_projection(hull.corner_count, hull.corners.data(),
                                                        hull.goal.data(), projected);
    BitRow record{"PHX_MG_HULL", {static_cast<std::uint64_t>(hull.corner_count)}};
    append_bits(record, hull.corners.data(), 9);
    append_bits(record, hull.goal.data(), 3);
    record.fields.push_back(valid ? 1U : 0U);
    append_bits(record, projected, 3);
    rows.push_back(record);
  }
}

void survivor_rows(std::vector<BitRow>& rows) {
  struct Star {
    int corner_count;
    std::array<double, 9> corners;
    std::array<double, 18> metrics;
    std::vector<double> points;  // survivor, removed vertex, link
  };
  std::vector<Star> stars;
  const auto metrics_of = [](const Metric& first, const Metric& second) {
    std::array<double, 18> both{};
    for (std::size_t index = 0; index < 9; ++index) {
      both[index] = first[index];
      both[9 + index] = second[index];
    }
    return both;
  };
  const Metric quadruple = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
  const Metric indefinite = {1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 9> none = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const std::array<double, 9> plane = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  const std::array<double, 9> line = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<double> octahedron = {0.0, 0.0,  0.0, 1.0, 0.0, 0.0, 2.0,  0.0, 0.0,
                                          0.0, 2.0,  0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0,
                                          0.0, -2.0, 0.0, 0.0, 0.0, -2.0};
  stars.push_back({0, none, metrics_of(kIdentity, kIdentity), octahedron});
  stars.push_back({0, none, metrics_of(kIdentity, quadruple), octahedron});
  stars.push_back({3, plane, metrics_of(kIdentity, quadruple), octahedron});
  stars.push_back({2, line, metrics_of(quadruple, kIdentity), octahedron});
  // No link: the first two goals stand, the third has no ball.
  stars.push_back({0, none, metrics_of(kIdentity, quadruple), {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}});
  // A tensor that is not a metric: no mean, so no third goal.
  stars.push_back({0, none, metrics_of(kIdentity, indefinite), octahedron});
  IntegerGenerator generator{0x510e527f9b05688cULL};
  for (int row = 0; row < 30; ++row) {
    Star star{0, none, metrics_of(generated_metric(generator), generated_metric(generator)), {}};
    star.corner_count = row % 3 == 0 ? 0 : 1 + row % 3;
    for (int corner = 0; corner < star.corner_count; ++corner) {
      const Point point = generated_point(generator);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        star.corners[3 * static_cast<std::size_t>(corner) + axis] = point[axis];
      }
    }
    const std::int64_t count = generator.integer(5, 16);
    for (std::int64_t index = 0; index < count; ++index) {
      const Point point = generated_point(generator);
      star.points.insert(star.points.end(), point.begin(), point.end());
    }
    stars.push_back(star);
  }
  for (const Star& star : stars) {
    const int64_t count = static_cast<int64_t>(star.points.size() / 3);
    double goals[9];
    int8_t valid[3] = {-1, -1, -1};
    PHX_CHECK(phx_mc_metric_survivor_goals(count, star.points.data(), star.metrics.data(),
                                           star.corner_count,
                                           star.corner_count == 0 ? nullptr : star.corners.data(),
                                           goals, valid) == PHX_MC_OK);
    BitRow record{"PHX_MG_SURVIVOR",
                  {static_cast<std::uint64_t>(count),
                   static_cast<std::uint64_t>(star.corner_count)}};
    append_bits(record, star.metrics.data(), 18);
    append_bits(record, star.corners.data(), 9);
    append_bits(record, star.points.data(), star.points.size());
    for (const int8_t flag : valid) record.fields.push_back(static_cast<std::uint64_t>(flag));
    append_bits(record, goals, 9);
    rows.push_back(record);
  }
}

std::vector<BitRow> evaluate() {
  std::vector<BitRow> rows;
  midpoint_rows(rows);
  center_rows(rows);
  hull_rows(rows);
  survivor_rows(rows);
  return rows;
}

void emit(const std::vector<BitRow>& rows) {
  std::printf(
      "// Bit vectors of the metric edit goals, written by `test_metric_goal --emit`.\n"
      "// One row per input, in the order the test generates them. A value is its\n"
      "// IEEE 754 binary64 bit pattern; counts and flags are integers.\n"
      "// PHX_MG_MIDPOINT(p, q, nine values of the metric at p, nine at q,\n"
      "//                 parameter, the point at that parameter)\n"
      "// PHX_MG_CENTER(count, nine metric values, count points, valid, center)\n"
      "// PHX_MG_HULL(corner count, three corners, goal, valid, projected goal)\n"
      "//   Only the first `corner count` corners are data.\n"
      "// PHX_MG_SURVIVOR(count, corner count, the metrics of the survivor and of\n"
      "//                 the removed vertex, three corners, count points\n"
      "//                 (survivor, removed vertex, link), three flags, three goals)\n");
  phx::mc::test::emit_rows(rows);
}

void guards_arguments() {
  const double points[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const double metrics[18] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                              1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double goals[9] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
  int8_t valid[3] = {7, 7, 7};
  double parameter = 7.0;
  PHX_CHECK(phx_mc_metric_midpoints(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) ==
            PHX_MC_OK);
  PHX_CHECK(phx_mc_metric_midpoints(1, points, points + 3, metrics, metrics + 9, &parameter,
                                    nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_midpoints(-1, points, points + 3, metrics, metrics + 9, &parameter,
                                    goals) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_survivor_goals(1, points, metrics, 0, nullptr, goals, valid) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_survivor_goals(2, points, metrics, 1, points, goals, valid) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_survivor_goals(2, points, metrics, 3, nullptr, goals, valid) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_survivor_goals(2, points, nullptr, 0, nullptr, goals, valid) ==
            PHX_MC_INVALID_ARGUMENT);
  // A refused call writes nothing.
  for (const double value : goals) PHX_CHECK(value == 7.0);
  PHX_CHECK(parameter == 7.0 && valid[0] == 7 && valid[1] == 7 && valid[2] == 7);
}

// Values that follow from the definitions without rounding.
void exact_cases() {
  const double points[9] = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 5.0, 0.0};
  const double metrics[18] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                              1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double goals[9];
  int8_t valid[3];
  PHX_CHECK(phx_mc_metric_survivor_goals(3, points, metrics, 0, nullptr, goals, valid) ==
            PHX_MC_OK);
  PHX_CHECK(valid[0] == 1 && valid[1] == 1 && valid[2] == 1);
  // Equal metrics: the midpoint of the edge, its quarter point, and the one
  // link vertex as the center of its own ball.
  const double expected[9] = {1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 5.0, 0.0};
  for (std::size_t index = 0; index < 9; ++index) PHX_CHECK(goals[index] == expected[index]);
  const double cube[24] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                           0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double center[3];
  PHX_CHECK(phx::mc::metric::enclosing_center(8, cube, metrics, center));
  for (const double value : center) PHX_CHECK_NEAR(value, 0.5, 0x1p-40);
}

}  // namespace

int main(int argc, char** argv) {
  const phx::mc::GradualNearestScope mode;
  PHX_CHECK(mode.established());
  const std::vector<BitRow> rows = evaluate();
  if (argc > 1 && std::strcmp(argv[1], "--emit") == 0) {
    emit(rows);
    return 0;
  }
  phx::mc::test::compare_rows(rows, kRecorded);
  guards_arguments();
  exact_cases();
  return phx::mc::test::finish("test_metric_goal");
}
