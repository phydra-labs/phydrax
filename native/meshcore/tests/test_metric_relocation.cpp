//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Candidate positions of a relocated vertex through the C boundary: every
// candidate is a carrier of the source whose recorded lengths, quality and
// verdict the separate measurement entries reproduce; the sequence is finite
// and the same for two searches of one star; the modes order the carriers of
// a goal as stated; malformed arguments and a star without a measured length
// are refused. The composition against its definition is checked from the
// host (tests/unit/geometry/test_metric_relocation.py).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "check.hpp"
#include "metric_float.hpp"
#include "metric_relocation.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr int64_t kRoom = 54;
constexpr double kFloor = 0x1p-22;
constexpr double kImprovement = 1.0e-6;

struct Source {
  phx_mc_metric_source* handle = nullptr;
  std::vector<double> points;
  ~Source() { phx_mc_metric_source_free(handle); }
};

// The cell with corners (0, 0, 0), (9, 0, 0), (0, 10, 0), (0, 0, 11) over 8,
// every tensor `scale` times the identity.
void create(Source& source, double scale) {
  const int64_t numerators[12] = {0, 0, 0, 9, 0, 0, 0, 10, 0, 0, 0, 11};
  std::vector<uint32_t> words;
  std::vector<int64_t> offsets = {0};
  std::vector<int8_t> signs;
  for (const int64_t value : numerators) {
    source.points.push_back(static_cast<double>(value) / 8.0);
    signs.push_back(1);
    if (value != 0) words.push_back(static_cast<uint32_t>(value));
    offsets.push_back(static_cast<int64_t>(words.size()));
  }
  words.push_back(8);
  offsets.push_back(static_cast<int64_t>(words.size()));
  std::vector<double> tensors(36, 0.0);
  std::vector<double> logarithms(36, 0.0);
  for (std::size_t point = 0; point < 4; ++point) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      tensors[9 * point + 4 * axis] = scale;
    }
  }
  int8_t valid[4] = {0, 0, 0, 0};
  PHX_CHECK(phx_mc_metric_logarithms(4, tensors.data(), logarithms.data(), valid) == PHX_MC_OK);
  const int64_t cells[4] = {0, 1, 2, 3};
  PHX_CHECK(phx_mc_metric_source_create(4, source.points.data(), words.data(), offsets.data(),
                                        signs.data(), 1, cells, tensors.data(),
                                        logarithms.data(), &source.handle) == PHX_MC_OK);
}

// The star of a vertex inside the cell: its four corners as neighbors.
struct Star {
  std::vector<double> points;
  std::vector<double> metrics;
  std::vector<int8_t> measured = {1, 1, 1, 1};
  std::vector<double> lengths = std::vector<double>(4);
  std::vector<int64_t> cells = {0, 2, 3, 4, 1, 0, 3, 4, 1, 2, 0, 4, 1, 2, 3, 0};
  double quality = 0.0;
};

void measure(const std::vector<double>& points, const std::vector<double>& metrics,
             const std::vector<int64_t>& cells, double* lengths, double& quality) {
  const int64_t spokes[8] = {0, 1, 0, 2, 0, 3, 0, 4};
  PHX_CHECK(phx_mc_metric_lengths(4, spokes, 5, points.data(), metrics.data(), lengths) ==
            PHX_MC_OK);
  double qualities[4];
  PHX_CHECK(phx_mc_metric_tetra_quality(4, cells.data(), 5, points.data(), metrics.data(),
                                        qualities) == PHX_MC_OK);
  quality = qualities[0];
  for (const double value : qualities) quality = value < quality ? value : quality;
}

Star star_of(const Source& source, double scale, double x, double y, double z) {
  Star star;
  star.points = {x, y, z};
  star.points.insert(star.points.end(), source.points.begin(), source.points.end());
  star.metrics.assign(45, 0.0);
  for (std::size_t point = 0; point < 5; ++point) {
    for (std::size_t axis = 0; axis < 3; ++axis) star.metrics[9 * point + 4 * axis] = scale;
  }
  measure(star.points, star.metrics, star.cells, star.lengths.data(), star.quality);
  return star;
}

struct Group {
  int32_t end = -1;
  int64_t count = -1;
  std::vector<int64_t> rows = std::vector<int64_t>(4 * kRoom);
  std::vector<int64_t> weights = std::vector<int64_t>(4 * kRoom);
  std::vector<int32_t> bits = std::vector<int32_t>(kRoom);
  std::vector<double> points = std::vector<double>(3 * kRoom);
  std::vector<double> metrics = std::vector<double>(9 * kRoom);
  std::vector<double> qualities = std::vector<double>(kRoom);
  std::vector<double> lengths = std::vector<double>(4 * kRoom);
};

struct Search {
  phx_mc_metric_relocation* handle = nullptr;
  ~Search() { phx_mc_metric_relocation_free(handle); }
  Group next() {
    Group group;
    PHX_CHECK(phx_mc_metric_relocation_next(handle, kRoom, group.rows.data(),
                                            group.weights.data(), group.bits.data(),
                                            group.points.data(), group.metrics.data(),
                                            group.qualities.data(), group.lengths.data(),
                                            &group.count, &group.end) == PHX_MC_OK);
    return group;
  }
};

int32_t open(Search& search, const Source& source, const Star& star,
             const std::vector<double>& pulls, int32_t flags) {
  return phx_mc_metric_relocation_create(
      source.handle, 5, star.points.data(), star.metrics.data(), star.measured.data(),
      star.lengths.data(), star.quality, 4, star.cells.data(), kFloor, kImprovement,
      static_cast<int64_t>(pulls.size() / 3), pulls.data(), nullptr, nullptr, 0, nullptr,
      nullptr, flags, &search.handle);
}

const std::vector<double> kPulls = {0.25,  0.0,  0.0,   0.0,  0.25, 0.0,  0.0,  0.0,
                                    0.25,  -0.1, -0.1,  -0.1, 0.2,  0.15, 0.1,  0.25,
                                    0.0,   0.0,  0.125, 0.25, 0.3};

// Every candidate of every group is what the measurement entries say of the
// star with the vertex at the carrier, and is a carrier of the source.
std::int64_t drain_and_check(const Source& source, const Star& star, int32_t flags,
                             std::vector<double>* sequence = nullptr) {
  Search search;
  PHX_CHECK(open(search, source, star, kPulls, flags) == PHX_MC_OK);
  std::int64_t candidates = 0;
  for (int guard = 0; guard < 100000; ++guard) {
    const Group group = search.next();
    if (group.end == 1) {
      PHX_CHECK(group.count == 0);
      // Exhausted stays exhausted.
      PHX_CHECK(search.next().end == 1);
      return candidates;
    }
    PHX_CHECK(group.end == 0 && group.count >= 1 && group.count <= kRoom);
    for (std::int64_t index = 0; index < group.count; ++index) {
      const std::size_t at = static_cast<std::size_t>(index);
      // Weights over distinct ascending rows that sum to 2^bits.
      std::int64_t total = 0;
      for (std::size_t slot = 0; slot < 4; ++slot) {
        const std::int64_t row = group.rows[4 * at + slot];
        if (row < 0) continue;
        PHX_CHECK(group.weights[4 * at + slot] > 0);
        PHX_CHECK(slot == 0 || group.rows[4 * at + slot - 1] < row);
        total += group.weights[4 * at + slot];
      }
      PHX_CHECK(total == (std::int64_t{1} << group.bits[at]));
      // The constant field leaves the tensor as it is.
      for (std::size_t entry = 0; entry < 9; ++entry) {
        PHX_CHECK(group.metrics[9 * at + entry] == star.metrics[entry]);
      }
      std::vector<double> moved = star.points;
      for (std::size_t axis = 0; axis < 3; ++axis) moved[axis] = group.points[3 * at + axis];
      double lengths[4];
      double quality = 0.0;
      measure(moved, star.metrics, star.cells, lengths, quality);
      for (std::size_t spoke = 0; spoke < 4; ++spoke) {
        PHX_CHECK(group.lengths[4 * at + spoke] == lengths[spoke]);
      }
      PHX_CHECK(group.qualities[at] == quality);
      if (sequence != nullptr) {
        sequence->insert(sequence->end(), moved.begin(), moved.begin() + 3);
      }
      ++candidates;
    }
  }
  PHX_CHECK(false);
  return candidates;
}

void sequences() {
  Source source;
  create(source, 4.0);
  // Off the center of the cell: with tensor 4 I the spokes are out of the
  // band on both sides.
  const Star star = star_of(source, 4.0, 0.2, 0.3, 0.25);
  for (const int32_t flags : {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3}) {
    std::vector<double> first;
    std::vector<double> second;
    const std::int64_t candidates = drain_and_check(source, star, flags, &first);
    PHX_CHECK(drain_and_check(source, star, flags, &second) == candidates);
    PHX_CHECK(first == second);
    // A refining search revisits its goals; a seed search has at least one.
    if (flags == 8) PHX_CHECK(candidates >= 1);
    if (flags == 10) PHX_CHECK(candidates > 3);
  }
  // No candidate repeats a position within one search of any mode.
  std::vector<double> positions;
  drain_and_check(source, star, 11, &positions);
  for (std::size_t a = 0; a < positions.size(); a += 3) {
    for (std::size_t b = a + 3; b < positions.size(); b += 3) {
      PHX_CHECK(positions[a] != positions[b] || positions[a + 1] != positions[b + 1] ||
                positions[a + 2] != positions[b + 2]);
    }
  }
}

void reversed_spoke_is_not_a_band_improvement() {
  namespace metric = phx::mc::metric;
  using phx::mc::Dyadic;
  constexpr double h = 0x1.b7f92f0969ff9p-1;
  const double x = std::nextafter(h, 1.0);
  std::vector<double> source_points{h, h, h, 0, 0, 0, 0, 1, 0, 1, 0, 0};
  std::vector<Dyadic> numerators;
  for (double point : source_points) numerators.push_back(Dyadic(point).scaled_power(53));
  const metric::CarrierSource source(source_points, std::move(numerators),
                                     Dyadic(1.0).scaled_power(53), {0, 1, 2, 3});
  auto points = source_points;
  points[0] = x;
  const std::array<double, 9> moving{1.25, 0, 0, 0, 1.25, 0, 0, 0, 1.25};
  const std::array<double, 9> neighbor{0.64, 0, 0, 0, 0.64, 0, 0, 0, 0.64};
  std::array<double, 36> source_metrics{};
  std::array<double, 36> source_logs{};
  for (std::size_t row = 0; row < 4; ++row) {
    std::copy(moving.begin(), moving.end(), source_metrics.data() + 9 * row);
    PHX_CHECK(metric::logarithm(moving.data(), source_logs.data() + 9 * row));
  }
  auto metrics = source_metrics;
  std::copy(neighbor.begin(), neighbor.end(), metrics.data() + 9);
  const double before = metric::length(points.data() + 3, points.data(), neighbor.data(),
                                        moving.data());
  const double after = metric::length(points.data() + 3, source_points.data(), neighbor.data(),
                                       moving.data());
  const double reversed = metric::length(source_points.data(), points.data() + 3, moving.data(),
                                          neighbor.data());
  PHX_CHECK(before > metric::kUpperLength && after > metric::kUpperLength &&
            reversed <= metric::kUpperLength);
  const auto carriers = source.carriers(source_points.data(), points.data(), {}, -1,
                                       metric::CarrierOrder::kCoarseFirst, 1);
  PHX_CHECK(carriers.size() == 1);
  const std::array<double, 3> goal{h, h, h};
  PHX_CHECK(!carriers.empty() && carriers.front().point == goal);
  double determinant = 0.0, relative = 0.0;
  metric::relative_determinant(source_points.data(), determinant, relative);
  PHX_CHECK(determinant > 0.0 && relative > kFloor);
  double before_long = 0, before_short = 0, after_long = 0, after_short = 0;
  metric::band_quanta(before, before_long, before_short);
  metric::band_quanta(after, after_long, after_short);
  PHX_CHECK(before_long == after_long && before_short == after_short);

  const std::int8_t measured[3]{1, 0, 0};
  const std::uint8_t neighbor_first[3]{1, 0, 0};
  const double lengths[3]{before, 1.0, 1.0};
  const std::int64_t cells[4]{0, 1, 2, 3};
  const double pull[3]{h - x, 0, 0};
  metric::RelocationStar star;
  star.point_count = 4;
  star.points = points.data();
  star.metrics = metrics.data();
  star.measured = measured;
  star.before_lengths = lengths;
  star.neighbor_first = neighbor_first;
  star.cell_count = 1;
  star.cells = cells;
  star.determinant_floor = kFloor;
  metric::RelocationMode mode;
  mode.band_only = true;
  metric::RelocationSearch search(source, source_metrics.data(), source_logs.data(), star,
                                  1, pull, nullptr, nullptr, {}, mode);
  std::vector<metric::RelocationCandidate> group;
  std::vector<double> candidate_lengths;
  PHX_CHECK(search.next(group, candidate_lengths) == metric::RelocationEnd::kExhausted);
}

void shape_relocation_keeps_the_requested_lower_bound() {
  namespace metric = phx::mc::metric;
  using phx::mc::Dyadic;
  constexpr double old_height = 17.0 / 16.0;
  constexpr double goal_height = 13.0 / 16.0;
  constexpr double side = 7.0 / 8.0;
  const std::vector<double> points{0, 0, old_height, 0, 0, 0, 0, side, 0, side, 0, 0};
  auto source_points = points;
  source_points[2] = goal_height;
  std::vector<Dyadic> numerators;
  for (double coordinate : source_points) {
    numerators.push_back(Dyadic(coordinate).scaled_power(4));
  }
  const metric::CarrierSource source(source_points, std::move(numerators), Dyadic(16.0),
                                     {0, 1, 2, 3});
  std::array<double, 36> tensors{}, logs{};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t axis = 0; axis < 3; ++axis) tensors[9 * row + 4 * axis] = 1.0;
  }
  const std::int8_t measured[]{1, 1, 1};
  const auto diagonal = std::hypot(side, old_height);
  const double lengths[]{old_height, diagonal, diagonal};
  const std::int64_t cells[]{0, 1, 2, 3};
  const double pull[]{0, 0, goal_height - old_height};
  metric::RelocationStar star;
  star.point_count = 4;
  star.points = points.data();
  star.metrics = tensors.data();
  star.measured = measured;
  star.before_lengths = lengths;
  star.before_quality = metric::tetra_quality(points.data(), tensors.data());
  star.cell_count = 1;
  star.cells = cells;
  star.determinant_floor = kFloor;
  star.improvement = kImprovement;
  for (const double lower : {metric::kLowerLength, 0.9}) {
    metric::RelocationMode mode;
    mode.shape_lower = lower;
    metric::RelocationSearch search(source, tensors.data(), logs.data(), star, 1, pull,
                                    nullptr, nullptr, {}, mode);
    std::vector<metric::RelocationCandidate> group;
    std::vector<double> candidate_lengths;
    const auto verdict = search.next(group, candidate_lengths);
    if (lower > goal_height) {
      PHX_CHECK(verdict == metric::RelocationEnd::kExhausted);
    } else {
      PHX_CHECK(verdict == metric::RelocationEnd::kGroup && group.size() == 1);
      PHX_CHECK(!group.empty() && group.front().carrier.point[2] == goal_height);
    }
  }
}

void exhaustive_steps_reach_a_goal_beyond_the_coarse_prefix() {
  namespace metric = phx::mc::metric;
  using phx::mc::Dyadic;
  const std::vector<double> source_points{0, 0, 1, 0, 0, 0, 0, 2, 0, 2, 0, 0};
  std::vector<Dyadic> numerators;
  for (double coordinate : source_points) numerators.emplace_back(coordinate);
  const metric::CarrierSource source(source_points, std::move(numerators), Dyadic(1.0),
                                     {0, 1, 2, 3});
  auto points = source_points;
  points[2] = 1.0 / 8.0;
  std::array<double, 36> tensors{}, logs{};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t axis = 0; axis < 3; ++axis) tensors[9 * row + 4 * axis] = 1.0;
  }
  const std::int8_t measured[]{0, 1, 1};
  const double length = metric::length(points.data(), points.data() + 6, tensors.data(),
                                        tensors.data() + 18);
  const double lengths[]{points[2], length, length};
  const std::int64_t cells[]{0, 1, 2, 3};
  const double pull[]{0, 0, -8};
  metric::RelocationStar star;
  star.point_count = 4;
  star.points = points.data();
  star.metrics = tensors.data();
  star.measured = measured;
  star.before_lengths = lengths;
  star.cell_count = 1;
  star.cells = cells;
  star.determinant_floor = kFloor;

  // Every coarse goal crosses the wall. An exhaustive step can improve the
  // two long spokes without crossing it; exhaustion of the prefix is local.
  metric::RelocationMode mode;
  mode.deep = true;
  mode.band_only = true;
  mode.coarse_steps = true;
  std::vector<metric::RelocationCandidate> group;
  std::vector<double> candidate_lengths;
  metric::RelocationSearch coarse(source, tensors.data(), logs.data(), star, 1, pull, nullptr,
                                  nullptr, {}, mode);
  PHX_CHECK(coarse.next(group, candidate_lengths) == metric::RelocationEnd::kExhausted);
  mode.coarse_steps = false;
  metric::RelocationSearch full(source, tensors.data(), logs.data(), star, 1, pull, nullptr,
                                nullptr, {}, mode);
  PHX_CHECK(full.next(group, candidate_lengths) == metric::RelocationEnd::kGroup);
  PHX_CHECK(!group.empty() && group.front().carrier.point[2] > 0.0 &&
            group.front().carrier.point[2] < points[2]);
  PHX_CHECK(candidate_lengths.size() >= 3 && candidate_lengths[1] < length &&
            candidate_lengths[2] < length);
}

void refuses() {
  Source source;
  create(source, 4.0);
  Star star = star_of(source, 4.0, 0.2, 0.3, 0.25);
  Search search;
  PHX_CHECK(open(search, source, star, kPulls, 16) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(open(search, source, star, kPulls, -1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(search.handle == nullptr);
  PHX_CHECK(phx_mc_metric_relocation_create(nullptr, 5, star.points.data(), star.metrics.data(),
                                            star.measured.data(), star.lengths.data(),
                                            star.quality, 4, star.cells.data(), kFloor,
                                            kImprovement, 0, nullptr, nullptr, nullptr, 0,
                                            nullptr, nullptr, 0,
                                            &search.handle) == PHX_MC_INVALID_ARGUMENT);
  // A frame is an origin and a projector together.
  const double origin[3] = {0.0, 0.0, 0.0};
  PHX_CHECK(phx_mc_metric_relocation_create(source.handle, 5, star.points.data(),
                                            star.metrics.data(), star.measured.data(),
                                            star.lengths.data(), star.quality, 4,
                                            star.cells.data(), kFloor, kImprovement, 0, nullptr,
                                            origin, nullptr, 0, nullptr, nullptr, 0,
                                            &search.handle) == PHX_MC_INVALID_ARGUMENT);
  // A cell names an absent star row; a measured spoke has no finite length.
  Star absent = star;
  absent.cells[3] = 5;
  PHX_CHECK(open(search, source, absent, kPulls, 0) == PHX_MC_INVALID_INPUT);
  Star unmeasured = star;
  unmeasured.lengths[2] = std::numeric_limits<double>::infinity();
  PHX_CHECK(open(search, source, unmeasured, kPulls, 0) == PHX_MC_INVALID_INPUT);
  unmeasured.measured[2] = 0;
  PHX_CHECK(open(search, source, unmeasured, kPulls, 0) == PHX_MC_OK);
  // Too little room for the carriers of one goal.
  Group group;
  PHX_CHECK(phx_mc_metric_relocation_next(search.handle, kRoom - 1, group.rows.data(),
                                          group.weights.data(), group.bits.data(),
                                          group.points.data(), group.metrics.data(),
                                          group.qualities.data(), group.lengths.data(),
                                          &group.count, &group.end) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_relocation_next(nullptr, kRoom, group.rows.data(),
                                          group.weights.data(), group.bits.data(),
                                          group.points.data(), group.metrics.data(),
                                          group.qualities.data(), group.lengths.data(),
                                          &group.count, &group.end) == PHX_MC_INVALID_ARGUMENT);
  phx_mc_metric_relocation_free(nullptr);
}

}  // namespace

int main() {
  sequences();
  reversed_spoke_is_not_a_band_improvement();
  shape_relocation_keeps_the_requested_lower_bound();
  exhaustive_steps_reach_a_goal_beyond_the_coarse_prefix();
  refuses();
  return phx::mc::test::finish("test_metric_relocation");
}
