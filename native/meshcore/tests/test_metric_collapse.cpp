//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Positions of a collapse survivor and the search of its moves. Verdicts are
// checked on stars whose lengths and orientations follow by hand (unit
// metric, spokes along an axis). The search is checked against its
// definition: the goals, the carriers of each goal, their tensors and their
// verdicts from the public entries, composed in order.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr double kUpper = 0x1.6a09e667f3bcdp+0;  // fl(sqrt 2)
constexpr double kFloor = 0x1p-22;
constexpr double kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

struct Star {
  std::vector<double> points;
  std::vector<double> metrics;
  std::vector<std::int64_t> spokes;
  std::vector<double> before;
  std::vector<std::int64_t> cells;
  double floor = kFloor;

  std::int64_t point_count() const { return static_cast<std::int64_t>(points.size() / 3); }
  std::int64_t spoke_count() const { return static_cast<std::int64_t>(spokes.size()); }
};

Star unit_star(const std::vector<double>& points) {
  Star star;
  star.points = points;
  for (std::size_t row = 0; row < points.size() / 3; ++row) {
    star.metrics.insert(star.metrics.end(), kIdentity, kIdentity + 9);
  }
  return star;
}

struct Verdicts {
  std::vector<std::int32_t> verdicts;
  std::vector<double> qualities;
  std::vector<double> lengths;
};

Verdicts evaluate(const Star& star, const std::vector<double>& positions,
                  const std::vector<double>& tensors) {
  const std::int64_t count = static_cast<std::int64_t>(positions.size() / 3);
  Verdicts out;
  out.verdicts.assign(static_cast<std::size_t>(count), -1);
  out.qualities.assign(static_cast<std::size_t>(count), -1.0);
  out.lengths.assign(static_cast<std::size_t>(count * star.spoke_count()), -1.0);
  PHX_CHECK(phx_mc_metric_collapse_positions(
                star.point_count(), star.points.data(), star.metrics.data(), star.spoke_count(),
                star.spokes.data(), static_cast<std::int64_t>(star.before.size()),
                star.before.data(), static_cast<std::int64_t>(star.cells.size() / 4),
                star.cells.data(), star.floor, count, positions.data(), tensors.data(),
                out.verdicts.data(), out.qualities.data(), out.lengths.data()) == PHX_MC_OK);
  return out;
}

Verdicts evaluate_unit(const Star& star, const std::vector<double>& positions) {
  std::vector<double> tensors;
  for (std::size_t row = 0; row < positions.size() / 3; ++row) {
    tensors.insert(tensors.end(), kIdentity, kIdentity + 9);
  }
  return evaluate(star, positions, tensors);
}

// Survivor far away, one spoke to the origin, one cell (origin, survivor,
// e_y, e_z) that is positive for a survivor at (x, 0, 0) with x > 0.
Star axis_star() {
  Star star = unit_star({9, 9, 9, 8, 8, 8, 0, 0, 0, 0, 1, 0, 0, 0, 1});
  star.spokes = {2};
  star.cells = {2, 0, 3, 4};
  return star;
}

void band_edge() {
  // Before: one long edge in the first violation quantum. A position
  // improves the objective only if its spoke is in the band, and the
  // decision is sharp at the band edge.
  Star star = axis_star();
  star.before = {kUpper * (1.0 + 0x1p-20)};
  const double inside = 0x1.6a09e667f3bccp+0;
  const double outside = 0x1.6a09e667f3bcep+0;
  const Verdicts out = evaluate_unit(
      star, {1.0, 0, 0, inside, 0, 0, kUpper, 0, 0, outside, 0, 0, kUpper * (1.0 + 0x1p-20), 0,
             0, 3.0, 0, 0, 0.0, 0, 0});
  const std::int32_t expected[7] = {0, 0, 0, 2, 2, 2, 1};
  for (std::size_t row = 0; row < 7; ++row) PHX_CHECK(out.verdicts[row] == expected[row]);
  // Unit metric along an axis: the spoke length is the coordinate.
  PHX_CHECK(out.lengths[0] == 1.0 && out.lengths[1] == inside && out.lengths[2] == kUpper);
  PHX_CHECK(out.lengths[3] == outside && out.lengths[5] == 3.0);
  PHX_CHECK(out.lengths[6] == 0.0);
  for (std::size_t row = 0; row < 3; ++row) PHX_CHECK(out.qualities[row] > 0.0);
}

void violation_order() {
  // Spokes to the origin and to (4, 0, 0): at (3, 0, 0) the lengths are 3 and
  // 1, at (2, 0, 0) they are 2 and 2.
  Star star = unit_star({9, 9, 9, 8, 8, 8, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 0, 0});
  star.spokes = {2, 5};
  star.cells = {2, 0, 3, 4};
  const std::vector<double> positions = {3, 0, 0, 2, 0, 0};
  // A proper prefix is smaller: (q3) against (q3, q2).
  star.before = {3.0, 2.0};
  Verdicts out = evaluate_unit(star, positions);
  PHX_CHECK(out.verdicts[0] == 0 && out.verdicts[1] == 0);
  // Equal long violations and no short one on either side: no decrease.
  // (q2, q2) is smaller than (q3) by its first entry.
  star.before = {3.0};
  out = evaluate_unit(star, positions);
  PHX_CHECK(out.verdicts[0] == 2 && out.verdicts[1] == 0);
  // Equal long violations: fewer short violations decide.
  star.before = {3.0, 0.5};
  out = evaluate_unit(star, positions);
  PHX_CHECK(out.verdicts[0] == 0);
  // The largest violation is compared first: (q2, q2) against (q2).
  star.before = {2.0};
  out = evaluate_unit(star, positions);
  PHX_CHECK(out.verdicts[0] == 2 && out.verdicts[1] == 2);
}

void cells_decide() {
  Star star = axis_star();
  star.before = {3.0};
  const std::vector<double> position = {1, 0, 0};
  PHX_CHECK(evaluate_unit(star, position).verdicts[0] == 0);
  // The same cell with two corners exchanged is negative.
  star.cells = {0, 2, 3, 4};
  PHX_CHECK(evaluate_unit(star, position).verdicts[0] == 3);
  // One good and one flat cell (a corner named twice).
  star.cells = {2, 0, 3, 4, 2, 0, 3, 3};
  PHX_CHECK(evaluate_unit(star, position).verdicts[0] == 3);
  // A floor no cell clears.
  star.cells = {2, 0, 3, 4};
  star.floor = 2.0;
  PHX_CHECK(evaluate_unit(star, position).verdicts[0] == 3);
  star.floor = kFloor;
  // A tensor that is not positive at a corner gives no quality.
  std::vector<double> tensors(kIdentity, kIdentity + 9);
  star.metrics[9 * 3] = -1.0;
  PHX_CHECK(evaluate(star, position, tensors).verdicts[0] == 4);
  star.metrics[9 * 3] = 1.0;
  // The quality is that of the worst cell.
  const double one = evaluate_unit(star, position).qualities[0];
  star.cells = {2, 0, 3, 4, 2, 0, 3, 5};
  star.points.insert(star.points.end(), {0, 0, 9});
  star.metrics.insert(star.metrics.end(), kIdentity, kIdentity + 9);
  const Verdicts two = evaluate_unit(star, position);
  PHX_CHECK(two.verdicts[0] == 0 && two.qualities[0] > 0.0 && two.qualities[0] < one);
}

void unmeasured_and_domain() {
  Star star = axis_star();
  const std::vector<double> positions = {1, 0, 0, 0, 0, 0};
  // A length before the collapse that was not measured: no verdict for a
  // position with spokes to measure, coincidence still first.
  for (const double length : {0.0, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    star.before = {3.0, length};
    const Verdicts out = evaluate_unit(star, positions);
    PHX_CHECK(out.verdicts[0] == 5 && out.verdicts[1] == 1);
  }
  // A spoke of the position that has no length (a tensor that is not a metric).
  star.before = {3.0};
  std::vector<double> tensors(18, 0.0);
  tensors[0] = -1.0;
  tensors[9] = 1.0;
  tensors[13] = 1.0;
  tensors[17] = 1.0;
  Verdicts out = evaluate(star, {1, 0, 0, 1, 0, 0}, tensors);
  PHX_CHECK(out.verdicts[0] == 5 && out.verdicts[1] == 0);
  // A coordinate below the exact predicate domain, reached only by a
  // position that decreases the violations.
  star.points[3 * 4 + 1] = 0x1p-130;
  out = evaluate_unit(star, {1, 0, 0, 3, 0, 0});
  PHX_CHECK(out.verdicts[0] == 6 && out.verdicts[1] == 2);
}

struct Source {
  phx_mc_metric_source* handle = nullptr;
  std::vector<double> points;
  std::vector<double> metrics;
  std::vector<double> logarithms;
  ~Source() { phx_mc_metric_source_free(handle); }
};

// Integer points and the cells of a source; the tensor of point p is
// (1 + p / 8) times the identity.
void create(Source& source, const std::vector<std::int64_t>& integers,
            const std::vector<std::int64_t>& cells) {
  const std::int64_t point_count = static_cast<std::int64_t>(integers.size() / 3);
  std::vector<std::uint32_t> words;
  std::vector<std::int64_t> offsets = {0};
  std::vector<std::int8_t> signs;
  for (const std::int64_t value : integers) {
    source.points.push_back(static_cast<double>(value));
    signs.push_back(value < 0 ? -1 : 1);
    if (value != 0) words.push_back(static_cast<std::uint32_t>(value < 0 ? -value : value));
    offsets.push_back(static_cast<std::int64_t>(words.size()));
  }
  words.push_back(1);
  offsets.push_back(static_cast<std::int64_t>(words.size()));
  for (std::int64_t point = 0; point < point_count; ++point) {
    for (std::size_t entry = 0; entry < 9; ++entry) {
      source.metrics.push_back(kIdentity[entry] * (1.0 + static_cast<double>(point) / 8.0));
    }
  }
  source.logarithms.resize(source.metrics.size());
  std::vector<std::int8_t> valid(static_cast<std::size_t>(point_count));
  PHX_CHECK(phx_mc_metric_logarithms(point_count, source.metrics.data(),
                                     source.logarithms.data(), valid.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_metric_source_create(
                point_count, source.points.data(), words.data(), offsets.data(), signs.data(),
                static_cast<std::int64_t>(cells.size() / 4), cells.data(), source.metrics.data(),
                source.logarithms.data(), &source.handle) == PHX_MC_OK);
}

struct Entry {
  std::int32_t goal = -1;
  std::int64_t rows[4] = {-1, -1, -1, -1};
  std::int64_t weights[4] = {0, 0, 0, 0};
  std::int32_t bits = 0;
  double point[3] = {0, 0, 0};
  double metric[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  double quality = 0.0;
  std::vector<double> lengths;
};

struct Search {
  std::int32_t end = -1;
  std::vector<Entry> entries;
};

constexpr std::int64_t kCapacity = 159;

Search search(const Source& source, const Star& star, const std::vector<std::int64_t>& simplices,
              const std::vector<std::int8_t>& declared, bool every_resolution) {
  const std::size_t room = static_cast<std::size_t>(kCapacity);
  const std::size_t spokes = static_cast<std::size_t>(star.spoke_count());
  std::vector<std::int32_t> goals(room), bits(room);
  std::vector<std::int64_t> rows(4 * room), weights(4 * room);
  std::vector<double> points(3 * room), metrics(9 * room), qualities(room),
      lengths(room * spokes);
  std::int64_t count = -1;
  Search out;
  PHX_CHECK(phx_mc_metric_source_survivor_moves(
                source.handle, star.point_count(), star.points.data(), star.metrics.data(),
                star.spoke_count(), star.spokes.data(),
                static_cast<std::int64_t>(star.before.size()), star.before.data(),
                static_cast<std::int64_t>(star.cells.size() / 4), star.cells.data(), star.floor,
                static_cast<std::int64_t>(declared.size()), simplices.data(), declared.data(),
                every_resolution ? 1 : 0, kCapacity, goals.data(), rows.data(), weights.data(),
                bits.data(), points.data(), metrics.data(), qualities.data(), lengths.data(),
                &count, &out.end) == PHX_MC_OK);
  for (std::size_t index = 0; index < static_cast<std::size_t>(count); ++index) {
    Entry entry;
    entry.goal = goals[index];
    entry.bits = bits[index];
    entry.quality = qualities[index];
    for (std::size_t slot = 0; slot < 4; ++slot) {
      entry.rows[slot] = rows[4 * index + slot];
      entry.weights[slot] = weights[4 * index + slot];
    }
    for (std::size_t axis = 0; axis < 3; ++axis) entry.point[axis] = points[3 * index + axis];
    for (std::size_t slot = 0; slot < 9; ++slot) entry.metric[slot] = metrics[9 * index + slot];
    entry.lengths.assign(lengths.begin() + static_cast<std::ptrdiff_t>(index * spokes),
                         lengths.begin() + static_cast<std::ptrdiff_t>((index + 1) * spokes));
    out.entries.push_back(entry);
  }
  return out;
}

bool same3(const double* a, const double* b) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

// The search by its definition, from the public entries.
Search composed(const Source& source, const Star& star,
                const std::vector<std::int64_t>& simplices,
                const std::vector<std::int8_t>& declared, bool every_resolution) {
  Search out;
  out.end = 0;
  const std::int64_t source_count = static_cast<std::int64_t>(source.points.size() / 3);
  double corners[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::int32_t corner_count = 0;
  if (!declared.empty()) {
    for (std::size_t slot = 0; slot < 3 && simplices[slot] >= 0; ++slot) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        corners[3 * slot + axis] =
            source.points[3 * static_cast<std::size_t>(simplices[slot]) + axis];
      }
      ++corner_count;
    }
  }
  double goals[9];
  std::int8_t valid[3];
  PHX_CHECK(phx_mc_metric_survivor_goals(star.point_count(), star.points.data(),
                                         star.metrics.data(), corner_count,
                                         corner_count == 0 ? nullptr : corners, goals,
                                         valid) == PHX_MC_OK);
  const std::int64_t limit = every_resolution ? 53 : 1;
  for (std::int32_t goal = 0; goal < 3; ++goal) {
    if (valid[goal] == 0) {
      out.end = 2;
      return out;
    }
    std::int64_t rows[4 * 53], weights[4 * 53], count = 0;
    std::int32_t bits[53];
    double points[3 * 53];
    std::int8_t exact[53];
    PHX_CHECK(phx_mc_metric_source_carriers(
                  source.handle, goals + 3 * goal, star.points.data(),
                  static_cast<std::int64_t>(declared.size()), simplices.data(), declared.data(),
                  0, limit, rows, weights, bits, points, exact, &count) == PHX_MC_OK);
    for (std::int64_t index = 0; index < count; ++index) {
      Entry entry;
      entry.goal = goal;
      entry.bits = bits[index];
      std::int64_t support[4];
      double coefficients[4];
      std::int64_t size = 0;
      for (std::size_t slot = 0; slot < 4; ++slot) {
        entry.rows[slot] = rows[4 * index + static_cast<std::int64_t>(slot)];
        entry.weights[slot] = weights[4 * index + static_cast<std::int64_t>(slot)];
        if (entry.rows[slot] < 0) continue;
        support[size] = entry.rows[slot];
        coefficients[size] = static_cast<double>(entry.weights[slot]) /
                             static_cast<double>(std::int64_t{1} << entry.bits);
        ++size;
      }
      for (std::size_t axis = 0; axis < 3; ++axis) {
        entry.point[axis] = points[3 * index + static_cast<std::int64_t>(axis)];
      }
      if (same3(entry.point, star.points.data()) || same3(entry.point, star.points.data() + 3)) {
        continue;
      }
      const std::int64_t offsets[2] = {0, size};
      std::int8_t has_metric = 0;
      PHX_CHECK(phx_mc_metric_interpolate(1, offsets, support, coefficients, source_count,
                                          source.metrics.data(), source.logarithms.data(),
                                          entry.metric, &has_metric) == PHX_MC_OK);
      if (has_metric == 0) continue;
      const Verdicts verdict =
          evaluate(star, {entry.point[0], entry.point[1], entry.point[2]},
                   std::vector<double>(entry.metric, entry.metric + 9));
      entry.lengths = verdict.lengths;
      if (verdict.verdicts[0] == 0) {
        entry.quality = verdict.qualities[0];
        out.entries.push_back(entry);
        bool long_spoke = false;
        for (const double length : entry.lengths) long_spoke = long_spoke || length > kUpper;
        if (!long_spoke) {
          out.end = 1;
          return out;
        }
      } else if (verdict.verdicts[0] == 5 || verdict.verdicts[0] == 6) {
        out.entries.push_back(entry);
        out.end = verdict.verdicts[0] == 5 ? 3 : 4;
        return out;
      }
    }
  }
  return out;
}

void check_same(const Search& found, const Search& expected) {
  PHX_CHECK(found.end == expected.end);
  PHX_CHECK(found.entries.size() == expected.entries.size());
  if (found.entries.size() != expected.entries.size()) return;
  for (std::size_t index = 0; index < found.entries.size(); ++index) {
    const Entry& a = found.entries[index];
    const Entry& b = expected.entries[index];
    PHX_CHECK(a.goal == b.goal && a.bits == b.bits && same3(a.point, b.point));
    for (std::size_t slot = 0; slot < 4; ++slot) {
      PHX_CHECK(a.rows[slot] == b.rows[slot] && a.weights[slot] == b.weights[slot]);
    }
    for (std::size_t slot = 0; slot < 9; ++slot) PHX_CHECK(a.metric[slot] == b.metric[slot]);
    // Quality and lengths of an entry that is not a move are not specified.
    const bool move = index + 1 < found.entries.size() || found.end < 3;
    if (!move) continue;
    PHX_CHECK(a.quality == b.quality);
    PHX_CHECK(a.lengths == b.lengths);
  }
}

// One source cell (0, 8 e_x, 8 e_y, 8 e_z). The survivor at (1, 1, 1) and the
// removed vertex at (2, 1, 1); link vertices above, behind and ahead.
Star interior_star() {
  Star star = unit_star({1, 1, 1, 2, 1, 1, 1, 2.5, 1, 1, 1, 2.5, 0.25, 0.5, 0.5, 2.5, 1, 1});
  star.spokes = {2, 3, 5};
  star.cells = {0, 2, 3, 5};
  return star;
}

void search_moves() {
  Source source;
  create(source, {0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8}, {0, 1, 2, 3});

  // Long edges before; every resolution of every goal.
  Star star = interior_star();
  star.before = {3.0, 1.5, 1.5, 1.5};
  Search found = search(source, star, {}, {}, true);
  check_same(found, composed(source, star, {}, {}, true));
  PHX_CHECK(!found.entries.empty());
  for (const Entry& entry : found.entries) {
    // A move is an exact dyadic combination of source points.
    std::int64_t total = 0;
    for (const std::int64_t weight : entry.weights) total += weight;
    PHX_CHECK(total == (std::int64_t{1} << entry.bits));
    PHX_CHECK(entry.quality > 0.0);
  }

  // The coarsest carrier of each goal only.
  found = search(source, star, {}, {}, false);
  check_same(found, composed(source, star, {}, {}, false));
  PHX_CHECK(found.entries.size() <= 3);

  // One spoke, ahead of the survivor: the first move that brings it into the
  // band ends the search.
  star.spokes = {5};
  star.before = {1.5};
  found = search(source, star, {}, {}, true);
  check_same(found, composed(source, star, {}, {}, true));
  PHX_CHECK(found.end == 1 && !found.entries.empty());
  PHX_CHECK(found.entries.back().lengths[0] <= kUpper);

  // No link vertex: the third goal does not exist.
  Star pair = unit_star({1, 1, 1, 2, 1, 1, 1, 2.5, 1, 1, 1, 2.5, 2.5, 1, 1});
  pair.points.resize(6);
  pair.metrics.resize(18);
  pair.cells = {0, 1, 1, 1};
  pair.before = {3.0};
  found = search(source, pair, {}, {}, true);
  PHX_CHECK(found.end == 2);
  check_same(found, composed(source, pair, {}, {}, true));

  // A length before that was not measured: the first carrier is referred.
  star = interior_star();
  star.before = {0.0};
  found = search(source, star, {}, {}, true);
  PHX_CHECK(found.end == 3 && found.entries.size() == 1);
  check_same(found, composed(source, star, {}, {}, true));

  // A cell coordinate outside the predicate domain.
  star.before = {3.0, 1.5, 1.5, 1.5};
  star.points[3 * 2 + 2] = 0x1p-130;
  found = search(source, star, {}, {}, true);
  PHX_CHECK(found.end == 4 && !found.entries.empty());
  check_same(found, composed(source, star, {}, {}, true));
}

void search_on_simplices() {
  // The survivor on the face x = 0 of the source cell: it moves in the
  // triangle (0, 8 e_y, 8 e_z) or along its edge from the origin to 8 e_y.
  Source source;
  create(source, {0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8}, {0, 1, 2, 3});
  Star star = unit_star({0, 1, 1, 0, 2, 1, 0, 1, 2.5, 1.5, 1, 1, 0, 2.5, 1, 0, 0.25, 0.5});
  star.spokes = {2, 3, 4};
  // (survivor, link ahead, link above, link inward) is positive.
  star.cells = {0, 4, 2, 3};
  star.before = {3.0, 1.5, 1.5, 1.5};
  const std::vector<std::int64_t> triangle = {0, 2, 3};
  for (const std::int8_t flag : {std::int8_t{0}, std::int8_t{1}}) {
    const Search found = search(source, star, triangle, {flag}, false);
    check_same(found, composed(source, star, triangle, {flag}, false));
    for (const Entry& entry : found.entries) PHX_CHECK(entry.point[0] == 0.0);
  }
  const Search all = search(source, star, triangle, {1}, true);
  check_same(all, composed(source, star, triangle, {1}, true));
  PHX_CHECK(!all.entries.empty());

  // Along an edge: the survivor on the y axis.
  Star edge_star = unit_star({0, 1, 0, 0, 2, 0, 1, 1, 1.5, 1.5, 1, 0, 0, 2.5, 0, 0, 1, 1.5});
  edge_star.spokes = {2, 3, 4};
  edge_star.cells = {0, 3, 4, 5};
  edge_star.before = {3.0, 1.5, 1.5, 1.5};
  const std::vector<std::int64_t> edge = {0, 2, -1};
  const Search along = search(source, edge_star, edge, {0}, true);
  check_same(along, composed(source, edge_star, edge, {0}, true));
  for (const Entry& entry : along.entries) {
    PHX_CHECK(entry.point[0] == 0.0 && entry.point[2] == 0.0);
  }
}

void guards_arguments() {
  Source source;
  create(source, {0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8}, {0, 1, 2, 3});
  Star star = interior_star();
  star.before = {3.0};
  std::int32_t verdict = 7;
  double quality = 7.0;
  double lengths[3] = {7, 7, 7};
  const double position[3] = {1.5, 1, 1};
  const auto positions = [&](const Star& s) {
    return phx_mc_metric_collapse_positions(
        s.point_count(), s.points.data(), s.metrics.data(), s.spoke_count(), s.spokes.data(),
        static_cast<std::int64_t>(s.before.size()), s.before.data(),
        static_cast<std::int64_t>(s.cells.size() / 4), s.cells.data(), s.floor, 1, position,
        kIdentity, &verdict, &quality, lengths);
  };
  PHX_CHECK(positions(star) == PHX_MC_OK && verdict == 0);
  verdict = 7;
  quality = 7.0;
  lengths[0] = 7.0;
  Star bad = star;
  bad.spokes[0] = 0;  // a spoke to the survivor itself
  PHX_CHECK(positions(bad) == PHX_MC_INVALID_INPUT);
  bad = star;
  bad.spokes[0] = 6;
  PHX_CHECK(positions(bad) == PHX_MC_INVALID_INPUT);
  bad = star;
  bad.cells[3] = 6;
  PHX_CHECK(positions(bad) == PHX_MC_INVALID_INPUT);
  bad = star;
  bad.cells.clear();  // no cell: no quality to report
  PHX_CHECK(positions(bad) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(verdict == 7 && quality == 7.0 && lengths[0] == 7.0);

  const std::size_t room = static_cast<std::size_t>(kCapacity);
  std::vector<std::int32_t> goals(room), bits(room);
  std::vector<std::int64_t> rows(4 * room), weights(4 * room);
  std::vector<double> points(3 * room), metrics(9 * room), qualities(room), spoke(3 * room);
  std::int64_t count = 7;
  std::int32_t end = 7;
  const auto moves = [&](const phx_mc_metric_source* handle, const Star& s,
                         std::int64_t simplex_count, const std::int64_t* simplex_rows,
                         const std::int8_t* declared, std::int64_t capacity) {
    return phx_mc_metric_source_survivor_moves(
        handle, s.point_count(), s.points.data(), s.metrics.data(), s.spoke_count(),
        s.spokes.data(), static_cast<std::int64_t>(s.before.size()), s.before.data(),
        static_cast<std::int64_t>(s.cells.size() / 4), s.cells.data(), s.floor, simplex_count,
        simplex_rows, declared, 1, capacity, goals.data(), rows.data(), weights.data(),
        bits.data(), points.data(), metrics.data(), qualities.data(), spoke.data(), &count,
        &end);
  };
  PHX_CHECK(moves(nullptr, star, 0, nullptr, nullptr, kCapacity) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(moves(source.handle, star, 0, nullptr, nullptr, kCapacity - 1) ==
            PHX_MC_INVALID_ARGUMENT);
  const std::int64_t vertex[3] = {0, -1, -1};
  const std::int64_t absent[3] = {0, 9, -1};
  const std::int8_t flag = 0;
  PHX_CHECK(moves(source.handle, star, 1, vertex, &flag, kCapacity) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(moves(source.handle, star, 1, absent, &flag, kCapacity) == PHX_MC_INVALID_INPUT);
  Star lone = star;
  lone.points.resize(3);
  lone.metrics.resize(9);
  lone.spokes.clear();
  lone.cells = {0, 0, 0, 0};
  PHX_CHECK(moves(source.handle, lone, 0, nullptr, nullptr, kCapacity) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(count == 7 && end == 7);
}

}  // namespace

int main() {
  band_edge();
  violation_order();
  cells_decide();
  unmeasured_and_domain();
  search_moves();
  search_on_simplices();
  guards_arguments();
  return phx::mc::test::finish("test_metric_collapse");
}
