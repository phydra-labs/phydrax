//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact carriers of a goal in a tetrahedral source: cases whose weights and
// images follow from the definition by hand, the invariants of every carrier
// (positive integer weights that sum to 2^bits, an image closer to the goal
// than the current point), the three resolution orders and the argument
// guards. The search is integer arithmetic, so these expectations are exact.
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Source {
  phx_mc_metric_source* handle = nullptr;
  ~Source() { phx_mc_metric_source_free(handle); }
};

// A source whose exact points are small integers over `denominator`.
void create(Source& source, const std::vector<std::int64_t>& numerators, std::uint32_t denominator,
            const std::vector<std::int64_t>& cells) {
  const std::int64_t point_count = static_cast<std::int64_t>(numerators.size() / 3);
  std::vector<double> points;
  std::vector<std::uint32_t> words;
  std::vector<std::int64_t> offsets = {0};
  std::vector<std::int8_t> signs;
  for (const std::int64_t value : numerators) {
    points.push_back(static_cast<double>(value) / static_cast<double>(denominator));
    signs.push_back(value < 0 ? -1 : 1);
    if (value != 0) words.push_back(static_cast<std::uint32_t>(value < 0 ? -value : value));
    offsets.push_back(static_cast<std::int64_t>(words.size()));
  }
  words.push_back(denominator);
  offsets.push_back(static_cast<std::int64_t>(words.size()));
  // The carriers do not read the tensors of the source.
  const std::vector<double> tensors(static_cast<std::size_t>(9 * point_count), 0.0);
  PHX_CHECK(phx_mc_metric_source_create(
                point_count, points.data(), words.data(), offsets.data(), signs.data(),
                static_cast<std::int64_t>(cells.size() / 4), cells.data(), tensors.data(),
                tensors.data(), &source.handle) == PHX_MC_OK);
  PHX_CHECK(source.handle != nullptr);
}

struct Found {
  std::int64_t count = -1;
  std::vector<std::int64_t> rows, weights;
  std::vector<std::int32_t> bits;
  std::vector<double> points;
  std::vector<std::int8_t> exact;
};

Found search(const Source& source, const double* target, const double* current,
             const std::vector<std::int64_t>& simplex_rows, const std::vector<std::int8_t>& declared,
             std::int32_t order, std::int64_t limit, std::int64_t cell = -1) {
  Found found;
  const std::size_t room = static_cast<std::size_t>(limit);
  found.rows.resize(4 * room);
  found.weights.resize(4 * room);
  found.bits.resize(room);
  found.points.resize(3 * room);
  found.exact.resize(room);
  PHX_CHECK(phx_mc_metric_source_carriers(source.handle, target, current, cell,
                                          static_cast<std::int64_t>(declared.size()),
                                          simplex_rows.data(), declared.data(), order, limit,
                                          found.rows.data(), found.weights.data(),
                                          found.bits.data(), found.points.data(),
                                          found.exact.data(), &found.count) == PHX_MC_OK);
  return found;
}

// Every carrier: positive weights over distinct rows that sum to 2^bits.
void check_invariants(const Found& found) {
  for (std::int64_t index = 0; index < found.count; ++index) {
    const std::size_t at = static_cast<std::size_t>(index);
    std::int64_t sum = 0;
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const std::int64_t row = found.rows[4 * at + slot];
      const std::int64_t weight = found.weights[4 * at + slot];
      PHX_CHECK((row < 0) == (weight == 0));
      PHX_CHECK(weight >= 0);
      for (std::size_t other = 0; other < slot; ++other) {
        PHX_CHECK(row < 0 || found.rows[4 * at + other] != row);
      }
      sum += weight;
    }
    PHX_CHECK(found.bits[at] >= 1 && found.bits[at] <= 53);
    PHX_CHECK(sum == (std::int64_t{1} << found.bits[at]));
  }
}

void unit_cell() {
  // The unit corner tetrahedron; the goal is its centroid of weight 1/4 each.
  Source source;
  create(source, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, 1, {0, 1, 2, 3});
  const double target[3] = {0.25, 0.25, 0.25};
  const double current[3] = {0.0, 0.0, 0.0};
  // extent 1 over a move of sqrt(3)/4: the first resolution is 1. There the
  // partial sums 1/2, 1, 3/2 round (ties to even) to 0, 1, 2: the midpoint of
  // an edge, exactly as far from the goal as the current point, so it is not
  // a carrier. At resolution 2 the weights are 1, 1, 1, 1 over 4.
  const Found coarse = search(source, target, current, {}, {}, 0, 4);
  check_invariants(coarse);
  PHX_CHECK(coarse.count == 4);
  PHX_CHECK(coarse.bits[0] == 2 && coarse.bits[1] == 3);
  for (std::size_t slot = 0; slot < 4; ++slot) {
    PHX_CHECK(coarse.rows[slot] == static_cast<std::int64_t>(slot));
    PHX_CHECK(coarse.weights[slot] == 1);
  }
  for (std::size_t axis = 0; axis < 3; ++axis) PHX_CHECK(coarse.points[axis] == 0.25);
  PHX_CHECK(coarse.exact[0] == 1);
  // The same exact point at every finer resolution, weights doubled each time.
  PHX_CHECK(coarse.weights[4] == 2 && coarse.points[3] == 0.25);

  const Found fine = search(source, target, current, {}, {}, 1, 2);
  check_invariants(fine);
  PHX_CHECK(fine.count == 2 && fine.bits[0] == 53 && fine.bits[1] == 52);
  const Found both = search(source, target, current, {}, {}, 2, 3);
  check_invariants(both);
  PHX_CHECK(both.count == 3 && both.bits[0] == 2 && both.bits[1] == 53 && both.bits[2] == 52);

  // A goal on a face has no weight on the opposite vertex; one outside the
  // cell, one at the current point and a limit of zero give nothing.
  const double on_face[3] = {0.5, 0.5, 0.0};
  const Found face = search(source, on_face, current, {}, {}, 0, 1);
  check_invariants(face);
  PHX_CHECK(face.count == 1 && face.bits[0] == 1);
  PHX_CHECK(face.rows[0] == 1 && face.rows[1] == 2 && face.rows[2] == -1);
  const double outside[3] = {1.0, 1.0, 1.0};
  PHX_CHECK(search(source, outside, current, {}, {}, 0, 4).count == 0);
  PHX_CHECK(search(source, current, current, {}, {}, 0, 4).count == 0);
  PHX_CHECK(search(source, target, current, {}, {}, 0, 0).count == 0);
}

void rational_cell() {
  // The same tetrahedron over the denominator 3: an exact combination is a
  // binary64 value only when its numerator is divisible by three.
  Source source;
  create(source, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, {0, 1, 2, 3});
  const double target[3] = {0.0625, 0.0625, 0.0625};
  const double current[3] = {0.0, 0.0, 0.0};
  const Found found = search(source, target, current, {}, {}, 0, 53);
  check_invariants(found);
  PHX_CHECK(found.count > 0);
  // In a cell an image that is only the rounding of its exact point is a carrier.
  bool rounded = false;
  for (std::int64_t index = 0; index < found.count; ++index) {
    rounded = rounded || found.exact[static_cast<std::size_t>(index)] == 0;
  }
  PHX_CHECK(rounded);
}

void two_cells() {
  // Two cells sharing the face (1, 2, 3): a goal on that face is located in
  // the first cell that holds it, in search order.
  Source source;
  create(source, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, 1, {4, 1, 2, 3, 0, 1, 2, 3});
  const double target[3] = {0.5, 0.25, 0.25};
  const double current[3] = {0.5, 0.5, 0.0};
  const Found found = search(source, target, current, {}, {}, 0, 1);
  check_invariants(found);
  PHX_CHECK(found.count == 1);
  // Weights 2, 1, 1 over 4 on the shared face, in the order of the first cell.
  PHX_CHECK(found.bits[0] == 2);
  PHX_CHECK(found.rows[0] == 1 && found.rows[1] == 2 && found.rows[2] == 3 && found.rows[3] == -1);
  PHX_CHECK(found.weights[0] == 2 && found.weights[1] == 1 && found.weights[2] == 1);

  // Restricted to one cell, only that cell is tried. A goal inside the
  // second cell has no carrier in the first and the same carriers in the
  // second as in the search of both.
  const double inner[3] = {0.25, 0.25, 0.25};
  const double origin[3] = {0.0, 0.0, 0.0};
  const Found both = search(source, inner, origin, {}, {}, 0, 53);
  const Found second = search(source, inner, origin, {}, {}, 0, 53, 1);
  PHX_CHECK(both.count > 0 && second.count == both.count);
  PHX_CHECK(second.rows == both.rows && second.weights == both.weights);
  PHX_CHECK(second.points == both.points);
  PHX_CHECK(search(source, inner, origin, {}, {}, 0, 53, 0).count == 0);
  // The goal on the shared face is held by either cell.
  const Found shared = search(source, target, current, {}, {}, 0, 1, 1);
  PHX_CHECK(shared.count == 1 && shared.rows == found.rows && shared.weights == found.weights);
}

void simplices() {
  Source source;
  create(source, {0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2}, 1, {0, 1, 2, 3});
  const double current[3] = {0.0, 0.0, 0.0};
  // Projection on the edge (0, 1): the target above its quarter point.
  const double target[3] = {0.5, 1.0, 0.0};
  const Found edge = search(source, target, current, {0, 1, -1}, {0}, 0, 1);
  check_invariants(edge);
  PHX_CHECK(edge.count == 1 && edge.bits[0] == 2);
  PHX_CHECK(edge.rows[0] == 0 && edge.rows[1] == 1 && edge.weights[0] == 3 && edge.weights[1] == 1);
  PHX_CHECK(edge.points[0] == 0.5 && edge.points[1] == 0.0 && edge.points[2] == 0.0);
  PHX_CHECK(edge.exact[0] == 1);
  // A projection beyond the end of the edge is not on it.
  const double beyond[3] = {3.0, 0.0, 0.0};
  PHX_CHECK(search(source, beyond, current, {0, 1, -1}, {0}, 0, 1).count == 0);
  // A degenerate simplex is passed over for the next one.
  const Found passed = search(source, target, current, {0, 0, -1, 0, 1, -1}, {0, 0}, 0, 1);
  PHX_CHECK(passed.count == 1 && passed.weights[0] == 3 && passed.weights[1] == 1);
  // Projection on the triangle (0, 1, 2): the target above (1/2, 1/2, 0).
  const double above[3] = {0.5, 0.5, 7.0};
  const Found triangle = search(source, above, current, {0, 1, 2}, {0}, 0, 1);
  check_invariants(triangle);
  PHX_CHECK(triangle.count == 1 && triangle.bits[0] == 2);
  PHX_CHECK(triangle.weights[0] == 2 && triangle.weights[1] == 1 && triangle.weights[2] == 1);
  PHX_CHECK(triangle.points[0] == 0.5 && triangle.points[1] == 0.5 && triangle.points[2] == 0.0);
}

void declared_faces() {
  // Over the denominator 3 no interior combination of this triangle is
  // binary64, so only a declared face has carriers.
  Source source;
  create(source, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, {0, 1, 2, 3});
  const double target[3] = {0.0625, 0.0625, 0.0};
  const double current[3] = {0.0, 0.0, 0.0};
  const Found undeclared = search(source, target, current, {0, 1, 2}, {0}, 0, 53);
  check_invariants(undeclared);
  for (std::int64_t index = 0; index < undeclared.count; ++index) {
    PHX_CHECK(undeclared.exact[static_cast<std::size_t>(index)] == 1);
  }
  const Found declared = search(source, target, current, {0, 1, 2}, {1}, 0, 53);
  check_invariants(declared);
  PHX_CHECK(declared.count > undeclared.count);
}

void guards_arguments() {
  Source source;
  create(source, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, 1, {0, 1, 2, 3});
  const double target[3] = {0.25, 0.25, 0.25};
  const double current[3] = {0.0, 0.0, 0.0};
  std::int64_t rows[4], weights[4], count = 7;
  std::int32_t bits = 7;
  double points[3];
  std::int8_t exact = 7;
  const std::int64_t absent[3] = {0, 9, -1};
  const std::int8_t flag = 0;
  const double not_finite[3] = {0.25, 0.25, std::numeric_limits<double>::infinity()};
  PHX_CHECK(phx_mc_metric_source_carriers(nullptr, target, current, -1, 0, nullptr, nullptr, 0,
                                          1, rows, weights, &bits, points, &exact, &count) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_source_carriers(source.handle, target, current, -1, 0, nullptr,
                                          nullptr, 3, 1, rows, weights, &bits, points, &exact,
                                          &count) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_source_carriers(source.handle, target, current, -1, 0, nullptr,
                                          nullptr, 0, 1, nullptr, weights, &bits, points, &exact,
                                          &count) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_source_carriers(source.handle, target, current, -1, 1, absent, &flag,
                                          0, 1, rows, weights, &bits, points, &exact, &count) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_source_carriers(source.handle, not_finite, current, -1, 0, nullptr,
                                          nullptr, 0, 1, rows, weights, &bits, points, &exact,
                                          &count) == PHX_MC_NONFINITE_INPUT);
  // A cell that does not exist, and a cell together with simplices.
  const std::int64_t edge[3] = {0, 1, -1};
  PHX_CHECK(phx_mc_metric_source_carriers(source.handle, target, current, 1, 0, nullptr, nullptr,
                                          0, 1, rows, weights, &bits, points, &exact, &count) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_source_carriers(source.handle, target, current, 0, 1, edge, &flag, 0,
                                          1, rows, weights, &bits, points, &exact, &count) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(count == 7 && bits == 7 && exact == 7);

  // A cell naming an absent point and a zero denominator are refused.
  const double coordinates[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::uint32_t words[4] = {1, 1, 1, 1};
  const std::int64_t offsets[14] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4};
  const std::int64_t zero_offsets[14] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3};
  const std::int8_t signs[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const std::int64_t bad_cell[4] = {0, 1, 2, 4};
  const std::int64_t cell[4] = {0, 1, 2, 3};
  const double tensors[36] = {};
  phx_mc_metric_source* refused = nullptr;
  PHX_CHECK(phx_mc_metric_source_create(4, coordinates, words, offsets, signs, 1, bad_cell,
                                        tensors, tensors, &refused) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(refused == nullptr);
  PHX_CHECK(phx_mc_metric_source_create(4, coordinates, words, zero_offsets, signs, 1, cell,
                                        tensors, tensors, &refused) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(refused == nullptr);
  PHX_CHECK(phx_mc_metric_source_create(4, coordinates, words, offsets, signs, 1, cell, nullptr,
                                        tensors, &refused) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_source_create(4, coordinates, words, offsets, signs, 1, cell, tensors,
                                        tensors, nullptr) == PHX_MC_INVALID_ARGUMENT);
  phx_mc_metric_source_free(nullptr);
}

}  // namespace

int main() {
  unit_cell();
  rational_cell();
  two_cells();
  simplices();
  declared_faces();
  guards_arguments();
  return phx::mc::test::finish("test_metric_carrier");
}
