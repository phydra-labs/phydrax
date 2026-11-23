//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact carriers of a goal in a tetrahedral source: cases whose weights and
// images follow from the definition by hand, the invariants of every carrier
// (positive integer weights that sum to 2^bits, an image closer to the goal
// than the current point), the three resolution orders and the argument
// guards. The search is integer arithmetic, so these expectations are exact.
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "metric_carrier.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Source {
  phx_mc_metric_source* handle = nullptr;
  ~Source() { phx_mc_metric_source_free(handle); }
};

// The 32-bit limbs of value * 3^power, least significant first; none for zero.
std::vector<std::uint32_t> limbs_of(std::uint64_t value, int power) {
  std::vector<std::uint32_t> limbs;
  for (; value != 0; value >>= 32) limbs.push_back(static_cast<std::uint32_t>(value));
  for (int step = 0; step < power && !limbs.empty(); ++step) {
    std::uint64_t carry = 0;
    for (std::uint32_t& limb : limbs) {
      const std::uint64_t product = 3 * static_cast<std::uint64_t>(limb) + carry;
      limb = static_cast<std::uint32_t>(product);
      carry = product >> 32;
    }
    if (carry != 0) limbs.push_back(static_cast<std::uint32_t>(carry));
  }
  return limbs;
}

// A source whose exact points are small integers over `denominator`, written
// with numerators and denominator both multiplied by 3^power.
void create(Source& source, const std::vector<std::int64_t>& numerators, std::uint32_t denominator,
            const std::vector<std::int64_t>& cells, int power = 0) {
  const std::int64_t point_count = static_cast<std::int64_t>(numerators.size() / 3);
  std::vector<double> points;
  std::vector<std::uint32_t> words;
  std::vector<std::int64_t> offsets = {0};
  std::vector<std::int8_t> signs;
  for (const std::int64_t value : numerators) {
    points.push_back(static_cast<double>(value) / static_cast<double>(denominator));
    signs.push_back(value < 0 ? -1 : 1);
    const std::vector<std::uint32_t> limbs =
        limbs_of(static_cast<std::uint64_t>(value < 0 ? -value : value), power);
    words.insert(words.end(), limbs.begin(), limbs.end());
    offsets.push_back(static_cast<std::int64_t>(words.size()));
  }
  const std::vector<std::uint32_t> common = limbs_of(denominator, power);
  words.insert(words.end(), common.begin(), common.end());
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

// The exact points do not change when numerators and denominator share a
// factor, so neither does any carrier, however long the integers are: about
// 240 and about 1,110 bits of common factor here.
void common_factor(std::int64_t denominator, std::int64_t y_sign) {
  const std::vector<std::int64_t> numerators{
      0, 0, 0, 9, 0, 0, 0, y_sign * 10, 0, 0, 0, 11, 8, y_sign * 9, 10};
  const std::vector<std::int64_t> cells = {4, 1, 2, 3, 0, 1, 2, 3};
  double points[5][3];
  for (std::size_t row = 0; row < 5; ++row) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      points[row][axis] = static_cast<double>(numerators[3 * row + axis]) /
                          static_cast<double>(denominator);
    }
  }
  Source plain;
  create(plain, numerators, denominator, cells);
  std::uint64_t state = 20261118;
  const auto unit = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) * 0x1p-53;
  };
  std::int64_t compared = 0;
  for (const int power : {150, 700}) {
    Source scaled;
    create(scaled, numerators, denominator, cells, power);
    for (int query = 0; query < 40; ++query) {
      // A point of one cell, and a current point somewhere else.
      const std::int64_t* corners = cells.data() + 4 * (query % 2);
      double weights[4];
      double total = 0.0;
      for (double& weight : weights) {
        weight = unit();
        total += weight;
      }
      double target[3] = {0.0, 0.0, 0.0};
      double current[3];
      for (int axis = 0; axis < 3; ++axis) {
        for (int corner = 0; corner < 4; ++corner) {
          target[axis] += weights[corner] / total * points[corners[corner]][axis];
        }
        const double step = unit();
        current[axis] = target[axis] + (step - 0.5);
      }
      const std::vector<std::vector<std::int64_t>> simplex_rows = {{}, {1, 2, 3}, {0, 3, -1}};
      const std::vector<std::vector<std::int8_t>> declared = {{}, {1}, {0}};
      for (std::size_t kind = 0; kind < 3; ++kind) {
        for (std::int32_t order = 0; order < 3; ++order) {
          const Found expected =
              search(plain, target, current, simplex_rows[kind], declared[kind], order, 54);
          const Found found =
              search(scaled, target, current, simplex_rows[kind], declared[kind], order, 54);
          PHX_CHECK(found.count == expected.count);
          PHX_CHECK(found.rows == expected.rows);
          PHX_CHECK(found.weights == expected.weights);
          PHX_CHECK(found.bits == expected.bits);
          PHX_CHECK(found.exact == expected.exact);
          for (std::size_t index = 0; index < found.points.size(); ++index) {
            PHX_CHECK(std::bit_cast<std::uint64_t>(found.points[index]) ==
                      std::bit_cast<std::uint64_t>(expected.points[index]));
          }
          compared += expected.count;
        }
      }
    }
  }
  PHX_CHECK(compared > 1000);
}

void prepared_projection_box_preserves_carrier_order() {
  using phx::mc::Dyadic;
  using phx::mc::metric::CarrierOrder;
  using phx::mc::metric::CarrierSimplex;
  using phx::mc::metric::CarrierSource;
  // Two separated horizontal triangles followed by a sloped triangle.
  // The out-of-plane goal coordinate must not limit a projected lookup.
  const std::vector<double> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0,
      2, 0, 0, 3, 0, 0, 2, 1, 0,
      4, 0, 0, 5, 0, 1, 4, 1, 0};
  std::vector<Dyadic> numerators;
  for (double value : points) numerators.emplace_back(value);
  const CarrierSource source(points, std::move(numerators), Dyadic(1.0), {});
  const std::vector<CarrierSimplex> plain{
      {{0, 1, 2}, 3}, {{3, 5, 4}, 3}, {{6, 7, 8}, 3}};
  std::vector<CarrierSimplex> prepared;
  for (CarrierSimplex simplex : plain) prepared.push_back(source.prepare_simplex(simplex));

  const std::array<double, 3> current{0, 0, 0};
  // The second goal projects exactly onto a chart edge.
  const std::array<std::array<double, 3>, 3> goals{{
      {2.25, 0.25, 4.0}, {2.0, 0.5, -4.0}, {3.75, 0.25, 0.75}}};
  for (std::size_t row = 0; row < goals.size(); ++row) {
    const auto& target = goals[row];
    const auto expected =
        source.carriers(target.data(), current.data(), {plain}, -1, CarrierOrder::kFineFirst, 4);
    const auto found =
        source.carriers(target.data(), current.data(), {prepared}, -1, CarrierOrder::kFineFirst, 4);
    PHX_CHECK(!expected.empty() && found.size() == expected.size());
    if (found.size() != expected.size()) continue;
    for (std::size_t index = 0; index < found.size(); ++index) {
      PHX_CHECK(found[index].rows == expected[index].rows);
      PHX_CHECK(found[index].weights == expected[index].weights);
      PHX_CHECK(found[index].bits == expected[index].bits);
      PHX_CHECK(found[index].point == expected[index].point);
      PHX_CHECK(found[index].exact == expected[index].exact);
    }
  }

  const std::vector<CarrierSimplex> segment{source.prepare_simplex({{1, 0, -1}, 2})};
  const double target[3]{0.5, 2.0, -3.0};
  const auto found =
      source.carriers(target, current.data(), {segment}, -1, CarrierOrder::kFineFirst, 1);
  PHX_CHECK(found.size() == 1);
  if (found.size() == 1) {
    const std::array<double, 3> midpoint{0.5, 0, 0};
    PHX_CHECK(found.front().point == midpoint);
  }
  const double endpoint[3]{1.0, 2.0, -3.0};
  const double beyond[3]{std::nextafter(1.0, 2.0), 2.0, -3.0};
  PHX_CHECK(!source.carriers(endpoint, current.data(), {segment}, -1,
                             CarrierOrder::kFineFirst, 1).empty());
  PHX_CHECK(source.carriers(beyond, current.data(), {segment}, -1,
                            CarrierOrder::kFineFirst, 1).empty());
}

void rounded_source_coordinates_do_not_hide_an_oblique_hull() {
  using phx::mc::Dyadic;
  using phx::mc::metric::CarrierOrder;
  using phx::mc::metric::CarrierSimplex;
  using phx::mc::metric::CarrierSource;
  // z = 1 + 2^-54 x. All three z coordinates round to 1, but the projected
  // target lies in the simplex even though its original x is negative.
  const Dyadic unit = Dyadic(1.0).scaled_power(54);
  const Dyadic zero;
  const CarrierSource source({0, 0, 1, 1, 0, 1, 0, 1, 1},
                             {zero, zero, unit, unit, zero, unit + Dyadic(1.0),
                              zero, unit, unit},
                             unit, {});
  const CarrierSimplex face{{0, 1, 2}, 3, true};
  const double target[3]{-0x1p-56, 0.25, 5.0};
  const double current[3]{1.0, 0.0, 1.0};
  const std::array prepared{source.prepare_simplex(face)};
  const auto result = source.carriers(target, current, {prepared}, -1, CarrierOrder::kFineFirst, 1);
  PHX_CHECK(result.size() == 1);
  if (result.size() == 1) {
    PHX_CHECK(result.front().point[0] > 0.0);
    PHX_CHECK(result.front().point[1] == 0.25);
    PHX_CHECK(result.front().point[2] == 1.0 && !result.front().exact);
  }
}

void indexed_charts_preserve_exact_first_carrier_on_closed_boxes() {
  using phx::mc::Dyadic;
  using phx::mc::metric::CarrierChart;
  using phx::mc::metric::CarrierOrder;
  using phx::mc::metric::CarrierSource;
  // Two overlapping projected triangles, with different projections, followed
  // by enough separated triangles for the prepared bucket lookup.
  std::vector<double> points{0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 2, 2, 0, 2, 0, 1, 2};
  CarrierChart chart;
  chart.simplices = {{{0, 1, 2}, 3}, {{3, 4, 5}, 3}};
  for (std::int64_t row = 2; row < 34; ++row) {
    const double x = static_cast<double>(2 * row);
    points.insert(points.end(), {x, 0, 1, x + 1, 0, 1, x, 1, 1});
    chart.simplices.push_back({{3 * row, 3 * row + 1, 3 * row + 2}, 3});
  }
  const auto plain = chart.simplices;
  std::vector<Dyadic> numerators;
  for (const double value : points) numerators.emplace_back(value);
  const CarrierSource source(std::move(points), std::move(numerators), Dyadic(1.0), {});
  source.prepare_chart(chart);
  const double current[3]{-4, 3, 0};
  const std::array<std::array<double, 3>, 4> targets{{
      {0.25, 0.25, 5}, {1, 0, -5}, {std::nextafter(1.0, 2.0), 0, 5}, {40.25, 0.25, 5}}};
  const std::array<double, 4> heights{1, 1, 2, 1};
  for (std::size_t row = 0; row < targets.size(); ++row) {
    const double* target = targets[row].data();
    const auto expected = source.carriers(target, current, {plain}, -1, CarrierOrder::kFineFirst, 1);
    const auto actual = source.carriers(target, current, chart.view(), -1, CarrierOrder::kFineFirst, 1);
    PHX_CHECK(expected.size() == 1 && actual.size() == 1);
    if (expected.size() != 1 || actual.size() != 1) continue;
    PHX_CHECK(actual.front().point[2] == heights[row]);
    PHX_CHECK(actual.front().point == expected.front().point);
    PHX_CHECK(actual.front().rows == expected.front().rows);
    PHX_CHECK(actual.front().weights == expected.front().weights);
    PHX_CHECK(actual.front().bits == expected.front().bits);
    PHX_CHECK(actual.front().exact == expected.front().exact);
  }
}

void crowded_chart_still_searches_past_the_bucket_budget() {
  using phx::mc::Dyadic;
  using phx::mc::metric::CarrierChart;
  using phx::mc::metric::CarrierOrder;
  using phx::mc::metric::CarrierSource;
  std::vector<double> points{0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 2, 0, 1, 2, 1, 0, 2};
  std::vector<Dyadic> numerators;
  for (const double value : points) numerators.emplace_back(value);
  const CarrierSource source(std::move(points), std::move(numerators), Dyadic(1.0), {});
  CarrierChart chart;
  // The square boxes overlap everywhere, but only the last exact triangle
  // holds the target's projection. Partial bucket population loses it.
  chart.simplices.assign(224, {{0, 1, 2}, 3});
  chart.simplices.push_back({{3, 4, 5}, 3});
  source.prepare_chart(chart);
  const double target[3]{0.75, 0.75, 5};
  const double current[3]{0, 0, 0};
  const auto found = source.carriers(target, current, chart.view(), -1, CarrierOrder::kFineFirst, 1);
  PHX_CHECK(found.size() == 1);
  if (found.size() == 1) {
    const std::array<double, 3> projected{0.75, 0.75, 2};
    PHX_CHECK(found.front().point == projected && found.front().exact);
  }
}

void axis_aligned_projection_retains_the_rounded_authored_coordinate() {
  using phx::mc::Dyadic;
  using phx::mc::metric::CarrierOrder;
  using phx::mc::metric::CarrierSource;
  const Dyadic zero;
  const Dyadic denominator(3.0);
  const Dyadic exact_z(2.0);
  const double z = 2.0 / 3.0;
  const CarrierSource source({0, 0, z, 1, 0, z, 0, 1, z},
                             {zero, zero, exact_z, denominator, zero, exact_z,
                              zero, denominator, exact_z},
                             denominator, {});
  const std::array chart{source.prepare_simplex({{0, 1, 2}, 3, true})};
  const double target[3]{0.25, 0.25, 5};
  const auto result = source.carriers(target, target, {chart}, -1, CarrierOrder::kFineFirst, 1);
  PHX_CHECK(result.size() == 1);
  if (result.size() == 1) {
    const std::array<double, 3> projected{0.25, 0.25, z};
    PHX_CHECK(result.front().point == projected && !result.front().exact);
  }
}

}  // namespace

int main() {
  unit_cell();
  rational_cell();
  two_cells();
  simplices();
  declared_faces();
  guards_arguments();
  common_factor(7, 1);
  common_factor(8, -1);
  prepared_projection_box_preserves_carrier_order();
  rounded_source_coordinates_do_not_hide_an_oblique_hull();
  indexed_charts_preserve_exact_first_carrier_on_closed_boxes();
  crowded_chart_still_searches_past_the_bucket_budget();
  axis_aligned_projection_retains_the_rounded_authored_coordinate();
  return phx::mc::test::finish("test_metric_carrier");
}
