//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Bit vectors of the metric measurements. The inputs are produced here from
// integers only (small integer matrices and coordinates scaled by powers of
// two), so they are the same binary64 values on every platform.
// metric_float_vectors.inc holds each input with the output bits the kernels
// produced when the file was written; this test requires the same bits on
// every platform. `--emit` prints that file.
#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bit_vectors.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

using Row = std::vector<std::uint64_t>;
using Rows = std::vector<Row>;

enum Kind : std::size_t {
  k_log = 0,
  k_interpolate = 1,
  k_length = 2,
  k_band = 3,
  k_relative = 4,
  k_quality = 5,
  k_properties = 6,
  k_congruence = 7,
  k_kinds = 8
};
constexpr const char* kKindNames[k_kinds] = {
    "PHX_MF_LOG",      "PHX_MF_INTERPOLATE", "PHX_MF_LENGTH",     "PHX_MF_BAND",
    "PHX_MF_RELATIVE", "PHX_MF_QUALITY",     "PHX_MF_PROPERTIES", "PHX_MF_CONGRUENCE"};

#define PHX_MF_LOG(...) {"PHX_MF_LOG", {__VA_ARGS__}},
#define PHX_MF_INTERPOLATE(...) {"PHX_MF_INTERPOLATE", {__VA_ARGS__}},
#define PHX_MF_LENGTH(...) {"PHX_MF_LENGTH", {__VA_ARGS__}},
#define PHX_MF_BAND(...) {"PHX_MF_BAND", {__VA_ARGS__}},
#define PHX_MF_RELATIVE(...) {"PHX_MF_RELATIVE", {__VA_ARGS__}},
#define PHX_MF_QUALITY(...) {"PHX_MF_QUALITY", {__VA_ARGS__}},
#define PHX_MF_PROPERTIES(...) {"PHX_MF_PROPERTIES", {__VA_ARGS__}},
#define PHX_MF_CONGRUENCE(...) {"PHX_MF_CONGRUENCE", {__VA_ARGS__}},
const std::vector<phx::mc::test::BitRow> kRecorded = {
#include "metric_float_vectors.inc"
};

std::uint64_t bits_of(double value) { return std::bit_cast<std::uint64_t>(value); }
double value_of(std::uint64_t bits) { return std::bit_cast<double>(bits); }

constexpr std::uint64_t kNaN = 0x7ff8000000000000ULL;
constexpr std::uint64_t kInfinity = 0x7ff0000000000000ULL;
constexpr std::size_t kSupportSlots = 4;

// Knuth's 64-bit linear congruential generator: integer arithmetic only.
struct Generator {
  std::uint64_t state;
  std::uint64_t next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
  }
  // An integer in [low, high].
  std::int64_t integer(std::int64_t low, std::int64_t high) {
    return low + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(high - low + 1));
  }
};

// 2^exponent, exactly.
double power_of_two(int exponent) {
  return value_of(static_cast<std::uint64_t>(exponent + 1023) << 52);
}

// The metric bank, nine row-major values per tensor. Rows 0-7 are fixed
// cases (5-7 are not metrics); the rest are (A^T A + I) 2^e for an integer
// matrix A with entries in [-4, 4], evaluated in integers.
std::vector<double> metric_bank() {
  std::vector<double> bank = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25,
      0x1p+20, 0.0, 0.0, 0.0, 0x1p+20, 0.0, 0.0, 0.0, 0x1p+20,
      0x1p-20, 0.0, 0.0, 0.0, 0x1p-20, 0.0, 0.0, 0.0, 0x1p-20,
      4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0,
      // Indefinite, with a NaN entry, and zero.
      1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      1.0, 0.0, 0.0, 0.0, value_of(kNaN), 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Generator generator{0x452821e638d01377ULL};
  for (int row = 0; row < 32; ++row) {
    std::int64_t a[9];
    for (std::int64_t& entry : a) entry = generator.integer(-4, 4);
    const double scale = power_of_two(static_cast<int>(generator.integer(-6, 6)));
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        std::int64_t sum = i == j ? 1 : 0;
        for (int k = 0; k < 3; ++k) sum += a[3 * k + i] * a[3 * k + j];
        bank.push_back(static_cast<double>(sum) * scale);
      }
    }
  }
  // Rows 40-42: a form whose two terms overflow with opposite signs, and
  // metrics at the ends of the range the measurements are used on.
  const double extreme[27] = {
      0x1p+1000, 0.0, 0.0, 0.0, -0x1p+1000, 0.0, 0.0, 0.0, 1.0,
      0x1p+1000, 0.0, 0.0, 0.0, 0x1p+1000,  0.0, 0.0, 0.0, 0x1p+1000,
      0x1p-1000, 0.0, 0.0, 0.0, 0x1p-1000,  0.0, 0.0, 0.0, 0x1p-1000};
  bank.insert(bank.end(), extreme, extreme + 27);
  return bank;
}

struct Points {
  std::vector<double> coordinates;  // three per point
  std::vector<std::int64_t> bank;   // the metric bank row of each point
};

// Points 0-3 and 24-27 are fixed; the others have coordinates k / 16 for an
// integer k in [-64, 64] and one of the generated metrics.
Points point_bank() {
  Points points;
  const double fixed[4][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 3.0, 1.0}, {3.0, 2.0, 1.0}};
  // Point 2 carries the indefinite tensor: its form along (1, -1, 0) is negative.
  const std::int64_t fixed_bank[4] = {0, 0, 5, 4};
  for (int point = 0; point < 4; ++point) {
    points.coordinates.insert(points.coordinates.end(), fixed[point], fixed[point] + 3);
    points.bank.push_back(fixed_bank[point]);
  }
  Generator generator{0xbe5466cf34e90c6cULL};
  for (int point = 4; point < 24; ++point) {
    for (int axis = 0; axis < 3; ++axis) {
      points.coordinates.push_back(static_cast<double>(generator.integer(-64, 64)) * 0x1p-4);
    }
    points.bank.push_back(8 + (point - 4) % 32);
  }
  const double unit[4][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  for (int point = 0; point < 4; ++point) {
    points.coordinates.insert(points.coordinates.end(), unit[point], unit[point] + 3);
    points.bank.push_back(0);
  }
  // Points 28-33: coordinates whose squares and triple products overflow.
  const double far[6][3] = {{0x1p+600, 0.0, 0.0},      {0x1p+100, 0x1p+100, 0.0},
                            {0.0, 0.0, 0.0},           {0.0, 0x1p+600, 0.0},
                            {0.0, 0.0, 0x1p+600},      {0x1p+600, 0x1p+600, 0x1p+600}};
  const std::int64_t far_bank[6] = {0, 40, 40, 0, 0, 0};
  for (int point = 0; point < 6; ++point) {
    points.coordinates.insert(points.coordinates.end(), far[point], far[point] + 3);
    points.bank.push_back(far_bank[point]);
  }
  return points;
}

struct Support {
  std::vector<std::int64_t> rows;
  std::vector<double> weights;
};

std::vector<Support> supports() {
  std::vector<Support> result = {
      // A constant support returns its tensor.
      {{4}, {1.0}},
      {{4, 4}, {0.5, 0.5}},
      {{9, 9, 9}, {0.25, 0.25, 0.5}},
      {{0, 1}, {0.5, 0.5}},
      {{2, 3}, {0.5, 0.5}},
      {{8, 9}, {0.25, 0.75}},
      {{8, 9, 10}, {0.5, 0.25, 0.25}},
      {{8, 9, 10, 11}, {0.25, 0.25, 0.25, 0.25}},
      {{12, 13, 14, 15}, {0.125, 0.125, 0.25, 0.5}},
      // A zero weight still enters the sum.
      {{16, 17, 18}, {0.5, 0.0, 0.5}},
      // A support with a tensor that is not a metric has no mean.
      {{0, 5}, {0.5, 0.5}},
      {{6, 0}, {0.5, 0.5}},
      {{7, 1, 2}, {0.5, 0.25, 0.25}},
      // Nor does a constant support of such a tensor return it: an indefinite,
      // a zero and a non-finite tensor alone, and an indefinite one repeated.
      {{5}, {1.0}},
      {{7}, {1.0}},
      {{6}, {1.0}},
      {{5, 5}, {0.5, 0.5}}};
  Generator generator{0xc0ac29b7c97c50ddULL};
  for (int row = 0; row < 24; ++row) {
    Support support;
    const std::int64_t count = generator.integer(2, 4);
    std::int64_t remaining = 64;
    for (std::int64_t k = 0; k < count; ++k) {
      support.rows.push_back(generator.integer(8, 39));
      const std::int64_t numerator = k + 1 == count ? remaining : generator.integer(1, 15);
      remaining -= numerator;
      support.weights.push_back(static_cast<double>(numerator) * 0x1p-6);
    }
    result.push_back(support);
  }
  return result;
}

std::vector<std::array<std::int64_t, 2>> edges() {
  std::vector<std::array<std::int64_t, 2>> result = {
      {0, 1}, {1, 0}, {0, 0}, {2, 3}, {3, 2}, {0, 3}, {24, 25}, {25, 27}};
  for (std::int64_t first = 4; first < 16; ++first) {
    for (std::int64_t second = first + 1; second < 16; ++second) {
      result.push_back({first, second});
    }
  }
  for (std::int64_t point = 4; point < 10; ++point) result.push_back({0, point});
  // An infinite form at both ends, and a form whose terms cancel at infinity.
  result.push_back({0, 28});
  result.push_back({28, 0});
  result.push_back({30, 29});
  return result;
}

std::vector<std::array<std::int64_t, 4>> cells() {
  std::vector<std::array<std::int64_t, 4>> result = {
      {24, 25, 26, 27},  // the unit corner in the identity metric
      {25, 24, 26, 27},  // inverted
      {0, 1, 2, 3},      // a corner whose tensor is not a metric
      {24, 25, 26, 24},  // a repeated corner
      {0, 1, 24, 25}};   // flat
  for (std::int64_t first = 4; first + 3 < 24; ++first) {
    result.push_back({first, first + 1, first + 2, first + 3});
  }
  // An infinite determinant, and one whose cofactor terms cancel at infinity.
  result.push_back({0, 28, 31, 32});
  result.push_back({0, 33, 33, 33});
  return result;
}

std::vector<double> band_lengths(const std::vector<double>& measured) {
  std::vector<double> lengths = {
      1.0, 0x1.6a09e667f3bccp+0, 0x1.6a09e667f3bcdp+0, 0x1.6a09e667f3bcep+0,
      0x1.6a09e667f3bcbp-1, 0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bcdp-1,
      // The binary64 neighbors of one quantum above and one quantum below the band.
      0x1.6a647432ce57fp+0, 0x1.6a647432ce581p+0, 0x1.69af6f3db7a62p-1,
      0x1.69af6f3db7a64p-1, 1.5, 0.7, 2.0, 0.5,
      0x1p-40, 0x1p+40, 0x1p-1074, 0x1.fffffffffffffp+1023, 0.0,
      value_of(kInfinity), value_of(kNaN)};
  lengths.insert(lengths.end(), measured.begin(), measured.end());
  return lengths;
}

void append_bits(Row& row, const double* values, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) row.push_back(bits_of(values[index]));
}

std::array<Rows, k_kinds> evaluate() {
  std::array<Rows, k_kinds> rows;
  const std::vector<double> bank = metric_bank();
  const int64_t bank_count = static_cast<int64_t>(bank.size() / 9);

  std::vector<double> logarithms(bank.size());
  std::vector<int8_t> log_valid(static_cast<std::size_t>(bank_count));
  PHX_CHECK(phx_mc_metric_logarithms(bank_count, bank.data(), logarithms.data(),
                                     log_valid.data()) == PHX_MC_OK);
  for (int64_t row = 0; row < bank_count; ++row) {
    Row record;
    append_bits(record, bank.data() + 9 * row, 9);
    record.push_back(static_cast<std::uint64_t>(log_valid[static_cast<std::size_t>(row)]));
    append_bits(record, logarithms.data() + 9 * row, 9);
    rows[k_log].push_back(record);
    // A row does not depend on the rows evaluated with it.
    double alone[9];
    int8_t alone_valid = -1;
    PHX_CHECK(phx_mc_metric_logarithms(1, bank.data() + 9 * row, alone, &alone_valid) ==
              PHX_MC_OK);
    PHX_CHECK(alone_valid == log_valid[static_cast<std::size_t>(row)]);
    for (int index = 0; index < 9; ++index) {
      PHX_CHECK(bits_of(alone[index]) == bits_of(logarithms[9 * row + index]));
    }
  }

  const std::vector<Support> stencils = supports();
  std::vector<int64_t> offsets = {0};
  std::vector<int64_t> support_rows;
  std::vector<double> weights;
  for (const Support& support : stencils) {
    support_rows.insert(support_rows.end(), support.rows.begin(), support.rows.end());
    weights.insert(weights.end(), support.weights.begin(), support.weights.end());
    offsets.push_back(static_cast<int64_t>(support_rows.size()));
  }
  const int64_t stencil_count = static_cast<int64_t>(stencils.size());
  std::vector<double> means(9 * stencils.size());
  std::vector<int8_t> mean_valid(stencils.size());
  PHX_CHECK(phx_mc_metric_interpolate(stencil_count, offsets.data(), support_rows.data(),
                                      weights.data(), bank_count, bank.data(), nullptr,
                                      means.data(), mean_valid.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < stencils.size(); ++row) {
    const Support& support = stencils[row];
    Row record = {support.rows.size()};
    bool metrics_only = true;
    for (std::size_t slot = 0; slot < kSupportSlots; ++slot) {
      record.push_back(slot < support.rows.size()
                           ? static_cast<std::uint64_t>(support.rows[slot])
                           : 0);
    }
    for (std::size_t slot = 0; slot < kSupportSlots; ++slot) {
      record.push_back(slot < support.weights.size() ? bits_of(support.weights[slot]) : 0);
    }
    for (const std::int64_t source : support.rows) {
      if (log_valid[static_cast<std::size_t>(source)] == 0) metrics_only = false;
    }
    record.push_back(static_cast<std::uint64_t>(mean_valid[row]));
    append_bits(record, means.data() + 9 * row, 9);
    rows[k_interpolate].push_back(record);
    // The same row alone, and with the bank's stored logarithms.
    const int64_t width = static_cast<int64_t>(support.rows.size());
    const int64_t alone_offsets[2] = {0, width};
    for (const double* stored : {static_cast<const double*>(nullptr),
                                 static_cast<const double*>(logarithms.data())}) {
      if (stored != nullptr && !metrics_only) continue;
      double alone[9];
      int8_t alone_valid = -1;
      PHX_CHECK(phx_mc_metric_interpolate(1, alone_offsets, support.rows.data(),
                                          support.weights.data(), bank_count, bank.data(),
                                          stored, alone, &alone_valid) == PHX_MC_OK);
      PHX_CHECK(alone_valid == mean_valid[row]);
      for (std::size_t index = 0; index < 9; ++index) {
        PHX_CHECK(bits_of(alone[index]) == bits_of(means[9 * row + index]));
      }
    }
  }

  const Points points = point_bank();
  const int64_t point_count = static_cast<int64_t>(points.bank.size());
  std::vector<double> point_metrics;
  for (const std::int64_t source : points.bank) {
    point_metrics.insert(point_metrics.end(), bank.begin() + 9 * source,
                         bank.begin() + 9 * source + 9);
  }
  const std::vector<std::array<std::int64_t, 2>> segments = edges();
  std::vector<int64_t> flat_edges;
  for (const std::array<std::int64_t, 2>& edge : segments) {
    flat_edges.insert(flat_edges.end(), edge.begin(), edge.end());
  }
  std::vector<double> lengths(segments.size());
  PHX_CHECK(phx_mc_metric_lengths(static_cast<int64_t>(segments.size()), flat_edges.data(),
                                  point_count, points.coordinates.data(),
                                  point_metrics.data(), lengths.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < segments.size(); ++row) {
    const std::size_t first = static_cast<std::size_t>(segments[row][0]);
    const std::size_t second = static_cast<std::size_t>(segments[row][1]);
    Row record;
    append_bits(record, points.coordinates.data() + 3 * first, 3);
    append_bits(record, points.coordinates.data() + 3 * second, 3);
    record.push_back(static_cast<std::uint64_t>(points.bank[first]));
    record.push_back(static_cast<std::uint64_t>(points.bank[second]));
    record.push_back(bits_of(lengths[row]));
    rows[k_length].push_back(record);
    double alone = 0.0;
    PHX_CHECK(phx_mc_metric_lengths(1, flat_edges.data() + 2 * row, point_count,
                                    points.coordinates.data(), point_metrics.data(),
                                    &alone) == PHX_MC_OK);
    PHX_CHECK(bits_of(alone) == bits_of(lengths[row]));
  }

  const std::vector<double> band = band_lengths(lengths);
  std::vector<double> above(band.size()), below(band.size());
  PHX_CHECK(phx_mc_metric_band_quanta(static_cast<int64_t>(band.size()), band.data(),
                                      above.data(), below.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < band.size(); ++row) {
    rows[k_band].push_back({bits_of(band[row]), bits_of(above[row]), bits_of(below[row])});
    double alone_above = -1.0;
    double alone_below = -1.0;
    PHX_CHECK(phx_mc_metric_band_quanta(1, band.data() + row, &alone_above, &alone_below) ==
              PHX_MC_OK);
    PHX_CHECK(bits_of(alone_above) == bits_of(above[row]));
    PHX_CHECK(bits_of(alone_below) == bits_of(below[row]));
  }

  const std::vector<std::array<std::int64_t, 4>> tetrahedra = cells();
  std::vector<int64_t> flat_cells;
  std::vector<double> corners;
  for (const std::array<std::int64_t, 4>& cell : tetrahedra) {
    flat_cells.insert(flat_cells.end(), cell.begin(), cell.end());
    for (const std::int64_t point : cell) {
      corners.insert(corners.end(), points.coordinates.begin() + 3 * point,
                     points.coordinates.begin() + 3 * point + 3);
    }
  }
  // One more cell of raw corners with a coordinate that is not a number.
  std::vector<double> raw = corners;
  raw.insert(raw.end(), corners.begin(), corners.begin() + 12);
  raw[raw.size() - 12] = value_of(kNaN);
  const std::size_t raw_count = raw.size() / 12;
  std::vector<double> determinants(raw_count), relative(raw_count);
  PHX_CHECK(phx_mc_metric_relative_determinants(static_cast<int64_t>(raw_count), raw.data(),
                                                determinants.data(),
                                                relative.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < raw_count; ++row) {
    Row record;
    append_bits(record, raw.data() + 12 * row, 12);
    record.push_back(bits_of(determinants[row]));
    record.push_back(bits_of(relative[row]));
    rows[k_relative].push_back(record);
    double alone_determinant = 0.0;
    double alone_relative = -1.0;
    PHX_CHECK(phx_mc_metric_relative_determinants(1, raw.data() + 12 * row,
                                                  &alone_determinant,
                                                  &alone_relative) == PHX_MC_OK);
    PHX_CHECK(bits_of(alone_relative) == bits_of(relative[row]));
    PHX_CHECK(bits_of(alone_determinant) == bits_of(determinants[row]));
  }

  std::vector<double> quality(tetrahedra.size());
  PHX_CHECK(phx_mc_metric_tetra_quality(static_cast<int64_t>(tetrahedra.size()),
                                        flat_cells.data(), point_count,
                                        points.coordinates.data(), point_metrics.data(),
                                        quality.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < tetrahedra.size(); ++row) {
    Row record;
    append_bits(record, corners.data() + 12 * row, 12);
    for (const std::int64_t point : tetrahedra[row]) {
      record.push_back(static_cast<std::uint64_t>(points.bank[static_cast<std::size_t>(point)]));
    }
    record.push_back(bits_of(quality[row]));
    rows[k_quality].push_back(record);
    double alone = -1.0;
    PHX_CHECK(phx_mc_metric_tetra_quality(1, flat_cells.data() + 4 * row, point_count,
                                          points.coordinates.data(), point_metrics.data(),
                                          &alone) == PHX_MC_OK);
    PHX_CHECK(bits_of(alone) == bits_of(quality[row]));
  }
  // Field gates: the bank, then tensors at the two tolerances of the gate
  // (64 * 2^-52 of the largest eigenvalue magnitude) and beyond the range.
  std::vector<double> tensors = bank;
  const double gate_cases[][9] = {
      // An asymmetry of half the tolerance, of about the tolerance (which
      // grows with the largest eigenvalue, here just above 1) and of twice it.
      {1.0, 0x1p-47, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
      {1.0, 0x1p-46, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
      {1.0, 0x1p-45, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
      {1.0, 0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 1.0},
      // Smallest eigenvalue above, at and below the tolerance; singular; negative.
      {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0x1p-45},
      {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0x1p-46},
      {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0x1p-47},
      {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0},
      {-1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -3.0},
      // Not finite, and an eigenvalue beyond the binary64 range.
      {1.0, 0.0, 0.0, 0.0, 1.0, value_of(kInfinity), 0.0, value_of(kInfinity), 1.0},
      {0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023,
       0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023,
       0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023}};
  for (const auto& tensor : gate_cases) tensors.insert(tensors.end(), tensor, tensor + 9);
  const std::size_t tensor_count = tensors.size() / 9;
  std::vector<int8_t> symmetric(tensor_count), definite(tensor_count);
  PHX_CHECK(phx_mc_metric_tensor_properties(static_cast<int64_t>(tensor_count), tensors.data(),
                                            symmetric.data(), definite.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < tensor_count; ++row) {
    Row record;
    append_bits(record, tensors.data() + 9 * row, 9);
    record.push_back(static_cast<std::uint64_t>(symmetric[row]));
    record.push_back(static_cast<std::uint64_t>(definite[row]));
    rows[k_properties].push_back(record);
    int8_t alone_symmetric = -1;
    int8_t alone_definite = -1;
    PHX_CHECK(phx_mc_metric_tensor_properties(1, tensors.data() + 9 * row, &alone_symmetric,
                                              &alone_definite) == PHX_MC_OK);
    PHX_CHECK(alone_symmetric == symmetric[row] && alone_definite == definite[row]);
  }

  // Congruences A^T M A = R. A is a signed permutation (an isometry of the
  // axes), so the exact image of a bank metric is a signed rearrangement of
  // its entries; every second row then moves one entry of R by a small step.
  std::vector<double> maps, images, sources;
  Generator congruences{0x1f83d9abfb41bd6bULL};
  for (int row = 0; row < 36; ++row) {
    std::int64_t order[3] = {0, 1, 2};
    for (int last = 2; last > 0; --last) {
      const std::int64_t other = congruences.integer(0, last);
      const std::int64_t swap = order[last];
      order[last] = order[other];
      order[other] = swap;
    }
    double sign[3];
    for (double& value : sign) value = congruences.integer(0, 1) == 0 ? 1.0 : -1.0;
    // A e_i = sign_i e_{order_i}.
    double linear[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int column = 0; column < 3; ++column) linear[3 * order[column] + column] = sign[column];
    const double* metric = bank.data() + 9 * (8 + row % 32);
    double source[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        source[3 * i + j] = (sign[i] * sign[j]) * metric[3 * order[i] + order[j]];
      }
    }
    if (row % 2 == 1) {
      const std::int64_t entry = congruences.integer(0, 8);
      source[entry] =
          source[entry] * (1.0 + power_of_two(-static_cast<int>(congruences.integer(8, 50))));
    }
    maps.insert(maps.end(), linear, linear + 9);
    images.insert(images.end(), metric, metric + 9);
    sources.insert(sources.end(), source, source + 9);
  }
  // A map that is not an isometry, a singular one and values that are not numbers.
  const double extra_maps[3][9] = {{2.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0},
                                   {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                                   {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
  const std::size_t extra_images[3] = {4, 4, 6};
  for (int row = 0; row < 3; ++row) {
    maps.insert(maps.end(), extra_maps[row], extra_maps[row] + 9);
    const double* image = bank.data() + 9 * extra_images[row];
    images.insert(images.end(), image, image + 9);
    sources.insert(sources.end(), bank.data() + 36, bank.data() + 45);
  }
  const std::size_t congruence_count = maps.size() / 9;
  std::vector<double> errors(congruence_count), scales(congruence_count);
  PHX_CHECK(phx_mc_metric_congruence_defects(static_cast<int64_t>(congruence_count), maps.data(),
                                             images.data(), sources.data(), errors.data(),
                                             scales.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < congruence_count; ++row) {
    Row record;
    append_bits(record, maps.data() + 9 * row, 9);
    append_bits(record, images.data() + 9 * row, 9);
    append_bits(record, sources.data() + 9 * row, 9);
    record.push_back(bits_of(errors[row]));
    record.push_back(bits_of(scales[row]));
    rows[k_congruence].push_back(record);
    double alone_error = -1.0;
    double alone_scale = -1.0;
    PHX_CHECK(phx_mc_metric_congruence_defects(1, maps.data() + 9 * row, images.data() + 9 * row,
                                               sources.data() + 9 * row, &alone_error,
                                               &alone_scale) == PHX_MC_OK);
    PHX_CHECK(bits_of(alone_error) == bits_of(errors[row]));
    PHX_CHECK(bits_of(alone_scale) == bits_of(scales[row]));
  }
  return rows;
}

// The rows in the order of the recorded file: kind by kind.
std::vector<phx::mc::test::BitRow> in_file_order(const std::array<Rows, k_kinds>& rows) {
  std::vector<phx::mc::test::BitRow> ordered;
  for (std::size_t kind = 0; kind < k_kinds; ++kind) {
    for (const Row& row : rows[kind]) ordered.push_back({kKindNames[kind], row});
  }
  return ordered;
}

void emit(const std::array<Rows, k_kinds>& rows) {
  std::printf(
      "// Bit vectors of the metric measurements, written by `test_metric_float --emit`.\n"
      "// One row per input, in the order the test generates them. A value is its\n"
      "// IEEE 754 binary64 bit pattern; counts, bank rows and flags are integers.\n"
      "// PHX_MF_LOG(nine metric values, valid, nine logarithm values)\n"
      "//   Row r of this list is row r of the metric bank the other rows name.\n"
      "// PHX_MF_INTERPOLATE(count, four bank rows, four weights, valid, nine values)\n"
      "//   Only the first `count` bank rows and weights are the support.\n"
      "// PHX_MF_LENGTH(p, q, bank row at p, bank row at q, length)\n"
      "// PHX_MF_BAND(length, quanta above the band, quanta below the band)\n"
      "// PHX_MF_RELATIVE(twelve corner values, determinant, relative determinant)\n"
      "// PHX_MF_QUALITY(twelve corner values, four bank rows, quality)\n"
      "// PHX_MF_PROPERTIES(nine tensor values, symmetric, positive definite)\n"
      "// PHX_MF_CONGRUENCE(nine values each of A, M and R, the infinity norm of\n"
      "//                   A^T M A - R, the larger norm of A^T M A and R)\n");
  phx::mc::test::emit_rows(in_file_order(rows));
}

void compare(const std::array<Rows, k_kinds>& rows) {
  phx::mc::test::compare_rows(in_file_order(rows), kRecorded);
}

void guards_arguments() {
  const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const double points[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const double metrics[18] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                              1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double out[9] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
  double second = 7.0;
  int8_t valid = 7;
  PHX_CHECK(phx_mc_metric_logarithms(0, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(phx_mc_metric_logarithms(-1, identity, out, &valid) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_logarithms(1, identity, out, nullptr) == PHX_MC_INVALID_ARGUMENT);

  const int64_t offsets[2] = {0, 1};
  const int64_t empty_offsets[2] = {0, 0};
  const int64_t support = 0;
  const int64_t outside = 1;
  const double weight = 1.0;
  PHX_CHECK(phx_mc_metric_interpolate(1, nullptr, &support, &weight, 1, identity, nullptr,
                                      out, &valid) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_interpolate(1, offsets, nullptr, &weight, 1, identity, nullptr,
                                      out, &valid) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_interpolate(1, empty_offsets, &support, &weight, 1, identity,
                                      nullptr, out, &valid) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_interpolate(1, offsets, &outside, &weight, 1, identity, nullptr,
                                      out, &valid) == PHX_MC_INVALID_INPUT);

  const int64_t edge[2] = {0, 1};
  const int64_t bad_edge[2] = {0, 2};
  const int64_t negative_edge[2] = {-1, 1};
  PHX_CHECK(phx_mc_metric_lengths(0, nullptr, 0, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(phx_mc_metric_lengths(1, edge, 2, points, metrics, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_lengths(1, bad_edge, 2, points, metrics, out) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_lengths(1, negative_edge, 2, points, metrics, out) ==
            PHX_MC_INVALID_INPUT);

  PHX_CHECK(phx_mc_metric_band_quanta(1, points, out, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_band_quanta(0, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(phx_mc_metric_relative_determinants(1, nullptr, out, &second) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_relative_determinants(0, nullptr, nullptr, nullptr) == PHX_MC_OK);

  const int64_t bad_cell[4] = {0, 1, 0, 2};
  PHX_CHECK(phx_mc_metric_tetra_quality(1, bad_cell, 2, points, metrics, out) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_tetra_quality(1, nullptr, 2, points, metrics, out) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_tetra_quality_min(1, bad_cell, 2, points, metrics, 2, out) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_tetra_quality_min(1, bad_cell, 2, points, metrics, 0, out) ==
            PHX_MC_INVALID_ARGUMENT);
  // A refused call writes nothing.
  for (const double value : out) PHX_CHECK(value == 7.0);
  PHX_CHECK(second == 7.0 && valid == 7);
}

// Values that follow from the definitions without rounding.
void exact_cases() {
  const double points[12] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double metrics[36];
  for (int corner = 0; corner < 4; ++corner) {
    for (int index = 0; index < 9; ++index) metrics[9 * corner + index] = index % 4 == 0 ? 1.0 : 0.0;
  }
  const int64_t edge[2] = {0, 1};
  double length = 0.0;
  PHX_CHECK(phx_mc_metric_lengths(1, edge, 4, points, metrics, &length) == PHX_MC_OK);
  PHX_CHECK(length == 1.0);
  double determinant = 0.0;
  double relative = 0.0;
  PHX_CHECK(phx_mc_metric_relative_determinants(1, points, &determinant, &relative) ==
            PHX_MC_OK);
  PHX_CHECK(determinant == 1.0);
  double logarithm[9];
  int8_t valid = 0;
  PHX_CHECK(phx_mc_metric_logarithms(1, metrics, logarithm, &valid) == PHX_MC_OK);
  PHX_CHECK(valid == 1);
  for (const double value : logarithm) PHX_CHECK(value == 0.0);
  const double band[3] = {1.0, 0x1.6a09e667f3bcdp+0, 0x1.6a09e667f3bccp-1};
  double above[3];
  double below[3];
  PHX_CHECK(phx_mc_metric_band_quanta(3, band, above, below) == PHX_MC_OK);
  for (int row = 0; row < 3; ++row) PHX_CHECK(above[row] == 0.0 && below[row] == 0.0);
}

void minimum_quality_across_worker_blocks() {
  constexpr int64_t cell_count = 4109;
  const double points[15] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0, 0.0, 0.0, 0x1p-10};
  double metrics[45]{};
  for (int vertex = 0; vertex < 5; ++vertex) {
    for (int axis = 0; axis < 3; ++axis) {
      metrics[9 * vertex + 4 * axis] = 0.75 + 0.25 * vertex;
    }
  }
  const int64_t unit[4]{0, 1, 2, 3};
  std::vector<int64_t> cells(4 * cell_count);
  for (int64_t row = 0; row < cell_count; ++row) {
    std::copy_n(unit, 4, cells.data() + 4 * row);
  }
  cells.back() = 4;  // Lowest quality in the final partial block.
  std::vector<double> qualities(cell_count);
  PHX_CHECK(phx_mc_metric_tetra_quality(cell_count, cells.data(), 5, points, metrics,
                                      qualities.data()) == PHX_MC_OK);
  PHX_CHECK(qualities.back() > 0.0 && qualities.back() < qualities.front());
  for (const int32_t workers : {1, 2, 3, 4, 8}) {
    double minimum = -1.0;
    PHX_CHECK(phx_mc_metric_tetra_quality_min(cell_count, cells.data(), 5, points, metrics,
                                            workers, &minimum) == PHX_MC_OK);
    PHX_CHECK(bits_of(minimum) == bits_of(qualities.back()));
  }
}

void minimum_quality_retains_nonfinite_evidence() {
  constexpr double far = 0x1p+400;
  const double points[12] = {0.0, 0.0, 0.0, far, 0.0, 0.0, 0.0, far, 0.0, 0.0, 0.0, far};
  double metrics[36]{};
  for (int vertex = 0; vertex < 4; ++vertex) {
    for (int axis = 0; axis < 3; ++axis) metrics[9 * vertex + 4 * axis] = 1.0;
  }
  const int64_t cells[4]{0, 1, 2, 3};
  double quality = 0.0;
  double minimum = 0.0;
  PHX_CHECK(phx_mc_metric_tetra_quality(1, cells, 4, points, metrics, &quality) == PHX_MC_OK);
  PHX_CHECK(!std::isfinite(quality));
  PHX_CHECK(phx_mc_metric_tetra_quality_min(1, cells, 4, points, metrics, 1, &minimum) ==
            PHX_MC_OK);
  PHX_CHECK(std::isnan(minimum));
}

}  // namespace

int main(int argc, char** argv) {
  const std::array<Rows, k_kinds> rows = evaluate();
  if (argc > 1 && std::strcmp(argv[1], "--emit") == 0) {
    emit(rows);
    return 0;
  }
  compare(rows);
  guards_arguments();
  exact_cases();
  minimum_quality_across_worker_blocks();
  minimum_quality_retains_nonfinite_evidence();
  return phx::mc::test::finish("test_metric_float");
}
