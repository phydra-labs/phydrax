// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "verify_chart_partition.hpp"
#include "phydrax_meshcore.h"

#define CHECK PHX_CHECK

namespace {
struct Fixture {
  std::array<int32_t, 2> patch{0, 0};
  std::array<double, 12> tri{0,0, 1,0, 1,1, 0,0, 1,1, 0,1};
  std::array<int64_t, 4> use_patch{0,0,0,0}, chord_use{0,1,2,3}, loop{0,0,0,0};
  std::array<double, 8> range{0,1, 0,1, 0,1, 0,1};
  std::array<double, 8> chord_range{0,1, 0,1, 0,1, 0,1};
  std::array<double, 16> chord{0,0,1,0, 1,0,1,1, 1,1,0,1, 0,1,0,0};
  std::vector<phx_mc_chart_failure> failures = std::vector<phx_mc_chart_failure>(64);
  int64_t count = -1;
  int32_t run(int64_t cap = 64) {
    return phx_mc_verify_chart_partition(1, 2, patch.data(), tri.data(), 4,
        use_patch.data(), range.data(), 1, 4, chord_use.data(), loop.data(),
        chord.data(), chord_range.data(), cap, cap ? failures.data() : nullptr, &count);
  }
  bool has(int kind) const {
    return std::any_of(failures.begin(), failures.begin() + count,
                       [kind](const auto& f) { return f.kind == kind; });
  }
};
}

int main() {
  Fixture f;
  CHECK(f.run() == PHX_MC_OK && f.count == 0);
  // Reversed patch is still a valid oriented partition.
  for (int i = 0; i < 2; ++i) {
    std::swap(f.tri[6*i+2], f.tri[6*i+4]);
    std::swap(f.tri[6*i+3], f.tri[6*i+5]);
  }
  for (int i = 0; i < 4; ++i) {
    std::swap(f.chord[4*i], f.chord[4*i+2]);
    std::swap(f.chord[4*i+1], f.chord[4*i+3]);
  }
  CHECK(f.run() == PHX_MC_OK && f.count == 0);
  // One inverted triangle must not be accepted just because its area is nonzero.
  std::swap(f.tri[2], f.tri[4]); std::swap(f.tri[3], f.tri[5]);
  CHECK(f.run() == PHX_MC_OK && f.has(PHX_MC_CHART_ORIENTATION));
  CHECK(f.run(0) == PHX_MC_CAPACITY_EXCEEDED && f.count > 0);

  f = Fixture{};
  f.chord_range[1] = std::nextafter(1.0, 0.0);
  CHECK(f.run() == PHX_MC_OK && f.has(PHX_MC_CHART_USE_COVERAGE));
  f = Fixture{};
  f.chord[2] = std::nextafter(1.0, 2.0);
  CHECK(f.run() == PHX_MC_OK && f.has(PHX_MC_CHART_EDGE) && f.has(PHX_MC_CHART_LOOP));
  f = Fixture{};
  f.chord[0] = -0.0;
  CHECK(f.run() == PHX_MC_OK && f.has(PHX_MC_CHART_EDGE) && f.has(PHX_MC_CHART_LOOP));
  f = Fixture{};
  f.tri[0] = std::numeric_limits<double>::quiet_NaN();
  f.count = 123;
  // Two consecutive intervals on one use must cover the range bit-for-bit.
  {
    const int32_t patches[] = {0,0,0};
    const double triangles[] = {0,0, .5,0, 0,1, .5,0, 1,0, 1,1,
                                .5,0, 1,1, 0,1};
    const int64_t uses[] = {0,0,0,0}, chord_uses[] = {0,0,1,2,3};
    const int64_t loops[] = {0,0,0,0,0};
    const double ranges[] = {0,1, 1,0, 0,1, 0,1};
    const double chords[] = {0,0,.5,0, .5,0,1,0, 1,0,1,1,
                             1,1,0,1, 0,1,0,0};
    double intervals[] = {0,.5, .5,1, 1,0, 0,1, 0,1};
    int64_t n = -1;
    phx_mc_chart_failure failures[32];
    auto call = [&] {
      return phx_mc_verify_chart_partition(1, 3, patches, triangles, 4, uses,
          ranges, 1, 5, chord_uses, loops, chords, intervals, 32, failures, &n);
    };
    CHECK(call() == PHX_MC_OK && n == 0);
    intervals[2] = std::nextafter(.5, 1.0);
    CHECK(call() == PHX_MC_OK && n > 0);
    CHECK(std::any_of(failures, failures + n, [](const auto& e) {
      return e.kind == PHX_MC_CHART_USE_COVERAGE && e.row == 0;
    }));
  }
  // The determinant is exactly 2^-104; rounded product subtraction gives zero.
  {
    constexpr double e = 0x1p-52;
    const int32_t patches[] = {0};
    const int64_t uses[] = {0,0,0}, chord_uses[] = {0,1,2}, loops[] = {0,0,0};
    const double triangles[] = {0,0, 1,1+e, 1-e,1};
    const double chords[] = {0,0,1,1+e, 1,1+e,1-e,1, 1-e,1,0,0};
    const double ranges[] = {0,1, 0,1, 0,1};
    int64_t n = -1;
    phx_mc_chart_failure failures[16];
    CHECK(phx_mc_verify_chart_partition(1, 1, patches, triangles, 3, uses,
        ranges, 1, 3, chord_uses, loops, chords, ranges, 16, failures, &n) == PHX_MC_OK);
    CHECK(n == 0);
  }
  // Overlap (MAC review): two CCW squares [0,1]^2 and [.5,1.5]x[0,1] in one patch pass the
  // orientation and boundary-chain checks, yet [.5,1]x[0,1] is covered twice.
  {
    const int32_t patches[] = {0,0,0,0};
    const double triangles[] = {0,0, 1,0, 1,1,  0,0, 1,1, 0,1,
                                .5,0, 1.5,0, 1.5,1,  .5,0, 1.5,1, .5,1};
    const int64_t uses[] = {0,0,0,0,0,0,0,0}, chord_uses[] = {0,1,2,3,4,5,6,7};
    const int64_t loops[] = {0,0,0,0,1,1,1,1};
    const double ranges[] = {0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1};
    const double chords[] = {0,0,1,0, 1,0,1,1, 1,1,0,1, 0,1,0,0,
                             .5,0,1.5,0, 1.5,0,1.5,1, 1.5,1,.5,1, .5,1,.5,0};
    int64_t n = -1;
    phx_mc_chart_failure failures[32];
    CHECK(phx_mc_verify_chart_partition(1, 4, patches, triangles, 8, uses, ranges, 2, 8,
          chord_uses, loops, chords, ranges, 32, failures, &n) == PHX_MC_OK);
    CHECK(n > 0);
    CHECK(std::all_of(failures, failures + n,
                      [](const auto& e) { return e.kind == PHX_MC_CHART_OVERLAP; }));
    CHECK(std::any_of(failures, failures + n, [](const auto& e) {
      return e.patch == 0 && e.row < e.other && e.row < 2 && e.other >= 2;
    }));
  }
  // A single vertex-simple pentagram loop fanned from its centre winds twice.
  {
    constexpr double pi = 3.14159265358979323846;
    double star[10];
    for (int k = 0; k < 5; ++k) {
      star[2*k] = std::cos(pi/2 + 4*pi*k/5);
      star[2*k+1] = std::sin(pi/2 + 4*pi*k/5);
    }
    int32_t patches[5] = {};
    double triangles[30], chords[20], ranges[10];
    int64_t uses[5] = {}, chord_uses[5], loops[5] = {};
    for (int k = 0; k < 5; ++k) {
      const int j = (k + 1) % 5;
      const double row[6] = {0, 0, star[2*k], star[2*k+1], star[2*j], star[2*j+1]};
      std::copy(row, row + 6, triangles + 6*k);
      const double chord[4] = {star[2*k], star[2*k+1], star[2*j], star[2*j+1]};
      std::copy(chord, chord + 4, chords + 4*k);
      ranges[2*k] = 0; ranges[2*k+1] = 1;
      chord_uses[k] = k;
    }
    int64_t n = -1;
    phx_mc_chart_failure failures[32];
    CHECK(phx_mc_verify_chart_partition(1, 5, patches, triangles, 5, uses, ranges, 1, 5,
          chord_uses, loops, chords, ranges, 32, failures, &n) == PHX_MC_OK);
    CHECK(n > 0);
    CHECK(std::all_of(failures, failures + n,
                      [](const auto& e) { return e.kind == PHX_MC_CHART_OVERLAP; }));
  }
  // Consecutive chords of one use must join bitwise in UV even when their ranges join.
  {
    const int32_t patches[] = {0,0,0};
    const double triangles[] = {0,0, .5,0, 0,1, .5,0, 1,0, 1,1, .5,0, 1,1, 0,1};
    const int64_t uses[] = {0,0,0,0}, chord_uses[] = {0,0,1,2,3}, loops[] = {0,0,0,0,0};
    const double ranges[] = {0,1, 1,0, 0,1, 0,1};
    double chords[] = {0,0,.5,0, .5,0,1,0, 1,0,1,1, 1,1,0,1, 0,1,0,0};
    const double intervals[] = {0,.5, .5,1, 1,0, 0,1, 0,1};
    chords[4] = std::nextafter(.5, 1.0);
    int64_t n = -1;
    phx_mc_chart_failure failures[32];
    CHECK(phx_mc_verify_chart_partition(1, 3, patches, triangles, 4, uses, ranges, 1, 5,
          chord_uses, loops, chords, intervals, 32, failures, &n) == PHX_MC_OK);
    CHECK(std::any_of(failures, failures + n, [](const auto& e) {
      return e.kind == PHX_MC_CHART_USE_COVERAGE && e.row == 0;
    }));
  }
  CHECK(f.run() == PHX_MC_INVALID_INPUT && f.count == 123);
  return phx::mc::test::finish("verify_chart_partition");
}
