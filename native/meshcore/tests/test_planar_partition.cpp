// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Independent integer supporting-line definition against the screened proof.
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {
using Point = std::array<int64_t, 2>;
using Triangle = std::array<Point, 3>;
int64_t cross(const Point& a, const Point& b, const Point& c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}
bool outside(const Triangle& a, const Triangle& b) {
  const int64_t orientation = cross(a[0], a[1], a[2]);
  for (int i = 0; i < 3; ++i) {
    bool all = true;
    for (const Point& p : b) {
      const int64_t side = cross(a[i], a[(i + 1) % 3], p);
      if ((orientation > 0 && side > 0) || (orientation < 0 && side < 0)) all = false;
    }
    if (all) return true;
  }
  return false;
}
bool overlap(const Triangle& a, const Triangle& b) {
  return !outside(a, b) && !outside(b, a);
}
std::array<int64_t, 3> run(const std::vector<Triangle>& rows,
                           const std::vector<Triangle>& declared, int shift = 0) {
  std::vector<uint32_t> words;
  std::vector<int64_t> offsets{0};
  std::vector<int8_t> signs;
  std::vector<double> lower, upper;
  const auto put = [&](int64_t value) {
    const uint64_t magnitude = static_cast<uint64_t>(value < 0 ? -value : value);
    if (magnitude != 0) {
      for (int i = 0; i < shift / 32; ++i) words.push_back(0);
      const uint64_t shifted = magnitude << (shift % 32);
      words.push_back(static_cast<uint32_t>(shifted));
      if (shifted >> 32) words.push_back(static_cast<uint32_t>(shifted >> 32));
    }
    offsets.push_back(static_cast<int64_t>(words.size()));
  };
  for (const auto* bank : {&rows, &declared}) {
    for (const Triangle& triangle : *bank) {
      for (int axis = 0; axis < 2; ++axis) {
        int64_t lo = triangle[0][axis], hi = lo;
        for (const Point& point : triangle) { lo = std::min(lo, point[axis]); hi = std::max(hi, point[axis]); }
        lower.push_back(static_cast<double>(lo)); upper.push_back(static_cast<double>(hi));
      }
      for (const Point& point : triangle) for (int64_t value : point) {
        signs.push_back(value < 0 ? -1 : 1); put(value);
      }
    }
  }
  put(1);
  int32_t verdict = -1;
  int64_t first = -1, second = -1;
  PHX_CHECK(phx_mc_planar_partition(static_cast<int64_t>(rows.size()),
      static_cast<int64_t>(declared.size()), static_cast<int64_t>(words.size()),
      words.data(), offsets.data(), signs.data(), lower.data(), upper.data(),
      &verdict, &first, &second) == PHX_MC_OK);
  return {verdict, first, second};
}
void test_generated() {
  std::mt19937 generator(1979);
  for (int trial = 0; trial < 600; ++trial) {
    std::vector<Triangle> triangles;
    while (triangles.size() < 4) {
      Triangle t{};
      for (Point& p : t) for (int64_t& x : p) x = static_cast<int64_t>(generator() % 9) - 4;
      if (cross(t[0], t[1], t[2]) != 0) triangles.push_back(t);
    }
    int64_t first = -1;
    for (std::size_t i = 0; i < triangles.size(); ++i) {
      for (std::size_t j = 0; j < triangles.size(); ++j) {
        if (i != j && overlap(triangles[i], triangles[j])) {
          first = static_cast<int64_t>(i); break;
        }
      }
      if (first >= 0) break;
    }
    const std::array<int64_t, 3> expected = first < 0 ? std::array<int64_t, 3>{0, -1, -1}
                                                        : std::array<int64_t, 3>{2, first, -1};
    PHX_CHECK(run(triangles, triangles) == expected);
  }
}
void test_touching_and_overlap() {
  const Triangle a{{Point{0, 0}, Point{2, 0}, Point{0, 2}}};
  const Triangle b{{Point{2, 2}, Point{0, 2}, Point{2, 0}}};
  const Triangle c{{Point{0, 0}, Point{2, 0}, Point{2, 2}}};
  const std::array<int64_t, 3> good{0, -1, -1}, bad{3, 0, 1};
  PHX_CHECK(run({a, b}, {a, b}) == good);
  PHX_CHECK(run({a, c}, {a, b}) == bad);
  PHX_CHECK(run({a, c}, {a, b}, 1152) == bad);
}
}  // namespace
int main() { test_touching_and_overlap(); test_generated(); }
