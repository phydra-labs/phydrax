// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Independent integer supporting-line definition against the screened proof.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
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
struct Fraction {
  int64_t n;
  int64_t d;
  Fraction(int64_t numerator = 0, int64_t denominator = 1) : n(numerator), d(denominator) {
    PHX_CHECK(d != 0);
    if (d < 0) { n = -n; d = -d; }
    const int64_t divisor = std::gcd(n < 0 ? -n : n, d);
    n /= divisor; d /= divisor;
  }
};
Fraction operator+(Fraction a, Fraction b) { return {a.n * b.d + b.n * a.d, a.d * b.d}; }
Fraction operator-(Fraction a, Fraction b) { return {a.n * b.d - b.n * a.d, a.d * b.d}; }
Fraction operator*(Fraction a, Fraction b) { return {a.n * b.n, a.d * b.d}; }
Fraction operator/(Fraction a, Fraction b) { return {a.n * b.d, a.d * b.n}; }
bool operator==(Fraction a, Fraction b) { return a.n == b.n && a.d == b.d; }
struct RationalPoint { Fraction x, y; };
Fraction orient(RationalPoint a, RationalPoint b, RationalPoint c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
Fraction area(const std::vector<RationalPoint>& polygon) {
  Fraction total;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const auto& a = polygon[i];
    const auto& b = polygon[(i + 1) % polygon.size()];
    total = total + a.x * b.y - a.y * b.x;
  }
  if (total.n < 0) total.n = -total.n;
  return total;
}
std::vector<RationalPoint> points(const Triangle& triangle) {
  std::vector<RationalPoint> out;
  for (const Point& p : triangle) out.push_back({Fraction(p[0]), Fraction(p[1])});
  return out;
}
Fraction intersection(const Triangle& subject, const Triangle& clip) {
  std::vector<RationalPoint> polygon = points(subject);
  const auto boundary = points(clip);
  const int64_t sign = orient(boundary[0], boundary[1], boundary[2]).n;
  if (sign == 0) return {};
  for (int edge = 0; edge < 3 && !polygon.empty(); ++edge) {
    const auto a = boundary[edge];
    const auto b = boundary[(edge + 1) % 3];
    std::vector<RationalPoint> output;
    auto previous = polygon.back();
    Fraction old = orient(a, b, previous) * Fraction(sign);
    for (const auto& point : polygon) {
      const Fraction current = orient(a, b, point) * Fraction(sign);
      if ((old.n >= 0) != (current.n >= 0)) {
        const Fraction parameter = old / (old - current);
        output.push_back({previous.x + parameter * (point.x - previous.x),
                          previous.y + parameter * (point.y - previous.y)});
      }
      if (current.n >= 0) output.push_back(point);
      previous = point;
      old = current;
    }
    polygon = std::move(output);
  }
  return area(polygon);
}
std::array<int64_t, 3> reference(const std::vector<Triangle>& rows,
                                 const std::vector<Triangle>& declared) {
  Fraction target_area, source_area;
  for (const auto& t : rows) target_area = target_area + area(points(t));
  for (const auto& t : declared) source_area = source_area + area(points(t));
  if (declared.empty() || !(target_area == source_area)) return {1, -1, -1};
  for (std::size_t row = 0; row < rows.size(); ++row) {
    Fraction covered;
    for (const auto& source : declared) covered = covered + intersection(rows[row], source);
    if (!(covered == area(points(rows[row])))) return {2, static_cast<int64_t>(row), -1};
  }
  for (std::size_t first = 0; first < rows.size(); ++first)
    for (std::size_t second = first + 1; second < rows.size(); ++second)
      if (intersection(rows[first], rows[second]).n != 0)
        return {3, static_cast<int64_t>(first), static_cast<int64_t>(second)};
  return {0, -1, -1};
}
std::array<int64_t, 3> run(const std::vector<Triangle>& rows,
                           const std::vector<Triangle>& declared, int shift = 0,
                           int64_t denominator = 1, bool bad_bounds = false) {
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
        const double infinity = std::numeric_limits<double>::infinity();
        lower.push_back(denominator == 1 ? static_cast<double>(lo) : -infinity);
        upper.push_back(denominator == 1 ? static_cast<double>(hi) : infinity);
      }
      for (const Point& point : triangle) for (int64_t value : point) {
        signs.push_back(value < 0 ? -1 : 1); put(value);
      }
    }
  }
  put(denominator);
  if (bad_bounds) {
    lower[0] = upper[0] = lower[1] = upper[1] = 0;
    lower[2] = upper[2] = lower[3] = upper[3] = 1;
    for (std::size_t i = 4; i < lower.size(); ++i) {
      lower[i] = -std::numeric_limits<double>::infinity();
      upper[i] = std::numeric_limits<double>::infinity();
    }
  }
  int32_t verdict = -1;
  int64_t first = -1, second = -1;
  const int32_t status = phx_mc_planar_partition(static_cast<int64_t>(rows.size()),
      static_cast<int64_t>(declared.size()), static_cast<int64_t>(words.size()),
      words.data(), offsets.data(), signs.data(), lower.data(), upper.data(),
      &verdict, &first, &second);
  if (bad_bounds) {
    PHX_CHECK(status == PHX_MC_INVALID_ARGUMENT);
  } else {
    PHX_CHECK(status == PHX_MC_OK);
  }
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
  run({a, c}, {a, b}, 0, 1, true);
  PHX_CHECK(run({a, c}, {a, b}, 120) == bad);
  PHX_CHECK(run({a, c}, {a, b}, 127) == bad);
  PHX_CHECK(run({a, c}, {a, b}, 1152) == bad);
}
void test_distinct_banks() {
  const Triangle a{{Point{0, 0}, Point{4, 0}, Point{0, 4}}};
  const Triangle b{{Point{4, 4}, Point{0, 4}, Point{4, 0}}};
  const Triangle c{{Point{0, 0}, Point{4, 0}, Point{4, 4}}};
  const Triangle d{{Point{0, 0}, Point{4, 4}, Point{0, 4}}};
  const Triangle outside_overlap{{Point{1, 0}, Point{5, 0}, Point{1, 4}}};
  const std::vector<Triangle> declared{a, b};
  for (const std::vector<Triangle> rows :
       {std::vector<Triangle>{c, d}, std::vector<Triangle>{a, c},
        std::vector<Triangle>{a, outside_overlap}}) {
    const auto expected = reference(rows, declared);
    PHX_CHECK(run(rows, declared) == expected);
    PHX_CHECK(run(rows, declared, 0, 3) == expected);
    PHX_CHECK(run(rows, declared, 0, int64_t{1} << 33) == expected);
  }
  const std::array<int64_t, 3> earlier_coverage{2, 1, -1};
  PHX_CHECK(reference({a, outside_overlap}, declared) == earlier_coverage);
  std::mt19937 generator(10291);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<Triangle> source;
    while (source.size() < 2) {
      Triangle triangle{};
      for (Point& p : triangle)
        for (int64_t& x : p) x = static_cast<int64_t>(generator() % 5) - 2;
      if (cross(triangle[0], triangle[1], triangle[2]) != 0) source.push_back(triangle);
    }
    auto rows = source;
    for (Point& p : rows[trial % 2]) p[0] += 1;
    PHX_CHECK(run(rows, source, 0, trial % 2 == 0 ? 3 : 1) == reference(rows, source));
  }
}

// Translate all coordinates by 2^128. Products overflow signed 256-bit while
// their cancellations leave the same small determinants as the input fixture.
void test_high_offset_low_sign_bits() {
  const std::array<Triangle, 4> bank{{
      Triangle{{Point{0, 0}, Point{2, 0}, Point{0, 2}}},
      Triangle{{Point{0, 0}, Point{2, 0}, Point{2, 2}}},
      Triangle{{Point{0, 0}, Point{2, 0}, Point{0, 2}}},
      Triangle{{Point{2, 2}, Point{0, 2}, Point{2, 0}}},
  }};
  std::vector<uint32_t> words;
  std::vector<int64_t> offsets{0};
  std::vector<int8_t> signs;
  for (const auto& triangle : bank) for (const auto& point : triangle) {
    for (const int64_t value : point) {
      words.push_back(static_cast<uint32_t>(value));
      words.push_back(0); words.push_back(0); words.push_back(0); words.push_back(1);
      offsets.push_back(static_cast<int64_t>(words.size()));
      signs.push_back(1);
    }
  }
  words.push_back(1);
  offsets.push_back(static_cast<int64_t>(words.size()));
  std::vector<double> lower(8, -std::numeric_limits<double>::infinity());
  std::vector<double> upper(8, std::numeric_limits<double>::infinity());
  int32_t verdict = -1;
  int64_t first = -1, second = -1;
  PHX_CHECK(phx_mc_planar_partition(2, 2, static_cast<int64_t>(words.size()), words.data(),
      offsets.data(), signs.data(), lower.data(), upper.data(), &verdict, &first, &second) ==
      PHX_MC_OK);
  PHX_CHECK(verdict == 3 && first == 0 && second == 1);
}
// The true determinant is 2^255 + 2^127. A signed 256-bit wrapped
// determinant is negative even though the exact orientation is positive.
void test_overflowed_predicate_sign() {
  const std::array<std::array<int, 6>, 4> bank{{
      {{0, 0, 1, 0, 0, 1}}, {{0, 0, 1, 0, 1, 1}},
      {{0, 0, 1, 0, 0, 1}}, {{1, 1, 0, 1, 1, 0}},
  }};
  std::vector<uint32_t> words;
  std::vector<int64_t> offsets{0};
  std::vector<int8_t> signs;
  for (const auto& triangle : bank) {
    for (std::size_t coordinate = 0; coordinate < triangle.size(); ++coordinate) {
      const bool positive = triangle[coordinate] != 0;
      words.push_back(positive && coordinate % 2 == 1 ? 1 : 0);
      words.push_back(0); words.push_back(0);
      words.push_back(positive && coordinate % 2 == 0 ? uint32_t{1} << 31 : 0);
      words.push_back(positive && coordinate % 2 == 1 ? 1 : 0);
      offsets.push_back(static_cast<int64_t>(words.size()));
      signs.push_back(1);
    }
  }
  words.push_back(1);
  offsets.push_back(static_cast<int64_t>(words.size()));
  std::vector<double> lower(8, -std::numeric_limits<double>::infinity());
  std::vector<double> upper(8, std::numeric_limits<double>::infinity());
  int32_t verdict = -1;
  int64_t first = -1, second = -1;
  PHX_CHECK(phx_mc_planar_partition(2, 2, static_cast<int64_t>(words.size()), words.data(),
      offsets.data(), signs.data(), lower.data(), upper.data(), &verdict, &first, &second) ==
      PHX_MC_OK);
  PHX_CHECK(verdict == 3 && first == 0 && second == 1);
}
}  // namespace
int main() {
  test_touching_and_overlap();
  test_generated();
  test_distinct_banks();
  test_high_offset_low_sign_bits();
  test_overflowed_predicate_sign();
  return phx::mc::test::finish("test_planar_partition");
}
