//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact proofs of vertex-defined polyhedral cells on rational vertices:
//
// * phx_mc_polyhedral_star_validity reproduces the rational star proof of
//   phydrax `_cell_geometry_source_validity._polyhedron_proof`: the signed
//   volumes of the tetrahedra (cell vertex average, face vertex average, edge)
//   of every oriented face edge, their floor tests, the directed binary64
//   endpoints of their extremes and the total-volume refusal.
// * phx_mc_polyhedral_star_decomposition reproduces the exact star
//   decomposition of phydrax `_global_embedding._volume_tetrahedra`: coherent
//   outward face orientation, exact face planes, the vertex-average star point,
//   the exact ear-clipping triangulation of every face and the degree of the
//   radial projection along the same generic ray family.
//
// A cell arrives as integer points X = L x over one positive common
// denominator L of its rational vertices x (and, for the decomposition, every
// face again over the common denominator of its own vertices). Each decision
// is the sign of a polynomial homogeneous in the points, hence decided on the
// integer rows; the validity volumes carry L^3 into their exact quotients.
// Signs are first evaluated in outward-rounded binary64 interval arithmetic
// on the exact rational coordinates; a sign the enclosure does not decide is
// recomputed in exact big-integer arithmetic (Dyadic with nonnegative
// exponents), as is every equality statement (planarity, zero area).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

namespace {

using phx::mc::Dyadic;
using Vec = std::array<Dyadic, 3>;

// Vertex and face counts stay exact binary64 integers through m n^3 and the
// least common multiple of the face sizes.
constexpr int64_t kMaximumCount = 1 << 12;
constexpr double kMaximumMultiple = 0x1p40;
constexpr int kUnknown = 2;

// ---------------------------------------------------------------------------
// Exact integer rows.

// Signed integers of a word bank: value v is (-1)^(signs[v] < 0) times the
// little-endian 32-bit words [word_offsets[v], word_offsets[v + 1]).
struct IntegerBank {
  const int64_t* offsets;
  const uint32_t* words;
  const int8_t* signs;

  Dyadic value(int64_t index) const {
    const int64_t start = offsets[index];
    return Dyadic::integer(words + start, static_cast<std::size_t>(offsets[index + 1] - start),
                           signs[index]);
  }
  Vec point(int64_t index) const {
    return {value(3 * index), value(3 * index + 1), value(3 * index + 2)};
  }
};

bool valid_bank(int64_t count, const int64_t* offsets, const uint32_t* words, const int8_t* signs) {
  if (count < 0 || offsets == nullptr || offsets[0] != 0 || (count > 0 && signs == nullptr)) {
    return false;
  }
  for (int64_t value = 0; value < count; ++value) {
    if (offsets[value + 1] < offsets[value]) return false;
  }
  return offsets[count] == 0 || words != nullptr;
}

Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec add(const Vec& a, const Vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec scale(const Vec& a, const Dyadic& b) { return {a[0] * b, a[1] * b, a[2] * b}; }
Dyadic dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
bool nonzero(const Vec& a) { return a[0].sign() || a[1].sign() || a[2].sign(); }
int compare(const Dyadic& a, const Dyadic& b) { return (a - b).sign(); }
Dyadic magnitude(const Dyadic& a) { return a.sign() < 0 ? -a : a; }
Dyadic count_value(int64_t count) { return Dyadic(static_cast<double>(count)); }

// Index of the first largest |normal| component (Python's max over axes).
int dominant_axis(const Vec& normal) {
  int best = 0;
  for (int axis = 1; axis < 3; ++axis) {
    if (compare(magnitude(normal[axis]), magnitude(normal[best])) > 0) best = axis;
  }
  return best;
}

void projection_axes(int dropped, int& u, int& v) {
  u = dropped == 0 ? 1 : 0;
  v = dropped == 2 ? 1 : 2;
}

// ---------------------------------------------------------------------------
// Outward-rounded binary64 enclosures.

struct Interval {
  double lo = 0.0;
  double hi = 0.0;
};
using Box = std::array<Interval, 3>;

double down(double value) { return std::nextafter(value, -std::numeric_limits<double>::infinity()); }
double up(double value) { return std::nextafter(value, std::numeric_limits<double>::infinity()); }

Interval operator+(const Interval& a, const Interval& b) {
  return {down(a.lo + b.lo), up(a.hi + b.hi)};
}
Interval operator-(const Interval& a, const Interval& b) {
  return {down(a.lo - b.hi), up(a.hi - b.lo)};
}
Interval operator*(const Interval& a, const Interval& b) {
  const double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  return {down(std::min({p[0], p[1], p[2], p[3]})), up(std::max({p[0], p[1], p[2], p[3]}))};
}
Interval exact_interval(double value) { return {value, value}; }

// The sign the enclosure proves, or kUnknown (a non-finite bound proves none).
int sign_of(const Interval& value) {
  if (!(std::isfinite(value.lo) && std::isfinite(value.hi))) return kUnknown;
  if (value.lo > 0.0) return 1;
  if (value.hi < 0.0) return -1;
  return kUnknown;
}

Box sub(const Box& a, const Box& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Box add(const Box& a, const Box& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Box scale(const Box& a, const Interval& b) { return {a[0] * b, a[1] * b, a[2] * b}; }
Interval dot(const Box& a, const Box& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Box cross(const Box& a, const Box& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Interval quotient of a positive-denominator enclosure.
Interval divide(const Interval& a, const Interval& b) {
  const double p[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
  return {down(std::min({p[0], p[1], p[2], p[3]})), up(std::max({p[0], p[1], p[2], p[3]}))};
}

// Exact cell rows and the enclosures of the rational points X / L.
struct CellPoints {
  std::vector<Vec> exact;
  std::vector<Box> boxes;
  Dyadic common;
};

void load_cell(const IntegerBank& bank, int64_t first, int64_t count, int64_t common_index,
               CellPoints& cell) {
  cell.exact.clear();
  cell.boxes.clear();
  cell.common = bank.value(common_index);
  for (int64_t index = first; index < first + count; ++index) {
    cell.exact.push_back(bank.point(index));
    Box box;
    for (int axis = 0; axis < 3; ++axis) {
      const auto bounds = phx::mc::exact::directed_quotient_bounds(cell.exact.back()[axis], cell.common);
      box[axis] = {bounds.lower, bounds.upper};
    }
    cell.boxes.push_back(box);
  }
}

bool valid_cells(int64_t cell_count, const int64_t* point_offsets, const int64_t* cell_face_offsets,
                 const int64_t* face_vertex_offsets, const int64_t* face_vertex_values) {
  if (point_offsets[0] != 0 || cell_face_offsets[0] != 0 || face_vertex_offsets[0] != 0) {
    return false;
  }
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    if (point_offsets[cell + 1] < point_offsets[cell] ||
        cell_face_offsets[cell + 1] < cell_face_offsets[cell]) {
      return false;
    }
    const int64_t local = point_offsets[cell + 1] - point_offsets[cell];
    for (int64_t face = cell_face_offsets[cell]; face < cell_face_offsets[cell + 1]; ++face) {
      if (face_vertex_offsets[face + 1] < face_vertex_offsets[face]) return false;
      for (int64_t entry = face_vertex_offsets[face]; entry < face_vertex_offsets[face + 1];
           ++entry) {
        if (face_vertex_values[entry] < 0 || face_vertex_values[entry] >= local) return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Star validity.

struct ExactStar {
  Dyadic numerator;  // D: volume D / (m n^3 L^3)
  Dyadic scale;      // S: scale product S / (m n^3 L^3)^2
};

ExactStar exact_star(const CellPoints& cell, const Vec& center_sum, const Dyadic& n,
                     const Vec& face_sum, const Dyadic& face_size, int64_t first,
                     int64_t second) {
  const Vec a = sub(scale(face_sum, n), scale(center_sum, face_size));
  const Vec b = sub(scale(cell.exact[static_cast<std::size_t>(first)], n), center_sum);
  const Vec c = sub(scale(cell.exact[static_cast<std::size_t>(second)], n), center_sum);
  return {dot(a, cross(b, c)), dot(a, a) * dot(b, b) * dot(c, c)};
}

}  // namespace

extern "C" int32_t phx_mc_polyhedral_star_validity(
    int64_t cell_count, const int64_t* point_offsets, const int64_t* word_offsets,
    const uint32_t* words, const int8_t* signs, const int64_t* cell_face_offsets,
    const int64_t* face_vertex_offsets, const int64_t* face_vertex_values,
    const int64_t* star_offsets, const int64_t* stars, double floor, int8_t* decided,
    int32_t* status, double* lower, double* upper) {
  return phx::mc::guarded([&]() -> int32_t {
    if (cell_count < 0 || point_offsets == nullptr || cell_face_offsets == nullptr ||
        face_vertex_offsets == nullptr || star_offsets == nullptr || !(floor >= 0.0) ||
        !std::isfinite(floor) ||
        (cell_count > 0 &&
         (decided == nullptr || status == nullptr || lower == nullptr || upper == nullptr)) ||
        !valid_bank(3 * point_offsets[cell_count] + cell_count, word_offsets, words, signs) ||
        (face_vertex_offsets[cell_face_offsets[cell_count]] > 0 && face_vertex_values == nullptr) ||
        star_offsets[0] != 0 || (star_offsets[cell_count] > 0 && stars == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!valid_cells(cell_count, point_offsets, cell_face_offsets, face_vertex_offsets,
                     face_vertex_values)) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      if (star_offsets[cell + 1] < star_offsets[cell]) return PHX_MC_INVALID_INPUT;
      const int64_t local = point_offsets[cell + 1] - point_offsets[cell];
      const int64_t faces = cell_face_offsets[cell + 1] - cell_face_offsets[cell];
      for (int64_t star = star_offsets[cell]; star < star_offsets[cell + 1]; ++star) {
        if (stars[3 * star] < 0 || stars[3 * star] >= faces || stars[3 * star + 1] < 0 ||
            stars[3 * star + 1] >= local || stars[3 * star + 2] < 0 ||
            stars[3 * star + 2] >= local) {
          return PHX_MC_INVALID_INPUT;
        }
      }
    }
    const phx::mc::GradualNearestScope mode;
    const IntegerBank bank{word_offsets, words, signs};
    const int64_t scale_base = 3 * point_offsets[cell_count];
    const Dyadic floor_value(floor);
    const Dyadic floor_squared = floor_value * floor_value;
    const Interval floor_interval = exact_interval(floor);
    const Interval floor_square_interval = floor_interval * floor_interval;
    CellPoints cell_points;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      decided[cell] = 0;
      const int64_t first_point = point_offsets[cell];
      const int64_t vertex_total = point_offsets[cell + 1] - first_point;
      const int64_t first_face = cell_face_offsets[cell];
      const int64_t first_star = star_offsets[cell];
      const int64_t star_total = star_offsets[cell + 1] - first_star;
      if (star_total == 0) {
        status[cell] = 2;
        lower[cell] = std::numeric_limits<double>::quiet_NaN();
        upper[cell] = std::numeric_limits<double>::quiet_NaN();
        decided[cell] = 1;
        continue;
      }
      if (vertex_total < 1 || vertex_total > kMaximumCount) continue;
      double multiple = 1.0;
      bool representable = true;
      for (int64_t star = first_star; star < first_star + star_total && representable; ++star) {
        const int64_t face = first_face + stars[3 * star];
        const int64_t size = face_vertex_offsets[face + 1] - face_vertex_offsets[face];
        representable = size >= 1 && size <= kMaximumCount;
        if (representable) {
          multiple = static_cast<double>(std::lcm(static_cast<int64_t>(multiple), size));
          representable = multiple <= kMaximumMultiple;
        }
      }
      if (!representable) continue;
      load_cell(bank, first_point, vertex_total, scale_base + cell, cell_points);
      const double count = static_cast<double>(vertex_total);
      Box center;
      for (const Box& box : cell_points.boxes) center = add(center, box);
      center = {divide(center[0], exact_interval(count)), divide(center[1], exact_interval(count)),
                divide(center[2], exact_interval(count))};
      // Enclosures of every star volume v_r and of v_r^2 - floor^2 w_r.
      std::map<int64_t, Box> face_centers;
      std::vector<Interval> volumes(static_cast<std::size_t>(star_total));
      bool enclosed_positive = true;
      for (int64_t row = 0; row < star_total; ++row) {
        const int64_t star = first_star + row;
        const int64_t face = first_face + stars[3 * star];
        auto found = face_centers.find(face);
        if (found == face_centers.end()) {
          Box sum;
          for (int64_t entry = face_vertex_offsets[face]; entry < face_vertex_offsets[face + 1];
               ++entry) {
            sum = add(sum, cell_points.boxes[static_cast<std::size_t>(face_vertex_values[entry])]);
          }
          const Interval size = exact_interval(
              static_cast<double>(face_vertex_offsets[face + 1] - face_vertex_offsets[face]));
          found = face_centers
                      .emplace(face, Box{divide(sum[0], size), divide(sum[1], size),
                                         divide(sum[2], size)})
                      .first;
        }
        const Box a = sub(found->second, center);
        const Box b = sub(cell_points.boxes[static_cast<std::size_t>(stars[3 * star + 1])], center);
        const Box c = sub(cell_points.boxes[static_cast<std::size_t>(stars[3 * star + 2])], center);
        const Interval volume = dot(a, cross(b, c));
        volumes[static_cast<std::size_t>(row)] = volume;
        if (enclosed_positive) {
          enclosed_positive =
              sign_of(volume) == 1 &&
              (floor == 0.0 ||
               sign_of(volume * volume -
                       floor_square_interval * (dot(a, a) * dot(b, b) * dot(c, c))) == 1);
        }
      }
      const Dyadic n = count_value(vertex_total);
      const Dyadic cube = n * n * n * cell_points.common * cell_points.common * cell_points.common;
      Vec center_sum;
      for (const Vec& point : cell_points.exact) center_sum = add(center_sum, point);
      std::map<int64_t, std::pair<Vec, Dyadic>> face_sums;
      const auto face_sum = [&](int64_t face) -> const std::pair<Vec, Dyadic>& {
        auto found = face_sums.find(face);
        if (found != face_sums.end()) return found->second;
        Vec sum;
        for (int64_t entry = face_vertex_offsets[face]; entry < face_vertex_offsets[face + 1];
             ++entry) {
          sum = add(sum, cell_points.exact[static_cast<std::size_t>(face_vertex_values[entry])]);
        }
        return face_sums
            .emplace(face, std::make_pair(std::move(sum),
                                          count_value(face_vertex_offsets[face + 1] -
                                                      face_vertex_offsets[face])))
            .first->second;
      };
      std::vector<ExactStar> exact(static_cast<std::size_t>(star_total));
      std::vector<int8_t> computed(static_cast<std::size_t>(star_total), 0);
      const auto star_value = [&](int64_t row) -> const ExactStar& {
        auto& value = exact[static_cast<std::size_t>(row)];
        if (!computed[static_cast<std::size_t>(row)]) {
          const int64_t star = first_star + row;
          const auto& [sum, size] = face_sum(first_face + stars[3 * star]);
          value = exact_star(cell_points, center_sum, n, sum, size, stars[3 * star + 1],
                             stars[3 * star + 2]);
          computed[static_cast<std::size_t>(row)] = 1;
        }
        return value;
      };
      const auto size_of = [&](int64_t row) {
        const int64_t face = first_face + stars[3 * (first_star + row)];
        return face_vertex_offsets[face + 1] - face_vertex_offsets[face];
      };
      // D_r / m_r against D_s / m_s (common n^3 L^3).
      const auto exact_compare = [&](int64_t left, int64_t right) {
        return compare(star_value(left).numerator * count_value(size_of(right)),
                       star_value(right).numerator * count_value(size_of(left)));
      };
      // The extremes: only stars whose enclosures reach the extreme enclosure
      // are compared exactly.
      double least_upper = std::numeric_limits<double>::infinity();
      double greatest_lower = -std::numeric_limits<double>::infinity();
      for (const Interval& volume : volumes) {
        least_upper = std::min(least_upper, volume.hi);
        greatest_lower = std::max(greatest_lower, volume.lo);
      }
      int64_t smallest = -1;
      int64_t largest = -1;
      for (int64_t row = 0; row < star_total; ++row) {
        const Interval& volume = volumes[static_cast<std::size_t>(row)];
        if (!(volume.lo > least_upper) &&
            (smallest < 0 || exact_compare(row, smallest) < 0)) {
          smallest = row;
        }
        if (!(volume.hi < greatest_lower) &&
            (largest < 0 || exact_compare(row, largest) > 0)) {
          largest = row;
        }
      }
      lower[cell] = phx::mc::exact::directed_quotient_bounds(
                        star_value(smallest).numerator, count_value(size_of(smallest)) * cube)
                        .lower;
      upper[cell] = phx::mc::exact::directed_quotient_bounds(
                        star_value(largest).numerator, count_value(size_of(largest)) * cube)
                        .upper;
      bool positive = enclosed_positive;
      if (!positive) {
        positive = true;
        for (int64_t row = 0; row < star_total && positive; ++row) {
          const ExactStar& value = star_value(row);
          positive = value.numerator.sign() > 0 &&
                     (floor == 0.0 ||
                      (value.numerator * value.numerator - floor_squared * value.scale).sign() > 0);
        }
      }
      if (positive) {
        status[cell] = 0;
        decided[cell] = 1;
        continue;
      }
      // Total volume T / (M n^3 L^3) over the common face-size multiple M;
      // the widest scale product S_w / m_w^2 is compared exactly.
      Dyadic total;
      int64_t widest = 0;
      for (int64_t row = 0; row < star_total; ++row) {
        const ExactStar& value = star_value(row);
        total = total + value.numerator * Dyadic(multiple / static_cast<double>(size_of(row)));
        const Dyadic row_size = count_value(size_of(row));
        const Dyadic widest_size = count_value(size_of(widest));
        if (compare(value.scale * widest_size * widest_size,
                    star_value(widest).scale * row_size * row_size) > 0) {
          widest = row;
        }
      }
      bool below = total.sign() < 0;
      const ExactStar& wide = star_value(widest);
      if (floor > 0.0 && wide.scale.sign() > 0) {
        if (total.sign() <= 0) {
          below = true;
        } else {
          const Dyadic widest_size = count_value(size_of(widest));
          const Dyadic common_multiple(multiple);
          below = compare(total * total * widest_size * widest_size,
                          floor_squared * wide.scale * common_multiple * common_multiple) < 0;
        }
      }
      status[cell] = below ? 1 : 2;
      decided[cell] = 1;
    }
    return PHX_MC_OK;
  });
}

namespace {

// ---------------------------------------------------------------------------
// Star decomposition.

enum DecompositionCode : int32_t {
  kDecomposed = 0,
  kNoFaces = 1,
  kFaceIncidence = 2,
  kNotClosedManifold = 3,
  kNotOrientable = 4,
  kZeroVolume = 5,
  kFaceZeroArea = 6,
  kFaceFanFolded = 7,
  kPolygonVertices = 8,
  kPolygonDegenerate = 9,
  kPolygonNotCoplanar = 10,
  kPolygonZeroArea = 11,
  kPolygonSelfContact = 12,
  kPolygonNoEar = 13,
  kPolygonDegenerateFinal = 14,
  kStarDegree = 15,
  kFaceFanWinding = 16,
  kUndecided = -1,
};

using Point2 = std::array<Dyadic, 2>;

int orient2(const Point2& a, const Point2& b, const Point2& c) {
  return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).sign();
}

bool between(const Dyadic& value, const Dyadic& first, const Dyadic& second) {
  const bool ordered = compare(first, second) <= 0;
  const Dyadic& low = ordered ? first : second;
  const Dyadic& high = ordered ? second : first;
  return compare(low, value) <= 0 && compare(value, high) <= 0;
}

// Closed coplanar segments meet (`segment_contact_points3` is nonempty); the
// projection drops a nonzero normal axis, so it is injective on their plane.
bool segments_meet(const Point2& p, const Point2& q, const Point2& r, const Point2& s) {
  const int o1 = orient2(p, q, r);
  const int o2 = orient2(p, q, s);
  const int o3 = orient2(r, s, p);
  const int o4 = orient2(r, s, q);
  const auto on = [](const Point2& a, const Point2& b, const Point2& c) {
    return between(c[0], a[0], b[0]) && between(c[1], a[1], b[1]);
  };
  if (o1 == 0 && on(p, q, r)) return true;
  if (o2 == 0 && on(p, q, s)) return true;
  if (o3 == 0 && on(r, s, p)) return true;
  if (o4 == 0 && on(r, s, q)) return true;
  return o1 * o2 < 0 && o3 * o4 < 0;
}

bool point_in_triangle2(const Point2& p, const Point2& a, const Point2& b, const Point2& c) {
  const int s0 = orient2(a, b, p);
  const int s1 = orient2(b, c, p);
  const int s2 = orient2(c, a, p);
  return !((s0 > 0 || s1 > 0 || s2 > 0) && (s0 < 0 || s1 < 0 || s2 < 0));
}

// `triangulate_planar_polygon` of one face loop (exact integer rows over any
// positive common scale): local index triangles in the rational proof's
// order, or its refusal. This is the single face-triangle source of the
// decomposition.
int32_t face_triangles(const std::vector<Vec>& loop,
                       std::vector<std::array<int32_t, 3>>& triangles) {
  const std::size_t count = loop.size();
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = first + 1; second < count; ++second) {
      if (!nonzero(sub(loop[first], loop[second]))) return kPolygonVertices;
    }
  }
  if (count < 3) return kPolygonVertices;
  const Vec& origin = loop[0];
  Vec normal;
  bool found = false;
  for (std::size_t first = 1; first < count && !found; ++first) {
    const Vec a = sub(loop[first], origin);
    for (std::size_t second = first + 1; second < count; ++second) {
      Vec candidate = cross(a, sub(loop[second], origin));
      if (nonzero(candidate)) {
        normal = std::move(candidate);
        found = true;
        break;
      }
    }
  }
  if (!found) return kPolygonDegenerate;
  for (const Vec& vertex : loop) {
    if (dot(normal, sub(vertex, origin)).sign()) return kPolygonNotCoplanar;
  }
  int u = 0, v = 0;
  projection_axes(dominant_axis(normal), u, v);
  std::vector<Point2> projected;
  projected.reserve(count);
  for (const Vec& vertex : loop) projected.push_back({vertex[u], vertex[v]});
  Dyadic area;
  for (std::size_t index = 0; index < count; ++index) {
    const Point2& a = projected[index];
    const Point2& b = projected[(index + 1) % count];
    area = area + (a[0] * b[1] - a[1] * b[0]);
  }
  const int orientation = area.sign();
  if (orientation == 0) return kPolygonZeroArea;
  for (std::size_t first = 0; first < count; ++first) {
    const std::size_t first_next = (first + 1) % count;
    for (std::size_t second = first + 1; second < count; ++second) {
      const std::size_t second_next = (second + 1) % count;
      if (first_next == second || second_next == first) continue;
      if (segments_meet(projected[first], projected[first_next], projected[second],
                        projected[second_next])) {
        return kPolygonSelfContact;
      }
    }
  }
  std::vector<int32_t> remaining(count);
  std::iota(remaining.begin(), remaining.end(), 0);
  while (remaining.size() > 3) {
    bool ear_found = false;
    const std::size_t size = remaining.size();
    for (std::size_t position = 0; position < size; ++position) {
      const int32_t previous = remaining[(position + size - 1) % size];
      const int32_t current = remaining[position];
      const int32_t following = remaining[(position + 1) % size];
      if (orient2(projected[previous], projected[current], projected[following]) !=
          orientation) {
        continue;
      }
      bool blocked = false;
      for (const int32_t index : remaining) {
        if (index == previous || index == current || index == following) continue;
        if (point_in_triangle2(projected[index], projected[previous], projected[current],
                               projected[following])) {
          blocked = true;
          break;
        }
      }
      if (blocked) continue;
      triangles.push_back({previous, current, following});
      remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(position));
      ear_found = true;
      break;
    }
    if (!ear_found) return kPolygonNoEar;
  }
  if (orient2(projected[remaining[0]], projected[remaining[1]], projected[remaining[2]]) == 0) {
    return kPolygonDegenerateFinal;
  }
  triangles.push_back({remaining[0], remaining[1], remaining[2]});
  return kDecomposed;
}

bool on_segment(const Vec& point, const Vec& first, const Vec& second) {
  const Vec direction = sub(second, first);
  if (nonzero(cross(direction, sub(point, first)))) return false;
  for (int axis = 0; axis < 3; ++axis) {
    if (direction[axis].sign()) return between(point[axis], first[axis], second[axis]);
  }
  return !nonzero(sub(point, first));
}

enum class RayHit { kMiss, kCross, kDegenerate };

// One triangle of `_star_ray_degree` in exact arithmetic. The corners are
// a, b, c / weight (weight > 0; a fan apex is its face's vertex sum over the
// face size) and the star point center_sum / n; everything is scaled by
// weight * n * denominator > 0: the hit point center + (normal . (a - center)
// / denominator) direction against the closed triangle and its edges.
RayHit exact_ray_hit(const Vec& center_sum, const Dyadic& n, const Vec& direction, const Vec& a,
                     const Vec& b, const Vec& c, const Dyadic& weight) {
  const Vec normal = cross(sub(b, a), sub(c, a));
  const Dyadic denominator = dot(normal, direction);
  if (denominator.sign() <= 0) return RayHit::kMiss;
  const Dyadic factor = denominator * n;
  const Dyadic numerator = dot(normal, sub(scale(a, n), scale(center_sum, weight)));
  const Vec hit = add(scale(center_sum, denominator * weight), scale(direction, numerator));
  const std::array<Vec, 3> corners = {scale(a, factor), scale(b, factor), scale(c, factor)};
  int u = 0, v = 0;
  projection_axes(dominant_axis(normal), u, v);
  const Point2 p = {hit[u], hit[v]};
  if (!point_in_triangle2(p, {corners[0][u], corners[0][v]}, {corners[1][u], corners[1][v]},
                          {corners[2][u], corners[2][v]})) {
    return RayHit::kMiss;
  }
  if (on_segment(hit, corners[0], corners[1]) || on_segment(hit, corners[1], corners[2]) ||
      on_segment(hit, corners[2], corners[0])) {
    return RayHit::kDegenerate;
  }
  return RayHit::kCross;
}

// The same triangle decided on enclosures when they are conclusive. Whether
// the hit lies in the closed triangle, strictly inside it, or outside does not
// depend on the injective projection, so any axis with a nonzero normal
// component serves.
int interval_ray_hit(const Box& center, const Box& direction, const Box& a, const Box& b,
                     const Box& c, RayHit& result) {
  const Box normal = cross(sub(b, a), sub(c, a));
  const Interval denominator = dot(normal, direction);
  const int facing = sign_of(denominator);
  if (facing == kUnknown) return kUnknown;
  if (facing < 0) {
    result = RayHit::kMiss;
    return 0;
  }
  const Interval parameter = divide(dot(normal, sub(a, center)), denominator);
  const Box hit = add(center, scale(direction, parameter));
  int dropped = -1;
  double best = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    if (sign_of(normal[axis]) == kUnknown) continue;
    const double size = std::min(std::fabs(normal[axis].lo), std::fabs(normal[axis].hi));
    if (dropped < 0 || size > best) {
      dropped = axis;
      best = size;
    }
  }
  if (dropped < 0) return kUnknown;
  int u = 0, v = 0;
  projection_axes(dropped, u, v);
  const auto orient = [&](const Box& p, const Box& q) {
    return sign_of((q[u] - p[u]) * (hit[v] - p[v]) - (q[v] - p[v]) * (hit[u] - p[u]));
  };
  const int s0 = orient(a, b);
  const int s1 = orient(b, c);
  const int s2 = orient(c, a);
  const bool positive = s0 == 1 || s1 == 1 || s2 == 1;
  const bool negative = s0 == -1 || s1 == -1 || s2 == -1;
  if (positive && negative) {
    result = RayHit::kMiss;
    return 0;
  }
  if (s0 == kUnknown || s1 == kUnknown || s2 == kUnknown) return kUnknown;
  // Strictly inside: no edge line contains the hit, so no edge segment does.
  result = RayHit::kCross;
  return 0;
}

// A face triangle: exact corners over the cell scale times `weight` > 0 (the
// fan apex of a face of m vertices is its exact vertex sum, weight m) and the
// enclosures of the rational corners.
struct Triangle {
  std::array<Vec, 3> exact;
  std::array<Box, 3> boxes;
  Dyadic weight;
};

// `_star_ray_degree` about the star point center_sum / n over the same ray
// family; -1 when no direction of the family is generic.
int64_t star_ray_degree(const Vec& center_sum, const Box& center, const Dyadic& n,
                        const std::vector<Triangle>& triangles) {
  const int64_t attempts = 8 * static_cast<int64_t>(triangles.size()) + 1;
  for (int64_t exponent = 0; exponent < attempts; ++exponent) {
    const double k = static_cast<double>(exponent);
    const Vec direction = {Dyadic(1.0), Dyadic(k), Dyadic(k * k + 1.0)};
    const Box direction_box = {exact_interval(1.0), exact_interval(k),
                               exact_interval(k * k + 1.0)};
    int64_t crossings = 0;
    bool generic = true;
    for (const Triangle& triangle : triangles) {
      RayHit hit = RayHit::kMiss;
      if (interval_ray_hit(center, direction_box, triangle.boxes[0], triangle.boxes[1],
                           triangle.boxes[2], hit) != 0) {
        hit = exact_ray_hit(center_sum, n, direction, triangle.exact[0], triangle.exact[1],
                            triangle.exact[2], triangle.weight);
      }
      if (hit == RayHit::kDegenerate) {
        generic = false;
        break;
      }
      if (hit == RayHit::kCross) ++crossings;
    }
    if (generic) return crossings;
  }
  return -1;
}

Vec newell(const std::vector<Vec>& loop) {
  Vec normal;
  for (std::size_t index = 0; index < loop.size(); ++index) {
    const Vec& start = loop[index];
    const Vec& stop = loop[(index + 1) % loop.size()];
    normal[0] = normal[0] + (start[1] - stop[1]) * (start[2] + stop[2]);
    normal[1] = normal[1] + (start[2] - stop[2]) * (start[0] + stop[0]);
    normal[2] = normal[2] + (start[0] - stop[0]) * (start[1] + stop[1]);
  }
  return normal;
}

Box newell(const std::vector<const Box*>& loop) {
  Box normal;
  for (std::size_t index = 0; index < loop.size(); ++index) {
    const Box& start = *loop[index];
    const Box& stop = *loop[(index + 1) % loop.size()];
    normal[0] = normal[0] + (start[1] - stop[1]) * (start[2] + stop[2]);
    normal[1] = normal[1] + (start[2] - stop[2]) * (start[0] + stop[0]);
    normal[2] = normal[2] + (start[0] - stop[0]) * (start[1] + stop[1]);
  }
  return normal;
}

// polytope_face_routes / polytope_face_triangles of one oriented loop: rows
// are the loop over the cell scale, face_rows the same loop over its own
// face's scale. Planar loops get their ear triangulation, others the fan
// (kApex, i, i + 1) from the exact vertex average, after the fold and
// single-winding checks; local triangles or the refusal.
constexpr int32_t kApex = -1;

int32_t polytope_face(const std::vector<Vec>& rows, const std::vector<Vec>& face_rows,
                      std::vector<std::array<int32_t, 3>>& triangles, bool& fan) {
  triangles.clear();
  fan = false;
  const Vec normal = newell(rows);
  if (!nonzero(normal)) return kFaceZeroArea;
  for (const Vec& row : rows) {
    if (dot(normal, sub(row, rows[0])).sign()) {
      fan = true;
      break;
    }
  }
  if (!fan) return face_triangles(face_rows, triangles);
  const std::size_t count = rows.size();
  const Dyadic m = count_value(static_cast<int64_t>(count));
  Vec sum;
  for (const Vec& row : rows) sum = add(sum, row);
  // Spokes m (P_i - c): positive multiples of the rational ones.
  std::vector<Vec> spokes;
  spokes.reserve(count);
  for (const Vec& row : rows) spokes.push_back(sub(scale(row, m), sum));
  const auto turn = [&](const Vec& u, const Vec& w) { return dot(normal, cross(u, w)).sign(); };
  for (std::size_t i = 0; i < count; ++i) {
    if (turn(spokes[i], spokes[(i + 1) % count]) <= 0) return kFaceFanFolded;
  }
  const Vec& first = spokes[0];
  const Dyadic norm2 = dot(normal, normal);
  const auto half = [&](const Vec& u) {
    const int side = turn(first, u);
    if (side) return side > 0 ? 0 : 1;
    const int along = (dot(first, u) * norm2 - dot(first, normal) * dot(u, normal)).sign();
    return along > 0 ? 0 : 1;
  };
  int64_t wraps = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const Vec& u = spokes[i];
    const Vec& w = spokes[(i + 1) % count];
    const int hu = half(u), hw = half(w);
    if (!(hu < hw || (hu == hw && turn(u, w) > 0))) ++wraps;
  }
  if (wraps != 1) return kFaceFanWinding;
  for (std::size_t i = 0; i < count; ++i) {
    triangles.push_back(
        {kApex, static_cast<int32_t>(i), static_cast<int32_t>((i + 1) % count)});
  }
  return kDecomposed;
}

// One cell. face_points[f] are the rows of face f over its own common scale,
// in the given loop order.
int32_t decompose(const CellPoints& cell, const int64_t* face_offsets, const int64_t* face_values,
                  int64_t face_count, const std::vector<std::vector<Vec>>& face_points,
                  int8_t* reversed_out, int32_t* triangles_out, int64_t& triangle_count) {
  const int64_t point_count = static_cast<int64_t>(cell.exact.size());
  if (point_count > kMaximumCount) return kUndecided;
  if (face_count == 0) return kNoFaces;
  // _oriented_closed_faces.
  std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<int64_t, int64_t>>> edges;
  for (int64_t face = 0; face < face_count; ++face) {
    const int64_t start = face_offsets[face];
    const int64_t size = face_offsets[face + 1] - start;
    std::vector<int64_t> sorted(face_values + start, face_values + start + size);
    std::sort(sorted.begin(), sorted.end());
    if (size < 3 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      return kFaceIncidence;
    }
    for (int64_t index = 0; index < size; ++index) {
      const int64_t first = face_values[start + index];
      const int64_t second = face_values[start + (index + 1) % size];
      edges[{std::min(first, second), std::max(first, second)}].push_back({face, first});
    }
  }
  for (const auto& entry : edges) {
    if (entry.second.size() != 2) return kNotClosedManifold;
  }
  std::vector<int8_t> reversed(static_cast<std::size_t>(face_count), -1);
  for (int64_t root = 0; root < face_count; ++root) {
    if (reversed[static_cast<std::size_t>(root)] >= 0) continue;
    reversed[static_cast<std::size_t>(root)] = 0;
    std::vector<int64_t> queue{root};
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const int64_t current = queue[head];
      const int64_t start = face_offsets[current];
      const int64_t size = face_offsets[current + 1] - start;
      for (int64_t index = 0; index < size; ++index) {
        const int64_t first = face_values[start + index];
        const int64_t second = face_values[start + (index + 1) % size];
        for (const auto& [neighbor, neighbor_start] :
             edges[{std::min(first, second), std::max(first, second)}]) {
          if (neighbor == current) continue;
          const int8_t flipped = static_cast<int8_t>(
              reversed[static_cast<std::size_t>(current)] ^ (neighbor_start == first ? 1 : 0));
          int8_t& known = reversed[static_cast<std::size_t>(neighbor)];
          if (known < 0) {
            known = flipped;
            queue.push_back(neighbor);
          } else if (known != flipped) {
            return kNotOrientable;
          }
        }
      }
    }
  }
  // Oriented loops (cell rows, face rows) and their polytope_face triangles.
  std::vector<std::vector<int64_t>> local(static_cast<std::size_t>(face_count));
  std::vector<std::vector<Vec>> rows(local.size()), loops(local.size());
  std::vector<std::vector<std::array<int32_t, 3>>> face_triangles_of(local.size());
  std::vector<char> fans(local.size(), 0);
  const auto triangulate = [&]() -> int32_t {
    for (std::size_t face = 0; face < local.size(); ++face) {
      auto& route = local[face];
      route.assign(face_values + face_offsets[face], face_values + face_offsets[face + 1]);
      loops[face] = face_points[face];
      if (reversed[face]) {
        std::reverse(route.begin(), route.end());
        std::reverse(loops[face].begin(), loops[face].end());
      }
      rows[face].clear();
      for (const int64_t index : route) {
        rows[face].push_back(cell.exact[static_cast<std::size_t>(index)]);
      }
      bool fan = false;
      const int32_t code = polytope_face(rows[face], loops[face], face_triangles_of[face], fan);
      if (code != kDecomposed) return code;
      fans[face] = fan ? 1 : 0;
    }
    return kDecomposed;
  };
  if (const int32_t code = triangulate(); code != kDecomposed) return code;
  // Sign of the exact volume about the first point. A planar face's part is
  // that of its loop's fan from its first vertex; a fan face of m vertices
  // contributes (1 / m) sum (S - m O) . ((P_i - O) x (P_{i+1} - O)), so every
  // part is scaled by the product of the fan sizes.
  const Vec& origin = cell.exact[0];
  const bool any_fan = std::any_of(fans.begin(), fans.end(), [](char fan) { return fan != 0; });
  int orientation = kUnknown;
  if (!any_fan) {
    Interval volume_box;
    const Box& origin_box = cell.boxes[0];
    for (const auto& route : local) {
      const Box base = sub(cell.boxes[static_cast<std::size_t>(route[0])], origin_box);
      for (std::size_t index = 1; index + 1 < route.size(); ++index) {
        volume_box =
            volume_box +
            dot(cross(base, sub(cell.boxes[static_cast<std::size_t>(route[index])], origin_box)),
                sub(cell.boxes[static_cast<std::size_t>(route[index + 1])], origin_box));
      }
    }
    orientation = sign_of(volume_box);
  }
  if (orientation == kUnknown) {
    Dyadic volume;
    for (std::size_t face = 0; face < local.size(); ++face) {
      const auto& loop = rows[face];
      Dyadic part;
      if (fans[face]) {
        const Dyadic m = count_value(static_cast<int64_t>(loop.size()));
        Vec sum;
        for (const Vec& row : loop) sum = add(sum, row);
        const Vec apex = sub(sum, scale(origin, m));
        for (std::size_t i = 0; i < loop.size(); ++i) {
          part = part + dot(apex, cross(sub(loop[i], origin),
                                        sub(loop[(i + 1) % loop.size()], origin)));
        }
      } else {
        const Vec base = sub(loop[0], origin);
        for (std::size_t index = 1; index + 1 < loop.size(); ++index) {
          part = part + dot(cross(base, sub(loop[index], origin)), sub(loop[index + 1], origin));
        }
      }
      // Scale by the fan sizes of every other fan face (all of them for a
      // planar face).
      for (std::size_t other = 0; other < local.size(); ++other) {
        if (other != face && fans[other]) {
          part = part * count_value(static_cast<int64_t>(rows[other].size()));
        }
      }
      volume = volume + part;
    }
    orientation = volume.sign();
  }
  if (orientation == 0) return kZeroVolume;
  if (orientation < 0) {
    // Every loop reversed and triangulated again: the outward loops.
    for (auto& flag : reversed) flag = static_cast<int8_t>(flag ^ 1);
    if (const int32_t code = triangulate(); code != kDecomposed) return code;
  }
  // The face triangles, a fan apex carried exactly as (vertex sum, size).
  std::vector<Triangle> triangles;
  for (std::size_t face = 0; face < local.size(); ++face) {
    const auto& loop = rows[face];
    const auto& route = local[face];
    const Dyadic m = count_value(static_cast<int64_t>(loop.size()));
    Vec sum;
    Box apex_box;
    if (fans[face]) {
      for (const Vec& row : loop) sum = add(sum, row);
      for (const int64_t index : route) {
        apex_box = add(apex_box, cell.boxes[static_cast<std::size_t>(index)]);
      }
      const Interval size = exact_interval(static_cast<double>(loop.size()));
      apex_box = {divide(apex_box[0], size), divide(apex_box[1], size),
                  divide(apex_box[2], size)};
    }
    for (const auto& local_triangle : face_triangles_of[face]) {
      Triangle triangle;
      triangle.weight = fans[face] ? m : Dyadic(1.0);
      for (std::size_t k = 0; k < 3; ++k) {
        const int32_t corner = local_triangle[k];
        if (corner == kApex) {
          triangle.exact[k] = sum;
          triangle.boxes[k] = apex_box;
        } else {
          const auto index = static_cast<std::size_t>(route[static_cast<std::size_t>(corner)]);
          triangle.exact[k] = fans[face] ? scale(cell.exact[index], m) : cell.exact[index];
          triangle.boxes[k] = cell.boxes[index];
        }
      }
      triangles.push_back(std::move(triangle));
    }
  }
  // _star_center: the vertex average, strictly inside every face triangle's
  // plane (one plane per planar face). The kernel-polytope fallback stays
  // with the rational proof.
  const Dyadic n = count_value(point_count);
  const double count = static_cast<double>(point_count);
  Box center;
  for (const Box& box : cell.boxes) center = add(center, box);
  center = {divide(center[0], exact_interval(count)), divide(center[1], exact_interval(count)),
            divide(center[2], exact_interval(count))};
  Vec center_sum;
  for (const Vec& point : cell.exact) center_sum = add(center_sum, point);
  std::size_t next = 0;
  for (std::size_t face = 0; face < local.size(); ++face) {
    const std::size_t face_size = face_triangles_of[face].size();
    if (!fans[face]) {
      std::vector<const Box*> boxes;
      for (const int64_t index : local[face]) {
        boxes.push_back(&cell.boxes[static_cast<std::size_t>(index)]);
      }
      const Box normal = newell(boxes);
      int side = sign_of(dot(normal, sub(center, *boxes[0])));
      if (side == kUnknown) {
        const Vec exact_normal = newell(rows[face]);
        side = compare(dot(exact_normal, center_sum), dot(exact_normal, rows[face][0]) * n);
      }
      if (side >= 0) return kUndecided;
    } else {
      for (std::size_t k = next; k < next + face_size; ++k) {
        const Triangle& triangle = triangles[k];
        const Box normal_box = cross(sub(triangle.boxes[1], triangle.boxes[0]),
                                     sub(triangle.boxes[2], triangle.boxes[0]));
        int side = sign_of(dot(normal_box, sub(center, triangle.boxes[0])));
        if (side == kUnknown) {
          const Vec normal = cross(sub(triangle.exact[1], triangle.exact[0]),
                                   sub(triangle.exact[2], triangle.exact[0]));
          side = compare(dot(normal, scale(center_sum, triangle.weight)),
                         dot(normal, triangle.exact[0]) * n);
        }
        if (side >= 0) return kUndecided;
      }
    }
    next += face_size;
  }
  if (star_ray_degree(center_sum, center, n, triangles) != 1) return kStarDegree;
  int32_t* out = triangles_out;
  for (std::size_t face = 0; face < local.size(); ++face) {
    for (const auto& triangle : face_triangles_of[face]) {
      *out++ = static_cast<int32_t>(face);
      *out++ = triangle[0];
      *out++ = triangle[1];
      *out++ = triangle[2];
    }
  }
  triangle_count = static_cast<int64_t>(triangles.size());
  std::copy(reversed.begin(), reversed.end(), reversed_out);
  return kDecomposed;
}

}  // namespace

extern "C" int32_t phx_mc_polyhedral_star_decomposition(
    int64_t cell_count, const int64_t* point_offsets, const int64_t* word_offsets,
    const uint32_t* words, const int8_t* signs, const int64_t* face_word_offsets,
    const uint32_t* face_words, const int8_t* face_signs, const int64_t* cell_face_offsets,
    const int64_t* face_vertex_offsets, const int64_t* face_vertex_values,
    const int64_t* triangle_offsets, int32_t* codes, int8_t* face_reversed, int32_t* triangles,
    int64_t* triangle_counts) {
  return phx::mc::guarded([&]() -> int32_t {
    if (cell_count < 0 || point_offsets == nullptr || cell_face_offsets == nullptr ||
        face_vertex_offsets == nullptr || triangle_offsets == nullptr ||
        (cell_count > 0 && (codes == nullptr || triangle_counts == nullptr)) ||
        triangle_offsets[0] != 0 ||
        !valid_bank(3 * point_offsets[cell_count] + cell_count, word_offsets, words, signs) ||
        !valid_bank(3 * face_vertex_offsets[cell_face_offsets[cell_count]], face_word_offsets,
                    face_words, face_signs) ||
        (face_vertex_offsets[cell_face_offsets[cell_count]] > 0 && face_vertex_values == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!valid_cells(cell_count, point_offsets, cell_face_offsets, face_vertex_offsets,
                     face_vertex_values)) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      int64_t expected = 0;
      for (int64_t face = cell_face_offsets[cell]; face < cell_face_offsets[cell + 1]; ++face) {
        expected += face_vertex_offsets[face + 1] - face_vertex_offsets[face];
      }
      if (triangle_offsets[cell + 1] - triangle_offsets[cell] != expected) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    if ((cell_face_offsets[cell_count] > 0 && face_reversed == nullptr) ||
        (triangle_offsets[cell_count] > 0 && triangles == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::GradualNearestScope mode;
    std::fill(triangle_counts, triangle_counts + cell_count, int64_t{0});
    const IntegerBank bank{word_offsets, words, signs};
    const IntegerBank face_bank{face_word_offsets, face_words, face_signs};
    const int64_t scale_base = 3 * point_offsets[cell_count];
    CellPoints cell_points;
    std::vector<int64_t> offsets;
    std::vector<std::vector<Vec>> face_points;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      const int64_t first_face = cell_face_offsets[cell];
      const int64_t face_count = cell_face_offsets[cell + 1] - first_face;
      if (point_offsets[cell + 1] - point_offsets[cell] > kMaximumCount) {
        codes[cell] = kUndecided;
        continue;
      }
      if (point_offsets[cell + 1] == point_offsets[cell]) {
        codes[cell] = face_count == 0 ? kNoFaces : kFaceIncidence;
        continue;
      }
      load_cell(bank, point_offsets[cell], point_offsets[cell + 1] - point_offsets[cell],
                scale_base + cell, cell_points);
      offsets.clear();
      face_points.assign(static_cast<std::size_t>(face_count), {});
      for (int64_t face = first_face; face <= cell_face_offsets[cell + 1]; ++face) {
        offsets.push_back(face_vertex_offsets[face] - face_vertex_offsets[first_face]);
        if (face == cell_face_offsets[cell + 1]) break;
        auto& rows = face_points[static_cast<std::size_t>(face - first_face)];
        for (int64_t entry = face_vertex_offsets[face]; entry < face_vertex_offsets[face + 1];
             ++entry) {
          rows.push_back(face_bank.point(entry));
        }
      }
      codes[cell] = decompose(cell_points, offsets.data(),
                              face_vertex_values + face_vertex_offsets[first_face], face_count,
                              face_points, face_reversed + first_face,
                              triangles + 4 * triangle_offsets[cell], triangle_counts[cell]);
    }
    return PHX_MC_OK;
  });
}

extern "C" int32_t phx_mc_face_triangulation(int64_t face_count,
                                             const int64_t* face_vertex_offsets,
                                             const int64_t* word_offsets, const uint32_t* words,
                                             const int8_t* signs, int32_t* codes,
                                             int32_t* triangles) {
  return phx::mc::guarded([&]() -> int32_t {
    if (face_count < 0 || face_vertex_offsets == nullptr || face_vertex_offsets[0] != 0 ||
        (face_count > 0 && codes == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int64_t triangle_count = 0;
    for (int64_t face = 0; face < face_count; ++face) {
      const int64_t size = face_vertex_offsets[face + 1] - face_vertex_offsets[face];
      if (size < 0) return PHX_MC_INVALID_INPUT;
      triangle_count += std::max<int64_t>(size - 2, 0);
    }
    if (!valid_bank(3 * face_vertex_offsets[face_count], word_offsets, words, signs) ||
        (triangle_count > 0 && triangles == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::GradualNearestScope mode;
    const IntegerBank bank{word_offsets, words, signs};
    std::vector<Vec> loop;
    std::vector<std::array<int32_t, 3>> local;
    int32_t* out = triangles;
    for (int64_t face = 0; face < face_count; ++face) {
      const int64_t first = face_vertex_offsets[face];
      const int64_t size = face_vertex_offsets[face + 1] - first;
      loop.clear();
      local.clear();
      for (int64_t entry = first; entry < first + size; ++entry) loop.push_back(bank.point(entry));
      codes[face] = face_triangles(loop, local);
      // A face's rows are fixed by its size, so refusals leave theirs at -1.
      for (int64_t row = 0; row < std::max<int64_t>(size - 2, 0); ++row) {
        const bool kept = codes[face] == kDecomposed;
        for (int k = 0; k < 3; ++k) {
          *out++ = kept ? local[static_cast<std::size_t>(row)][static_cast<std::size_t>(k)] : -1;
        }
      }
    }
    return PHX_MC_OK;
  });
}
