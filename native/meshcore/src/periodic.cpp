//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Periodic Delaunay triangulation on successively enlarged certified image
// neighborhoods; see periodic.hpp for quotient sufficiency and closure.
#include "periodic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "exact_scalar.hpp"
#include "expansion.hpp"
#include "filtered.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"
#include "cavity.hpp"
#include <map>
#include <set>
#include <tuple>

namespace phx::mc {
namespace {

// Rounding slack of the input fractional coordinates; it only admits more
// images, so every periodic point inside the certified box is present.
constexpr double kImageSlack = 0x1p-30;
// Relative slack on the constructed circumball: a certificate needs the
// rounded ball to clear the box by this much, which dominates the rounding of
// the circumcenter of any cell whose circumcenter is not ill-conditioned; an
// ill-conditioned construction only fails to certify and widens the margin.
constexpr double kCertificateSlack = 0x1p-26;

struct PeriodicIndexRangeExceeded {};
struct DuplicatePoint {
  int32_t representative;
};
int32_t checked_shift(int64_t value);

using exact::det3;

double add_up(double left, double right) {
  return std::nextafter(left + right, std::numeric_limits<double>::infinity());
}

double subtract_up(double left, double right) {
  return std::nextafter(left - right, std::numeric_limits<double>::infinity());
}

double upper_sqrt_ratio(const Dyadic& numerator, const Dyadic& denominator) {
  if (numerator.sign() < 0 || denominator.sign() <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  if (numerator.is_zero()) return 0.0;
  int64_t numerator_exponent = 0;
  int64_t denominator_exponent = 0;
  const double numerator_fraction = numerator.fraction(numerator_exponent);
  const double denominator_fraction = denominator.fraction(denominator_exponent);
  const int64_t exponent = numerator_exponent - denominator_exponent;
  int64_t half_exponent = exponent / 2;
  if (exponent < 0 && exponent % 2 != 0) --half_exponent;
  const int64_t odd_exponent = exponent - 2 * half_exponent;
  const double scaled =
      numerator_fraction / denominator_fraction * (odd_exponent == 0 ? 1.0 : 2.0);
  double upper = 0.0;
  if (half_exponent > std::numeric_limits<int>::max()) {
    return std::numeric_limits<double>::infinity();
  }
  if (half_exponent >= std::numeric_limits<int>::min()) {
    upper = std::ldexp(std::sqrt(scaled), static_cast<int>(half_exponent));
  }
  if (!std::isfinite(upper)) return std::numeric_limits<double>::infinity();
  for (;;) {
    const Dyadic candidate(upper);
    if ((candidate * candidate * denominator - numerator).sign() >= 0) return upper;
    upper = std::nextafter(upper, std::numeric_limits<double>::infinity());
    if (!std::isfinite(upper)) return std::numeric_limits<double>::infinity();
  }
}

template <int D>
class ExactLatticeTransform final {
 public:
  explicit ExactLatticeTransform(const double* lattice) {
    std::array<std::array<Dyadic, D>, D> matrix{};
    for (int row = 0; row < D; ++row) {
      for (int column = 0; column < D; ++column) {
        matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)] =
            Dyadic(lattice[row * D + column]);
      }
    }
    if constexpr (D == 2) {
      const Dyadic& a = matrix[0][0];
      const Dyadic& b = matrix[0][1];
      const Dyadic& c = matrix[1][0];
      const Dyadic& d = matrix[1][1];
      denominator_ = a * d - b * c;
      inverse_numerator_[0][0] = d;
      inverse_numerator_[0][1] = -b;
      inverse_numerator_[1][0] = -c;
      inverse_numerator_[1][1] = a;
    } else {
      const Dyadic& a = matrix[0][0];
      const Dyadic& b = matrix[0][1];
      const Dyadic& c = matrix[0][2];
      const Dyadic& d = matrix[1][0];
      const Dyadic& e = matrix[1][1];
      const Dyadic& f = matrix[1][2];
      const Dyadic& g = matrix[2][0];
      const Dyadic& h = matrix[2][1];
      const Dyadic& i = matrix[2][2];
      denominator_ =
          exact::det3(matrix[0].data(), matrix[1].data(), matrix[2].data());
      inverse_numerator_[0][0] = e * i - f * h;
      inverse_numerator_[0][1] = c * h - b * i;
      inverse_numerator_[0][2] = b * f - c * e;
      inverse_numerator_[1][0] = f * g - d * i;
      inverse_numerator_[1][1] = a * i - c * g;
      inverse_numerator_[1][2] = c * d - a * f;
      inverse_numerator_[2][0] = d * h - e * g;
      inverse_numerator_[2][1] = b * g - a * h;
      inverse_numerator_[2][2] = a * e - b * d;
    }
    if (denominator_.sign() == 0) return;
    if (denominator_.sign() < 0) {
      denominator_.negate();
      for (auto& row : inverse_numerator_) {
        for (Dyadic& value : row) value.negate();
      }
    }
    denominator_squared_ = denominator_ * denominator_;
    for (int column = 0; column < D; ++column) {
      for (int row = 0; row < D; ++row) {
        const Dyadic& value =
            inverse_numerator_[static_cast<std::size_t>(row)]
                              [static_cast<std::size_t>(column)];
        inverse_norm_numerator_[static_cast<std::size_t>(column)] =
            inverse_norm_numerator_[static_cast<std::size_t>(column)] +
            value * value;
      }
    }
    valid_ = true;
  }

  bool valid() const { return valid_; }
  const Dyadic& denominator() const { return denominator_; }
  const Dyadic& denominator_squared() const { return denominator_squared_; }
  const Dyadic& inverse_numerator(int row, int column) const {
    return inverse_numerator_[static_cast<std::size_t>(row)]
                             [static_cast<std::size_t>(column)];
  }
  const Dyadic& inverse_norm_numerator(int column) const {
    return inverse_norm_numerator_[static_cast<std::size_t>(column)];
  }

  Dyadic point_fractional_numerator(const double* point, int column) const {
    Dyadic numerator;
    for (int row = 0; row < D; ++row) {
      numerator = numerator +
                  Dyadic(point[row]) * inverse_numerator(row, column);
    }
    return numerator;
  }

  exact::QuotientBounds point_fractional_bounds(const double* point,
                                                int column) const {
    return exact::directed_quotient_bounds(
        point_fractional_numerator(point, column), denominator_);
  }

 private:
  bool valid_ = false;
  Dyadic denominator_;
  Dyadic denominator_squared_;
  std::array<std::array<Dyadic, D>, D> inverse_numerator_{};
  std::array<Dyadic, D> inverse_norm_numerator_{};
};

template <int D>
class PeriodicEngine {
 public:
  using Cell = std::array<int32_t, D + 1>;
  struct Vertex {
    int32_t representative;  // -1 for a bounding vertex
    std::array<int32_t, D> shift;
  };
  struct Candidate {
    std::vector<int32_t> key;  // sorted (representative, relative shift) rows
    Cell ordered;              // positively oriented, anchor-relative order
    std::array<std::array<int32_t, D>, D + 1> shifts;
    double required_margin;
  };

  PeriodicEngine(const PeriodicDelaunayInput& input, std::vector<Vertex> images)
      : input_(input), vertices_(std::move(images)) {}

  int64_t slots() const { return static_cast<int64_t>(cells_.size()); }

  void build() {
    const int32_t image_count = static_cast<int32_t>(vertices_.size());
    std::vector<double> approximate(static_cast<std::size_t>(image_count) * D);
    for (int32_t index = 0; index < image_count; ++index) {
      for (int c = 0; c < D; ++c) {
        approximate[static_cast<std::size_t>(index) * D + c] = position(index, c);
      }
    }
    add_bounding_simplex(approximate);
    std::vector<int32_t> candidates(static_cast<std::size_t>(image_count));
    for (int32_t index = 0; index < image_count; ++index) {
      candidates[static_cast<std::size_t>(index)] = index;
    }
    for (const int32_t index : brio_hilbert_order(approximate.data(), D, candidates)) {
      insert(index);
    }
  }

  // Anchor-extracted cells with their certificates, sorted by orbit key, and
  // the number of extracted cells that are uncertified or not closed.
  int64_t extract(double margin, std::vector<Candidate>& extracted, double& required) {
    extracted.clear();
    required = 0.0;
    std::unordered_map<std::size_t, std::vector<int32_t>> index;
    int64_t uncertified = 0;
    for (int32_t cell = 0; cell < slots(); ++cell) {
      if (!alive_[static_cast<std::size_t>(cell)] || touches_bound(cell)) {
        continue;
      }
      const int32_t anchor = anchor_of(cells_[static_cast<std::size_t>(cell)]);
      if (!is_base(anchor)) {
        continue;
      }
      Candidate candidate = canonical(cell);
      candidate.required_margin = required_margin(cells_[static_cast<std::size_t>(cell)]);
      required = std::max(required, candidate.required_margin);
      if (!(candidate.required_margin <= margin)) {
        ++uncertified;
      }
      index[hash(candidate.key)].push_back(static_cast<int32_t>(extracted.size()));
      extracted.push_back(std::move(candidate));
    }
    if (uncertified == 0) {
      uncertified = closure_misses(extracted, index);
    }
    std::sort(extracted.begin(), extracted.end(),
              [](const Candidate& left, const Candidate& right) { return left.key < right.key; });
    return uncertified;
  }

  int support_orientation_sign(const Cell& cell) const {
    const Approx approx = orientation<Approx>(
        cell, [this](int32_t a, int32_t b, int c) { return approx_difference(a, b, c); });
    const int sign = approx.certified_sign();
    if (sign != 2) return sign;
    return orientation<Expansion>(
               cell, [this](int32_t a, int32_t b, int c) {
                 return exact_difference(a, b, c);
               })
        .sign();
  }

  int support_insphere_sign(const Cell& cell, int32_t query_representative,
                            const std::array<int32_t, D>& query_shift) {
    Vertex query = vertices_[static_cast<std::size_t>(query_representative)];
    query.shift = query_shift;
    vertices_.push_back(query);
    const int32_t query_vertex = static_cast<int32_t>(vertices_.size() - 1);
    try {
      const int sign = insphere_sign(cell, query_vertex);
      vertices_.pop_back();
      return sign;
    } catch (...) {
      vertices_.pop_back();
      throw;
    }
  }


  double support_fractional_bounds(
      const Cell& cell, const ExactLatticeTransform<D>& transform,
      std::array<double, D>& lower, std::array<double, D>& upper) const {
    return support_bounds_exact(cell, transform, lower, upper);
  }

 private:
  const PeriodicDelaunayInput& input_;
  std::vector<Vertex> vertices_;
  int32_t bound_base_ = 0;
  std::array<double, (D + 1) * D> bound_ = {};
  std::vector<Cell> cells_;
  std::vector<Cell> neighbors_;
  std::vector<uint8_t> alive_;
  std::vector<int32_t> free_;
  std::vector<uint64_t> cavity_mark_;
  std::vector<uint64_t> outside_mark_;
  uint64_t stamp_ = 0;
  int32_t last_ = 0;

  // Whether a vertex is its representative's base image: the one whose
  // fractional coordinates are wrapped into [0, 1) by -floor(fractional).
  bool is_base(int32_t vertex) const {
    const Vertex& item = vertices_[static_cast<std::size_t>(vertex)];
    for (int j = 0; j < D; ++j) {
      const double fractional =
          input_.fractional[static_cast<int64_t>(item.representative) * D + j];
      if (item.shift[static_cast<std::size_t>(j)] !=
          -static_cast<int32_t>(std::floor(fractional))) {
        return false;
      }
    }
    return true;
  }

  static std::size_t hash(const std::vector<int32_t>& key) {
    uint64_t value = 0x243F6A8885A308D3ULL;
    for (const int32_t item : key) {
      value = splitmix64(value ^ static_cast<uint32_t>(item));
    }
    return static_cast<std::size_t>(value);
  }

  double base(int32_t vertex, int c) const {
    const Vertex& item = vertices_[static_cast<std::size_t>(vertex)];
    return item.representative >= 0
               ? input_.points[static_cast<int64_t>(item.representative) * D + c]
               : bound_[static_cast<std::size_t>((vertex - bound_base_) * D + c)];
  }

  double position(int32_t vertex, int c) const {
    double value = base(vertex, c);
    const Vertex& item = vertices_[static_cast<std::size_t>(vertex)];
    for (int j = 0; j < D; ++j) {
      value += static_cast<double>(item.shift[static_cast<std::size_t>(j)]) *
               input_.lattice[j * D + c];
    }
    return value;
  }

  Approx approx_difference(int32_t a, int32_t b, int c) const {
    Approx value = Approx::exact(base(a, c)) - Approx::exact(base(b, c));
    const Vertex& va = vertices_[static_cast<std::size_t>(a)];
    const Vertex& vb = vertices_[static_cast<std::size_t>(b)];
    for (int j = 0; j < D; ++j) {
      const int64_t delta = static_cast<int64_t>(va.shift[static_cast<std::size_t>(j)]) -
                            vb.shift[static_cast<std::size_t>(j)];
      if (delta != 0) {
        value = value + Approx::exact(static_cast<double>(delta)) *
                            Approx::exact(input_.lattice[j * D + c]);
      }
    }
    return value;
  }

  Expansion exact_difference(int32_t a, int32_t b, int c) const {
    Expansion value = Expansion::difference(base(a, c), base(b, c));
    const Vertex& va = vertices_[static_cast<std::size_t>(a)];
    const Vertex& vb = vertices_[static_cast<std::size_t>(b)];
    for (int j = 0; j < D; ++j) {
      const int64_t delta = static_cast<int64_t>(va.shift[static_cast<std::size_t>(j)]) -
                            vb.shift[static_cast<std::size_t>(j)];
      if (delta != 0) {
        value = value + Expansion::product(static_cast<double>(delta), input_.lattice[j * D + c]);
      }
    }
    return value;
  }

  // det[v1 - v0, ..., vD - v0]; positive for a positively oriented simplex.
  template <class T, class Difference>
  static T orientation(const Cell& v, const Difference& difference) {
    T rows[D][D];
    for (int i = 0; i < D; ++i) {
      for (int c = 0; c < D; ++c) {
        rows[i][c] = difference(v[static_cast<std::size_t>(i + 1)], v[0], c);
      }
    }
    if constexpr (D == 2) {
      return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0];
    } else {
      return det3(rows[0], rows[1], rows[2]);
    }
  }

  // Positive iff q lies inside the circumsphere of the positively oriented v.
  template <class T, class Difference>
  static T insphere(const Cell& v, int32_t q, const Difference& difference) {
    T rows[D + 1][D + 1];
    for (int i = 0; i <= D; ++i) {
      for (int c = 0; c < D; ++c) {
        rows[i][c] = difference(v[static_cast<std::size_t>(i)], q, c);
      }
      T lifted = rows[i][0] * rows[i][0];
      for (int c = 1; c < D; ++c) {
        lifted = lifted + rows[i][c] * rows[i][c];
      }
      rows[i][D] = lifted;
    }
    if constexpr (D == 2) {
      return det3(rows[0], rows[1], rows[2]);
    } else {
      // Negated 4x4 lifted determinant, expanded along the lifted column.
      return rows[0][3] * det3(rows[1], rows[2], rows[3]) -
             rows[1][3] * det3(rows[0], rows[2], rows[3]) +
             rows[2][3] * det3(rows[0], rows[1], rows[3]) -
             rows[3][3] * det3(rows[0], rows[1], rows[2]);
    }
  }

  int orient_sign(const Cell& v) {
    const Approx approx = orientation<Approx>(
        v, [this](int32_t a, int32_t b, int c) { return approx_difference(a, b, c); });
    const int sign = approx.certified_sign();
    if (sign != 2) {
      return sign;
    }
    return orientation<Expansion>(
               v, [this](int32_t a, int32_t b, int c) { return exact_difference(a, b, c); })
        .sign();
  }

  int insphere_sign(const Cell& v, int32_t q) {
    const Approx approx = insphere<Approx>(
        v, q, [this](int32_t a, int32_t b, int c) { return approx_difference(a, b, c); });
    const int sign = approx.certified_sign();
    if (sign != 2) {
      return sign;
    }
    return insphere<Expansion>(
               v, q, [this](int32_t a, int32_t b, int c) { return exact_difference(a, b, c); })
        .sign();
  }

  // Exact lexicographic order of positions: translation invariant.
  int lexicographic(int32_t a, int32_t b) {
    for (int c = 0; c < D; ++c) {
      int sign = approx_difference(a, b, c).certified_sign();
      if (sign == 2) {
        sign = exact_difference(a, b, c).sign();
      }
      if (sign != 0) {
        return sign;
      }
    }
    return 0;
  }

  bool same_point(int32_t a, int32_t b) { return lexicographic(a, b) == 0; }

  // Perturbed conflict test (Devillers-Teillaud): positive iff q conflicts
  // with the cell.  On an exact tie the lexicographically largest point of the
  // cell plus q decides through one or two orientation tests.
  int conflict(int32_t cell, int32_t q) {
    const Cell& v = cells_[static_cast<std::size_t>(cell)];
    const int sign = insphere_sign(v, q);
    if (sign != 0) {
      return sign;
    }
    std::array<int32_t, D + 2> order;
    for (int i = 0; i <= D; ++i) {
      order[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(i)];
    }
    order[D + 1] = q;
    std::sort(order.begin(), order.end(),
              [this](int32_t a, int32_t b) { return lexicographic(a, b) < 0; });
    for (int rank = D + 1; rank > D - 1; --rank) {
      const int32_t largest = order[static_cast<std::size_t>(rank)];
      if (largest == q) {
        return -1;
      }
      for (int k = 0; k <= D; ++k) {
        if (v[static_cast<std::size_t>(k)] == largest) {
          Cell replaced = v;
          replaced[static_cast<std::size_t>(k)] = q;
          const int orientation_sign = orient_sign(replaced);
          if (orientation_sign != 0) {
            return orientation_sign;
          }
        }
      }
    }
    throw std::logic_error("periodic perturbation left a tie");
  }

  bool touches_bound(int32_t cell) const {
    for (const int32_t vertex : cells_[static_cast<std::size_t>(cell)]) {
      if (vertices_[static_cast<std::size_t>(vertex)].representative < 0) {
        return true;
      }
    }
    return false;
  }

  bool precedes(int32_t a, int32_t b) const {
    const Vertex& va = vertices_[static_cast<std::size_t>(a)];
    const Vertex& vb = vertices_[static_cast<std::size_t>(b)];
    if (va.representative != vb.representative) {
      return va.representative < vb.representative;
    }
    return va.shift < vb.shift;
  }

  int32_t anchor_of(const Cell& cell) const {
    int32_t anchor = cell[0];
    for (int i = 1; i <= D; ++i) {
      if (precedes(cell[static_cast<std::size_t>(i)], anchor)) {
        anchor = cell[static_cast<std::size_t>(i)];
      }
    }
    return anchor;
  }

  // Orbit key of a finite cell: its vertices as (representative, shift
  // relative to the anchor) rows in increasing order.  The published order is
  // the key order, with the last two entries swapped when that permutation of
  // the positively oriented cell is odd.
  Candidate canonical(int32_t cell) const {
    const Cell& v = cells_[static_cast<std::size_t>(cell)];
    const Vertex& anchor = vertices_[static_cast<std::size_t>(anchor_of(v))];
    std::array<int, D + 1> order;
    for (int i = 0; i <= D; ++i) {
      order[static_cast<std::size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int left, int right) {
      return precedes(v[static_cast<std::size_t>(left)], v[static_cast<std::size_t>(right)]);
    });
    int inversions = 0;
    for (int i = 0; i <= D; ++i) {
      for (int j = i + 1; j <= D; ++j) {
        inversions += order[static_cast<std::size_t>(i)] > order[static_cast<std::size_t>(j)];
      }
    }
    Candidate candidate;
    candidate.key.reserve(static_cast<std::size_t>((D + 1) * (D + 1)));
    for (int i = 0; i <= D; ++i) {
      const int32_t vertex = v[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
      const Vertex& item = vertices_[static_cast<std::size_t>(vertex)];
      candidate.key.push_back(item.representative);
      candidate.ordered[static_cast<std::size_t>(i)] = item.representative;
      for (int j = 0; j < D; ++j) {
        const int32_t relative = checked_shift(
            static_cast<int64_t>(item.shift[static_cast<std::size_t>(j)]) -
            anchor.shift[static_cast<std::size_t>(j)]);
        candidate.key.push_back(relative);
        candidate.shifts[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
            item.shift[static_cast<std::size_t>(j)];
      }
    }
    if (inversions % 2 == 1) {
      std::swap(candidate.ordered[D - 1], candidate.ordered[D]);
      std::swap(candidate.shifts[D - 1], candidate.shifts[D]);
    }
    return candidate;
  }

  // Fractional margin that certifies the circumball of the cell, including
  // the construction slack; +inf when the circumcenter is not constructible.
  double required_margin(const Cell& v) const {
    double edges[D][D];
    for (int i = 0; i < D; ++i) {
      for (int c = 0; c < D; ++c) {
        edges[i][c] = approx_difference(v[static_cast<std::size_t>(i + 1)], v[0], c).value;
      }
    }
    double center[D];
    if constexpr (D == 2) {
      const double det = edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0];
      const double b0 = 0.5 * (edges[0][0] * edges[0][0] + edges[0][1] * edges[0][1]);
      const double b1 = 0.5 * (edges[1][0] * edges[1][0] + edges[1][1] * edges[1][1]);
      center[0] = (b0 * edges[1][1] - b1 * edges[0][1]) / det;
      center[1] = (edges[0][0] * b1 - edges[1][0] * b0) / det;
    } else {
      const double* a = edges[0];
      const double* b = edges[1];
      const double* c = edges[2];
      const double bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                            b[0] * c[1] - b[1] * c[0]};
      const double ca[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2],
                            c[0] * a[1] - c[1] * a[0]};
      const double ab[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0]};
      const double aa = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
      const double bb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
      const double cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
      const double denominator = 2.0 * (a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]);
      for (int axis = 0; axis < 3; ++axis) {
        center[axis] = (aa * bc[axis] + bb * ca[axis] + cc * ab[axis]) / denominator;
      }
    }
    double radius = 0.0;
    for (int c = 0; c < D; ++c) {
      radius += center[c] * center[c];
    }
    radius = std::sqrt(radius);
    const Vertex& origin = vertices_[static_cast<std::size_t>(v[0])];
    double required = 0.0;
    double scale = 1.0;
    for (int j = 0; j < D; ++j) {
      double fractional =
          input_.fractional[static_cast<int64_t>(origin.representative) * D + j] +
          static_cast<double>(origin.shift[static_cast<std::size_t>(j)]);
      double norm = 0.0;
      for (int c = 0; c < D; ++c) {
        fractional += center[c] * input_.inverse[c * D + j];
        norm += input_.inverse[c * D + j] * input_.inverse[c * D + j];
      }
      const double reach = radius * std::sqrt(norm);
      required = std::max({required, reach - fractional, fractional + reach - 1.0});
      scale = std::max({scale, std::fabs(fractional), reach});
    }
    const double margin = required + kCertificateSlack * scale;
    return std::isfinite(margin) ? margin : std::numeric_limits<double>::infinity();
  }

  // Exact dyadic center/radius bounds keep the translated-image search complete.
  double support_bounds_exact(const Cell& v,
                              const ExactLatticeTransform<D>& transform,
                              std::array<double, D>& lower_bound,
                              std::array<double, D>& upper_bound) const {
    if (!transform.valid()) return std::numeric_limits<double>::infinity();
    std::array<std::array<Dyadic, D>, D> edges{};
    for (int row = 0; row < D; ++row) {
      for (int axis = 0; axis < D; ++axis) {
        edges[static_cast<std::size_t>(row)][static_cast<std::size_t>(axis)] =
            exact::to_dyadic(exact_difference(
                v[static_cast<std::size_t>(row + 1)], v[0], axis));
      }
    }
    std::array<Dyadic, D> center_numerator{};
    Dyadic center_denominator;
    if constexpr (D == 2) {
      const Dyadic& ax = edges[0][0];
      const Dyadic& ay = edges[0][1];
      const Dyadic& bx = edges[1][0];
      const Dyadic& by = edges[1][1];
      const Dyadic det = ax * by - ay * bx;
      const Dyadic rhs_a = (ax * ax + ay * ay).scaled_power(-1);
      const Dyadic rhs_b = (bx * bx + by * by).scaled_power(-1);
      center_numerator[0] = rhs_a * by - rhs_b * ay;
      center_numerator[1] = ax * rhs_b - bx * rhs_a;
      center_denominator = det;
    } else {
      const Dyadic& ax = edges[0][0];
      const Dyadic& ay = edges[0][1];
      const Dyadic& az = edges[0][2];
      const Dyadic& bx = edges[1][0];
      const Dyadic& by = edges[1][1];
      const Dyadic& bz = edges[1][2];
      const Dyadic& cx = edges[2][0];
      const Dyadic& cy = edges[2][1];
      const Dyadic& cz = edges[2][2];
      const std::array<Dyadic, 3> cross_bc{
          by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx};
      const std::array<Dyadic, 3> cross_ca{
          cy * az - cz * ay, cz * ax - cx * az, cx * ay - cy * ax};
      const std::array<Dyadic, 3> cross_ab{
          ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx};
      const Dyadic norm_a = ax * ax + ay * ay + az * az;
      const Dyadic norm_b = bx * bx + by * by + bz * bz;
      const Dyadic norm_c = cx * cx + cy * cy + cz * cz;
      const Dyadic determinant =
          ax * cross_bc[0] + ay * cross_bc[1] + az * cross_bc[2];
      center_denominator = determinant.scaled_power(1);
      for (int axis = 0; axis < 3; ++axis) {
        center_numerator[static_cast<std::size_t>(axis)] =
            norm_a * cross_bc[static_cast<std::size_t>(axis)] +
            norm_b * cross_ca[static_cast<std::size_t>(axis)] +
            norm_c * cross_ab[static_cast<std::size_t>(axis)];
      }
    }
    if (center_denominator.sign() == 0) {
      return std::numeric_limits<double>::infinity();
    }
    if (center_denominator.sign() < 0) {
      center_denominator.negate();
      for (Dyadic& value : center_numerator) value.negate();
    }

    Dyadic radius_numerator;
    for (const Dyadic& component : center_numerator) {
      radius_numerator = radius_numerator + component * component;
    }
    const Dyadic radius_denominator =
        center_denominator * center_denominator;
    const Vertex& origin = vertices_[static_cast<std::size_t>(v[0])];
    double required = -std::numeric_limits<double>::infinity();
    double scale = 1.0;
    for (int axis = 0; axis < D; ++axis) {
      const int64_t origin_row = origin.representative;
      const double* origin_point = input_.points + origin_row * D;
      Dyadic fractional_origin_numerator =
          transform.point_fractional_numerator(origin_point, axis) +
          Dyadic(static_cast<double>(origin.shift[static_cast<std::size_t>(axis)])) *
              transform.denominator();
      Dyadic fractional_numerator = fractional_origin_numerator * center_denominator;
      for (int component = 0; component < D; ++component) {
        fractional_numerator =
            fractional_numerator +
            center_numerator[static_cast<std::size_t>(component)] *
                transform.inverse_numerator(component, axis);
      }
      const Dyadic fractional_denominator =
          transform.denominator() * center_denominator;
      const exact::QuotientBounds fractional =
          exact::directed_quotient_bounds(fractional_numerator,
                                          fractional_denominator);
      if (!std::isfinite(fractional.lower) || !std::isfinite(fractional.upper)) {
        return std::numeric_limits<double>::infinity();
      }
      const double reach = upper_sqrt_ratio(
          radius_numerator * transform.inverse_norm_numerator(axis),
          radius_denominator * transform.denominator_squared());
      if (!std::isfinite(reach)) return std::numeric_limits<double>::infinity();
      const double lower =
          std::nextafter(fractional.lower - reach,
                         -std::numeric_limits<double>::infinity());
      const double upper = add_up(fractional.upper, reach);
      if (!std::isfinite(lower) || !std::isfinite(upper)) {
        return std::numeric_limits<double>::infinity();
      }
      lower_bound[static_cast<std::size_t>(axis)] = lower;
      upper_bound[static_cast<std::size_t>(axis)] = upper;
      const double lower_requirement = subtract_up(0.0, lower);
      const double upper_requirement = subtract_up(upper, 1.0);
      required = std::max({required, lower_requirement, upper_requirement});
      scale = std::max({scale, std::fabs(fractional.lower),
                        std::fabs(fractional.upper), reach});
    }
    const double slack =
        std::nextafter(kCertificateSlack * scale,
                       std::numeric_limits<double>::infinity());
    const double margin = add_up(required, slack);
    return std::isfinite(margin) ? margin : std::numeric_limits<double>::infinity();
  }

  int64_t closure_misses(const std::vector<Candidate>& extracted,
                         const std::unordered_map<std::size_t, std::vector<int32_t>>& index) {
    // Map every extracted slot to its orbit, then check each finite neighbor.
    std::vector<int32_t> orbit_of(static_cast<std::size_t>(slots()), -1);
    for (int32_t cell = 0; cell < slots(); ++cell) {
      if (!alive_[static_cast<std::size_t>(cell)] || touches_bound(cell)) {
        continue;
      }
      const std::vector<int32_t> key = canonical(cell).key;
      const auto found = index.find(hash(key));
      if (found == index.end()) {
        continue;
      }
      for (const int32_t candidate : found->second) {
        if (extracted[static_cast<std::size_t>(candidate)].key == key) {
          orbit_of[static_cast<std::size_t>(cell)] = candidate;
        }
      }
    }
    int64_t misses = 0;
    for (int32_t cell = 0; cell < slots(); ++cell) {
      if (!alive_[static_cast<std::size_t>(cell)] || touches_bound(cell)) {
        continue;
      }
      const int32_t anchor = anchor_of(cells_[static_cast<std::size_t>(cell)]);
      if (!is_base(anchor)) {
        continue;
      }
      for (const int32_t neighbor : neighbors_[static_cast<std::size_t>(cell)]) {
        if (neighbor < 0 || orbit_of[static_cast<std::size_t>(neighbor)] < 0) {
          ++misses;
        }
      }
    }
    return misses;
  }

  void add_bounding_simplex(const std::vector<double>& approximate) {
    std::array<double, D> lower;
    std::array<double, D> upper;
    lower.fill(std::numeric_limits<double>::infinity());
    upper.fill(-std::numeric_limits<double>::infinity());
    for (std::size_t index = 0; index < approximate.size() / D; ++index) {
      for (int c = 0; c < D; ++c) {
        lower[static_cast<std::size_t>(c)] =
            std::min(lower[static_cast<std::size_t>(c)], approximate[index * D + c]);
        upper[static_cast<std::size_t>(c)] =
            std::max(upper[static_cast<std::size_t>(c)], approximate[index * D + c]);
      }
    }
    double extent = 0.0;
    for (int c = 0; c < D; ++c) {
      extent = std::max(extent, upper[static_cast<std::size_t>(c)] - lower[static_cast<std::size_t>(c)]);
    }
    extent = extent > 0.0 ? extent : 1.0;
    // The simplex {x_c >= center_c - 32 R, sum_c (x_c - center_c) <= 32 R}
    // contains the box of half-width R, and it lies 30 R beyond every image.
    bound_base_ = static_cast<int32_t>(vertices_.size());
    for (int k = 0; k <= D; ++k) {
      for (int c = 0; c < D; ++c) {
        const double center =
            0.5 * (lower[static_cast<std::size_t>(c)] + upper[static_cast<std::size_t>(c)]);
        const double offset = (k == c + 1) ? (32.0 * D) * extent : -32.0 * extent;
        bound_[static_cast<std::size_t>(k * D + c)] = center + offset;
      }
      Vertex bound;
      bound.representative = -1;
      bound.shift.fill(0);
      vertices_.push_back(bound);
    }
    Cell cell;
    for (int k = 0; k <= D; ++k) {
      cell[static_cast<std::size_t>(k)] = bound_base_ + k;
    }
    if (orient_sign(cell) < 0) {
      std::swap(cell[0], cell[1]);
    }
    Cell none;
    none.fill(-1);
    last_ = allocate(cell);
    neighbors_[static_cast<std::size_t>(last_)] = none;
  }

  int32_t allocate(const Cell& cell) {
    if (!free_.empty()) {
      const int32_t slot = free_.back();
      free_.pop_back();
      cells_[static_cast<std::size_t>(slot)] = cell;
      alive_[static_cast<std::size_t>(slot)] = 1;
      return slot;
    }
    if (slots() >= INT32_MAX) throw PeriodicIndexRangeExceeded{};
    cells_.push_back(cell);
    Cell none;
    none.fill(-1);
    neighbors_.push_back(none);
    alive_.push_back(1);
    cavity_mark_.push_back(0);
    outside_mark_.push_back(0);
    return static_cast<int32_t>(cells_.size() - 1);
  }

  int32_t locate(int32_t q) {
    int32_t cell = last_;
    const int64_t limit = 4 * slots() + 64;
    for (int64_t step = 0; step <= limit; ++step) {
      const int start =
          static_cast<int>(splitmix64(static_cast<uint64_t>(q) * 0x9E37ULL + step) % (D + 1));
      bool moved = false;
      for (int k = 0; k <= D; ++k) {
        const int i = (start + k) % (D + 1);
        Cell probe = cells_[static_cast<std::size_t>(cell)];
        probe[static_cast<std::size_t>(i)] = q;
        if (orient_sign(probe) < 0) {
          const int32_t next = neighbors_[static_cast<std::size_t>(cell)][static_cast<std::size_t>(i)];
          if (next < 0) {
            throw std::logic_error("periodic walk left the bounding simplex");
          }
          cell = next;
          moved = true;
          break;
        }
      }
      if (!moved) {
        return cell;
      }
    }
    throw std::logic_error("periodic walk did not terminate");
  }

  static uint64_t ridge_key(const Cell& cell, int skip_first, int skip_second) {
    uint64_t low = std::numeric_limits<uint64_t>::max();
    uint64_t high = 0;
    for (int k = 0; k <= D; ++k) {
      if (k == skip_first || k == skip_second) {
        continue;
      }
      const uint64_t value = static_cast<uint32_t>(cell[static_cast<std::size_t>(k)]);
      low = std::min(low, value);
      high = std::max(high, value);
    }
    return D == 2 ? low : (low << 32) | high;
  }

  void insert(int32_t q) {
    const int32_t start = locate(q);
    for (const int32_t vertex : cells_[static_cast<std::size_t>(start)]) {
      if (same_point(vertex, q)) {
        throw DuplicatePoint{vertices_[static_cast<std::size_t>(q)].representative};
      }
    }
    ++stamp_;
    std::vector<int32_t> cavity{start};
    std::vector<std::pair<int32_t, int>> boundary;
    cavity_mark_[static_cast<std::size_t>(start)] = stamp_;
    for (std::size_t head = 0; head < cavity.size(); ++head) {
      const int32_t cell = cavity[head];
      for (int i = 0; i <= D; ++i) {
        const int32_t neighbor = neighbors_[static_cast<std::size_t>(cell)][static_cast<std::size_t>(i)];
        if (neighbor >= 0 && cavity_mark_[static_cast<std::size_t>(neighbor)] == stamp_) {
          continue;
        }
        if (neighbor >= 0 && outside_mark_[static_cast<std::size_t>(neighbor)] != stamp_ &&
            conflict(neighbor, q) > 0) {
          cavity_mark_[static_cast<std::size_t>(neighbor)] = stamp_;
          cavity.push_back(neighbor);
          continue;
        }
        if (neighbor >= 0) {
          outside_mark_[static_cast<std::size_t>(neighbor)] = stamp_;
        }
        boundary.emplace_back(cell, i);
      }
    }
    std::unordered_map<uint64_t, std::pair<int32_t, int>> ridges;
    ridges.reserve(boundary.size() * D);
    int32_t created = -1;
    for (const auto& [cell, face] : boundary) {
      Cell vertices = cells_[static_cast<std::size_t>(cell)];
      vertices[static_cast<std::size_t>(face)] = q;
      if (orient_sign(vertices) <= 0) {
        throw std::logic_error("periodic cavity is not star-shaped");
      }
      const int32_t outside = neighbors_[static_cast<std::size_t>(cell)][static_cast<std::size_t>(face)];
      created = allocate(vertices);
      Cell& links = neighbors_[static_cast<std::size_t>(created)];
      links.fill(-1);
      links[static_cast<std::size_t>(face)] = outside;
      if (outside >= 0) {
        for (int32_t& back : neighbors_[static_cast<std::size_t>(outside)]) {
          if (back == cell) {
            back = created;
          }
        }
      }
      for (int j = 0; j <= D; ++j) {
        if (j == face) {
          continue;
        }
        const uint64_t key = ridge_key(vertices, face, j);
        const auto found = ridges.find(key);
        if (found == ridges.end()) {
          ridges.emplace(key, std::make_pair(created, j));
          continue;
        }
        const auto [other, other_face] = found->second;
        neighbors_[static_cast<std::size_t>(created)][static_cast<std::size_t>(j)] = other;
        neighbors_[static_cast<std::size_t>(other)][static_cast<std::size_t>(other_face)] = created;
        ridges.erase(found);
      }
    }
    if (!ridges.empty() || created < 0) {
      throw std::logic_error("periodic cavity boundary is not closed");
    }
    for (const int32_t cell : cavity) {
      alive_[static_cast<std::size_t>(cell)] = 0;
      free_.push_back(cell);
    }
    last_ = created;
  }
};

template <int D>
int64_t count_images(const PeriodicDelaunayInput& input, double margin,
                     std::vector<typename PeriodicEngine<D>::Vertex>* images) {
  const double lower = -margin - kImageSlack;
  const double upper = 1.0 + margin + kImageSlack;
  if (!std::isfinite(lower) || !std::isfinite(upper)) return -1;
  int64_t total = 0;
  for (int64_t point = 0; point < input.point_count; ++point) {
    std::array<int32_t, D> first{}, last{};
    int64_t count = 1;
    bool present = true;
    for (int axis = 0; axis < D; ++axis) {
      const double fractional = input.fractional[point * D + axis];
      const double lo = std::ceil(lower - fractional);
      const double hi = std::floor(upper - fractional);
      if (!std::isfinite(lo) || !std::isfinite(hi) ||
          lo < static_cast<double>(INT32_MIN) || lo > static_cast<double>(INT32_MAX) ||
          hi < static_cast<double>(INT32_MIN) || hi > static_cast<double>(INT32_MAX)) {
        return -1;
      }
      if (lo > hi) {
        present = false;
        count = 0;
        break;
      }
      first[static_cast<std::size_t>(axis)] = static_cast<int32_t>(lo);
      last[static_cast<std::size_t>(axis)] = static_cast<int32_t>(hi);
      const int64_t span = static_cast<int64_t>(last[static_cast<std::size_t>(axis)]) -
                           first[static_cast<std::size_t>(axis)] + 1;
      if (span <= 0 || count > (INT32_MAX - (D + 1) - total) / span) return -1;
      count *= span;
    }
    if (!present) continue;
    total += count;
    if (images == nullptr) continue;
    typename PeriodicEngine<D>::Vertex vertex;
    vertex.representative = static_cast<int32_t>(point);
    vertex.shift = first;
    while (true) {
      images->push_back(vertex);
      int axis = 0;
      while (axis < D) {
        int32_t& value = vertex.shift[static_cast<std::size_t>(axis)];
        if (value < last[static_cast<std::size_t>(axis)]) {
          ++value;
          break;
        }
        value = first[static_cast<std::size_t>(axis)];
        ++axis;
      }
      if (axis == D) break;
    }
  }
  return total;
}

template <int D>
int32_t run(const PeriodicDelaunayInput& input, PeriodicDelaunayResult& result) {
  double margin = input.initial_margin;
  for (;;) {
    const int64_t count = count_images<D>(input, margin, nullptr);
    if (count < 0) return PHX_MC_INDEX_RANGE_ERROR;
    std::vector<typename PeriodicEngine<D>::Vertex> images;
    images.reserve(static_cast<std::size_t>(count));
    if (count_images<D>(input, margin, &images) != count) return PHX_MC_INTERNAL_ERROR;
    PeriodicEngine<D> engine(input, std::move(images));
    try {
      engine.build();
    } catch (const DuplicatePoint&) {
      return PHX_MC_INVALID_INPUT;
    } catch (const PeriodicIndexRangeExceeded&) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    std::vector<typename PeriodicEngine<D>::Candidate> extracted;
    double required = 0.0;
    int64_t uncertified = 0;
    try {
      uncertified = engine.extract(margin, extracted, required);
    } catch (const PeriodicIndexRangeExceeded&) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    if (uncertified == 0 && !extracted.empty()) {
      result.dimension = D;
      result.certified_margin = margin;
      result.vertices.clear();
      result.shifts.clear();
      result.vertices.reserve(extracted.size() * (D + 1));
      result.shifts.reserve(extracted.size() * (D + 1) * D);
      for (const auto& candidate : extracted) {
        for (int i = 0; i <= D; ++i) {
          result.vertices.push_back(candidate.ordered[static_cast<std::size_t>(i)]);
          for (int j = 0; j < D; ++j) {
            result.shifts.push_back(
                candidate.shifts[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
          }
        }
      }
      return PHX_MC_OK;
    }
    if (margin > std::numeric_limits<double>::max() / 2.0) return PHX_MC_UNRESOLVED;
    margin *= 2.0;
  }
}


}  // namespace

int32_t periodic_delaunay(const PeriodicDelaunayInput& input, PeriodicDelaunayResult& result) {
  switch (input.dimension) {
    case 2:
      return run<2>(input, result);
    case 3:
      return run<3>(input, result);
    default:
      return PHX_MC_INVALID_ARGUMENT;
  }
}

namespace {

uint64_t periodic_cell_hash_word(uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

template <int D>
int32_t local_delaunay_run(const PeriodicLocalDelaunayInput& input,
                           const ExactLatticeTransform<D>& transform,
                           PeriodicLocalDelaunayResult& result) {
  const PeriodicDelaunayInput& source = input.triangulation;
  const int64_t count = source.point_count;
  std::vector<int32_t> sorted_to_source(static_cast<std::size_t>(count));
  for (int64_t row = 0; row < count; ++row) {
    sorted_to_source[static_cast<std::size_t>(row)] = static_cast<int32_t>(row);
  }
  const bool already_sorted =
      std::is_sorted(input.point_global_ids, input.point_global_ids + count);
  if (!already_sorted) {
    std::sort(sorted_to_source.begin(), sorted_to_source.end(),
              [&input](int32_t left, int32_t right) {
                return input.point_global_ids[left] < input.point_global_ids[right];
              });
  }

  std::vector<double> sorted_points;
  std::vector<double> sorted_fractional;
  const double* points = source.points;
  const double* fractional = source.fractional;
  if (!already_sorted) {
    const std::size_t item_count = static_cast<std::size_t>(count) * D;
    sorted_points.resize(item_count);
    sorted_fractional.resize(item_count);
    for (int64_t row = 0; row < count; ++row) {
      const int32_t original = sorted_to_source[static_cast<std::size_t>(row)];
      for (int axis = 0; axis < D; ++axis) {
        const std::size_t sorted_index = static_cast<std::size_t>(row) * D + axis;
        const std::size_t original_index = static_cast<std::size_t>(original) * D + axis;
        sorted_points[sorted_index] = source.points[original_index];
        sorted_fractional[sorted_index] = source.fractional[original_index];
      }
    }
    points = sorted_points.data();
    fractional = sorted_fractional.data();
  }

  PeriodicDelaunayInput sorted_input = source;
  sorted_input.points = points;
  sorted_input.fractional = fractional;
  PeriodicDelaunayResult triangulation;
  const int32_t status = run<D>(sorted_input, triangulation);
  if (status != PHX_MC_OK) return status;
  std::array<double, D> owner_lower;
  std::array<double, D> owner_upper;
  owner_lower.fill(std::numeric_limits<double>::infinity());
  owner_upper.fill(-std::numeric_limits<double>::infinity());
  bool has_owned_sites = false;
  for (int64_t row = 0; row < count; ++row) {
    if (input.point_owner_ranks[row] != input.local_owner_rank) continue;
    const double* point = source.points + row * D;
    for (int axis = 0; axis < D; ++axis) {
      const exact::QuotientBounds bounds =
          transform.point_fractional_bounds(point, axis);
      if (!std::isfinite(bounds.lower) || !std::isfinite(bounds.upper)) {
        return PHX_MC_UNRESOLVED;
      }
      owner_lower[static_cast<std::size_t>(axis)] =
          std::min(owner_lower[static_cast<std::size_t>(axis)], bounds.lower);
      owner_upper[static_cast<std::size_t>(axis)] =
          std::max(owner_upper[static_cast<std::size_t>(axis)], bounds.upper);
    }
    has_owned_sites = true;
  }

  constexpr int arity = D + 1;
  if (triangulation.vertices.size() % arity != 0 ||
      triangulation.vertices.size() >
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return PHX_MC_INTERNAL_ERROR;
  }
  const int64_t cell_count =
      static_cast<int64_t>(triangulation.vertices.size()) / arity;
  if (!addressable(cell_count, arity * D, sizeof(int32_t)) ||
      triangulation.shifts.size() !=
          static_cast<std::size_t>(cell_count) * arity * D) {
    return PHX_MC_INTERNAL_ERROR;
  }
  if (!addressable(cell_count, arity, sizeof(int32_t)) ||
      !addressable(cell_count, arity, sizeof(int64_t))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const int64_t support_vertex_count = cell_count * arity;
  if (support_vertex_count > INT32_MAX - (D + 1) ||
      !addressable(cell_count, D, sizeof(double))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  std::vector<typename PeriodicEngine<D>::Vertex> support_vertices(
      static_cast<std::size_t>(support_vertex_count));
  for (int64_t vertex = 0; vertex < support_vertex_count; ++vertex) {
    auto& image = support_vertices[static_cast<std::size_t>(vertex)];
    image.representative = triangulation.vertices[static_cast<std::size_t>(vertex)];
    for (int axis = 0; axis < D; ++axis) {
      image.shift[static_cast<std::size_t>(axis)] =
          triangulation.shifts[static_cast<std::size_t>(vertex) * D + axis];
    }
  }
  PeriodicEngine<D> support_engine(sorted_input, std::move(support_vertices));
  std::vector<double> support_lower;
  std::vector<double> support_upper;
  support_lower.reserve(static_cast<std::size_t>(cell_count * D));
  support_upper.reserve(static_cast<std::size_t>(cell_count * D));
  for (int64_t cell_row = 0; cell_row < cell_count; ++cell_row) {
    typename PeriodicEngine<D>::Cell cell{};
    for (int vertex = 0; vertex < arity; ++vertex) {
      cell[static_cast<std::size_t>(vertex)] =
          static_cast<int32_t>(cell_row * arity + vertex);
    }
    const int orientation = support_engine.support_orientation_sign(cell);
    if (orientation == 0) return PHX_MC_INTERNAL_ERROR;
    if (orientation < 0) {
      std::swap(cell[static_cast<std::size_t>(D - 1)],
                cell[static_cast<std::size_t>(D)]);
    }
    std::array<double, D> lower{};
    std::array<double, D> upper{};
    const double support_margin =
        support_engine.support_fractional_bounds(cell, transform, lower, upper);
    if (!std::isfinite(support_margin)) return PHX_MC_UNRESOLVED;
    support_lower.insert(support_lower.end(), lower.begin(), lower.end());
    support_upper.insert(support_upper.end(), upper.begin(), upper.end());
  }

  result = PeriodicLocalDelaunayResult{};
  result.dimension = D;
  result.certified_margin = triangulation.certified_margin;
  result.has_owned_sites = has_owned_sites ? 1 : 0;
  if (has_owned_sites) {
    for (int axis = 0; axis < D; ++axis) {
      result.owner_fractional_lower[static_cast<std::size_t>(axis)] =
          owner_lower[static_cast<std::size_t>(axis)];
      result.owner_fractional_upper[static_cast<std::size_t>(axis)] =
          owner_upper[static_cast<std::size_t>(axis)];
    }
  }
  result.support_fractional_lower = std::move(support_lower);
  result.support_fractional_upper = std::move(support_upper);
  result.vertices.reserve(static_cast<std::size_t>(cell_count * arity));
  result.shifts.reserve(static_cast<std::size_t>(cell_count * arity * D));
  result.cell_ids.reserve(static_cast<std::size_t>(cell_count));
  result.cell_key_ids.reserve(static_cast<std::size_t>(cell_count * arity));
  result.cell_key_relative_shifts.reserve(
      static_cast<std::size_t>(cell_count * arity * D));
  result.cell_owner_ranks.reserve(static_cast<std::size_t>(cell_count));
  result.support_cell_mask.reserve(static_cast<std::size_t>(cell_count));

  std::map<int64_t, std::vector<int64_t>> ids_by_key;
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    struct KeyVertex {
      int32_t source_row;
      int64_t global_id;
      std::array<int32_t, D> shift;
      std::array<int32_t, D> relative_shift;
    };
    std::array<KeyVertex, arity> vertices{};
    int anchor = 0;
    for (int vertex = 0; vertex < arity; ++vertex) {
      const std::size_t offset = static_cast<std::size_t>(cell) * arity + vertex;
      const int32_t sorted_row = triangulation.vertices[offset];
      const int32_t source_row =
          sorted_to_source[static_cast<std::size_t>(sorted_row)];
      KeyVertex& item = vertices[static_cast<std::size_t>(vertex)];
      item.source_row = source_row;
      item.global_id = input.point_global_ids[source_row];
      for (int axis = 0; axis < D; ++axis) {
        item.shift[static_cast<std::size_t>(axis)] =
            triangulation.shifts[offset * D + axis];
      }
      const KeyVertex& current_anchor = vertices[static_cast<std::size_t>(anchor)];
      if (item.global_id < current_anchor.global_id ||
          (item.global_id == current_anchor.global_id && item.shift < current_anchor.shift)) {
        anchor = vertex;
      }
    }
    const KeyVertex& anchor_vertex = vertices[static_cast<std::size_t>(anchor)];
    for (KeyVertex& item : vertices) {
      for (int axis = 0; axis < D; ++axis) {
        const int64_t relative =
            static_cast<int64_t>(item.shift[static_cast<std::size_t>(axis)]) -
            anchor_vertex.shift[static_cast<std::size_t>(axis)];
        if (relative < INT32_MIN || relative > INT32_MAX) {
          return PHX_MC_INDEX_RANGE_ERROR;
        }
        item.relative_shift[static_cast<std::size_t>(axis)] =
            static_cast<int32_t>(relative);
      }
    }
    std::array<int, arity> key_order{};
    for (int vertex = 0; vertex < arity; ++vertex) {
      key_order[static_cast<std::size_t>(vertex)] = vertex;
    }
    std::sort(key_order.begin(), key_order.end(), [&vertices](int left, int right) {
      const KeyVertex& a = vertices[static_cast<std::size_t>(left)];
      const KeyVertex& b = vertices[static_cast<std::size_t>(right)];
      if (a.global_id != b.global_id) return a.global_id < b.global_id;
      return a.relative_shift < b.relative_shift;
    });

    uint64_t hash = 0x243F6A8885A308D3ULL;
    std::vector<int64_t> semantic_key;
    semantic_key.reserve(static_cast<std::size_t>(arity * (D + 1)));
    bool owned_support = false;
    for (int position = 0; position < arity; ++position) {
      const KeyVertex& item =
          vertices[static_cast<std::size_t>(key_order[static_cast<std::size_t>(position)])];
      semantic_key.push_back(item.global_id);
      result.cell_key_ids.push_back(item.global_id);
      hash = periodic_cell_hash_word(hash ^ static_cast<uint64_t>(item.global_id));
      owned_support =
          owned_support ||
          input.point_owner_ranks[item.source_row] == input.local_owner_rank;
      for (int axis = 0; axis < D; ++axis) {
        const int32_t relative = item.relative_shift[static_cast<std::size_t>(axis)];
        semantic_key.push_back(relative);
        result.cell_key_relative_shifts.push_back(relative);
        hash = periodic_cell_hash_word(hash ^ static_cast<uint32_t>(relative));
      }
    }
    const int64_t cell_id = static_cast<int64_t>(hash & 0x7FFFFFFFFFFFFFFFULL);
    const auto [found, inserted] = ids_by_key.emplace(cell_id, semantic_key);
    if (!inserted) {
      return found->second == semantic_key ? PHX_MC_INTERNAL_ERROR
                                           : PHX_MC_INDEX_RANGE_ERROR;
    }
    result.cell_ids.push_back(cell_id);
    const KeyVertex& owner_vertex =
        vertices[static_cast<std::size_t>(key_order.front())];
    result.cell_owner_ranks.push_back(input.point_owner_ranks[owner_vertex.source_row]);
    result.support_cell_mask.push_back(owned_support ? 1 : 0);

    for (int vertex = 0; vertex < arity; ++vertex) {
      const std::size_t offset = static_cast<std::size_t>(cell) * arity + vertex;
      const int32_t sorted_row = triangulation.vertices[offset];
      result.vertices.push_back(
          sorted_to_source[static_cast<std::size_t>(sorted_row)]);
      for (int axis = 0; axis < D; ++axis) {
        result.shifts.push_back(triangulation.shifts[offset * D + axis]);
      }
    }
  }
  return PHX_MC_OK;
}

template <int D>
int32_t owner_support_query_run(const PeriodicSupportQueryInput& input,
                                PeriodicSupportQueryResult& result) {
  constexpr int arity = D + 1;
  const ExactLatticeTransform<D> transform(input.lattice);
  if (!transform.valid()) return PHX_MC_INVALID_INPUT;
  if (input.cell_count > (INT64_MAX - input.point_count) / arity) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const int64_t cell_vertex_count = input.cell_count * arity;
  const int64_t total_points = cell_vertex_count + input.point_count;
  if (total_points > INT32_MAX - (D + 1) ||
      !addressable(total_points, D, sizeof(double))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  std::vector<double> points(static_cast<std::size_t>(total_points) * D);
  std::vector<typename PeriodicEngine<D>::Vertex> images(
      static_cast<std::size_t>(total_points));
  for (int64_t vertex = 0; vertex < cell_vertex_count; ++vertex) {
    for (int axis = 0; axis < D; ++axis) {
      const std::size_t target = static_cast<std::size_t>(vertex) * D + axis;
      points[target] = input.cell_vertex_points[target];
    }
    auto& image = images[static_cast<std::size_t>(vertex)];
    image.representative = static_cast<int32_t>(vertex);
    for (int axis = 0; axis < D; ++axis) {
      image.shift[static_cast<std::size_t>(axis)] =
          input.cell_vertex_shifts[static_cast<std::size_t>(vertex) * D + axis];
    }
  }
  for (int64_t site = 0; site < input.point_count; ++site) {
    const int64_t representative = cell_vertex_count + site;
    for (int axis = 0; axis < D; ++axis) {
      const std::size_t target = static_cast<std::size_t>(representative) * D + axis;
      const std::size_t source = static_cast<std::size_t>(site) * D + axis;
      points[target] = input.points[source];
    }
    auto& image = images[static_cast<std::size_t>(representative)];
    image.representative = static_cast<int32_t>(representative);
    image.shift.fill(0);
  }
  PeriodicDelaunayInput predicate_input;
  predicate_input.dimension = D;
  predicate_input.point_count = total_points;
  predicate_input.points = points.data();
  predicate_input.fractional = nullptr;
  predicate_input.lattice = input.lattice;
  predicate_input.inverse = nullptr;
  predicate_input.initial_margin = 1.0;
  PeriodicEngine<D> engine(predicate_input, std::move(images));

  struct Hit {
    int64_t cell_row;
    int64_t point_id;
    std::array<int32_t, D> image_shift;
    int8_t sign;
  };
  std::vector<Hit> hits;
  for (int64_t cell_row = 0; cell_row < input.cell_count; ++cell_row) {
    typename PeriodicEngine<D>::Cell cell{};
    const int64_t first_vertex = cell_row * arity;
    for (int vertex = 0; vertex < arity; ++vertex) {
      cell[static_cast<std::size_t>(vertex)] =
          static_cast<int32_t>(first_vertex + vertex);
    }
    const int orientation = engine.support_orientation_sign(cell);
    if (orientation == 0) return PHX_MC_INVALID_INPUT;
    if (orientation < 0) {
      std::swap(cell[static_cast<std::size_t>(D - 1)],
                cell[static_cast<std::size_t>(D)]);
    }
    std::array<double, D> lower_fractional{};
    std::array<double, D> upper_fractional{};
    const double margin =
        engine.support_fractional_bounds(cell, transform, lower_fractional,
                                         upper_fractional);
    if (!std::isfinite(margin)) return PHX_MC_UNRESOLVED;
    for (int64_t site = 0; site < input.point_count; ++site) {
      std::array<int64_t, D> first_shift{};
      std::array<int64_t, D> last_shift{};
      bool has_images = true;
      const double* source_point = input.points + site * D;
      for (int axis = 0; axis < D; ++axis) {
        const exact::QuotientBounds source_fractional =
            transform.point_fractional_bounds(source_point, axis);
        if (!std::isfinite(source_fractional.lower) ||
            !std::isfinite(source_fractional.upper)) {
          return PHX_MC_UNRESOLVED;
        }
        const double lower = std::ceil(std::nextafter(
            lower_fractional[static_cast<std::size_t>(axis)] - source_fractional.upper,
            -std::numeric_limits<double>::infinity()));
        const double upper = std::floor(std::nextafter(
            upper_fractional[static_cast<std::size_t>(axis)] - source_fractional.lower,
            std::numeric_limits<double>::infinity()));
        if (!std::isfinite(lower) || !std::isfinite(upper) ||
            lower < static_cast<double>(INT32_MIN) ||
            lower > static_cast<double>(INT32_MAX) ||
            upper < static_cast<double>(INT32_MIN) ||
            upper > static_cast<double>(INT32_MAX)) {
          return PHX_MC_INDEX_RANGE_ERROR;
        }
        first_shift[static_cast<std::size_t>(axis)] = static_cast<int64_t>(lower);
        last_shift[static_cast<std::size_t>(axis)] = static_cast<int64_t>(upper);
        has_images = has_images && lower <= upper;
      }
      if (!has_images) continue;

      std::array<int32_t, D> shift{};
      const int32_t query_representative =
          static_cast<int32_t>(cell_vertex_count + site);
      auto visit_images = [&](auto&& self, int axis) -> void {
        if (axis == D) {
          for (int vertex = 0; vertex < arity; ++vertex) {
            const std::size_t cell_vertex =
                static_cast<std::size_t>(first_vertex + vertex);
            bool same_image = input.cell_vertex_global_ids[cell_vertex] ==
                              input.point_global_ids[site];
            for (int component = 0; component < D; ++component) {
              same_image =
                  same_image &&
                  input.cell_vertex_shifts[cell_vertex * D + component] ==
                      shift[static_cast<std::size_t>(component)];
            }
            if (same_image) return;
          }
          const int sign = engine.support_insphere_sign(cell, query_representative, shift);
          if (sign >= 0) {
            hits.push_back(
                {cell_row, input.point_global_ids[site], shift, static_cast<int8_t>(sign)});
          }
          return;
        }
        const std::size_t component = static_cast<std::size_t>(axis);
        for (int64_t image = first_shift[component]; image <= last_shift[component]; ++image) {
          shift[component] = static_cast<int32_t>(image);
          self(self, axis + 1);
        }
      };
      visit_images(visit_images, 0);
    }
  }
  std::sort(hits.begin(), hits.end(), [](const Hit& left, const Hit& right) {
    if (left.cell_row != right.cell_row) return left.cell_row < right.cell_row;
    if (left.point_id != right.point_id) return left.point_id < right.point_id;
    return left.image_shift < right.image_shift;
  });
  if (hits.size() > static_cast<std::size_t>(INT64_MAX)) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const int64_t hit_count = static_cast<int64_t>(hits.size());
  if (!addressable(hit_count, 1, sizeof(int64_t)) ||
      !addressable(hit_count, D, sizeof(int32_t))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  result = PeriodicSupportQueryResult{};
  result.dimension = D;
  result.candidate_revision = input.candidate_revision;
  result.cell_rows.reserve(hits.size());
  result.point_global_ids.reserve(hits.size());
  result.image_shifts.reserve(hits.size() * D);
  result.predicate_signs.reserve(hits.size());
  for (const Hit& hit : hits) {
    result.cell_rows.push_back(hit.cell_row);
    result.point_global_ids.push_back(hit.point_id);
    result.image_shifts.insert(result.image_shifts.end(), hit.image_shift.begin(),
                               hit.image_shift.end());
    result.predicate_signs.push_back(hit.sign);
  }
  return PHX_MC_OK;
}

}  // namespace

int32_t periodic_delaunay_local(const PeriodicLocalDelaunayInput& input,
                                PeriodicLocalDelaunayResult& result) {
  const int dimension = input.triangulation.dimension;
  if (dimension != 2 && dimension != 3) return PHX_MC_INVALID_ARGUMENT;
  result = PeriodicLocalDelaunayResult{};
  if (dimension == 2) {
    const ExactLatticeTransform<2> transform(input.triangulation.lattice);
    if (!transform.valid()) return PHX_MC_INVALID_INPUT;
    if (input.triangulation.point_count == 0) {
      result.dimension = dimension;
      return PHX_MC_OK;
    }
    return local_delaunay_run<2>(input, transform, result);
  }
  const ExactLatticeTransform<3> transform(input.triangulation.lattice);
  if (!transform.valid()) return PHX_MC_INVALID_INPUT;
  if (input.triangulation.point_count == 0) {
    result.dimension = dimension;
    return PHX_MC_OK;
  }
  return local_delaunay_run<3>(input, transform, result);
}

int32_t periodic_owner_support_query(const PeriodicSupportQueryInput& input,
                                     PeriodicSupportQueryResult& result) {
  switch (input.dimension) {
    case 2:
      return owner_support_query_run<2>(input, result);
    case 3:
      return owner_support_query_run<3>(input, result);
    default:
      return PHX_MC_INVALID_ARGUMENT;
  }
}

}  // namespace phx::mc

struct phx_mc_periodic_triangulation {
  phx::mc::PeriodicDelaunayResult result;
};

struct phx_mc_periodic_local_delaunay {
  phx::mc::PeriodicLocalDelaunayResult result;
};

struct phx_mc_periodic_support_query_result {
  phx::mc::PeriodicSupportQueryResult result;
};

namespace {

int32_t validate_periodic_input(const phx::mc::PeriodicDelaunayInput& input) {
  const int d = input.dimension;
  if ((d != 2 && d != 3) || input.point_count < 1 || input.points == nullptr ||
      input.fractional == nullptr || input.lattice == nullptr || input.inverse == nullptr ||
      !phx::mc::addressable(input.point_count, d, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (input.point_count > INT32_MAX - 1) return PHX_MC_INDEX_RANGE_ERROR;
  if (!std::isfinite(input.initial_margin) || !(input.initial_margin > 0.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t items = input.point_count * d;
  for (int64_t index = 0; index < items; ++index) {
    if (!std::isfinite(input.points[index]) || !std::isfinite(input.fractional[index])) {
      return PHX_MC_NONFINITE_INPUT;
    }
  }
  for (int index = 0; index < d * d; ++index) {
    if (!std::isfinite(input.lattice[index]) || !std::isfinite(input.inverse[index])) {
      return PHX_MC_NONFINITE_INPUT;
    }
  }
  for (int64_t index = 0; index < items; ++index) {
    if (!phx::mc::coordinate_in_domain(input.points[index])) return PHX_MC_RANGE_ERROR;
  }
  for (int index = 0; index < d * d; ++index) {
    if (!phx::mc::coordinate_in_domain(input.lattice[index])) return PHX_MC_RANGE_ERROR;
  }
  for (int64_t index = 0; index < items; ++index) {
    if (!(std::fabs(input.fractional[index]) <= 0x1p30)) return PHX_MC_RANGE_ERROR;
  }
  return PHX_MC_OK;
}

int32_t validate_periodic_local_request(
    const phx_mc_periodic_local_delaunay_request& request) {
  const int dimension = request.dimension;
  if ((dimension != 2 && dimension != 3) || request.point_count < 0 ||
      request.local_owner_rank < 0 || request.lattice == nullptr ||
      request.inverse == nullptr || !std::isfinite(request.initial_margin) ||
      !(request.initial_margin > 0.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (request.point_count > 0 &&
      (request.points == nullptr || request.fractional == nullptr ||
       request.point_global_ids == nullptr || request.point_owner_ranks == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  phx::mc::PeriodicDelaunayInput periodic;
  periodic.dimension = dimension;
  periodic.point_count = request.point_count;
  periodic.points = request.points;
  periodic.fractional = request.fractional;
  periodic.lattice = request.lattice;
  periodic.inverse = request.inverse;
  periodic.initial_margin = request.initial_margin;
  int32_t status = PHX_MC_OK;
  if (request.point_count > 0) {
    status = validate_periodic_input(periodic);
    if (status != PHX_MC_OK) return status;
  } else {
    for (int index = 0; index < dimension * dimension; ++index) {
      if (!std::isfinite(request.lattice[index]) ||
          !std::isfinite(request.inverse[index])) {
        return PHX_MC_NONFINITE_INPUT;
      }
      if (!phx::mc::coordinate_in_domain(request.lattice[index])) {
        return PHX_MC_RANGE_ERROR;
      }
    }
  }
  std::set<int64_t> identifiers;
  for (int64_t row = 0; row < request.point_count; ++row) {
    if (request.point_global_ids[row] < 0 || request.point_owner_ranks[row] < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!identifiers.insert(request.point_global_ids[row]).second) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

int32_t validate_periodic_support_request(
    const phx_mc_periodic_support_query_request& request) {
  const int dimension = request.dimension;
  if ((dimension != 2 && dimension != 3) || request.cell_count < 0 ||
      request.point_count < 0 || request.lattice == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (request.point_count > INT32_MAX - (dimension + 1) ||
      request.cell_count >
          (INT32_MAX - (dimension + 1) - request.point_count) / (dimension + 1)) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const int64_t cell_vertex_width = (dimension + 1) * dimension;
  if (!phx::mc::addressable(request.cell_count, cell_vertex_width, sizeof(double)) ||
      !phx::mc::addressable(request.cell_count, dimension + 1, sizeof(int64_t)) ||
      !phx::mc::addressable(request.point_count, dimension, sizeof(double))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  if (request.cell_count > 0 &&
      (request.cell_vertex_points == nullptr ||
       request.cell_vertex_global_ids == nullptr ||
       request.cell_vertex_shifts == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (request.point_count > 0 &&
      (request.points == nullptr || request.point_global_ids == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t cell_components = request.cell_count * cell_vertex_width;
  const int64_t point_components = request.point_count * dimension;
  for (int64_t index = 0; index < cell_components; ++index) {
    if (!std::isfinite(request.cell_vertex_points[index])) {
      return PHX_MC_NONFINITE_INPUT;
    }
    if (!phx::mc::coordinate_in_domain(request.cell_vertex_points[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  for (int64_t index = 0; index < point_components; ++index) {
    if (!std::isfinite(request.points[index])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(request.points[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  for (int index = 0; index < dimension * dimension; ++index) {
    if (!std::isfinite(request.lattice[index])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(request.lattice[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  for (int64_t vertex = 0; vertex < request.cell_count * (dimension + 1); ++vertex) {
    if (request.cell_vertex_global_ids[vertex] < 0) return PHX_MC_INVALID_ARGUMENT;
  }
  std::set<int64_t> identifiers;
  for (int64_t site = 0; site < request.point_count; ++site) {
    if (request.point_global_ids[site] < 0) return PHX_MC_INVALID_ARGUMENT;
    if (!identifiers.insert(request.point_global_ids[site]).second) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_periodic_delaunay(int32_t dimension, int64_t point_count,
                                 const double* points, const double* fractional,
                                 const double* lattice, const double* inverse,
                                 double initial_margin,
                                 phx_mc_periodic_triangulation** triangulation) {
  return phx::mc::guarded([&]() -> int32_t {
    if (triangulation == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *triangulation = nullptr;
    phx::mc::PeriodicDelaunayInput input;
    input.dimension = dimension;
    input.point_count = point_count;
    input.points = points;
    input.fractional = fractional;
    input.lattice = lattice;
    input.inverse = inverse;
    input.initial_margin = initial_margin;
    const int32_t valid = validate_periodic_input(input);
    if (valid != PHX_MC_OK) return valid;
    auto handle = std::make_unique<phx_mc_periodic_triangulation>();
    const int32_t status = phx::mc::periodic_delaunay(input, handle->result);
    if (status == PHX_MC_OK) *triangulation = handle.release();
    return status;
  });
}

int64_t phx_mc_periodic_cell_count(const phx_mc_periodic_triangulation* triangulation) {
  if (triangulation == nullptr) return -1;
  if (triangulation->result.dimension == 0) return 0;
  return static_cast<int64_t>(triangulation->result.vertices.size()) /
         (triangulation->result.dimension + 1);
}

int32_t phx_mc_periodic_copy_cells(const phx_mc_periodic_triangulation* triangulation,
                                   int64_t cell_capacity, int32_t* vertices,
                                   int32_t* shifts) {
  return phx::mc::guarded([&]() -> int32_t {
    if (triangulation == nullptr || cell_capacity < 0 ||
        !phx::mc::addressable(cell_capacity, triangulation->result.dimension + 1,
                              sizeof(int32_t)) ||
        !phx::mc::addressable(cell_capacity,
                              (triangulation->result.dimension + 1) *
                                  triangulation->result.dimension,
                              sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int64_t count = phx_mc_periodic_cell_count(triangulation);
    if (cell_capacity < count) return PHX_MC_CAPACITY_EXCEEDED;
    if (count > 0 && (vertices == nullptr || shifts == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::copy(triangulation->result.vertices.begin(), triangulation->result.vertices.end(),
              vertices);
    std::copy(triangulation->result.shifts.begin(), triangulation->result.shifts.end(), shifts);
    return PHX_MC_OK;
  });
}

void phx_mc_periodic_free(phx_mc_periodic_triangulation* triangulation) {
  delete triangulation;
}

int32_t phx_mc_periodic_delaunay_local(
    const phx_mc_periodic_local_delaunay_request* request) {
  return phx::mc::guarded([&]() -> int32_t {
    if (request == nullptr || request->result == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *request->result = nullptr;
    const int32_t valid = validate_periodic_local_request(*request);
    if (valid != PHX_MC_OK) return valid;
    phx::mc::PeriodicLocalDelaunayInput input;
    input.triangulation.dimension = request->dimension;
    input.triangulation.point_count = request->point_count;
    input.triangulation.points = request->points;
    input.triangulation.fractional = request->fractional;
    input.triangulation.lattice = request->lattice;
    input.triangulation.inverse = request->inverse;
    input.triangulation.initial_margin = request->initial_margin;
    input.point_global_ids = request->point_global_ids;
    input.point_owner_ranks = request->point_owner_ranks;
    input.local_owner_rank = request->local_owner_rank;
    auto handle = std::make_unique<phx_mc_periodic_local_delaunay>();
    const int32_t status = phx::mc::periodic_delaunay_local(input, handle->result);
    if (status == PHX_MC_OK) *request->result = handle.release();
    return status;
  });
}

int32_t phx_mc_periodic_local_delaunay_sizes(
    const phx_mc_periodic_local_delaunay* result, int64_t* cell_count,
    double* certified_margin) {
  if (result == nullptr || cell_count == nullptr || certified_margin == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *cell_count = static_cast<int64_t>(result->result.cell_ids.size());
  *certified_margin = result->result.certified_margin;
  return PHX_MC_OK;
}

int32_t phx_mc_periodic_local_delaunay_owner_bounds(
    const phx_mc_periodic_local_delaunay* result, int8_t* has_owned_sites,
    double* fractional_lower, double* fractional_upper) {
  if (result == nullptr || has_owned_sites == nullptr ||
      fractional_lower == nullptr || fractional_upper == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int dimension = result->result.dimension;
  if ((dimension != 2 && dimension != 3) ||
      !phx::mc::addressable(1, dimension, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *has_owned_sites = result->result.has_owned_sites;
  std::copy_n(result->result.owner_fractional_lower.begin(), dimension,
              fractional_lower);
  std::copy_n(result->result.owner_fractional_upper.begin(), dimension,
              fractional_upper);
  return PHX_MC_OK;
}


int32_t phx_mc_periodic_local_delaunay_copy(
    const phx_mc_periodic_local_delaunay* result, int64_t cell_capacity,
    int32_t* vertices, int32_t* shifts, int64_t* cell_ids, int64_t* cell_key_ids,
    int32_t* cell_key_relative_shifts, int32_t* cell_owner_ranks,
    int8_t* support_cell_mask, double* support_fractional_lower,
    double* support_fractional_upper) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || cell_capacity < 0) return PHX_MC_INVALID_ARGUMENT;
    const int dimension = result->result.dimension;
    if ((dimension != 2 && dimension != 3) ||
        !phx::mc::addressable(cell_capacity, dimension + 1, sizeof(int32_t)) ||
        !phx::mc::addressable(cell_capacity, (dimension + 1) * dimension, sizeof(int32_t)) ||
        !phx::mc::addressable(cell_capacity, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(cell_capacity, dimension + 1, sizeof(int64_t)) ||
        !phx::mc::addressable(cell_capacity, (dimension + 1) * dimension,
                              sizeof(int32_t)) ||
        !phx::mc::addressable(cell_capacity, 1, sizeof(int32_t)) ||
        !phx::mc::addressable(cell_capacity, 1, sizeof(int8_t)) ||
        !phx::mc::addressable(cell_capacity, dimension, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int64_t count = static_cast<int64_t>(result->result.cell_ids.size());
    if (cell_capacity < count) return PHX_MC_CAPACITY_EXCEEDED;
    if (count > 0 &&
        (vertices == nullptr || shifts == nullptr || cell_ids == nullptr ||
         cell_key_ids == nullptr || cell_key_relative_shifts == nullptr ||
         cell_owner_ranks == nullptr || support_cell_mask == nullptr ||
         support_fractional_lower == nullptr || support_fractional_upper == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0) {
      std::copy(result->result.vertices.begin(), result->result.vertices.end(), vertices);
      std::copy(result->result.shifts.begin(), result->result.shifts.end(), shifts);
      std::copy(result->result.cell_ids.begin(), result->result.cell_ids.end(), cell_ids);
      std::copy(result->result.cell_key_ids.begin(), result->result.cell_key_ids.end(),
                cell_key_ids);
      std::copy(result->result.cell_key_relative_shifts.begin(),
                result->result.cell_key_relative_shifts.end(), cell_key_relative_shifts);
      std::copy(result->result.cell_owner_ranks.begin(),
                result->result.cell_owner_ranks.end(), cell_owner_ranks);
      std::copy(result->result.support_cell_mask.begin(),
                result->result.support_cell_mask.end(), support_cell_mask);
      std::copy(result->result.support_fractional_lower.begin(),
                result->result.support_fractional_lower.end(), support_fractional_lower);
      std::copy(result->result.support_fractional_upper.begin(),
                result->result.support_fractional_upper.end(), support_fractional_upper);
    }
    return PHX_MC_OK;
  });
}

void phx_mc_periodic_local_delaunay_free(
    phx_mc_periodic_local_delaunay* result) {
  delete result;
}

int32_t phx_mc_periodic_owner_support_query(
    const phx_mc_periodic_support_query_request* request) {
  return phx::mc::guarded([&]() -> int32_t {
    if (request == nullptr || request->result == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *request->result = nullptr;
    const int32_t valid = validate_periodic_support_request(*request);
    if (valid != PHX_MC_OK) return valid;
    phx::mc::PeriodicSupportQueryInput input;
    input.dimension = request->dimension;
    input.cell_count = request->cell_count;
    input.cell_vertex_global_ids = request->cell_vertex_global_ids;
    input.cell_vertex_points = request->cell_vertex_points;
    input.cell_vertex_shifts = request->cell_vertex_shifts;
    input.point_count = request->point_count;
    input.point_global_ids = request->point_global_ids;
    input.points = request->points;
    input.lattice = request->lattice;
    input.candidate_revision = request->candidate_revision;
    auto result = std::make_unique<phx_mc_periodic_support_query_result>();
    const int32_t status = phx::mc::periodic_owner_support_query(input, result->result);
    if (status == PHX_MC_OK) *request->result = result.release();
    return status;
  });
}

int32_t phx_mc_periodic_support_query_sizes(
    const phx_mc_periodic_support_query_result* result, int64_t* hit_count,
    uint64_t* candidate_revision) {
  if (result == nullptr || hit_count == nullptr || candidate_revision == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *hit_count = static_cast<int64_t>(result->result.cell_rows.size());
  *candidate_revision = result->result.candidate_revision;
  return PHX_MC_OK;
}

int32_t phx_mc_periodic_support_query_copy_hits(
    const phx_mc_periodic_support_query_result* result, int64_t hit_capacity,
    int64_t* cell_rows, int64_t* point_global_ids, int32_t* image_shifts,
    int8_t* predicate_signs) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || hit_capacity < 0 ||
        !phx::mc::addressable(hit_capacity, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(hit_capacity, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(hit_capacity, result->result.dimension, sizeof(int32_t)) ||
        !phx::mc::addressable(hit_capacity, 1, sizeof(int8_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int64_t count = static_cast<int64_t>(result->result.cell_rows.size());
    if (hit_capacity < count) return PHX_MC_CAPACITY_EXCEEDED;
    if (count > 0 &&
        (cell_rows == nullptr || point_global_ids == nullptr || image_shifts == nullptr ||
         predicate_signs == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0) {
      std::copy(result->result.cell_rows.begin(), result->result.cell_rows.end(), cell_rows);
      std::copy(result->result.point_global_ids.begin(),
                result->result.point_global_ids.end(), point_global_ids);
      std::copy(result->result.image_shifts.begin(), result->result.image_shifts.end(),
                image_shifts);
      std::copy(result->result.predicate_signs.begin(),
                result->result.predicate_signs.end(), predicate_signs);
    }
    return PHX_MC_OK;
  });
}

void phx_mc_periodic_support_query_free(
    phx_mc_periodic_support_query_result* result) {
  delete result;
}
}  // extern "C"

namespace phx::mc {
namespace {

using QEdge = std::array<int32_t, 4>;
struct QImage {
  int32_t representative;
  std::array<int32_t, 2> shift;
  auto operator<=>(const QImage&) const = default;
};
struct QTriangle {
  std::array<QImage, 3> v;
  bool live = true;
};
struct QIncidence {
  int32_t cell;
  int32_t side;
};
using QSides = std::vector<QIncidence>;
using QBank = std::map<QEdge, QSides>;
using QMerit = std::tuple<int64_t, int64_t, double, int64_t, double>;
int32_t checked_shift(int64_t value) {
  if (value < INT32_MIN || value > INT32_MAX) throw PeriodicIndexRangeExceeded{};
  return static_cast<int32_t>(value);
}

int32_t checked_floor_shift(double value) {
  if (!std::isfinite(value)) throw PeriodicIndexRangeExceeded{};
  const double floor_value = std::floor(value);
  if (floor_value < static_cast<double>(INT32_MIN) ||
      floor_value > static_cast<double>(INT32_MAX)) throw PeriodicIndexRangeExceeded{};
  return static_cast<int32_t>(floor_value);
}

int32_t checked_nearest_shift(double value) {
  const double rounded = std::nearbyint(value);
  if (!std::isfinite(rounded) || rounded < static_cast<double>(INT32_MIN) ||
      rounded > static_cast<double>(INT32_MAX)) throw PeriodicIndexRangeExceeded{};
  return static_cast<int32_t>(rounded);
}

QEdge quotient_edge(QImage a, QImage b) {
  if (b < a) std::swap(a, b);
  return {a.representative, b.representative,
          checked_shift(static_cast<int64_t>(b.shift[0]) - a.shift[0]),
          checked_shift(static_cast<int64_t>(b.shift[1]) - a.shift[1])};
}
QEdge triangle_edge(const QTriangle& t, int side) {
  return quotient_edge(t.v[side], t.v[(side + 1) % 3]);
}

// Augmented deterministic treap. Candidate readers never mutate it. Ranks and
// selection are logarithmic; the serial owner reserves new nodes before edit.
class QuotientRanks final {
 public:
  using Key = std::pair<double, QEdge>;
  struct Node {
    Key key;
    uint64_t priority;
    int32_t left = -1, right = -1;
    int64_t size = 1;
    double sum = 0.0;
  };
  void reserve(std::size_t additions) {
    reserve_for(nodes_, nodes_.size() + additions);
    reserve_for(free_, nodes_.size() + additions);
  }
  int64_t size() const { return count(root_); }
  int64_t rank(const Key& key) const {
    int64_t result = 0;
    for (int32_t id = root_; id >= 0;) {
      const Node& n = nodes_[id];
      if (n.key < key) {
        result += count(n.left) + 1;
        id = n.right;
      } else id = n.left;
    }
    return result;
  }
  double excess(double threshold) const {
    double result=0.0;
    for (int32_t id=root_;id>=0;) {
      const Node& n=nodes_[id];
      if (n.key.first>threshold) {
        result+=n.key.first-threshold+sum(n.right)-
                threshold*static_cast<double>(count(n.right));
        id=n.left;
      } else id=n.right;
    }
    return result;
  }
  Key at(int64_t rank) const {
    for (int32_t id = root_; id >= 0;) {
      const Node& n = nodes_[id];
      const int64_t left = count(n.left);
      if (rank == left) return n.key;
      if (rank < left) id = n.left;
      else { rank -= left + 1; id = n.right; }
    }
    throw std::logic_error("periodic edge rank outside bank");
  }
  void insert(const Key& key) {
    int32_t id;
    if (free_.empty()) {
      id = static_cast<int32_t>(nodes_.size());
      nodes_.push_back(Node{key, priority(key.second), -1, -1, 1, key.first});
    } else {
      id = free_.back(); free_.pop_back();
      nodes_[id] = Node{key, priority(key.second), -1, -1, 1, key.first};
    }
    root_ = put(root_, id);
  }
  void erase(const Key& key) { root_ = remove(root_, key); }
 private:
  std::vector<Node> nodes_;
  std::vector<int32_t> free_;
  int32_t root_ = -1;
  int64_t count(int32_t id) const { return id < 0 ? 0 : nodes_[id].size; }
  double sum(int32_t id) const { return id<0 ? 0.0 : nodes_[id].sum; }
  void update(int32_t id) {
    nodes_[id].size = 1 + count(nodes_[id].left) + count(nodes_[id].right);
    nodes_[id].sum = nodes_[id].key.first + sum(nodes_[id].left) + sum(nodes_[id].right);
  }
  static uint64_t priority(const QEdge& key) {
    uint64_t h = 0x243f6a8885a308d3ULL;
    for (int32_t x : key) h = splitmix64(h ^ static_cast<uint32_t>(x));
    return h;
  }
  int32_t merge(int32_t a, int32_t b) {
    if (a < 0) return b;
    if (b < 0) return a;
    if (nodes_[a].priority < nodes_[b].priority) {
      nodes_[a].right = merge(nodes_[a].right, b); update(a); return a;
    }
    nodes_[b].left = merge(a, nodes_[b].left); update(b); return b;
  }
  int32_t put(int32_t root, int32_t id) {
    if (root < 0) return id;
    if (nodes_[id].key < nodes_[root].key) {
      nodes_[root].left = put(nodes_[root].left, id);
      const int32_t child = nodes_[root].left;
      if (nodes_[child].priority < nodes_[root].priority) {
        nodes_[root].left = nodes_[child].right;
        nodes_[child].right = root; update(root); update(child); return child;
      }
    } else {
      nodes_[root].right = put(nodes_[root].right, id);
      const int32_t child = nodes_[root].right;
      if (nodes_[child].priority < nodes_[root].priority) {
        nodes_[root].right = nodes_[child].left;
        nodes_[child].left = root; update(root); update(child); return child;
      }
    }
    update(root); return root;
  }
  int32_t remove(int32_t root, const Key& key) {
    if (root < 0) throw std::logic_error("periodic missing edge rank");
    if (nodes_[root].key == key) {
      const int32_t next = merge(nodes_[root].left, nodes_[root].right);
      free_.push_back(root); return next;
    }
    if (key < nodes_[root].key) nodes_[root].left = remove(nodes_[root].left, key);
    else nodes_[root].right = remove(nodes_[root].right, key);
    update(root); return root;
  }
};

// Rank arithmetic on a local deletion/insertion bank, without copying or
// sorting the global bank. Each requested order statistic costs O(k log n).
class HypotheticalRanks final {
 public:
  using Key = QuotientRanks::Key;
  HypotheticalRanks(const QuotientRanks& base, std::vector<Key> removed,
                    std::vector<Key> added)
      : base_(base), removed_(std::move(removed)), added_(std::move(added)) {
    std::sort(removed_.begin(), removed_.end());
    std::sort(added_.begin(), added_.end());
    for (const Key& key : removed_) removed_ranks_.push_back(base_.rank(key));
  }
  int64_t size() const {
    return base_.size() - static_cast<int64_t>(removed_.size()) +
           static_cast<int64_t>(added_.size());
  }
  int64_t rank(const Key& key) const {
    return base_.rank(key) -
           (std::lower_bound(removed_.begin(), removed_.end(), key) - removed_.begin()) +
           (std::lower_bound(added_.begin(), added_.end(), key) - added_.begin());
  }
  double at(int64_t r) const {
    int64_t preceding = 0;
    for (std::size_t i = 0; i < added_.size(); ++i) {
      const int64_t position = base_.rank(added_[i]) -
          (std::lower_bound(removed_.begin(), removed_.end(), added_[i]) - removed_.begin()) +
          static_cast<int64_t>(i);
      if (position == r) return added_[i].first;
      if (position > r) break;
      ++preceding;
    }
    int64_t original = r - preceding;
    for (int64_t removed : removed_ranks_) if (removed <= original) ++original;
    return base_.at(original).first;
  }
  double quantile(double q) const {
    const double r = q * static_cast<double>(size() - 1);
    const int64_t lo = static_cast<int64_t>(std::floor(r));
    const int64_t hi = static_cast<int64_t>(std::ceil(r));
    const double a = at(lo), b = at(hi), fraction = r - static_cast<double>(lo);
    // NumPy's linear interpolation avoids loss near the upper endpoint.
    return fraction >= 0.5 ? b - (b - a) * (1.0 - fraction) : a + (b - a) * fraction;
  }
  double excess(double threshold) const {
    double result=base_.excess(threshold);
    for (const Key& key:removed_)
      result-=std::max(0.0,key.first-threshold);
    for (const Key& key:added_)
      result+=std::max(0.0,key.first-threshold);
    return std::max(0.0,result);
  }
 private:
  const QuotientRanks& base_;
  std::vector<Key> removed_, added_;
  std::vector<int64_t> removed_ranks_;
};

class QuotientPredicates final {
 public:
  QuotientPredicates(const std::vector<double>& points, const double* lattice,
                     const std::array<double, 2>* candidate = nullptr)
      : points_(points), lattice_(lattice), candidate_(candidate) {}
  int64_t exact = 0, perturbed = 0;
  double base(QImage a, int c) const {
    return a.representative == static_cast<int32_t>(points_.size() / 2)
               ? (*candidate_)[c] : points_[2 * static_cast<std::size_t>(a.representative) + c];
  }
  template <class T> T difference(QImage a, QImage b, int c) const {
    T value;
    if constexpr (std::is_same_v<T, Approx>)
      value = Approx::exact(base(a, c)) - Approx::exact(base(b, c));
    else value = Expansion::difference(base(a, c), base(b, c));
    for (int j = 0; j < 2; ++j) {
      const int64_t delta = static_cast<int64_t>(a.shift[j]) - b.shift[j];
      if (delta==0) continue;
      if constexpr (std::is_same_v<T, Approx>)
        value = value + Approx::exact(static_cast<double>(delta)) * Approx::exact(lattice_[2*j+c]);
      else value = value + Expansion::product(static_cast<double>(delta), lattice_[2*j+c]);
    }
    return value;
  }
  template <class T> T orientation(const std::array<QImage,3>& v) const {
    return difference<T>(v[1],v[0],0)*difference<T>(v[2],v[0],1) -
           difference<T>(v[1],v[0],1)*difference<T>(v[2],v[0],0);
  }
  int orient(const std::array<QImage,3>& v) {
    const int sign = orientation<Approx>(v).certified_sign();
    if (sign != 2) return sign;
    ++exact; return orientation<Expansion>(v).sign();
  }
  template <class T> T circle(const std::array<QImage,3>& v, QImage q) const {
    T rows[3][3];
    for (int i=0;i<3;++i) {
      rows[i][0]=difference<T>(v[i],q,0); rows[i][1]=difference<T>(v[i],q,1);
      rows[i][2]=rows[i][0]*rows[i][0]+rows[i][1]*rows[i][1];
    }
    return det3(rows[0],rows[1],rows[2]);
  }
  int lex(QImage a,QImage b) {
    for (int c=0;c<2;++c) {
      int sign=difference<Approx>(a,b,c).certified_sign();
      if (sign==2) { ++exact; sign=difference<Expansion>(a,b,c).sign(); }
      if (sign) return sign;
    }
    return 0;
  }
  int conflict(const std::array<QImage,3>& v,QImage q) {
    int sign=circle<Approx>(v,q).certified_sign();
    if (sign==2) { ++exact; sign=circle<Expansion>(v,q).sign(); }
    if (sign) return sign;
    ++perturbed;
    std::array<QImage,4> order={v[0],v[1],v[2],q};
    std::sort(order.begin(),order.end(),[&](QImage a,QImage b){return lex(a,b)<0;});
    for (int r=3;r>1;--r) {
      if (order[r]==q) return -1;
      for (int k=0;k<3;++k) if (v[k]==order[r]) {
        auto replaced=v; replaced[k]=q;
        sign=orient(replaced); if (sign) return sign;
      }
    }
    throw std::logic_error("periodic quotient perturbation left a tie");
  }
 private:
  const std::vector<double>& points_;
  const double* lattice_;
  const std::array<double,2>* candidate_;
};

struct QuotientCandidate {
  std::array<double,2> site;
  int32_t seed;
  std::vector<int32_t> removed;
  std::vector<QTriangle> proposal;
  QBank edges;
  std::map<int32_t,std::array<int32_t,2>> lifts;
  std::map<QEdge,double> lengths;
  std::map<int32_t,double> growth;
  QMerit merit{};
  int32_t status=PHX_MC_OK;
};


struct QuotientInterval {
  double lo,hi;
  static QuotientInterval exact(double x) {return {x,x};}
  static QuotientInterval filtered(Approx x) {
    return {std::nextafter(x.value-x.bound,-INFINITY),
            std::nextafter(x.value+x.bound,INFINITY)};
  }
};
QuotientInterval operator+(QuotientInterval a,QuotientInterval b) {
  return {std::nextafter(a.lo+b.lo,-INFINITY),std::nextafter(a.hi+b.hi,INFINITY)};
}
QuotientInterval operator-(QuotientInterval a,QuotientInterval b) {
  return {std::nextafter(a.lo-b.hi,-INFINITY),std::nextafter(a.hi-b.lo,INFINITY)};
}
QuotientInterval operator*(QuotientInterval a,QuotientInterval b) {
  const std::array<double,4> products={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
  if (std::any_of(products.begin(),products.end(),[](double x){return std::isnan(x);}))
    return {-INFINITY,INFINITY};
  return {std::nextafter(*std::min_element(products.begin(),products.end()),-INFINITY),
          std::nextafter(*std::max_element(products.begin(),products.end()),INFINITY)};
}
QuotientInterval operator/(QuotientInterval a,QuotientInterval b) {
  if (b.lo<=0.0 && b.hi>=0.0) return {-INFINITY,INFINITY};
  return a*QuotientInterval{std::nextafter(1.0/b.hi,-INFINITY),
                            std::nextafter(1.0/b.lo,INFINITY)};
}
using QuotientWindow=std::array<QuotientInterval,2>;

struct PeriodicRefinementInvalidInput {};

class PeriodicQuotientOwner final {
 public:
  explicit PeriodicQuotientOwner(const phx_mc_periodic_refinement_input& input)
      : input_(input), points_(input.points, input.points + 2 * input.point_count) {
    const auto scalar=[](double x){return QuotientInterval::exact(x);};
    const auto det=scalar(input.lattice[0])*scalar(input.lattice[3])-
                   scalar(input.lattice[1])*scalar(input.lattice[2]);
    exact_inverse_={scalar(input.lattice[3])/det,scalar(-input.lattice[1])/det,
                    scalar(-input.lattice[2])/det,scalar(input.lattice[0])/det};
    windows_.reserve(static_cast<std::size_t>(input.cell_count));
    vertex_cells_.resize(input.point_count);
    vertex_edges_.resize(input.point_count);
    lifts_.resize(input.point_count);
    ratios_.resize(input.point_count,1.0);
    cells_.reserve(static_cast<std::size_t>(input.cell_count));
    for (int64_t i=0;i<input.cell_count;++i) {
      QTriangle t;
      for (int j=0;j<3;++j) t.v[j]={input.vertices[3*i+j],
          {input.shifts[6*i+2*j],input.shifts[6*i+2*j+1]}};
      QuotientPredicates p(points_,input.lattice);
      if (p.orient(t.v) <= 0) throw PeriodicRefinementInvalidInput{};
      canonical(t,nullptr);
      cells_.push_back(t);
      windows_.push_back(window(t,nullptr));
      for (int j=0;j<3;++j) {
        edges_[triangle_edge(t,j)].push_back({static_cast<int32_t>(i),j});
        vertex_cells_[t.v[j].representative].insert(static_cast<int32_t>(i));
      }
    }
    for (const auto& [key,sides]:edges_) {
      if (sides.size() != 2) throw PeriodicRefinementInvalidInput{};
      vertex_edges_[key[0]].insert(key); vertex_edges_[key[1]].insert(key);
    }
    for (int64_t i=0;i<input.protected_edge_count;++i) {
      const int32_t* row=input.protected_edges+4*i;
      const QEdge key={row[0],row[1],row[2],row[3]};
      if (!edges_.contains(key) || !protected_edges_.insert(key).second)
        throw PeriodicRefinementInvalidInput{};
    }
    for (int32_t v=0;v<input.point_count;++v) lifts_[v]=reference(v,nullptr);
    ranks_.reserve(edges_.size());
    for (const auto& [key,sides]:edges_) {
      (void)sides;
      const double value=length(key,nullptr);
      if (!(value > 0.0) || !std::isfinite(value)) throw PeriodicRefinementInvalidInput{};
      lengths_[key]=value; ranks_.insert({value,key});
    }
    growth_ranks_.reserve(input.point_count);
    for (int32_t v=0;v<input.point_count;++v) {
      ratios_[v]=vertex_growth(v,nullptr);
      growth_ranks_.insert({ratios_[v],{v,0,0,0}});
    }
  }

  int32_t run(PeriodicRefinementResult& result) {
    HypotheticalRanks current_ranks(ranks_, {}, {});
    QMerit current = merit(current_ranks, growth_ranks_.at(growth_ranks_.size() - 1).first);
    double size = input_.controls[0];
    for (int64_t c = 1; c < input_.control_count; ++c) size = std::min(size, input_.controls[5 * c]);
    bool graded = false;
    bool unresolved = false;
    while (std::get<0>(current)) {
      // Exact rank comparisons remain authoritative; this guard only detects
      // when further finite improvements cannot be resolved in binary64.
      const double rounding = 32.0 * std::numeric_limits<double>::epsilon() *
                              static_cast<double>(input_.control_count) *
                              input_.statistic_count;
      if (std::get<2>(current) <= rounding) {
        unresolved = true;
        break;
      }
      if (points_.size() / 2 >= static_cast<std::size_t>(INT32_MAX - 1)) {
        return PHX_MC_INDEX_RANGE_ERROR;
      }
      std::vector<QuotientCandidate> candidates;
      generate(size, candidates, graded);
      if (candidates.empty()) {
        if (!graded) {
          graded = true;
          continue;
        }
        unresolved = true;
        break;
      }
      QuotientCandidate best;
      std::vector<QuotientCandidate> beam;
      bool improved = false;
      const int32_t status = select(candidates, current, best, improved, graded ? &beam : nullptr);
      if (status != PHX_MC_OK) return status;
      if (!improved) {
        if (!graded) {
          graded = true;
          continue;
        }
        std::array<double, 2> second_site{};
        const int32_t searched = lookahead(beam, current, size, best, second_site, improved);
        if (searched != PHX_MC_OK) return searched;
        if (!improved) {
          unresolved = true;
          break;
        }
        beam.clear();
        const int32_t first_commit = commit(best);
        if (first_commit != PHX_MC_OK) return first_commit;
        QuotientCandidate next;
        next.site = second_site;
        for (std::size_t i = 0; i < cells_.size(); ++i) {
          if (cells_[i].live) {
            next.seed = static_cast<int32_t>(i);
            break;
          }
        }
        std::vector<QuotientCandidate> second;
        second.push_back(std::move(next));
        const int32_t reevaluated = select(second, current, best, improved);
        if (reevaluated != PHX_MC_OK) return reevaluated;
        if (!improved) {
          unresolved = true;
          break;
        }
      }
      const int32_t committed = commit(best);
      if (committed != PHX_MC_OK) return committed;
      current = best.merit;
    }
    for (const QEdge& key : protected_edges_) {
      if (!edges_.contains(key) || edges_.at(key).size() != 2) return PHX_MC_INVALID_INPUT;
    }
    std::vector<int32_t> sorted;
    sorted.reserve(cells_.size());
    for (std::size_t id = 0; id < cells_.size(); ++id)
      if (cells_[id].live) sorted.push_back(static_cast<int32_t>(id));
    std::sort(sorted.begin(), sorted.end(), [&](int32_t a, int32_t b) {
      const auto& av = cells_[a].v;
      const auto& bv = cells_[b].v;
      if (av[0] != bv[0]) return av[0] < bv[0];
      const QImage& amin = std::min(av[1], av[2]);
      const QImage& bmin = std::min(bv[1], bv[2]);
      if (amin != bmin) return amin < bmin;
      return std::max(av[1], av[2]) < std::max(bv[1], bv[2]);
    });
    result.vertices.clear();
    result.shifts.clear();
    result.vertices.reserve(3 * sorted.size());
    result.shifts.reserve(6 * sorted.size());
    result.points = std::move(points_);
    for (int32_t id : sorted) {
      for (const QImage& v : cells_[id].v) {
        result.vertices.push_back(v.representative);
        result.shifts.insert(result.shifts.end(), v.shift.begin(), v.shift.end());
      }
    }
    return unresolved ? PHX_MC_UNRESOLVED : PHX_MC_OK;
  }

 private:
  const phx_mc_periodic_refinement_input& input_;
  std::vector<double> points_;
  std::vector<QTriangle> cells_;
  QBank edges_;
  std::set<QEdge> protected_edges_;
  std::map<QEdge,double> lengths_;
  std::vector<std::set<int32_t>> vertex_cells_;
  std::vector<std::set<QEdge>> vertex_edges_;
  std::vector<std::array<int32_t,2>> lifts_;
  std::vector<double> ratios_;
  QuotientRanks ranks_,growth_ranks_;
  std::vector<QuotientWindow> windows_;
  std::array<QuotientInterval,4> exact_inverse_;

  static auto primary_score(const QMerit& value) {
    return std::tuple{std::get<0>(value),std::get<1>(value),
                      std::get<2>(value),std::get<3>(value)};
  }
  static auto tail_score(const QMerit& value) {
    return std::tuple{std::get<0>(value),std::get<4>(value),
                      std::get<2>(value),std::get<1>(value),std::get<3>(value)};
  }
  int32_t select(std::vector<QuotientCandidate>& candidates, const QMerit& current,
                 QuotientCandidate& best, bool& improved,
                 std::vector<QuotientCandidate>* beam = nullptr) {
    int best_class = 2;
    const auto less = [](const QuotientCandidate& a, const QuotientCandidate& b) {
      return std::tuple{std::get<0>(a.merit), std::get<2>(a.merit), std::get<4>(a.merit),
                        std::get<1>(a.merit), std::get<3>(a.merit), a.site} <
             std::tuple{std::get<0>(b.merit), std::get<2>(b.merit), std::get<4>(b.merit),
                        std::get<1>(b.merit), std::get<3>(b.merit), b.site};
    };
    for (auto& candidate : candidates) {
      evaluate(candidate);
      if (candidate.status != PHX_MC_OK) continue;
      const int candidate_class =
          primary_score(candidate.merit) < primary_score(current) ? 0 :
          (tail_score(candidate.merit) < tail_score(current) ? 1 : 2);
      const bool better = candidate_class < best_class ||
          (candidate_class == best_class && candidate_class == 0 &&
           primary_score(candidate.merit) < primary_score(best.merit)) ||
          (candidate_class == best_class && candidate_class == 1 &&
           tail_score(candidate.merit) < tail_score(best.merit));
      if (better) {
        best = std::move(candidate);
        best_class = candidate_class;
      } else if (beam != nullptr && candidate_class == 2) {
        beam->push_back(std::move(candidate));
      }
    }
    if (beam != nullptr) std::sort(beam->begin(), beam->end(), less);
    improved = best_class < 2;
    return PHX_MC_OK;
  }
  int32_t lookahead(std::vector<QuotientCandidate>& beam,const QMerit& current,
                    double size,QuotientCandidate& first,std::array<double,2>& second_site,
                    bool& improved) {
    if (points_.size() / 2 >= static_cast<std::size_t>(INT32_MAX - 1))
      return PHX_MC_INDEX_RANGE_ERROR;
    int best_class=2;
    std::size_t selected=beam.size();
    QMerit best_merit{};
    for (std::size_t index=0;index<beam.size();++index) {
      const auto& candidate=beam[index];
      std::vector<double> points(points_);
      points.insert(points.end(),candidate.site.begin(),candidate.site.end());
      std::vector<int32_t> vertices,shifts;
      const auto append=[&](const QTriangle& triangle) {
        for (const auto& vertex:triangle.v) {
          vertices.push_back(vertex.representative);
          shifts.insert(shifts.end(),vertex.shift.begin(),vertex.shift.end());
        }
      };
      for (std::size_t id=0;id<cells_.size();++id)
        if (cells_[id].live &&
            !std::binary_search(candidate.removed.begin(),candidate.removed.end(),static_cast<int32_t>(id)))
          append(cells_[id]);
      for (const auto& triangle:candidate.proposal) append(triangle);
      auto input=input_;
      input.point_count=static_cast<int64_t>(points.size()/2);
      input.points=points.data();
      input.cell_count=static_cast<int64_t>(vertices.size()/3);
      input.vertices=vertices.data();input.shifts=shifts.data();
      PeriodicQuotientOwner trial(input);
      std::vector<QuotientCandidate> candidates;
      trial.generate(size,candidates,true);
      QuotientCandidate second;
      bool found=false;
      const int32_t status=trial.select(candidates,current,second,found);
      if (status!=PHX_MC_OK) return status;
      if (!found) continue;
      const int candidate_class=primary_score(second.merit)<primary_score(current) ? 0 : 1;
      if (candidate_class<best_class ||
          (candidate_class==best_class && candidate_class==0 &&
           primary_score(second.merit)<primary_score(best_merit)) ||
          (candidate_class==best_class && candidate_class==1 &&
           tail_score(second.merit)<tail_score(best_merit))) {
        selected=index;best_class=candidate_class;
        best_merit=second.merit;second_site=second.site;
      }
    }
    improved=selected<beam.size();
    if (improved) first=std::move(beam[selected]);
    return PHX_MC_OK;
  }

  std::array<double,2> position(QImage v,const QuotientCandidate* candidate=nullptr) const {
    std::array<double,2> p;
    for (int c=0;c<2;++c) p[c]=
        v.representative==static_cast<int32_t>(points_.size()/2) ? candidate->site[c] :
        points_[2*static_cast<std::size_t>(v.representative)+c];
    for (int c=0;c<2;++c) p[c]+=v.shift[0]*input_.lattice[c]+v.shift[1]*input_.lattice[2+c];
    return p;
  }
  std::array<double,2> fractional(std::array<double,2> p) const {
    return {p[0]*input_.inverse[0]+p[1]*input_.inverse[2],
            p[0]*input_.inverse[1]+p[1]*input_.inverse[3]};
  }
  void canonical(QTriangle& t,const QuotientCandidate* candidate) const {
    int anchor=0;
    for (int i=1;i<3;++i) if (t.v[i]<t.v[anchor]) anchor=i;
    const auto p=position({t.v[anchor].representative,{0,0}},candidate);
    const auto f=fractional(p);
    const std::array<int32_t, 2> delta = {
        checked_shift(-static_cast<int64_t>(checked_floor_shift(f[0])) -
                      t.v[anchor].shift[0]),
        checked_shift(-static_cast<int64_t>(checked_floor_shift(f[1])) -
                      t.v[anchor].shift[1])};
    for (QImage& v : t.v)
      for (int j = 0; j < 2; ++j)
        v.shift[j] = checked_shift(static_cast<int64_t>(v.shift[j]) + delta[j]);
    // Rotate rather than reflect the positive simplex.
    std::rotate(t.v.begin(),t.v.begin()+anchor,t.v.end());
  }
  const QTriangle& triangle(int32_t id,const QuotientCandidate& candidate) const {
    return id>=0 ? cells_[id] : candidate.proposal[-id-1];
  }
  QSides& mutable_edge(const QEdge& key,QuotientCandidate& candidate) const {
    auto [it,fresh]=candidate.edges.try_emplace(key);
    if (fresh) {
      auto original=edges_.find(key);
      if (original!=edges_.end()) it->second=original->second;
    }
    return it->second;
  }
  void remove(int32_t id,QuotientCandidate& candidate,std::set<QEdge>& frontier) const {
    const QTriangle t=triangle(id,candidate);
    for (int j=0;j<3;++j) {
      const QEdge key=triangle_edge(t,j);
      QSides& sides=mutable_edge(key,candidate);
      std::erase_if(sides,[&](const QIncidence& s){return s.cell==id && s.side==j;});
      frontier.insert(key);
    }
    if (id>=0) candidate.removed.push_back(id);
    else {
      candidate.proposal[-id-1].live=false;
    }
  }
  void add(QTriangle t,QuotientCandidate& candidate,std::set<QEdge>& frontier,
           QuotientPredicates& predicates) const {
    const int sign=predicates.orient(t.v);
    if (sign==0) return;
    if (sign<0) throw std::logic_error("periodic inverted cavity proposal");
    canonical(t,&candidate);
    if (candidate.proposal.size() >= static_cast<std::size_t>(INT32_MAX)) {
      throw PeriodicIndexRangeExceeded{};
    }
    const int32_t id=-static_cast<int32_t>(candidate.proposal.size())-1;
    candidate.proposal.push_back(t);
    for (int j=0;j<3;++j) {
      const QEdge key=triangle_edge(t,j);
      mutable_edge(key,candidate).push_back({id,j}); frontier.insert(key);
    }
  }
  static QTriangle translate(QTriangle t, std::array<int32_t, 2> delta) {
    for (QImage& v : t.v)
      for (int j = 0; j < 2; ++j)
        v.shift[j] = checked_shift(static_cast<int64_t>(v.shift[j]) + delta[j]);
    return t;
  }
  QTriangle align(const QTriangle& first,int side,const QTriangle& other,int back) const {
    const QImage a=first.v[side],b=first.v[(side+1)%3];
    const QImage oa=other.v[(back+1)%3],ob=other.v[back];
    if (a.representative!=oa.representative || b.representative!=ob.representative)
      throw std::logic_error("periodic inconsistent oriented edge");
    const std::array<int32_t, 2> delta = {
        checked_shift(static_cast<int64_t>(a.shift[0]) - oa.shift[0]),
        checked_shift(static_cast<int64_t>(a.shift[1]) - oa.shift[1])};
    QTriangle t=translate(other,delta);
    if (t.v[back]!=b) throw std::logic_error("periodic inconsistent edge translation");
    return t;
  }

  void cavity(QuotientCandidate& candidate) const {
    QuotientPredicates predicates(points_, input_.lattice, &candidate.site);
    const QImage q = {static_cast<int32_t>(points_.size() / 2), {0, 0}};
    int32_t seed = candidate.seed;
    QTriangle located = cells_[seed];
    const auto f = fractional(candidate.site);
    const auto center = fractional(position(located.v[0]));
    const std::array<int32_t, 2> delta = {
        checked_nearest_shift(f[0] - center[0]),
        checked_nearest_shift(f[1] - center[1])};
    located = translate(located, delta);
    std::set<std::tuple<int32_t, int32_t, int32_t>> walked;
    for (;;) {
      if (!walked.insert({seed, located.v[0].shift[0], located.v[0].shift[1]}).second) {
        throw std::logic_error("periodic point-location cycle");
      }
      int crossing = -1;
      for (int j = 0; j < 3; ++j) {
        if (predicates.orient({located.v[j], located.v[(j + 1) % 3], q}) < 0) {
          crossing = j;
          break;
        }
      }
      if (crossing < 0) break;
      const QSides& sides = edges_.at(triangle_edge(located, crossing));
      const QIncidence next = sides[0].cell == seed ? sides[1] : sides[0];
      located = align(located, crossing, cells_[next.cell], next.side);
      seed = next.cell;
    }
    std::set<QEdge> frontier;
    std::vector<std::pair<int32_t, QTriangle>> initial;
    initial.push_back({seed, located});
    for (int j = 0; j < 3; ++j) {
      if (predicates.orient({located.v[j], located.v[(j + 1) % 3], q}) != 0) continue;
      if (protected_edges_.contains(triangle_edge(located, j))) {
        candidate.status = PHX_MC_INVALID_INPUT;
        return;
      }
      const QSides& sides = edges_.at(triangle_edge(located, j));
      const QIncidence next = sides[0].cell == seed ? sides[1] : sides[0];
      initial.push_back({next.cell, align(located, j, cells_[next.cell], next.side)});
    }
    for (const auto& [id, triangle] : initial) remove(id, candidate, frontier);
    for (const auto& [id, triangle] : initial) {
      (void)id;
      for (int j = 0; j < 3; ++j)
        add({{triangle.v[j], triangle.v[(j + 1) % 3], q}}, candidate, frontier, predicates);
    }
    while (!frontier.empty()) {
      const QEdge key = *frontier.begin();
      frontier.erase(frontier.begin());
      if (protected_edges_.contains(key)) continue;
      const QSides sides = mutable_edge(key, candidate);
      if (sides.empty()) continue;
      if (sides.size() != 2) throw std::logic_error("periodic nonmanifold cavity frontier");
      const auto first = sides[0], second = sides[1];
      const QTriangle a = triangle(first.cell, candidate);
      const QTriangle b = align(a, first.side, triangle(second.cell, candidate), second.side);
      const QImage u = a.v[first.side], v = a.v[(first.side + 1) % 3];
      const QImage c = a.v[(first.side + 2) % 3], d = b.v[(second.side + 2) % 3];
      if (predicates.conflict(a.v, d) <= 0) continue;
      const QTriangle left = {{c, d, v}}, right = {{d, c, u}};
      if (predicates.orient(left.v) <= 0 || predicates.orient(right.v) <= 0) continue;
      remove(first.cell, candidate, frontier);
      remove(second.cell, candidate, frontier);
      add(left, candidate, frontier, predicates);
      add(right, candidate, frontier, predicates);
    }
    std::vector<QTriangle> compact;
    std::map<int32_t, int32_t> ids;
    for (std::size_t i = 0; i < candidate.proposal.size(); ++i) {
      if (candidate.proposal[i].live) {
        ids[-static_cast<int32_t>(i) - 1] = -static_cast<int32_t>(compact.size()) - 1;
        compact.push_back(candidate.proposal[i]);
      }
    }
    for (auto& [key, sides] : candidate.edges) {
      (void)key;
      for (QIncidence& side : sides) {
        if (side.cell < 0) side.cell = ids.at(side.cell);
      }
    }
    candidate.proposal = std::move(compact);
    std::sort(candidate.removed.begin(), candidate.removed.end());
  }

  std::array<int32_t,2> reference(int32_t vertex,const QuotientCandidate* candidate) const {
    std::array<int32_t,2> best={INT32_MAX,INT32_MAX};
    bool zero=false;
    const auto visit=[&](const QTriangle& t) {
      for (const QImage& image:t.v) if (image.representative==vertex) {
        best=std::min(best,image.shift);
        zero=zero || image.shift==std::array<int32_t,2>{0,0};
      }
    };
    if (vertex<static_cast<int32_t>(vertex_cells_.size()))
      for (int32_t id:vertex_cells_[vertex])
        if (!candidate || !std::binary_search(candidate->removed.begin(),candidate->removed.end(),id))
          visit(cells_[id]);
    if (candidate) {
      const auto proposal=candidate->lifts.find(vertex);
      if (proposal!=candidate->lifts.end()) {
        best=std::min(best,proposal->second);
        zero=zero || proposal->second==std::array<int32_t,2>{0,0};
      }
    }
    if (best[0]==INT32_MAX) throw std::logic_error("periodic unused orbit representative");
    return zero ? std::array<int32_t,2>{0,0} : best;
  }
  std::array<int32_t,2> lift(int32_t vertex,const QuotientCandidate* candidate) const {
    if (candidate) {
      const auto it=candidate->lifts.find(vertex);
      if (it!=candidate->lifts.end()) return it->second;
    }
    return lifts_[vertex];
  }
  double length(const QEdge& key,const QuotientCandidate* candidate) const {
    const auto a=lift(key[0],candidate),b=lift(key[1],candidate);
    const auto first=position({key[0],a},candidate),second=position({key[1],b},candidate);
    const int64_t sx=static_cast<int64_t>(key[2])-b[0]+a[0];
    const int64_t sy=static_cast<int64_t>(key[3])-b[1]+a[1];
    // Match published coordinates followed by relative @ lattice, not the
    // algebraically equivalent (and differently rounded) unlifted difference.
    const double x=(second[0]-first[0])+
        (static_cast<double>(sx)*input_.lattice[0]+static_cast<double>(sy)*input_.lattice[2]);
    const double y=(second[1]-first[1])+
        (static_cast<double>(sx)*input_.lattice[1]+static_cast<double>(sy)*input_.lattice[3]);
    return std::sqrt(x*x+y*y);
  }
  bool edge_present(const QEdge& key,const QuotientCandidate* candidate) const {
    if (candidate) {
      const auto it=candidate->edges.find(key);
      if (it!=candidate->edges.end()) return !it->second.empty();
    }
    return edges_.contains(key);
  }
  double vertex_growth(int32_t vertex,const QuotientCandidate* candidate,
                       std::span<const QEdge> additions={}) const {
    double lo=INFINITY,hi=0.0;
    const auto visit=[&](const QEdge& key) {
      if (!edge_present(key,candidate)) return;
      double value;
      if (candidate && candidate->lengths.contains(key)) value=candidate->lengths.at(key);
      else value=lengths_.at(key);
      lo=std::min(lo,value); hi=std::max(hi,value);
    };
    if (vertex<static_cast<int32_t>(vertex_edges_.size()))
      for (const QEdge& key:vertex_edges_[vertex]) visit(key);
    for (const QEdge& key:additions) visit(key);
    return hi==0.0 ? 1.0 : hi/lo;
  }
  double tolerance(double value) const {
    return input_.absolute_tolerance+input_.relative_tolerance*std::abs(value);
  }
  QMerit merit(const HypotheticalRanks& lengths,double growth) const {
    int64_t issues=0,longer=0,matched=0;
    double deviation=0.0,excess=0.0;
    const double minimum=lengths.at(0),maximum=lengths.at(lengths.size()-1);
    const QEdge lower={INT32_MIN,INT32_MIN,INT32_MIN,INT32_MIN};
    const QEdge upper={INT32_MAX,INT32_MAX,INT32_MAX,INT32_MAX};
    for (int64_t c=0;c<input_.control_count;++c) {
      const double* control=input_.controls+5*c;
      const double target=control[0],tol=tolerance(target);
      const bool hard=control[4]==1.0;
      longer+=lengths.size()-lengths.rank({target+tol,upper});
      excess+=lengths.excess(target+tol)/target;
      // Use the same abs(length-target) comparison as NumPy, including
      // cancellation at exactly representable tolerance endpoints.
      if (std::isinf(tol)) matched+=lengths.size();
      else {
        double lo=target-tol,hi=target+tol;
        while (std::abs(lo-target)>tol) lo=std::nextafter(lo,INFINITY);
        while (std::abs(std::nextafter(lo,-INFINITY)-target)<=tol)
          lo=std::nextafter(lo,-INFINITY);
        while (std::abs(hi-target)>tol) hi=std::nextafter(hi,-INFINITY);
        while (std::abs(std::nextafter(hi,INFINITY)-target)<=tol)
          hi=std::nextafter(hi,INFINITY);
        matched+=lengths.rank({hi,upper})-lengths.rank({lo,lower});
      }
      double statistic_deviation=0.0;
      for (int64_t s=0;s<input_.statistic_count;++s) {
        const double value=lengths.quantile(input_.quantiles[s]);
        issues+=hard && std::abs(value-target)>tol;
        statistic_deviation+=std::abs(value-target)/target;
      }
      deviation+=statistic_deviation;
      if (!std::isnan(control[2])) {
        issues+=hard && maximum>control[2]+tolerance(control[2]);
        deviation+=std::max(0.0,maximum-control[2])/control[2];
      }
      if (!std::isnan(control[1])) {
        issues+=hard && minimum<control[1]-tolerance(control[1]);
        deviation+=std::max(0.0,control[1]-minimum)/control[1];
      }
      if (!std::isnan(control[3])) {
        issues+=hard && growth>control[3]+tolerance(control[3]);
        deviation+=std::max(0.0,growth-control[3])/control[3];
      }
    }
    return {issues,longer,deviation,-matched,excess};
  }
  void evaluate_local(QuotientCandidate& candidate) const {
    cavity(candidate);
    if (candidate.status!=PHX_MC_OK) return;
    std::set<int32_t> touched;
    for (int32_t id:candidate.removed) for (QImage v:cells_[id].v) touched.insert(v.representative);
    for (const QTriangle& t:candidate.proposal) for (QImage v:t.v) touched.insert(v.representative);
    for (const QTriangle& t:candidate.proposal) for (QImage image:t.v) {
      auto [it,fresh]=candidate.lifts.try_emplace(image.representative,image.shift);
      if (!fresh && it->second!=std::array<int32_t,2>{0,0}) {
        if (image.shift==std::array<int32_t,2>{0,0}) it->second=image.shift;
        else it->second=std::min(it->second,image.shift);
      }
    }
    for (int32_t v:touched) candidate.lifts[v]=reference(v,&candidate);
    std::set<QEdge> changed;
    for (const auto& [key,sides]:candidate.edges) {
      (void)sides; changed.insert(key);
    }
    for (const auto& [v,image]:candidate.lifts)
      if (v<static_cast<int32_t>(lifts_.size()) && image!=lifts_[v])
        for (const QEdge& key:vertex_edges_[v]) changed.insert(key);
    std::vector<QuotientRanks::Key> removed,added;
    for (const QEdge& key:changed) {

      auto old=lengths_.find(key);
      if (old!=lengths_.end()) removed.push_back({old->second,key});
      if (edge_present(key,&candidate)) {
        const double value=length(key,&candidate);
        if (!(value>0.0) || !std::isfinite(value)) {
          candidate.status=PHX_MC_INVALID_INPUT; return;
        }
        candidate.lengths[key]=value; added.push_back({value,key});
      }
      touched.insert(key[0]); touched.insert(key[1]);
    }
    std::map<int32_t,std::vector<QEdge>> new_edges;
    for (const auto& [key,sides]:candidate.edges)
      if (!sides.empty() && !edges_.contains(key)) {
        new_edges[key[0]].push_back(key);
        if (key[0]!=key[1]) new_edges[key[1]].push_back(key);
      }
    std::vector<QuotientRanks::Key> growth_removed,growth_added;
    for (int32_t v:touched) {
      const double ratio=vertex_growth(v,&candidate,new_edges[v]);
      candidate.growth[v]=ratio;
      if (v<static_cast<int32_t>(ratios_.size()))
        growth_removed.push_back({ratios_[v],{v,0,0,0}});
      growth_added.push_back({ratio,{v,0,0,0}});
    }
    const HypotheticalRanks proposed(ranks_,std::move(removed),std::move(added));
    const HypotheticalRanks growth(growth_ranks_,std::move(growth_removed),std::move(growth_added));
    candidate.merit=merit(proposed,growth.at(growth.size()-1));
  }

  void evaluate(QuotientCandidate& output) const {
    QuotientCandidate candidate;
    candidate.site = output.site;
    candidate.seed = output.seed;
    evaluate_local(candidate);
    output = std::move(candidate);
  }

  std::array<double,2> frontal(const QEdge& key,double side,double size,double second_size) const {
    const auto first=position({key[0],{0,0}}),second=position({key[1],{key[2],key[3]}});
    const double dx=second[0]-first[0],dy=second[1]-first[1];
    const double norm=std::sqrt(dx*dx+dy*dy);
    const double height=std::sqrt(size*size-0.25*norm*norm);
    std::array<double,2> apex={
        0.5*(first[0]+second[0])+(side*(-dy)/norm)*height,
        0.5*(first[1]+second[1])+(side*dx/norm)*height};
    if (second_size!=size) {
      const double along=(size*size-second_size*second_size+norm*norm)/(2.0*norm);
      const double offset=std::sqrt(std::max(0.0,size*size-along*along));
      apex={first[0]+(dx/norm)*along+(side*(-dy)/norm)*offset,
            first[1]+(dy/norm)*along+(side*dx/norm)*offset};
    }
    const auto f=fractional(apex);
    const std::array<double,2> image={std::floor(f[0]),std::floor(f[1])};
    std::array<double,2> site={
        apex[0]-(image[0]*input_.lattice[0]+image[1]*input_.lattice[2]),
        apex[1]-(image[0]*input_.lattice[1]+image[1]*input_.lattice[3])};
    std::array<std::array<double,2>,5> neighbors;
    neighbors[0]=site;
    auto lower=site,upper=site;
    for (int i=0;i<2;++i) {
      for (int c=0;c<2;++c) {
        lower[c]=std::nextafter(lower[c],-INFINITY);
        upper[c]=std::nextafter(upper[c],INFINITY);
      }
      neighbors[1+2*i]=lower; neighbors[2+2*i]=upper;
    }
    std::tuple<int,int,double> best={3,0,INFINITY};
    for (const auto& x:neighbors) for (const auto& y:neighbors) {
      const std::array<double,2> trial={x[0],y[1]};
      std::array<double,2> legs;
      for (int leg=0;leg<2;++leg) {
        const int32_t vertex=key[leg];
        const double sx=image[0]-(leg ? key[2] : 0);
        const double sy=image[1]-(leg ? key[3] : 0);
        const double lx=(trial[0]-points_[2*static_cast<std::size_t>(vertex)])+
            (sx*input_.lattice[0]+sy*input_.lattice[2]);
        const double ly=(trial[1]-points_[2*static_cast<std::size_t>(vertex)+1])+
            (sx*input_.lattice[1]+sy*input_.lattice[3]);
        legs[leg]=std::sqrt(lx*lx+ly*ly);
      }
      const auto score=std::tuple{static_cast<int>(legs[0]>size)+static_cast<int>(legs[1]>size),
          -static_cast<int>(legs[0]==size)-static_cast<int>(legs[1]==size),
          std::abs(legs[0]-size)+std::abs(legs[1]-second_size)};
      if (score<best) {best=score;site=trial;}
    }
    return site;
  }
  QuotientWindow window(const QTriangle& t,const QuotientCandidate* candidate) const {
    QuotientPredicates predicates(points_,input_.lattice,candidate ? &candidate->site : nullptr);
    const Approx ux=predicates.difference<Approx>(t.v[1],t.v[0],0);
    const Approx uy=predicates.difference<Approx>(t.v[1],t.v[0],1);
    const Approx vx=predicates.difference<Approx>(t.v[2],t.v[0],0);
    const Approx vy=predicates.difference<Approx>(t.v[2],t.v[0],1);
    const Approx det=ux*vy-uy*vx;
    const Approx a=Approx::exact(0.5)*(ux*ux+uy*uy);
    const Approx b=Approx::exact(0.5)*(vx*vx+vy*vy);
    const auto cx=QuotientInterval::filtered(a*vy-b*uy)/QuotientInterval::filtered(det);
    const auto cy=QuotientInterval::filtered(ux*b-vx*a)/QuotientInterval::filtered(det);
    const double x=std::max(std::abs(cx.lo),std::abs(cx.hi));
    const double y=std::max(std::abs(cy.lo),std::abs(cy.hi));
    const double radius=std::nextafter(std::sqrt(std::nextafter(
        std::nextafter(x*x,INFINITY)+std::nextafter(y*y,INFINITY),INFINITY)),INFINITY);
    if (!std::isfinite(radius)) return {{{-INFINITY,INFINITY},{-INFINITY,INFINITY}}};
    std::array<QuotientInterval,2> center;
    for (int c=0;c<2;++c) {
      Approx anchor=Approx::exact(predicates.base(t.v[0],c));
      for (int j=0;j<2;++j)
        anchor=anchor+Approx::exact(static_cast<double>(t.v[0].shift[j]))*
                       Approx::exact(input_.lattice[2*j+c]);
      center[c]=QuotientInterval::filtered(anchor)+(c==0 ? cx : cy);
      center[c]=center[c]+QuotientInterval{-radius,radius};
    }
    return {center[0]*exact_inverse_[0]+center[1]*exact_inverse_[2],
            center[0]*exact_inverse_[1]+center[1]*exact_inverse_[3]};
  }
  uint64_t conflict_bound(const std::array<double,2>& site) {
    const auto qx=QuotientInterval::exact(site[0]),qy=QuotientInterval::exact(site[1]);
    const QuotientWindow fractional_site={qx*exact_inverse_[0]+qy*exact_inverse_[2],
                                         qx*exact_inverse_[1]+qy*exact_inverse_[3]};
    QuotientPredicates predicates(points_,input_.lattice,&site);
    uint64_t count=0;
    for (std::size_t id=0;id<cells_.size();++id) if (cells_[id].live) {
      const auto x=windows_[id][0]-fractional_site[0];
      const auto y=windows_[id][1]-fractional_site[1];
      const double lo_x=std::ceil(x.lo),hi_x=std::floor(x.hi);
      const double lo_y=std::ceil(y.lo),hi_y=std::floor(y.hi);
      if (hi_x<lo_x || hi_y<lo_y) continue;
      // An ill-conditioned envelope remains conservative, not a numerical
      // rejection. It contributes the cell without enumerating unbounded
      // possible images. The exact worker owns the scientific decision.
      if (!std::isfinite(lo_x) || !std::isfinite(hi_x) ||
          !std::isfinite(lo_y) || !std::isfinite(hi_y) ||
          lo_x < INT32_MIN || hi_x > INT32_MAX || lo_y < INT32_MIN || hi_y > INT32_MAX) {
        ++count;
        continue;
      }
      bool possible=false;
      for (int64_t sx=static_cast<int64_t>(lo_x);sx<=static_cast<int64_t>(hi_x) && !possible;++sx)
        for (int64_t sy=static_cast<int64_t>(lo_y);sy<=static_cast<int64_t>(hi_y);++sy) {
          const QImage image={static_cast<int32_t>(points_.size()/2),
                              {static_cast<int32_t>(sx),static_cast<int32_t>(sy)}};
          if (predicates.circle<Approx>(cells_[id].v,image).certified_sign()!=-1) {
            possible=true;break;
          }
        }
      count+=possible;
    }
    return count;
  }
  void generate(double size,std::vector<QuotientCandidate>& candidates,bool graded) {
    std::set<QEdge> front;
    for (const QTriangle& t:cells_) if (t.live) {
      bool open=false;
      for (int j=0;j<3;++j) open=open || lengths_.at(triangle_edge(t,j))>size;
      if (open) for (int j=0;j<3;++j) front.insert(triangle_edge(t,j));
    }
    for (const QEdge& key:front) {
      const auto first=position({key[0],{0,0}}),second=position({key[1],{key[2],key[3]}});
      const double dx=second[0]-first[0],dy=second[1]-first[1];
      const double length=std::sqrt(dx*dx+dy*dy);
      if (!(length>0.0 && length<=2.0*size)) continue;
      for (double ratio:{1.0,0.875,0.75,0.625,0.5,0.375,0.25}) {
        if (!graded && ratio!=1.0) continue;
        if (ratio!=1.0 &&
            !(std::abs(size-ratio*size)<length && length<size+ratio*size)) continue;
        for (int direction=0;direction<(ratio==1.0 ? 1 : 2);++direction) {
          const QEdge oriented = direction == 0
                                     ? key
                                     : QEdge{key[1], key[0], checked_shift(-int64_t{key[2]}),
                                             checked_shift(-int64_t{key[3]})};
          for (double side:{1.0,-1.0}) {
            const auto site=frontal(oriented,side,size,ratio*size);
            bool separated=true;
            for (std::size_t v=0;v<points_.size()/2;++v) {
              const auto gap=fractional({points_[2*v]-site[0],points_[2*v+1]-site[1]});
              const double a=gap[0]-std::nearbyint(gap[0]),b=gap[1]-std::nearbyint(gap[1]);
              const double x=a*input_.lattice[0]+b*input_.lattice[2];
              const double y=a*input_.lattice[1]+b*input_.lattice[3];
              if (std::sqrt(x*x+y*y)<0.25*size) {separated=false;break;}
            }
            if (!separated) continue;
            QuotientCandidate candidate;
            candidate.site=site; candidate.seed=edges_.at(key)[0].cell;
            candidates.push_back(std::move(candidate));
          }
        }
      }
    }
  }

  bool boundary_matches(std::span<const int32_t> removed,
                        std::span<const QTriangle> proposal) const {
    std::map<QEdge,int64_t> balance;
    const auto accumulate=[&](const QTriangle& t,int sign) {
      for (int j=0;j<3;++j) {
        const QImage a=t.v[j],b=t.v[(j+1)%3];
        balance[quotient_edge(a,b)]+=sign*(a<b ? 1 : -1);
      }
    };
    for (int32_t id:removed) accumulate(cells_[id],-1);
    for (const QTriangle& t:proposal) accumulate(t,1);
    return std::all_of(balance.begin(),balance.end(),[](const auto& row){return row.second==0;});
  }

  int32_t commit(const QuotientCandidate& candidate) {
    if (points_.size() / 2 >= static_cast<std::size_t>(INT32_MAX))
      return PHX_MC_INDEX_RANGE_ERROR;
    const int32_t vertex = static_cast<int32_t>(points_.size() / 2);
    // Allocate every secondary-index node before W01 enters its no-fail block.
    reserve_for(points_,points_.size()+2);
    reserve_for(vertex_cells_,vertex_cells_.size()+1);
    reserve_for(vertex_edges_,vertex_edges_.size()+1);
    reserve_for(lifts_,lifts_.size()+1); reserve_for(ratios_,ratios_.size()+1);
    ranks_.reserve(candidate.lengths.size()); growth_ranks_.reserve(candidate.growth.size());
    QuotientCavityEdit<QTriangle> transaction(cells_);
    const auto prepared = transaction.prepare(
        candidate.removed, candidate.proposal, static_cast<int64_t>(edges_.size() * 2 / 3),
        [&](const auto& removed, const auto& proposal) {
          return boundary_matches(removed, proposal);
        });
    if (prepared!=EditStatus::kOk) return edit_status_code(prepared);
    std::size_t window_count=windows_.size();
    for (int32_t id:transaction.created())
      window_count=std::max(window_count,static_cast<std::size_t>(id)+1);
    reserve_for(windows_,window_count);
    std::vector<QuotientWindow> window_nodes;
    window_nodes.reserve(candidate.proposal.size());
    for (const QTriangle& t:candidate.proposal) window_nodes.push_back(window(t,&candidate));
    QBank edge_nodes;
    for (const auto& [key,sides]:candidate.edges)
      edge_nodes[key].assign(sides.begin(),sides.end());
    for (auto& [key,sides]:edge_nodes) {
      (void)key;
      for (QIncidence& side:sides)
        if (side.cell<0) side.cell=transaction.created()[-side.cell-1];
    }
    std::map<QEdge,double> length_nodes(candidate.lengths.begin(),candidate.lengths.end());
    std::map<int32_t,std::set<int32_t>> cell_nodes;
    std::map<int32_t,std::set<QEdge>> vertex_edge_nodes;
    std::set<int32_t> affected;
    for (int32_t id:candidate.removed)
      for (QImage v:cells_[id].v) affected.insert(v.representative);
    for (const QTriangle& t:candidate.proposal)
      for (QImage v:t.v) affected.insert(v.representative);
    for (const auto& [key,sides]:candidate.edges) {
      (void)sides; affected.insert(key[0]);affected.insert(key[1]);
    }
    for (int32_t v:affected) {
      if (v<vertex) {
        cell_nodes[v]=vertex_cells_[v]; vertex_edge_nodes[v]=vertex_edges_[v];
      } else {
        cell_nodes[v];vertex_edge_nodes[v];
      }
    }
    for (int32_t id:candidate.removed)
      for (QImage v:cells_[id].v) cell_nodes[v.representative].erase(id);
    for (std::size_t i=0;i<candidate.proposal.size();++i)
      for (QImage v:candidate.proposal[i].v)
        cell_nodes[v.representative].insert(transaction.created()[i]);
    for (const auto& [key,sides]:candidate.edges)
      for (int32_t v:{key[0],key[1]}) {
        if (sides.empty()) vertex_edge_nodes[v].erase(key);
        else vertex_edge_nodes[v].insert(key);
      }
    const auto status=transaction.commit([&]() noexcept {
      points_.push_back(candidate.site[0]);points_.push_back(candidate.site[1]);
      vertex_cells_.emplace_back();vertex_edges_.emplace_back();
      lifts_.push_back({0,0});ratios_.push_back(1.0);
      windows_.resize(window_count);
      for (std::size_t i=0;i<window_nodes.size();++i)
        windows_[transaction.created()[i]]=window_nodes[i];
      for (const auto& [key,sides]:candidate.edges) {
        auto old=lengths_.find(key);
        if (old!=lengths_.end()) {ranks_.erase({old->second,key});lengths_.erase(old);}
        edges_.erase(key);
        if (!sides.empty()) edges_.insert(edge_nodes.extract(key));
      }
      // Reference-lift changes can also remeasure surviving outside edges.
      for (const auto& [key,value]:candidate.lengths) {
        auto old=lengths_.find(key);
        if (old!=lengths_.end()) {ranks_.erase({old->second,key});lengths_.erase(old);}
        ranks_.insert({value,key});lengths_.insert(length_nodes.extract(key));
      }
      for (const auto& [v,image]:candidate.lifts) lifts_[v]=image;
      for (const auto& [v,ratio]:candidate.growth) {
        if (v<vertex) growth_ranks_.erase({ratios_[v],{v,0,0,0}});
        ratios_[v]=ratio;growth_ranks_.insert({ratio,{v,0,0,0}});
      }
      for (auto& [v,nodes]:cell_nodes) vertex_cells_[v].swap(nodes);
      for (auto& [v,nodes]:vertex_edge_nodes) vertex_edges_[v].swap(nodes);
    });
    return edit_status_code(status);
  }
};

int32_t validate_refinement(const phx_mc_periodic_refinement_input& in) {
  if (in.point_count < 1 || in.cell_count < 1 || in.points == nullptr ||
      in.lattice == nullptr || in.inverse == nullptr || in.vertices == nullptr ||
      in.shifts == nullptr || in.control_count < 1 || in.controls == nullptr ||
      in.statistic_count < 1 || in.quantiles == nullptr || in.protected_edge_count < 0 ||
      (in.protected_edge_count > 0 && in.protected_edges == nullptr) ||
      !addressable(in.point_count, 2, sizeof(double)) ||
      !addressable(in.cell_count, 3, sizeof(int32_t)) ||
      !addressable(in.cell_count, 6, sizeof(int32_t)) ||
      !addressable(in.protected_edge_count, 4, sizeof(int32_t)) ||
      !addressable(in.control_count, 5, sizeof(double)) ||
      !addressable(in.statistic_count, 1, sizeof(double)) ||
      !std::isfinite(in.absolute_tolerance) || in.absolute_tolerance < 0.0 ||
      !std::isfinite(in.relative_tolerance) || in.relative_tolerance < 0.0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (in.point_count >= INT32_MAX || in.cell_count >= INT32_MAX) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  for (int64_t i = 0; i < in.protected_edge_count; ++i) {
    const int32_t* row = in.protected_edges + 4 * i;
    if (row[0] < 0 || row[0] >= in.point_count || row[1] < 0 ||
        row[1] >= in.point_count) return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t i = 0; i < 2 * in.point_count; ++i) {
    if (!std::isfinite(in.points[i])) return PHX_MC_NONFINITE_INPUT;
    if (!coordinate_in_domain(in.points[i])) return PHX_MC_RANGE_ERROR;
  }
  for (int j = 0; j < 4; ++j) {
    if (!std::isfinite(in.lattice[j]) || !std::isfinite(in.inverse[j])) {
      return PHX_MC_NONFINITE_INPUT;
    }
    if (!coordinate_in_domain(in.lattice[j])) return PHX_MC_RANGE_ERROR;
  }
  if ((Expansion::product(in.lattice[0], in.lattice[3]) -
       Expansion::product(in.lattice[1], in.lattice[2])).sign() == 0) {
    return PHX_MC_INVALID_INPUT;
  }
  for (int64_t i = 0; i < in.point_count; ++i) {
    for (int j = 0; j < 2; ++j) {
      const double fractional =
          in.points[2 * i] * in.inverse[j] + in.points[2 * i + 1] * in.inverse[2 + j];
      if (!(std::abs(fractional) <= 0x1p30)) return PHX_MC_RANGE_ERROR;
    }
  }
  for (int64_t i = 0; i < 3 * in.cell_count; ++i)
    if (in.vertices[i] < 0 || in.vertices[i] >= in.point_count) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t i = 0; i < 6 * in.cell_count; ++i)
    if (in.shifts[i] == INT32_MIN) return PHX_MC_INDEX_RANGE_ERROR;
  for (int64_t i = 0; i < in.control_count; ++i) {
    const double* control = in.controls + 5 * i;
    if (!std::isfinite(control[0]) || control[0] <= 0.0 ||
        (control[4] != 0.0 && control[4] != 1.0)) return PHX_MC_INVALID_ARGUMENT;
    for (int j = 1; j < 4; ++j)
      if (!std::isnan(control[j]) && (!std::isfinite(control[j]) || control[j] <= 0.0))
        return PHX_MC_INVALID_ARGUMENT;
    if ((!std::isnan(control[1]) && !std::isnan(control[2]) && control[1] > control[2]) ||
        (!std::isnan(control[3]) && control[3] < 1.0)) return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t i = 0; i < in.statistic_count; ++i)
    if (!std::isfinite(in.quantiles[i]) || in.quantiles[i] < 0.0 || in.quantiles[i] > 1.0)
      return PHX_MC_INVALID_ARGUMENT;
  return PHX_MC_OK;
}
}  // namespace

int32_t periodic_refine_2d(const phx_mc_periodic_refinement_input& input,
                           PeriodicRefinementResult& result) {
  const int32_t valid = validate_refinement(input);
  if (valid != PHX_MC_OK) return valid;
  result = PeriodicRefinementResult{};
  try {
    PeriodicQuotientOwner owner(input);
    return owner.run(result);
  } catch (const PeriodicRefinementInvalidInput&) {
    return PHX_MC_INVALID_INPUT;
  } catch (const PeriodicIndexRangeExceeded&) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
}
}  // namespace phx::mc

struct phx_mc_periodic_refinement {
  phx::mc::PeriodicRefinementResult result;
};

extern "C" {

int32_t phx_mc_periodic_refine_2d(const phx_mc_periodic_refinement_input* input,
                                 phx_mc_periodic_refinement** refinement) {
  if (refinement != nullptr) *refinement = nullptr;
  if (input == nullptr || refinement == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    phx::mc::PeriodicRefinementResult result;
    const int32_t status = phx::mc::periodic_refine_2d(*input, result);
    if (status == PHX_MC_OK || status == PHX_MC_UNRESOLVED) {
      auto handle = std::make_unique<phx_mc_periodic_refinement>();
      handle->result = std::move(result);
      *refinement = handle.release();
    }
    return status;
  });
}

int32_t phx_mc_periodic_refinement_sizes(
    const phx_mc_periodic_refinement* refinement, int64_t* sizes) {
  if (refinement == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  sizes[0] = static_cast<int64_t>(refinement->result.points.size() / 2);
  sizes[1] = static_cast<int64_t>(refinement->result.vertices.size() / 3);
  return PHX_MC_OK;
}

int32_t phx_mc_periodic_refinement_copy_points(
    const phx_mc_periodic_refinement* refinement, int64_t point_capacity, double* points) {
  if (refinement == nullptr || point_capacity < 0 ||
      !phx::mc::addressable(point_capacity, 2, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t count = static_cast<int64_t>(refinement->result.points.size() / 2);
  if (point_capacity < count) return PHX_MC_CAPACITY_EXCEEDED;
  if (count > 0 && points == nullptr) return PHX_MC_INVALID_ARGUMENT;
  std::copy(refinement->result.points.begin(), refinement->result.points.end(), points);
  return PHX_MC_OK;
}

int32_t phx_mc_periodic_refinement_copy_cells(
    const phx_mc_periodic_refinement* refinement, int64_t cell_capacity,
    int32_t* vertices, int32_t* shifts) {
  if (refinement == nullptr || cell_capacity < 0 ||
      !phx::mc::addressable(cell_capacity, 3, sizeof(int32_t)) ||
      !phx::mc::addressable(cell_capacity, 6, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t count = static_cast<int64_t>(refinement->result.vertices.size() / 3);
  if (cell_capacity < count) return PHX_MC_CAPACITY_EXCEEDED;
  if (count > 0 && (vertices == nullptr || shifts == nullptr)) return PHX_MC_INVALID_ARGUMENT;
  std::copy(refinement->result.vertices.begin(), refinement->result.vertices.end(), vertices);
  std::copy(refinement->result.shifts.begin(), refinement->result.shifts.end(), shifts);
  return PHX_MC_OK;
}

void phx_mc_periodic_refinement_free(phx_mc_periodic_refinement* refinement) {
  delete refinement;
}

}  // extern "C"
