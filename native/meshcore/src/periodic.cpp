//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Periodic Delaunay triangulation on bounded certified image neighborhoods;
// see periodic.hpp for the construction and its certificate.
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
#include "expansion.hpp"
#include "filtered.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

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
// Largest admitted margin; lattice shifts stay far inside int32.
constexpr double kMaximumMargin = 0x1p20;

struct CellBudgetExceeded {};
struct DuplicatePoint {
  int32_t representative;
};

template <class T>
T det3(const T* a, const T* b, const T* c) {
  return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

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
  int64_t exact_evaluations() const { return exact_; }
  int64_t perturbed_decisions() const { return perturbed_; }

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
  int64_t exact_ = 0;
  int64_t perturbed_ = 0;

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
      const int32_t delta = va.shift[static_cast<std::size_t>(j)] - vb.shift[static_cast<std::size_t>(j)];
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
      const int32_t delta = va.shift[static_cast<std::size_t>(j)] - vb.shift[static_cast<std::size_t>(j)];
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
    ++exact_;
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
    ++exact_;
    return insphere<Expansion>(
               v, q, [this](int32_t a, int32_t b, int c) { return exact_difference(a, b, c); })
        .sign();
  }

  // Exact lexicographic order of positions: translation invariant.
  int lexicographic(int32_t a, int32_t b) {
    for (int c = 0; c < D; ++c) {
      int sign = approx_difference(a, b, c).certified_sign();
      if (sign == 2) {
        ++exact_;
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
    ++perturbed_;
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
        const int32_t relative =
            item.shift[static_cast<std::size_t>(j)] - anchor.shift[static_cast<std::size_t>(j)];
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
    if (slots() >= input_.max_cells) {
      throw CellBudgetExceeded{};
    }
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

// Number of images of every representative inside the slack box of margin m,
// or -1 once it exceeds `limit`.
template <int D>
int64_t count_images(const PeriodicDelaunayInput& input, double margin, int64_t limit,
                     std::vector<typename PeriodicEngine<D>::Vertex>* images) {
  const double lower = -margin - kImageSlack;
  const double upper = 1.0 + margin + kImageSlack;
  int64_t total = 0;
  for (int64_t point = 0; point < input.point_count; ++point) {
    std::array<int32_t, D> first;
    std::array<int32_t, D> last;
    int64_t count = 1;
    for (int j = 0; j < D; ++j) {
      const double fractional = input.fractional[point * D + j];
      first[static_cast<std::size_t>(j)] = static_cast<int32_t>(std::ceil(lower - fractional));
      last[static_cast<std::size_t>(j)] = static_cast<int32_t>(std::floor(upper - fractional));
      count *= static_cast<int64_t>(last[static_cast<std::size_t>(j)] -
                                    first[static_cast<std::size_t>(j)] + 1);
    }
    total += count;
    if (total > limit) {
      return -1;
    }
    if (images == nullptr) {
      continue;
    }
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
      if (axis == D) {
        break;
      }
    }
  }
  return total;
}

template <int D>
int32_t run(const PeriodicDelaunayInput& input, PeriodicDelaunayResult& result) {
  double* evidence = result.evidence;
  evidence[kPeriodicDuplicatePoint] = -1.0;
  double margin = input.initial_margin;
  for (int round = 1;; ++round) {
    evidence[kPeriodicRounds] = round;
    evidence[kPeriodicMargin] = margin;
    const int64_t count =
        margin > kMaximumMargin ? -1 : count_images<D>(input, margin, input.max_images, nullptr);
    if (count < 0) {
      evidence[kPeriodicExhaustedLimit] = 1.0;
      evidence[kPeriodicImages] = static_cast<double>(input.max_images) + 1.0;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    evidence[kPeriodicImages] = static_cast<double>(count);
    std::vector<typename PeriodicEngine<D>::Vertex> images;
    images.reserve(static_cast<std::size_t>(count));
    count_images<D>(input, margin, input.max_images, &images);
    PeriodicEngine<D> engine(input, std::move(images));
    try {
      engine.build();
    } catch (const CellBudgetExceeded&) {
      evidence[kPeriodicFiniteCells] = static_cast<double>(engine.slots());
      evidence[kPeriodicExactEvaluations] += static_cast<double>(engine.exact_evaluations());
      evidence[kPeriodicPerturbedDecisions] += static_cast<double>(engine.perturbed_decisions());
      evidence[kPeriodicExhaustedLimit] = 2.0;
      return PHX_MC_CAPACITY_EXCEEDED;
    } catch (const DuplicatePoint& duplicate) {
      evidence[kPeriodicDuplicatePoint] = duplicate.representative;
      return PHX_MC_INVALID_INPUT;
    }
    evidence[kPeriodicFiniteCells] = static_cast<double>(engine.slots());
    evidence[kPeriodicExactEvaluations] += static_cast<double>(engine.exact_evaluations());
    evidence[kPeriodicPerturbedDecisions] += static_cast<double>(engine.perturbed_decisions());
    std::vector<typename PeriodicEngine<D>::Candidate> extracted;
    double required = 0.0;
    const int64_t uncertified = engine.extract(margin, extracted, required);
    evidence[kPeriodicUncertified] = static_cast<double>(uncertified);
    evidence[kPeriodicRequiredMargin] = required;
    if (uncertified == 0 && !extracted.empty()) {
      result.dimension = D;
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

}  // namespace phx::mc

struct phx_mc_periodic_triangulation {
  phx::mc::PeriodicDelaunayResult result;
};

namespace {

int32_t validate_periodic_input(const phx::mc::PeriodicDelaunayInput& input) {
  const int d = input.dimension;
  if ((d != 2 && d != 3) || input.point_count < 1 || input.point_count > 2147483646 ||
      input.points == nullptr || input.fractional == nullptr || input.lattice == nullptr ||
      input.inverse == nullptr || input.max_images < 1 || input.max_cells < 1 ||
      !phx::mc::addressable(input.point_count, d, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
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
    if (!phx::mc::coordinate_in_domain(input.points[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  for (int index = 0; index < d * d; ++index) {
    if (!phx::mc::coordinate_in_domain(input.lattice[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  for (int64_t index = 0; index < items; ++index) {
    if (!(std::fabs(input.fractional[index]) <= 0x1p30)) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_periodic_delaunay(int32_t dimension, int64_t point_count, const double* points,
                                 const double* fractional, const double* lattice,
                                 const double* inverse, double initial_margin,
                                 int64_t max_images, int64_t max_cells, double* evidence,
                                 phx_mc_periodic_triangulation** triangulation) {
  return phx::mc::guarded([&]() -> int32_t {
    if (evidence == nullptr || triangulation == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *triangulation = nullptr;
    for (int slot = 0; slot < PHX_MC_PERIODIC_EVIDENCE; ++slot) {
      evidence[slot] = 0.0;
    }
    phx::mc::PeriodicDelaunayInput input;
    input.dimension = dimension;
    input.point_count = point_count;
    input.points = points;
    input.fractional = fractional;
    input.lattice = lattice;
    input.inverse = inverse;
    input.initial_margin = initial_margin;
    input.max_images = max_images;
    input.max_cells = max_cells;
    const int32_t valid = validate_periodic_input(input);
    if (valid != PHX_MC_OK) {
      return valid;
    }
    auto handle = std::make_unique<phx_mc_periodic_triangulation>();
    const int32_t status = phx::mc::periodic_delaunay(input, handle->result);
    for (int slot = 0; slot < PHX_MC_PERIODIC_EVIDENCE; ++slot) {
      evidence[slot] = handle->result.evidence[slot];
    }
    if (status == PHX_MC_OK) {
      *triangulation = handle.release();
    }
    return status;
  });
}

int64_t phx_mc_periodic_cell_count(const phx_mc_periodic_triangulation* triangulation) {
  if (triangulation == nullptr || triangulation->result.dimension == 0) {
    return 0;
  }
  return static_cast<int64_t>(triangulation->result.vertices.size()) /
         (triangulation->result.dimension + 1);
}

void phx_mc_periodic_copy_cells(const phx_mc_periodic_triangulation* triangulation,
                                int32_t* vertices, int32_t* shifts) {
  if (triangulation == nullptr) {
    return;
  }
  std::copy(triangulation->result.vertices.begin(), triangulation->result.vertices.end(), vertices);
  std::copy(triangulation->result.shifts.begin(), triangulation->result.shifts.end(), shifts);
}

void phx_mc_periodic_free(phx_mc_periodic_triangulation* triangulation) {
  delete triangulation;
}

}  // extern "C"
