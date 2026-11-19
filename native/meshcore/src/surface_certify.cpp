// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Surface certification primitives; contract in surface_certify.hpp.
#include "surface_certify.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "exact_scalar.hpp"
#include "geometry3.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"

namespace phx::mc::remesh {

bool legal_triangle_contact(const std::array<Point, 3>& first, const Triangle& first_ids,
                            const std::array<Point, 3>& second, const Triangle& second_ids) {
  const double* a[3] = {first[0].data(), first[1].data(), first[2].data()};
  const double* b[3] = {second[0].data(), second[1].data(), second[2].data()};
  if (first_ids[0] != first_ids[1] && first_ids[1] != first_ids[2] &&
      first_ids[2] != first_ids[0] && second_ids[0] != second_ids[1] &&
      second_ids[1] != second_ids[2] && second_ids[2] != second_ids[0]) {
    int shared = 0, others[3], other_count = 0;
    bool matching = true;
    for (int j = 0; j < 3; ++j) {
      int match = -1;
      for (int i = 0; i < 3; ++i)
        if (first_ids[i] == second_ids[j]) match = i;
      if (match < 0) others[other_count++] = j;
      else {
        ++shared;
        matching = matching && phx::mc::geometry::same_point(a[match], b[j]);
      }
    }
    // Two triangles in different planes which share two vertices meet only on
    // their common edge. The exact nonzero determinant also proves both are
    // nondegenerate; coplanar or out-of-domain cases use the full predicate.
    if (shared == 2 && matching && phx::mc::geometry::point_in_domain(a[0]) &&
        phx::mc::geometry::point_in_domain(a[1]) && phx::mc::geometry::point_in_domain(a[2]) &&
        phx::mc::geometry::point_in_domain(b[others[0]]) &&
        phx::mc::orient3d(a[0], a[1], a[2], b[others[0]]) != 0)
      return true;
    // With one shared vertex, two strictly equal plane-side signs show that
    // the rest of the second triangle cannot touch the first triangle's plane.
    if (shared == 1 && matching && phx::mc::geometry::point_in_domain(a[0]) &&
        phx::mc::geometry::point_in_domain(a[1]) && phx::mc::geometry::point_in_domain(a[2]) &&
        phx::mc::geometry::point_in_domain(b[others[0]]) &&
        phx::mc::geometry::point_in_domain(b[others[1]])) {
      const int first_side = phx::mc::orient3d(a[0], a[1], a[2], b[others[0]]);
      if (first_side != 0 &&
          first_side == phx::mc::orient3d(a[0], a[1], a[2], b[others[1]]) &&
          !phx::mc::geometry::collinear(b[0], b[1], b[2]))
        return true;
    }
  }
  phx::mc::Contact contact;
  return phx::mc::intersect_triangles(a, b, first_ids.data(), second_ids.data(), contact) ==
             PHX_MC_OK &&
         (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
          contact.kind == PHX_MC_SHARED_EDGE);
}

namespace {

// An upper bound on sqrt(numerator / denominator) for exact positive dyadics:
// the directed upper quotient, then the correctly rounded square root raised
// until its exact square covers it.
double upper_sqrt_quotient(const Dyadic& numerator, const Dyadic& denominator) {
  const double square = exact::directed_quotient_bounds(numerator, denominator).upper;
  if (!std::isfinite(square)) return std::numeric_limits<double>::infinity();
  const Dyadic bound(square);
  double root = std::sqrt(square);
  while ((Dyadic(root) * Dyadic(root) - bound).sign() < 0) {
    root = phx::mc::geometry::next_up(root);
    if (!std::isfinite(root)) return root;
  }
  return root;
}

}  // namespace

// Every point of the triangle lies within the returned radius of a corner,
// plus the caller's absolute `roundoff`. The squared sides and the squared
// cross product are exact dyadics of the binary64 corners, so the obtuse
// classification and the radius are exact before one directed rounding: a
// floating cross product loses relative accuracy of about eps / theta at an
// apex angle theta and cannot bound a needle's circumradius. A zero cross
// product (degenerate triangle) or a nonfinite corner has no free radius.
double certified_free_radius(const Point& a, const Point& b, const Point& c,
                             double roundoff) {
  for (const Point* corner : {&a, &b, &c})
    for (int axis = 0; axis < 3; ++axis)
      if (!std::isfinite((*corner)[axis])) return std::numeric_limits<double>::infinity();
  std::array<Dyadic, 3> ab, ac, bc;
  for (int axis = 0; axis < 3; ++axis) {
    ab[axis] = Dyadic(b[axis]) - Dyadic(a[axis]);
    ac[axis] = Dyadic(c[axis]) - Dyadic(a[axis]);
    bc[axis] = Dyadic(c[axis]) - Dyadic(b[axis]);
  }
  const auto squared = [](const std::array<Dyadic, 3>& v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  };
  const std::array<Dyadic, 3> normal{ab[1] * ac[2] - ab[2] * ac[1],
                                     ab[2] * ac[0] - ab[0] * ac[2],
                                     ab[0] * ac[1] - ab[1] * ac[0]};
  const Dyadic twice_area_squared = squared(normal);
  if (twice_area_squared.sign() <= 0) return std::numeric_limits<double>::infinity();
  const std::array<Dyadic, 3> squares{squared(bc), squared(ac), squared(ab)};  // opposite a, b, c
  int obtuse = -1;
  for (int k = 0; k < 3; ++k)
    if ((squares[k] - squares[(k + 1) % 3] - squares[(k + 2) % 3]).sign() > 0) obtuse = k;
  double radius;
  if (obtuse < 0) {
    // Circumradius: R^2 = |bc|^2 |ca|^2 |ab|^2 / (4 |ab x ac|^2).
    radius = upper_sqrt_quotient(squares[0] * squares[1] * squares[2],
                                 twice_area_squared * Dyadic(4.0));
  } else {
    // Obtuse at corner `obtuse`: the bisectors of its two sides meet the
    // longest side at distance side / (2 cos(adjacent angle)) from that side's
    // ends; with S the squared sides, that distance squared is
    // S_other^2 S_obtuse / (S_obtuse + S_other - S_k)^2 for the angle at k.
    radius = 0;
    for (int k = 0; k < 3; ++k) {
      if (k == obtuse) continue;
      const int other = 3 - k - obtuse;
      const Dyadic twice_cosine_product = squares[obtuse] + squares[other] - squares[k];
      if (twice_cosine_product.sign() <= 0) return std::numeric_limits<double>::infinity();
      radius = std::max(radius, upper_sqrt_quotient(
                                    squares[other] * squares[other] * squares[obtuse],
                                    twice_cosine_product * twice_cosine_product));
    }
  }
  if (!std::isfinite(radius)) return radius;
  return phx::mc::geometry::next_up(radius + roundoff);
}

TriangleKey triangle_key(const Point& a, const Point& b, const Point& c) {
  if (b < a && b < c) return {b, c, a};
  if (c < a && c < b) return {c, a, b};
  return {a, b, c};
}

bool closed_manifold(const std::vector<Triangle>& triangles, size_t vertex_count) {
  const size_t corners = 3 * triangles.size();
  // (tail, head, corner) of every directed edge, sorted.
  std::vector<std::array<int64_t, 3>> directed(corners);
  std::vector<int64_t> first(vertex_count, -1), count(vertex_count, 0);
  for (size_t c = 0; c < corners; ++c) {
    const int64_t a = triangles[c / 3][c % 3], b = triangles[c / 3][(c % 3 + 1) % 3];
    if (a < 0 || b < 0 || a == b || static_cast<size_t>(a) >= vertex_count ||
        static_cast<size_t>(b) >= vertex_count)
      return false;
    directed[c] = {a, b, static_cast<int64_t>(c)};
    if (first[static_cast<size_t>(a)] < 0) first[static_cast<size_t>(a)] = static_cast<int64_t>(c);
    ++count[static_cast<size_t>(a)];
  }
  std::sort(directed.begin(), directed.end());
  std::vector<int64_t> opposite(corners, -1);
  for (size_t i = 0; i < corners; ++i) {
    if (i + 1 < corners && directed[i][0] == directed[i + 1][0] &&
        directed[i][1] == directed[i + 1][1])
      return false;
    const std::array<int64_t, 3> reverse{directed[i][1], directed[i][0], -1};
    const auto found = std::lower_bound(directed.begin(), directed.end(), reverse);
    if (found == directed.end() || (*found)[0] != reverse[0] || (*found)[1] != reverse[1])
      return false;
    opposite[static_cast<size_t>(directed[i][2])] = (*found)[2];
  }
  // The ring of v (outgoing corner c, then the twin of the edge entering v in
  // c's triangle) returns to its first corner after all of v's corners.
  for (size_t v = 0; v < vertex_count; ++v) {
    if (first[v] < 0) continue;
    int64_t c = first[v], steps = 0;
    do {
      const int64_t entering = c - c % 3 + (c % 3 + 2) % 3;
      c = opposite[static_cast<size_t>(entering)];
      ++steps;
    } while (c != first[v] && steps <= count[v]);
    if (steps != count[v]) return false;
  }
  return true;
}

// Broad phase: a uniform grid of the mean box extent, every triangle listed in
// each cell its box meets. A pair is classified once, in the first cell (by
// coordinates) of the intersection of the two cell ranges, and only when the
// boxes meet. A triangle whose box meets more than kGridCells cells (far
// larger than the mean) is not listed; it is tested by box against every
// triangle instead, which keeps the grid at most kGridCells entries per
// triangle.
int64_t refuse_illegal_contacts(const std::vector<Point>& points,
                                const std::vector<Triangle>& triangles,
                                const std::vector<uint8_t>* changed,
                                std::vector<uint8_t>& refused) {
  const size_t m = triangles.size();
  refused.assign(m, 0);
  std::vector<std::array<Point, 3>> corners(m);
  std::vector<std::array<double, 6>> boxes(m);
  std::vector<uint8_t> finite(m, 0);
  Point origin{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()};
  Point top{-origin[0], -origin[1], -origin[2]};
  double extents = 0;
  size_t counted = 0;
  for (size_t t = 0; t < m; ++t) {
    bool ok = true;
    for (size_t k = 0; k < 3; ++k) {
      corners[t][k] = points[static_cast<size_t>(triangles[t][k])];
      for (int d = 0; d < 3; ++d) ok = ok && std::isfinite(corners[t][k][d]);
    }
    if (!ok || phx::mc::geometry::collinear(corners[t][0], corners[t][1], corners[t][2])) {
      refused[t] = 1;
      continue;
    }
    finite[t] = 1;
    double extent = 0;
    for (int d = 0; d < 3; ++d) {
      const double lo = std::min({corners[t][0][d], corners[t][1][d], corners[t][2][d]});
      const double hi = std::max({corners[t][0][d], corners[t][1][d], corners[t][2][d]});
      boxes[t][static_cast<size_t>(d)] = lo;
      boxes[t][static_cast<size_t>(d) + 3] = hi;
      origin[d] = std::min(origin[d], lo);
      top[d] = std::max(top[d], hi);
      extent = std::max(extent, hi - lo);
    }
    extents += extent;
    ++counted;
  }
  if (counted == 0) return static_cast<int64_t>(m);
  // Cell coordinates stay within 2^20 per axis.
  double cell = extents / static_cast<double>(counted);
  for (int d = 0; d < 3; ++d) cell = std::max(cell, (top[d] - origin[d]) * 0x1p-20);
  if (!(cell > 0)) cell = 1;
  const auto coordinate = [&](double value, int d) {
    return static_cast<int64_t>(std::floor((value - origin[d]) / cell));
  };
  constexpr int64_t kGridCells = 64;
  std::vector<std::array<int64_t, 6>> ranges(m);
  std::vector<std::pair<uint64_t, int32_t>> entries;
  std::vector<size_t> large;
  entries.reserve(4 * counted);
  const auto key = [](int64_t x, int64_t y, int64_t z) {
    return (static_cast<uint64_t>(x) << 42) | (static_cast<uint64_t>(y) << 21) |
           static_cast<uint64_t>(z);
  };
  for (size_t t = 0; t < m; ++t) {
    if (!finite[t]) continue;
    int64_t cells = 1;
    for (int d = 0; d < 3; ++d) {
      ranges[t][static_cast<size_t>(d)] = coordinate(boxes[t][static_cast<size_t>(d)], d);
      ranges[t][static_cast<size_t>(d) + 3] = coordinate(boxes[t][static_cast<size_t>(d) + 3], d);
      cells *= ranges[t][static_cast<size_t>(d) + 3] - ranges[t][static_cast<size_t>(d)] + 1;
    }
    if (cells > kGridCells) {
      large.push_back(t);
      continue;
    }
    for (int64_t x = ranges[t][0]; x <= ranges[t][3]; ++x)
      for (int64_t y = ranges[t][1]; y <= ranges[t][4]; ++y)
        for (int64_t z = ranges[t][2]; z <= ranges[t][5]; ++z)
          entries.push_back({key(x, y, z), static_cast<int32_t>(t)});
  }
  const auto classify = [&](size_t s, size_t u) {
    if (changed != nullptr && !(*changed)[s] && !(*changed)[u]) return;
    for (size_t d = 0; d < 3; ++d)
      if (!(boxes[s][d] <= boxes[u][d + 3] && boxes[u][d] <= boxes[s][d + 3])) return;
    if (!legal_triangle_contact(corners[s], triangles[s], corners[u], triangles[u]))
      refused[s] = refused[u] = 1;
  };
  // Large triangles against every other triangle (each large pair once).
  std::vector<uint8_t> is_large(m, 0);
  for (const size_t s : large) is_large[s] = 1;
  for (const size_t s : large)
    for (size_t u = 0; u < m; ++u)
      if (finite[u] && u != s && (!is_large[u] || u > s)) classify(s, u);
  std::sort(entries.begin(), entries.end());
  for (size_t begin = 0; begin < entries.size();) {
    size_t end = begin + 1;
    while (end < entries.size() && entries[end].first == entries[begin].first) ++end;
    const uint64_t here = entries[begin].first;
    for (size_t i = begin; i < end; ++i) {
      const auto s = static_cast<size_t>(entries[i].second);
      for (size_t j = i + 1; j < end; ++j) {
        const auto u = static_cast<size_t>(entries[j].second);
        if (key(std::max(ranges[s][0], ranges[u][0]), std::max(ranges[s][1], ranges[u][1]),
                std::max(ranges[s][2], ranges[u][2])) != here)
          continue;
        classify(s, u);
      }
    }
    begin = end;
  }
  return std::count(refused.begin(), refused.end(), uint8_t{1});
}

}  // namespace phx::mc::remesh
