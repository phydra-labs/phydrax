// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Surface certification primitives: exact legal contact, the certified free
// radius, the closed-manifold check and the illegal-contact refusal.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "check.hpp"
#include "dyadic.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"
#include "surface_certify.hpp"

namespace {
using phx::mc::remesh::Point;
using phx::mc::remesh::Triangle;

void exact_contact_parity() {
  using phx::mc::remesh::legal_triangle_contact;
  const std::array<Point, 3> first{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
  const Triangle first_ids{0, 1, 2};
  std::mt19937_64 generator(461);
  std::uniform_real_distribution<double> coordinate(-2.0, 2.0);
  for (int sample = 0; sample < 2000; ++sample) {
    std::array<Point, 3> second{{first[1], first[0],
                                 {coordinate(generator), coordinate(generator),
                                  sample % 4 == 0 ? 0.0 : coordinate(generator)}}};
    Triangle second_ids{1, 0, 3};
    if (sample % 17 == 0) second[0][0] += 0.25;  // inconsistent shared ID
    if (sample % 19 == 0) second_ids[2] = second_ids[1];  // repeated ID
    const double* a[3] = {first[0].data(), first[1].data(), first[2].data()};
    const double* b[3] = {second[0].data(), second[1].data(), second[2].data()};
    phx::mc::Contact contact;
    const auto status = phx::mc::intersect_triangles(a, b, first_ids.data(), second_ids.data(), contact);
    const bool expected = status == PHX_MC_OK &&
        (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
         contact.kind == PHX_MC_SHARED_EDGE);
    PHX_CHECK(legal_triangle_contact(first, first_ids, second, second_ids) == expected);
  }
  for (int sample = 0; sample < 2000; ++sample) {
    std::array<Point, 3> second{{first[0],
                                 {coordinate(generator), coordinate(generator), coordinate(generator)},
                                 {coordinate(generator), coordinate(generator), coordinate(generator)}}};
    Triangle second_ids{0, 3, 4};
    if (sample % 5 == 0) second[2] = second[1];  // degenerate face
    if (sample % 7 == 0) second[1][2] = second[2][2] = 0;  // coplanar
    if (sample % 17 == 0) second[0][0] += 0.25;  // inconsistent shared ID
    const double* a[3] = {first[0].data(), first[1].data(), first[2].data()};
    const double* b[3] = {second[0].data(), second[1].data(), second[2].data()};
    phx::mc::Contact contact;
    const auto status = phx::mc::intersect_triangles(a, b, first_ids.data(), second_ids.data(), contact);
    const bool expected = status == PHX_MC_OK &&
        (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
         contact.kind == PHX_MC_SHARED_EDGE);
    PHX_CHECK(legal_triangle_contact(first, first_ids, second, second_ids) == expected);
  }
}

// Exact squared free radius of binary64 corners (independent reference):
// non-obtuse, the circumradius |bc|^2 |ca|^2 |ab|^2 / (4 |ab x ac|^2); obtuse at
// corner o, the maximum over the other corners k of S_other^2 S_o /
// (S_o + S_other - S_k)^2 with S the squared sides. Returns numerator and
// denominator.
std::pair<phx::mc::Dyadic, phx::mc::Dyadic> exact_free_radius_squared(const Point& a,
                                                                     const Point& b,
                                                                     const Point& c) {
  using phx::mc::Dyadic;
  std::array<Dyadic, 3> ab, ac, bc;
  for (int axis = 0; axis < 3; ++axis) {
    ab[axis] = Dyadic(b[axis]) - Dyadic(a[axis]);
    ac[axis] = Dyadic(c[axis]) - Dyadic(a[axis]);
    bc[axis] = Dyadic(c[axis]) - Dyadic(b[axis]);
  }
  const auto dot = [](const std::array<Dyadic, 3>& u, const std::array<Dyadic, 3>& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const std::array<Dyadic, 3> n{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                                ab[0] * ac[1] - ab[1] * ac[0]};
  const std::array<Dyadic, 3> s{dot(bc, bc), dot(ac, ac), dot(ab, ab)};
  int obtuse = -1;
  for (int k = 0; k < 3; ++k)
    if ((s[k] - s[(k + 1) % 3] - s[(k + 2) % 3]).sign() > 0) obtuse = k;
  if (obtuse < 0) return {s[0] * s[1] * s[2], Dyadic(4.0) * dot(n, n)};
  std::pair<Dyadic, Dyadic> best{Dyadic(), Dyadic(1.0)};
  for (int k = 0; k < 3; ++k) {
    if (k == obtuse) continue;
    const int other = 3 - k - obtuse;
    const Dyadic d = s[obtuse] + s[other] - s[k];
    const Dyadic numerator = s[other] * s[other] * s[obtuse], denominator = d * d;
    if ((numerator * best.second - best.first * denominator).sign() > 0)
      best = {numerator, denominator};
  }
  return best;
}

bool covers(double radius, const std::pair<phx::mc::Dyadic, phx::mc::Dyadic>& exact) {
  const phx::mc::Dyadic r(radius);
  return (r * r * exact.second - exact.first).sign() >= 0;
}

double envelope_roundoff(const Point& a, const Point& b, const Point& c) {
  double scale = 0;
  for (const Point* p : {&a, &b, &c})
    for (double v : *p) scale = std::max(scale, std::abs(v));
  return std::nextafter(std::max(256 * std::numeric_limits<double>::epsilon() * scale,
                                 256 * std::numeric_limits<double>::denorm_min()),
                        std::numeric_limits<double>::infinity());
}

// The free radius must bound the exact one for binary64 needles, whose
// floating cross product loses relative accuracy of about eps / theta. The
// first rows are the Windows audit's escapes (C8); the random needles and caps
// repeat its probe.
void certified_free_radius_is_exact() {
  using phx::mc::remesh::certified_free_radius;
  const std::array<std::array<Point, 3>, 5> escapes{{
      {{{-0x1.7156d5fa24df0p-1, 0x1.baf3c9d8ec664p-2, -0x1.856534f685164p-2},
        {-0x1.0af5efa4a7ba6p-1, 0x1.2176fc2305e7ap-1, -0x1.50629dffa805ap-1},
        {-0x1.0af600e997fdcp-1, 0x1.2177007bfa29ap-1, -0x1.5062a8640476ep-1}}},
      {{{0x1.8b3ce171aaa0ap-1, -0x1.10a8fb78985c0p-1, 0x1.522e4628d3c58p-1},
        {0x1.f830fb2ad305fp-1, -0x1.41fda84f50363p-1, 0x1.ab9e617334331p-2},
        {0x1.f830fb37251a1p-1, -0x1.41fda847cd213p-1, 0x1.ab9e6182d4cf9p-2}}},
      {{{0x1.83ec42ffac8c0p-5, -0x1.e174b90083ad4p-1, -0x1.f4b2c8b7ca1fcp-1},
        {-0x1.b63b05759e842p-2, -0x1.76298e9efd6cbp+0, -0x1.391e328c39dbbp+0},
        {-0x1.b638625b4e03ap-2, -0x1.762a5257200d5p+0, -0x1.391dd9a81f49fp+0}}},
      {{{0x1.8ed948f0188b0p-1, -0x1.ad390989186e4p-1, -0x1.f4c6549956f58p-3},
        {0x1.83b919c2a19adp-1, -0x1.bcbe2b1fdefa4p-1, -0x1.d01e2bdf83c57p-3},
        {0x1.83b90c5e43ec1p-1, -0x1.bcbe226dbefeep-1, -0x1.d01e320037e35p-3}}},
      {{{0x1.c0cdb5943415cp-2, -0x1.9646364cc6752p-1, 0x1.4ddf18f4f3a5ep-1},
        {0x1.6f9bb4f0f5d75p-2, -0x1.15443b8523851p-1, 0x1.6d5f7b85a308fp-1},
        {0x1.6f9bb4fba105bp-2, -0x1.15443842e5a03p-1, 0x1.6d5f6e340a1afp-1}}},
  }};
  for (const auto& [a, b, c] : escapes) {
    const double roundoff = envelope_roundoff(a, b, c);
    const double radius = certified_free_radius(a, b, c, roundoff);
    const auto exact = exact_free_radius_squared(a, b, c);
    PHX_CHECK(std::isfinite(radius));
    PHX_CHECK(covers(radius, exact));
    // Only the caller's absolute roundoff and directed rounding separate the
    // bound from the exact radius.
    PHX_CHECK(!covers(radius - 2 * roundoff, exact));
  }
  // An obtuse triangle takes the bisector bound (exactly 0.265625 here), not
  // its circumradius (1.0625); a negligible roundoff leaves directed rounding.
  {
    const Point a{0, 0, 0}, b{1, 0, 0}, c{0.5, 0.125, 0};
    const double radius = certified_free_radius(a, b, c, 0x1p-60);
    const auto exact = exact_free_radius_squared(a, b, c);
    PHX_CHECK(covers(radius, exact));
    double below = radius;
    for (int step = 0; step < 4; ++step)
      below = std::nextafter(below, -std::numeric_limits<double>::infinity());
    PHX_CHECK(!covers(below, exact));
    PHX_CHECK(radius < 0.5);
  }
  // An exactly zero cross product or a nonfinite corner has no free radius.
  PHX_CHECK(std::isinf(certified_free_radius({0, 0, 0}, {0.25, 0.5, 1}, {0.5, 1, 2}, 0x1p-60)));
  PHX_CHECK(std::isinf(certified_free_radius({0, 0, 0}, {1, 0, 0},
                                             {0, std::numeric_limits<double>::quiet_NaN(), 0},
                                             0x1p-60)));
  std::mt19937_64 generator(20261106);
  std::uniform_real_distribution<double> unit(-1.0, 1.0), exponent(0.0, 1.0);
  std::normal_distribution<double> normal;
  int escaped = 0, finite = 0;
  for (int sample = 0; sample < 20000; ++sample) {
    const Point apex{unit(generator), unit(generator), unit(generator)};
    Point direction{normal(generator), normal(generator), normal(generator)};
    const double direction_length = std::sqrt(direction[0] * direction[0] +
                                              direction[1] * direction[1] +
                                              direction[2] * direction[2]);
    for (double& v : direction) v /= direction_length;
    Point side{normal(generator), normal(generator), normal(generator)};
    const double along = side[0] * direction[0] + side[1] * direction[1] + side[2] * direction[2];
    for (int axis = 0; axis < 3; ++axis) side[axis] -= along * direction[axis];
    const double side_length = std::sqrt(side[0] * side[0] + side[1] * side[1] + side[2] * side[2]);
    for (double& v : side) v /= side_length;
    const double length = std::pow(10.0, -3.0 * exponent(generator));
    const double theta = std::pow(10.0, -9.0 + 8.0 * exponent(generator));
    Point b, c;
    for (int axis = 0; axis < 3; ++axis) {
      if (sample % 2 == 0) {
        b[axis] = apex[axis] + length * direction[axis] + 0.5 * length * theta * side[axis];
        c[axis] = apex[axis] + length * direction[axis] - 0.5 * length * theta * side[axis];
      } else {
        b[axis] = apex[axis] + length * direction[axis];
        c[axis] = apex[axis] + 0.5 * length * direction[axis] + length * theta * side[axis];
      }
    }
    const double radius = certified_free_radius(apex, b, c, envelope_roundoff(apex, b, c));
    if (!std::isfinite(radius)) continue;
    ++finite;
    if (!covers(radius, exact_free_radius_squared(apex, b, c))) ++escaped;
  }
  PHX_CHECK(finite == 20000);
  PHX_CHECK(escaped == 0);
}

// Boundary of the tetrahedron with corners at `origin` + `size` times the
// unit axes, outward, appended to (points, triangles).
void append_tetrahedron(const Point& origin, std::vector<Point>& points,
                        std::vector<Triangle>& triangles, double size = 1) {
  const auto base = static_cast<int64_t>(points.size());
  points.push_back(origin);
  for (int d = 0; d < 3; ++d) {
    Point corner = origin;
    corner[static_cast<size_t>(d)] += size;
    points.push_back(corner);
  }
  for (const Triangle& face : {Triangle{0, 2, 1}, Triangle{0, 1, 3}, Triangle{0, 3, 2},
                               Triangle{1, 2, 3}})
    triangles.push_back({base + face[0], base + face[1], base + face[2]});
}

// Closed oriented 2-manifolds pass, and every returned twin reverses its
// corner's edge; a border, a repeated directed edge, a reversed triangle and
// two closed surfaces sharing one vertex (two fans at it) do not.
void closed_manifold_cases() {
  using phx::mc::remesh::closed_manifold;
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  append_tetrahedron({0, 0, 0}, points, triangles);
  std::vector<int64_t> twins;
  PHX_CHECK(closed_manifold(triangles, points.size(), &twins));
  PHX_CHECK(twins.size() == 3 * triangles.size());
  for (size_t c = 0; c < twins.size(); ++c) {
    const auto g = static_cast<size_t>(twins[c]);
    PHX_CHECK(triangles[c / 3][c % 3] == triangles[g / 3][(g % 3 + 1) % 3]);
    PHX_CHECK(triangles[c / 3][(c % 3 + 1) % 3] == triangles[g / 3][g % 3]);
  }
  std::vector<Triangle> open = triangles;
  open.pop_back();
  PHX_CHECK(!closed_manifold(open, points.size()));
  std::vector<Triangle> repeated = triangles;
  repeated.push_back(triangles[0]);
  PHX_CHECK(!closed_manifold(repeated, points.size()));
  std::vector<Triangle> flipped = triangles;
  std::swap(flipped[0][1], flipped[0][2]);
  PHX_CHECK(!closed_manifold(flipped, points.size()));
  // A second tetrahedron, reflected through vertex 0 of the first and sharing
  // it.
  std::vector<Point> pinched = points;
  for (const Point& corner : {Point{-1, 0, 0}, Point{0, -1, 0}, Point{0, 0, -1}})
    pinched.push_back(corner);
  std::vector<Triangle> pinched_triangles = triangles;
  for (const Triangle& face : {Triangle{0, 4, 5}, Triangle{0, 6, 4}, Triangle{0, 5, 6},
                               Triangle{4, 6, 5}})
    pinched_triangles.push_back(face);
  PHX_CHECK(!closed_manifold(pinched_triangles, pinched.size()));
  PHX_CHECK(!closed_manifold(triangles, 3));  // an index outside the vertices
}

// Two tetrahedra overlapping in space are flagged (only triangles of a
// crossing pair); with only unchanged triangles nothing is classified; apart,
// nothing is flagged; a degenerate triangle is flagged on its own.
void illegal_contact_cases() {
  using phx::mc::remesh::refuse_illegal_contacts;
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  append_tetrahedron({0, 0, 0}, points, triangles);
  append_tetrahedron({0.25, 0.25, 0.25}, points, triangles);
  std::vector<uint8_t> refused;
  const int64_t crossing = refuse_illegal_contacts(points, triangles, nullptr, refused);
  PHX_CHECK(crossing > 0 && crossing <= 8);
  PHX_CHECK(refused.size() == triangles.size());
  const std::vector<uint8_t> unchanged(triangles.size(), 0);
  PHX_CHECK(refuse_illegal_contacts(points, triangles, &unchanged, refused) == 0);
  std::vector<uint8_t> second(triangles.size(), 0);
  for (size_t t = 4; t < triangles.size(); ++t) second[t] = 1;
  PHX_CHECK(refuse_illegal_contacts(points, triangles, &second, refused) == crossing);
  std::vector<Point> apart_points;
  std::vector<Triangle> apart;
  append_tetrahedron({0, 0, 0}, apart_points, apart);
  append_tetrahedron({3, 0, 0}, apart_points, apart);
  PHX_CHECK(refuse_illegal_contacts(apart_points, apart, nullptr, refused) == 0);
  // Vertices 0, 2, 1 of the first tetrahedron made collinear.
  apart_points[1] = Point{0.25, 0.25, 0};
  apart_points[2] = Point{0.5, 0.5, 0};
  PHX_CHECK(refuse_illegal_contacts(apart_points, apart, nullptr, refused) > 0);
  PHX_CHECK(refused[0] == 1);
  // A tetrahedron far larger than the rest (it meets more grid cells than
  // any listed triangle may) crossing one small tetrahedron among many apart.
  std::vector<Point> graded_points;
  std::vector<Triangle> graded;
  for (int k = 0; k < 40; ++k) append_tetrahedron({3.0 * k, 0, 0}, graded_points, graded);
  PHX_CHECK(refuse_illegal_contacts(graded_points, graded, nullptr, refused) == 0);
  append_tetrahedron({30.25, 0.25, 0.25}, graded_points, graded, 60);
  const int64_t graded_crossing = refuse_illegal_contacts(graded_points, graded, nullptr, refused);
  PHX_CHECK(graded_crossing > 0);
  PHX_CHECK(refused[graded.size() - 1] + refused[graded.size() - 2] + refused[graded.size() - 3] +
                refused[graded.size() - 4] >
            0);
  PHX_CHECK(refused[0] == 0);
}

}  // namespace

int main() {
  exact_contact_parity();
  certified_free_radius_is_exact();
  closed_manifold_cases();
  illegal_contact_cases();
  return phx::mc::test::finish("test_surface_certify");
}
