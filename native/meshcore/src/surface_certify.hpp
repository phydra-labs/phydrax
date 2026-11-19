// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Certification of a constructed closed triangle surface.
//
// Construction (surface_adapt.hpp) proves nothing. A surface is certified
// when every triangle has
//  * fidelity: its source certifier (TriangleCertifier) states a finite
//    bound for the closed triangle, or it is one of the carrier's own
//    triangles (the same corner coordinates in the same cyclic order),
//    accepted at the carrier's declared distance;
//  * embedding: every other triangle whose box meets its box has only legal
//    contact with it (disjoint, a shared vertex, or a shared edge), by exact
//    classification.
// certify_surface (surface_adapt.hpp) runs both and repairs the refused
// triangles locally.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace phx::mc::remesh {

using Point = std::array<double, 3>;
using Triangle = std::array<int64_t, 3>;

// Certified per-vertex enclosure of a source field; its meaning is owned by
// the certifier (offset band: bounds of the source distance; zero set: a
// point enclosure of f).
struct VertexEvidence {
  double lower = 0;
  double upper = 0;
};

// Whether two indexed triangles have only legal disjoint/shared-feature contact.
bool legal_triangle_contact(const std::array<Point, 3>& first, const Triangle& first_ids,
                            const std::array<Point, 3>& second, const Triangle& second_ids);

// Upper bound on the distance from any point of the closed triangle to its
// nearest corner (circumradius for a non-obtuse triangle, else the larger
// perpendicular-bisector crossing of the longest side), rounded outward and
// widened by `roundoff`; +inf for a degenerate triangle.
double certified_free_radius(const Point& a, const Point& b, const Point& c, double roundoff);

// A source's certificate for one closed triangle.
class TriangleCertifier {
 public:
  virtual ~TriangleCertifier() = default;
  // The certified bound the source states for the triangle (a distance from
  // the source, or 0 for a band membership), +inf when it cannot certify it.
  virtual double bound(const Point& a, const Point& b, const Point& c) const = 0;
};

// A triangle's corners in its own cyclic order, rotated to start at the
// lexicographically least corner: equal for the same oriented triangle
// whatever corner it lists first.
using TriangleKey = std::array<Point, 3>;
TriangleKey triangle_key(const Point& a, const Point& b, const Point& c);

// The triangles of the surface that was remeshed, accepted without a proof at
// `distance` (its producer's evidence).
struct CarrierEvidence {
  std::vector<TriangleKey> keys;  // sorted
  double distance = 0;
};

// Whether the triangles form a closed, consistently oriented 2-manifold over
// vertices [0, vertex_count): every directed edge held once and reversed
// once, and the triangles around every used vertex one fan.
bool closed_manifold(const std::vector<Triangle>& triangles, size_t vertex_count);

// Resets `refused` to one flag per triangle and sets it for every triangle
// that is degenerate (collinear or nonfinite corners) or in an illegal
// contact with another triangle whose box meets its own; with `changed`
// given, only pairs with at least one changed triangle are classified.
// Returns the number of flagged triangles.
int64_t refuse_illegal_contacts(const std::vector<Point>& points,
                                const std::vector<Triangle>& triangles,
                                const std::vector<uint8_t>* changed,
                                std::vector<uint8_t>& refused);

}  // namespace phx::mc::remesh
