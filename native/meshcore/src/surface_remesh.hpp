// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Source-agnostic certified remeshing of a closed, consistently oriented,
// embedded triangle surface.
//
// The remesher owns the topology (edge split, edge flip, edge collapse, vertex
// relocation), the quantized quality score and its worst-first acceptance, and
// the manifold and self-intersection checks. What the surface REPRESENTS is
// declared by a TriangleAdmission: every created triangle must be admitted by
// it, so the remeshed surface keeps exactly the certificate the admission
// states for its input.
//
// Admission interface (one implementation per declared AdmissionKind):
//  * evaluate(position): certified per-vertex evidence for a vertex placed at
//    `position`; the remesher caches it per vertex and restores it when an
//    edit is refused, so admit() never re-evaluates unchanged vertices;
//  * admits(a, b, c, evidence): whether the CLOSED triangle (a, b, c) with the
//    cached corner evidence satisfies the declared predicate. It must be
//    sound (never admit a triangle violating the predicate); incompleteness
//    only leaves edits refused;
//  * project(position): a candidate placement on the represented surface for
//    split points and relocations (only a proposal; admission decides);
//  * normal_cone(a, b, c, evidence): a cone (unit axis, half-angle in
//    radians, pi when nothing is known) holding every unit normal of the
//    represented source over the triangle's band piece, oriented like the
//    surface and widened by how far an admitted triangle's own normal may
//    legitimately depart from them. It only grades quality (never admission),
//    so it may be generous; a tighter cone exposes more folds.
// Declared kinds:
//  * kOffsetBand (surface envelope): every point of the triangle lies in the
//    certified band lower <= d_source <= upper. d_source is 1-Lipschitz, so on
//    a triangle with vertex enclosures [L_i, U_i] and every point within R of
//    a corner (R = certified_free_radius), d_source lies in
//    [min L_i - R, max U_i + R]; evidence = (L_i, U_i). Its normal cone holds
//    the directions from every source point that can be nearest to a point of
//    the triangle (the gradients of d_source there).
//  * kZeroSetBand (implicit surface): the triangle's certified distance to the
//    zero set of f is at most the declared surface tolerance, from an interval
//    enclosure of |f| and a lower bound on |grad f| over the triangle's
//    neighborhood (native interval program), plus a certified transversal
//    normal; evidence = point enclosure of f. Its normal cone holds the
//    gradient enclosure over the triangle's box, widened by the tilt a
//    triangle within the tolerance of the zero set may have. See
//    zero_set_admission.hpp.
//
// Every admitted edit also keeps the surface an embedded, consistently
// oriented closed 2-manifold: link condition for collapses, no duplicate edge
// or face, orientation agreeing with the replaced patch, binary64 positions in
// the representable domain, and exact triangle-triangle contact
// classification against every nearby triangle (only shared vertices and
// shared edges are legal contacts).
//
// Passes and termination (no iteration caps):
//  * quality: worst-first over triangles below the full score; an edit is
//    admitted only if every created score strictly exceeds the replaced
//    patch's worst score, and every surviving edge neighbor of the created
//    patch (whose fold factor the edit changes) either keeps its score or
//    also exceeds it, so the multiset of quantized scores decreases in the
//    well-founded multiset order;
//  * regularization: collapse of an edge shorter than 4/5 of the target
//    length, or a flip that strictly lowers the squared valence deviation
//    from six over its four vertices, admitted only if no created or changed
//    neighbor score falls below the replaced patch's worst. Collapses lower
//    the vertex count and flips the integer valence energy, so (vertex
//    count, valence energy) decreases lexicographically and the sweeps
//    terminate;
//  * the passes run quality, regularization, quality.
// The score of a triangle is floor(2^20 * min(1, angle_min / minimum_angle) *
// min(1, (4/3) h / edge_max) * fold). Fold is the minimum over the triangle's
// edges of min(1, (pi - turn) / (pi - allowed)), turn being the angle between
// the unit normals of the triangle and its edge neighbor and allowed the
// largest angle between normals of the two triangles' source cones (the angle
// between their axes plus both half-angles, at most pi). A turn the source
// admits scores full: genuine creases (an implicit CSG edge, an offset's
// concave ridge) are kept. A turn beyond it is a fold the source does not
// have; without this factor a folded cluster of well-shaped triangles scores
// full, and two boundary triangles meeting at turn t bound every tetrahedron
// on their edge to a dihedral angle of at most pi - t.
// `unmet` counts surviving triangles below 2^20.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace phx::mc::remesh {

using Point = std::array<double, 3>;
using Triangle = std::array<int64_t, 3>;

enum class AdmissionKind : int32_t { kOffsetBand = 0, kZeroSetBand = 1 };

// Certified per-vertex enclosure cached by the remesher; its meaning is owned by
// the admission (offset band: lower/upper bounds of the source distance).
struct VertexEvidence {
  double lower = 0;
  double upper = 0;
};

// Half-angle of a cone that constrains nothing (pi).
inline constexpr double kOpenCone = 3.141592653589793;

// Source normal cone of a triangle's band piece: every unit source normal
// there lies within `half_angle` radians of the unit `axis`; the default open
// cone constrains nothing.
struct NormalCone {
  Point axis{0, 0, 0};
  double half_angle = kOpenCone;
};

class TriangleAdmission {
 public:
  virtual ~TriangleAdmission() = default;
  virtual AdmissionKind kind() const = 0;
  virtual VertexEvidence evaluate(const Point& position) const = 0;
  virtual bool admits(const Point& a, const Point& b, const Point& c,
                      const std::array<VertexEvidence, 3>& evidence) const = 0;
  virtual Point project(const Point& position) const = 0;
  virtual NormalCone normal_cone(const Point& a, const Point& b, const Point& c,
                                 const std::array<VertexEvidence, 3>& evidence) const = 0;
  // A cheap lower bound on normal_cone's half-angle (pi when it is open):
  // edges turning within the two floors need no cones.
  virtual double normal_cone_floor(const Point& a, const Point& b, const Point& c,
                                   const std::array<VertexEvidence, 3>& evidence) const = 0;
};

// Largest angle between a unit vector of `first` and one of `second`, in [0, pi].
double cone_separation(const NormalCone& first, const NormalCone& second);

struct RemeshTargets {
  double edge_length = 0;              // h
  double minimum_angle_degrees = 30;   // full-score minimum angle
};

struct RemeshCounters {
  int64_t splits = 0, collapses = 0, flips = 0, relocations = 0, unmet = 0;
};

// Upper bound on the distance from any point of the closed triangle to its
// nearest corner (circumradius for a non-obtuse triangle, else the larger
// perpendicular-bisector crossing of the longest side), rounded outward and
// widened by `roundoff`; +inf for a degenerate triangle.
double certified_free_radius(const Point& a, const Point& b, const Point& c, double roundoff);

// Remeshes (points, triangles) in place; the output is compacted (only vertices
// of surviving triangles, in first-use order). Status codes are PHX_MC_*.
int32_t remesh_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                       const TriangleAdmission& admission, const RemeshTargets& targets,
                       RemeshCounters& counters);

}  // namespace phx::mc::remesh
