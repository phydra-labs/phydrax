// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Floating-point remeshing of a closed, consistently oriented triangle surface
// toward a smooth source surface: construction without proofs.
//
// The adapter owns flat arrays (positions, unit source normals, target sizes,
// triangle corners and the opposite half-edge of every corner) and runs
// worklist sweeps of edge splits, edge collapses, edge flips and tangential
// relocations, every new or moved vertex projected onto the source by the
// oracle. Exact predicates decide nothing here and no edit is proved:
// certify_surface certifies the finished surface (surface_certify.hpp) and
// repairs what fails locally. The oracle is floating point too.
//
// Target size: h at every vertex, lowered where the source curves so that a
// well-shaped triangle of that size keeps its sagitta within the tolerance
// (kappa L^2 / 6 for an equilateral triangle of side L on a surface of normal
// curvature kappa). Thresholds follow MMGS: split above 4/3 of the size,
// collapse below 4/5 of it.
//
// Passes: a size sweep (splits, then collapses, then valence flips, then
// tangential relocation toward the area-weighted ring centroid) repeated a
// fixed number of times; then a quality sweep (Delaunay flips, and for each
// triangle below the minimum angle: relocation of its corners, collapse of
// its shortest edge, or split of its longest) repeated while it improves;
// then a fidelity sweep splitting the longest edge of every triangle whose
// sampled distance (corners, edge midpoints, centroid) exceeds the tolerance
// with a margin. Every edit keeps the surface a closed, consistently oriented
// 2-manifold (link condition for collapses, no duplicate edge for flips) and
// refuses a created triangle whose normal turns more than 60 degrees from the
// source normal at its corners.
#pragma once

#include <cstdint>
#include <vector>

#include "surface_certify.hpp"

namespace phx::mc::remesh {

// A point of the source surface with its oriented unit normal and its two
// principal curvatures (in either order, signed).
struct SurfacePoint {
  Point on{0, 0, 0};
  Point normal{0, 0, 0};
  double curvature_first = 0;
  double curvature_second = 0;
};

// A smooth closed surface the adapter projects onto. Floating point: its
// answers steer construction and bound nothing.
class SurfaceOracle {
 public:
  virtual ~SurfaceOracle() = default;
  // A point of the surface near `position`, its unit normal oriented like the
  // surface being remeshed, and its curvature; false when none is found.
  virtual bool project(const Point& position, SurfacePoint& out) const = 0;
  // project() for every position; found[i] says whether out[i] holds a point.
  // Implementations may evaluate in blocks, which is cheaper than one call per
  // position; the results are those of project().
  virtual void project_all(const std::vector<Point>& positions, std::vector<SurfacePoint>& out,
                           std::vector<uint8_t>& found) const = 0;
  // First-order distance estimates of every position from the surface.
  virtual void distances(const std::vector<Point>& positions, std::vector<double>& out) const = 0;
};

struct AdaptTargets {
  double edge_length = 0;              // h
  double tolerance = 0;                // distance target, both directions
  double minimum_angle_degrees = 30;   // quality target
  // Keep the input's triangles around creases (edges turning more than 45
  // degrees) bitwise; off for sources without creases.
  bool keep_creases = true;
};

struct AdaptCounters {
  int64_t splits = 0, collapses = 0, flips = 0, relocations = 0;
  // Triangles left below the minimum angle, and triangles whose sampled
  // distance is still above the tolerance.
  int64_t below_angle = 0, above_tolerance = 0;
};

// An owned surface returned through the C ABI (phx_mc_surface_result_*);
// `carried` is empty or holds one flag per triangle.
struct SurfaceResult {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  std::vector<uint8_t> carried;
};

// Remeshes (points, triangles) in place and compacts the output (only
// vertices of surviving triangles, in first-use order). The input must be a
// closed, consistently oriented 2-manifold (PHX_MC_INVALID_INPUT otherwise);
// a vertex the oracle cannot project is PHX_MC_UNRESOLVED. On any status but
// PHX_MC_OK the vectors hold no result.
int32_t adapt_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                      const SurfaceOracle& oracle, const AdaptTargets& targets,
                      AdaptCounters& counters);

struct CertifyCounters {
  double distance = 0;            // maximum certified or carrier bound
  int64_t refused_fidelity = 0;   // certifier refusals, over all rounds
  int64_t refused_embedding = 0;  // degenerate or illegal-contact triangles, over all rounds
  int64_t carrier = 0;            // triangles accepted from the carrier
  int64_t repair_splits = 0;
  int64_t rounds = 0;             // certification rounds run
  // Per triangle of the last round (the certified surface on PHX_MC_OK):
  // 1 when it was accepted from the carrier.
  std::vector<uint8_t> carried;
};

// Certifies (points, triangles) (surface_certify.hpp) and repairs refused
// triangles locally: the longest edge of each that is not kept is split at
// its projected midpoint, Delaunay flips restore the angles, and only the
// triangles that changed (or were refused) are decided again, for at most
// four rounds. Vertices are not moved otherwise. With `repair_certifier`
// (a tighter band, leaving the other direction room), a triangle a repair
// created must also be proved by it: one proved only by `certifier` is
// repaired again but not refused, and when only such triangles remain after
// the last round (or no split takes) the last surface without a refusal
// stands. PHX_MC_OK leaves the certified surface (compacted after a
// repair); PHX_MC_UNRESOLVED leaves the last state with a refused triangle.
// Counters hold the evidence either way.
int32_t certify_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                        const SurfaceOracle& oracle, const AdaptTargets& targets,
                        const TriangleCertifier& certifier, const CarrierEvidence& carrier,
                        const TriangleCertifier* repair_certifier, CertifyCounters& counters);

// certify_surface's local repair, for refusals decided elsewhere: the longest
// edge of each flagged triangle that is not kept is split at its projected
// midpoint and Delaunay flips follow; `splits` counts the splits. The
// surface is compacted when anything was split and unchanged otherwise.
int32_t repair_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                       const SurfaceOracle& oracle, const AdaptTargets& targets,
                       const std::vector<uint8_t>& refused, int64_t& splits);

// Restores angles after a repair: Delaunay flips, and the corners of every
// triangle below the minimum angle relocated (tangentially, then projected)
// while that lifts their stars; no collapses or splits. `relocations` counts
// the moves. With any, the surface is stored (compacted, flips included);
// with none it is left unchanged.
int32_t relax_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                      const SurfaceOracle& oracle, const AdaptTargets& targets,
                      int64_t& relocations);

}  // namespace phx::mc::remesh
