// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// kZeroSetBand admission for the certified surface remesher (surface_remesh.hpp):
// a triangle represents the zero set Z = {f = 0} of a lowered scalar field f when
//  * distance: every point of the closed triangle T lies within the declared
//    tolerance of Z. Let F enclose f over T and G enclose grad f over the box B
//    = bbox(T) widened by r, with m = min |g| over g in G > 0. Following the
//    normalized gradient flow from x in T, |f| falls at rate >= m while the
//    path stays in the ball of radius r around x (inside B), so Z is reached
//    within |f(x)| / m. Hence sup |F| <= r m certifies dist(x, Z) <= sup |F| / m
//    <= r for every x in T; r is tried at twice the bbox estimate and at the
//    tolerance itself. F intersects the natural enclosure over bbox(T) with
//    corner-anchored mean-value forms along the triangle's edges. A second
//    proof outward-encloses T-d, T+d and their complete extrusion: uniform
//    opposite endpoint signs and a strict sign of grad f . d throughout it
//    isolate a zero along every point's segment, at distance <= |d|. Directed
//    normalization bounds |d| and retains the subdivision-rounding slack.
//    An undecided piece splits into four until certified, refuted (|f| exceeds
//    the gradient-norm upper bound times the tolerance), or at binary64
//    resolution, which refuses;
//  * transversality: the triangle normal n satisfies orientation * (n . g) > 0
//    for every g enclosing grad f over bbox(T). This is the strict-gradient
//    sign witness the adaptive scaffold carries per leaf cell: T is a local
//    graph over Z along the gradient lines, with the carrier's orientation.
// f and grad f are evaluated by the native straight-line interval executor
// (interval_program.cpp) on one three-dimensional box per call; both are
// assumed C^1 wherever the gradient enclosure is finite.
// The admission is sound but incomplete: an undecided enclosure refuses.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "interval_program.hpp"
#include "surface_remesh.hpp"

namespace phx::mc::remesh {

struct Interval {
  double lower = 0;
  double upper = 0;
};

// One validated interval program over a single 3-d input box. The workspace
// template carries the program's constant seeds; entries [0, 3) receive the
// box and `outputs` name the workspace entries of the result components.
class IntervalBoxProgram {
 public:
  // PHX_MC_OK, or the refusal status for an invalid program or extent.
  int32_t bind(const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
               int64_t index_count, const double* parameters, int64_t parameter_count,
               const double* workspace_lower, const double* workspace_upper,
               int64_t workspace_size, const int64_t* outputs, int64_t output_count,
               const int64_t* guards, int64_t guard_count);
  // Encloses every output over the box; false when execution fails or an
  // output is NaN. `result` must hold output_count() intervals.
  bool evaluate(const Point& lower, const Point& upper, Interval* result) const;
  int64_t output_count() const { return static_cast<int64_t>(outputs_.size()); }
  const phx::mc::PreparedIntervalProgram& program() const noexcept { return program_; }
  const std::vector<int64_t>& outputs() const noexcept { return outputs_; }
  const std::vector<int64_t>& guards() const noexcept { return guards_; }
  const std::vector<double>& template_lower() const noexcept { return template_lower_; }
  const std::vector<double>& template_upper() const noexcept { return template_upper_; }

 private:
  phx::mc::PreparedIntervalProgram program_;
  std::vector<int64_t> outputs_, guards_;
  std::vector<double> template_lower_, template_upper_;
  mutable std::vector<double> lower_, upper_;  // single-threaded scratch workspace
};

class ZeroSetBandAdmission final : public TriangleAdmission {
 public:
  ZeroSetBandAdmission(const IntervalBoxProgram& value, const IntervalBoxProgram& gradient,
                       double tolerance, int orientation)
      : value_(value), gradient_(gradient), tolerance_(tolerance), orientation_(orientation) {}

  AdmissionKind kind() const override { return AdmissionKind::kZeroSetBand; }
  // Point enclosure of f at the vertex (mean-value anchor for admits()).
  VertexEvidence evaluate(const Point& position) const override;
  bool admits(const Point& a, const Point& b, const Point& c,
              const std::array<VertexEvidence, 3>& evidence) const override;
  // Interval Newton along the gradient line through `position`; returns the
  // position unchanged when no root is isolated (a proposal only).
  Point project(const Point& position) const override;
  // Gradient-enclosure cone over bbox(T) widened by the tolerance, widened by
  // the tilt a triangle within the tolerance of Z may have.
  NormalCone normal_cone(const Point& a, const Point& b, const Point& c,
                         const std::array<VertexEvidence, 3>& evidence) const override;
  // The tilt term every cone carries.
  double normal_cone_floor(const Point& a, const Point& b, const Point& c,
                           const std::array<VertexEvidence, 3>& evidence) const override;

  // Certified upper bound on max_{x in T} dist(x, Z), +inf when undecided.
  double distance_bound(const Point& a, const Point& b, const Point& c,
                        const std::array<VertexEvidence, 3>& evidence) const;
  // Certified sign of n . grad f over bbox(T): +1, -1, or 0 when undecided.
  int transversal_sign(const Point& a, const Point& b, const Point& c) const;

  // Refused admissions by first failing condition (evidence for the caller).
  struct Refusals {
    int64_t transversality = 0;
    int64_t distance = 0;
  };
  const Refusals& refusals() const { return refusals_; }
  // Proposal sample from the first rejecting subdivision piece, in the
  // original triangle's vertex order; not a distance extremum or certificate.
  const std::optional<Point>& distance_barycentric() const { return distance_barycentric_; }

 private:
  // One unsplit piece: the residual/gradient bound over its (widened) box.
  double piece_distance(const Point& a, const Point& b, const Point& c,
                        const std::array<VertexEvidence, 3>& evidence) const;
  bool translated_field(const Point& a, const Point& b, const Point& c,
                        const Point& displacement, double sign, Interval& field) const;
  double extrusion_distance(const Point& a, const Point& b, const Point& c,
                            double radius) const;
  // Certified: no point of the piece lies within the tolerance of Z.
  bool piece_refuted(const Point& a, const Point& b, const Point& c) const;
  double split_distance(const Point& a, const Point& b, const Point& c,
                        const std::array<VertexEvidence, 3>& evidence, double slack) const;

  const IntervalBoxProgram& value_;
  const IntervalBoxProgram& gradient_;
  double tolerance_;
  int orientation_;
  mutable Refusals refusals_;  // single-threaded, like the program scratch
  mutable std::optional<Point> distance_barycentric_;
};

}  // namespace phx::mc::remesh
