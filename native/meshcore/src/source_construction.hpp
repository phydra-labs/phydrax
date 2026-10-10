//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Ancestry-backed construction on a represented source entity, shared by PLC
// recovery and tetrahedral refinement (implemented in feature_protection.cpp).
//
// A witness names one source segment row (A, B) with the binary64 line
// parameter t, S = A + t (B - A), or one source triangle row (A, B, C) with
// the binary64 barycentric weights (l1, l2), S = A + l1 (B - A) + l2 (C - A).
// S is evaluated exactly in expansion arithmetic and lies on the closed
// entity whenever 0 <= t <= 1, respectively l1, l2 >= 0 and l1 + l2 <= 1
// (checked exactly). The carrier of a witness is the correctly rounded
// binary64 point nearest S, one coordinate at a time.
//
// The deviation of a position from its witness is zero exactly when the
// position lies on the closed source entity (exact collinearity/interval or
// exact triangle location); otherwise it is an outward upper bound of the
// Euclidean distance |position - S|, hence of the distance to the entity.
// Witnesses carry scientific source identity; deviation never permits a
// change of source stratum, region or orientation, which remain decided by
// exact predicates and validated cavity transactions.
#pragma once

#include <cstdint>

#include "expansion.hpp"

namespace phx::mc {

enum class SourceStratum : int8_t { kNone = 0, kSegment = 1, kFacet = 2 };

struct SourceWitness {
  SourceStratum stratum = SourceStratum::kNone;
  int32_t entity = -1;               // source segment or triangle row; -1 when unnamed
  double parameters[2] = {0.0, 0.0};  // t, or (l1, l2)
  double deviation = 0.0;            // outward bound; 0 iff exact membership
};

// Number of corners of the entity of a segment (2) or facet (3) stratum.
inline int source_corner_count(SourceStratum stratum) {
  return stratum == SourceStratum::kSegment ? 2 : 3;
}

// Whether the parameters name a point of the closed entity (exact test).
bool source_parameters_valid(SourceStratum stratum, const double* parameters);

// Exact source point S of the witness parameters; false when invalid.
bool source_point(const double* const* corners, SourceStratum stratum,
                  const double* parameters, Expansion* point);

// The binary64 value nearest to an exact expansion (ties to even).
double nearest_double(const Expansion& value);

// Smallest binary64 value >= |value|.
double outward_magnitude(const Expansion& value);

// Outward upper bound of |position - point| for an exact point.
double outward_distance(const double* position, const Expansion* point);

// Whether a position lies exactly on the closed source entity.
bool on_source_entity(const double* const* corners, SourceStratum stratum,
                      const double* position);

// Certified deviation of a position from the witness parameters, or a
// negative value when the parameters do not name a point of the entity.
double source_deviation(const double* const* corners, SourceStratum stratum,
                        const double* parameters, const double* position);

// Correctly rounded carrier of the witness point and its certified deviation;
// false when the parameters are invalid or the carrier leaves the coordinate
// domain.
bool source_carrier(const double* const* corners, SourceStratum stratum,
                    const double* parameters, double* position, double& deviation);

// Approximate parameters of a position, optionally clamped to the closed entity.
// only rank constructions; a witness built from them is exact by definition.
void source_locator(const double* const* corners, SourceStratum stratum,
                    const double* position, double* parameters, bool clamp = true);

// Clamps parameters onto the closed entity (the complementary barycentric
// weight is checked exactly), e.g. after a rounded convex combination.
void clamp_source_parameters(SourceStratum stratum, double* parameters);

}  // namespace phx::mc
