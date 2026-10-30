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
// Entity is a row in the exact canonical source bank: source_segments for
// kSegment and source_faces for kFacet. It is never a recovery-carrier row,
// face group, or external identity; kNone uses -1.
struct SourceWitness {
  SourceStratum stratum = SourceStratum::kNone;
  int32_t entity = -1;
  double parameters[2] = {0.0, 0.0};
  double deviation = 0.0;
};

// Recovery witnesses use transient rows: PlcRecovery::plc_edges for kSegment
// and PlcRecovery::input_triangles for kFacet. Translate these through their
// exact source endpoint/identity relations before constructing SourceWitness.
struct PlcRecoveryWitness {
  SourceStratum stratum = SourceStratum::kNone;
  int32_t entity = -1;
  double parameters[2] = {0.0, 0.0};
  double deviation = 0.0;
};

inline int source_corner_count(SourceStratum stratum) {
  return stratum == SourceStratum::kSegment ? 2 : 3;
}

bool source_parameters_valid(SourceStratum stratum, const double* parameters);
bool source_point(const double* const* corners, SourceStratum stratum,
                  const double* parameters, Expansion* point);
double nearest_double(const Expansion& value);
double outward_magnitude(const Expansion& value);
double outward_distance(const double* position, const Expansion* point);
bool on_source_entity(const double* const* corners, SourceStratum stratum,
                      const double* position);
double source_deviation(const double* const* corners, SourceStratum stratum,
                        const double* parameters, const double* position);
bool source_carrier(const double* const* corners, SourceStratum stratum,
                    const double* parameters, double* position, double& deviation);
void source_locator(const double* const* corners, SourceStratum stratum,
                    const double* position, double* parameters, bool clamp = true);
void clamp_source_parameters(SourceStratum stratum, double* parameters);

}  // namespace phx::mc
