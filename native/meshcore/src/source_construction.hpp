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

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "expansion.hpp"

namespace phx::mc {

enum class SourceStratum : int8_t { kNone = 0, kSegment = 1, kFacet = 2, kPlane = 3 };
// Entity is a row in the exact canonical source bank: source_segments for
// kSegment and source_faces for kFacet and kPlane. It is never a
// recovery-carrier row, face group, or external identity; kNone uses -1. A
// kPlane witness lies on the declared exact plane of its face row's facet:
// origin + parameters[0] * first_axis + parameters[1] * second_axis.
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

bool source_parameters_valid(SourceStratum stratum, const double* parameters);
bool source_point(const double* const* corners, SourceStratum stratum,
                  const double* parameters, Expansion* point);
double nearest_double(const Expansion& value);
double outward_magnitude(const Expansion& value);
double outward_distance(const double* position, const Expansion* point);
// Outward physical norm of the difference between an index-coordinate carrier
// and exact index witness under the row-major linear map.
double outward_affine_distance(const double* matrix, const double* position,
                               const Expansion* point);
bool on_source_entity(const double* const* corners, SourceStratum stratum,
                      const double* position);
double source_deviation(const double* const* corners, SourceStratum stratum,
                        const double* parameters, const double* position);
bool source_carrier(const double* const* corners, SourceStratum stratum,
                    const double* parameters, double* position, double& deviation);
void source_locator(const double* const* corners, SourceStratum stratum,
                    const double* position, double* parameters, bool clamp = true);
void clamp_source_parameters(SourceStratum stratum, double* parameters);

// Exactly witnessed vertices. A source-segment split point S = A + t (B - A)
// with a binary64 parameter t is a dyadic rational point of the segment but
// usually not a binary64 point; its carrier is only the rounding of S. Every
// predicate involving such a vertex evaluates S itself in expansion
// arithmetic, so incidences with the source (collinearity with A and B,
// coplanarity with every facet through AB) hold exactly.
//
// A predicate argument: the carrier, and the exact position when the carrier
// is only its rounding (null when the carrier is the exact position), with
// upper bounds of |exact - carrier| per coordinate when known (null: the
// predicates derive them).
struct ExactPointRef {
  const double* carrier = nullptr;
  const Expansion* exact = nullptr;
  const double* deviation = nullptr;
};

// Exact positions of the witnessed vertices of one vertex bank, keyed by
// vertex index; vertices without an entry are exact at their carrier.
class ExactVertexBank {
 public:
  void clear() {
    slots_.clear();
    positions_.clear();
    deviations_.clear();
  }
  bool empty() const { return positions_.empty(); }
  // Three exact coordinates of `vertex`, or null when its carrier is exact.
  const Expansion* position(int32_t vertex) const {
    const int32_t slot = slot_of(vertex);
    return slot < 0 ? nullptr : positions_[static_cast<std::size_t>(slot)].data();
  }
  // The exact position of `vertex` and the deviation bounds of its carrier.
  void assign(int32_t vertex, const Expansion* position, const double* carrier) {
    if (static_cast<std::size_t>(vertex) >= slots_.size()) {
      slots_.resize(static_cast<std::size_t>(vertex) + 1, -1);
    }
    int32_t& slot = slots_[static_cast<std::size_t>(vertex)];
    if (slot < 0) {
      slot = static_cast<int32_t>(positions_.size());
      positions_.emplace_back();
      deviations_.emplace_back();
    }
    for (int axis = 0; axis < 3; ++axis) {
      positions_[static_cast<std::size_t>(slot)][static_cast<std::size_t>(axis)] =
          position[axis];
      deviations_[static_cast<std::size_t>(slot)][static_cast<std::size_t>(axis)] =
          outward_magnitude(position[axis] - Expansion(carrier[axis]));
    }
  }
  void erase(int32_t vertex) {
    if (vertex >= 0 && static_cast<std::size_t>(vertex) < slots_.size()) {
      slots_[static_cast<std::size_t>(vertex)] = -1;
    }
  }
  ExactPointRef reference(const double* carriers, int32_t vertex) const {
    const int32_t slot = slot_of(vertex);
    if (slot < 0) return {carriers + 3 * static_cast<std::ptrdiff_t>(vertex), nullptr, nullptr};
    return {carriers + 3 * static_cast<std::ptrdiff_t>(vertex),
            positions_[static_cast<std::size_t>(slot)].data(),
            deviations_[static_cast<std::size_t>(slot)].data()};
  }

 private:
  int32_t slot_of(int32_t vertex) const {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= slots_.size()) return -1;
    return slots_[static_cast<std::size_t>(vertex)];
  }

  std::vector<int32_t> slots_;
  // Deques keep returned pointers valid while entries are added.
  std::deque<std::array<Expansion, 3>> positions_;
  std::deque<std::array<double, 3>> deviations_;
};

// Exact predicates on references; all-carrier arguments use the filtered
// binary64 predicates, otherwise expansion arithmetic on the exact positions.
int orient3d(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
             const ExactPointRef& d);
int orient3d_sos(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                 const ExactPointRef& d, int64_t ia, int64_t ib, int64_t ic, int64_t id);
int insphere_sos(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                 const ExactPointRef& d, const ExactPointRef& e, int64_t ia, int64_t ib,
                 int64_t ic, int64_t id, int64_t ie);
// Affine physical insphere, retaining the index-space orientation convention.
// orientation is the exact sign of the linear map's determinant.
int affine_insphere_sos(const double* linear, int orientation,
                        const ExactPointRef& a, const ExactPointRef& b,
                        const ExactPointRef& c, const ExactPointRef& d,
                        const ExactPointRef& e, int64_t ia, int64_t ib,
                        int64_t ic, int64_t id, int64_t ie);
// orient2d of the projections that drop coordinate `axis`, in the cyclic
// coordinate order (axis + 1, axis + 2).
int orient2d_projected(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                       int axis);
bool same_point(const ExactPointRef& a, const ExactPointRef& b);
bool collinear(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c);
// Exact coordinate `axis` of a reference.
Expansion coordinate(const ExactPointRef& point, int axis);
// Feature of the nondegenerate triangle containing x (the codes of
// intersections.hpp: vertex k, 3 + k for the edge opposite vertex k,
// kFeatureInterior, kFeatureNone); side receives orient3d(t0, t1, t2, x).
int8_t locate_on_triangle(const ExactPointRef& x, const ExactPointRef* triangle, int& side);
// Exact position of the source point of a segment witness.
void segment_witness_position(const double* first, const double* second, double parameter,
                              Expansion* position);

}  // namespace phx::mc
