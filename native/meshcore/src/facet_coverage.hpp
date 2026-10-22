#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "bounded_memory.hpp"

namespace phx::mc {

// Exact planar union proof over explicitly selected original face rows. Only
// positive-area coverage contributes; every subject fragment must have exactly
// one owner. Positive-area source contributors must have a consistent
// orientation, which may oppose the subject. Noncoplanar rows are ignored
// (they cannot cover area in the subject plane). Geometric refusal leaves
// contributors empty; resource
// failures propagate through the caller's native execution/memory scope.
// original_points and original_faces must be valid, finite triangle banks.
// Before any arrangement is built, exact orientation predicates (one work unit
// each) decide two configurations and return the arrangement's result for
// every such configuration: a subject inside one closed coplanar row whose
// other coplanar rows are each weakly separated by an edge line, and a subject
// inside the union of pairwise separated, consistently oriented contributors
// whose unshared union edges all miss the open subject.
bool exact_facet_coverage(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, NativeVector<int32_t>& contributors);

// The same admission and arrangement proof without the predicate proofs: the
// independent reference compared by the native tests.
bool exact_facet_coverage_by_arrangement(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, NativeVector<int32_t>& contributors);

}  // namespace phx::mc
