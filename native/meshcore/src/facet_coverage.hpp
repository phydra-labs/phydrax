#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace phx::mc {

// Exact planar union proof over explicitly selected original face rows. Only
// positive-area coverage contributes; every subject fragment has exactly one
// owner. Positive-area source contributors have consistent orientation. A
// geometric refusal leaves contributors empty. Original banks contain finite
// triangles. Subject and source points use their represented binary64 values.
bool exact_facet_coverage(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, std::vector<int32_t>& contributors);

// Arrangement-only reference for independently checking the fast exact proof.
bool exact_facet_coverage_by_arrangement(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, std::vector<int32_t>& contributors);

// Chronological exact union: epochs correspond to sorted, unique source rows.
// Newer accepted rows supersede older rows on overlap; two rows in the same
// epoch may not overlap. Epoch zero retains authored orientation. Later
// delegated epochs use geometric coverage after their material-side ordering
// was validated by the accepting transaction.
bool exact_facet_history_coverage(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, std::span<const int64_t> epochs,
    std::vector<int32_t>& contributors);

}  // namespace phx::mc
