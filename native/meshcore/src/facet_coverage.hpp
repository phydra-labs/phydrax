#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace phx::mc {
// Optional construction-local metadata, aligned with original_faces.
struct FacetCoverageBounds {
  std::array<double, 3> lower;
  std::array<double, 3> upper;
  bool nondegenerate;
};
// Construction-local broad phase over validated, bank-ordered source rows.
// The face, bound and row spans must remain alive and unchanged for its life.
class FacetCoverageIndex {
 public:
  FacetCoverageIndex() = default;
  FacetCoverageIndex(const FacetCoverageIndex&) = delete;
  FacetCoverageIndex& operator=(const FacetCoverageIndex&) = delete;
  FacetCoverageIndex(FacetCoverageIndex&&) = default;
  FacetCoverageIndex& operator=(FacetCoverageIndex&&) = default;
  bool build(std::span<const std::array<int32_t, 4>> faces,
             std::span<const int32_t> rows,
             std::span<const FacetCoverageBounds> bounds);
  bool valid() const noexcept { return valid_; }
  std::span<const std::array<int32_t, 4>> faces() const noexcept { return faces_; }
  std::span<const int32_t> rows() const noexcept { return rows_; }
  std::span<const FacetCoverageBounds> bounds() const noexcept { return bounds_; }
  void candidates(const std::array<double, 3>& lower, const std::array<double, 3>& upper,
                  std::vector<std::size_t>& result) const;

 private:
  struct Node {
    std::array<double, 3> lower;
    std::array<double, 3> upper;
    std::size_t begin;
    std::size_t end;
    std::size_t left;
    std::size_t right;
  };
  std::size_t build_node(std::size_t begin, std::size_t end, unsigned depth);
  void collect(std::size_t node, const std::array<double, 3>& lower,
               const std::array<double, 3>& upper, std::vector<std::size_t>& result) const;
  std::span<const std::array<int32_t, 4>> faces_;
  std::span<const int32_t> rows_;
  std::span<const FacetCoverageBounds> bounds_;
  std::vector<std::size_t> order_;
  std::vector<std::size_t> degenerate_;
  std::vector<Node> nodes_;
  bool valid_ = false;
};



// Exact planar union proof over explicitly selected original face rows. Only
// positive-area coverage contributes; every subject fragment has exactly one
// owner. Positive-area source contributors have consistent orientation. A
// geometric refusal leaves contributors empty. Original banks contain finite
// triangles. Subject and source points use their represented binary64 values.
bool exact_facet_coverage(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, std::vector<int32_t>& contributors,
    std::span<const FacetCoverageBounds> bounds = {});
// The index binds the validated, contiguous source rows and their bounds.
bool exact_facet_coverage_prepared(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    const FacetCoverageIndex& index, std::vector<std::size_t>& scratch,
    std::vector<int32_t>& contributors);


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
