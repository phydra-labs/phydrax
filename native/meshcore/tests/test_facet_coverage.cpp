//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact source-facet union coverage and chronological ancestry behavior.
#include <array>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "facet_coverage.hpp"

namespace {

using namespace phx::mc;

void test_exact_partition_and_arrangement_reference_agree() {
  const double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.5, 0.5, 0.0};
  const double subject[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  const std::array<std::array<int32_t, 4>, 2> faces{{
      {0, 1, 3, 7}, {0, 3, 2, 7}}};
  const std::array<int32_t, 2> rows{0, 1};
  std::vector<int32_t> fast, reference;
  PHX_CHECK(exact_facet_coverage(subject, points, faces, rows, fast));
  PHX_CHECK(exact_facet_coverage_by_arrangement(subject, points, faces, rows, reference));
  PHX_CHECK(fast == std::vector<int32_t>({0, 1}));
  PHX_CHECK(reference == fast);

  const std::array<int32_t, 1> incomplete{0};
  PHX_CHECK(!exact_facet_coverage(subject, points, faces, incomplete, fast));
  PHX_CHECK(fast.empty());
}

void test_noncontiguous_rows_and_disjoint_degenerate_partner() {
  const double subject[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const double points[] = {
      0, 0, 1, 1, 0, 1, 0, 1, 1,
      2, 0, 0, 3, 0, 0, 4, 0, 0,
      0, 1, 0, 0, 0, 0, 1, 0, 0,
      2, 2, 0, 3, 2, 0, 2, 3, 0,
      1, 0, 0, 2, 0, 0, 1, 1, 0};
  const std::array<std::array<int32_t, 4>, 5> faces{{
      {0, 1, 2, 0}, {3, 4, 5, 0}, {7, 8, 6, 0}, {9, 10, 11, 0}, {12, 13, 14, 0}}};
  const std::array<FacetCoverageBounds, 5> bounds{{
      {{{0, 0, 1}}, {{1, 1, 1}}, true},
      {{{2, 0, 0}}, {{4, 0, 0}}, false},
      {{{0, 0, 0}}, {{1, 1, 0}}, true},
      {{{2, 2, 0}}, {{3, 3, 0}}, true},
      {{{1, 0, 0}}, {{2, 1, 0}}, true},
  }};
  const std::array<int32_t, 4> selected{4, 3, 2, 0};
  std::vector<int32_t> contributors;
  PHX_CHECK(exact_facet_coverage(subject, points, faces, selected, contributors));
  PHX_CHECK(contributors == std::vector<int32_t>({2}));
  PHX_CHECK(exact_facet_coverage(subject, points, faces, selected, contributors, bounds));
  PHX_CHECK(contributors == std::vector<int32_t>({2}));
  const std::array<int32_t, 3> degenerate{2, 1, 0};
  PHX_CHECK(!exact_facet_coverage(subject, points, faces, degenerate, contributors));
  PHX_CHECK(contributors.empty());
  PHX_CHECK(!exact_facet_coverage(subject, points, faces, degenerate, contributors, bounds));
  PHX_CHECK(contributors.empty());
  const std::array<int32_t, 1> far_coplanar{3};
  const std::array<int32_t, 1> far_noncoplanar{0};
  PHX_CHECK(!exact_facet_coverage(subject, points, faces, far_coplanar, contributors, bounds));
  PHX_CHECK(contributors.empty());
  PHX_CHECK(!exact_facet_coverage(subject, points, faces, far_noncoplanar, contributors, bounds));
  PHX_CHECK(contributors.empty());
  const std::array<int32_t, 2> repeated_sorted{2, 2};
  const std::array<int32_t, 3> repeated_unsorted{2, 0, 2};
  PHX_CHECK(!exact_facet_coverage(subject, points, faces, repeated_sorted, contributors));
  PHX_CHECK(!exact_facet_coverage(subject, points, faces, repeated_unsorted, contributors));
  for (unsigned mask = 1; mask < (1U << faces.size()); ++mask) {
    std::vector<int32_t> subset;
    for (int row = static_cast<int>(faces.size()) - 1; row >= 0; --row) {
      if ((mask & (1U << row)) != 0) subset.push_back(row);
    }
    std::vector<int32_t> unfiltered, filtered;
    const bool expected = exact_facet_coverage(subject, points, faces, subset, unfiltered);
    const bool actual = exact_facet_coverage(subject, points, faces, subset, filtered, bounds);
    PHX_CHECK(actual == expected);
    PHX_CHECK(filtered == unfiltered);
  }
}

void test_history_supersedes_only_older_covered_fragments() {
  const double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.5, 0.5, 0.0};
  const double subject[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  const std::array<std::array<int32_t, 4>, 3> faces{{
      {0, 1, 3, 7}, {0, 3, 2, 7}, {0, 1, 2, 7}}};
  const std::array<int32_t, 3> rows{0, 1, 2};
  const std::array<int64_t, 3> epochs{0, 0, 1};
  std::vector<int32_t> contributors;
  PHX_CHECK(exact_facet_history_coverage(subject, points, faces, rows, epochs, contributors));
  PHX_CHECK(contributors == std::vector<int32_t>({2}));

  const std::array<int64_t, 3> overlapping_epoch{0, 0, 0};
  PHX_CHECK(!exact_facet_history_coverage(subject, points, faces, rows,
                                           overlapping_epoch, contributors));
  PHX_CHECK(contributors.empty());
}

}  // namespace

int main() {
  test_exact_partition_and_arrangement_reference_agree();
  test_noncontiguous_rows_and_disjoint_degenerate_partner();
  test_history_supersedes_only_older_covered_fragments();
  return phx::mc::test::finish("test_facet_coverage");
}
