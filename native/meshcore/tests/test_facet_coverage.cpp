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
  test_history_supersedes_only_older_covered_fragments();
  return phx::mc::test::finish("test_facet_coverage");
}
