// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// phx_mc_welded_topology against counts of phydrax's host welded-topology
// audit: a closed pair of tetrahedra, tetrahedra pinched at one vertex and
// one edge, and argument refusal.
#include <array>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

// Outward-oriented facets of a positive tetrahedron (a, b, c, d), as the host
// reference-cell topology lists them, and its six edges.
void tetrahedron(const std::array<int64_t, 4>& v, int64_t cell, std::vector<int64_t>& cells,
                 std::vector<int64_t>& facets, std::vector<int64_t>& facet_cells,
                 std::vector<int64_t>& edges, std::vector<int64_t>& edge_cells) {
  cells.insert(cells.end(), v.begin(), v.end());
  const int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  for (const auto& face : faces) {
    for (const int local : face) facets.push_back(v[local]);
    facet_cells.push_back(cell);
  }
  const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  for (const auto& pair : pairs) {
    edges.push_back(v[pair[0]]);
    edges.push_back(v[pair[1]]);
    edge_cells.push_back(cell);
  }
}

struct Result {
  int32_t status;
  std::array<int64_t, 7> counts;
  std::vector<int8_t> boundary;
};

Result audit(const std::vector<std::array<int64_t, 4>>& rows, int64_t key_count) {
  std::vector<int64_t> cells, facets, facet_cells, edges, edge_cells;
  for (std::size_t cell = 0; cell < rows.size(); ++cell) {
    tetrahedron(rows[cell], static_cast<int64_t>(cell), cells, facets, facet_cells, edges,
                edge_cells);
  }
  const int64_t cell_count = static_cast<int64_t>(rows.size());
  const int64_t facet_count = 4 * cell_count;
  std::vector<int8_t> signs(static_cast<std::size_t>(facet_count), 0);
  Result result{};
  result.boundary.assign(static_cast<std::size_t>(facet_count), -1);
  result.status = phx_mc_welded_topology(
      3, 0, 1, 1, key_count, cell_count, cell_count, 4, cells.data(), facet_count, facet_count,
      3, 3, facets.data(), facet_cells.data(), signs.data(), 6 * cell_count, edges.data(),
      edge_cells.data(), -1, nullptr, result.counts.data(), result.boundary.data());
  return result;
}

}  // namespace

int main() {
  // Two tetrahedra glued on face (1, 2, 3): one shared facet, six boundary
  // facets whose nine ridges are each shared twice.
  {
    const Result result = audit({{0, 1, 2, 3}, {4, 1, 3, 2}}, 5);
    PHX_CHECK(result.status == PHX_MC_OK);
    PHX_CHECK((result.counts == std::array<int64_t, 7>{0, 0, 0, 0, 0, 0, 0}));
    int boundary = 0;
    for (const int8_t value : result.boundary) boundary += value;
    PHX_CHECK(boundary == 6);
  }
  // Two tetrahedra sharing only vertex 0: one pinched vertex.
  {
    const Result result = audit({{0, 1, 2, 3}, {0, 5, 4, 6}}, 7);
    PHX_CHECK(result.status == PHX_MC_OK);
    PHX_CHECK((result.counts == std::array<int64_t, 7>{0, 0, 0, 0, 0, 1, 0}));
  }
  // Two tetrahedra sharing only edge (0, 1): a pinched edge and two pinched
  // vertices.
  {
    const Result result = audit({{0, 1, 2, 3}, {0, 1, 4, 5}}, 6);
    PHX_CHECK(result.status == PHX_MC_OK);
    PHX_CHECK(result.counts[5] == 2);
    PHX_CHECK(result.counts[6] == 1);
  }
  // A stacked duplicate: a duplicate cell, four doubled facets with equal
  // orientation, no boundary.
  {
    const Result result = audit({{0, 1, 2, 3}, {0, 1, 2, 3}}, 4);
    PHX_CHECK(result.status == PHX_MC_OK);
    PHX_CHECK(result.counts[1] == 1);
    PHX_CHECK(result.counts[3] == 4);
    PHX_CHECK(result.counts[4] == 0);
  }
  // Two empty facet loops pair consistently (the host reads index 0); a
  // 2-D complex needs facet rows of width two.
  {
    const std::vector<int64_t> cells{0, 0}, facets{-1, -1, -1, -1, -1, -1}, owners{0, 1};
    const std::vector<int8_t> signs{0, 0};
    std::array<int64_t, 7> counts{};
    std::array<int8_t, 2> boundary{};
    PHX_CHECK(phx_mc_welded_topology(3, 0, 0, 0, 1, 2, 2, 1, cells.data(), 2, 2, 3, 3,
                                     facets.data(), owners.data(), signs.data(), 0, nullptr,
                                     nullptr, -1, nullptr, counts.data(),
                                     boundary.data()) == PHX_MC_OK);
    PHX_CHECK(counts[3] == 0);
    PHX_CHECK(phx_mc_welded_topology(2, 0, 0, 0, 1, 2, 2, 1, cells.data(), 2, 2, 1, 1,
                                     facets.data(), owners.data(), signs.data(), 0, nullptr,
                                     nullptr, -1, nullptr, counts.data(),
                                     boundary.data()) == PHX_MC_INVALID_ARGUMENT);
  }
  // Keys at or beyond key_count are refused.
  PHX_CHECK(audit({{0, 1, 2, 3}}, 3).status == PHX_MC_INVALID_ARGUMENT);
  return phx::mc::test::finish("test_welded_topology");
}
