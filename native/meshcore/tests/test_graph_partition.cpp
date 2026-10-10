//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Multilevel graph partition: cut quality against known optimal cuts, balance,
// disconnected and heavy-vertex graphs, determinism, and input refusals.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Csr {
  std::vector<int64_t> offsets{0};
  std::vector<int32_t> neighbors;
  std::vector<int64_t> edge_weights;
  std::vector<int64_t> vertex_weights;
};

// Builds canonical CSR from undirected (u, v, w) edges.
Csr from_edges(int32_t n, const std::vector<std::vector<int64_t>>& edges) {
  std::vector<std::vector<std::pair<int32_t, int64_t>>> rows(n);
  for (const auto& edge : edges) {
    rows[edge[0]].emplace_back(static_cast<int32_t>(edge[1]), edge[2]);
    rows[edge[1]].emplace_back(static_cast<int32_t>(edge[0]), edge[2]);
  }
  Csr csr;
  for (auto& row : rows) {
    std::sort(row.begin(), row.end());
    for (const auto& [v, w] : row) {
      csr.neighbors.push_back(v);
      csr.edge_weights.push_back(w);
    }
    csr.offsets.push_back(static_cast<int64_t>(csr.neighbors.size()));
  }
  csr.vertex_weights.assign(n, 1);
  return csr;
}

std::vector<std::vector<int64_t>> grid_edges(int32_t nx, int32_t ny, int32_t offset) {
  std::vector<std::vector<int64_t>> edges;
  for (int32_t j = 0; j < ny; ++j) {
    for (int32_t i = 0; i < nx; ++i) {
      const int64_t v = offset + j * nx + i;
      if (i + 1 < nx) {
        edges.push_back({v, v + 1, 1});
      }
      if (j + 1 < ny) {
        edges.push_back({v, v + nx, 1});
      }
    }
  }
  return edges;
}

struct Outcome {
  int32_t status;
  std::vector<int32_t> parts;
  std::vector<int64_t> counters;
};

Outcome run(const Csr& g, int32_t k, const std::vector<int64_t>& targets,
            const std::vector<int64_t>& capacities, int32_t require_nonempty = 1,
            int64_t work_limit = std::numeric_limits<int64_t>::max()) {
  const int64_t n = static_cast<int64_t>(g.vertex_weights.size());
  Outcome out{0, std::vector<int32_t>(n, -1),
              std::vector<int64_t>(PHX_MC_GRAPH_PARTITION_COUNTERS, -1)};
  out.status = phx_mc_graph_partition(n, g.offsets.data(), g.neighbors.data(),
                                      g.edge_weights.data(), g.vertex_weights.data(), k,
                                      targets.data(), capacities.data(), require_nonempty, 8,
                                      work_limit, out.parts.data(), out.counters.data());
  return out;
}

int64_t cut(const Csr& g, const std::vector<int32_t>& parts) {
  int64_t total = 0;
  for (std::size_t v = 0; v + 1 < g.offsets.size(); ++v) {
    for (int64_t e = g.offsets[v]; e < g.offsets[v + 1]; ++e) {
      total += parts[g.neighbors[e]] != parts[v] ? g.edge_weights[e] : 0;
    }
  }
  return total / 2;
}

std::vector<int64_t> part_weights(const Csr& g, const std::vector<int32_t>& parts, int32_t k) {
  std::vector<int64_t> weights(k, 0);
  for (std::size_t v = 0; v < parts.size(); ++v) {
    weights[parts[v]] += g.vertex_weights[v];
  }
  return weights;
}

bool within(const std::vector<int64_t>& weights, const std::vector<int64_t>& capacities) {
  for (std::size_t q = 0; q < weights.size(); ++q) {
    if (weights[q] > capacities[q]) {
      return false;
    }
  }
  return true;
}

void test_grid_bisection_finds_the_strip_cut() {
  const Csr g = from_edges(32 * 32, grid_edges(32, 32, 0));
  const Outcome out = run(g, 2, {512, 512}, {537, 537});
  PHX_CHECK(out.status == PHX_MC_OK);
  // The optimal balanced bisection of a 32 x 32 grid cuts one 32-edge line.
  PHX_CHECK(cut(g, out.parts) <= 36);
  PHX_CHECK(within(part_weights(g, out.parts, 2), {537, 537}));
  PHX_CHECK(out.counters[0] >= 1);
  PHX_CHECK(out.counters[1] <= 40 * 2);
  PHX_CHECK(out.counters[PHX_MC_GRAPH_PARTITION_VISITS] > 0);
}

void test_grid_four_way_is_balanced_and_near_quadrants() {
  const Csr g = from_edges(32 * 32, grid_edges(32, 32, 0));
  const std::vector<int64_t> capacities(4, 269);
  const Outcome out = run(g, 4, {256, 256, 256, 256}, capacities);
  PHX_CHECK(out.status == PHX_MC_OK);
  // Quadrants cut 64 edges; strips cut 96.
  PHX_CHECK(cut(g, out.parts) <= 80);
  PHX_CHECK(within(part_weights(g, out.parts, 4), capacities));
}

void test_partition_is_deterministic() {
  const Csr g = from_edges(24 * 20, grid_edges(24, 20, 0));
  const std::vector<int64_t> targets{160, 160, 160};
  const std::vector<int64_t> capacities{168, 168, 168};
  const Outcome first = run(g, 3, targets, capacities);
  const Outcome second = run(g, 3, targets, capacities);
  PHX_CHECK(first.status == PHX_MC_OK);
  PHX_CHECK(first.parts == second.parts);
  PHX_CHECK(first.counters == second.counters);
}

void test_disconnected_components_are_not_cut() {
  std::vector<std::vector<int64_t>> edges;
  for (int32_t component = 0; component < 4; ++component) {
    const auto block = grid_edges(6, 6, component * 36);
    edges.insert(edges.end(), block.begin(), block.end());
  }
  const Csr g = from_edges(4 * 36, edges);
  const std::vector<int64_t> capacities(4, 37);
  const Outcome out = run(g, 4, {36, 36, 36, 36}, capacities);
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK(cut(g, out.parts) == 0);
  PHX_CHECK(within(part_weights(g, out.parts, 4), capacities));
}

void test_weighted_edges_steer_the_cut() {
  // Two 8 x 8 grids joined along a full column of heavy edges and, on the
  // other side, a light seam: the partition must cut the light seam.
  std::vector<std::vector<int64_t>> edges = grid_edges(16, 8, 0);
  for (auto& edge : edges) {
    const int64_t u = edge[0] % 16;
    const int64_t v = edge[1] % 16;
    if (u == 7 && v == 8) {
      edge[2] = 1;
    } else {
      edge[2] = 50;
    }
  }
  const Csr g = from_edges(128, edges);
  const Outcome out = run(g, 2, {64, 64}, {64, 64});
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK(cut(g, out.parts) == 8);
}

void test_heavy_vertex_stays_whole_and_is_measurable() {
  std::vector<std::vector<int64_t>> edges;
  for (int64_t v = 0; v + 1 < 12; ++v) {
    edges.push_back({v, v + 1, 1});
  }
  Csr g = from_edges(12, edges);
  g.vertex_weights[5] = 40;
  // Total 51; no part of capacity 27 can hold the 40-weight vertex.
  const Outcome out = run(g, 2, {26, 25}, {27, 27});
  PHX_CHECK(out.status == PHX_MC_OK);
  const std::vector<int64_t> weights = part_weights(g, out.parts, 2);
  PHX_CHECK(std::max(weights[0], weights[1]) >= 40);
  PHX_CHECK(std::min(weights[0], weights[1]) >= 1);
}

void test_every_part_is_nonempty() {
  std::vector<std::vector<int64_t>> edges;
  for (int64_t v = 0; v + 1 < 5; ++v) {
    edges.push_back({v, v + 1, 1});
  }
  Csr g = from_edges(5, edges);
  g.vertex_weights = {1, 0, 0, 0, 0};
  const Outcome out = run(g, 5, {1, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  PHX_CHECK(out.status == PHX_MC_OK);
  std::vector<int32_t> sorted = out.parts;
  std::sort(sorted.begin(), sorted.end());
  PHX_CHECK((sorted == std::vector<int32_t>{0, 1, 2, 3, 4}));
}

void test_work_limit_is_a_resource_refusal() {
  const Csr g = from_edges(32 * 32, grid_edges(32, 32, 0));
  const Outcome out = run(g, 2, {512, 512}, {537, 537}, 1, 100);
  PHX_CHECK(out.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(out.counters[PHX_MC_GRAPH_PARTITION_VISITS] + out.counters[10] > 100);
  PHX_CHECK(std::all_of(out.parts.begin(), out.parts.end(), [](int32_t q) { return q == -1; }));
}

void test_late_work_refusal_does_not_publish_a_partial_partition() {
  Csr g = from_edges(4, {{0, 1, 1}, {2, 3, 1}});
  g.vertex_weights = {8, 1, 8, 1};
  const std::vector<int64_t> targets{5, 4, 5, 4}, capacities(4, 5);
  const Outcome full = run(g, 4, targets, capacities);
  PHX_CHECK(full.status == PHX_MC_OK);
  const int64_t budget = full.counters[PHX_MC_GRAPH_PARTITION_VISITS] + full.counters[10] - 1;
  const Outcome refused = run(g, 4, targets, capacities, 1, budget);
  PHX_CHECK(refused.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(std::all_of(refused.parts.begin(), refused.parts.end(),
                       [](int32_t q) { return q == -1; }));
}

void test_isolated_vertex_work_is_bounded() {
  const Csr g = from_edges(100, {});
  const Outcome out = run(g, 2, {50, 50}, {50, 50}, 1, 5);
  PHX_CHECK(out.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(out.counters[PHX_MC_GRAPH_PARTITION_VISITS] == 0);
  PHX_CHECK(out.counters[10] > 5);
}

void test_weighted_disconnected_packing_respects_exact_capacity() {
  Csr g = from_edges(5, {});
  g.vertex_weights = {8, 7, 6, 5, 4};
  const Outcome out = run(g, 2, {15, 15}, {15, 15});
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK((part_weights(g, out.parts, 2) == std::vector<int64_t>{15, 15}));
  PHX_CHECK(cut(g, out.parts) == 0);
}

void test_fine_graph_never_relaxes_exact_capacity() {
  const Csr g = from_edges(32 * 32, grid_edges(32, 32, 0));
  const std::vector<int64_t> targets(8, 128);
  const Outcome out = run(g, 8, targets, targets);
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK(part_weights(g, out.parts, 8) == targets);
}

void test_impossible_packing_is_returned_without_capacity_relaxation() {
  Csr g = from_edges(3, {});
  g.vertex_weights = {2, 2, 2};
  const Outcome out = run(g, 2, {3, 3}, {3, 3});
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK(!within(part_weights(g, out.parts, 2), {3, 3}));
  PHX_CHECK(std::all_of(out.parts.begin(), out.parts.end(),
                       [](int32_t q) { return q == 0 || q == 1; }));
  PHX_CHECK(*std::min_element(out.parts.begin(), out.parts.end()) == 0);
  PHX_CHECK(*std::max_element(out.parts.begin(), out.parts.end()) == 1);
}

void test_disconnected_kway_matches_independent_grid_cut() {
  auto edges = grid_edges(16, 16, 0);
  const auto second = grid_edges(16, 16, 256);
  edges.insert(edges.end(), second.begin(), second.end());
  const Csr g = from_edges(512, edges);
  const std::vector<int64_t> targets(32, 16);
  const Outcome out = run(g, 32, targets, targets);
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK(part_weights(g, out.parts, 32) == targets);
  // Each component admits sixteen 4x4 blocks, cutting 96 edges.
  PHX_CHECK(cut(g, out.parts) <= 192);
}

void test_disconnected_spatial_kway_matches_independent_grid_cut() {
  std::vector<std::vector<int64_t>> edges;
  for (int32_t block = 0; block < 2; ++block) {
    for (int32_t v = 0; v < 512; ++v) {
      const int32_t u = v + block * 512;
      if (v % 8 < 7) {
        edges.push_back({u, u + 1, 1});
      }
      if ((v / 8) % 8 < 7) {
        edges.push_back({u, u + 8, 1});
      }
      if (v / 64 < 7) {
        edges.push_back({u, u + 64, 1});
      }
    }
  }
  const Csr g = from_edges(1024, edges);
  const std::vector<int64_t> targets(32, 32);
  const Outcome out = run(g, 32, targets, targets);
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK(part_weights(g, out.parts, 32) == targets);
  // Each 8x8x8 component admits a 4x2x2 block arrangement cutting 320 edges.
  PHX_CHECK(cut(g, out.parts) <= 640);
}

void test_component_weights_match_unequal_capacity_parts() {
  Csr g = from_edges(4, {{0, 1, 1}, {2, 3, 1}});
  g.vertex_weights = {4, 4, 1, 1};
  const Outcome out = run(g, 2, {2, 8}, {2, 8});
  PHX_CHECK(out.status == PHX_MC_OK);
  PHX_CHECK((part_weights(g, out.parts, 2) == std::vector<int64_t>{2, 8}));
  PHX_CHECK(cut(g, out.parts) == 0);
}

void test_malformed_inputs_are_refused() {
  const Csr good = from_edges(4, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}});
  const std::vector<int64_t> targets{2, 2};
  const std::vector<int64_t> capacities{2, 2};
  PHX_CHECK(run(good, 2, targets, capacities).status == PHX_MC_OK);

  Csr asymmetric = good;
  asymmetric.edge_weights[0] = 3;
  PHX_CHECK(run(asymmetric, 2, targets, capacities).status == PHX_MC_INVALID_INPUT);

  Csr unsorted = from_edges(3, {{0, 1, 1}, {0, 2, 1}});
  std::swap(unsorted.neighbors[0], unsorted.neighbors[1]);
  PHX_CHECK(run(unsorted, 1, {3}, {3}).status == PHX_MC_INVALID_INPUT);

  Csr self_loop = good;
  self_loop.neighbors[0] = 0;
  PHX_CHECK(run(self_loop, 2, targets, capacities).status == PHX_MC_INVALID_INPUT);

  Csr negative = good;
  negative.vertex_weights[2] = -1;
  PHX_CHECK(run(negative, 2, {1, 0}, {2, 2}).status == PHX_MC_INVALID_INPUT);

  PHX_CHECK(run(good, 2, {2, 1}, capacities).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(run(good, 2, targets, {1, 2}).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(run(good, 5, {1, 1, 1, 1, 0}, {1, 1, 1, 1, 1}).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(run(good, 5, {1, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, 0).status == PHX_MC_OK);
  PHX_CHECK(run(good, 0, targets, capacities).status == PHX_MC_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  test_grid_bisection_finds_the_strip_cut();
  test_grid_four_way_is_balanced_and_near_quadrants();
  test_partition_is_deterministic();
  test_disconnected_components_are_not_cut();
  test_weighted_edges_steer_the_cut();
  test_heavy_vertex_stays_whole_and_is_measurable();
  test_every_part_is_nonempty();
  test_work_limit_is_a_resource_refusal();
  test_late_work_refusal_does_not_publish_a_partial_partition();
  test_isolated_vertex_work_is_bounded();
  test_weighted_disconnected_packing_respects_exact_capacity();
  test_fine_graph_never_relaxes_exact_capacity();
  test_impossible_packing_is_returned_without_capacity_relaxation();
  test_disconnected_kway_matches_independent_grid_cut();
  test_disconnected_spatial_kway_matches_independent_grid_cut();
  test_component_weights_match_unequal_capacity_parts();
  test_malformed_inputs_are_refused();
  return phx::mc::test::finish("test_graph_partition");
}
