// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "metric_adapt3d.hpp"

namespace phx::mc::metric {

inline constexpr std::int64_t kMaxLayerSourceCells = 250'000;

// A proposed conforming tetrahedral grid. Carrier rows refer to the immutable
// source; tensors are the transported values at those exact dyadic carriers.
struct MetricLayerGrid {
  std::vector<Carrier> vertices;
  std::vector<std::array<double, 9>> tensors;
  std::vector<std::array<std::int32_t, 4>> cells;
};

// For a box with constant Cartesian tangential metrics and one varying normal
// metric, generate and measure a graded grid. The transaction owner proves
// the box and its source groups; neither generator nor scales identify faces.
std::optional<MetricLayerGrid> propose_metric_layer_grid(
    const AdaptInput& input, const CarrierSource& source,
    const std::array<double, 3>& low, const std::array<double, 3>& high);

// An ineligible source or unusable proposal leaves state and output unchanged.
// A committed whole-volume operation writes its genuine stencil, cell
// ancestry, generation and status evidence. Fatal transaction errors propagate.
std::optional<TetEditStatus> adapt_metric_layers(TetMesh& mesh, const AdaptInput& input,
                                                const CarrierSource& source, AdaptOutput& output);

}  // namespace phx::mc::metric
