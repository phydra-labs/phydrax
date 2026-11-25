// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "metric_adapt3d.hpp"

namespace phx::mc::metric {

using BoxPoint = std::array<double, 3>;
using BoxCell = std::array<std::int32_t, 4>;
struct ScalarBoxSample {
  std::int32_t cell = -1;
  double log_rate = 0.0;  // Half the physical log metric (isotropic).
  BoxPoint gradient{};
};

// Numeric proposals only. The transaction separately binds exact source
// carriers, transported metric, walls, volume coverage and ancestry.
class ScalarBoxField {
 public:
  bool prepare(const AdaptInput& input, const BoxPoint& low, const BoxPoint& high);
  bool sample(const BoxPoint& x, ScalarBoxSample& sample) const;
  BoxPoint low{}, high{};
  double maximum_size = 0.0;
 private:
  struct Element {
    BoxPoint origin;
    std::array<BoxPoint, 4> gradients;
    double log_rate;
    BoxPoint log_gradient;
  };
  std::vector<Element> tets_;
  CarrierBoxGrid boxes_;
};

struct ScalarBoxSites {
  std::vector<BoxPoint> points;
  std::vector<ScalarBoxSample> samples;
  std::vector<BoxCell> cells;
};

inline constexpr std::size_t kMaxScalarSites = 100'000;
inline constexpr std::size_t kMaxScalarCells = 600'000;

// Expected geometric/resource refusals decline a proposal. Fatal native
// errors must reach the C boundary rather than trigger another algorithm.
bool scalar_topology_accepted(std::int32_t status);

// Seed with separated metric edge midpoints; rebuild the 12 seams in
// integrated metric length. Candidate rows are deterministic, not original IDs.
bool seed_scalar_box(const AdaptInput& input, const ScalarBoxField& field, ScalarBoxSites& sites);
bool fill_scalar_wall_gaps(const ScalarBoxField& field, ScalarBoxSites& sites);

// Alternate exact point-set triangulation with source-weighted circumcenter
// relocation on each point's actual box stratum; leave the last *pre-move*
// connectivity in sites.cells, as used by the subsequent continuous repair.
bool relax_scalar_box(const ScalarBoxField& field, ScalarBoxSites& sites, int sweeps);

// Replace only changed local chambers with the same oriented exterior;
// unchanged chambers and the entire rejected cell bank retain their rows.
bool retile_scalar_defects(const ScalarBoxField& field, std::span<const BoxPoint> points,
                           std::vector<BoxCell>& cells, std::span<const std::int32_t> seed_cells);

// Correct strict edge-band and Euclidean mean-ratio deficits on a fixed
// positive connectivity; every trial recomputes the log-P1 source field.
bool correct_scalar_box(const ScalarBoxField& field, ScalarBoxSites& sites, double lower);

}  // namespace phx::mc::metric
