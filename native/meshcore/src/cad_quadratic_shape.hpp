// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cad_curving.hpp"

namespace phx::mc {

// After P2 lifting, polish only interior edge nodes and count cells missing
// each enabled degree-six sampled target (scaled Jacobian, determinant ratio).
// Zero disables a target. On UNRESOLVED the node bank is unchanged and unmet
// describes that bank. On OK, touched_cells lists all cells incident to an
// actually changed node, in ascending order; the caller must exact-decide them.
std::int32_t polish_cad_quadratic_shape(const CadQuadraticInput& input, double length_scale,
                                       double minimum_scaled, double minimum_ratio,
                                       CadQuadraticMesh& mesh,
                                       std::vector<std::int64_t>& touched_cells,
                                       std::array<std::int64_t, 2>& unmet,
                                       std::vector<std::int64_t>* moved_edges = nullptr);

}  // namespace phx::mc
