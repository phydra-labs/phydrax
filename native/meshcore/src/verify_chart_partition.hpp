// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <vector>

#include "phydrax_meshcore.h"

// The use-coverage and loop checks of phx_mc_verify_chart_partition, shared
// with the CAD fidelity verifier (which runs them as its precondition).
// Arguments are already validated (indices in range, finite ranges).
namespace phx::mc::chart {
// Per use: its chords, ordered along the use, chain bitwise from the first to
// the last use_range value in one loop (PHX_MC_CHART_USE_COVERAGE otherwise).
void check_uses(int64_t use_count, const int64_t* use_patch, const double* use_range,
                int64_t chord_count, const int64_t* chord_use, const int64_t* chord_loop,
                const double* chord_uv, const double* chord_range,
                std::vector<phx_mc_chart_failure>& failures);
// Per loop: its chords on one patch form one closed bitwise UV cycle
// (PHX_MC_CHART_LOOP otherwise).
void check_loops(int64_t loop_count, int64_t chord_count, const int64_t* chord_loop,
                 const int64_t* chord_use, const int64_t* use_patch, const double* uv,
                 std::vector<phx_mc_chart_failure>& failures);
// Per patch: the chord cycles' signed UV area is nonzero, and every
// triangle's orientation agrees with its sign (PHX_MC_CHART_ORIENTATION
// otherwise); oriented[i] = 1 for the agreeing triangles.
void check_orientation(int64_t patch_count, int64_t triangle_count,
                       const int32_t* triangle_patch, const double* triangle_uv,
                       int64_t chord_count, const int64_t* chord_use, const int64_t* use_patch,
                       const double* chord_uv, std::vector<uint8_t>& oriented,
                       std::vector<phx_mc_chart_failure>& failures);
}  // namespace phx::mc::chart
