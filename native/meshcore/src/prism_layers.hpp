//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Prism layers grown from a closed oriented triangle wall: one straight column
// per wall vertex (the SMOOTH corner policy), construction in floating point,
// certification after construction with exact predicates and local repair by
// thinning (see phx_mc_prism_layers in phydrax_meshcore.h).
#pragma once

#include <cstdint>

namespace phx::mc::prism_layers {

inline constexpr int kStatCount = 21;

struct Input {
  int64_t vertex_count;
  const double* vertices;
  int64_t triangle_count;
  const int32_t* triangles;
  int32_t layer_count;
  const double* heights;
  double feature_angle_degrees;
  double minimum_scale;
  int32_t prove;
  const double* source_normals;          // (m, 3, 3) per triangle corner, or null
  const double* source_shape_operators;  // (m, 3, 6) per triangle corner, or null
};

struct Output {
  double* points;
  int32_t* prisms;
  double* displacement;
  double* scale;
  uint8_t* vertex_kind;
  double* stats;
};

// Validated input (closed, consistently oriented 2-manifold wall; increasing
// positive heights). Returns PHX_MC_OK when every prism is constructed (and,
// with prove, certified), PHX_MC_UNRESOLVED when repair reached the minimum
// scale with a failure left (the outputs then hold the last construction),
// or PHX_MC_INVALID_INPUT / PHX_MC_DEGENERATE_INPUT for a wall that is not
// a closed oriented manifold or has a zero-area triangle.
int32_t grow(const Input& input, const Output& output);

}  // namespace phx::mc::prism_layers
