//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C entry of the prism layers: argument validation, then prism_layers.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "prism_layers.hpp"

namespace {

int32_t validate(const phx_mc_prism_layers_request* in, const phx_mc_prism_layers_result* out) {
  if (in == nullptr || out == nullptr || in->vertex_count < 3 || in->triangle_count < 4 ||
      in->layer_count < 1 || in->vertices == nullptr || in->triangles == nullptr ||
      in->heights == nullptr || out->points == nullptr || out->prisms == nullptr ||
      out->displacement == nullptr || out->scale == nullptr || out->vertex_kind == nullptr ||
      out->stats == nullptr || (in->prove != 0 && in->prove != 1) ||
      (in->source_normals == nullptr) != (in->source_shape_operators == nullptr) ||
      !(in->feature_angle_degrees > 0.0 && in->feature_angle_degrees < 180.0) ||
      !(in->minimum_scale > 0.0 && in->minimum_scale <= 1.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t levels = static_cast<int64_t>(in->layer_count) + 1;
  if (in->vertex_count > std::numeric_limits<int32_t>::max() / levels ||
      !phx::mc::addressable(in->vertex_count * levels, 3, sizeof(double)) ||
      !phx::mc::addressable(in->triangle_count, 3 * levels, sizeof(int32_t)) ||
      in->triangle_count > std::numeric_limits<int64_t>::max() / 3) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  for (int64_t k = 0; k < 3 * in->vertex_count; ++k) {
    if (!std::isfinite(in->vertices[k])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(in->vertices[k])) return PHX_MC_RANGE_ERROR;
  }
  double previous = 0.0;
  for (int32_t k = 0; k < in->layer_count; ++k) {
    if (!std::isfinite(in->heights[k])) return PHX_MC_NONFINITE_INPUT;
    if (!(in->heights[k] > previous)) return PHX_MC_INVALID_ARGUMENT;
    previous = in->heights[k];
  }
  for (int64_t k = 0; k < 3 * in->triangle_count; ++k) {
    if (in->triangles[k] < 0 || in->triangles[k] >= in->vertex_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (in->source_normals != nullptr) {
    // A corner frame is a unit normal, or a zero normal and zero S (none).
    for (int64_t c = 0; c < 3 * in->triangle_count; ++c) {
      const double* g = in->source_normals + 3 * c;
      const double* s = in->source_shape_operators + 6 * c;
      double length = 0.0, curvature = 0.0;
      for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(g[k])) return PHX_MC_NONFINITE_INPUT;
        length += g[k] * g[k];
      }
      for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(s[k])) return PHX_MC_NONFINITE_INPUT;
        curvature = std::max(curvature, std::fabs(s[k]));
      }
      if (length == 0.0 ? curvature != 0.0 : !(std::fabs(length - 1.0) <= 2e-10)) {
        return PHX_MC_INVALID_ARGUMENT;
      }
    }
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_prism_layers(const phx_mc_prism_layers_request* request,
                            const phx_mc_prism_layers_result* result) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const int32_t status = validate(request, result);
    if (status != PHX_MC_OK) return status;
    const phx::mc::prism_layers::Input input{
        request->vertex_count,  request->vertices,      request->triangle_count,
        request->triangles,     request->layer_count,   request->heights,
        request->feature_angle_degrees, request->minimum_scale, request->prove,
        request->source_normals, request->source_shape_operators};
    const phx::mc::prism_layers::Output output{result->points, result->prisms,
                                               result->displacement, result->scale,
                                               result->vertex_kind, result->stats};
    return phx::mc::prism_layers::grow(input, output);
  });
}

}  // extern "C"
