//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C entry of hex-dominant meshing: argument validation, then hex_dominant.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "hex_dominant.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

struct phx_mc_hex_dominant_mesh {
  phx::mc::hex_dominant::Mesh mesh;
};

namespace {

int32_t validate(const phx_mc_hex_dominant_request* in) {
  if (in == nullptr || in->result == nullptr || in->vertex_count < 4 || in->triangle_count < 4 ||
      in->vertices == nullptr || in->triangles == nullptr || in->triangle_facets == nullptr ||
      in->layer_count < 0 || (in->layer_count > 0 && in->layer_thicknesses == nullptr) ||
      (in->prove != 0 && in->prove != 1) ||
      !(in->feature_angle_degrees > 0.0 && in->feature_angle_degrees < 180.0) ||
      !(in->boundary_size > 0.0) || !(in->core_size >= in->boundary_size) ||
      !std::isfinite(in->core_size)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  // core_size = boundary_size * 2^k exactly.
  int levels = 0;
  while (std::ldexp(in->boundary_size, levels) < in->core_size && levels < 30) ++levels;
  if (std::ldexp(in->boundary_size, levels) != in->core_size) return PHX_MC_INVALID_ARGUMENT;
  if (in->vertex_count > std::numeric_limits<int32_t>::max() ||
      in->triangle_count > std::numeric_limits<int32_t>::max() / 3 ||
      !phx::mc::addressable(in->vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(in->triangle_count, 3, sizeof(int32_t))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  for (int64_t k = 0; k < 3 * in->vertex_count; ++k) {
    if (!std::isfinite(in->vertices[k])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(in->vertices[k])) return PHX_MC_RANGE_ERROR;
  }
  for (int32_t k = 0; k < in->layer_count; ++k) {
    if (!std::isfinite(in->layer_thicknesses[k])) return PHX_MC_NONFINITE_INPUT;
    if (!(in->layer_thicknesses[k] > 0.0)) return PHX_MC_INVALID_ARGUMENT;
  }
  std::unique_ptr<uint8_t[]> used(new uint8_t[static_cast<std::size_t>(in->vertex_count)]());
  for (int64_t k = 0; k < 3 * in->triangle_count; ++k) {
    if (in->triangles[k] < 0 || in->triangles[k] >= in->vertex_count) return PHX_MC_INVALID_INPUT;
    used[in->triangles[k]] = 1;
  }
  for (int64_t v = 0; v < in->vertex_count; ++v) {
    if (!used[v]) return PHX_MC_INVALID_INPUT;
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_hex_dominant(const phx_mc_hex_dominant_request* request) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (request != nullptr && request->result != nullptr) *request->result = nullptr;
    const int32_t status = validate(request);
    if (status != PHX_MC_OK) return status;
    auto result = std::make_unique<phx_mc_hex_dominant_mesh>();
    const phx::mc::hex_dominant::Input input{
        request->vertex_count,   request->vertices,         request->triangle_count,
        request->triangles,      request->triangle_facets,  request->core_size,
        request->boundary_size,  request->feature_angle_degrees, request->layer_count,
        request->layer_thicknesses, request->prove};
    const int32_t built = phx::mc::hex_dominant::build(input, result->mesh);
    if (built == PHX_MC_OK || built == PHX_MC_UNRESOLVED) *request->result = result.release();
    return built;
  });
}

int32_t phx_mc_hex_dominant_sizes(const phx_mc_hex_dominant_mesh* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& mesh = result->mesh;
  sizes[0] = static_cast<int64_t>(mesh.points.size() / 3);
  sizes[1] = static_cast<int64_t>(mesh.face_owner.size());
  sizes[2] = static_cast<int64_t>(mesh.face_vertices.size());
  sizes[3] = static_cast<int64_t>(mesh.cell_kinds.size());
  sizes[4] = static_cast<int64_t>(mesh.cell_vertices.size());
  return PHX_MC_OK;
}

int32_t phx_mc_hex_dominant_export(const phx_mc_hex_dominant_mesh* result,
                                   const phx_mc_hex_dominant_arrays* arrays) {
  if (result == nullptr || arrays == nullptr || arrays->points == nullptr ||
      arrays->face_offsets == nullptr || arrays->face_vertices == nullptr ||
      arrays->face_owner == nullptr || arrays->face_neighbor == nullptr ||
      arrays->face_facets == nullptr || arrays->cell_kinds == nullptr ||
      arrays->cell_roles == nullptr || arrays->cell_vertex_offsets == nullptr ||
      (!result->mesh.cell_vertices.empty() && arrays->cell_vertices == nullptr) ||
      arrays->vertex_triangles == nullptr || arrays->vertex_kinds == nullptr ||
      arrays->stats == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& mesh = result->mesh;
  std::copy(mesh.points.begin(), mesh.points.end(), arrays->points);
  std::copy(mesh.face_offsets.begin(), mesh.face_offsets.end(), arrays->face_offsets);
  std::copy(mesh.face_vertices.begin(), mesh.face_vertices.end(), arrays->face_vertices);
  std::copy(mesh.face_owner.begin(), mesh.face_owner.end(), arrays->face_owner);
  std::copy(mesh.face_neighbor.begin(), mesh.face_neighbor.end(), arrays->face_neighbor);
  std::copy(mesh.face_facets.begin(), mesh.face_facets.end(), arrays->face_facets);
  std::copy(mesh.cell_kinds.begin(), mesh.cell_kinds.end(), arrays->cell_kinds);
  std::copy(mesh.cell_roles.begin(), mesh.cell_roles.end(), arrays->cell_roles);
  std::copy(mesh.cell_vertex_offsets.begin(), mesh.cell_vertex_offsets.end(), arrays->cell_vertex_offsets);
  std::copy(mesh.cell_vertices.begin(), mesh.cell_vertices.end(), arrays->cell_vertices);
  std::copy(mesh.vertex_triangles.begin(), mesh.vertex_triangles.end(), arrays->vertex_triangles);
  std::copy(mesh.vertex_kinds.begin(), mesh.vertex_kinds.end(), arrays->vertex_kinds);
  std::copy(std::begin(mesh.stats), std::end(mesh.stats), arrays->stats);
  return PHX_MC_OK;
}

int32_t phx_mc_hex_dominant_certify(const phx_mc_hex_dominant_mesh* result,
                                    int64_t* failed_cells) {
  if (result == nullptr || failed_cells == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    *failed_cells = phx::mc::hex_dominant::uncertified_cells(result->mesh);
    return *failed_cells == 0 ? PHX_MC_OK : PHX_MC_UNRESOLVED;
  });
}

void phx_mc_hex_dominant_free(phx_mc_hex_dominant_mesh* result) { delete result; }

}  // extern "C"
