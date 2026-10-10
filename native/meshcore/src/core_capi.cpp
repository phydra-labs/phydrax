//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: library identity, batched predicates, and mesh handle accessors.
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

#ifndef PHX_MC_VERSION
#error "PHX_MC_VERSION must be defined by the build."
#endif
#ifndef PHX_MC_BUILD_HASH
#error "PHX_MC_BUILD_HASH must be defined by the build."
#endif
#ifndef PHX_MC_BUILD_CONFIGURATION
#error "PHX_MC_BUILD_CONFIGURATION must be defined by the build."
#endif
#ifndef PHX_MC_ABI_CONTRACT
#error "PHX_MC_ABI_CONTRACT must be defined by the build."
#endif

namespace {

using phx::mc::addressable;
using phx::mc::coordinate_in_domain;
using phx::mc::guarded;

int32_t validate_coordinates(int64_t count, int width, const double* const* arrays, int arity) {
  for (int k = 0; k < arity; ++k) {
    for (int64_t index = 0; index < count * width; ++index) {
      phx::mc::native_execution_charge(0);
      if (!std::isfinite(arrays[k][index])) {
        return PHX_MC_NONFINITE_INPUT;
      }
    }
  }
  for (int k = 0; k < arity; ++k) {
    for (int64_t index = 0; index < count * width; ++index) {
      phx::mc::native_execution_charge(0);
      if (!coordinate_in_domain(arrays[k][index])) {
        return PHX_MC_RANGE_ERROR;
      }
    }
  }
  return PHX_MC_OK;
}

// Rejects a negative or unaddressable count before any `count * width` offset.
int32_t validate_batch(int64_t count, int width, const double* const* arrays, int arity,
                       const void* output) {
  if (count < 0 || output == nullptr || !addressable(count, width, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int k = 0; k < arity; ++k) {
    if (count > 0 && arrays[k] == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  return validate_coordinates(count, width, arrays, arity);
}

int32_t validate_ids(int64_t count, int arity, const int64_t* ids) {
  if ((count > 0 && ids == nullptr) || !addressable(count, arity, sizeof(int64_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t row = 0; row < count; ++row) {
    phx::mc::native_execution_charge(0);
    const int64_t* r = ids + row * arity;
    for (int i = 0; i < arity; ++i) {
      for (int j = i + 1; j < arity; ++j) {
        if (r[i] == r[j]) {
          return PHX_MC_INVALID_INPUT;
        }
      }
    }
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

const char* phx_mc_version(void) { return PHX_MC_VERSION; }

const char* phx_mc_abi_contract(void) { return PHX_MC_ABI_CONTRACT; }

const char* phx_mc_build_hash(void) { return PHX_MC_BUILD_HASH; }
const char* phx_mc_build_configuration(void) { return PHX_MC_BUILD_CONFIGURATION; }

void phx_mc_exact_domain(int32_t* min_exponent, int32_t* max_exponent) {
  if (min_exponent != nullptr) {
    *min_exponent = phx::mc::kCoordinateMinExponent;
  }
  if (max_exponent != nullptr) {
    *max_exponent = phx::mc::kCoordinateMaxExponent;
  }
}

int32_t phx_mc_orient2d(int64_t count, const double* a, const double* b, const double* c,
                        int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[3] = {a, b, c};
    const int32_t status = validate_batch(count, 2, arrays, 3, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(phx::mc::orient2d(a + 2 * i, b + 2 * i, c + 2 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_orient3d(int64_t count, const double* a, const double* b, const double* c,
                        const double* d, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    const int32_t status = validate_batch(count, 3, arrays, 4, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(
          phx::mc::orient3d(a + 3 * i, b + 3 * i, c + 3 * i, d + 3 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_incircle(int64_t count, const double* a, const double* b, const double* c,
                        const double* d, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    const int32_t status = validate_batch(count, 2, arrays, 4, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(
          phx::mc::incircle(a + 2 * i, b + 2 * i, c + 2 * i, d + 2 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_insphere(int64_t count, const double* a, const double* b, const double* c,
                        const double* d, const double* e, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[5] = {a, b, c, d, e};
    const int32_t status = validate_batch(count, 3, arrays, 5, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(
          phx::mc::insphere(a + 3 * i, b + 3 * i, c + 3 * i, d + 3 * i, e + 3 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_orient2d_sos(int64_t count, const double* a, const double* b, const double* c,
                            const int64_t* ids, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[3] = {a, b, c};
    int32_t status = validate_batch(count, 2, arrays, 3, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 3, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 3 * i;
      signs[i] = static_cast<int8_t>(
          phx::mc::orient2d_sos(a + 2 * i, b + 2 * i, c + 2 * i, r[0], r[1], r[2]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_orient3d_sos(int64_t count, const double* a, const double* b, const double* c,
                            const double* d, const int64_t* ids, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    int32_t status = validate_batch(count, 3, arrays, 4, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 4, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 4 * i;
      signs[i] = static_cast<int8_t>(phx::mc::orient3d_sos(
          a + 3 * i, b + 3 * i, c + 3 * i, d + 3 * i, r[0], r[1], r[2], r[3]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_incircle_sos(int64_t count, const double* a, const double* b, const double* c,
                            const double* d, const int64_t* ids, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    int32_t status = validate_batch(count, 2, arrays, 4, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 4, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 4 * i;
      signs[i] = static_cast<int8_t>(phx::mc::incircle_sos(
          a + 2 * i, b + 2 * i, c + 2 * i, d + 2 * i, r[0], r[1], r[2], r[3]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_insphere_sos(int64_t count, const double* a, const double* b, const double* c,
                            const double* d, const double* e, const int64_t* ids,
                            int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[5] = {a, b, c, d, e};
    int32_t status = validate_batch(count, 3, arrays, 5, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 5, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 5 * i;
      signs[i] = static_cast<int8_t>(phx::mc::insphere_sos(a + 3 * i, b + 3 * i, c + 3 * i,
                                                           d + 3 * i, e + 3 * i, r[0], r[1],
                                                           r[2], r[3], r[4]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_dimension(const phx_mc_mesh* mesh) { return mesh->dimension; }

int64_t phx_mc_mesh_point_count(const phx_mc_mesh* mesh) { return mesh->point_count(); }

int64_t phx_mc_mesh_cell_count(const phx_mc_mesh* mesh) { return mesh->cell_count(); }

int64_t phx_mc_mesh_input_point_count(const phx_mc_mesh* mesh) {
  return mesh->input_point_count;
}

void phx_mc_mesh_copy_points(const phx_mc_mesh* mesh, double* points) {
  std::copy(mesh->points.begin(), mesh->points.end(), points);
}

void phx_mc_mesh_copy_cells(const phx_mc_mesh* mesh, int32_t* cells) {
  std::copy(mesh->cells.begin(), mesh->cells.end(), cells);
}

void phx_mc_mesh_copy_vertex_map(const phx_mc_mesh* mesh, int32_t* vertex_map) {
  std::copy(mesh->vertex_map.begin(), mesh->vertex_map.end(), vertex_map);
}

void phx_mc_mesh_copy_cell_constraints(const phx_mc_mesh* mesh, int32_t* cell_constraints) {
  if (mesh->cell_constraints.empty()) {
    std::fill_n(cell_constraints, mesh->cell_count() * (mesh->dimension + 1), -1);
  } else {
    std::copy(mesh->cell_constraints.begin(), mesh->cell_constraints.end(), cell_constraints);
  }
}

void phx_mc_mesh_copy_cell_regions(const phx_mc_mesh* mesh, int32_t* cell_regions) {
  if (mesh->cell_regions.empty()) {
    std::fill_n(cell_regions, mesh->cell_count(), -1);
  } else {
    std::copy(mesh->cell_regions.begin(), mesh->cell_regions.end(), cell_regions);
  }
}

void phx_mc_mesh_free(phx_mc_mesh* mesh) { phx::mc::destroy_native_object(mesh); }

}  // extern "C"
