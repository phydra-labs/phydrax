//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Delaunay triangulation and refinement of translationally periodic point sets
// on the flat torus R^d / L. Image neighborhoods grow until their quotient
// topology is certified by circum-ball sufficiency and facet closure.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "phydrax_meshcore.h"

namespace phx::mc {

struct PeriodicDelaunayInput {
  int dimension = 0;
  int64_t point_count = 0;
  const double* points = nullptr;      // (n, d) Cartesian representatives
  const double* fractional = nullptr;  // (n, d) lattice coordinates
  const double* lattice = nullptr;     // (d, d) rows are lattice vectors
  const double* inverse = nullptr;     // (d, d) with fractional = x @ inverse
  double initial_margin = 0.0;
};

struct PeriodicDelaunayResult {
  int dimension = 0;
  std::vector<int32_t> vertices;  // (T, d + 1) representatives
  std::vector<int32_t> shifts;    // (T, d + 1, d) image shifts of anchored lifts
  double certified_margin = 0.0;
};

struct PeriodicLocalDelaunayInput {
  PeriodicDelaunayInput triangulation;
  const int64_t* point_global_ids = nullptr;
  const int32_t* point_owner_ranks = nullptr;
  int32_t local_owner_rank = -1;
};

struct PeriodicLocalDelaunayResult {
  int dimension = 0;
  std::vector<int32_t> vertices;  // (T, d + 1) rows into the supplied local point bank
  std::vector<int32_t> shifts;    // (T, d + 1, d) absolute source image windings
  std::vector<int64_t> cell_ids;  // deterministic 63-bit hashes of canonical cell keys
  std::vector<int64_t> cell_key_ids;
  std::vector<int32_t> cell_key_relative_shifts;
  std::vector<int32_t> cell_owner_ranks;
  std::vector<int8_t> support_cell_mask;
  std::vector<double> support_fractional_lower;
  std::vector<double> support_fractional_upper;
  std::int8_t has_owned_sites = 0;
  std::array<double, 3> owner_fractional_lower{};
  std::array<double, 3> owner_fractional_upper{};
  double certified_margin = 0.0;
};

struct PeriodicSupportQueryInput {
  int dimension = 0;
  int64_t cell_count = 0;
  const double* cell_vertex_points = nullptr;
  const int64_t* cell_vertex_global_ids = nullptr;
  const int32_t* cell_vertex_shifts = nullptr;
  int64_t point_count = 0;
  const double* points = nullptr;
  const int64_t* point_global_ids = nullptr;
  const double* lattice = nullptr;
  uint64_t candidate_revision = 0;
};

struct PeriodicSupportQueryResult {
  int dimension = 0;
  std::vector<int64_t> cell_rows;
  std::vector<int64_t> point_global_ids;
  std::vector<int32_t> image_shifts;
  std::vector<int8_t> predicate_signs;
  uint64_t candidate_revision = 0;
};

int32_t periodic_delaunay_local(const PeriodicLocalDelaunayInput& input,
                                PeriodicLocalDelaunayResult& result);
int32_t periodic_owner_support_query(const PeriodicSupportQueryInput& input,
                                     PeriodicSupportQueryResult& result);

int32_t periodic_delaunay(const PeriodicDelaunayInput& input,
                          PeriodicDelaunayResult& result);

struct PeriodicRefinementResult {
  std::vector<double> points;
  std::vector<int32_t> vertices;
  std::vector<int32_t> shifts;
};

int32_t periodic_refine_2d(const phx_mc_periodic_refinement_input& input,
                           PeriodicRefinementResult& result);

}  // namespace phx::mc
