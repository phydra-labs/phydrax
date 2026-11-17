//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the metric edit goals. Every entry establishes
// round-to-nearest with gradual underflow for its duration.
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "metric_goal.hpp"
#include "phydrax_meshcore.h"

#include <cstddef>
#include <cstdint>

using phx::mc::addressable;

extern "C" {

int32_t phx_mc_metric_midpoints(int64_t count, const double* first_points,
                                const double* second_points, const double* first_metrics,
                                const double* second_metrics, double* parameters,
                                double* points) {
  if (!addressable(count, 9, sizeof(double)) ||
      (count > 0 && (first_points == nullptr || second_points == nullptr ||
                     first_metrics == nullptr || second_metrics == nullptr ||
                     parameters == nullptr || points == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) {
    const double* first = first_points + 3 * row;
    const double* second = second_points + 3 * row;
    const double parameter = phx::mc::metric::midpoint_parameter(
        first, second, first_metrics + 9 * row, second_metrics + 9 * row);
    parameters[row] = parameter;
    for (int64_t axis = 0; axis < 3; ++axis) {
      points[3 * row + axis] = first[axis] + parameter * (second[axis] - first[axis]);
    }
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_survivor_goals(int64_t point_count, const double* points,
                                     const double* metrics, int32_t corner_count,
                                     const double* corners, double* goals, int8_t* valid) {
  if (!addressable(point_count, 3, sizeof(double)) || point_count < 2 || points == nullptr ||
      metrics == nullptr || goals == nullptr || valid == nullptr ||
      !(corner_count == 0 || corner_count == 2 || corner_count == 3) ||
      (corner_count > 0 && corners == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  return phx::mc::guarded([&]() -> int32_t {
    double result[9];
    bool admitted[3];
    phx::mc::metric::survivor_goals(point_count, points, metrics, corner_count, corners, result,
                                    admitted);
    for (std::size_t index = 0; index < 9; ++index) goals[index] = result[index];
    for (std::size_t goal = 0; goal < 3; ++goal) valid[goal] = admitted[goal] ? 1 : 0;
    return PHX_MC_OK;
  });
}

int32_t phx_mc_metric_hull_frame(int32_t corner_count, const double* corners, double* basis,
                                 double* projector, int32_t* rank) {
  if (!(corner_count == 2 || corner_count == 3) || corners == nullptr || basis == nullptr ||
      projector == nullptr || rank == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  double vectors[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double result[9];
  *rank = phx::mc::metric::hull_frame(corner_count, corners, vectors, result);
  for (std::size_t index = 0; index < 6; ++index) basis[index] = vectors[index];
  for (std::size_t index = 0; index < 9; ++index) projector[index] = result[index];
  return PHX_MC_OK;
}

int32_t phx_mc_metric_project_goals(int64_t count, const double* goals, const double* origin,
                                    const double* projector, double* projected) {
  if (!addressable(count, 3, sizeof(double)) || origin == nullptr || projector == nullptr ||
      (count > 0 && (goals == nullptr || projected == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  phx::mc::metric::project_goals(count, goals, origin, projector, projected);
  return PHX_MC_OK;
}

int32_t phx_mc_metric_relocation_pulls(int64_t neighbor_count, const double* points,
                                       const double* metric, const double* lengths,
                                       const int8_t* measured, int32_t deep, int32_t interior,
                                       int32_t basis_count, const double* basis,
                                       const double* projector, const double* jacobian,
                                       int64_t capacity, double* pulls, int64_t* count) {
  if (!addressable(neighbor_count, 3, sizeof(double)) || neighbor_count < 1 ||
      !addressable(capacity, 3, sizeof(double)) || points == nullptr || metric == nullptr ||
      lengths == nullptr || measured == nullptr || count == nullptr ||
      (capacity > 0 && pulls == nullptr) || basis_count < 0 || basis_count > 3 ||
      (jacobian != nullptr && (basis_count == 0 || basis == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  return phx::mc::guarded([&]() -> int32_t {
    const std::int64_t written = phx::mc::metric::relocation_pulls(
        neighbor_count, points, metric, lengths, measured, deep != 0, interior != 0, basis_count,
        basis, projector, jacobian, capacity, pulls);
    if (written < 0) return PHX_MC_CAPACITY_EXCEEDED;
    *count = written;
    return PHX_MC_OK;
  });
}

}  // extern "C"
