//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the metric measurements. Every entry establishes
// round-to-nearest with gradual underflow for its duration, validates every
// index before it writes, and evaluates its rows independently, in order.
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "metric_float.hpp"
#include "phydrax_meshcore.h"

#include <cstddef>
#include <cstdint>

namespace {

using phx::mc::addressable;

bool indices_in_range(const int64_t* indices, int64_t count, int64_t bound) noexcept {
  for (int64_t index = 0; index < count; ++index) {
    if (indices[index] < 0 || indices[index] >= bound) return false;
  }
  return true;
}

}  // namespace

extern "C" {

int32_t phx_mc_metric_logarithms(int64_t count, const double* metrics, double* logarithms,
                                 int8_t* valid) {
  if (!addressable(count, 9, sizeof(double)) ||
      (count > 0 && (metrics == nullptr || logarithms == nullptr || valid == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) {
    double result[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    valid[row] = phx::mc::metric::logarithm(metrics + 9 * row, result) ? 1 : 0;
    for (std::size_t index = 0; index < 9; ++index) {
      logarithms[9 * row + static_cast<int64_t>(index)] = result[index];
    }
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_interpolate(int64_t row_count, const int64_t* offsets,
                                  const int64_t* support, const double* weights,
                                  int64_t source_count, const double* source_metrics,
                                  const double* source_logarithms, double* metrics,
                                  int8_t* valid) {
  if (!addressable(row_count, 9, sizeof(double)) ||
      !addressable(source_count, 9, sizeof(double)) || offsets == nullptr ||
      (row_count > 0 && (metrics == nullptr || valid == nullptr)) ||
      (source_count > 0 && source_metrics == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (offsets[0] != 0) return PHX_MC_INVALID_INPUT;
  for (int64_t row = 0; row < row_count; ++row) {
    // Every row has a support: the mean of no tensor is not defined.
    if (offsets[row + 1] <= offsets[row]) return PHX_MC_INVALID_INPUT;
  }
  const int64_t entries = offsets[row_count];
  if (!addressable(entries, 1, sizeof(double)) ||
      (entries > 0 && (support == nullptr || weights == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!indices_in_range(support, entries, source_count)) return PHX_MC_INVALID_INPUT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  // The gathered support of one row; its capacity grows to the widest row.
  int64_t widest = 0;
  for (int64_t row = 0; row < row_count; ++row) {
    const int64_t width = offsets[row + 1] - offsets[row];
    if (width > widest) widest = width;
  }
  return phx::mc::guarded([&]() -> int32_t {
    std::vector<double> gathered(static_cast<std::size_t>(widest) * 9);
    std::vector<double> gathered_logarithms(
        source_logarithms == nullptr ? 0 : static_cast<std::size_t>(widest) * 9);
    for (int64_t row = 0; row < row_count; ++row) {
      const int64_t begin = offsets[row];
      const int64_t width = offsets[row + 1] - begin;
      for (int64_t k = 0; k < width; ++k) {
        const int64_t source = support[begin + k];
        for (int64_t index = 0; index < 9; ++index) {
          gathered[static_cast<std::size_t>(9 * k + index)] = source_metrics[9 * source + index];
          if (source_logarithms != nullptr) {
            gathered_logarithms[static_cast<std::size_t>(9 * k + index)] =
                source_logarithms[9 * source + index];
          }
        }
      }
      double result[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      valid[row] = phx::mc::metric::interpolate(
                       width, weights + begin, gathered.data(),
                       source_logarithms == nullptr ? nullptr : gathered_logarithms.data(),
                       result)
                       ? 1
                       : 0;
      for (int64_t index = 0; index < 9; ++index) metrics[9 * row + index] = result[index];
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_metric_lengths(int64_t edge_count, const int64_t* edges, int64_t point_count,
                              const double* points, const double* metrics, double* lengths) {
  if (!addressable(edge_count, 2, sizeof(int64_t)) ||
      !addressable(point_count, 9, sizeof(double)) ||
      (edge_count > 0 && (edges == nullptr || lengths == nullptr)) ||
      (point_count > 0 && (points == nullptr || metrics == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!indices_in_range(edges, 2 * edge_count, point_count)) return PHX_MC_INVALID_INPUT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t edge = 0; edge < edge_count; ++edge) {
    const int64_t first = edges[2 * edge];
    const int64_t second = edges[2 * edge + 1];
    lengths[edge] = phx::mc::metric::length(points + 3 * first, points + 3 * second,
                                            metrics + 9 * first, metrics + 9 * second);
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_band_quanta(int64_t count, const double* lengths, double* above,
                                  double* below) {
  if (!addressable(count, 1, sizeof(double)) ||
      (count > 0 && (lengths == nullptr || above == nullptr || below == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) {
    phx::mc::metric::band_quanta(lengths[row], above[row], below[row]);
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_relative_determinants(int64_t cell_count, const double* corners,
                                            double* determinants, double* relative) {
  if (!addressable(cell_count, 12, sizeof(double)) ||
      (cell_count > 0 &&
       (corners == nullptr || determinants == nullptr || relative == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    phx::mc::metric::relative_determinant(corners + 12 * cell, determinants[cell],
                                          relative[cell]);
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_tetra_quality(int64_t cell_count, const int64_t* cells,
                                    int64_t point_count, const double* points,
                                    const double* metrics, double* quality) {
  if (!addressable(cell_count, 4, sizeof(int64_t)) ||
      !addressable(point_count, 9, sizeof(double)) ||
      (cell_count > 0 && (cells == nullptr || quality == nullptr)) ||
      (point_count > 0 && (points == nullptr || metrics == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!indices_in_range(cells, 4 * cell_count, point_count)) return PHX_MC_INVALID_INPUT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    double corners[12];
    double tensors[36];
    for (int64_t corner = 0; corner < 4; ++corner) {
      const int64_t point = cells[4 * cell + corner];
      for (int64_t axis = 0; axis < 3; ++axis) {
        corners[3 * corner + axis] = points[3 * point + axis];
      }
      for (int64_t index = 0; index < 9; ++index) {
        tensors[9 * corner + index] = metrics[9 * point + index];
      }
    }
    quality[cell] = phx::mc::metric::tetra_quality(corners, tensors);
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_tensor_properties(int64_t count, const double* tensors, int8_t* symmetric,
                                        int8_t* positive_definite) {
  if (!addressable(count, 9, sizeof(double)) ||
      (count > 0 &&
       (tensors == nullptr || symmetric == nullptr || positive_definite == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) {
    bool is_symmetric = false;
    bool is_positive_definite = false;
    phx::mc::metric::tensor_properties(tensors + 9 * row, is_symmetric, is_positive_definite);
    symmetric[row] = is_symmetric ? 1 : 0;
    positive_definite[row] = is_positive_definite ? 1 : 0;
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_congruence_defects(int64_t count, const double* linear,
                                         const double* metrics, const double* references,
                                         double* errors, double* scales) {
  if (!addressable(count, 9, sizeof(double)) ||
      (count > 0 && (linear == nullptr || metrics == nullptr || references == nullptr ||
                     errors == nullptr || scales == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) {
    phx::mc::metric::congruence_defect(linear + 9 * row, metrics + 9 * row, references + 9 * row,
                                       errors[row], scales[row]);
  }
  return PHX_MC_OK;
}

}  // extern "C"
