//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the metric measurements. Every entry establishes
// round-to-nearest with gradual underflow for its duration, validates every
// index before it writes, and evaluates its rows independently, in order.
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "metric_float.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace {

using phx::mc::addressable;
using phx::mc::ChunkContext;
using phx::mc::GradualNearestScope;
using phx::mc::ParallelRequest;

bool indices_in_range(const int64_t* indices, int64_t count, int64_t bound) noexcept {
  for (int64_t index = 0; index < count; ++index) {
    if (indices[index] < 0 || indices[index] >= bound) return false;
  }
  return true;
}

double indexed_tetra_quality(const int64_t* cell, const double* points,
                             const double* metrics, const double* logs) noexcept {
  double corners[12];
  double tensors[36];
  double cell_logs[4];
  for (int corner = 0; corner < 4; ++corner) {
    const int64_t row = cell[corner];
    std::copy_n(points + 3 * row, 3, corners + 3 * corner);
    std::copy_n(metrics + 9 * row, 9, tensors + 9 * corner);
    if (logs != nullptr) cell_logs[corner] = logs[row];
  }
  return logs == nullptr ? phx::mc::metric::tetra_quality(corners, tensors)
                         : phx::mc::metric::tetra_quality(corners, tensors, cell_logs);
}

template <class Measure>
int32_t nearest_ranges(std::size_t count, ParallelRequest request, const Measure& measure) {
  const auto run = [&](std::size_t begin, std::size_t end, ChunkContext& chunk) -> int32_t {
    const GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    measure(begin, end, chunk);
    return PHX_MC_OK;
  };
  if (request.parallelism == 1 || count <= request.chunk_size) {
    ChunkContext chunk{0};
    return run(0, count, chunk);
  }
  return phx::mc::parallel_for(count, request, run);
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
    quality[cell] = indexed_tetra_quality(cells + 4 * cell, points, metrics, nullptr);
  }
  return PHX_MC_OK;
}

int32_t phx_mc_metric_tetra_quality_min(int64_t cell_count, const int64_t* cells,
                                        int64_t point_count, const double* points,
                                        const double* metrics, int32_t parallelism,
                                        double* minimum) {
  if (cell_count <= 0 || parallelism < 1 || minimum == nullptr || cells == nullptr ||
      !addressable(cell_count, 4, sizeof(int64_t)) ||
      !addressable(point_count, 9, sizeof(double)) ||
      (point_count > 0 && (points == nullptr || metrics == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!indices_in_range(cells, 4 * cell_count, point_count)) return PHX_MC_INVALID_INPUT;
  return phx::mc::guarded([&]() -> int32_t {
    constexpr std::size_t chunk_size = 1024;
    const ParallelRequest request{chunk_size, static_cast<uint32_t>(parallelism)};
    // Reuse the exact per-point term on dense incidence. A smaller cell
    // selection does not justify a point-wide preparation pass.
    const auto log_count =
        point_count <= cell_count ? static_cast<std::size_t>(point_count) : 0;
    auto logs = log_count ? std::make_unique_for_overwrite<double[]>(log_count) : nullptr;
    if (logs != nullptr) {
      const int32_t status = nearest_ranges(
          log_count, request, [&](std::size_t begin, std::size_t end, const ChunkContext&) {
            for (std::size_t row = begin; row < end; ++row) {
              logs[row] = phx::mc::metric::log_determinant(metrics + 9 * row);
            }
          });
      if (status != PHX_MC_OK) return status;
    }
    const auto count = static_cast<std::size_t>(cell_count);
    const std::size_t chunks = parallelism == 1 ? 1 : (count - 1) / chunk_size + 1;
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double local;
    auto others = chunks > 1 ? std::make_unique_for_overwrite<double[]>(chunks) : nullptr;
    double* const partial = others != nullptr ? others.get() : &local;
    const int32_t status = nearest_ranges(
        count, request, [&](std::size_t begin, std::size_t end, const ChunkContext& chunk) {
          double least = infinity;
          for (std::size_t row = begin; row < end; ++row) {
            const double quality = indexed_tetra_quality(cells + 4 * row, points, metrics, logs.get());
            if (!std::isfinite(quality)) {
              least = nan;
              break;
            }
            least = std::min(least, quality);
          }
          partial[chunk.chunk_index] = least;
        });
    if (status != PHX_MC_OK) return status;
    double least = infinity;
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      const double value = partial[chunk];
      if (!std::isfinite(value)) {
        least = nan;
        break;
      }
      least = std::min(least, value);
    }
    *minimum = least;
    return PHX_MC_OK;
  });
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
