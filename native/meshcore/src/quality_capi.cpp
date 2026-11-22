//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the host cell-quality kernels: the rounded binary64
// operations of phydrax's host quality formulas (meshing/_quality.py), in
// their order, so every value is the same double. Every entry establishes
// round-to-nearest with gradual underflow for its duration; cells are
// independent and evaluated in order.
#include "angle.hpp"
#include "capi_guard.hpp"
#include "deterministic_float.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using phx::mc::addressable;

constexpr double kTiny = std::numeric_limits<double>::min();

// numpy.maximum(value, kTiny): NaN propagates.
double at_least_tiny(double value) noexcept {
  return std::isnan(value) ? value : std::max(value, kTiny);
}

// Cofactor expansion along the first row, (a - b) + c.
double determinant(const double matrix[3][3]) noexcept {
  return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
         matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
         matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

bool cell_arguments(int64_t cell_count, int64_t corner_count, const double* points,
                    int64_t row_count, int64_t row_width, const int64_t* rows) noexcept {
  if (corner_count <= 0 || corner_count > std::numeric_limits<int64_t>::max() / 3 ||
      !addressable(cell_count, corner_count * 3, sizeof(double)) ||
      !addressable(cell_count, row_count, sizeof(double)) ||
      !addressable(row_count, row_width, sizeof(int64_t)) ||
      (cell_count > 0 && points == nullptr) || (row_count > 0 && rows == nullptr)) {
    return false;
  }
  for (int64_t index = 0; index < row_count * row_width; ++index) {
    if (rows[index] < 0 || rows[index] >= corner_count) return false;
  }
  return true;
}

}  // namespace

extern "C" {

int32_t phx_mc_quality_dihedral_angles(int64_t cell_count, int64_t corner_count,
                                       const double* points, int64_t sample_count,
                                       const int64_t* samples, double* angles) {
  if (!cell_arguments(cell_count, corner_count, points, sample_count, 4, samples) ||
      (cell_count > 0 && sample_count > 0 && angles == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  constexpr double kTwoPi = 2.0 * phx::mc::angle::kPi;
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    const double* corners = points + cell * corner_count * 3;
    for (int64_t sample = 0; sample < sample_count; ++sample) {
      const int64_t* row = samples + sample * 4;
      const double* origin = corners + row[0] * 3;
      double edge[3];
      double first[3];
      double second[3];
      for (int axis = 0; axis < 3; ++axis) {
        edge[axis] = corners[row[1] * 3 + axis] - origin[axis];
        first[axis] = corners[row[2] * 3 + axis] - origin[axis];
        second[axis] = corners[row[3] * 3 + axis] - origin[axis];
      }
      const double scale = at_least_tiny(
          std::sqrt(edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]));
      const double unit[3] = {edge[0] / scale, edge[1] / scale, edge[2] / scale};
      // Each sum starts from zero, so a lone -0 term sums to +0.
      const double sine = ((0.0 + (first[1] * second[2] - first[2] * second[1]) * unit[0]) +
                           (first[2] * second[0] - first[0] * second[2]) * unit[1]) +
                          (first[0] * second[1] - first[1] * second[0]) * unit[2];
      const double dot =
          ((0.0 + first[0] * second[0]) + first[1] * second[1]) + first[2] * second[2];
      const double first_along =
          ((0.0 + first[0] * unit[0]) + first[1] * unit[1]) + first[2] * unit[2];
      const double second_along =
          ((0.0 + second[0] * unit[0]) + second[1] * unit[1]) + second[2] * unit[2];
      const double angle = phx::mc::angle::arctangent2(sine, dot - first_along * second_along);
      angles[cell * sample_count + sample] = angle < 0.0 ? angle + kTwoPi : angle;
    }
  }
  return PHX_MC_OK;
}

int32_t phx_mc_quality_solid_corners(int64_t cell_count, int64_t corner_count,
                                     const double* points, int64_t row_count,
                                     const int64_t* frames, const double* ideal_inverse,
                                     const double* ideal_scaled, double* determinants,
                                     double* scaled, double* ratios, double* conditions) {
  if (!cell_arguments(cell_count, corner_count, points, row_count, 4, frames) ||
      (row_count > 0 && (ideal_inverse == nullptr || ideal_scaled == nullptr)) ||
      (cell_count > 0 && row_count > 0 &&
       (determinants == nullptr || scaled == nullptr || ratios == nullptr ||
        conditions == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    const double* corners = points + cell * corner_count * 3;
    for (int64_t row = 0; row < row_count; ++row) {
      const int64_t* frame_row = frames + row * 4;
      const double* apex = corners + frame_row[0] * 3;
      const double* inverse = ideal_inverse + row * 9;
      // frame[a][d]: component a of the edge to neighbor d.
      double frame[3][3];
      for (int side = 0; side < 3; ++side) {
        for (int axis = 0; axis < 3; ++axis) {
          frame[axis][side] = corners[frame_row[side + 1] * 3 + axis] - apex[axis];
        }
      }
      const double signed_volume = determinant(frame);
      double lengths[3];
      for (int side = 0; side < 3; ++side) {
        lengths[side] = std::sqrt(frame[0][side] * frame[0][side] +
                                  frame[1][side] * frame[1][side] +
                                  frame[2][side] * frame[2][side]);
      }
      double target[3][3];
      for (int axis = 0; axis < 3; ++axis) {
        for (int column = 0; column < 3; ++column) {
          target[axis][column] = frame[axis][0] * inverse[column] +
                                 frame[axis][1] * inverse[3 + column] +
                                 frame[axis][2] * inverse[6 + column];
        }
      }
      double gram[3][3];
      for (int first = 0; first < 3; ++first) {
        for (int second = 0; second < 3; ++second) {
          gram[first][second] = target[0][first] * target[0][second] +
                                target[1][first] * target[1][second] +
                                target[2][first] * target[2][second];
        }
      }
      const double gram_determinant = at_least_tiny(determinant(gram));
      const double trace = ((0.0 + gram[0][0]) + gram[1][1]) + gram[2][2];
      const double adjugate_trace = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0] +
                                    gram[0][0] * gram[2][2] - gram[0][2] * gram[2][0] +
                                    gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1];
      const bool positive = signed_volume > 0.0;
      const int64_t out = cell * row_count + row;
      determinants[out] = signed_volume;
      scaled[out] =
          signed_volume / at_least_tiny(1.0 * lengths[0] * lengths[1] * lengths[2]) /
          ideal_scaled[row];
      ratios[out] = positive ? 3.0 * phx::mc::deterministic::cube_root(gram_determinant) /
                                   at_least_tiny(trace)
                             : 0.0;
      conditions[out] =
          positive ? std::sqrt(trace * adjugate_trace / gram_determinant) / 3.0 : kInfinity;
    }
  }
  return PHX_MC_OK;
}

}  // extern "C"
