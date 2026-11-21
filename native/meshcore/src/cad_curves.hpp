// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Curve discretization is a sampled construction target, not a certificate.
// The S2-F verifier must bound curve-edge chord and trim-lens error and repair
// a failing edge consistently on every use.
// Degree-aware sampling defends polynomial spans against aliasing; the Gauss
// screen is heuristic, and neither replaces the S2-F certificate.
#pragma once

#include <cstdint>
#include <vector>

#include "cad_curve_discretization.hpp"
#include "cad_geometry.hpp"
#include "cad_proximity.hpp"

namespace phx::mc::cad {

struct CurveControls {
  const double* face_sizes;
  const double* face_tolerances;
  const double* curve_tolerances;
  double tolerance_fraction;
  double grading;
  double minimum_size;
  // Internal test control; the construction path enables the sag screen.
  bool sag_check = true;
  // Solid route only (0: off): the volume stage's minimum dihedral target in
  // degrees. A curve between two faces whose true wedge exceeds it takes the
  // rim size of discretize_curves so the meshed wedge keeps a margin above it.
  double rim_dihedral_degrees = 0.0;
  // Solid route only (null: none): the proximity size field
  // (ProximitySizeControl), which bounds a scoped face's size at each point.
  const ProximityField* proximity = nullptr;
};

struct CurveFailure {
  int32_t curve;
  double first, last;
};

struct CurveStatistics {
  int64_t value_evaluations = 0;
  int64_t first_order_evaluations = 0;
  int64_t second_order_evaluations = 0;
  int64_t simpson_subdivisions = 0;
  int64_t sag_evaluations = 0;
  int64_t sag_splits = 0;
  std::vector<int64_t> sag_splits_per_curve;
  int64_t small_feature_resamples = 0;
  std::vector<CurveFailure> unresolved;
};

struct CurveDiscretization {
  // Normalized parameters, including both endpoints, shared by every use.
  std::vector<int64_t> offsets;
  std::vector<double> parameters;
  CurveStatistics statistics;

  CadCurveDiscretization view() const noexcept { return {offsets.data(), parameters.data()}; }
};

// Each representation span receives ceil(F_span) segments, at least one;
// inverse placement uses the monotone integral table, without point evaluation.
// Inputs are monotone tables of length table_count; output has segments + 1
// elements. This allocation-free inversion is also timed independently.
int32_t inverse_curve_table(const double* parameters, const double* integrals,
                            int64_t table_count, int64_t segments, double* output);

// Apply the sampled Gauss-point sag screen and bounded local repair to one
// curve. This operation is exposed separately for phase measurements.
int32_t repair_curve_sag(const CadGeometry& geometry, const CurveControls& controls,
                         int32_t curve, const std::vector<double>& input,
                         std::vector<double>& output, CurveStatistics& statistics);

int32_t discretize_curves(const CadGeometry& geometry, const CurveControls& controls,
                          CurveDiscretization& result);

}  // namespace phx::mc::cad
