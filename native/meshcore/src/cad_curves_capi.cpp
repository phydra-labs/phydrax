// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_curves.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

extern "C" int32_t phx_mc_cad_curves_test(
    const phx_mc_cad_model* model, const phx_mc_cad_curve_controls* controls,
    int64_t offset_capacity, int64_t* offsets, int64_t parameter_capacity, double* parameters,
    int64_t failure_capacity, phx_mc_cad_curve_failure* failures,
    int64_t split_capacity, int64_t* sag_splits,
    int64_t* metric_splits,
    phx_mc_cad_curve_statistics* statistics) {
  return phx::mc::guarded([&]() -> int32_t {
    if (model == nullptr || controls == nullptr || statistics == nullptr || offset_capacity < 0 ||
        parameter_capacity < 0 || failure_capacity < 0 || split_capacity < 0)
      return PHX_MC_INVALID_ARGUMENT;
    phx::mc::cad::CadGeometry geometry;
    const int32_t binding = geometry.bind(*model);
    if (binding != PHX_MC_OK) return binding;
    const phx::mc::cad::CurveControls native_controls{
        controls->face_sizes, controls->face_tolerances, controls->curve_tolerances,
        controls->tolerance_fraction, controls->grading, controls->minimum_size};
    phx::mc::cad::CurveDiscretization result;
    const int32_t status = phx::mc::cad::discretize_curves(geometry, native_controls, result);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    if (result.offsets.size() != static_cast<std::size_t>(model->curve_count) + 1)
      return status == PHX_MC_UNRESOLVED ? status : PHX_MC_INTERNAL_ERROR;
    if (result.offsets.size() > static_cast<uint64_t>(offset_capacity) ||
        result.parameters.size() > static_cast<uint64_t>(parameter_capacity) ||
        result.statistics.unresolved.size() > static_cast<uint64_t>(failure_capacity) ||
        result.statistics.sag_splits_per_curve.size() > static_cast<uint64_t>(split_capacity) ||
        result.statistics.metric_splits_per_curve.size() > static_cast<uint64_t>(split_capacity))
      return PHX_MC_CAPACITY_EXCEEDED;
    if ((!result.offsets.empty() && offsets == nullptr) ||
        (!result.parameters.empty() && parameters == nullptr) ||
        (!result.statistics.unresolved.empty() && failures == nullptr) ||
        (!result.statistics.sag_splits_per_curve.empty() && sag_splits == nullptr) ||
        (!result.statistics.metric_splits_per_curve.empty() && metric_splits == nullptr))
      return PHX_MC_INVALID_ARGUMENT;
    std::copy(result.offsets.begin(), result.offsets.end(), offsets);
    if (!result.parameters.empty())
      std::copy(result.parameters.begin(), result.parameters.end(), parameters);
    if (!result.statistics.sag_splits_per_curve.empty())
      std::copy(result.statistics.sag_splits_per_curve.begin(),
                result.statistics.sag_splits_per_curve.end(), sag_splits);
    if (!result.statistics.metric_splits_per_curve.empty())
      std::copy(result.statistics.metric_splits_per_curve.begin(),
                result.statistics.metric_splits_per_curve.end(), metric_splits);
    for (std::size_t i = 0; i < result.statistics.unresolved.size(); ++i) {
      const auto& failure = result.statistics.unresolved[i];
      failures[i] = {failure.curve, failure.first, failure.last, failure.face};
    }
    *statistics = {result.statistics.first_order_evaluations,
                   result.statistics.second_order_evaluations,
                   result.statistics.sag_evaluations,
                   result.statistics.small_feature_resamples,
                   result.statistics.sag_splits,
                   static_cast<int64_t>(result.parameters.size()),
                   static_cast<int64_t>(result.statistics.unresolved.size()),
                   result.statistics.metric_splits, result.statistics.metric_points_skipped};
    return status;
  });
}
