// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_scalar_box.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "deterministic_float.hpp"
#include "geometry3.hpp"
#include "metric_box_rate.hpp"

namespace phx::mc::metric {

bool scalar_topology_accepted(std::int32_t status) {
  if (status == PHX_MC_ALLOCATION_FAILURE) throw std::bad_alloc();
  if (status == PHX_MC_INDEX_RANGE_ERROR) throw std::length_error("Scalar box site index overflow.");
  if (status == PHX_MC_INTERNAL_ERROR) throw std::logic_error("Scalar box triangulation failed.");
  return status == PHX_MC_OK;
}

bool ScalarBoxField::prepare(const AdaptInput& input, const BoxPoint& minimum,
                             const BoxPoint& maximum) {
  const std::size_t vertices = input.source_points.size() / 3;
  const std::size_t cells = input.source_cells.size() / 4;
  if (vertices == 0 || vertices > kMaxScalarSites || cells == 0 || cells > kMaxScalarCells ||
      3 * vertices != input.source_points.size() || 4 * cells != input.source_cells.size() ||
      input.source_logarithms.size() != 9 * vertices || input.source_metrics.size() != 9 * vertices) {
    return false;
  }
  low = minimum;
  high = maximum;
  for (int k = 0; k < 3; ++k) if (!(low[k] < high[k])) return false;
  double min_metric = std::numeric_limits<double>::infinity();
  double max_metric = 0.0;
  for (std::size_t i = 0; i < vertices; ++i) {
    const auto* metric = input.source_metrics.data() + 9 * i;
    const auto* log = input.source_logarithms.data() + 9 * i;
    if (!(metric[0] > 0.0) || !std::isfinite(metric[0]) || !std::isfinite(log[0])) return false;
    for (int j = 1; j < 9; ++j) {
      if (j % 4 == 0 ? metric[j] != metric[0] || log[j] != log[0] :
                       metric[j] != 0.0 || log[j] != 0.0) return false;
    }
    min_metric = std::min(min_metric, metric[0]);
    max_metric = std::max(max_metric, metric[0]);
  }
  if (min_metric == max_metric) return false;
  maximum_size = 1.0 / std::sqrt(min_metric);
  std::vector<double> lower(3 * cells), upper(3 * cells);
  tets_.resize(cells);
  double estimated_work = 0.0;
  for (std::size_t row = 0; row < cells; ++row) {
    std::array<BoxPoint, 4> corners;
    std::array<double, 4> rates;
    for (int k = 0; k < 4; ++k) {
      const auto i = static_cast<std::size_t>(input.source_cells[4 * row + k]);
      if (i >= vertices) return false;
      std::copy_n(input.source_points.data() + 3 * i, 3, corners[k].data());
      if (!geometry::point_in_domain(corners[k])) return false;
      rates[k] = 0.5 * input.source_logarithms[9 * i];
    }
    auto& tet = tets_[row];
    tet.origin = corners[0];
    tet.log_rate = rates[0];
    if (!prepare_box_rate(corners, rates, tet.gradients, tet.log_gradient)) return false;
    for (int k = 0; k < 3; ++k) {
      lower[3 * row + k] = std::min({corners[0][k], corners[1][k], corners[2][k], corners[3][k]});
      upper[3 * row + k] = std::max({corners[0][k], corners[1][k], corners[2][k], corners[3][k]});
      if (lower[3 * row + k] < low[k] || upper[3 * row + k] > high[k]) return false;
    }
    const auto a = geometry::difference(corners[1], tet.origin);
    const auto b = geometry::difference(corners[2], tet.origin);
    const auto c = geometry::difference(corners[3], tet.origin);
    const double volume6 = std::abs(geometry::dot(a.data(), geometry::cross(b, c).data()));
    // Dimensionless metric volume, estimated at the barycenter. This only
    // selects a refinement construction; no bound or certificate uses it.
    const double mean3 = 0.75 * ((rates[0] + rates[1]) + (rates[2] + rates[3]));
    estimated_work += volume6 * deterministic::exp(mean3);
  }
  if (!(estimated_work > 2.0 * static_cast<double>(cells) &&
        estimated_work <= static_cast<double>(kMaxScalarCells))) return false;
  boxes_.prepare(lower, upper, 0b111, 4'000'000);
  return boxes_.indexed();
}

bool ScalarBoxField::sample(const BoxPoint& x, ScalarBoxSample& sample) const {
  if (!geometry::point_in_domain(x)) return false;
  const auto locate = [&](std::size_t id) {
    const auto& tet = tets_[id];
    const auto d = geometry::difference(x, tet.origin);
    for (int k = 0; k < 4; ++k) {
      if (geometry::dot(d.data(), tet.gradients[k].data()) + (k == 0 ? 1.0 : 0.0) < -1.e-10) return false;
    }
    const double value = tet.log_rate + geometry::dot(tet.log_gradient.data(), d.data());
    if (!std::isfinite(value)) return false;
    sample = {static_cast<std::int32_t>(id), value, tet.log_gradient};
    return true;
  };
  if (sample.cell >= 0 && static_cast<std::size_t>(sample.cell) < tets_.size() &&
      locate(static_cast<std::size_t>(sample.cell))) return true;
  for (auto cell : boxes_.candidates(x.data())) {
    if (locate(static_cast<std::size_t>(cell))) return true;
  }
  return false;
}

}  // namespace phx::mc::metric
