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
                             const BoxPoint& maximum, const CarrierBoxGrid* shared_grid) {
  const std::size_t vertices = input.source_points.size() / 3;
  const std::size_t cells = input.source_cells.size() / 4;
  // Sample hints are int32 cell rows; the transaction owns the source cap.
  if (vertices == 0 || cells == 0 || vertices > 4 * cells ||
      cells > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      3 * vertices != input.source_points.size() || 4 * cells != input.source_cells.size() ||
      input.source_logarithms.size() != 9 * vertices || input.source_metrics.size() != 9 * vertices) {
    return false;
  }
  low = minimum;
  high = maximum;
  for (int k = 0; k < 3; ++k) if (!(low[k] < high[k])) return false;
  minimum_log_rate = std::numeric_limits<double>::infinity();
  maximum_log_rate = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < vertices; ++i) {
    const auto* metric = input.source_metrics.data() + 9 * i;
    const auto* log = input.source_logarithms.data() + 9 * i;
    if (!(metric[0] > 0.0) || !std::isfinite(metric[0]) || !std::isfinite(log[0])) return false;
    for (int j = 1; j < 9; ++j) {
      if (j % 4 == 0 ? metric[j] != metric[0] || log[j] != log[0] :
                       metric[j] != 0.0 || log[j] != 0.0) return false;
    }
    minimum_log_rate = std::min(minimum_log_rate, 0.5 * log[0]);
    maximum_log_rate = std::max(maximum_log_rate, 0.5 * log[0]);
  }
  const bool build_grid = shared_grid == nullptr;
  std::vector<double> lower, upper;
  if (build_grid) {
    lower.resize(3 * cells);
    upper.resize(3 * cells);
  }
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
    const auto a = geometry::difference(corners[1], corners[0]);
    const auto b = geometry::difference(corners[2], corners[0]);
    const auto c = geometry::difference(corners[3], corners[0]);
    const double det = geometry::dot(a.data(), geometry::cross(b, c).data());
    const double scale = geometry::length(a) * geometry::length(b) * geometry::length(c);
    if (!(std::abs(det) > 64 * std::numeric_limits<double>::epsilon() * scale)) return false;
    for (int k = 0; k < 3; ++k) {
      const auto [lo, hi] = std::minmax({corners[0][k], corners[1][k], corners[2][k], corners[3][k]});
      if (lo < low[k] || hi > high[k]) return false;
      if (build_grid) {
        lower[3 * row + k] = lo;
        upper[3 * row + k] = hi;
      }
    }
    // Dimensionless metric volume, estimated at the barycenter. It bounds the
    // proposal's resources; no bound or certificate uses it.
    const double mean3 = 0.75 * ((rates[0] + rates[1]) + (rates[2] + rates[3]));
    estimated_work += std::abs(det) * deterministic::exp(mean3);
  }
  if (!(estimated_work > 0.0 && estimated_work <= static_cast<double>(kMaxScalarCells))) return false;
  source_points_ = input.source_points;
  source_cells_ = input.source_cells;
  source_logs_ = input.source_logarithms;
  tets_.resize(cells);
  ready_.assign(cells, 0);
  if (build_grid) {
    owned_boxes_.prepare(lower, upper, 0b111, 64 * cells);
    boxes_ = &owned_boxes_;
  } else {
    if (!shared_grid->indexed() || shared_grid->membership_count() > 64 * cells) return false;
    boxes_ = shared_grid;
  }
  return boxes_->indexed();
}

bool ScalarBoxField::ensure_element(std::size_t id) const {
  if (ready_[id] != 0) return true;
  std::array<BoxPoint, 4> corners;
  std::array<double, 4> rates;
  for (int k = 0; k < 4; ++k) {
    const auto i = static_cast<std::size_t>(source_cells_[4 * id + k]);
    std::copy_n(source_points_.data() + 3 * i, 3, corners[k].data());
    rates[k] = 0.5 * source_logs_[9 * i];
  }
  auto& tet = tets_[id];
  tet.origin = corners[0];
  tet.log_rate = rates[0];
  if (!prepare_box_rate(corners, rates, tet.gradients, tet.log_gradient)) return false;
  ready_[id] = 1;
  return true;
}

bool ScalarBoxField::sample(const BoxPoint& x, ScalarBoxSample& sample) const {
  if (!geometry::point_in_domain(x) || boxes_ == nullptr) return false;
  const auto locate = [&](std::size_t id) {
    if (!ensure_element(id)) return false;
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
  for (auto cell : boxes_->candidates(x.data())) {
    if (locate(static_cast<std::size_t>(cell))) return true;
  }
  return false;
}

}  // namespace phx::mc::metric
