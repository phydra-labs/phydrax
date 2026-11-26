//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

using Row = uint32_t;
using Slot = uint32_t;
using Entry = std::pair<double, Row>;

struct Neighbor {
  Row row;
  Slot edge;
};

struct Graph {
  Row rows;
  Slot edge_count;
  const int64_t* edges;
  const double* steps;
  std::vector<Slot> offsets;
  std::unique_ptr<Neighbor[]> neighbors;

  void build() {
    offsets.resize(static_cast<std::size_t>(rows) + 1);
    for (Slot slot = 0; slot < 2 * edge_count; ++slot) {
      ++offsets[edges[slot] + 1];
    }
    for (Row row = 0; row < rows; ++row) offsets[row + 1] += offsets[row];
    auto cursor = offsets;
    neighbors = std::make_unique_for_overwrite<Neighbor[]>(2 * edge_count);
    // Match the stable incidence order: first endpoints in input order,
    // then second endpoints in input order. Parallel edges can affect the
    // update count even though their order cannot affect the fixed point.
    for (Slot side = 0; side < 2; ++side) {
      for (Slot edge = 0; edge < edge_count; ++edge) {
        const auto source = static_cast<Row>(edges[2 * edge + side]);
        const auto target = static_cast<Row>(edges[2 * edge + (1 - side)]);
        neighbors[cursor[source]++] = {target, edge};
      }
    }
  }
};

struct Relaxation {
  std::vector<uint8_t> active;
  std::vector<Entry> heap;

  int64_t solve(Graph& graph, double* labels) {
    // A row whose initial value cannot relax a neighbor need not be queued
    // until it decreases: neighbor labels only decrease. If it never does,
    // its omitted pop is equivalent to a pop with no successful relaxation.
    for (Slot edge = 0; edge < graph.edge_count; ++edge) {
      auto source = static_cast<Row>(graph.edges[2 * edge]);
      auto target = static_cast<Row>(graph.edges[2 * edge + 1]);
      if (labels[source] > labels[target]) std::swap(source, target);
      if (labels[source] + graph.steps[edge] < labels[target]) {
        if (active.empty()) active.resize(graph.rows);
        active[source] = 1;
      }
    }
    if (active.empty()) return 0;
    for (Row row = 0; row < graph.rows; ++row) {
      if (active[row]) heap.emplace_back(labels[row], row);
    }
    std::fill(active.begin(), active.end(), 0);
    if (heap.empty()) return 0;
    if (!graph.neighbors) graph.build();

    const auto minimum_first = std::greater<Entry>{};
    std::make_heap(heap.begin(), heap.end(), minimum_first);
    int64_t updates = 0;
    while (!heap.empty()) {
      std::pop_heap(heap.begin(), heap.end(), minimum_first);
      const auto [value, row] = heap.back();
      heap.pop_back();
      if (value != labels[row]) continue;
      for (Slot slot = graph.offsets[row]; slot < graph.offsets[row + 1]; ++slot) {
        const auto [target, edge] = graph.neighbors[slot];
        const double candidate = value + graph.steps[edge];
        if (candidate >= labels[target]) continue;
        labels[target] = candidate;
        ++updates;
        heap.emplace_back(candidate, target);
        std::push_heap(heap.begin(), heap.end(), minimum_first);
      }
    }
    return updates;
  }
};

double directed_quotient(double value, double root, double target_inverse_root,
                         double step) noexcept {
  const double factor = 1.0 + root * step;
  const double grown = value / (factor * factor);
  return (target_inverse_root * grown) * target_inverse_root;
}

// In the coordinate eigenbasis the largest eigenvalue of
// M_target^(-1/2) G_source M_target^(-1/2) is just the maximum axis quotient.
double maximum_quotient(const Graph& graph, const double* values, int32_t axes,
                        const double* roots, const double* inverse_roots) noexcept {
  double maximum = 1.0;
  for (Slot edge = 0; edge < graph.edge_count; ++edge) {
    const int64_t p = graph.edges[2 * edge];
    const int64_t q = graph.edges[2 * edge + 1];
    const double step = graph.steps[edge];
    maximum = std::max(maximum, directed_quotient(values[p * axes], roots[p], inverse_roots[q], step));
    maximum = std::max(maximum, directed_quotient(values[q * axes], roots[q], inverse_roots[p], step));
  }
  return maximum;
}

bool within_bounds(const double* values, int32_t axes, const double* bounds) noexcept {
  const auto [smallest, largest] = std::minmax_element(values, values + axes);
  const double slack = 32.0 * std::numeric_limits<double>::epsilon() * *largest;
  return *smallest + slack >= bounds[0] && *largest - slack <= bounds[1] &&
         *largest - slack <= (bounds[2] * bounds[2]) * (*smallest + slack);
}

void grade_diagonal(Graph& graph, int32_t axes, const double* initial, const double* bounds,
                    double minimum_step, double* graded, int64_t* stats,
                    double& violation) {
  auto labels = std::make_unique_for_overwrite<double[]>(graph.rows);
  auto roots = std::make_unique_for_overwrite<double[]>(graph.rows);
  auto inverse_roots = std::make_unique_for_overwrite<double[]>(graph.rows);
  Relaxation relaxation;
  double quotient = 1.0;
  std::copy_n(initial, static_cast<int64_t>(graph.rows) * axes, graded);
  for (int32_t axis = 0; axis < axes; ++axis) {
    bool constant = true;
    for (Row row = 1; row < graph.rows; ++row) {
      if (initial[static_cast<int64_t>(row) * axes + axis] != initial[axis]) {
        constant = false;
        break;
      }
    }
    if (constant) {
      // Every quotient is a monotone decreasing function of the step on this
      // axis; use the shortest step without reconstructing any eigenvalues.
      const double root = std::sqrt(initial[axis]);
      quotient = std::max(quotient, directed_quotient(initial[axis], root, 1.0 / root, minimum_step));
      continue;
    }
    // The physical growth l/(1 + sqrt(l)*step)^2 is h' = h + step for
    // h = 1/sqrt(l). Diagonal intersection takes the componentwise maximum
    // eigenvalue, hence the minimum size. The uncapped greatest graded size
    // bank is therefore the independent scalar fixed point on each axis.
    for (Row row = 0; row < graph.rows; ++row) {
      const double root = std::sqrt(initial[static_cast<int64_t>(row) * axes + axis]);
      roots[row] = root;
      labels[row] = inverse_roots[row] = 1.0 / root;
    }
    stats[1] += relaxation.solve(graph, labels.get());
    for (Row row = 0; row < graph.rows; ++row) {
      if (labels[row] >= inverse_roots[row]) continue;
      const int64_t index = static_cast<int64_t>(row) * axes + axis;
      const double size = labels[row];
      const double value = 1.0 / (size * size);
      // Inverting the rounded size must not coarsen the authored metric.
      if (value > graded[index]) {
        graded[index] = value;
        roots[row] = std::sqrt(value);
        inverse_roots[row] = 1.0 / roots[row];
      }
    }
    quotient = std::max(quotient, maximum_quotient(graph, graded + axis, axes,
                                                 roots.get(), inverse_roots.get()));
  }
  violation = quotient - 1.0;
  for (Row row = 0; row < graph.rows; ++row) {
    const int64_t index = static_cast<int64_t>(row) * axes;
    stats[2] += !std::equal(graded + index, graded + index + axes, initial + index);
    stats[3] &= static_cast<int64_t>(within_bounds(graded + index, axes, bounds));
  }
}

}  // namespace

extern "C" int32_t phx_mc_metric_grade_physical(
    int64_t row_count, int32_t axes, const double* initial, int64_t edge_count,
    const int64_t* edges, const double* steps, const double* bounds, double* graded,
    int64_t* stats, double* maximum_violation) {
  using phx::mc::addressable;
  if (axes < 1 || axes > 3 || !addressable(row_count, axes, sizeof(double)) ||
      !addressable(edge_count, 2, sizeof(int64_t)) ||
      row_count >= std::numeric_limits<Row>::max() ||
      edge_count > std::numeric_limits<Slot>::max() / 2 ||
      (row_count > 0 && (initial == nullptr || graded == nullptr || initial == graded)) ||
      (edge_count > 0 && (edges == nullptr || steps == nullptr)) ||
      (axes == 1 ? bounds != nullptr : bounds == nullptr) ||
      stats == nullptr || maximum_violation == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  if (axes != 1 && (!std::isfinite(bounds[0]) || !std::isfinite(bounds[1]) ||
                    !std::isfinite(bounds[2]) || !(0.0 < bounds[0] && bounds[0] <= bounds[1]) ||
                    bounds[2] < 1.0)) {
    return PHX_MC_INVALID_INPUT;
  }
  for (int64_t index = 0; index < row_count * axes; ++index) {
    const double value = initial[index];
    if (!std::isfinite(value)) return PHX_MC_NONFINITE_INPUT;
    if (value <= 0.0 || (axes != 1 && (value < 0x1p-80 || value > 0x1p80))) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  double minimum_step = std::numeric_limits<double>::infinity();
  for (int64_t edge = 0; edge < edge_count; ++edge) {
    const int64_t first = edges[2 * edge];
    const int64_t second = edges[2 * edge + 1];
    if (first < 0 || first >= row_count || second < 0 || second >= row_count ||
        !(steps[edge] >= 0.0)) {
      return PHX_MC_INVALID_INPUT;
    }
    minimum_step = std::min(minimum_step, steps[edge]);
  }
  return phx::mc::guarded([&]() -> int32_t {
    // Every row finalizes once, including omitted pops with no relaxation.
    stats[0] = edge_count == 0 ? 0 : row_count * axes;
    stats[1] = stats[2] = 0;
    stats[3] = 1;
    *maximum_violation = 0.0;
    Graph graph{static_cast<Row>(row_count), static_cast<Slot>(edge_count), edges, steps, {}, {}};
    if (axes != 1 && edge_count != 0) {
      grade_diagonal(graph, axes, initial, bounds, minimum_step, graded, stats, *maximum_violation);
    } else if (row_count != 0) {
      std::copy_n(initial, row_count * axes, graded);
      if (axes == 1) {
        stats[1] = Relaxation{}.solve(graph, graded);
        for (int64_t row = 0; row < row_count; ++row) stats[2] += graded[row] < initial[row];
      } else {
        for (int64_t row = 0; row < row_count; ++row) {
          stats[3] &= static_cast<int64_t>(within_bounds(graded + row * axes, axes, bounds));
        }
      }
    }
    return PHX_MC_OK;
  });
}
