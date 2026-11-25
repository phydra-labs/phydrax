// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "dyadic.hpp"
#include "metric_adapt3d.hpp"
#include "metric_carrier.hpp"

namespace phx::mc::metric {

using Point2 = std::array<double, 2>;
using Tri2 = std::array<std::int32_t, 3>;

inline constexpr std::size_t kMaxScalar2DSites = 500'000;
inline constexpr std::size_t kMaxScalar2DCells = 1'000'000;
inline constexpr std::size_t kMaxScalar2DSourceCells = 2'000'000;

struct ScalarSample2D {
  std::int32_t cell = -1;
  double log_rate = 0.0;  // Half of log(m) where M = m * I_2.
  Point2 gradient{};
};

// Numeric log-P1 scalar field on a 2D source triangle mesh covering a Cartesian
// rectangle. Proposals use this field; final coordinates, stencils, metrics,
// coverage, and cell ancestry are certified in exact arithmetic.
class ScalarField2D {
 public:
  bool prepare(std::span<const double> points, std::span<const std::int64_t> cells,
               std::span<const double> log_rates, const Point2& low, const Point2& high);
  bool sample(const Point2& x, ScalarSample2D& out) const;
  std::int32_t locate_exact(const Point2& x, std::int32_t hint) const;
  std::span<const std::int64_t> box_candidates(const Point2& x) const noexcept;

  Point2 low{}, high{};
  double minimum_log_rate = 0.0;
  double maximum_log_rate = 0.0;

 private:
  struct Element {
    Point2 origin;
    std::array<Point2, 3> gradients;
    double log_rate;
    Point2 log_gradient;
  };
  bool ensure_element(std::size_t id) const;

  std::span<const double> points_;
  std::span<const std::int64_t> cells_;
  std::span<const double> log_rates_;
  mutable std::vector<Element> tris_;
  mutable std::vector<std::uint8_t> ready_;
  CarrierBoxGrid boxes_;
};

struct ScalarSites2D {
  std::vector<Point2> points;
  std::vector<ScalarSample2D> samples;
  std::vector<Tri2> cells;
};

struct Adapt2DBoxInput {
  std::span<const double> points;
  std::span<const double> scalar_metrics;
  std::span<const std::uint64_t> vertex_ids;
  std::span<const std::int8_t> fixed_vertices;
  std::span<const std::int64_t> cells;
  std::span<const std::uint64_t> cell_ids;
  std::span<const std::int64_t> edges;
  std::span<const std::int64_t> edge_classes;
  std::span<const std::int8_t> protected_edges;
  std::uint64_t next_vertex_id = 0;
  std::uint64_t next_cell_id = 0;
  double lower = 0.0;
};

struct Adapt2DBoxOutput {
  std::vector<double> points;
  std::vector<std::uint64_t> vertex_ids;
  std::vector<double> scalar_metrics;
  std::vector<std::int64_t> stencil_offsets;
  std::vector<std::int32_t> stencil_rows;
  std::vector<double> stencil_weights;
  std::vector<std::int32_t> cells;
  std::vector<std::uint64_t> cell_ids;
  std::vector<std::int64_t> ancestry_offsets;
  std::vector<std::uint64_t> ancestry_ids;
  std::vector<std::uint64_t> edge_relation_sources;
  std::vector<std::uint64_t> edge_relation_targets;
  std::vector<std::int32_t> edge_relation_kinds;
  std::int64_t measured_edges = 0;
  std::array<double, 4> measurements{};  // min_len, max_len, unit_fraction, min_quality
};

enum class Adapt2DBoxOutcome : std::int32_t {
  kDeclined = 0,
  kAdapted = 1,
  kAlreadyUnit = 2,
};

// Proposal stages (implemented in metric_scalar2d_proposal.cpp).
bool propose_scalar_box_2d(const ScalarField2D& field, double lower, ScalarSites2D& sites);

// Full eligibility check, proposal, exact dyadic carrier binding, exact area
// and boundary certification, and exact 2D cell/wall ancestry.
Adapt2DBoxOutcome adapt_metric_2d_box(const Adapt2DBoxInput& input, Adapt2DBoxOutput& output);

}  // namespace phx::mc::metric
