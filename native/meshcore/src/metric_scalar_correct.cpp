// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_scalar_box.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "bounded_lbfgs.hpp"
#include "deterministic_float.hpp"
#include "geometry3.hpp"
#include "metric_float.hpp"
#include "topology_keys.hpp"

namespace phx::mc::metric {
namespace {

using Row = std::int32_t;
using Edge = std::array<Row, 2>;
using geometry::cross;
using geometry::difference;
using geometry::dot;

constexpr double kInvalidMerit = std::numeric_limits<double>::quiet_NaN();
constexpr double kShapeFloor = 0.60;
constexpr double kShapeTarget = 0.605;
constexpr double kPenaltyGrowth = 2.5;
constexpr int kMaxStages = 7;
constexpr int kStepsPerStage = 65;

struct Band {
  double low, high;
};

struct Node {
  BoxPoint point;
  ScalarBoxSample sample;
  std::array<Row, 3> dof{-1, -1, -1};
  double rate = 0.0;
  double size = 0.0;
  bool changed = false;
};

struct EdgeConstraint {
  Edge rows;
  double low_shift = 0.0, high_shift = 0.0;
  double distance = 0.0, mean_rate = 0.0, length = 0.0;
};

struct ShapeConstraint {
  double shift = 0.0;
  double fatness = 0.0;
  double inverse_denominator = 0.0, inverse_squares = 0.0;
};

struct CellGeometry {
  BoxPoint u, v, w;
  BoxPoint du;

  CellGeometry(const BoxPoint& a, const BoxPoint& b, const BoxPoint& c, const BoxPoint& d)
      : u(difference(b, a)), v(difference(c, a)), w(difference(d, a)), du(cross(v, w)) {}
};

bool sample_at(const ScalarBoxField& field, const BoxPoint& x, Node& node) {
  auto sample = node.sample;
  if (!field.sample(x, sample)) return false;
  if (node.rate == 0.0 || sample.log_rate != node.sample.log_rate) {
    const double rate = deterministic::exp(sample.log_rate);
    if (!(rate > 0.0) || !std::isfinite(rate)) return false;
    node.rate = rate;
  }
  node.sample = sample;
  node.point = x;
  return true;
}

// exp(a) and exp(b) are cached per node. With w=(b-a)/2, the log mean
// divided by the arithmetic mean of the endpoint rates is tanh(w)/w.
constexpr double kNearEqual = 0.002;

double mean_rate(const Node& a, const Node& b) {
  const double gap = b.sample.log_rate - a.sample.log_rate;
  if (std::abs(gap) < kNearEqual) {
    const double w2 = (0.25 * gap) * gap;
    return (0.5 * a.rate + 0.5 * b.rate) * (1.0 + w2 * (-1.0 / 3.0 + w2 * (2.0 / 15.0)));
  }
  return (b.rate - a.rate) / gap;
}

std::array<double, 2> mean_rate_gradient(const Node& a, const Node& b, double mean) {
  const double gap = b.sample.log_rate - a.sample.log_rate;
  if (std::abs(gap) < kNearEqual) {
    const double w = 0.5 * gap;
    const double odd = (0.5 * a.rate + 0.5 * b.rate) * (w * (1.0 / 3.0 - w * w * (2.0 / 15.0)));
    return {0.5 * (mean - odd), 0.5 * (mean + odd)};
  }
  return {(mean - a.rate) / gap, (b.rate - mean) / gap};
}

bool update_edge(const Node& a, const Node& b, EdgeConstraint& edge) {
  edge.distance = std::sqrt(geometry::squared_distance(a.point.data(), b.point.data()));
  edge.mean_rate = mean_rate(a, b);
  edge.length = edge.distance * edge.mean_rate;
  return edge.length > 0.0 && std::isfinite(edge.length);
}

bool update_shape(const CellGeometry& t, double limit, ShapeConstraint& shape) {
  const double determinant = dot(t.u.data(), t.du.data());
  const double squares = dot(t.u.data(), t.u.data()) + dot(t.v.data(), t.v.data()) +
                         dot(t.w.data(), t.w.data()) +
                         geometry::squared_distance(t.u.data(), t.v.data()) +
                         geometry::squared_distance(t.u.data(), t.w.data()) +
                         geometry::squared_distance(t.v.data(), t.w.data());
  const double mean_square = squares / 12.0;
  const double denominator = (2.0 * mean_square) * std::sqrt(mean_square);
  if (!(denominator > 0.0) || !std::isfinite(denominator) || !std::isfinite(determinant)) {
    return false;
  }
  // Positive determinant above this guarded cutoff also clears the hard
  // floor. No division or volume gradient is needed in the inactive tail.
  if (determinant > (limit * (1.0 + 0x1p-40)) * denominator) {
    shape.fatness = limit;
  } else {
    shape.fatness = determinant / denominator;
    shape.inverse_denominator = 1.0 / denominator;
    shape.inverse_squares = 1.0 / squares;
  }
  return true;
}

class ScalarRepair {
 public:
  ScalarRepair(const ScalarBoxField& field, const ScalarBoxSites& sites, Band hard)
      : field_(field), sites_(sites), hard_(hard),
        target_{hard.low + std::min(0.0009, (hard.high - hard.low) / 8.0),
                hard.high - std::min(0.0012, (hard.high - hard.low) / 8.0)} {}

  bool prepare(bool local);
  double evaluate(std::span<const double> x, std::span<double> gradient);
  void increase_penalty();
  void publish(ScalarBoxSites& sites) const;

  std::span<const Band> bounds() const { return bounds_; }
  bool sparse() const { return sparse_; }
  bool feasible() const { return feasible_; }
  bool interior() const { return interior_; }
  bool retile(ScalarBoxSites& sites) const;

 private:
  bool prepare_workset(std::vector<std::uint8_t>& selected);
  bool select_coordinates(std::span<const std::uint8_t> selected);
  template <class Visit>
  bool visit_rows(std::size_t size, std::span<const Row> workset, Visit&& visit) const {
    if (sparse_) {
      for (Row row : workset) {
        if (!visit(row)) return false;
      }
    } else {
      for (std::size_t row = 0; row < size; ++row) {
        if (!visit(row)) return false;
      }
    }
    return true;
  }
  bool update_node(std::size_t row, std::span<const double> x, bool force);
  void edge_gradient(const EdgeConstraint& edge, double action, std::span<double> gradient) const;
  void shape_gradient(const BoxCell& cell, const CellGeometry& t, const ShapeConstraint& shape,
                      double action, std::span<double> gradient) const;

  const ScalarBoxField& field_;
  const ScalarBoxSites& sites_;
  const Band hard_, target_;
  const double shape_floor_ = kShapeFloor * std::sqrt(kShapeFloor);
  const double shape_target_ = kShapeTarget * std::sqrt(kShapeTarget);
  std::array<Band, 3> wall_interior_;
  std::vector<Node> nodes_;
  std::vector<Band> bounds_;
  std::vector<EdgeConstraint> edges_;
  std::vector<ShapeConstraint> shapes_;
  std::vector<Row> active_nodes_, active_edges_, active_cells_;
  bool sparse_ = false;
  bool cache_valid_ = false;
  bool feasible_ = false;
  bool interior_ = false;
};

bool ScalarRepair::prepare(bool local) {
  for (int k = 0; k < 3; ++k) {
    if (!(field_.low[k] < field_.high[k])) return false;
    wall_interior_[k] = {std::nextafter(field_.low[k], field_.high[k]),
                         std::nextafter(field_.high[k], field_.low[k])};
  }
  nodes_.resize(sites_.points.size());
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    node.sample = sites_.samples[i];
    const auto& origin = sites_.points[i];
    if (!sample_at(field_, origin, node)) return false;
    node.size = 1.0 / node.rate;
    if (!std::isfinite(node.size)) return false;
  }

  std::vector<Edge> keys;
  keys.reserve(6 * sites_.cells.size());
  for (const auto& cell : sites_.cells) {
    for (Row r : cell) {
      if (r < 0 || static_cast<std::size_t>(r) >= nodes_.size()) return false;
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        if (cell[a] == cell[b]) return false;
        keys.push_back(topology::undirected_edge_vertices(cell[a], cell[b]));
      }
    }
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  edges_.reserve(keys.size());
  for (const auto& edge : keys) edges_.push_back({edge});
  shapes_.resize(sites_.cells.size());
  std::vector<std::uint8_t> selected;
  if (local && !prepare_workset(selected)) return false;
  return select_coordinates(selected);
}

bool ScalarRepair::prepare_workset(std::vector<std::uint8_t>& selected) {
  selected.resize(nodes_.size());
  for (auto& edge : edges_) {
    const auto [a, b] = edge.rows;
    if (!update_edge(nodes_[a], nodes_[b], edge)) return false;
    if (edge.length < target_.low || edge.length > target_.high) {
      selected[a] = selected[b] = 1;
    }
  }
  for (std::size_t row = 0; row < shapes_.size(); ++row) {
    const auto& c = sites_.cells[row];
    const CellGeometry t(nodes_[c[0]].point, nodes_[c[1]].point,
                         nodes_[c[2]].point, nodes_[c[3]].point);
    auto& shape = shapes_[row];
    if (!update_shape(t, shape_target_, shape)) return false;
    if (shape.fatness < shape_target_) {
      for (Row r : c) selected[r] = 1;
    }
  }
  cache_valid_ = true;

  // Expand simultaneous rings in the tetrahedral edge graph. The ring
  // number prevents a newly selected vertex from spreading in this pass.
  for (std::uint8_t ring = 1; ring <= 2; ++ring) {
    for (const auto& cell : sites_.cells) {
      const auto frontier = [&](Row r) { return selected[r] == ring; };
      if (std::any_of(cell.begin(), cell.end(), frontier)) {
        for (Row r : cell) {
          if (selected[r] == 0) selected[r] = ring + 1;
        }
      }
    }
  }
  for (std::size_t row = 0; row < nodes_.size(); ++row) {
    if (selected[row] == 0) continue;
    const auto& p = sites_.points[row];
    bool movable = false;
    for (int k = 0; k < 3; ++k) {
      movable |= p[k] != field_.low[k] && p[k] != field_.high[k];
    }
    selected[row] = movable;
  }
  const auto moves = [&](Row r) { return selected[r] != 0; };

  // Include the full incident halo, plus target violations that have no
  // free coordinates. Every omitted row was measured inside its target
  // and is constant: it contributes neither merit nor a penalty shift.
  std::vector<Row> cells;
  for (std::size_t row = 0; row < shapes_.size(); ++row) {
    const auto& cell = sites_.cells[row];
    if (shapes_[row].fatness < shape_target_ || std::any_of(cell.begin(), cell.end(), moves)) {
      cells.push_back(static_cast<Row>(row));
      if (cells.size() * 4 >= shapes_.size() * 3) return true;
    }
  }
  for (std::size_t row = 0; row < edges_.size(); ++row) {
    const auto& edge = edges_[row];
    if (moves(edge.rows[0]) || moves(edge.rows[1]) ||
        edge.length < target_.low || edge.length > target_.high) {
      active_edges_.push_back(static_cast<Row>(row));
    }
  }
  active_cells_.swap(cells);
  sparse_ = true;
  return true;
}

bool ScalarRepair::select_coordinates(std::span<const std::uint8_t> selected) {
  if (!sparse_) bounds_.reserve(3 * nodes_.size());
  for (std::size_t row = 0; row < nodes_.size(); ++row) {
    auto& node = nodes_[row];
    const auto& origin = sites_.points[row];
    const bool choose = !sparse_ || selected[row] != 0;
    for (int k = 0; k < 3; ++k) {
      if (origin[k] == field_.low[k] || origin[k] == field_.high[k]) continue;
      const auto [lo, hi] = wall_interior_[k];
      if (!(origin[k] >= lo && origin[k] <= hi)) return false;
      const Band bound{(lo - origin[k]) / node.size, (hi - origin[k]) / node.size};
      if (!std::isfinite(bound.low) || !std::isfinite(bound.high)) return false;
      if (!choose) continue;
      node.dof[k] = static_cast<Row>(bounds_.size());
      bounds_.push_back(bound);
    }
    if (sparse_ && choose) active_nodes_.push_back(static_cast<Row>(row));
  }
  return true;
}

bool ScalarRepair::update_node(std::size_t row, std::span<const double> x, bool force) {
  auto& node = nodes_[row];
  auto q = sites_.points[row];
  for (int k = 0; k < 3; ++k) {
    if (node.dof[k] < 0) continue;
    const auto [lo, hi] = wall_interior_[k];
    // Projection uses the local-size units. The second clamp just guards
    // rounding in the reconstruction and preserves the original stratum.
    q[k] = std::clamp(q[k] + node.size * x[node.dof[k]], lo, hi);
  }
  const bool moved = q != node.point;
  node.changed = force || moved;
  return !moved || sample_at(field_, q, node);
}

void ScalarRepair::edge_gradient(const EdgeConstraint& edge, double action,
                                 std::span<double> gradient) const {
  const auto& a = nodes_[edge.rows[0]];
  const auto& b = nodes_[edge.rows[1]];
  const auto dm = mean_rate_gradient(a, b, edge.mean_rate);
  const double coordinate = (action * edge.mean_rate) / edge.distance;
  const double rate_a = (action * edge.distance) * dm[0];
  const double rate_b = (action * edge.distance) * dm[1];
  for (int k = 0; k < 3; ++k) {
    const double direction = coordinate * (b.point[k] - a.point[k]);
    if (a.dof[k] >= 0) gradient[a.dof[k]] += a.size * (rate_a * a.sample.gradient[k] - direction);
    if (b.dof[k] >= 0) gradient[b.dof[k]] += b.size * (rate_b * b.sample.gradient[k] + direction);
  }
}

void ScalarRepair::shape_gradient(const BoxCell& cell, const CellGeometry& t,
                                  const ShapeConstraint& shape, double action,
                                  std::span<double> gradient) const {
  const auto dv = cross(t.w, t.u);
  const auto dw = cross(t.u, t.v);
  const double volume = -action * shape.inverse_denominator;
  const double scale = (12.0 * action * shape.fatness) * shape.inverse_squares;
  const auto add = [&](int corner, int k, double derivative) {
    const auto& node = nodes_[cell[corner]];
    if (node.dof[k] >= 0) gradient[node.dof[k]] += node.size * derivative;
  };
  for (int k = 0; k < 3; ++k) {
    const double center = 0.25 * (t.u[k] + t.v[k] + t.w[k]);
    add(0, k, -volume * (t.du[k] + dv[k] + dw[k]) - scale * center);
    add(1, k, volume * t.du[k] + scale * (t.u[k] - center));
    add(2, k, volume * dv[k] + scale * (t.v[k] - center));
    add(3, k, volume * dw[k] + scale * (t.w[k] - center));
  }
}

double ScalarRepair::evaluate(std::span<const double> x, std::span<double> gradient) {
  const bool force = !cache_valid_;
  cache_valid_ = false;
  feasible_ = interior_ = false;
  if (!visit_rows(nodes_.size(), active_nodes_, [&](std::size_t row) {
        return update_node(row, x, force);
      })) return kInvalidMerit;
  std::fill(gradient.begin(), gradient.end(), 0.0);
  const double inside_low = 0.5 * (hard_.low + target_.low);
  const double inside_high = 0.5 * (hard_.high + target_.high);
  const double inside_shape = 0.5 * (shape_floor_ + shape_target_);
  bool feasible = true, interior = true;
  double value = 0.0;
  const auto edge_row = [&](std::size_t row) {
    auto& edge = edges_[row];
    const auto& a = nodes_[edge.rows[0]];
    const auto& b = nodes_[edge.rows[1]];
    if ((a.changed || b.changed) && !update_edge(a, b, edge)) return false;
    feasible &= edge.length >= hard_.low && edge.length <= hard_.high;
    interior &= edge.length >= inside_low && edge.length <= inside_high;
    const double low = std::max(0.0, (target_.low + edge.low_shift) - edge.length);
    const double high = std::max(0.0, edge.length - (target_.high - edge.high_shift));
    value += low * low + high * high;
    if (const double action = high - low; action != 0.0) edge_gradient(edge, action, gradient);
    return true;
  };
  const auto cell_row = [&](std::size_t i) {
    auto& shape = shapes_[i];
    const auto& cell = sites_.cells[i];
    const auto& a = nodes_[cell[0]];
    const auto& b = nodes_[cell[1]];
    const auto& c = nodes_[cell[2]];
    const auto& d = nodes_[cell[3]];
    const double limit = shape_target_ + shape.shift;
    const bool changed = a.changed || b.changed || c.changed || d.changed;
    if (changed || shape.fatness < limit) {
      const CellGeometry t(a.point, b.point, c.point, d.point);
      if (changed && !update_shape(t, limit, shape)) return false;
      if (const double action = limit - shape.fatness; action > 0.0) {
        value += action * action;
        shape_gradient(cell, t, shape, action, gradient);
      }
    }
    // A positive q^(3/2) gate includes signed-volume positivity, even
    // though the optimizer can cross through zero on its rejected trials.
    feasible &= shape.fatness >= shape_floor_;
    interior &= shape.fatness >= inside_shape;
    return true;
  };
  if (!visit_rows(edges_.size(), active_edges_, edge_row) ||
      !visit_rows(shapes_.size(), active_cells_, cell_row)) return kInvalidMerit;
  if (!std::isfinite(value)) return kInvalidMerit;
  cache_valid_ = true;
  feasible_ = feasible;
  interior_ = interior;
  return 0.5 * value;
}

void ScalarRepair::increase_penalty() {
  // Divide the augmented Lagrangian by its penalty parameter. These are
  // lambda / mu; the gradient and line search retain the seed's units.
  visit_rows(edges_.size(), active_edges_, [&](std::size_t row) {
    auto& edge = edges_[row];
    edge.low_shift = std::max(0.0, (target_.low + edge.low_shift) - edge.length) / kPenaltyGrowth;
    edge.high_shift = std::max(0.0, edge.length - (target_.high - edge.high_shift)) / kPenaltyGrowth;
    return true;
  });
  visit_rows(shapes_.size(), active_cells_, [&](std::size_t row) {
    auto& shape = shapes_[row];
    const double deficit = (shape_target_ + shape.shift) - shape.fatness;
    // A screened value equals its old limit. Its limit cannot increase
    // here, so that cached lower bound is still sufficient if unmoved.
    shape.shift = std::max(0.0, deficit) / kPenaltyGrowth;
    return true;
  });
}

bool ScalarRepair::retile(ScalarBoxSites& sites) const {
  // Restrict topology changes to sparse remaining defects. If the global
  // proposal still needs widespread repair, the local adapter owns it.
  constexpr std::size_t cap = 256;
  std::vector<Edge> bad;
  for (const auto& e : edges_) {
    if (e.length < hard_.low || e.length > hard_.high) {
      if (bad.size() == cap) return false;
      bad.push_back(e.rows);
    }
  }
  std::vector<Row> seeds;
  for (std::size_t row = 0; row < sites.cells.size(); ++row) {
    const auto& c = sites.cells[row];
    bool selected = shapes_[row].fatness < shape_floor_;
    for (int a = 0; a < 4 && !selected && !bad.empty(); ++a) {
      for (int b = a + 1; b < 4 && !selected; ++b) {
        selected = std::binary_search(bad.begin(), bad.end(), topology::undirected_edge_vertices(c[a], c[b]));
      }
    }
    if (!selected) continue;
    if (seeds.size() == cap) return false;
    seeds.push_back(static_cast<Row>(row));
  }
  if (seeds.empty()) return false;
  std::vector<BoxPoint> moved;
  moved.reserve(nodes_.size());
  for (const auto& n : nodes_) moved.push_back(n.point);
  if (!retile_scalar_defects(field_, moved, sites.cells, seeds)) return false;
  publish(sites);
  return true;
}

void ScalarRepair::publish(ScalarBoxSites& sites) const {
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    sites.points[i] = nodes_[i].point;
    sites.samples[i] = nodes_[i].sample;
  }
}

bool bounded_repair(ScalarRepair& repair, std::span<double> x, std::span<double> work) {
  const auto result = detail::projected_bounded_lbfgs(
      x, repair.bounds(), 0.15, kStepsPerStage, 1e-10, work,
      [&](std::span<const double> trial, std::span<double> gradient) {
        return repair.evaluate(trial, gradient);
      },
      [&] { return repair.interior(); });
  return result.status != detail::BoundedLbfgsStatus::kInvalidObjective;
}

}  // namespace

enum class RepairEnd { kAccepted, kRetiled, kRefused, kRetryDense };

static RepairEnd repair_connectivity(const ScalarBoxField& field, ScalarBoxSites& sites,
                                     double lower, bool allow_retile, bool local) {
  if (sites.points.empty() || sites.points.size() > kMaxScalarSites ||
      sites.samples.size() != sites.points.size() || sites.cells.empty() ||
      sites.cells.size() > kMaxScalarCells || !(lower > 0.0 && lower < kUpperLength)) {
    return RepairEnd::kRefused;
  }
  ScalarRepair repair(field, sites, {std::max(lower, kLowerLength), kUpperLength});
  if (!repair.prepare(local)) return RepairEnd::kRefused;
  const std::size_t n = repair.bounds().size();
  constexpr auto kWorkBuffers = 5 + 2 * detail::BoundedLbfgsHistory::kCapacity;
  const auto storage = std::make_unique_for_overwrite<double[]>(kWorkBuffers * n);
  const std::span<double> work{storage.get(), kWorkBuffers * n};
  std::vector<double> position(n, 0.0);
  const auto failure = repair.sparse() ? RepairEnd::kRetryDense : RepairEnd::kRefused;
  for (int stage = 0; stage < kMaxStages; ++stage) {
    if (!bounded_repair(repair, position, work)) return failure;
    if (repair.feasible()) {
      repair.publish(sites);
      return RepairEnd::kAccepted;
    }
    if (position.empty()) return failure;
    if (allow_retile && (stage == 0 || stage == 2) && repair.retile(sites)) return RepairEnd::kRetiled;
    if (stage + 1 < kMaxStages) repair.increase_penalty();
  }
  return failure;
}

bool correct_scalar_box(const ScalarBoxField& field, ScalarBoxSites& sites, double lower) {
  for (int round = 0; round < 3; ++round) {
    auto end = repair_connectivity(field, sites, lower, round < 2, round != 0);
    // A restricted solve is a proposal, not a convergence criterion. Its
    // failure leaves sites untouched; retry from that seed with every DOF.
    if (end == RepairEnd::kRetryDense) {
      end = repair_connectivity(field, sites, lower, round < 2, false);
    }
    if (end != RepairEnd::kRetiled) return end == RepairEnd::kAccepted;
  }
  return false;
}

}  // namespace phx::mc::metric
