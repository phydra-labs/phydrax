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
#include "metric_box_edges.hpp"
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
constexpr std::size_t kWorkBuffers = 5 + 2 * detail::BoundedLbfgsHistory::kCapacity;
constexpr double kLinearTrust = 0.125;  // In local-size coordinates.
constexpr double kLinearEdgeMargin = 0.08;
constexpr double kLinearShapeMargin = 0.075;  // In q^(3/2) units.
constexpr int kLinearModels = 4;
constexpr int kLinearSteps = 12;

struct Band {
  double low, high;
};

enum class LinearEnd { kUnchanged, kAdvanced, kInvalid };

// Each row is an affine interval on displacement from the frozen iterate.
// An edge contributes at most six entries, a tetrahedron at most twelve.
struct LinearModel {
  struct Entry {
    Row column;
    double derivative;
  };
  struct Constraint {
    std::size_t end;
    Band interval;
  };
  std::vector<Entry> jacobian;
  std::vector<Constraint> rows;

  template <class Emit>
  bool add(Band interval, std::span<const Row> columns, Emit&& emit) {
    const auto first = jacobian.size();
    double squares = 0.0;
    emit([&](Row dof, double derivative) {
      const Row column = columns[dof];
      if (column < 0 || derivative == 0.0) return;
      jacobian.push_back({column, derivative});
      squares += derivative * derivative;
    });
    if (first == jacobian.size()) return true;
    if (!std::isfinite(squares)) return false;
    // Balance high-sensitivity rows without amplifying flat ones.
    const double scale = 1.0 / std::max(1.0, std::sqrt(squares));
    for (std::size_t i = first; i < jacobian.size(); ++i) jacobian[i].derivative *= scale;
    rows.push_back({jacobian.size(), {interval.low * scale, interval.high * scale}});
    return true;
  }

  double evaluate(std::span<const double> shift, std::span<double> gradient) const {
    std::fill(gradient.begin(), gradient.end(), 0.0);
    double value = 0.0;
    std::size_t first = 0;
    for (const auto& row : rows) {
      double change = 0.0;
      for (std::size_t i = first; i < row.end; ++i) {
        const auto& entry = jacobian[i];
        change += entry.derivative * shift[entry.column];
      }
      const double action = change - std::clamp(change, row.interval.low, row.interval.high);
      if (action != 0.0) {
        value += action * action;
        for (std::size_t i = first; i < row.end; ++i) {
          const auto& entry = jacobian[i];
          gradient[entry.column] += action * entry.derivative;
        }
      }
      first = row.end;
    }
    return 0.5 * value;
  }
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

  bool prepare(bool local, bool linearize);
  double evaluate(std::span<const double> x, std::span<double> gradient, double shape_margin = 0.0);
  LinearEnd linearized_repair(std::span<double> x);
  void increase_penalty();
  void publish(ScalarBoxSites& sites) const;

  std::span<const Band> bounds() const { return bounds_; }
  bool sparse() const { return sparse_; }
  bool feasible() const { return feasible_; }
  bool interior() const { return interior_; }
  bool retile(ScalarBoxSites& sites) const;

 private:
  bool prepare_workset(std::vector<std::uint8_t>& selected, bool linearize);
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
  std::size_t select_linear_columns(std::span<Row> columns, std::vector<Row>& dofs) const;
  bool linearize(LinearModel& model, std::span<const Row> columns) const;
  template <class Emit>
  void edge_gradient(const EdgeConstraint& edge, double action, Emit&& emit) const;
  template <class Emit>
  void shape_gradient(const BoxCell& cell, const CellGeometry& t, const ShapeConstraint& shape,
                      double action, Emit&& emit) const;

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

bool ScalarRepair::prepare(bool local, bool linearize) {
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

  edges_.reserve(sites_.cells.size() + 2 * nodes_.size());
  if (!visit_box_edges(nodes_.size(), sites_.cells, [&](Edge e) { edges_.push_back({e}); })) {
    return false;
  }
  shapes_.resize(sites_.cells.size());
  std::vector<std::uint8_t> selected;
  if (local && !prepare_workset(selected, linearize)) return false;
  return select_coordinates(selected);
}

bool ScalarRepair::prepare_workset(std::vector<std::uint8_t>& selected, bool linearize) {
  selected.resize(nodes_.size());
  for (auto& edge : edges_) {
    const auto [a, b] = edge.rows;
    if (!update_edge(nodes_[a], nodes_[b], edge)) return false;
    if (edge.length < target_.low || edge.length > target_.high) {
      selected[a] = selected[b] = 1;
    }
  }
  const double limit = shape_target_ + (linearize ? kLinearShapeMargin : 0.0);
  for (std::size_t row = 0; row < shapes_.size(); ++row) {
    const auto& c = sites_.cells[row];
    const CellGeometry t(nodes_[c[0]].point, nodes_[c[1]].point,
                         nodes_[c[2]].point, nodes_[c[3]].point);
    auto& shape = shapes_[row];
    if (!update_shape(t, limit, shape)) return false;
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

template <class Emit>
void ScalarRepair::edge_gradient(const EdgeConstraint& edge, double action, Emit&& emit) const {
  const auto& a = nodes_[edge.rows[0]];
  const auto& b = nodes_[edge.rows[1]];
  const auto dm = mean_rate_gradient(a, b, edge.mean_rate);
  const double coordinate = (action * edge.mean_rate) / edge.distance;
  const double rate_a = (action * edge.distance) * dm[0];
  const double rate_b = (action * edge.distance) * dm[1];
  for (int k = 0; k < 3; ++k) {
    const double direction = coordinate * (b.point[k] - a.point[k]);
    if (a.dof[k] >= 0) emit(a.dof[k], a.size * (rate_a * a.sample.gradient[k] - direction));
    if (b.dof[k] >= 0) emit(b.dof[k], b.size * (rate_b * b.sample.gradient[k] + direction));
  }
}

template <class Emit>
void ScalarRepair::shape_gradient(const BoxCell& cell, const CellGeometry& t,
                                  const ShapeConstraint& shape, double action, Emit&& emit) const {
  // The action multiplies the derivative of the deficit, i.e. -q^(3/2).
  const auto dv = cross(t.w, t.u);
  const auto dw = cross(t.u, t.v);
  const double volume = -action * shape.inverse_denominator;
  const double scale = (12.0 * action * shape.fatness) * shape.inverse_squares;
  const auto add = [&](int corner, int k, double derivative) {
    const auto& node = nodes_[cell[corner]];
    if (node.dof[k] >= 0) emit(node.dof[k], node.size * derivative);
  };
  for (int k = 0; k < 3; ++k) {
    const double center = 0.25 * (t.u[k] + t.v[k] + t.w[k]);
    add(0, k, -volume * (t.du[k] + dv[k] + dw[k]) - scale * center);
    add(1, k, volume * t.du[k] + scale * (t.u[k] - center));
    add(2, k, volume * dv[k] + scale * (t.v[k] - center));
    add(3, k, volume * dw[k] + scale * (t.w[k] - center));
  }
}

double ScalarRepair::evaluate(std::span<const double> x, std::span<double> gradient,
                              double shape_margin) {
  const bool force = !cache_valid_;
  cache_valid_ = false;
  feasible_ = interior_ = false;
  if (!visit_rows(nodes_.size(), active_nodes_, [&](std::size_t row) {
        return update_node(row, x, force);
      })) return kInvalidMerit;
  std::fill(gradient.begin(), gradient.end(), 0.0);
  const auto emit = [&](Row dof, double derivative) { gradient[dof] += derivative; };
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
    if (const double action = high - low; !gradient.empty() && action != 0.0) {
      edge_gradient(edge, action, emit);
    }
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
    double action = limit - shape.fatness;
    if (changed || (!gradient.empty() && action > 0.0)) {
      const CellGeometry t(a.point, b.point, c.point, d.point);
      if (changed) {
        if (!update_shape(t, limit + shape_margin, shape)) return false;
        action = limit - shape.fatness;
      }
      if (!gradient.empty() && action > 0.0) shape_gradient(cell, t, shape, action, emit);
    }
    if (action > 0.0) value += action * action;
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

std::size_t ScalarRepair::select_linear_columns(std::span<Row> columns, std::vector<Row>& dofs) const {
  for (const Row dof : dofs) columns[dof] = -1;
  dofs.clear();
  const auto select = [&](Row vertex) {
    for (Row dof : nodes_[vertex].dof) {
      if (dof < 0 || columns[dof] >= 0) continue;
      columns[dof] = static_cast<Row>(dofs.size());
      dofs.push_back(dof);
    }
  };
  std::size_t violations = 0;
  visit_rows(edges_.size(), active_edges_, [&](std::size_t i) {
    const auto& edge = edges_[i];
    if (edge.length < target_.low || edge.length > target_.high) {
      ++violations;
      for (Row row : edge.rows) select(row);
    }
    return true;
  });
  visit_rows(shapes_.size(), active_cells_, [&](std::size_t i) {
    if (shapes_[i].fatness < shape_target_) {
      ++violations;
      for (Row row : sites_.cells[i]) select(row);
    }
    return true;
  });
  return violations;
}

bool ScalarRepair::linearize(LinearModel& model, std::span<const Row> columns) const {
  model.rows.clear();
  model.jacobian.clear();
  const auto moves = [&](Row row) {
    for (Row dof : nodes_[row].dof) {
      if (dof >= 0 && columns[dof] >= 0) return true;
    }
    return false;
  };
  const auto edge_row = [&](std::size_t i) {
    const auto& edge = edges_[i];
    const Band interval{target_.low - edge.length, target_.high - edge.length};
    if ((interval.low <= -kLinearEdgeMargin && interval.high >= kLinearEdgeMargin) ||
        (!moves(edge.rows[0]) && !moves(edge.rows[1]))) return true;
    return model.add(interval, columns, [&](const auto& emit) { edge_gradient(edge, 1.0, emit); });
  };
  const auto shape_row = [&](std::size_t i) {
    const auto& shape = shapes_[i];
    const auto& cell = sites_.cells[i];
    if (shape.fatness >= shape_target_ + kLinearShapeMargin ||
        std::none_of(cell.begin(), cell.end(), moves)) return true;
    const CellGeometry t(nodes_[cell[0]].point, nodes_[cell[1]].point,
                         nodes_[cell[2]].point, nodes_[cell[3]].point);
    return model.add({-std::numeric_limits<double>::infinity(), shape.fatness - shape_target_},
                     columns, [&](const auto& emit) { shape_gradient(cell, t, shape, 1.0, emit); });
  };
  return visit_rows(edges_.size(), active_edges_, edge_row) &&
         visit_rows(shapes_.size(), active_cells_, shape_row);
}

LinearEnd ScalarRepair::linearized_repair(std::span<double> x) {
  double merit = evaluate(x, {}, kLinearShapeMargin);
  if (!std::isfinite(merit)) return LinearEnd::kInvalid;
  if (interior_ || x.empty()) return LinearEnd::kUnchanged;
  std::vector<Row> columns(x.size(), -1), dofs;
  std::vector<Band> trust;
  std::vector<double> shift;
  std::unique_ptr<double[]> storage;
  std::size_t capacity = 0;
  LinearModel model;
  auto end = LinearEnd::kUnchanged;
  for (int iteration = 0; iteration < kLinearModels; ++iteration) {
    const auto violations = select_linear_columns(columns, dofs);
    // This predictor is for a defect tail, not another dense repair. Skip
    // widespread defects before constructing or solving any frozen model.
    if (dofs.empty() ||
        violations > std::max<std::size_t>(256, (edges_.size() + shapes_.size()) / 32) ||
        !linearize(model, columns) || model.rows.empty()) break;
    trust.clear();
    for (const Row dof : dofs) {
      const auto [lo, hi] = bounds_[dof];
      trust.push_back({std::max(lo - x[dof], -kLinearTrust), std::min(hi - x[dof], kLinearTrust)});
    }
    shift.assign(dofs.size(), 0.0);
    const auto count = kWorkBuffers * dofs.size();
    if (count > capacity) {
      storage = std::make_unique_for_overwrite<double[]>(count);
      capacity = count;
    }
    const auto result = detail::projected_bounded_lbfgs(
        std::span<double>(shift), std::span<const Band>(trust), 0.35, kLinearSteps, 1e-9,
        std::span<double>(storage.get(), count),
        [&](std::span<const double> trial, std::span<double> gradient) {
          return model.evaluate(trial, gradient);
        }, [] { return false; });
    if (result.status == detail::BoundedLbfgsStatus::kInvalidObjective || result.steps == 0) break;

    // The frozen model is never a feasibility certificate. Re-sample P1
    // carriers and check all incident nonlinear rows before keeping its step.
    for (std::size_t i = 0; i < dofs.size(); ++i) {
      const Row dof = dofs[i];
      const auto [lo, hi] = bounds_[dof];
      storage[i] = x[dof];
      x[dof] = std::clamp(x[dof] + shift[i], lo, hi);
    }
    const double next = evaluate(x, {}, kLinearShapeMargin);
    if (!std::isfinite(next) || !(next < merit)) {
      for (std::size_t i = 0; i < dofs.size(); ++i) x[dofs[i]] = storage[i];
      if (!std::isfinite(evaluate(x, {}, kLinearShapeMargin))) return LinearEnd::kInvalid;
      break;
    }
    end = LinearEnd::kAdvanced;
    if (interior_ || next > 0.9 * merit) break;
    merit = next;
  }
  return end;
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

enum class RepairEnd { kAccepted, kRetiled, kLocalize, kRefused, kRetryDense };

static RepairEnd repair_connectivity(const ScalarBoxField& field, ScalarBoxSites& sites,
                                     double lower, bool allow_retile, bool local, bool linearize) {
  if (sites.points.empty() || sites.points.size() > kMaxScalarSites ||
      sites.samples.size() != sites.points.size() || sites.cells.empty() ||
      sites.cells.size() > kMaxScalarCells || !(lower > 0.0 && lower < kUpperLength)) {
    return RepairEnd::kRefused;
  }
  ScalarRepair repair(field, sites, {std::max(lower, kLowerLength), kUpperLength});
  if (!repair.prepare(local, linearize)) return RepairEnd::kRefused;
  const auto failure = repair.sparse() ? RepairEnd::kRetryDense : RepairEnd::kRefused;
  std::vector<double> position(repair.bounds().size(), 0.0);
  if (linearize) {
    const auto end = repair.linearized_repair(position);
    if (end == LinearEnd::kInvalid) return failure;
    if (repair.interior()) {
      repair.publish(sites);
      return RepairEnd::kAccepted;
    }
    if (allow_retile && end == LinearEnd::kAdvanced && !repair.feasible() && repair.retile(sites)) {
      return RepairEnd::kRetiled;
    }
    if (!local && end == LinearEnd::kAdvanced && !repair.feasible()) {
      // The accepted affine predictor has removed the widespread defects.
      // Rebind the true nonlinear solve at the remaining stars rather than
      // allocating a history for all site coordinates.
      repair.publish(sites);
      return RepairEnd::kLocalize;
    }
  }
  const std::size_t n = repair.bounds().size();
  if (n == 0 && linearize && !repair.feasible()) return failure;
  const auto storage = std::make_unique_for_overwrite<double[]>(kWorkBuffers * n);
  const std::span<double> work{storage.get(), kWorkBuffers * n};
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
    auto end = repair_connectivity(field, sites, lower, round < 2, round != 0, true);
    if (end == RepairEnd::kLocalize) {
      end = repair_connectivity(field, sites, lower, round < 2, true, false);
    }
    // A restricted solve is a proposal, not a convergence criterion. Its
    // failure leaves sites untouched; retry from that seed with every DOF.
    if (end == RepairEnd::kRetryDense) {
      end = repair_connectivity(field, sites, lower, round < 2, false, false);
    }
    if (end != RepairEnd::kRetiled) return end == RepairEnd::kAccepted;
  }
  return false;
}

}  // namespace phx::mc::metric
