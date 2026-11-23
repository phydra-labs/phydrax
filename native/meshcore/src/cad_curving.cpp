// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_curving.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {
namespace {

using Point = std::array<double, 3>;
using Edge = std::array<int32_t, 2>;
using Face = std::array<int32_t, 3>;
constexpr int kEdgeCorners[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
constexpr int kFaceCorners[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};

// Jacobian columns of the P2 tetrahedron in the derivative_weights layout of
// phx_mc_quadratic_tetrahedron_validity: column c (d/dx_c on the reference
// tetrahedron), monomial p (1, x, y, z), node (corners 0..3, edges 01 02 03
// 12 13 23). With l0 = 1 - x - y - z and lk = x_k, a corner shape
// l_i (2 l_i - 1) has derivative (4 l_i - 1) dl_i and an edge shape 4 l_i l_j
// has 4 (l_j dl_i + l_i dl_j).
struct DerivativeWeights {
  std::array<int32_t, 120> values{};
  DerivativeWeights() {
    const int linear[4][4] = {{1, -1, -1, -1}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    for (int c = 0; c < 3; ++c) {
      const auto gradient = [c](int i) { return i == 0 ? -1 : (i == c + 1 ? 1 : 0); };
      for (int p = 0; p < 4; ++p) {
        for (int i = 0; i < 4; ++i)
          values[(c * 4 + p) * 10 + i] =
              gradient(i) * (4 * linear[i][p] - (p == 0 ? 1 : 0));
        for (int e = 0; e < 6; ++e) {
          const int i = kEdgeCorners[e][0], j = kEdgeCorners[e][1];
          values[(c * 4 + p) * 10 + 4 + e] =
              4 * (gradient(i) * linear[j][p] + gradient(j) * linear[i][p]);
        }
      }
    }
  }
};

Point point_at(const double* points, int32_t v) noexcept {
  const int64_t i = 3 * static_cast<int64_t>(v);
  return {points[i], points[i + 1], points[i + 2]};
}

Point chord_midpoint(const double* points, int32_t a, int32_t b) noexcept {
  const Point p = point_at(points, a), q = point_at(points, b);
  return {0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
}

struct PoleAxis {
  int32_t face, corner, axis;
};

template <std::size_t N>
std::array<double, N> third_point(const std::array<double, N>& a,
                                  const std::array<double, N>& b) noexcept {
  std::array<double, N> p;
  for (std::size_t i = 0; i < N; ++i) p[i] = std::lerp(a[i], b[i], 1.0 / 3.0);
  return p;
}

template <std::size_t N>
std::array<double, N> centroid(const std::array<double, N>& a, const std::array<double, N>& b,
                              const std::array<double, N>& c) noexcept {
  std::array<double, N> p;
  // A constant chart coordinate stays exact, even on a domain boundary.
  for (std::size_t i = 0; i < N; ++i)
    p[i] = std::lerp(std::midpoint(a[i], b[i]), c[i], 1.0 / 3.0);
  return p;
}

template <std::size_t N>
bool finite(const std::array<double, N>& p) noexcept {
  return std::all_of(p.begin(), p.end(), [](double x) { return std::isfinite(x); });
}

enum class EdgeClaim { skip, lift, conflict };

class CurvingSource {
 public:
  CurvingSource(const cad::CadGeometry& geometry, const CadQuadraticInput& input)
      : geometry_(geometry), input_(input) {}

  template <int Order>
  int32_t validate() const {
    const phx_mc_cad_model& model = geometry_.model();
    const int64_t points = input_.point_count;
    if (points > std::numeric_limits<int32_t>::max() || input_.surface_point_count > points)
      return PHX_MC_INVALID_INPUT;
    for (int64_t i = 0; i < 3 * points; ++i)
      if (!std::isfinite(input_.points[i])) return PHX_MC_NONFINITE_INPUT;
    for (int64_t i = 0; i < 4 * input_.tetrahedron_count; ++i)
      if (input_.tetrahedra[i] < 0 || input_.tetrahedra[i] >= points) return PHX_MC_INVALID_INPUT;
    for (int64_t t = 0; t < input_.tetrahedron_count; ++t) {
      const int32_t* cell = input_.tetrahedra + 4 * t;
      for (int e = 0; e < 6; ++e)
        if (cell[kEdgeCorners[e][0]] == cell[kEdgeCorners[e][1]]) return PHX_MC_INVALID_INPUT;
    }
    for (int64_t i = 0; i < input_.surface_point_count; ++i) {
      const int8_t dimension = input_.vertex_dimensions[i];
      if (dimension < 0 || dimension > 2) return PHX_MC_INVALID_INPUT;
      if constexpr (Order == 3) {
        const int64_t entity_counts[]{model.corner_count, model.curve_count, model.surface_count};
        const int32_t entity = input_.vertex_entities[i];
        if (entity < 0 || entity >= entity_counts[dimension]) return PHX_MC_INVALID_INPUT;
      }
    }
    for (int64_t t = 0; t < input_.triangle_count; ++t) {
      for (int k = 0; k < 3; ++k) {
        const int32_t v = input_.triangles[3 * t + k];
        if (v < 0 || v >= input_.surface_point_count) return PHX_MC_INVALID_INPUT;
      }
      const int32_t face = input_.triangle_patches[t];
      if (face < 0 || face >= model.surface_count) return PHX_MC_INVALID_INPUT;
      for (int k = 0; k < 6; ++k)
        if (!std::isfinite(input_.triangle_uv[6 * t + k])) return PHX_MC_NONFINITE_INPUT;
    }
    for (int64_t e = 0; e < input_.curve_edge_count; ++e) {
      for (int k = 0; k < 2; ++k) {
        const int32_t v = input_.curve_edges[2 * e + k];
        if (v < 0 || v >= input_.surface_point_count) return PHX_MC_INVALID_INPUT;
        if (!std::isfinite(input_.curve_edge_parameters[2 * e + k])) return PHX_MC_NONFINITE_INPUT;
      }
      const int32_t curve = input_.curve_edge_curves[e];
      if (curve < 0 || curve >= model.curve_count) return PHX_MC_INVALID_INPUT;
    }
    return PHX_MC_OK;
  }

  template <class Route>
  void enumerate_edges(std::vector<Edge>& edges, Route route) const {
    const int64_t cells = input_.tetrahedron_count;
    // Pack only the point ID bits in the key; equal edges keep their
    // original order by cell and local edge.
    const auto last_point = static_cast<uint64_t>(std::max(int64_t{1}, input_.point_count - 1));
    const auto bits = std::bit_width(last_point);
    const uint64_t mask = (uint64_t{1} << bits) - 1;
    std::vector<SpatialKey> slots, scratch;
    slots.reserve(static_cast<std::size_t>(6 * cells));
    for (int64_t t = 0; t < cells; ++t) {
      const int32_t* cell = input_.tetrahedra + 4 * t;
      for (int e = 0; e < 6; ++e) {
        const int32_t a = cell[kEdgeCorners[e][0]], b = cell[kEdgeCorners[e][1]];
        const auto key = (static_cast<uint64_t>(std::min(a, b)) << bits) |
                         static_cast<uint64_t>(std::max(a, b));
        const auto slot = static_cast<uint64_t>(6 * t + e);
        slots.push_back({key, std::bit_cast<int32_t>(static_cast<uint32_t>(slot)),
                         static_cast<uint32_t>(slot >> 32)});
      }
    }
    radix_sort_keys(slots, scratch, true, false);
    for (const SpatialKey& entry : slots) {
      const Edge edge{static_cast<int32_t>(entry.key >> bits),
                      static_cast<int32_t>(entry.key & mask)};
      if (edges.empty() || edges.back() != edge) edges.push_back(edge);
      const uint64_t slot = (static_cast<uint64_t>(entry.round) << 32) |
                            static_cast<uint32_t>(entry.index);
      route(slot, static_cast<int64_t>(edges.size()) - 1);
    }
  }

  template <int Order, class Claim>
  int32_t lift_edges(const std::vector<Edge>& edges, std::vector<Point>& nodes, Claim claim) const {
    constexpr int stride = Order - 1;
    cad::EvaluationWorkspace workspace;
    for (int64_t row = 0; row < input_.curve_edge_count; ++row) {
      const int32_t a = input_.curve_edges[2 * row], b = input_.curve_edges[2 * row + 1];
      const int64_t edge = find_edge(edges, a, b);
      if (edge < 0) return PHX_MC_INVALID_INPUT;
      const int32_t curve = input_.curve_edge_curves[row];
      const EdgeClaim choice = claim(edge, PHX_MC_CAD_EDGE_CURVE, curve);
      if (choice == EdgeClaim::conflict) return PHX_MC_INVALID_INPUT;
      if (choice == EdgeClaim::skip) continue;
      const auto [ta, tb] = lifted_curve_parameters(row, curve);
      const int32_t status = lift_edge<Order>(geometry_.curve_node(curve), std::array{ta},
                                             std::array{tb}, a > b, nodes.data() + stride * edge,
                                             workspace);
      if (status != PHX_MC_OK) return status;
    }
    std::vector<PoleAxis> poles;
    const int32_t status = pole_axes<Order>(poles);
    if (status != PHX_MC_OK) return status;
    for (int64_t row = 0; row < input_.triangle_count; ++row) {
      const int32_t face = input_.triangle_patches[row];
      const int32_t* triangle = input_.triangles + 3 * row;
      const double* uv = input_.triangle_uv + 6 * row;
      for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3;
        const int32_t a = triangle[i], b = triangle[j];
        const int64_t edge = find_edge(edges, a, b);
        if (edge < 0) return PHX_MC_INVALID_INPUT;
        const EdgeClaim choice = claim(edge, PHX_MC_CAD_EDGE_SURFACE, face);
        if (choice == EdgeClaim::conflict) return PHX_MC_INVALID_INPUT;
        if (choice == EdgeClaim::skip) continue;
        std::array<double, 2> p{uv[2 * i], uv[2 * i + 1]}, q{uv[2 * j], uv[2 * j + 1]};
        const int pa = pole_axis(poles, face, a), pb = pole_axis(poles, face, b);
        if (pa >= 0 && pb < 0) p[pa] = q[pa];
        if (pb >= 0 && pa < 0) q[pb] = p[pb];
        const int32_t placed = lift_edge<Order>(geometry_.surface_node(face), p, q, a > b,
                                               nodes.data() + stride * edge, workspace);
        if (placed != PHX_MC_OK) return placed;
      }
    }
    return PHX_MC_OK;
  }

 private:
  static int64_t find_edge(const std::vector<Edge>& edges, int32_t a, int32_t b) {
    const Edge key{std::min(a, b), std::max(a, b)};
    const auto it = std::lower_bound(edges.begin(), edges.end(), key);
    return it != edges.end() && *it == key ? static_cast<int64_t>(it - edges.begin()) : -1;
  }

  std::array<double, 2> lifted_curve_parameters(int64_t row, int32_t curve) const {
    const phx_mc_cad_model& model = geometry_.model();
    double ta = input_.curve_edge_parameters[2 * row], tb = input_.curve_edge_parameters[2 * row + 1];
    const int64_t offset = 2 * static_cast<int64_t>(curve);
    const double first = model.curve_ranges[offset], last = model.curve_ranges[offset + 1];
    const double period = std::abs(last - first);
    if (model.curve_corners[offset] == model.curve_corners[offset + 1] &&
        std::abs(tb - ta) > 0.5 * period) {
      (ta < tb ? ta : tb) += period;
    }
    return {ta, tb};
  }

  template <int Order, std::size_t N>
  int32_t lift_edge(int32_t carrier, const std::array<double, N>& a, const std::array<double, N>& b,
                    bool reverse, Point* nodes, cad::EvaluationWorkspace& workspace) const {
    if constexpr (Order == 2) {
      std::array<double, N> middle;
      for (std::size_t i = 0; i < N; ++i) middle[i] = 0.5 * (a[i] + b[i]);
      Point xyz;
      if (!geometry_.jets(carrier, middle.data(), 0, xyz.data(), workspace))
        return PHX_MC_INVALID_INPUT;
      nodes[0] = xyz;
    } else {
      // Compute in physical endpoint order, but never go outside the owning
      // chart interval even for equal parameters or at rounding limits.
      const auto& p = reverse ? b : a;
      const auto& q = reverse ? a : b;
      if (!finite(p) || !finite(q)) return PHX_MC_NONFINITE_INPUT;
      const auto near_p = third_point(p, q);
      const auto near_q = third_point(q, p);
      if (!geometry_.jets(carrier, near_p.data(), 0, nodes[0].data(), workspace) ||
          !geometry_.jets(carrier, near_q.data(), 0, nodes[1].data(), workspace))
        return PHX_MC_INVALID_INPUT;
      if (!finite(nodes[0]) || !finite(nodes[1])) return PHX_MC_NONFINITE_INPUT;
    }
    return PHX_MC_OK;
  }

  // Free p-curve axis of each pole use, keyed by (face, corner).
  template <int Order>
  int32_t pole_axes(std::vector<PoleAxis>& poles) const {
    const phx_mc_cad_model& model = geometry_.model();
    cad::EvaluationWorkspace workspace;
    for (int64_t loop = 0; loop < model.loop_count; ++loop) {
      for (int64_t use = model.loop_use_offsets[loop]; use < model.loop_use_offsets[loop + 1]; ++use) {
        if (model.use_kinds[use] != 1) continue;
        const double t = model.use_ranges[2 * use];
        double jet[4];
        if (!geometry_.jets(geometry_.pcurve_node(use), &t, 1, jet, workspace))
          return PHX_MC_INVALID_INPUT;
        if constexpr (Order == 3) {
          if (!std::all_of(jet, jet + 4, [](double x) { return std::isfinite(x); }))
            return PHX_MC_NONFINITE_INPUT;
        }
        poles.push_back({model.loop_surfaces[loop], model.use_entities[use],
                         std::abs(jet[2]) >= std::abs(jet[3]) ? 0 : 1});
      }
    }
    return PHX_MC_OK;
  }

  int pole_axis(const std::vector<PoleAxis>& poles, int32_t face, int32_t vertex) const {
    if (input_.vertex_dimensions[vertex] != 0) return -1;
    for (const PoleAxis& pole : poles)
      if (pole.face == face && pole.corner == input_.vertex_entities[vertex]) return pole.axis;
    return -1;
  }

  const cad::CadGeometry& geometry_;
  const CadQuadraticInput& input_;
};

class QuadraticCurving {
 public:
  QuadraticCurving(const CadQuadraticInput& input, CadQuadraticMesh& result)
      : input_(input), result_(result) {}

  int32_t run(const cad::CadGeometry& geometry) {
    const CurvingSource source(geometry, input_);
    int32_t status = source.validate<2>();
    if (status != PHX_MC_OK) return status;
    result_.cell_edges.resize(static_cast<std::size_t>(input_.tetrahedron_count));
    source.enumerate_edges(result_.edges, [&](uint64_t slot, int64_t edge) {
      result_.cell_edges[static_cast<std::size_t>(slot / 6)][slot % 6] = edge;
    });
    status = place_nodes(source);
    if (status != PHX_MC_OK) return status;
    return repair();
  }

 private:
  void prepare_incidence() {
    // Edge-to-cell incidence (CSR) in ascending cell order.
    const std::size_t edges = result_.edges.size();
    cell_offsets_.assign(edges + 1, 0);
    for (const auto& row : result_.cell_edges)
      for (int64_t e : row) ++cell_offsets_[static_cast<std::size_t>(e) + 1];
    for (std::size_t e = 0; e < edges; ++e) cell_offsets_[e + 1] += cell_offsets_[e];
    edge_cells_.resize(static_cast<std::size_t>(cell_offsets_[edges]));
    std::vector<int64_t> cursor(cell_offsets_.begin(), cell_offsets_.end() - 1);
    for (std::size_t t = 0; t < result_.cell_edges.size(); ++t)
      for (int64_t e : result_.cell_edges[static_cast<std::size_t>(t)])
        edge_cells_[static_cast<std::size_t>(cursor[static_cast<std::size_t>(e)]++)] =
            static_cast<int64_t>(t);
  }

  int32_t place_nodes(const CurvingSource& source) {
    const std::size_t edges = result_.edges.size();
    result_.nodes.resize(edges);
    result_.edge_kinds.assign(edges, PHX_MC_CAD_EDGE_INTERIOR);
    result_.edge_entities.assign(edges, -1);
    curved_.assign(edges, 0);
    for (std::size_t e = 0; e < edges; ++e)
      result_.nodes[e] = chord_midpoint(input_.points, result_.edges[e][0], result_.edges[e][1]);
    const int32_t status = source.lift_edges<2>(
        result_.edges, result_.nodes, [&](int64_t edge, int8_t kind, int32_t entity) {
          if (result_.edge_kinds[edge] != PHX_MC_CAD_EDGE_INTERIOR) return EdgeClaim::skip;
          result_.edge_kinds[edge] = kind;
          result_.edge_entities[edge] = entity;
          curved_[edge] = 1;
          return EdgeClaim::lift;
        });
    if (status != PHX_MC_OK) return status;
    for (std::size_t e = 0; e < edges; ++e) {
      if (result_.edge_kinds[e] == PHX_MC_CAD_EDGE_CURVE) ++result_.stats.curve_edges;
      else if (result_.edge_kinds[e] == PHX_MC_CAD_EDGE_SURFACE) ++result_.stats.surface_edges;
      else ++result_.stats.interior_edges;
    }
    return PHX_MC_OK;
  }

  // Decide `cells` (ascending) into result_.cell_decisions.
  int32_t decide(const std::vector<int64_t>& cells) {
    if (cells.empty()) return PHX_MC_OK;
    static const DerivativeWeights weights;
    buffer_.resize(30 * cells.size());
    for (std::size_t k = 0; k < cells.size(); ++k) {
      const int32_t* cell = input_.tetrahedra + 4 * cells[k];
      double* row = buffer_.data() + 30 * k;
      for (int i = 0; i < 4; ++i) {
        const double* point = input_.points + 3 * static_cast<int64_t>(cell[i]);
        for (int c = 0; c < 3; ++c) row[3 * i + c] = point[c];
      }
      const auto& edges = result_.cell_edges[static_cast<std::size_t>(cells[k])];
      for (int e = 0; e < 6; ++e)
        for (int c = 0; c < 3; ++c) row[12 + 3 * e + c] = result_.nodes[edges[e]][c];
    }
    decisions_.resize(cells.size());
    const int32_t status = phx_mc_quadratic_tetrahedron_validity(
        static_cast<int64_t>(cells.size()), buffer_.data(), weights.values.data(),
        input_.relative_floor, decisions_.data(), nullptr, nullptr);
    if (status != PHX_MC_OK) return status;
    for (std::size_t k = 0; k < cells.size(); ++k)
      result_.cell_decisions[static_cast<std::size_t>(cells[k])] = decisions_[k];
    return PHX_MC_OK;
  }

  bool valid(int64_t cell) const {
    return result_.cell_decisions[static_cast<std::size_t>(cell)] == PHX_MC_QUADRATIC_VALIDITY_VALID;
  }

  // Ascending unique cells incident to `edges`.
  std::vector<int64_t> incident_cells(const std::vector<int64_t>& edges) const {
    std::vector<int64_t> cells;
    for (int64_t e : edges)
      for (int64_t k = cell_offsets_[e]; k < cell_offsets_[e + 1]; ++k)
        cells.push_back(edge_cells_[static_cast<std::size_t>(k)]);
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
  }

  // One Jacobi pass of harmonic extension of the boundary displacement over
  // the interior nodes of the failing cells and their edge neighbors: each
  // such node takes the mean displacement of the other edges of its cells.
  int32_t harmonic_pass(const std::vector<int64_t>& failing) {
    std::vector<int64_t> failing_edges;
    for (int64_t cell : failing)
      for (int64_t e : result_.cell_edges[static_cast<std::size_t>(cell)]) failing_edges.push_back(e);
    std::vector<int64_t> moved;
    for (int64_t cell : incident_cells(failing_edges))
      for (int64_t e : result_.cell_edges[static_cast<std::size_t>(cell)])
        if (result_.edge_kinds[e] == PHX_MC_CAD_EDGE_INTERIOR) moved.push_back(e);
    std::sort(moved.begin(), moved.end());
    moved.erase(std::unique(moved.begin(), moved.end()), moved.end());
    const auto displacement = [&](int64_t e) {
      const Point middle = chord_midpoint(input_.points, result_.edges[e][0], result_.edges[e][1]);
      const Point& node = result_.nodes[e];
      return Point{node[0] - middle[0], node[1] - middle[1], node[2] - middle[2]};
    };
    std::vector<Point> updated(moved.size());
    for (std::size_t k = 0; k < moved.size(); ++k) {
      const int64_t e = moved[k];
      Point sum{0.0, 0.0, 0.0};
      int64_t count = 0;
      for (int64_t i = cell_offsets_[e]; i < cell_offsets_[e + 1]; ++i) {
        for (int64_t other : result_.cell_edges[static_cast<std::size_t>(edge_cells_[i])]) {
          if (other == e) continue;
          const Point d = displacement(other);
          for (int c = 0; c < 3; ++c) sum[c] += d[c];
          ++count;
        }
      }
      const Point middle = chord_midpoint(input_.points, result_.edges[e][0], result_.edges[e][1]);
      for (int c = 0; c < 3; ++c) updated[k][c] = middle[c] + sum[c] / static_cast<double>(count);
    }
    std::vector<int64_t> changed;
    for (std::size_t k = 0; k < moved.size(); ++k) {
      const int64_t e = moved[k];
      if (updated[k] == result_.nodes[e]) continue;
      result_.nodes[e] = updated[k];
      curved_[e] = 1;
      changed.push_back(e);
    }
    result_.stats.harmonic_nodes = static_cast<int64_t>(changed.size());
    return decide(incident_cells(changed));
  }

  int32_t repair() {
    const int64_t cells = input_.tetrahedron_count;
    result_.cell_decisions.assign(static_cast<std::size_t>(cells), PHX_MC_QUADRATIC_VALIDITY_VALID);
    std::vector<int64_t> all(static_cast<std::size_t>(cells));
    for (int64_t t = 0; t < cells; ++t) all[static_cast<std::size_t>(t)] = t;
    int32_t status = decide(all);
    if (status != PHX_MC_OK) return status;
    result_.initial_decisions = result_.cell_decisions;
    std::vector<int64_t> failing;
    for (int64_t t = 0; t < cells; ++t) {
      const int32_t decision = result_.cell_decisions[static_cast<std::size_t>(t)];
      if (decision == PHX_MC_QUADRATIC_VALIDITY_INVALID) ++result_.stats.initial_invalid;
      if (decision == PHX_MC_QUADRATIC_VALIDITY_UNRESOLVED) ++result_.stats.initial_unresolved;
      if (decision != PHX_MC_QUADRATIC_VALIDITY_VALID) failing.push_back(t);
    }
    if (!failing.empty()) {
      prepare_incidence();
      status = harmonic_pass(failing);
      if (status != PHX_MC_OK) return status;
      for (int64_t t : failing) result_.stats.harmonic_repaired_cells += valid(t) ? 1 : 0;
    }
    // Straighten every curved node of each failing cell with a curved node,
    // in cell order, until none remains.
    const auto has_curved = [&](int64_t cell) {
      for (int64_t e : result_.cell_edges[static_cast<std::size_t>(cell)])
        if (curved_[e]) return true;
      return false;
    };
    std::vector<int64_t> queue;
    for (int64_t t = 0; t < cells; ++t)
      if (!valid(t) && has_curved(t)) queue.push_back(t);
    while (!queue.empty()) {
      std::vector<int64_t> changed;
      for (int64_t cell : queue) {
        for (int64_t e : result_.cell_edges[static_cast<std::size_t>(cell)]) {
          if (!curved_[e]) continue;
          curved_[e] = 0;
          result_.nodes[e] = chord_midpoint(input_.points, result_.edges[e][0], result_.edges[e][1]);
          if (result_.edge_kinds[e] != PHX_MC_CAD_EDGE_INTERIOR)
            result_.straightened.push_back({e, cell});
          changed.push_back(e);
        }
      }
      status = decide(incident_cells(changed));
      if (status != PHX_MC_OK) return status;
      queue.clear();
      for (int64_t cell : incident_cells(changed))
        if (!valid(cell) && has_curved(cell)) queue.push_back(cell);
    }
    result_.stats.straightened_edges = static_cast<int64_t>(result_.straightened.size());
    for (int32_t decision : result_.cell_decisions) {
      if (decision == PHX_MC_QUADRATIC_VALIDITY_INVALID) ++result_.stats.final_invalid;
      if (decision == PHX_MC_QUADRATIC_VALIDITY_UNRESOLVED) ++result_.stats.final_unresolved;
    }
    return PHX_MC_OK;
  }

  const CadQuadraticInput& input_;
  CadQuadraticMesh& result_;
  std::vector<int64_t> cell_offsets_, edge_cells_;
  std::vector<uint8_t> curved_;
  std::vector<double> buffer_;
  std::vector<int32_t> decisions_;
};

bool cubic_banks_present(const CadQuadraticInput& input) {
  const auto present = [](const void* p, int64_t count) { return count == 0 || p != nullptr; };
  return addressable(input.point_count, 3, sizeof(double)) &&
         addressable(input.surface_point_count, 1, sizeof(int32_t)) &&
         addressable(input.tetrahedron_count, 16, sizeof(int64_t)) &&
         addressable(input.triangle_count, 6, sizeof(double)) &&
         addressable(input.curve_edge_count, 2, sizeof(double)) &&
         present(input.points, input.point_count) &&
         present(input.vertex_dimensions, input.surface_point_count) &&
         present(input.vertex_entities, input.surface_point_count) &&
         present(input.tetrahedra, input.tetrahedron_count) &&
         present(input.triangles, input.triangle_count) &&
         present(input.triangle_patches, input.triangle_count) &&
         present(input.triangle_uv, input.triangle_count) &&
         present(input.curve_edges, input.curve_edge_count) &&
         present(input.curve_edge_curves, input.curve_edge_count) &&
         present(input.curve_edge_parameters, input.curve_edge_count);
}

Face sorted_face(Face corners) {
  std::sort(corners.begin(), corners.end());
  return corners;
}

std::array<double, 2> chart_centroid(const int32_t* triangle, const double* uv) {
  std::array<int, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](int a, int b) { return triangle[a] < triangle[b]; });
  const auto corner = [&](int i) {
    return std::array{uv[2 * order[i]], uv[2 * order[i] + 1]};
  };
  return centroid(corner(0), corner(1), corner(2));
}

int32_t place_cubic_faces(const cad::CadGeometry& geometry, const CadQuadraticInput& input,
                          std::size_t edge_nodes, CadCubicMesh& result) {
  // Sort physical triples with CAD chart declarations (negative slots) before
  // cell incidences. A CAD interface may be owned by two cells; the exterior
  // must have a chart, and a face cannot declare two distinct owning rows.
  using FaceSlot = std::pair<Face, int64_t>;
  std::vector<FaceSlot> slots;
  const auto n = static_cast<std::size_t>(4 * input.tetrahedron_count + input.triangle_count);
  if (n > slots.max_size()) return PHX_MC_INDEX_RANGE_ERROR;
  slots.reserve(n);
  for (int64_t t = 0; t < input.tetrahedron_count; ++t) {
    const int32_t* cell = input.tetrahedra + 4 * t;
    for (int f = 0; f < 4; ++f) {
      const int* corners = kFaceCorners[f];
      const Face key = sorted_face({cell[corners[0]], cell[corners[1]], cell[corners[2]]});
      slots.emplace_back(key, 4 * t + f);
    }
  }
  for (int64_t t = 0; t < input.triangle_count; ++t) {
    const int32_t* triangle = input.triangles + 3 * t;
    slots.emplace_back(sorted_face({triangle[0], triangle[1], triangle[2]}), ~t);
  }
  std::sort(slots.begin(), slots.end());
  std::size_t face_count = 0;
  for (std::size_t i = 0; i < slots.size(); ++i)
    if (i == 0 || slots[i - 1].first != slots[i].first) ++face_count;
  if (edge_nodes > result.nodes.max_size() ||
      face_count > result.nodes.max_size() - edge_nodes)
    return PHX_MC_INDEX_RANGE_ERROR;
  result.nodes.resize(edge_nodes + face_count);

  cad::EvaluationWorkspace workspace;
  std::size_t node = edge_nodes;
  for (std::size_t first = 0; first < slots.size(); ++node) {
    const auto& [key, owner] = slots[first];
    const bool on_cad = owner < 0;
    const std::size_t begin = first + on_cad;
    std::size_t end = first + 1;
    while (end < slots.size() && slots[end].first == key) ++end;
    const std::size_t incidence = end - begin;
    if (incidence == 0 || incidence > 2 || (!on_cad && incidence == 1) ||
        slots[begin].second < 0)
      return PHX_MC_INVALID_INPUT;

    Point& xyz = result.nodes[node];
    if (on_cad) {
      const int64_t row = ~owner;
      const auto uv = chart_centroid(input.triangles + 3 * row, input.triangle_uv + 6 * row);
      const int32_t face = geometry.surface_node(input.triangle_patches[row]);
      if (!geometry.jets(face, uv.data(), 0, xyz.data(), workspace)) return PHX_MC_INVALID_INPUT;
    } else {
      xyz = centroid(point_at(input.points, key[0]), point_at(input.points, key[1]),
                     point_at(input.points, key[2]));
    }
    if (!finite(xyz)) return PHX_MC_NONFINITE_INPUT;
    for (std::size_t i = begin; i < end; ++i) {
      const auto slot = static_cast<std::size_t>(slots[i].second);
      result.cell_nodes[slot / 4][12 + slot % 4] = static_cast<int64_t>(node);
    }
    first = end;
  }
  return PHX_MC_OK;
}

int32_t place_cubic_edges(const CurvingSource& source, const CadQuadraticInput& input,
                          const std::vector<Edge>& edges, std::vector<Point>& nodes) {
  constexpr int32_t interior = -1, unused_curve = -2, used_curve = -3;
  // Nonnegative values are the single owning patch of a non-curve edge.
  std::vector<int32_t> owners(edges.size(), interior);
  const int32_t status = source.lift_edges<3>(
      edges, nodes, [&](int64_t edge, int8_t kind, int32_t entity) {
        int32_t& owner = owners[edge];
        if (kind == PHX_MC_CAD_EDGE_CURVE) {
          if (owner != interior) return EdgeClaim::conflict;
          owner = unused_curve;
        } else if (owner == unused_curve || owner == used_curve) {
          owner = used_curve;
          return EdgeClaim::skip;
        } else if (owner != interior) {
          return owner == entity ? EdgeClaim::skip : EdgeClaim::conflict;
        } else {
          owner = entity;
        }
        return EdgeClaim::lift;
      });
  if (status != PHX_MC_OK) return status;
  for (std::size_t e = 0; e < edges.size(); ++e) {
    if (owners[e] == unused_curve) return PHX_MC_INVALID_INPUT;
    if (owners[e] != interior) continue;
    const Point a = point_at(input.points, edges[e][0]), b = point_at(input.points, edges[e][1]);
    nodes[2 * e] = third_point(a, b);
    nodes[2 * e + 1] = third_point(b, a);
  }
  return PHX_MC_OK;
}

}  // namespace

int32_t cad_quadratic_nodes(const cad::CadGeometry& geometry, const CadQuadraticInput& input,
                            CadQuadraticMesh& result) {
  result = CadQuadraticMesh{};
  return QuadraticCurving(input, result).run(geometry);
}

int32_t cad_cubic_nodes(const cad::CadGeometry& geometry, const CadQuadraticInput& input,
                        CadCubicMesh& result) {
  result = CadCubicMesh{};
  if (!cubic_banks_present(input)) return PHX_MC_INVALID_ARGUMENT;
  const CurvingSource source(geometry, input);
  const int32_t valid_input = source.validate<3>();
  if (valid_input != PHX_MC_OK) return valid_input;

  std::vector<Edge> edges;
  result.cell_nodes.resize(static_cast<std::size_t>(input.tetrahedron_count));
  source.enumerate_edges(edges, [&](uint64_t slot, int64_t edge) {
    const auto t = static_cast<std::size_t>(slot / 6);
    const auto e = static_cast<std::size_t>(slot % 6);
    const int32_t* cell = input.tetrahedra + 4 * t;
    const bool reverse = cell[kEdgeCorners[e][0]] > cell[kEdgeCorners[e][1]];
    result.cell_nodes[t][2 * e] = 2 * edge + reverse;
    result.cell_nodes[t][2 * e + 1] = 2 * edge + !reverse;
  });
  const int32_t faces = place_cubic_faces(geometry, input, 2 * edges.size(), result);
  if (faces != PHX_MC_OK) return faces;
  return place_cubic_edges(source, input, edges, result.nodes);
}

}  // namespace phx::mc

extern "C" int32_t phx_mc_cad_quadratic_nodes(const phx_mc_cad_quadratic_request* request) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!request || !request->model || !request->edge_count || !request->record_count)
      return PHX_MC_INVALID_ARGUMENT;
    const phx_mc_cad_quadratic_request& r = *request;
    using phx::mc::addressable;
    if (!addressable(r.point_count, 3, sizeof(double)) ||
        !addressable(r.tetrahedron_count, 6, sizeof(int64_t)) ||
        !addressable(r.triangle_count, 6, sizeof(double)) ||
        !addressable(r.curve_edge_count, 2, sizeof(double)) || r.surface_point_count < 0 ||
        !addressable(r.edge_capacity, 3, sizeof(double)) ||
        !addressable(r.record_capacity, 2, sizeof(int64_t)) || !std::isfinite(r.relative_floor) ||
        r.relative_floor < 0.0 || r.relative_floor >= 1.0)
      return PHX_MC_INVALID_ARGUMENT;
    const auto present = [](const void* p, int64_t n) { return n == 0 || p != nullptr; };
    if (!present(r.points, r.point_count) || !present(r.vertex_dimensions, r.surface_point_count) ||
        !present(r.vertex_entities, r.surface_point_count) ||
        !present(r.tetrahedra, r.tetrahedron_count) || !present(r.triangles, r.triangle_count) ||
        !present(r.triangle_patches, r.triangle_count) ||
        !present(r.triangle_uv, r.triangle_count) || !present(r.curve_edges, r.curve_edge_count) ||
        !present(r.curve_edge_curves, r.curve_edge_count) ||
        !present(r.curve_edge_parameters, r.curve_edge_count) ||
        !present(r.cell_edges, r.tetrahedron_count) ||
        !present(r.cell_decisions, r.tetrahedron_count) || !present(r.edges, r.edge_capacity) ||
        !present(r.nodes, r.edge_capacity) || !present(r.edge_kinds, r.edge_capacity) ||
        !present(r.edge_entities, r.edge_capacity) || !present(r.records, r.record_capacity))
      return PHX_MC_INVALID_ARGUMENT;
    phx::mc::cad::CadGeometry geometry;
    const int32_t bound = geometry.bind(*r.model);
    if (bound != PHX_MC_OK) return bound;
    const phx::mc::CadQuadraticInput input{
        r.point_count, r.points, r.surface_point_count, r.vertex_dimensions, r.vertex_entities,
        r.tetrahedron_count, r.tetrahedra, r.triangle_count, r.triangles, r.triangle_patches,
        r.triangle_uv, r.curve_edge_count, r.curve_edges, r.curve_edge_curves,
        r.curve_edge_parameters, r.relative_floor};
    phx::mc::CadQuadraticMesh mesh;
    const int32_t status = phx::mc::cad_quadratic_nodes(geometry, input, mesh);
    if (status != PHX_MC_OK) return status;
    const auto edges = static_cast<int64_t>(mesh.edges.size());
    const auto records = static_cast<int64_t>(mesh.straightened.size());
    *r.edge_count = edges;
    *r.record_count = records;
    if (edges > r.edge_capacity || records > r.record_capacity) return PHX_MC_CAPACITY_EXCEEDED;
    for (int64_t e = 0; e < edges; ++e) {
      const auto k = static_cast<std::size_t>(e);
      r.edges[2 * e] = mesh.edges[k][0];
      r.edges[2 * e + 1] = mesh.edges[k][1];
      for (int c = 0; c < 3; ++c) r.nodes[3 * e + c] = mesh.nodes[k][c];
      r.edge_kinds[e] = mesh.edge_kinds[k];
      r.edge_entities[e] = mesh.edge_entities[k];
    }
    for (int64_t t = 0; t < r.tetrahedron_count; ++t) {
      const auto k = static_cast<std::size_t>(t);
      for (int e = 0; e < 6; ++e) r.cell_edges[6 * t + e] = mesh.cell_edges[k][e];
      r.cell_decisions[t] = mesh.cell_decisions[k];
    }
    for (int64_t i = 0; i < records; ++i) {
      r.records[2 * i] = mesh.straightened[static_cast<std::size_t>(i)][0];
      r.records[2 * i + 1] = mesh.straightened[static_cast<std::size_t>(i)][1];
    }
    if (r.stats) *r.stats = mesh.stats;
    return PHX_MC_OK;
  });
}
