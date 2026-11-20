// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_curving.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"

namespace phx::mc {
namespace {

using Point = std::array<double, 3>;
constexpr int kEdgeCorners[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

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

uint64_t edge_key(int32_t a, int32_t b) noexcept {
  const auto low = static_cast<uint64_t>(static_cast<uint32_t>(std::min(a, b)));
  const auto high = static_cast<uint64_t>(static_cast<uint32_t>(std::max(a, b)));
  return (low << 32) | high;
}

Point point_at(const double* points, int32_t v) noexcept {
  return {points[3 * v], points[3 * v + 1], points[3 * v + 2]};
}

Point chord_midpoint(const double* points, int32_t a, int32_t b) noexcept {
  const Point p = point_at(points, a), q = point_at(points, b);
  return {0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
}

struct PoleAxis {
  int32_t face, corner, axis;
};

class Curving {
 public:
  Curving(const cad::CadGeometry& geometry, const CadQuadraticInput& input,
          CadQuadraticMesh& result)
      : geometry_(geometry), input_(input), result_(result) {}

  int32_t run() {
    int32_t status = validate();
    if (status != PHX_MC_OK) return status;
    status = enumerate_edges();
    if (status != PHX_MC_OK) return status;
    status = place_nodes();
    if (status != PHX_MC_OK) return status;
    return repair();
  }

 private:
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

  int64_t find_edge(int32_t a, int32_t b) const {
    const uint64_t key = edge_key(a, b);
    const auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
    return it != keys_.end() && *it == key ? static_cast<int64_t>(it - keys_.begin()) : -1;
  }

  int32_t enumerate_edges() {
    const int64_t cells = input_.tetrahedron_count;
    std::vector<std::pair<uint64_t, int64_t>> slots;
    slots.reserve(static_cast<std::size_t>(6 * cells));
    for (int64_t t = 0; t < cells; ++t) {
      const int32_t* cell = input_.tetrahedra + 4 * t;
      for (int e = 0; e < 6; ++e)
        slots.emplace_back(edge_key(cell[kEdgeCorners[e][0]], cell[kEdgeCorners[e][1]]), 6 * t + e);
    }
    std::sort(slots.begin(), slots.end());
    result_.cell_edges.assign(static_cast<std::size_t>(cells), {});
    keys_.clear();
    for (const auto& [key, slot] : slots) {
      if (keys_.empty() || keys_.back() != key) {
        keys_.push_back(key);
        result_.edges.push_back({static_cast<int32_t>(key >> 32),
                                 static_cast<int32_t>(key & 0xffffffffu)});
      }
      result_.cell_edges[static_cast<std::size_t>(slot / 6)][slot % 6] =
          static_cast<int64_t>(keys_.size()) - 1;
    }
    // Edge-to-cell incidence (CSR) in ascending cell order.
    const std::size_t edges = keys_.size();
    cell_offsets_.assign(edges + 1, 0);
    for (const auto& row : result_.cell_edges)
      for (int64_t e : row) ++cell_offsets_[static_cast<std::size_t>(e) + 1];
    for (std::size_t e = 0; e < edges; ++e) cell_offsets_[e + 1] += cell_offsets_[e];
    edge_cells_.resize(static_cast<std::size_t>(cell_offsets_[edges]));
    std::vector<int64_t> cursor(cell_offsets_.begin(), cell_offsets_.end() - 1);
    for (int64_t t = 0; t < cells; ++t)
      for (int64_t e : result_.cell_edges[static_cast<std::size_t>(t)])
        edge_cells_[static_cast<std::size_t>(cursor[static_cast<std::size_t>(e)]++)] = t;
    return PHX_MC_OK;
  }

  // Free p-curve axis of each pole use, keyed by (face, corner).
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
        poles.push_back({model.loop_surfaces[loop], model.use_entities[use],
                         std::abs(jet[2]) >= std::abs(jet[3]) ? 0 : 1});
      }
    }
    return PHX_MC_OK;
  }

  int32_t place_nodes() {
    const phx_mc_cad_model& model = geometry_.model();
    const std::size_t edges = keys_.size();
    result_.nodes.resize(edges);
    result_.edge_kinds.assign(edges, PHX_MC_CAD_EDGE_INTERIOR);
    result_.edge_entities.assign(edges, -1);
    curved_.assign(edges, 0);
    for (std::size_t e = 0; e < edges; ++e)
      result_.nodes[e] = chord_midpoint(input_.points, result_.edges[e][0], result_.edges[e][1]);
    cad::EvaluationWorkspace workspace;
    for (int64_t row = 0; row < input_.curve_edge_count; ++row) {
      const int32_t a = input_.curve_edges[2 * row], b = input_.curve_edges[2 * row + 1];
      const int64_t e = find_edge(a, b);
      if (e < 0) return PHX_MC_INVALID_INPUT;
      if (result_.edge_kinds[e] == PHX_MC_CAD_EDGE_CURVE) continue;
      const int32_t curve = input_.curve_edge_curves[row];
      double ta = input_.curve_edge_parameters[2 * row], tb = input_.curve_edge_parameters[2 * row + 1];
      const double first = model.curve_ranges[2 * curve], last = model.curve_ranges[2 * curve + 1];
      const double period = std::abs(last - first);
      if (model.curve_corners[2 * curve] == model.curve_corners[2 * curve + 1] &&
          std::abs(tb - ta) > 0.5 * period) {
        (ta < tb ? ta : tb) += period;
      }
      const double t = 0.5 * (ta + tb);
      Point xyz;
      if (!geometry_.jets(geometry_.curve_node(curve), &t, 0, xyz.data(), workspace))
        return PHX_MC_INVALID_INPUT;
      result_.nodes[e] = xyz;
      result_.edge_kinds[e] = PHX_MC_CAD_EDGE_CURVE;
      result_.edge_entities[e] = curve;
      curved_[e] = 1;
    }
    std::vector<PoleAxis> poles;
    const int32_t status = pole_axes(poles);
    if (status != PHX_MC_OK) return status;
    const auto pole_axis = [&](int32_t face, int32_t vertex) {
      if (input_.vertex_dimensions[vertex] != 0) return -1;
      for (const PoleAxis& pole : poles)
        if (pole.face == face && pole.corner == input_.vertex_entities[vertex]) return pole.axis;
      return -1;
    };
    for (int64_t t = 0; t < input_.triangle_count; ++t) {
      const int32_t face = input_.triangle_patches[t];
      const double* uv = input_.triangle_uv + 6 * t;
      for (int k = 0; k < 3; ++k) {
        const int i = k, j = (k + 1) % 3;
        const int32_t a = input_.triangles[3 * t + i], b = input_.triangles[3 * t + j];
        const int64_t e = find_edge(a, b);
        if (e < 0) return PHX_MC_INVALID_INPUT;
        if (result_.edge_kinds[e] != PHX_MC_CAD_EDGE_INTERIOR) continue;
        std::array<double, 2> p{uv[2 * i], uv[2 * i + 1]}, q{uv[2 * j], uv[2 * j + 1]};
        const int pa = pole_axis(face, a), pb = pole_axis(face, b);
        if (pa >= 0 && pb < 0) p[pa] = q[pa];
        if (pb >= 0 && pa < 0) q[pb] = p[pb];
        const std::array<double, 2> middle{0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])};
        Point xyz;
        if (!geometry_.jets(geometry_.surface_node(face), middle.data(), 0, xyz.data(), workspace))
          return PHX_MC_INVALID_INPUT;
        result_.nodes[e] = xyz;
        result_.edge_kinds[e] = PHX_MC_CAD_EDGE_SURFACE;
        result_.edge_entities[e] = face;
        curved_[e] = 1;
      }
    }
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
      for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 3; ++c) row[3 * i + c] = input_.points[3 * cell[i] + c];
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
  // the interior nodes of the failing cells and their edge neighbours: each
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

  const cad::CadGeometry& geometry_;
  const CadQuadraticInput& input_;
  CadQuadraticMesh& result_;
  std::vector<uint64_t> keys_;
  std::vector<int64_t> cell_offsets_, edge_cells_;
  std::vector<uint8_t> curved_;
  std::vector<double> buffer_;
  std::vector<int32_t> decisions_;
};

}  // namespace

int32_t cad_quadratic_nodes(const cad::CadGeometry& geometry, const CadQuadraticInput& input,
                            CadQuadraticMesh& result) {
  result = CadQuadraticMesh{};
  return Curving(geometry, input, result).run();
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
