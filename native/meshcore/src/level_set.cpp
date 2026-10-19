//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// P1 zero insertion on an existing tetrahedral complex. Every source face is
// constrained and every source edge is a subsegment: roots cannot replace the
// source interpolation or cross a source material interface. TetMesh owns root
// insertion and validated protected cavity transactions. Only a complete split
// publishes output; a refusal discards the private native state.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "capi_guard.hpp"
#include "counter_records.hpp"
#include "geometry3.hpp"
#include "tet_mesh.hpp"

namespace {
struct LevelSetProvenance {
  std::vector<std::array<int32_t, 2>> sources;
  std::vector<std::array<double, 2>> weights;
};

int32_t validate_level_set_request(const phx_mc_level_set_split_3d_request& request) {
  if (request.point_count < 4 || request.cell_count < 1 || request.face_count < 4 ||
      request.edge_count < 6 || request.max_vertices < request.point_count ||
      request.max_vertices > std::numeric_limits<int32_t>::max() - 1 ||
      request.max_cells < request.cell_count ||
      request.cell_count > std::numeric_limits<int32_t>::max() ||
      request.face_count > std::numeric_limits<int32_t>::max() ||
      request.edge_count > std::numeric_limits<int32_t>::max() || request.work_limit < 0 ||
      request.cavity_limit < 1 || request.points == nullptr || request.values == nullptr ||
      request.cells == nullptr || request.faces == nullptr || request.edges == nullptr ||
      request.protected_edges == nullptr || request.out_points == nullptr ||
      request.out_cells == nullptr || request.out_parents == nullptr ||
      request.out_sources == nullptr || request.out_weights == nullptr ||
      request.counters == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return PHX_MC_OK;
}

int32_t prepare_level_set_sources(const phx_mc_level_set_split_3d_request& request,
                                  std::vector<int32_t>& regions, std::vector<int32_t>& face_sources,
                                  std::vector<int32_t>& edge_sources,
                                  phx::mc::LevelSetCounters& state) {
  for (int64_t i = 0; i < request.point_count; ++i) {
    if (!std::isfinite(request.values[i])) return PHX_MC_INVALID_INPUT;
    state.zero_vertices += request.values[i] == 0.0 ? 1 : 0;
    for (int k = 0; k < 3; ++k) {
      if (!phx::mc::coordinate_in_domain(request.points[3 * i + k])) return PHX_MC_INVALID_INPUT;
    }
  }
  for (int32_t i = 0; i < request.cell_count; ++i) regions[static_cast<std::size_t>(i)] = i;
  for (int32_t i = 0; i < request.face_count; ++i) face_sources[static_cast<std::size_t>(i)] = i;
  for (int32_t i = 0; i < request.edge_count; ++i) edge_sources[static_cast<std::size_t>(i)] = i;
  return PHX_MC_OK;
}

int32_t construct_level_set_root(const phx_mc_level_set_split_3d_request& request, int32_t a,
                                 int32_t b, double& t, double* root) {
  // The unique P1 root in binary64 on every platform. Halving both
  // magnitudes only when their sum overflows keeps every finite-sum case
  // equal to the unscaled quotient.
  const double first = std::abs(request.values[a]);
  const double second = std::abs(request.values[b]);
  const double sum = first + second;
  t = std::isfinite(sum) ? first / sum : (0.5 * first) / (0.5 * first + 0.5 * second);
  for (int k = 0; k < 3; ++k) {
    root[k] = (1.0 - t) * request.points[3 * a + k] + t * request.points[3 * b + k];
    if (!phx::mc::coordinate_in_domain(root[k])) return PHX_MC_INVALID_INPUT;
  }
  // Never perturb the root off a protected source stratum to force success.
  if (!(t > 0.0 && t < 1.0) ||
      !phx::mc::geometry::collinear(request.points + 3 * a, request.points + 3 * b, root, 1)) {
    return PHX_MC_CONSTRAINT_INTERSECTION;
  }
  return PHX_MC_OK;
}

int32_t insert_level_set_roots(const phx_mc_level_set_split_3d_request& request,
                               phx::mc::TetMesh& mesh, LevelSetProvenance& provenance,
                               std::vector<int32_t>& star, int64_t input_work,
                               phx::mc::LevelSetCounters& state) {
  for (int64_t e = 0; e < request.edge_count; ++e) {
    if (!mesh.spend(1)) {
      state.work_units = input_work + mesh.work();
      state.failed_edge = e;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    state.work_units = input_work + mesh.work();
    const int32_t a = request.edges[2 * e], b = request.edges[2 * e + 1];
    if (a < 0 || b < 0 || a >= request.point_count || b >= request.point_count || a == b) {
      return PHX_MC_INVALID_INPUT;
    }
    if (request.values[a] == 0.0 || request.values[b] == 0.0 ||
        std::signbit(request.values[a]) == std::signbit(request.values[b])) {
      continue;
    }
    state.failed_edge = e;
    if (request.protected_edges[e] != 0) return PHX_MC_CONSTRAINT_INTERSECTION;
    double t, root[3];
    const int32_t status = construct_level_set_root(request, a, b, t, root);
    if (status != PHX_MC_OK) return status;
    if (!mesh.vertex_star(a, star)) return PHX_MC_INTERNAL_ERROR;
    int32_t origin = -1;
    for (int32_t cell : star) {
      if (!phx::mc::is_ghost(mesh.tet(cell)) && phx::mc::vertex_slot(mesh.tet(cell), b) >= 0) {
        origin = cell;
        break;
      }
    }
    if (origin < 0) return PHX_MC_INTERNAL_ERROR;
    phx::mc::Insertion insertion = mesh.prepare(root, origin, a, b);
    if (insertion == phx::mc::Insertion::kOk &&
        mesh.cavity().size() > static_cast<std::size_t>(request.cavity_limit)) {
      mesh.abandon();
      state.work_units = input_work + mesh.work();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (insertion == phx::mc::Insertion::kOk) {
      insertion = mesh.commit(phx::mc::InsertKind::kSubsegment, 0.0);
    }
    if (insertion != phx::mc::Insertion::kOk) {
      mesh.abandon();
      state.work_units = input_work + mesh.work();
      return insertion == phx::mc::Insertion::kCapacity ? PHX_MC_CAPACITY_EXCEEDED
                                                        : PHX_MC_CONSTRAINT_INTERSECTION;
    }
    provenance.sources.push_back({a, b});
    provenance.weights.push_back({1.0 - t, t});
    ++state.inserted;
  }
  return PHX_MC_OK;
}

int32_t certify_level_set_cells(const phx_mc_level_set_split_3d_request& request,
                                phx::mc::TetMesh& mesh,
                                const std::vector<std::array<int32_t, 5>>& target) {
  // Every original opposite-sign pair was split as a constrained segment.
  // Thus each finite target simplex must lie wholly on one side (zeros shared).
  for (const auto& cell : target) {
    bool negative = false, positive = false;
    for (int k = 0; k < 4; ++k) {
      const int32_t v = cell[static_cast<std::size_t>(k)];
      if (v < request.point_count) {
        negative = negative || request.values[v] < 0.0;
        positive = positive || request.values[v] > 0.0;
      }
    }
    if (negative == positive) return PHX_MC_INVALID_INPUT;
  }
  if (!mesh.complex().audit()) return PHX_MC_INTERNAL_ERROR;
  return PHX_MC_OK;
}

void publish_level_set_split(const phx_mc_level_set_split_3d_request& request,
                             phx::mc::TetMesh& mesh, const LevelSetProvenance& provenance,
                             const std::vector<std::array<int32_t, 5>>& target, int64_t input_work,
                             phx::mc::LevelSetCounters& state) {
  state.vertices = mesh.vertex_count();
  state.cells = static_cast<int64_t>(target.size());
  state.work_units = input_work + mesh.work();
  state.failed_edge = -1;
  for (int32_t i = 0; i < mesh.vertex_count(); ++i) {
    std::copy_n(mesh.point(i), 3, request.out_points + 3 * i);
    std::copy_n(provenance.sources[static_cast<std::size_t>(i)].data(), 2,
                request.out_sources + 2 * i);
    std::copy_n(provenance.weights[static_cast<std::size_t>(i)].data(), 2,
                request.out_weights + 2 * i);
  }
  for (std::size_t i = 0; i < target.size(); ++i) {
    std::copy_n(target[i].data(), 4, request.out_cells + 4 * i);
    request.out_parents[i] = target[i][4];
  }
}
}  // namespace

extern "C" PHX_MC_API int32_t
phx_mc_level_set_split_3d(const phx_mc_level_set_split_3d_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    int32_t status = validate_level_set_request(*request);
    if (status != PHX_MC_OK) return status;
    phx::mc::LevelSetCounters state;
    phx::mc::CounterPublisher publisher{state, request->counters};
    state.failed_edge = -1;
    const int64_t input_work = request->point_count + 4 * request->cell_count +
                               3 * request->face_count + 2 * request->edge_count;
    if (input_work > request->work_limit) return PHX_MC_CAPACITY_EXCEEDED;
    state.work_units = input_work;
    std::vector<int32_t> regions(static_cast<std::size_t>(request->cell_count));
    std::vector<int32_t> face_sources(static_cast<std::size_t>(request->face_count));
    std::vector<int32_t> edge_sources(static_cast<std::size_t>(request->edge_count));
    status = prepare_level_set_sources(*request, regions, face_sources, edge_sources, state);
    if (status != PHX_MC_OK) return status;
    phx::mc::TetMesh mesh(phx::mc::BoundaryPolicy::kConforming, request->max_vertices,
                          request->max_cells);
    status = mesh.build(request->point_count, request->points, request->cell_count, request->cells,
                        regions.data(), request->face_count, request->faces, face_sources.data(),
                        request->edge_count, request->edges, edge_sources.data(), nullptr);
    if (status != PHX_MC_OK) return status;
    mesh.set_work_limit(request->work_limit - input_work);
    LevelSetProvenance provenance;
    provenance.sources.reserve(static_cast<std::size_t>(
        std::min(request->max_vertices, request->point_count + request->edge_count)));
    provenance.weights.reserve(provenance.sources.capacity());
    for (int32_t i = 0; i < request->point_count; ++i) {
      provenance.sources.push_back({i, i});
      provenance.weights.push_back({1.0, 0.0});
    }
    std::vector<int32_t> star;
    status = insert_level_set_roots(*request, mesh, provenance, star, input_work, state);
    if (status != PHX_MC_OK) return status;
    std::vector<std::array<int32_t, 5>> target;
    mesh.collect_cells(target);
    status = certify_level_set_cells(*request, mesh, target);
    if (status != PHX_MC_OK) return status;
    publish_level_set_split(*request, mesh, provenance, target, input_work, state);
    return PHX_MC_OK;
  });
}
