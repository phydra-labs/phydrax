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
#include "predicates.hpp"
#include "tet_mesh.hpp"

extern "C" PHX_MC_API int32_t phx_mc_level_set_split_3d(
    int64_t point_count, const double* points, const double* values,
    int64_t cell_count, const int32_t* cells, int64_t face_count,
    const int32_t* faces, int64_t edge_count, const int32_t* edges,
    const int8_t* protected_edges, int64_t max_vertices, int64_t max_cells,
    int64_t work_limit, int64_t cavity_limit, double* output_points, int32_t* output_cells,
    int32_t* output_parents, int32_t* output_sources, double* output_weights,
    int64_t* counters) {
  return phx::mc::guarded([&]() -> int32_t {
    if (point_count < 4 || cell_count < 1 || face_count < 4 || edge_count < 6 ||
        max_vertices < point_count || max_vertices > std::numeric_limits<int32_t>::max() - 1 ||
        max_cells < cell_count || cell_count > std::numeric_limits<int32_t>::max() ||
        face_count > std::numeric_limits<int32_t>::max() ||
        edge_count > std::numeric_limits<int32_t>::max() || work_limit < 0 || cavity_limit < 1 ||
        points == nullptr || values == nullptr || cells == nullptr || faces == nullptr ||
        edges == nullptr || protected_edges == nullptr || output_points == nullptr ||
        output_cells == nullptr || output_parents == nullptr || output_sources == nullptr ||
        output_weights == nullptr || counters == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::fill_n(counters, 6, int64_t{0});
    counters[5] = -1;
    const int64_t input_work = point_count + 4 * cell_count + 3 * face_count + 2 * edge_count;
    if (input_work > work_limit) return PHX_MC_CAPACITY_EXCEEDED;
    counters[4] = input_work;
    std::vector<int32_t> regions(static_cast<std::size_t>(cell_count));
    std::vector<int32_t> face_sources(static_cast<std::size_t>(face_count));
    std::vector<int32_t> edge_sources(static_cast<std::size_t>(edge_count));
    for (int64_t i = 0; i < point_count; ++i) {
      if (!std::isfinite(values[i])) return PHX_MC_INVALID_INPUT;
      counters[3] += values[i] == 0.0 ? 1 : 0;
      for (int k = 0; k < 3; ++k) {
        if (!phx::mc::coordinate_in_domain(points[3 * i + k])) return PHX_MC_INVALID_INPUT;
      }
    }
    for (int32_t i = 0; i < cell_count; ++i) regions[static_cast<std::size_t>(i)] = i;
    for (int32_t i = 0; i < face_count; ++i) face_sources[static_cast<std::size_t>(i)] = i;
    for (int32_t i = 0; i < edge_count; ++i) edge_sources[static_cast<std::size_t>(i)] = i;
    phx::mc::TetMesh mesh(phx::mc::BoundaryPolicy::kConforming, max_vertices, max_cells);
    int32_t status = mesh.build(point_count, points, cell_count, cells, regions.data(),
                                face_count, faces, face_sources.data(), edge_count, edges,
                                edge_sources.data(), nullptr);
    if (status != PHX_MC_OK) return status;
    mesh.set_work_limit(work_limit - input_work);
    std::vector<std::array<int32_t, 2>> sources;
    std::vector<std::array<double, 2>> weights;
    sources.reserve(static_cast<std::size_t>(std::min(max_vertices, point_count + edge_count)));
    weights.reserve(sources.capacity());
    for (int32_t i = 0; i < point_count; ++i) {
      sources.push_back({i, i});
      weights.push_back({1.0, 0.0});
    }
    std::vector<int32_t> star;
    for (int64_t e = 0; e < edge_count; ++e) {
      if (!mesh.spend(1)) {
        counters[4] = input_work + mesh.work();
        counters[5] = e;
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      counters[4] = input_work + mesh.work();
      const int32_t a = edges[2 * e], b = edges[2 * e + 1];
      if (a < 0 || b < 0 || a >= point_count || b >= point_count || a == b) {
        return PHX_MC_INVALID_INPUT;
      }
      if (values[a] == 0.0 || values[b] == 0.0 || std::signbit(values[a]) == std::signbit(values[b])) {
        continue;
      }
      counters[5] = e;
      if (protected_edges[e] != 0) return PHX_MC_CONSTRAINT_INTERSECTION;
      // The unique P1 root is computed without overflow in the endpoint sum.
      const long double first = std::abs(static_cast<long double>(values[a]));
      const long double second = std::abs(static_cast<long double>(values[b]));
      const double t = static_cast<double>(first / (first + second));
      double root[3];
      for (int k = 0; k < 3; ++k) {
        root[k] = static_cast<double>((1.0L - static_cast<long double>(t)) * points[3 * a + k] +
                                     static_cast<long double>(t) * points[3 * b + k]);
        if (!phx::mc::coordinate_in_domain(root[k])) return PHX_MC_INVALID_INPUT;
      }
      // Never perturb the root off a protected source stratum to force success.
      if (!(t > 0.0 && t < 1.0) ||
          !phx::mc::collinear3d(points + 3 * a, points + 3 * b, root)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
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
          mesh.cavity().size() > static_cast<std::size_t>(cavity_limit)) {
        mesh.abandon();
        counters[4] = input_work + mesh.work();
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      if (insertion == phx::mc::Insertion::kOk) {
        insertion = mesh.commit(phx::mc::InsertKind::kSubsegment, 0.0);
      }
      if (insertion != phx::mc::Insertion::kOk) {
        mesh.abandon();
        counters[4] = input_work + mesh.work();
        return insertion == phx::mc::Insertion::kCapacity ? PHX_MC_CAPACITY_EXCEEDED
                                                        : PHX_MC_CONSTRAINT_INTERSECTION;
      }
      sources.push_back({a, b});
      weights.push_back({1.0 - t, t});
      ++counters[2];
    }
    std::vector<std::array<int32_t, 5>> target;
    mesh.collect_cells(target);
    // Every original opposite-sign pair was split as a constrained segment.
    // Thus each finite target simplex must lie wholly on one side (zeros shared).
    for (const auto& cell : target) {
      bool negative = false, positive = false;
      for (int k = 0; k < 4; ++k) {
        const int32_t v = cell[static_cast<std::size_t>(k)];
        if (v < point_count) {
          negative = negative || values[v] < 0.0;
          positive = positive || values[v] > 0.0;
        }
      }
      if (negative == positive) return PHX_MC_INVALID_INPUT;
    }
    if (!mesh.complex().audit()) return PHX_MC_INTERNAL_ERROR;
    counters[0] = mesh.vertex_count();
    counters[1] = static_cast<int64_t>(target.size());
    counters[4] = input_work + mesh.work();
    counters[5] = -1;
    for (int32_t i = 0; i < mesh.vertex_count(); ++i) {
      std::copy_n(mesh.point(i), 3, output_points + 3 * i);
      std::copy_n(sources[static_cast<std::size_t>(i)].data(), 2, output_sources + 2 * i);
      std::copy_n(weights[static_cast<std::size_t>(i)].data(), 2, output_weights + 2 * i);
    }
    for (std::size_t i = 0; i < target.size(); ++i) {
      std::copy_n(target[i].data(), 4, output_cells + 4 * i);
      output_parents[i] = target[i][4];
    }
    return PHX_MC_OK;
  });
}
