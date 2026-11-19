//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Chart-embedded surface triangulations with local physical-space
// reconnection.  A surface patch is triangulated in its chart (a planar
// parameter domain) and simultaneously carries the physical position of every
// vertex.  Ordinary decisions that keep it valid (point location, split
// legality, flip legality) use the exact orient2d predicate.  A terminal
// producer-owned rational edge split may carry a noncollinear rounded chart
// execution view; its explicit reciprocal edge and open componentwise bounds
// own the four-child topology, while the producer retains and validates the
// exact coordinate.  Quality decisions (which diagonal a quadrilateral should
// use) are physical: the Delaunay criterion is evaluated on the physical
// triangles, so metric distortion of the chart does not leak into the
// published surface mesh.  Physical quantities remain guarded by either the
// exact chart invariant or that explicit exact-restriction contract.
//
// Vertices may share physical coordinates (a chart side collapsing to a pole).
// A quadrilateral with two coincident corners has no meaningful physical
// diagonal, so it is never flipped.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "chart_surface.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "topology_keys.hpp"

namespace phx::mc {
namespace {


bool chart_in_domain(const double* uv) {
  return std::isfinite(uv[0]) && std::isfinite(uv[1]) && coordinate_in_domain(uv[0]) &&
         coordinate_in_domain(uv[1]);
}

// Count bounds and addressability precede every allocation and buffer access.
bool reconnect_dimensions_valid(const phx_mc_surface_reconnect_request& r) {
  return !(r.vertex_count < 3 || r.vertex_count >= INT32_MAX || r.triangle_count < 1 ||
           r.triangle_count >= INT32_MAX || r.insert_count < 0 ||
           !addressable(r.vertex_count, 3, sizeof(double)) ||
           !addressable(r.triangle_count, 3, sizeof(int32_t)) ||
           !addressable(r.triangle_count, 3, sizeof(int8_t)) ||
           !addressable(r.triangle_capacity, 3, sizeof(int32_t)) ||
           !addressable(r.triangle_capacity, 3, sizeof(int8_t)) ||
           !addressable(r.insert_count, 3, sizeof(double)) ||
           !addressable(r.insert_count, 1, sizeof(int32_t)) ||
           !addressable(r.vertex_count + r.insert_count, 3, sizeof(double)));
}

bool reconnect_source_buffers_valid(const phx_mc_surface_reconnect_request& r) {
  return !(r.charts == nullptr || r.points == nullptr || r.normals == nullptr ||
           r.triangles == nullptr || r.constrained == nullptr ||
           (r.vertex_metrics == nullptr) != (r.insert_metrics == nullptr) ||
           (r.vertex_metrics != nullptr &&
            !addressable(r.vertex_count + r.insert_count, 9, sizeof(double))) ||
           (r.vertex_pole_ids != nullptr &&
            !addressable(r.vertex_count, 1, sizeof(int64_t))) ||
           (r.insert_pole_ids != nullptr &&
            !addressable(r.insert_count, 1, sizeof(int64_t))) ||
           (r.insert_edges != nullptr &&
            !addressable(r.insert_count, 2, sizeof(int32_t))));
}

bool reconnect_result_buffers_valid(const phx_mc_surface_reconnect_request& r) {
  return !(r.triangles_out == nullptr || r.constrained_out == nullptr ||
           r.triangle_count_out == nullptr ||
           (r.insert_count > 0 &&
            (r.insert_charts == nullptr || r.insert_points == nullptr ||
             r.insert_normals == nullptr || r.insert_spacing == nullptr ||
             r.insert_hints == nullptr || r.vertex_ids == nullptr || r.item_status == nullptr)));
}

// Preserve reservation order and source metadata refusals before geometry
// validation. Explicit edge intent alone requires the incident-fan index.
int32_t load_surface_vertices(SurfaceTriangulation& surface,
                              const phx_mc_surface_reconnect_request& r) {
  const int64_t vertex_capacity = r.vertex_count + r.insert_count;
  const int64_t cell_capacity = std::min(r.triangle_capacity, r.triangle_count + 2 * r.insert_count);
  surface.charts.reserve(static_cast<std::size_t>(2 * vertex_capacity));
  surface.points.reserve(static_cast<std::size_t>(3 * vertex_capacity));
  surface.normals.reserve(static_cast<std::size_t>(3 * vertex_capacity));
  surface.pole_ids.reserve(static_cast<std::size_t>(vertex_capacity));
  if (r.insert_edges != nullptr &&
      std::any_of(r.insert_edges, r.insert_edges + 2 * r.insert_count,
                  [](int32_t vertex) { return vertex >= 0; })) {
    surface.vertex_cells.reserve(static_cast<std::size_t>(vertex_capacity));
    surface.vertex_cells.assign(static_cast<std::size_t>(r.vertex_count), -1);
  }
  if (r.vertex_pole_ids == nullptr) {
    surface.pole_ids.assign(static_cast<std::size_t>(r.vertex_count), -1);
  } else {
    for (int64_t vertex = 0; vertex < r.vertex_count; ++vertex) {
      if (r.vertex_pole_ids[vertex] < -1) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    surface.pole_ids.assign(r.vertex_pole_ids, r.vertex_pole_ids + r.vertex_count);
  }
  surface.cells.reserve(static_cast<std::size_t>(cell_capacity));
  surface.neighbors.reserve(static_cast<std::size_t>(cell_capacity));
  surface.constrained.reserve(static_cast<std::size_t>(cell_capacity));
  surface.charts.assign(r.charts, r.charts + 2 * r.vertex_count);
  surface.points.assign(r.points, r.points + 3 * r.vertex_count);
  surface.normals.assign(r.normals, r.normals + 3 * r.vertex_count);
  if (r.vertex_metrics != nullptr) {
    for (int64_t vertex = 0; vertex < r.vertex_count; ++vertex) {
      if (!metric_valid(r.vertex_metrics + 9 * vertex)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    surface.metrics.reserve(static_cast<std::size_t>(9 * vertex_capacity));
    surface.metrics.assign(r.vertex_metrics, r.vertex_metrics + 9 * r.vertex_count);
  }
  return PHX_MC_OK;
}

int32_t check_surface_coordinates(const phx_mc_surface_reconnect_request& r) {
  for (int64_t vertex = 0; vertex < r.vertex_count; ++vertex) {
    if (!chart_in_domain(r.charts + 2 * vertex)) {
      return std::isfinite(r.charts[2 * vertex]) && std::isfinite(r.charts[2 * vertex + 1])
                 ? PHX_MC_RANGE_ERROR
                 : PHX_MC_NONFINITE_INPUT;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(r.points[3 * vertex + axis]) ||
          !std::isfinite(r.normals[3 * vertex + axis])) {
        return PHX_MC_NONFINITE_INPUT;
      }
    }
  }
  return PHX_MC_OK;
}

int32_t load_surface_cells(SurfaceTriangulation& surface,
                           const phx_mc_surface_reconnect_request& r) {
  surface.cells.resize(static_cast<std::size_t>(r.triangle_count));
  surface.constrained.resize(static_cast<std::size_t>(r.triangle_count));
  for (int64_t cell = 0; cell < r.triangle_count; ++cell) {
    for (int k = 0; k < 3; ++k) {
      const int32_t vertex = r.triangles[3 * cell + k];
      if (vertex < 0 || vertex >= r.vertex_count) {
        return PHX_MC_INVALID_INPUT;
      }
      surface.cells[cell][k] = vertex;
      surface.constrained[cell][k] = r.constrained[3 * cell + k] != 0 ? 1 : 0;
    }
    const auto& t = surface.cells[cell];
    if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

struct SurfaceInsertion {
  const double* chart;
  const double* point;
  const double* normal;
  double spacing;
  const int32_t* requested;
};

bool insertion_coordinates_valid(const SurfaceInsertion& input) {
  const double* uv = input.chart;
  const double* xyz = input.point;
  const double* n = input.normal;
  return !(!std::isfinite(uv[0]) || !std::isfinite(uv[1]) || !std::isfinite(xyz[0]) ||
           !std::isfinite(xyz[1]) || !std::isfinite(xyz[2]) || !std::isfinite(n[0]) ||
           !std::isfinite(n[1]) || !std::isfinite(n[2]) || !std::isfinite(input.spacing) ||
           input.spacing < 0.0);
}

// Endpoints refer to source rows, even after earlier items append vertices.
bool requested_edge_valid(const int32_t* requested, int64_t vertex_count) {
  return requested == nullptr ||
         ((requested[0] == -1 && requested[1] == -1) ||
          (requested[0] >= 0 && requested[1] >= 0 &&
           requested[0] < vertex_count && requested[1] < vertex_count &&
           requested[0] != requested[1]));
}

// Validate item metadata before topology changes.
int32_t insertion_context_status(const SurfaceTriangulation& surface,
                                 const SurfaceInsertion& input, int64_t vertex_count) {
  if (surface.insertion_metric != nullptr && !metric_valid(surface.insertion_metric)) {
    return PHX_MC_NONFINITE_INPUT;
  }
  if (surface.insertion_pole_id < -1) {
    return PHX_MC_INVALID_INPUT;
  }
  if (!chart_in_domain(input.chart)) {
    return PHX_MC_RANGE_ERROR;
  }
  if (!requested_edge_valid(input.requested, vertex_count)) {
    return PHX_MC_INVALID_INPUT;
  }
  return PHX_MC_OK;
}

void execute_surface_insertions(SurfaceTriangulation& surface,
                                const phx_mc_surface_reconnect_request& r) {
  int32_t last = 0;
  for (int64_t item = 0; item < r.insert_count; ++item) {
    const SurfaceInsertion input{
        r.insert_charts + 2 * item, r.insert_points + 3 * item,
        r.insert_normals + 3 * item, r.insert_spacing[item],
        r.insert_edges == nullptr ? nullptr : r.insert_edges + 2 * item};
    surface.insertion_metric =
        r.insert_metrics == nullptr ? nullptr : r.insert_metrics + 9 * item;
    surface.insertion_pole_id = r.insert_pole_ids == nullptr ? -1 : r.insert_pole_ids[item];
    const int32_t status = insertion_coordinates_valid(input)
        ? insertion_context_status(surface, input, r.vertex_count) : PHX_MC_NONFINITE_INPUT;
    SurfaceTriangulation::InsertionResult result{-1, status};
    if (status == PHX_MC_OK) {
      const int32_t hint = r.insert_hints[item] >= 0 ? r.insert_hints[item] : last;
      result = surface.insert(input.chart, input.point, input.normal, input.spacing,
                              hint, input.requested, r.sweep != 0, r.triangle_capacity);
      if (result.vertex >= 0) {
        last = static_cast<int32_t>(surface.cells.size()) - 1;
      }
    }
    r.vertex_ids[item] = result.vertex;
    r.item_status[item] = result.status;
  }
}

void publish_surface_topology(const SurfaceTriangulation& surface,
                              const phx_mc_surface_reconnect_request& r) {
  const std::size_t produced = surface.cells.size();
  for (std::size_t cell = 0; cell < produced; ++cell) {
    for (int k = 0; k < 3; ++k) {
      r.triangles_out[3 * cell + k] = surface.cells[cell][k];
      r.constrained_out[3 * cell + k] = surface.constrained[cell][k];
    }
  }
  *r.triangle_count_out = static_cast<int64_t>(produced);
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_surface_reconnect(
    const phx_mc_surface_reconnect_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    using phx::mc::SurfaceTriangulation;
    const auto& r = *request;
    if (r.vertex_count >= INT32_MAX || r.triangle_count >= INT32_MAX ||
        r.insert_count >= INT32_MAX ||
        (r.vertex_count >= 0 && r.insert_count >= 0 &&
         r.vertex_count + r.insert_count >= INT32_MAX)) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    if (!phx::mc::reconnect_dimensions_valid(r) ||
        !phx::mc::reconnect_source_buffers_valid(r) ||
        !phx::mc::reconnect_result_buffers_valid(r)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (r.triangle_capacity < r.triangle_count) return PHX_MC_CAPACITY_EXCEEDED;
    SurfaceTriangulation surface;
    const int32_t vertices = phx::mc::load_surface_vertices(surface, r);
    if (vertices != PHX_MC_OK) {
      return vertices;
    }
    const int32_t coordinates = phx::mc::check_surface_coordinates(r);
    if (coordinates != PHX_MC_OK) {
      return coordinates;
    }
    const int32_t cells = phx::mc::load_surface_cells(surface, r);
    if (cells != PHX_MC_OK) {
      return cells;
    }
    const int32_t prepared = surface.prepare();
    if (prepared != PHX_MC_OK) {
      return prepared;
    }
    phx::mc::execute_surface_insertions(surface, r);
    if (r.sweep != 0) {
      surface.sweep();
    }
    phx::mc::publish_surface_topology(surface, r);
    return PHX_MC_OK;
  });
}

}  // extern "C"