// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "phydrax_meshcore.h"
#include "expansion.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {
namespace {

using Point = std::array<double, 3>;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double curve_parameter(const phx_mc_cad_model& model, int32_t curve, double s) {
  const double first = model.curve_ranges[2 * curve], last = model.curve_ranges[2 * curve + 1];
  return s == 0.0 ? first : s == 1.0 ? last : first + s * (last - first);
}

double chord_parameter(const phx_mc_cad_model& model, int32_t use, double s) {
  const double first = model.use_ranges[2 * use], last = model.use_ranges[2 * use + 1];
  return s == 0.0 ? first : s == 1.0 ? last : first + s * (last - first);
}

int32_t add_vertex(CadSurfaceMesh& mesh, const Point& xyz, int8_t dimension, int32_t entity,
                   std::array<double, 2> parameters) {
  if (mesh.points.size() >= static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) return -1;
  const int32_t row = static_cast<int32_t>(mesh.points.size());
  mesh.points.push_back(xyz);
  mesh.vertex_dimensions.push_back(dimension);
  mesh.vertex_entities.push_back(entity);
  mesh.vertex_parameters.push_back(parameters);
  return row;
}

int32_t material_sides(const phx_mc_cad_model& model, std::vector<std::array<int32_t, 2>>& sides) {
  sides.assign(static_cast<std::size_t>(model.surface_count), {-1, -1});
  if (model.region_count > std::numeric_limits<int32_t>::max()) return PHX_MC_INDEX_RANGE_ERROR;
  for (int64_t region = 0; region < model.region_count; ++region) {
    for (int64_t i = model.region_surface_offsets[region]; i < model.region_surface_offsets[region + 1]; ++i) {
      const int32_t face = model.region_surfaces[i];
      const int side = model.region_orientations[i] > 0 ? 1 : 0;
      if (sides[face][side] != -1) return PHX_MC_INVALID_INPUT;
      sides[face][side] = static_cast<int32_t>(region);
    }
  }
  return PHX_MC_OK;
}

std::uint64_t morton(const Point& p, const Point& low, const Point& high) {
  std::uint64_t q[3];
  for (int k = 0; k < 3; ++k) {
    const double span = high[k] - low[k];
    const double unit = span > 0 ? std::clamp((p[k] - low[k]) / span, 0.0, 1.0) : 0.0;
    q[k] = static_cast<std::uint64_t>(unit * 2097151.0);
  }
  return spread_bits_3(q[0]) | (spread_bits_3(q[1]) << 1) | (spread_bits_3(q[2]) << 2);
}

template <class T>
void reorder(std::vector<T>& rows, const std::vector<SpatialKey>& order) {
  std::vector<T> next;
  next.reserve(order.size());
  for (const auto& key : order) next.push_back(std::move(rows[key.index]));
  rows = std::move(next);
}

void morton_reorder(CadSurfaceMesh& mesh) {
  if (mesh.points.empty()) return;
  Point low = mesh.points.front(), high = low;
  for (const Point& p : mesh.points) for (int k = 0; k < 3; ++k) {
    low[k] = std::min(low[k], p[k]); high[k] = std::max(high[k], p[k]);
  }
  std::vector<SpatialKey> order, scratch;
  order.reserve(mesh.points.size());
  for (std::size_t i = 0; i < mesh.points.size(); ++i)
    order.push_back({morton(mesh.points[i], low, high), static_cast<int32_t>(i), 0});
  radix_sort_keys(order, scratch, true, false);
  std::vector<int32_t> inverse(order.size());
  for (std::size_t i = 0; i < order.size(); ++i) inverse[order[i].index] = static_cast<int32_t>(i);
  reorder(mesh.points, order); reorder(mesh.vertex_dimensions, order);
  reorder(mesh.vertex_entities, order); reorder(mesh.vertex_parameters, order);
  for (auto& triangle : mesh.triangles) for (int32_t& v : triangle) v = inverse[v];
  for (auto& edge : mesh.curve_edges) for (int32_t& v : edge) v = inverse[v];
  for (auto& triangle : mesh.chart.triangles)
    for (int64_t& v : triangle) v = inverse[static_cast<std::size_t>(v)];
  order.clear();
  order.reserve(mesh.triangles.size());
  for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
    const auto& t = mesh.triangles[i];
    Point center{};
    for (int k = 0; k < 3; ++k)
      center[k] = (mesh.points[t[0]][k] + mesh.points[t[1]][k] + mesh.points[t[2]][k]) / 3.0;
    order.push_back({morton(center, low, high), static_cast<int32_t>(i), 0});
  }
  radix_sort_keys(order, scratch, true, false);
  std::vector<int32_t> triangle_inverse(order.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    triangle_inverse[order[i].index] = static_cast<int32_t>(i);
  for (int64_t& t : mesh.chart.chart_to_mesh)
    if (t >= 0) t = triangle_inverse[static_cast<std::size_t>(t)];
  reorder(mesh.triangles, order); reorder(mesh.triangle_patches, order);
  reorder(mesh.triangle_uv, order); reorder(mesh.triangle_edge_uses, order);
  reorder(mesh.facet_regions, order);
}

}  // namespace

int32_t cad_mesh_surface(const cad::CadGeometry& geometry, const CadCurveDiscretization& curves,
                         const double* sizes, const double* tolerances, double theta,
                         CadSurfaceMesh& output) {
  const auto& model = geometry.model();
  if ((model.surface_count && (!sizes || !tolerances)) || !(theta > 0.0) || !std::isfinite(theta))
    return PHX_MC_INVALID_ARGUMENT;
  if (model.surface_count > std::numeric_limits<int32_t>::max() ||
      model.curve_count > std::numeric_limits<int32_t>::max() ||
      model.corner_count > std::numeric_limits<int32_t>::max()) return PHX_MC_INDEX_RANGE_ERROR;
  const int32_t valid = validate_cad_curve_discretization(model, curves);
  if (valid != PHX_MC_OK) return valid;
  for (int64_t face = 0; face < model.surface_count; ++face)
    if (!(sizes[face] > 0.0) || !(tolerances[face] > 0.0) ||
        !std::isfinite(sizes[face]) || !std::isfinite(tolerances[face])) return PHX_MC_INVALID_INPUT;
  CadSurfaceMesh mesh;
  std::vector<std::array<int32_t, 2>> sides;
  int32_t status = material_sides(model, sides);
  if (status != PHX_MC_OK) return status;
  std::vector<int32_t> use_loop(static_cast<std::size_t>(model.use_count), -1);
  mesh.chart.use_patch.resize(static_cast<std::size_t>(model.use_count));
  mesh.chart.use_range.resize(static_cast<std::size_t>(model.use_count));
  for (int64_t loop = 0; loop < model.loop_count; ++loop) {
    for (int64_t use = model.loop_use_offsets[loop]; use < model.loop_use_offsets[loop+1]; ++use) {
      use_loop[use] = static_cast<int32_t>(loop);
      mesh.chart.use_patch[use] = model.loop_surfaces[loop];
      const bool forward = model.use_senses[use] > 0;
      mesh.chart.use_range[use] = {model.use_ranges[2*use+(forward ? 0 : 1)],
                                   model.use_ranges[2*use+(forward ? 1 : 0)]};
    }
  }
  for (int64_t c = 0; c < model.corner_count; ++c) {
    const double* p = model.corner_points + 3 * c;
    if (add_vertex(mesh, {p[0], p[1], p[2]}, 0, static_cast<int32_t>(c), {kNaN,kNaN}) < 0)
      return PHX_MC_INDEX_RANGE_ERROR;
  }
  cad::EvaluationWorkspace workspace;
  std::vector<int32_t> curve_base(static_cast<std::size_t>(model.curve_count));
  for (int32_t c = 0; c < model.curve_count; ++c) {
    curve_base[c] = static_cast<int32_t>(mesh.points.size());
    const int64_t first = curves.offsets[c], count = curves.offsets[c + 1] - first;
    for (int64_t j = 1; j + 1 < count; ++j) {
      const double t = curve_parameter(model, c, curves.parameters[first + j]);
      Point xyz;
      if (!geometry.jets(geometry.curve_node(c), &t, 0, xyz.data(), workspace)) return PHX_MC_INVALID_INPUT;
      if (add_vertex(mesh, xyz, 1, c, {t,kNaN}) < 0) return PHX_MC_INDEX_RANGE_ERROR;
    }
    const auto row = [&](int64_t index) -> int32_t {
      if (index == 0) return model.curve_corners[2 * c];
      if (index == count - 1) return model.curve_corners[2 * c + 1];
      return curve_base[c] + static_cast<int32_t>(index - 1);
    };
    for (int64_t j = 0; j + 1 < count; ++j) {
      mesh.curve_edges.push_back({row(j), row(j + 1)});
      mesh.curve_edge_curves.push_back(c);
      mesh.curve_edge_parameters.push_back({curve_parameter(model,c,curves.parameters[first+j]),
                                            curve_parameter(model,c,curves.parameters[first+j+1])});
    }
  }
  int32_t overall = PHX_MC_OK;
  for (int32_t face = 0; face < model.surface_count; ++face) {
    CadFaceMesh local;
    status = cad_mesh_face(geometry, curves, face, sizes[face], tolerances[face], theta, local);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    if (local.failure.reason != 0) {
      output = CadSurfaceMesh{};
      output.face_failures = std::move(mesh.face_failures);
      output.face_failures.push_back(local.failure);
      return PHX_MC_UNRESOLVED;
    }
    mesh.face_failures.insert(mesh.face_failures.end(), local.refinement_failures.begin(),
                              local.refinement_failures.end());
    if (status == PHX_MC_UNRESOLVED) overall = status;
    mesh.face_stats.push_back(local.stats);
    mesh.omitted_pole_points.push_back(local.omitted_pole_points);
    for (const auto& pole : local.pole_sectors)
      mesh.pole_sectors.push_back({face, pole.use, pole.fixed, pole.free});
    mesh.periodic_corner_gaps.push_back(local.periodic_corner_gaps);
    mesh.max_periodic_corner_gap_ulps.push_back(local.max_periodic_corner_gap_ulps);
    std::vector<int32_t> global(local.vertices.size());
    for (std::size_t j = 0; j < local.vertices.size(); ++j) {
      const auto& vertex = local.vertices[j];
      if (vertex.dimension == 0) global[j] = vertex.entity;
      else if (vertex.dimension == 1) global[j] = curve_base[vertex.entity] + vertex.curve_index - 1;
      else {
        Point xyz;
        if (!geometry.jets(geometry.surface_node(face), vertex.uv.data(), 0, xyz.data(), workspace))
          return PHX_MC_INVALID_INPUT;
        global[j] = add_vertex(mesh, xyz, 2, face, vertex.uv);
        if (global[j] < 0) return PHX_MC_INDEX_RANGE_ERROR;
      }
    }
    using Boundary = std::pair<std::pair<int32_t,int32_t>, int64_t>;
    std::vector<Boundary> boundary;
    // Chord direction must agree with the oriented physical triangles, even
    // when the input loop winding and surface orientation differ independently.
    Expansion area;
    for (const auto& edge : local.boundary_edges) {
      const auto& a = local.vertices[edge.vertices[0]].uv;
      const auto& b = local.vertices[edge.vertices[1]].uv;
      area = area + Expansion::product(a[0], b[1]) - Expansion::product(a[1], b[0]);
    }
    const bool reverse_chords = area.sign() != 0 &&
                                area.sign() != model.surface_orientations[face];
    if (reverse_chords) {
      for (int64_t loop = 0; loop < model.loop_count; ++loop) {
        if (model.loop_surfaces[loop] != face) continue;
        for (int64_t use = model.loop_use_offsets[loop]; use < model.loop_use_offsets[loop+1]; ++use)
          std::swap(mesh.chart.use_range[use][0], mesh.chart.use_range[use][1]);
      }
    }
    boundary.reserve(local.boundary_edges.size());
    for (const auto& edge : local.boundary_edges) {
      const bool forward = !reverse_chords;
      const auto& a = local.vertices[edge.vertices[forward ? 0 : 1]].uv;
      const auto& b = local.vertices[edge.vertices[forward ? 1 : 0]].uv;
      const int64_t chord = static_cast<int64_t>(mesh.chart.chord_use.size());
      boundary.push_back({std::minmax(edge.vertices[0], edge.vertices[1]), chord});
      mesh.chart.chord_use.push_back(edge.use);
      mesh.chart.chord_loop.push_back(use_loop[edge.use]);
      mesh.chart.chord_uv.push_back({a[0], a[1], b[0], b[1]});
      mesh.chart.chord_range.push_back({
          chord_parameter(model, edge.use, edge.parameters[forward ? 0 : 1]),
          chord_parameter(model, edge.use, edge.parameters[forward ? 1 : 0])});
    }
    std::sort(boundary.begin(), boundary.end());
    const auto uv_record = [&](const std::array<int32_t,3>& t) {
      return std::array<double,6>{local.vertices[t[0]].uv[0],local.vertices[t[0]].uv[1],
                                  local.vertices[t[1]].uv[0],local.vertices[t[1]].uv[1],
                                  local.vertices[t[2]].uv[0],local.vertices[t[2]].uv[1]};
    };
    const auto chart_record = [&](const std::array<int32_t, 3>& t, int64_t mesh_row) {
      mesh.chart.triangles.push_back({global[t[0]], global[t[1]], global[t[2]]});
      mesh.chart.patches.push_back(face);
      mesh.chart.uv.push_back(uv_record(t));
      mesh.chart.chart_to_mesh.push_back(mesh_row);
      std::array<int64_t, 3> chords{-1, -1, -1};
      for (int k = 0; k < 3; ++k) {
        const std::pair<int32_t, int32_t> key = std::minmax(t[k], t[(k+1)%3]);
        const auto it = std::lower_bound(boundary.begin(), boundary.end(), Boundary{key,-1});
        if (it != boundary.end() && it->first == key) chords[k] = it->second;
      }
      mesh.chart.edge_chord.push_back(chords);
      return chords;
    };
    for (auto t : local.triangles) {
      if (model.surface_orientations[face] < 0) std::swap(t[1], t[2]);
      const auto chords = chart_record(t, static_cast<int64_t>(mesh.triangles.size()));
      mesh.triangles.push_back({global[t[0]],global[t[1]],global[t[2]]});
      mesh.triangle_patches.push_back(face);
      mesh.triangle_uv.push_back(uv_record(t));
      std::array<int32_t,3> uses{-1,-1,-1};
      for (int k = 0; k < 3; ++k)
        if (chords[k] >= 0) uses[k] = static_cast<int32_t>(mesh.chart.chord_use[chords[k]]);
      mesh.triangle_edge_uses.push_back(uses);
      mesh.facet_regions.push_back(sides[face]);
    }
    for (auto sliver : local.slivers) {
      if (model.surface_orientations[face] < 0) std::swap(sliver[1], sliver[2]);
      chart_record(sliver, -1);
      mesh.sliver_uv.push_back(uv_record(sliver));
      mesh.sliver_patches.push_back(face);
    }
  }
  morton_reorder(mesh);
  output = std::move(mesh);
  return overall;
}

}  // namespace phx::mc
