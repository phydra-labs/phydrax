// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_mesh.hpp"

#include <chrono>
#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include "phydrax_meshcore.h"
#include "expansion.hpp"
#include "spatial_sort.hpp"
#include "cavity.hpp"
#include "improve3d.hpp"
#include "plc3d.hpp"
#include "refine3d.hpp"
#include "tet_mesh_handle.hpp"

namespace phx::mc {
namespace {

using Point = std::array<double, 3>;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

int64_t elapsed_ns(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                              since).count();
}

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
                         CadSurfaceMesh& output, bool graded_pole_rings,
                         const cad::ProximityField* proximity) {
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
  const auto surface_start = std::chrono::steady_clock::now();
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
    const auto face_start = std::chrono::steady_clock::now();
    status = cad_mesh_face(geometry, curves, face, sizes[face], tolerances[face], theta, local,
                           nullptr, nullptr, graded_pole_rings, proximity);
    mesh.face_ns += elapsed_ns(face_start);
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
  mesh.assembly_ns = elapsed_ns(surface_start) - mesh.face_ns;
  output = std::move(mesh);
  return overall;
}

namespace {

// Shifts uv by whole periods of `face` to the copy nearest `near`.
void nearest_period(const phx_mc_cad_model& model, int32_t face, const std::array<double, 2>& near,
                    std::array<double, 2>& uv) {
  if (!model.surface_periods) return;
  for (int k = 0; k < 2; ++k) {
    const double period = model.surface_periods[2 * face + k];
    if (period > 0.0 && std::isfinite(period))
      uv[k] += period * std::nearbyint((near[k] - uv[k]) / period);
  }
}

}  // namespace

int32_t cad_split_surface_edge(const cad::CadGeometry& geometry, cad::CurveDiscretization& curves,
                               CadSurfaceMesh& mesh, int32_t a, int32_t b,
                               phx_mc_cad_boundary_split& record) {
  const auto& model = geometry.model();
  const auto count = static_cast<int64_t>(mesh.points.size());
  if (a < 0 || b < 0 || a >= count || b >= count || a == b) return PHX_MC_INVALID_ARGUMENT;
  struct Incident {
    std::size_t row;
    int k;
  };
  std::vector<Incident> incident;
  for (std::size_t t = 0; t < mesh.triangles.size(); ++t)
    for (int k = 0; k < 3; ++k) {
      const int32_t x = mesh.triangles[t][k], y = mesh.triangles[t][(k + 1) % 3];
      if ((x == a && y == b) || (x == b && y == a)) incident.push_back({t, k});
    }
  if (incident.size() != 2) return PHX_MC_UNRESOLVED;
  int64_t curve_edge = -1;
  for (std::size_t e = 0; e < mesh.curve_edges.size() && curve_edge < 0; ++e) {
    const auto& edge = mesh.curve_edges[e];
    if ((edge[0] == a && edge[1] == b) || (edge[0] == b && edge[1] == a))
      curve_edge = static_cast<int64_t>(e);
  }
  if (curve_edge < 0 &&
      mesh.triangle_patches[incident[0].row] != mesh.triangle_patches[incident[1].row])
    return PHX_MC_UNRESOLVED;
  // Chart rows of the two triangles; a curve edge must be a published chord
  // edge of both (an edge at a pole, carried by a collapsed sliver, is not
  // split). Checked before anything changes.
  auto& chart = mesh.chart;
  std::array<std::size_t, 2> chart_rows{};
  for (int i = 0; i < 2; ++i) {
    chart_rows[i] = chart.chart_to_mesh.size();
    for (std::size_t r = 0; r < chart.chart_to_mesh.size(); ++r)
      if (chart.chart_to_mesh[r] == static_cast<int64_t>(incident[i].row)) { chart_rows[i] = r; break; }
    if (chart_rows[i] == chart.chart_to_mesh.size()) return PHX_MC_INTERNAL_ERROR;
    if (curve_edge >= 0 && chart.edge_chord[chart_rows[i]][incident[i].k] < 0)
      return PHX_MC_UNRESOLVED;
  }
  cad::EvaluationWorkspace workspace;
  Point xyz{};
  int8_t dimension = 2;
  int32_t carrier = -1;
  std::array<double, 2> parameters{kNaN, kNaN};
  const auto corner_uv = [](const std::array<double, 6>& record_uv, int j) {
    return std::array<double, 2>{record_uv[2 * (j % 3)], record_uv[2 * (j % 3) + 1]};
  };
  const auto record_midpoint = [&](const Incident& use) {
    const auto& uv = mesh.triangle_uv[use.row];
    const auto p = corner_uv(uv, use.k), q = corner_uv(uv, use.k + 1);
    return std::array<double, 2>{0.5 * p[0] + 0.5 * q[0], 0.5 * p[1] + 0.5 * q[1]};
  };
  if (curve_edge >= 0) {
    carrier = mesh.curve_edge_curves[curve_edge];
    const auto& range = mesh.curve_edge_parameters[curve_edge];
    const double t = 0.5 * range[0] + 0.5 * range[1];
    if (!geometry.jets(geometry.curve_node(carrier), &t, 0, xyz.data(), workspace))
      return PHX_MC_INVALID_INPUT;
    dimension = 1;
    parameters = {t, kNaN};
  } else {
    carrier = mesh.triangle_patches[incident[0].row];
    parameters = record_midpoint(incident[0]);
    if (!geometry.jets(geometry.surface_node(carrier), parameters.data(), 0, xyz.data(), workspace))
      return PHX_MC_INVALID_INPUT;
  }
  const int32_t m = add_vertex(mesh, xyz, dimension, carrier, parameters);
  if (m < 0) return PHX_MC_INDEX_RANGE_ERROR;
  for (int i = 0; i < 2; ++i) {
    const Incident& use = incident[i];
    const std::size_t row = use.row;
    const int k = use.k;
    const auto triangle = mesh.triangles[row];
    const int32_t x = triangle[k], y = triangle[(k + 1) % 3], opposite = triangle[(k + 2) % 3];
    const auto record = mesh.triangle_uv[row];
    const auto ux = corner_uv(record, k), uy = corner_uv(record, k + 1),
               uc = corner_uv(record, k + 2);
    const int32_t face = mesh.triangle_patches[row];
    const std::array<double, 2> near = record_midpoint(use);
    std::array<double, 2> um = near;
    const std::size_t chart_row = chart_rows[i];
    const auto chords = chart.edge_chord[chart_row];
    int64_t chord_xm = -1, chord_my = -1;
    if (curve_edge >= 0) {
      const int64_t chord = chords[k];
      const auto range = chart.chord_range[chord];
      const auto chord_uv = chart.chord_uv[chord];
      const double p = 0.5 * range[0] + 0.5 * range[1];
      if (!geometry.jets(geometry.pcurve_node(chart.chord_use[chord]), &p, 0, um.data(), workspace))
        return PHX_MC_INVALID_INPUT;
      nearest_period(model, face, near, um);
      const bool forward = chord_uv[0] == ux[0] && chord_uv[1] == ux[1];
      chart.chord_uv[chord] = {chord_uv[0], chord_uv[1], um[0], um[1]};
      chart.chord_range[chord] = {range[0], p};
      const auto added = static_cast<int64_t>(chart.chord_use.size());
      chart.chord_use.push_back(chart.chord_use[chord]);
      chart.chord_loop.push_back(chart.chord_loop[chord]);
      chart.chord_uv.push_back({um[0], um[1], chord_uv[2], chord_uv[3]});
      chart.chord_range.push_back({p, range[1]});
      chord_xm = forward ? chord : added;
      chord_my = forward ? added : chord;
    }
    const auto uv_record = [](const std::array<double, 2>& p, const std::array<double, 2>& q,
                              const std::array<double, 2>& r) {
      return std::array<double, 6>{p[0], p[1], q[0], q[1], r[0], r[1]};
    };
    const auto new_row = static_cast<int64_t>(mesh.triangles.size());
    const auto uses = mesh.triangle_edge_uses[row];
    mesh.triangles[row] = {x, m, opposite};
    mesh.triangle_uv[row] = uv_record(ux, um, uc);
    mesh.triangle_edge_uses[row] = {uses[k], -1, uses[(k + 2) % 3]};
    mesh.triangles.push_back({m, y, opposite});
    mesh.triangle_patches.push_back(face);
    mesh.triangle_uv.push_back(uv_record(um, uy, uc));
    mesh.triangle_edge_uses.push_back({uses[k], uses[(k + 1) % 3], -1});
    mesh.facet_regions.push_back(mesh.facet_regions[row]);
    chart.triangles[chart_row] = {x, m, opposite};
    chart.uv[chart_row] = uv_record(ux, um, uc);
    chart.edge_chord[chart_row] = {chord_xm, -1, chords[(k + 2) % 3]};
    chart.triangles.push_back({m, y, opposite});
    chart.patches.push_back(face);
    chart.uv.push_back(uv_record(um, uy, uc));
    chart.chart_to_mesh.push_back(new_row);
    chart.edge_chord.push_back({chord_my, chords[(k + 1) % 3], -1});
  }
  if (curve_edge >= 0) {
    const auto edge = mesh.curve_edges[curve_edge];
    const auto range = mesh.curve_edge_parameters[curve_edge];
    const double t = parameters[0];
    mesh.curve_edges[curve_edge] = {edge[0], m};
    mesh.curve_edge_parameters[curve_edge] = {range[0], t};
    mesh.curve_edges.push_back({m, edge[1]});
    mesh.curve_edge_curves.push_back(carrier);
    mesh.curve_edge_parameters.push_back({t, range[1]});
    // The shared discretization keeps its normalized parameters sorted.
    const double first = model.curve_ranges[2 * carrier], last = model.curve_ranges[2 * carrier + 1];
    const double s = (t - first) / (last - first);
    const auto begin = curves.parameters.begin() + curves.offsets[carrier];
    const auto end = curves.parameters.begin() + curves.offsets[carrier + 1];
    curves.parameters.insert(std::lower_bound(begin, end, s), s);
    for (std::size_t c = static_cast<std::size_t>(carrier) + 1; c < curves.offsets.size(); ++c)
      ++curves.offsets[c];
  }
  record.vertices[0] = a;
  record.vertices[1] = b;
  record.vertex = m;
  record.dimension = dimension;
  record.carrier = carrier;
  record.parameters[0] = parameters[0];
  record.parameters[1] = parameters[1];
  return PHX_MC_OK;
}

namespace {

struct PlcResultDeleter {
  void operator()(phx_mc_plc3d_result* r) const noexcept { phx_mc_plc3d_free(r); }
};
struct TetMeshDeleter {
  void operator()(phx_mc_tet_mesh* m) const noexcept { phx_mc_tet_mesh_free(m); }
};
using PlcHandle = std::unique_ptr<phx_mc_plc3d_result, PlcResultDeleter>;

// S3 fixed-boundary recovery: one facet per surface triangle (its row), the
// triangle's region sides, curve segments as protected PLC edges.
int32_t recover_surface(const CadSurfaceMesh& s, PlcHandle& handle) {
  const auto count = static_cast<int64_t>(s.triangles.size());
  std::vector<int64_t> offsets(static_cast<std::size_t>(count) + 1);
  std::vector<int32_t> facets(static_cast<std::size_t>(count));
  for (int64_t t = 0; t <= count; ++t) offsets[static_cast<std::size_t>(t)] = 3 * t;
  for (int64_t t = 0; t < count; ++t) facets[static_cast<std::size_t>(t)] = static_cast<int32_t>(t);
  phx_mc_plc3d_result* result = nullptr;
  phx_mc_plc3d_prepare_source_request request{};
  request.point_count = static_cast<int64_t>(s.points.size());
  request.points = reinterpret_cast<const double*>(s.points.data());
  request.polygon_count = count;
  request.polygon_vertex_count = 3 * count;
  request.polygon_offsets = offsets.data();
  request.polygon_vertices = reinterpret_cast<const int32_t*>(s.triangles.data());
  request.polygon_facets = facets.data();
  request.facet_count = count;
  request.facet_regions = reinterpret_cast<const int32_t*>(s.facet_regions.data());
  request.segment_count = static_cast<int64_t>(s.curve_edges.size());
  request.segments = reinterpret_cast<const int32_t*>(s.curve_edges.data());
  request.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  request.result = &result;
  const int32_t status = phx_mc_plc3d_recover(&request);
  handle.reset(result);
  return status;
}

double squared_length(const Point& p, const Point& q) {
  const double x = p[0] - q[0], y = p[1] - q[1], z = p[2] - q[2];
  return x * x + y * y + z * z;
}

// Unmet cells split by cause (stats.unmet_forced_cells and
// unmet_unforced_cells), decided per unmet cell. A cell on a surface curve
// edge AB lies in the meshed wedge between the edge's two boundary triangles
// ABC, ABD, so its dihedral at AB is at most that wedge, which tends to the
// true wedge w between the two faces at the curve as the mesh refines: when
// w is below the minimum dihedral target, no tetrahedralization of a
// boundary resolving that curve meets it there. w at an end point of AB is
// the angle in the plane across the curve tangent T (the curve's own
// derivative) between the face conormals m = n x T (n = Su x Sv of the face
// at the triangle's uv), each turned toward its triangle's apex; it is the
// reflex 2 pi - angle when the cell's centroid falls outside the cone of the
// two conormals. An end where the tangent or a normal vanishes (a pole)
// defines no wedge; a seam (both triangles on one face) has none. A curve
// edge is a forced edge when w is below the target at each end that defines
// it. An unmet cell is forced only when every one of its records is a
// dihedral record and every edge of it whose dihedral is below the target is
// a forced edge; any other failure (radius-edge, size, validity, a small
// dihedral off the rim) leaves it unforced.
// The radius-edge bound is never forced by the wedge: with AB = AC = BD = 1,
// C on one face across A and D on the other across B, the circumcenter is
// (1/2, 1/2, tan(w/2)/2) in the frame of A, AB, AC, so the radius-edge ratio
// is sqrt(1/2 + tan^2(w/2)/4) <= 2 for every w up to 2 atan(sqrt 14) (150
// deg): any wedge admits a cell meeting the radius-edge bound 2 there.
// A rim below the target forces more than its own cells, which is why the
// run stays UNRESOLVED and the rim is reported (curve_wedges): a cell on rim
// edge AB (length L) meeting the radius-edge bound has circumradius >= L/2,
// so every edge >= L/4 and its other corners C, D lie >= L/4 from the rim;
// the cell across its face ACD lies in the solid within O(L) of the rim,
// whose thickness there is O(w L), so its height over an edge >= L/4 is
// O(w L) and one of its dihedrals, at an edge off the rim, is O(w): below
// the target for a small w whatever the apex placement. Either the rim cells
// fail radius-edge or their neighbors fail the dihedral off the rim, so the
// dihedral target is unattainable in a neighborhood of such a rim for any
// mesh; only relaxing the target resolves it. That neighborhood is not
// decidable cell by cell, so its cells stay unforced.
// curve_wedges (when given, one per model curve, NaN where undefined): the
// smallest true wedge in degrees over the curve edges' defined ends, filled
// only when some unmet cell is unforced.
int32_t count_forced_cells(const cad::CadGeometry& geometry, const CadSurfaceMesh& surface,
                           const TetMesh& state, const std::vector<TetUnmetRecord>& records,
                           double target_degrees, int64_t& forced, int64_t& unforced,
                           std::vector<double>* curve_wedges) {
  forced = 0;
  unforced = 0;
  const double target = target_degrees * (std::numbers::pi / 180.0);
  const auto surface_points = static_cast<int32_t>(surface.points.size());
  using Edge = std::array<int32_t, 3>;  // low, high, row
  std::vector<Edge> curve_edges;
  curve_edges.reserve(surface.curve_edges.size());
  for (std::size_t k = 0; k < surface.curve_edges.size(); ++k) {
    const auto [a, b] = std::minmax(surface.curve_edges[k][0], surface.curve_edges[k][1]);
    curve_edges.push_back({a, b, static_cast<int32_t>(k)});
  }
  std::sort(curve_edges.begin(), curve_edges.end());
  const auto find_edge = [&](int32_t a, int32_t b) -> int32_t {
    const auto [low, high] = std::minmax(a, b);
    const auto it = std::lower_bound(curve_edges.begin(), curve_edges.end(), Edge{low, high, -1});
    return it != curve_edges.end() && (*it)[0] == low && (*it)[1] == high ? (*it)[2] : -1;
  };
  // Surface triangles on each curve edge: (curve edge row, triangle).
  std::vector<std::array<int32_t, 2>> sides;
  for (std::size_t t = 0; t < surface.triangles.size(); ++t)
    for (int k = 0; k < 3; ++k) {
      const int32_t row = find_edge(surface.triangles[t][k], surface.triangles[t][(k + 1) % 3]);
      if (row >= 0) sides.push_back({row, static_cast<int32_t>(t)});
    }
  std::sort(sides.begin(), sides.end());
  cad::EvaluationWorkspace workspace;
  const auto cross = [](const double* x, const double* y, double* z) {
    z[0] = x[1] * y[2] - x[2] * y[1];
    z[1] = x[2] * y[0] - x[0] * y[2];
    z[2] = x[0] * y[1] - x[1] * y[0];
  };
  const auto dot = [](const double* x, const double* y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  };
  // The true wedge at end `end` of curve edge `row` on the side of `inside`
  // (a point of the cell), NaN where undefined; false when the geometry
  // cannot be evaluated.
  const auto wedge = [&](int32_t row, int end, const std::array<int32_t, 2>& triangles,
                         const double* inside, double& angle) {
    const auto& edge = surface.curve_edges[row];
    const int32_t vertex = edge[end];
    const double t = surface.curve_edge_parameters[row][end];
    double curve[6];
    if (!geometry.jets(geometry.curve_node(surface.curve_edge_curves[row]), &t, 1, curve,
                       workspace))
      return false;
    double tangent[3] = {curve[3], curve[4], curve[5]};
    const double speed = std::sqrt(dot(tangent, tangent));
    angle = std::numeric_limits<double>::quiet_NaN();
    if (!(speed > 0) || !std::isfinite(speed)) return true;
    for (double& x : tangent) x /= speed;
    const double* p = surface.points[vertex].data();
    double conormals[2][3];
    for (int side = 0; side < 2; ++side) {
      const auto& triangle = surface.triangles[triangles[side]];
      int corner = 0, apex = 0;
      for (int k = 0; k < 3; ++k) {
        if (triangle[k] == vertex) corner = k;
        if (triangle[k] != edge[0] && triangle[k] != edge[1]) apex = triangle[k];
      }
      const auto& uv = surface.triangle_uv[triangles[side]];
      const double parameters[2] = {uv[2 * corner], uv[2 * corner + 1]};
      double jet[9], normal[3], m[3], toward[3];
      if (!geometry.jets(geometry.surface_node(surface.triangle_patches[triangles[side]]),
                         parameters, 1, jet, workspace))
        return false;
      cross(jet + 3, jet + 6, normal);
      cross(normal, tangent, m);
      for (int k = 0; k < 3; ++k) toward[k] = surface.points[apex][k] - p[k];
      const double length = std::sqrt(dot(m, m)) * (dot(m, toward) < 0 ? -1.0 : 1.0);
      if (!(std::abs(length) > 0) || !std::isfinite(length)) return true;
      for (int k = 0; k < 3; ++k) conormals[side][k] = m[k] / length;
    }
    double g[3];
    for (int k = 0; k < 3; ++k) g[k] = inside[k] - p[k];
    const double along = dot(g, tangent);
    for (int k = 0; k < 3; ++k) g[k] -= along * tangent[k];
    const double c = std::clamp(dot(conormals[0], conormals[1]), -1.0, 1.0);
    const double g0 = dot(g, conormals[0]), g1 = dot(g, conormals[1]);
    const double determinant = 1.0 - c * c;
    const bool within = determinant > 0 ? g0 - c * g1 >= 0 && g1 - c * g0 >= 0 : g0 > 0;
    angle = within ? std::acos(c) : 2 * std::numbers::pi - std::acos(c);
    return true;
  };
  // Records grouped by cell: a cell is forced only if all its records are
  // dihedral records.
  std::vector<std::pair<uint64_t, bool>> cells;
  cells.reserve(records.size());
  for (const TetUnmetRecord& record : records)
    cells.push_back({record.cell_id, record.criterion == PHX_MC_TET_UNMET_DIHEDRAL});
  std::sort(cells.begin(), cells.end());
  std::size_t next = 0;
  for (std::size_t first = 0; first < cells.size(); first = next) {
    const uint64_t id = cells[first].first;
    bool dihedral_only = true;
    for (next = first; next < cells.size() && cells[next].first == id; ++next)
      dihedral_only = dihedral_only && cells[next].second;
    const int32_t slot = state.cell_slot(id);
    bool is_forced = dihedral_only && slot >= 0;
    const Tetrahedron* tet = is_forced ? &state.tet(slot) : nullptr;
    const double* q[4] = {};
    for (int k = 0; k < 4 && is_forced; ++k) {
      q[k] = tet->v[k] == kDeadVertex ? nullptr : state.point(tet->v[k]);
      is_forced = q[k] != nullptr;
    }
    double centroid[3] = {0, 0, 0};
    for (int k = 0; k < 4 && is_forced; ++k)
      for (int axis = 0; axis < 3; ++axis) centroid[axis] += 0.25 * q[k][axis];
    bool any_small = false;
    for (int i = 0; i < 4 && is_forced; ++i)
      for (int j = i + 1; j < 4 && is_forced; ++j) {
        // Dihedral at edge ij: the angle between the face normals of
        // (i, j, k) and (i, j, l) taken with the same edge vector.
        int k = 0;
        while (k == i || k == j) ++k;
        const int l = 6 - i - j - k;
        double e[3], f[3], g[3], nk[3], nl[3];
        for (int axis = 0; axis < 3; ++axis) {
          e[axis] = q[j][axis] - q[i][axis];
          f[axis] = q[k][axis] - q[i][axis];
          g[axis] = q[l][axis] - q[i][axis];
        }
        cross(e, f, nk);
        cross(e, g, nl);
        const double scale = std::sqrt(dot(nk, nk) * dot(nl, nl));
        const double dihedral =
            scale > 0 ? std::acos(std::clamp(dot(nk, nl) / scale, -1.0, 1.0)) : 0.0;
        if (!(dihedral < target)) continue;
        any_small = true;
        is_forced = false;
        if (tet->v[i] >= surface_points || tet->v[j] >= surface_points) continue;
        const int32_t row = find_edge(tet->v[i], tet->v[j]);
        if (row < 0) continue;
        const auto side = std::lower_bound(sides.begin(), sides.end(),
                                           std::array<int32_t, 2>{row, -1});
        if (sides.end() - side < 2 || (*side)[0] != row || side[1][0] != row ||
            (sides.end() - side > 2 && side[2][0] == row))
          continue;
        const std::array<int32_t, 2> triangles{(*side)[1], side[1][1]};
        if (surface.triangle_patches[triangles[0]] == surface.triangle_patches[triangles[1]])
          continue;
        double at_start = 0, at_end = 0;
        if (!wedge(row, 0, triangles, centroid, at_start) ||
            !wedge(row, 1, triangles, centroid, at_end))
          return PHX_MC_INVALID_INPUT;
        // NaN compares false: an undefined end neither forces nor clears.
        is_forced = (std::isnan(at_start) || at_start < target) &&
                    (std::isnan(at_end) || at_end < target) &&
                    !(std::isnan(at_start) && std::isnan(at_end));
      }
    // A dihedral record whose small angle this binary64 check does not see
    // (at the threshold) is not explained by a forced edge.
    is_forced = is_forced && any_small;
    (is_forced ? forced : unforced) += 1;
  }
  if (unforced == 0 || !curve_wedges) return PHX_MC_OK;
  // Every rim edge once, its side decided by the centroid of a cell on it.
  std::vector<bool> seen(surface.curve_edges.size(), false);
  const auto& complex = state.complex();
  for (std::size_t slot = 0; slot < complex.tets.size(); ++slot) {
    const Tetrahedron& tet = complex.tets[slot];
    if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
    double centroid[3] = {0, 0, 0};
    for (int k = 0; k < 4; ++k)
      for (int axis = 0; axis < 3; ++axis) centroid[axis] += 0.25 * state.point(tet.v[k])[axis];
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j) {
        if (tet.v[i] >= surface_points || tet.v[j] >= surface_points) continue;
        const int32_t row = find_edge(tet.v[i], tet.v[j]);
        if (row < 0 || seen[static_cast<std::size_t>(row)]) continue;
        seen[static_cast<std::size_t>(row)] = true;
        const auto side = std::lower_bound(sides.begin(), sides.end(),
                                           std::array<int32_t, 2>{row, -1});
        if (sides.end() - side < 2 || (*side)[0] != row || side[1][0] != row ||
            (sides.end() - side > 2 && side[2][0] == row))
          continue;
        const std::array<int32_t, 2> triangles{(*side)[1], side[1][1]};
        if (surface.triangle_patches[triangles[0]] == surface.triangle_patches[triangles[1]])
          continue;
        double& smallest = (*curve_wedges)[static_cast<std::size_t>(
            surface.curve_edge_curves[static_cast<std::size_t>(row)])];
        for (int end = 0; end < 2; ++end) {
          double angle = 0;
          if (!wedge(row, end, triangles, centroid, angle)) return PHX_MC_INVALID_INPUT;
          const double degrees = angle * (180.0 / std::numbers::pi);
          if (!std::isnan(degrees) && !(degrees >= smallest)) smallest = degrees;
        }
      }
  }
  return PHX_MC_OK;
}

}  // namespace

int32_t cad_mesh_volume(const cad::CadGeometry& geometry, cad::CurveDiscretization& curves,
                        CadSurfaceMesh& surface, const double* sizes,
                        const CadVolumeControls& controls, CadVolumeMesh& result,
                        const CadForcedRefusal* forced) {
  result = CadVolumeMesh{};
  auto& stats = result.stats;
  const auto& model = geometry.model();
  if (!sizes || (controls.geometry_order != 1 && controls.geometry_order != 2))
    return PHX_MC_INVALID_ARGUMENT;
  const auto volume_start = std::chrono::steady_clock::now();
  // A fixed boundary triangle that is flat to rounding (a zero-length edge or
  // a zero-area sliver) leaves no tetrahedron beside it that can meet the
  // quality targets, and improvement then churns without end (measured: the
  // sphere's coincident equator rings, 2 ulps apart, ran 115 s to 30k
  // improvement insertions). Its relative area r = 2 A / l_max^2 (sqrt(3)/2
  // for an equilateral triangle) is refused before recovery when r^2 is below
  // the determinant floor 2^-22, the improvement's floor for cells.
  for (std::size_t t = 0; t < surface.triangles.size(); ++t) {
    const auto& tri = surface.triangles[t];
    const Point& a = surface.points[tri[0]];
    const Point& b = surface.points[tri[1]];
    const Point& c = surface.points[tri[2]];
    const Point u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const Point v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const Point n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    const double longest =
        std::max({squared_length(a, b), squared_length(b, c), squared_length(c, a)});
    const double area2 = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const double relative = longest > 0.0 ? area2 / longest : 0.0;
    if (!(relative * relative >= kMetricDeterminantFloor)) {
      stats.recovery_status = PHX_MC_UNRESOLVED;
      stats.recovery_reason = PHX_MC_PLC_INVALID_FACET;
      stats.failure_kind = PHX_MC_PLC_ENTITY_FACET;
      stats.failure_entity = static_cast<int64_t>(t);
      stats.failure_measure = relative;
      stats.volume_ns = elapsed_ns(volume_start);
      return PHX_MC_UNRESOLVED;
    }
  }
  PlcHandle recovery;
  for (int attempt = 0;; ++attempt) {
    phx_mc_cad_boundary_split split{};
    int32_t a = -1, b = -1;
    if (attempt == 0 && forced && forced->a >= 0) {
      split.entity_kind = PHX_MC_PLC_ENTITY_EDGE;
      split.entity = -1;
      a = forced->a;
      b = forced->b;
    } else {
      ++stats.recovery_attempts;
      const int32_t status = recover_surface(surface, recovery);
      stats.recovery_status = status;
      if (status == PHX_MC_OK) break;
      if (!recovery) return status;
      int32_t reason = 0, first_kind = 0, second_kind = 0;
      int64_t first = -1, second = -1;
      double deviation = 0.0, bound = 0.0;
      phx_mc_plc3d_failure(recovery.get(), &reason, &first_kind, &first, &second_kind, &second,
                           &deviation, &bound);
      stats.recovery_reason = reason;
      stats.failure_kind = first_kind;
      stats.failure_entity = first;
      if (reason == PHX_MC_PLC_FIXED_SEGMENT || reason == PHX_MC_PLC_FIXED_FACET ||
          reason == PHX_MC_PLC_STEINER_SUPPRESSION) {
        const auto& edges = recovery->recovery.plc_edges;
        const auto points = static_cast<int32_t>(surface.points.size());
        if (first_kind == PHX_MC_PLC_ENTITY_EDGE && first >= 0 &&
            static_cast<std::size_t>(2 * first + 1) < edges.size()) {
          a = edges[static_cast<std::size_t>(2 * first)];
          b = edges[static_cast<std::size_t>(2 * first + 1)];
          if (a >= points || b >= points) a = b = -1;
        } else if (first_kind == PHX_MC_PLC_ENTITY_FACET && first >= 0 &&
                   static_cast<std::size_t>(first) < surface.triangles.size()) {
          const auto& t = surface.triangles[static_cast<std::size_t>(first)];
          double longest = -1.0;
          for (int k = 0; k < 3; ++k) {
            const double length = squared_length(surface.points[t[k]], surface.points[t[(k + 1) % 3]]);
            if (length > longest) { longest = length; a = t[k]; b = t[(k + 1) % 3]; }
          }
        }
      }
      recovery.reset();
      split.reason = reason;
      split.entity_kind = first_kind;
      split.entity = first;
      if (attempt > 0 || a < 0) {
        stats.volume_ns = elapsed_ns(volume_start);
        return PHX_MC_UNRESOLVED;
      }
    }
    const int32_t split_status = cad_split_surface_edge(geometry, curves, surface, a, b, split);
    if (split_status != PHX_MC_OK) {
      stats.volume_ns = elapsed_ns(volume_start);
      return split_status == PHX_MC_UNRESOLVED ? PHX_MC_UNRESOLVED : split_status;
    }
    result.splits.push_back(split);
    ++stats.boundary_splits;
  }
  // S3's size is an upper bound on each cell's circumradius; the request h is
  // passed as that bound unchanged. Refinement seeds the interior with the
  // coarsest lattice meeting it (refine3d.cpp): on Gmsh's quadric surface at
  // h = 0.02286 every cell has circumradius at most h, interior edges median
  // 1.52 h (HXT's 1.45 h), at 0.69x HXT's tet count; the regular-tet value
  // h * sqrt(6) / 4 would give several times that. Each surface point takes
  // the smallest h of its faces, the rest the smallest h overall; refinement
  // interpolates inside.
  const auto surface_points = surface.points.size();
  const auto& recovered = recovery->recovery;
  const std::size_t point_count = recovered.points.size() / 3;
  double smallest = std::numeric_limits<double>::infinity();
  for (int64_t f = 0; f < model.surface_count; ++f) smallest = std::min(smallest, sizes[f]);
  std::vector<double> targets(point_count, std::numeric_limits<double>::infinity());
  for (std::size_t t = 0; t < surface.triangles.size(); ++t) {
    const int32_t face = surface.triangle_patches[t];
    for (int32_t v : surface.triangles[t]) {
      targets[v] = std::min(targets[v], sizes[face]);
      // A proximity control's size at the point bounds it too.
      if (controls.proximity)
        targets[v] = std::min(targets[v], controls.proximity->size(face, surface.points[v].data()));
    }
  }
  for (double& target : targets) target = std::isfinite(target) ? target : smallest;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = static_cast<int64_t>(point_count);
  create.tetrahedron_count = static_cast<int64_t>(recovered.tets.size() / 4);
  create.face_count = static_cast<int64_t>(recovered.faces.size() / 3);
  create.face_sources = recovered.face_sources.data();
  create.segment_count = static_cast<int64_t>(recovered.segments.size() / 2);
  create.segment_sources = recovered.segment_sources.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  phx_mc_tet_mesh* raw = nullptr;
  create.result = &raw;
  const int32_t created = phx_mc_tet_mesh_create_from_plc(recovery.get(), &create, nullptr, 0);
  std::unique_ptr<phx_mc_tet_mesh, TetMeshDeleter> handle(raw);
  recovery.reset();
  if (created != PHX_MC_OK) return created;
  TetMesh& state = tet_mesh_state(*handle);
  const int32_t refined = refine_mesh(state, controls.radius_edge_bound, targets,
                                      stats.refine_inserted, stats.refine_unmet);
  if (refined != PHX_MC_OK && refined != PHX_MC_UNRESOLVED) return refined;
  const TetQualityRequest quality{controls.radius_edge_bound, controls.minimum_dihedral_degrees,
                                  controls.minimum_relative_determinant};
  const int32_t improved = improve_mesh(state, quality, {}, stats.improve_inserted,
                                        stats.improve_collapsed, stats.improve_relocated,
                                        stats.unmet_cells);
  if (improved != PHX_MC_OK && improved != PHX_MC_UNRESOLVED) return improved;
  result.curve_rim_wedges.assign(static_cast<std::size_t>(model.curve_count),
                                 std::numeric_limits<double>::quiet_NaN());
  if (stats.unmet_cells > 0) {
    // The split uses improve_mesh's own final records (one or more per
    // unmet cell, the set unmet_cells counts), not a second evaluation:
    // evaluate_mesh_quality also checks size targets and a determinant
    // floor, so it can name cells improve_mesh does not count.
    const int32_t counted = count_forced_cells(geometry, surface, state, state.unmet_records(),
                                               controls.minimum_dihedral_degrees,
                                               stats.unmet_forced_cells,
                                               stats.unmet_unforced_cells,
                                               &result.curve_rim_wedges);
    if (counted != PHX_MC_OK) return counted;
    if (stats.unmet_forced_cells + stats.unmet_unforced_cells != stats.unmet_cells)
      return PHX_MC_INTERNAL_ERROR;
  }
  // Live vertex slots in order; the surface keeps its rows.
  std::vector<int32_t> rows(state.vertex_slot_count(), -1);
  for (std::size_t slot = 0; slot < rows.size(); ++slot) {
    const double* p = state.point(static_cast<int32_t>(slot));
    if (!p) continue;
    rows[slot] = static_cast<int32_t>(result.points.size());
    result.points.push_back({p[0], p[1], p[2]});
  }
  if (result.points.size() < surface_points) return PHX_MC_INTERNAL_ERROR;
  for (std::size_t v = 0; v < surface_points; ++v)
    if (rows[v] != static_cast<int32_t>(v) || result.points[v] != surface.points[v])
      return PHX_MC_INTERNAL_ERROR;
  const auto& complex = state.complex();
  for (std::size_t slot = 0; slot < complex.tets.size(); ++slot) {
    const Tetrahedron& tet = complex.tets[slot];
    if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
    result.tetrahedra.push_back({rows[tet.v[0]], rows[tet.v[1]], rows[tet.v[2]], rows[tet.v[3]]});
    result.regions.push_back(complex.region(static_cast<int32_t>(slot)));
  }
  handle.reset();
  // The boundary face of each surface triangle: faces whose corners are all
  // surface points, matched by sorted corners; a triangle between two cells
  // takes the cell of region facet_regions[t][0] when that is a region.
  using Key = std::array<int32_t, 3>;
  struct Face {
    Key key;
    int32_t cell, local;
  };
  const auto sorted = [](Key k) {
    std::sort(k.begin(), k.end());
    return k;
  };
  std::vector<Face> faces;
  const auto limit = static_cast<int32_t>(surface_points);
  for (std::size_t c = 0; c < result.tetrahedra.size(); ++c) {
    const auto& v = result.tetrahedra[c];
    for (int local = 0; local < 4; ++local) {
      const Key key{v[(local + 1) % 4], v[(local + 2) % 4], v[(local + 3) % 4]};
      if (key[0] < limit && key[1] < limit && key[2] < limit)
        faces.push_back({sorted(key), static_cast<int32_t>(c), local});
    }
  }
  std::sort(faces.begin(), faces.end(), [](const Face& x, const Face& y) {
    return x.key != y.key ? x.key < y.key : x.cell < y.cell;
  });
  result.boundary_faces.resize(surface.triangles.size());
  for (std::size_t t = 0; t < surface.triangles.size(); ++t) {
    const Key key = sorted(surface.triangles[t]);
    auto it = std::lower_bound(faces.begin(), faces.end(), key,
                               [](const Face& f, const Key& k) { return f.key < k; });
    if (it == faces.end() || it->key != key) return PHX_MC_INTERNAL_ERROR;
    const auto next = std::next(it);
    if (next != faces.end() && next->key == key &&
        result.regions[next->cell] == surface.facet_regions[t][0] &&
        result.regions[it->cell] != surface.facet_regions[t][0])
      it = next;
    result.boundary_faces[t] = {it->cell, it->local};
  }
  stats.volume_ns = elapsed_ns(volume_start);
  // UNRESOLVED iff some unmet cell is not forced by the input
  // (count_forced_cells); forced cells are reported, not unmet.
  int32_t overall = stats.unmet_unforced_cells > 0 ? PHX_MC_UNRESOLVED : PHX_MC_OK;
  if (controls.geometry_order == 2) {
    const auto quadratic_start = std::chrono::steady_clock::now();
    const CadQuadraticInput input{
        static_cast<int64_t>(result.points.size()), reinterpret_cast<const double*>(result.points.data()),
        static_cast<int64_t>(surface_points), surface.vertex_dimensions.data(),
        surface.vertex_entities.data(), static_cast<int64_t>(result.tetrahedra.size()),
        reinterpret_cast<const int32_t*>(result.tetrahedra.data()),
        static_cast<int64_t>(surface.triangles.size()),
        reinterpret_cast<const int32_t*>(surface.triangles.data()), surface.triangle_patches.data(),
        reinterpret_cast<const double*>(surface.triangle_uv.data()),
        static_cast<int64_t>(surface.curve_edges.size()),
        reinterpret_cast<const int32_t*>(surface.curve_edges.data()),
        surface.curve_edge_curves.data(),
        reinterpret_cast<const double*>(surface.curve_edge_parameters.data()),
        controls.minimum_relative_determinant};
    const int32_t curved = cad_quadratic_nodes(geometry, input, result.quadratic);
    stats.quadratic_ns = elapsed_ns(quadratic_start);
    if (curved != PHX_MC_OK) return curved;
    stats.quadratic = result.quadratic.stats;
    if (stats.quadratic.final_invalid > 0 || stats.quadratic.final_unresolved > 0)
      overall = PHX_MC_UNRESOLVED;
  }
  return overall;
}

}  // namespace phx::mc
