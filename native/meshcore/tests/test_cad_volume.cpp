// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// CAD solid route, volume stage: fixed-boundary tetrahedra of the face-stage
// surface (valid, boundary faces equal to the surface triangles), P2 nodes on
// the surfaces, determinism, and the boundary-split retry.
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "cad_curves.hpp"
#include "cad_geometry.hpp"
#include "cad_mesh.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "cad_quadric_model.inc"
#include "cad_sphere_equator_rings.inc"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTurn = 2.0 * kPi;

struct Bank {
  std::vector<int32_t> kinds, surface_nodes, curve_nodes, pcurve_nodes;
  std::vector<int64_t> integer_offsets{0}, integers, real_offsets{0};
  std::vector<double> reals, corners, ranges, use_ranges, periods, domains;
  std::vector<int32_t> curve_corners, use_entities, loop_surfaces;
  std::vector<int8_t> use_kinds, senses, orientation;
  std::vector<int64_t> loop_offsets{0}, region_offsets{0, 1};
  std::vector<int32_t> region_surfaces{0};
  std::vector<int8_t> region_orientations{1};

  int32_t add(const char* name, const std::vector<double>& values) {
    const int32_t kind = phx::mc::cad::template_index(name);
    PHX_CHECK(kind >= 0);
    kinds.push_back(PHX_MC_CAD_NODE_TEMPLATE);
    integers.push_back(kind);
    integer_offsets.push_back(static_cast<int64_t>(integers.size()));
    reals.insert(reals.end(), values.begin(), values.end());
    real_offsets.push_back(static_cast<int64_t>(reals.size()));
    return static_cast<int32_t>(kinds.size() - 1);
  }

  phx_mc_cad_model model() const {
    phx_mc_cad_model m{};
    m.node_count = static_cast<int64_t>(kinds.size());
    m.node_kinds = kinds.data();
    m.node_integer_offsets = integer_offsets.data();
    m.node_integers = integers.data();
    m.node_real_offsets = real_offsets.data();
    m.node_reals = reals.data();
    m.corner_count = static_cast<int64_t>(corners.size() / 3);
    m.corner_points = corners.data();
    m.surface_count = static_cast<int64_t>(surface_nodes.size());
    m.surface_nodes = surface_nodes.data();
    m.surface_orientations = orientation.data();
    m.surface_periods = periods.data();
    m.surface_parameter_domains = domains.data();
    m.curve_count = static_cast<int64_t>(curve_nodes.size());
    m.curve_nodes = curve_nodes.data();
    m.curve_corners = curve_corners.data();
    m.curve_ranges = ranges.data();
    m.loop_count = static_cast<int64_t>(loop_surfaces.size());
    m.loop_use_offsets = loop_offsets.data();
    m.loop_surfaces = loop_surfaces.data();
    m.use_count = static_cast<int64_t>(use_kinds.size());
    m.use_kinds = use_kinds.data();
    m.use_entities = use_entities.data();
    m.use_senses = senses.data();
    m.use_ranges = use_ranges.data();
    m.use_pcurve_nodes = pcurve_nodes.data();
    m.region_count = 1;
    m.region_surface_offsets = region_offsets.data();
    m.region_surfaces = region_surfaces.data();
    m.region_orientations = region_orientations.data();
    return m;
  }
};

const std::vector<double> kFrame{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

// Unit sphere: one meridian seam curve between the two pole corners.
Bank sphere_bank() {
  Bank bank;
  std::vector<double> frame = kFrame;
  frame.push_back(1.0);
  bank.surface_nodes = {bank.add("sphere", frame)};
  bank.curve_nodes = {bank.add("circle3", {0, 0, 0, 1, 0, 0, 0, 0, 1, 1})};
  bank.corners = {0, 0, -1, 0, 0, 1};
  bank.curve_corners = {0, 1};
  bank.ranges = {-kPi / 2, kPi / 2};
  bank.pcurve_nodes = {bank.add("line2", {0, -kPi / 2, 1, 0}), bank.add("line2", {kTurn, 0, 0, 1}),
                       bank.add("line2", {0, kPi / 2, 1, 0}), bank.add("line2", {0, 0, 0, 1})};
  bank.use_kinds = {1, 0, 1, 0};
  bank.use_entities = {0, 0, 1, 0};
  bank.senses = {1, 1, -1, -1};
  bank.use_ranges = {0, kTurn, -kPi / 2, kPi / 2, 0, kTurn, -kPi / 2, kPi / 2};
  bank.loop_offsets = {0, 4};
  bank.loop_surfaces = {0};
  bank.orientation = {1};
  bank.periods = {kTurn, 0};
  bank.domains = {0, kTurn, -kPi / 2, kPi / 2};
  return bank;
}


phx_mc_cad_model quadric_model() {
  phx_mc_cad_model model{};
  model.node_count = quadric_node_kinds.size();
  model.node_kinds = quadric_node_kinds.data();
  model.node_integer_offsets = quadric_node_integer_offsets.data();
  model.node_integers = quadric_node_integers.data();
  model.node_real_offsets = quadric_node_real_offsets.data();
  model.node_reals = quadric_node_reals.data();
  model.corner_count = quadric_corner_points.size() / 3;
  model.corner_points = quadric_corner_points.data();
  model.surface_count = quadric_surface_nodes.size();
  model.surface_nodes = quadric_surface_nodes.data();
  model.surface_orientations = quadric_surface_orientations.data();
  model.surface_periods = quadric_surface_periods.data();
  model.surface_parameter_domains = quadric_surface_parameter_domains.data();
  model.curve_count = quadric_curve_nodes.size();
  model.curve_nodes = quadric_curve_nodes.data();
  model.curve_corners = quadric_curve_corners.data();
  model.curve_ranges = quadric_curve_ranges.data();
  model.loop_count = quadric_loop_surfaces.size();
  model.loop_use_offsets = quadric_loop_use_offsets.data();
  model.loop_surfaces = quadric_loop_surfaces.data();
  model.use_count = quadric_use_kinds.size();
  model.use_kinds = quadric_use_kinds.data();
  model.use_entities = quadric_use_entities.data();
  model.use_senses = quadric_use_senses.data();
  model.use_ranges = quadric_use_ranges.data();
  model.use_pcurve_nodes = quadric_use_pcurve_nodes.data();
  model.region_count = quadric_region_surface_offsets.size() - 1;
  model.region_surface_offsets = quadric_region_surface_offsets.data();
  model.region_surfaces = quadric_region_surfaces.data();
  model.region_orientations = quadric_region_orientations.data();
  return model;
}

// Exported solid (public C API).
struct Solid {
  int32_t status = -1;
  std::vector<double> points, nodes, uv, surface_points;
  std::vector<double> cubic_nodes;
  std::vector<int32_t> tetrahedra, regions, boundary, edges, entities, decisions, triangles,
      patches;
  std::vector<int8_t> kinds;
  std::vector<int64_t> cell_edges, records;
  std::vector<int64_t> cubic_cell_nodes;
  std::vector<phx_mc_cad_boundary_split> splits;
  phx_mc_cad_volume_stats stats{};
};

Solid solid(const phx_mc_cad_model& model, double size, int32_t order,
            double minimum_dihedral = 15.5) {
  Solid s;
  const std::vector<double> sizes(static_cast<std::size_t>(model.surface_count), size);
  const std::vector<double> tolerances(static_cast<std::size_t>(model.surface_count), 0.4);
  const std::vector<double> curve_tolerances(static_cast<std::size_t>(model.curve_count),
                                             std::numeric_limits<double>::infinity());
  const phx_mc_cad_surface_controls controls{sizes.data(), tolerances.data(),
                                             curve_tolerances.data(), 0.5, 1.3, 1e-8};
  const phx_mc_cad_volume_controls volume{2.0, minimum_dihedral, 0.0, order};
  phx_mc_cad_surface_result* handle = nullptr;
  s.status = phx_mc_cad_mesh_volume(&model, &controls, &volume, &handle, 1);
  PHX_CHECK(handle != nullptr);
  if (!handle) return s;
  int64_t n[9]{}, m[7]{};
  PHX_CHECK(phx_mc_cad_surface_result_sizes(handle, n) == PHX_MC_OK);
  PHX_CHECK(phx_mc_cad_volume_bank_sizes(handle, m) == PHX_MC_OK);
  const auto z = [](int64_t v) { return static_cast<std::size_t>(v); };
  // The surface rows the volume refers to.
  std::vector<double> parameters(2 * z(n[0])), sliver_uv(6 * z(n[2])), edge_parameters(2 * z(n[3])),
      curve_parameters(z(n[6]));
  std::vector<int8_t> dimensions(z(n[0]));
  std::vector<int32_t> vertex_entities(z(n[0])), uses(3 * z(n[1])), facet_regions(2 * z(n[1])),
      sliver_patches(z(n[2])), curve_edges(2 * z(n[3])), edge_curves(z(n[3]));
  std::vector<phx_mc_cad_face_stats> face_stats(z(n[4]));
  std::vector<int64_t> offsets(z(n[5]) + 1), sag(z(n[5]));
  std::vector<phx_mc_cad_curve_failure> curve_failures(z(n[7]));
  std::vector<phx_mc_cad_face_failure> face_failures(z(n[8]));
  phx_mc_cad_curve_statistics curve_stats{};
  s.surface_points.resize(3 * z(n[0]));
  s.triangles.resize(3 * z(n[1]));
  s.patches.resize(z(n[1]));
  s.uv.resize(6 * z(n[1]));
  const phx_mc_cad_surface_arrays arrays{
      n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8],
      s.surface_points.data(), dimensions.data(), vertex_entities.data(), parameters.data(),
      s.triangles.data(), s.patches.data(), s.uv.data(), uses.data(), facet_regions.data(),
      sliver_uv.data(), sliver_patches.data(), curve_edges.data(), edge_curves.data(),
      edge_parameters.data(), face_stats.data(), offsets.data(), curve_parameters.data(),
      sag.data(), curve_failures.data(), &curve_stats, face_failures.data()};
  PHX_CHECK(phx_mc_cad_surface_result_export(handle, &arrays) == PHX_MC_OK);
  s.points.resize(3 * z(m[0]));
  s.tetrahedra.resize(4 * z(m[1]));
  s.regions.resize(z(m[1]));
  s.boundary.resize(2 * z(m[2]));
  s.edges.resize(2 * z(m[3]));
  s.nodes.resize(3 * z(m[3]));
  s.kinds.resize(z(m[3]));
  s.entities.resize(z(m[3]));
  s.cell_edges.resize(m[3] ? 6 * z(m[1]) : 0);
  s.decisions.resize(m[3] ? z(m[1]) : 0);
  s.records.resize(2 * z(m[4]));
  s.splits.resize(z(m[5]));
  s.cubic_nodes.resize(3 * z(m[6]));
  s.cubic_cell_nodes.resize(m[6] ? 16 * z(m[1]) : 0);
  const phx_mc_cad_volume_arrays out{
      m[0], m[1], m[2], m[3], m[4], m[5], s.points.data(), s.tetrahedra.data(), s.regions.data(),
      s.boundary.data(), s.edges.data(), s.nodes.data(), s.kinds.data(), s.entities.data(),
      s.cell_edges.data(), s.decisions.data(), s.records.data(), s.splits.data(), &s.stats,
      0, nullptr, m[6], s.cubic_nodes.data(), s.cubic_cell_nodes.data()};
  PHX_CHECK(phx_mc_cad_volume_copy_banks(handle, &out) == PHX_MC_OK);
  phx_mc_cad_volume_arrays small = out;
  small.point_capacity = m[0] - 1;
  if (m[0] > 0)
    PHX_CHECK(phx_mc_cad_volume_copy_banks(handle, &small) == PHX_MC_CAPACITY_EXCEEDED);
  phx_mc_cad_surface_result_free(handle);
  return s;
}

// Cells valid and embedded (verify_tetrahedra), each surface triangle the
// face its boundary row names, and the boundary of the cells exactly the
// surface.
void check_volume(const char* name, const std::vector<double>& points,
                  const std::vector<double>& surface_points, const std::vector<int32_t>& cells,
                  const std::vector<int32_t>& triangles, const std::vector<int32_t>& boundary) {
  const auto count = static_cast<int64_t>(cells.size() / 4);
  PHX_CHECK(count > 0);
  PHX_CHECK(points.size() >= surface_points.size());
  PHX_CHECK(std::memcmp(points.data(), surface_points.data(),
                        surface_points.size() * sizeof(double)) == 0);
  phx_mc_tetrahedra_verdict verdict{};
  PHX_CHECK(phx_mc_verify_tetrahedra(static_cast<int64_t>(points.size() / 3), points.data(), count,
                                     cells.data(), &verdict) == PHX_MC_OK);
  PHX_CHECK(verdict.status == PHX_MC_VERDICT_CERTIFIED);
  const auto triangle_count = static_cast<int64_t>(triangles.size() / 3);
  PHX_CHECK(verdict.boundary_faces == triangle_count);
  PHX_CHECK(boundary.size() == 2 * triangles.size() / 3);
  for (int64_t t = 0; t < triangle_count; ++t) {
    const int32_t cell = boundary[2 * t], local = boundary[2 * t + 1];
    PHX_CHECK(cell >= 0 && cell < count && local >= 0 && local < 4);
    if (cell < 0 || cell >= count || local < 0 || local >= 4) return;
    std::array<int32_t, 3> face{}, triangle{};
    for (int k = 0; k < 3; ++k) {
      face[k] = cells[4 * cell + (local + 1 + k) % 4];
      triangle[k] = triangles[3 * t + k];
    }
    std::sort(face.begin(), face.end());
    std::sort(triangle.begin(), triangle.end());
    PHX_CHECK(face == triangle);
  }
  std::printf("%s: %" PRId64 " cells, %" PRId64 " triangles\n", name, count, triangle_count);
}

template <class OnSurface>
void check_nodes(const Solid& s, OnSurface on_surface) {
  int64_t curved = 0;
  for (std::size_t e = 0; e < s.kinds.size(); ++e) {
    if (s.kinds[e] == PHX_MC_CAD_EDGE_INTERIOR) continue;
    ++curved;
    PHX_CHECK(on_surface(s.nodes.data() + 3 * e));
  }
  PHX_CHECK(curved > 0);
  for (int32_t decision : s.decisions) PHX_CHECK(decision == PHX_MC_QUADRATIC_VALIDITY_VALID);
  PHX_CHECK(s.stats.quadratic.final_invalid == 0 && s.stats.quadratic.final_unresolved == 0);
}

bool on_sphere(const double* x) {
  return std::fabs(std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) - 1.0) < 1e-12;
}

// z cylinder radius 1 (|z| <= 1.2), x cylinder radius 0.6 (|x| <= 1.5),
// their planar caps.
bool on_quadric(const double* x) {
  const double tol = 1e-12;
  return std::fabs(std::hypot(x[0], x[1]) - 1.0) < tol ||
         std::fabs(std::hypot(x[1], x[2]) - 0.6) < tol || std::fabs(std::fabs(x[2]) - 1.2) < tol ||
         std::fabs(std::fabs(x[0]) - 1.5) < tol;
}

template <class OnSurface>
void solid_case(const char* name, const phx_mc_cad_model& model, double size, OnSurface on_surface) {
  const Solid first = solid(model, size, 2);
  PHX_CHECK(first.status == PHX_MC_OK || first.status == PHX_MC_UNRESOLVED);
  PHX_CHECK(first.stats.recovery_status == PHX_MC_OK && first.stats.recovery_attempts == 1);
  PHX_CHECK(first.splits.empty());
  check_volume(name, first.points, first.surface_points, first.tetrahedra, first.triangles,
               first.boundary);
  check_nodes(first, on_surface);
  for (int32_t region : first.regions) PHX_CHECK(region == 0);
  PHX_CHECK(first.stats.volume_ns > 0 && first.stats.quadratic_ns > 0 &&
            first.stats.face_ns > 0 && first.stats.curve_ns > 0);
  std::printf("%s: unmet %" PRId64 ", refine +%" PRId64 ", improve +%" PRId64 "\n", name,
              first.stats.unmet_cells, first.stats.refine_inserted, first.stats.improve_inserted);
  // One set: forced and unforced cells partition unmet_cells, and the
  // status is UNRESOLVED exactly when some unmet cell is unforced.
  PHX_CHECK(first.stats.unmet_forced_cells + first.stats.unmet_unforced_cells ==
            first.stats.unmet_cells);
  PHX_CHECK((first.status == PHX_MC_UNRESOLVED) == (first.stats.unmet_unforced_cells > 0));
  // Determinism: the same arrays bit for bit.
  const Solid second = solid(model, size, 2);
  PHX_CHECK(second.status == first.status);
  PHX_CHECK(second.points.size() == first.points.size() &&
            std::memcmp(second.points.data(), first.points.data(),
                        first.points.size() * sizeof(double)) == 0);
  PHX_CHECK(second.tetrahedra == first.tetrahedra && second.boundary == first.boundary &&
            second.edges == first.edges);
  PHX_CHECK(second.nodes.size() == first.nodes.size() &&
            std::memcmp(second.nodes.data(), first.nodes.data(),
                        first.nodes.size() * sizeof(double)) == 0);
  // Order 1: the same cells, no P2 rows.
  const Solid linear = solid(model, size, 1);
  PHX_CHECK(linear.tetrahedra == first.tetrahedra && linear.edges.empty() &&
            linear.decisions.empty());
}

void cubic_sphere_exports_curved_boundary_face_nodes(const phx_mc_cad_model& model) {
  const Solid mesh = solid(model, 0.3, 3);
  PHX_CHECK(mesh.status == PHX_MC_OK);
  if (mesh.status != PHX_MC_OK) return;
  PHX_CHECK(!mesh.tetrahedra.empty() && !mesh.triangles.empty());
  PHX_CHECK(mesh.cubic_cell_nodes.size() == 4 * mesh.tetrahedra.size());
  if (mesh.cubic_cell_nodes.size() != 4 * mesh.tetrahedra.size()) return;
  for (std::size_t row = 0; row < mesh.boundary.size() / 2; ++row) {
    const int32_t cell = mesh.boundary[2 * row], local = mesh.boundary[2 * row + 1];
    const auto slot = 16 * static_cast<std::size_t>(cell) + 12 + local;
    PHX_CHECK(slot < mesh.cubic_cell_nodes.size());
    if (slot >= mesh.cubic_cell_nodes.size()) continue;
    const auto node = static_cast<std::size_t>(mesh.cubic_cell_nodes[slot]);
    PHX_CHECK(node < mesh.cubic_nodes.size() / 3);
    if (node >= mesh.cubic_nodes.size() / 3) continue;
    PHX_CHECK(on_sphere(mesh.cubic_nodes.data() + 3 * node));
  }
}

// The boundary-split retry: the first recovery is refused as if it needed a
// boundary Steiner point on (a, b); the edge is split on the source and the
// second recovery succeeds.
template <class OnSurface>
void forced_split(const char* name, const phx_mc_cad_model& model, double size, bool curve_edge,
                  OnSurface on_surface) {
  phx::mc::cad::CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const std::vector<double> sizes(static_cast<std::size_t>(model.surface_count), size);
  const std::vector<double> tolerances(static_cast<std::size_t>(model.surface_count), 0.4);
  const std::vector<double> curve_tolerances(static_cast<std::size_t>(model.curve_count),
                                             std::numeric_limits<double>::infinity());
  phx::mc::cad::CurveDiscretization curves;
  const phx::mc::cad::CurveControls controls{sizes.data(), tolerances.data(),
                                             curve_tolerances.data(), 0.5, 1.3, 1e-8};
  PHX_CHECK(phx::mc::cad::discretize_curves(geometry, controls, curves) == PHX_MC_OK);
  phx::mc::CadSurfaceMesh surface;
  PHX_CHECK(phx::mc::cad_mesh_surface(geometry, curves.view(), sizes.data(),
                                      tolerances.data(), 0.5, surface) == PHX_MC_OK);
  const std::size_t triangles = surface.triangles.size(), points = surface.points.size();
  const std::size_t parameters = curves.parameters.size();
  phx::mc::CadForcedRefusal forced;
  if (curve_edge) {
    // The first curve edge between two curve-interior samples.
    for (const auto& e : surface.curve_edges)
      if (forced.a < 0 && surface.vertex_dimensions[e[0]] == 1 &&
          surface.vertex_dimensions[e[1]] == 1) {
        forced.a = e[0];
        forced.b = e[1];
      }
  } else {
    // The first triangle edge between two face-interior vertices.
    for (const auto& t : surface.triangles) {
      for (int k = 0; k < 3 && forced.a < 0; ++k)
        if (surface.vertex_dimensions[t[k]] == 2 && surface.vertex_dimensions[t[(k + 1) % 3]] == 2) {
          forced.a = t[k];
          forced.b = t[(k + 1) % 3];
        }
      if (forced.a >= 0) break;
    }
  }
  PHX_CHECK(forced.a >= 0);
  phx::mc::CadVolumeMesh volume;
  const int32_t status = phx::mc::cad_mesh_volume(geometry, curves, surface, sizes.data(),
                                                  phx::mc::CadVolumeControls{2.0, 15.5, 0.0, 2},
                                                  volume, &forced);
  PHX_CHECK(status == PHX_MC_OK || status == PHX_MC_UNRESOLVED);
  PHX_CHECK(volume.splits.size() == 1 && volume.stats.boundary_splits == 1);
  PHX_CHECK(volume.stats.recovery_attempts == 1 && volume.stats.recovery_status == PHX_MC_OK);
  PHX_CHECK(surface.triangles.size() == triangles + 2 && surface.points.size() == points + 1);
  if (volume.splits.size() != 1) return;
  const auto& split = volume.splits[0];
  PHX_CHECK(split.vertex == static_cast<int32_t>(points));
  PHX_CHECK(split.dimension == (curve_edge ? 1 : 2));
  PHX_CHECK(on_surface(surface.points[points].data()));
  PHX_CHECK(curves.parameters.size() == parameters + (curve_edge ? 1 : 0));
  for (std::size_t c = 0; c + 1 < curves.offsets.size(); ++c)
    PHX_CHECK(std::is_sorted(curves.parameters.begin() + curves.offsets[c],
                             curves.parameters.begin() + curves.offsets[c + 1]));
  // The split edge is gone; its halves are surface (and curve) edges.
  for (const auto& t : surface.triangles)
    for (int k = 0; k < 3; ++k) {
      const int32_t x = t[k], y = t[(k + 1) % 3];
      PHX_CHECK(!((x == forced.a && y == forced.b) || (x == forced.b && y == forced.a)));
    }
  std::vector<double> flat_points, flat_surface;
  for (const auto& p : volume.points) flat_points.insert(flat_points.end(), p.begin(), p.end());
  for (const auto& p : surface.points) flat_surface.insert(flat_surface.end(), p.begin(), p.end());
  std::vector<int32_t> cells, faces, boundary;
  for (const auto& c : volume.tetrahedra) cells.insert(cells.end(), c.begin(), c.end());
  for (const auto& t : surface.triangles) faces.insert(faces.end(), t.begin(), t.end());
  for (const auto& b : volume.boundary_faces) boundary.insert(boundary.end(), b.begin(), b.end());
  check_volume(name, flat_points, flat_surface, cells, faces, boundary);
  // The new vertex's P2 neighbors stay on the source.
  for (std::size_t e = 0; e < volume.quadratic.edges.size(); ++e)
    if (volume.quadratic.edge_kinds[e] != PHX_MC_CAD_EDGE_INTERIOR)
      PHX_CHECK(on_surface(volume.quadratic.nodes[e].data()));
  PHX_CHECK(volume.quadratic.stats.final_invalid == 0);
  // The chart evidence still verifies after the split.
  const auto& chart = surface.chart;
  std::vector<int64_t> chart_triangles;
  for (const auto& t : chart.triangles) chart_triangles.insert(chart_triangles.end(), t.begin(), t.end());
  std::vector<double> chart_uv, use_range, chord_uv, chord_range;
  for (const auto& r : chart.uv) chart_uv.insert(chart_uv.end(), r.begin(), r.end());
  for (const auto& r : chart.use_range) use_range.insert(use_range.end(), r.begin(), r.end());
  for (const auto& r : chart.chord_uv) chord_uv.insert(chord_uv.end(), r.begin(), r.end());
  for (const auto& r : chart.chord_range) chord_range.insert(chord_range.end(), r.begin(), r.end());
  int64_t evidence = -1;
  std::vector<phx_mc_chart_failure> failures(16);
  PHX_CHECK(phx_mc_verify_chart_partition(
                model.surface_count, static_cast<int64_t>(chart.patches.size()), chart.patches.data(),
                chart_uv.data(), static_cast<int64_t>(chart.use_patch.size()), chart.use_patch.data(),
                use_range.data(), model.loop_count, static_cast<int64_t>(chart.chord_use.size()),
                chart.chord_use.data(), chart.chord_loop.data(), chord_uv.data(), chord_range.data(),
                static_cast<int64_t>(failures.size()), failures.data(), &evidence) == PHX_MC_OK);
  PHX_CHECK(evidence == 0);
}

// A surface with zero-area triangles (the sphere's former coincident equator
// rings) is refused by recovery (S3's flat-facet refusal), naming the first
// such triangle and its smallest-angle sine, instead of minutes of improvement.
void flat_surface_triangle() {
  const Bank sphere = sphere_bank();
  const phx_mc_cad_model model = sphere.model();
  phx::mc::cad::CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const std::vector<double> sizes{0.3}, tolerances{0.05};
  const std::vector<double> curve_tolerances(static_cast<std::size_t>(model.curve_count), 0.05);
  phx::mc::cad::CurveDiscretization curves;
  const phx::mc::cad::CurveControls controls{sizes.data(), tolerances.data(),
                                             curve_tolerances.data(), 0.8, 1.3, 1e-8};
  PHX_CHECK(phx::mc::cad::discretize_curves(geometry, controls, curves) == PHX_MC_OK);
  phx::mc::CadSurfaceMesh surface;
  surface.points = kEquatorRingPoints;
  surface.triangles = kEquatorRingTriangles;
  surface.triangle_patches.assign(surface.triangles.size(), 0);
  surface.facet_regions.assign(surface.triangles.size(), {0, -1});
  phx::mc::CadVolumeMesh volume;
  const int32_t status = phx::mc::cad_mesh_volume(
      geometry, curves, surface, sizes.data(),
      phx::mc::CadVolumeControls{2.0, 15.5, 0x1p-35, 1}, volume);
  PHX_CHECK(status == PHX_MC_UNRESOLVED);
  PHX_CHECK(volume.tetrahedra.empty() && volume.points.empty());
  const auto& stats = volume.stats;
  PHX_CHECK(stats.recovery_attempts == 1 && stats.refine_inserted == 0 &&
            stats.improve_inserted == 0);
  PHX_CHECK(stats.recovery_status == PHX_MC_UNRESOLVED);
  PHX_CHECK(stats.recovery_reason == PHX_MC_PLC_INVALID_FACET);
  PHX_CHECK(stats.failure_kind == PHX_MC_PLC_ENTITY_FACET && stats.failure_entity == 8);
  PHX_CHECK(stats.failure_measure >= 0.0 && stats.failure_measure < 1e-12);
  PHX_CHECK(stats.volume_ns < 100000000);
}

// A dihedral target of 40 deg the interior cannot meet at h 0.2293: unmet
// cells exist, and forced plus unforced is exactly unmet_cells (fails when
// the split is taken from a different evaluation than the count, e.g. one
// adding size records: 259 split cells for 256 unmet measured on a cone).
void strict_schedule() {
  const Solid strict = solid(quadric_model(), 0.2293, 1, 40.0);
  PHX_CHECK(strict.stats.unmet_cells > 0);
  PHX_CHECK(strict.stats.unmet_forced_cells + strict.stats.unmet_unforced_cells ==
            strict.stats.unmet_cells);
  PHX_CHECK(strict.stats.unmet_unforced_cells > 0 && strict.status == PHX_MC_UNRESOLVED);
}

// Invalid controls are refused before any work.
void refusals() {
  const phx_mc_cad_model model = quadric_model();
  const std::vector<double> sizes(static_cast<std::size_t>(model.surface_count), 0.3);
  const std::vector<double> tolerances(static_cast<std::size_t>(model.surface_count), 0.4);
  const std::vector<double> curve_tolerances(static_cast<std::size_t>(model.curve_count),
                                             std::numeric_limits<double>::infinity());
  const phx_mc_cad_surface_controls controls{sizes.data(), tolerances.data(),
                                             curve_tolerances.data(), 0.5, 1.3, 1e-8};
  const phx_mc_cad_volume_controls bad{2.0, 15.5, 0.0, 4};
  phx_mc_cad_surface_result* handle = reinterpret_cast<phx_mc_cad_surface_result*>(&handle);
  PHX_CHECK(phx_mc_cad_mesh_volume(&model, &controls, &bad, &handle, 1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(handle == nullptr);
  PHX_CHECK(phx_mc_cad_mesh_volume(&model, &controls, nullptr, &handle, 1) == PHX_MC_INVALID_ARGUMENT);
  const phx_mc_cad_volume_controls linear{2.0, 15.5, 0.0, 1};
  PHX_CHECK(phx_mc_cad_mesh_volume(&model, &controls, &linear, &handle, 0) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(handle == nullptr);
  int64_t sizes_out[7];
  PHX_CHECK(phx_mc_cad_volume_bank_sizes(nullptr, sizes_out) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_cad_volume_copy_banks(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
}

// The proximity field over two parallel unit squares 0.25 apart (face 0 at
// z = 0 facing -z, face 1 at z = 0.25 facing +z; face 2 unscoped): a point's
// size is its gap to the other square less that square's deviation bound,
// over elements_per_gap, clamped; a face no control scopes is unbounded.
void proximity_field() {
  using phx::mc::cad::ProximityControl;
  using phx::mc::cad::ProximityField;
  const std::vector<std::array<double, 3>> points{
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
      {0, 0, 0.25}, {1, 0, 0.25}, {1, 1, 0.25}, {0, 1, 0.25}};
  const std::vector<std::array<int32_t, 3>> facing{{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}};
  const std::vector<int32_t> faces{0, 0, 1, 1};
  const std::vector<double> deviations{0.0, 0.0, 0.05, 0.05};
  const double bottom[3] = {0.5, 0.5, 0.0}, top[3] = {0.5, 0.5, 0.25};
  ProximityControl control;
  control.source_faces = {0};
  control.target_faces = {1};
  ProximityField field;
  PHX_CHECK(field.build({control}, points, facing, faces, deviations, 3) == PHX_MC_OK);
  PHX_CHECK(std::abs(field.size(0, bottom) - 0.20) < 1e-15);
  PHX_CHECK(std::abs(field.size(1, top) - 0.25) < 1e-15);
  PHX_CHECK(std::isinf(field.size(2, top)));
  ProximityControl halved = control;
  halved.elements_per_gap = 2;
  ProximityControl clamped = control;
  clamped.maximum_size = 0.1;
  PHX_CHECK(field.build({halved}, points, facing, faces, deviations, 3) == PHX_MC_OK);
  PHX_CHECK(std::abs(field.size(1, top) - 0.125) < 1e-15);
  PHX_CHECK(field.build({halved, clamped}, points, facing, faces, deviations, 3) == PHX_MC_OK);
  PHX_CHECK(field.size(1, top) == 0.1);
  // The top square turned to face -z no longer faces the bottom one: no gap.
  const std::vector<std::array<int32_t, 3>> turned{{0, 2, 1}, {0, 3, 2}, {4, 6, 5}, {4, 7, 6}};
  PHX_CHECK(field.build({control}, points, turned, faces, deviations, 3) == PHX_MC_OK);
  PHX_CHECK(std::isinf(field.size(0, bottom)));
  control.opposite_normals_only = false;
  PHX_CHECK(field.build({control}, points, turned, faces, deviations, 3) == PHX_MC_OK);
  PHX_CHECK(std::abs(field.size(0, bottom) - 0.20) < 1e-15);
  // A gap bound reaching zero gives zero without minimum_size.
  const std::vector<double> loose{0.0, 0.0, 0.3, 0.3};
  PHX_CHECK(field.build({control}, points, turned, faces, loose, 3) == PHX_MC_OK);
  PHX_CHECK(field.size(0, bottom) == 0.0);
  control.minimum_size = 0.01;
  PHX_CHECK(field.build({control}, points, turned, faces, loose, 3) == PHX_MC_OK);
  PHX_CHECK(field.size(0, bottom) == 0.01);
  // Overlapping scopes, an out-of-range face and an empty scope are refused.
  ProximityControl overlapping = control;
  overlapping.target_faces = {0, 1};
  PHX_CHECK(field.build({overlapping}, points, facing, faces, deviations, 3) ==
            PHX_MC_INVALID_INPUT);
  ProximityControl outside = control;
  outside.target_faces = {3};
  PHX_CHECK(field.build({outside}, points, facing, faces, deviations, 3) == PHX_MC_INVALID_INPUT);
  ProximityControl empty = control;
  empty.target_faces = {2};
  PHX_CHECK(field.build({empty}, points, facing, faces, deviations, 3) == PHX_MC_INVALID_INPUT);
  // The deviation bound: zero on a plane, the cap sag 1 - sqrt(1 - R^2) on a
  // unit sphere, the tolerance beyond it.
  PHX_CHECK(phx::mc::cad::proxy_deviation(0.3, 0.0, 1.0) == 0.0);
  PHX_CHECK(std::abs(phx::mc::cad::proxy_deviation(0.3, 1.0, 1.0) -
                     (1.0 - std::sqrt(1.0 - 0.09))) < 1e-15);
  PHX_CHECK(phx::mc::cad::proxy_deviation(0.3, 1.0, 0.01) == 0.01);
  PHX_CHECK(phx::mc::cad::proxy_deviation(2.0, 1.0, 0.5) == 0.5);
}

}  // namespace

int main() {
  const Bank sphere = sphere_bank();
  const phx_mc_cad_model sphere_model = sphere.model();
  const phx_mc_cad_model quadric = quadric_model();
  solid_case("sphere", sphere_model, 0.3, on_sphere);
  solid_case("quadric", quadric, 0.2293, on_quadric);
  cubic_sphere_exports_curved_boundary_face_nodes(sphere_model);
  forced_split("sphere curve split", sphere_model, 0.3, true, on_sphere);
  forced_split("quadric face split", quadric, 0.2293, false, on_quadric);
  forced_split("quadric curve split", quadric, 0.2293, true, on_quadric);
  flat_surface_triangle();
  strict_schedule();
  refusals();
  proximity_field();
  return phx::mc::test::finish("test_cad_volume");
}
