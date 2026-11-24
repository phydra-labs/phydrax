// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// P2 curving from stored parameters: nodes on closed-form surfaces and curves,
// certified validity, forced repairs, determinism.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "cad_cubic_shape.hpp"
#include "cad_curving.hpp"
#include "cad_geometry.hpp"
#include "floating_point_mode.hpp"
#include "improve3d.hpp"
#include "refine3d.hpp"
#include "quadratic_validity_screen.hpp"
#include "tet_mesh_handle.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "cad_quadric_model.inc"

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

// Torus R = 2, r = 0.5: its inner side is concave (negative Gaussian curvature).
Bank torus_bank() {
  Bank bank;
  std::vector<double> coefficients = kFrame;
  coefficients.insert(coefficients.end(), {2.0, 0.5});
  bank.surface_nodes = {bank.add("torus", coefficients)};
  bank.curve_nodes = {bank.add("circle3", {0, 0, 0, 1, 0, 0, 0, 1, 0, 2.5}),
                      bank.add("circle3", {2, 0, 0, 1, 0, 0, 0, 0, 1, 0.5})};
  bank.corners = {2.5, 0, 0};
  bank.curve_corners = {0, 0, 0, 0};
  bank.ranges = {0, kTurn, 0, kTurn};
  bank.pcurve_nodes = {bank.add("line2", {0, 0, 1, 0}), bank.add("line2", {kTurn, 0, 0, 1}),
                       bank.add("line2", {0, kTurn, 1, 0}), bank.add("line2", {0, 0, 0, 1})};
  bank.use_kinds = {0, 0, 0, 0};
  bank.use_entities = {0, 1, 0, 1};
  bank.senses = {1, 1, -1, -1};
  bank.use_ranges = {0, kTurn, 0, kTurn, 0, kTurn, 0, kTurn};
  bank.loop_offsets = {0, 4};
  bank.loop_surfaces = {0};
  bank.orientation = {1};
  bank.periods = {kTurn, kTurn};
  bank.domains = {0, kTurn, 0, kTurn};
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

// The face stage's exported surface (public C API).
struct Surface {
  std::vector<double> points, parameters, uv, sliver_uv, edge_parameters, curve_parameters;
  std::vector<int8_t> dimensions;
  std::vector<int32_t> entities, triangles, patches, uses, regions, sliver_patches, edges,
      edge_curves;
  std::vector<phx_mc_cad_face_stats> stats;
  std::vector<int64_t> curve_offsets, sag_splits;
  std::vector<phx_mc_cad_curve_failure> curve_failures;
  std::vector<phx_mc_cad_face_failure> face_failures;
  phx_mc_cad_curve_statistics curve_stats{};
};

bool surface_mesh(const phx_mc_cad_model& model, double size, Surface& s) {
  const std::vector<double> sizes(static_cast<std::size_t>(model.surface_count), size);
  const std::vector<double> tolerances(static_cast<std::size_t>(model.surface_count), 0.4);
  const std::vector<double> curve_tolerances(static_cast<std::size_t>(model.curve_count),
                                             std::numeric_limits<double>::infinity());
  const phx_mc_cad_surface_controls controls{sizes.data(), tolerances.data(),
                                             curve_tolerances.data(), 0.5, 1.3, 1e-8};
  phx_mc_cad_surface_result* handle = nullptr;
  const int32_t status = phx_mc_cad_mesh_surface(&model, &controls, &handle, 1);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK || handle == nullptr) {
    phx_mc_cad_surface_result_free(handle);
    return false;
  }
  int64_t n[9]{};
  PHX_CHECK(phx_mc_cad_surface_result_sizes(handle, n) == PHX_MC_OK);
  const auto z = [](int64_t v) { return static_cast<std::size_t>(v); };
  s.points.resize(3 * z(n[0])); s.dimensions.resize(z(n[0])); s.entities.resize(z(n[0]));
  s.parameters.resize(2 * z(n[0])); s.triangles.resize(3 * z(n[1])); s.patches.resize(z(n[1]));
  s.uv.resize(6 * z(n[1])); s.uses.resize(3 * z(n[1])); s.regions.resize(2 * z(n[1]));
  s.sliver_uv.resize(6 * z(n[2])); s.sliver_patches.resize(z(n[2]));
  s.edges.resize(2 * z(n[3])); s.edge_curves.resize(z(n[3])); s.edge_parameters.resize(2 * z(n[3]));
  s.stats.resize(z(n[4])); s.curve_offsets.resize(z(n[5]) + 1);
  s.curve_parameters.resize(z(n[6])); s.sag_splits.resize(z(n[5]));
  s.curve_failures.resize(z(n[7])); s.face_failures.resize(z(n[8]));
  const phx_mc_cad_surface_arrays arrays{
      n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8],
      s.points.data(), s.dimensions.data(), s.entities.data(), s.parameters.data(),
      s.triangles.data(), s.patches.data(), s.uv.data(), s.uses.data(), s.regions.data(),
      s.sliver_uv.data(), s.sliver_patches.data(), s.edges.data(), s.edge_curves.data(),
      s.edge_parameters.data(), s.stats.data(), s.curve_offsets.data(), s.curve_parameters.data(),
      s.sag_splits.data(), s.curve_failures.data(), &s.curve_stats, s.face_failures.data()};
  PHX_CHECK(phx_mc_cad_surface_result_export(handle, &arrays) == PHX_MC_OK);
  phx_mc_cad_surface_result_free(handle);
  return true;
}

struct TetMesh {
  std::vector<double> points;
  std::vector<int32_t> tetrahedra;
};

double orientation(const double* points, const int32_t* cell) {
  const double* a = points + 3 * cell[0];
  double u[3][3];
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c) u[k][c] = points[3 * cell[k + 1] + c] - a[c];
  return u[0][0] * (u[1][1] * u[2][2] - u[1][2] * u[2][1]) -
         u[0][1] * (u[1][0] * u[2][2] - u[1][2] * u[2][0]) +
         u[0][2] * (u[1][0] * u[2][1] - u[1][1] * u[2][0]);
}

// Fixed-boundary tetrahedralization of the closed face-stage surface by the
// S3 PLC recovery (phx_mc_plc3d_recover, PHX_MC_TET_MESH_BOUNDARY_FIXED).
bool fill(const Surface& s, TetMesh& mesh, bool quality) {
  const auto count = static_cast<int64_t>(s.patches.size());
  std::vector<int64_t> offsets(static_cast<std::size_t>(count) + 1);
  std::vector<int32_t> facets(static_cast<std::size_t>(count));
  for (int64_t t = 0; t <= count; ++t) offsets[static_cast<std::size_t>(t)] = 3 * t;
  for (int64_t t = 0; t < count; ++t) facets[static_cast<std::size_t>(t)] = static_cast<int32_t>(t);
  phx_mc_plc3d_result* result = nullptr;
  phx_mc_plc3d_prepare_source_request request{};
  request.point_count = static_cast<int64_t>(s.dimensions.size());
  request.points = s.points.data();
  request.polygon_count = count;
  request.polygon_vertex_count = 3 * count;
  request.polygon_offsets = offsets.data();
  request.polygon_vertices = s.triangles.data();
  request.polygon_facets = facets.data();
  request.facet_count = count;
  request.facet_regions = s.regions.data();
  request.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  request.result = &result;
  const int32_t status = phx_mc_plc3d_recover(&request);
  if (status != PHX_MC_OK) {
    int32_t reason = 0, first_kind = 0, second_kind = 0;
    int64_t first = 0, second = 0;
    double deviation = 0, bound = 0;
    if (result)
      phx_mc_plc3d_failure(result, &reason, &first_kind, &first, &second_kind, &second, &deviation,
                           &bound);
    std::fprintf(stderr, "plc recover status=%d reason=%d first=(%d,%lld)\n", status, reason,
                 first_kind, static_cast<long long>(first));
  }
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) {
    phx_mc_plc3d_free(result);
    return false;
  }
  int64_t n[PHX_MC_PLC3D_SIZES]{};
  PHX_CHECK(phx_mc_plc3d_sizes(result, n) == PHX_MC_OK);
  const auto z = [](int64_t v) { return static_cast<std::size_t>(v); };
  mesh.points.resize(3 * z(n[0]));
  mesh.tetrahedra.resize(4 * z(n[1]));
  std::vector<int32_t> regions(z(n[1])), faces(3 * z(n[2])), face_sources(z(n[2]));
  std::vector<int8_t> face_orientations(z(n[2])), dimensions(z(n[0]));
  std::vector<int32_t> segments(2 * z(n[3])), segment_sources(z(n[3])), plc_edges(2 * z(n[4]));
  std::vector<int32_t> input_triangles(3 * z(n[5])), input_polygons(z(n[5]));
  std::vector<double> protection(z(n[0]));
  PHX_CHECK(phx_mc_plc3d_export(result, n[0], n[1], n[2], n[3], n[4], n[5], mesh.points.data(),
                                mesh.tetrahedra.data(), regions.data(), faces.data(),
                                face_sources.data(), face_orientations.data(), segments.data(),
                                segment_sources.data(), protection.data(), dimensions.data(),
                                plc_edges.data(), input_triangles.data(),
                                input_polygons.data()) == PHX_MC_OK);
  if (!quality) {
    phx_mc_plc3d_free(result);
  } else {
    // S3 persistent mesh adopted from the recovery, refined to radius-edge 2
    // and improved (fixed boundary: the surface triangles stay).
    phx_mc_tet_mesh_create_request create{};
    create.point_count = n[0];
    create.tetrahedron_count = n[1];
    create.face_count = n[2];
    create.face_sources = face_sources.data();
    create.segment_count = n[3];
    create.segment_sources = segment_sources.data();
    create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
    phx_mc_tet_mesh* handle = nullptr;
    create.result = &handle;
    const int32_t created = phx_mc_tet_mesh_create_from_plc(result, &create, nullptr, 0);
    phx_mc_plc3d_free(result);
    PHX_CHECK(created == PHX_MC_OK);
    if (created != PHX_MC_OK) return false;
    phx::mc::TetMesh& state = phx::mc::tet_mesh_state(*handle);
    int64_t inserted = 0, collapsed = 0, relocated = 0, unmet = 0;
    const int32_t refined = phx::mc::refine_mesh(state, 2.0, {}, inserted, unmet);
    PHX_CHECK(refined == PHX_MC_OK || refined == PHX_MC_UNRESOLVED);
    const phx::mc::TetQualityRequest quality_request{2.0, 10.0, 0.0};
    const int32_t improved = phx::mc::improve_mesh(state, quality_request, {}, inserted,
                                                   collapsed, relocated, unmet);
    PHX_CHECK(improved == PHX_MC_OK || improved == PHX_MC_UNRESOLVED);
    phx::mc::TetMeshSnapshot snapshot;
    PHX_CHECK(state.snapshot(snapshot) == phx::mc::TetEditStatus::kApplied);
    phx_mc_tet_mesh_free(handle);
    std::map<uint64_t, int32_t> rows;
    mesh.points.clear();
    for (const auto& vertex : snapshot.vertices) {
      rows[vertex.id] = static_cast<int32_t>(rows.size());
      mesh.points.insert(mesh.points.end(), vertex.carrier.begin(), vertex.carrier.end());
    }
    mesh.tetrahedra.clear();
    for (const auto& cell : snapshot.cells)
      for (uint64_t id : cell.vertices) mesh.tetrahedra.push_back(rows.at(id));
  }
  // The surface vertices keep their rows; volume points follow.
  PHX_CHECK(mesh.points.size() >= s.points.size());
  PHX_CHECK(std::memcmp(mesh.points.data(), s.points.data(), s.points.size() * sizeof(double)) == 0);
  for (std::size_t t = 0; t < mesh.tetrahedra.size(); t += 4) {
    if (orientation(mesh.points.data(), mesh.tetrahedra.data() + t) < 0)
      std::swap(mesh.tetrahedra[t + 2], mesh.tetrahedra[t + 3]);
  }
  return true;
}

struct Quadratic {
  int32_t status = -1;
  std::vector<int32_t> edges, edge_entities, cell_decisions, initial_decisions;
  std::vector<double> nodes;
  std::vector<int8_t> edge_kinds;
  std::vector<int64_t> cell_edges, records;
  phx_mc_cad_quadratic_stats stats{};
};

Quadratic curve(const phx_mc_cad_model& model, const Surface& s, const TetMesh& mesh) {
  Quadratic q;
  const int64_t cells = static_cast<int64_t>(mesh.tetrahedra.size() / 4);
  phx_mc_cad_quadratic_request r{};
  r.model = &model;
  r.point_count = static_cast<int64_t>(mesh.points.size() / 3);
  r.points = mesh.points.data();
  r.surface_point_count = static_cast<int64_t>(s.dimensions.size());
  r.vertex_dimensions = s.dimensions.data();
  r.vertex_entities = s.entities.data();
  r.tetrahedron_count = cells;
  r.tetrahedra = mesh.tetrahedra.data();
  r.triangle_count = static_cast<int64_t>(s.patches.size());
  r.triangles = s.triangles.data();
  r.triangle_patches = s.patches.data();
  r.triangle_uv = s.uv.data();
  r.curve_edge_count = static_cast<int64_t>(s.edge_curves.size());
  r.curve_edges = s.edges.data();
  r.curve_edge_curves = s.edge_curves.data();
  r.curve_edge_parameters = s.edge_parameters.data();
  int64_t edge_count = -1, record_count = -1;
  r.edge_count = &edge_count;
  r.record_count = &record_count;
  q.cell_edges.assign(static_cast<std::size_t>(6 * cells), -7);
  q.cell_decisions.assign(static_cast<std::size_t>(cells), -7);
  r.cell_edges = q.cell_edges.data();
  r.cell_decisions = q.cell_decisions.data();
  // Counts first: zero capacity reports them without writing anything else.
  PHX_CHECK(phx_mc_cad_quadratic_nodes(&r) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(edge_count > 0 && record_count >= 0);
  PHX_CHECK(cells == 0 || q.cell_decisions[0] == -7);
  const auto e = static_cast<std::size_t>(edge_count);
  q.edges.resize(2 * e); q.nodes.resize(3 * e); q.edge_kinds.resize(e); q.edge_entities.resize(e);
  q.records.resize(2 * static_cast<std::size_t>(record_count));
  r.edge_capacity = edge_count;
  r.record_capacity = record_count;
  r.edges = q.edges.data();
  r.nodes = q.nodes.data();
  r.edge_kinds = q.edge_kinds.data();
  r.edge_entities = q.edge_entities.data();
  r.records = q.records.data();
  r.stats = &q.stats;
  q.status = phx_mc_cad_quadratic_nodes(&r);
  // The per-cell decisions before repair come from the internal entry point
  // (the C API reports only their counts).
  const phx::mc::GradualNearestScope mode;
  phx::mc::cad::CadGeometry geometry;
  PHX_CHECK(mode.established() && geometry.bind(model) == PHX_MC_OK);
  const phx::mc::CadQuadraticInput input{
      r.point_count, r.points, r.surface_point_count, r.vertex_dimensions, r.vertex_entities,
      r.tetrahedron_count, r.tetrahedra, r.triangle_count, r.triangles, r.triangle_patches,
      r.triangle_uv, r.curve_edge_count, r.curve_edges, r.curve_edge_curves,
      r.curve_edge_parameters, r.relative_floor};
  phx::mc::CadQuadraticMesh internal;
  PHX_CHECK(phx::mc::cad_quadratic_nodes(geometry, input, internal) == q.status);
  q.initial_decisions = std::move(internal.initial_decisions);
  return q;
}

std::array<double, 3> midpoint(const std::vector<double>& points, int32_t a, int32_t b) {
  return {0.5 * (points[3 * a] + points[3 * b]), 0.5 * (points[3 * a + 1] + points[3 * b + 1]),
          0.5 * (points[3 * a + 2] + points[3 * b + 2])};
}

double length(const std::vector<double>& points, int32_t a, int32_t b) {
  return std::hypot(std::hypot(points[3 * a] - points[3 * b], points[3 * a + 1] - points[3 * b + 1]),
                    points[3 * a + 2] - points[3 * b + 2]);
}

// Shared structure checks: sorted unique edges, cell edge table, interior
// nodes at the straight midpoint, curved nodes near their chord (a wrong UV
// lift or curve unwrap lands across the body), every cell certified VALID.
template <class OnSurface>
void check_quadratic(const Surface& s, const TetMesh& mesh, const Quadratic& q,
                     OnSurface on_surface, const char* name, bool quality) {
  PHX_CHECK(q.status == PHX_MC_OK);
  if (q.status != PHX_MC_OK) return;
  const auto edges = q.edge_kinds.size();
  for (std::size_t e = 0; e < edges; ++e) {
    PHX_CHECK(q.edges[2 * e] < q.edges[2 * e + 1]);
    if (e > 0)
      PHX_CHECK(std::make_pair(q.edges[2 * e - 2], q.edges[2 * e - 1]) <
                std::make_pair(q.edges[2 * e], q.edges[2 * e + 1]));
  }
  constexpr int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  for (std::size_t t = 0; t < q.cell_decisions.size(); ++t) {
    for (int k = 0; k < 6; ++k) {
      const auto e = static_cast<std::size_t>(q.cell_edges[6 * t + k]);
      const int32_t a = mesh.tetrahedra[4 * t + pairs[k][0]], b = mesh.tetrahedra[4 * t + pairs[k][1]];
      PHX_CHECK(q.edges[2 * e] == std::min(a, b) && q.edges[2 * e + 1] == std::max(a, b));
    }
  }
  // Independent edge classes from the surface arrays.
  std::map<std::pair<int32_t, int32_t>, int> expected;
  for (std::size_t t = 0; t < s.patches.size(); ++t)
    for (int k = 0; k < 3; ++k) {
      const int32_t a = s.triangles[3 * t + k], b = s.triangles[3 * t + (k + 1) % 3];
      expected[std::minmax(a, b)] = PHX_MC_CAD_EDGE_SURFACE;
    }
  for (std::size_t i = 0; i < s.edge_curves.size(); ++i)
    expected[std::minmax(s.edges[2 * i], s.edges[2 * i + 1])] = PHX_MC_CAD_EDGE_CURVE;
  int64_t counts[3]{};
  double worst = 0.0;
  // Straightened records: boundary edges back at the chord midpoint, each
  // recorded once, naming a cell that contains the edge.
  std::vector<char> straight(edges, 0);
  for (std::size_t i = 0; i < q.records.size(); i += 2) {
    const auto e = static_cast<std::size_t>(q.records[i]);
    const auto cell = static_cast<std::size_t>(q.records[i + 1]);
    PHX_CHECK(q.edge_kinds[e] != PHX_MC_CAD_EDGE_INTERIOR && !straight[e]);
    straight[e] = 1;
    bool contained = false;
    for (int k = 0; k < 6; ++k) contained = contained || q.cell_edges[6 * cell + k] == q.records[i];
    PHX_CHECK(contained);
  }
  PHX_CHECK(static_cast<int64_t>(q.records.size() / 2) == q.stats.straightened_edges);
  for (std::size_t e = 0; e < edges; ++e) {
    const int32_t a = q.edges[2 * e], b = q.edges[2 * e + 1];
    const auto it = expected.find({a, b});
    const int kind = it == expected.end() ? PHX_MC_CAD_EDGE_INTERIOR : it->second;
    PHX_CHECK(q.edge_kinds[e] == kind);
    ++counts[kind];
    const double* node = q.nodes.data() + 3 * e;
    const auto middle = midpoint(mesh.points, a, b);
    if (kind == PHX_MC_CAD_EDGE_INTERIOR) {
      PHX_CHECK(q.edge_entities[e] == -1);
      if (q.stats.harmonic_nodes == 0)
        PHX_CHECK(node[0] == middle[0] && node[1] == middle[1] && node[2] == middle[2]);
      continue;
    }
    if (straight[e]) {
      PHX_CHECK(node[0] == middle[0] && node[1] == middle[1] && node[2] == middle[2]);
      continue;
    }
    worst = std::max(worst, on_surface(node, static_cast<int>(kind), q.edge_entities[e]));
    const double offset = std::hypot(std::hypot(node[0] - middle[0], node[1] - middle[1]),
                                     node[2] - middle[2]);
    PHX_CHECK(offset <= 0.25 * length(mesh.points, a, b));
  }
  PHX_CHECK(worst <= 1e-14);
  PHX_CHECK(q.stats.interior_edges == counts[0] && q.stats.surface_edges == counts[1] &&
            q.stats.curve_edges == counts[2]);
  PHX_CHECK(counts[1] > 0 && counts[2] > 0);
  // On the refined mesh every cell is certified. On the raw recovery a cell
  // may stay undecided only when all its nodes are back on their chords: the
  // linear cell itself (a flat boundary sliver) is the obstruction.
  int64_t linear_failures = 0;
  for (std::size_t t = 0; t < q.cell_decisions.size(); ++t) {
    if (q.cell_decisions[t] == PHX_MC_QUADRATIC_VALIDITY_VALID) continue;
    ++linear_failures;
    PHX_CHECK(!quality);
    for (int k = 0; k < 6; ++k) {
      const auto e = static_cast<std::size_t>(q.cell_edges[6 * t + k]);
      const auto middle = midpoint(mesh.points, q.edges[2 * e], q.edges[2 * e + 1]);
      PHX_CHECK(q.nodes[3 * e] == middle[0] && q.nodes[3 * e + 1] == middle[1] &&
                q.nodes[3 * e + 2] == middle[2]);
    }
  }
  PHX_CHECK(q.stats.final_invalid + q.stats.final_unresolved == linear_failures);
  // The first lift can fail before harmonic repair, including in cells
  // incident on interior vertices. A quality fill must finish certified.
  int64_t initial_failures = 0;
  PHX_CHECK(q.initial_decisions.size() == q.cell_decisions.size());
  for (std::size_t t = 0; t < q.initial_decisions.size(); ++t) {
    if (q.initial_decisions[t] == PHX_MC_QUADRATIC_VALIDITY_VALID) continue;
    ++initial_failures;
  }
  PHX_CHECK(initial_failures == q.stats.initial_invalid + q.stats.initial_unresolved);
  if (quality) PHX_CHECK(q.stats.final_invalid + q.stats.final_unresolved == 0);
  std::printf("%s%s: points=%zu cells=%zu edges=%zu (curve %lld surface %lld interior %lld) "
              "initial invalid=%lld unresolved=%lld harmonic nodes=%lld repaired=%lld "
              "straightened=%lld "
              "max closed-form residual=%.3g linear failures=%lld\n",
              name, quality ? "" : " (raw recovery)", mesh.points.size() / 3,
              q.cell_decisions.size(), edges,
              static_cast<long long>(counts[2]), static_cast<long long>(counts[1]),
              static_cast<long long>(counts[0]), static_cast<long long>(q.stats.initial_invalid),
              static_cast<long long>(q.stats.initial_unresolved),
              static_cast<long long>(q.stats.harmonic_nodes),
              static_cast<long long>(q.stats.harmonic_repaired_cells),
              static_cast<long long>(q.stats.straightened_edges), worst,
              static_cast<long long>(linear_failures));
}

void check_deterministic(const Quadratic& a, const Quadratic& b) {
  PHX_CHECK(a.edges == b.edges && a.edge_kinds == b.edge_kinds &&
            a.edge_entities == b.edge_entities && a.cell_edges == b.cell_edges &&
            a.cell_decisions == b.cell_decisions && a.records == b.records);
  PHX_CHECK(a.nodes.size() == b.nodes.size() &&
            std::memcmp(a.nodes.data(), b.nodes.data(), a.nodes.size() * sizeof(double)) == 0);
}

// P2 Jacobian weights in the node order of phx_mc_quadratic_tetrahedron_validity
// (corners, then edges 01 02 03 12 13 23), written from the P2 shape functions.
std::array<int32_t, 120> p2_derivative_weights() {
  constexpr int corners[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  const int linear[4][4] = {{1, -1, -1, -1}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  std::array<int32_t, 120> values{};
  for (int c = 0; c < 3; ++c) {
    const auto gradient = [c](int i) { return i == 0 ? -1 : (i == c + 1 ? 1 : 0); };
    for (int p = 0; p < 4; ++p) {
      for (int i = 0; i < 4; ++i)
        values[(c * 4 + p) * 10 + i] = gradient(i) * (4 * linear[i][p] - (p == 0 ? 1 : 0));
      for (int e = 0; e < 6; ++e) {
        const int i = corners[e][0], j = corners[e][1];
        values[(c * 4 + p) * 10 + 4 + e] =
            4 * (gradient(i) * linear[j][p] + gradient(j) * linear[i][p]);
      }
    }
  }
  return values;
}

// The binary64 screen on the final P2 cells: verdicts without bounds (screen
// enabled) equal verdicts with bounds (exact path only), and every screened
// cell is exactly VALID. Catches an error constant too small for the
// operation tree (a screened cell whose exact control is not positive).
void check_screen(const TetMesh& mesh, const Quadratic& q, const char* name, bool quality) {
  const auto weights = p2_derivative_weights();
  const std::size_t cells = mesh.tetrahedra.size() / 4;
  std::vector<double> nodes(30 * cells);
  for (std::size_t k = 0; k < cells; ++k) {
    double* row = nodes.data() + 30 * k;
    for (int i = 0; i < 4; ++i)
      for (int c = 0; c < 3; ++c)
        row[3 * i + c] = mesh.points[3 * static_cast<std::size_t>(mesh.tetrahedra[4 * k + i]) + c];
    for (int e = 0; e < 6; ++e)
      for (int c = 0; c < 3; ++c)
        row[12 + 3 * e + c] = q.nodes[3 * static_cast<std::size_t>(q.cell_edges[6 * k + e]) + c];
  }
  const auto count = static_cast<int64_t>(cells);
  const phx::mc::QuadraticDerivativeStencil stencil(weights.data());
  for (double floor : {0.0, 1e-3, 0.05}) {
    std::vector<int32_t> screened_verdicts(cells, -1), exact_verdicts(cells, -1);
    std::vector<double> lower(cells), upper(cells);
    PHX_CHECK(phx_mc_quadratic_tetrahedron_validity(count, nodes.data(), weights.data(), floor,
                                                    screened_verdicts.data(), nullptr,
                                                    nullptr) == PHX_MC_OK);
    PHX_CHECK(phx_mc_quadratic_tetrahedron_validity(count, nodes.data(), weights.data(), floor,
                                                    exact_verdicts.data(), lower.data(),
                                                    upper.data()) == PHX_MC_OK);
    PHX_CHECK(screened_verdicts == exact_verdicts);
    int64_t screened = 0, valid = 0;
    for (std::size_t k = 0; k < cells; ++k) {
      valid += exact_verdicts[k] == PHX_MC_QUADRATIC_VALIDITY_VALID;
      if (!phx::mc::quadratic_validity_screen(nodes.data() + 30 * k, stencil, floor))
        continue;
      ++screened;
      PHX_CHECK(exact_verdicts[k] == PHX_MC_QUADRATIC_VALIDITY_VALID);
    }
    std::printf("%s%s screen: floor=%g cells=%zu exactly_valid=%lld screened=%lld exact_path=%lld\n",
                name, quality ? "" : " (raw fill)", floor, cells, static_cast<long long>(valid),
                static_cast<long long>(screened), static_cast<long long>(count - screened));
  }
}

void sphere(bool quality) {
  const Bank bank = sphere_bank();
  const phx_mc_cad_model model = bank.model();
  Surface s;
  TetMesh mesh;
  if (!surface_mesh(model, 0.35, s) || !fill(s, mesh, quality)) return;
  const Quadratic q = curve(model, s, mesh);
  check_screen(mesh, q, "sphere", quality);
  check_quadratic(s, mesh, q, [](const double* x, int, int32_t) {
    return std::abs(std::hypot(std::hypot(x[0], x[1]), x[2]) - 1.0);
  }, "sphere", quality);
  check_deterministic(q, curve(model, s, mesh));
  // Seam curve nodes lie on the meridian plane y = 0, x >= 0; surface edges
  // at a pole lie in the plane of the pole and the other end (a wrong pole u
  // leaves that great circle).
  // Straightened edges sit on their chords (checked above).
  std::vector<char> straight(q.edge_kinds.size(), 0);
  for (std::size_t i = 0; i < q.records.size(); i += 2)
    straight[static_cast<std::size_t>(q.records[i])] = 1;
  int64_t pole_edges = 0;
  for (std::size_t e = 0; e < q.edge_kinds.size(); ++e) {
    const double* x = q.nodes.data() + 3 * e;
    if (straight[e]) continue;
    if (q.edge_kinds[e] == PHX_MC_CAD_EDGE_CURVE) PHX_CHECK(std::abs(x[1]) <= 1e-15 && x[0] >= -1e-15);
    if (q.edge_kinds[e] != PHX_MC_CAD_EDGE_SURFACE) continue;
    for (int end = 0; end < 2; ++end) {
      const int32_t v = q.edges[2 * e + end], w = q.edges[2 * e + 1 - end];
      if (s.dimensions[static_cast<std::size_t>(v)] != 0) continue;
      ++pole_edges;
      const double* p = s.points.data() + 3 * v;
      const double* o = s.points.data() + 3 * w;
      const double triple = p[0] * (o[1] * x[2] - o[2] * x[1]) - p[1] * (o[0] * x[2] - o[2] * x[0]) +
                            p[2] * (o[0] * x[1] - o[1] * x[0]);
      PHX_CHECK(std::abs(triple) <= 1e-14);
    }
  }
  PHX_CHECK(pole_edges > 0);
}

void stretched_sphere_places_nodes_in_the_source_chart() {
  Bank bank = sphere_bank();
  Surface surface;
  TetMesh mesh;
  if (!surface_mesh(bank.model(), 0.35, surface) || !fill(surface, mesh, true)) return;
  constexpr double scale[]{1.5, 0.75, 1.25};
  constexpr double shift[]{0.25, -0.5, 0.125};
  const auto place = [&](int32_t child) {
    const auto node = static_cast<int32_t>(bank.kinds.size());
    bank.kinds.push_back(PHX_MC_CAD_NODE_AFFINE);
    bank.integers.insert(bank.integers.end(), {child, 3});
    bank.integer_offsets.push_back(static_cast<int64_t>(bank.integers.size()));
    bank.reals.insert(bank.reals.end(), {scale[0], 0, 0, 0, scale[1], 0, 0, 0, scale[2],
                                         shift[0], shift[1], shift[2]});
    bank.real_offsets.push_back(static_cast<int64_t>(bank.reals.size()));
    return node;
  };
  bank.surface_nodes[0] = place(bank.surface_nodes[0]);
  bank.curve_nodes[0] = place(bank.curve_nodes[0]);
  for (auto* bank_points : {&bank.corners, &mesh.points})
    for (std::size_t i = 0; i < bank_points->size(); ++i)
      (*bank_points)[i] = scale[i % 3] * (*bank_points)[i] + shift[i % 3];

  const Quadratic curved = curve(bank.model(), surface, mesh);
  PHX_CHECK(curved.status == PHX_MC_OK);
  if (curved.status != PHX_MC_OK) return;
  PHX_CHECK(curved.stats.final_invalid == 0 && curved.stats.final_unresolved == 0);
  std::vector<bool> straight(curved.edge_kinds.size());
  for (std::size_t i = 0; i < curved.records.size(); i += 2)
    straight[static_cast<std::size_t>(curved.records[i])] = true;
  std::size_t measured = 0;
  for (std::size_t e = 0; e < curved.edge_kinds.size(); ++e) {
    if (straight[e] || curved.edge_kinds[e] != PHX_MC_CAD_EDGE_SURFACE) continue;
    auto direction = midpoint(mesh.points, curved.edges[2 * e], curved.edges[2 * e + 1]);
    for (int axis = 0; axis < 3; ++axis)
      direction[axis] = (direction[axis] - shift[axis]) / scale[axis];
    const double norm = std::hypot(direction[0], direction[1], direction[2]);
    for (int axis = 0; axis < 3; ++axis)
      PHX_CHECK_NEAR(curved.nodes[3 * e + axis],
                     shift[axis] + scale[axis] * direction[axis] / norm, 2e-14);
    ++measured;
  }
  PHX_CHECK(measured > 0);
}

void torus(bool quality) {
  const Bank bank = torus_bank();
  const phx_mc_cad_model model = bank.model();
  Surface s;
  TetMesh mesh;
  if (!surface_mesh(model, 0.3, s) || !fill(s, mesh, quality)) return;
  const Quadratic q = curve(model, s, mesh);
  check_screen(mesh, q, "torus", quality);
  check_quadratic(s, mesh, q, [](const double* x, int, int32_t) {
    const double radial = std::hypot(x[0], x[1]);
    return std::abs(std::hypot(radial - 2.0, x[2]) - 0.5);
  }, "torus", quality);
  check_deterministic(q, curve(model, s, mesh));
  // The inner side (radial < 2) has negative Gaussian curvature: its curved
  // nodes bend into the solid along the toroidal direction; those cells are
  // certified above.
  int64_t inner = 0;
  for (std::size_t e = 0; e < q.edge_kinds.size(); ++e)
    if (q.edge_kinds[e] == PHX_MC_CAD_EDGE_SURFACE &&
        std::hypot(q.nodes[3 * e], q.nodes[3 * e + 1]) < 1.75)
      ++inner;
  PHX_CHECK(inner > 0);
}

void quadric(bool quality) {
  const phx_mc_cad_model model = quadric_model();
  Surface s;
  TetMesh mesh;
  if (!surface_mesh(model, 0.12, s) || !fill(s, mesh, quality)) return;
  const Quadratic q = curve(model, s, mesh);
  check_screen(mesh, q, "quadric", quality);
  check_quadratic(s, mesh, q, [](const double* x, int, int32_t) {
    const double first = std::abs(x[0] * x[0] + x[1] * x[1] - 1.0);
    const double second = std::abs(x[1] * x[1] + x[2] * x[2] - 0.36);
    return std::min(first, second);
  }, "quadric", quality);
  check_deterministic(q, curve(model, s, mesh));
}

// A bore: the unit cylinder S(u, v) = (cos u, sin u, v) with the solid
// outside, so curved nodes bend into the solid. One tet on the surface
// triangle A = S(-1/2, 0), B = S(1/2, 0), C = S(0, 1) whose apex D sits
// `height` outside the triangle's centroid. AB is a segment of the closed
// circle v = 0 given across its parameter wrap (2 pi - 1/2, 1/2).
Quadratic bore(double height, std::vector<double>& points, std::vector<int32_t>& cells,
               std::vector<int32_t> triangle = {0, 1, 2}) {
  Bank bank;
  std::vector<double> frame = kFrame;
  frame.push_back(1.0);
  bank.surface_nodes = {bank.add("cylinder", frame)};
  bank.curve_nodes = {bank.add("circle3", {0, 0, 0, 1, 0, 0, 0, 1, 0, 1})};
  bank.corners = {1, 0, 0};
  bank.curve_corners = {0, 0};
  bank.ranges = {0, kTurn};
  bank.orientation = {-1};
  bank.periods = {kTurn, 0};
  bank.domains = {0, kTurn, 0, 1};
  const phx_mc_cad_model model = bank.model();
  const double c = std::cos(0.5), s = std::sin(0.5);
  points = {c, -s, 0, c, s, 0, 1, 0, 1, (2 * c + 1) / 3 + height, 0, 1.0 / 3};
  cells = {0, 1, 2, 3};
  if (orientation(points.data(), cells.data()) < 0) std::swap(cells[2], cells[3]);
  const std::vector<int8_t> dimensions{1, 1, 2};
  const std::vector<int32_t> entities{0, 0, 0};
  const std::vector<int32_t> patch{0}, curve_edge{0, 1}, curve_row{0};
  const std::vector<double> uv{-0.5, 0, 0.5, 0, 0, 1}, parameters{kTurn - 0.5, 0.5};
  Quadratic q;
  phx_mc_cad_quadratic_request r{};
  r.model = &model;
  r.point_count = 4;
  r.points = points.data();
  r.surface_point_count = 3;
  r.vertex_dimensions = dimensions.data();
  r.vertex_entities = entities.data();
  r.tetrahedron_count = 1;
  r.tetrahedra = cells.data();
  r.triangle_count = 1;
  r.triangles = triangle.data();
  r.triangle_patches = patch.data();
  r.triangle_uv = uv.data();
  r.curve_edge_count = 1;
  r.curve_edges = curve_edge.data();
  r.curve_edge_curves = curve_row.data();
  r.curve_edge_parameters = parameters.data();
  q.edges.resize(12); q.nodes.resize(18); q.edge_kinds.resize(6); q.edge_entities.resize(6);
  q.cell_edges.resize(6); q.cell_decisions.resize(1); q.records.resize(12);
  int64_t edge_count = -1, record_count = -1;
  r.edge_capacity = 6;
  r.record_capacity = 6;
  r.edges = q.edges.data(); r.nodes = q.nodes.data(); r.edge_kinds = q.edge_kinds.data();
  r.edge_entities = q.edge_entities.data(); r.cell_edges = q.cell_edges.data();
  r.cell_decisions = q.cell_decisions.data(); r.records = q.records.data();
  r.edge_count = &edge_count; r.record_count = &record_count; r.stats = &q.stats;
  q.status = phx_mc_cad_quadratic_nodes(&r);
  PHX_CHECK(q.status != PHX_MC_OK || edge_count == 6);
  q.records.resize(2 * static_cast<std::size_t>(std::max<int64_t>(record_count, 0)));
  return q;
}

// det J of the P2 map at reference point (x, y, z) by central differences of
// the textbook shape functions (corners l(2l - 1), edges 4 l_i l_j).
double p2_determinant(const std::array<std::array<double, 3>, 10>& nodes, double x, double y,
                      double z) {
  constexpr int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  const auto map = [&](double a, double b, double c) {
    const double l[4] = {1 - a - b - c, a, b, c};
    std::array<double, 3> out{};
    for (int i = 0; i < 4; ++i)
      for (int k = 0; k < 3; ++k) out[k] += l[i] * (2 * l[i] - 1) * nodes[i][k];
    for (int e = 0; e < 6; ++e)
      for (int k = 0; k < 3; ++k) out[k] += 4 * l[pairs[e][0]] * l[pairs[e][1]] * nodes[4 + e][k];
    return out;
  };
  constexpr double step = 1e-6;
  double j[3][3];
  for (int c = 0; c < 3; ++c) {
    const double d[3] = {c == 0 ? step : 0, c == 1 ? step : 0, c == 2 ? step : 0};
    const auto plus = map(x + d[0], y + d[1], z + d[2]), minus = map(x - d[0], y - d[1], z - d[2]);
    for (int k = 0; k < 3; ++k) j[k][c] = (plus[k] - minus[k]) / (2 * step);
  }
  return j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) -
         j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) +
         j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
}

// Smallest sampled det J over the reference tetrahedron (lattice step 1/12).
double sampled_minimum(const std::array<std::array<double, 3>, 10>& nodes) {
  double minimum = std::numeric_limits<double>::infinity();
  for (int a = 0; a <= 12; ++a)
    for (int b = 0; a + b <= 12; ++b)
      for (int c = 0; a + b + c <= 12; ++c)
        minimum = std::min(minimum, p2_determinant(nodes, a / 12.0, b / 12.0, c / 12.0));
  return minimum;
}

std::array<std::array<double, 3>, 10> cell_nodes(const std::vector<double>& points,
                                                 const std::vector<int32_t>& cells,
                                                 const Quadratic& q) {
  std::array<std::array<double, 3>, 10> nodes{};
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 3; ++k) nodes[i][k] = points[3 * cells[i] + k];
  for (int e = 0; e < 6; ++e)
    for (int k = 0; k < 3; ++k)
      nodes[4 + e][k] = q.nodes[3 * static_cast<std::size_t>(q.cell_edges[e]) + k];
  return nodes;
}

int64_t edge_of(const Quadratic& q, int32_t a, int32_t b) {
  for (std::size_t e = 0; e < q.edge_kinds.size(); ++e)
    if (q.edges[2 * e] == std::min(a, b) && q.edges[2 * e + 1] == std::max(a, b))
      return static_cast<int64_t>(e);
  return -1;
}

void bore_repairs() {
  const std::array<double, 3> ab{1, 0, 0}, bc{std::cos(0.25), std::sin(0.25), 0.5},
      ca{std::cos(0.25), -std::sin(0.25), 0.5};
  std::vector<double> points;
  std::vector<int32_t> cells;

  // Far apex: no repair; the wrapped curve segment's node is C(0) = (1, 0, 0)
  // (without the unwrap it would be C(pi) = (-1, 0, 0)); the surface nodes
  // are S(+-1/4, 1/2) from the triangle's UV record.
  const Quadratic far = bore(0.5, points, cells);
  PHX_CHECK(far.status == PHX_MC_OK && far.cell_decisions[0] == PHX_MC_QUADRATIC_VALIDITY_VALID);
  PHX_CHECK(far.stats.initial_invalid == 0 && far.records.empty());
  const auto near = [](const double* x, const std::array<double, 3>& y) {
    return std::abs(x[0] - y[0]) <= 1e-15 && std::abs(x[1] - y[1]) <= 1e-15 &&
           std::abs(x[2] - y[2]) <= 1e-15;
  };
  const int64_t e_ab = edge_of(far, 0, 1), e_bc = edge_of(far, 1, 2), e_ca = edge_of(far, 0, 2);
  PHX_CHECK(e_ab >= 0 && e_bc >= 0 && e_ca >= 0);
  if (e_ab < 0 || e_bc < 0 || e_ca < 0) return;
  PHX_CHECK(far.edge_kinds[e_ab] == PHX_MC_CAD_EDGE_CURVE && near(&far.nodes[3 * e_ab], ab));
  PHX_CHECK(far.edge_kinds[e_bc] == PHX_MC_CAD_EDGE_SURFACE && near(&far.nodes[3 * e_bc], bc));
  PHX_CHECK(far.edge_kinds[e_ca] == PHX_MC_CAD_EDGE_SURFACE && near(&far.nodes[3 * e_ca], ca));
  PHX_CHECK(sampled_minimum(cell_nodes(points, cells, far)) > 0);

  // Apex 0.05 outside: the curved face passes the apex. The unrepaired map
  // (independent reconstruction) inverts; harmonic extension cannot move a
  // vertex, so the three curved edges are straightened and recorded.
  const Quadratic close = bore(0.05, points, cells);
  PHX_CHECK(close.status == PHX_MC_OK);
  auto curved = cell_nodes(points, cells, close);
  for (int e = 0; e < 6; ++e) {
    const int32_t a = cells[e < 3 ? 0 : e < 5 ? 1 : 2];
    const int32_t b = cells[e == 0 ? 1 : e == 1 || e == 3 ? 2 : 3];
    const auto middle = midpoint(points, a, b);
    const int32_t low = std::min(a, b), high = std::max(a, b);
    curved[4 + e] = low == 0 && high == 1 ? ab : low == 1 && high == 2 ? bc
                  : low == 0 && high == 2 ? ca : middle;
  }
  PHX_CHECK(sampled_minimum(curved) < 0);
  PHX_CHECK(close.stats.initial_invalid == 1 && close.stats.harmonic_repaired_cells == 0);
  PHX_CHECK(close.stats.straightened_edges == 3 && close.records.size() == 6);
  for (std::size_t i = 0; i < close.records.size(); i += 2) PHX_CHECK(close.records[i + 1] == 0);
  std::vector<int64_t> recorded;
  for (std::size_t i = 0; i < close.records.size(); i += 2) recorded.push_back(close.records[i]);
  std::sort(recorded.begin(), recorded.end());
  std::vector<int64_t> boundary{e_ab, e_bc, e_ca};
  std::sort(boundary.begin(), boundary.end());
  PHX_CHECK(recorded == boundary);
  for (std::size_t e = 0; e < 6; ++e) {
    const auto middle = midpoint(points, close.edges[2 * e], close.edges[2 * e + 1]);
    PHX_CHECK(close.nodes[3 * e] == middle[0] && close.nodes[3 * e + 1] == middle[1] &&
              close.nodes[3 * e + 2] == middle[2]);
  }
  PHX_CHECK(close.cell_decisions[0] == PHX_MC_QUADRATIC_VALIDITY_VALID);
  PHX_CHECK(close.stats.final_invalid == 0 && close.stats.final_unresolved == 0);

  // Apex 0.2 outside: one Jacobi pass moves each apex edge node by the mean
  // displacement of the cell's other five edges, which repairs the cell; the
  // curved boundary keeps its nodes.
  const Quadratic mid = bore(0.2, points, cells);
  PHX_CHECK(mid.status == PHX_MC_OK);
  PHX_CHECK(mid.stats.initial_invalid + mid.stats.initial_unresolved == 1);
  PHX_CHECK(mid.stats.harmonic_nodes == 3 && mid.stats.harmonic_repaired_cells == 1);
  PHX_CHECK(mid.records.empty() && mid.cell_decisions[0] == PHX_MC_QUADRATIC_VALIDITY_VALID);
  PHX_CHECK(near(&mid.nodes[3 * e_ab], ab) && near(&mid.nodes[3 * e_bc], bc) &&
            near(&mid.nodes[3 * e_ca], ca));
  std::array<double, 3> boundary_shift{};
  for (int64_t e : {e_ab, e_bc, e_ca}) {
    const auto middle = midpoint(points, mid.edges[2 * e], mid.edges[2 * e + 1]);
    for (int k = 0; k < 3; ++k) boundary_shift[k] += mid.nodes[3 * e + k] - middle[k];
  }
  for (int32_t corner = 0; corner < 3; ++corner) {
    const int64_t e = edge_of(mid, corner, 3);
    PHX_CHECK(e >= 0 && mid.edge_kinds[e] == PHX_MC_CAD_EDGE_INTERIOR);
    if (e < 0) continue;
    const auto middle = midpoint(points, corner, 3);
    for (int k = 0; k < 3; ++k)
      PHX_CHECK_NEAR(mid.nodes[3 * e + k], middle[k] + boundary_shift[k] / 5, 1e-15);
  }
  PHX_CHECK(sampled_minimum(cell_nodes(points, cells, mid)) > 0);

  // A surface triangle naming a volume point is refused before any output.
  const Quadratic refused = bore(0.5, points, cells, {0, 1, 3});
  PHX_CHECK(refused.status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_cad_quadratic_nodes(nullptr) == PHX_MC_INVALID_ARGUMENT);
}

void cubic_shape_is_independent_of_global_size_scale() {
  constexpr double points[]{0, 0, 0, 1, 0, 0, 0.32, 0.95, 0, 0, 0, 1};
  constexpr int32_t tetrahedron[]{0, 1, 2, 3};
  constexpr int edges[][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  phx::mc::CadQuadraticInput input{};
  input.point_count = 4;
  input.points = points;
  input.tetrahedron_count = 1;
  input.tetrahedra = tetrahedron;
  input.length_scale = std::ldexp(0.85, -341);
  input.sampled_minimum_scaled_jacobian = 1.4;
  phx::mc::CadCubicMesh mesh;
  mesh.cell_nodes.resize(1);
  const auto add = [&](const std::array<int, 3>& corners) {
    mesh.cell_nodes[0][mesh.nodes.size()] = static_cast<int64_t>(mesh.nodes.size());
    auto& node = mesh.nodes.emplace_back();
    for (int axis = 0; axis < 3; ++axis)
      node[axis] = (points[3 * corners[0] + axis] + points[3 * corners[1] + axis] +
                    points[3 * corners[2] + axis]) / 3.0;
  };
  for (const auto& edge : edges) {
    add({edge[0], edge[0], edge[1]});
    add({edge[0], edge[1], edge[1]});
  }
  for (int omitted = 0; omitted < 4; ++omitted)
    add({(omitted + 1) % 4, (omitted + 2) % 4, (omitted + 3) % 4});
  const auto shape = phx::mc::sample_cad_cubic_shape(input, mesh);
  const double independent = std::sqrt(2.0) * 0.95 / std::hypot(0.32, 0.95);
  PHX_CHECK(shape.unmet[0] == 1);
  PHX_CHECK(std::abs(shape.minimum_scaled - independent) < 1e-13);
}

}  // namespace

int main() {
  for (bool quality : {true, false}) {
    sphere(quality);
    torus(quality);
    quadric(quality);
  }
  bore_repairs();
  stretched_sphere_places_nodes_in_the_source_chart();
  cubic_shape_is_independent_of_global_size_scale();
  return phx::mc::test::finish("test_cad_curving");
}
