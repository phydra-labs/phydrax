//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>
#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"
#include "facet_coverage.hpp"
#include "predicates.hpp"
#include "tet_mesh.hpp"

namespace {
struct TetGeometrySnapshot {
  std::vector<double> points;
  std::vector<uint64_t> cell_ids;
  std::vector<uint64_t> cell_vertices;
};

void capture_vertex_edge_witnesses(const phx_mc_tet_mesh* mesh,
                                  std::vector<uint64_t>& vertex_ids,
                                  std::vector<int8_t>& witness_present,
                                  std::vector<uint64_t>& witness_endpoints,
                                  std::vector<double>& witness_parameters,
                                  uint64_t& generation,
                                  TetGeometrySnapshot* geometry = nullptr) {
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  const int32_t sizes_status = phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data());
  PHX_CHECK(sizes_status == PHX_MC_OK);
  if (sizes_status != PHX_MC_OK) return;
  const std::size_t vertices = static_cast<std::size_t>(sizes[0]);
  const std::size_t cells = static_cast<std::size_t>(sizes[1]);
  const std::size_t facets = static_cast<std::size_t>(sizes[2]);
  const std::size_t segments = static_cast<std::size_t>(sizes[3]);
  const std::size_t source_points_count = static_cast<std::size_t>(sizes[4]);
  const std::size_t source_faces_count = static_cast<std::size_t>(sizes[5]);
  const std::size_t source_segments_count = static_cast<std::size_t>(sizes[6]);
  const std::size_t vertex_ancestors = static_cast<std::size_t>(sizes[17]);
  const std::size_t cell_ancestors = static_cast<std::size_t>(sizes[18]);
  vertex_ids.resize(vertices);
  witness_present.resize(vertices);
  witness_endpoints.resize(2 * vertices);
  witness_parameters.resize(vertices);
  std::vector<double> points(3 * vertices);
  std::vector<double> snapshot_vertex_sizes(vertices);
  std::vector<double> snapshot_cell_sizes(cells);
  std::vector<int8_t> witness_strata(vertices), vertex_dimensions(vertices),
      protected_vertices(vertices), facet_orientations(facets),
      protected_facets(facets), protected_segments(segments);
  std::vector<int32_t> witness_entities(vertices), cell_regions(cells),
      facet_sources(facets), segment_sources(segments), source_faces(3 * source_faces_count),
      source_face_groups(source_faces_count), source_segments(2 * source_segments_count),
      source_segment_groups(source_segments_count), cell_edit_kinds(cells);
  std::vector<int64_t> source_face_ids(source_faces_count);
  std::vector<int64_t> source_segment_ids(source_segments_count);
  std::vector<double> witness_values(2 * vertices), witness_deviations(vertices),
      protection_radii(vertices), vertex_edge_parameters(vertices),
      vertex_ancestry_weights(vertex_ancestors), source_points(3 * source_points_count),
      source_face_tolerances(source_faces_count), source_segment_tolerances(source_segments_count),
      source_face_planes(9 * source_faces_count), source_face_plane_tolerances(source_faces_count);
  std::vector<uint64_t> cells_ids(cells), cell_vertices(4 * cells), facet_ids(facets),
      facet_vertices(3 * facets), segment_ids(segments), segment_vertices(2 * segments),
      vertex_ancestor_ids(vertex_ancestors), cell_ancestor_ids(cell_ancestors);
  std::vector<int64_t> vertex_ancestry_offsets(vertices + 1), cell_ancestry_offsets(cells + 1),
      vertex_ancestor_winding(3 * vertex_ancestors);
  phx_mc_tet_mesh_snapshot_request snapshot{};
  snapshot.vertex_capacity = sizes[0];
  snapshot.cell_capacity = sizes[1];
  snapshot.facet_capacity = sizes[2];
  snapshot.segment_capacity = sizes[3];
  snapshot.source_point_capacity = sizes[4];
  snapshot.source_face_capacity = sizes[5];
  snapshot.source_segment_capacity = sizes[6];
  snapshot.periodic_orbit_capacity = sizes[7];
  snapshot.periodic_member_capacity = sizes[8];
  snapshot.periodic_quotient_cell_capacity = sizes[9];
  snapshot.periodic_quotient_lifted_capacity = sizes[10];
  snapshot.periodic_entity_map_capacity = sizes[11];
  snapshot.periodic_entity_lifted_capacity = sizes[12];
  snapshot.periodic_entity_quotient_capacity = sizes[13];
  snapshot.periodic_entity_winding_capacity = sizes[14];
  snapshot.periodic_entity_corner_capacity = sizes[15];
  snapshot.periodic_entity_representative_capacity = sizes[16];
  snapshot.vertex_ancestry_capacity = sizes[17];
  snapshot.cell_ancestor_capacity = sizes[18];
  snapshot.vertex_ids = vertex_ids.data();
  snapshot.points = points.data();
  snapshot.vertex_size_targets = snapshot_vertex_sizes.data();
  snapshot.witness_strata = witness_strata.data();
  snapshot.witness_entities = witness_entities.data();
  snapshot.witness_parameters = witness_values.data();
  snapshot.witness_deviations = witness_deviations.data();
  snapshot.vertex_dimensions = vertex_dimensions.data();
  snapshot.protection_radii = protection_radii.data();
  snapshot.protected_vertices = protected_vertices.data();
  snapshot.vertex_edge_witness_present = witness_present.data();
  snapshot.vertex_edge_witness_endpoints = witness_endpoints.data();
  snapshot.vertex_edge_parameters = witness_parameters.data();
  snapshot.vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  snapshot.vertex_ancestor_ids = vertex_ancestor_ids.data();
  snapshot.vertex_ancestor_winding = vertex_ancestor_winding.data();
  snapshot.vertex_ancestry_weights = vertex_ancestry_weights.data();
  snapshot.cell_ancestry_offsets = cell_ancestry_offsets.data();
  snapshot.cell_ancestor_ids = cell_ancestor_ids.data();
  snapshot.cell_ancestry_edit_kinds = cell_edit_kinds.data();
  snapshot.cell_ids = cells_ids.data();
  snapshot.cell_vertices = cell_vertices.data();
  snapshot.cell_regions = cell_regions.data();
  snapshot.cell_size_targets = snapshot_cell_sizes.data();
  snapshot.facet_ids = facet_ids.data();
  snapshot.facet_vertices = facet_vertices.data();
  snapshot.facet_sources = facet_sources.data();
  snapshot.facet_orientations = facet_orientations.data();
  snapshot.protected_facets = protected_facets.data();
  snapshot.segment_ids = segment_ids.data();
  snapshot.segment_vertices = segment_vertices.data();
  snapshot.segment_sources = segment_sources.data();
  snapshot.protected_segments = protected_segments.data();
  snapshot.source_points = source_points.data();
  snapshot.source_faces = source_faces.data();
  snapshot.source_face_groups = source_face_groups.data();
  snapshot.source_face_ids = source_face_ids.data();
  snapshot.source_face_tolerances = source_face_tolerances.data();
  snapshot.source_face_planes = source_face_planes.data();
  snapshot.source_face_plane_tolerances = source_face_plane_tolerances.data();
  snapshot.source_segments = source_segments.data();
  snapshot.source_segment_groups = source_segment_groups.data();
  snapshot.source_segment_ids = source_segment_ids.data();
  snapshot.source_segment_tolerances = source_segment_tolerances.data();
  generation = 0;
  snapshot.generation = &generation;
  const int32_t snapshot_status = phx_mc_tet_mesh_snapshot(mesh, &snapshot);
  PHX_CHECK(snapshot_status == PHX_MC_OK);
  if (snapshot_status == PHX_MC_OK && geometry != nullptr) {
    geometry->points.swap(points);
    geometry->cell_ids.swap(cells_ids);
    geometry->cell_vertices.swap(cell_vertices);
  }
}
void persistent_create_snapshot_and_rejection() {
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr std::array<int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> sources{0, 1, 2, 3};
  constexpr std::array<uint64_t, 4> initial_vertex_ids{101, 211, 307, 409};
  constexpr std::array<uint64_t, 1> initial_cell_ids{9001};
  constexpr std::array<uint64_t, 4> initial_facet_ids{5101, 5102, 5103, 5104};
  constexpr std::array<int8_t, 4> initial_protected_vertices{0, 0, 0, 1};
  constexpr char source_id[] = "cad:shape:α";
  constexpr char source_revision[] = "revision:β";
  constexpr char source_map_id[] = "map:surface:γ";
  constexpr char frame_id[] = "frame:world";

  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 4;
  create.points = points.data();
  create.vertex_ids = initial_vertex_ids.data();
  create.tetrahedron_count = 1;
  create.tetrahedra = tetrahedra.data();
  create.cell_ids = initial_cell_ids.data();
  create.face_count = 4;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.facet_ids = initial_facet_ids.data();
  create.protected_vertices = initial_protected_vertices.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.identity_bound = 1;
  create.source_id = source_id;
  create.source_id_length = sizeof(source_id) - 1;
  create.source_revision = source_revision;
  create.source_revision_length = sizeof(source_revision) - 1;
  create.source_map_id = source_map_id;
  create.source_map_id_length = sizeof(source_map_id) - 1;
  create.frame_id = frame_id;
  create.frame_id_length = sizeof(frame_id) - 1;
  create.result = &mesh;
  constexpr char invalid_create_utf8[] = {
      static_cast<char>(0xc0), static_cast<char>(0xaf)};
  create.frame_id = invalid_create_utf8;
  create.frame_id_length = sizeof(invalid_create_utf8);
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(mesh == nullptr);
  create.frame_id = frame_id;
  create.frame_id_length = sizeof(frame_id) - 1;
  std::array<uint64_t, 4> duplicate_vertex_ids = initial_vertex_ids;
  duplicate_vertex_ids[3] = duplicate_vertex_ids[2];
  create.vertex_ids = duplicate_vertex_ids.data();
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
  constexpr std::array<int8_t, 4> invalid_protected_vertices{0, 0, 2, 0};
  create.protected_vertices = invalid_protected_vertices.data();
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(mesh == nullptr);
  create.protected_vertices = initial_protected_vertices.data();
  create.vertex_ids = initial_vertex_ids.data();
  std::array<double, 4> bad_initial_sizes{0.1, 0.2, -0.1, 0.4};
  create.initial_vertex_sizes = bad_initial_sizes.data();
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
  bad_initial_sizes[2] = std::numeric_limits<double>::quiet_NaN();
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
  create.initial_vertex_sizes = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  constexpr std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> expected_sizes{
      4, 1, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes == expected_sizes);

  std::array<uint64_t, 4> vertex_ids{};
  std::array<double, 12> copied_points{};
  std::array<int8_t, 4> witness_strata{};
  std::array<int32_t, 4> witness_entities{};
  std::array<double, 8> witness_parameters{};
  std::array<double, 4> witness_deviations{};
  std::array<int8_t, 4> vertex_dimensions{};
  std::array<double, 4> protection_radii{};
  std::array<double, 4> snapshot_vertex_sizes{};
  std::array<int8_t, 4> protected_vertices{};
  std::array<int8_t, 4> edge_witness_present{};
  std::array<uint64_t, 8> edge_witness_endpoints{};
  std::array<double, 4> edge_parameters{};
  std::array<int64_t, 5> vertex_ancestry_offsets{};
  std::array<int64_t, 2> cell_ancestry_offsets{};
  std::array<int32_t, 1> cell_ancestry_edit_kinds{};
  std::array<uint64_t, 1> cell_ids{};
  std::array<uint64_t, 4> cell_vertices{};
  std::array<int32_t, 1> cell_regions{};
  std::array<double, 1> snapshot_cell_sizes{};
  std::array<uint64_t, 4> facet_ids{};
  std::array<uint64_t, 12> facet_vertices{};
  std::array<int32_t, 4> facet_sources{};
  std::array<int8_t, 4> facet_orientations{};
  std::array<int8_t, 4> protected_facets{};
  std::array<double, 12> copied_source_points{};
  std::array<int32_t, 12> source_faces{};
  std::array<int32_t, 4> source_face_groups{};
  std::array<int64_t, 4> source_face_ids{};
  std::array<double, 4> source_face_tolerances{};
  std::array<double, 36> source_face_planes{};
  std::array<double, 4> source_face_plane_tolerances{};
  uint64_t generation = 99;
  phx_mc_tet_mesh_snapshot_request snapshot{};
  snapshot.vertex_capacity = 4;
  snapshot.cell_capacity = 1;
  snapshot.facet_capacity = 4;
  snapshot.source_point_capacity = 4;
  snapshot.source_face_capacity = 4;
  snapshot.vertex_ids = vertex_ids.data();
  snapshot.points = copied_points.data();
  snapshot.vertex_size_targets = snapshot_vertex_sizes.data();
  snapshot.witness_strata = witness_strata.data();
  snapshot.witness_entities = witness_entities.data();
  snapshot.witness_parameters = witness_parameters.data();
  snapshot.witness_deviations = witness_deviations.data();
  snapshot.vertex_dimensions = vertex_dimensions.data();
  snapshot.protection_radii = protection_radii.data();
  snapshot.protected_vertices = protected_vertices.data();
  snapshot.vertex_edge_witness_present = edge_witness_present.data();
  snapshot.vertex_edge_witness_endpoints = edge_witness_endpoints.data();
  snapshot.vertex_edge_parameters = edge_parameters.data();
  snapshot.vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  snapshot.cell_ancestry_offsets = cell_ancestry_offsets.data();
  snapshot.cell_ancestry_edit_kinds = cell_ancestry_edit_kinds.data();
  snapshot.cell_ids = cell_ids.data();
  snapshot.cell_vertices = cell_vertices.data();
  snapshot.cell_regions = cell_regions.data();
  snapshot.cell_size_targets = snapshot_cell_sizes.data();
  snapshot.facet_ids = facet_ids.data();
  snapshot.facet_vertices = facet_vertices.data();
  snapshot.facet_sources = facet_sources.data();
  snapshot.facet_orientations = facet_orientations.data();
  snapshot.protected_facets = protected_facets.data();
  snapshot.source_points = copied_source_points.data();
  snapshot.source_faces = source_faces.data();
  snapshot.source_face_groups = source_face_groups.data();
  snapshot.source_face_ids = source_face_ids.data();
  snapshot.source_face_tolerances = source_face_tolerances.data();
  snapshot.source_face_planes = source_face_planes.data();
  snapshot.source_face_plane_tolerances = source_face_plane_tolerances.data();
  snapshot.generation = &generation;
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_OK);
  PHX_CHECK(generation == 0);
  PHX_CHECK(protected_vertices == initial_protected_vertices);
  PHX_CHECK(protection_radii[3] == 0.0 && protected_vertices[3] == 1);
  constexpr std::array<int64_t, 5> empty_vertex_ancestry_offsets{0, 0, 0, 0, 0};
  constexpr std::array<int64_t, 2> empty_cell_ancestry_offsets{0, 0};
  PHX_CHECK(vertex_ancestry_offsets == empty_vertex_ancestry_offsets);
  PHX_CHECK(cell_ancestry_offsets == empty_cell_ancestry_offsets);
  PHX_CHECK(cell_ancestry_edit_kinds[0] == -1);
  for (std::size_t row = 0; row < edge_witness_present.size(); ++row) {
    PHX_CHECK(edge_witness_present[row] == 0);
    PHX_CHECK(edge_witness_endpoints[2 * row] == 0 &&
              edge_witness_endpoints[2 * row + 1] == 0);
    PHX_CHECK(edge_parameters[row] == 0.0);
  }
  int32_t identity_bound = -1;
  int64_t source_id_length = -1;
  int64_t source_revision_length = -1;
  int64_t source_map_id_length = -1;
  int64_t frame_id_length = -1;
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &source_id_length,
                &source_revision_length, &source_map_id_length,
                &frame_id_length) == PHX_MC_OK);
  PHX_CHECK(generation == 0 && identity_bound == 1);
  PHX_CHECK(source_id_length == sizeof(source_id) - 1 &&
            source_revision_length == sizeof(source_revision) - 1 &&
            source_map_id_length == sizeof(source_map_id) - 1 &&
            frame_id_length == sizeof(frame_id) - 1);
  std::array<char, sizeof(source_id) - 1> copied_source_id{};
  std::array<char, sizeof(source_revision) - 1> copied_source_revision{};
  std::array<char, sizeof(source_map_id) - 1> copied_source_map_id{};
  std::array<char, sizeof(frame_id) - 1> copied_frame_id{};
  PHX_CHECK(phx_mc_tet_mesh_identity(
                mesh, copied_source_id.size(), copied_source_id.data(),
                copied_source_revision.size(), copied_source_revision.data(),
                copied_source_map_id.size(), copied_source_map_id.data(),
                copied_frame_id.size(), copied_frame_id.data()) == PHX_MC_OK);
  PHX_CHECK(std::string_view(copied_source_id.data(), copied_source_id.size()) ==
            std::string_view(source_id, sizeof(source_id) - 1));
  PHX_CHECK(std::string_view(copied_source_revision.data(), copied_source_revision.size()) ==
            std::string_view(source_revision, sizeof(source_revision) - 1));
  PHX_CHECK(std::string_view(copied_source_map_id.data(), copied_source_map_id.size()) ==
            std::string_view(source_map_id, sizeof(source_map_id) - 1));
  PHX_CHECK(std::string_view(copied_frame_id.data(), copied_frame_id.size()) ==
            std::string_view(frame_id, sizeof(frame_id) - 1));
  constexpr std::array<uint64_t, 4> expected_cell_vertices{101, 211, 307, 409};
  constexpr std::array<int64_t, 4> expected_source_ids{0, 1, 2, 3};
  PHX_CHECK(cell_vertices == expected_cell_vertices);
  PHX_CHECK(facet_ids == initial_facet_ids);
  PHX_CHECK(vertex_ids == initial_vertex_ids && cell_ids == initial_cell_ids);
  PHX_CHECK(source_faces == faces && source_face_groups == sources &&
            source_face_ids == expected_source_ids);
  PHX_CHECK(copied_points == points && copied_source_points == points);

  constexpr std::array<uint64_t, 4> untouched_expected{77, 77, 77, 77};
  std::array<uint64_t, 4> untouched = untouched_expected;
  snapshot.vertex_capacity = 3;
  snapshot.vertex_ids = untouched.data();
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(untouched == untouched_expected);

  snapshot.vertex_capacity = 4;
  std::array<double, 12> untouched_points{};
  untouched_points.fill(91.0);
  snapshot.points = untouched_points.data();
  snapshot.cell_vertices = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(untouched == untouched_expected);
  PHX_CHECK(untouched_points[0] == 91.0 && untouched_points[11] == 91.0);

  int64_t unmet_count = -1;
  PHX_CHECK(phx_mc_tet_mesh_unmet_count(mesh, &unmet_count) == PHX_MC_OK);
  PHX_CHECK(unmet_count == 0);
  PHX_CHECK(phx_mc_tet_mesh_unmet(mesh, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, nullptr, nullptr) == PHX_MC_OK);

  phx_mc_tet_mesh_edit_request edit{};
  int32_t edit_status = -1;
  edit.expected_generation = 8;
  edit.kind = PHX_MC_TET_EDIT_RECONNECT;
  edit.identity_bound = 1;
  edit.source_id = source_id;
  edit.source_id_length = sizeof(source_id) - 1;
  edit.source_revision = source_revision;
  edit.source_revision_length = sizeof(source_revision) - 1;
  edit.source_map_id = source_map_id;
  edit.source_map_id_length = sizeof(source_map_id) - 1;
  edit.frame_id = frame_id;
  edit.frame_id_length = sizeof(frame_id) - 1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_STALE_GENERATION);
  edit.expected_generation = 0;
  constexpr char invalid_utf8[] = {static_cast<char>(0xc0), static_cast<char>(0xaf)};
  edit.frame_id = invalid_utf8;
  edit.frame_id_length = sizeof(invalid_utf8);
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_INVALID_ARGUMENT);
  edit.frame_id = frame_id;
  edit.frame_id_length = sizeof(frame_id) - 1;
  constexpr char stale_revision[] = "revision:old";
  edit.source_revision = stale_revision;
  edit.source_revision_length = sizeof(stale_revision) - 1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_STALE_GENERATION);
  edit.source_revision = source_revision;
  edit.source_revision_length = sizeof(source_revision) - 1;
  edit.expected_generation = 0;
  edit.kind = PHX_MC_TET_EDIT_RELOCATE;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  edit.kind = PHX_MC_TET_EDIT_RECONNECT;
  edit.removed_cell_count = 1;
  const uint64_t removed_cell = 9001;
  edit.removed_cell_ids = &removed_cell;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  PHX_CHECK(generation == 0);
  phx_mc_tet_mesh_edit_request split{};
  split.expected_generation = 0;
  split.identity_bound = 1;
  split.source_id = source_id;
  split.source_id_length = sizeof(source_id) - 1;
  split.source_revision = source_revision;
  split.source_revision_length = sizeof(source_revision) - 1;
  split.source_map_id = source_map_id;
  split.source_map_id_length = sizeof(source_map_id) - 1;
  split.frame_id = frame_id;
  split.frame_id_length = sizeof(frame_id) - 1;
  split.kind = PHX_MC_TET_EDIT_SPLIT;
  const std::array<uint64_t, 2> split_edge{101, 211};
  const uint64_t parent_cell = 9001;
  const uint64_t inserted_vertex = 8001;
  std::array<double, 3> inserted_point{0.5, 0.0, 0.0};
  const int8_t has_edge_witness = 1;
  const std::array<uint64_t, 2> witness_edge{101, 211};
  const double edge_parameter = 0.25;
  const std::array<int64_t, 2> ancestry_offsets{0, 2};
  const std::array<uint64_t, 2> ancestor_ids{101, 211};
  const std::array<int64_t, 6> ancestor_winding{};
  std::array<double, 2> ancestry_weights{0.75, 0.25};
  split.split_edge_count = 1;
  split.split_edges = split_edge.data();
  split.removed_cell_count = 1;
  split.removed_cell_ids = &parent_cell;
  split.changed_vertex_count = 1;
  split.changed_vertex_ids = &inserted_vertex;
  split.changed_vertex_points = inserted_point.data();
  split.changed_edge_witness_present = &has_edge_witness;
  split.changed_edge_witness_endpoints = witness_edge.data();
  split.changed_edge_parameters = &edge_parameter;
  split.changed_vertex_ancestry_count = 2;
  split.changed_vertex_ancestry_offsets = ancestry_offsets.data();
  split.changed_vertex_ancestor_ids = ancestor_ids.data();
  split.changed_vertex_ancestor_winding = ancestor_winding.data();
  split.changed_vertex_ancestry_weights = ancestry_weights.data();
  constexpr std::array<uint64_t, 2> split_replacement_cell_ids{9002, 9003};
  constexpr std::array<uint64_t, 8> split_replacement_vertices{
      101, 8001, 307, 409, 8001, 211, 307, 409};
  constexpr std::array<int32_t, 8> split_replacement_facet_sources{
      -1, 1, 2, 3, 0, -1, 2, 3};
  constexpr std::array<int32_t, 2> split_replacement_regions{0, 0};
  constexpr std::array<int64_t, 3> split_cell_ancestry_offsets{0, 1, 2};
  constexpr std::array<uint64_t, 2> split_cell_ancestor_ids{9001, 9001};
  split.replacement_cell_count = static_cast<int64_t>(split_replacement_cell_ids.size());
  split.replacement_cell_ids = split_replacement_cell_ids.data();
  split.replacement_cell_vertices = split_replacement_vertices.data();
  split.replacement_cell_facet_sources = split_replacement_facet_sources.data();
  split.replacement_regions = split_replacement_regions.data();
  split.replacement_cell_ancestry_count =
      static_cast<int64_t>(split_cell_ancestor_ids.size());
  split.replacement_cell_ancestry_offsets = split_cell_ancestry_offsets.data();
  split.replacement_cell_ancestor_ids = split_cell_ancestor_ids.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &split, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_INVALID_GEOMETRY);
  inserted_point[0] = 0.25;
  ancestry_weights = {0.5, 0.5};
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &split, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_INVALID_GEOMETRY);
  ancestry_weights = {0.75, 0.25};
  split.replacement_cell_count = 0;
  split.replacement_cell_ids = nullptr;
  split.replacement_cell_vertices = nullptr;
  split.replacement_cell_facet_sources = nullptr;
  split.replacement_regions = nullptr;
  split.replacement_cell_ancestry_count = 0;
  split.replacement_cell_ancestry_offsets = nullptr;
  split.replacement_cell_ancestor_ids = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &split, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 4 && sizes[1] == 1);
  int32_t run_status = -1;
  int64_t inserted_vertices = 0;
  int64_t remaining_cells = 0;
  PHX_CHECK(phx_mc_tet_mesh_refine(
                mesh, generation, identity_bound, source_id, sizeof(source_id) - 1,
                source_revision, sizeof(source_revision) - 1, source_map_id,
                sizeof(source_map_id) - 1, frame_id, sizeof(frame_id) - 1,
                0.5, 0, nullptr, &edit_status, &run_status, &inserted_vertices,
                &remaining_cells) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  // A radius-edge bound below the regular tetrahedron's is unattainable: the
  // run ends unresolved after circumcenter insertions (if any) that keep a
  // valid, ancestry-consistent state.
  PHX_CHECK(run_status == PHX_MC_UNRESOLVED && inserted_vertices >= 0);
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> refined_sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, refined_sizes.data()) == PHX_MC_OK);
  PHX_CHECK(refined_sizes[0] == 4 + inserted_vertices && refined_sizes[1] >= 1);
  const std::size_t refined_vertex_count = static_cast<std::size_t>(refined_sizes[0]);
  const std::size_t refined_cell_count = static_cast<std::size_t>(refined_sizes[1]);
  const std::size_t refined_vertex_ancestry = static_cast<std::size_t>(refined_sizes[17]);
  const std::size_t refined_cell_ancestry = static_cast<std::size_t>(refined_sizes[18]);
  std::vector<uint64_t> refined_vertex_ids(refined_vertex_count);
  std::vector<double> refined_points(3 * refined_vertex_count);
  std::vector<double> refined_vertex_size_targets(refined_vertex_count);
  std::vector<int8_t> refined_witness_strata(refined_vertex_count);
  std::vector<int32_t> refined_witness_entities(refined_vertex_count);
  std::vector<double> refined_witness_parameters(2 * refined_vertex_count);
  std::vector<double> refined_witness_deviations(refined_vertex_count);
  std::vector<int8_t> refined_vertex_dimensions(refined_vertex_count);
  std::vector<double> refined_protection_radii(refined_vertex_count);
  std::vector<int8_t> refined_protected_vertices(refined_vertex_count);
  std::vector<int8_t> refined_edge_witness_present(refined_vertex_count);
  std::vector<uint64_t> refined_edge_witness_endpoints(2 * refined_vertex_count);
  std::vector<double> refined_edge_parameters(refined_vertex_count);
  std::vector<int64_t> refined_vertex_ancestry_offsets(refined_vertex_count + 1);
  std::vector<uint64_t> refined_vertex_ancestor_ids(refined_vertex_ancestry + 1);
  std::vector<int64_t> refined_vertex_ancestor_winding(3 * refined_vertex_ancestry + 3);
  std::vector<double> refined_vertex_ancestry_weights(refined_vertex_ancestry + 1);
  std::vector<uint64_t> refined_cell_ids(refined_cell_count);
  std::vector<uint64_t> refined_cell_vertices(4 * refined_cell_count);
  std::vector<int32_t> refined_cell_regions(refined_cell_count);
  std::vector<double> refined_cell_size_targets(refined_cell_count);
  std::vector<int64_t> refined_cell_ancestry_offsets(refined_cell_count + 1);
  std::vector<uint64_t> refined_cell_ancestor_ids(refined_cell_ancestry + 1);
  std::vector<int32_t> refined_cell_ancestry_edit_kinds(refined_cell_count);
  uint64_t refined_generation = 0;
  phx_mc_tet_mesh_snapshot_request refined_snapshot{};
  refined_snapshot.vertex_capacity = refined_sizes[0];
  refined_snapshot.cell_capacity = refined_sizes[1];
  refined_snapshot.facet_capacity = 4;
  refined_snapshot.source_point_capacity = 4;
  refined_snapshot.source_face_capacity = 4;
  refined_snapshot.vertex_ancestry_capacity = refined_sizes[17];
  refined_snapshot.cell_ancestor_capacity = refined_sizes[18];
  refined_snapshot.vertex_ids = refined_vertex_ids.data();
  refined_snapshot.points = refined_points.data();
  refined_snapshot.vertex_size_targets = refined_vertex_size_targets.data();
  refined_snapshot.witness_strata = refined_witness_strata.data();
  refined_snapshot.witness_entities = refined_witness_entities.data();
  refined_snapshot.witness_parameters = refined_witness_parameters.data();
  refined_snapshot.witness_deviations = refined_witness_deviations.data();
  refined_snapshot.vertex_dimensions = refined_vertex_dimensions.data();
  refined_snapshot.protection_radii = refined_protection_radii.data();
  refined_snapshot.protected_vertices = refined_protected_vertices.data();
  refined_snapshot.vertex_edge_witness_present = refined_edge_witness_present.data();
  refined_snapshot.vertex_edge_witness_endpoints = refined_edge_witness_endpoints.data();
  refined_snapshot.vertex_edge_parameters = refined_edge_parameters.data();
  refined_snapshot.vertex_ancestry_offsets = refined_vertex_ancestry_offsets.data();
  refined_snapshot.vertex_ancestor_ids = refined_vertex_ancestor_ids.data();
  refined_snapshot.vertex_ancestor_winding = refined_vertex_ancestor_winding.data();
  refined_snapshot.vertex_ancestry_weights = refined_vertex_ancestry_weights.data();
  refined_snapshot.cell_ids = refined_cell_ids.data();
  refined_snapshot.cell_vertices = refined_cell_vertices.data();
  refined_snapshot.cell_regions = refined_cell_regions.data();
  refined_snapshot.cell_size_targets = refined_cell_size_targets.data();
  refined_snapshot.cell_ancestry_offsets = refined_cell_ancestry_offsets.data();
  refined_snapshot.cell_ancestor_ids = refined_cell_ancestor_ids.data();
  refined_snapshot.cell_ancestry_edit_kinds = refined_cell_ancestry_edit_kinds.data();
  refined_snapshot.facet_ids = facet_ids.data();
  refined_snapshot.facet_vertices = facet_vertices.data();
  refined_snapshot.facet_sources = facet_sources.data();
  refined_snapshot.facet_orientations = facet_orientations.data();
  refined_snapshot.protected_facets = protected_facets.data();
  refined_snapshot.source_points = copied_source_points.data();
  refined_snapshot.source_faces = source_faces.data();
  refined_snapshot.source_face_groups = source_face_groups.data();
  refined_snapshot.source_face_ids = source_face_ids.data();
  refined_snapshot.source_face_tolerances = source_face_tolerances.data();
  refined_snapshot.source_face_planes = source_face_planes.data();
  refined_snapshot.source_face_plane_tolerances = source_face_plane_tolerances.data();
  refined_snapshot.generation = &refined_generation;
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &refined_snapshot) == PHX_MC_OK);
  PHX_CHECK(refined_generation == generation + static_cast<uint64_t>(inserted_vertices));
  // Circumcenters carry no affine stencil, and every refined cell names the
  // cavity cells it replaced as ancestors.
  PHX_CHECK(refined_sizes[17] == 0);
  for (std::size_t row = 0; row < refined_cell_count; ++row) {
    const std::array<const double*, 4> corners{
        [&] {
          std::array<const double*, 4> result{};
          for (std::size_t corner = 0; corner < 4; ++corner) {
            const uint64_t id = refined_cell_vertices[4 * row + corner];
            const auto found =
                std::find(refined_vertex_ids.begin(), refined_vertex_ids.end(), id);
            PHX_CHECK(found != refined_vertex_ids.end());
            result[corner] = refined_points.data() +
                             3 * static_cast<std::size_t>(found - refined_vertex_ids.begin());
          }
          return result;
        }()};
    PHX_CHECK(phx::mc::orient3d(corners[0], corners[1], corners[2], corners[3]) > 0);
    if (refined_cell_ancestry_edit_kinds[row] == PHX_MC_TET_EDIT_REFINE_CELL) {
      PHX_CHECK(refined_cell_ancestry_offsets[row + 1] > refined_cell_ancestry_offsets[row]);
    }
  }
  for (std::size_t row = 0; row < refined_cell_ancestry; ++row) {
    PHX_CHECK(refined_cell_ancestor_ids[row] >= 9001);
  }
  int64_t report_count = 0;
  PHX_CHECK(phx_mc_tet_mesh_unmet_count(mesh, &report_count) == PHX_MC_OK);
  PHX_CHECK(report_count > 0 && report_count >= remaining_cells);
  const uint64_t untouched_report_id = 987;
  uint64_t short_report_id = untouched_report_id;
  PHX_CHECK(phx_mc_tet_mesh_unmet(mesh, report_count - 1, &short_report_id,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(short_report_id == untouched_report_id);
  std::vector<uint64_t> report_cells(static_cast<std::size_t>(report_count));
  std::vector<int32_t> report_criteria(static_cast<std::size_t>(report_count));
  std::vector<int32_t> report_reasons(static_cast<std::size_t>(report_count));
  std::vector<double> report_values(static_cast<std::size_t>(report_count));
  std::vector<uint64_t> report_apexes(static_cast<std::size_t>(report_count));
  std::vector<double> report_angles(static_cast<std::size_t>(report_count));
  std::vector<uint64_t> report_facets(static_cast<std::size_t>(report_count));
  std::vector<double> report_facet_ratios(static_cast<std::size_t>(report_count));
  PHX_CHECK(phx_mc_tet_mesh_unmet(mesh, report_count, report_cells.data(),
                                  report_criteria.data(), report_reasons.data(),
                                  report_values.data(), report_apexes.data(),
                                  report_angles.data(), report_facets.data(),
                                  report_facet_ratios.data()) == PHX_MC_OK);
  // The domain is one tetrahedron: its dihedral angles are acute, so a
  // boundary split near an edge is protected and its cell carries the apex
  // (a vertex present at refinement) and the angle as evidence.
  for (std::size_t row = 0; row < report_cells.size(); ++row) {
    PHX_CHECK(report_criteria[row] == PHX_MC_TET_UNMET_RADIUS_EDGE);
    PHX_CHECK(report_reasons[row] == PHX_MC_TET_UNMET_NO_IMPROVEMENT ||
              report_reasons[row] == PHX_MC_TET_UNMET_FIXED_BOUNDARY ||
              report_reasons[row] == PHX_MC_TET_UNMET_REFUSED ||
              report_reasons[row] == PHX_MC_TET_UNMET_INPUT_ANGLE);
    PHX_CHECK(report_values[row] > 0.5);
    if (report_reasons[row] == PHX_MC_TET_UNMET_INPUT_ANGLE) {
      PHX_CHECK(std::find(refined_vertex_ids.begin(), refined_vertex_ids.end(),
                          report_apexes[row]) != refined_vertex_ids.end());
      PHX_CHECK(report_angles[row] > 0.0 && report_angles[row] < 90.0);
    } else {
      PHX_CHECK(report_apexes[row] == PHX_MC_TET_UNMET_NO_APEX &&
                std::isnan(report_angles[row]));
    }
    // Refinement records never carry an improvement forcing facet.
    PHX_CHECK(report_facets[row] == PHX_MC_TET_UNMET_NO_FACET &&
              std::isnan(report_facet_ratios[row]));
  }
  int64_t collapsed_vertices = 0;
  int64_t relocated_vertices = 0;
  double score_quantum = 0.0;
  PHX_CHECK(phx_mc_tet_mesh_improve_quality(
                mesh, refined_generation, identity_bound, source_id, sizeof(source_id) - 1,
                source_revision, sizeof(source_revision) - 1, source_map_id,
                sizeof(source_map_id) - 1, frame_id, sizeof(frame_id) - 1,
                2.0, 0.0, 0.0, 0, 0, nullptr, 0, nullptr, &edit_status, &run_status, &inserted_vertices,
                &collapsed_vertices, &relocated_vertices, &remaining_cells,
                &score_quantum) == PHX_MC_OK);
  PHX_CHECK(score_quantum == 0x1p-20);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  // Status and per-cell records agree: success exactly when no cell is unmet.
  PHX_CHECK((run_status == PHX_MC_OK && remaining_cells == 0) ||
            (run_status == PHX_MC_UNRESOLVED && remaining_cells > 0));
  PHX_CHECK(phx_mc_tet_mesh_unmet_count(mesh, &report_count) == PHX_MC_OK);
  PHX_CHECK((report_count == 0) == (remaining_cells == 0));
  report_cells.resize(static_cast<std::size_t>(report_count));
  report_criteria.resize(static_cast<std::size_t>(report_count));
  report_reasons.resize(static_cast<std::size_t>(report_count));
  report_values.resize(static_cast<std::size_t>(report_count));
  report_apexes.resize(static_cast<std::size_t>(report_count));
  report_angles.resize(static_cast<std::size_t>(report_count));
  report_facets.resize(static_cast<std::size_t>(report_count));
  report_facet_ratios.resize(static_cast<std::size_t>(report_count));
  PHX_CHECK(phx_mc_tet_mesh_unmet(mesh, report_count, report_cells.data(),
                                  report_criteria.data(), report_reasons.data(),
                                  report_values.data(), report_apexes.data(),
                                  report_angles.data(), report_facets.data(),
                                  report_facet_ratios.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < report_cells.size(); ++row) {
    PHX_CHECK(report_criteria[row] == PHX_MC_TET_UNMET_RADIUS_EDGE);
    PHX_CHECK(report_reasons[row] == PHX_MC_TET_UNMET_NO_IMPROVEMENT ||
              report_reasons[row] == PHX_MC_TET_UNMET_FIXED_BOUNDARY);
    PHX_CHECK(report_values[row] > 2.0);
    // A forcing facet is recorded exactly for FIXED_BOUNDARY rows, and its
    // own ratio bounds the cell's from below.
    if (report_reasons[row] == PHX_MC_TET_UNMET_FIXED_BOUNDARY) {
      PHX_CHECK(report_facets[row] != PHX_MC_TET_UNMET_NO_FACET &&
                report_facet_ratios[row] > 2.0 &&
                report_facet_ratios[row] <= report_values[row] * (1.0 + 1.0e-12));
    } else {
      PHX_CHECK(report_facets[row] == PHX_MC_TET_UNMET_NO_FACET &&
                std::isnan(report_facet_ratios[row]));
    }
  }
  phx_mc_tet_mesh_free(mesh);
  mesh = nullptr;
  constexpr std::array<int32_t, 2> optional_segments{0, 1};
  constexpr std::array<int32_t, 1> optional_segment_sources{0};
  create.vertex_ids = nullptr;
  create.cell_ids = nullptr;
  create.facet_ids = nullptr;
  create.segment_count = 1;
  create.segments = optional_segments.data();
  create.segment_sources = optional_segment_sources.data();
  create.segment_ids = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  std::array<uint64_t, 1> optional_segment_ids{};
  std::array<uint64_t, 2> optional_segment_vertices{};
  std::array<int32_t, 1> optional_segment_source_rows{};
  std::array<int8_t, 1> optional_protected_segments{};
  std::array<int32_t, 2> optional_source_segments{};
  std::array<int32_t, 1> optional_source_segment_groups{};
  std::array<int64_t, 1> optional_source_segment_ids{};
  std::array<double, 1> optional_source_segment_tolerances{};
  snapshot.segment_capacity = 1;
  snapshot.source_segment_capacity = 1;
  snapshot.segment_ids = optional_segment_ids.data();
  snapshot.segment_vertices = optional_segment_vertices.data();
  snapshot.segment_sources = optional_segment_source_rows.data();
  snapshot.protected_segments = optional_protected_segments.data();
  snapshot.source_segments = optional_source_segments.data();
  snapshot.source_segment_groups = optional_source_segment_groups.data();
  snapshot.source_segment_ids = optional_source_segment_ids.data();
  snapshot.source_segment_tolerances = optional_source_segment_tolerances.data();
  snapshot.vertex_ids = vertex_ids.data();
  snapshot.points = copied_points.data();
  snapshot.cell_vertices = cell_vertices.data();
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> optional_id_sizes{};
  constexpr std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES>
      expected_optional_id_sizes{
          4, 1, 4, 1, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, optional_id_sizes.data()) ==
            PHX_MC_OK);
  PHX_CHECK(optional_id_sizes == expected_optional_id_sizes);
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_OK);
  constexpr std::array<uint64_t, 4> row_vertex_ids{0, 1, 2, 3};
  constexpr std::array<uint64_t, 1> row_cell_ids{0};
  constexpr std::array<uint64_t, 4> row_facet_ids{0, 1, 2, 3};
  PHX_CHECK(vertex_ids == row_vertex_ids && cell_ids == row_cell_ids &&
            facet_ids == row_facet_ids);
  constexpr std::array<uint64_t, 2> row_segment_vertices{0, 1};
  PHX_CHECK(optional_segment_ids[0] == 0 &&
            optional_segment_vertices == row_segment_vertices &&
            optional_segment_source_rows[0] == 0 &&
            optional_source_segments == optional_segments &&
            optional_source_segment_groups[0] == 0 &&
            optional_source_segment_ids[0] == 0 &&
            optional_source_segment_tolerances[0] == 0.0);
  phx_mc_tet_mesh_free(mesh);
}
void persistent_periodic_state_and_atomic_relocation() {
  constexpr std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 0.25, 0.25, 0.25};
  constexpr std::array<int32_t, 16> tetrahedra{
      1, 3, 2, 4, 0, 2, 3, 4, 0, 3, 1, 4, 0, 1, 2, 4};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr std::array<int32_t, 4> regions{0, 0, 0, 0};
  constexpr std::array<uint64_t, 5> vertex_ids{101, 211, 307, 409, 500};
  constexpr std::array<uint64_t, 4> cell_ids{9001, 9002, 9003, 9004};
  constexpr std::array<uint64_t, 4> facet_ids{5101, 5102, 5103, 5104};
  constexpr std::array<uint64_t, 29> entity_lifted_ids{
      101, 211, 307, 409, 500,
      11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010,
      5101, 5102, 5103, 5104, 12001, 12002, 12003, 12004, 12005, 12006,
      9001, 9002, 9003, 9004};
  constexpr std::array<int64_t, 4> entity_counts{5, 10, 10, 4};
  constexpr char source_mesh_id[] = "mesh:α";
  constexpr char source_topology_id[] = "topology:β";
  constexpr char lifted_topology_id[] = "lifted:γ";
  constexpr char periodic_topology_id[] = "periodic:δ";
  constexpr char cell_entity_set_id[] = "cells:ε";
  constexpr char edge_entity_set_id[] = "edges:ζ";

  std::array<uint64_t, 29> entity_quotient_ids{};
  std::array<int32_t, 29> entity_lifted_to_quotient{};
  std::array<int32_t, 29> entity_orientation_signs{};
  std::array<int64_t, 29> entity_winding_shifts{};
  std::array<int32_t, 71> entity_corner_permutations{};
  std::array<int32_t, 29> entity_representative_rows{};
  std::array<phx_mc_tet_periodic_entity_orbit_map, 4> entity_maps{};
  std::size_t lifted_offset = 0;
  std::size_t quotient_offset = 0;
  std::size_t winding_offset = 0;
  std::size_t corner_offset = 0;
  std::size_t representative_offset = 0;
  for (std::size_t dimension = 0; dimension < entity_maps.size(); ++dimension) {
    phx_mc_tet_periodic_entity_orbit_map& map = entity_maps[dimension];
    const std::size_t count = static_cast<std::size_t>(entity_counts[dimension]);
    const std::size_t arity = dimension + 1;
    map.dimension = static_cast<int32_t>(dimension);
    map.arity = static_cast<int32_t>(arity);
    map.rank = 1;
    map.lifted_offset = static_cast<int64_t>(lifted_offset);
    map.lifted_count = static_cast<int64_t>(count);
    map.quotient_offset = static_cast<int64_t>(quotient_offset);
    map.quotient_count = static_cast<int64_t>(count);
    map.winding_offset = static_cast<int64_t>(winding_offset);
    map.corner_permutation_offset = static_cast<int64_t>(corner_offset);
    map.representative_offset = static_cast<int64_t>(representative_offset);
    for (std::size_t row = 0; row < count; ++row) {
      entity_quotient_ids[quotient_offset + row] = dimension == 3
          ? 50001 + row : 30001 + quotient_offset + row;
      entity_lifted_to_quotient[lifted_offset + row] = static_cast<int32_t>(row);
      entity_orientation_signs[lifted_offset + row] = 1;
      entity_representative_rows[representative_offset + row] = static_cast<int32_t>(row);
      for (std::size_t corner = 0; corner < arity; ++corner) {
        entity_corner_permutations[corner_offset + row * arity + corner] =
            static_cast<int32_t>(corner);
      }
    }
    lifted_offset += count;
    quotient_offset += count;
    winding_offset += count;
    corner_offset += count * arity;
    representative_offset += count;
  }

  phx_mc_tet_periodic_identity_map_view identity_map{};
  identity_map.identity_bound = 1;
  identity_map.source_mesh_id = source_mesh_id;
  identity_map.source_mesh_id_length = sizeof(source_mesh_id) - 1;
  identity_map.source_topology_id = source_topology_id;
  identity_map.source_topology_id_length = sizeof(source_topology_id) - 1;
  identity_map.lifted_topology_id = lifted_topology_id;
  identity_map.lifted_topology_id_length = sizeof(lifted_topology_id) - 1;
  identity_map.periodic_topology_id = periodic_topology_id;
  identity_map.periodic_topology_id_length = sizeof(periodic_topology_id) - 1;
  identity_map.cell_entity_set_id = cell_entity_set_id;
  identity_map.cell_entity_set_id_length = sizeof(cell_entity_set_id) - 1;
  identity_map.edge_entity_set_id = edge_entity_set_id;
  identity_map.edge_entity_set_id_length = sizeof(edge_entity_set_id) - 1;
  identity_map.entity_map_count = static_cast<int64_t>(entity_maps.size());
  identity_map.entity_maps = entity_maps.data();
  identity_map.entity_lifted_count = static_cast<int64_t>(entity_lifted_ids.size());
  identity_map.entity_lifted_ids = entity_lifted_ids.data();
  identity_map.entity_lifted_to_quotient = entity_lifted_to_quotient.data();
  identity_map.entity_orientation_signs = entity_orientation_signs.data();
  identity_map.entity_quotient_count = static_cast<int64_t>(entity_quotient_ids.size());
  identity_map.entity_quotient_ids = entity_quotient_ids.data();
  identity_map.entity_winding_count = static_cast<int64_t>(entity_winding_shifts.size());
  identity_map.entity_winding_shifts = entity_winding_shifts.data();
  identity_map.entity_corner_count = static_cast<int64_t>(entity_corner_permutations.size());
  identity_map.entity_corner_permutations = entity_corner_permutations.data();
  identity_map.entity_representative_count =
      static_cast<int64_t>(entity_representative_rows.size());
  identity_map.entity_representative_rows = entity_representative_rows.data();

  constexpr std::array<double, 12> identity_transform{
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  std::array<phx_mc_tet_periodic_orbit, 5> orbits{};
  std::array<phx_mc_tet_periodic_vertex_member, 5> members{};
  for (std::size_t row = 0; row < orbits.size(); ++row) {
    orbits[row].id = 20001 + row;
    orbits[row].representative_vertex_id = vertex_ids[row];
    orbits[row].member_offset = static_cast<int64_t>(row);
    orbits[row].member_count = 1;
    members[row].vertex_id = vertex_ids[row];
    for (std::size_t value = 0; value < identity_transform.size(); ++value) {
      members[row].representative_to_member[value] = identity_transform[value];
    }
  }
  std::array<phx_mc_tet_periodic_quotient_cell, 4> quotient_cells{};
  for (std::size_t row = 0; row < quotient_cells.size(); ++row) {
    quotient_cells[row].orbit_id = 50001 + row;
    quotient_cells[row].id = 50001 + row;
    quotient_cells[row].region = regions[row];
    quotient_cells[row].lifted_cell_offset = static_cast<int64_t>(row);
    quotient_cells[row].lifted_cell_count = 1;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      quotient_cells[row].representative_vertices[corner] =
          vertex_ids[static_cast<std::size_t>(tetrahedra[4 * row + corner])];
    }
  }
  phx_mc_tet_periodic_state_view periodic_state{};
  periodic_state.identity_map = identity_map;
  periodic_state.periodic_orbit_count = static_cast<int64_t>(orbits.size());
  periodic_state.periodic_orbits = orbits.data();
  periodic_state.periodic_member_count = static_cast<int64_t>(members.size());
  periodic_state.periodic_members = members.data();
  periodic_state.periodic_quotient_cell_count =
      static_cast<int64_t>(quotient_cells.size());
  periodic_state.periodic_quotient_cells = quotient_cells.data();
  periodic_state.periodic_quotient_lifted_count = static_cast<int64_t>(cell_ids.size());
  periodic_state.periodic_quotient_lifted_cell_ids = cell_ids.data();

  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = static_cast<int64_t>(points.size() / 3);
  create.points = points.data();
  create.vertex_ids = vertex_ids.data();
  create.tetrahedron_count = static_cast<int64_t>(tetrahedra.size() / 4);
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.cell_ids = cell_ids.data();
  create.face_count = static_cast<int64_t>(faces.size() / 3);
  create.faces = faces.data();
  create.face_sources = face_sources.data();
  create.facet_ids = facet_ids.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.periodic_state = &periodic_state;
  create.result = &mesh;
  quotient_cells[0].id = 50009;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
  quotient_cells[0].id = 50001;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  constexpr std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> expected_sizes{
      5, 4, 4, 0, 5, 4, 0, 5, 5, 4, 4, 4, 29, 29, 29, 71, 29, 0, 0};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes == expected_sizes);

  int32_t periodic_identity_bound = 0;
  std::array<int64_t, 6> identity_lengths{};
  PHX_CHECK(phx_mc_tet_mesh_periodic_identity_sizes(
                mesh, &periodic_identity_bound, &identity_lengths[0],
                &identity_lengths[1], &identity_lengths[2], &identity_lengths[3],
                &identity_lengths[4], &identity_lengths[5]) == PHX_MC_OK);
  PHX_CHECK(periodic_identity_bound == 1);
  PHX_CHECK(identity_lengths[0] == sizeof(source_mesh_id) - 1 &&
            identity_lengths[1] == sizeof(source_topology_id) - 1 &&
            identity_lengths[2] == sizeof(lifted_topology_id) - 1 &&
            identity_lengths[3] == sizeof(periodic_topology_id) - 1 &&
            identity_lengths[4] == sizeof(cell_entity_set_id) - 1 &&
            identity_lengths[5] == sizeof(edge_entity_set_id) - 1);
  std::array<char, sizeof(source_mesh_id) - 1> copied_source_mesh_id{};
  std::array<char, sizeof(source_topology_id) - 1> copied_source_topology_id{};
  std::array<char, sizeof(lifted_topology_id) - 1> copied_lifted_topology_id{};
  std::array<char, sizeof(periodic_topology_id) - 1> copied_periodic_topology_id{};
  std::array<char, sizeof(cell_entity_set_id) - 1> copied_cell_entity_set_id{};
  std::array<char, sizeof(edge_entity_set_id) - 1> copied_edge_entity_set_id{};
  PHX_CHECK(phx_mc_tet_mesh_periodic_identity(
                mesh, copied_source_mesh_id.size(), copied_source_mesh_id.data(),
                copied_source_topology_id.size(), copied_source_topology_id.data(),
                copied_lifted_topology_id.size(), copied_lifted_topology_id.data(),
                copied_periodic_topology_id.size(), copied_periodic_topology_id.data(),
                copied_cell_entity_set_id.size(), copied_cell_entity_set_id.data(),
                copied_edge_entity_set_id.size(), copied_edge_entity_set_id.data()) ==
            PHX_MC_OK);
  PHX_CHECK(std::string_view(copied_source_mesh_id.data(), copied_source_mesh_id.size()) ==
            std::string_view(source_mesh_id, sizeof(source_mesh_id) - 1));
  PHX_CHECK(std::string_view(copied_source_topology_id.data(),
                             copied_source_topology_id.size()) ==
            std::string_view(source_topology_id, sizeof(source_topology_id) - 1));
  PHX_CHECK(std::string_view(copied_lifted_topology_id.data(),
                             copied_lifted_topology_id.size()) ==
            std::string_view(lifted_topology_id, sizeof(lifted_topology_id) - 1));
  PHX_CHECK(std::string_view(copied_periodic_topology_id.data(),
                             copied_periodic_topology_id.size()) ==
            std::string_view(periodic_topology_id, sizeof(periodic_topology_id) - 1));
  PHX_CHECK(std::string_view(copied_cell_entity_set_id.data(),
                             copied_cell_entity_set_id.size()) ==
            std::string_view(cell_entity_set_id, sizeof(cell_entity_set_id) - 1));
  PHX_CHECK(std::string_view(copied_edge_entity_set_id.data(),
                             copied_edge_entity_set_id.size()) ==
            std::string_view(edge_entity_set_id, sizeof(edge_entity_set_id) - 1));

  phx_mc_tet_periodic_edit_view periodic_edit{};
  periodic_edit.target_identity_map = &identity_map;
  phx_mc_tet_mesh_edit_request edit{};
  edit.expected_generation = 0;
  edit.kind = PHX_MC_TET_EDIT_RELOCATE;
  edit.periodic = &periodic_edit;
  constexpr uint64_t moved_vertex_id = 500;
  constexpr std::array<double, 3> moved_point{0.26, 0.25, 0.25};
  edit.changed_vertex_count = 1;
  edit.changed_vertex_ids = &moved_vertex_id;
  edit.changed_vertex_points = moved_point.data();
  edit.removed_cell_count = static_cast<int64_t>(cell_ids.size());
  edit.removed_cell_ids = cell_ids.data();
  int32_t edit_status = -1;
  // The target identity names the edited state; the generation guards staleness.
  edit.expected_generation = 1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_STALE_GENERATION);
  edit.expected_generation = 0;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  uint64_t generation = 99;
  int32_t source_identity_bound = -1;
  int64_t ignored_lengths[4]{};
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &source_identity_bound, &ignored_lengths[0],
                &ignored_lengths[1], &ignored_lengths[2], &ignored_lengths[3]) == PHX_MC_OK);
  PHX_CHECK(generation == 0);

  periodic_edit.periodic_member_count = 1;
  periodic_edit.periodic_members = &members[4];
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &source_identity_bound, &ignored_lengths[0],
                &ignored_lengths[1], &ignored_lengths[2], &ignored_lengths[3]) == PHX_MC_OK);
  PHX_CHECK(generation == 1);

  std::array<uint64_t, 5> snapshot_vertex_ids{};
  std::array<double, 15> snapshot_points{};
  std::array<int8_t, 5> witness_strata{};
  std::array<int32_t, 5> witness_entities{};
  std::array<double, 10> witness_parameters{};
  std::array<double, 5> witness_deviations{};
  std::array<int8_t, 5> vertex_dimensions{};
  std::array<double, 5> protection_radii{};
  std::array<double, 5> snapshot_vertex_size_targets{};
  std::array<int8_t, 5> protected_vertices{};
  std::array<int8_t, 5> edge_witness_present{};
  std::array<uint64_t, 10> edge_witness_endpoints{};
  std::array<double, 5> edge_parameters{};
  std::array<int64_t, 6> vertex_ancestry_offsets{};
  std::array<int64_t, 5> cell_ancestry_offsets{};
  std::array<uint64_t, 4> cell_ancestor_ids{};
  std::array<int32_t, 4> cell_ancestry_edit_kinds{};
  std::array<uint64_t, 4> snapshot_cell_ids{};
  std::array<uint64_t, 16> snapshot_cell_vertices{};
  std::array<double, 4> snapshot_cell_size_targets{};
  std::array<int32_t, 4> snapshot_regions{};
  std::array<uint64_t, 4> snapshot_facet_ids{};
  std::array<uint64_t, 12> snapshot_facet_vertices{};
  std::array<int32_t, 4> snapshot_facet_sources{};
  std::array<int8_t, 4> snapshot_facet_orientations{};
  std::array<int8_t, 4> protected_facets{};
  std::array<double, 15> snapshot_source_points{};
  std::array<int32_t, 12> snapshot_source_faces{};
  std::array<int32_t, 4> snapshot_source_face_groups{};
  std::array<int64_t, 4> snapshot_source_face_ids{};
  std::array<double, 4> snapshot_source_face_tolerances{};
  std::array<double, 36> snapshot_source_face_planes{};
  std::array<double, 4> snapshot_source_face_plane_tolerances{};
  std::array<phx_mc_tet_periodic_orbit, 5> snapshot_orbits{};
  std::array<phx_mc_tet_periodic_vertex_member, 5> snapshot_members{};
  std::array<phx_mc_tet_periodic_quotient_cell, 4> snapshot_quotient_cells{};
  std::array<uint64_t, 4> snapshot_quotient_lifted_ids{};
  std::array<phx_mc_tet_periodic_entity_orbit_map, 4> snapshot_entity_maps{};
  std::array<uint64_t, 29> snapshot_entity_lifted_ids{};
  std::array<int32_t, 29> snapshot_lifted_to_quotient{};
  std::array<int32_t, 29> snapshot_orientation_signs{};
  std::array<uint64_t, 29> snapshot_entity_quotient_ids{};
  std::array<int64_t, 29> snapshot_winding_shifts{};
  std::array<int32_t, 71> snapshot_corner_permutations{};
  std::array<int32_t, 29> snapshot_representative_rows{};
  uint64_t snapshot_generation = 0;
  phx_mc_tet_mesh_snapshot_request snapshot{};
  snapshot.vertex_capacity = 5;
  snapshot.cell_capacity = 4;
  snapshot.facet_capacity = 4;
  snapshot.source_point_capacity = 5;
  snapshot.source_face_capacity = 4;
  snapshot.periodic_orbit_capacity = 5;
  snapshot.periodic_member_capacity = 5;
  snapshot.periodic_quotient_cell_capacity = 4;
  snapshot.periodic_quotient_lifted_capacity = 4;
  snapshot.periodic_entity_map_capacity = 4;
  snapshot.periodic_entity_lifted_capacity = 29;
  snapshot.periodic_entity_quotient_capacity = 29;
  snapshot.periodic_entity_winding_capacity = 29;
  snapshot.periodic_entity_corner_capacity = 71;
  snapshot.periodic_entity_representative_capacity = 29;
  snapshot.cell_ancestor_capacity = 4;
  snapshot.vertex_ids = snapshot_vertex_ids.data();
  snapshot.points = snapshot_points.data();
  snapshot.vertex_size_targets = snapshot_vertex_size_targets.data();
  snapshot.witness_strata = witness_strata.data();
  snapshot.witness_entities = witness_entities.data();
  snapshot.witness_parameters = witness_parameters.data();
  snapshot.witness_deviations = witness_deviations.data();
  snapshot.vertex_dimensions = vertex_dimensions.data();
  snapshot.protection_radii = protection_radii.data();
  snapshot.protected_vertices = protected_vertices.data();
  snapshot.vertex_edge_witness_present = edge_witness_present.data();
  snapshot.vertex_edge_witness_endpoints = edge_witness_endpoints.data();
  snapshot.vertex_edge_parameters = edge_parameters.data();
  snapshot.vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  snapshot.periodic_orbits = snapshot_orbits.data();
  snapshot.periodic_members = snapshot_members.data();
  snapshot.periodic_quotient_cells = snapshot_quotient_cells.data();
  snapshot.periodic_quotient_lifted_cell_ids = snapshot_quotient_lifted_ids.data();
  snapshot.periodic_entity_maps = snapshot_entity_maps.data();
  snapshot.periodic_entity_lifted_ids = snapshot_entity_lifted_ids.data();
  snapshot.periodic_entity_lifted_to_quotient = snapshot_lifted_to_quotient.data();
  snapshot.periodic_entity_orientation_signs = snapshot_orientation_signs.data();
  snapshot.periodic_entity_quotient_ids = snapshot_entity_quotient_ids.data();
  snapshot.periodic_entity_winding_shifts = snapshot_winding_shifts.data();
  snapshot.periodic_entity_corner_permutations = snapshot_corner_permutations.data();
  snapshot.periodic_entity_representative_rows = snapshot_representative_rows.data();
  snapshot.cell_ancestry_offsets = cell_ancestry_offsets.data();
  snapshot.cell_ancestor_ids = cell_ancestor_ids.data();
  snapshot.cell_ancestry_edit_kinds = cell_ancestry_edit_kinds.data();
  snapshot.cell_ids = snapshot_cell_ids.data();
  snapshot.cell_vertices = snapshot_cell_vertices.data();
  snapshot.cell_regions = snapshot_regions.data();
  snapshot.cell_size_targets = snapshot_cell_size_targets.data();
  snapshot.facet_ids = snapshot_facet_ids.data();
  snapshot.facet_vertices = snapshot_facet_vertices.data();
  snapshot.facet_sources = snapshot_facet_sources.data();
  snapshot.facet_orientations = snapshot_facet_orientations.data();
  snapshot.protected_facets = protected_facets.data();
  snapshot.source_points = snapshot_source_points.data();
  snapshot.source_faces = snapshot_source_faces.data();
  snapshot.source_face_groups = snapshot_source_face_groups.data();
  snapshot.source_face_ids = snapshot_source_face_ids.data();
  snapshot.source_face_tolerances = snapshot_source_face_tolerances.data();
  snapshot.source_face_planes = snapshot_source_face_planes.data();
  snapshot.source_face_plane_tolerances = snapshot_source_face_plane_tolerances.data();
  snapshot.generation = &snapshot_generation;
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_OK);
  PHX_CHECK(snapshot_generation == 1 && snapshot_vertex_ids[4] == moved_vertex_id);
  PHX_CHECK(snapshot_points[12] == moved_point[0] &&
            snapshot_points[13] == moved_point[1] &&
            snapshot_points[14] == moved_point[2]);
  PHX_CHECK(snapshot_orbits[4].id == orbits[4].id &&
            snapshot_members[4].vertex_id == moved_vertex_id);
  PHX_CHECK(snapshot_quotient_cells[3].id == quotient_cells[3].id &&
            snapshot_quotient_lifted_ids[3] == cell_ids[3]);
  PHX_CHECK(snapshot_entity_lifted_ids == entity_lifted_ids &&
            snapshot_entity_quotient_ids == entity_quotient_ids);
  PHX_CHECK(snapshot_lifted_to_quotient == entity_lifted_to_quotient &&
            snapshot_orientation_signs == entity_orientation_signs &&
            snapshot_winding_shifts == entity_winding_shifts &&
            snapshot_corner_permutations == entity_corner_permutations &&
            snapshot_representative_rows == entity_representative_rows);
  constexpr std::array<int64_t, 4> target_counts{4, 6, 4, 1};
  constexpr std::array<uint64_t, 15> target_lifted_ids{
      101, 211, 307, 409, 11001, 11002, 11003, 11004, 11005, 11006,
      5101, 5102, 5103, 5104, 9005};
  constexpr std::array<uint64_t, 15> target_quotient_ids{
      30001, 30002, 30003, 30004,
      30006, 30007, 30008, 30009, 30010, 30011,
      30016, 30017, 30018, 30019, 50005};
  std::array<int32_t, 15> target_lifted_to_quotient{};
  std::array<int32_t, 15> target_orientation_signs{};
  std::array<int64_t, 15> target_winding_shifts{};
  std::array<int32_t, 32> target_corner_permutations{};
  std::array<int32_t, 15> target_representative_rows{};
  std::array<phx_mc_tet_periodic_entity_orbit_map, 4> target_entity_maps{};
  std::size_t target_lifted_offset = 0;
  std::size_t target_quotient_offset = 0;
  std::size_t target_winding_offset = 0;
  std::size_t target_corner_offset = 0;
  std::size_t target_representative_offset = 0;
  for (std::size_t dimension = 0; dimension < target_entity_maps.size(); ++dimension) {
    phx_mc_tet_periodic_entity_orbit_map& map = target_entity_maps[dimension];
    const std::size_t count = static_cast<std::size_t>(target_counts[dimension]);
    const std::size_t arity = dimension + 1;
    map.dimension = static_cast<int32_t>(dimension);
    map.arity = static_cast<int32_t>(arity);
    map.rank = 1;
    map.lifted_offset = static_cast<int64_t>(target_lifted_offset);
    map.lifted_count = static_cast<int64_t>(count);
    map.quotient_offset = static_cast<int64_t>(target_quotient_offset);
    map.quotient_count = static_cast<int64_t>(count);
    map.winding_offset = static_cast<int64_t>(target_winding_offset);
    map.corner_permutation_offset = static_cast<int64_t>(target_corner_offset);
    map.representative_offset = static_cast<int64_t>(target_representative_offset);
    for (std::size_t row = 0; row < count; ++row) {
      target_lifted_to_quotient[target_lifted_offset + row] = static_cast<int32_t>(row);
      target_orientation_signs[target_lifted_offset + row] = 1;
      target_representative_rows[target_representative_offset + row] = static_cast<int32_t>(row);
      for (std::size_t corner = 0; corner < arity; ++corner) {
        target_corner_permutations[target_corner_offset + row * arity + corner] =
            static_cast<int32_t>(corner);
      }
    }
    target_lifted_offset += count;
    target_quotient_offset += count;
    target_winding_offset += count;
    target_corner_offset += count * arity;
    target_representative_offset += count;
  }
  phx_mc_tet_periodic_identity_map_view target_identity_map = identity_map;
  target_identity_map.entity_map_count = 4;
  target_identity_map.entity_maps = target_entity_maps.data();
  target_identity_map.entity_lifted_count = 15;
  target_identity_map.entity_lifted_ids = target_lifted_ids.data();
  target_identity_map.entity_lifted_to_quotient = target_lifted_to_quotient.data();
  target_identity_map.entity_orientation_signs = target_orientation_signs.data();
  target_identity_map.entity_quotient_count = 15;
  target_identity_map.entity_quotient_ids = target_quotient_ids.data();
  target_identity_map.entity_winding_count = 15;
  target_identity_map.entity_winding_shifts = target_winding_shifts.data();
  target_identity_map.entity_corner_count = 32;
  target_identity_map.entity_corner_permutations = target_corner_permutations.data();
  target_identity_map.entity_representative_count = 15;
  target_identity_map.entity_representative_rows = target_representative_rows.data();

  constexpr std::array<uint64_t, 1> collapse_removed_vertex{500};
  constexpr std::array<uint64_t, 1> collapse_removed_id{500};
  constexpr std::array<uint64_t, 1> collapse_survivor_id{101};
  constexpr std::array<uint64_t, 4> removed_quotient_ids{50001, 50002, 50003, 50004};
  constexpr std::array<uint64_t, 1> replacement_cell_id{9005};
  constexpr std::array<uint64_t, 4> replacement_cell_vertices{101, 211, 307, 409};
  constexpr std::array<int64_t, 2> replacement_ancestry_offsets{0, 4};
  constexpr int32_t replacement_region = 0;
  phx_mc_tet_periodic_quotient_cell merged_quotient_cell{};
  merged_quotient_cell.orbit_id = 50005;
  merged_quotient_cell.id = 50005;
  merged_quotient_cell.representative_vertices[0] = 101;
  merged_quotient_cell.representative_vertices[1] = 211;
  merged_quotient_cell.representative_vertices[2] = 307;
  merged_quotient_cell.representative_vertices[3] = 409;
  merged_quotient_cell.region = 0;
  merged_quotient_cell.lifted_cell_offset = 0;
  merged_quotient_cell.lifted_cell_count = 1;
  constexpr uint64_t replacement_lifted_cell_id = 9005;

  phx_mc_tet_periodic_edit_view collapse_periodic{};
  collapse_periodic.periodic_member_count = static_cast<int64_t>(members.size());
  collapse_periodic.periodic_members = members.data();
  collapse_periodic.removed_periodic_orbit_count = 1;
  collapse_periodic.removed_periodic_orbit_ids = &orbits[4].id;
  collapse_periodic.removed_quotient_cell_count =
      static_cast<int64_t>(removed_quotient_ids.size());
  collapse_periodic.removed_quotient_cell_ids = removed_quotient_ids.data();
  collapse_periodic.replacement_quotient_cell_count = 1;
  collapse_periodic.replacement_quotient_cells = &merged_quotient_cell;
  collapse_periodic.replacement_quotient_lifted_count = 1;
  collapse_periodic.replacement_quotient_lifted_cell_ids = &replacement_lifted_cell_id;
  collapse_periodic.target_identity_map = &target_identity_map;

  phx_mc_tet_mesh_edit_request periodic_collapse{};
  periodic_collapse.expected_generation = 1;
  periodic_collapse.kind = PHX_MC_TET_EDIT_COLLAPSE;
  periodic_collapse.periodic = &collapse_periodic;
  periodic_collapse.collapse_pair_count = 1;
  periodic_collapse.collapse_removed_ids = collapse_removed_id.data();
  periodic_collapse.collapse_survivor_ids = collapse_survivor_id.data();
  periodic_collapse.removed_vertex_count = 1;
  periodic_collapse.removed_vertex_ids = collapse_removed_vertex.data();
  periodic_collapse.removed_cell_count = static_cast<int64_t>(cell_ids.size());
  periodic_collapse.removed_cell_ids = cell_ids.data();
  periodic_collapse.replacement_cell_count = 1;
  periodic_collapse.replacement_cell_ids = replacement_cell_id.data();
  periodic_collapse.replacement_cell_vertices = replacement_cell_vertices.data();
  periodic_collapse.replacement_cell_facet_sources = face_sources.data();
  periodic_collapse.replacement_regions = &replacement_region;
  periodic_collapse.replacement_cell_ancestry_count =
      static_cast<int64_t>(cell_ids.size());
  periodic_collapse.replacement_cell_ancestry_offsets = replacement_ancestry_offsets.data();
  periodic_collapse.replacement_cell_ancestor_ids = cell_ids.data();
  // The collapse produces a new source mesh: its target identity is bound.
  constexpr char collapsed_source_mesh_id[] = "mesh:collapsed";
  target_identity_map.source_mesh_id = collapsed_source_mesh_id;
  target_identity_map.source_mesh_id_length = sizeof(collapsed_source_mesh_id) - 1;
  // A moved survivor is admitted on a periodic lift like any changed vertex
  // and checked exactly at its new carrier: moving 101 off its source facets
  // is invalid geometry, not a refused periodic topology edit.
  constexpr std::array<double, 3> off_facet_survivor{0x1p-4, 0x1p-4, 0x1p-4};
  phx_mc_tet_mesh_edit_request moved_survivor = periodic_collapse;
  moved_survivor.changed_vertex_count = 1;
  moved_survivor.changed_vertex_ids = collapse_survivor_id.data();
  moved_survivor.changed_vertex_points = off_facet_survivor.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &moved_survivor, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_INVALID_GEOMETRY);
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &periodic_collapse, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  std::array<int64_t, 6> rebound_lengths{};
  PHX_CHECK(phx_mc_tet_mesh_periodic_identity_sizes(
                mesh, &periodic_identity_bound, &rebound_lengths[0], &rebound_lengths[1],
                &rebound_lengths[2], &rebound_lengths[3], &rebound_lengths[4],
                &rebound_lengths[5]) == PHX_MC_OK);
  PHX_CHECK(periodic_identity_bound == 1 &&
            rebound_lengths[0] == sizeof(collapsed_source_mesh_id) - 1);
  std::array<char, sizeof(collapsed_source_mesh_id) - 1> rebound_source_mesh_id{};
  PHX_CHECK(phx_mc_tet_mesh_periodic_identity(
                mesh, rebound_source_mesh_id.size(), rebound_source_mesh_id.data(),
                copied_source_topology_id.size(), copied_source_topology_id.data(),
                copied_lifted_topology_id.size(), copied_lifted_topology_id.data(),
                copied_periodic_topology_id.size(), copied_periodic_topology_id.data(),
                copied_cell_entity_set_id.size(), copied_cell_entity_set_id.data(),
                copied_edge_entity_set_id.size(), copied_edge_entity_set_id.data()) ==
            PHX_MC_OK);
  PHX_CHECK(std::string_view(rebound_source_mesh_id.data(), rebound_source_mesh_id.size()) ==
            std::string_view(collapsed_source_mesh_id, sizeof(collapsed_source_mesh_id) - 1));
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> merged_sizes{};
  constexpr std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> expected_merged_sizes{
      4, 1, 4, 0, 5, 4, 0, 4, 4, 1, 1, 4, 15, 15, 15, 32, 15, 0, 4};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, merged_sizes.data()) == PHX_MC_OK);
  PHX_CHECK(merged_sizes == expected_merged_sizes);
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_OK);
  PHX_CHECK(snapshot_generation == 2 && snapshot_vertex_ids[3] == 409);
  PHX_CHECK(snapshot_quotient_cells[0].orbit_id == merged_quotient_cell.orbit_id &&
            snapshot_quotient_cells[0].id == merged_quotient_cell.id &&
            snapshot_quotient_lifted_ids[0] == replacement_lifted_cell_id);
  PHX_CHECK(snapshot_entity_maps[3].lifted_count == 1 &&
            snapshot_entity_maps[3].quotient_count == 1 &&
            snapshot_entity_lifted_ids[14] == replacement_lifted_cell_id &&
            snapshot_entity_quotient_ids[14] == merged_quotient_cell.id);
  PHX_CHECK(snapshot_cell_ids[0] == replacement_cell_id[0] &&
            cell_ancestor_ids == cell_ids &&
            snapshot.cell_ancestry_edit_kinds[0] == PHX_MC_TET_EDIT_COLLAPSE);
  phx_mc_tet_mesh_free(mesh);
}

void persistent_joint_quality_transition_preserves_identity() {
  constexpr std::array<double, 24> points{
      0.020000000035049353, 0.019999999999999997, 0.019999999999999997,
      0.9799999997734662, 0.02, 0.02,
      0.9799999999649507, 0.98, 0.019999999999999997,
      0.019999999999999997, 0.9799999999649507, 0.019999999999999997,
      0.019999999999999997, 0.020000000035049353, 0.98,
      0.98, 0.02, 0.9799999998417303,
      0.9800000000000019, 0.9799999999649525, 0.9800000000000019,
      0.02, 0.98, 0.9799999998417303};
  constexpr std::array<std::int64_t, 13> offsets{
      0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36};
  constexpr std::array<std::int32_t, 36> triangles{
      2, 3, 0, 5, 6, 1, 7, 4, 3, 5, 1, 0,
      6, 5, 4, 6, 7, 2, 6, 2, 1, 4, 0, 3,
      1, 2, 0, 4, 5, 0, 7, 6, 4, 7, 3, 2};
  constexpr std::array<std::int32_t, 12> polygon_facets{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  constexpr std::array<std::int32_t, 24> facet_regions{
      0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
      0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1};
  constexpr std::array<std::int32_t, 24> segments{
      0, 1, 1, 2, 2, 3, 0, 3, 4, 5, 5, 6,
      6, 7, 4, 7, 0, 4, 1, 5, 2, 6, 3, 7};
  phx::mc::PlcInput input;
  input.point_count = 8;
  input.points = points.data();
  input.polygon_count = 12;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 12;
  input.facet_regions = facet_regions.data();
  input.segment_count = 12;
  input.segments = segments.data();
  input.policy = phx::mc::BoundaryPolicy::kFixed;

  phx::mc::PlcRecovery recovered;
  const std::int32_t recovery_status = phx::mc::recover_plc(input, recovered);
  PHX_CHECK(recovery_status == PHX_MC_OK);
  if (recovery_status != PHX_MC_OK) return;
  std::vector<std::int8_t> protected_vertices(recovered.points.size() / 3, 1);
  std::vector<double> vertex_sizes(recovered.points.size() / 3, 0.75);

  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = static_cast<int64_t>(recovered.points.size() / 3);
  create.points = recovered.points.data();
  create.tetrahedron_count = static_cast<int64_t>(recovered.tets.size() / 4);
  create.tetrahedra = recovered.tets.data();
  create.tetrahedron_regions = recovered.tet_regions.data();
  create.face_count = static_cast<int64_t>(recovered.faces.size() / 3);
  create.faces = recovered.faces.data();
  create.face_sources = recovered.face_sources.data();
  create.face_orientations = recovered.face_orientations.data();
  create.segment_count = static_cast<int64_t>(recovered.segments.size() / 2);
  create.segments = recovered.segments.data();
  create.segment_sources = recovered.segment_sources.data();
  create.protection_radii = recovered.protection.data();
  create.protected_vertices = protected_vertices.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  uint64_t generation = 0;
  int32_t identity_bound = -1;
  std::array<int64_t, 4> identity_lengths{};
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &identity_lengths[0],
                &identity_lengths[1], &identity_lengths[2],
                &identity_lengths[3]) == PHX_MC_OK);
  PHX_CHECK(generation == 0 && identity_bound == 0);

  int32_t edit_status = -1;
  int32_t run_status = -1;
  int64_t inserted = 0;
  int64_t refinement_unmet = 0;
  PHX_CHECK(phx_mc_tet_mesh_refine(
                mesh, generation, identity_bound, nullptr, 0, nullptr, 0,
                nullptr, 0, nullptr, 0, 2.0,
                static_cast<int64_t>(vertex_sizes.size()), vertex_sizes.data(),
                &edit_status, &run_status, &inserted,
                &refinement_unmet) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(run_status == PHX_MC_OK || run_status == PHX_MC_UNRESOLVED);

  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> before_sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, before_sizes.data()) == PHX_MC_OK);
  PHX_CHECK(before_sizes[2] == static_cast<int64_t>(recovered.faces.size() / 3));
  uint64_t generation_before_quality = 0;
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation_before_quality, &identity_bound,
                &identity_lengths[0], &identity_lengths[1], &identity_lengths[2],
                &identity_lengths[3]) == PHX_MC_OK);
  PHX_CHECK(identity_bound == 0);

  int64_t collapsed = 0;
  int64_t relocated = 0;
  int64_t improve_unmet = 0;
  double score_quantum = 0.0;
  // No `sizes`: refinement's retained size field gates every improvement edit.
  PHX_CHECK(phx_mc_tet_mesh_improve_quality(
                mesh, generation_before_quality, identity_bound, nullptr, 0,
                nullptr, 0, nullptr, 0, nullptr, 0, 2.0, 10.0, 1.0e-12, 0, 0, nullptr, 0, nullptr,
                &edit_status, &run_status, &inserted, &collapsed, &relocated,
                &improve_unmet, &score_quantum) == PHX_MC_OK);
  PHX_CHECK(score_quantum == 0x1p-20);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(run_status == PHX_MC_OK && improve_unmet == 0);
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> after_sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, after_sizes.data()) == PHX_MC_OK);
  PHX_CHECK(after_sizes[2] == before_sizes[2]);
  uint64_t generation_after_quality = 0;
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation_after_quality, &identity_bound,
                &identity_lengths[0], &identity_lengths[1], &identity_lengths[2],
                &identity_lengths[3]) == PHX_MC_OK);
  PHX_CHECK(generation_after_quality > generation_before_quality &&
            identity_bound == 0);
  int64_t report_count = 0;
  PHX_CHECK(phx_mc_tet_mesh_unmet_count(mesh, &report_count) == PHX_MC_OK);
  PHX_CHECK(report_count == 0);
  phx_mc_tet_mesh_free(mesh);
}

void persistent_collapse_replaces_vertex_star() {
  constexpr std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 0.001, 0.001, 0.001};
  constexpr std::array<int32_t, 16> tetrahedra{
      1, 3, 2, 4, 0, 2, 3, 4, 0, 3, 1, 4, 0, 1, 2, 4};
  constexpr std::array<int32_t, 4> regions{0, 0, 0, 0};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr std::array<uint64_t, 5> vertex_ids{101, 211, 307, 409, 500};
  constexpr std::array<uint64_t, 4> cell_ids{9001, 9002, 9003, 9004};
  constexpr std::array<uint64_t, 4> facet_ids{5101, 5102, 5103, 5104};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 5;
  create.points = points.data();
  create.vertex_ids = vertex_ids.data();
  create.tetrahedron_count = 4;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.cell_ids = cell_ids.data();
  create.face_count = 4;
  create.faces = faces.data();
  create.face_sources = face_sources.data();
  create.facet_ids = facet_ids.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;
  uint64_t generation_before_collapse = 0;
  int32_t identity_bound_before_collapse = -1;
  std::array<int64_t, 4> identity_lengths_before_collapse{};
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation_before_collapse,
                &identity_bound_before_collapse,
                &identity_lengths_before_collapse[0],
                &identity_lengths_before_collapse[1],
                &identity_lengths_before_collapse[2],
                &identity_lengths_before_collapse[3]) == PHX_MC_OK);
  PHX_CHECK(generation_before_collapse == 0 &&
            identity_bound_before_collapse == 0);

  constexpr std::array<uint64_t, 1> removed_vertex_ids{500};
  constexpr std::array<uint64_t, 1> collapse_removed_ids{500};
  constexpr std::array<uint64_t, 1> collapse_survivor_ids{101};
  constexpr std::array<uint64_t, 1> replacement_cell_ids{9005};
  constexpr std::array<uint64_t, 4> replacement_vertices{101, 211, 307, 409};
  constexpr std::array<int32_t, 4> replacement_facet_sources{0, 1, 2, 3};
  constexpr std::array<int32_t, 1> replacement_regions{0};
  constexpr std::array<int64_t, 2> ancestry_offsets{0, 4};
  phx_mc_tet_mesh_edit_request edit{};
  edit.expected_generation = generation_before_collapse;
  edit.kind = PHX_MC_TET_EDIT_COLLAPSE;
  edit.collapse_pair_count = 1;
  edit.collapse_removed_ids = collapse_removed_ids.data();
  edit.collapse_survivor_ids = collapse_survivor_ids.data();
  edit.removed_vertex_count = 1;
  edit.removed_vertex_ids = removed_vertex_ids.data();
  edit.removed_cell_count = 4;
  edit.removed_cell_ids = cell_ids.data();
  edit.replacement_cell_count = 1;
  edit.replacement_cell_ids = replacement_cell_ids.data();
  edit.replacement_cell_vertices = replacement_vertices.data();
  edit.replacement_cell_facet_sources = replacement_facet_sources.data();
  edit.replacement_regions = replacement_regions.data();
  edit.replacement_cell_ancestry_count = 4;
  edit.replacement_cell_ancestry_offsets = ancestry_offsets.data();
  edit.replacement_cell_ancestor_ids = cell_ids.data();
  int32_t edit_status = -1;
  constexpr std::array<uint64_t, 1> mismatched_collapse_ids{501};
  edit.collapse_removed_ids = mismatched_collapse_ids.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) ==
            PHX_MC_INVALID_ARGUMENT);
  edit.collapse_removed_ids = collapse_removed_ids.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  constexpr std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> expected_sizes{
      4, 1, 4, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes == expected_sizes);

  std::array<uint64_t, 4> snapshot_vertex_ids{};
  std::array<double, 12> snapshot_points{};
  std::array<int8_t, 4> witness_strata{};
  std::array<int32_t, 4> witness_entities{};
  std::array<double, 8> witness_parameters{};
  std::array<double, 4> witness_deviations{};
  std::array<int8_t, 4> vertex_dimensions{};
  std::array<double, 4> protection_radii{};
  std::array<double, 4> snapshot_vertex_size_targets{};
  std::array<int8_t, 4> protected_vertices{};
  std::array<int8_t, 4> edge_witness_present{};
  std::array<uint64_t, 8> edge_witness_endpoints{};
  std::array<double, 4> edge_parameters{};
  std::array<int64_t, 5> vertex_ancestry_offsets{};
  std::array<int64_t, 2> cell_ancestry_offsets{};
  std::array<uint64_t, 4> cell_ancestor_ids{};
  std::array<int32_t, 1> cell_ancestry_edit_kinds{};
  std::array<uint64_t, 1> snapshot_cell_ids{};
  std::array<uint64_t, 4> snapshot_cell_vertices{};
  std::array<int32_t, 1> snapshot_regions{};
  std::array<double, 1> snapshot_cell_size_targets{};
  std::array<uint64_t, 4> snapshot_facet_ids{};
  std::array<uint64_t, 12> snapshot_facet_vertices{};
  std::array<int32_t, 4> snapshot_facet_sources{};
  std::array<int8_t, 4> snapshot_facet_orientations{};
  std::array<int8_t, 4> protected_facets{};
  std::array<double, 15> snapshot_source_points{};
  std::array<int32_t, 12> snapshot_source_faces{};
  std::array<int32_t, 4> snapshot_source_face_groups{};
  std::array<int64_t, 4> snapshot_source_face_ids{};
  std::array<double, 4> snapshot_source_face_tolerances{};
  std::array<double, 36> snapshot_source_face_planes{};
  std::array<double, 4> snapshot_source_face_plane_tolerances{};
  uint64_t generation = 99;
  phx_mc_tet_mesh_snapshot_request snapshot{};
  snapshot.vertex_capacity = 4;
  snapshot.cell_capacity = 1;
  snapshot.facet_capacity = 4;
  snapshot.source_point_capacity = 5;
  snapshot.source_face_capacity = 4;
  snapshot.cell_ancestor_capacity = 4;
  snapshot.vertex_ids = snapshot_vertex_ids.data();
  snapshot.points = snapshot_points.data();
  snapshot.vertex_size_targets = snapshot_vertex_size_targets.data();
  snapshot.witness_strata = witness_strata.data();
  snapshot.witness_entities = witness_entities.data();
  snapshot.witness_parameters = witness_parameters.data();
  snapshot.witness_deviations = witness_deviations.data();
  snapshot.vertex_dimensions = vertex_dimensions.data();
  snapshot.protection_radii = protection_radii.data();
  snapshot.protected_vertices = protected_vertices.data();
  snapshot.vertex_edge_witness_present = edge_witness_present.data();
  snapshot.vertex_edge_witness_endpoints = edge_witness_endpoints.data();
  snapshot.vertex_edge_parameters = edge_parameters.data();
  snapshot.vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  snapshot.cell_ancestry_offsets = cell_ancestry_offsets.data();
  snapshot.cell_ancestor_ids = cell_ancestor_ids.data();
  snapshot.cell_ancestry_edit_kinds = cell_ancestry_edit_kinds.data();
  snapshot.cell_ids = snapshot_cell_ids.data();
  snapshot.cell_vertices = snapshot_cell_vertices.data();
  snapshot.cell_regions = snapshot_regions.data();
  snapshot.cell_size_targets = snapshot_cell_size_targets.data();
  snapshot.facet_ids = snapshot_facet_ids.data();
  snapshot.facet_vertices = snapshot_facet_vertices.data();
  snapshot.facet_sources = snapshot_facet_sources.data();
  snapshot.facet_orientations = snapshot_facet_orientations.data();
  snapshot.protected_facets = protected_facets.data();
  snapshot.source_points = snapshot_source_points.data();
  snapshot.source_faces = snapshot_source_faces.data();
  snapshot.source_face_groups = snapshot_source_face_groups.data();
  snapshot.source_face_ids = snapshot_source_face_ids.data();
  snapshot.source_face_tolerances = snapshot_source_face_tolerances.data();
  snapshot.source_face_planes = snapshot_source_face_planes.data();
  snapshot.source_face_plane_tolerances = snapshot_source_face_plane_tolerances.data();
  snapshot.generation = &generation;
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_OK);
  constexpr std::array<int64_t, 2> expected_ancestry_offsets{0, 4};
  constexpr std::array<uint64_t, 4> expected_vertex_ids{101, 211, 307, 409};
  constexpr std::array<double, 12> expected_points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  PHX_CHECK(generation == generation_before_collapse + 1 &&
            snapshot_vertex_ids == expected_vertex_ids);
  PHX_CHECK(snapshot_cell_ids[0] == replacement_cell_ids[0] &&
            snapshot_cell_vertices == replacement_vertices &&
            snapshot_regions[0] == 0);
  PHX_CHECK(cell_ancestry_offsets == expected_ancestry_offsets &&
            cell_ancestor_ids == cell_ids &&
            cell_ancestry_edit_kinds[0] == PHX_MC_TET_EDIT_COLLAPSE);
  PHX_CHECK(snapshot_facet_ids == facet_ids);
  PHX_CHECK(snapshot_points == expected_points);
  phx_mc_tet_mesh_free(mesh);
}

void refinement_preserves_source_region_conflict_refusal() {
  constexpr std::array<double, 24> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 1.0, 0.0, 3.0, 0.0, 1.0};
  constexpr std::array<int32_t, 8> tetrahedra{0, 1, 2, 3, 4, 5, 6, 7};
  constexpr std::array<int32_t, 24> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2,
      5, 7, 6, 4, 6, 7, 4, 7, 5, 4, 5, 6};
  constexpr std::array<int32_t, 2> regions{0, 1};
  constexpr std::array<int32_t, 8> sources{1, 2, 3, 0, 4, 5, 6, 0};
  constexpr std::array<double, 33> source_points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 1.0, 0.0, 3.0, 0.0, 1.0,
      -1.0, -1.0, 0.0, 6.0, -1.0, 0.0, -1.0, 6.0, 0.0};
  constexpr std::array<int32_t, 21> source_faces{
      8, 9, 10, 1, 3, 2, 0, 2, 3, 0, 3, 1, 5, 7, 6, 4, 6, 7, 4, 7, 5};
  constexpr std::array<int32_t, 7> source_groups{0, 1, 2, 3, 4, 5, 6};
  constexpr std::array<int64_t, 7> source_ids{0, 1, 2, 3, 4, 5, 6};
  constexpr std::array<double, 7> source_tolerances{};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 8;
  create.points = points.data();
  create.tetrahedron_count = 2;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.face_count = 8;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.source_point_count = 11;
  create.source_points = source_points.data();
  create.source_face_count = 7;
  create.source_faces = source_faces.data();
  create.source_face_groups = source_groups.data();
  create.source_face_ids = source_ids.data();
  create.source_face_tolerances = source_tolerances.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  // Both components bind the same oriented source side to different regions.
  // Refinement replaces cells, not facets, but must still refuse that conflict.
  int32_t edit_status = -1;
  int32_t run_status = -1;
  int64_t inserted = -1;
  int64_t unmet = -1;
  PHX_CHECK(phx_mc_tet_mesh_refine(
                mesh, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
                0.5, 0, nullptr, &edit_status, &run_status, &inserted, &unmet) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(run_status == PHX_MC_UNRESOLVED);
  PHX_CHECK(inserted == 0 && unmet == 2);
  uint64_t generation = 99;
  int32_t identity_bound = -1;
  int64_t source_length = -1, revision_length = -1, map_length = -1, frame_length = -1;
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &source_length,
                &revision_length, &map_length, &frame_length) == PHX_MC_OK);
  PHX_CHECK(generation == 0);
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 8 && sizes[1] == 2);
  phx_mc_tet_mesh_free(mesh);
}

void persistent_quality_query_is_read_only() {
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr std::array<int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> sources{0, 1, 2, 3};
  constexpr std::array<double, 4> loose_sizes{2.0, 2.0, 2.0, 2.0};
  constexpr std::array<double, 4> target_sizes{0.1, 0.1, 0.1, 0.1};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 4;
  create.points = points.data();
  create.tetrahedron_count = 1;
  create.tetrahedra = tetrahedra.data();
  create.face_count = 4;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;
  int32_t edit_status = -1, run_status = -1;
  int64_t inserted = -1, unmet = -1, quality_count = -1;
  PHX_CHECK(phx_mc_tet_mesh_refine(
                mesh, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
                2.0, 4, loose_sizes.data(), &edit_status, &run_status, &inserted,
                &unmet) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED && run_status == PHX_MC_OK &&
            inserted == 0 && unmet == 0);
  PHX_CHECK(phx_mc_tet_mesh_evaluate_quality(
                mesh, 0, 2.0, 10.0, 0.0, 0, &quality_count,
                nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(quality_count == 0);
  PHX_CHECK(phx_mc_tet_mesh_refine(
                mesh, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
                0.5, 4, target_sizes.data(), &edit_status, &run_status, &inserted,
                &unmet) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  uint64_t generation = 99;
  int32_t identity_bound = -1;
  int64_t source_length = -1, revision_length = -1, map_length = -1, frame_length = -1;
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &source_length,
                &revision_length, &map_length, &frame_length) == PHX_MC_OK);
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  int64_t prior_count = 0;
  PHX_CHECK(phx_mc_tet_mesh_unmet_count(mesh, &prior_count) == PHX_MC_OK);
  PHX_CHECK(prior_count > 0);
  const auto n = static_cast<std::size_t>(prior_count);
  std::vector<uint64_t> prior_cells(n), prior_apexes(n), prior_facets(n);
  std::vector<int32_t> prior_criteria(n), prior_reasons(n);
  std::vector<double> prior_values(n), prior_angles(n), prior_ratios(n);
  PHX_CHECK(phx_mc_tet_mesh_unmet(
                mesh, prior_count, prior_cells.data(), prior_criteria.data(),
                prior_reasons.data(), prior_values.data(), prior_apexes.data(),
                prior_angles.data(), prior_facets.data(), prior_ratios.data()) == PHX_MC_OK);

  std::vector<uint64_t> vertex_ids_before, vertex_ids_after;
  std::vector<int8_t> edge_present_before, edge_present_after;
  std::vector<uint64_t> edge_endpoints_before, edge_endpoints_after;
  std::vector<double> edge_parameters_before, edge_parameters_after;
  uint64_t before_generation = 99, after_generation = 99;
  TetGeometrySnapshot geometry_before, geometry_after;
  capture_vertex_edge_witnesses(mesh, vertex_ids_before, edge_present_before,
                                edge_endpoints_before, edge_parameters_before,
                                before_generation, &geometry_before);

  // A loose shape request still reports the retained, unsatisfied size field.
  quality_count = -1;
  PHX_CHECK(phx_mc_tet_mesh_evaluate_quality(
                mesh, generation, 2.0, 10.0, 0.0, 0, &quality_count,
                nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(quality_count > 0);
  std::vector<uint64_t> quality_cells(static_cast<std::size_t>(quality_count));
  std::vector<int32_t> quality_criteria(static_cast<std::size_t>(quality_count));
  std::vector<int32_t> quality_reasons(static_cast<std::size_t>(quality_count));
  std::vector<double> quality_values(static_cast<std::size_t>(quality_count));
  PHX_CHECK(phx_mc_tet_mesh_evaluate_quality(
                mesh, generation, 2.0, 10.0, 0.0, quality_count, &quality_count,
                quality_cells.data(), quality_criteria.data(), quality_reasons.data(),
                quality_values.data()) == PHX_MC_OK);
  bool saw_size = false;
  for (std::size_t row = 0; row < quality_cells.size(); ++row) {
    PHX_CHECK(quality_reasons[row] == PHX_MC_TET_UNMET_NO_IMPROVEMENT ||
              quality_reasons[row] == PHX_MC_TET_UNMET_NONREPRESENTABLE);
    if (quality_criteria[row] == PHX_MC_TET_UNMET_SIZE) {
      saw_size = true;
      PHX_CHECK(quality_values[row] > 1.0);
    }
  }
  PHX_CHECK(saw_size);

  // Strict shape thresholds independently expose all three geometric criteria.
  int64_t strict_count = 0;
  PHX_CHECK(phx_mc_tet_mesh_evaluate_quality(
                mesh, generation, 0.5, 70.0, 0.9, 0, &strict_count,
                nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(strict_count > 1);
  if (strict_count <= 1) {
    phx_mc_tet_mesh_free(mesh);
    return;
  }
  std::vector<uint64_t> strict_cells(static_cast<std::size_t>(strict_count));
  std::vector<int32_t> strict_criteria(static_cast<std::size_t>(strict_count));
  std::vector<int32_t> strict_reasons(static_cast<std::size_t>(strict_count));
  std::vector<double> strict_values(static_cast<std::size_t>(strict_count));
  std::vector<uint64_t> short_cells(static_cast<std::size_t>(strict_count), 987);
  std::vector<int32_t> short_criteria(static_cast<std::size_t>(strict_count), -7);
  std::vector<int32_t> short_reasons(static_cast<std::size_t>(strict_count), -8);
  std::vector<double> short_values(static_cast<std::size_t>(strict_count), -9.0);
  int64_t short_count = -1;
  PHX_CHECK(phx_mc_tet_mesh_evaluate_quality(
                mesh, generation, 0.5, 70.0, 0.9, strict_count - 1, &short_count,
                short_cells.data(), short_criteria.data(), short_reasons.data(),
                short_values.data()) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(short_count == strict_count);
  PHX_CHECK(std::all_of(short_cells.begin(), short_cells.end(),
                        [](uint64_t value) { return value == 987; }));
  PHX_CHECK(std::all_of(short_criteria.begin(), short_criteria.end(),
                        [](int32_t value) { return value == -7; }));
  PHX_CHECK(std::all_of(short_reasons.begin(), short_reasons.end(),
                        [](int32_t value) { return value == -8; }));
  PHX_CHECK(std::all_of(short_values.begin(), short_values.end(),
                        [](double value) { return value == -9.0; }));
  PHX_CHECK(phx_mc_tet_mesh_evaluate_quality(
                mesh, generation + 1, 0.5, 70.0, 0.9, strict_count, &short_count,
                strict_cells.data(), strict_criteria.data(), strict_reasons.data(),
                strict_values.data()) == PHX_MC_STALE_GENERATION);
  PHX_CHECK(phx_mc_tet_mesh_evaluate_quality(
                mesh, generation, 0.5, 70.0, 0.9, strict_count, &strict_count,
                strict_cells.data(), strict_criteria.data(), strict_reasons.data(),
                strict_values.data()) == PHX_MC_OK);
  std::array<int, 5> criterion_counts{};
  for (std::size_t row = 0; row < strict_cells.size(); ++row) {
    const int32_t criterion = strict_criteria[row];
    PHX_CHECK(strict_reasons[row] == PHX_MC_TET_UNMET_NO_IMPROVEMENT ||
              strict_reasons[row] == PHX_MC_TET_UNMET_NONREPRESENTABLE);
    if (criterion >= 0 && criterion < static_cast<int32_t>(criterion_counts.size())) {
      ++criterion_counts[static_cast<std::size_t>(criterion)];
    }
    if (criterion == PHX_MC_TET_UNMET_RADIUS_EDGE) PHX_CHECK(strict_values[row] > 0.5);
    if (criterion == PHX_MC_TET_UNMET_SIZE) PHX_CHECK(strict_values[row] > 1.0);
    if (criterion == PHX_MC_TET_UNMET_DIHEDRAL) PHX_CHECK(strict_values[row] < 70.0);
    if (criterion == PHX_MC_TET_UNMET_VALIDITY) PHX_CHECK(strict_values[row] <= 0.0);
  }
  PHX_CHECK(criterion_counts[PHX_MC_TET_UNMET_RADIUS_EDGE] > 0);
  PHX_CHECK(criterion_counts[PHX_MC_TET_UNMET_SIZE] > 0);
  PHX_CHECK(criterion_counts[PHX_MC_TET_UNMET_DIHEDRAL] > 0);
  PHX_CHECK(criterion_counts[PHX_MC_TET_UNMET_VALIDITY] > 0);

  int64_t after_count = -1;
  PHX_CHECK(phx_mc_tet_mesh_unmet_count(mesh, &after_count) == PHX_MC_OK);
  PHX_CHECK(after_count == prior_count);
  std::vector<uint64_t> after_cells(n), after_apexes(n), after_facets(n);
  std::vector<int32_t> after_criteria(n), after_reasons(n);
  std::vector<double> after_values(n), after_angles(n), after_ratios(n);
  PHX_CHECK(phx_mc_tet_mesh_unmet(
                mesh, prior_count, after_cells.data(), after_criteria.data(),
                after_reasons.data(), after_values.data(), after_apexes.data(),
                after_angles.data(), after_facets.data(), after_ratios.data()) == PHX_MC_OK);
  PHX_CHECK(after_cells == prior_cells && after_criteria == prior_criteria &&
            after_reasons == prior_reasons && after_values == prior_values &&
            after_apexes == prior_apexes && after_facets == prior_facets);
  for (std::size_t row = 0; row < n; ++row) {
    PHX_CHECK(after_angles[row] == prior_angles[row] ||
              (std::isnan(after_angles[row]) && std::isnan(prior_angles[row])));
    PHX_CHECK(after_ratios[row] == prior_ratios[row] ||
              (std::isnan(after_ratios[row]) && std::isnan(prior_ratios[row])));
  }
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> after_sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, after_sizes.data()) == PHX_MC_OK);
  capture_vertex_edge_witnesses(mesh, vertex_ids_after, edge_present_after,
                                edge_endpoints_after, edge_parameters_after,
                                after_generation, &geometry_after);
  PHX_CHECK(after_sizes == sizes && before_generation == generation &&
            after_generation == generation && vertex_ids_after == vertex_ids_before &&
            edge_present_after == edge_present_before &&
            edge_endpoints_after == edge_endpoints_before &&
            edge_parameters_after == edge_parameters_before);
  PHX_CHECK(geometry_before.points == geometry_after.points &&
            geometry_before.cell_ids == geometry_after.cell_ids &&
            geometry_before.cell_vertices == geometry_after.cell_vertices);
  phx_mc_tet_mesh_free(mesh);
}

void retriangulation_rejects_undeclared_region_jumps() {
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr std::array<int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<int32_t, 1> initial_regions{0};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> sources{0, 1, 2, 3};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 4;
  create.points = points.data();
  create.tetrahedron_count = 1;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = initial_regions.data();
  create.face_count = 4;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  constexpr uint64_t removed_cell = 0;
  constexpr uint64_t new_vertex = 20;
  constexpr std::array<double, 3> center{0.25, 0.25, 0.25};
  constexpr std::array<int64_t, 2> vertex_ancestry_offsets{0, 4};
  constexpr std::array<uint64_t, 4> vertex_ancestor_ids{0, 1, 2, 3};
  constexpr std::array<int64_t, 12> vertex_ancestor_winding{};
  constexpr std::array<double, 4> vertex_ancestry_weights{0.25, 0.25, 0.25, 0.25};
  constexpr std::array<uint64_t, 4> replacement_cell_ids{10, 11, 12, 13};
  constexpr std::array<uint64_t, 16> replacement_vertices{
      1, 3, 2, 20, 0, 2, 3, 20, 0, 3, 1, 20, 0, 1, 2, 20};
  constexpr std::array<int32_t, 16> replacement_sources{
      -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3};
  std::array<int32_t, 4> replacement_regions{1, 0, 0, 0};
  constexpr std::array<int64_t, 5> cell_ancestry_offsets{0, 1, 2, 3, 4};
  constexpr std::array<uint64_t, 4> cell_ancestor_ids{0, 0, 0, 0};
  phx_mc_tet_mesh_edit_request edit{};
  edit.expected_generation = 0;
  edit.kind = PHX_MC_TET_EDIT_RETRIANGULATE;
  edit.removed_cell_count = 1;
  edit.removed_cell_ids = &removed_cell;
  edit.changed_vertex_count = 1;
  edit.changed_vertex_ids = &new_vertex;
  edit.changed_vertex_points = center.data();
  edit.changed_vertex_ancestry_count = 4;
  edit.changed_vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  edit.changed_vertex_ancestor_ids = vertex_ancestor_ids.data();
  edit.changed_vertex_ancestor_winding = vertex_ancestor_winding.data();
  edit.changed_vertex_ancestry_weights = vertex_ancestry_weights.data();
  edit.replacement_cell_count = 4;
  edit.replacement_cell_ids = replacement_cell_ids.data();
  edit.replacement_cell_vertices = replacement_vertices.data();
  edit.replacement_cell_facet_sources = replacement_sources.data();
  edit.replacement_regions = replacement_regions.data();
  edit.replacement_cell_ancestry_count = 4;
  edit.replacement_cell_ancestry_offsets = cell_ancestry_offsets.data();
  edit.replacement_cell_ancestor_ids = cell_ancestor_ids.data();
  int32_t edit_status = -1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  uint64_t generation = 99;
  int32_t identity_bound = -1;
  int64_t source_length = -1, revision_length = -1, map_length = -1, frame_length = -1;
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &source_length,
                &revision_length, &map_length, &frame_length) == PHX_MC_OK);
  PHX_CHECK(generation == 0);
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 4 && sizes[1] == 1);

  replacement_regions = {0, 0, 0, 0};
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &source_length,
                &revision_length, &map_length, &frame_length) == PHX_MC_OK);
  PHX_CHECK(generation == 1);
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 5 && sizes[1] == 4);
  phx_mc_tet_mesh_free(mesh);
}

void retriangulation_rejects_closed_interface_region_relabel() {
  constexpr std::array<double, 24> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      0.5, 0.5, 0.5, -0.25, 0.25, 0.25, 0.25, -0.25, 0.25, 0.25, 0.25, -0.25};
  constexpr std::array<int32_t, 20> tetrahedra{
      0, 1, 2, 3, 1, 2, 3, 4, 0, 3, 2, 5, 0, 1, 3, 6, 0, 2, 1, 7};
  constexpr std::array<int32_t, 5> initial_regions{1, 0, 0, 0, 0};
  constexpr std::array<int32_t, 48> faces{
      1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1,
      2, 4, 3, 1, 3, 4, 1, 4, 2,
      3, 5, 2, 0, 2, 5, 0, 5, 3,
      1, 6, 3, 0, 3, 6, 0, 6, 1,
      2, 7, 1, 0, 1, 7, 0, 7, 2};
  constexpr std::array<int32_t, 16> sources{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 8;
  create.points = points.data();
  create.tetrahedron_count = 5;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = initial_regions.data();
  create.face_count = 16;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  constexpr uint64_t inserted_vertex = 8;
  constexpr std::array<double, 3> center{0.25, 0.25, 0.25};
  constexpr std::array<int64_t, 2> vertex_ancestry_offsets{0, 4};
  constexpr std::array<uint64_t, 4> vertex_ancestor_ids{0, 1, 2, 3};
  constexpr std::array<int64_t, 12> vertex_ancestor_winding{};
  constexpr std::array<double, 4> vertex_ancestry_weights{0.25, 0.25, 0.25, 0.25};
  constexpr std::array<uint64_t, 5> removed_cells{0, 1, 2, 3, 4};
  constexpr std::array<uint64_t, 8> replacement_cell_ids{10, 11, 12, 13, 14, 15, 16, 17};
  constexpr std::array<uint64_t, 32> replacement_vertices{
      1, 3, 2, 8, 0, 2, 3, 8, 0, 3, 1, 8, 0, 1, 2, 8,
      1, 2, 3, 4, 0, 3, 2, 5, 0, 1, 3, 6, 0, 2, 1, 7};
  constexpr std::array<int32_t, 32> replacement_sources{
      -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3,
      4, 5, 6, 0, 7, 8, 9, 1, 10, 11, 12, 2, 13, 14, 15, 3};
  std::array<int32_t, 8> replacement_regions{2, 2, 2, 2, 0, 0, 0, 0};
  constexpr std::array<int64_t, 9> cell_ancestry_offsets{0, 1, 2, 3, 4, 5, 6, 7, 8};
  constexpr std::array<uint64_t, 8> cell_ancestor_ids{0, 0, 0, 0, 1, 2, 3, 4};
  phx_mc_tet_mesh_edit_request edit{};
  edit.expected_generation = 0;
  edit.kind = PHX_MC_TET_EDIT_RETRIANGULATE;
  edit.removed_cell_count = 5;
  edit.removed_cell_ids = removed_cells.data();
  edit.changed_vertex_count = 1;
  edit.changed_vertex_ids = &inserted_vertex;
  edit.changed_vertex_points = center.data();
  edit.changed_vertex_ancestry_count = 4;
  edit.changed_vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  edit.changed_vertex_ancestor_ids = vertex_ancestor_ids.data();
  edit.changed_vertex_ancestor_winding = vertex_ancestor_winding.data();
  edit.changed_vertex_ancestry_weights = vertex_ancestry_weights.data();
  edit.replacement_cell_count = 8;
  edit.replacement_cell_ids = replacement_cell_ids.data();
  edit.replacement_cell_vertices = replacement_vertices.data();
  edit.replacement_cell_facet_sources = replacement_sources.data();
  edit.replacement_regions = replacement_regions.data();
  edit.replacement_cell_ancestry_count = 8;
  edit.replacement_cell_ancestry_offsets = cell_ancestry_offsets.data();
  edit.replacement_cell_ancestor_ids = cell_ancestor_ids.data();
  int32_t edit_status = -1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  uint64_t generation = 99;
  int32_t identity_bound = -1;
  int64_t source_length = -1, revision_length = -1, map_length = -1, frame_length = -1;
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &source_length,
                &revision_length, &map_length, &frame_length) == PHX_MC_OK);
  PHX_CHECK(generation == 0);
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 8 && sizes[1] == 5 && sizes[2] == 16);

  replacement_regions = {1, 1, 1, 1, 0, 0, 0, 0};
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &source_length,
                &revision_length, &map_length, &frame_length) == PHX_MC_OK);
  PHX_CHECK(generation == 1);
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 9 && sizes[1] == 8 && sizes[2] == 16);
  phx_mc_tet_mesh_free(mesh);
}

void grouped_orthogonal_source_faces_validate_each_contributor() {
  constexpr std::array<double, 24> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0};
  constexpr std::array<int32_t, 24> tetrahedra{
      0, 1, 2, 6, 0, 2, 3, 6, 0, 3, 7, 6,
      0, 7, 4, 6, 0, 4, 5, 6, 0, 5, 1, 6};
  constexpr std::array<int32_t, 6> regions{0, 0, 0, 0, 0, 0};
  constexpr std::array<int32_t, 36> faces{
      0, 1, 2, 0, 2, 3, 1, 6, 2, 2, 6, 3,
      0, 3, 7, 3, 6, 7, 0, 7, 4, 7, 6, 4,
      0, 4, 5, 4, 6, 5, 0, 5, 1, 5, 6, 1};
  constexpr std::array<int32_t, 12> sources{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // One authored facet ID intentionally spans all six orthogonal cube sides.
  constexpr std::array<int64_t, 12> source_ids{
      100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
  constexpr std::array<double, 12> source_tolerances{};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 8;
  create.points = points.data();
  create.tetrahedron_count = 6;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.face_count = 12;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.source_point_count = 8;
  create.source_points = points.data();
  create.source_face_count = 12;
  create.source_faces = faces.data();
  create.source_face_groups = sources.data();
  create.source_face_ids = source_ids.data();
  create.source_face_tolerances = source_tolerances.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  constexpr uint64_t inserted_vertex = 8;
  constexpr std::array<double, 3> center{0.75, 0.5, 0.25};
  constexpr std::array<int64_t, 2> vertex_ancestry_offsets{0, 4};
  constexpr std::array<uint64_t, 4> vertex_ancestor_ids{0, 1, 2, 6};
  constexpr std::array<int64_t, 12> vertex_ancestor_winding{};
  constexpr std::array<double, 4> vertex_ancestry_weights{0.25, 0.25, 0.25, 0.25};
  constexpr std::array<uint64_t, 6> removed_cells{0, 1, 2, 3, 4, 5};
  constexpr std::array<int64_t, 10> cell_ancestry_offsets{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr std::array<uint64_t, 9> cell_ancestor_ids{0, 0, 0, 0, 1, 2, 3, 4, 5};
  constexpr std::array<uint64_t, 9> replacement_cell_ids{
      20, 21, 22, 23, 24, 25, 26, 27, 28};
  constexpr std::array<uint64_t, 36> replacement_vertices{
      1, 6, 2, 8, 0, 2, 6, 8, 0, 6, 1, 8, 0, 1, 2, 8,
      0, 2, 3, 6, 0, 3, 7, 6, 0, 7, 4, 6, 0, 4, 5, 6, 0, 5, 1, 6};
  constexpr std::array<int32_t, 36> replacement_sources{
      -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,
      0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0};
  constexpr std::array<int32_t, 9> replacement_regions{0, 0, 0, 0, 0, 0, 0, 0, 0};
  constexpr std::array<uint64_t, 12> removed_facets{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  constexpr std::array<uint64_t, 12> replacement_facets{
      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  constexpr std::array<int8_t, 12> replacement_orientations{
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::array<uint64_t, 36> replacement_face_vertices{};
  for (std::size_t index = 0; index < faces.size(); ++index) {
    replacement_face_vertices[index] = static_cast<uint64_t>(faces[index]);
  }
  phx_mc_tet_mesh_edit_request edit{};
  edit.expected_generation = 0;
  edit.kind = PHX_MC_TET_EDIT_RETRIANGULATE;
  edit.removed_cell_count = 6;
  edit.removed_cell_ids = removed_cells.data();
  edit.changed_vertex_count = 1;
  edit.changed_vertex_ids = &inserted_vertex;
  edit.changed_vertex_points = center.data();
  edit.changed_vertex_ancestry_count = 4;
  edit.changed_vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  edit.changed_vertex_ancestor_ids = vertex_ancestor_ids.data();
  edit.changed_vertex_ancestor_winding = vertex_ancestor_winding.data();
  edit.changed_vertex_ancestry_weights = vertex_ancestry_weights.data();
  edit.replacement_cell_count = 9;
  edit.replacement_cell_ids = replacement_cell_ids.data();
  edit.replacement_cell_vertices = replacement_vertices.data();
  edit.replacement_cell_facet_sources = replacement_sources.data();
  edit.replacement_regions = replacement_regions.data();
  edit.replacement_cell_ancestry_count = 9;
  edit.replacement_cell_ancestry_offsets = cell_ancestry_offsets.data();
  edit.replacement_cell_ancestor_ids = cell_ancestor_ids.data();
  edit.removed_facet_count = 12;
  edit.removed_facet_ids = removed_facets.data();
  edit.replacement_facet_count = 12;
  edit.replacement_facet_ids = replacement_facets.data();
  edit.replacement_facet_vertices = replacement_face_vertices.data();
  edit.replacement_facet_sources = sources.data();
  edit.replacement_facet_orientations = replacement_orientations.data();
  int32_t edit_status = -1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  phx_mc_tet_mesh_free(mesh);
}

void relocation_preserves_or_clears_edge_witnesses_explicitly() {
  constexpr std::array<double, 18> points{
      0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
      0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
  constexpr std::array<int32_t, 16> tetrahedra{
      0, 1, 2, 3, 0, 1, 3, 4, 0, 1, 4, 5, 0, 1, 5, 2};
  constexpr std::array<int32_t, 4> regions{0, 0, 0, 0};
  constexpr std::array<int32_t, 24> faces{
      1, 3, 2, 0, 2, 3, 1, 4, 3, 0, 3, 4,
      1, 5, 4, 0, 4, 5, 1, 2, 5, 0, 5, 2};
  constexpr std::array<int32_t, 8> sources{0, 1, 2, 3, 4, 5, 6, 7};
  constexpr std::array<uint64_t, 6> vertex_ids{101, 211, 307, 409, 500, 600};
  constexpr std::array<uint64_t, 4> parent_cells{9001, 9002, 9003, 9004};
  constexpr std::array<int8_t, 6> protected_vertices{0, 0, 0, 0, 0, 0};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 6;
  create.points = points.data();
  create.vertex_ids = vertex_ids.data();
  create.tetrahedron_count = 4;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.cell_ids = parent_cells.data();
  create.face_count = 8;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.protected_vertices = protected_vertices.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  constexpr std::array<uint64_t, 2> split_edge{101, 211};
  constexpr uint64_t inserted_vertex = 8001;
  constexpr std::array<double, 3> inserted_point{0.0, 0.0, 0.0};
  constexpr int8_t has_edge_witness = 1;
  constexpr std::array<uint64_t, 2> witness_edge{101, 211};
  constexpr double edge_parameter = 0.5;
  constexpr std::array<int64_t, 2> vertex_ancestry_offsets{0, 2};
  constexpr std::array<uint64_t, 2> vertex_ancestor_ids{101, 211};
  constexpr std::array<int64_t, 6> vertex_ancestor_winding{};
  constexpr std::array<double, 2> vertex_ancestry_weights{0.5, 0.5};
  constexpr std::array<uint64_t, 8> split_cell_ids{
      9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108};
  constexpr std::array<uint64_t, 32> split_vertices{
      101, 8001, 307, 409, 8001, 211, 307, 409,
      101, 8001, 409, 500, 8001, 211, 409, 500,
      101, 8001, 500, 600, 8001, 211, 500, 600,
      101, 8001, 600, 307, 8001, 211, 600, 307};
  constexpr std::array<int32_t, 32> split_sources{
      -1, 1, -1, -1, 0, -1, -1, -1,
      -1, 3, -1, -1, 2, -1, -1, -1,
      -1, 5, -1, -1, 4, -1, -1, -1,
      -1, 7, -1, -1, 6, -1, -1, -1};
  constexpr std::array<int32_t, 8> split_regions{0, 0, 0, 0, 0, 0, 0, 0};
  constexpr std::array<int64_t, 9> cell_ancestry_offsets{0, 1, 2, 3, 4, 5, 6, 7, 8};
  constexpr std::array<uint64_t, 8> cell_ancestor_ids{
      9001, 9001, 9002, 9002, 9003, 9003, 9004, 9004};
  phx_mc_tet_mesh_edit_request split{};
  split.expected_generation = 0;
  split.kind = PHX_MC_TET_EDIT_SPLIT;
  split.split_edge_count = 1;
  split.split_edges = split_edge.data();
  split.removed_cell_count = 4;
  split.removed_cell_ids = parent_cells.data();
  split.changed_vertex_count = 1;
  split.changed_vertex_ids = &inserted_vertex;
  split.changed_vertex_points = inserted_point.data();
  split.changed_edge_witness_present = &has_edge_witness;
  split.changed_edge_witness_endpoints = witness_edge.data();
  split.changed_edge_parameters = &edge_parameter;
  split.changed_vertex_ancestry_count = 2;
  split.changed_vertex_ancestry_offsets = vertex_ancestry_offsets.data();
  split.changed_vertex_ancestor_ids = vertex_ancestor_ids.data();
  split.changed_vertex_ancestor_winding = vertex_ancestor_winding.data();
  split.changed_vertex_ancestry_weights = vertex_ancestry_weights.data();
  split.replacement_cell_count = 8;
  split.replacement_cell_ids = split_cell_ids.data();
  split.replacement_cell_vertices = split_vertices.data();
  split.replacement_cell_facet_sources = split_sources.data();
  split.replacement_regions = split_regions.data();
  split.replacement_cell_ancestry_count = 8;
  split.replacement_cell_ancestry_offsets = cell_ancestry_offsets.data();
  split.replacement_cell_ancestor_ids = cell_ancestor_ids.data();
  int32_t edit_status = -1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &split, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);

  std::vector<uint64_t> snapshot_vertex_ids;
  std::vector<int8_t> edge_present;
  std::vector<uint64_t> edge_endpoints;
  std::vector<double> edge_parameters;
  uint64_t generation = 0;
  capture_vertex_edge_witnesses(mesh, snapshot_vertex_ids, edge_present,
                                edge_endpoints, edge_parameters, generation);
  const std::size_t inserted_row = snapshot_vertex_ids.size() - 1;
  PHX_CHECK(generation == 1 && snapshot_vertex_ids.back() == inserted_vertex);
  PHX_CHECK(edge_present[inserted_row] == 1 &&
            edge_endpoints[2 * inserted_row] == 101 &&
            edge_endpoints[2 * inserted_row + 1] == 211 &&
            edge_parameters[inserted_row] == 0.5);

  constexpr std::array<double, 3> unchanged_point{0.0, 0.0, 0.0};
  constexpr std::array<int8_t, 1> clear_witness{0};
  constexpr std::array<uint64_t, 2> cleared_endpoints{};
  constexpr std::array<double, 1> cleared_parameter{0.0};
  phx_mc_tet_mesh_edit_request relocation{};
  relocation.expected_generation = 0;
  relocation.kind = PHX_MC_TET_EDIT_RELOCATE;
  relocation.removed_cell_count = 8;
  relocation.removed_cell_ids = split_cell_ids.data();
  relocation.changed_vertex_count = 1;
  relocation.changed_vertex_ids = &inserted_vertex;
  relocation.changed_vertex_points = unchanged_point.data();
  relocation.changed_edge_witness_present = clear_witness.data();
  relocation.changed_edge_witness_endpoints = cleared_endpoints.data();
  relocation.changed_edge_parameters = cleared_parameter.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &relocation, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_STALE_GENERATION);
  capture_vertex_edge_witnesses(mesh, snapshot_vertex_ids, edge_present,
                                edge_endpoints, edge_parameters, generation);
  PHX_CHECK(generation == 1 && edge_present[inserted_row] == 1 &&
            edge_parameters[inserted_row] == 0.5);

  relocation.expected_generation = 1;
  relocation.changed_edge_witness_present = nullptr;
  relocation.changed_edge_witness_endpoints = nullptr;
  relocation.changed_edge_parameters = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &relocation, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  capture_vertex_edge_witnesses(mesh, snapshot_vertex_ids, edge_present,
                                edge_endpoints, edge_parameters, generation);
  PHX_CHECK(generation == 2 && edge_present[inserted_row] == 1 &&
            edge_parameters[inserted_row] == 0.5);

  relocation.expected_generation = 2;
  relocation.changed_edge_witness_present = clear_witness.data();
  relocation.changed_edge_witness_endpoints = cleared_endpoints.data();
  relocation.changed_edge_parameters = cleared_parameter.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &relocation, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  capture_vertex_edge_witnesses(mesh, snapshot_vertex_ids, edge_present,
                                edge_endpoints, edge_parameters, generation);
  PHX_CHECK(generation == 3 && edge_present[inserted_row] == 0 &&
            edge_parameters[inserted_row] == 0.0);
  PHX_CHECK(edge_endpoints[2 * inserted_row] == 0 &&
            edge_endpoints[2 * inserted_row + 1] == 0);
  phx_mc_tet_mesh_free(mesh);
}

void replacement_facet_requires_its_own_source_geometry() {
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr std::array<int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> sources{0, 1, 2, 3};
  constexpr std::array<int32_t, 1> regions{0};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 4;
  create.points = points.data();
  create.tetrahedron_count = 1;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.face_count = 4;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  constexpr std::array<uint64_t, 1> removed_cells{0};
  constexpr std::array<uint64_t, 1> replacement_cells{1};
  constexpr std::array<uint64_t, 4> replacement_vertices{0, 1, 2, 3};
  std::array<int32_t, 4> replacement_sources{1, 1, 2, 3};
  constexpr std::array<int64_t, 2> ancestor_offsets{0, 1};
  constexpr std::array<uint64_t, 1> removed_facets{0};
  constexpr std::array<uint64_t, 1> replacement_facets{4};
  constexpr std::array<uint64_t, 3> replacement_face{1, 3, 2};
  std::array<int32_t, 1> replacement_face_source{1};
  phx_mc_tet_mesh_edit_request edit{};
  edit.kind = PHX_MC_TET_EDIT_RECONNECT;
  edit.removed_cell_count = 1;
  edit.removed_cell_ids = removed_cells.data();
  edit.replacement_cell_count = 1;
  edit.replacement_cell_ids = replacement_cells.data();
  edit.replacement_cell_vertices = replacement_vertices.data();
  edit.replacement_cell_facet_sources = replacement_sources.data();
  edit.replacement_regions = regions.data();
  edit.replacement_cell_ancestry_count = 1;
  edit.replacement_cell_ancestry_offsets = ancestor_offsets.data();
  edit.replacement_cell_ancestor_ids = removed_cells.data();
  edit.removed_facet_count = 1;
  edit.removed_facet_ids = removed_facets.data();
  edit.replacement_facet_count = 1;
  edit.replacement_facet_ids = replacement_facets.data();
  edit.replacement_facet_vertices = replacement_face.data();
  edit.replacement_facet_sources = replacement_face_source.data();
  int32_t edit_status = -1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_INVALID_GEOMETRY);

  // The rejected edit must not consume IDs or advance the generation.
  replacement_sources[0] = 0;
  replacement_face_source[0] = 0;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  phx_mc_tet_mesh_free(mesh);
}

void split_edge_preserves_exact_ghost_boundary_coverage() {
  using namespace phx::mc;
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0};
  constexpr std::array<std::int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 1> regions{0};
  constexpr std::array<std::int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<std::int32_t, 4> face_sources{0, 0, 0, 0};
  constexpr std::array<std::int8_t, 4> face_orientations{1, 1, 1, 1};
  constexpr std::array<std::int32_t, 4> source_face_groups{0, 0, 0, 0};
  constexpr std::array<std::int64_t, 4> source_face_ids{300, 301, 302, 303};
  constexpr std::array<double, 4> source_face_tolerances{};
  constexpr std::array<std::int32_t, 0> no_segments{};
  constexpr std::array<std::int64_t, 0> no_segment_ids{};
  constexpr std::array<double, 0> no_segment_tolerances{};
  constexpr std::array<SourceWitness, 0> no_witnesses{};
  const SourceComplexView source{
      std::span<const double>(points),
      std::span<const std::int32_t>(faces),
      std::span<const std::int32_t>(source_face_groups),
      std::span<const std::int64_t>(source_face_ids),
      std::span<const double>(source_face_tolerances),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int64_t>(no_segment_ids),
      std::span<const double>(no_segment_tolerances),
      std::span<const SourceWitness>(no_witnesses)};

  const auto transact = [&](int variant, TetMeshSnapshot* snapshot) {
    TetMesh mesh(BoundaryPolicy::kConforming);
    const std::int32_t build_status = mesh.build(
        points, tetrahedra, regions, faces, face_sources, face_orientations,
        no_segments, no_segments, std::span<const double>{}, &source);
    PHX_CHECK(build_status == PHX_MC_OK);
    if (build_status != PHX_MC_OK) return TetEditStatus::kInvalidGeometry;
    constexpr std::array<std::uint64_t, 4> vertex_ids{0, 1, 2, 3};
    constexpr std::array<std::uint64_t, 1> cell_ids{100};
    constexpr std::array<std::uint64_t, 4> facet_ids{200, 201, 202, 203};
    PHX_CHECK(mesh.assign_initial_ids(vertex_ids, cell_ids, facet_ids,
                                      std::span<const std::uint64_t>{}) ==
              PHX_MC_OK);

    TetEditVertex midpoint;
    midpoint.id = 4;
    midpoint.carrier = {0.5, 0.5, 0.0};
    midpoint.witness.stratum = SourceStratum::kFacet;
    midpoint.witness.entity = 3;
    midpoint.witness.parameters[0] = 0.5;
    midpoint.witness.parameters[1] = 0.0;
    midpoint.witness.deviation = 0.0;
    midpoint.ancestry = {{0, {}, 0.5}, {1, {}, 0.5}};

    TetEditCell first_child;
    first_child.id = 101;
    first_child.vertices = {0, 4, 2, 3};
    first_child.facet_sources = {kNoConstraint, 0, 0, 0};
    first_child.region = 0;
    first_child.ancestor_cell_ids = {100};
    TetEditCell second_child;
    second_child.id = 102;
    second_child.vertices = {4, 1, 2, 3};
    second_child.facet_sources = {0, kNoConstraint, 0, 0};
    second_child.region = 0;
    second_child.ancestor_cell_ids = {100};

    std::vector<TetEditFacet> replacement_facets(4);
    replacement_facets[0] = {204, {1, 4, 3}, 0, 1, false};
    replacement_facets[1] = {205, {4, 0, 3}, 0, 1, false};
    replacement_facets[2] = {206, {0, 4, 2}, 0, 1, false};
    replacement_facets[3] = {207, {4, 1, 2}, 0, 1, false};
    if (variant == 1) replacement_facets.pop_back();
    if (variant == 2) {
      std::swap(replacement_facets[0].vertices[1],
                replacement_facets[0].vertices[2]);
    }
    if (variant == 3) {
      replacement_facets.push_back(
          {208, replacement_facets[0].vertices, 0, 1, false});
    }

    TetEdit operation;
    operation.kind = TetEditKind::kSplit;
    operation.split_edges.push_back({0, 1});
    operation.removed_cell_ids.push_back(100);
    operation.changed_vertices.push_back(midpoint);
    operation.replacement_cells = {first_child, second_child};
    operation.removed_facet_ids = {202, 203};
    operation.replacement_facets = std::move(replacement_facets);
    TetEditBatch batch;
    batch.expected_generation = mesh.generation();
    batch.source_revision = mesh.source_revision();
    batch.frame_id = mesh.frame_id();
    batch.identity_bound = mesh.identity_bound();
    batch.operations.push_back(std::move(operation));
    batch.source_id = mesh.source_id();
    batch.source_map_id = mesh.source_map_id();
    batch.periodic_identity = mesh.periodic_identity();
    std::vector<TetEditResult> results;
    const TetEditStatus batch_status = mesh.apply_edit_batch(batch, results);
    if (batch_status != TetEditStatus::kApplied) return batch_status;
    if (results.size() != 1) return TetEditStatus::kRejectedTopology;
    if (snapshot != nullptr && results.front().status == TetEditStatus::kApplied) {
      PHX_CHECK(mesh.snapshot(*snapshot) == TetEditStatus::kApplied);
    }
    return results.front().status;
  };

  TetMeshSnapshot snapshot;
  PHX_CHECK(transact(0, &snapshot) == TetEditStatus::kApplied);
  PHX_CHECK(snapshot.vertices.size() == 5);
  PHX_CHECK(snapshot.cells.size() == 2);
  PHX_CHECK(snapshot.constrained_facets.size() == 6);
  PHX_CHECK(snapshot.constrained_segments.empty());
  for (const TetSnapshotCell& cell : snapshot.cells) {
    PHX_CHECK(cell.region == 0);
    PHX_CHECK(cell.ancestor_cell_ids == std::vector<std::uint64_t>{100});
  }
  for (const TetSnapshotFacet& facet : snapshot.constrained_facets) {
    PHX_CHECK(facet.source == 0 && facet.orientation == 1);
  }
  const auto midpoint = std::find_if(
      snapshot.vertices.begin(), snapshot.vertices.end(),
      [](const TetSnapshotVertex& vertex) { return vertex.id == 4; });
  PHX_CHECK(midpoint != snapshot.vertices.end());
  if (midpoint != snapshot.vertices.end()) {
    PHX_CHECK(midpoint->witness.stratum == SourceStratum::kFacet);
    PHX_CHECK(midpoint->witness.entity == 3);
    PHX_CHECK(midpoint->witness.parameters[0] == 0.5);
    PHX_CHECK(midpoint->witness.parameters[1] == 0.0);
    PHX_CHECK(!midpoint->has_edge_parameter);
  }
  PHX_CHECK(transact(1, nullptr) != TetEditStatus::kApplied);
  PHX_CHECK(transact(2, nullptr) != TetEditStatus::kApplied);
  PHX_CHECK(transact(3, nullptr) != TetEditStatus::kApplied);
}

// An edge on the shared diagonal of two coplanar source facets in different
// groups. A split carrier that is only the rounding of its edge point leaves
// the diagonal: the exact coverage predicate refuses it with certainty
// (invalid geometry, not an uncertain predicate). A short dyadic parameter
// whose edge point is exact lies on the diagonal and is accepted.
void split_on_shared_source_diagonal_needs_an_exact_edge_point() {
  using namespace phx::mc;
  // A, B span the diagonal of the z = 0 square; C, D close it; E lies below.
  constexpr std::array<double, 15> points{
      0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.25, 0.75, -0.5};
  constexpr std::array<std::int32_t, 8> tetrahedra{0, 1, 2, 4, 0, 3, 1, 4};
  constexpr std::array<std::int32_t, 2> regions{0, 0};
  constexpr std::array<std::int32_t, 18> faces{
      0, 1, 2, 0, 3, 1, 1, 4, 2, 0, 2, 4, 3, 4, 1, 0, 4, 3};
  constexpr std::array<std::int32_t, 6> face_sources{0, 1, 2, 3, 4, 5};
  constexpr std::array<std::int8_t, 6> face_orientations{1, 1, 1, 1, 1, 1};
  constexpr std::array<std::int32_t, 6> source_face_groups{0, 1, 2, 3, 4, 5};
  constexpr std::array<std::int64_t, 6> source_face_ids{300, 301, 302, 303, 304, 305};
  constexpr std::array<double, 6> source_face_tolerances{};
  constexpr std::array<std::int32_t, 0> no_segments{};
  constexpr std::array<std::int64_t, 0> no_segment_ids{};
  constexpr std::array<double, 0> no_segment_tolerances{};
  constexpr std::array<SourceWitness, 0> no_witnesses{};
  const SourceComplexView source{
      std::span<const double>(points),
      std::span<const std::int32_t>(faces),
      std::span<const std::int32_t>(source_face_groups),
      std::span<const std::int64_t>(source_face_ids),
      std::span<const double>(source_face_tolerances),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int64_t>(no_segment_ids),
      std::span<const double>(no_segment_tolerances),
      std::span<const SourceWitness>(no_witnesses)};

  const auto split = [&](double parameter) {
    TetMesh mesh(BoundaryPolicy::kConforming);
    const std::int32_t build_status = mesh.build(
        points, tetrahedra, regions, faces, face_sources, face_orientations,
        no_segments, no_segments, std::span<const double>{}, &source);
    PHX_CHECK(build_status == PHX_MC_OK);
    if (build_status != PHX_MC_OK) return TetEditStatus::kRejectedTopology;
    constexpr std::array<std::uint64_t, 5> vertex_ids{0, 1, 2, 3, 4};
    constexpr std::array<std::uint64_t, 2> cell_ids{100, 101};
    constexpr std::array<std::uint64_t, 6> facet_ids{200, 201, 202, 203, 204, 205};
    PHX_CHECK(mesh.assign_initial_ids(vertex_ids, cell_ids, facet_ids,
                                      std::span<const std::uint64_t>{}) == PHX_MC_OK);

    TetEditVertex inserted;
    inserted.id = 5;
    // Each coordinate is one correctly rounded operation on exact values.
    inserted.carrier = {0.5 * parameter, 0.5 + 0.5 * parameter, 0.0};
    inserted.has_edge_parameter = true;
    inserted.edge_witness_supplied = true;
    inserted.edge_endpoints = {0, 1};
    inserted.edge_parameter = parameter;

    const auto cell = [](std::uint64_t id, std::array<std::uint64_t, 4> vertices,
                         std::array<std::int32_t, 4> sources, std::uint64_t parent) {
      TetEditCell result;
      result.id = id;
      result.vertices = vertices;
      result.facet_sources = sources;
      result.region = 0;
      result.ancestor_cell_ids = {parent};
      return result;
    };
    TetEdit operation;
    operation.kind = TetEditKind::kSplit;
    operation.split_edges.push_back({0, 1});
    operation.removed_cell_ids = {100, 101};
    operation.changed_vertices.push_back(inserted);
    operation.replacement_cells = {
        cell(102, {0, 5, 2, 4}, {kNoConstraint, 3, kNoConstraint, 0}, 100),
        cell(103, {5, 1, 2, 4}, {2, kNoConstraint, kNoConstraint, 0}, 100),
        cell(104, {0, 3, 5, 4}, {kNoConstraint, kNoConstraint, 5, 1}, 101),
        cell(105, {5, 3, 1, 4}, {4, kNoConstraint, kNoConstraint, 1}, 101)};
    operation.removed_facet_ids = {200, 201};
    operation.replacement_facets = {
        {206, {0, 5, 2}, 0, 1, false},
        {207, {5, 1, 2}, 0, 1, false},
        {208, {1, 5, 3}, 1, 1, false},
        {209, {5, 0, 3}, 1, 1, false}};
    TetEditBatch batch;
    batch.expected_generation = mesh.generation();
    batch.source_revision = mesh.source_revision();
    batch.frame_id = mesh.frame_id();
    batch.identity_bound = mesh.identity_bound();
    batch.operations.push_back(std::move(operation));
    batch.source_id = mesh.source_id();
    batch.source_map_id = mesh.source_map_id();
    batch.periodic_identity = mesh.periodic_identity();
    std::vector<TetEditResult> results;
    const TetEditStatus batch_status = mesh.apply_edit_batch(batch, results);
    if (batch_status != TetEditStatus::kApplied) return batch_status;
    if (results.size() != 1) return TetEditStatus::kRejectedTopology;
    return results.front().status;
  };

  // The full-precision metric midpoint rounds y off the diagonal y = x + 0.5
  // (y - 0.5 is exact for y in [0.5, 1)).
  constexpr double rounded = 0x1.9aed20c368972p-2;
  PHX_CHECK((0.5 + 0.5 * rounded) - 0.5 != 0.5 * rounded);
  PHX_CHECK(split(rounded) == TetEditStatus::kInvalidGeometry);
  PHX_CHECK(split(13.0 / 32.0) == TetEditStatus::kApplied);
}

// A vertex V inside source facet ABC (z = 0) stars three cells under apex E.
// Without a source witness V is its own exact source point: it may move within
// ABC (every incident facet is revalidated exactly) but not off the plane or
// out of the facet, and it may collapse into A, retiring its boundary facets
// for the single facet BCA, which exactly covers the same source facet.
void boundary_vertex_moves_and_collapses_within_its_source_facet() {
  using namespace phx::mc;
  constexpr std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.25, 1.0, 0.25, 0.25, 0.0};
  // A = 0, B = 1, C = 2, E = 3, V = 4.
  constexpr std::array<std::int32_t, 12> tetrahedra{0, 1, 4, 3, 1, 2, 4, 3, 2, 0, 4, 3};
  constexpr std::array<std::int32_t, 3> regions{0, 0, 0};
  constexpr std::array<std::int32_t, 18> faces{
      0, 1, 4, 1, 2, 4, 2, 0, 4, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  constexpr std::array<std::int32_t, 6> face_sources{0, 0, 0, 1, 2, 3};
  constexpr std::array<std::int8_t, 6> face_orientations{1, 1, 1, 1, 1, 1};
  constexpr std::array<std::int32_t, 12> source_faces{0, 1, 2, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  constexpr std::array<std::int32_t, 4> source_face_groups{0, 1, 2, 3};
  constexpr std::array<std::int64_t, 4> source_face_ids{300, 301, 302, 303};
  constexpr std::array<double, 4> source_face_tolerances{};
  constexpr std::array<std::int32_t, 0> no_segments{};
  constexpr std::array<std::int64_t, 0> no_segment_ids{};
  constexpr std::array<double, 0> no_segment_tolerances{};
  constexpr std::array<SourceWitness, 0> no_witnesses{};
  const SourceComplexView source{
      std::span<const double>(points),
      std::span<const std::int32_t>(source_faces),
      std::span<const std::int32_t>(source_face_groups),
      std::span<const std::int64_t>(source_face_ids),
      std::span<const double>(source_face_tolerances),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int64_t>(no_segment_ids),
      std::span<const double>(no_segment_tolerances),
      std::span<const SourceWitness>(no_witnesses)};

  const auto apply = [&](TetEdit operation) {
    TetMesh mesh(BoundaryPolicy::kConforming);
    const std::int32_t build_status = mesh.build(
        points, tetrahedra, regions, faces, face_sources, face_orientations,
        no_segments, no_segments, std::span<const double>{}, &source);
    PHX_CHECK(build_status == PHX_MC_OK);
    if (build_status != PHX_MC_OK) return TetEditStatus::kRejectedTopology;
    constexpr std::array<std::uint64_t, 5> vertex_ids{0, 1, 2, 3, 4};
    constexpr std::array<std::uint64_t, 3> cell_ids{100, 101, 102};
    constexpr std::array<std::uint64_t, 6> facet_ids{200, 201, 202, 203, 204, 205};
    PHX_CHECK(mesh.assign_initial_ids(vertex_ids, cell_ids, facet_ids,
                                      std::span<const std::uint64_t>{}) == PHX_MC_OK);
    TetEditBatch batch;
    batch.expected_generation = mesh.generation();
    batch.source_revision = mesh.source_revision();
    batch.frame_id = mesh.frame_id();
    batch.identity_bound = mesh.identity_bound();
    batch.operations.push_back(std::move(operation));
    batch.source_id = mesh.source_id();
    batch.source_map_id = mesh.source_map_id();
    batch.periodic_identity = mesh.periodic_identity();
    std::vector<TetEditResult> results;
    const TetEditStatus batch_status = mesh.apply_edit_batch(batch, results);
    if (batch_status != TetEditStatus::kApplied) return batch_status;
    if (results.size() != 1) return TetEditStatus::kRejectedTopology;
    return results.front().status;
  };
  const auto relocation = [](std::array<double, 3> carrier) {
    TetEditVertex moved;
    moved.id = 4;
    moved.carrier = carrier;
    moved.edge_witness_supplied = true;
    TetEdit operation;
    operation.kind = TetEditKind::kRelocate;
    operation.removed_cell_ids = {100, 101, 102};
    operation.changed_vertices.push_back(moved);
    return operation;
  };
  PHX_CHECK(apply(relocation({0.375, 0.25, 0.0})) == TetEditStatus::kApplied);
  PHX_CHECK(apply(relocation({0.375, 0.25, 0.0625})) == TetEditStatus::kInvalidGeometry);
  PHX_CHECK(apply(relocation({0.375, -0.0625, 0.0})) != TetEditStatus::kApplied);

  TetEdit collapse;
  collapse.kind = TetEditKind::kCollapse;
  collapse.collapse_pairs.push_back({4, 0});
  collapse.removed_vertex_ids = {4};
  collapse.removed_cell_ids = {100, 101, 102};
  TetEditCell restored;
  restored.id = 103;
  restored.vertices = {1, 2, 0, 3};
  restored.facet_sources = {3, 1, 2, 0};
  restored.region = 0;
  restored.ancestor_cell_ids = {100, 101, 102};
  collapse.replacement_cells = {restored};
  collapse.removed_facet_ids = {200, 201, 202};
  collapse.replacement_facets = {{206, {1, 2, 0}, 0, 1, false}};
  PHX_CHECK(apply(collapse) == TetEditStatus::kApplied);
  // Retiring the boundary needs the exact replacement cover.
  TetEdit uncovered = collapse;
  uncovered.replacement_facets.clear();
  PHX_CHECK(apply(uncovered) != TetEditStatus::kApplied);
}

// Boundary vertices V and W lie inside source facet ABC (z = 0), five cells
// cone the facet's triangulation to apex E. Collapsing V into W may move W to
// the edge midpoint M: the edit replaces star(V) and star(W), and the
// replacement cells, the replacement facet CAW and the retained facets ABW and
// BCW are checked at M. W may not leave the facet plane or the facet, and its
// whole star must be replaced.
void collapse_relocates_its_survivor_within_its_source_facet() {
  using namespace phx::mc;
  constexpr std::array<double, 18> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.25, 0.25, 1.0, 0.25, 0.25, 0.0, 0.5, 0.25, 0.0};
  // A = 0, B = 1, C = 2, E = 3, V = 4, W = 5.
  constexpr std::array<std::int32_t, 20> tetrahedra{
      0, 1, 5, 3, 1, 2, 5, 3, 2, 0, 4, 3, 0, 5, 4, 3, 5, 2, 4, 3};
  constexpr std::array<std::int32_t, 5> regions{0, 0, 0, 0, 0};
  constexpr std::array<std::int32_t, 24> faces{
      0, 1, 5, 1, 2, 5, 2, 0, 4, 0, 5, 4, 5, 2, 4, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  constexpr std::array<std::int32_t, 8> face_sources{0, 0, 0, 0, 0, 1, 2, 3};
  constexpr std::array<std::int8_t, 8> face_orientations{1, 1, 1, 1, 1, 1, 1, 1};
  constexpr std::array<std::int32_t, 12> source_faces{0, 1, 2, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  constexpr std::array<std::int32_t, 4> source_face_groups{0, 1, 2, 3};
  constexpr std::array<std::int64_t, 4> source_face_ids{300, 301, 302, 303};
  constexpr std::array<double, 4> source_face_tolerances{};
  constexpr std::array<std::int32_t, 0> no_segments{};
  constexpr std::array<std::int64_t, 0> no_segment_ids{};
  constexpr std::array<double, 0> no_segment_tolerances{};
  constexpr std::array<SourceWitness, 0> no_witnesses{};
  const SourceComplexView source{
      std::span<const double>(points),
      std::span<const std::int32_t>(source_faces),
      std::span<const std::int32_t>(source_face_groups),
      std::span<const std::int64_t>(source_face_ids),
      std::span<const double>(source_face_tolerances),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int32_t>(no_segments),
      std::span<const std::int64_t>(no_segment_ids),
      std::span<const double>(no_segment_tolerances),
      std::span<const SourceWitness>(no_witnesses)};

  std::array<double, 3> survivor_point{};
  const auto apply = [&](TetEdit operation) {
    TetMesh mesh(BoundaryPolicy::kConforming);
    const std::int32_t build_status = mesh.build(
        points, tetrahedra, regions, faces, face_sources, face_orientations,
        no_segments, no_segments, std::span<const double>{}, &source);
    PHX_CHECK(build_status == PHX_MC_OK);
    if (build_status != PHX_MC_OK) return TetEditStatus::kRejectedTopology;
    constexpr std::array<std::uint64_t, 6> vertex_ids{0, 1, 2, 3, 4, 5};
    constexpr std::array<std::uint64_t, 5> cell_ids{100, 101, 102, 103, 104};
    constexpr std::array<std::uint64_t, 8> facet_ids{
        200, 201, 202, 203, 204, 205, 206, 207};
    PHX_CHECK(mesh.assign_initial_ids(vertex_ids, cell_ids, facet_ids,
                                      std::span<const std::uint64_t>{}) == PHX_MC_OK);
    TetEditBatch batch;
    batch.expected_generation = mesh.generation();
    batch.source_revision = mesh.source_revision();
    batch.frame_id = mesh.frame_id();
    batch.identity_bound = mesh.identity_bound();
    batch.operations.push_back(std::move(operation));
    batch.source_id = mesh.source_id();
    batch.source_map_id = mesh.source_map_id();
    batch.periodic_identity = mesh.periodic_identity();
    std::vector<TetEditResult> results;
    const TetEditStatus batch_status = mesh.apply_edit_batch(batch, results);
    const double* survivor = mesh.point(mesh.vertex_slot(5));
    std::copy_n(survivor, 3, survivor_point.begin());
    if (batch_status != TetEditStatus::kApplied) return batch_status;
    if (results.size() != 1) return TetEditStatus::kRejectedTopology;
    return results.front().status;
  };
  const auto collapse = [](std::array<double, 3> carrier) {
    TetEdit operation;
    operation.kind = TetEditKind::kCollapse;
    operation.collapse_pairs.push_back({4, 5});
    operation.removed_vertex_ids = {4};
    operation.removed_cell_ids = {100, 101, 102, 103, 104};
    TetEditVertex moved;
    moved.id = 5;
    moved.carrier = carrier;
    moved.edge_witness_supplied = true;
    operation.changed_vertices.push_back(moved);
    TetEditCell abw;
    abw.id = 105;
    abw.vertices = {0, 1, 5, 3};
    abw.facet_sources = {kNoConstraint, kNoConstraint, 1, 0};
    abw.region = 0;
    abw.ancestor_cell_ids = {100};
    TetEditCell bcw;
    bcw.id = 106;
    bcw.vertices = {1, 2, 5, 3};
    bcw.facet_sources = {kNoConstraint, kNoConstraint, 2, 0};
    bcw.region = 0;
    bcw.ancestor_cell_ids = {101};
    TetEditCell caw;
    caw.id = 107;
    caw.vertices = {2, 0, 5, 3};
    caw.facet_sources = {kNoConstraint, kNoConstraint, 3, 0};
    caw.region = 0;
    caw.ancestor_cell_ids = {102, 103, 104};
    operation.replacement_cells = {abw, bcw, caw};
    operation.removed_facet_ids = {202, 203, 204};
    operation.replacement_facets = {{208, {2, 0, 5}, 0, 1, false}};
    return operation;
  };
  // The midpoint of VW keeps W in ABC: applied, and W is at M afterwards.
  PHX_CHECK(apply(collapse({0.375, 0.25, 0.0})) == TetEditStatus::kApplied);
  PHX_CHECK(survivor_point[0] == 0.375 && survivor_point[1] == 0.25 &&
            survivor_point[2] == 0.0);
  // The plain collapse (W stays) is still a collapse.
  TetEdit plain = collapse({0.5, 0.25, 0.0});
  plain.changed_vertices.clear();
  PHX_CHECK(apply(plain) == TetEditStatus::kApplied);
  PHX_CHECK(survivor_point[0] == 0.5);
  // Off the facet plane: the retained facets ABW and BCW refuse it exactly.
  PHX_CHECK(apply(collapse({0.375, 0.25, 0x1p-20})) == TetEditStatus::kInvalidGeometry);
  // Out of the facet (below AB): the cell ABWE inverts.
  PHX_CHECK(apply(collapse({0.375, -0.0625, 0.0})) == TetEditStatus::kInvalidGeometry);
  // A survivor whose star is not wholly replaced cannot move.
  TetEdit partial = collapse({0.375, 0.25, 0.0});
  partial.removed_cell_ids = {100, 102, 103, 104};
  partial.replacement_cells.erase(partial.replacement_cells.begin() + 1);
  PHX_CHECK(apply(partial) == TetEditStatus::kRejectedTopology);
  // Only survivors move: the removed vertex cannot be a changed vertex.
  TetEdit removed_moves = collapse({0.375, 0.25, 0.0});
  removed_moves.changed_vertices.front().id = 4;
  PHX_CHECK(apply(removed_moves) != TetEditStatus::kApplied);
}

void relocation_checks_incident_source_facets_atomically() {
  using namespace phx::mc;
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0};
  constexpr std::array<std::int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 1> regions{0};
  constexpr std::array<std::int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<std::int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr std::array<std::int8_t, 4> face_orientations{1, 1, 1, 1};
  constexpr std::array<std::int32_t, 4> face_groups{0, 1, 2, 3};
  constexpr std::array<std::int64_t, 4> source_face_ids{300, 301, 302, 303};
  constexpr std::array<double, 4> face_tolerances{};
  constexpr std::array<std::int32_t, 0> no_segments{};
  constexpr std::array<std::int64_t, 0> no_segment_ids{};
  constexpr std::array<double, 0> no_segment_tolerances{};
  constexpr std::array<SourceWitness, 0> no_witnesses{};
  const SourceComplexView source{
      points, faces, face_groups, source_face_ids, face_tolerances,
      no_segments, no_segments, no_segment_ids, no_segment_tolerances,
      no_witnesses};
  TetMesh mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(mesh.build(points, tetrahedra, regions, faces, face_sources,
                       face_orientations, no_segments, no_segments, {}, &source) ==
            PHX_MC_OK);
  constexpr std::array<std::uint64_t, 4> vertex_ids{0, 1, 2, 3};
  constexpr std::array<std::uint64_t, 1> cell_ids{100};
  constexpr std::array<std::uint64_t, 4> facet_ids{200, 201, 202, 203};
  PHX_CHECK(mesh.assign_initial_ids(vertex_ids, cell_ids, facet_ids, {}) ==
            PHX_MC_OK);

  TetEditVertex midpoint;
  midpoint.id = 4;
  midpoint.carrier = {0.5, 0.5, 0.0};
  midpoint.witness.stratum = SourceStratum::kFacet;
  midpoint.witness.entity = 3;
  midpoint.witness.parameters[0] = 0.5;
  midpoint.witness.parameters[1] = 0.0;
  midpoint.witness.deviation = 0.0;
  midpoint.ancestry = {{0, {}, 0.5}, {1, {}, 0.5}};
  TetEditCell first_child;
  first_child.id = 101;
  first_child.vertices = {0, 4, 2, 3};
  first_child.facet_sources = {kNoConstraint, 1, 2, 3};
  first_child.region = 0;
  first_child.ancestor_cell_ids = {100};
  TetEditCell second_child;
  second_child.id = 102;
  second_child.vertices = {4, 1, 2, 3};
  second_child.facet_sources = {0, kNoConstraint, 2, 3};
  second_child.region = 0;
  second_child.ancestor_cell_ids = {100};
  TetEdit operation;
  operation.kind = TetEditKind::kSplit;
  operation.split_edges.push_back({0, 1});
  operation.removed_cell_ids.push_back(100);
  operation.changed_vertices.push_back(midpoint);
  operation.replacement_cells = {first_child, second_child};
  operation.removed_facet_ids = {202, 203};
  operation.replacement_facets = {
      {204, {1, 4, 3}, 2, 1, false},
      {205, {4, 0, 3}, 2, 1, false},
      {206, {0, 4, 2}, 3, 1, false},
      {207, {4, 1, 2}, 3, 1, false}};
  TetEditBatch split;
  split.expected_generation = mesh.generation();
  split.identity_bound = mesh.identity_bound();
  split.source_id = mesh.source_id();
  split.source_revision = mesh.source_revision();
  split.source_map_id = mesh.source_map_id();
  split.frame_id = mesh.frame_id();
  split.periodic_identity = mesh.periodic_identity();
  split.operations.push_back(operation);
  std::vector<TetEditResult> split_results;
  PHX_CHECK(mesh.apply_edit_batch(split, split_results) == TetEditStatus::kApplied);
  PHX_CHECK(split_results.size() == 1 &&
            split_results.front().status == TetEditStatus::kApplied);
  TetMeshSnapshot split_state;
  PHX_CHECK(mesh.snapshot(split_state) == TetEditStatus::kApplied);

  const auto incident_cells = [&](std::uint64_t vertex_id) {
    std::vector<std::uint64_t> affected;
    for (const TetSnapshotCell& cell : split_state.cells) {
      if (std::find(cell.vertices.begin(), cell.vertices.end(), vertex_id) !=
          cell.vertices.end()) {
        affected.push_back(cell.id);
      }
    }
    return affected;
  };
  const std::int32_t midpoint_slot = mesh.vertex_slot(4);
  TetEditVertex valid_move;
  valid_move.id = 4;
  valid_move.carrier = {0.4, 0.4, 0.0};
  valid_move.witness = mesh.witness(midpoint_slot);
  valid_move.witness.parameters[0] = 0.4;
  valid_move.witness.parameters[1] = 0.0;
  valid_move.protection_radius = mesh.protection_radius(midpoint_slot);
  valid_move.protected_vertex = mesh.protected_vertex(midpoint_slot);
  const std::vector<std::uint64_t> affected = incident_cells(4);
  PHX_CHECK(mesh.relocate(std::span(&valid_move, 1), affected) ==
            TetEditStatus::kApplied);
  TetMeshSnapshot after_valid;
  PHX_CHECK(mesh.snapshot(after_valid) == TetEditStatus::kApplied);
  PHX_CHECK(after_valid.generation > split_state.generation);
  PHX_CHECK(after_valid.vertices.size() == split_state.vertices.size());
  PHX_CHECK(after_valid.constrained_facets.size() ==
            split_state.constrained_facets.size());
  PHX_CHECK(after_valid.cells.size() == split_state.cells.size());
  for (std::size_t row = 0; row < after_valid.cells.size(); ++row) {
    PHX_CHECK(after_valid.cells[row].id == split_state.cells[row].id);
    PHX_CHECK(after_valid.cells[row].ancestor_cell_ids ==
              split_state.cells[row].ancestor_cell_ids);
    PHX_CHECK(after_valid.cells[row].edit_kind == split_state.cells[row].edit_kind);
  }

  const TetMeshSnapshot before_invalid = after_valid;
  PHX_CHECK(affected.size() == 2);
  if (affected.size() > 1) {
    TetEditVertex incomplete_star_move = valid_move;
    incomplete_star_move.carrier = {0.35, 0.35, 0.0};
    incomplete_star_move.witness.parameters[0] = 0.35;
    const std::array<std::uint64_t, 1> incomplete_star{affected.front()};
    PHX_CHECK(mesh.relocate(std::span(&incomplete_star_move, 1),
                            incomplete_star) ==
              TetEditStatus::kRejectedTopology);
    TetMeshSnapshot after_incomplete_star;
    PHX_CHECK(mesh.snapshot(after_incomplete_star) == TetEditStatus::kApplied);
    PHX_CHECK(after_incomplete_star.generation == before_invalid.generation);
    const auto unchanged_midpoint = std::find_if(
        after_incomplete_star.vertices.begin(), after_incomplete_star.vertices.end(),
        [](const TetSnapshotVertex& vertex) { return vertex.id == 4; });
    PHX_CHECK(unchanged_midpoint != after_incomplete_star.vertices.end());
    if (unchanged_midpoint != after_incomplete_star.vertices.end()) {
      PHX_CHECK(unchanged_midpoint->carrier == valid_move.carrier);
    }
  }
  TetEditVertex first_candidate = valid_move;
  first_candidate.carrier = {0.35, 0.35, 0.0};
  first_candidate.witness.parameters[0] = 0.35;
  TetEditVertex duplicate_candidate = first_candidate;
  duplicate_candidate.carrier = {0.3, 0.3, 0.0};
  duplicate_candidate.witness.parameters[0] = 0.3;
  const std::array<TetEditVertex, 2> duplicate_moves{
      first_candidate, duplicate_candidate};
  PHX_CHECK(mesh.relocate(duplicate_moves, affected) ==
            TetEditStatus::kRejectedTopology);
  TetMeshSnapshot after_duplicate_refusal;
  PHX_CHECK(mesh.snapshot(after_duplicate_refusal) == TetEditStatus::kApplied);
  PHX_CHECK(after_duplicate_refusal.generation == before_invalid.generation);
  const auto midpoint_after_duplicate = std::find_if(
      after_duplicate_refusal.vertices.begin(), after_duplicate_refusal.vertices.end(),
      [](const TetSnapshotVertex& vertex) { return vertex.id == 4; });
  PHX_CHECK(midpoint_after_duplicate != after_duplicate_refusal.vertices.end());
  if (midpoint_after_duplicate != after_duplicate_refusal.vertices.end()) {
    PHX_CHECK(midpoint_after_duplicate->carrier == valid_move.carrier);
  }
  TetEditVertex off_adjacent_source;
  off_adjacent_source.id = 4;
  off_adjacent_source.carrier = {0.5, 0.4, 0.0};
  off_adjacent_source.witness = mesh.witness(midpoint_slot);
  off_adjacent_source.witness.parameters[0] = 0.4;
  off_adjacent_source.witness.parameters[1] = 0.1;
  off_adjacent_source.protection_radius = mesh.protection_radius(midpoint_slot);
  off_adjacent_source.protected_vertex = mesh.protected_vertex(midpoint_slot);
  PHX_CHECK(mesh.relocate(std::span(&off_adjacent_source, 1), affected) ==
            TetEditStatus::kInvalidGeometry);
  TetMeshSnapshot after_rejection;
  PHX_CHECK(mesh.snapshot(after_rejection) == TetEditStatus::kApplied);
  PHX_CHECK(after_rejection.generation == before_invalid.generation);
  PHX_CHECK(after_rejection.vertices.size() == before_invalid.vertices.size());
  PHX_CHECK(after_rejection.cells.size() == before_invalid.cells.size());
  PHX_CHECK(after_rejection.constrained_facets.size() ==
            before_invalid.constrained_facets.size());
  PHX_CHECK(after_rejection.source_points == before_invalid.source_points);
  PHX_CHECK(after_rejection.source_faces == before_invalid.source_faces);
  PHX_CHECK(after_rejection.source_face_groups == before_invalid.source_face_groups);
  PHX_CHECK(after_rejection.source_face_ids == before_invalid.source_face_ids);
  PHX_CHECK(after_rejection.source_face_tolerances ==
            before_invalid.source_face_tolerances);
  PHX_CHECK(after_rejection.source_segments == before_invalid.source_segments);
  PHX_CHECK(after_rejection.source_segment_groups ==
            before_invalid.source_segment_groups);
  PHX_CHECK(after_rejection.source_segment_ids ==
            before_invalid.source_segment_ids);
  PHX_CHECK(after_rejection.source_segment_tolerances ==
            before_invalid.source_segment_tolerances);
  for (std::size_t row = 0; row < after_rejection.vertices.size(); ++row) {
    PHX_CHECK(after_rejection.vertices[row].id == before_invalid.vertices[row].id);
    PHX_CHECK(after_rejection.vertices[row].carrier ==
              before_invalid.vertices[row].carrier);
    PHX_CHECK(after_rejection.vertices[row].stratum_dimension ==
              before_invalid.vertices[row].stratum_dimension);
    PHX_CHECK(after_rejection.vertices[row].protection_radius ==
              before_invalid.vertices[row].protection_radius);
    PHX_CHECK(after_rejection.vertices[row].protected_vertex ==
              before_invalid.vertices[row].protected_vertex);
    PHX_CHECK(after_rejection.vertices[row].witness.stratum ==
              before_invalid.vertices[row].witness.stratum);
    PHX_CHECK(after_rejection.vertices[row].witness.entity ==
              before_invalid.vertices[row].witness.entity);
    PHX_CHECK(after_rejection.vertices[row].witness.parameters[0] ==
              before_invalid.vertices[row].witness.parameters[0]);
    PHX_CHECK(after_rejection.vertices[row].witness.parameters[1] ==
              before_invalid.vertices[row].witness.parameters[1]);
    PHX_CHECK(after_rejection.vertices[row].witness.deviation ==
              before_invalid.vertices[row].witness.deviation);
  }
  for (std::size_t row = 0; row < after_rejection.cells.size(); ++row) {
    PHX_CHECK(after_rejection.cells[row].id == before_invalid.cells[row].id);
    PHX_CHECK(after_rejection.cells[row].vertices ==
              before_invalid.cells[row].vertices);
    PHX_CHECK(after_rejection.cells[row].region ==
              before_invalid.cells[row].region);
    PHX_CHECK(after_rejection.cells[row].ancestor_cell_ids ==
              before_invalid.cells[row].ancestor_cell_ids);
    PHX_CHECK(after_rejection.cells[row].edit_kind ==
              before_invalid.cells[row].edit_kind);
  }
  for (std::size_t row = 0; row < after_rejection.constrained_facets.size(); ++row) {
    PHX_CHECK(after_rejection.constrained_facets[row].id ==
              before_invalid.constrained_facets[row].id);
    PHX_CHECK(after_rejection.constrained_facets[row].vertices ==
              before_invalid.constrained_facets[row].vertices);
    PHX_CHECK(after_rejection.constrained_facets[row].source ==
              before_invalid.constrained_facets[row].source);
    PHX_CHECK(after_rejection.constrained_facets[row].orientation ==
              before_invalid.constrained_facets[row].orientation);
    PHX_CHECK(after_rejection.constrained_facets[row].protected_facet ==
              before_invalid.constrained_facets[row].protected_facet);
  }
}
void exact_boundary_subface_coverage_rejects_gap_reversal_and_overlap() {
  using namespace phx::mc;
  constexpr std::array<double, 9> subject{
      0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0};
  constexpr std::array<double, 18> split_children{
      0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0,
      0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0};
  constexpr std::array<std::array<std::int32_t, 4>, 2> two_faces{{
      {0, 1, 2, 0}, {3, 4, 5, 1}}};
  constexpr std::array<std::int32_t, 2> two_rows{0, 1};
  std::vector<std::int32_t> contributors;
  PHX_CHECK(exact_facet_coverage(subject.data(), split_children.data(),
                                 two_faces, two_rows, contributors));
  const std::array<std::int32_t, 1> missing_rows{0};
  PHX_CHECK(!exact_facet_coverage(subject.data(), split_children.data(),
                                  two_faces, missing_rows, contributors));

  constexpr std::array<double, 18> reversed_children{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0,
      0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0};
  PHX_CHECK(!exact_facet_coverage(subject.data(), reversed_children.data(),
                                  two_faces, two_rows, contributors));

  constexpr std::array<double, 27> overlapping_children{
      0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0,
      0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0};
  constexpr std::array<std::array<std::int32_t, 4>, 3> three_faces{{
      {0, 1, 2, 0}, {3, 4, 5, 1}, {6, 7, 8, 2}}};
  constexpr std::array<std::int32_t, 3> three_rows{0, 1, 2};
  PHX_CHECK(!exact_facet_coverage(subject.data(), overlapping_children.data(),
                                  three_faces, three_rows, contributors));
}

// Face row 3 of the tetrahedron (1, 1, 1) + unit simplex lies on z = 1 and
// declares the plane through vertex 0 with axes (1, 0, slope) and (0, 1, 0);
// the other rows declare none. Vertex 1 = (2, 1, 1) is the only corner off
// the plane, at distance slope / sqrt(1 + slope^2) ~ slope.
int32_t create_declared_tet(double slope, double tolerance, phx_mc_tet_mesh** mesh,
                            phx_mc_tet_plane_refusal* refusal) {
  static constexpr std::array<double, 12> points{
      1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0};
  static constexpr std::array<int32_t, 4> tetrahedra{0, 1, 2, 3};
  static constexpr std::array<int32_t, 1> regions{0};
  static constexpr std::array<int32_t, 12> faces{1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  static constexpr std::array<int32_t, 4> sources{0, 1, 2, 3};
  static constexpr std::array<int64_t, 4> source_ids{101, 102, 103, 104};
  static constexpr std::array<double, 4> source_tolerances{};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::array<double, 36> planes{};
  planes.fill(nan);
  const std::array<double, 9> declared{1.0, 1.0, 1.0, 1.0, 0.0, slope, 0.0, 1.0, 0.0};
  std::copy(declared.begin(), declared.end(), planes.begin() + 27);
  const std::array<double, 4> plane_tolerances{0.0, 0.0, 0.0, tolerance};
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 4;
  create.points = points.data();
  create.tetrahedron_count = 1;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.face_count = 4;
  create.faces = faces.data();
  create.face_sources = sources.data();
  create.source_point_count = 4;
  create.source_points = points.data();
  create.source_face_count = 4;
  create.source_faces = faces.data();
  create.source_face_groups = sources.data();
  create.source_face_ids = source_ids.data();
  create.source_face_tolerances = source_tolerances.data();
  create.source_face_planes = planes.data();
  create.source_face_plane_tolerances = plane_tolerances.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.plane_refusal = refusal;
  create.result = mesh;
  return phx_mc_tet_mesh_create(&create);
}

void declared_plane_admission_refuses_and_records_vertices() {
  constexpr double tilt = 0x1p-40;
  // Refusal: the plane misses vertex 1's rounding box and no tolerance is
  // declared. The typed record names the vertex, the plane row, the rounded
  // projection parameters and an outward distance bound; nothing is created.
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_plane_refusal refusal{};
  PHX_CHECK(create_declared_tet(tilt, 0.0, &mesh, &refusal) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
  PHX_CHECK(refusal.refused == 1);
  PHX_CHECK(refusal.vertex == 1 && refusal.face_row == 3);
  PHX_CHECK(refusal.parameters[0] == 1.0 && refusal.parameters[1] == 0.0);
  PHX_CHECK(refusal.deviation >= tilt * (1.0 - 0x1p-30) &&
            refusal.deviation <= tilt * (1.0 + 0x1p-30));
  PHX_CHECK(refusal.tolerance == 0.0);

  // Tolerance admission: the same plane within the declared tolerance.
  refusal = phx_mc_tet_plane_refusal{};
  PHX_CHECK(create_declared_tet(tilt, 2.0 * tilt, &mesh, &refusal) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr && refusal.refused == 0);
  if (mesh == nullptr) return;
  int64_t count = -1;
  PHX_CHECK(phx_mc_tet_mesh_plane_record_count(mesh, &count) == PHX_MC_OK);
  PHX_CHECK(count == 3);
  std::array<uint64_t, 3> ids{};
  std::array<int32_t, 3> rows{};
  std::array<double, 6> parameters{};
  std::array<double, 3> deviations{};
  PHX_CHECK(phx_mc_tet_mesh_plane_records(mesh, 2, ids.data(), rows.data(), parameters.data(),
                                          deviations.data()) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx_mc_tet_mesh_plane_records(mesh, 3, ids.data(), rows.data(), parameters.data(),
                                          deviations.data()) == PHX_MC_OK);
  PHX_CHECK((ids == std::array<uint64_t, 3>{0, 1, 2}));
  PHX_CHECK((rows == std::array<int32_t, 3>{3, 3, 3}));
  PHX_CHECK((parameters == std::array<double, 6>{0.0, 0.0, 1.0, 0.0, 0.0, 1.0}));
  PHX_CHECK(deviations[0] == 0.0 && deviations[2] == 0.0);
  PHX_CHECK(deviations[1] >= tilt * (1.0 - 0x1p-30) && deviations[1] <= 2.0 * tilt);
  phx_mc_tet_mesh_free(mesh);

  // Rounding-box admission: a plane 2^-60 off vertex 1 meets the box of
  // reals rounding to its carrier (half an ulp of 1 is 2^-53), so it is
  // admitted without a tolerance and recorded with its nonzero distance.
  mesh = nullptr;
  refusal = phx_mc_tet_plane_refusal{};
  PHX_CHECK(create_declared_tet(0x1p-60, 0.0, &mesh, &refusal) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr && refusal.refused == 0);
  if (mesh == nullptr) return;
  PHX_CHECK(phx_mc_tet_mesh_plane_records(mesh, 3, ids.data(), rows.data(), parameters.data(),
                                          deviations.data()) == PHX_MC_OK);
  PHX_CHECK(deviations[1] > 0.0 && deviations[1] <= 0x1p-59);
  phx_mc_tet_mesh_free(mesh);
}

// Vertices 101 and 211 are two copies of one orbit (211 = 101 + e_x). The edit
// retriangulates the one cell that holds copy 101 but not copy 211, as a split
// next to one face of a fundamental-domain lift does: the outside copy needs no
// change, but it must still carry its exact edit member.
void periodic_closure_accepts_edit_touching_one_orbit_copy() {
  constexpr std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 0.25, 0.25, 0.25};
  constexpr std::array<int32_t, 16> tetrahedra{
      1, 3, 2, 4, 0, 2, 3, 4, 0, 3, 1, 4, 0, 1, 2, 4};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr std::array<int32_t, 4> regions{0, 0, 0, 0};
  constexpr std::array<uint64_t, 5> vertex_ids{101, 211, 307, 409, 500};
  constexpr std::array<uint64_t, 4> cell_ids{9001, 9002, 9003, 9004};
  constexpr std::array<uint64_t, 4> facet_ids{5101, 5102, 5103, 5104};
  constexpr std::array<int64_t, 4> lifted_counts{5, 10, 10, 4};
  constexpr std::array<int64_t, 4> quotient_counts{4, 10, 10, 4};
  constexpr std::array<int64_t, 5> vertex_windings{0, 1, 0, 0, 0};
  constexpr std::array<int32_t, 5> vertex_orbits{0, 0, 1, 2, 3};
  constexpr std::array<int32_t, 4> vertex_representatives{0, 2, 3, 4};
  constexpr char text[] = "periodic-copy";

  struct Bank {
    std::array<uint64_t, 29> lifted_ids{};
    std::array<uint64_t, 28> quotient_ids{};
    std::array<int32_t, 29> lifted_to_quotient{};
    std::array<int32_t, 29> orientation_signs{};
    std::array<int64_t, 29> winding_shifts{};
    std::array<int32_t, 71> corner_permutations{};
    std::array<int32_t, 28> representative_rows{};
    std::array<phx_mc_tet_periodic_entity_orbit_map, 4> maps{};
    phx_mc_tet_periodic_identity_map_view view{};
  };
  const auto fill = [&](Bank& bank, const std::array<uint64_t, 4>& cells,
                        const std::array<uint64_t, 4>& quotient_cell_ids) {
    std::size_t lifted = 0, quotient = 0, corner = 0;
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
      phx_mc_tet_periodic_entity_orbit_map& map = bank.maps[dimension];
      const std::size_t count = static_cast<std::size_t>(lifted_counts[dimension]);
      const std::size_t arity = dimension + 1;
      map.dimension = static_cast<int32_t>(dimension);
      map.arity = static_cast<int32_t>(arity);
      map.rank = 1;
      map.lifted_offset = static_cast<int64_t>(lifted);
      map.lifted_count = static_cast<int64_t>(count);
      map.quotient_offset = static_cast<int64_t>(quotient);
      map.quotient_count = quotient_counts[dimension];
      map.winding_offset = static_cast<int64_t>(lifted);
      map.corner_permutation_offset = static_cast<int64_t>(corner);
      map.representative_offset = static_cast<int64_t>(quotient);
      for (std::size_t row = 0; row < count; ++row) {
        const bool vertex = dimension == 0;
        bank.lifted_ids[lifted + row] = vertex ? vertex_ids[row]
            : dimension == 3 ? cells[row] : 11001 + 100 * dimension + row;
        bank.lifted_to_quotient[lifted + row] =
            vertex ? vertex_orbits[row] : static_cast<int32_t>(row);
        bank.orientation_signs[lifted + row] = 1;
        bank.winding_shifts[lifted + row] = vertex ? vertex_windings[row] : 0;
        for (std::size_t local = 0; local < arity; ++local) {
          bank.corner_permutations[corner + row * arity + local] =
              static_cast<int32_t>(local);
        }
      }
      for (std::size_t row = 0; row < static_cast<std::size_t>(quotient_counts[dimension]);
           ++row) {
        bank.quotient_ids[quotient + row] = dimension == 3
            ? quotient_cell_ids[row] : 30001 + 100 * dimension + row;
        bank.representative_rows[quotient + row] =
            dimension == 0 ? vertex_representatives[row] : static_cast<int32_t>(row);
      }
      lifted += count;
      quotient += static_cast<std::size_t>(quotient_counts[dimension]);
      corner += count * arity;
    }
    phx_mc_tet_periodic_identity_map_view& view = bank.view;
    view.identity_bound = 1;
    view.source_mesh_id = view.source_topology_id = view.lifted_topology_id = text;
    view.periodic_topology_id = view.cell_entity_set_id = view.edge_entity_set_id = text;
    view.source_mesh_id_length = view.source_topology_id_length =
        view.lifted_topology_id_length = view.periodic_topology_id_length =
            view.cell_entity_set_id_length = view.edge_entity_set_id_length =
                sizeof(text) - 1;
    view.entity_map_count = 4;
    view.entity_maps = bank.maps.data();
    view.entity_lifted_count = static_cast<int64_t>(bank.lifted_ids.size());
    view.entity_lifted_ids = bank.lifted_ids.data();
    view.entity_lifted_to_quotient = bank.lifted_to_quotient.data();
    view.entity_orientation_signs = bank.orientation_signs.data();
    view.entity_quotient_count = static_cast<int64_t>(bank.quotient_ids.size());
    view.entity_quotient_ids = bank.quotient_ids.data();
    view.entity_winding_count = static_cast<int64_t>(bank.winding_shifts.size());
    view.entity_winding_shifts = bank.winding_shifts.data();
    view.entity_corner_count = static_cast<int64_t>(bank.corner_permutations.size());
    view.entity_corner_permutations = bank.corner_permutations.data();
    view.entity_representative_count =
        static_cast<int64_t>(bank.representative_rows.size());
    view.entity_representative_rows = bank.representative_rows.data();
  };
  Bank source;
  fill(source, cell_ids, {50001, 50002, 50003, 50004});

  constexpr std::array<double, 12> identity{
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  std::array<phx_mc_tet_periodic_vertex_member, 5> members{};
  for (std::size_t row = 0; row < members.size(); ++row) {
    members[row].vertex_id = vertex_ids[row];
    members[row].winding[0] = vertex_windings[row];
    for (std::size_t value = 0; value < identity.size(); ++value) {
      members[row].representative_to_member[value] = identity[value];
    }
  }
  members[1].representative_to_member[3] = 1.0;
  std::array<phx_mc_tet_periodic_orbit, 4> orbits{};
  for (std::size_t row = 0; row < orbits.size(); ++row) {
    orbits[row].id = 20001 + row;
    orbits[row].representative_vertex_id = vertex_ids[vertex_representatives[row]];
    orbits[row].member_offset = row == 0 ? 0 : static_cast<int64_t>(row + 1);
    orbits[row].member_count = row == 0 ? 2 : 1;
  }
  std::array<phx_mc_tet_periodic_quotient_cell, 4> quotient_cells{};
  for (std::size_t row = 0; row < quotient_cells.size(); ++row) {
    quotient_cells[row].orbit_id = quotient_cells[row].id = 50001 + row;
    quotient_cells[row].region = regions[row];
    quotient_cells[row].lifted_cell_offset = static_cast<int64_t>(row);
    quotient_cells[row].lifted_cell_count = 1;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      const std::size_t vertex = static_cast<std::size_t>(tetrahedra[4 * row + corner]);
      quotient_cells[row].representative_vertices[corner] =
          vertex_ids[vertex_representatives[vertex_orbits[vertex]]];
      quotient_cells[row].winding[3 * corner] = vertex_windings[vertex];
    }
  }
  phx_mc_tet_periodic_state_view state{};
  state.identity_map = source.view;
  state.periodic_orbit_count = static_cast<int64_t>(orbits.size());
  state.periodic_orbits = orbits.data();
  state.periodic_member_count = static_cast<int64_t>(members.size());
  state.periodic_members = members.data();
  state.periodic_quotient_cell_count = static_cast<int64_t>(quotient_cells.size());
  state.periodic_quotient_cells = quotient_cells.data();
  state.periodic_quotient_lifted_count = static_cast<int64_t>(cell_ids.size());
  state.periodic_quotient_lifted_cell_ids = cell_ids.data();

  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = static_cast<int64_t>(points.size() / 3);
  create.points = points.data();
  create.vertex_ids = vertex_ids.data();
  create.tetrahedron_count = static_cast<int64_t>(tetrahedra.size() / 4);
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.cell_ids = cell_ids.data();
  create.face_count = static_cast<int64_t>(faces.size() / 3);
  create.faces = faces.data();
  create.face_sources = face_sources.data();
  create.facet_ids = facet_ids.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.periodic_state = &state;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;

  // Cell 9002 = (101, 307, 409, 500) is retriangulated under a fresh cell and
  // quotient-cell identity; copy 211 lies outside the cavity.
  Bank target;
  fill(target, {9001, 9005, 9003, 9004}, {50001, 50005, 50003, 50004});
  constexpr uint64_t removed_cell = 9002;
  constexpr uint64_t removed_quotient = 50002;
  constexpr uint64_t replacement_cell = 9005;
  constexpr std::array<uint64_t, 4> replacement_vertices{101, 307, 409, 500};
  constexpr std::array<int32_t, 4> replacement_sources{-1, -1, -1, 1};
  constexpr int32_t replacement_region = 0;
  constexpr std::array<int64_t, 2> ancestry_offsets{0, 1};
  phx_mc_tet_periodic_quotient_cell replacement_quotient = quotient_cells[1];
  replacement_quotient.orbit_id = replacement_quotient.id = 50005;
  replacement_quotient.lifted_cell_offset = 0;
  phx_mc_tet_periodic_edit_view periodic{};
  periodic.periodic_member_count = static_cast<int64_t>(members.size());
  periodic.periodic_members = members.data();
  periodic.removed_quotient_cell_count = 1;
  periodic.removed_quotient_cell_ids = &removed_quotient;
  periodic.replacement_quotient_cell_count = 1;
  periodic.replacement_quotient_cells = &replacement_quotient;
  periodic.replacement_quotient_lifted_count = 1;
  periodic.replacement_quotient_lifted_cell_ids = &replacement_cell;
  periodic.target_identity_map = &target.view;
  phx_mc_tet_mesh_edit_request edit{};
  edit.expected_generation = 0;
  edit.kind = PHX_MC_TET_EDIT_RECONNECT;
  edit.periodic = &periodic;
  edit.removed_cell_count = 1;
  edit.removed_cell_ids = &removed_cell;
  edit.replacement_cell_count = 1;
  edit.replacement_cell_ids = &replacement_cell;
  edit.replacement_cell_vertices = replacement_vertices.data();
  edit.replacement_cell_facet_sources = replacement_sources.data();
  edit.replacement_regions = &replacement_region;
  edit.replacement_cell_ancestry_count = 1;
  edit.replacement_cell_ancestry_offsets = ancestry_offsets.data();
  edit.replacement_cell_ancestor_ids = &removed_cell;
  int32_t edit_status = -1;

  // The outside copy keeps its winding and exact image checks.
  std::array<phx_mc_tet_periodic_vertex_member, 5> wrong_winding = members;
  wrong_winding[1].winding[0] = 0;
  periodic.periodic_members = wrong_winding.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  periodic.periodic_member_count = 4;
  std::array<phx_mc_tet_periodic_vertex_member, 4> missing_copy{
      members[0], members[2], members[3], members[4]};
  periodic.periodic_members = missing_copy.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);

  periodic.periodic_member_count = static_cast<int64_t>(members.size());
  periodic.periodic_members = members.data();
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  uint64_t generation = 99;
  int32_t identity_bound = -1;
  int64_t ignored_lengths[4]{};
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(
                mesh, &generation, &identity_bound, &ignored_lengths[0],
                &ignored_lengths[1], &ignored_lengths[2], &ignored_lengths[3]) == PHX_MC_OK);
  PHX_CHECK(generation == 1);
  phx_mc_tet_mesh_free(mesh);
}

void affine_create_c_api_guards() {
  const std::array<double, 12> points{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::array<int32_t, 4> tets{0, 1, 2, 3};
  const std::array<int32_t, 12> faces{1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  const std::array<int32_t, 4> sources{0, 1, 2, 3};
  const std::array<double, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::array<double, 9> reflected{-1, 0, 0, 0, 2, 0, 0, 0, 1};
  const std::array<double, 9> singular{1, 0, 0, 0, 1, 0, 0, 0, 0};
  auto nonfinite = identity;
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request request{};
  request.point_count = 4;
  request.points = points.data();
  request.tetrahedron_count = 1;
  request.tetrahedra = tets.data();
  request.face_count = 4;
  request.faces = faces.data();
  request.face_sources = sources.data();
  request.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  request.result = &mesh;

  PHX_CHECK(phx_mc_tet_mesh_create_with_affine_metric(nullptr, identity.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  auto invalid = [&](const phx_mc_tet_mesh_create_request& bad, const double* matrix) {
    mesh = reinterpret_cast<phx_mc_tet_mesh*>(static_cast<uintptr_t>(1));
    PHX_CHECK(phx_mc_tet_mesh_create_with_affine_metric(&bad, matrix) ==
              PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(mesh == nullptr);
  };
  invalid(request, singular.data());
  nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
  invalid(request, nonfinite.data());
  nonfinite[0] = std::numeric_limits<double>::infinity();
  invalid(request, nonfinite.data());
  auto bad = request;
  bad.points = nullptr;
  invalid(bad, identity.data());
  bad = request;
  bad.tetrahedra = nullptr;
  invalid(bad, identity.data());
  bad = request;
  bad.point_count = -1;
  invalid(bad, identity.data());
  bad = request;
  bad.point_count = std::numeric_limits<int32_t>::max();
  invalid(bad, identity.data());
  const std::array<int32_t, 4> bad_tets{0, 1, 2, 4};
  bad = request;
  bad.tetrahedra = bad_tets.data();
  mesh = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_create_with_affine_metric(&bad, identity.data()) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
  bad = request;
  bad.result = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_create_with_affine_metric(&bad, identity.data()) ==
            PHX_MC_INVALID_ARGUMENT);

  for (const double* matrix : {static_cast<const double*>(nullptr), identity.data(),
                               reflected.data()}) {
    mesh = nullptr;
    PHX_CHECK(phx_mc_tet_mesh_create_with_affine_metric(&request, matrix) == PHX_MC_OK);
    PHX_CHECK(mesh != nullptr);
    phx_mc_tet_mesh_free(mesh);
  }
  mesh = nullptr;
  PHX_CHECK(phx_mc_tet_mesh_create(&request) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr);
  phx_mc_tet_mesh_free(mesh);
}

void selective_facet_protection_is_atomic_and_source_bound() {
  using phx::mc::BoundaryPolicy;
  using phx::mc::TetEditStatus;
  using phx::mc::TetMesh;
  using phx::mc::TetMeshSnapshot;
  constexpr std::array<double, 12> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr std::array<std::int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 1> regions{0};
  constexpr std::array<std::int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<std::int32_t, 4> sources{0, 1, 2, 3};
  TetMesh mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(mesh.build(points, tetrahedra, regions, faces, sources, {}, {}) == PHX_MC_OK);
  const std::array<std::int32_t, 1> absent{99};
  PHX_CHECK(!mesh.protect_facet_sources(absent));
  PHX_CHECK(mesh.generation() == 0);
  const std::array<std::int32_t, 1> selected{0};
  PHX_CHECK(mesh.protect_facet_sources(selected));
  TetMeshSnapshot snapshot;
  PHX_CHECK(mesh.snapshot(snapshot) == TetEditStatus::kApplied);
  PHX_CHECK(snapshot.generation == 1);
  for (const auto& facet : snapshot.constrained_facets) {
    PHX_CHECK(facet.protected_facet == (facet.source == 0));
  }
  std::size_t protected_count = 0;
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.protected_vertex) ++protected_count;
  }
  PHX_CHECK(protected_count == 3);
}

void implicit_source_binding_owns_programs_and_refuses_wrong_orientation() {
  constexpr std::array<double, 12> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr std::array<int32_t, 4> tetrahedra{0, 1, 2, 3};
  constexpr std::array<int32_t, 12> faces{
      1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr char source_id[] = "implicit-source";
  constexpr char binding[] = "binding";
  constexpr char revision[] = "revision";
  constexpr char frame[] = "world-frame";
  phx_mc_tet_mesh_create_request request{};
  request.point_count = 4;
  request.points = points.data();
  request.tetrahedron_count = 1;
  request.tetrahedra = tetrahedra.data();
  request.face_count = 4;
  request.faces = faces.data();
  request.face_sources = face_sources.data();
  request.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  request.identity_bound = 1;
  request.source_id = source_id;
  request.source_id_length = sizeof(source_id) - 1;
  request.source_revision = revision;
  request.source_revision_length = sizeof(revision) - 1;
  request.source_map_id = binding;
  request.source_map_id_length = sizeof(binding) - 1;
  request.frame_id = frame;
  request.frame_id_length = sizeof(frame) - 1;
  phx_mc_tet_mesh* mesh = nullptr;
  request.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&request) == PHX_MC_OK);
  if (mesh == nullptr) return;

  // f(x,y,z)=z, grad f=(0,0,1); the selected source facet lies on z=0.
  std::array<int64_t, 8> value_code{2, 4, 1, 2, -1, 3, -1, 0};
  std::array<double, 5> value_workspace{0, 0, 0, 0, 0};
  std::array<int64_t, 1> value_outputs{4};
  std::array<double, 6> gradient_workspace{0, 0, 0, 0, 0, 1};
  std::array<int64_t, 3> gradient_outputs{3, 4, 5};
  std::array<int32_t, 1> groups{3};
  phx_mc_tet_implicit_source source{};
  source.value = {value_code.data(), 1, nullptr, 0, nullptr, 0, value_workspace.data(),
                  value_workspace.data(), 5, value_outputs.data(), 1, nullptr, 0};
  source.gradient = {nullptr, 0, nullptr, 0, nullptr, 0, gradient_workspace.data(),
                     gradient_workspace.data(), 6, gradient_outputs.data(), 3, nullptr, 0};
  source.groups = groups.data();
  source.group_count = 1;
  source.tolerance = 0.01;
  source.orientation = -1;
  source.source_id = source_id;
  source.source_id_length = sizeof(source_id) - 1;
  source.source_revision = revision;
  source.source_revision_length = sizeof(revision) - 1;
  source.binding_id = binding;
  source.binding_id_length = sizeof(binding) - 1;
  source.frame_id = frame;
  source.frame_id_length = sizeof(frame) - 1;
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(mesh, &source) == PHX_MC_INVALID_INPUT);
  source.orientation = 0;
  constexpr char wrong_frame[] = "wrong-frame";
  source.frame_id = wrong_frame;
  source.frame_id_length = sizeof(wrong_frame) - 1;
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(mesh, &source) == PHX_MC_INVALID_ARGUMENT);
  source.frame_id = frame;
  source.frame_id_length = sizeof(frame) - 1;
  constexpr char wrong_identity[] = "different";
  source.source_id = wrong_identity;
  source.source_id_length = sizeof(wrong_identity) - 1;
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(mesh, &source) == PHX_MC_INVALID_ARGUMENT);
  source.source_id = source_id;
  source.source_id_length = sizeof(source_id) - 1;
  source.source_revision = wrong_identity;
  source.source_revision_length = sizeof(wrong_identity) - 1;
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(mesh, &source) == PHX_MC_INVALID_ARGUMENT);
  source.source_revision = revision;
  source.source_revision_length = sizeof(revision) - 1;
  source.binding_id = wrong_identity;
  source.binding_id_length = sizeof(wrong_identity) - 1;
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(mesh, &source) == PHX_MC_INVALID_ARGUMENT);
  source.binding_id = binding;
  source.binding_id_length = sizeof(binding) - 1;
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(mesh, &source) == PHX_MC_OK);
  phx_mc_tet_implicit_source snapshot{};
  PHX_CHECK(phx_mc_tet_mesh_implicit_source(mesh, &snapshot) == PHX_MC_OK);
  PHX_CHECK(snapshot.group_count == 1 && snapshot.groups[0] == 3);
  PHX_CHECK(snapshot.orientation == 1 && snapshot.tolerance == 0.01);
  PHX_CHECK(std::string_view(snapshot.frame_id, snapshot.frame_id_length) == frame);
  PHX_CHECK(snapshot.value.instruction_count == 1 && snapshot.value.instructions[0] == 2);
  value_code[0] = 99;
  groups[0] = 99;
  PHX_CHECK(snapshot.value.instructions[0] == 2 && snapshot.groups[0] == 3);
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(mesh, &source) != PHX_MC_OK);
  value_code[0] = 2;
  groups[0] = 3;
  phx_mc_tet_mesh* restored = nullptr;
  request.result = &restored;
  source.orientation = snapshot.orientation;
  PHX_CHECK(phx_mc_tet_mesh_create_with_implicit(&request, &source) == PHX_MC_OK);
  PHX_CHECK(restored != nullptr);
  phx_mc_tet_mesh_free(restored);
  restored = nullptr;
  value_workspace[3] = 1.0;
  PHX_CHECK(phx_mc_tet_mesh_create_with_implicit(&request, &source) == PHX_MC_UNRESOLVED);
  PHX_CHECK(restored == nullptr);
  value_workspace[3] = 0.0;
  constexpr std::array<double, 9> reflection{-1, 0, 0, 0, 1, 0, 0, 0, 1};
  PHX_CHECK(phx_mc_tet_mesh_create_with_affine_metric(&request, reflection.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(restored, &source) == PHX_MC_INVALID_ARGUMENT);
  phx_mc_tet_mesh_free(restored);
  // Restore a locally moved implicit face against its original non-moved PWL
  // source bank: the PWL constructor refuses, while the source-bound interval
  // owner admits every actual replacement face within the declared band.
  std::array<double, 15> source_points{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, .25, .25, 0};
  auto moved_points = source_points;
  moved_points[14] = 0.001;
  constexpr std::array<int32_t, 12> cells{0, 1, 4, 3, 1, 2, 4, 3, 2, 0, 4, 3};
  constexpr std::array<int32_t, 18> moved_faces{
      0, 1, 4, 1, 2, 4, 2, 0, 4, 1, 3, 2, 0, 2, 3, 0, 3, 1};
  constexpr std::array<int32_t, 6> moved_groups{0, 0, 0, 1, 2, 3};
  constexpr std::array<int64_t, 6> moved_source_ids{0, 0, 0, 1, 2, 3};
  std::array<double, 54> planes;
  planes.fill(std::numeric_limits<double>::quiet_NaN());
  constexpr std::array<double, 9> base_plane{0, 0, 0, 1, 0, 0, 0, 1, 0};
  constexpr std::array<double, 9> side_plane{1, 0, 0, -1, 0, 1, -1, 1, 0};
  for (std::size_t row = 0; row < 3; ++row) {
    std::copy(base_plane.begin(), base_plane.end(), planes.begin() + 9 * row);
  }
  std::copy(side_plane.begin(), side_plane.end(), planes.begin() + 27);
  constexpr std::array<double, 6> plane_tolerances{};
  phx_mc_tet_mesh_create_request moved{};
  moved.point_count = 5;
  moved.points = moved_points.data();
  moved.tetrahedron_count = 3;
  moved.tetrahedra = cells.data();
  moved.face_count = 6;
  moved.faces = moved_faces.data();
  moved.face_sources = moved_groups.data();
  moved.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  moved.identity_bound = 1;
  moved.source_id = source_id;
  moved.source_id_length = sizeof(source_id) - 1;
  moved.source_revision = revision;
  moved.source_revision_length = sizeof(revision) - 1;
  moved.source_map_id = binding;
  moved.source_map_id_length = sizeof(binding) - 1;
  moved.frame_id = frame;
  moved.frame_id_length = sizeof(frame) - 1;
  moved.source_point_count = 5;
  moved.source_points = source_points.data();
  moved.source_face_count = 6;
  moved.source_faces = moved_faces.data();
  moved.source_face_groups = moved_groups.data();
  moved.source_face_ids = moved_source_ids.data();
  moved.source_face_planes = planes.data();
  moved.source_face_plane_tolerances = plane_tolerances.data();
  restored = nullptr;
  moved.result = &restored;
  PHX_CHECK(phx_mc_tet_mesh_create(&moved) == PHX_MC_INVALID_INPUT);
  groups[0] = 0;
  source.orientation = 1;
  PHX_CHECK(phx_mc_tet_mesh_create_with_implicit(&moved, &source) == PHX_MC_OK);
  PHX_CHECK(restored != nullptr);
  phx_mc_tet_implicit_facet_refusal refusal{};
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 0);
  const std::array<uint64_t, 3> affected{0, 1, 2};
  const uint64_t moved_id = 4;
  std::array<double, 3> proposal{.25, .25, .1};
  phx_mc_tet_mesh_edit_request edit{};
  edit.identity_bound = 1;
  edit.source_id = source_id;
  edit.source_id_length = sizeof(source_id) - 1;
  edit.source_revision = revision;
  edit.source_revision_length = sizeof(revision) - 1;
  edit.source_map_id = binding;
  edit.source_map_id_length = sizeof(binding) - 1;
  edit.frame_id = frame;
  edit.frame_id_length = sizeof(frame) - 1;
  edit.kind = PHX_MC_TET_EDIT_RELOCATE;
  edit.removed_cell_count = affected.size();
  edit.removed_cell_ids = affected.data();
  edit.changed_vertex_count = 1;
  edit.changed_vertex_ids = &moved_id;
  edit.changed_vertex_points = proposal.data();
  int32_t decided = 0, edit_status = 0;
  PHX_CHECK(phx_mc_tet_mesh_probe_edit(restored, &edit, &decided, &edit_status) == PHX_MC_OK);
  PHX_CHECK(decided == 1 && edit_status == PHX_MC_TET_EDIT_UNCERTAIN_PREDICATE);
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 1 && refusal.generation == edit.expected_generation);
  PHX_CHECK(refusal.facet_id < 3 && refusal.facet_source == 0 && refusal.orientation == 1);
  PHX_CHECK(refusal.distance_refusals > 0 && refusal.status == edit_status);
  bool found_proposal = false;
  for (int corner = 0; corner < 3; ++corner) {
    if (refusal.vertex_ids[corner] == moved_id) {
      found_proposal = std::equal(proposal.begin(), proposal.end(), refusal.points + 3 * corner);
    }
  }
  PHX_CHECK(found_proposal);
  edit.kind = PHX_MC_TET_EDIT_SPLIT;
  PHX_CHECK(phx_mc_tet_mesh_probe_edit(restored, &edit, &decided, &edit_status) == PHX_MC_OK);
  PHX_CHECK(decided == 0);
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 0);
  edit.kind = PHX_MC_TET_EDIT_RELOCATE;
  PHX_CHECK(phx_mc_tet_mesh_probe_edit(restored, &edit, &decided, &edit_status) == PHX_MC_OK);
  PHX_CHECK(decided == 1);
  edit.expected_generation = 1;
  PHX_CHECK(phx_mc_tet_mesh_probe_edit(restored, &edit, &decided, &edit_status) == PHX_MC_OK);
  PHX_CHECK(decided == 0);
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 0);
  edit.expected_generation = 0;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(restored, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_UNCERTAIN_PREDICATE);
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 1 && refusal.distance_refusals > 0);
  PHX_CHECK(phx_mc_tet_mesh_clear_implicit_facet_refusal(restored) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 0);
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(restored, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_UNCERTAIN_PREDICATE);
  PHX_CHECK(phx_mc_tet_mesh_bind_implicit(restored, &source) != PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 0);
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(restored, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_UNCERTAIN_PREDICATE);
  proposal[2] = .001;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(restored, &edit, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(phx_mc_tet_mesh_implicit_facet_refusal(restored, &refusal) == PHX_MC_OK);
  PHX_CHECK(refusal.refused == 0);
  phx_mc_tet_mesh_free(restored);
  restored = nullptr;
  moved_points[3] = 1.001;
  PHX_CHECK(phx_mc_tet_mesh_create_with_implicit(&moved, &source) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(restored == nullptr);
  moved_points[3] = 1.0;
  restored = nullptr;
  moved_points[14] = 0.1;
  PHX_CHECK(phx_mc_tet_mesh_create_with_implicit(&moved, &source) == PHX_MC_UNRESOLVED);
  PHX_CHECK(restored == nullptr);
  using namespace phx::mc;
  constexpr std::array<int32_t, 3> regions{0, 0, 0};
  constexpr std::array<int8_t, 6> orientations{1, 1, 1, 1, 1, 1};
  constexpr std::array<double, 6> tolerances{};
  constexpr std::array<int32_t, 0> no_segments{};
  constexpr std::array<int64_t, 0> no_segment_ids{};
  constexpr std::array<double, 0> no_segment_tolerances{};
  constexpr std::array<SourceWitness, 0> no_witnesses{};
  const SourceComplexView source_view{
      source_points, moved_faces, moved_groups, moved_source_ids, tolerances,
      no_segments, no_segments, no_segment_ids, no_segment_tolerances, no_witnesses};
  const auto make_owner = [&]() {
    auto result = std::make_shared<TetImplicitSource>();
    const auto bind = [](remesh::IntervalBoxProgram& destination,
                         const phx_mc_interval_box_program& program) {
      return destination.bind(program.instructions, program.instruction_count,
                              program.indices, program.index_count, program.parameters,
                              program.parameter_count, program.workspace_lower,
                              program.workspace_upper, program.workspace_size,
                              program.outputs, program.output_count, program.guards,
                              program.guard_count);
    };
    PHX_CHECK(bind(result->value, source.value) == PHX_MC_OK);
    PHX_CHECK(bind(result->gradient, source.gradient) == PHX_MC_OK);
    result->groups = {0};
    result->tolerance = source.tolerance;
    result->source_id = source_id;
    result->source_revision = revision;
    result->binding_id = binding;
    result->frame_id = frame;
    return result;
  };
  TetMesh tracked(BoundaryPolicy::kConforming);
  PHX_CHECK(tracked.bind_source_identity(source_id, revision, binding, frame, true));
  PHX_CHECK(tracked.build(source_points, cells, regions, moved_faces, moved_groups,
                          orientations, no_segments, no_segments, {}, &source_view) == PHX_MC_OK);
  TetEditBatch stale{};
  stale.expected_generation = tracked.generation();
  stale.identity_bound = tracked.identity_bound();
  stale.source_id = tracked.source_id();
  stale.source_revision = tracked.source_revision();
  stale.source_map_id = tracked.source_map_id();
  stale.frame_id = tracked.frame_id();
  stale.periodic_identity = tracked.periodic_identity();
  PHX_CHECK(tracked.bind_implicit_source(make_owner()) == PHX_MC_OK);
  PHX_CHECK(tracked.generation() == stale.expected_generation + 1);
  std::vector<TetEditResult> stale_results;
  PHX_CHECK(tracked.apply_edit_batch(stale, stale_results) == TetEditStatus::kStaleGeneration);
  PHX_CHECK(!tracked.bind_source_identity("different", revision, binding, frame, true));
  TetEditVertex candidate{};
  candidate.id = 4;
  candidate.carrier = {.25, .25, .001};
  constexpr std::array<std::uint64_t, 3> star{0, 1, 2};
  PHX_CHECK(tracked.relocate(std::span(&candidate, 1), star) == TetEditStatus::kApplied);
  const auto accepted_generation = tracked.generation();
  candidate.carrier[2] = 0.1;
  PHX_CHECK(tracked.relocate(std::span(&candidate, 1), star) == TetEditStatus::kUncertainPredicate);
  PHX_CHECK(tracked.generation() == accepted_generation);
  PHX_CHECK(tracked.implicit_source()->distance_refusals > 0);
  PHX_CHECK(tracked.implicit_facet_refusal().has_value());
  if (tracked.implicit_facet_refusal()) {
    const auto& actual = *tracked.implicit_facet_refusal();
    PHX_CHECK(actual.generation == accepted_generation && actual.facet_source == 0);
    PHX_CHECK(actual.distance_refusals > 0 && actual.status == TetEditStatus::kUncertainPredicate);
    const auto found = std::find(actual.vertex_ids.begin(), actual.vertex_ids.end(), candidate.id);
    PHX_CHECK(found != actual.vertex_ids.end());
    if (found != actual.vertex_ids.end()) {
      PHX_CHECK(actual.points[static_cast<std::size_t>(found - actual.vertex_ids.begin())] ==
                candidate.carrier);
    }
  }
  PHX_CHECK(tracked.apply_edit_batch(stale, stale_results) == TetEditStatus::kStaleGeneration);
  PHX_CHECK(!tracked.implicit_facet_refusal());
  PHX_CHECK(tracked.relocate(std::span(&candidate, 1), star) == TetEditStatus::kUncertainPredicate);
  PHX_CHECK(!tracked.bind_source_identity("different", revision, binding, frame, true));
  PHX_CHECK(!tracked.implicit_facet_refusal());
  TetEditBatch staged_batch{};
  staged_batch.expected_generation = accepted_generation;
  staged_batch.identity_bound = tracked.identity_bound();
  staged_batch.source_id = tracked.source_id();
  staged_batch.source_revision = tracked.source_revision();
  staged_batch.source_map_id = tracked.source_map_id();
  staged_batch.frame_id = tracked.frame_id();
  staged_batch.periodic_identity = tracked.periodic_identity();
  TetEdit staged_move{};
  staged_move.kind = TetEditKind::kRelocate;
  staged_move.removed_cell_ids.assign(star.begin(), star.end());
  TetEditVertex staged_vertex = candidate;
  staged_vertex.carrier[2] = .002;
  staged_move.changed_vertices = {staged_vertex};
  staged_batch.operations.push_back(staged_move);
  staged_move.changed_vertices = {candidate};
  staged_batch.operations.push_back(staged_move);
  std::vector<TetEditResult> staged_results;
  PHX_CHECK(tracked.apply_edit_batch(staged_batch, staged_results) ==
            TetEditStatus::kUncertainPredicate);
  PHX_CHECK(staged_results.size() == 2 &&
            staged_results[1].status == TetEditStatus::kUncertainPredicate);
  PHX_CHECK(tracked.implicit_facet_refusal().has_value());
  if (tracked.implicit_facet_refusal()) {
    PHX_CHECK(tracked.implicit_facet_refusal()->generation == accepted_generation);
    PHX_CHECK(tracked.implicit_facet_refusal()->distance_refusals > 0);
  }
  TetMeshSnapshot accepted_snapshot;
  PHX_CHECK(tracked.snapshot(accepted_snapshot) == TetEditStatus::kApplied);
  PHX_CHECK(accepted_snapshot.vertices[4].carrier[2] == .001);
  TetEdit collapse{};
  collapse.kind = TetEditKind::kCollapse;
  collapse.collapse_pairs.push_back({4, 0});
  collapse.removed_vertex_ids = {4};
  collapse.removed_cell_ids = {0, 1, 2};
  TetEditCell coarse{};
  coarse.id = 3;
  coarse.vertices = {1, 2, 0, 3};
  coarse.facet_sources = {2, 3, 1, 0};
  coarse.region = 0;
  coarse.ancestor_cell_ids = {0, 1, 2};
  collapse.replacement_cells = {coarse};
  collapse.removed_facet_ids = {0, 1, 2};
  collapse.replacement_facets = {{6, {1, 2, 0}, 0, 1, false}};
  TetEditBatch batch{};
  batch.expected_generation = tracked.generation();
  batch.identity_bound = tracked.identity_bound();
  batch.source_id = tracked.source_id();
  batch.source_revision = tracked.source_revision();
  batch.source_map_id = tracked.source_map_id();
  batch.frame_id = tracked.frame_id();
  batch.periodic_identity = tracked.periodic_identity();
  batch.operations.push_back(collapse);
  std::vector<TetEditResult> results;
  PHX_CHECK(tracked.apply_edit_batch(batch, results) == TetEditStatus::kApplied);
  PHX_CHECK(results.size() == 1 && results.front().status == TetEditStatus::kApplied);
  TetMesh protected_mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(protected_mesh.bind_source_identity(source_id, revision, binding, frame, true));
  constexpr std::array<int8_t, 5> protected_vertices{0, 0, 0, 0, 1};
  PHX_CHECK(protected_mesh.build(source_points, cells, regions, moved_faces, moved_groups,
                                orientations, no_segments, no_segments, {}, &source_view,
                                protected_vertices) == PHX_MC_OK);
  PHX_CHECK(protected_mesh.bind_implicit_source(make_owner()) == PHX_MC_OK);
  candidate.carrier[2] = .001;
  PHX_CHECK(protected_mesh.relocate(std::span(&candidate, 1), star) ==
            TetEditStatus::kRejectedProtected);
  phx_mc_tet_mesh_free(mesh);
}

// A real bank compared by bit pattern, so undeclared (NaN) plane rows compare
// equal to themselves and signed zeros stay distinct.
struct ExactReals : std::vector<double> {
  bool operator==(const ExactReals& other) const {
    return std::equal(begin(), end(), other.begin(), other.end(), [](double a, double b) {
      return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b);
    });
  }
};

// Every non-periodic snapshot bank of a mesh, compared exactly across modes.
struct TetMeshBanks {
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> sizes{};
  uint64_t generation = 0;
  std::vector<uint64_t> vertex_ids, cell_ids, cell_vertices, facet_ids, facet_vertices,
      segment_ids, segment_vertices, edge_witness_endpoints, vertex_ancestor_ids,
      cell_ancestor_ids;
  ExactReals points, vertex_sizes, cell_sizes, witness_parameters, witness_deviations,
      protection_radii, edge_parameters, vertex_ancestry_weights, source_points,
      source_face_tolerances, source_face_planes, source_face_plane_tolerances,
      source_segment_tolerances;
  std::vector<int8_t> witness_strata, vertex_dimensions, protected_vertices,
      edge_witness_present, facet_orientations, protected_facets, protected_segments;
  std::vector<int32_t> witness_entities, cell_regions, cell_edit_kinds, facet_sources,
      segment_sources, source_faces, source_face_groups, source_segments,
      source_segment_groups;
  std::vector<int64_t> vertex_ancestry_offsets, vertex_ancestor_winding, cell_ancestry_offsets,
      source_face_ids, source_segment_ids;

  bool operator==(const TetMeshBanks&) const = default;
};

TetMeshBanks capture_banks(const phx_mc_tet_mesh* mesh) {
  TetMeshBanks banks;
  PHX_CHECK(phx_mc_tet_mesh_snapshot_sizes(mesh, banks.sizes.data()) == PHX_MC_OK);
  const auto rows = [&](int bank, int width) {
    return static_cast<std::size_t>(banks.sizes[static_cast<std::size_t>(bank)] * width);
  };
  banks.vertex_ids.resize(rows(0, 1));
  banks.points.resize(rows(0, 3));
  banks.vertex_sizes.resize(rows(0, 1));
  banks.witness_strata.resize(rows(0, 1));
  banks.witness_entities.resize(rows(0, 1));
  banks.witness_parameters.resize(rows(0, 2));
  banks.witness_deviations.resize(rows(0, 1));
  banks.vertex_dimensions.resize(rows(0, 1));
  banks.protection_radii.resize(rows(0, 1));
  banks.protected_vertices.resize(rows(0, 1));
  banks.edge_witness_present.resize(rows(0, 1));
  banks.edge_witness_endpoints.resize(rows(0, 2));
  banks.edge_parameters.resize(rows(0, 1));
  banks.vertex_ancestry_offsets.resize(rows(0, 1) + 1);
  banks.vertex_ancestor_ids.resize(rows(17, 1));
  banks.vertex_ancestor_winding.resize(rows(17, 3));
  banks.vertex_ancestry_weights.resize(rows(17, 1));
  banks.cell_ancestry_offsets.resize(rows(1, 1) + 1);
  banks.cell_ancestor_ids.resize(rows(18, 1));
  banks.cell_edit_kinds.resize(rows(1, 1));
  banks.cell_ids.resize(rows(1, 1));
  banks.cell_vertices.resize(rows(1, 4));
  banks.cell_regions.resize(rows(1, 1));
  banks.cell_sizes.resize(rows(1, 1));
  banks.facet_ids.resize(rows(2, 1));
  banks.facet_vertices.resize(rows(2, 3));
  banks.facet_sources.resize(rows(2, 1));
  banks.facet_orientations.resize(rows(2, 1));
  banks.protected_facets.resize(rows(2, 1));
  banks.segment_ids.resize(rows(3, 1));
  banks.segment_vertices.resize(rows(3, 2));
  banks.segment_sources.resize(rows(3, 1));
  banks.protected_segments.resize(rows(3, 1));
  banks.source_points.resize(rows(4, 3));
  banks.source_faces.resize(rows(5, 3));
  banks.source_face_groups.resize(rows(5, 1));
  banks.source_face_ids.resize(rows(5, 1));
  banks.source_face_tolerances.resize(rows(5, 1));
  banks.source_face_planes.resize(rows(5, 9));
  banks.source_face_plane_tolerances.resize(rows(5, 1));
  banks.source_segments.resize(rows(6, 2));
  banks.source_segment_groups.resize(rows(6, 1));
  banks.source_segment_ids.resize(rows(6, 1));
  banks.source_segment_tolerances.resize(rows(6, 1));
  phx_mc_tet_mesh_snapshot_request snapshot{};
  snapshot.vertex_capacity = banks.sizes[0];
  snapshot.cell_capacity = banks.sizes[1];
  snapshot.facet_capacity = banks.sizes[2];
  snapshot.segment_capacity = banks.sizes[3];
  snapshot.source_point_capacity = banks.sizes[4];
  snapshot.source_face_capacity = banks.sizes[5];
  snapshot.source_segment_capacity = banks.sizes[6];
  snapshot.vertex_ancestry_capacity = banks.sizes[17];
  snapshot.cell_ancestor_capacity = banks.sizes[18];
  snapshot.vertex_ids = banks.vertex_ids.data();
  snapshot.points = banks.points.data();
  snapshot.vertex_size_targets = banks.vertex_sizes.data();
  snapshot.witness_strata = banks.witness_strata.data();
  snapshot.witness_entities = banks.witness_entities.data();
  snapshot.witness_parameters = banks.witness_parameters.data();
  snapshot.witness_deviations = banks.witness_deviations.data();
  snapshot.vertex_dimensions = banks.vertex_dimensions.data();
  snapshot.protection_radii = banks.protection_radii.data();
  snapshot.protected_vertices = banks.protected_vertices.data();
  snapshot.vertex_edge_witness_present = banks.edge_witness_present.data();
  snapshot.vertex_edge_witness_endpoints = banks.edge_witness_endpoints.data();
  snapshot.vertex_edge_parameters = banks.edge_parameters.data();
  snapshot.vertex_ancestry_offsets = banks.vertex_ancestry_offsets.data();
  snapshot.vertex_ancestor_ids = banks.vertex_ancestor_ids.data();
  snapshot.vertex_ancestor_winding = banks.vertex_ancestor_winding.data();
  snapshot.vertex_ancestry_weights = banks.vertex_ancestry_weights.data();
  snapshot.cell_ancestry_offsets = banks.cell_ancestry_offsets.data();
  snapshot.cell_ancestor_ids = banks.cell_ancestor_ids.data();
  snapshot.cell_ancestry_edit_kinds = banks.cell_edit_kinds.data();
  snapshot.cell_ids = banks.cell_ids.data();
  snapshot.cell_vertices = banks.cell_vertices.data();
  snapshot.cell_regions = banks.cell_regions.data();
  snapshot.cell_size_targets = banks.cell_sizes.data();
  snapshot.facet_ids = banks.facet_ids.data();
  snapshot.facet_vertices = banks.facet_vertices.data();
  snapshot.facet_sources = banks.facet_sources.data();
  snapshot.facet_orientations = banks.facet_orientations.data();
  snapshot.protected_facets = banks.protected_facets.data();
  snapshot.segment_ids = banks.segment_ids.data();
  snapshot.segment_vertices = banks.segment_vertices.data();
  snapshot.segment_sources = banks.segment_sources.data();
  snapshot.protected_segments = banks.protected_segments.data();
  snapshot.source_points = banks.source_points.data();
  snapshot.source_faces = banks.source_faces.data();
  snapshot.source_face_groups = banks.source_face_groups.data();
  snapshot.source_face_ids = banks.source_face_ids.data();
  snapshot.source_face_tolerances = banks.source_face_tolerances.data();
  snapshot.source_face_planes = banks.source_face_planes.data();
  snapshot.source_face_plane_tolerances = banks.source_face_plane_tolerances.data();
  snapshot.source_segments = banks.source_segments.data();
  snapshot.source_segment_groups = banks.source_segment_groups.data();
  snapshot.source_segment_ids = banks.source_segment_ids.data();
  snapshot.source_segment_tolerances = banks.source_segment_tolerances.data();
  snapshot.generation = &banks.generation;
  PHX_CHECK(phx_mc_tet_mesh_snapshot(mesh, &snapshot) == PHX_MC_OK);
  return banks;
}

// Statuses, quality rows, and published banks of one pass over the persistent
// tetrahedral-mesh C API entries.
struct TetHazardRun {
  std::vector<int32_t> statuses;
  std::vector<int64_t> counts;
  std::vector<uint64_t> quality_cells;
  std::vector<int32_t> quality_criteria;
  std::vector<int32_t> quality_reasons;
  ExactReals quality_values;
  std::vector<TetMeshBanks> banks;
  ExactReals reals;

  bool operator==(const TetHazardRun&) const = default;
};

// Inputs of one pass, built before any hazard mode is installed.
struct TetHazardInputs {
  std::array<double, 15> star_points;
  std::array<double, 15> candidate_points;
  std::array<double, 3> relocated_point;
  std::array<double, 12> corner_points;
};

void append_quality(TetHazardRun& run, const phx_mc_tet_mesh* mesh, uint64_t generation,
                    const double* candidate) {
  int64_t count = -1;
  const auto evaluate = [&](int64_t capacity, uint64_t* cells, int32_t* criteria,
                            int32_t* reasons, double* values) {
    return candidate == nullptr
               ? phx_mc_tet_mesh_evaluate_quality(mesh, generation, 0.5, 70.0, 0.9, capacity,
                                                  &count, cells, criteria, reasons, values)
               : phx_mc_tet_mesh_evaluate_candidate_quality(
                     mesh, generation, 5, candidate, 0.5, 70.0, 0.9, capacity, &count, cells,
                     criteria, reasons, values);
  };
  run.statuses.push_back(evaluate(0, nullptr, nullptr, nullptr, nullptr));
  run.counts.push_back(count);
  if (run.statuses.back() != PHX_MC_OK || count <= 0) {
    run.statuses.push_back(-1);
    return;
  }
  const std::size_t rows = static_cast<std::size_t>(count);
  const std::size_t offset = run.quality_cells.size();
  run.quality_cells.resize(offset + rows);
  run.quality_criteria.resize(offset + rows);
  run.quality_reasons.resize(offset + rows);
  run.quality_values.resize(offset + rows);
  run.statuses.push_back(evaluate(count, run.quality_cells.data() + offset,
                                  run.quality_criteria.data() + offset,
                                  run.quality_reasons.data() + offset,
                                  run.quality_values.data() + offset));
}

// Creates the interior-vertex star of the unit corner (affine create too),
// evaluates current and candidate quality, relocates the interior vertex,
// refines, and improves; then creates and binds the z = 0 implicit source.
TetHazardRun tet_hazard_run(const TetHazardInputs& inputs) {
  constexpr std::array<int32_t, 16> tetrahedra{1, 3, 2, 4, 0, 2, 3, 4, 0, 3, 1, 4, 0, 1, 2, 4};
  constexpr std::array<int32_t, 4> regions{0, 0, 0, 0};
  constexpr std::array<int32_t, 12> faces{1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
  constexpr std::array<int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr std::array<uint64_t, 5> vertex_ids{101, 211, 307, 409, 500};
  constexpr std::array<uint64_t, 4> cell_ids{9001, 9002, 9003, 9004};
  constexpr std::array<double, 9> metric{-1, 0, 0, 0, 2, 0, 0, 0, 1};
  constexpr std::array<double, 5> sizes{0.5, 0.5, 0.5, 0.5, 0.5};
  TetHazardRun run;
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 5;
  create.points = inputs.star_points.data();
  create.vertex_ids = vertex_ids.data();
  create.tetrahedron_count = 4;
  create.tetrahedra = tetrahedra.data();
  create.tetrahedron_regions = regions.data();
  create.cell_ids = cell_ids.data();
  create.face_count = 4;
  create.faces = faces.data();
  create.face_sources = face_sources.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  run.statuses.push_back(phx_mc_tet_mesh_create_with_affine_metric(&create, metric.data()));
  if (mesh != nullptr) {
    run.banks.push_back(capture_banks(mesh));
    phx_mc_tet_mesh_free(mesh);
    mesh = nullptr;
  }
  run.statuses.push_back(phx_mc_tet_mesh_create(&create));
  if (mesh != nullptr) {
    run.banks.push_back(capture_banks(mesh));
    append_quality(run, mesh, 0, nullptr);
    append_quality(run, mesh, 0, inputs.candidate_points.data());
    constexpr std::array<uint64_t, 1> moved_vertex{500};
    phx_mc_tet_mesh_edit_request relocation{};
    relocation.expected_generation = 0;
    relocation.kind = PHX_MC_TET_EDIT_RELOCATE;
    relocation.removed_cell_count = 4;
    relocation.removed_cell_ids = cell_ids.data();
    relocation.changed_vertex_count = 1;
    relocation.changed_vertex_ids = moved_vertex.data();
    relocation.changed_vertex_points = inputs.relocated_point.data();
    int32_t edit_status = -1;
    int32_t run_status = -1;
    std::array<int64_t, 5> edits{-1, -1, -1, -1, -1};
    double score_quantum = -1.0;
    run.statuses.push_back(phx_mc_tet_mesh_apply_edit(mesh, &relocation, &edit_status));
    run.statuses.push_back(edit_status);
    run.banks.push_back(capture_banks(mesh));
    run.statuses.push_back(phx_mc_tet_mesh_refine(
        mesh, run.banks.back().generation, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
        2.0, static_cast<int64_t>(sizes.size()), sizes.data(), &edit_status, &run_status,
        &edits[0], &edits[1]));
    run.statuses.push_back(edit_status);
    run.statuses.push_back(run_status);
    run.banks.push_back(capture_banks(mesh));
    run.statuses.push_back(phx_mc_tet_mesh_improve_quality(
        mesh, run.banks.back().generation, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
        2.0, 0.0, 0.0, 0, 0, nullptr, 0, nullptr, &edit_status, &run_status, &edits[2],
        &edits[3], &edits[4], &edits[1], &score_quantum));
    run.statuses.push_back(edit_status);
    run.statuses.push_back(run_status);
    run.counts.insert(run.counts.end(), edits.begin(), edits.end());
    run.reals.push_back(score_quantum);
    run.banks.push_back(capture_banks(mesh));
    phx_mc_tet_mesh_free(mesh);
    mesh = nullptr;
  }

  // f(x,y,z) = z, grad f = (0,0,1) on the unit corner's z = 0 source facet.
  constexpr std::array<int32_t, 4> corner_tetrahedron{0, 1, 2, 3};
  constexpr char source_id[] = "implicit-source";
  constexpr char binding[] = "binding";
  constexpr char revision[] = "revision";
  constexpr char frame[] = "world-frame";
  phx_mc_tet_mesh_create_request corner{};
  corner.point_count = 4;
  corner.points = inputs.corner_points.data();
  corner.tetrahedron_count = 1;
  corner.tetrahedra = corner_tetrahedron.data();
  corner.face_count = 4;
  corner.faces = faces.data();
  corner.face_sources = face_sources.data();
  corner.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  corner.identity_bound = 1;
  corner.source_id = source_id;
  corner.source_id_length = sizeof(source_id) - 1;
  corner.source_revision = revision;
  corner.source_revision_length = sizeof(revision) - 1;
  corner.source_map_id = binding;
  corner.source_map_id_length = sizeof(binding) - 1;
  corner.frame_id = frame;
  corner.frame_id_length = sizeof(frame) - 1;
  corner.result = &mesh;
  const std::array<int64_t, 8> value_code{2, 4, 1, 2, -1, 3, -1, 0};
  std::array<double, 5> value_workspace{0, 0, 0, 0, 0};
  const std::array<int64_t, 1> value_outputs{4};
  std::array<double, 6> gradient_workspace{0, 0, 0, 0, 0, 1};
  const std::array<int64_t, 3> gradient_outputs{3, 4, 5};
  const std::array<int32_t, 1> groups{3};
  phx_mc_tet_implicit_source source{};
  source.value = {value_code.data(), 1, nullptr, 0, nullptr, 0, value_workspace.data(),
                  value_workspace.data(), 5, value_outputs.data(), 1, nullptr, 0};
  source.gradient = {nullptr, 0, nullptr, 0, nullptr, 0, gradient_workspace.data(),
                     gradient_workspace.data(), 6, gradient_outputs.data(), 3, nullptr, 0};
  source.groups = groups.data();
  source.group_count = 1;
  source.tolerance = 0.01;
  source.orientation = 1;
  source.source_id = source_id;
  source.source_id_length = sizeof(source_id) - 1;
  source.source_revision = revision;
  source.source_revision_length = sizeof(revision) - 1;
  source.binding_id = binding;
  source.binding_id_length = sizeof(binding) - 1;
  source.frame_id = frame;
  source.frame_id_length = sizeof(frame) - 1;
  run.statuses.push_back(phx_mc_tet_mesh_create_with_implicit(&corner, &source));
  if (mesh != nullptr) {
    run.banks.push_back(capture_banks(mesh));
    phx_mc_tet_mesh_free(mesh);
    mesh = nullptr;
  }
  run.statuses.push_back(phx_mc_tet_mesh_create(&corner));
  if (mesh != nullptr) {
    source.orientation = 0;
    run.statuses.push_back(phx_mc_tet_mesh_bind_implicit(mesh, &source));
    phx_mc_tet_implicit_source bound{};
    run.statuses.push_back(phx_mc_tet_mesh_implicit_source(mesh, &bound));
    run.statuses.push_back(bound.orientation);
    run.reals.push_back(bound.tolerance);
    run.banks.push_back(capture_banks(mesh));
    phx_mc_tet_mesh_free(mesh);
  }
  return run;
}

// The persistent mesh entries give their round-to-nearest statuses, quality
// rows, and banks in every caller mode. A subnormal created point or relocation
// target lies outside the exact domain; a subnormal candidate point is
// evaluated as given.
void persistent_entries_under_floating_point_hazards() {
  TetHazardInputs regular{};
  regular.star_points = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.25, 0.25, 0.25};
  regular.candidate_points = regular.star_points;
  regular.candidate_points[12] = 0.125;
  regular.relocated_point = {0.125, 0.25, 0.25};
  regular.corner_points = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  TetHazardInputs subnormal = regular;
  subnormal.star_points[0] = 0x1p-1060;
  subnormal.corner_points[0] = 0x1p-1060;
  TetHazardInputs subnormal_motion = regular;
  subnormal_motion.candidate_points[12] = 0x1p-1060;
  subnormal_motion.relocated_point[0] = 0x1p-1060;

  const TetHazardRun reference = tet_hazard_run(regular);
  // [0] affine create, [1] create, [2..3] quality count/rows, [4..5] candidate
  // count/rows, [6..7] edit and its status, [8..10] refine, [11..13] improve,
  // [14] implicit create, [15] create, [16] bind, [17] implicit snapshot,
  // [18] bound orientation.
  PHX_CHECK(reference.statuses.size() == 19);
  PHX_CHECK(reference.banks.size() == 7);
  if (reference.statuses.size() == 19) {
    PHX_CHECK(reference.statuses[0] == PHX_MC_OK && reference.statuses[1] == PHX_MC_OK);
    PHX_CHECK(reference.statuses[2] == PHX_MC_OK && reference.statuses[3] == PHX_MC_OK);
    PHX_CHECK(reference.statuses[4] == PHX_MC_OK && reference.statuses[5] == PHX_MC_OK);
    PHX_CHECK(reference.statuses[6] == PHX_MC_OK &&
              reference.statuses[7] == PHX_MC_TET_EDIT_APPLIED);
    PHX_CHECK(reference.statuses[8] == PHX_MC_OK &&
              reference.statuses[9] == PHX_MC_TET_EDIT_APPLIED);
    PHX_CHECK(reference.statuses[11] == PHX_MC_OK &&
              reference.statuses[12] == PHX_MC_TET_EDIT_APPLIED);
    PHX_CHECK(reference.statuses[14] == PHX_MC_OK && reference.statuses[15] == PHX_MC_OK);
    PHX_CHECK(reference.statuses[16] == PHX_MC_OK && reference.statuses[17] == PHX_MC_OK);
    PHX_CHECK(reference.statuses[18] == 1);
  }
  PHX_CHECK(reference.counts.size() >= 2 && reference.counts[0] > 0 && reference.counts[1] > 0);
  if (reference.banks.size() == 7) {
    PHX_CHECK(std::equal(reference.banks[1].points.begin(), reference.banks[1].points.end(),
                         regular.star_points.begin(), regular.star_points.end()));
    PHX_CHECK(reference.banks[2].generation == 1 &&
              reference.banks[2].points[12] == regular.relocated_point[0]);
  }
  const TetHazardRun subnormal_reference = tet_hazard_run(subnormal);
  PHX_CHECK(subnormal_reference.statuses ==
            (std::vector<int32_t>{PHX_MC_RANGE_ERROR, PHX_MC_RANGE_ERROR, PHX_MC_RANGE_ERROR,
                                  PHX_MC_RANGE_ERROR}));
  const TetHazardRun motion_reference = tet_hazard_run(subnormal_motion);
  PHX_CHECK(motion_reference.statuses.size() == 19);
  if (motion_reference.statuses.size() == 19) {
    PHX_CHECK(motion_reference.statuses[4] == PHX_MC_OK);
    PHX_CHECK(motion_reference.statuses[6] == PHX_MC_OK &&
              motion_reference.statuses[7] == PHX_MC_TET_EDIT_INVALID_GEOMETRY);
  }
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    PHX_CHECK(tet_hazard_run(regular) == reference);
    PHX_CHECK(tet_hazard_run(subnormal) == subnormal_reference);
    PHX_CHECK(tet_hazard_run(subnormal_motion) == motion_reference);
  });
}

}  // namespace

int main() {
  implicit_source_binding_owns_programs_and_refuses_wrong_orientation();
  persistent_entries_under_floating_point_hazards();
  selective_facet_protection_is_atomic_and_source_bound();
  affine_create_c_api_guards();
  persistent_create_snapshot_and_rejection();
  persistent_periodic_state_and_atomic_relocation();
  persistent_joint_quality_transition_preserves_identity();
  persistent_collapse_replaces_vertex_star();
  refinement_preserves_source_region_conflict_refusal();
  persistent_quality_query_is_read_only();
  retriangulation_rejects_undeclared_region_jumps();
  retriangulation_rejects_closed_interface_region_relabel();
  grouped_orthogonal_source_faces_validate_each_contributor();
  relocation_preserves_or_clears_edge_witnesses_explicitly();
  replacement_facet_requires_its_own_source_geometry();
  split_edge_preserves_exact_ghost_boundary_coverage();
  split_on_shared_source_diagonal_needs_an_exact_edge_point();
  boundary_vertex_moves_and_collapses_within_its_source_facet();
  collapse_relocates_its_survivor_within_its_source_facet();
  relocation_checks_incident_source_facets_atomically();
  exact_boundary_subface_coverage_rejects_gap_reversal_and_overlap();
  declared_plane_admission_refuses_and_records_vertices();
  periodic_closure_accepts_edit_touching_one_orbit_copy();
  return phx::mc::test::finish("test_tet_mesh");
}
