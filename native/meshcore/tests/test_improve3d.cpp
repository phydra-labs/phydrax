//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <bit>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <map>
#include <limits>
#include <span>
#include <vector>
#include <set>
#include <utility>

#include "check.hpp"
#include "cavity.hpp"
#include "geometry3.hpp"
#include "improve3d.hpp"
#include "local_incidence.hpp"
#include "refine3d.hpp"
#include "phydrax_meshcore.h"
#include "plc3d.hpp"
#include "tet_mesh.hpp"

namespace {

using namespace phx::mc;

bool minimum_quality(const TetMesh& mesh, const TetMeshSnapshot& snapshot,
                     double& minimum_angle, double& minimum_determinant,
                     double& maximum_radius_edge) {
  minimum_angle = 180.0;
  minimum_determinant = std::numeric_limits<double>::infinity();
  maximum_radius_edge = 0.0;
  if (snapshot.cells.empty()) return false;
  for (const TetSnapshotCell& cell : snapshot.cells) {
    std::array<std::int32_t, 4> slots{};
    std::array<const double*, 4> points{};
    for (int corner = 0; corner < 4; ++corner) {
      slots[static_cast<std::size_t>(corner)] =
          mesh.vertex_slot(cell.vertices[static_cast<std::size_t>(corner)]);
      if (!mesh.live_vertex(slots[static_cast<std::size_t>(corner)])) return false;
      points[static_cast<std::size_t>(corner)] =
          mesh.point(slots[static_cast<std::size_t>(corner)]);
    }
    if (!mesh.positive(slots.data())) return false;
    geometry::TetrahedronShape shape;
    if (!geometry::tetrahedron_shape(points.data(), shape)) return false;
    const double angle = geometry::minimum_dihedral_degrees(points.data());
    const double determinant = geometry::relative_determinant(points.data());
    if (!std::isfinite(angle) || !std::isfinite(determinant) ||
        !std::isfinite(shape.radius_edge_ratio)) return false;
    minimum_angle = std::min(minimum_angle, angle);
    minimum_determinant = std::min(minimum_determinant, determinant);
    maximum_radius_edge = std::max(maximum_radius_edge, shape.radius_edge_ratio);
  }
  return true;
}

double pairwise_minimum_dihedral_reference(
    const std::array<const double*, 4>& points) {
  constexpr int faces[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  std::array<geometry::Vector3, 4> normals{};
  for (int face = 0; face < 4; ++face) {
    const int* vertices = faces[face];
    const geometry::Vector3 first_edge{
        points[vertices[1]][0] - points[vertices[0]][0],
        points[vertices[1]][1] - points[vertices[0]][1],
        points[vertices[1]][2] - points[vertices[0]][2]};
    const geometry::Vector3 second_edge{
        points[vertices[2]][0] - points[vertices[0]][0],
        points[vertices[2]][1] - points[vertices[0]][1],
        points[vertices[2]][2] - points[vertices[0]][2]};
    normals[static_cast<std::size_t>(face)] =
        geometry::cross(first_edge, second_edge);
    const double magnitude =
        std::sqrt(geometry::dot(normals[static_cast<std::size_t>(face)],
                                normals[static_cast<std::size_t>(face)]));
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) return 0.0;
    geometry::Vector3& normal = normals[static_cast<std::size_t>(face)];
    normal.x /= magnitude;
    normal.y /= magnitude;
    normal.z /= magnitude;
  }
  double minimum = 180.0;
  constexpr double kRadiansToDegrees = 57.295779513082320876798154814105;
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      const double cosine = std::clamp(
          -geometry::dot(normals[static_cast<std::size_t>(first)],
                         normals[static_cast<std::size_t>(second)]),
          -1.0, 1.0);
      minimum = std::min(minimum, std::acos(cosine) * kRadiansToDegrees);
    }
  }
  return std::isfinite(minimum) ? minimum : 0.0;
}

bool same_source_banks(const TetMeshSnapshot& first, const TetMeshSnapshot& second) {
  if (first.constrained_facets.size() != second.constrained_facets.size() ||
      first.constrained_segments.size() != second.constrained_segments.size()) {
    return false;
  }
  std::map<std::uint64_t, TetSnapshotFacet> facets;
  for (const TetSnapshotFacet& facet : first.constrained_facets) {
    facets.emplace(facet.id, facet);
  }
  for (const TetSnapshotFacet& facet : second.constrained_facets) {
    const auto found = facets.find(facet.id);
    if (found == facets.end() || found->second.vertices != facet.vertices ||
        found->second.source != facet.source ||
        found->second.orientation != facet.orientation ||
        found->second.protected_facet != facet.protected_facet) {
      return false;
    }
  }
  std::map<std::uint64_t, TetSnapshotSegment> segments;
  for (const TetSnapshotSegment& segment : first.constrained_segments) {
    segments.emplace(segment.id, segment);
  }
  for (const TetSnapshotSegment& segment : second.constrained_segments) {
    const auto found = segments.find(segment.id);
    if (found == segments.end() || found->second.vertices != segment.vertices ||
        found->second.source != segment.source ||
        found->second.protected_segment != segment.protected_segment) {
      return false;
    }
  }
  return true;
}

bool same_vertex_carriers(const TetMeshSnapshot& first, const TetMeshSnapshot& second,
                          std::span<const std::uint64_t> source_vertex_ids) {
  const auto carrier = [](const TetMeshSnapshot& snapshot, std::uint64_t id)
      -> const std::array<double, 3>* {
    const auto found = std::find_if(snapshot.vertices.begin(), snapshot.vertices.end(),
                                    [id](const TetSnapshotVertex& vertex) {
                                      return vertex.id == id;
                                    });
    return found == snapshot.vertices.end() ? nullptr : &found->carrier;
  };
  for (std::uint64_t id : source_vertex_ids) {
    const auto* first_carrier = carrier(first, id);
    const auto* second_carrier = carrier(second, id);
    if (first_carrier == nullptr || second_carrier == nullptr ||
        *first_carrier != *second_carrier) {
      return false;
    }
  }
  return true;
}

bool same_vertex_state(const TetSnapshotVertex& first, const TetSnapshotVertex& second) {
  if (first.carrier != second.carrier || first.stratum_dimension != second.stratum_dimension ||
      first.protection_radius != second.protection_radius ||
      first.protected_vertex != second.protected_vertex ||
      first.has_edge_parameter != second.has_edge_parameter ||
      first.edge_endpoints != second.edge_endpoints ||
      first.edge_parameter != second.edge_parameter ||
      first.witness.stratum != second.witness.stratum ||
      first.witness.entity != second.witness.entity ||
      first.witness.parameters[0] != second.witness.parameters[0] ||
      first.witness.parameters[1] != second.witness.parameters[1] ||
      first.witness.deviation != second.witness.deviation ||
      first.ancestry.size() != second.ancestry.size()) {
    return false;
  }
  for (std::size_t row = 0; row < first.ancestry.size(); ++row) {
    if (first.ancestry[row].source_vertex_id != second.ancestry[row].source_vertex_id ||
        first.ancestry[row].source_winding != second.ancestry[row].source_winding ||
        first.ancestry[row].weight != second.ancestry[row].weight) {
      return false;
    }
  }
  return true;
}

bool same_mesh_state(const TetMeshSnapshot& first, const TetMeshSnapshot& second) {
  if (first.generation != second.generation || first.identity_bound != second.identity_bound ||
      first.source_id != second.source_id || first.source_revision != second.source_revision ||
      first.frame_id != second.frame_id || first.source_map_id != second.source_map_id ||
      first.periodic_identity != second.periodic_identity ||
      first.periodic_orbits.size() != second.periodic_orbits.size() ||
      first.quotient_cells.size() != second.quotient_cells.size() ||
      first.periodic_entity_maps.size() != second.periodic_entity_maps.size() ||
      first.vertices.size() != second.vertices.size() || first.cells.size() != second.cells.size() ||
      first.source_points != second.source_points || first.source_faces != second.source_faces ||
      first.source_face_groups != second.source_face_groups ||
      first.source_face_ids != second.source_face_ids ||
      first.source_face_tolerances != second.source_face_tolerances ||
      first.source_segments != second.source_segments ||
      first.source_segment_groups != second.source_segment_groups ||
      first.source_segment_ids != second.source_segment_ids ||
      first.source_segment_tolerances != second.source_segment_tolerances ||
      !same_source_banks(first, second)) {
    return false;
  }
  std::map<std::uint64_t, const TetSnapshotVertex*> vertices;
  for (const TetSnapshotVertex& vertex : first.vertices) vertices.emplace(vertex.id, &vertex);
  for (const TetSnapshotVertex& vertex : second.vertices) {
    const auto found = vertices.find(vertex.id);
    if (found == vertices.end() || !same_vertex_state(*found->second, vertex)) return false;
  }
  std::map<std::uint64_t, const TetSnapshotCell*> cells;
  for (const TetSnapshotCell& cell : first.cells) cells.emplace(cell.id, &cell);
  for (const TetSnapshotCell& cell : second.cells) {
    const auto found = cells.find(cell.id);
    if (found == cells.end()) return false;
    const TetSnapshotCell& prior = *found->second;
    if (prior.vertices != cell.vertices || prior.region != cell.region ||
        prior.ancestor_cell_ids != cell.ancestor_cell_ids || prior.edit_kind != cell.edit_kind) {
      return false;
    }
  }
  return true;
}
struct ImprovementFixture {
  std::vector<double> points;
  std::vector<std::array<std::int32_t, 4>> cells;
  std::vector<std::array<std::int32_t, 3>> faces;
  std::vector<std::int32_t> face_sources;
};

ImprovementFixture make_improvement_fixture(
    std::vector<double> points,
    std::vector<std::array<std::int32_t, 4>> cells) {
  ImprovementFixture fixture;
  fixture.points = std::move(points);
  fixture.cells = std::move(cells);
  struct FaceUse {
    std::array<std::int32_t, 3> oriented{};
    std::size_t count = 0;
  };
  std::map<std::array<std::int32_t, 3>, FaceUse> uses;
  for (std::array<std::int32_t, 4>& cell : fixture.cells) {
    const auto point = [&](std::int32_t vertex) {
      return fixture.points.data() + 3 * vertex;
    };
    const std::int32_t orientation =
        orient3d(point(cell[0]), point(cell[1]), point(cell[2]), point(cell[3]));
    if (orientation == 0) return {};
    if (orientation < 0) std::swap(cell[0], cell[1]);
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      std::array<std::int32_t, 3> face{};
      oriented_facet(cell.data(), face_slot, face.data());
      auto key = face;
      std::sort(key.begin(), key.end());
      FaceUse& use = uses[key];
      if (use.count == 0) use.oriented = face;
      ++use.count;
    }
  }
  for (const auto& [key, use] : uses) {
    (void)key;
    if (use.count != 1) continue;
    fixture.faces.push_back(use.oriented);
    fixture.face_sources.push_back(
        static_cast<std::int32_t>(fixture.face_sources.size()));
  }
  return fixture;
}

ImprovementFixture make_bipyramid_fixture(std::span<const double> radii,
                                          double half_height) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const std::size_t ring_size = radii.size();
  std::vector<double> points(3 * (ring_size + 2));
  for (std::size_t vertex = 0; vertex < ring_size; ++vertex) {
    const double angle = kTwoPi * static_cast<double>(vertex) /
                         static_cast<double>(ring_size);
    points[3 * vertex] = radii[vertex] * std::cos(angle);
    points[3 * vertex + 1] = radii[vertex] * std::sin(angle);
  }
  points[3 * ring_size + 2] = -half_height;
  points[3 * (ring_size + 1) + 2] = half_height;
  std::vector<std::array<std::int32_t, 4>> cells;
  cells.reserve(ring_size);
  const auto first_apex = static_cast<std::int32_t>(ring_size);
  const auto second_apex = static_cast<std::int32_t>(ring_size + 1);
  for (std::size_t edge = 0; edge < ring_size; ++edge) {
    cells.push_back({first_apex, second_apex,
                     static_cast<std::int32_t>(edge),
                     static_cast<std::int32_t>((edge + 1) % ring_size)});
  }
  return make_improvement_fixture(std::move(points), std::move(cells));
}

ImprovementFixture make_interior_collapse_fixture() {
  constexpr std::int32_t side = 5;
  constexpr std::int32_t grid_vertices = side * side * side;
  const auto vertex_id = [side](std::int32_t x, std::int32_t y,
                                std::int32_t z) {
    return (x * side + y) * side + z;
  };
  std::vector<double> raw_points(3 * grid_vertices);
  for (std::int32_t x = 0; x < side; ++x) {
    for (std::int32_t y = 0; y < side; ++y) {
      for (std::int32_t z = 0; z < side; ++z) {
        const std::int32_t id = vertex_id(x, y, z);
        raw_points[3 * static_cast<std::size_t>(id)] = static_cast<double>(x);
        raw_points[3 * static_cast<std::size_t>(id) + 1] = static_cast<double>(y);
        raw_points[3 * static_cast<std::size_t>(id) + 2] = static_cast<double>(z);
      }
    }
  }
  constexpr std::array<std::array<std::int32_t, 3>, 6> axis_orders{{
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
      {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
  std::vector<std::array<std::int32_t, 4>> raw_cells;
  for (std::int32_t x = 0; x + 1 < side; ++x) {
    for (std::int32_t y = 0; y + 1 < side; ++y) {
      for (std::int32_t z = 0; z + 1 < side; ++z) {
        for (const auto& order : axis_orders) {
          std::array<std::int32_t, 3> coordinate{x, y, z};
          std::array<std::int32_t, 4> cell{};
          cell[0] = vertex_id(coordinate[0], coordinate[1], coordinate[2]);
          for (int step = 0; step < 3; ++step) {
            ++coordinate[static_cast<std::size_t>(order[static_cast<std::size_t>(step)])];
            cell[static_cast<std::size_t>(step + 1)] =
                vertex_id(coordinate[0], coordinate[1], coordinate[2]);
          }
          raw_cells.push_back(cell);
        }
      }
    }
  }

  std::map<std::array<std::int32_t, 3>, std::vector<std::size_t>> face_owners;
  for (std::size_t row = 0; row < raw_cells.size(); ++row) {
    for (int opposite = 0; opposite < 4; ++opposite) {
      std::array<std::int32_t, 3> face{};
      int corner = 0;
      for (int index = 0; index < 4; ++index) {
        if (index != opposite) face[static_cast<std::size_t>(corner++)] =
            raw_cells[row][static_cast<std::size_t>(index)];
      }
      std::sort(face.begin(), face.end());
      face_owners[face].push_back(row);
    }
  }
  const auto strictly_interior = [&](std::int32_t vertex) {
    const std::int32_t x = vertex / (side * side);
    const std::int32_t y = (vertex / side) % side;
    const std::int32_t z = vertex % side;
    return x > 0 && x + 1 < side && y > 0 && y + 1 < side &&
           z > 0 && z + 1 < side;
  };
  std::array<std::int32_t, 3> shared_face{};
  std::array<std::size_t, 2> parent_rows{};
  bool found_parent = false;
  for (const auto& [face, owners] : face_owners) {
    if (owners.size() != 2 ||
        !std::all_of(face.begin(), face.end(), strictly_interior)) {
      continue;
    }
    if (!std::all_of(owners.begin(), owners.end(), [&](std::size_t row) {
          return std::all_of(raw_cells[row].begin(), raw_cells[row].end(),
                             strictly_interior);
        })) {
      continue;
    }
    shared_face = face;
    parent_rows = {owners[0], owners[1]};
    found_parent = true;
    break;
  }
  if (!found_parent) return {};

  std::array<double, 3> center{};
  for (std::int32_t vertex : shared_face) {
    for (int axis = 0; axis < 3; ++axis) {
      center[static_cast<std::size_t>(axis)] +=
          raw_points[3 * static_cast<std::size_t>(vertex) +
                     static_cast<std::size_t>(axis)] / 3.0;
    }
  }
  const std::int32_t target = shared_face[0];
  for (int axis = 0; axis < 3; ++axis) {
    const double target_coordinate =
        raw_points[3 * static_cast<std::size_t>(target) +
                   static_cast<std::size_t>(axis)];
    center[static_cast<std::size_t>(axis)] =
        target_coordinate + 0.02 * (center[static_cast<std::size_t>(axis)] -
                                    target_coordinate);
  }
  raw_points.insert(raw_points.end(), center.begin(), center.end());
  const std::int32_t center_id = grid_vertices;
  std::vector<std::array<std::int32_t, 4>> cells;
  cells.reserve(raw_cells.size() + 4);
  for (std::size_t row = 0; row < raw_cells.size(); ++row) {
    if (row != parent_rows[0] && row != parent_rows[1]) {
      cells.push_back(raw_cells[row]);
    }
  }
  for (std::size_t row : parent_rows) {
    const std::int32_t opposite = *std::find_if(
        raw_cells[row].begin(), raw_cells[row].end(),
        [&](std::int32_t vertex) {
          return !std::binary_search(shared_face.begin(), shared_face.end(),
                                     vertex);
        });
    cells.push_back({center_id, shared_face[0], shared_face[1], opposite});
    cells.push_back({center_id, shared_face[1], shared_face[2], opposite});
    cells.push_back({center_id, shared_face[2], shared_face[0], opposite});
  }

  std::vector<std::int32_t> logical_ids(static_cast<std::size_t>(grid_vertices + 1));
  logical_ids[static_cast<std::size_t>(center_id)] = 0;
  logical_ids[static_cast<std::size_t>(target)] = 1;
  std::int32_t next_id = 2;
  for (std::int32_t vertex = 0; vertex < grid_vertices; ++vertex) {
    if (vertex != target) logical_ids[static_cast<std::size_t>(vertex)] = next_id++;
  }
  std::vector<double> points(raw_points.size());
  for (std::int32_t vertex = 0; vertex <= grid_vertices; ++vertex) {
    const std::int32_t logical = logical_ids[static_cast<std::size_t>(vertex)];
    std::copy_n(raw_points.data() + 3 * static_cast<std::size_t>(vertex), 3,
                points.data() + 3 * static_cast<std::size_t>(logical));
  }
  for (std::array<std::int32_t, 4>& cell : cells) {
    for (std::int32_t& vertex : cell) {
      vertex = logical_ids[static_cast<std::size_t>(vertex)];
    }
  }
  return make_improvement_fixture(std::move(points), std::move(cells));
}

std::int32_t build_improvement_mesh(
    TetMesh& mesh, const ImprovementFixture& fixture,
    std::span<const std::int32_t> regions_by_cell,
    std::span<const std::uint64_t> slot_vertex_ids,
    std::span<const std::uint64_t> row_cell_ids,
    std::span<const std::uint64_t> row_facet_ids,
    std::span<const std::int8_t> protected_by_vertex) {
  const std::size_t vertex_count = fixture.points.size() / 3;
  if (slot_vertex_ids.size() != vertex_count ||
      row_cell_ids.size() != fixture.cells.size() ||
      row_facet_ids.size() != fixture.faces.size() ||
      regions_by_cell.size() != fixture.cells.size() ||
      protected_by_vertex.size() != vertex_count) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::vector<std::size_t> local_vertex(vertex_count);
  std::vector<double> points(fixture.points.size());
  std::vector<std::int8_t> protected_vertices(vertex_count);
  for (std::size_t slot = 0; slot < vertex_count; ++slot) {
    const std::uint64_t id = slot_vertex_ids[slot];
    if (id >= vertex_count) return PHX_MC_INVALID_ARGUMENT;
    local_vertex[static_cast<std::size_t>(id)] = slot;
    std::copy_n(fixture.points.data() + 3 * static_cast<std::size_t>(id), 3,
                points.data() + 3 * slot);
    protected_vertices[slot] = protected_by_vertex[static_cast<std::size_t>(id)];
  }
  std::vector<std::int32_t> tetrahedra(4 * fixture.cells.size());
  std::vector<std::int32_t> regions(fixture.cells.size());
  for (std::size_t row = 0; row < fixture.cells.size(); ++row) {
    const std::uint64_t id = row_cell_ids[row];
    if (id >= fixture.cells.size()) return PHX_MC_INVALID_ARGUMENT;
    regions[row] = regions_by_cell[static_cast<std::size_t>(id)];
    for (int corner = 0; corner < 4; ++corner) {
      const std::int32_t vertex =
          fixture.cells[static_cast<std::size_t>(id)][static_cast<std::size_t>(corner)];
      tetrahedra[4 * row + static_cast<std::size_t>(corner)] =
          static_cast<std::int32_t>(local_vertex[static_cast<std::size_t>(vertex)]);
    }
  }
  std::vector<std::int32_t> faces(3 * fixture.faces.size());
  std::vector<std::int32_t> face_sources(fixture.faces.size());
  for (std::size_t row = 0; row < fixture.faces.size(); ++row) {
    const std::uint64_t id = row_facet_ids[row];
    if (id >= fixture.faces.size()) return PHX_MC_INVALID_ARGUMENT;
    face_sources[row] = fixture.face_sources[static_cast<std::size_t>(id)];
    for (int corner = 0; corner < 3; ++corner) {
      const std::int32_t vertex =
          fixture.faces[static_cast<std::size_t>(id)][static_cast<std::size_t>(corner)];
      faces[3 * row + static_cast<std::size_t>(corner)] =
          static_cast<std::int32_t>(local_vertex[static_cast<std::size_t>(vertex)]);
    }
  }
  constexpr std::array<std::int32_t, 0> no_indices{};
  constexpr std::array<std::int8_t, 0> no_orientations{};
  constexpr std::array<double, 0> no_protection_radii{};
  const std::int32_t status = mesh.build(
      points, tetrahedra, regions, faces, face_sources, no_orientations,
      no_indices, no_indices, no_protection_radii, nullptr, protected_vertices);
  if (status != PHX_MC_OK) return status;
  TetMeshSnapshot unassigned;
  if (mesh.snapshot(unassigned) != TetEditStatus::kApplied) {
    return PHX_MC_INTERNAL_ERROR;
  }
  using CellKey = std::array<std::int32_t, 4>;
  using FaceKey = std::array<std::int32_t, 3>;
  std::map<CellKey, std::uint64_t> canonical_cells;
  for (std::size_t id = 0; id < fixture.cells.size(); ++id) {
    CellKey key = fixture.cells[id];
    std::sort(key.begin(), key.end());
    canonical_cells.emplace(key, static_cast<std::uint64_t>(id));
  }
  std::vector<std::uint64_t> storage_cell_ids;
  storage_cell_ids.reserve(unassigned.cells.size());
  for (const TetSnapshotCell& cell : unassigned.cells) {
    CellKey key{};
    for (int corner = 0; corner < 4; ++corner) {
      const std::uint64_t local = cell.vertices[static_cast<std::size_t>(corner)];
      key[static_cast<std::size_t>(corner)] =
          static_cast<std::int32_t>(slot_vertex_ids[static_cast<std::size_t>(local)]);
    }
    std::sort(key.begin(), key.end());
    const auto found = canonical_cells.find(key);
    if (found == canonical_cells.end()) return PHX_MC_INTERNAL_ERROR;
    storage_cell_ids.push_back(found->second);
  }
  std::map<FaceKey, std::uint64_t> canonical_facets;
  for (std::size_t id = 0; id < fixture.faces.size(); ++id) {
    FaceKey key = fixture.faces[id];
    std::sort(key.begin(), key.end());
    canonical_facets.emplace(key, static_cast<std::uint64_t>(id));
  }
  std::vector<std::uint64_t> storage_facet_ids;
  storage_facet_ids.reserve(unassigned.constrained_facets.size());
  for (const TetSnapshotFacet& facet : unassigned.constrained_facets) {
    FaceKey key{};
    for (int corner = 0; corner < 3; ++corner) {
      const std::uint64_t local = facet.vertices[static_cast<std::size_t>(corner)];
      key[static_cast<std::size_t>(corner)] =
          static_cast<std::int32_t>(slot_vertex_ids[static_cast<std::size_t>(local)]);
    }
    std::sort(key.begin(), key.end());
    const auto found = canonical_facets.find(key);
    if (found == canonical_facets.end()) return PHX_MC_INTERNAL_ERROR;
    storage_facet_ids.push_back(found->second);
  }
  if (storage_cell_ids.size() != fixture.cells.size() ||
      storage_facet_ids.size() != fixture.faces.size()) {
    return PHX_MC_INTERNAL_ERROR;
  }
  return mesh.assign_initial_ids(slot_vertex_ids, storage_cell_ids,
                                 storage_facet_ids,
                                 std::span<const std::uint64_t>{});
}

bool same_logical_snapshot(const TetMeshSnapshot& first,
                           const TetMeshSnapshot& second) {
  if (first.generation != second.generation ||
      first.identity_bound != second.identity_bound ||
      first.source_id != second.source_id ||
      first.source_revision != second.source_revision ||
      first.source_map_id != second.source_map_id ||
      first.frame_id != second.frame_id ||
      first.vertices.size() != second.vertices.size() ||
      first.cells.size() != second.cells.size() ||
      !same_source_banks(first, second)) {
    return false;
  }
  std::map<std::uint64_t, const TetSnapshotVertex*> vertices;
  for (const TetSnapshotVertex& vertex : first.vertices) {
    vertices.emplace(vertex.id, &vertex);
  }
  for (const TetSnapshotVertex& vertex : second.vertices) {
    const auto found = vertices.find(vertex.id);
    if (found == vertices.end() || !same_vertex_state(*found->second, vertex)) {
      return false;
    }
  }
  std::map<std::uint64_t, const TetSnapshotCell*> cells;
  for (const TetSnapshotCell& cell : first.cells) cells.emplace(cell.id, &cell);
  for (const TetSnapshotCell& cell : second.cells) {
    const auto found = cells.find(cell.id);
    if (found == cells.end()) return false;
    const TetSnapshotCell& prior = *found->second;
    if (prior.vertices != cell.vertices || prior.region != cell.region ||
        prior.ancestor_cell_ids != cell.ancestor_cell_ids ||
        prior.edit_kind != cell.edit_kind) {
      return false;
    }
  }
  return true;
}

std::vector<std::uint64_t> sequential_ids(std::size_t count) {
  std::vector<std::uint64_t> ids(count);
  for (std::size_t index = 0; index < count; ++index) {
    ids[index] = static_cast<std::uint64_t>(index);
  }
  return ids;
}


void convex_bipyramid_reports_unresolved_radius_edge() {
  constexpr std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.42, 0.17, 0.08, -0.2, -0.01, -1.0};
  constexpr std::array<std::int32_t, 8> tetrahedra{0, 3, 1, 2, 0, 4, 2, 1};
  constexpr std::array<std::int32_t, 2> regions{7, 7};
  std::map<std::array<std::int32_t, 3>, std::int32_t> face_counts;
  for (std::size_t row = 0; row < tetrahedra.size() / 4; ++row) {
    const std::int32_t* tetrahedron = tetrahedra.data() + 4 * row;
    for (int opposite = 0; opposite < 4; ++opposite) {
      std::int32_t face[3];
      oriented_facet(tetrahedron, opposite, face);
      std::array<std::int32_t, 3> key{face[0], face[1], face[2]};
      std::sort(key.begin(), key.end());
      ++face_counts[key];
    }
  }
  std::vector<std::int32_t> faces;
  std::map<std::array<std::int32_t, 2>, bool> boundary_edges;
  for (const auto& [face, count] : face_counts) {
    if (count != 1) continue;
    faces.insert(faces.end(), face.begin(), face.end());
    for (int first = 0; first < 3; ++first) {
      for (int second = first + 1; second < 3; ++second) {
        boundary_edges.emplace(std::array<std::int32_t, 2>{face[first], face[second]}, true);
      }
    }
  }
  std::vector<std::int32_t> face_sources(faces.size() / 3);
  for (std::size_t row = 0; row < face_sources.size(); ++row) {
    face_sources[row] = static_cast<std::int32_t>(row);
  }
  std::vector<std::int32_t> segments;
  segments.reserve(boundary_edges.size() * 2);
  for (const auto& [edge, unused] : boundary_edges) {
    (void)unused;
    segments.insert(segments.end(), edge.begin(), edge.end());
  }
  std::vector<std::int32_t> segment_sources(segments.size() / 2);
  for (std::size_t row = 0; row < segment_sources.size(); ++row) {
    segment_sources[row] = static_cast<std::int32_t>(row);
  }

  TetMesh mesh(BoundaryPolicy::kFixed);
  const std::int32_t build_status = mesh.build(
      points, tetrahedra, regions, faces, face_sources, {}, segments, segment_sources,
      {}, nullptr);
  PHX_CHECK(build_status == PHX_MC_OK);
  if (build_status != PHX_MC_OK) return;
  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);
  if (before.cells.size() != 2) return;
  double before_angle = 0.0;
  double before_determinant = 0.0;
  double before_radius = 0.0;
  PHX_CHECK(minimum_quality(mesh, before, before_angle, before_determinant,
                            before_radius));
  PHX_CHECK(before_radius > 2.0);

  std::int64_t inserted = 0;
  std::int64_t collapsed = 0;
  std::int64_t relocated = 0;
  std::int64_t unmet = 0;
  const std::int32_t improve_status =
      improve_mesh(mesh, TetQualityRequest{2.0, 10.0, 1.0e-12}, {},
                   inserted, collapsed, relocated, unmet);
  if (improve_status != PHX_MC_UNRESOLVED || unmet != 1 ||
      mesh.unmet_records().size() != 1 ||
      mesh.unmet_records().front().criterion != PHX_MC_TET_UNMET_RADIUS_EDGE) {
    std::fprintf(stderr, "joint-quality status=%d unmet_cells=%lld\n",
                  improve_status, static_cast<long long>(unmet));
    for (const TetUnmetRecord& record : mesh.unmet_records()) {
      std::fprintf(stderr, "  cell=%llu criterion=%d value=%.17g reason=%d\n",
                   static_cast<unsigned long long>(record.cell_id),
                   record.criterion, record.value, record.reason);
    }
  }
  PHX_CHECK(improve_status == PHX_MC_UNRESOLVED);
  PHX_CHECK(collapsed == 0 && relocated == 0 && unmet == 1);
  PHX_CHECK(mesh.unmet_records().size() == 1);
  if (mesh.unmet_records().size() == 1) {
    PHX_CHECK(mesh.unmet_records().front().criterion == PHX_MC_TET_UNMET_RADIUS_EDGE);
    PHX_CHECK(mesh.unmet_records().front().value > 2.0);
    PHX_CHECK(mesh.unmet_records().front().reason == PHX_MC_TET_UNMET_NO_IMPROVEMENT);
  }
  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  PHX_CHECK(after.generation > before.generation && after.cells.size() == 3);
  double after_angle = 0.0;
  double after_determinant = 0.0;
  double after_radius = 0.0;
  PHX_CHECK(minimum_quality(mesh, after, after_angle, after_determinant,
                            after_radius));
  PHX_CHECK(after_angle > before_angle && after_angle >= 10.0);
  PHX_CHECK(after_radius > 2.0);
  PHX_CHECK(after_determinant > before_determinant);
  PHX_CHECK(same_source_banks(before, after));
  for (const TetSnapshotCell& cell : after.cells) {
    PHX_CHECK(cell.region == 7);
  }
}

void relocation_clears_stale_carrier_ancestry() {
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr std::array<std::int32_t, 4> tetrahedron{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 1> regions{3};
  constexpr std::array<std::int32_t, 0> no_indices{};
  std::array<std::int32_t, 12> faces{};
  for (int opposite = 0; opposite < 4; ++opposite) {
    std::int32_t face[3];
    oriented_facet(tetrahedron.data(), opposite, face);
    std::copy_n(face, 3, faces.begin() + 3 * opposite);
  }
  constexpr std::array<std::int32_t, 4> face_sources{0, 1, 2, 3};
  TetMesh mesh(BoundaryPolicy::kFixed);
  const std::int32_t build_status = mesh.build(
      points, tetrahedron, regions, faces, face_sources, {}, no_indices, no_indices,
      {}, nullptr);
  PHX_CHECK(build_status == PHX_MC_OK);
  if (build_status != PHX_MC_OK) return;
  // An explicit stellar edit seeds an interior vertex with an exact
  // barycentric stencil (refinement inserts circumcenters without one).
  TetMeshSnapshot initial;
  PHX_CHECK(mesh.snapshot(initial) == TetEditStatus::kApplied);
  if (initial.cells.size() != 1) return;
  const TetSnapshotCell& parent = initial.cells.front();
  std::uint64_t next_vertex_id = 0;
  for (const TetSnapshotVertex& vertex : initial.vertices) {
    next_vertex_id = std::max(next_vertex_id, vertex.id + 1);
  }
  TetEdit stellar;
  stellar.kind = TetEditKind::kRefineCell;
  stellar.removed_cell_ids.push_back(parent.id);
  TetEditVertex centroid;
  centroid.id = next_vertex_id;
  centroid.carrier = {0.25, 0.25, 0.25};
  for (std::uint64_t vertex : parent.vertices) {
    centroid.ancestry.push_back({vertex, {}, 0.25});
  }
  stellar.changed_vertices.push_back(centroid);
  const std::int32_t parent_slot = mesh.cell_slot(parent.id);
  for (int face = 0; face < 4; ++face) {
    std::int32_t oriented[3];
    oriented_facet(mesh.tet(parent_slot).v, face, oriented);
    TetEditCell child;
    child.id = parent.id + 1 + static_cast<std::uint64_t>(face);
    child.vertices = {mesh.vertex_id(oriented[0]), mesh.vertex_id(oriented[1]),
                      mesh.vertex_id(oriented[2]), centroid.id};
    child.facet_sources[3] = mesh.complex().constraint(parent_slot, face);
    child.region = parent.region;
    child.ancestor_cell_ids.push_back(parent.id);
    stellar.replacement_cells.push_back(child);
  }
  TetEditBatch batch{mesh.generation(), mesh.source_revision(), mesh.frame_id(),
                     mesh.identity_bound(), {std::move(stellar)}, mesh.source_id(),
                     mesh.source_map_id(), mesh.periodic_identity()};
  std::vector<TetEditResult> results;
  PHX_CHECK(mesh.apply_edit_batch(batch, results) == TetEditStatus::kApplied);
  if (results.size() != 1 || results.front().status != TetEditStatus::kApplied) return;
  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);
  const auto steiner = std::find_if(before.vertices.begin(), before.vertices.end(),
                                    [](const TetSnapshotVertex& vertex) {
                                      return !vertex.ancestry.empty();
                                    });
  PHX_CHECK(steiner != before.vertices.end());
  if (steiner == before.vertices.end()) return;
  const std::int32_t slot = mesh.vertex_slot(steiner->id);
  TetEditVertex moved;
  moved.id = steiner->id;
  std::copy_n(mesh.point(slot), 3, moved.carrier.begin());
  moved.carrier[0] += 0.01;
  moved.witness = mesh.witness(slot);
  moved.protection_radius = mesh.protection_radius(slot);
  moved.protected_vertex = mesh.protected_vertex(slot);
  std::vector<std::uint64_t> affected_cells;
  for (const TetSnapshotCell& cell : before.cells) {
    if (std::find(cell.vertices.begin(), cell.vertices.end(), steiner->id) !=
        cell.vertices.end()) {
      affected_cells.push_back(cell.id);
    }
  }
  PHX_CHECK(mesh.relocate(std::span(&moved, 1), affected_cells) ==
            TetEditStatus::kApplied);
  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  const auto relocated = std::find_if(after.vertices.begin(), after.vertices.end(),
                                      [&](const TetSnapshotVertex& vertex) {
                                        return vertex.id == steiner->id;
                                      });
  PHX_CHECK(relocated != after.vertices.end());
  if (relocated != after.vertices.end()) {
    PHX_CHECK(relocated->carrier == moved.carrier);
    PHX_CHECK(relocated->ancestry.empty());
  }
  PHX_CHECK(same_source_banks(before, after));
}

void unresolved_radius_target_reports_residuals_without_losing_constraints() {
  constexpr std::array<double, 12> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr std::array<std::int32_t, 4> tetrahedron{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 1> regions{5};
  constexpr std::array<std::int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 0> no_indices{};
  std::array<std::int32_t, 12> faces{};
  for (int opposite = 0; opposite < 4; ++opposite) {
    std::int32_t face[3];
    oriented_facet(tetrahedron.data(), opposite, face);
    std::copy_n(face, 3, faces.begin() + 3 * opposite);
  }
  TetMesh mesh(BoundaryPolicy::kFixed);
  const std::int32_t build_status = mesh.build(
      points, tetrahedron, regions, faces, face_sources, {}, no_indices,
      no_indices, {}, nullptr);
  PHX_CHECK(build_status == PHX_MC_OK);
  if (build_status != PHX_MC_OK) return;
  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);
  if (before.cells.size() != 1) return;

  std::int64_t inserted = 0;
  std::int64_t collapsed = 0;
  std::int64_t relocated = 0;
  std::int64_t unmet = 0;
  const std::int32_t status = improve_mesh(
      mesh, TetQualityRequest{0.5, 10.0, 0.0}, {}, inserted, collapsed, relocated, unmet);
  PHX_CHECK(status == PHX_MC_UNRESOLVED);
  PHX_CHECK(inserted >= 0 && collapsed >= 0 && relocated >= 0 && unmet > 0);
  PHX_CHECK(!mesh.unmet_records().empty());
  PHX_CHECK(std::any_of(mesh.unmet_records().begin(), mesh.unmet_records().end(),
                        [](const TetUnmetRecord& record) {
                          return record.criterion == PHX_MC_TET_UNMET_RADIUS_EDGE &&
                                 record.value > 0.5;
                        }));
  const std::int32_t angle_status = improve_mesh(
      mesh, TetQualityRequest{2.0, 70.49, 0.0}, {}, inserted, collapsed, relocated, unmet);
  PHX_CHECK(angle_status == PHX_MC_UNRESOLVED);
  PHX_CHECK(std::any_of(mesh.unmet_records().begin(), mesh.unmet_records().end(),
                        [](const TetUnmetRecord& record) {
                          return record.criterion == PHX_MC_TET_UNMET_DIHEDRAL &&
                                 record.value < 70.49;
                        }));


  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  PHX_CHECK(same_source_banks(before, after));
  for (const TetSnapshotCell& cell : after.cells) {
    PHX_CHECK(cell.region == 5);
  }
}

// A rounded cocircular torus-grid sliver: exactly positive, but the floating
// circumcenter solve meets a zero pivot. Improvement measures it with an
// unbounded radius-edge ratio instead of refusing its input.
void unrepresentable_positive_sliver_is_measured_not_refused() {
  const std::array<double, 12> points{
      -0x1.c70b87954e637p-3, -0x1.c5ed025972287p-1, -0x1.32dd9dbad3c19p-2,
      -0x1.d04e9d763be24p-3, -0x1.cf2a434460531p-1, -0x1.31619e74336d0p-2,
      -0x1.aaaea448179fdp-3, -0x1.c7a4ec59e9057p-1, -0x1.32dd9dbad3c19p-2,
      -0x1.b35df02262e1bp-3, -0x1.d0eb21831421bp-1, -0x1.31619e74336d0p-2};
  const double* corners[4]{points.data(), points.data() + 3, points.data() + 6,
                           points.data() + 9};
  PHX_CHECK(orient3d(corners[0], corners[1], corners[2], corners[3]) > 0);
  geometry::TetrahedronShape shape;
  PHX_CHECK(!geometry::tetrahedron_shape(corners, shape));
  constexpr std::array<std::int32_t, 4> tetrahedron{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 1> regions{0};
  constexpr std::array<std::int32_t, 4> face_sources{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 0> no_indices{};
  std::array<std::int32_t, 12> faces{};
  for (int opposite = 0; opposite < 4; ++opposite) {
    std::int32_t face[3];
    oriented_facet(tetrahedron.data(), opposite, face);
    std::copy_n(face, 3, faces.begin() + 3 * opposite);
  }
  TetMesh mesh(BoundaryPolicy::kFixed);
  PHX_CHECK(mesh.build(points, tetrahedron, regions, faces, face_sources, {}, no_indices,
                       no_indices, {}, nullptr) == PHX_MC_OK);
  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);
  if (before.cells.size() != 1) return;
  std::int64_t inserted = 0;
  std::int64_t collapsed = 0;
  std::int64_t relocated = 0;
  std::int64_t unmet = 0;
  const std::int32_t status = improve_mesh(
      mesh, TetQualityRequest{2.0, 10.0, 1.0e-12}, {}, inserted, collapsed, relocated, unmet);
  PHX_CHECK(status == PHX_MC_UNRESOLVED);
  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  const std::uint64_t sliver = before.cells.front().id;
  const bool survives = std::any_of(after.cells.begin(), after.cells.end(),
                                    [&](const TetSnapshotCell& cell) { return cell.id == sliver; });
  PHX_CHECK(survives == std::any_of(
                            mesh.unmet_records().begin(), mesh.unmet_records().end(),
                            [&](const TetUnmetRecord& record) {
                              return record.cell_id == sliver &&
                                     record.criterion == PHX_MC_TET_UNMET_RADIUS_EDGE &&
                                     record.reason == PHX_MC_TET_UNMET_NONREPRESENTABLE &&
                                     std::isinf(record.value);
                            }));
  PHX_CHECK(same_source_banks(before, after));
}


void fixed_mapped_cube_slivers_are_improved_transactionally() {
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
  PlcInput input;
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
  input.policy = BoundaryPolicy::kFixed;

  PlcRecovery recovered;
  const std::int32_t recovery_status = recover_plc(input, recovered);
  PHX_CHECK(recovery_status == PHX_MC_OK);
  if (recovery_status != PHX_MC_OK) return;
  PHX_CHECK(recovered.points.size() == points.size());
  PHX_CHECK(!recovered.tets.empty() && recovered.tets.size() % 4 == 0);
  PHX_CHECK(recovered.segment_sources.size() >= segments.size() / 2);
  if (recovered.points.size() != points.size() || recovered.tets.empty() ||
      recovered.tets.size() % 4 != 0) return;

  std::vector<std::int8_t> protected_vertices(recovered.points.size() / 3, 1);
  TetMesh mesh(BoundaryPolicy::kFixed);
  const std::int32_t build_status = mesh.build(
      recovered.points, recovered.tets, recovered.tet_regions, recovered.faces,
      recovered.face_sources, recovered.face_orientations, recovered.segments,
      recovered.segment_sources, recovered.protection, nullptr, protected_vertices);
  PHX_CHECK(build_status == PHX_MC_OK);
  if (build_status != PHX_MC_OK) return;

  TetMeshSnapshot recovered_state;
  PHX_CHECK(mesh.snapshot(recovered_state) == TetEditStatus::kApplied);
  PHX_CHECK(recovered_state.vertices.size() == recovered.points.size() / 3);
  PHX_CHECK(recovered_state.cells.size() == recovered.tets.size() / 4);
  if (recovered_state.vertices.size() != recovered.points.size() / 3 ||
      recovered_state.cells.size() != recovered.tets.size() / 4) return;
  std::vector<std::uint64_t> source_vertex_ids;
  source_vertex_ids.reserve(recovered_state.vertices.size());
  for (const TetSnapshotVertex& vertex : recovered_state.vertices) {
    source_vertex_ids.push_back(vertex.id);
  }

  // Match the layer-core fixture's declared soft uniform size control.
  const std::vector<double> vertex_sizes(recovered.points.size() / 3, 0.75);
  std::int64_t inserted = 0;
  std::int64_t refinement_unmet = 0;
  const std::int32_t refinement_status =
      refine_mesh(mesh, 2.0, vertex_sizes, inserted, refinement_unmet);
  PHX_CHECK(refinement_status == PHX_MC_OK || refinement_status == PHX_MC_UNRESOLVED);
  PHX_CHECK(refinement_unmet >= 0);
  if (refinement_status != PHX_MC_OK && refinement_status != PHX_MC_UNRESOLVED) return;

  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);
  PHX_CHECK(before.vertices.size() >= recovered_state.vertices.size());
  PHX_CHECK(before.cells.size() >= recovered_state.cells.size());
  if (before.vertices.size() < recovered_state.vertices.size() ||
      before.cells.size() < recovered_state.cells.size()) return;
  const std::size_t added_vertices = before.vertices.size() - recovered_state.vertices.size();
  PHX_CHECK(added_vertices == static_cast<std::size_t>(inserted));
  // The refined cube stays a triangulated 3-ball: V - E + F - T = 1.
  {
    std::set<std::pair<std::uint64_t, std::uint64_t>> edges;
    std::set<std::array<std::uint64_t, 3>> faces;
    for (const TetSnapshotCell& cell : before.cells) {
      for (int first = 0; first < 4; ++first) {
        for (int second = first + 1; second < 4; ++second) {
          edges.insert(std::minmax(cell.vertices[first], cell.vertices[second]));
        }
        std::array<std::uint64_t, 3> face{};
        int offset = 0;
        for (int corner = 0; corner < 4; ++corner) {
          if (corner != first) face[offset++] = cell.vertices[corner];
        }
        std::sort(face.begin(), face.end());
        faces.insert(face);
      }
    }
    PHX_CHECK(static_cast<std::int64_t>(before.vertices.size()) -
                  static_cast<std::int64_t>(edges.size()) +
                  static_cast<std::int64_t>(faces.size()) -
                  static_cast<std::int64_t>(before.cells.size()) ==
              1);
  }
  double before_angle = 0.0;
  double before_determinant = 0.0;
  double before_radius = 0.0;
  PHX_CHECK(minimum_quality(mesh, before, before_angle, before_determinant,
                            before_radius));
  PHX_CHECK(before_determinant < 1.0e-12);

  std::int64_t improvement_inserted = 0;
  std::int64_t collapsed = 0;
  std::int64_t relocated = 0;
  std::int64_t unmet = 0;
  const std::int32_t improve_status =
      improve_mesh(mesh, TetQualityRequest{2.0, 10.0, 1.0e-12}, {},
                   improvement_inserted, collapsed, relocated, unmet);
  PHX_CHECK(improve_status == PHX_MC_OK);
  PHX_CHECK(improvement_inserted >= 0 && collapsed >= 0 && relocated >= 0 && unmet == 0);
  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  PHX_CHECK(!after.cells.empty() && !after.vertices.empty());
  if (after.cells.empty() || after.vertices.empty()) return;
  double after_angle = 0.0;
  double after_determinant = 0.0;
  double after_radius = 0.0;
  PHX_CHECK(minimum_quality(mesh, after, after_angle, after_determinant,
                            after_radius));
  PHX_CHECK(after.generation > before.generation);
  PHX_CHECK(after_angle > before_angle);
  PHX_CHECK(after_angle >= 10.0);
  PHX_CHECK(after_determinant > before_determinant);
  PHX_CHECK(after_determinant > 1.0e-12);
  PHX_CHECK(after_radius <= 2.0);
  PHX_CHECK(after.cells.size() == before.constrained_facets.size());
  PHX_CHECK(after.vertices.size() == source_vertex_ids.size() + 1);
  for (const TetSnapshotCell& cell : after.cells) {
    PHX_CHECK(cell.edit_kind == static_cast<std::int32_t>(TetEditKind::kRetriangulate));
    PHX_CHECK(cell.ancestor_cell_ids.empty());
  }
  const auto steiner = std::find_if(after.vertices.begin(), after.vertices.end(),
                                    [&](const TetSnapshotVertex& vertex) {
                                      return std::find(source_vertex_ids.begin(),
                                                       source_vertex_ids.end(), vertex.id) ==
                                             source_vertex_ids.end();
                                    });
  PHX_CHECK(steiner != after.vertices.end());
  if (steiner != after.vertices.end()) {
    PHX_CHECK(steiner->ancestry.size() == source_vertex_ids.size());
    double weight_sum = 0.0;
    for (const TetVertexAncestry& ancestor : steiner->ancestry) {
      PHX_CHECK(std::find(source_vertex_ids.begin(), source_vertex_ids.end(),
                          ancestor.source_vertex_id) != source_vertex_ids.end());
      PHX_CHECK(ancestor.weight == 1.0 / static_cast<double>(source_vertex_ids.size()));
      weight_sum += ancestor.weight;
    }
    PHX_CHECK(weight_sum == 1.0);
  }
  PHX_CHECK(same_source_banks(before, after));
  PHX_CHECK(same_vertex_carriers(before, after, source_vertex_ids));
  PHX_CHECK(!before.cells.empty());
  if (!before.cells.empty()) {
    for (const TetSnapshotCell& cell : after.cells) {
      PHX_CHECK(cell.region == before.cells.front().region);
    }
  }

  const auto protected_vertex = std::find_if(
      after.vertices.begin(), after.vertices.end(),
      [](const TetSnapshotVertex& vertex) { return vertex.protected_vertex; });
  PHX_CHECK(protected_vertex != after.vertices.end());
  if (protected_vertex == after.vertices.end()) return;
  const std::uint64_t protected_id = protected_vertex->id;
  const std::int32_t protected_slot = mesh.vertex_slot(protected_id);
  PHX_CHECK(mesh.live_vertex(protected_slot));
  if (!mesh.live_vertex(protected_slot)) return;
  TetEditVertex moved;
  moved.id = protected_id;
  std::copy_n(mesh.point(protected_slot), 3, moved.carrier.begin());
  moved.carrier[0] += 1.0e-6;
  moved.witness = mesh.witness(protected_slot);
  moved.protection_radius = mesh.protection_radius(protected_slot);
  moved.protected_vertex = true;
  std::vector<std::uint64_t> affected_cells;
  for (const TetSnapshotCell& cell : after.cells) {
    if (std::find(cell.vertices.begin(), cell.vertices.end(), protected_id) !=
        cell.vertices.end()) {
      affected_cells.push_back(cell.id);
    }
  }
  const TetEditStatus protected_status =
      mesh.relocate(std::span(&moved, 1), affected_cells);
  PHX_CHECK(protected_status == TetEditStatus::kRejectedProtected);
  TetMeshSnapshot after_protected_rejection;
  PHX_CHECK(mesh.snapshot(after_protected_rejection) == TetEditStatus::kApplied);
  PHX_CHECK(same_mesh_state(after, after_protected_rejection));

  TetEditBatch stale;
  stale.expected_generation = before.generation;
  stale.identity_bound = mesh.identity_bound();
  stale.source_id = mesh.source_id();
  stale.source_revision = mesh.source_revision();
  stale.source_map_id = mesh.source_map_id();
  stale.frame_id = mesh.frame_id();
  stale.periodic_identity = mesh.periodic_identity();
  std::vector<TetEditResult> stale_results;
  PHX_CHECK(mesh.apply_edit_batch(stale, stale_results) == TetEditStatus::kStaleGeneration);
  TetMeshSnapshot after_stale_rejection;
  PHX_CHECK(mesh.snapshot(after_stale_rejection) == TetEditStatus::kApplied);
  PHX_CHECK(same_mesh_state(after, after_stale_rejection));

  constexpr std::array<double, 15> invalid_points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 0.25, 0.25, 0.25};
  constexpr std::array<std::int32_t, 4> parent_tetrahedron{0, 1, 2, 3};
  constexpr std::array<std::int32_t, 4> invalid_regions{1, 1, 1, 1};
  constexpr std::array<std::int32_t, 4> invalid_face_sources{0, 1, 2, 3};
  constexpr std::array<std::int8_t, 4> invalid_face_orientations{1, 1, 1, 1};
  constexpr std::array<std::int32_t, 0> no_indices{};
  std::array<std::int32_t, 16> invalid_tetrahedra{};
  std::array<std::int32_t, 12> invalid_faces{};
  for (int face = 0; face < 4; ++face) {
    std::int32_t oriented[3];
    oriented_facet(parent_tetrahedron.data(), face, oriented);
    for (int corner = 0; corner < 3; ++corner) {
      invalid_faces[static_cast<std::size_t>(3 * face + corner)] = oriented[corner];
      invalid_tetrahedra[static_cast<std::size_t>(4 * face + corner)] = oriented[corner];
    }
    invalid_tetrahedra[static_cast<std::size_t>(4 * face + 3)] = 4;
  }
  TetMesh invalid_mesh(BoundaryPolicy::kFixed);
  const std::int32_t invalid_build_status = invalid_mesh.build(
      invalid_points, invalid_tetrahedra, invalid_regions, invalid_faces,
      invalid_face_sources, invalid_face_orientations, no_indices, no_indices, {});
  PHX_CHECK(invalid_build_status == PHX_MC_OK);
  if (invalid_build_status != PHX_MC_OK) return;
  TetMeshSnapshot invalid_before;
  PHX_CHECK(invalid_mesh.snapshot(invalid_before) == TetEditStatus::kApplied);
  if (invalid_before.cells.empty()) return;
  const std::uint64_t moved_id = invalid_before.vertices.back().id;
  const std::uint64_t coincident_id = invalid_before.cells.front().vertices[0];
  const std::int32_t moved_slot = invalid_mesh.vertex_slot(moved_id);
  const std::int32_t coincident_slot = invalid_mesh.vertex_slot(coincident_id);
  TetEditVertex invalid_move;
  invalid_move.id = moved_id;
  std::copy_n(invalid_mesh.point(coincident_slot), 3, invalid_move.carrier.begin());
  invalid_move.witness = invalid_mesh.witness(moved_slot);
  invalid_move.protection_radius = invalid_mesh.protection_radius(moved_slot);
  invalid_move.protected_vertex = invalid_mesh.protected_vertex(moved_slot);
  std::vector<std::uint64_t> invalid_affected_cells;
  for (const TetSnapshotCell& cell : invalid_before.cells) {
    if (std::find(cell.vertices.begin(), cell.vertices.end(), moved_id) !=
        cell.vertices.end()) {
      invalid_affected_cells.push_back(cell.id);
    }
  }
  PHX_CHECK(invalid_mesh.relocate(std::span(&invalid_move, 1), invalid_affected_cells) ==
            TetEditStatus::kInvalidGeometry);
  TetMeshSnapshot invalid_after;
  PHX_CHECK(invalid_mesh.snapshot(invalid_after) == TetEditStatus::kApplied);
  PHX_CHECK(same_mesh_state(invalid_before, invalid_after));
}

void dihedral_extremum_matches_pairwise_reference() {
  constexpr std::array<std::array<double, 12>, 4> samples{{
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.5, 0.86602540378443864676, 0.0,
       0.5, 0.28867513459481288225, 0.81649658092772603273},
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.1, 1.2, 0.0,
       0.31, 0.27, 0.83},
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.2, 0.3, 1.0e-10},
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.5, 0.5, 0.0}}};
  for (const auto& sample : samples) {
    std::array<const double*, 4> points{};
    for (int corner = 0; corner < 4; ++corner) {
      points[static_cast<std::size_t>(corner)] =
          sample.data() + 3 * static_cast<std::size_t>(corner);
    }
    const double reference = pairwise_minimum_dihedral_reference(points);
    PHX_CHECK(geometry::minimum_dihedral_degrees(points.data()) == reference);
  }
}
void relocation_metric_cache_is_bitwise_exact() {
  constexpr std::array<std::array<double, 12>, 4> samples{{
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.5, 0.86602540378443864676, 0.0,
       0.5, 0.28867513459481288225, 0.81649658092772603273},
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.1, 1.2, 0.0,
       0.31, 0.27, 0.83},
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.2, 0.3, 1.0e-10},
      {0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.5, 0.5, 0.0}}};
  const auto same_bits = [](double first, double second) {
    return std::bit_cast<std::uint64_t>(first) ==
           std::bit_cast<std::uint64_t>(second);
  };
  for (const auto& sample : samples) {
    std::array<const double*, 4> base_points{};
    for (int corner = 0; corner < 4; ++corner) {
      base_points[static_cast<std::size_t>(corner)] =
          sample.data() + 3 * static_cast<std::size_t>(corner);
    }
    for (std::int32_t moved = 0; moved < 4; ++moved) {
      geometry::RelocationMetricCache cache;
      if (!geometry::prepare_relocation_metric_cache(base_points.data(), moved, cache)) {
        PHX_CHECK(false);
        continue;
      }
      std::array<std::array<double, 3>, 5> candidate_sites{};
      for (int axis = 0; axis < 3; ++axis) {
        const double original = base_points[moved][axis];
        candidate_sites[0][static_cast<std::size_t>(axis)] = original;
        candidate_sites[1][static_cast<std::size_t>(axis)] =
            original + (axis + 1) * 0.037;
        for (int corner = 0; corner < 4; ++corner) {
          if (corner == moved) continue;
          candidate_sites[2][static_cast<std::size_t>(axis)] +=
              base_points[corner][axis] / 3.0;
        }
        candidate_sites[3][static_cast<std::size_t>(axis)] =
            base_points[(moved + 1) % 4][axis];
      }
      candidate_sites[4][0] = std::numeric_limits<double>::quiet_NaN();
      for (const auto& candidate : candidate_sites) {
        std::array<const double*, 4> points = base_points;
        points[static_cast<std::size_t>(moved)] = candidate.data();

        geometry::TetrahedronShape reference_shape;
        const bool reference_shape_valid =
            geometry::tetrahedron_shape(points.data(), reference_shape);
        geometry::TetrahedronShape cached_shape;
        const bool cached_shape_valid = geometry::relocation_tetrahedron_shape(
            points.data(), cache, cached_shape);
        PHX_CHECK(cached_shape_valid == reference_shape_valid);
        if (cached_shape_valid && reference_shape_valid) {
          PHX_CHECK(same_bits(cached_shape.radius_edge_ratio,
                              reference_shape.radius_edge_ratio));
          PHX_CHECK(same_bits(cached_shape.circumradius, reference_shape.circumradius));
        }
        PHX_CHECK(same_bits(
            geometry::relocation_minimum_dihedral_degrees(points.data(), cache),
            geometry::minimum_dihedral_degrees(points.data())));
        PHX_CHECK(same_bits(
            geometry::relocation_relative_determinant(points.data(), cache),
            geometry::relative_determinant(points.data())));
      }
    }
  }
}

void hexagonal_ring_reconnects_and_is_slot_invariant() {
  constexpr std::array<double, 6> radii{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const ImprovementFixture fixture = make_bipyramid_fixture(radii, 1.5);
  const std::vector<std::int32_t> regions(fixture.cells.size(), 7);
  const std::vector<std::int8_t> protected_vertices(fixture.points.size() / 3, 1);
  const std::vector<std::uint64_t> vertex_ids = sequential_ids(fixture.points.size() / 3);
  const std::vector<std::uint64_t> cell_ids = sequential_ids(fixture.cells.size());
  const std::vector<std::uint64_t> facet_ids = sequential_ids(fixture.faces.size());
  std::vector<std::uint64_t> permuted_vertices = vertex_ids;
  std::vector<std::uint64_t> permuted_cells = cell_ids;
  std::vector<std::uint64_t> permuted_facets = facet_ids;
  std::reverse(permuted_vertices.begin(), permuted_vertices.end());
  std::reverse(permuted_cells.begin(), permuted_cells.end());
  std::reverse(permuted_facets.begin(), permuted_facets.end());

  TetMesh first_mesh(BoundaryPolicy::kConforming);
  TetMesh second_mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(build_improvement_mesh(first_mesh, fixture, regions, vertex_ids,
                                   cell_ids, facet_ids, protected_vertices) ==
            PHX_MC_OK);
  PHX_CHECK(build_improvement_mesh(second_mesh, fixture, regions,
                                   permuted_vertices, permuted_cells,
                                   permuted_facets, protected_vertices) ==
            PHX_MC_OK);
  TetMeshSnapshot first_before;
  TetMeshSnapshot second_before;
  PHX_CHECK(first_mesh.snapshot(first_before) == TetEditStatus::kApplied);
  PHX_CHECK(second_mesh.snapshot(second_before) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(first_before, second_before));

  std::int64_t first_inserted = 0;
  std::int64_t first_collapsed = 0;
  std::int64_t first_relocated = 0;
  std::int64_t first_unmet = 0;
  std::int64_t second_inserted = 0;
  std::int64_t second_collapsed = 0;
  std::int64_t second_relocated = 0;
  std::int64_t second_unmet = 0;
  const TetQualityRequest request{1.5, 10.0, 1.0e-12};
  const std::int32_t first_status =
      improve_mesh(first_mesh, request, {}, first_inserted, first_collapsed,
                   first_relocated, first_unmet);
  const std::int32_t second_status =
      improve_mesh(second_mesh, request, {}, second_inserted, second_collapsed,
                   second_relocated, second_unmet);
  PHX_CHECK(first_status == PHX_MC_OK && second_status == PHX_MC_OK);
  PHX_CHECK(first_unmet == 0 && second_unmet == 0);
  PHX_CHECK(first_collapsed == 0 && second_collapsed == 0);
  PHX_CHECK(first_relocated == 0 && second_relocated == 0);

  TetMeshSnapshot first_after;
  TetMeshSnapshot second_after;
  PHX_CHECK(first_mesh.snapshot(first_after) == TetEditStatus::kApplied);
  PHX_CHECK(second_mesh.snapshot(second_after) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(first_after, second_after));
  PHX_CHECK(first_after.generation > first_before.generation);
  PHX_CHECK(first_after.cells.size() == 8 && first_after.vertices.size() == 8);
  PHX_CHECK(first_after.constrained_facets.size() == 12);
  PHX_CHECK(same_source_banks(first_before, first_after));
  std::size_t six_ring_replacements = 0;
  for (const TetSnapshotCell& cell : first_after.cells) {
    PHX_CHECK(cell.region == 7);
    if (cell.edit_kind == static_cast<std::int32_t>(TetEditKind::kRingReconnect) &&
        cell.ancestor_cell_ids.size() == 6) {
      ++six_ring_replacements;
    }
    PHX_CHECK(std::find(cell.vertices.begin(), cell.vertices.end(), 6) ==
                  cell.vertices.end() ||
              std::find(cell.vertices.begin(), cell.vertices.end(), 7) ==
                  cell.vertices.end());
  }
  PHX_CHECK(six_ring_replacements == 8);
  double minimum_angle = 0.0;
  double minimum_determinant = 0.0;
  double maximum_radius_edge = 0.0;
  PHX_CHECK(minimum_quality(first_mesh, first_after, minimum_angle,
                            minimum_determinant, maximum_radius_edge));
  PHX_CHECK(minimum_angle >= 10.0 && maximum_radius_edge <= 1.5);
  PHX_CHECK(minimum_determinant > 1.0e-12);
}

void concave_ring_uses_only_oriented_dp_cavities() {
  constexpr std::array<double, 5> radii{
      0.11529319288025494, 1.4369737537733915, 0.130614163275431,
      0.9198292292368534, 1.1186118593318968};
  const ImprovementFixture fixture = make_bipyramid_fixture(radii, 1.5);
  const std::vector<std::int32_t> regions(fixture.cells.size(), 3);
  const std::vector<std::int8_t> protected_vertices(fixture.points.size() / 3, 1);
  const std::vector<std::uint64_t> vertex_ids = sequential_ids(fixture.points.size() / 3);
  const std::vector<std::uint64_t> cell_ids = sequential_ids(fixture.cells.size());
  const std::vector<std::uint64_t> facet_ids = sequential_ids(fixture.faces.size());
  TetMesh mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(build_improvement_mesh(mesh, fixture, regions, vertex_ids, cell_ids,
                                   facet_ids, protected_vertices) == PHX_MC_OK);
  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);

  std::int64_t inserted = 0;
  std::int64_t collapsed = 0;
  std::int64_t relocated = 0;
  std::int64_t unmet = 0;
  const std::int32_t status = improve_mesh(
      mesh, TetQualityRequest{5.0, 10.0, 1.0e-12}, {}, inserted, collapsed, relocated, unmet);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(inserted == 0 && collapsed == 0 && relocated == 0 && unmet == 0);
  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  PHX_CHECK(after.generation > before.generation);
  PHX_CHECK(after.cells.size() == 6 && after.vertices.size() == 7);
  PHX_CHECK(after.constrained_facets.size() == 10);
  PHX_CHECK(same_source_banks(before, after));
  std::size_t five_ring_replacements = 0;
  for (const TetSnapshotCell& cell : after.cells) {
    PHX_CHECK(cell.region == 3);
    if (cell.edit_kind == static_cast<std::int32_t>(TetEditKind::kRingReconnect) &&
        cell.ancestor_cell_ids.size() == 5) {
      ++five_ring_replacements;
    }
  }
  PHX_CHECK(five_ring_replacements == 6);
  double minimum_angle = 0.0;
  double minimum_determinant = 0.0;
  double maximum_radius_edge = 0.0;
  PHX_CHECK(minimum_quality(mesh, after, minimum_angle, minimum_determinant,
                            maximum_radius_edge));
  PHX_CHECK(minimum_angle >= 10.0 && maximum_radius_edge <= 5.0);
  PHX_CHECK(minimum_determinant > 1.0e-12);
}

void mixed_region_high_ring_refuses_transaction() {
  constexpr std::array<double, 6> radii{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const ImprovementFixture fixture = make_bipyramid_fixture(radii, 1.5);
  const std::array<std::int32_t, 6> regions{0, 1, 2, 3, 4, 5};
  const std::vector<std::int8_t> protected_vertices(fixture.points.size() / 3, 1);
  const std::vector<std::uint64_t> vertex_ids = sequential_ids(fixture.points.size() / 3);
  const std::vector<std::uint64_t> cell_ids = sequential_ids(fixture.cells.size());
  const std::vector<std::uint64_t> facet_ids = sequential_ids(fixture.faces.size());
  TetMesh mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(build_improvement_mesh(mesh, fixture, regions, vertex_ids, cell_ids,
                                   facet_ids, protected_vertices) == PHX_MC_OK);
  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);

  std::int64_t inserted = 0;
  std::int64_t collapsed = 0;
  std::int64_t relocated = 0;
  std::int64_t unmet = 0;
  const std::int32_t status = improve_mesh(
      mesh, TetQualityRequest{1.5, 10.0, 1.0e-12}, {}, inserted, collapsed, relocated, unmet);
  PHX_CHECK(status == PHX_MC_UNRESOLVED && unmet > 0);
  PHX_CHECK(inserted == 0 && collapsed == 0 && relocated == 0);
  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  PHX_CHECK(same_mesh_state(before, after));
}

void interior_collapse_is_slot_order_invariant() {
  const ImprovementFixture fixture = make_interior_collapse_fixture();
  const std::vector<std::int32_t> regions(fixture.cells.size(), 1);
  std::vector<std::int8_t> protected_vertices(fixture.points.size() / 3, 1);
  protected_vertices[0] = 0;
  protected_vertices[1] = 0;
  const std::vector<std::uint64_t> vertex_ids = sequential_ids(fixture.points.size() / 3);
  const std::vector<std::uint64_t> cell_ids = sequential_ids(fixture.cells.size());
  const std::vector<std::uint64_t> facet_ids = sequential_ids(fixture.faces.size());
  std::vector<std::uint64_t> permuted_vertices = vertex_ids;
  std::vector<std::uint64_t> permuted_cells = cell_ids;
  std::vector<std::uint64_t> permuted_facets = facet_ids;
  std::reverse(permuted_vertices.begin(), permuted_vertices.end());
  std::reverse(permuted_cells.begin(), permuted_cells.end());
  std::reverse(permuted_facets.begin(), permuted_facets.end());

  TetMesh first_mesh(BoundaryPolicy::kConforming);
  TetMesh second_mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(build_improvement_mesh(first_mesh, fixture, regions, vertex_ids,
                                   cell_ids, facet_ids, protected_vertices) ==
            PHX_MC_OK);
  PHX_CHECK(build_improvement_mesh(second_mesh, fixture, regions,
                                   permuted_vertices, permuted_cells,
                                   permuted_facets, protected_vertices) ==
            PHX_MC_OK);
  TetMeshSnapshot first_before;
  TetMeshSnapshot second_before;
  PHX_CHECK(first_mesh.snapshot(first_before) == TetEditStatus::kApplied);
  PHX_CHECK(second_mesh.snapshot(second_before) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(first_before, second_before));

  std::int64_t first_inserted = 0;
  std::int64_t first_collapsed = 0;
  std::int64_t first_relocated = 0;
  std::int64_t first_unmet = 0;
  std::int64_t second_inserted = 0;
  std::int64_t second_collapsed = 0;
  std::int64_t second_relocated = 0;
  std::int64_t second_unmet = 0;
  const TetQualityRequest request{2.0, 10.0, 1.0e-12};
  const std::int32_t first_status =
      improve_mesh(first_mesh, request, {}, first_inserted, first_collapsed,
                   first_relocated, first_unmet);
  const std::int32_t second_status =
      improve_mesh(second_mesh, request, {}, second_inserted, second_collapsed,
                   second_relocated, second_unmet);
  PHX_CHECK(first_status == second_status);
  PHX_CHECK(first_collapsed > 0 && second_collapsed > 0);
  PHX_CHECK(first_unmet == second_unmet);
  TetMeshSnapshot first_after;
  TetMeshSnapshot second_after;
  PHX_CHECK(first_mesh.snapshot(first_after) == TetEditStatus::kApplied);
  PHX_CHECK(second_mesh.snapshot(second_after) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(first_after, second_after));
  PHX_CHECK(first_after.generation > first_before.generation);
  PHX_CHECK(first_inserted == second_inserted);
  PHX_CHECK(first_after.vertices.size() + static_cast<std::size_t>(first_collapsed) ==
            first_before.vertices.size() + static_cast<std::size_t>(first_inserted));
  PHX_CHECK(same_source_banks(first_before, first_after));
}

// Positively oriented replacement cell whose cavity-boundary faces carry the
// sources of the matching constrained facets.
TetEditCell sourced_cell(const TetMesh& mesh, const TetMeshSnapshot& snapshot,
                         std::array<std::uint64_t, 4> vertices, std::uint64_t id,
                         std::span<const std::uint64_t> ancestors) {
  std::array<std::int32_t, 4> slots{};
  for (std::size_t corner = 0; corner < 4; ++corner) {
    slots[corner] = mesh.vertex_slot(vertices[corner]);
  }
  if (!mesh.positive(slots.data())) {
    std::swap(vertices[0], vertices[1]);
    std::swap(slots[0], slots[1]);
  }
  TetEditCell cell;
  cell.id = id;
  cell.vertices = vertices;
  cell.region = 7;
  cell.ancestor_cell_ids.assign(ancestors.begin(), ancestors.end());
  for (int corner = 0; corner < 4; ++corner) {
    std::int32_t face[3];
    oriented_facet(slots.data(), corner, face);
    std::array<std::uint64_t, 3> key{mesh.vertex_id(face[0]), mesh.vertex_id(face[1]),
                                     mesh.vertex_id(face[2])};
    std::sort(key.begin(), key.end());
    for (const TetSnapshotFacet& facet : snapshot.constrained_facets) {
      std::array<std::uint64_t, 3> facet_key = facet.vertices;
      std::sort(facet_key.begin(), facet_key.end());
      if (facet_key == key) {
        cell.facet_sources[static_cast<std::size_t>(corner)] = facet.source;
      }
    }
  }
  return cell;
}

bool incidence_matches_mesh(const TetMesh& mesh,
                            const detail::LocalIncidence& incidence) {
  TetMeshSnapshot snapshot;
  if (mesh.snapshot(snapshot) != TetEditStatus::kApplied) return false;
  detail::LocalIncidence rebuilt;
  if (!rebuilt.initialize(mesh, snapshot)) return false;
  for (const TetSnapshotVertex& vertex : snapshot.vertices) {
    const std::int32_t slot = mesh.vertex_slot(vertex.id);
    const detail::LocalIncidence::Star maintained = incidence.vertex_star(slot);
    const detail::LocalIncidence::Star expected = rebuilt.vertex_star(slot);
    if (!std::equal(maintained.begin(), maintained.end(), expected.begin(),
                    expected.end())) {
      return false;
    }
  }
  return true;
}

void incidence_route_matches_public_scan_after_collapse() {
  const ImprovementFixture fixture = make_interior_collapse_fixture();
  const std::vector<std::int32_t> regions(fixture.cells.size(), 1);
  std::vector<std::int8_t> protected_vertices(fixture.points.size() / 3, 1);
  protected_vertices[0] = 0;
  protected_vertices[1] = 0;
  const std::vector<std::uint64_t> vertex_ids = sequential_ids(fixture.points.size() / 3);
  const std::vector<std::uint64_t> cell_ids = sequential_ids(fixture.cells.size());
  const std::vector<std::uint64_t> facet_ids = sequential_ids(fixture.faces.size());
  TetMesh indexed_mesh(BoundaryPolicy::kConforming);
  TetMesh scanned_mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(build_improvement_mesh(indexed_mesh, fixture, regions, vertex_ids,
                                   cell_ids, facet_ids, protected_vertices) ==
            PHX_MC_OK);
  PHX_CHECK(build_improvement_mesh(scanned_mesh, fixture, regions, vertex_ids,
                                   cell_ids, facet_ids, protected_vertices) ==
            PHX_MC_OK);
  TetMeshSnapshot initial;
  PHX_CHECK(indexed_mesh.snapshot(initial) == TetEditStatus::kApplied);
  detail::LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(indexed_mesh, initial));
  const std::int32_t initial_survivor_slot = indexed_mesh.vertex_slot(1);
  const detail::LocalIncidence::Star stale_star =
      incidence.vertex_star(initial_survivor_slot);
  std::vector<std::uint64_t> stale_affected_cells;
  for (std::int32_t cell : stale_star) {
    stale_affected_cells.push_back(indexed_mesh.cell_id(cell));
  }

  constexpr std::uint64_t removed_id = 0;
  constexpr std::uint64_t survivor_id = 1;
  const std::int32_t removed_slot = indexed_mesh.vertex_slot(removed_id);
  std::uint64_t next_cell_id = 0;
  for (const TetSnapshotCell& cell : initial.cells) {
    next_cell_id = std::max(next_cell_id, cell.id + 1);
  }
  std::vector<std::uint64_t> removed_cell_ids;
  std::vector<TetEditCell> replacements;
  for (const TetSnapshotCell& cell : initial.cells) {
    const std::int32_t slot = indexed_mesh.cell_slot(cell.id);
    const Tetrahedron& old = indexed_mesh.tet(slot);
    int removed_corner = -1;
    bool contains_survivor = false;
    for (int corner = 0; corner < 4; ++corner) {
      if (indexed_mesh.vertex_id(old.v[corner]) == removed_id) removed_corner = corner;
      if (indexed_mesh.vertex_id(old.v[corner]) == survivor_id) contains_survivor = true;
    }
    if (removed_corner < 0) continue;
    removed_cell_ids.push_back(cell.id);
    if (contains_survivor) continue;

    TetEditCell replacement;
    replacement.id = next_cell_id++;
    replacement.region = indexed_mesh.complex().region(slot);
    replacement.ancestor_cell_ids = {cell.id};
    for (int corner = 0; corner < 4; ++corner) {
      const std::int32_t vertex = old.v[corner];
      replacement.vertices[static_cast<std::size_t>(corner)] =
          vertex == removed_slot ? survivor_id : indexed_mesh.vertex_id(vertex);
      const std::int32_t source = indexed_mesh.complex().constraint(slot, corner);
      if (corner == removed_corner) {
        replacement.facet_sources[static_cast<std::size_t>(corner)] = source;
      } else {
        if (source != kNoConstraint) PHX_CHECK(false);
        replacement.facet_sources[static_cast<std::size_t>(corner)] = kNoConstraint;
      }
    }
    replacements.push_back(std::move(replacement));
  }
  PHX_CHECK(!removed_cell_ids.empty() && !replacements.empty());
  TetEdit collapse;
  collapse.kind = TetEditKind::kCollapse;
  collapse.collapse_pairs.push_back({removed_id, survivor_id});
  collapse.removed_vertex_ids.push_back(removed_id);
  collapse.removed_cell_ids = removed_cell_ids;
  collapse.replacement_cells = replacements;
  // Only the index's current star of the removed vertex proves the cavity.
  PHX_CHECK(incidence.replace_cavity(indexed_mesh, collapse, &stale_star) ==
            TetEditStatus::kRejectedTopology);
  const detail::LocalIncidence::Star removed_star = incidence.vertex_star(removed_slot);
  const TetEditStatus indexed_collapse =
      incidence.replace_cavity(indexed_mesh, collapse, &removed_star);
  const TetEditStatus scanned_collapse = scanned_mesh.collapse_vertex(
      removed_id, survivor_id, removed_cell_ids, replacements, {}, {});
  PHX_CHECK(indexed_collapse == scanned_collapse);
  PHX_CHECK(indexed_collapse == TetEditStatus::kApplied);
  if (indexed_collapse != TetEditStatus::kApplied) return;
  PHX_CHECK(incidence_matches_mesh(indexed_mesh, incidence));
  TetMeshSnapshot after_collapse;
  TetMeshSnapshot scanned_after_collapse;
  PHX_CHECK(indexed_mesh.snapshot(after_collapse) == TetEditStatus::kApplied);
  PHX_CHECK(scanned_mesh.snapshot(scanned_after_collapse) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(after_collapse, scanned_after_collapse));

  const std::int32_t survivor_slot = indexed_mesh.vertex_slot(survivor_id);
  PHX_CHECK(indexed_mesh.live_vertex(survivor_slot));
  const detail::LocalIncidence::Star star =
      incidence.vertex_star(survivor_slot);
  PHX_CHECK(!star.empty());
  TetEditVertex moved;
  moved.id = survivor_id;
  std::copy_n(indexed_mesh.point(survivor_slot), 3, moved.carrier.begin());
  moved.carrier[0] += 1.0e-6;
  moved.witness = indexed_mesh.witness(survivor_slot);
  moved.protection_radius = indexed_mesh.protection_radius(survivor_slot);
  moved.protected_vertex = indexed_mesh.protected_vertex(survivor_slot);
  const TetEditStatus stale_indexed =
      incidence.relocate(indexed_mesh, std::span(&moved, 1), stale_star);
  const TetEditStatus stale_scanned =
      scanned_mesh.relocate(std::span(&moved, 1), stale_affected_cells);
  PHX_CHECK(stale_indexed == stale_scanned);
  PHX_CHECK(stale_indexed == TetEditStatus::kRejectedTopology);
  TetMeshSnapshot after_stale_refusal;
  TetMeshSnapshot scanned_after_stale_refusal;
  PHX_CHECK(indexed_mesh.snapshot(after_stale_refusal) == TetEditStatus::kApplied);
  PHX_CHECK(scanned_mesh.snapshot(scanned_after_stale_refusal) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(after_stale_refusal,
                                  scanned_after_stale_refusal));
  std::vector<std::uint64_t> affected_cells;
  for (std::int32_t cell : star) affected_cells.push_back(indexed_mesh.cell_id(cell));
  const TetEditStatus indexed_move =
      incidence.relocate(indexed_mesh, std::span(&moved, 1), star);
  const TetEditStatus scanned_move =
      scanned_mesh.relocate(std::span(&moved, 1), affected_cells);
  PHX_CHECK(indexed_move == scanned_move);
  PHX_CHECK(indexed_move == TetEditStatus::kApplied);
  TetMeshSnapshot after_move;
  TetMeshSnapshot scanned_after_move;
  PHX_CHECK(indexed_mesh.snapshot(after_move) == TetEditStatus::kApplied);
  PHX_CHECK(scanned_mesh.snapshot(scanned_after_move) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(after_move, scanned_after_move));

  const auto duplicate_vertex = std::find_if(
      after_move.vertices.begin(), after_move.vertices.end(),
      [&](const TetSnapshotVertex& vertex) {
        return vertex.id != survivor_id &&
               std::any_of(vertex.carrier.begin(), vertex.carrier.end(),
                           [](double coordinate) { return coordinate == 0.0; });
      });
  PHX_CHECK(duplicate_vertex != after_move.vertices.end());
  if (duplicate_vertex == after_move.vertices.end()) return;
  TetEditVertex duplicate_move = moved;
  duplicate_move.carrier = duplicate_vertex->carrier;
  for (double& coordinate : duplicate_move.carrier) {
    if (coordinate == 0.0) coordinate = -0.0;
  }
  const TetEditStatus indexed_duplicate =
      incidence.relocate(indexed_mesh, std::span(&duplicate_move, 1), star);
  const TetEditStatus scanned_duplicate =
      scanned_mesh.relocate(std::span(&duplicate_move, 1), affected_cells);
  PHX_CHECK(indexed_duplicate == scanned_duplicate);
  PHX_CHECK(indexed_duplicate == TetEditStatus::kInvalidGeometry);
  TetMeshSnapshot after_duplicate;
  TetMeshSnapshot scanned_after_duplicate;
  PHX_CHECK(indexed_mesh.snapshot(after_duplicate) == TetEditStatus::kApplied);
  PHX_CHECK(scanned_mesh.snapshot(scanned_after_duplicate) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(after_duplicate, scanned_after_duplicate));
}

void cavity_local_reconnection_matches_public_batch() {
  constexpr std::array<double, 6> radii{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const ImprovementFixture fixture = make_bipyramid_fixture(radii, 1.5);
  const std::vector<std::int32_t> regions(fixture.cells.size(), 7);
  const std::vector<std::int8_t> protected_vertices(fixture.points.size() / 3, 1);
  const std::vector<std::uint64_t> vertex_ids = sequential_ids(fixture.points.size() / 3);
  const std::vector<std::uint64_t> cell_ids = sequential_ids(fixture.cells.size());
  const std::vector<std::uint64_t> facet_ids = sequential_ids(fixture.faces.size());
  TetMesh local_mesh(BoundaryPolicy::kConforming);
  TetMesh public_mesh(BoundaryPolicy::kConforming);
  PHX_CHECK(build_improvement_mesh(local_mesh, fixture, regions, vertex_ids, cell_ids,
                                   facet_ids, protected_vertices) == PHX_MC_OK);
  PHX_CHECK(build_improvement_mesh(public_mesh, fixture, regions, vertex_ids, cell_ids,
                                   facet_ids, protected_vertices) == PHX_MC_OK);
  TetMeshSnapshot initial;
  PHX_CHECK(local_mesh.snapshot(initial) == TetEditStatus::kApplied);
  detail::LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(local_mesh, initial));

  // Replace the six cells around the apex edge (6, 7) by a fan from ring vertex 0.
  TetEdit fan;
  fan.kind = TetEditKind::kRingReconnect;
  for (const TetSnapshotCell& cell : initial.cells) fan.removed_cell_ids.push_back(cell.id);
  std::sort(fan.removed_cell_ids.begin(), fan.removed_cell_ids.end());
  std::uint64_t next_cell_id = fan.removed_cell_ids.back() + 1;
  for (std::uint64_t ring = 1; ring + 1 < 6; ++ring) {
    for (std::uint64_t apex : {6ULL, 7ULL}) {
      fan.replacement_cells.push_back(sourced_cell(local_mesh, initial,
                                                   {0, ring, ring + 1, apex},
                                                   next_cell_id++, fan.removed_cell_ids));
    }
  }

  // Refusals leave the mesh and the index unchanged.
  TetEdit incomplete = fan;
  incomplete.replacement_cells.pop_back();
  PHX_CHECK(incidence.replace_cavity(local_mesh, incomplete, nullptr) ==
            TetEditStatus::kRejectedTopology);
  TetEdit collapse_without_star = fan;
  collapse_without_star.kind = TetEditKind::kCollapse;
  PHX_CHECK(incidence.replace_cavity(local_mesh, collapse_without_star, nullptr) ==
            TetEditStatus::kRejectedTopology);
  TetEdit inverted = fan;
  std::swap(inverted.replacement_cells.front().vertices[0],
            inverted.replacement_cells.front().vertices[1]);
  PHX_CHECK(incidence.replace_cavity(local_mesh, inverted, nullptr) ==
            TetEditStatus::kInvalidGeometry);
  TetMeshSnapshot after_refusals;
  PHX_CHECK(local_mesh.snapshot(after_refusals) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(initial, after_refusals));
  PHX_CHECK(incidence_matches_mesh(local_mesh, incidence));

  PHX_CHECK(incidence.replace_cavity(local_mesh, fan, nullptr) == TetEditStatus::kApplied);
  PHX_CHECK(public_mesh.reconnect(fan.removed_cell_ids, fan.replacement_cells) ==
            TetEditStatus::kApplied);
  TetMeshSnapshot local_fan;
  TetMeshSnapshot public_fan;
  PHX_CHECK(local_mesh.snapshot(local_fan) == TetEditStatus::kApplied);
  PHX_CHECK(public_mesh.snapshot(public_fan) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(local_fan, public_fan));
  PHX_CHECK(local_fan.cells.size() == 8);
  PHX_CHECK(incidence_matches_mesh(local_mesh, incidence));

  // The public batch audits the complete state before it commits, so restoring
  // the original ring on the locally edited mesh also audits the local commit.
  std::vector<std::uint64_t> fan_ids;
  for (const TetEditCell& cell : fan.replacement_cells) fan_ids.push_back(cell.id);
  std::vector<TetEditCell> ring_cells;
  for (const TetSnapshotCell& cell : initial.cells) {
    ring_cells.push_back(sourced_cell(local_mesh, local_fan, cell.vertices, next_cell_id++,
                                      fan_ids));
  }
  PHX_CHECK(local_mesh.reconnect(fan_ids, ring_cells) == TetEditStatus::kApplied);
  PHX_CHECK(public_mesh.reconnect(fan_ids, ring_cells) == TetEditStatus::kApplied);
  TetMeshSnapshot local_ring;
  TetMeshSnapshot public_ring;
  PHX_CHECK(local_mesh.snapshot(local_ring) == TetEditStatus::kApplied);
  PHX_CHECK(public_mesh.snapshot(public_ring) == TetEditStatus::kApplied);
  PHX_CHECK(same_logical_snapshot(local_ring, public_ring));
}

}  // namespace

int main() {
  convex_bipyramid_reports_unresolved_radius_edge();
  relocation_clears_stale_carrier_ancestry();
  unresolved_radius_target_reports_residuals_without_losing_constraints();
  unrepresentable_positive_sliver_is_measured_not_refused();
  fixed_mapped_cube_slivers_are_improved_transactionally();
  dihedral_extremum_matches_pairwise_reference();
  relocation_metric_cache_is_bitwise_exact();
  hexagonal_ring_reconnects_and_is_slot_invariant();
  concave_ring_uses_only_oriented_dp_cavities();
  mixed_region_high_ring_refuses_transaction();
  interior_collapse_is_slot_order_invariant();
  incidence_route_matches_public_scan_after_collapse();
  cavity_local_reconnection_matches_public_batch();
  return phx::mc::test::finish("test_improve3d");
}
