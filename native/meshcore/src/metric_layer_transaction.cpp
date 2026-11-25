// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_layer_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "key_map.hpp"
#include "topology_keys.hpp"

#include "clip3d.hpp"
#include "metric_float.hpp"
#include "metric_layer_overlap.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc::metric {
namespace {

using Id = std::uint64_t;
using Point = std::array<double, 3>;
using Face = std::array<std::int32_t, 3>;
using TetPoints = std::array<const double*, 4>;
using TetCoordinates = std::array<double, 12>;
using layer_detail::StencilPoint;
using layer_detail::Tet;

TetPoints tet_points(const TetCoordinates& coordinates) {
  const double* p = coordinates.data();
  return {p, p + 3, p + 6, p + 9};
}

// Exported IDs must also survive the public stable int64 identity boundary.
constexpr Id kLastId = std::numeric_limits<std::int64_t>::max();
constexpr auto kPreserved = static_cast<std::int32_t>(AdaptLineage::kPreserved);
constexpr auto kRefined = static_cast<std::int32_t>(AdaptLineage::kRefinedFrom);
constexpr auto kRetessellated = static_cast<std::int32_t>(AdaptLineage::kSwappedFrom);

bool fits_ids(Id first, std::size_t count) {
  return count == 0 || (first <= kLastId && count - 1 <= kLastId - first);
}

bool strict_ids(const std::vector<Id>& ids) {
  return std::adjacent_find(ids.begin(), ids.end(), std::greater_equal<>{}) == ids.end();
}

int face_direction(const Face& first, const Face& second) {
  return FacetKey::of(first.data()) == FacetKey::of(second.data()) ? 1 : -1;
}

Face face_key(Face face) {
  std::sort(face.begin(), face.end());
  return face;
}

struct Bounds {
  Point low;
  Point high;
};

struct Box {
  Bounds bounds;
  std::array<Expansion, 6> twice_area;

  int side(const double* a, const double* b, const double* c) const {
    for (int axis = 0; axis < 3; ++axis) {
      if (a[axis] == b[axis] && a[axis] == c[axis]) {
        if (a[axis] == bounds.low[axis]) return 2 * axis;
        if (a[axis] == bounds.high[axis]) return 2 * axis + 1;
      }
    }
    return -1;
  }

  bool contains(const Point& point) const {
    for (int axis = 0; axis < 3; ++axis) {
      if (!(bounds.low[axis] <= point[axis] && point[axis] <= bounds.high[axis])) return false;
    }
    return true;
  }

  bool corner(const Point& point) const {
    for (int axis = 0; axis < 3; ++axis) {
      if (point[axis] != bounds.low[axis] && point[axis] != bounds.high[axis]) return false;
    }
    return true;
  }

  // Native positive-tet side order points *into* its finite owner.
  static int inward(int side) { return side % 2 == 0 ? 1 : -1; }

  bool add(int side, const double* a, const double* b, const double* c) {
    if (side < 0) return false;
    const int u = (side / 2 + 1) % 3;
    const int v = (side / 2 + 2) % 3;
    const Expansion bu = Expansion::difference(b[u], a[u]);
    const Expansion bv = Expansion::difference(b[v], a[v]);
    const Expansion cu = Expansion::difference(c[u], a[u]);
    const Expansion cv = Expansion::difference(c[v], a[v]);
    const Expansion area = bu * cv - bv * cu;
    if (area.sign() != inward(side)) return false;
    twice_area[side] = twice_area[side] + (inward(side) > 0 ? area : -area);
    return true;
  }

  // A closed, consistently oriented boundary chain supported on the box
  // sphere has constant multiplicity there. Unit area multiplicity on every
  // side proves full coverage without overlaps. Positive tetrahedra bound
  // this chain, so their degree is one inside and zero outside the box.
  bool covered() const {
    for (int axis = 0; axis < 3; ++axis) {
      const int u = (axis + 1) % 3;
      const int v = (axis + 2) % 3;
      const Expansion area = Expansion::difference(bounds.high[u], bounds.low[u]).scaled(2.0) *
                             Expansion::difference(bounds.high[v], bounds.low[v]);
      if (!(twice_area[2 * axis] - area).is_zero() ||
          !(twice_area[2 * axis + 1] - area).is_zero()) return false;
    }
    return true;
  }
};

struct Wall {
  std::int32_t group = kNoConstraint;
  std::int64_t source_id = -1;
  std::int8_t orientation = 0;
  bool declared = false;
};

struct SourceBox {
  Box box;
  std::array<Wall, 6> walls;
  std::map<Point, std::size_t> point_rows;
  std::vector<std::int32_t> row_by_slot;
  std::int32_t block = -1;
  std::int32_t region = kNoRegion;
  Id next_facet_id = 0;
};

bool original_witness(const TetSnapshotVertex& vertex, std::int32_t row,
                      const TetMeshSnapshot& snapshot) {
  if (vertex.has_edge_parameter || !vertex.ancestry.empty() || vertex.witness.deviation != 0.0 ||
      vertex.protection_radius != 0.0) return false;
  const SourceWitness& witness = vertex.witness;
  if (witness.stratum == SourceStratum::kNone) return witness.entity == -1;
  if (witness.stratum != SourceStratum::kSegment || witness.entity < 0 ||
      static_cast<std::size_t>(witness.entity) >= snapshot.source_segments.size() / 2) return false;
  const double t = witness.parameters[0];
  return (t == 0.0 || t == 1.0) &&
         snapshot.source_segments[2 * static_cast<std::size_t>(witness.entity) +
                                  static_cast<std::size_t>(t)] == row;
}

bool bind_original_vertices(const TetMesh& mesh, const AdaptInput& input,
                            const TetMeshSnapshot& snapshot, SourceBox& source) {
  const std::size_t count = input.vertex_ids.size();
  if (count < 8 || count >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      count != snapshot.vertices.size() || count != mesh.vertex_slot_count() ||
      input.source_vertex_ids.size() != count || input.source_points != snapshot.source_points ||
      input.source_points.size() != 3 * count || input.source_metrics.size() != 9 * count ||
      input.source_logarithms.size() != 9 * count || !strict_ids(input.vertex_ids) ||
      input.vertex_target_ids != input.vertex_ids || input.stencil_offsets.size() != count + 1 ||
      input.stencil_rows.size() != count || input.stencil_weights.size() != count ||
      input.stencil_offsets.back() != static_cast<std::int64_t>(count) ||
      input.next_vertex_id <= input.vertex_ids.back() ||
      input.next_target_vertex <= input.vertex_target_ids.back()) return false;
  source.row_by_slot.resize(count);
  source.box.bounds.low.fill(std::numeric_limits<double>::infinity());
  source.box.bounds.high.fill(-std::numeric_limits<double>::infinity());
  for (std::size_t at = 0; at < count; ++at) {
    const std::int32_t slot = mesh.vertex_slot(input.vertex_ids[at]);
    const std::int32_t row = input.stencil_rows[at];
    if (!mesh.live_vertex(slot) || row < 0 || static_cast<std::size_t>(row) >= count ||
        input.stencil_offsets[at] != static_cast<std::int64_t>(at) ||
        input.stencil_weights[at] != 1.0 || input.source_vertex_ids[row] != input.vertex_ids[at] ||
        mesh.vertex_size(slot) != 0.0) return false;
    const TetSnapshotVertex& vertex = snapshot.vertices[slot];
    if (vertex.id != input.vertex_ids[at] || !original_witness(vertex, row, snapshot) ||
        !std::equal(vertex.carrier.begin(), vertex.carrier.end(), input.source_points.data() + 3 * row) ||
        !source.point_rows.emplace(vertex.carrier, row).second) return false;
    source.row_by_slot[slot] = row;
    for (int axis = 0; axis < 3; ++axis) {
      source.box.bounds.low[axis] = std::min(source.box.bounds.low[axis], vertex.carrier[axis]);
      source.box.bounds.high[axis] = std::max(source.box.bounds.high[axis], vertex.carrier[axis]);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!(source.box.bounds.low[axis] < source.box.bounds.high[axis])) return false;
  }
  return std::all_of(snapshot.vertices.begin(), snapshot.vertices.end(), [&](const auto& vertex) {
    return !vertex.protected_vertex || source.box.corner(vertex.carrier);
  });
}

// Inactive source rows used as witnesses for Cartesian ridges are not live
// constrained segments. Accept only exact, represented box-ridge subsegments.
bool exact_ridge_rows(const TetMeshSnapshot& snapshot, const Box& box) {
  for (std::size_t row = 0; row < snapshot.source_segments.size() / 2; ++row) {
    const double* a = snapshot.source_points.data() + 3 * snapshot.source_segments[2 * row];
    const double* b = snapshot.source_points.data() + 3 * snapshot.source_segments[2 * row + 1];
    int fixed = 0;
    for (int axis = 0; axis < 3; ++axis) {
      fixed += a[axis] == b[axis] &&
               (a[axis] == box.bounds.low[axis] || a[axis] == box.bounds.high[axis]);
    }
    if (fixed != 2 || snapshot.source_segment_tolerances[row] != 0.0) return false;
  }
  return true;
}

bool bind_wall(const TetMeshSnapshot& snapshot, std::size_t row, const Box& box, int side,
               std::int8_t orientation, Wall& wall) {
  const auto group = snapshot.source_face_groups[row];
  const auto source_id = snapshot.source_face_ids[row];
  const double* plane = snapshot.source_face_planes.data() + 9 * row;
  const bool declared = !std::isnan(plane[0]);
  const int axis = side / 2;
  const double level = side % 2 == 0 ? box.bounds.low[axis] : box.bounds.high[axis];
  if (group < 0 || source_id < 0 || snapshot.source_face_tolerances[row] != 0.0 ||
      snapshot.source_face_plane_tolerances[row] != 0.0 ||
      (declared && (plane[axis] != level || plane[3 + axis] != 0.0 || plane[6 + axis] != 0.0))) {
    return false;
  }
  if (wall.group == kNoConstraint) wall = {group, source_id, orientation, declared};
  return wall.group == group && wall.source_id == source_id && wall.orientation == orientation &&
         wall.declared == declared;
}

bool bind_original_cells(const TetMesh& mesh, const AdaptInput& input,
                         const TetMeshSnapshot& snapshot, SourceBox& source) {
  const std::size_t count = input.cell_ids.size();
  if (count == 0 || count != snapshot.cells.size() || !strict_ids(input.cell_ids) ||
      input.source_cells.size() != 4 * count || input.cell_target_ids != input.cell_ids ||
      input.cell_blocks.size() != count || input.ancestry_offsets.size() != count + 1 ||
      input.ancestry_offsets.back() != static_cast<std::int64_t>(count) ||
      input.ancestry_ids != input.cell_ids || input.ancestry_kinds.size() != count ||
      input.next_cell_id <= input.cell_ids.back() || input.next_target_cell <= input.cell_ids.back()) {
    return false;
  }
  source.block = input.cell_blocks.front();
  source.region = snapshot.cells.front().region;
  if (source.block < 0) return false;
  std::vector<std::uint8_t> referenced(source.row_by_slot.size(), 0);
  for (std::size_t row = 0; row < count; ++row) {
    const std::int32_t slot = mesh.cell_slot(input.cell_ids[row]);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot)) ||
        mesh.complex().region(slot) != source.region || input.cell_blocks[row] != source.block ||
        input.ancestry_offsets[row] != static_cast<std::int64_t>(row) ||
        input.ancestry_kinds[row] != kPreserved) return false;
    const Tetrahedron& tet = mesh.tet(slot);
    std::array<std::int64_t, 4> live_rows;
    std::array<std::int64_t, 4> source_rows;
    for (int k = 0; k < 4; ++k) {
      live_rows[k] = source.row_by_slot[tet.v[k]];
      referenced[live_rows[k]] = 1;
    }
    std::copy_n(input.source_cells.data() + 4 * row, 4, source_rows.begin());
    std::sort(live_rows.begin(), live_rows.end());
    std::sort(source_rows.begin(), source_rows.end());
    if (live_rows != source_rows ||
        orient3d(mesh.point(tet.v[0]), mesh.point(tet.v[1]), mesh.point(tet.v[2]),
                 mesh.point(tet.v[3])) <= 0) return false;
  }
  return std::find(referenced.begin(), referenced.end(), 0) == referenced.end() &&
         std::all_of(snapshot.cells.begin(), snapshot.cells.end(), [](const auto& cell) {
           return cell.edit_kind == -1 && cell.ancestor_cell_ids.empty();
         });
}

bool bind_original_boundary(const TetMesh& mesh, const TetMeshSnapshot& snapshot, SourceBox& source) {
  const std::size_t count = snapshot.constrained_facets.size();
  if (count < 12 || snapshot.source_faces.size() != 3 * count ||
      !exact_ridge_rows(snapshot, source.box)) return false;
  struct FacetBinding {
    const TetSnapshotFacet* facet;
    Face vertices;
    std::size_t row;
  };
  std::map<Face, FacetBinding> live_faces;
  for (const auto& facet : snapshot.constrained_facets) {
    if (facet.protected_facet || facet.id > kLastId ||
        (facet.orientation != 1 && facet.orientation != -1)) return false;
    Face rows;
    for (int k = 0; k < 3; ++k) {
      const std::int32_t slot = mesh.vertex_slot(facet.vertices[k]);
      if (!mesh.live_vertex(slot)) return false;
      rows[k] = source.row_by_slot[slot];
    }
    if (!live_faces.emplace(face_key(rows), FacetBinding{&facet, rows, count}).second) return false;
    source.next_facet_id = std::max(source.next_facet_id, facet.id + 1);
  }

  // Every source row must name an actual constrained, exterior triangle, not
  // just a matching plane. Erasing the rows proves both directions of the bind.
  for (std::size_t row = 0; row < count; ++row) {
    Face triangle;
    std::copy_n(snapshot.source_faces.data() + 3 * row, 3, triangle.begin());
    const auto active = live_faces.find(face_key(triangle));
    if (active == live_faces.end() || active->second.row != count) return false;
    active->second.row = row;
  }
  const auto& complex = mesh.complex();
  for (std::size_t cell = 0; cell < complex.tets.size(); ++cell) {
    const auto slot = static_cast<std::int32_t>(cell);
    if (!complex.live(slot) || is_ghost(complex.tets[cell])) continue;
    const auto& tet = complex.tets[cell];
    for (int k = 0; k < 4; ++k) {
      const auto group = complex.constraint(slot, k);
      if (!complex.live(tet.n[k])) return false;
      const auto& neighbor = mesh.tet(tet.n[k]);
      const int opposite = neighbor_slot(neighbor, slot);
      Face rows, twin;
      oriented_facet(tet.v, k, rows.data());
      if (opposite < 0 || complex.constraint(tet.n[k], opposite) != group) return false;
      oriented_facet(neighbor.v, opposite, twin.data());
      if (!(FacetKey::of(rows.data()) == FacetKey::of(twin.data()).reversed())) return false;
      if (!is_ghost(neighbor)) {
        if (group != kNoConstraint) return false;
        continue;
      }
      for (auto& vertex : rows) vertex = source.row_by_slot[vertex];
      const auto face = live_faces.find(face_key(rows));
      if (face == live_faces.end()) return false;
      const auto [facet, facet_rows, row] = face->second;
      if (facet->source != group || snapshot.source_face_groups[row] != group) return false;
      const Face original{snapshot.source_faces[3 * row], snapshot.source_faces[3 * row + 1],
                          snapshot.source_faces[3 * row + 2]};
      if (face_direction(facet_rows, original) != facet->orientation) return false;
      const double* a = snapshot.source_points.data() + 3 * rows[0];
      const double* b = snapshot.source_points.data() + 3 * rows[1];
      const double* c = snapshot.source_points.data() + 3 * rows[2];
      const int side = source.box.side(a, b, c);
      const auto direction = static_cast<std::int8_t>(face_direction(rows, original));
      if (!source.box.add(side, a, b, c) ||
          !bind_wall(snapshot, row, source.box, side, direction, source.walls[side])) return false;
      live_faces.erase(face);
    }
  }
  if (!live_faces.empty() || !source.box.covered()) return false;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < i; ++j) {
      if (source.walls[i].group == source.walls[j].group ||
          source.walls[i].source_id == source.walls[j].source_id) return false;
    }
  }
  return true;
}

struct VertexBinding {
  Id native;
  Id target;
};

struct VertexOutput {
  std::vector<VertexBinding> ids;
  std::vector<std::size_t> order;
};

std::optional<TetEditStatus> bind_grid_vertices(
    const MetricBoxGrid& grid, const AdaptInput& input, const SourceBox& source,
    const TetMeshSnapshot& snapshot, VertexOutput& vertices, TetEdit& edit) {
  const std::size_t count = grid.vertices.size();
  if (count == 0 || grid.tensors.size() != count) return std::nullopt;
  vertices.ids.resize(count);
  vertices.order.reserve(count);
  std::vector<std::uint8_t> retained(input.source_vertex_ids.size(), 0);
  std::set<Point> grid_points;
  std::vector<std::size_t> fresh_rows;
  for (std::size_t row = 0; row < count; ++row) {
    const Carrier& carrier = grid.vertices[row];
    if (!source.box.contains(carrier.point) || !grid_points.insert(carrier.point).second) {
      return std::nullopt;
    }
    const auto original = source.point_rows.find(carrier.point);
    if (original == source.point_rows.end()) {
      fresh_rows.push_back(row);
    } else {
      const auto prior = original->second;
      if (carrier.size != 1 || carrier.rows[0] != static_cast<std::int64_t>(prior) ||
          !std::equal(grid.tensors[row].begin(), grid.tensors[row].end(),
                      input.source_metrics.data() + 9 * prior)) return std::nullopt;
      const Id id = input.source_vertex_ids[prior];
      vertices.ids[row] = {id, id};
      retained[prior] = 1;
    }
    vertices.order.push_back(row);
  }
  const auto fresh = fresh_rows.size();
  if (fresh == 0) return std::nullopt;
  if (!fits_ids(input.next_vertex_id, fresh) || !fits_ids(input.next_target_vertex, fresh)) {
    return TetEditStatus::kIndexOverflow;
  }
  for (const auto& vertex : snapshot.vertices) {
    const std::size_t row = source.point_rows.find(vertex.carrier)->second;
    if (retained[row]) continue;
    if (vertex.protected_vertex) return std::nullopt;
    edit.removed_vertex_ids.push_back(vertex.id);
  }
  std::sort(edit.removed_vertex_ids.begin(), edit.removed_vertex_ids.end());
  edit.changed_vertices.reserve(fresh);
  Id next_native = input.next_vertex_id;
  Id next_target = input.next_target_vertex;
  for (std::size_t row : fresh_rows) {
    const Carrier& carrier = grid.vertices[row];
    vertices.ids[row] = {next_native++, next_target++};
    TetEditVertex& vertex = edit.changed_vertices.emplace_back();
    vertex.id = vertices.ids[row].native;
    vertex.carrier = carrier.point;
  }
  std::sort(vertices.order.begin(), vertices.order.end(), [&](std::size_t a, std::size_t b) {
    return vertices.ids[a].native < vertices.ids[b].native;
  });
  return TetEditStatus::kApplied;
}

bool write_vertices(const MetricBoxGrid& grid, const AdaptInput& input,
                    const Box& box, const VertexOutput& vertices,
                    std::vector<StencilPoint>& lifted, TetEdit& edit, AdaptOutput& output) {
  const auto count = static_cast<std::int64_t>(input.source_vertex_ids.size());
  output.vertex_ids.reserve(vertices.order.size());
  output.vertex_target_ids.reserve(vertices.order.size());
  output.stencil_offsets.reserve(vertices.order.size() + 1);
  std::size_t terms_count = 0;
  for (const Carrier& carrier : grid.vertices) {
    if (carrier.size < 1 || carrier.size > 4) return false;
    terms_count += static_cast<std::size_t>(carrier.size);
  }
  output.stencil_rows.reserve(terms_count);
  output.stencil_weights.reserve(terms_count);
  output.metrics.reserve(9 * vertices.order.size());
  output.stencil_offsets.push_back(0);
  lifted.resize(grid.vertices.size());
  std::vector<std::pair<Id, Id>> relations;
  auto changed = edit.changed_vertices.begin();
  for (std::size_t row : vertices.order) {
    const auto [native, target] = vertices.ids[row];
    const Carrier& carrier = grid.vertices[row];
    if (carrier.bits < 0 || carrier.bits > 53) return false;
    const auto quantum = std::int64_t{1} << carrier.bits;
    auto remaining = quantum;
    StencilPoint& point = lifted[row];
    const bool fresh = changed != edit.changed_vertices.end() && changed->id == native;
    if (fresh) changed->ancestry.reserve(carrier.size);
    for (int k = 0; k < carrier.size; ++k) {
      const auto source_row = carrier.rows[k];
      const auto weight = carrier.weights[k];
      if (source_row < 0 || source_row >= count || (k > 0 && source_row <= carrier.rows[k - 1]) ||
          weight <= 0 || weight > remaining) return false;
      remaining -= weight;
      const double coefficient = std::ldexp(static_cast<double>(weight), -carrier.bits);
      output.stencil_rows.push_back(static_cast<std::int32_t>(source_row));
      output.stencil_weights.push_back(coefficient);
      const double* corner = input.source_points.data() + 3 * source_row;
      for (int axis = 0; axis < 3; ++axis) {
        point.position[axis] = point.position[axis] + Expansion::product(coefficient, corner[axis]);
      }
      if (fresh) {
        const Id ancestor = input.source_vertex_ids[source_row];
        changed->ancestry.push_back({ancestor, {}, coefficient});
        relations.emplace_back(ancestor, target);
      }
    }
    if (remaining != 0) return false;
    for (int axis = 0; axis < 3; ++axis) {
      const double rounded = carrier.point[axis];
      if (nearest_double(point.position[axis]) != rounded) return false;
      const int direction = (point.position[axis] - Expansion(rounded)).sign();
      if (direction == 0) continue;
      // The host preserves exact stencil images, but native geometry uses
      // rounded carriers. Both retain their represented coordinate on a wall.
      if (rounded == box.bounds.low[axis] || rounded == box.bounds.high[axis]) return false;
      const double neighbor = std::nextafter(rounded, direction < 0 ?
          -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity());
      point.displacement[axis] = neighbor - rounded;
      point.exact = false;
    }
    if (fresh) {
      std::sort(changed->ancestry.begin(), changed->ancestry.end(), [](const auto& a, const auto& b) {
        return a.source_vertex_id < b.source_vertex_id;
      });
      ++changed;
    }
    output.vertex_ids.push_back(native);
    output.vertex_target_ids.push_back(target);
    output.stencil_offsets.push_back(static_cast<std::int64_t>(output.stencil_rows.size()));
    output.metrics.insert(output.metrics.end(), grid.tensors[row].begin(), grid.tensors[row].end());
  }
  std::sort(relations.begin(), relations.end());
  output.vertex_relation_sources.reserve(relations.size());
  output.vertex_relation_targets.reserve(relations.size());
  output.vertex_relation_kinds.assign(relations.size(), kRefined);
  for (const auto& [source_id, target_id] : relations) {
    output.vertex_relation_sources.push_back(source_id);
    output.vertex_relation_targets.push_back(target_id);
  }
  return changed == edit.changed_vertices.end();
}

struct FaceUse {
  Face key;
  std::size_t cell;
  int face;
};

Face grid_face(const MetricBoxGrid& grid, const FaceUse& use) {
  Face rows;
  oriented_facet(grid.cells[use.cell].data(), use.face, rows.data());
  return rows;
}

std::optional<TetEditStatus> bind_grid_cells(const MetricBoxGrid& grid, const AdaptInput& input,
                                            const SourceBox& source, const VertexOutput& vertices,
                                            TetEdit& edit) {
  const std::size_t count = grid.cells.size();
  if (count == 0) return std::nullopt;
  if (!fits_ids(input.next_cell_id, count) || !fits_ids(input.next_target_cell, count) ||
      count >= static_cast<std::size_t>(kMaxTetrahedronSlots)) return TetEditStatus::kIndexOverflow;
  edit.replacement_cells.resize(count);
  std::vector<FaceUse> uses;
  std::vector<unsigned> walls(grid.vertices.size());
  for (std::size_t vertex = 0; vertex < grid.vertices.size(); ++vertex) {
    for (int k = 0; k < 3; ++k) {
      const auto x = grid.vertices[vertex].point[k];
      if (x == source.box.bounds.low[k]) walls[vertex] |= 1U << (2 * k);
      else if (x == source.box.bounds.high[k]) walls[vertex] |= 2U << (2 * k);
    }
  }
  std::vector<std::uint8_t> referenced(grid.vertices.size(), 0);
  const double lower = std::max(input.lower, kLowerLength);
  KeyMap edges(std::min(6 * count, count + 2 * grid.vertices.size()));
  for (std::size_t cell = 0; cell < count; ++cell) {
    TetEditCell& record = edit.replacement_cells[cell];
    record.id = input.next_cell_id + cell;
    record.region = source.region;
    const auto& rows = grid.cells[cell];
    TetPoints p;
    for (int k = 0; k < 4; ++k) {
      if (rows[k] < 0 || static_cast<std::size_t>(rows[k]) >= grid.vertices.size()) return std::nullopt;
      referenced[rows[k]] = 1;
      record.vertices[k] = vertices.ids[rows[k]].native;
      p[k] = grid.vertices[rows[k]].point.data();
    }
    if (orient3d(p[0], p[1], p[2], p[3]) <= 0) return std::nullopt;
    for (int k = 0; k < 4; ++k) {
      Face tri;
      oriented_facet(rows.data(), k, tri.data());
      // Only wall facets need source labels. The canonical whole-cavity
      // validator pairs every interior side, including unmarked ones.
      if ((walls[tri[0]] & walls[tri[1]] & walls[tri[2]]) != 0) {
        uses.push_back({face_key(tri), cell, k});
      }
      for (int j = k + 1; j < 4; ++j) {
        auto a = rows[k], b = rows[j];
        auto& seen = edges.slot(topology::undirected_edge(a, b));
        if (seen != -1) continue;
        seen = 1;
        if (vertices.ids[b].native < vertices.ids[a].native) std::swap(a, b);
        const double measured = length(grid.vertices[a].point.data(), grid.vertices[b].point.data(),
                                       grid.tensors[a].data(), grid.tensors[b].data());
        if (!(measured >= lower && measured <= kUpperLength)) return std::nullopt;
      }
    }
  }
  if (std::find(referenced.begin(), referenced.end(), 0) != referenced.end()) return std::nullopt;

  std::sort(uses.begin(), uses.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
  Box shell{source.box.bounds, {}};
  Id next_facet = source.next_facet_id;
  for (auto it = uses.begin(); it != uses.end();) {
    const Face rows = grid_face(grid, *it);
    const auto next = it + 1;
    if (next != uses.end() && it->key == next->key) return std::nullopt;
    const double* a = grid.vertices[rows[0]].point.data();
    const double* b = grid.vertices[rows[1]].point.data();
    const double* c = grid.vertices[rows[2]].point.data();
    const int side = shell.side(a, b, c);
    if (!shell.add(side, a, b, c)) return std::nullopt;
    if (!fits_ids(next_facet, 1)) return TetEditStatus::kIndexOverflow;
    const Wall& wall = source.walls[side];
    TetEditFacet& facet = edit.replacement_facets.emplace_back();
    facet.id = next_facet++;
    facet.source = wall.group;
    facet.orientation = wall.orientation;
    for (int k = 0; k < 3; ++k) facet.vertices[k] = vertices.ids[rows[k]].native;
    edit.replacement_cells[it->cell].facet_sources[it->face] = wall.group;
    it = next;
  }
  if (!shell.covered()) return std::nullopt;
  return TetEditStatus::kApplied;
}

Bounds tet_bounds(const TetPoints& points) {
  Bounds result;
  for (int axis = 0; axis < 3; ++axis) {
    const auto [low, high] = std::minmax({points[0][axis], points[1][axis],
                                        points[2][axis], points[3][axis]});
    result.low[axis] = low;
    result.high[axis] = high;
  }
  return result;
}

bool open_overlap(const Bounds& a, const Bounds& b) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a.low[axis] >= b.high[axis] || b.low[axis] >= a.high[axis]) return false;
  }
  return true;
}

class CellIndex {
 public:
  void build(std::span<const TetCoordinates> points) {
    const std::size_t count = points.size();
    const std::size_t capacity = 2 * count;
    bounds_.reserve(count);
    std::vector<double> low(3 * count), high(3 * count);
    for (std::size_t row = 0; row < count; ++row) {
      const Bounds& b = bounds_.emplace_back(tet_bounds(tet_points(points[row])));
      std::copy(b.low.begin(), b.low.end(), low.begin() + 3 * row);
      std::copy(b.high.begin(), b.high.end(), high.begin() + 3 * row);
    }
    order_.resize(count);
    std::vector<std::int64_t> start(capacity), stop(capacity), left(capacity), right(capacity),
        offsets(capacity);
    std::int64_t nodes = 0;
    std::int64_t levels = 0;
    const int status = phx_mc_bvh_build(PHX_MC_BVH_BUILD_MEDIAN, static_cast<std::int64_t>(count),
                                       3, low.data(), high.data(), nullptr, 8, 0, order_.data(),
                                       start.data(), stop.data(), left.data(), right.data(),
                                       offsets.data(), &nodes, &levels);
    if (status == PHX_MC_ALLOCATION_FAILURE) throw std::bad_alloc();
    if (status != PHX_MC_OK) throw std::logic_error("Could not index source tetrahedra.");
    nodes_.resize(nodes);
    for (std::int64_t row = nodes; row-- > 0;) {
      Node& node = nodes_[row];
      node.start = start[row];
      node.stop = stop[row];
      node.left = left[row];
      node.right = right[row];
      if (node.left >= 0) {
        node.box = enclose(nodes_[node.left].box, nodes_[node.right].box);
      } else {
        node.box = bounds_[order_[node.start]];
        for (auto at = node.start + 1; at < node.stop; ++at) {
          node.box = enclose(node.box, bounds_[order_[at]]);
        }
      }
    }
  }

  template <class Visit>
  bool visit(const Bounds& box, Visit&& visitor) const {
    // Median bisection of an int64 item count cannot exceed 64 levels.
    std::array<std::int64_t, 64> stack{0};
    std::size_t size = 1;
    while (size > 0) {
      const Node& node = nodes_[stack[--size]];
      if (!open_overlap(box, node.box)) continue;
      if (node.left >= 0) {
        stack[size++] = node.right;
        stack[size++] = node.left;
      } else {
        for (auto at = node.start; at < node.stop; ++at) {
          const auto row = order_[at];
          if (open_overlap(box, bounds_[row]) &&
              !visitor(static_cast<std::size_t>(row), bounds_[row])) return false;
        }
      }
    }
    return true;
  }

  const Bounds& bounds(std::size_t row) const { return bounds_[row]; }

 private:
  struct Node {
    Bounds box;
    std::int64_t start, stop, left, right;
  };

  static Bounds enclose(Bounds a, const Bounds& b) {
    for (int k = 0; k < 3; ++k) {
      a.low[k] = std::min(a.low[k], b.low[k]);
      a.high[k] = std::max(a.high[k], b.high[k]);
    }
    return a;
  }

  std::vector<Bounds> bounds_;
  std::vector<std::int64_t> order_;
  std::vector<Node> nodes_;
};

struct SourceCellPlanes {
  std::array<Point, 4> normals;
  std::array<std::int32_t, 4> neighbors;
  std::array<std::int64_t, 4> vertices;

  bool supports(const Carrier& vertex) const {
    return std::includes(vertices.begin(), vertices.end(),
                         vertex.rows.begin(), vertex.rows.begin() + vertex.size);
  }
};

struct SourceSite {
  std::int32_t owner = -1;
  bool closed = false;
  bool interior = false;
};

// The existing determinant bound depends only on coordinate ranges and
// maximum displacements. All rounded vertices are admitted in this box.
// These extremes bound every face query for both embeddings and let us
// reuse the normals and the error across targets.
double box_face_error(const Box& box, std::span<const StencilPoint> lifted) {
  StencilPoint envelope;
  for (const auto& p : lifted) {
    if (p.exact) continue;
    for (int k = 0; k < 3; ++k) {
      envelope.displacement[k] = std::max(envelope.displacement[k], std::abs(p.displacement[k]));
    }
  }
  const double* low = box.bounds.low.data();
  const double* high = box.bounds.high.data();
  const Tet extremes{{{low, &envelope}, {high}, {low}, {high}}};
  return layer_detail::common_distance_error(extremes, extremes);
}

// Returns -1 outside, 0 on the boundary, and 1 in the interior. This checks
// the rounded position independently of any known exact stencil support.
int rounded_site_location(const SourceCellPlanes& cell, const TetCoordinates& source,
                          const double* point, double error) {
  const auto p = tet_points(source);
  int location = 1;
  for (int k = 0; k < 4; ++k) {
    const auto& face = kOrientedFacet[k];
    const double* origin = p[face[0]];
    if (error > 0.0) {
      const double distance = layer_detail::projected_distance(cell.normals[k].data(), origin, point);
      if (distance > error) continue;
      if (distance < -error) return -1;
    }
    const int side = orient3d(origin, p[face[1]], p[face[2]], point);
    if (side < 0) return -1;
    if (side == 0) location = 0;
  }
  return location;
}

bool make_source_planes(const TetMesh& mesh, const AdaptInput& input,
                        std::span<TetCoordinates> points, std::span<SourceCellPlanes> cells) {
  bool adjacent = true;
  for (std::size_t row = 0; row < cells.size(); ++row) {
    const auto& tet = mesh.tet(mesh.cell_slot(input.cell_ids[row]));
    auto& cell = cells[row];
    std::copy_n(input.source_cells.data() + 4 * row, 4, cell.vertices.begin());
    std::sort(cell.vertices.begin(), cell.vertices.end());
    for (int k = 0; k < 4; ++k) std::copy_n(mesh.point(tet.v[k]), 3, points[row].data() + 3 * k);
    const auto p = tet_points(points[row]);
    for (int k = 0; k < 4; ++k) {
      const auto& face = kOrientedFacet[k];
      double u[3], v[3];
      geometry::difference(p[face[1]], p[face[0]], u);
      geometry::difference(p[face[2]], p[face[0]], v);
      geometry::cross(u, v, cell.normals[k].data());
      cell.neighbors[k] = -1;
      const auto slot = tet.n[k];
      if (is_ghost(mesh.tet(slot))) continue;
      const Id id = mesh.cell_id(slot);
      const auto at = std::lower_bound(input.cell_ids.begin(), input.cell_ids.end(), id);
      if (at == input.cell_ids.end() || *at != id) {
        adjacent = false;
        continue;
      }
      cell.neighbors[k] = static_cast<std::int32_t>(at - input.cell_ids.begin());
    }
  }
  return adjacent;
}

std::vector<SourceSite> bind_source_sites(const MetricBoxGrid& grid, const AdaptInput& input,
                                         std::span<const TetCoordinates> points,
                                         std::span<const SourceCellPlanes> cells, double error) {
  // The source cap keeps all four incidences per cell in the int32 range.
  // A stencil can only be supported by a cell incident to its first row.
  std::vector<std::int32_t> first(input.source_vertex_ids.size(), -1);
  std::vector<std::int32_t> next(input.source_cells.size());
  for (std::int32_t at = 0; at < static_cast<std::int32_t>(next.size()); ++at) {
    next[at] = std::exchange(first[input.source_cells[at]], at);
  }
  std::vector<SourceSite> sites(grid.vertices.size());
  for (std::size_t row = 0; row < grid.vertices.size(); ++row) {
    const auto& vertex = grid.vertices[row];
    for (auto at = first[vertex.rows[0]]; at >= 0; at = next[at]) {
      const auto owner = at / 4;
      if (!cells[owner].supports(vertex)) continue;
      const int location = rounded_site_location(cells[owner], points[owner], vertex.point.data(),
                                                error);
      sites[row] = {owner, location >= 0, location > 0 && vertex.size == 4};
      if (location >= 0) break;
    }
  }
  return sites;
}

bool singleton_ancestor(std::int32_t owner, const std::array<std::int32_t, 4>& vertices,
                        const MetricBoxGrid& grid, std::span<const SourceSite> sites,
                        const SourceCellPlanes& cell, const TetCoordinates& points, double error) {
  // write_vertices proved that every exact stencil is convex. Inclusion of
  // its support proves exact containment; independently check every rounded
  // site not already tested against this cell. A target tet of positive
  // volume contained in one source tet has no other positive ancestor.
  for (const auto row : vertices) {
    const auto& site = sites[row];
    if (site.owner == owner) {
      if (!site.closed) return false;
    } else if (grid.vertices[row].size == 4 || !cell.supports(grid.vertices[row])) {
      return false;
    }
  }
  for (const auto row : vertices) {
    if (sites[row].owner != owner &&
        rounded_site_location(cell, points, grid.vertices[row].point.data(), error) < 0) return false;
  }
  return true;
}

struct FaceEvidence {
  layer_detail::Contact contact = layer_detail::Contact::kEdges;
  unsigned neighbors = 15;
};

FaceEvidence source_face_evidence(const SourceCellPlanes& cell, const TetCoordinates& source,
                                  const Tet& target, double error) {
  using layer_detail::Contact;
  if (error == 0.0) return {};
  const auto p = tet_points(source);
  FaceEvidence evidence;
  unsigned strict = 15;
  bool centroid_inside = true;
  double distance[4][4];
  for (int k = 0; k < 4; ++k) {
    auto& d = distance[k];
    const double* origin = p[kOrientedFacet[k][0]];
    unsigned inside = 0, outside = 0;
    for (int j = 0; j < 4; ++j) {
      d[j] = layer_detail::projected_distance(cell.normals[k].data(), origin, target[j].rounded);
      if (d[j] > error) inside |= 1U << j;
      if (d[j] < -error) outside |= 1U << j;
    }
    if (outside == 15) return {Contact::kDisjoint, 0};
    // Both target hulls are entirely on the owner's side of this plane.
    if (inside == 15) evidence.neighbors &= ~(1U << k);
    strict &= inside;
    centroid_inside = centroid_inside && ((d[0] + d[1]) + (d[2] + d[3]) > 8.0 * error);
  }
  if (strict != 0 || centroid_inside || layer_detail::certified_crossing(distance, error)) {
    evidence.contact = Contact::kPositive;
  }
  return evidence;
}

Bounds expanded_bounds(Bounds box, const Tet& target) {
  for (const auto& point : target) {
    if (point.stencil == nullptr) continue;
    for (int axis = 0; axis < 3; ++axis) {
      const double outward = point.rounded[axis] + point.stencil->displacement[axis];
      box.low[axis] = std::min(box.low[axis], outward);
      box.high[axis] = std::max(box.high[axis], outward);
    }
  }
  return box;
}

bool collect_ancestor(TetrahedronClipper& clipper, const TetCoordinates& source,
                      const Tet& exact_target,
                      bool native_boxes_overlap, Id ancestor,
                      std::vector<Id>& native_ancestry, std::vector<Id>& exact_ancestry) {
  const auto p = tet_points(source);
  const Tet first{{{p[0]}, {p[1]}, {p[2]}, {p[3]}}};
  layer_detail::CommonSigns signs;
  const bool positive = layer_detail::positive_overlap(first, exact_target, signs);
  if (positive) exact_ancestry.push_back(ancestor);
  if (!native_boxes_overlap) return true;
  bool native_positive = positive;
  if (!signs.same()) {
    TetCoordinates rounded_target;
    for (int k = 0; k < 4; ++k) {
      std::copy_n(exact_target[k].rounded, 3, rounded_target.data() + 3 * k);
    }
    const auto status = clipper.overlaps(source.data(), rounded_target.data(), native_positive);
    if (status == PHX_MC_ALLOCATION_FAILURE) throw std::bad_alloc();
    if (status == PHX_MC_INDEX_RANGE_ERROR) throw std::length_error("Intersection index overflow.");
    if (status == PHX_MC_INTERNAL_ERROR) throw std::logic_error("Exact intersection failed.");
    if (status != PHX_MC_OK) return false;
  }
  if (native_positive) native_ancestry.push_back(ancestor);
  return true;
}

bool write_cell_ancestry(const TetMesh& mesh, const MetricBoxGrid& grid, const AdaptInput& input,
                         const SourceBox& source, std::span<const StencilPoint> lifted,
                         TetEdit& edit, AdaptOutput& output) {
  std::vector<TetCoordinates> source_points(input.cell_ids.size());
  std::vector<SourceCellPlanes> planes(input.cell_ids.size());
  const bool adjacent = make_source_planes(mesh, input, source_points, planes);
  const double face_error = box_face_error(source.box, lifted);
  const auto sites = bind_source_sites(grid, input, source_points, planes, face_error);
  std::vector<std::size_t> seen(planes.size());
  std::vector<std::pair<std::int32_t, unsigned>> queue;
  CellIndex index;
  index.build(source_points);
  TetrahedronClipper clipper;
  std::vector<Id> exact_ancestry;
  const std::size_t count = edit.replacement_cells.size();
  output.cell_ids.reserve(count);
  output.cell_target_ids.reserve(count);
  output.cell_blocks.assign(count, source.block);
  output.ancestry_offsets.reserve(count + 1);
  output.ancestry_offsets.push_back(0);
  output.ancestry_ids.reserve(count);
  output.ancestry_kinds.assign(count, kRetessellated);
  output.cell_relocated.assign(count, 0);
  for (std::size_t row = 0; row < count; ++row) {
    const auto& vertices = grid.cells[row];
    TetPoints rounded;
    Tet target;
    for (int k = 0; k < 4; ++k) {
      const auto vertex = vertices[k];
      rounded[k] = grid.vertices[vertex].point.data();
      target[k] = {rounded[k], lifted[vertex].exact ? nullptr : &lifted[vertex]};
    }
    // The rounded and exact shells are homotopic inside the same box strata
    // because their vertices differ only in free coordinates. Positive exact
    // cells and the already-proven shell degree prove the exact cover too.
    layer_detail::CommonSigns signs;
    if (signs.orientation(target[0], target[1], target[2], target[3]) <= 0) return false;
    std::int32_t seed = -1;
    bool strict_seed = false;
    for (const auto vertex : vertices) {
      const auto& site = sites[vertex];
      if (site.interior) {
        seed = site.owner;
        strict_seed = true;
        break;
      }
      if (seed < 0 && site.closed) seed = site.owner;
    }
    TetEditCell& cell = edit.replacement_cells[row];
    exact_ancestry.clear();
    const auto append_both = [&](std::size_t candidate) {
      const Id id = input.cell_ids[candidate];
      cell.ancestor_cell_ids.push_back(id);
      exact_ancestry.push_back(id);
    };
    if (seed >= 0 && singleton_ancestor(seed, vertices, grid, sites, planes[seed],
                                       source_points[seed], face_error)) {
      append_both(seed);
    } else {
      using layer_detail::Contact;
      const Bounds native_box = tet_bounds(rounded);
      const Bounds exact_box = expanded_bounds(native_box, target);
      const auto collect = [&](std::size_t candidate, const Bounds& source_box) {
        return collect_ancestor(clipper, source_points[candidate], target,
                                open_overlap(native_box, source_box), input.cell_ids[candidate],
                                cell.ancestor_cell_ids, exact_ancestry);
      };
      const auto seed_faces = !adjacent || seed < 0 ? FaceEvidence{} :
          source_face_evidence(planes[seed], source_points[seed], target, face_error);
      if (adjacent && seed >= 0 && (strict_seed || seed_faces.contact == Contact::kPositive)) {
        // In either embedding, positive overlaps meet the open target interior.
        // Join interior witnesses inside that convex open set by a generic
        // path avoiding the finite source edge/vertex skeleton. It crosses
        // source facets only in their relative interiors, so the positive
        // cells for each embedding are face-connected to this common seed.
        append_both(seed);
        queue.clear();
        queue.emplace_back(seed, seed_faces.neighbors);
        seen[seed] = row + 1;
        for (std::size_t at = 0; at < queue.size(); ++at) {
          const auto [current, faces] = queue[at];
          for (int k = 0; k < 4; ++k) {
            const auto candidate = planes[current].neighbors[k];
            if (!(faces & (1U << k)) || candidate < 0 || seen[candidate] == row + 1) continue;
            seen[candidate] = row + 1;
            const auto& bounds = index.bounds(candidate);
            if (!open_overlap(exact_box, bounds)) continue;
            const auto proof = source_face_evidence(planes[candidate], source_points[candidate],
                                                    target, face_error);
            if (proof.contact == Contact::kDisjoint) continue;
            if (proof.contact == Contact::kPositive) {
              append_both(candidate);
            } else {
              const auto before = cell.ancestor_cell_ids.size() + exact_ancestry.size();
              if (!collect(candidate, bounds)) return false;
              if (before == cell.ancestor_cell_ids.size() + exact_ancestry.size()) continue;
            }
            queue.emplace_back(candidate, proof.neighbors);
          }
        }
      } else if (!index.visit(exact_box, collect)) {
        return false;
      }
    }
    if (exact_ancestry.empty() || cell.ancestor_cell_ids.empty()) return false;
    std::sort(cell.ancestor_cell_ids.begin(), cell.ancestor_cell_ids.end());
    std::sort(exact_ancestry.begin(), exact_ancestry.end());
    output.cell_ids.push_back(cell.id);
    output.cell_target_ids.push_back(input.next_target_cell + row);
    output.ancestry_ids.insert(output.ancestry_ids.end(), exact_ancestry.begin(), exact_ancestry.end());
    output.ancestry_offsets.push_back(static_cast<std::int64_t>(output.ancestry_ids.size()));
  }
  return true;
}

bool fatal(TetEditStatus status) {
  return status == TetEditStatus::kStaleGeneration || status == TetEditStatus::kAllocationFailure ||
         status == TetEditStatus::kIndexOverflow;
}

}  // namespace

std::optional<TetEditStatus> adapt_metric_box(TetMesh& mesh, const AdaptInput& input,
                                             const CarrierSource& source, bool scalar,
                                             AdaptOutput& output) {
  if (source.cell_count() > kMaxBoxSourceCells) return std::nullopt;
  if (!input.topology_operations || !input.relocation || input.exact_source != nullptr ||
      !input.protected_edges.empty() || mesh.generation() != 0 || mesh.implicit_source() ||
      mesh.periodic_identity().bound || mesh.metric_active() || !mesh.unmet_records().empty() ||
      mesh.policy() != BoundaryPolicy::kConforming || !mesh.source_map_id().empty()) {
    return std::nullopt;
  }
  TetMeshSnapshot snapshot;
  const TetEditStatus captured = mesh.snapshot(snapshot);
  if (captured != TetEditStatus::kApplied) {
    return fatal(captured) ? std::optional{captured} : std::nullopt;
  }
  SourceBox original;
  if (!snapshot.constrained_segments.empty() || !snapshot.periodic_orbits.empty() ||
      !snapshot.quotient_cells.empty() || !snapshot.periodic_entity_maps.empty() ||
      !bind_original_vertices(mesh, input, snapshot, original) ||
      !bind_original_cells(mesh, input, snapshot, original) ||
      !bind_original_boundary(mesh, snapshot, original) ||
      !source.matches_roundings(input.source_points, input.source_cells)) return std::nullopt;

  const auto propose = scalar ? propose_scalar_box_grid : propose_metric_layer_grid;
  const auto grid = propose(input, source, original.box.bounds.low, original.box.bounds.high);
  if (!grid) return std::nullopt;
  TetEditBatch batch;
  batch.expected_generation = snapshot.generation;
  batch.identity_bound = snapshot.identity_bound;
  batch.source_id = std::move(snapshot.source_id);
  batch.source_revision = std::move(snapshot.source_revision);
  batch.source_map_id = std::move(snapshot.source_map_id);
  batch.frame_id = std::move(snapshot.frame_id);
  batch.periodic_identity = std::move(snapshot.periodic_identity);
  TetEdit& edit = batch.operations.emplace_back();
  edit.kind = TetEditKind::kRetriangulate;
  VertexOutput vertices;
  auto status = bind_grid_vertices(*grid, input, original, snapshot, vertices, edit);
  if (status != TetEditStatus::kApplied) return status;
  status = bind_grid_cells(*grid, input, original, vertices, edit);
  if (status != TetEditStatus::kApplied) return status;
  edit.removed_cell_ids = input.cell_ids;
  edit.removed_facet_ids.reserve(snapshot.constrained_facets.size());
  for (const auto& facet : snapshot.constrained_facets) edit.removed_facet_ids.push_back(facet.id);
  std::sort(edit.removed_facet_ids.begin(), edit.removed_facet_ids.end());

  AdaptOutput staged;
  std::vector<StencilPoint> lifted;
  if (!write_vertices(*grid, input, original.box, vertices, lifted, edit, staged)) return std::nullopt;
  if (!write_cell_ancestry(mesh, *grid, input, original, lifted, edit, staged)) return std::nullopt;
  staged.status = AdaptStatus::kComplete;
  staged.passes = 1;
  staged.counters.retriangulations = 1;
  std::vector<TetEditResult> results;
  const TetEditStatus applied = mesh.apply_edit_batch(batch, results, input.source_vertex_ids);
  if (applied != TetEditStatus::kApplied) {
    return fatal(applied) ? std::optional{applied} : std::nullopt;
  }
  output = std::move(staged);
  return TetEditStatus::kApplied;
}

}  // namespace phx::mc::metric
