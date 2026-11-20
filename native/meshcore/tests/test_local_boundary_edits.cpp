//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// LocalIncidence::collapse_boundary and swap_boundary_edge against
// TetMesh::apply_edit_batch: random single-operation boundary collapses (in
// place and with a moved survivor) and coplanar boundary edge swaps, valid
// and perturbed, on small PLC boxes (six side facet groups of two source
// triangles, twelve source segments, an interior interface, protected
// vertices, the fixed policy) must give the batch's verdict and, when
// applied, its state. Their cost is star-local: the time per edit does not
// grow with the mesh.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

#include "cavity.hpp"
#include "check.hpp"
#include "local_incidence.hpp"
#include "phydrax_meshcore.h"
#include "plc3d.hpp"
#include "predicates.hpp"
#include "tet_mesh.hpp"

using namespace phx::mc;
using detail::LocalIncidence;

namespace {

using FaceIds = std::array<std::uint64_t, 3>;
using EdgeIds = std::array<std::uint64_t, 2>;
using Point = std::array<double, 3>;

FaceIds sorted_face(FaceIds face) {
  std::sort(face.begin(), face.end());
  return face;
}

EdgeIds sorted_edge(std::uint64_t a, std::uint64_t b) { return {std::min(a, b), std::max(a, b)}; }

Point cross(const Point& u, const Point& v) {
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
}

Point minus(const double* p, const double* q) { return {p[0] - q[0], p[1] - q[1], p[2] - q[2]}; }

double dot(const Point& u, const Point& v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

// A Kuhn-grid box [0, n0] x [0, n1] x [0, n2] (powers of two, so every
// witness parameter is exact) whose six sides are facet groups of two source
// triangles each and whose twelve edges are source segments; with
// `two_regions` the cells with x > n0 / 2 are region 1, the plane x = n0 / 2
// is an interior interface group and the four sides it crosses are split
// into one group per region.
struct Piece {
  int axis = 0;
  double value = 0.0;
  double low = 0.0;
  double high = 0.0;
};
struct Box {
  std::array<int, 3> n{};
  bool two_regions = false;
  std::vector<double> points;
  std::vector<std::int32_t> tetrahedra;
  std::vector<std::int32_t> regions;
  std::vector<std::int32_t> faces;
  std::vector<std::int32_t> face_sources;
  std::vector<std::int8_t> orientations;
  std::vector<std::int32_t> segments;
  std::vector<std::int32_t> segment_sources;
  std::vector<double> source_points;
  std::vector<std::int32_t> source_faces;
  std::vector<std::int32_t> face_groups;
  std::vector<std::int64_t> face_ids;
  std::vector<double> face_tolerances;
  std::vector<std::int32_t> source_segments;
  std::vector<std::int32_t> segment_groups;
  std::vector<std::int64_t> segment_ids;
  std::vector<double> segment_tolerances;
  std::vector<SourceWitness> witnesses;
  // Source points of each face group and segment group, and each face
  // group's plane piece (its x range).
  std::map<std::int32_t, std::set<std::int32_t>> face_group_points;
  std::map<std::int32_t, std::set<std::int32_t>> segment_group_points;
  std::map<std::int32_t, Piece> pieces;
};

bool on_piece(const Box& box, std::int32_t group, const Point& p) {
  const Piece& piece = box.pieces.at(group);
  return p[static_cast<std::size_t>(piece.axis)] == piece.value && p[0] >= piece.low &&
         p[0] <= piece.high;
}

// The source witness of a point on face group `group`, on the source row
// holding it.
std::optional<SourceWitness> facet_witness(const Box& box, std::int32_t group, const Point& p) {
  const int axis = box.pieces.at(group).axis;
  const int b = (axis + 1) % 3;
  const int c = (axis + 2) % 3;
  for (std::size_t row = 0; row < box.face_groups.size(); ++row) {
    if (box.face_groups[row] != group) continue;
    const double* q0 = box.source_points.data() + 3 * box.source_faces[3 * row];
    const double* q1 = box.source_points.data() + 3 * box.source_faces[3 * row + 1];
    const double* q2 = box.source_points.data() + 3 * box.source_faces[3 * row + 2];
    const double e1b = q1[b] - q0[b], e1c = q1[c] - q0[c];
    const double e2b = q2[b] - q0[b], e2c = q2[c] - q0[c];
    const double db = p[static_cast<std::size_t>(b)] - q0[b];
    const double dc = p[static_cast<std::size_t>(c)] - q0[c];
    const double det = e1b * e2c - e1c * e2b;
    const double u = (db * e2c - dc * e2b) / det;
    const double v = (e1b * dc - e1c * db) / det;
    if (u < 0.0 || v < 0.0 || u + v > 1.0) continue;
    SourceWitness witness;
    witness.stratum = SourceStratum::kFacet;
    witness.entity = static_cast<std::int32_t>(row);
    witness.parameters[0] = u;
    witness.parameters[1] = v;
    return witness;
  }
  return std::nullopt;
}

std::optional<SourceWitness> segment_witness(const Box& box, std::int32_t row, const Point& p) {
  const double* a = box.source_points.data() + 3 * box.source_segments[2 * row];
  const double* b = box.source_points.data() + 3 * box.source_segments[2 * row + 1];
  int axis = 0;
  while (a[axis] == b[axis]) ++axis;
  for (int other = 0; other < 3; ++other) {
    if (other != axis && p[static_cast<std::size_t>(other)] != a[other]) return std::nullopt;
  }
  const double t = (p[static_cast<std::size_t>(axis)] - a[axis]) / (b[axis] - a[axis]);
  if (t < 0.0 || t > 1.0) return std::nullopt;
  SourceWitness witness;
  witness.stratum = SourceStratum::kSegment;
  witness.entity = row;
  witness.parameters[0] = t;
  return witness;
}

Box make_box(std::array<int, 3> n, bool two_regions) {
  Box box;
  box.n = n;
  box.two_regions = two_regions;
  const auto index = [&](int i, int j, int k) { return i + (n[0] + 1) * (j + (n[1] + 1) * k); };
  for (int k = 0; k <= n[2]; ++k) {
    for (int j = 0; j <= n[1]; ++j) {
      for (int i = 0; i <= n[0]; ++i) box.points.insert(box.points.end(), {double(i), double(j), double(k)});
    }
  }
  // Source points: the eight corners (bit i + 2 j + 4 k), then the
  // interface's four (8 + bit y + 2 bit z).
  for (int corner = 0; corner < 8; ++corner) {
    box.source_points.insert(box.source_points.end(),
                             {double((corner & 1) * n[0]), double(((corner >> 1) & 1) * n[1]),
                              double(((corner >> 2) & 1) * n[2])});
  }
  for (int corner = 0; corner < 4; ++corner) {
    box.source_points.insert(box.source_points.end(),
                             {0.5 * n[0], double((corner & 1) * n[1]), double((corner >> 1) * n[2])});
  }
  // The source point at x step `xs` (0, the middle, n0) and y, z bits.
  const auto point_index = [](int xs, int yb, int zb) {
    return xs == 1 ? 8 + yb + 2 * zb : (xs / 2) + 2 * yb + 4 * zb;
  };
  const auto add_piece = [&](std::int32_t group, int axis, int bit, int low, int high) {
    Piece piece;
    piece.axis = axis;
    piece.value = axis == 0 ? 0.5 * low * n[0] : bit * double(n[static_cast<std::size_t>(axis)]);
    piece.low = 0.5 * low * n[0];
    piece.high = 0.5 * high * n[0];
    box.pieces[group] = piece;
    // The quad in cyclic order over the piece's two in-plane axes.
    std::array<std::int32_t, 4> quad{};
    const std::array<std::array<int, 2>, 4> cycle{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    for (std::size_t q = 0; q < 4; ++q) {
      if (axis == 0) {
        quad[q] = point_index(low, cycle[q][0], cycle[q][1]);
      } else {
        const int xs = cycle[q][0] == 0 ? low : high;
        quad[q] = axis == 1 ? point_index(xs, bit, cycle[q][1]) : point_index(xs, cycle[q][1], bit);
      }
    }
    for (const auto& triangle : {std::array<std::int32_t, 3>{quad[0], quad[1], quad[2]},
                                 std::array<std::int32_t, 3>{quad[0], quad[2], quad[3]}}) {
      box.source_faces.insert(box.source_faces.end(), triangle.begin(), triangle.end());
      box.face_groups.push_back(group);
      box.face_ids.push_back(static_cast<std::int64_t>(box.face_ids.size()));
      box.face_tolerances.push_back(0.0);
      box.face_group_points[group].insert(triangle.begin(), triangle.end());
    }
  };
  // Groups 0..5 the sides (x = 0 and n0 by low = high), 6 the interface,
  // 7..10 the region-1 halves of sides 2..5.
  add_piece(0, 0, 0, 0, 0);
  add_piece(1, 0, 1, 2, 2);
  for (std::int32_t side = 2; side < 6; ++side) {
    add_piece(side, side / 2, side % 2, 0, two_regions ? 1 : 2);
    if (two_regions) add_piece(side + 5, side / 2, side % 2, 1, 2);
  }
  if (two_regions) add_piece(6, 0, 0, 1, 1);
  for (int axis = 0; axis < 3; ++axis) {
    for (int bits = 0; bits < 4; ++bits) {
      const int b = (axis + 1) % 3;
      const int c = (axis + 2) % 3;
      const int base = ((bits & 1) << b) | (((bits >> 1) & 1) << c);
      const std::int32_t row = static_cast<std::int32_t>(box.segment_groups.size());
      box.source_segments.insert(box.source_segments.end(), {base, base | (1 << axis)});
      box.segment_groups.push_back(row);
      box.segment_ids.push_back(row);
      box.segment_tolerances.push_back(0.0);
      box.segment_group_points[row] = {base, base | (1 << axis)};
    }
  }

  // Kuhn cells.
  constexpr int kAxes[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        for (const auto& axes : kAxes) {
          int corner[3]{i, j, k};
          std::array<std::int32_t, 4> cell{index(i, j, k), 0, 0, 0};
          for (int step = 0; step < 3; ++step) {
            ++corner[axes[step]];
            cell[static_cast<std::size_t>(step + 1)] = index(corner[0], corner[1], corner[2]);
          }
          const double* p[4];
          for (int q = 0; q < 4; ++q) p[q] = box.points.data() + 3 * cell[static_cast<std::size_t>(q)];
          if (orient3d(p[0], p[1], p[2], p[3]) < 0) std::swap(cell[2], cell[3]);
          box.tetrahedra.insert(box.tetrahedra.end(), cell.begin(), cell.end());
          box.regions.push_back(two_regions && i >= n[0] / 2 ? 1 : 0);
        }
      }
    }
  }

  // Constrained faces: the hull and the interface.
  std::map<std::array<std::int32_t, 3>, std::vector<std::pair<std::size_t, int>>> owners;
  const std::size_t cells = box.tetrahedra.size() / 4;
  for (std::size_t row = 0; row < cells; ++row) {
    for (int face = 0; face < 4; ++face) {
      std::array<std::int32_t, 3> corners{};
      oriented_facet(box.tetrahedra.data() + 4 * row, face, corners.data());
      std::sort(corners.begin(), corners.end());
      owners[corners].emplace_back(row, face);
    }
  }
  for (const auto& [key, sides] : owners) {
    std::int32_t group = -1;
    if (sides.size() == 1) {
      for (const auto& [candidate, piece] : box.pieces) {
        if (candidate == 6 || group >= 0) continue;
        bool on = true;
        for (std::int32_t vertex : key) {
          on = on && on_piece(box, candidate,
                              {box.points[3 * vertex], box.points[3 * vertex + 1],
                               box.points[3 * vertex + 2]});
        }
        if (on) group = candidate;
      }
    } else if (two_regions && box.regions[sides[0].first] != box.regions[sides[1].first]) {
      group = 6;
    }
    if (group < 0) continue;
    const auto [row, face] =
        sides.size() == 1 || box.regions[sides[0].first] == 0 ? sides[0] : sides[1];
    std::array<std::int32_t, 3> corners{};
    oriented_facet(box.tetrahedra.data() + 4 * row, face, corners.data());
    box.faces.insert(box.faces.end(), corners.begin(), corners.end());
    box.face_sources.push_back(group);
    std::size_t source_row = 0;
    while (box.face_groups[source_row] != group) ++source_row;
    const double* q0 = box.source_points.data() + 3 * box.source_faces[3 * source_row];
    const double* q1 = box.source_points.data() + 3 * box.source_faces[3 * source_row + 1];
    const double* q2 = box.source_points.data() + 3 * box.source_faces[3 * source_row + 2];
    const double* p0 = box.points.data() + 3 * corners[0];
    const double* p1 = box.points.data() + 3 * corners[1];
    const double* p2 = box.points.data() + 3 * corners[2];
    box.orientations.push_back(
        dot(cross(minus(p1, p0), minus(p2, p0)), cross(minus(q1, q0), minus(q2, q0))) > 0.0 ? 1 : -1);
  }
  // Mesh segments along the source segments.
  for (std::size_t row = 0; row < box.segment_groups.size(); ++row) {
    const double* a = box.source_points.data() + 3 * box.source_segments[2 * row];
    const double* b = box.source_points.data() + 3 * box.source_segments[2 * row + 1];
    int axis = 0;
    while (a[axis] == b[axis]) ++axis;
    for (int step = 0; step < n[static_cast<std::size_t>(axis)]; ++step) {
      int first[3] = {int(a[0]), int(a[1]), int(a[2])};
      first[axis] = step;
      int second[3] = {first[0], first[1], first[2]};
      second[axis] = step + 1;
      box.segments.insert(box.segments.end(), {index(first[0], first[1], first[2]),
                                               index(second[0], second[1], second[2])});
      box.segment_sources.push_back(static_cast<std::int32_t>(row));
    }
  }
  // Witnesses: corners none, edge points on their segment, side and
  // interface points on their facet.
  const std::size_t vertex_count = box.points.size() / 3;
  box.witnesses.resize(vertex_count);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    const Point p{box.points[3 * vertex], box.points[3 * vertex + 1], box.points[3 * vertex + 2]};
    std::vector<int> sides;
    for (int axis = 0; axis < 3; ++axis) {
      if (p[static_cast<std::size_t>(axis)] == 0.0 ||
          p[static_cast<std::size_t>(axis)] == n[static_cast<std::size_t>(axis)]) {
        sides.push_back(axis);
      }
    }
    if (sides.size() == 2) {
      for (std::int32_t row = 0; row < 12; ++row) {
        if (const auto witness = segment_witness(box, row, p)) box.witnesses[vertex] = *witness;
      }
    } else if (sides.size() == 1) {
      for (const auto& [group, piece] : box.pieces) {
        if (group != 6 && on_piece(box, group, p)) {
          box.witnesses[vertex] = *facet_witness(box, group, p);
          break;
        }
      }
    } else if (sides.empty() && two_regions && p[0] == 0.5 * n[0]) {
      box.witnesses[vertex] = *facet_witness(box, 6, p);
    }
  }
  return box;
}

bool build(TetMesh& mesh, const Box& box, std::span<const std::int8_t> protected_vertices) {
  const SourceComplexView source{box.source_points,   box.source_faces,     box.face_groups,
                                 box.face_ids,        box.face_tolerances,  box.source_segments,
                                 box.segment_groups,  box.segment_ids,      box.segment_tolerances,
                                 box.witnesses};
  const std::int32_t status =
      mesh.build(box.points, box.tetrahedra, box.regions, box.faces, box.face_sources,
                 box.orientations, box.segments, box.segment_sources, {}, &source,
                 protected_vertices);
  PHX_CHECK(status == PHX_MC_OK);
  return status == PHX_MC_OK;
}

// The constrained facets and segments by sorted vertex IDs, kept with the
// local mesh.
struct Banks {
  std::map<FaceIds, TetSnapshotFacet> facets;
  std::map<EdgeIds, TetSnapshotSegment> segments;

  explicit Banks(const TetMesh& mesh) {
    TetMeshSnapshot snapshot;
    PHX_CHECK(mesh.snapshot(snapshot) == TetEditStatus::kApplied);
    for (const auto& facet : snapshot.constrained_facets) facets[sorted_face(facet.vertices)] = facet;
    for (const auto& segment : snapshot.constrained_segments) {
      segments[sorted_edge(segment.vertices[0], segment.vertices[1])] = segment;
    }
  }
  void apply(const TetEdit& operation) {
    for (std::uint64_t id : operation.removed_facet_ids) {
      std::erase_if(facets, [&](const auto& entry) { return entry.second.id == id; });
    }
    for (std::uint64_t id : operation.removed_segment_ids) {
      std::erase_if(segments, [&](const auto& entry) { return entry.second.id == id; });
    }
    for (const TetEditFacet& facet : operation.replacement_facets) {
      facets[sorted_face(facet.vertices)] =
          TetSnapshotFacet{facet.id, facet.vertices, facet.source, facet.orientation, false};
    }
    for (const TetEditSegment& segment : operation.replacement_segments) {
      segments[sorted_edge(segment.vertices[0], segment.vertices[1])] =
          TetSnapshotSegment{segment.id, segment.vertices, segment.source, false};
    }
  }
};

enum class Perturbation { kNone, kFlipCell, kRegion, kDropFacet, kProtectFacet, kAncestor,
                          kReuseFacetId, kLabel, kReverseFacet };

struct Planned {
  TetEdit operation;
  std::int32_t removed = -1;
  std::int32_t survivor = -1;
  bool moved = false;
  bool out_of_domain = false;
};

class Generator {
 public:
  Generator(const Box& box, std::uint32_t seed) : box_(box), random_(seed) {}

  std::uint64_t fresh_id() { return next_id_++; }
  bool chance(double p) { return std::uniform_real_distribution<double>(0.0, 1.0)(random_) < p; }
  std::size_t pick(std::size_t count) {
    return std::uniform_int_distribution<std::size_t>(0, count - 1)(random_);
  }

  std::vector<std::int32_t> link(const TetMesh& mesh, const LocalIncidence& incidence,
                                 std::int32_t vertex) {
    std::set<std::int32_t> vertices;
    for (std::int32_t cell : incidence.vertex_star(vertex)) {
      for (std::int32_t other : mesh.tet(cell).v) {
        if (other != vertex) vertices.insert(other);
      }
    }
    return {vertices.begin(), vertices.end()};
  }

  // The source points of a witness's support (none: not a source point).
  std::set<std::int32_t> support(const TetMesh& mesh, std::int32_t slot,
                                 const SourceWitness& witness, const double* carrier) {
    std::set<std::int32_t> points;
    const auto row = static_cast<std::size_t>(witness.entity);
    if (witness.stratum == SourceStratum::kSegment) {
      if (witness.parameters[0] != 1.0) points.insert(box_.source_segments[2 * row]);
      if (witness.parameters[0] != 0.0) points.insert(box_.source_segments[2 * row + 1]);
    } else if (witness.stratum == SourceStratum::kFacet) {
      if (witness.parameters[0] + witness.parameters[1] != 1.0) {
        points.insert(box_.source_faces[3 * row]);
      }
      if (witness.parameters[0] != 0.0) points.insert(box_.source_faces[3 * row + 1]);
      if (witness.parameters[1] != 0.0) points.insert(box_.source_faces[3 * row + 2]);
    } else {
      for (std::size_t point = 0; point < box_.source_points.size() / 3; ++point) {
        if (std::equal(carrier, carrier + 3, box_.source_points.data() + 3 * point)) {
          points.insert(static_cast<std::int32_t>(point));
        }
      }
      if (points.empty()) points.insert(-1 - slot);
    }
    return points;
  }

  // A collapse of a facet or segment vertex into a link vertex, in the
  // contract's domain unless `out_of_domain`.
  std::optional<Planned> collapse(const TetMesh& mesh, const LocalIncidence& incidence,
                                  const Banks& banks, Perturbation perturbation,
                                  bool allow_move, bool allow_out_of_domain) {
    Planned plan;
    std::int32_t r = -1;
    for (int attempt = 0; attempt < 64 && r < 0; ++attempt) {
      const auto slot = static_cast<std::int32_t>(pick(mesh.vertex_slot_count()));
      const std::int8_t dimension = mesh.vertex_dimension(slot);
      if (mesh.live_vertex(slot) && (dimension == 1 || dimension == 2)) r = slot;
    }
    if (r < 0) return std::nullopt;
    const std::vector<std::int32_t> neighbors = link(mesh, incidence, r);
    const std::int32_t s = neighbors[pick(neighbors.size())];
    if (s < 0) return std::nullopt;
    const std::uint64_t r_id = mesh.vertex_id(r);
    const std::uint64_t s_id = mesh.vertex_id(s);
    const LocalIncidence::Star r_star = incidence.vertex_star(r);

    // r's facets and segments, and the source rule.
    std::vector<TetSnapshotFacet> r_facets;
    std::vector<TetSnapshotSegment> r_segments;
    for (const auto& [key, facet] : banks.facets) {
      if (std::find(key.begin(), key.end(), r_id) != key.end()) r_facets.push_back(facet);
    }
    for (const auto& [key, segment] : banks.segments) {
      if (key[0] == r_id || key[1] == r_id) r_segments.push_back(segment);
    }
    std::optional<std::set<std::int32_t>> entity;
    const auto narrow = [&](const std::set<std::int32_t>& points) {
      if (!entity) {
        entity = points;
        return;
      }
      std::set<std::int32_t> common;
      std::set_intersection(entity->begin(), entity->end(), points.begin(), points.end(),
                            std::inserter(common, common.end()));
      entity = common;
    };
    for (const auto& facet : r_facets) narrow(box_.face_group_points.at(facet.source));
    for (const auto& segment : r_segments) narrow(box_.segment_group_points.at(segment.source));

    // The survivor's move: to the midpoint with a vertex on its stratum.
    std::optional<TetEditVertex> move;
    if (allow_move && chance(0.35)) {
      const SourceWitness& witness = mesh.witness(s);
      const std::vector<std::int32_t> others = link(mesh, incidence, s);
      const std::int32_t x = others[pick(others.size())];
      if (x >= 0 && witness.stratum != SourceStratum::kNone) {
        Point midpoint{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
          midpoint[axis] = 0.5 * (mesh.point(s)[axis] + mesh.point(x)[axis]);
        }
        std::optional<SourceWitness> moved;
        if (witness.stratum == SourceStratum::kFacet) {
          const std::int32_t group = box_.face_groups[static_cast<std::size_t>(witness.entity)];
          if (on_piece(box_, group, midpoint)) moved = facet_witness(box_, group, midpoint);
        } else {
          moved = segment_witness(box_, witness.entity, midpoint);
        }
        if (chance(0.1)) {
          // Off its source: the carrier no longer realizes the witness.
          midpoint[0] += 0.25;
          if (!moved) moved = witness;
        }
        if (moved) {
          TetEditVertex vertex;
          vertex.id = s_id;
          vertex.carrier = midpoint;
          vertex.witness = *moved;
          vertex.protection_radius = mesh.protection_radius(s);
          vertex.protected_vertex = mesh.protected_vertex(s);
          move = vertex;
        }
      }
    }
    const SourceWitness& s_witness = move ? move->witness : mesh.witness(s);
    const double* s_carrier = move ? move->carrier.data() : mesh.point(s);
    const std::set<std::int32_t> stencil = support(mesh, s, s_witness, s_carrier);
    bool in_domain = entity && entity->size() >= 2 &&
                     std::includes(entity->begin(), entity->end(), stencil.begin(), stencil.end());
    if (!r_segments.empty() && !banks.segments.contains(sorted_edge(r_id, s_id))) in_domain = false;
    if (!in_domain && !(allow_out_of_domain && chance(0.15))) return std::nullopt;
    plan.out_of_domain = !in_domain;
    plan.removed = r;
    plan.survivor = s;
    plan.moved = move.has_value();

    TetEdit& operation = plan.operation;
    operation.kind = TetEditKind::kCollapse;
    operation.collapse_pairs.push_back({r_id, s_id});
    operation.removed_vertex_ids.push_back(r_id);
    if (move) operation.changed_vertices.push_back(*move);
    const auto add_cells = [&](const LocalIncidence::Star& star, std::int32_t skip, bool image) {
      for (std::int32_t cell : star) {
        const std::uint64_t cell_id = mesh.cell_id(cell);
        if (std::find(operation.removed_cell_ids.begin(), operation.removed_cell_ids.end(),
                      cell_id) == operation.removed_cell_ids.end()) {
          operation.removed_cell_ids.push_back(cell_id);
        }
        const Tetrahedron& tet = mesh.tet(cell);
        if (vertex_slot(tet, skip) >= 0) continue;
        TetEditCell replacement;
        replacement.id = fresh_id();
        replacement.region = mesh.complex().region(cell);
        replacement.ancestor_cell_ids = {cell_id};
        for (std::size_t corner = 0; corner < 4; ++corner) {
          const std::int32_t vertex = tet.v[corner];
          replacement.vertices[corner] = image && vertex == r ? s_id : mesh.vertex_id(vertex);
          replacement.facet_sources[corner] = mesh.complex().constraint(cell, static_cast<int>(corner));
        }
        operation.replacement_cells.push_back(replacement);
      }
    };
    add_cells(r_star, s, true);
    if (move) add_cells(incidence.vertex_star(s), r, false);
    const auto image = [&](std::uint64_t id) { return id == r_id ? s_id : id; };
    for (const auto& facet : r_facets) {
      operation.removed_facet_ids.push_back(facet.id);
      if (std::find(facet.vertices.begin(), facet.vertices.end(), s_id) != facet.vertices.end()) {
        continue;
      }
      TetEditFacet replacement;
      replacement.id = fresh_id();
      replacement.vertices = {image(facet.vertices[0]), image(facet.vertices[1]),
                              image(facet.vertices[2])};
      replacement.source = facet.source;
      replacement.orientation = facet.orientation;
      operation.replacement_facets.push_back(replacement);
    }
    for (const auto& segment : r_segments) {
      operation.removed_segment_ids.push_back(segment.id);
      if (segment.vertices[0] == s_id || segment.vertices[1] == s_id) continue;
      TetEditSegment replacement;
      replacement.id = fresh_id();
      replacement.vertices = {image(segment.vertices[0]), image(segment.vertices[1])};
      replacement.source = segment.source;
      operation.replacement_segments.push_back(replacement);
    }
    perturb(operation, perturbation, banks);
    return plan;
  }

  // A swap of a boundary edge between two facets of one group.
  std::optional<Planned> swap(const TetMesh& mesh, const LocalIncidence& incidence,
                              const Banks& banks, Perturbation perturbation) {
    auto chosen = banks.facets.begin();
    std::advance(chosen, static_cast<std::ptrdiff_t>(pick(banks.facets.size())));
    const TetSnapshotFacet first = chosen->second;
    const std::size_t corner = pick(3);
    const std::uint64_t a_id = first.vertices[corner];
    const std::uint64_t b_id = first.vertices[(corner + 1) % 3];
    if (banks.segments.contains(sorted_edge(a_id, b_id))) return std::nullopt;
    std::vector<TetSnapshotFacet> around;
    for (const auto& [key, facet] : banks.facets) {
      if (std::find(key.begin(), key.end(), a_id) != key.end() &&
          std::find(key.begin(), key.end(), b_id) != key.end()) {
        around.push_back(facet);
      }
    }
    if (around.size() != 2 || around[0].source != around[1].source ||
        around[0].orientation != around[1].orientation) {
      return std::nullopt;
    }
    const std::int32_t a = mesh.vertex_slot(a_id);
    const std::int32_t b = mesh.vertex_slot(b_id);
    const auto apex = [&](const TetSnapshotFacet& facet) {
      for (std::uint64_t id : facet.vertices) {
        if (id != a_id && id != b_id) return mesh.vertex_slot(id);
      }
      return -1;
    };
    const TetSnapshotFacet& abc = around[0].id == first.id ? around[0] : around[1];
    const TetSnapshotFacet& abd = around[0].id == first.id ? around[1] : around[0];
    const std::int32_t c = apex(abc);
    const std::int32_t d = apex(abd);
    if (!incidence.edge_star(c, d).empty()) return std::nullopt;
    const std::vector<std::int32_t> fan(incidence.edge_star(a, b).begin(),
                                        incidence.edge_star(a, b).end());
    std::map<std::int32_t, std::vector<std::int32_t>> ring;
    for (std::int32_t cell : fan) {
      std::vector<std::int32_t> rest;
      for (std::int32_t vertex : mesh.tet(cell).v) {
        if (vertex != a && vertex != b) rest.push_back(vertex);
      }
      ring[rest[0]].push_back(rest[1]);
      ring[rest[1]].push_back(rest[0]);
    }
    if (ring[c].size() != 1 || ring[d].size() != 1) return std::nullopt;
    std::vector<std::int32_t> path{c};
    while (path.back() != d && path.size() <= ring.size()) {
      const std::vector<std::int32_t>& next = ring[path.back()];
      const std::int32_t previous = path.size() > 1 ? path[path.size() - 2] : -1;
      path.push_back(next[0] != previous ? next[0] : next[1]);
    }
    if (path.back() != d || path.size() != ring.size()) return std::nullopt;

    Planned plan;
    TetEdit& operation = plan.operation;
    operation.kind = TetEditKind::kRingReconnect;
    for (std::int32_t cell : fan) operation.removed_cell_ids.push_back(mesh.cell_id(cell));
    operation.removed_facet_ids = {abc.id, abd.id};
    if (chance(0.5)) std::swap(operation.removed_facet_ids[0], operation.removed_facet_ids[1]);
    // acd and bcd, oriented like abc.
    const auto normal = [&](std::int32_t p, std::int32_t q, std::int32_t t) {
      return cross(minus(mesh.point(q), mesh.point(p)), minus(mesh.point(t), mesh.point(p)));
    };
    const Point reference = normal(mesh.vertex_slot(abc.vertices[0]),
                                   mesh.vertex_slot(abc.vertices[1]),
                                   mesh.vertex_slot(abc.vertices[2]));
    std::map<FaceIds, std::int32_t> new_faces;
    for (std::int32_t top : {a, b}) {
      TetEditFacet facet;
      facet.id = fresh_id();
      facet.source = abc.source;
      facet.orientation = abc.orientation;
      const bool aligned = dot(normal(top, c, d), reference) >= 0.0;
      facet.vertices = {mesh.vertex_id(top), mesh.vertex_id(aligned ? c : d),
                        mesh.vertex_id(aligned ? d : c)};
      new_faces[sorted_face(facet.vertices)] = facet.source;
      operation.replacement_facets.push_back(facet);
    }
    // The old faces' labels.
    std::map<FaceIds, std::int32_t> old_faces;
    for (std::int32_t cell : fan) {
      for (int face = 0; face < 4; ++face) {
        std::int32_t corners[3];
        oriented_facet(mesh.tet(cell).v, face, corners);
        old_faces[sorted_face({mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]),
                               mesh.vertex_id(corners[2])})] = mesh.complex().constraint(cell, face);
      }
    }
    // The ring c..d (closed by dc) triangulated by the fan from its first
    // vertex whose cones over a and b are consistently oriented.
    const std::size_t ring_size = path.size();
    const auto orientation = [&](std::int32_t p, std::int32_t q, std::int32_t t, std::int32_t u) {
      return orient3d(mesh.point(p), mesh.point(q), mesh.point(t), mesh.point(u));
    };
    std::size_t center = 0;
    int sign = 0;
    for (std::size_t candidate = 0; candidate < ring_size && sign == 0; ++candidate) {
      int common = 0;
      bool valid = true;
      for (std::size_t edge = 0; edge < ring_size && valid; ++edge) {
        const std::size_t next = (edge + 1) % ring_size;
        if (edge == candidate || next == candidate) continue;
        const int above = orientation(a, path[candidate], path[edge], path[next]);
        const int below = orientation(b, path[candidate], path[edge], path[next]);
        if (common == 0) common = above;
        valid = above != 0 && above == common && below == -common;
      }
      if (valid && common != 0) {
        center = candidate;
        sign = common;
      }
    }
    for (std::size_t edge = 0; edge < ring_size; ++edge) {
      const std::size_t next = (edge + 1) % ring_size;
      if (edge == center || next == center) continue;
      for (std::int32_t top : {a, b}) {
        std::array<std::int32_t, 4> cell{top, path[center], path[edge], path[next]};
        const int side = sign != 0 ? (top == a ? sign : -sign)
                                   : orientation(cell[0], cell[1], cell[2], cell[3]);
        if (side < 0) std::swap(cell[2], cell[3]);
        TetEditCell replacement;
        replacement.id = fresh_id();
        replacement.region = mesh.complex().region(fan.front());
        for (std::int32_t parent : fan) replacement.ancestor_cell_ids.push_back(mesh.cell_id(parent));
        for (std::size_t q = 0; q < 4; ++q) replacement.vertices[q] = mesh.vertex_id(cell[q]);
        for (int face = 0; face < 4; ++face) {
          std::int32_t corners[3];
          oriented_facet(cell.data(), face, corners);
          const FaceIds key = sorted_face(
              {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])});
          const auto made = new_faces.find(key);
          const auto old = old_faces.find(key);
          replacement.facet_sources[static_cast<std::size_t>(face)] =
              made != new_faces.end() ? made->second
              : old != old_faces.end() ? old->second
                                       : kNoConstraint;
        }
        operation.replacement_cells.push_back(replacement);
      }
    }
    perturb(operation, perturbation, banks);
    // A swap replaces both facets.
    plan.out_of_domain = perturbation == Perturbation::kDropFacet;
    return plan;
  }

 private:
  void perturb(TetEdit& operation, Perturbation perturbation, const Banks& banks) {
    if (operation.replacement_cells.empty()) return;
    TetEditCell& cell = operation.replacement_cells[pick(operation.replacement_cells.size())];
    switch (perturbation) {
      case Perturbation::kNone:
        break;
      case Perturbation::kFlipCell:
        std::swap(cell.vertices[0], cell.vertices[1]);
        std::swap(cell.facet_sources[0], cell.facet_sources[1]);
        break;
      case Perturbation::kRegion:
        ++cell.region;
        break;
      case Perturbation::kDropFacet:
        if (!operation.replacement_facets.empty()) operation.replacement_facets.pop_back();
        break;
      case Perturbation::kProtectFacet:
        if (!operation.replacement_facets.empty()) {
          operation.replacement_facets.front().protected_facet = true;
        }
        break;
      case Perturbation::kAncestor:
        cell.ancestor_cell_ids.push_back(cell.ancestor_cell_ids.empty()
                                             ? 0
                                             : cell.ancestor_cell_ids.front());
        break;
      case Perturbation::kReuseFacetId:
        if (!operation.replacement_facets.empty()) {
          operation.replacement_facets.front().id = banks.facets.begin()->second.id;
        }
        break;
      case Perturbation::kLabel:
        for (std::int32_t& source : cell.facet_sources) {
          if (source == kNoConstraint) {
            source = 0;
            break;
          }
        }
        break;
      case Perturbation::kReverseFacet:
        if (!operation.replacement_facets.empty()) {
          auto& vertices = operation.replacement_facets.front().vertices;
          std::swap(vertices[1], vertices[2]);
        }
        break;
    }
  }

  const Box& box_;
  std::mt19937 random_;
  std::uint64_t next_id_ = std::uint64_t{1} << 40;
};

bool same_witness(const SourceWitness& a, const SourceWitness& b) {
  return a.stratum == b.stratum && a.entity == b.entity && a.parameters[0] == b.parameters[0] &&
         a.parameters[1] == b.parameters[1] && a.deviation == b.deviation;
}

// The two states hold the same records (banks compared in ID order).
bool same_state(const TetMesh& first, const TetMesh& second) {
  TetMeshSnapshot x;
  TetMeshSnapshot y;
  if (first.snapshot(x) != TetEditStatus::kApplied ||
      second.snapshot(y) != TetEditStatus::kApplied || x.generation != y.generation ||
      x.vertices.size() != y.vertices.size() || x.cells.size() != y.cells.size() ||
      x.constrained_facets.size() != y.constrained_facets.size() ||
      x.constrained_segments.size() != y.constrained_segments.size()) {
    return false;
  }
  std::map<std::uint64_t, const TetSnapshotVertex*> vertices;
  for (const auto& vertex : x.vertices) vertices[vertex.id] = &vertex;
  for (const auto& vertex : y.vertices) {
    const auto found = vertices.find(vertex.id);
    if (found == vertices.end()) return false;
    const TetSnapshotVertex& other = *found->second;
    if (other.carrier != vertex.carrier || !same_witness(other.witness, vertex.witness) ||
        other.stratum_dimension != vertex.stratum_dimension ||
        other.protection_radius != vertex.protection_radius ||
        other.protected_vertex != vertex.protected_vertex ||
        other.has_edge_parameter != vertex.has_edge_parameter ||
        other.edge_endpoints != vertex.edge_endpoints ||
        other.edge_parameter != vertex.edge_parameter ||
        other.ancestry.size() != vertex.ancestry.size()) {
      return false;
    }
  }
  std::map<std::uint64_t, const TetSnapshotCell*> cells;
  for (const auto& cell : x.cells) cells[cell.id] = &cell;
  for (const auto& cell : y.cells) {
    const auto found = cells.find(cell.id);
    if (found == cells.end() || found->second->vertices != cell.vertices ||
        found->second->region != cell.region ||
        found->second->ancestor_cell_ids != cell.ancestor_cell_ids ||
        found->second->edit_kind != cell.edit_kind) {
      return false;
    }
  }
  const auto by_id = [](const auto& p, const auto& q) { return p.id < q.id; };
  std::sort(x.constrained_facets.begin(), x.constrained_facets.end(), by_id);
  std::sort(y.constrained_facets.begin(), y.constrained_facets.end(), by_id);
  for (std::size_t row = 0; row < x.constrained_facets.size(); ++row) {
    const auto& p = x.constrained_facets[row];
    const auto& q = y.constrained_facets[row];
    if (p.id != q.id || p.vertices != q.vertices || p.source != q.source ||
        p.orientation != q.orientation || p.protected_facet != q.protected_facet) {
      return false;
    }
  }
  std::sort(x.constrained_segments.begin(), x.constrained_segments.end(), by_id);
  std::sort(y.constrained_segments.begin(), y.constrained_segments.end(), by_id);
  for (std::size_t row = 0; row < x.constrained_segments.size(); ++row) {
    const auto& p = x.constrained_segments[row];
    const auto& q = y.constrained_segments[row];
    if (p.id != q.id || p.vertices != q.vertices || p.source != q.source ||
        p.protected_segment != q.protected_segment) {
      return false;
    }
  }
  return true;
}

// The complex is a consistent labeled triangulation: symmetric adjacency
// across equal faces with equal labels, every hull face labeled, and the
// labeled faces exactly the constrained facets with their sources.
bool consistent_complex(const TetMesh& mesh) {
  TetMeshSnapshot snapshot;
  if (mesh.snapshot(snapshot) != TetEditStatus::kApplied) return false;
  const TetrahedralComplex& complex = mesh.complex();
  std::map<FaceIds, std::int32_t> labeled;
  std::size_t finite = 0;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const auto slot = static_cast<std::int32_t>(row);
    if (!complex.live(slot)) continue;
    const Tetrahedron& tet = complex.tets[row];
    if (!is_ghost(tet)) ++finite;
    for (int face = 0; face < 4; ++face) {
      const std::int32_t neighbor = tet.n[face];
      if (!complex.live(neighbor)) return false;
      const int back = neighbor_slot(complex.tets[static_cast<std::size_t>(neighbor)], slot);
      if (back < 0) return false;
      std::array<std::int32_t, 3> mine{};
      std::array<std::int32_t, 3> theirs{};
      oriented_facet(tet.v, face, mine.data());
      oriented_facet(complex.tets[static_cast<std::size_t>(neighbor)].v, back, theirs.data());
      if (!(FacetKey::of(mine.data()) == FacetKey::of(theirs.data()).reversed()) ||
          complex.constraint(slot, face) != complex.constraint(neighbor, back)) {
        return false;
      }
      if (is_ghost(tet) || std::find(mine.begin(), mine.end(), kGhostVertex) != mine.end()) {
        continue;
      }
      const std::int32_t source = complex.constraint(slot, face);
      if (is_ghost(complex.tets[static_cast<std::size_t>(neighbor)]) && source == kNoConstraint) {
        return false;
      }
      if (source != kNoConstraint) {
        labeled[sorted_face({mesh.vertex_id(mine[0]), mesh.vertex_id(mine[1]),
                             mesh.vertex_id(mine[2])})] = source;
      }
    }
  }
  std::map<FaceIds, std::int32_t> facets;
  for (const auto& facet : snapshot.constrained_facets) {
    facets[sorted_face(facet.vertices)] = facet.source;
  }
  return finite == snapshot.cells.size() && labeled == facets;
}

bool incidence_current(const TetMesh& mesh, const LocalIncidence& incidence) {
  LocalIncidence rebuilt;
  if (!rebuilt.initialize(mesh)) return false;
  for (std::size_t slot = 0; slot < mesh.vertex_slot_count(); ++slot) {
    const auto vertex = static_cast<std::int32_t>(slot);
    if (!mesh.live_vertex(vertex)) continue;
    const LocalIncidence::Star maintained = incidence.vertex_star(vertex);
    const LocalIncidence::Star expected = rebuilt.vertex_star(vertex);
    if (!std::equal(maintained.begin(), maintained.end(), expected.begin(), expected.end())) {
      return false;
    }
  }
  return true;
}

TetEditStatus apply_batch(TetMesh& mesh, const TetEdit& operation) {
  const TetEditBatch batch{mesh.generation(), mesh.source_revision(), mesh.frame_id(),
                           mesh.identity_bound(), {operation}, mesh.source_id(),
                           mesh.source_map_id(), mesh.periodic_identity(), std::nullopt};
  std::vector<TetEditResult> results;
  const TetEditStatus status = mesh.apply_edit_batch(batch, results);
  return status == TetEditStatus::kApplied ? results.front().status : status;
}

TetEditStatus apply_local(TetMesh& mesh, LocalIncidence& incidence, const Planned& plan) {
  if (plan.operation.kind == TetEditKind::kRingReconnect) {
    return incidence.swap_boundary_edge(mesh, plan.operation);
  }
  const LocalIncidence::Star removed = incidence.vertex_star(plan.removed);
  if (!plan.moved) return incidence.collapse_boundary(mesh, plan.operation, removed, nullptr);
  const LocalIncidence::Star survivor = incidence.vertex_star(plan.survivor);
  return incidence.collapse_boundary(mesh, plan.operation, removed, &survivor);
}

const char* status_name(TetEditStatus status) {
  switch (status) {
    case TetEditStatus::kApplied: return "applied";
    case TetEditStatus::kRejectedTopology: return "topology";
    case TetEditStatus::kRejectedProtected: return "protected";
    case TetEditStatus::kInvalidGeometry: return "geometry";
    default: return "other";
  }
}

// The constrained segments' bank rows by sorted vertex IDs.
TetSegmentIndex segment_rows(const TetMesh& mesh) {
  TetMeshSnapshot snapshot;
  PHX_CHECK(mesh.snapshot(snapshot) == TetEditStatus::kApplied);
  TetSegmentIndex rows;
  for (std::size_t row = 0; row < snapshot.constrained_segments.size(); ++row) {
    const auto& ends = snapshot.constrained_segments[row].vertices;
    rows[sorted_edge(ends[0], ends[1])] = row;
  }
  return rows;
}

// Verdict counts by (swap, status).
using Tally = std::map<std::pair<bool, TetEditStatus>, int>;
// Applied collapses whose survivor moved.
int moved_collapses = 0;

// Random edits on two copies of one box, one edited locally and one by
// batches, compared after every edit.
void differential(const char* name, std::array<int, 3> n, bool two_regions, BoundaryPolicy policy,
                  double protected_share, int edits, std::uint32_t seed, Tally& tally) {
  const Box box = make_box(n, two_regions);
  std::mt19937 random(seed);
  std::vector<std::int8_t> protected_vertices(box.points.size() / 3, 0);
  for (std::int8_t& flag : protected_vertices) {
    flag = std::uniform_real_distribution<double>(0.0, 1.0)(random) < protected_share ? 1 : 0;
  }
  TetMesh local(policy);
  TetMesh batch(policy);
  if (!build(local, box, protected_vertices) || !build(batch, box, protected_vertices)) return;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(local));
  // The caller's segment index (bank rows), which the collapses keep current.
  TetSegmentIndex segment_index = segment_rows(local);
  incidence.bind_segment_index(segment_index);
  Banks banks(local);
  Generator generator(box, seed);
  int compared = 0;
  int outside = 0;
  for (int edit = 0; edit < edits; ++edit) {
    const bool swapping = generator.chance(0.5);
    const auto perturbation =
        generator.chance(0.7) ? Perturbation::kNone
                              : static_cast<Perturbation>(1 + generator.pick(8));
    const std::optional<Planned> plan =
        swapping ? generator.swap(local, incidence, banks, perturbation)
                 : generator.collapse(local, incidence, banks, perturbation, true, true);
    if (!plan) continue;
    const TetEditStatus local_status = apply_local(local, incidence, *plan);
    if (plan->out_of_domain) {
      ++outside;
      PHX_CHECK(local_status == TetEditStatus::kRejectedTopology);
      PHX_CHECK(same_state(local, batch));
      continue;
    }
    const TetEditStatus batch_status = apply_batch(batch, plan->operation);
    ++compared;
    ++tally[{swapping, batch_status}];
    if (local_status != batch_status) {
      std::fprintf(stderr, "%s edit %d (%s, perturbation %d): local %s, batch %s\n", name, edit,
                   swapping ? "swap" : plan->moved ? "moved collapse" : "collapse",
                   static_cast<int>(perturbation), status_name(local_status),
                   status_name(batch_status));
      for (std::uint64_t id : plan->operation.removed_facet_ids) {
        for (const auto& [key, facet] : banks.facets) {
          if (facet.id != id) continue;
          std::fprintf(stderr, "  removed facet %llu: source %d orientation %d vertices",
                       static_cast<unsigned long long>(id), facet.source, facet.orientation);
          for (std::uint64_t vertex : facet.vertices) {
            const double* p = local.point(local.vertex_slot(vertex));
            std::fprintf(stderr, " (%g %g %g)", p[0], p[1], p[2]);
          }
          std::fprintf(stderr, "\n");
        }
      }
      for (const TetEditFacet& facet : plan->operation.replacement_facets) {
        std::fprintf(stderr, "  new facet: source %d orientation %d vertices", facet.source,
                     facet.orientation);
        for (std::uint64_t vertex : facet.vertices) {
          const double* p = local.point(local.vertex_slot(vertex));
          std::fprintf(stderr, " (%g %g %g)", p[0], p[1], p[2]);
        }
        std::fprintf(stderr, "\n");
      }
      for (const TetEditCell& cell : plan->operation.replacement_cells) {
        std::fprintf(stderr, "  new cell region %d sources %d %d %d %d\n", cell.region,
                     cell.facet_sources[0], cell.facet_sources[1], cell.facet_sources[2],
                     cell.facet_sources[3]);
      }
      PHX_CHECK(local_status == batch_status);
      return;
    }
    if (local_status == TetEditStatus::kApplied) {
      banks.apply(plan->operation);
      moved_collapses += plan->moved ? 1 : 0;
      PHX_CHECK(segment_index == segment_rows(local));
    }
    if (!same_state(local, batch)) {
      std::fprintf(stderr, "%s edit %d: states differ after %s\n", name, edit,
                   status_name(local_status));
      PHX_CHECK(false);
      return;
    }
    if (local_status == TetEditStatus::kApplied &&
        (!consistent_complex(local) || !incidence_current(local, incidence))) {
      std::fprintf(stderr, "%s edit %d: inconsistent local complex or index\n", name, edit);
      PHX_CHECK(false);
      return;
    }
  }
  std::printf("%s: %d edits compared, %d outside the contract refused\n", name, compared, outside);
  PHX_CHECK(compared > edits / 4);
}

void boundary_edits_match_apply_edit_batch() {
  Tally tally;
  differential("box", {4, 4, 4}, false, BoundaryPolicy::kConforming, 0.0, 1500, 11, tally);
  differential("protected box", {4, 4, 2}, false, BoundaryPolicy::kConforming, 0.15, 600, 12,
               tally);
  differential("two-region box", {4, 4, 4}, true, BoundaryPolicy::kConforming, 0.0, 1200, 13,
               tally);
  differential("fixed box", {2, 2, 2}, false, BoundaryPolicy::kFixed, 0.0, 200, 14, tally);
  for (const bool swapping : {false, true}) {
    for (const TetEditStatus status :
         {TetEditStatus::kApplied, TetEditStatus::kRejectedTopology,
          TetEditStatus::kRejectedProtected, TetEditStatus::kInvalidGeometry}) {
      std::printf("  %s %s: %d\n", swapping ? "swap" : "collapse", status_name(status),
                  tally[{swapping, status}]);
      PHX_CHECK((tally[{swapping, status}] > 0));
    }
  }
  std::printf("  moved-survivor collapses applied: %d\n", moved_collapses);
  PHX_CHECK(moved_collapses > 0);
}

// Periodic and implicit-bound meshes are outside the contract.
void unsupported_meshes_are_refused() {
  const Box box = make_box({2, 2, 2}, false);
  TetMesh mesh(BoundaryPolicy::kConforming);
  if (!build(mesh, box, {})) return;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(mesh));
  Banks banks(mesh);
  Generator generator(box, 5);
  std::optional<Planned> collapse;
  std::optional<Planned> swap;
  for (int attempt = 0; attempt < 400 && (!collapse || !swap); ++attempt) {
    if (!collapse) collapse = generator.collapse(mesh, incidence, banks, Perturbation::kNone, false, false);
    if (!swap) swap = generator.swap(mesh, incidence, banks, Perturbation::kNone);
  }
  PHX_CHECK(collapse && swap);
  if (!collapse || !swap) return;
  TetMeshSnapshot before;
  PHX_CHECK(mesh.snapshot(before) == TetEditStatus::kApplied);
  TetPeriodicIdentity identity;
  identity.bound = true;
  identity.ids = {"lattice", "frame", "source", "revision", "map", "orbits"};
  PHX_CHECK(mesh.bind_periodic_identity(identity));
  PHX_CHECK(apply_local(mesh, incidence, *collapse) == TetEditStatus::kRejectedTopology);
  PHX_CHECK(apply_local(mesh, incidence, *swap) == TetEditStatus::kRejectedTopology);
  PHX_CHECK(mesh.bind_periodic_identity(TetPeriodicIdentity{}));
  TetMeshSnapshot after;
  PHX_CHECK(mesh.snapshot(after) == TetEditStatus::kApplied);
  PHX_CHECK(after.generation == before.generation && after.cells.size() == before.cells.size());
  // A dimension-three vertex is the interior collapse's (replace_cavity).
  TetEdit interior = collapse->operation;
  std::int32_t inside = -1;
  for (std::size_t slot = 0; slot < mesh.vertex_slot_count() && inside < 0; ++slot) {
    if (mesh.vertex_dimension(static_cast<std::int32_t>(slot)) == 3) {
      inside = static_cast<std::int32_t>(slot);
    }
  }
  PHX_CHECK(inside >= 0);
  interior.removed_vertex_ids = {mesh.vertex_id(inside)};
  interior.collapse_pairs = {{mesh.vertex_id(inside), interior.collapse_pairs.front().survivor_vertex_id}};
  PHX_CHECK(incidence.collapse_boundary(mesh, interior, incidence.vertex_star(inside), nullptr) ==
            TetEditStatus::kRejectedTopology);
  // A stale star proves no cavity: any applied edit changes the topology.
  const LocalIncidence::Star stale = incidence.vertex_star(collapse->removed);
  TetEditStatus swapped = apply_local(mesh, incidence, *swap);
  for (int attempt = 0; attempt < 400 && swapped != TetEditStatus::kApplied; ++attempt) {
    if (const auto next = generator.swap(mesh, incidence, banks, Perturbation::kNone)) {
      swapped = apply_local(mesh, incidence, *next);
    }
  }
  PHX_CHECK(swapped == TetEditStatus::kApplied);
  PHX_CHECK(incidence.collapse_boundary(mesh, collapse->operation, stale, nullptr) ==
            TetEditStatus::kRejectedTopology);

  // An implicit-bound mesh (one tetrahedron whose bottom face is the
  // implicit group z = 0).
  const std::vector<double> points{0, 0, 0, 2, 0, 0, 0, 2, 0, .5, .5, 1};
  std::array<std::int32_t, 4> cell{0, 1, 2, 3};
  if (orient3d(points.data(), points.data() + 3, points.data() + 6, points.data() + 9) < 0) {
    std::swap(cell[0], cell[1]);
  }
  std::vector<std::int32_t> faces;
  std::vector<std::int32_t> face_sources;
  std::vector<std::int32_t> source_faces;
  std::vector<std::int32_t> source_groups;
  std::vector<std::int64_t> source_ids;
  for (int side = 0; side < 4; ++side) {
    std::array<std::int32_t, 3> face{};
    oriented_facet(cell.data(), side, face.data());
    faces.insert(faces.end(), face.begin(), face.end());
    face_sources.push_back(side);
    source_faces.insert(source_faces.end(), face.begin(), face.end());
    source_groups.push_back(side);
    source_ids.push_back(side);
  }
  const std::vector<double> tolerances(4, 0.0);
  const SourceComplexView source{points, source_faces, source_groups, source_ids, tolerances,
                                 {}, {}, {}, {}, std::span<const SourceWitness>{}};
  auto implicit = std::make_shared<TetImplicitSource>();
  const std::array<int64_t, 8> value_code{2, 4, 1, 2, -1, 3, -1, 0};
  const std::array<double, 5> value_workspace{0, 0, 0, 0, 0};
  const std::array<int64_t, 1> value_outputs{4};
  const std::array<double, 6> gradient_workspace{0, 0, 0, 0, 0, 1};
  const std::array<int64_t, 3> gradient_outputs{3, 4, 5};
  PHX_CHECK(implicit->value.bind(value_code.data(), 1, nullptr, 0, nullptr, 0,
                                 value_workspace.data(), value_workspace.data(), 5,
                                 value_outputs.data(), 1, nullptr, 0) == PHX_MC_OK);
  PHX_CHECK(implicit->gradient.bind(nullptr, 0, nullptr, 0, nullptr, 0, gradient_workspace.data(),
                                    gradient_workspace.data(), 6, gradient_outputs.data(), 3,
                                    nullptr, 0) == PHX_MC_OK);
  std::int32_t bottom = -1;
  for (int side = 0; side < 4; ++side) {
    bool flat = true;
    for (int corner = 0; corner < 3; ++corner) flat = flat && points[3 * faces[3 * side + corner] + 2] == 0.0;
    if (flat) bottom = side;
  }
  implicit->groups = {bottom};
  implicit->tolerance = .02;
  implicit->source_id = "implicit";
  implicit->source_revision = "revision";
  implicit->binding_id = "binding";
  implicit->frame_id = "world";
  TetMesh bound(BoundaryPolicy::kConforming);
  PHX_CHECK(bound.prepare_implicit_source(implicit) == PHX_MC_OK);
  PHX_CHECK(bound.bind_source_identity("implicit", "revision", "binding", "world", true));
  const std::array<std::int32_t, 1> regions{0};
  PHX_CHECK(bound.build(points, cell, regions, faces, face_sources, {}, {}, {}, {}, &source) ==
            PHX_MC_OK);
  LocalIncidence bound_incidence;
  PHX_CHECK(bound_incidence.initialize(bound));
  TetEdit operation;
  operation.kind = TetEditKind::kCollapse;
  operation.collapse_pairs.push_back({0, 1});
  operation.removed_vertex_ids.push_back(0);
  operation.removed_cell_ids.push_back(0);
  PHX_CHECK(bound_incidence.collapse_boundary(bound, operation, bound_incidence.vertex_star(0),
                                              nullptr) == TetEditStatus::kRejectedTopology);
  TetEdit ring;
  ring.removed_cell_ids.push_back(0);
  ring.removed_facet_ids = {0, 1};
  ring.replacement_facets.resize(2);
  PHX_CHECK(bound_incidence.swap_boundary_edge(bound, ring) == TetEditStatus::kRejectedTopology);
  PHX_CHECK(bound.generation() == 0);
}

// Time per local edit on boxes of about 1e4 and 1e5 cells: in-place
// collapses and swaps, each with its own valid operation.
double seconds_per_edit(std::array<int, 3> n, std::uint32_t seed) {
  const Box box = make_box(n, false);
  TetMesh mesh(BoundaryPolicy::kConforming);
  if (!build(mesh, box, {})) return 0.0;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(mesh));
  Banks banks(mesh);
  Generator generator(box, seed);
  using Clock = std::chrono::steady_clock;
  Clock::duration spent{};
  int applied = 0;
  int timed = 0;
  bool warmed = false;
  for (int attempt = 0; attempt < 20000 && timed < 400; ++attempt) {
    const bool swapping = (attempt % 2) == 0;
    const std::optional<Planned> plan =
        swapping ? generator.swap(mesh, incidence, banks, Perturbation::kNone)
                 : generator.collapse(mesh, incidence, banks, Perturbation::kNone, false, false);
    if (!plan) continue;
    // The caller's star lookup is part of the edit.
    const Clock::time_point start = Clock::now();
    const TetEditStatus status = apply_local(mesh, incidence, *plan);
    const Clock::duration elapsed = Clock::now() - start;
    if (status == TetEditStatus::kApplied) {
      banks.apply(plan->operation);
      ++applied;
    }
    // The first edit builds the bank indexes.
    if (!warmed) {
      warmed = true;
      continue;
    }
    spent += elapsed;
    ++timed;
  }
  PHX_CHECK(applied > 100);
  return timed == 0 ? 0.0 : std::chrono::duration<double>(spent).count() / timed;
}

void boundary_edits_are_star_local() {
  const double small = seconds_per_edit({16, 16, 8}, 21);   // 12288 cells
  const double large = seconds_per_edit({32, 32, 16}, 21);  // 98304 cells
  std::printf("boundary edit: %.2f us per edit at 1.2e4 cells, %.2f us at 9.8e4 cells\n",
              1e6 * small, 1e6 * large);
  PHX_CHECK(small > 0.0 && large < 3.0 * small);
}

}  // namespace

int main() {
  boundary_edits_match_apply_edit_batch();
  unsupported_meshes_are_refused();
  boundary_edits_are_star_local();
  return phx::mc::test::finish("test_local_boundary_edits");
}
