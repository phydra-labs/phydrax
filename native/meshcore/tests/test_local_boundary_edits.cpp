//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// LocalIncidence::collapse_boundary, swap_boundary_edge, split_boundary_edge
// and relocate (boundary vertices) against TetMesh::apply_edit_batch and
// apply_relocation_edit: random single-operation boundary collapses (in
// place and with a moved survivor), coplanar boundary edge swaps, boundary
// edge splits and facet or segment vertex relocations, valid and perturbed,
// on small PLC boxes (six side facet groups of two source triangles, twelve
// source segments, an interior interface, protected vertices, the fixed
// policy) and Gmsh-like cubes (arbitrary binary64 coordinates, declared
// side planes) must give the batch's verdict and, when applied, its state.
// Their cost is star-local: the time per edit does not grow with the mesh.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

#include "cavity.hpp"
#include "check.hpp"
#include "dyadic.hpp"
#include "local_incidence.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "plc3d.hpp"
#include "predicates.hpp"
#include "source_construction.hpp"
#include "tet_mesh.hpp"

using namespace phx::mc;
using detail::LocalIncidence;
using detail::SourceStencil;

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
  // Declared exact planes per source face row (empty: none).
  std::vector<double> face_planes;
  std::vector<double> face_plane_tolerances;
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

// The point of source segment `row` at the open-segment parameter `t`,
// rounded into `carrier`, and its witness; both are left unchanged when
// there is none.
bool segment_point(const Box& box, std::int32_t row, double t, double* carrier,
                   SourceWitness& witness) {
  if (!(t > 0.0 && t < 1.0)) return false;
  const double* corners[3] = {box.source_points.data() + 3 * box.source_segments[2 * row],
                              box.source_points.data() + 3 * box.source_segments[2 * row + 1],
                              nullptr};
  const double parameters[2] = {t, 0.0};
  double position[3];
  double deviation = 0.0;
  if (!source_carrier(corners, SourceStratum::kSegment, parameters, position, deviation)) {
    return false;
  }
  std::copy_n(position, 3, carrier);
  witness = SourceWitness{SourceStratum::kSegment, row, {t, 0.0}, deviation};
  return true;
}

// The parameter of vertex `x` on source segment `row`: its witness's, or 0
// or 1 at the segment's ends.
bool segment_parameter(const Box& box, const TetMesh& mesh, std::int32_t x, std::int32_t row,
                       double& t) {
  const SourceWitness& witness = mesh.witness(x);
  if (witness.stratum == SourceStratum::kSegment && witness.entity == row) {
    t = witness.parameters[0];
    return true;
  }
  for (const int end : {0, 1}) {
    const double* corner = box.source_points.data() + 3 * box.source_segments[2 * row + end];
    if (std::equal(corner, corner + 3, mesh.point(x))) {
      t = end;
      return true;
    }
  }
  return false;
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
                                 box.witnesses,       box.face_planes,      box.face_plane_tolerances};
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
  std::mt19937& random() { return random_; }
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
  // contract's domain unless `out_of_domain`; with `fixed` (r, s), of r into
  // s, without a move, whatever its domain.
  std::optional<Planned> collapse(const TetMesh& mesh, const LocalIncidence& incidence,
                                  const Banks& banks, Perturbation perturbation,
                                  bool allow_move, bool allow_out_of_domain,
                                  std::optional<std::pair<std::int32_t, std::int32_t>> fixed =
                                      std::nullopt) {
    Planned plan;
    std::int32_t r = fixed ? fixed->first : -1;
    for (int attempt = 0; attempt < 64 && r < 0; ++attempt) {
      const auto slot = static_cast<std::int32_t>(pick(mesh.vertex_slot_count()));
      const std::int8_t dimension = mesh.vertex_dimension(slot);
      if (mesh.live_vertex(slot) && (dimension == 1 || dimension == 2)) r = slot;
    }
    if (r < 0) return std::nullopt;
    const std::vector<std::int32_t> neighbors = link(mesh, incidence, r);
    const std::int32_t s = fixed ? fixed->second : neighbors[pick(neighbors.size())];
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
    if (!in_domain && !fixed && !(allow_out_of_domain && chance(0.15))) return std::nullopt;
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
  if (!plan.moved) return incidence.collapse_boundary(mesh, plan.operation, removed, nullptr).value();
  const LocalIncidence::Star survivor = incidence.vertex_star(plan.survivor);
  return incidence.collapse_boundary(mesh, plan.operation, removed, &survivor).value();
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

// Applied boundary insertions by kind.
int inserted_facet_points = 0;
int inserted_segment_points = 0;

// Splits a random boundary edge at its midpoint on its facet group (or its
// segment) through insert_boundary_point on `local` and the equivalent
// refinement batch on `batch`: 0 when the local insertion is refused (the
// state unchanged), 1 for a facet point, 2 for a segment point, -1 when the
// batch does not apply it.
int insert_split(const Box& box, Generator& generator, TetMesh& local, TetMesh& batch,
                 LocalIncidence& incidence, Banks& banks) {
  TetMeshSnapshot before;
  PHX_CHECK(local.snapshot(before) == TetEditStatus::kApplied);
  const TetSnapshotFacet& chosen =
      before.constrained_facets[generator.pick(before.constrained_facets.size())];
  const std::size_t corner = generator.pick(3);
  const std::uint64_t a_id = chosen.vertices[corner];
  const std::uint64_t b_id = chosen.vertices[(corner + 1) % 3];
  const std::int32_t a = local.vertex_slot(a_id);
  const std::int32_t b = local.vertex_slot(b_id);
  std::vector<std::size_t> facet_rows;
  for (std::size_t row = 0; row < before.constrained_facets.size(); ++row) {
    const auto& vertices = before.constrained_facets[row].vertices;
    if (std::count(vertices.begin(), vertices.end(), a_id) == 1 &&
        std::count(vertices.begin(), vertices.end(), b_id) == 1) {
      facet_rows.push_back(row);
    }
  }
  std::vector<std::size_t> segment_rows_split;
  for (std::size_t row = 0; row < before.constrained_segments.size(); ++row) {
    if (sorted_edge(before.constrained_segments[row].vertices[0],
                    before.constrained_segments[row].vertices[1]) == sorted_edge(a_id, b_id)) {
      segment_rows_split.push_back(row);
    }
  }
  Point midpoint{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    midpoint[axis] = 0.5 * (local.point(a)[axis] + local.point(b)[axis]);
  }
  std::optional<SourceWitness> witness;
  if (!segment_rows_split.empty()) {
    witness = segment_witness(box, before.constrained_segments[segment_rows_split[0]].source,
                              midpoint);
  } else if (on_piece(box, chosen.source, midpoint)) {
    witness = facet_witness(box, chosen.source, midpoint);
  }
  if (!witness) return 0;
  // The cavity: the cells around ab and the hull ghosts across their faces
  // through ab.
  const std::vector<std::int32_t> star(incidence.edge_star(a, b).begin(),
                                       incidence.edge_star(a, b).end());
  std::vector<std::int32_t> cavity = star;
  for (std::int32_t cell : star) {
    const Tetrahedron& tet = local.tet(cell);
    for (int face = 0; face < 4; ++face) {
      if (tet.v[face] != a && tet.v[face] != b && is_ghost(local.tet(tet.n[face]))) {
        cavity.push_back(tet.n[face]);
      }
    }
  }
  TetEditVertex vertex;
  vertex.id = generator.fresh_id();
  vertex.carrier = midpoint;
  vertex.witness = *witness;
  std::vector<std::uint64_t> created;
  std::vector<std::uint64_t> removed_ids;
  for (std::int32_t cell : star) removed_ids.push_back(local.cell_id(cell));
  const TetEditStatus status = incidence.insert_boundary_point(local, vertex, cavity, facet_rows,
                                                               segment_rows_split, created);
  if (status != TetEditStatus::kApplied) return same_state(local, batch) ? 0 : -1;

  // The same edit as a refinement batch: the created cells, facets and
  // segments with their identities.
  TetMeshSnapshot after;
  PHX_CHECK(local.snapshot(after) == TetEditStatus::kApplied);
  TetEdit operation;
  // insert_boundary_point records its cells as refinement.
  operation.kind = TetEditKind::kRefineCell;
  operation.removed_cell_ids = removed_ids;
  for (const TetSnapshotCell& cell : after.cells) {
    if (std::find(created.begin(), created.end(), cell.id) == created.end()) continue;
    TetEditCell replacement;
    replacement.id = cell.id;
    replacement.vertices = cell.vertices;
    replacement.region = cell.region;
    replacement.ancestor_cell_ids = cell.ancestor_cell_ids;
    const std::int32_t slot = local.cell_slot(cell.id);
    for (int face = 0; face < 4; ++face) {
      // Snapshot corners are the cell's slot order.
      replacement.facet_sources[static_cast<std::size_t>(face)] =
          local.complex().constraint(slot, face);
    }
    operation.replacement_cells.push_back(replacement);
  }
  for (std::size_t row : facet_rows) {
    operation.removed_facet_ids.push_back(before.constrained_facets[row].id);
  }
  for (std::size_t row : segment_rows_split) {
    const TetSnapshotSegment& old = before.constrained_segments[row];
    operation.removed_segment_ids.push_back(old.id);
    const TetSnapshotSegment* retired = nullptr;
    PHX_CHECK(incidence.boundary_segment(old.vertices, retired) == TetEditStatus::kApplied);
    PHX_CHECK(retired == nullptr);
  }
  std::set<std::uint64_t> old_facets;
  for (const auto& facet : before.constrained_facets) old_facets.insert(facet.id);
  for (const auto& facet : after.constrained_facets) {
    if (old_facets.contains(facet.id)) continue;
    operation.replacement_facets.push_back(
        TetEditFacet{facet.id, facet.vertices, facet.source, facet.orientation, false});
  }
  std::set<std::uint64_t> old_segments;
  for (const auto& segment : before.constrained_segments) old_segments.insert(segment.id);
  for (const auto& segment : after.constrained_segments) {
    // The row may have been overwritten by the first half or appended.
    const TetSnapshotSegment* record = nullptr;
    PHX_CHECK(incidence.boundary_segment({segment.vertices[1], segment.vertices[0]}, record) ==
              TetEditStatus::kApplied);
    PHX_CHECK(record != nullptr && record->id == segment.id &&
              record->vertices == segment.vertices && record->source == segment.source);
    if (old_segments.contains(segment.id)) continue;
    operation.replacement_segments.push_back(
        TetEditSegment{segment.id, segment.vertices, segment.source, false});
  }
  operation.changed_vertices.push_back(vertex);
  const TetEditStatus batch_status = apply_batch(batch, operation);
  if (batch_status != TetEditStatus::kApplied) {
    std::fprintf(stderr, "  boundary insertion: batch %s\n", status_name(batch_status));
    return -1;
  }
  banks.apply(operation);
  return segment_rows_split.empty() ? 1 : 2;
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
    if (generator.chance(0.15)) {
      // A boundary split, so later edits run on an index that went through
      // insert_boundary_point.
      const int split = insert_split(box, generator, local, batch, incidence, banks);
      if (split < 0) {
        std::fprintf(stderr, "%s edit %d: boundary insertion diverged\n", name, edit);
        PHX_CHECK(false);
        return;
      }
      if (split > 0) {
        (split == 1 ? inserted_facet_points : inserted_segment_points) += 1;
        segment_index = segment_rows(local);
        if (!same_state(local, batch) || !consistent_complex(local) ||
            !incidence_current(local, incidence)) {
          std::fprintf(stderr, "%s edit %d: states differ after a boundary insertion\n", name,
                       edit);
          PHX_CHECK(false);
          return;
        }
      }
      continue;
    }
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
  std::printf("  boundary insertions applied: %d facet points, %d segment points\n",
              inserted_facet_points, inserted_segment_points);
  PHX_CHECK(inserted_facet_points > 0 && inserted_segment_points > 0);
}

// A Gmsh-like cube: the box's topology on arbitrary binary64 coordinates
// (y and z from origin 0.3 by step 0.1; x dyadic, so the seam x = n0 / 2
// is the exact middle of the x edges; every coordinate off the sides
// jittered), with witness-free vertices but on the edges, and each side
// group declaring its exact plane (tolerance 1e-9). With `split_sides`
// four sides are two groups each, on one region without an interface.
Box make_cube(std::array<int, 3> n, bool split_sides, std::uint32_t seed) {
  Box box = make_box(n, split_sides);
  if (split_sides) {
    std::fill(box.regions.begin(), box.regions.end(), 0);
    std::vector<std::int32_t> faces;
    std::vector<std::int32_t> sources;
    std::vector<std::int8_t> orientations;
    for (std::size_t row = 0; row < box.face_sources.size(); ++row) {
      if (box.face_sources[row] == 6) continue;
      faces.insert(faces.end(), box.faces.begin() + static_cast<std::ptrdiff_t>(3 * row),
                   box.faces.begin() + static_cast<std::ptrdiff_t>(3 * row + 3));
      sources.push_back(box.face_sources[row]);
      orientations.push_back(box.orientations[row]);
    }
    box.faces.swap(faces);
    box.face_sources.swap(sources);
    box.orientations.swap(orientations);
  }
  std::mt19937 random(seed);
  std::uniform_real_distribution<double> jitter(-0.012, 0.012);
  const auto map = [](std::size_t axis, double x) {
    return axis == 0 ? 0.25 + 0.125 * x : 0.3 + 0.1 * x;
  };
  for (std::size_t vertex = 0; vertex < box.points.size() / 3; ++vertex) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      double& x = box.points[3 * vertex + axis];
      const bool fixed = x == 0.0 || x == n[axis] || (split_sides && axis == 0 && x == 0.5 * n[0]);
      x = map(axis, x) + (fixed ? 0.0 : jitter(random));
    }
  }
  for (std::size_t row = 0; row < box.source_points.size(); ++row) {
    box.source_points[row] = map(row % 3, box.source_points[row]);
  }
  for (auto& [group, piece] : box.pieces) {
    piece.value = map(static_cast<std::size_t>(piece.axis), piece.value);
    piece.low = map(0, piece.low);
    piece.high = map(0, piece.high);
  }
  // Edge vertices keep their dyadic segment parameters, realized by
  // rounding; the rest are their own exact points.
  for (std::size_t vertex = 0; vertex < box.witnesses.size(); ++vertex) {
    SourceWitness& witness = box.witnesses[vertex];
    if (witness.stratum != SourceStratum::kSegment) {
      witness = SourceWitness{};
      continue;
    }
    PHX_CHECK(segment_point(box, witness.entity, witness.parameters[0],
                            box.points.data() + 3 * vertex, witness));
  }
  for (std::size_t row = 0; row < box.face_groups.size(); ++row) {
    const int axis = box.pieces.at(box.face_groups[row]).axis;
    const double* origin = box.source_points.data() + 3 * box.source_faces[3 * row];
    std::array<double, 9> plane{origin[0], origin[1], origin[2], 0, 0, 0, 0, 0, 0};
    plane[3 + static_cast<std::size_t>((axis + 1) % 3)] = 1.0;
    plane[6 + static_cast<std::size_t>((axis + 2) % 3)] = 1.0;
    box.face_planes.insert(box.face_planes.end(), plane.begin(), plane.end());
    box.face_plane_tolerances.push_back(1e-9);
  }
  for (std::size_t cell = 0; cell < box.tetrahedra.size() / 4; ++cell) {
    const std::int32_t* v = box.tetrahedra.data() + 4 * cell;
    PHX_CHECK(orient3d(box.points.data() + 3 * v[0], box.points.data() + 3 * v[1],
                       box.points.data() + 3 * v[2], box.points.data() + 3 * v[3]) > 0);
  }
  return box;
}

// How a split point is placed on the boundary edge ab between the facets
// abc and abd: the rounded midpoint (with or without its exact stencil),
// that point one ulp off the plane, or a point of the plane inside or
// outside the quadrilateral acbd.
enum class SplitKind { kMidpoint, kOffPlane, kInside, kOutside };

struct SplitPlan {
  std::int32_t a = -1;
  std::int32_t b = -1;
  TetEditVertex vertex;
  SplitKind kind = SplitKind::kMidpoint;
  // abc and abd belong to two face groups.
  bool two_groups = false;
};

// A split of a random facet edge (with `segment_ends`, of one whose ends
// both lie on segments or corners but which is no segment).
std::optional<SplitPlan> plan_split(const TetMesh& mesh, const Banks& banks, Generator& generator,
                                    bool valid_only, bool segment_ends = false) {
  std::uint64_t a_id = 0;
  std::uint64_t b_id = 0;
  if (segment_ends) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
    for (const auto& [key, facet] : banks.facets) {
      for (std::size_t corner = 0; corner < 3; ++corner) {
        const std::uint64_t first = facet.vertices[corner];
        const std::uint64_t second = facet.vertices[(corner + 1) % 3];
        if (mesh.vertex_dimension(mesh.vertex_slot(first)) <= 1 &&
            mesh.vertex_dimension(mesh.vertex_slot(second)) <= 1 &&
            !banks.segments.contains(sorted_edge(first, second))) {
          edges.emplace_back(first, second);
        }
      }
    }
    if (edges.empty()) return std::nullopt;
    std::tie(a_id, b_id) = edges[generator.pick(edges.size())];
  } else {
    auto chosen = banks.facets.begin();
    std::advance(chosen, static_cast<std::ptrdiff_t>(generator.pick(banks.facets.size())));
    const TetSnapshotFacet& facet = chosen->second;
    const std::size_t corner = generator.pick(3);
    a_id = facet.vertices[corner];
    b_id = facet.vertices[(corner + 1) % 3];
  }
  if (banks.segments.contains(sorted_edge(a_id, b_id))) return std::nullopt;
  std::vector<std::int32_t> apexes;
  std::set<std::int32_t> groups;
  for (const auto& [key, other] : banks.facets) {
    if (std::find(key.begin(), key.end(), a_id) == key.end() ||
        std::find(key.begin(), key.end(), b_id) == key.end()) {
      continue;
    }
    groups.insert(other.source);
    for (std::uint64_t id : key) {
      if (id != a_id && id != b_id) apexes.push_back(mesh.vertex_slot(id));
    }
  }
  if (apexes.size() != 2) return std::nullopt;
  SplitPlan plan;
  plan.two_groups = groups.size() == 2;
  plan.a = mesh.vertex_slot(a_id);
  plan.b = mesh.vertex_slot(b_id);
  const double* pa = mesh.point(plan.a);
  const double* pb = mesh.point(plan.b);
  // The plane's axis: the coordinate shared by a, b and both apexes.
  int axis = -1;
  for (int candidate = 0; candidate < 3; ++candidate) {
    bool shared = pa[candidate] == pb[candidate];
    for (std::int32_t apex : apexes) shared = shared && mesh.point(apex)[candidate] == pa[candidate];
    if (shared) axis = candidate;
  }
  if (axis < 0) return std::nullopt;
  const double draw = std::uniform_real_distribution<double>(0.0, 1.0)(generator.random());
  plan.kind = valid_only || draw < 0.5 ? SplitKind::kMidpoint
              : draw < 0.65            ? SplitKind::kOffPlane
              : draw < 0.85            ? SplitKind::kInside
                                       : SplitKind::kOutside;
  TetEditVertex& vertex = plan.vertex;
  vertex.id = generator.fresh_id();
  for (std::size_t k = 0; k < 3; ++k) vertex.carrier[k] = 0.5 * (pa[k] + pb[k]);
  switch (plan.kind) {
    case SplitKind::kMidpoint:
      if (generator.chance(0.5)) vertex.ancestry = {{a_id, {}, 0.5}, {b_id, {}, 0.5}};
      break;
    case SplitKind::kOffPlane:
      vertex.carrier[static_cast<std::size_t>(axis)] =
          std::nextafter(vertex.carrier[static_cast<std::size_t>(axis)], 2.0);
      break;
    case SplitKind::kInside:
    case SplitKind::kOutside: {
      const double* apex = mesh.point(apexes[generator.pick(2)]);
      const double s = plan.kind == SplitKind::kInside
                           ? std::uniform_real_distribution<double>(0.05, 0.9)(generator.random())
                           : std::uniform_real_distribution<double>(1.1, 1.6)(generator.random());
      const std::array<double, 3> middle = vertex.carrier;
      for (std::size_t k = 0; k < 3; ++k) vertex.carrier[k] = middle[k] + s * (apex[k] - middle[k]);
      vertex.carrier[static_cast<std::size_t>(axis)] = pa[axis];
      break;
    }
  }
  return plan;
}

// The kSplit edit of split_boundary_edge's contract for `plan` on `mesh`,
// with fresh identities.
TetEdit contract_split(const TetMesh& mesh, const LocalIncidence& incidence, const Banks& banks,
                       const SplitPlan& plan, Generator& generator) {
  const std::uint64_t a_id = mesh.vertex_id(plan.a);
  const std::uint64_t b_id = mesh.vertex_id(plan.b);
  TetEdit operation;
  operation.kind = TetEditKind::kSplit;
  operation.split_edges.push_back({a_id, b_id});
  operation.changed_vertices.push_back(plan.vertex);
  const auto shell = incidence.edge_star(plan.a, plan.b);
  std::vector<std::int32_t> star(shell.begin(), shell.end());
  std::sort(star.begin(), star.end(), [&](std::int32_t p, std::int32_t q) {
    return mesh.cell_id(p) < mesh.cell_id(q);
  });
  for (std::int32_t cell : star) {
    const Tetrahedron& tet = mesh.tet(cell);
    operation.removed_cell_ids.push_back(mesh.cell_id(cell));
    const int slot_a = vertex_slot(tet, plan.a);
    const int slot_b = vertex_slot(tet, plan.b);
    for (const int replaced : {slot_b, slot_a}) {
      TetEditCell child;
      child.id = generator.fresh_id();
      child.region = mesh.complex().region(cell);
      child.ancestor_cell_ids = {mesh.cell_id(cell)};
      for (int corner = 0; corner < 4; ++corner) {
        const auto index = static_cast<std::size_t>(corner);
        child.vertices[index] = corner == replaced ? plan.vertex.id : mesh.vertex_id(tet.v[corner]);
        child.facet_sources[index] = corner == (replaced == slot_b ? slot_a : slot_b)
                                         ? kNoConstraint
                                         : mesh.complex().constraint(cell, corner);
      }
      operation.replacement_cells.push_back(child);
    }
  }
  std::vector<const TetSnapshotFacet*> parents;
  for (const auto& [key, facet] : banks.facets) {
    if (std::find(key.begin(), key.end(), a_id) == key.end() ||
        std::find(key.begin(), key.end(), b_id) == key.end()) {
      continue;
    }
    parents.push_back(&facet);
  }
  std::ranges::sort(parents, {}, [](const TetSnapshotFacet* parent) { return parent->id; });
  for (const TetSnapshotFacet* parent : parents) {
    const auto& facet = *parent;
    operation.removed_facet_ids.push_back(facet.id);
    for (const std::uint64_t replaced : {b_id, a_id}) {
      TetEditFacet half{generator.fresh_id(), facet.vertices, facet.source, facet.orientation, false};
      std::replace(half.vertices.begin(), half.vertices.end(), replaced, plan.vertex.id);
      operation.replacement_facets.push_back(half);
    }
  }
  return operation;
}

// A relocation of a facet or segment vertex toward (or past) a link vertex
// on its carrier, or toward an interior link vertex (off the carrier).
enum class MoveKind { kWithin, kPast, kOff };

struct MovePlan {
  LocalIncidence::Star star;
  TetEditVertex vertex;
  MoveKind kind = MoveKind::kWithin;
  std::int8_t dimension = 0;
};

std::optional<MovePlan> plan_move(const Box& box, bool witnessed, const TetMesh& mesh,
                                  const LocalIncidence& incidence, const Banks& banks,
                                  Generator& generator) {
  std::int32_t v = -1;
  for (int attempt = 0; attempt < 64 && v < 0; ++attempt) {
    const auto slot = static_cast<std::int32_t>(generator.pick(mesh.vertex_slot_count()));
    const std::int8_t dimension = mesh.vertex_dimension(slot);
    if (mesh.live_vertex(slot) && (dimension == 1 || dimension == 2)) v = slot;
  }
  if (v < 0) return std::nullopt;
  const std::uint64_t v_id = mesh.vertex_id(v);
  const std::int8_t dimension = mesh.vertex_dimension(v);
  std::set<std::int32_t> v_groups;
  for (const auto& [key, facet] : banks.facets) {
    if (std::find(key.begin(), key.end(), v_id) != key.end()) v_groups.insert(facet.source);
  }
  const double draw = std::uniform_real_distribution<double>(0.0, 1.0)(generator.random());
  const MoveKind kind = draw < 0.6 ? MoveKind::kWithin : draw < 0.8 ? MoveKind::kPast : MoveKind::kOff;
  // Link vertices on the carrier: along a segment through v, or on a facet
  // of each of v's groups; off it: interior vertices.
  std::vector<std::int32_t> targets;
  for (std::int32_t x : generator.link(mesh, incidence, v)) {
    const std::uint64_t x_id = mesh.vertex_id(x);
    bool on = true;
    if (dimension == 1) {
      on = banks.segments.contains(sorted_edge(v_id, x_id));
    } else {
      for (std::int32_t group : v_groups) {
        bool shares = false;
        for (const auto& [key, facet] : banks.facets) {
          shares = shares || (facet.source == group &&
                              std::find(key.begin(), key.end(), x_id) != key.end());
        }
        on = on && shares;
      }
    }
    if (kind == MoveKind::kOff ? mesh.vertex_dimension(x) == 3 : on) targets.push_back(x);
  }
  if (targets.empty()) return std::nullopt;
  const std::int32_t x = targets[generator.pick(targets.size())];
  MovePlan plan{incidence.vertex_star(v), {}, kind, dimension};
  TetEditVertex& vertex = plan.vertex;
  vertex.id = v_id;
  vertex.protection_radius = mesh.protection_radius(v);
  vertex.protected_vertex = mesh.protected_vertex(v);
  vertex.witness = mesh.witness(v);
  // Witnessed boxes move by dyadic steps, so the witness stays exact.
  const double t = kind == MoveKind::kPast ? (witnessed ? 2.0 : 2.5)
                   : witnessed          ? 0.5
                                        : std::uniform_real_distribution<double>(0.2, 0.7)(generator.random());
  const double* p = mesh.point(v);
  const double* q = mesh.point(x);
  for (std::size_t k = 0; k < 3; ++k) vertex.carrier[k] = p[k] + t * (q[k] - p[k]);
  const SourceWitness& witness = mesh.witness(v);
  if (witness.stratum == SourceStratum::kSegment) {
    // Along the segment by dyadic parameter steps toward (or past) x; off
    // it, or past its end, the old witness no longer matches the carrier.
    double end = 0.0;
    if (kind != MoveKind::kOff && segment_parameter(box, mesh, x, witness.entity, end)) {
      const double step = kind == MoveKind::kPast ? 2.0 : 0.5;
      segment_point(box, witness.entity, witness.parameters[0] + step * (end - witness.parameters[0]),
                    vertex.carrier.data(), vertex.witness);
    }
    return plan;
  }
  if (!witnessed) {
    // A cube vertex keeps its sides' coordinates exactly on the carrier.
    if (kind != MoveKind::kOff) {
      for (std::size_t k = 0; k < 3; ++k) {
        if (p[k] == q[k]) vertex.carrier[k] = p[k];
      }
    }
    return plan;
  }
  const Point target{vertex.carrier[0], vertex.carrier[1], vertex.carrier[2]};
  std::optional<SourceWitness> moved;
  if (witness.stratum == SourceStratum::kFacet) {
    const std::int32_t group = box.face_groups[static_cast<std::size_t>(witness.entity)];
    if (on_piece(box, group, target)) moved = facet_witness(box, group, target);
  }
  // Off its source the old witness no longer matches the carrier.
  if (moved) vertex.witness = *moved;
  return plan;
}

TetEdit relocation_edit(const TetMesh& mesh, const MovePlan& plan) {
  TetEdit operation;
  operation.kind = TetEditKind::kRelocate;
  for (std::int32_t cell : plan.star) operation.removed_cell_ids.push_back(mesh.cell_id(cell));
  operation.changed_vertices.push_back(plan.vertex);
  return operation;
}

// Verdicts by edit (split kind, or 10 + move kind for a dimension-one and
// 20 + move kind for a dimension-two relocation) and status.
using EditTally = std::map<std::pair<int, TetEditStatus>, int>;
// Applied midpoint splits whose point is off the exact edge (no per-facet
// exact coverage).
int inexact_splits = 0;
// Refused splits of an edge between facets of two groups.
int two_group_refusals = 0;
// Splits of an edge whose ends lie on segments or corners, by verdict.
int segment_end_splits_applied = 0;
int segment_end_splits_refused = 0;

// One random split through split_boundary_edge on `local` and its contract
// edit through apply_edit_batch on `batch`: false when they disagree.
bool compare_split(const char* name, int edit, TetMesh& local, TetMesh& batch,
                   LocalIncidence& incidence, Banks& banks, Generator& generator, EditTally& tally,
                   bool valid_only) {
  const bool segment_ends = generator.chance(0.25);
  const std::optional<SplitPlan> plan =
      plan_split(local, banks, generator, valid_only, segment_ends);
  if (!plan) return true;
  TetEdit operation = contract_split(local, incidence, banks, *plan, generator);
  const double* endpoints[3] = {local.point(plan->a), local.point(plan->b), nullptr};
  const bool exact = on_source_entity(endpoints, SourceStratum::kSegment, plan->vertex.carrier.data());
  std::vector<std::uint64_t> created;
  const TetEditStatus local_status =
      incidence.split_boundary_edge(local, plan->a, plan->b, plan->vertex, created);
  if (local_status == TetEditStatus::kApplied) {
    // The local edit's identities: its cells in contract order, its facets
    // by their vertices.
    PHX_CHECK(created.size() == operation.replacement_cells.size());
    if (created.size() != operation.replacement_cells.size()) return false;
    for (std::size_t row = 0; row < created.size(); ++row) {
      operation.replacement_cells[row].id = created[row];
    }
    TetMeshSnapshot after;
    PHX_CHECK(local.snapshot(after) == TetEditStatus::kApplied);
    for (const auto& [key, old] : banks.facets) {
      if (std::find(operation.removed_facet_ids.begin(), operation.removed_facet_ids.end(),
                    old.id) == operation.removed_facet_ids.end()) continue;
      const TetSnapshotFacet* retired = nullptr;
      PHX_CHECK(incidence.boundary_facet(key, retired) == TetEditStatus::kApplied);
      PHX_CHECK(retired == nullptr);
    }
    std::optional<std::uint64_t> previous_facet_id;
    for (TetEditFacet& facet : operation.replacement_facets) {
      const auto found = std::find_if(
          after.constrained_facets.begin(), after.constrained_facets.end(),
          [&](const TetSnapshotFacet& made) { return made.vertices == facet.vertices; });
      PHX_CHECK(found != after.constrained_facets.end());
      if (found == after.constrained_facets.end()) return false;
      const TetSnapshotFacet* record = nullptr;
      PHX_CHECK(incidence.boundary_facet({facet.vertices[2], facet.vertices[0], facet.vertices[1]},
                                          record) == TetEditStatus::kApplied);
      PHX_CHECK(record != nullptr && record->id == found->id &&
                record->vertices == found->vertices && record->source == found->source &&
                record->orientation == found->orientation);
      PHX_CHECK(!previous_facet_id || found->id > *previous_facet_id);
      previous_facet_id = found->id;
      facet.id = found->id;
    }
  }
  const TetEditStatus batch_status = apply_batch(batch, operation);
  ++tally[{static_cast<int>(plan->kind), batch_status}];
  if (local_status != batch_status) {
    std::fprintf(stderr, "%s edit %d (split kind %d): local %s, batch %s\n", name, edit,
                 static_cast<int>(plan->kind), status_name(local_status),
                 status_name(batch_status));
    return false;
  }
  // Two groups' faces split only at an exact point of their edge.
  if (plan->two_groups) {
    PHX_CHECK(local_status != TetEditStatus::kApplied || exact);
    two_group_refusals += local_status != TetEditStatus::kApplied ? 1 : 0;
  }
  if (segment_ends) {
    (local_status == TetEditStatus::kApplied ? segment_end_splits_applied
                                             : segment_end_splits_refused) += 1;
  }
  if (local_status == TetEditStatus::kApplied) {
    banks.apply(operation);
    inexact_splits += plan->kind == SplitKind::kMidpoint && !exact ? 1 : 0;
  }
  return true;
}

// One random boundary relocation through LocalIncidence::relocate on
// `local` and apply_relocation_edit (odd edits) or apply_edit_batch on
// `batch`: false when they disagree.
bool compare_move(const char* name, int edit, const Box& box, bool witnessed, TetMesh& local,
                  TetMesh& batch, LocalIncidence& incidence, const Banks& banks,
                  Generator& generator, EditTally& tally) {
  const std::optional<MovePlan> plan = plan_move(box, witnessed, local, incidence, banks, generator);
  if (!plan) return true;
  const TetEdit operation = relocation_edit(local, *plan);
  const TetEditStatus local_status =
      incidence.relocate(local, std::span<const TetEditVertex>(&plan->vertex, 1), plan->star);
  TetEditStatus batch_status = TetEditStatus::kApplied;
  if (edit % 2 == 1) {
    PHX_CHECK(batch.apply_relocation_edit(operation, batch_status));
  } else {
    batch_status = apply_batch(batch, operation);
  }
  ++tally[{10 * plan->dimension + static_cast<int>(plan->kind), batch_status}];
  if (local_status != batch_status) {
    std::fprintf(stderr, "%s edit %d (dimension %d move kind %d): local %s, batch %s\n", name,
                 edit, plan->dimension, static_cast<int>(plan->kind), status_name(local_status),
                 status_name(batch_status));
    return false;
  }
  return true;
}

// Random boundary splits and relocations on two copies of one mesh, one
// edited locally and one by the batch paths, compared after every edit.
void split_relocate_differential(const char* name, const Box& box, bool witnessed,
                                 BoundaryPolicy policy, double protected_share,
                                 double split_share, int edits, std::uint32_t seed,
                                 EditTally& tally) {
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
  Banks banks(local);
  Generator generator(box, seed);
  for (int edit = 0; edit < edits; ++edit) {
    const bool agreed =
        generator.chance(split_share)
            ? compare_split(name, edit, local, batch, incidence, banks, generator, tally, false)
            : compare_move(name, edit, box, witnessed, local, batch, incidence, banks, generator,
                           tally);
    if (!agreed || !same_state(local, batch)) {
      std::fprintf(stderr, "%s edit %d: %s\n", name, edit,
                   agreed ? "states differ" : "verdicts differ");
      PHX_CHECK(false);
      return;
    }
    if (local.generation() != batch.generation() || !consistent_complex(local) ||
        !incidence_current(local, incidence)) {
      std::fprintf(stderr, "%s edit %d: inconsistent local complex or index\n", name, edit);
      PHX_CHECK(false);
      return;
    }
  }
  std::printf("%s: %d edits compared\n", name, edits);
}

void splits_and_relocations_match_the_batch() {
  EditTally tally;
  split_relocate_differential("cube", make_cube({4, 4, 4}, false, 31), false,
                              BoundaryPolicy::kConforming, 0.0, 0.5, 900, 31, tally);
  split_relocate_differential("split-side cube", make_cube({4, 4, 2}, true, 32), false,
                              BoundaryPolicy::kConforming, 0.1, 0.6, 700, 32, tally);
  split_relocate_differential("fixed cube", make_cube({2, 2, 2}, false, 33), false,
                              BoundaryPolicy::kFixed, 0.0, 0.5, 150, 33, tally);
  split_relocate_differential("box", make_box({4, 4, 4}, false), true,
                              BoundaryPolicy::kConforming, 0.1, 0.0, 700, 34, tally);
  split_relocate_differential("two-region box", make_box({4, 4, 4}, true), true,
                              BoundaryPolicy::kConforming, 0.0, 0.0, 700, 35, tally);
  for (const auto& [key, count] : tally) {
    std::printf("  edit %d %s: %d\n", key.first, status_name(key.second), count);
  }
  const auto count = [&](int edit, TetEditStatus status) {
    const auto found = tally.find({edit, status});
    return found == tally.end() ? 0 : found->second;
  };
  constexpr auto applied = TetEditStatus::kApplied;
  constexpr auto geometry = TetEditStatus::kInvalidGeometry;
  constexpr auto topology = TetEditStatus::kRejectedTopology;
  // Declared-pair splits apply off the exact edge; off the plane (within
  // its tolerance) the pair proof refuses them, and inverted halves, points
  // outside acbd and pairs of two groups are refused.
  std::printf("  declared splits applied off the exact edge: %d\n", inexact_splits);
  PHX_CHECK(count(0, applied) > 0 && inexact_splits > 0);
  PHX_CHECK(count(1, topology) > 0);
  PHX_CHECK(count(2, applied) > 0 && count(2, geometry) > 0);
  PHX_CHECK(count(3, geometry) > 0);
  std::printf("  two-group splits refused: %d\n", two_group_refusals);
  PHX_CHECK(two_group_refusals > 0);
  std::printf("  splits between segment vertices: %d applied, %d refused\n",
              segment_end_splits_applied, segment_end_splits_refused);
  PHX_CHECK(segment_end_splits_applied > 0 && segment_end_splits_refused > 0);
  // Relocations within the carrier apply; past the link they invert a star
  // cell or leave the carrier, and toward the interior they leave it.
  for (const int dimension : {10, 20}) {
    PHX_CHECK(count(dimension, applied) > 0);
    PHX_CHECK(count(dimension + 1, geometry) > 0);
    PHX_CHECK(count(dimension + 2, geometry) > 0);
  }
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
// collapses and swaps, each with its own valid operation. The cell arrays
// are reserved before timing (a built mesh can be exactly at capacity, and
// the growth would land in one timed edit), and each kind's first edit is a
// warm-up that builds the bank indexes.
double seconds_per_edit(std::array<int, 3> n, std::uint32_t seed) {
  const Box box = make_box(n, false);
  TetMesh mesh(BoundaryPolicy::kConforming);
  if (!build(mesh, box, {})) return 0.0;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(mesh));
  Banks banks(mesh);
  Generator generator(box, seed);
  // Room for every cell the timed edits can add.
  mesh.reserve_cells(16384);
  using Clock = std::chrono::steady_clock;
  std::array<Clock::duration, 2> spent{};
  std::array<int, 2> timed{};
  std::array<bool, 2> warmed{};
  int applied = 0;
  for (int attempt = 0; attempt < 20000 && timed[0] + timed[1] < 400; ++attempt) {
    const std::size_t kind = attempt % 2;
    const bool swapping = kind == 0;
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
    if (!warmed[kind]) {
      warmed[kind] = true;
      continue;
    }
    spent[kind] += elapsed;
    ++timed[kind];
  }
  PHX_CHECK(applied > 100);
  // Enough samples per kind that one slow call cannot set the mean.
  PHX_CHECK(timed[0] >= 50 && timed[1] >= 50);
  return std::chrono::duration<double>(spent[0] + spent[1]).count() / (timed[0] + timed[1]);
}

// Time per local split and per relocation on Gmsh-like cubes of about 1e4
// and 1e5 cells: declared-pair midpoint splits and moves within the carrier.
// As above, the cell arrays are reserved and each kind has its own warm-up.
std::array<double, 3> seconds_per_split_and_move(std::array<int, 3> n, std::uint32_t seed) {
  const Box box = make_cube(n, false, seed);
  TetMesh mesh(BoundaryPolicy::kConforming);
  if (!build(mesh, box, {})) return {};
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(mesh));
  Banks banks(mesh);
  Generator generator(box, seed);
  mesh.reserve_cells(32768);
  using Clock = std::chrono::steady_clock;
  // Splits, relocations, and splits of edges between segment vertices.
  std::array<Clock::duration, 3> spent{};
  std::array<int, 3> timed{};
  int splits = 0;
  int moves = 0;
  std::array<bool, 3> warmed{};
  for (int attempt = 0; attempt < 30000 && timed[0] + timed[1] + timed[2] < 900; ++attempt) {
    Clock::duration elapsed{};
    if (attempt % 3 != 1) {
      const std::optional<SplitPlan> plan =
          plan_split(mesh, banks, generator, true, attempt % 3 == 2);
      if (!plan) continue;
      const TetEdit operation = contract_split(mesh, incidence, banks, *plan, generator);
      std::vector<std::uint64_t> created;
      const Clock::time_point start = Clock::now();
      const TetEditStatus status =
          incidence.split_boundary_edge(mesh, plan->a, plan->b, plan->vertex, created);
      elapsed = Clock::now() - start;
      if (status == TetEditStatus::kApplied) {
        // The banks need only the facets' vertices and sources.
        banks.apply(operation);
        ++splits;
      }
    } else {
      const std::optional<MovePlan> plan = plan_move(box, false, mesh, incidence, banks, generator);
      if (!plan || plan->kind != MoveKind::kWithin) continue;
      // The caller's star lookup is part of the edit.
      const Clock::time_point start = Clock::now();
      const LocalIncidence::Star star = incidence.vertex_star(mesh.vertex_slot(plan->vertex.id));
      const TetEditStatus status =
          incidence.relocate(mesh, std::span<const TetEditVertex>(&plan->vertex, 1), star);
      elapsed = Clock::now() - start;
      moves += status == TetEditStatus::kApplied ? 1 : 0;
    }
    // Each kind's first edit is a warm-up (bank indexes, first-use caches).
    const std::size_t kind = attempt % 3;
    if (!warmed[kind]) {
      warmed[kind] = true;
      continue;
    }
    spent[kind] += elapsed;
    ++timed[kind];
  }
  PHX_CHECK(splits > 50 && moves > 50);
  // Enough samples per kind that one slow call cannot set the mean.
  PHX_CHECK(timed[0] >= 50 && timed[1] >= 50 && timed[2] >= 10);
  std::array<double, 3> seconds{};
  for (std::size_t kind = 0; kind < 3; ++kind) {
    seconds[kind] = std::chrono::duration<double>(spent[kind]).count() / timed[kind];
  }
  return seconds;
}

// apply_edit_batch's time for one batch of `count` disjoint valid boundary
// swaps on a box (informational: the batch copies the state and rebuilds
// the candidate complex per edit).
double batch_seconds(std::array<int, 3> n, int count, std::uint32_t seed) {
  const Box box = make_box(n, false);
  TetMesh mesh(BoundaryPolicy::kConforming);
  // Each swap is checked alone on a scratch copy; disjoint swaps stay valid
  // in one batch.
  TetMesh scratch(BoundaryPolicy::kConforming);
  if (!build(mesh, box, {}) || !build(scratch, box, {})) return 0.0;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(mesh));
  Banks banks(mesh);
  Generator generator(box, seed);
  TetEditBatch batch{mesh.generation(), mesh.source_revision(), mesh.frame_id(),
                     mesh.identity_bound(), {}, mesh.source_id(),
                     mesh.source_map_id(), mesh.periodic_identity(), std::nullopt};
  std::set<std::int32_t> touched;
  for (int attempt = 0; attempt < 20000 && static_cast<int>(batch.operations.size()) < count;
       ++attempt) {
    const std::optional<Planned> plan = generator.swap(mesh, incidence, banks, Perturbation::kNone);
    if (!plan) continue;
    std::set<std::int32_t> vertices;
    for (std::uint64_t id : plan->operation.removed_cell_ids) {
      for (std::int32_t vertex : mesh.tet(mesh.cell_slot(id)).v) vertices.insert(vertex);
    }
    if (std::any_of(vertices.begin(), vertices.end(),
                    [&](std::int32_t vertex) { return touched.contains(vertex); }) ||
        apply_batch(scratch, plan->operation) != TetEditStatus::kApplied) {
      continue;
    }
    touched.insert(vertices.begin(), vertices.end());
    batch.operations.push_back(plan->operation);
  }
  std::vector<TetEditResult> results;
  const auto start = std::chrono::steady_clock::now();
  const TetEditStatus status = mesh.apply_edit_batch(batch, results);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  PHX_CHECK(status == TetEditStatus::kApplied && static_cast<int>(batch.operations.size()) == count);
  return std::chrono::duration<double>(elapsed).count();
}

void boundary_edits_are_star_local() {
  const double small = seconds_per_edit({16, 16, 8}, 21);   // 12288 cells
  const double large = seconds_per_edit({32, 32, 16}, 21);  // 98304 cells
  std::printf("boundary edit: %.2f us per edit at 1.2e4 cells, %.2f us at 9.8e4 cells\n",
              1e6 * small, 1e6 * large);
  PHX_CHECK(small > 0.0 && large < 3.0 * small);
  const std::array<double, 3> small_cd = seconds_per_split_and_move({16, 16, 8}, 22);
  const std::array<double, 3> large_cd = seconds_per_split_and_move({32, 32, 16}, 22);
  std::printf("boundary split: %.2f us per edit at 1.2e4 cells, %.2f us at 9.8e4 cells\n",
              1e6 * small_cd[0], 1e6 * large_cd[0]);
  std::printf("boundary relocation: %.2f us per edit at 1.2e4 cells, %.2f us at 9.8e4 cells\n",
              1e6 * small_cd[1], 1e6 * large_cd[1]);
  std::printf("split between segment vertices: %.2f us per edit at 1.2e4 cells, %.2f us at "
              "9.8e4 cells\n",
              1e6 * small_cd[2], 1e6 * large_cd[2]);
  for (std::size_t kind = 0; kind < 3; ++kind) {
    PHX_CHECK(small_cd[kind] > 0.0 && large_cd[kind] < 3.0 * small_cd[kind]);
  }
  for (const std::array<int, 3> n : {std::array<int, 3>{16, 16, 8}, std::array<int, 3>{32, 32, 16}}) {
    const double one = batch_seconds(n, 1, 23);
    const double many = batch_seconds(n, 32, 23);
    std::printf("apply_edit_batch at %d cells: %.2f ms for 1 swap, %.2f ms for 32 swaps\n",
                6 * n[0] * n[1] * n[2], 1e3 * one, 1e3 * many);
  }
}

// A boundary split's cavity (TetMesh::boundary_insertion_cavity) grows by
// insphere tests from the cells beside its disk faces and never across a
// constrained face, but it can wrap around the free edges of an interior
// constrained facet and hold the cells on both of its sides. The edit
// (insert_boundary_point) refuses a cavity that would remove a constrained
// face other than a disk face, so such a split was refused and its edge
// stayed long. Every hull edge of a small Delaunay mesh is split against
// every interior face constrained as a one-triangle fin: a found cavity
// never joins two of its cells through a constrained face off the disk.
void boundary_cavities_never_hold_a_non_disk_constrained_face() {
  std::vector<double> input;
  std::uint32_t state = 12345;
  const auto jitter = [&]() {
    state = state * 1664525U + 1013904223U;
    return static_cast<double>(static_cast<int>((state >> 16) % 5U) - 2);
  };
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        input.push_back(16.0 * i + jitter());
        input.push_back(16.0 * j + jitter());
        input.push_back(16.0 * k + jitter());
      }
    }
  }
  phx_mc_mesh* delaunay = nullptr;
  const std::int32_t triangulated =
      phx_mc_delaunay_3d(static_cast<std::int64_t>(input.size() / 3), input.data(), &delaunay, 1);
  PHX_CHECK(triangulated == PHX_MC_OK);
  if (triangulated != PHX_MC_OK) return;
  const std::vector<double> points = delaunay->points;
  const std::vector<std::int32_t> tetrahedra = delaunay->cells;
  phx_mc_mesh_free(delaunay);
  const auto face_of = [](const std::int32_t* cell, int skip) {
    std::array<std::int32_t, 3> face{};
    int at = 0;
    for (int corner = 0; corner < 4; ++corner) {
      if (corner != skip) face[static_cast<std::size_t>(at++)] = cell[corner];
    }
    std::sort(face.begin(), face.end());
    return face;
  };
  std::map<std::array<std::int32_t, 3>, int> uses;
  for (std::size_t cell = 0; cell < tetrahedra.size() / 4; ++cell) {
    for (int skip = 0; skip < 4; ++skip) ++uses[face_of(tetrahedra.data() + 4 * cell, skip)];
  }
  std::vector<std::array<std::int32_t, 3>> hull;
  std::vector<std::array<std::int32_t, 3>> interior;
  for (const auto& [face, count] : uses) (count == 1 ? hull : interior).push_back(face);
  std::set<std::array<std::int32_t, 2>> hull_edges;
  for (const auto& face : hull) {
    for (int corner = 0; corner < 3; ++corner) {
      const std::int32_t a = face[static_cast<std::size_t>(corner)];
      const std::int32_t b = face[static_cast<std::size_t>((corner + 1) % 3)];
      hull_edges.insert({std::min(a, b), std::max(a, b)});
    }
  }
  const std::vector<std::int32_t> regions(tetrahedra.size() / 4, 0);
  std::int64_t found = 0;
  for (const auto& fin : interior) {
    std::vector<std::int32_t> faces;
    std::vector<std::int32_t> sources;
    for (const auto& face : hull) {
      faces.insert(faces.end(), face.begin(), face.end());
      sources.push_back(static_cast<std::int32_t>(sources.size()));
    }
    faces.insert(faces.end(), fin.begin(), fin.end());
    sources.push_back(static_cast<std::int32_t>(sources.size()));
    TetMesh mesh(BoundaryPolicy::kConforming);
    if (mesh.build(points, tetrahedra, regions, faces, sources, {}, {}) != PHX_MC_OK) continue;
    const TetrahedralComplex& complex = mesh.complex();
    for (const auto& [first, second] : hull_edges) {
      const std::int32_t a = mesh.vertex_slot(static_cast<std::uint64_t>(first));
      const std::int32_t b = mesh.vertex_slot(static_cast<std::uint64_t>(second));
      std::vector<std::pair<std::int32_t, int>> disk;
      std::set<std::array<std::int32_t, 3>> disk_faces;
      for (std::size_t slot = 0; slot < complex.tets.size(); ++slot) {
        const auto cell = static_cast<std::int32_t>(slot);
        if (!complex.live(cell) || is_ghost(mesh.tet(cell))) continue;
        const Tetrahedron& tetrahedron = mesh.tet(cell);
        if (vertex_slot(tetrahedron, a) < 0 || vertex_slot(tetrahedron, b) < 0) continue;
        for (int face = 0; face < 4; ++face) {
          if (tetrahedron.v[face] == a || tetrahedron.v[face] == b ||
              !is_ghost(mesh.tet(tetrahedron.n[face]))) {
            continue;
          }
          disk.emplace_back(cell, face);
          disk_faces.insert(face_of(tetrahedron.v, face));
        }
      }
      TetEditVertex vertex;
      vertex.id = mesh.vertex_slot_count();
      for (int axis = 0; axis < 3; ++axis) {
        vertex.carrier[static_cast<std::size_t>(axis)] =
            0.5 * (points[3 * static_cast<std::size_t>(first) + static_cast<std::size_t>(axis)] +
                   points[3 * static_cast<std::size_t>(second) + static_cast<std::size_t>(axis)]);
      }
      std::vector<std::int32_t> cells;
      if (mesh.boundary_insertion_cavity(vertex, disk, {-1, -1}, {}, cells, nullptr) !=
          TetCavityStatus::kFound) {
        continue;
      }
      ++found;
      for (const std::int32_t cell : cells) {
        const Tetrahedron& tetrahedron = mesh.tet(cell);
        for (int face = 0; face < 4; ++face) {
          if (!std::binary_search(cells.begin(), cells.end(), tetrahedron.n[face]) ||
              complex.constraint(cell, face) == kNoConstraint) {
            continue;
          }
          PHX_CHECK(disk_faces.contains(face_of(tetrahedron.v, face)));
        }
      }
    }
  }
  PHX_CHECK(found > 0);
}

// The corners (u0, v0), (u1, v0), (u0, v1) of the in-plane bounding
// rectangle of `points` (plane axis `axis`, u and v the next two axes), when
// all three are among them.
std::optional<std::array<std::int32_t, 3>> rectangle_corners(const Box& box,
                                                             const std::set<std::int32_t>& points,
                                                             int axis) {
  const auto u = static_cast<std::size_t>((axis + 1) % 3);
  const auto v = static_cast<std::size_t>((axis + 2) % 3);
  const auto at = [&](std::int32_t row, std::size_t k) {
    return box.source_points[3 * static_cast<std::size_t>(row) + k];
  };
  double u0 = 1e300, u1 = -1e300, v0 = 1e300, v1 = -1e300;
  for (std::int32_t row : points) {
    u0 = std::min(u0, at(row, u));
    u1 = std::max(u1, at(row, u));
    v0 = std::min(v0, at(row, v));
    v1 = std::max(v1, at(row, v));
  }
  std::array<std::int32_t, 3> corners{-1, -1, -1};
  for (std::int32_t row : points) {
    if (at(row, u) == u0 && at(row, v) == v0) corners[0] = row;
    if (at(row, u) == u1 && at(row, v) == v0) corners[1] = row;
    if (at(row, u) == u0 && at(row, v) == v1) corners[2] = row;
  }
  if (corners[0] < 0 || corners[1] < 0 || corners[2] < 0 || u0 == u1 || v0 == v1) {
    return std::nullopt;
  }
  return corners;
}

// Whether `weights` on `corners` are an exact affine combination equal to p.
bool exact_stencil(const Box& box, const std::array<std::int32_t, 3>& corners,
                   const std::array<double, 3>& weights, const double* p) {
  Dyadic sum(0.0);
  std::array<Dyadic, 3> combination{Dyadic(0.0), Dyadic(0.0), Dyadic(0.0)};
  for (std::size_t k = 0; k < 3; ++k) {
    const Dyadic weight(weights[k]);
    sum = sum + weight;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      combination[axis] =
          combination[axis] +
          Dyadic(box.source_points[3 * static_cast<std::size_t>(corners[k]) + axis]) * weight;
    }
  }
  if (!(sum - Dyadic(1.0)).is_zero()) return false;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (!(combination[axis] - Dyadic(p[axis])).is_zero()) return false;
  }
  return true;
}

struct StencilTally {
  int applied = 0;
  int refused_without_stencil = 0;
  int refused_inexact = 0;
  int refused_outside = 0;
};

// Moves witness-free facet vertices s to dyadic points of their face
// group's rectangle (no source point), then collapses a witness-free
// neighbour r of the same group into s: refused without a stencil, refused
// with a stencil whose combination misses s (one weight quantum moved
// between two points) or whose exact support leaves r's carrier, and with
// the exact stencil equal to apply_edit_batch in verdict and state.
void stencil_collapse_case(const char* name, const Box& box, std::uint32_t seed,
                           StencilTally& tally) {
  TetMesh local(BoundaryPolicy::kConforming);
  TetMesh batch(BoundaryPolicy::kConforming);
  if (!build(local, box, {}) || !build(batch, box, {})) return;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(local));
  Banks banks(local);
  Generator generator(box, seed);
  const auto source_point = [&](const double* p) {
    for (std::size_t row = 0; row < box.source_points.size() / 3; ++row) {
      if (std::equal(p, p + 3, box.source_points.data() + 3 * row)) return true;
    }
    return false;
  };
  // The face group of a vertex on one group's facets and no segment.
  const auto facet_group = [&](std::uint64_t id) -> std::optional<std::int32_t> {
    std::optional<std::int32_t> group;
    for (const auto& [key, facet] : banks.facets) {
      if (std::find(key.begin(), key.end(), id) == key.end()) continue;
      if (group && *group != facet.source) return std::nullopt;
      group = facet.source;
    }
    for (const auto& [key, segment] : banks.segments) {
      if (key[0] == id || key[1] == id) return std::nullopt;
    }
    return group;
  };
  const auto slots = local.vertex_slot_count();
  for (std::size_t slot = 0; slot < slots; ++slot) {
    const auto s = static_cast<std::int32_t>(slot);
    if (!local.live_vertex(s) || local.vertex_dimension(s) != 2 ||
        local.witness(s).stratum != SourceStratum::kNone) {
      continue;
    }
    const std::optional<std::int32_t> group = facet_group(local.vertex_id(s));
    if (!group) continue;
    const Piece& piece = box.pieces.at(*group);
    const auto corners = rectangle_corners(box, box.face_group_points.at(*group), piece.axis);
    if (!corners) continue;
    const auto axis = static_cast<std::size_t>(piece.axis);
    const auto u = (axis + 1) % 3;
    const auto v = (axis + 2) % 3;
    const double* p0 = box.source_points.data() + 3 * static_cast<std::size_t>((*corners)[0]);
    const double* p1 = box.source_points.data() + 3 * static_cast<std::size_t>((*corners)[1]);
    const double* p2 = box.source_points.data() + 3 * static_cast<std::size_t>((*corners)[2]);
    const auto parameter = [](double x, double low, double high) {
      return std::clamp(std::round((x - low) / (high - low) * 64.0) / 64.0, 1.0 / 64.0,
                        63.0 / 64.0);
    };
    const double a = parameter(local.point(s)[u], p0[u], p1[u]);
    const double b = parameter(local.point(s)[v], p0[v], p2[v]);
    TetEditVertex moved;
    moved.id = local.vertex_id(s);
    moved.carrier[axis] = p0[axis];
    moved.carrier[u] = p0[u] + a * (p1[u] - p0[u]);
    moved.carrier[v] = p0[v] + b * (p2[v] - p0[v]);
    moved.protection_radius = local.protection_radius(s);
    moved.protected_vertex = local.protected_vertex(s);
    const std::array<double, 3> weights{1.0 - a - b, a, b};
    if (source_point(moved.carrier.data()) ||
        !exact_stencil(box, *corners, weights, moved.carrier.data())) {
      continue;
    }
    const LocalIncidence::Star star = incidence.vertex_star(s);
    TetEdit relocation;
    relocation.kind = TetEditKind::kRelocate;
    for (std::int32_t cell : star) relocation.removed_cell_ids.push_back(local.cell_id(cell));
    relocation.changed_vertices.push_back(moved);
    if (incidence.relocate(local, std::span<const TetEditVertex>(&moved, 1), star) !=
        TetEditStatus::kApplied) {
      continue;
    }
    TetEditStatus relocated = TetEditStatus::kApplied;
    PHX_CHECK(batch.apply_relocation_edit(relocation, relocated));
    PHX_CHECK(relocated == TetEditStatus::kApplied && same_state(local, batch));

    // r: a witness-free link vertex on the same group alone.
    std::int32_t r = -1;
    for (std::int32_t other : generator.link(local, incidence, s)) {
      if (other >= 0 && local.vertex_dimension(other) == 2 &&
          local.witness(other).stratum == SourceStratum::kNone &&
          facet_group(local.vertex_id(other)) == group) {
        r = other;
        break;
      }
    }
    if (r < 0) continue;
    const std::optional<Planned> plan = generator.collapse(
        local, incidence, banks, Perturbation::kNone, false, false, std::pair{r, s});
    if (!plan) continue;
    const LocalIncidence::Star removed = incidence.vertex_star(r);
    const std::array<std::int32_t, 3> points = *corners;

    // Refusals leave the mesh as it is.
    const TetEditStatus bare =
        incidence.collapse_boundary(local, plan->operation, removed, nullptr).value();
    PHX_CHECK(bare == TetEditStatus::kRejectedTopology);
    tally.refused_without_stencil += bare == TetEditStatus::kRejectedTopology ? 1 : 0;
    const std::array<double, 3> inexact{weights[0] - 0x1p-52, weights[1] + 0x1p-52, weights[2]};
    PHX_CHECK(!exact_stencil(box, points, inexact, moved.carrier.data()));
    const SourceStencil off{points, inexact};
    const TetEditStatus missed =
        incidence.collapse_boundary(local, plan->operation, removed, nullptr, &off).value();
    PHX_CHECK(missed == TetEditStatus::kRejectedTopology);
    tally.refused_inexact += missed == TetEditStatus::kRejectedTopology ? 1 : 0;
    // The same point through the corners of the whole plane, when one of
    // them lies outside r's carrier (a side split into two groups).
    std::set<std::int32_t> plane;
    for (std::size_t row = 0; row < box.source_points.size() / 3; ++row) {
      if (box.source_points[3 * row + axis] == p0[axis]) plane.insert(static_cast<std::int32_t>(row));
    }
    const auto wide = rectangle_corners(box, plane, piece.axis);
    if (wide && std::any_of(wide->begin(), wide->end(), [&](std::int32_t row) {
          return !box.face_group_points.at(*group).contains(row);
        })) {
      const double* q0 = box.source_points.data() + 3 * static_cast<std::size_t>((*wide)[0]);
      const double* q1 = box.source_points.data() + 3 * static_cast<std::size_t>((*wide)[1]);
      const double* q2 = box.source_points.data() + 3 * static_cast<std::size_t>((*wide)[2]);
      const double wa = (moved.carrier[u] - q0[u]) / (q1[u] - q0[u]);
      const double wb = (moved.carrier[v] - q0[v]) / (q2[v] - q0[v]);
      const std::array<double, 3> wide_weights{1.0 - wa - wb, wa, wb};
      // Only an exact combination isolates the support rule.
      if (wa != 0.0 && wb != 0.0 &&
          exact_stencil(box, *wide, wide_weights, moved.carrier.data())) {
        const SourceStencil outside{*wide, wide_weights};
        const TetEditStatus left =
            incidence.collapse_boundary(local, plan->operation, removed, nullptr, &outside).value();
        PHX_CHECK(left == TetEditStatus::kRejectedTopology);
        tally.refused_outside += left == TetEditStatus::kRejectedTopology ? 1 : 0;
      }
    }
    PHX_CHECK(same_state(local, batch) && incidence_current(local, incidence));

    // The exact stencil: the local verdict and state are the batch's.
    const SourceStencil exact{points, weights};
    const TetEditStatus local_status =
        incidence.collapse_boundary(local, plan->operation, removed, nullptr, &exact).value();
    const TetEditStatus batch_status = apply_batch(batch, plan->operation);
    if (local_status != batch_status) {
      std::fprintf(stderr, "%s stencil collapse: local %s, batch %s\n", name,
                   status_name(local_status), status_name(batch_status));
    }
    PHX_CHECK(local_status == batch_status);
    PHX_CHECK(same_state(local, batch) && consistent_complex(local) &&
              incidence_current(local, incidence));
    if (local_status == TetEditStatus::kApplied) {
      banks.apply(plan->operation);
      ++tally.applied;
    }
  }
}

// A box whose facet vertices are witness-free (their own exact points);
// segment vertices keep their segment witnesses.
Box witness_free_box(std::array<int, 3> n, bool two_regions) {
  Box box = make_box(n, two_regions);
  for (SourceWitness& witness : box.witnesses) {
    if (witness.stratum != SourceStratum::kSegment) witness = SourceWitness{};
  }
  return box;
}

void stencil_collapses_match_the_batch() {
  StencilTally tally;
  stencil_collapse_case("box", witness_free_box({4, 4, 4}, false), 41, tally);
  stencil_collapse_case("two-region box", witness_free_box({4, 4, 2}, true), 42, tally);
  std::printf(
      "stencil collapses: %d applied; refused %d without a stencil, %d inexact, %d off the "
      "carrier\n",
      tally.applied, tally.refused_without_stencil, tally.refused_inexact, tally.refused_outside);
  PHX_CHECK(tally.applied > 0 && tally.refused_inexact > 0 && tally.refused_outside > 0);
}

enum class RoundedControl { kCovered, kOblique, kAuthoredSegment, kOppositeZeroSign };

void rounded_survivor_matches_batch_across_chart_boundary(RoundedControl control) {
  const bool oblique = control == RoundedControl::kOblique;
  const bool admit = control == RoundedControl::kCovered;
  Box box = witness_free_box({4, 4, 2}, false);
  const auto transform = [oblique](double* point) {
    point[0] *= 1.25;
    point[1] *= 1.75;
    if (oblique) point[2] += 0.125 * point[0];
  };
  for (auto* points : {&box.points, &box.source_points}) {
    for (std::size_t row = 0; row < points->size(); row += 3) transform(points->data() + row);
  }
  for (std::int32_t group : box.face_groups) {
    const Piece& piece = box.pieces.at(group);
    const auto axis = static_cast<std::size_t>(piece.axis);
    std::array<double, 9> plane{};
    plane[axis] = piece.value;
    plane[3 + (axis + 1) % 3] = 1.0;
    plane[6 + (axis + 2) % 3] = 1.0;
    for (std::size_t row = 0; row < 9; row += 3) transform(plane.data() + row);
    box.face_planes.insert(box.face_planes.end(), plane.begin(), plane.end());
    box.face_plane_tolerances.push_back(0.0);
  }
  // A coordinate duplicate is not a source row of the bottom face group.
  const auto foreign = static_cast<std::int32_t>(box.source_points.size() / 3);
  box.source_points.insert(box.source_points.end(), {0.0, 0.0, 0.0});
  if (control == RoundedControl::kAuthoredSegment) {
    // Assign semantic source identity to the formerly unprotected internal
    // diagonal; the live survivor's incident facets and vertices are unchanged.
    const auto segment = static_cast<std::int32_t>(box.segment_groups.size());
    box.source_segments.insert(box.source_segments.end(), {0, 3});
    box.segment_groups.push_back(segment);
    box.segment_ids.push_back(segment);
    box.segment_tolerances.push_back(0.0);
  }
  TetMesh local(BoundaryPolicy::kConforming);
  TetMesh batch(BoundaryPolicy::kConforming);
  if (!build(local, box, {}) || !build(batch, box, {})) return;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(local));
  constexpr std::int32_t kept = 18;  // (3, 3, 0) before the transform
  constexpr double a = 0x1.999999999999cp-1;
  constexpr double b = 0x1.999999999999dp-1;
  const double x = 5.0 * a;
  const double y = 7.0 * b;
  // The image lies to one side of the diagonal, its rounding to the other.
  PHX_CHECK((Dyadic(a) - Dyadic(b)).sign() < 0);
  PHX_CHECK((Dyadic(7.0) * Dyadic(x) - Dyadic(5.0) * Dyadic(y)).sign() > 0);
  const std::array<std::int32_t, 3> rows{0, 2, 3};
  const std::array<double, 3> weights{1.0 - b, b - a, a};
  constexpr auto rounded = detail::SourceStencilImage::kRoundedPlane;
  const SourceStencil stencil{rows, weights, rounded};
  const std::array<std::int32_t, 3> wrong_group{foreign, rows[1], rows[2]};
  const std::array<double, 3> wrong_point{weights[0] - 0x1p-49, weights[1], weights[2] + 0x1p-49};
  const std::array bad_stencils{
      SourceStencil{wrong_group, weights, rounded},
      SourceStencil{rows, wrong_point, rounded},
      SourceStencil{rows, std::span(weights).first<2>(), rounded},
  };
  MovePlan move{incidence.vertex_star(kept), {}};
  move.vertex.id = local.vertex_id(kept);
  const bool wrong_zero = control == RoundedControl::kOppositeZeroSign;
  const double z = oblique ? 0.125 * x : (wrong_zero ? -0.0 : 0.0);
  move.vertex.carrier = {x, y, z};
  const auto refuses_move = [&](Point carrier, const SourceStencil& proof,
                                TetEditStatus expected = TetEditStatus::kRejectedTopology) {
    TetEditVertex candidate = move.vertex;
    candidate.carrier = carrier;
    PHX_CHECK(incidence.relocate(local, std::span(&candidate, 1), move.star, &proof) == expected);
    PHX_CHECK(same_state(local, batch) && incidence_current(local, incidence));
  };
  if (admit) {
    for (const SourceStencil& bad : bad_stencils) refuses_move(move.vertex.carrier, bad);
    const std::array<std::int32_t, 2> rim_rows{1, 3};
    const std::array<double, 2> rim_weights{0.1875, 0.8125};
    refuses_move({5.0, 7.0 * 0.8125, 0.0}, {rim_rows, rim_weights, rounded});

    // A supported image inside the source group can still invert the star.
    const std::array<std::int32_t, 3> inverted_rows{0, 1, 3};
    const std::array<double, 3> inverted_weights{0.75, 0.125, 0.125};
    refuses_move({1.25, 0.875, 0.0}, {inverted_rows, inverted_weights, rounded},
                 TetEditStatus::kInvalidGeometry);
  } else {
    refuses_move(move.vertex.carrier, stencil);
  }
  PHX_CHECK(incidence.relocate(local, std::span(&move.vertex, 1), move.star,
                               admit ? &stencil : nullptr) == TetEditStatus::kApplied);
  if (wrong_zero) PHX_CHECK(std::signbit(local.point(kept)[2]));
  TetEditStatus status = TetEditStatus::kApplied;
  PHX_CHECK(batch.apply_relocation_edit(relocation_edit(batch, move), status));
  PHX_CHECK(status == TetEditStatus::kApplied && same_state(local, batch));
  Banks banks(local);
  Generator generator(box, 37);
  for (std::int32_t removed : generator.link(local, incidence, kept)) {
    if (removed < 0 || local.vertex_dimension(removed) != 2) continue;
    const double* candidate = local.point(removed);
    if (candidate[2] != (oblique ? 0.125 * candidate[0] : 0.0)) continue;
    const auto plan = generator.collapse(local, incidence, banks, Perturbation::kNone, false,
                                         false, std::pair{removed, kept});
    if (!plan) continue;
    const auto star = incidence.vertex_star(removed);
    if (admit) {
      for (const SourceStencil& bad : bad_stencils) {
        PHX_CHECK(!incidence.collapse_boundary(local, plan->operation, star, nullptr, &bad));
        PHX_CHECK(same_state(local, batch));
      }
      if (!plan->operation.replacement_facets.empty()) {
        TetEdit reversed = plan->operation;
        reversed.replacement_facets.front().orientation *= -1;
        const auto refusal = apply_batch(batch, reversed);
        PHX_CHECK(refusal != TetEditStatus::kApplied);
        PHX_CHECK(incidence.collapse_boundary(local, reversed, star, nullptr, &stencil) ==
                  refusal);
        PHX_CHECK(same_state(local, batch));
      }
    }
    const TetEditStatus reference = apply_batch(batch, plan->operation);
    const auto generation = local.generation();
    const auto actual = incidence.collapse_boundary(local, plan->operation, star, nullptr, &stencil);
    if (!admit) {
      PHX_CHECK(!actual && local.generation() == generation);
      if (reference == TetEditStatus::kApplied) return;
    } else {
      PHX_CHECK(actual == reference && same_state(local, batch));
      if (reference == TetEditStatus::kApplied) {
        PHX_CHECK(consistent_complex(local) && incidence_current(local, incidence));
        return;
      }
    }
  }
  PHX_CHECK(false);  // the fixture must exercise a batch-admissible collapse
}

void boundary_lookups_require_bound_mesh() {
  LocalIncidence incidence;
  const TetSnapshotFacet* facet = nullptr;
  const TetSnapshotSegment* segment = nullptr;
  PHX_CHECK(incidence.boundary_facet({0, 1, 2}, facet) == TetEditStatus::kStaleGeneration);
  PHX_CHECK(incidence.boundary_segment({0, 1}, segment) == TetEditStatus::kStaleGeneration);
}

void stencil_relocations_verify_the_candidate() {
  const Box box = witness_free_box({4, 4, 4}, true);
  TetMesh local(BoundaryPolicy::kConforming);
  TetMesh batch(BoundaryPolicy::kConforming);
  if (!build(local, box, {}) || !build(batch, box, {})) return;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(local));
  const auto plan_at = [&](std::int32_t slot, Point candidate) {
    MovePlan plan{incidence.vertex_star(slot), {}};
    plan.vertex.id = local.vertex_id(slot);
    plan.vertex.carrier = candidate;
    plan.vertex.witness = local.witness(slot);
    return plan;
  };
  constexpr auto topology = TetEditStatus::kRejectedTopology;
  const auto refuses = [&](const MovePlan& plan, SourceStencil stencil,
                           TetEditStatus expected = TetEditStatus::kRejectedTopology) {
    PHX_CHECK(incidence.relocate(local, std::span(&plan.vertex, 1), plan.star, &stencil) ==
              expected);
    PHX_CHECK(same_state(local, batch) && incidence_current(local, incidence));
  };

  // Move (1, 1, 0) on the left half of z = 0; row 9 is its corner at (2, 4, 0).
  const MovePlan move = plan_at(6, {1.25, 1.0, 0.0});
  const std::array<std::int32_t, 3> rows{0, 8, 2};
  const std::array<double, 3> weights{0.125, 0.625, 0.25};
  constexpr std::int32_t far_corner = 9;
  refuses(move, {});
  refuses(move, {rows, std::span(weights).first<2>()});
  refuses(move, {rows, std::array{std::numeric_limits<double>::infinity(), 0.625, 0.25}});
  // Changing the origin's weight preserves the coordinates but not the exact sum.
  refuses(move, {rows, std::array{std::nextafter(weights[0], 1.0), 0.625, 0.25}});
  // These weights certify the old coordinates, not the candidate.
  refuses(move, {rows, std::array{0.25, 0.5, 0.25}});
  // The sum is exactly one; the reconstructed x and y round to the candidate but differ exactly.
  refuses(move, {std::array{0, 8, 2, far_corner},
                 std::array{0.125 - 0x1p-56, 0.625, 0.25, 0x1p-56}});
  // Even rows of zero weight must exist and be distinct.
  const auto zero_weight = std::array{0.125, 0.625, 0.25, 0.0};
  refuses(move, {std::array{0, 8, 2, 0}, zero_weight});
  refuses(move, {std::array{0, 8, 2, -1}, zero_weight});
  refuses(move, {std::array{0, 8, 2, 12}, zero_weight});
  // Row 1 is on the same plane, but not in the incident source group.
  refuses(move, {std::array{0, 1, 2}, std::array{0.4375, 0.3125, 0.25}});

  // An exact, supported affine extrapolation is not proof of source facet coverage.
  const MovePlan outside = plan_at(6, {2.25, 1.0, 0.0});
  refuses(outside, {rows, std::array{-0.375, 1.125, 0.25}}, TetEditStatus::kInvalidGeometry);

  // Stencils cannot silently fall through the interior or witnessed relocation paths.
  const MovePlan interior = plan_at(31, {1.25, 1.0, 1.0});
  refuses(interior, {std::array{0, 1, 2, 4}, std::array{0.1875, 0.3125, 0.25, 0.25}});
  MovePlan witnessed = plan_at(5, {0.0, 1.25, 0.0});
  PHX_CHECK(segment_point(box, witnessed.vertex.witness.entity, 0.3125,
                          witnessed.vertex.carrier.data(), witnessed.vertex.witness));
  const std::array<std::int32_t, 2> edge_rows{0, 2};
  const std::array<double, 2> edge_weights{0.6875, 0.3125};
  refuses(witnessed, {edge_rows, edge_weights});
  witnessed.vertex.witness = {};
  refuses(witnessed, {edge_rows, edge_weights}, TetEditStatus::kRejectedProtected);

  // Affine weights may be signed; a zero-weight row outside the group is not support.
  const std::array<std::int32_t, 5> exact_rows{far_corner, 8, 0, 2, 1};
  const std::array<double, 5> exact_weights{-0.25, 0.875, -0.125, 0.5, -0.0};
  const SourceStencil stencil{exact_rows, exact_weights};
  PHX_CHECK(incidence.relocate(local, std::span(&move.vertex, 1), move.star, &stencil) ==
            TetEditStatus::kApplied);
  TetEditStatus status = topology;
  PHX_CHECK(batch.apply_relocation_edit(relocation_edit(batch, move), status));
  PHX_CHECK(status == TetEditStatus::kApplied && same_state(local, batch) &&
            consistent_complex(local) && incidence_current(local, incidence));
}

void prepared_relocation_rebinds_to_new_source() {
  LocalIncidence incidence;
  for (double offset : {0.0, 16.0}) {
    Box box = witness_free_box({4, 4, 4}, true);
    for (auto* points : {&box.points, &box.source_points}) {
      for (std::size_t row = 0; row < points->size(); row += 3) (*points)[row] += offset;
    }
    for (std::int32_t group : box.face_groups) {
      const Piece& piece = box.pieces.at(group);
      const auto axis = static_cast<std::size_t>(piece.axis);
      std::array<double, 9> plane{offset};
      plane[axis] = piece.value + (axis == 0 ? offset : 0.0);
      plane[3 + (axis + 1) % 3] = 1.0;
      plane[6 + (axis + 2) % 3] = 1.0;
      box.face_planes.insert(box.face_planes.end(), plane.begin(), plane.end());
      box.face_plane_tolerances.push_back(0.0);
    }
    TetMesh mesh(BoundaryPolicy::kConforming);
    if (!build(mesh, box, {})) return;
    PHX_CHECK(incidence.initialize(mesh));
    constexpr std::int32_t slot = 6;
    const auto generation = mesh.generation();
    const auto star = incidence.vertex_star(slot);
    const std::array<std::int32_t, 3> rows{0, 8, 2};
    const std::array<double, 3> extrapolated{-0.375, 1.125, 0.25};
    const TetEditVertex outside{mesh.vertex_id(slot), {offset + 2.25, 1.0, 0.0}};
    const SourceStencil unsupported{rows, extrapolated};
    PHX_CHECK(incidence.relocate(mesh, std::span(&outside, 1), star, &unsupported) ==
              TetEditStatus::kInvalidGeometry);
    PHX_CHECK(mesh.generation() == generation && mesh.point(slot)[0] == offset + 1.0);
    const std::array<double, 3> weights{0.125, 0.625, 0.25};
    const TetEditVertex inside{mesh.vertex_id(slot), {offset + 1.25, 1.0, 0.0}};
    const SourceStencil supported{rows, weights};
    PHX_CHECK(incidence.relocate(mesh, std::span(&inside, 1), star, &supported) ==
              TetEditStatus::kApplied);
    PHX_CHECK(mesh.generation() == generation + 1 && mesh.point(slot)[0] == inside.carrier[0]);
  }
}

void relocation_rejects_a_hole_or_notch_in_declared_source(int patch_x) {
  Box box = make_box({4, 4, 4}, false);
  box.source_points = box.points;
  box.source_faces = box.faces;
  box.segments.clear();
  box.segment_sources.clear();
  box.source_segments.clear();
  box.segment_groups.clear();
  box.segment_ids.clear();
  box.segment_tolerances.clear();
  box.witnesses.assign(box.points.size() / 3, SourceWitness{});
  box.face_ids.resize(box.face_sources.size());
  for (std::size_t row = 0; row < box.face_ids.size(); ++row) {
    box.face_ids[row] = static_cast<std::int64_t>(row);
  }
  box.face_tolerances.assign(box.face_sources.size(), 0.0);
  box.face_plane_tolerances.assign(box.face_sources.size(), 0.0);
  box.orientations.assign(box.face_sources.size(), 1);
  int patch_faces = 0;
  for (std::size_t row = 0; row < box.face_sources.size(); ++row) {
    const auto& piece = box.pieces.at(box.face_sources[row]);
    const auto axis = static_cast<std::size_t>(piece.axis);
    std::array<double, 9> plane{};
    plane[axis] = piece.value;
    plane[3 + (axis + 1) % 3] = 1.0;
    plane[6 + (axis + 2) % 3] = 1.0;
    box.face_planes.insert(box.face_planes.end(), plane.begin(), plane.end());
    const auto triangle = std::span(box.faces).subspan(3 * row, 3);
    if (axis == 2 && piece.value == 0.0 &&
        std::ranges::all_of(triangle, [&](std::int32_t vertex) {
          const double* p = box.points.data() + 3 * vertex;
          return p[0] >= patch_x && p[0] <= patch_x + 1 && p[1] >= 2.0 && p[1] <= 3.0;
        })) {
      box.face_sources[row] = 6;
      ++patch_faces;
    }
  }
  PHX_CHECK(patch_faces == 2);
  box.face_groups = box.face_sources;
  TetMesh local(BoundaryPolicy::kConforming);
  TetMesh batch(BoundaryPolicy::kConforming);
  if (!build(local, box, {}) || !build(batch, box, {})) return;
  LocalIncidence incidence;
  PHX_CHECK(incidence.initialize(local));
  const std::int32_t vertex = patch_x + 2 * 5;
  const TetEditVertex moved{local.vertex_id(vertex), {patch_x + 0.25, 2.25, 0.0}};
  const auto star = incidence.vertex_star(vertex);
  for (std::int32_t cell : star) {
    const auto& tetrahedron = local.tet(cell);
    const double* p[4];
    for (int k = 0; k < 4; ++k) {
      const auto slot = tetrahedron.v[k];
      p[k] = slot == vertex ? moved.carrier.data() : local.point(slot);
    }
    PHX_CHECK(orient3d(p[0], p[1], p[2], p[3]) > 0);
  }
  const std::array<std::int32_t, 3> rows{vertex, vertex + 1, vertex + 5};
  const std::array<double, 3> weights{0.5, 0.25, 0.25};
  const SourceStencil stencil{rows, weights};
  const TetEditStatus current = incidence.relocate(local, std::span(&moved, 1), star, &stencil);
  const MovePlan plan{incidence.vertex_star(vertex), moved};
  TetEditStatus reference = TetEditStatus::kApplied;
  PHX_CHECK(batch.apply_relocation_edit(relocation_edit(batch, plan), reference));
  // Inside the hull, but live facets assigned to the outer group would
  // enter the patch that belongs to the other source.
  PHX_CHECK(current == TetEditStatus::kInvalidGeometry && current == reference);
  PHX_CHECK(same_state(local, batch) && incidence_current(local, incidence));
}

void stencil_relocations_respect_incident_segments() {
  for (bool split_faces : {false, true}) {
    Box box = witness_free_box({4, 4, 4}, split_faces);
    box.witnesses[1] = {};  // (1, 0, 0), on the x-axis source segment.
    TetMesh local(BoundaryPolicy::kConforming);
    TetMesh batch(BoundaryPolicy::kConforming);
    if (!build(local, box, {}) || !build(batch, box, {})) return;
    LocalIncidence incidence;
    PHX_CHECK(incidence.initialize(local));
    MovePlan move{incidence.vertex_star(1), {}};
    move.vertex.id = local.vertex_id(1);
    move.vertex.carrier = {1.25, 0.0, 0.0};
    const std::span<const TetEditVertex> candidate(&move.vertex, 1);
    const std::array<std::int32_t, 2> segment_rows{0, 1};
    const std::array<double, 2> segment_weights{0.6875, 0.3125};
    const SourceStencil segment_stencil{segment_rows, segment_weights};
    if (split_faces) {
      // Each incident facet group ends at row 8; the source segment ends at row 1.
      // Neither exact stencil lies in the intersection of all incident groups.
      const std::array<std::int32_t, 2> face_rows{0, 8};
      const std::array<double, 2> face_weights{0.375, 0.625};
      const SourceStencil face_stencil{face_rows, face_weights};
      for (const SourceStencil* stencil : {&face_stencil, &segment_stencil}) {
        PHX_CHECK(incidence.relocate(local, candidate, move.star, stencil) ==
                  TetEditStatus::kRejectedTopology);
        PHX_CHECK(same_state(local, batch) && incidence_current(local, incidence));
      }
    }
    PHX_CHECK(incidence.relocate(local, candidate, move.star,
                                 split_faces ? nullptr : &segment_stencil) ==
              TetEditStatus::kApplied);
    TetEditStatus status = TetEditStatus::kRejectedTopology;
    PHX_CHECK(batch.apply_relocation_edit(relocation_edit(batch, move), status));
    PHX_CHECK(status == TetEditStatus::kApplied && same_state(local, batch) &&
              consistent_complex(local) && incidence_current(local, incidence));
  }
}

}  // namespace

int main() {
  boundary_edits_match_apply_edit_batch();
  unsupported_meshes_are_refused();
  splits_and_relocations_match_the_batch();
  boundary_edits_are_star_local();
  boundary_cavities_never_hold_a_non_disk_constrained_face();
  stencil_collapses_match_the_batch();
  rounded_survivor_matches_batch_across_chart_boundary(RoundedControl::kCovered);
  rounded_survivor_matches_batch_across_chart_boundary(RoundedControl::kOblique);
  rounded_survivor_matches_batch_across_chart_boundary(RoundedControl::kAuthoredSegment);
  rounded_survivor_matches_batch_across_chart_boundary(RoundedControl::kOppositeZeroSign);
  boundary_lookups_require_bound_mesh();
  stencil_relocations_verify_the_candidate();
  prepared_relocation_rebinds_to_new_source();
  relocation_rejects_a_hole_or_notch_in_declared_source(2);  // bounded hole
  relocation_rejects_a_hole_or_notch_in_declared_source(3);  // open notch
  stencil_relocations_respect_incident_segments();
  return phx::mc::test::finish("test_local_boundary_edits");
}
