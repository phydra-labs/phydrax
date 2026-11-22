//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Metric adaptation of a TetMesh in place. Stencil combination is checked
// against exact dyadic arithmetic by hand. Runs adapt Kuhn cubes of [0, 1]^3
// (cube sides as six coplanar facet groups) under constant tensors whose
// target sizes need refinement, coarsening and boundary edits, and check the
// result independently of the loop: every edge in the band, every cell
// positively oriented (exact predicate), the unit volume conserved, and every
// boundary face on its cube side.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "check.hpp"
#include "metric_adapt3d.hpp"
#include "metric_float.hpp"
#include "plc3d.hpp"
#include "predicates.hpp"
#include "tet_mesh.hpp"

namespace {

using namespace phx::mc;
namespace metric = phx::mc::metric;

constexpr double kLower = 0x1.6a09e667f3bcdp-1;  // fl(1 / sqrt 2)
constexpr double kUpper = 0x1.6a09e667f3bcdp+0;  // fl(sqrt 2)

void check_combine_stencils() {
  metric::Stencil combined;
  PHX_CHECK(metric::combine_stencils({{0, 1.0}}, {{1, 1.0}}, 0.25, combined));
  PHX_CHECK(combined.size() == 2);
  if (combined.size() == 2) {
    PHX_CHECK(combined[0].row == 0 && combined[0].weight == 0.75);
    PHX_CHECK(combined[1].row == 1 && combined[1].weight == 0.25);
  }
  // A term whose combined weight is zero is dropped.
  PHX_CHECK(metric::combine_stencils({{0, 0.5}, {1, 0.5}}, {{1, 1.0}}, 1.0, combined));
  PHX_CHECK(combined.size() == 1 && combined[0].row == 1 && combined[0].weight == 1.0);
  // 1/2 + 2^-61 needs 61 significant bits: not a binary64 value.
  PHX_CHECK(!metric::combine_stencils({{0, 1.0}}, {{0, 0x1p-60}, {1, 1.0}}, 0.5, combined));
}

// The Kuhn subdivision of an n^3 grid of [0, 1]^3, and its boundary faces
// with the cube side (0..5: x = 0, x = 1, y = 0, y = 1, z = 0, z = 1) as group.
struct Cube {
  std::vector<double> points;
  std::vector<std::int32_t> tetrahedra;
  std::vector<std::int32_t> faces;
  std::vector<std::int32_t> sides;
};

Cube kuhn_cube(int n) {
  Cube cube;
  const auto index = [n](int i, int j, int k) { return (i * (n + 1) + j) * (n + 1) + k; };
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      for (int k = 0; k <= n; ++k) {
        cube.points.push_back(static_cast<double>(i) / n);
        cube.points.push_back(static_cast<double>(j) / n);
        cube.points.push_back(static_cast<double>(k) / n);
      }
    }
  }
  const std::array<std::array<int, 3>, 6> orders{{{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                                   {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
  // Faces by sorted key: their count and their outward orientation.
  std::map<std::array<std::int32_t, 3>, std::pair<int, std::array<std::int32_t, 3>>> face_count;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        for (const std::array<int, 3>& order : orders) {
          std::array<int, 3> corner{i, j, k};
          std::array<std::int32_t, 4> cell{};
          cell[0] = index(corner[0], corner[1], corner[2]);
          for (std::size_t step = 0; step < 3; ++step) {
            ++corner[static_cast<std::size_t>(order[step])];
            cell[step + 1] = index(corner[0], corner[1], corner[2]);
          }
          const double* a = cube.points.data() + 3 * cell[0];
          if (orient3d(a, cube.points.data() + 3 * cell[1], cube.points.data() + 3 * cell[2],
                       cube.points.data() + 3 * cell[3]) < 0) {
            std::swap(cell[1], cell[2]);
          }
          cube.tetrahedra.insert(cube.tetrahedra.end(), cell.begin(), cell.end());
          // The faces of a positive cell, each oriented outward.
          const std::array<std::array<std::size_t, 3>, 4> outward{
              {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}}};
          for (const std::array<std::size_t, 3>& rows : outward) {
            const std::array<std::int32_t, 3> face{cell[rows[0]], cell[rows[1]], cell[rows[2]]};
            std::array<std::int32_t, 3> key = face;
            std::sort(key.begin(), key.end());
            auto& entry = face_count[key];
            ++entry.first;
            entry.second = face;
          }
        }
      }
    }
  }
  for (const auto& [key, entry] : face_count) {
    const auto& [count, face] = entry;
    if (count != 1) continue;
    int side = -1;
    for (int axis = 0; axis < 3 && side < 0; ++axis) {
      for (int end = 0; end < 2 && side < 0; ++end) {
        const double value = static_cast<double>(end);
        bool on = true;
        for (std::int32_t vertex : face) on = on && cube.points[3 * vertex + axis] == value;
        if (on) side = 2 * axis + end;
      }
    }
    cube.faces.insert(cube.faces.end(), face.begin(), face.end());
    cube.sides.push_back(side);
  }
  return cube;
}

double volume(const double* a, const double* b, const double* c, const double* d) {
  const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double w[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
  return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
          u[2] * (v[0] * w[1] - v[1] * w[0])) /
         6.0;
}

struct Run {
  metric::AdaptOutput output;
  TetMeshSnapshot before;
  TetMeshSnapshot after;
  TetEditStatus status = TetEditStatus::kApplied;
};

// Adapts the Kuhn cube of n^3 cells under the constant tensor c I.
Run adapt_cube(int n, double c, bool witnessed = false) {
  Cube cube = kuhn_cube(n);
  const std::int32_t moved = n + 2;  // (0, 1/n, 1/n)
  if (witnessed) cube.points[3 * moved + 1] *= 0.625;
  Run run;
  TetMesh mesh(BoundaryPolicy::kConforming);
  const std::vector<std::int32_t> regions(cube.tetrahedra.size() / 4, 0);
  // Each cube side declares its exact plane (origin, two axes): the plane of
  // its first face, whose axes are exact binary64 differences.
  const std::size_t faces = cube.sides.size();
  std::vector<double> planes(9 * faces);
  std::map<std::int32_t, std::size_t> first_face;
  for (std::size_t row = 0; row < faces; ++row) first_face.emplace(cube.sides[row], row);
  for (std::size_t row = 0; row < faces; ++row) {
    const std::size_t first = first_face.at(cube.sides[row]);
    const double* a = cube.points.data() + 3 * cube.faces[3 * first];
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const double* p = cube.points.data() + 3 * cube.faces[3 * first + corner];
      for (std::size_t axis = 0; axis < 3; ++axis) {
        planes[9 * row + 3 * corner + axis] = corner == 0 ? a[axis] : p[axis] - a[axis];
      }
    }
  }
  const std::vector<std::int64_t> face_ids(cube.sides.begin(), cube.sides.end());
  const std::vector<double> zeros(faces, 0.0);
  SourceComplexView source;
  source.points = cube.points;
  source.faces = cube.faces;
  source.face_group_rows = cube.sides;
  source.face_ids = face_ids;
  source.face_tolerances = zeros;
  source.face_planes = planes;
  source.face_plane_tolerances = zeros;
  std::vector<SourceWitness> witnesses;
  std::vector<std::int8_t> fixed;
  const std::size_t count = cube.points.size() / 3;
  if (witnessed) {
    witnesses.resize(count);
    fixed.assign(count, 1);
    fixed[static_cast<std::size_t>(moved)] = 0;
    for (std::size_t row = 0; row < faces; ++row) {
      const auto first = cube.faces.begin() + static_cast<std::ptrdiff_t>(3 * row);
      const auto corner = std::find(first, first + 3, moved);
      if (corner == first + 3) continue;
      witnesses[static_cast<std::size_t>(moved)] = {
          SourceStratum::kFacet, static_cast<std::int32_t>(row),
          {corner == first + 1 ? 1.0 : 0.0, corner == first + 2 ? 1.0 : 0.0}};
      break;
    }
    source.witnesses = witnesses;
  }
  PHX_CHECK(mesh.build(cube.points, cube.tetrahedra, regions, cube.faces, cube.sides, {}, {}, {},
                       {}, &source, fixed) == PHX_MC_OK);
  PHX_CHECK(mesh.snapshot(run.before) == TetEditStatus::kApplied);
  metric::AdaptInput input;
  input.source_points = cube.points;
  const double tensor[9] = {c, 0, 0, 0, c, 0, 0, 0, c};
  const double logarithm[9] = {std::log(c), 0, 0, 0, std::log(c), 0, 0, 0, std::log(c)};
  for (std::size_t row = 0; row < count; ++row) {
    input.source_metrics.insert(input.source_metrics.end(), tensor, tensor + 9);
    input.source_logarithms.insert(input.source_logarithms.end(), logarithm, logarithm + 9);
    input.source_vertex_ids.push_back(row);
  }
  input.source_cells.assign(cube.tetrahedra.begin(), cube.tetrahedra.end());
  // The live mesh is the source: each vertex its own source row.
  input.stencil_offsets.push_back(0);
  for (const auto& vertex : run.before.vertices) {
    input.vertex_ids.push_back(vertex.id);
    input.stencil_rows.push_back(static_cast<std::int32_t>(vertex.id));
    input.stencil_weights.push_back(1.0);
    input.stencil_offsets.push_back(static_cast<std::int64_t>(input.stencil_rows.size()));
    input.vertex_target_ids.push_back(vertex.id);
  }
  input.ancestry_offsets.push_back(0);
  for (const auto& cell : run.before.cells) {
    input.cell_ids.push_back(cell.id);
    input.cell_blocks.push_back(0);
    input.cell_target_ids.push_back(cell.id);
    input.ancestry_ids.push_back(cell.id);
    input.ancestry_kinds.push_back(static_cast<std::int32_t>(metric::AdaptLineage::kPreserved));
    input.ancestry_offsets.push_back(static_cast<std::int64_t>(input.ancestry_ids.size()));
    input.next_cell_id = std::max(input.next_cell_id, cell.id + 1);
  }
  input.next_vertex_id = count;
  input.next_target_vertex = count;
  input.next_target_cell = input.next_cell_id;
  input.lower = kLower;
  input.topology_operations = !witnessed;
  run.status = metric::adapt_metric(mesh, input, run.output);
  PHX_CHECK(mesh.snapshot(run.after) == TetEditStatus::kApplied);
  return run;
}

// The result's contract, checked from its snapshot alone.
void check_result(const Run& run, double c) {
  PHX_CHECK(run.status == TetEditStatus::kApplied);
  PHX_CHECK(run.output.status == metric::AdaptStatus::kComplete);
  const TetMeshSnapshot& mesh = run.after;
  std::map<std::uint64_t, const double*> carrier;
  for (const auto& vertex : mesh.vertices) carrier[vertex.id] = vertex.carrier.data();
  const auto point = [&](std::uint64_t id) { return carrier.at(id); };
  double total = 0.0;
  std::set<std::pair<std::uint64_t, std::uint64_t>> edges;
  bool positive = true;
  for (const auto& cell : mesh.cells) {
    const auto& v = cell.vertices;
    positive = positive && orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) > 0;
    total += volume(point(v[0]), point(v[1]), point(v[2]), point(v[3]));
    for (std::size_t first = 0; first < 4; ++first) {
      for (std::size_t second = first + 1; second < 4; ++second) {
        edges.emplace(std::min(v[first], v[second]), std::max(v[first], v[second]));
      }
    }
  }
  PHX_CHECK(positive);
  PHX_CHECK(std::abs(total - 1.0) <= 1.0e-12);
  bool in_band = true;
  for (const auto& [a, b] : edges) {
    const double* p = point(a);
    const double* q = point(b);
    const double length = std::sqrt(c) * std::sqrt((p[0] - q[0]) * (p[0] - q[0]) +
                                                   (p[1] - q[1]) * (p[1] - q[1]) +
                                                   (p[2] - q[2]) * (p[2] - q[2]));
    in_band = in_band && length >= kLower * (1.0 - 1.0e-12) && length <= kUpper * (1.0 + 1.0e-12);
  }
  PHX_CHECK(in_band);
  // Every constrained facet lies on the cube side of its group.
  bool on_side = true;
  for (const auto& facet : mesh.constrained_facets) {
    const int axis = facet.source / 2;
    const double value = static_cast<double>(facet.source % 2);
    for (std::uint64_t id : facet.vertices) on_side = on_side && point(id)[axis] == value;
  }
  PHX_CHECK(on_side);
}

std::size_t boundary_vertices(const TetMeshSnapshot& mesh) {
  std::set<std::uint64_t> vertices;
  for (const auto& facet : mesh.constrained_facets) {
    vertices.insert(facet.vertices.begin(), facet.vertices.end());
  }
  return vertices.size();
}

void check_refinement() {
  // h = 1/2 against the target 1/4: interior and boundary splits.
  const Run run = adapt_cube(2, 16.0);
  check_result(run, 16.0);
  PHX_CHECK(run.output.counters.splits > 0);
  PHX_CHECK(run.after.cells.size() > run.before.cells.size());
}

void check_coarsening() {
  // h = 1/4 against the target 1/2: interior and boundary collapses (the
  // cube sides keep fewer vertices).
  const Run run = adapt_cube(4, 4.0);
  check_result(run, 4.0);
  PHX_CHECK(run.output.counters.collapses > 0);
  PHX_CHECK(boundary_vertices(run.after) < boundary_vertices(run.before));
}

void check_witnessed_boundary_relocation() {
  const Run run = adapt_cube(2, 4.0, true);
  PHX_CHECK(run.status == TetEditStatus::kApplied);
  PHX_CHECK(run.output.counters.relocations > 0);
  if (run.before.vertices.size() <= 4 || run.after.vertices.size() <= 4) {
    PHX_CHECK(false);
    return;
  }
  const auto& before = run.before.vertices[4];
  const auto& after = run.after.vertices[4];
  PHX_CHECK(before.id == after.id && before.carrier != after.carrier);
  const SourceWitness& witness = after.witness;
  PHX_CHECK(witness.stratum == SourceStratum::kFacet && witness.entity >= 0);
  if (witness.stratum != SourceStratum::kFacet || witness.entity < 0) return;
  const std::size_t face = 3 * static_cast<std::size_t>(witness.entity);
  PHX_CHECK(face + 2 < run.before.source_faces.size());
  if (face + 2 >= run.before.source_faces.size()) return;
  const double* corners[3];
  for (std::size_t corner = 0; corner < 3; ++corner) {
    corners[corner] = run.before.source_points.data() + 3 * run.before.source_faces[face + corner];
  }
  std::array<double, 3> point{};
  double deviation = 0.0;
  PHX_CHECK(source_carrier(corners, witness.stratum, witness.parameters, point.data(), deviation));
  PHX_CHECK(after.carrier == point && after.carrier[0] == 0.0);
  PHX_CHECK(witness.deviation == 0.0 && deviation == 0.0);
}

Run adapt_direction_sensitive_edge(bool permuted) {
  // Only the logical body diagonal is editable. Its two evaluation
  // directions straddle the exact binary64 completion threshold.
  Cube cube = kuhn_cube(1);
  for (double& value : cube.points) value *= 0x1.b7f92f0969ff9p-1;
  const std::array<std::uint64_t, 8> ids{41, 719, 203, 367, 691, 27, 109, 533};
  const std::array<std::uint64_t, 6> cell_ids{110, 145, 192, 211, 297, 350};
  metric::AdaptInput input;
  input.source_points = cube.points;
  input.source_vertex_ids.assign(ids.begin(), ids.end());
  input.source_cells.assign(cube.tetrahedra.begin(), cube.tetrahedra.end());
  for (std::size_t row = 0; row < ids.size(); ++row) {
    const double c = row == 0 ? 0.64 : 1.25;
    const std::array<double, 9> tensor{c, 0, 0, 0, c, 0, 0, 0, c};
    std::array<double, 9> logarithm{};
    PHX_CHECK(metric::logarithm(tensor.data(), logarithm.data()));
    input.source_metrics.insert(input.source_metrics.end(), tensor.begin(), tensor.end());
    input.source_logarithms.insert(input.source_logarithms.end(), logarithm.begin(),
                                   logarithm.end());
  }
  const double* a = input.source_points.data();
  const double* b = a + 21;
  const double* ma = input.source_metrics.data();
  const double* mb = ma + 63;
  PHX_CHECK(metric::length(a, b, ma, mb) > metric::kUpperLength);
  PHX_CHECK(metric::length(b, a, mb, ma) <= metric::kUpperLength);
  std::set<std::array<std::uint64_t, 2>> protected_edges;
  for (std::size_t cell = 0; cell < cube.tetrahedra.size(); cell += 4) {
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = i + 1; j < 4; ++j) {
        const auto [low, high] = std::minmax(ids[cube.tetrahedra[cell + i]],
                                             ids[cube.tetrahedra[cell + j]]);
        if (low != ids.front() || high != ids.back()) protected_edges.insert({low, high});
      }
    }
  }
  for (const auto& key : protected_edges) {
    input.protected_edges.insert(input.protected_edges.end(), key.begin(), key.end());
  }
  auto physical_ids = ids;
  if (permuted) {
    std::reverse(physical_ids.begin(), physical_ids.end());
    for (std::size_t row = 0; row < ids.size(); ++row) {
      std::copy_n(input.source_points.data() + 3 * (ids.size() - row - 1), 3,
                  cube.points.data() + 3 * row);
    }
    for (auto* bank : {&cube.tetrahedra, &cube.faces}) {
      for (std::int32_t& row : *bank) row = 7 - row;
    }
  }
  TetMesh mesh(BoundaryPolicy::kConforming);
  const std::vector<std::int32_t> regions(cell_ids.size(), 0);
  Run run;
  const auto setup = mesh.build(cube.points, cube.tetrahedra, regions, cube.faces, cube.sides,
                                 {}, {}, {}, {});
  PHX_CHECK(setup == PHX_MC_OK);
  if (setup != PHX_MC_OK) return run;
  PHX_CHECK(mesh.assign_initial_ids(physical_ids, cell_ids, {}, {}) == PHX_MC_OK);
  PHX_CHECK(mesh.snapshot(run.before) == TetEditStatus::kApplied);
  input.vertex_ids.assign(ids.begin(), ids.end());
  std::sort(input.vertex_ids.begin(), input.vertex_ids.end());
  input.stencil_offsets.push_back(0);
  for (std::uint64_t id : input.vertex_ids) {
    const auto row = std::find(ids.begin(), ids.end(), id) - ids.begin();
    input.stencil_rows.push_back(static_cast<std::int32_t>(row));
    input.stencil_weights.push_back(1.0);
    input.stencil_offsets.push_back(static_cast<std::int64_t>(input.stencil_rows.size()));
  }
  input.vertex_target_ids = input.vertex_ids;
  input.cell_ids.assign(cell_ids.begin(), cell_ids.end());
  input.cell_target_ids = input.cell_ids;
  input.cell_blocks.assign(cell_ids.size(), 0);
  input.ancestry_ids = input.cell_ids;
  input.ancestry_kinds.assign(cell_ids.size(), static_cast<std::int32_t>(metric::AdaptLineage::kPreserved));
  for (std::size_t row = 0; row <= cell_ids.size(); ++row) {
    input.ancestry_offsets.push_back(static_cast<std::int64_t>(row));
  }
  input.next_vertex_id = input.next_target_vertex = 800;
  input.next_cell_id = input.next_target_cell = 800;
  input.lower = 0.6;
  input.relocation = false;
  run.status = metric::adapt_metric(mesh, input, run.output);
  PHX_CHECK(mesh.snapshot(run.after) == TetEditStatus::kApplied);
  return run;
}

void metric_worksets_ignore_physical_vertex_order() {
  const Run first = adapt_direction_sensitive_edge(false);
  const Run second = adapt_direction_sensitive_edge(true);
  PHX_CHECK(first.status == TetEditStatus::kApplied && second.status == first.status);
  PHX_CHECK(first.output.status == metric::AdaptStatus::kComplete &&
            second.output.status == first.output.status);
  PHX_CHECK(first.output.counters.splits == 1 && second.output.counters.splits == 1);
  PHX_CHECK(first.output.vertex_ids == second.output.vertex_ids);
  PHX_CHECK(first.output.stencil_rows == second.output.stencil_rows);
  PHX_CHECK(first.output.stencil_weights == second.output.stencil_weights);
  PHX_CHECK(first.output.ancestry_ids == second.output.ancestry_ids);
  const auto cells = [](const Run& run) {
    std::map<std::uint64_t, std::array<std::uint64_t, 4>> result;
    for (const auto& cell : run.after.cells) result.emplace(cell.id, cell.vertices);
    return result;
  };
  PHX_CHECK(cells(first) == cells(second));
}

}  // namespace

int main() {
  check_combine_stencils();
  check_refinement();
  check_coarsening();
  check_witnessed_boundary_relocation();
  metric_worksets_ignore_physical_vertex_order();
  return phx::mc::test::finish("test_metric_adapt3d");
}
