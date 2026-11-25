// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_scalar_box.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "mesh.hpp"
#include "topology_keys.hpp"

namespace phx::mc::metric {
namespace {

using Row = std::int32_t;
using Flags = std::vector<unsigned char>;
constexpr std::size_t kPatchSites = 4096;
constexpr std::size_t kPatchCells = 8192;
constexpr std::size_t kMaxPatches = 16;

void join(std::vector<Row>& parent, Row a, Row b) {
  a = topology::find_root(parent, a);
  b = topology::find_root(parent, b);
  if (a != b) parent[std::max(a, b)] = std::min(a, b);
}

struct Patch {
  std::vector<Row> cells, vertices;
  bool oversized = false;
};

bool form_patches(std::span<const BoxCell> cells, std::size_t point_count,
                  std::span<const Row> seeds, Flags& seeded, std::vector<Patch>& patches) {
  for (const auto& tet : cells) {
    for (Row v : tet) if (static_cast<std::size_t>(v) >= point_count) return false;
  }
  Flags front(point_count, 0), next(point_count, 0);
  for (Row row : seeds) {
    if (static_cast<std::size_t>(row) >= cells.size()) return false;
    seeded[row] = 1;
    for (Row v : cells[row]) front[v] = 1;
  }
  // Take two simultaneous vertex-star rings. Do not let a newly encountered
  // site extend the current ring as a function of the source cell order.
  for (const auto& tet : cells) {
    if (front[tet[0]] || front[tet[1]] || front[tet[2]] || front[tet[3]]) {
      for (Row v : tet) next[v] = 1;
    }
  }
  std::vector<Row> rows;
  for (std::size_t r = 0; r < cells.size(); ++r) {
    const auto& t = cells[r];
    if (!next[t[0]] && !next[t[1]] && !next[t[2]] && !next[t[3]]) continue;
    if (rows.size() == kMaxPatches * kPatchCells) return false;
    rows.push_back(static_cast<Row>(r));
  }
  // Patches are disjoint even at vertices, so a repair in one cannot reuse
  // any simplex from a different patch. Number them by first old cell row.
  std::vector<Row> parent(rows.size()), owner(point_count, -1);
  std::iota(parent.begin(), parent.end(), Row{0});
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (Row v : cells[rows[r]]) {
      if (owner[v] < 0) owner[v] = static_cast<Row>(r);
      else join(parent, owner[v], static_cast<Row>(r));
    }
  }
  std::vector<Row> patch_of_root(rows.size(), -1);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto& id = patch_of_root[topology::find_root(parent, static_cast<Row>(r))];
    if (id < 0) {
      if (patches.size() == kMaxPatches) return false;
      id = static_cast<Row>(patches.size());
      patches.emplace_back();
    }
    auto& patch = patches[id];
    if (patch.cells.size() == kPatchCells) patch.oversized = true;
    else patch.cells.push_back(rows[r]);
  }
  for (std::size_t v = 0; v < owner.size(); ++v) {
    if (owner[v] < 0) continue;
    auto& patch = patches[patch_of_root[topology::find_root(parent, owner[v])]];
    if (patch.vertices.size() == kPatchSites) patch.oversized = true;
    if (!patch.oversized) patch.vertices.push_back(static_cast<Row>(v));
  }
  return true;
}

// The point-set owner enforces this chamber's finite-cell budget while
// inserting sites, not after allocating an unbounded Delaunay complex.
std::unique_ptr<phx_mc_mesh> patch_delaunay(const Patch& patch, std::span<const BoxPoint> points) {
  const auto n = patch.vertices.size();
  if (n < 4) return {};
  std::vector<double> coordinates(3 * n);
  for (std::size_t k = 0; k < n; ++k) {
    std::copy_n(points[patch.vertices[k]].data(), 3, coordinates.data() + 3 * k);
  }
  phx_mc_mesh* raw = nullptr;
  const auto status = triangulate_3d(static_cast<std::int64_t>(n), coordinates.data(), nullptr,
                                     false, true, &raw, 1, kPatchCells);
  std::unique_ptr<phx_mc_mesh> mesh(raw);
  if (!scalar_topology_accepted(status)) return {};
  for (std::size_t k = 0; k < n; ++k) {
    if (mesh->vertex_map[k] != static_cast<Row>(k)) return {};
  }
  return mesh;
}

using Triangle = std::array<Row, 3>;
struct Face {
  Triangle key;
  // Incident rows with the outward sorted loop, or its reverse, respectively.
  std::array<Row, 2> side{-1, -1};
};
using FaceBank = std::vector<Face>;

bool positive(std::span<const BoxPoint> points, const BoxCell& t) {
  return orient3d_filtered(points[t[0]].data(), points[t[1]].data(),
                           points[t[2]].data(), points[t[3]].data()) > 0;
}

bool face_bank(std::span<const BoxCell> cells, std::span<const BoxPoint> points, FaceBank& faces) {
  faces.reserve(4 * cells.size());
  for (std::size_t row = 0; row < cells.size(); ++row) {
    const auto& c = cells[row];
    if (!positive(points, c)) return false;
    for (const auto& ord : topology::kTetFaceOrder) {
      // The helper is inward; reverse it before computing permutation parity.
      Face face{{c[ord[0]], c[ord[2]], c[ord[1]]}};
      auto& [a, b, d] = face.key;
      bool reverse = false;
      if (a > b) { std::swap(a, b); reverse = !reverse; }
      if (b > d) { std::swap(b, d); reverse = !reverse; }
      if (a > b) { std::swap(a, b); reverse = !reverse; }
      face.side[reverse] = static_cast<Row>(row);
      faces.push_back(face);
    }
  }
  std::sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) { return a.key < b.key; });
  std::size_t out = 0;
  for (const auto& face : faces) {
    if (out == 0 || faces[out - 1].key != face.key) {
      faces[out++] = face;
      continue;
    }
    auto& gathered = faces[out - 1];
    for (int side = 0; side < 2; ++side) {
      if (face.side[side] < 0) continue;
      if (gathered.side[side] >= 0) return false;
      gathered.side[side] = face.side[side];
    }
  }
  faces.resize(out);
  return true;
}

// Cut both meshes on every common triangle. All other paired faces connect
// chambers, not outer facets; they must not be mistaken for exchange seams.
std::vector<Row> chambers(const FaceBank& faces, const FaceBank& other, std::size_t count) {
  std::vector<Row> parent(count);
  std::iota(parent.begin(), parent.end(), Row{0});
  auto next = other.begin();
  for (const auto& face : faces) {
    if (face.side[0] < 0 || face.side[1] < 0) continue;
    while (next != other.end() && next->key < face.key) ++next;
    if (next == other.end() || next->key != face.key) join(parent, face.side[0], face.side[1]);
  }
  for (std::size_t k = 0; k < count; ++k) parent[k] = topology::find_root(parent, static_cast<Row>(k));
  return parent;
}

using Chain = std::vector<std::pair<Triangle, int>>;
struct Boundary {
  Chain interface;
  Chain wall_rim;  // (wall number, lower endpoint row, higher endpoint row)
  bool operator==(const Boundary&) const = default;
};

bool boundary_chain(const ScalarBoxField& field, std::span<const BoxPoint> points,
                    const FaceBank& faces, const Flags& selected, Boundary& boundary) {
  for (const auto& face : faces) {
    const auto [a, b] = face.side;
    const bool forward = a >= 0 && selected[a], reverse = b >= 0 && selected[b];
    if (forward == reverse) continue;
    const int coefficient = forward ? 1 : -1;
    bool on_wall = false;
    for (int axis = 0; axis < 3 && !on_wall; ++axis) {
      const auto& [u, v, w] = face.key;
      const double x = points[u][axis];
      if (x != points[v][axis] || x != points[w][axis] ||
          (x != field.low[axis] && x != field.high[axis])) continue;
      const int i = (axis + 1) % 3, j = (axis + 2) % 3;
      const double p[2] = {points[u][i], points[u][j]};
      const double q[2] = {points[v][i], points[v][j]};
      const double r[2] = {points[w][i], points[w][j]};
      const bool high = x == field.high[axis];
      if (coefficient * orient2d_filtered(p, q, r) != (high ? 1 : -1)) return false;
      const int wall = 2 * axis + (high ? 1 : 0);
      boundary.wall_rim.emplace_back(Triangle{wall, u, v}, coefficient);
      boundary.wall_rim.emplace_back(Triangle{wall, v, w}, coefficient);
      boundary.wall_rim.emplace_back(Triangle{wall, u, w}, -coefficient);
      on_wall = true;
    }
    if (!on_wall) boundary.interface.emplace_back(face.key, coefficient);
  }
  auto& rim = boundary.wall_rim;
  std::sort(rim.begin(), rim.end());
  std::size_t out = 0;
  for (std::size_t i = 0; i < rim.size();) {
    const auto key = rim[i].first;
    int coefficient = 0;
    do { coefficient += rim[i++].second; } while (i < rim.size() && rim[i].first == key);
    if (coefficient < -1 || coefficient > 1) return false;
    if (coefficient != 0) rim[out++] = {key, coefficient};
  }
  rim.resize(out);
  return true;
}

BoxCell canonical(BoxCell cell) {
  // Ascending rows, except the last pair when needed to preserve parity.
  constexpr int network[5][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}};
  bool odd = false;
  for (const auto& e : network) {
    if (cell[e[0]] > cell[e[1]]) {
      std::swap(cell[e[0]], cell[e[1]]);
      odd = !odd;
    }
  }
  if (odd) std::swap(cell[2], cell[3]);
  return cell;
}

bool exchange_patch(const ScalarBoxField& field, std::span<const BoxPoint> points,
                    std::span<const BoxCell> cells, const Flags& seeded, const Patch& patch,
                    std::vector<Row>& removed, std::vector<BoxCell>& added) {
  std::vector<BoxCell> before, after;
  before.reserve(patch.cells.size());
  for (Row row : patch.cells) before.push_back(cells[row]);
  FaceBank old_faces, new_faces;
  if (!face_bank(before, points, old_faces)) return false;
  {
    const auto mesh = patch_delaunay(patch, points);
    if (!mesh) return false;
    after.resize(static_cast<std::size_t>(mesh->cell_count()));
    for (std::size_t row = 0; row < after.size(); ++row) {
      for (int k = 0; k < 4; ++k) after[row][k] = patch.vertices[mesh->cells[4 * row + k]];
    }
  }
  if (!face_bank(after, points, new_faces)) return false;
  const auto old_group = chambers(old_faces, new_faces, before.size());
  const auto new_group = chambers(new_faces, old_faces, after.size());
  Flags old_selected(before.size(), 0), new_selected(after.size(), 0);
  for (std::size_t i = 0; i < before.size(); ++i) {
    if (seeded[patch.cells[i]]) old_selected[old_group[i]] = 1;
  }
  for (std::size_t i = 0; i < before.size(); ++i) old_selected[i] = old_selected[old_group[i]];

  auto next = new_faces.begin();
  for (const auto& face : old_faces) {
    while (next != new_faces.end() && next->key < face.key) ++next;
    if (next == new_faces.end() || next->key != face.key) continue;
    for (int side = 0; side < 2; ++side) {
      const Row a = face.side[side], b = next->side[side];
      if (a >= 0 && b >= 0 && old_selected[a]) new_selected[new_group[b]] = 1;
    }
  }
  for (std::size_t i = 0; i < after.size(); ++i) new_selected[i] = new_selected[new_group[i]];
  if (std::find(new_selected.begin(), new_selected.end(), 1) == new_selected.end()) return false;

  // The entire interface must agree as an oriented triangle chain. Each box
  // wall can instead use another triangulation of the same oriented rim: the
  // difference is a compact planar 2-chain with zero boundary.
  Boundary outgoing, incoming;
  if (!boundary_chain(field, points, old_faces, old_selected, outgoing) ||
      !boundary_chain(field, points, new_faces, new_selected, incoming) ||
      !(outgoing == incoming)) return false;

  before.clear();
  for (std::size_t i = 0; i < old_selected.size(); ++i) {
    if (old_selected[i]) before.push_back(canonical(cells[patch.cells[i]]));
  }
  std::size_t end = 0;
  for (std::size_t i = 0; i < after.size(); ++i) if (new_selected[i]) after[end++] = after[i];
  after.resize(end);
  if (before.size() == after.size()) {
    std::sort(before.begin(), before.end());
    if (before == after) return false;
  }
  for (std::size_t i = 0; i < old_selected.size(); ++i) {
    if (old_selected[i]) removed.push_back(patch.cells[i]);
  }
  added.insert(added.end(), after.begin(), after.end());
  return true;
}

}  // namespace

bool retile_scalar_defects(const ScalarBoxField& field, std::span<const BoxPoint> points,
                           std::vector<BoxCell>& cells, std::span<const std::int32_t> seed_cells) {
  if (points.size() < 4 || points.size() > kMaxScalarSites || cells.empty() ||
      cells.size() > kMaxScalarCells || seed_cells.empty() || seed_cells.size() > cells.size()) {
    return false;
  }
  const GradualNearestScope mode;
  if (!mode.established()) return false;
  for (const auto& p : points) {
    if (!geometry::point_in_domain(p)) return false;
    for (int i = 0; i < 3; ++i) if (!(p[i] >= field.low[i] && p[i] <= field.high[i])) return false;
  }
  Flags seeded(cells.size(), 0);
  std::vector<Patch> patches;
  if (!form_patches(cells, points.size(), seed_cells, seeded, patches)) return false;
  std::vector<Row> removed;
  std::vector<BoxCell> added;
  for (const auto& patch : patches) {
    if (!patch.oversized) exchange_patch(field, points, cells, seeded, patch, removed, added);
  }
  if (removed.empty()) return false;
  const std::size_t count = cells.size() - removed.size() + added.size();
  if (count > kMaxScalarCells) return false;
  std::sort(removed.begin(), removed.end());
  std::sort(added.begin(), added.end());
  std::vector<BoxCell> result;
  result.reserve(count);
  auto skip = removed.begin();
  for (std::size_t row = 0; row < cells.size(); ++row) {
    if (skip != removed.end() && *skip == static_cast<Row>(row)) ++skip;
    else result.push_back(canonical(cells[row]));
  }
  const auto retained = static_cast<std::ptrdiff_t>(result.size());
  result.insert(result.end(), added.begin(), added.end());
  if (std::is_sorted(result.begin(), result.begin() + retained)) {
    std::inplace_merge(result.begin(), result.begin() + retained, result.end());
  } else {
    std::sort(result.begin(), result.end());
  }
  if (std::adjacent_find(result.begin(), result.end()) != result.end()) return false;
  Flags touched(points.size(), 0);
  for (const auto& cell : result) {
    if (!positive(points, cell)) return false;
    for (Row v : cell) touched[v] = 1;
  }
  if (std::find(touched.begin(), touched.end(), 0) != touched.end()) return false;
  cells.swap(result);
  return true;
}

}  // namespace phx::mc::metric
