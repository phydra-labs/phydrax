//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact certificates of affine tetrahedral meshes, checked after construction
// on the stored binary64 coordinates.
//
// phx_mc_verify_tetrahedra (global embedding).  Checked exactly: every cell is
// positively oriented (orient3d); every face belongs to at most two cells, and
// two cells sharing a face induce opposite orientations on it; the boundary B
// (the faces of exactly one cell, outward) has every edge in exactly two of
// its triangles, traversed in opposite directions; two triangles of B meet
// only in shared vertices or a shared edge (exact contact classification);
// and the winding number W of B is 0 on the front and 1 on the back side of
// every shell (edge-connected component) of B.
//
// Why that suffices: the cells are an oriented chain whose boundary is B, so
// at every point p off the cell faces the number of (positive) cells covering
// p is the winding number W(p) of B.  B is an embedded surface; resolving its
// vertex pinches separates it into embedded closed manifolds without changing
// its complement, so each shell S has winding number s_S (its orientation,
// +1 outward) inside and 0 outside, and W is constant on each complement
// component of B.  Every bounded component touches the front or back side of
// some shell, where W is 0 or 1; hence W is 0 or 1 everywhere, every point is
// covered by at most one cell, and the cells tile the solid {W = 1} bounded by
// B without overlap or gap.  W on the front side of S is
//   sum over shells S' != S of w_S'(x_S)  -  [s_S < 0],
// with x_S a vertex of S on no other shell, w_S'(x_S) the exact winding number
// along a ray, and s_S the sign of the exact signed volume of S.  A boundary
// edge in more than two triangles, or a shell without a private vertex, is
// outside the theorem: reported unresolved, not violated.
//
// phx_mc_verify_volume (region coverage).  Checked exactly: positivity and
// face pairing as above; every boundary face is a declared surface triangle
// with the cell on the side its region names and the exterior on the other;
// every face between cells of different regions is a declared triangle with
// the two regions on its two sides; faces between cells of one region carry
// no declared triangle; every declared triangle is a face of the mesh.  With
// the mesh globally embedded (phx_mc_verify_tetrahedra) the cells of region r
// form a chain whose boundary is the declared oriented boundary of r, so they
// tile exactly the points where that boundary has winding number one.
//
// phx_mc_tetrahedra_validity: the exact sign of the affine Jacobian
// determinant d and of d^2 - floor^2 |e1|^2 |e2|^2 |e3|^2 (edge vectors e_k
// from the first vertex), with a rigorous enclosure of d.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <tuple>
#include <vector>

#include "capi_guard.hpp"
#include "expansion.hpp"
#include "filtered.hpp"
#include "floating_point_mode.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {

using phx::mc::Approx;
using phx::mc::Expansion;

// Outward faces of a positively oriented cell (v0, v1, v2, v3): the face
// opposite local vertex k, ordered so that (face, k) is an odd permutation.
constexpr std::array<std::array<int, 3>, 4> kOutwardFaces{
    {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}}};

constexpr int64_t kNone = -1;

struct FaceRecord {
  std::array<int32_t, 3> sorted;  // vertex indices in increasing order
  int64_t owner;                  // cell index, or -1 - surface triangle index
  int32_t local;                  // local face of the owning cell (0..3), or -1
  bool even;                      // the oriented triangle is an even permutation of `sorted`
};

FaceRecord make_record(int32_t a, int32_t b, int32_t c, int64_t owner, int32_t local) {
  std::array<int32_t, 3> v{a, b, c};
  bool even = true;
  if (v[0] > v[1]) {
    std::swap(v[0], v[1]);
    even = !even;
  }
  if (v[1] > v[2]) {
    std::swap(v[1], v[2]);
    even = !even;
  }
  if (v[0] > v[1]) {
    std::swap(v[0], v[1]);
    even = !even;
  }
  return {v, owner, local, even};
}

bool face_less(const FaceRecord& first, const FaceRecord& second) {
  if (first.sorted[1] != second.sorted[1]) return first.sorted[1] < second.sorted[1];
  if (first.sorted[2] != second.sorted[2]) return first.sorted[2] < second.sorted[2];
  return first.owner < second.owner;
}

// Face records of every cell (and surface triangle), grouped by vertex set:
// bucketed by the smallest vertex, each bucket ordered by the other two and
// then by owner, so surface records (negative owners) lead their group.
std::vector<FaceRecord> sorted_faces(int64_t point_count, int64_t cell_count,
                                     const int32_t* cells, int64_t surface_count,
                                     const int32_t* surface) {
  std::vector<FaceRecord> records;
  records.reserve(static_cast<std::size_t>(4 * cell_count + surface_count));
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    const int32_t* v = cells + 4 * cell;
    for (int32_t local = 0; local < 4; ++local) {
      const auto& face = kOutwardFaces[static_cast<std::size_t>(local)];
      records.push_back(make_record(v[face[0]], v[face[1]], v[face[2]], cell, local));
    }
  }
  for (int64_t triangle = 0; triangle < surface_count; ++triangle) {
    const int32_t* v = surface + 3 * triangle;
    records.push_back(make_record(v[0], v[1], v[2], -1 - triangle, -1));
  }
  std::vector<int64_t> offsets(static_cast<std::size_t>(point_count) + 1, 0);
  for (const FaceRecord& record : records) ++offsets[static_cast<std::size_t>(record.sorted[0]) + 1];
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  std::vector<FaceRecord> sorted(records.size());
  std::vector<int64_t> cursor(offsets.begin(), offsets.end() - 1);
  for (const FaceRecord& record : records) {
    sorted[static_cast<std::size_t>(cursor[static_cast<std::size_t>(record.sorted[0])]++)] = record;
  }
  for (std::size_t vertex = 0; vertex + 1 < offsets.size(); ++vertex) {
    std::sort(sorted.begin() + offsets[vertex], sorted.begin() + offsets[vertex + 1], face_less);
  }
  return sorted;
}

bool same_face(const FaceRecord& first, const FaceRecord& second) {
  return first.sorted == second.sorted;
}

void note(int64_t& count, int64_t& first, int64_t index) {
  if (count == 0 || index < first) first = index;
  ++count;
}

const double* point_at(const double* points, int32_t vertex) { return points + 3 * vertex; }

bool used_points_in_domain(int64_t point_count, const double* points, int64_t cell_count,
                           const int32_t* cells, int64_t& outside, int64_t& first_outside) {
  std::vector<uint8_t> used(static_cast<std::size_t>(point_count), 0);
  for (int64_t k = 0; k < 4 * cell_count; ++k) used[static_cast<std::size_t>(cells[k])] = 1;
  for (int64_t vertex = 0; vertex < point_count; ++vertex) {
    if (used[static_cast<std::size_t>(vertex)] == 0) continue;
    const double* x = points + 3 * vertex;
    if (!phx::mc::coordinate_in_domain(x[0]) || !phx::mc::coordinate_in_domain(x[1]) ||
        !phx::mc::coordinate_in_domain(x[2])) {
      note(outside, first_outside, vertex);
    }
  }
  return outside == 0;
}

bool indices_valid(int64_t point_count, int64_t count, int64_t width, const int32_t* rows) {
  for (int64_t k = 0; k < width * count; ++k) {
    if (rows[k] < 0 || rows[k] >= point_count) return false;
  }
  return true;
}

void orientation_findings(const double* points, int64_t cell_count, const int32_t* cells,
                          int64_t& count, int64_t& first) {
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    const int32_t* v = cells + 4 * cell;
    const int sign = phx::mc::orient3d(point_at(points, v[0]), point_at(points, v[1]),
                                       point_at(points, v[2]), point_at(points, v[3]));
    if (sign <= 0) note(count, first, cell);
  }
}

// ------------------------------------------------------------------ boundary shells

struct BoundaryFace {
  std::array<int32_t, 3> v;  // outward
  int64_t cell;
};

struct Box {
  double lower[3];
  double upper[3];
};

Box triangle_box(const double* points, const std::array<int32_t, 3>& v) {
  Box box{};
  for (int axis = 0; axis < 3; ++axis) {
    const double a = points[3 * v[0] + axis];
    const double b = points[3 * v[1] + axis];
    const double c = points[3 * v[2] + axis];
    box.lower[axis] = std::min({a, b, c});
    box.upper[axis] = std::max({a, b, c});
  }
  return box;
}

bool boxes_meet(const Box& a, const Box& b) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a.upper[axis] < b.lower[axis] || b.upper[axis] < a.lower[axis]) return false;
  }
  return true;
}

class ContactSearch {
 public:
  ContactSearch(const double* points, const std::vector<BoundaryFace>& faces)
      : points_(points), faces_(faces) {
    boxes_.reserve(faces.size());
    for (const BoundaryFace& face : faces) boxes_.push_back(triangle_box(points, face.v));
  }

  // Counts the pairs of triangles that meet in anything but shared vertices or
  // a shared edge; `first` and `second` receive the smallest such pair (by
  // owning cell, then face row).
  void run(int64_t& count, int64_t& first, int64_t& second) {
    count_ = 0;
    if (!grid()) sweep();
    count = count_;
    first = count_ == 0 ? kNone : faces_[best_.first].cell;
    second = count_ == 0 ? kNone : faces_[best_.second].cell;
  }

 private:
  bool separated(const std::array<int32_t, 3>& a, const std::array<int32_t, 3>& b) const {
    int shared = 0;
    for (int32_t vertex : a) shared += std::find(b.begin(), b.end(), vertex) != b.end() ? 1 : 0;
    const auto apart = [&](const std::array<int32_t, 3>& plane, const std::array<int32_t, 3>& other) {
      int side = 0;
      for (int32_t vertex : other) {
        if (std::find(plane.begin(), plane.end(), vertex) != plane.end()) continue;
        const int sign = phx::mc::orient3d(point_at(points_, plane[0]), point_at(points_, plane[1]),
                                           point_at(points_, plane[2]), point_at(points_, vertex));
        if (sign == 0 || (side != 0 && sign != side)) return false;
        side = sign;
      }
      return side != 0;
    };
    if (shared == 2) return apart(a, b);
    return shared < 2 && (apart(a, b) || apart(b, a));
  }

  void test(std::size_t i, std::size_t j) {
    if (!boxes_meet(boxes_[i], boxes_[j])) return;
    const auto& a = faces_[i].v;
    const auto& b = faces_[j].v;
    if (separated(a, b)) return;
    const double* first[3] = {point_at(points_, a[0]), point_at(points_, a[1]),
                              point_at(points_, a[2])};
    const double* second[3] = {point_at(points_, b[0]), point_at(points_, b[1]),
                               point_at(points_, b[2])};
    const int64_t first_ids[3] = {a[0], a[1], a[2]};
    const int64_t second_ids[3] = {b[0], b[1], b[2]};
    phx::mc::Contact contact;
    const int32_t status =
        phx::mc::intersect_triangles(first, second, first_ids, second_ids, contact);
    if (status == PHX_MC_OK &&
        (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
         contact.kind == PHX_MC_SHARED_EDGE)) {
      return;
    }
    std::pair<std::size_t, std::size_t> pair = std::minmax(i, j);
    const auto key = [&](const std::pair<std::size_t, std::size_t>& value) {
      return std::make_tuple(faces_[value.first].cell, faces_[value.second].cell, value.first,
                             value.second);
    };
    if (count_ == 0 || key(pair) < key(best_)) best_ = pair;
    ++count_;
  }

  // Every pair whose closed boxes overlap, once, through a uniform grid of
  // cells of the median triangle extent: both boxes meet the cell of the
  // larger lower corner on each axis (the cell index is monotone in the
  // coordinate), where the pair is tested.  Triangles spanning more than 64
  // cells are tested against every triangle instead.  False when the grid
  // does not apply (the caller sweeps instead).
  bool grid() {
    constexpr int64_t kAxisCells = int64_t{1} << 20;
    constexpr int64_t kLargeCells = 64;
    const std::size_t n = faces_.size();
    if (n < 64) return false;
    std::vector<double> extents;
    extents.reserve(n);
    double origin[3] = {boxes_[0].lower[0], boxes_[0].lower[1], boxes_[0].lower[2]};
    for (const Box& box : boxes_) {
      double extent = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        extent = std::max(extent, box.upper[axis] - box.lower[axis]);
        origin[axis] = std::min(origin[axis], box.lower[axis]);
      }
      extents.push_back(extent);
    }
    const auto middle = extents.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(extents.begin(), middle, extents.end());
    const double size = *middle;
    if (!(size > 0.0) || !std::isfinite(size)) return false;
    // The grid index of a coordinate, or kAxisCells when its quotient is not
    // below kAxisCells (or not finite): the quotient is range-checked as a
    // double, since in-domain extents make it reach about 2^293 and a cast
    // of an out-of-range value is undefined. Every value is at least the
    // origin, so the quotient is not negative.
    const auto cell = [&](double value, int axis) {
      const double quotient = std::floor((value - origin[axis]) / size);
      return quotient < static_cast<double>(kAxisCells) ? static_cast<int64_t>(quotient)
                                                         : kAxisCells;
    };
    std::vector<std::pair<uint64_t, uint32_t>> entries;
    std::vector<std::size_t> large;
    for (std::size_t row = 0; row < n; ++row) {
      int64_t low[3];
      int64_t high[3];
      int64_t cells = 1;
      for (int axis = 0; axis < 3; ++axis) {
        low[axis] = cell(boxes_[row].lower[axis], axis);
        high[axis] = cell(boxes_[row].upper[axis], axis);
        if (high[axis] >= kAxisCells) return false;
        cells *= high[axis] - low[axis] + 1;
      }
      if (cells > kLargeCells) {
        large.push_back(row);
        continue;
      }
      for (int64_t x = low[0]; x <= high[0]; ++x) {
        for (int64_t y = low[1]; y <= high[1]; ++y) {
          for (int64_t z = low[2]; z <= high[2]; ++z) {
            entries.push_back({static_cast<uint64_t>(x << 40 | y << 20 | z),
                               static_cast<uint32_t>(row)});
          }
        }
      }
    }
    if (large.size() > n / 8 + 64) return false;
    std::sort(entries.begin(), entries.end());
    for (std::size_t begin = 0; begin < entries.size();) {
      std::size_t end = begin + 1;
      while (end < entries.size() && entries[end].first == entries[begin].first) ++end;
      for (std::size_t a = begin; a < end; ++a) {
        for (std::size_t b = a + 1; b < end; ++b) {
          const Box& first = boxes_[entries[a].second];
          const Box& second = boxes_[entries[b].second];
          uint64_t key = 0;
          for (int axis = 0; axis < 3; ++axis) {
            const double corner = std::max(first.lower[axis], second.lower[axis]);
            key = key << 20 | static_cast<uint64_t>(cell(corner, axis));
          }
          if (key == entries[begin].first) test(entries[a].second, entries[b].second);
        }
      }
      begin = end;
    }
    std::vector<uint8_t> is_large(n, 0);
    for (std::size_t row : large) is_large[row] = 1;
    for (std::size_t row : large) {
      for (std::size_t other = 0; other < n; ++other) {
        if (other == row || (is_large[other] != 0 && other < row)) continue;
        test(row, other);
      }
    }
    return true;
  }

  // Sort-and-sweep along the first axis.
  void sweep() {
    std::vector<std::size_t> order(faces_.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return boxes_[a].lower[0] < boxes_[b].lower[0] ||
             (boxes_[a].lower[0] == boxes_[b].lower[0] && a < b);
    });
    for (std::size_t k = 0; k < order.size(); ++k) {
      for (std::size_t l = k + 1; l < order.size(); ++l) {
        if (boxes_[order[l]].lower[0] > boxes_[order[k]].upper[0]) break;
        test(order[k], order[l]);
      }
    }
  }

  const double* points_;
  const std::vector<BoundaryFace>& faces_;
  std::vector<Box> boxes_;
  int64_t count_ = 0;
  std::pair<std::size_t, std::size_t> best_{0, 0};
};

std::size_t find_root(std::vector<std::size_t>& parent, std::size_t row) {
  while (parent[row] != row) {
    parent[row] = parent[parent[row]];
    row = parent[row];
  }
  return row;
}

// Sign of the perturbed orient2d(u, v, x + (eps, eps^2)) on the first two
// coordinates; u and v differ in that projection.
int perturbed_side(const double* u, const double* v, const double* x) {
  const int sign = phx::mc::orient2d(u, v, x);
  if (sign != 0) return sign;
  if (v[1] != u[1]) return v[1] > u[1] ? -1 : 1;
  return v[0] > u[0] ? 1 : -1;
}

// Exact winding number at x of the closed triangle chain `rows` of `faces`,
// counted along the ray from x + (eps, eps^2, 0) in direction +z.  False
// when x lies on one of the triangles.
bool winding_number(const double* points, const std::vector<BoundaryFace>& faces,
                    const std::vector<std::size_t>& rows, const double* x, int64_t& value) {
  value = 0;
  for (std::size_t row : rows) {
    const auto& v = faces[row].v;
    const double* a = point_at(points, v[0]);
    const double* b = point_at(points, v[1]);
    const double* c = point_at(points, v[2]);
    if (std::max({a[2], b[2], c[2]}) < x[2] || std::min({a[0], b[0], c[0]}) > x[0] ||
        std::max({a[0], b[0], c[0]}) < x[0] || std::min({a[1], b[1], c[1]}) > x[1] ||
        std::max({a[1], b[1], c[1]}) < x[1]) {
      continue;
    }
    const int projected = phx::mc::orient2d(a, b, c);
    if (projected == 0) continue;
    if (perturbed_side(a, b, x) != projected || perturbed_side(b, c, x) != projected ||
        perturbed_side(c, a, x) != projected) {
      continue;
    }
    const int side = phx::mc::orient3d(a, b, c, x);
    if (side == 0) return false;
    // The normal (b - a) x (c - a) has the sign `projected` along the ray, so
    // the crossing lies ahead of x exactly when x is behind the plane.
    if (side * projected < 0) value += projected;
  }
  return true;
}

// Sign of the exact signed volume enclosed by the shell `rows`: -sign of the
// sum of orient3d(t, o) over its triangles (outward shells are positive).
int shell_orientation(const double* points, const std::vector<BoundaryFace>& faces,
                      const std::vector<std::size_t>& rows, const double* o) {
  Approx total = Approx::exact(0.0);
  for (std::size_t row : rows) {
    const auto& v = faces[row].v;
    total = total + phx::mc::orient3d_approx(point_at(points, v[0]), point_at(points, v[1]),
                                             point_at(points, v[2]), o);
  }
  int sign = total.certified_sign();
  if (sign == 2) {
    Expansion exact;
    for (std::size_t row : rows) {
      const auto& v = faces[row].v;
      exact = exact + phx::mc::orient3d_exact(point_at(points, v[0]), point_at(points, v[1]),
                                              point_at(points, v[2]), o);
    }
    sign = exact.sign();
  }
  return -sign;
}

void boundary_findings(const double* points, int64_t point_count,
                       const std::vector<BoundaryFace>& faces, phx_mc_tetrahedra_verdict& verdict) {
  // Edge pairing and shells.
  struct EdgeUse {
    uint64_t key;
    bool forward;
    std::size_t face;
  };
  std::vector<EdgeUse> uses;
  uses.reserve(3 * faces.size());
  for (std::size_t row = 0; row < faces.size(); ++row) {
    const auto& v = faces[row].v;
    for (int k = 0; k < 3; ++k) {
      const auto a = static_cast<uint32_t>(v[static_cast<std::size_t>(k)]);
      const auto b = static_cast<uint32_t>(v[static_cast<std::size_t>((k + 1) % 3)]);
      uses.push_back({static_cast<uint64_t>(std::min(a, b)) << 32 | std::max(a, b), a < b, row});
    }
  }
  std::sort(uses.begin(), uses.end(), [](const EdgeUse& first, const EdgeUse& second) {
    return first.key < second.key || (first.key == second.key && first.face < second.face);
  });
  std::vector<std::size_t> parent(faces.size());
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  for (std::size_t begin = 0; begin < uses.size();) {
    std::size_t end = begin + 1;
    while (end < uses.size() && uses[end].key == uses[begin].key) ++end;
    if (end - begin == 2 && uses[begin].forward != uses[begin + 1].forward) {
      const std::size_t a = find_root(parent, uses[begin].face);
      const std::size_t b = find_root(parent, uses[begin + 1].face);
      if (a != b) parent[std::max(a, b)] = std::min(a, b);
    } else {
      for (std::size_t k = begin; k < end; ++k) {
        note(verdict.nonmanifold_boundary_edges, verdict.first_nonmanifold_boundary_cell,
             faces[uses[k].face].cell);
      }
    }
    begin = end;
  }
  std::vector<std::size_t> shell_of(faces.size());
  std::vector<std::vector<std::size_t>> shells;
  std::vector<std::size_t> shell_index(faces.size(), faces.size());
  for (std::size_t row = 0; row < faces.size(); ++row) {
    const std::size_t root = find_root(parent, row);
    if (shell_index[root] == faces.size()) {
      shell_index[root] = shells.size();
      shells.emplace_back();
    }
    shell_of[row] = shell_index[root];
    shells[shell_index[root]].push_back(row);
  }
  verdict.boundary_shells = static_cast<int64_t>(shells.size());

  ContactSearch(points, faces).run(verdict.boundary_contacts, verdict.first_contact_cell,
                                   verdict.second_contact_cell);
  if (verdict.nonmanifold_boundary_edges != 0 || verdict.boundary_contacts != 0 ||
      shells.size() < 2) {
    return;
  }
  // A private vertex of each shell: the smallest vertex on no other shell.
  constexpr int64_t kUnseen = -1;
  constexpr int64_t kShared = -2;
  std::vector<int64_t> vertex_shell(static_cast<std::size_t>(point_count), kUnseen);
  for (std::size_t row = 0; row < faces.size(); ++row) {
    for (int32_t vertex : faces[row].v) {
      int64_t& owner = vertex_shell[static_cast<std::size_t>(vertex)];
      const auto shell = static_cast<int64_t>(shell_of[row]);
      owner = owner == kUnseen ? shell : (owner == shell ? shell : kShared);
    }
  }
  std::vector<int32_t> private_vertex(shells.size(), -1);
  for (int64_t vertex = 0; vertex < point_count; ++vertex) {
    const int64_t shell = vertex_shell[static_cast<std::size_t>(vertex)];
    if (shell >= 0 && private_vertex[static_cast<std::size_t>(shell)] < 0) {
      private_vertex[static_cast<std::size_t>(shell)] = static_cast<int32_t>(vertex);
    }
  }
  for (std::size_t shell = 0; shell < shells.size(); ++shell) {
    const int64_t cell = faces[shells[shell].front()].cell;
    if (private_vertex[shell] < 0) {
      note(verdict.unresolved_shells, verdict.first_unresolved_shell_cell, cell);
      continue;
    }
    const double* x = point_at(points, private_vertex[shell]);
    int64_t front = shell_orientation(points, faces, shells[shell], x) < 0 ? -1 : 0;
    bool decided = true;
    for (std::size_t other = 0; other < shells.size() && decided; ++other) {
      if (other == shell) continue;
      int64_t winding = 0;
      decided = winding_number(points, faces, shells[other], x, winding);
      front += winding;
    }
    if (!decided) {
      note(verdict.unresolved_shells, verdict.first_unresolved_shell_cell, cell);
    } else if (front != 0) {
      note(verdict.misnested_shells, verdict.first_misnested_shell_cell, cell);
    }
  }
}

// ------------------------------------------------------------------ validity

template <class T, class Difference>
void validity_values(const double* a, const double* b, const double* c, const double* d,
                     double floor, Difference difference, T& determinant, T& margin) {
  const double* p[3] = {b, c, d};
  T e[3][3];
  for (int k = 0; k < 3; ++k) {
    for (int axis = 0; axis < 3; ++axis) e[k][axis] = difference(p[k][axis], a[axis]);
  }
  determinant = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
  T scale = difference(floor, 0.0) * difference(floor, 0.0);
  for (int k = 0; k < 3; ++k) {
    scale = scale * (e[k][0] * e[k][0] + e[k][1] * e[k][1] + e[k][2] * e[k][2]);
  }
  margin = determinant * determinant - scale;
}

// The largest binary64 value at most (upper: the smallest at least) `value`.
double directed(const Expansion& value, bool upper) {
  double x = value.estimate();
  const auto below = [&](double y) { return (value - Expansion(y)).sign(); };
  if (upper) {
    while (below(x) > 0) x = std::nextafter(x, std::numeric_limits<double>::infinity());
    while (below(std::nextafter(x, -std::numeric_limits<double>::infinity())) <= 0) {
      x = std::nextafter(x, -std::numeric_limits<double>::infinity());
    }
  } else {
    while (below(x) < 0) x = std::nextafter(x, -std::numeric_limits<double>::infinity());
    while (below(std::nextafter(x, std::numeric_limits<double>::infinity())) >= 0) {
      x = std::nextafter(x, std::numeric_limits<double>::infinity());
    }
  }
  return x;
}

}  // namespace

extern "C" int32_t phx_mc_verify_tetrahedra(int64_t point_count, const double* points,
                                            int64_t cell_count, const int32_t* cells,
                                            phx_mc_tetrahedra_verdict* verdict) {
  return phx::mc::guarded([&]() -> int32_t {
    if (verdict == nullptr || point_count < 0 || cell_count < 0 ||
        (point_count > 0 && points == nullptr) || (cell_count > 0 && cells == nullptr) ||
        point_count > std::numeric_limits<int32_t>::max() ||
        !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(cell_count, 4, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!indices_valid(point_count, cell_count, 4, cells)) return PHX_MC_INVALID_INPUT;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    phx_mc_tetrahedra_verdict result{};
    result.first_outside_domain_point = kNone;
    result.first_nonpositive_cell = kNone;
    result.first_nonmanifold_cell = kNone;
    result.first_nonmanifold_boundary_cell = kNone;
    result.first_contact_cell = kNone;
    result.second_contact_cell = kNone;
    result.first_misnested_shell_cell = kNone;
    result.first_unresolved_shell_cell = kNone;
    if (used_points_in_domain(point_count, points, cell_count, cells, result.outside_domain_points,
                              result.first_outside_domain_point)) {
      orientation_findings(points, cell_count, cells, result.nonpositive_cells,
                           result.first_nonpositive_cell);
      const std::vector<FaceRecord> records =
          sorted_faces(point_count, cell_count, cells, 0, nullptr);
      std::vector<BoundaryFace> boundary;
      for (std::size_t begin = 0; begin < records.size();) {
        std::size_t end = begin + 1;
        while (end < records.size() && same_face(records[begin], records[end])) ++end;
        if (end - begin == 1) {
          const FaceRecord& record = records[begin];
          const int32_t* v = cells + 4 * record.owner;
          const auto& face = kOutwardFaces[static_cast<std::size_t>(record.local)];
          boundary.push_back({{v[face[0]], v[face[1]], v[face[2]]}, record.owner});
        } else if (end - begin > 2 || records[begin].even == records[begin + 1].even) {
          for (std::size_t k = begin; k < end; ++k) {
            note(result.nonmanifold_faces, result.first_nonmanifold_cell, records[k].owner);
          }
        }
        begin = end;
      }
      result.boundary_faces = static_cast<int64_t>(boundary.size());
      if (result.nonpositive_cells == 0) boundary_findings(points, point_count, boundary, result);
    }
    const bool violated = result.nonpositive_cells != 0 || result.nonmanifold_faces != 0 ||
                          result.boundary_contacts != 0 || result.misnested_shells != 0;
    const bool unresolved = result.outside_domain_points != 0 ||
                            result.nonmanifold_boundary_edges != 0 ||
                            result.unresolved_shells != 0;
    result.status = violated ? PHX_MC_VERDICT_VIOLATED
                             : (unresolved ? PHX_MC_VERDICT_UNRESOLVED : PHX_MC_VERDICT_CERTIFIED);
    *verdict = result;
    return PHX_MC_OK;
  });
}

extern "C" int32_t phx_mc_verify_volume(int64_t point_count, const double* points,
                                        int64_t cell_count, const int32_t* cells,
                                        const int32_t* cell_regions, int64_t surface_count,
                                        const int32_t* surface, const int32_t* surface_regions,
                                        phx_mc_volume_verdict* verdict) {
  return phx::mc::guarded([&]() -> int32_t {
    if (verdict == nullptr || point_count < 0 || cell_count < 0 || surface_count < 0 ||
        (point_count > 0 && points == nullptr) || (cell_count > 0 && cells == nullptr) ||
        (surface_count > 0 && surface == nullptr) ||
        point_count > std::numeric_limits<int32_t>::max() ||
        !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(cell_count, 4, sizeof(int32_t)) ||
        !phx::mc::addressable(surface_count, 3, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!indices_valid(point_count, cell_count, 4, cells) ||
        !indices_valid(point_count, surface_count, 3, surface)) {
      return PHX_MC_INVALID_INPUT;
    }
    const auto region_of = [&](int64_t cell) {
      return cell_regions == nullptr ? 0 : cell_regions[cell];
    };
    // Declared regions on the two sides: the normal of the oriented triangle
    // points from the first into the second; -1 is the exterior.
    const auto sides = [&](int64_t triangle) {
      return surface_regions == nullptr
                 ? std::array<int32_t, 2>{0, -1}
                 : std::array<int32_t, 2>{surface_regions[2 * triangle],
                                          surface_regions[2 * triangle + 1]};
    };
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      if (region_of(cell) < 0) return PHX_MC_INVALID_INPUT;
    }
    for (int64_t triangle = 0; triangle < surface_count; ++triangle) {
      const auto pair = sides(triangle);
      if (pair[0] < -1 || pair[1] < -1 || pair[0] == pair[1]) return PHX_MC_INVALID_INPUT;
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    phx_mc_volume_verdict result{};
    result.first_outside_domain_point = kNone;
    result.first_nonpositive_cell = kNone;
    result.first_nonmanifold_cell = kNone;
    result.first_unmatched_cell = kNone;
    result.first_unmatched_surface_triangle = kNone;
    result.first_region_mismatch_cell = kNone;
    if (used_points_in_domain(point_count, points, cell_count, cells, result.outside_domain_points,
                              result.first_outside_domain_point)) {
      orientation_findings(points, cell_count, cells, result.nonpositive_cells,
                           result.first_nonpositive_cell);
    }
    const std::vector<FaceRecord> records =
        sorted_faces(point_count, cell_count, cells, surface_count, surface);
    for (std::size_t begin = 0; begin < records.size();) {
      std::size_t end = begin + 1;
      while (end < records.size() && same_face(records[begin], records[end])) ++end;
      // Within a group, surface records (owner < 0) precede cell records.
      std::size_t first_cell = begin;
      while (first_cell < end && records[first_cell].owner < 0) ++first_cell;
      const std::size_t surface_records = first_cell - begin;
      const std::size_t cell_records = end - first_cell;
      for (std::size_t k = begin + 1; k < first_cell; ++k) {
        note(result.unmatched_surface_triangles, result.first_unmatched_surface_triangle,
             -1 - records[k].owner);
      }
      const FaceRecord* declared = surface_records > 0 ? &records[begin] : nullptr;
      // The region a cell record's cell must carry for the declared triangle.
      const auto expected = [&](const FaceRecord& cell) {
        const auto pair = sides(-1 - declared->owner);
        return cell.even == declared->even ? pair[0] : pair[1];
      };
      if (cell_records > 2 ||
          (cell_records == 2 && records[first_cell].even == records[first_cell + 1].even)) {
        for (std::size_t k = first_cell; k < end; ++k) {
          note(result.nonmanifold_faces, result.first_nonmanifold_cell, records[k].owner);
        }
      } else if (cell_records == 2) {
        const FaceRecord& a = records[first_cell];
        const FaceRecord& b = records[first_cell + 1];
        if (declared == nullptr) {
          if (region_of(a.owner) != region_of(b.owner)) {
            note(result.region_mismatches, result.first_region_mismatch_cell,
                 std::min(a.owner, b.owner));
          }
        } else if (expected(a) != region_of(a.owner) || expected(b) != region_of(b.owner)) {
          note(result.region_mismatches, result.first_region_mismatch_cell,
               std::min(a.owner, b.owner));
        }
      } else if (cell_records == 1) {
        ++result.boundary_faces;
        const FaceRecord& a = records[first_cell];
        if (declared == nullptr) {
          note(result.unmatched_boundary_faces, result.first_unmatched_cell, a.owner);
        } else {
          const auto pair = sides(-1 - declared->owner);
          const bool exterior = (a.even == declared->even ? pair[1] : pair[0]) == -1;
          if (!exterior) {
            note(result.unmatched_boundary_faces, result.first_unmatched_cell, a.owner);
            note(result.unmatched_surface_triangles, result.first_unmatched_surface_triangle,
                 -1 - declared->owner);
          } else if (expected(a) != region_of(a.owner)) {
            note(result.region_mismatches, result.first_region_mismatch_cell, a.owner);
          }
        }
      } else if (declared != nullptr) {
        note(result.unmatched_surface_triangles, result.first_unmatched_surface_triangle,
             -1 - declared->owner);
      }
      begin = end;
    }
    result.certified = result.outside_domain_points == 0 && result.nonpositive_cells == 0 &&
                               result.nonmanifold_faces == 0 &&
                               result.unmatched_boundary_faces == 0 &&
                               result.unmatched_surface_triangles == 0 &&
                               result.region_mismatches == 0
                           ? 1
                           : 0;
    *verdict = result;
    return PHX_MC_OK;
  });
}

extern "C" int32_t phx_mc_tetrahedra_validity(int64_t point_count, const double* points,
                                              int64_t cell_count, const int32_t* cells,
                                              double floor, int8_t* valid, double* lower,
                                              double* upper) {
  return phx::mc::guarded([&]() -> int32_t {
    // The margin has degree eight in coordinate differences and the floor;
    // magnitudes in [2^-60, 2^60] keep every exact product free of overflow
    // and underflow.
    const auto validity_domain = [](double x) {
      const double magnitude = std::fabs(x);
      return magnitude == 0.0 || (magnitude >= 0x1p-60 && magnitude <= 0x1p60);
    };
    if (point_count < 0 || cell_count < 0 || (point_count > 0 && points == nullptr) ||
        (cell_count > 0 && (cells == nullptr || valid == nullptr || lower == nullptr ||
                            upper == nullptr)) ||
        !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(cell_count, 4, sizeof(int32_t)) || !(floor >= 0.0) ||
        !(floor < 1.0) || !validity_domain(floor)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!indices_valid(point_count, cell_count, 4, cells)) return PHX_MC_INVALID_INPUT;
    for (int64_t k = 0; k < 4 * cell_count; ++k) {
      const double* x = point_at(points, cells[k]);
      if (!validity_domain(x[0]) || !validity_domain(x[1]) || !validity_domain(x[2])) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const auto approx_difference = [](double x, double y) {
      return Approx::exact(x) - Approx::exact(y);
    };
    const auto exact_difference = [](double x, double y) { return Expansion::difference(x, y); };
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      const int32_t* v = cells + 4 * cell;
      const double* a = point_at(points, v[0]);
      const double* b = point_at(points, v[1]);
      const double* c = point_at(points, v[2]);
      const double* d = point_at(points, v[3]);
      Approx determinant;
      Approx margin;
      validity_values(a, b, c, d, floor, approx_difference, determinant, margin);
      int determinant_sign = determinant.certified_sign();
      int margin_sign = margin.certified_sign();
      if (determinant_sign != 2 && margin_sign != 2) {
        lower[cell] = determinant.bound == 0.0
                          ? determinant.value
                          : std::nextafter(determinant.value - determinant.bound, -kInfinity);
        upper[cell] = determinant.bound == 0.0
                          ? determinant.value
                          : std::nextafter(determinant.value + determinant.bound, kInfinity);
      } else {
        Expansion exact_determinant;
        Expansion exact_margin;
        validity_values(a, b, c, d, floor, exact_difference, exact_determinant, exact_margin);
        determinant_sign = exact_determinant.sign();
        margin_sign = exact_margin.sign();
        lower[cell] = directed(exact_determinant, false);
        upper[cell] = directed(exact_determinant, true);
      }
      valid[cell] = determinant_sign > 0 && margin_sign > 0 ? 1 : 0;
    }
    return PHX_MC_OK;
  });
}
