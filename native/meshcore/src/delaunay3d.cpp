//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Incremental 3D Delaunay and regular triangulations (Bowyer-Watson).
//
// The triangulation of the convex hull is closed by ghost tetrahedra: every
// hull facet carries one tetrahedron whose remaining vertex is kInfinite.
// Every tetrahedron (v0, v1, v2, v3) is positively oriented: orient3d > 0 for
// finite ones, and for ghosts substituting a point strictly beyond the hull
// facet for the infinite vertex gives orient3d > 0.  n[k] is the neighbor
// across the facet opposite v[k]; the structure is always a closed
// pseudomanifold (every facet shared by exactly two tetrahedra).
//
// Conflicts are exact and index-ordered: a finite tetrahedron conflicts with p
// iff insphere_sos (power3d_sos for weighted points) > 0.  A ghost conflicts
// iff p lies strictly beyond its hull facet; when p is coplanar with the facet
// it conflicts iff its finite neighbor does (the neighbor's circumsphere, or
// orthosphere, restricted to the facet plane is the facet's circumcircle, so
// this is the limit of the perturbed predicate at the hull).  The triangulation
// is the regular triangulation of the symbolically perturbed lifting, hence
// the conflict region of p is connected and star-shaped from p, and the
// visibility walk is acyclic.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {
namespace {

constexpr int32_t kInfinite = -1;
constexpr int32_t kDead = -2;
constexpr std::size_t kMaxTetSlots = static_cast<std::size_t>(std::numeric_limits<int32_t>::max());

// Per-insertion conflict cache states, valid while Tet::stamp equals the
// current insertion stamp.
enum : std::uint8_t { kNoConflict = 0, kConflict = 1, kInCavity = 2 };

struct Tet {
  int32_t v[4];
  int32_t n[4];
  std::uint32_t stamp;
  std::uint8_t state;
};

bool is_ghost(const Tet& tet) {
  return tet.v[0] == kInfinite || tet.v[1] == kInfinite || tet.v[2] == kInfinite ||
         tet.v[3] == kInfinite;
}

int vertex_slot(const Tet& tet, int32_t vertex) {
  for (int k = 0; k < 4; ++k) {
    if (tet.v[k] == vertex) {
      return k;
    }
  }
  return -1;
}

int neighbor_slot(const Tet& tet, int32_t neighbor) {
  for (int k = 0; k < 4; ++k) {
    if (tet.n[k] == neighbor) {
      return k;
    }
  }
  return -1;
}

// Unordered vertex pair key; the infinite vertex maps to 0.
std::uint64_t edge_key(int32_t a, int32_t b) {
  const int32_t low = std::min(a, b);
  const int32_t high = std::max(a, b);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low + 1)) << 32) |
         static_cast<std::uint64_t>(static_cast<std::uint32_t>(high + 1));
}

// (b - a) x (c - a) vanishes iff the orientations of the three coordinate-plane
// projections vanish; each is an exact orient2d.
bool collinear(const double* a, const double* b, const double* c) {
  for (int axis = 0; axis < 3; ++axis) {
    const int i = (axis + 1) % 3;
    const int j = (axis + 2) % 3;
    const double pa[2] = {a[i], a[j]};
    const double pb[2] = {b[i], b[j]};
    const double pc[2] = {c[i], c[j]};
    if (orient2d(pa, pb, pc) != 0) {
      return false;
    }
  }
  return true;
}

class Triangulation3D {
 public:
  Triangulation3D(const double* points, const double* weights, int64_t point_count,
                  int64_t max_tetrahedra)
      : points_(points),
        weights_(weights),
        max_tetrahedra_(max_tetrahedra),
        vertex_stamp_(static_cast<std::size_t>(point_count), 0U) {}

  // Inserts `order` (distinct points) incrementally.  alive[v] is cleared for
  // redundant or removed weighted vertices.
  int32_t build(const std::vector<int32_t>& order, std::vector<char>& alive) {
    const std::size_t count = order.size();
    if (count < 4) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    const int32_t a = order[0];
    int32_t b = order[1];
    std::size_t third = 2;
    while (third < count && collinear(point(a), point(b), point(order[third]))) {
      ++third;
    }
    if (third == count) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    int32_t c = order[third];
    std::size_t fourth = third + 1;
    int sign = 0;
    while (fourth < count &&
           (sign = orient3d(point(a), point(b), point(c), point(order[fourth]))) == 0) {
      ++fourth;
    }
    if (fourth == count) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    const int32_t d = order[fourth];
    if (sign < 0) {
      std::swap(b, c);
    }
    tets_.reserve(7 * count + 16);
    int32_t status = initialize(a, b, c, d);
    for (std::size_t k = 2; k < count && status == PHX_MC_OK; ++k) {
      if (k != third && k != fourth) {
        status = insert(order[k], alive);
      }
    }
    return status;
  }

  void collect_cells(std::vector<int32_t>& cells) const {
    cells.clear();
    cells.reserve(static_cast<std::size_t>(finite_count_) * 4);
    for (const Tet& tet : tets_) {
      if (tet.v[0] != kDead && !is_ghost(tet)) {
        cells.insert(cells.end(), tet.v, tet.v + 4);
      }
    }
  }

 private:
  struct BoundaryFace {
    int32_t tet;       // cavity tetrahedron
    int32_t slot;      // facet opposite tets_[tet].v[slot]
    int32_t outside;   // non-conflicting neighbor across the facet
    int32_t back;      // slot of `tet` in tets_[outside].n
  };

  // Open-addressing slot of the facet table; valid while stamp == link_stamp_.
  struct LinkSlot {
    std::uint64_t key;
    int32_t tet;  // -1 once both occurrences are linked
    int32_t slot;
    std::uint32_t stamp;
  };

  const double* point(int32_t vertex) const {
    return points_ + 3 * static_cast<int64_t>(vertex);
  }

  // orient3d of `tet` with v[slot] replaced by p.
  int orient_with(const Tet& tet, int slot, int32_t p) const {
    const double* q[4];
    for (int k = 0; k < 4; ++k) {
      q[k] = point(k == slot ? p : tet.v[k]);
    }
    return orient3d(q[0], q[1], q[2], q[3]);
  }

  bool finite_conflict(const Tet& tet, int32_t p) const {
    const int32_t* v = tet.v;
    if (weights_ == nullptr) {
      return insphere_sos(point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p), v[0],
                          v[1], v[2], v[3], p) > 0;
    }
    return power3d_sos(point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p),
                       weights_[v[0]], weights_[v[1]], weights_[v[2]], weights_[v[3]],
                       weights_[p], v[0], v[1], v[2], v[3], p) > 0;
  }

  bool conflict(int32_t t, int32_t p) {
    Tet& tet = tets_[static_cast<std::size_t>(t)];
    if (tet.stamp == stamp_value_) {
      return tet.state != kNoConflict;
    }
    const int slot = vertex_slot(tet, kInfinite);
    bool result = false;
    if (slot < 0) {
      result = finite_conflict(tet, p);
    } else {
      const int side = orient_with(tet, slot, p);
      result = side > 0 || (side == 0 && conflict(tet.n[slot], p));
    }
    tet.stamp = stamp_value_;
    tet.state = result ? kConflict : kNoConflict;
    return result;
  }

  // Visibility walk from the hint: returns a finite tetrahedron containing p
  // (closed) or a ghost whose hull facet has p strictly beyond it.
  int32_t locate(int32_t p) {
    int32_t t = hint_;
    {
      const Tet& start = tets_[static_cast<std::size_t>(t)];
      const int slot = vertex_slot(start, kInfinite);
      if (slot >= 0) {
        t = start.n[slot];
      }
    }
    int32_t previous = -1;
    for (;;) {
      const Tet& tet = tets_[static_cast<std::size_t>(t)];
      const int first = static_cast<int>(splitmix64(walk_step_++) & 3U);
      int32_t next = -1;
      for (int i = 0; i < 4; ++i) {
        const int k = (first + i) & 3;
        // p is strictly on this side of the facet just crossed.
        if (tet.n[k] == previous) {
          continue;
        }
        if (orient_with(tet, k, p) < 0) {
          next = tet.n[k];
          break;
        }
      }
      if (next < 0) {
        return t;
      }
      previous = t;
      t = next;
      if (is_ghost(tets_[static_cast<std::size_t>(t)])) {
        return t;
      }
    }
  }

  int32_t allocate() {
    if (!free_.empty()) {
      const int32_t id = free_.back();
      free_.pop_back();
      return id;
    }
    if (tets_.size() >= kMaxTetSlots) {
      return -1;
    }
    tets_.push_back(Tet{});
    return static_cast<int32_t>(tets_.size() - 1);
  }

  // Links the facets through `center` of the tetrahedra `ids` (a closed fan
  // around center): each such facet is keyed by its two other vertices and
  // must occur exactly twice.
  bool link_star(const std::vector<int32_t>& ids, int32_t center) {
    std::size_t capacity = 64;
    while (capacity < 6 * ids.size()) {
      capacity <<= 1;
    }
    if (link_table_.size() < capacity) {
      link_table_.assign(capacity, LinkSlot{0, 0, 0, 0U});
      link_stamp_ = 0;
    }
    ++link_stamp_;
    const std::size_t mask = link_table_.size() - 1;
    std::size_t pending = 0;
    for (int32_t id : ids) {
      Tet& tet = tets_[static_cast<std::size_t>(id)];
      const int apex = vertex_slot(tet, center);
      for (int s = 0; s < 4; ++s) {
        if (s == apex) {
          continue;
        }
        int32_t pair[2];
        int count = 0;
        for (int r = 0; r < 4; ++r) {
          if (r != s && r != apex) {
            pair[count++] = tet.v[r];
          }
        }
        const std::uint64_t key = edge_key(pair[0], pair[1]);
        std::size_t h = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        for (;;) {
          LinkSlot& entry = link_table_[h];
          if (entry.stamp != link_stamp_) {
            entry = LinkSlot{key, id, s, link_stamp_};
            ++pending;
            break;
          }
          if (entry.key == key) {
            if (entry.tet < 0) {
              return false;
            }
            tet.n[s] = entry.tet;
            tets_[static_cast<std::size_t>(entry.tet)].n[entry.slot] = id;
            entry.tet = -1;
            --pending;
            break;
          }
          h = (h + 1) & mask;
        }
      }
    }
    return pending == 0;
  }

  int32_t initialize(int32_t a, int32_t b, int32_t c, int32_t d) {
    if (max_tetrahedra_ < 1) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    const int32_t root = allocate();
    tets_[static_cast<std::size_t>(root)] = Tet{{a, b, c, d}, {-1, -1, -1, -1}, 0U, kNoConflict};
    std::vector<int32_t> ghosts;
    for (int k = 0; k < 4; ++k) {
      const int32_t ghost = allocate();
      Tet tet = tets_[static_cast<std::size_t>(root)];
      tet.v[k] = kInfinite;
      // Odd permutation of the facet: the outside becomes the positive side.
      const int first = k == 0 ? 1 : 0;
      const int second = k <= 1 ? 2 : 1;
      std::swap(tet.v[first], tet.v[second]);
      tet.n[0] = tet.n[1] = tet.n[2] = tet.n[3] = -1;
      tet.n[k] = root;
      tets_[static_cast<std::size_t>(ghost)] = tet;
      tets_[static_cast<std::size_t>(root)].n[k] = ghost;
      ghosts.push_back(ghost);
    }
    if (!link_star(ghosts, kInfinite)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    finite_count_ = 1;
    hint_ = root;
    return PHX_MC_OK;
  }

  int32_t insert(int32_t p, std::vector<char>& alive) {
    ++stamp_value_;
    const int32_t origin = locate(p);
    if (!conflict(origin, p)) {
      // Only a weighted point lying above the lower hull can be conflict-free.
      if (weights_ == nullptr) {
        return PHX_MC_INTERNAL_ERROR;
      }
      alive[static_cast<std::size_t>(p)] = 0;
      hint_ = origin;
      return PHX_MC_OK;
    }

    cavity_.clear();
    boundary_.clear();
    stack_.clear();
    tets_[static_cast<std::size_t>(origin)].state = kInCavity;
    cavity_.push_back(origin);
    stack_.push_back(origin);
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      for (int k = 0; k < 4; ++k) {
        const int32_t neighbor = tets_[static_cast<std::size_t>(t)].n[k];
        if (tets_[static_cast<std::size_t>(neighbor)].stamp == stamp_value_ &&
            tets_[static_cast<std::size_t>(neighbor)].state == kInCavity) {
          continue;
        }
        if (conflict(neighbor, p)) {
          tets_[static_cast<std::size_t>(neighbor)].state = kInCavity;
          cavity_.push_back(neighbor);
          stack_.push_back(neighbor);
        } else {
          const int back = neighbor_slot(tets_[static_cast<std::size_t>(neighbor)], t);
          if (back < 0) {
            return PHX_MC_INTERNAL_ERROR;
          }
          boundary_.push_back({t, k, neighbor, back});
        }
      }
    }

    // Vertices of the cavity absent from its boundary lose their whole star.
    for (const BoundaryFace& face : boundary_) {
      const Tet& tet = tets_[static_cast<std::size_t>(face.tet)];
      for (int s = 0; s < 4; ++s) {
        if (s != face.slot && tet.v[s] >= 0) {
          vertex_stamp_[static_cast<std::size_t>(tet.v[s])] = stamp_value_;
        }
      }
    }
    removed_.clear();
    int64_t removed_finite = 0;
    for (int32_t t : cavity_) {
      const Tet& tet = tets_[static_cast<std::size_t>(t)];
      if (!is_ghost(tet)) {
        ++removed_finite;
      }
      for (int s = 0; s < 4; ++s) {
        const int32_t vertex = tet.v[s];
        if (vertex >= 0 && vertex_stamp_[static_cast<std::size_t>(vertex)] != stamp_value_) {
          vertex_stamp_[static_cast<std::size_t>(vertex)] = stamp_value_;
          removed_.push_back(vertex);
        }
      }
    }
    if (!removed_.empty() && weights_ == nullptr) {
      return PHX_MC_INTERNAL_ERROR;
    }

    created_.clear();
    int64_t created_finite = 0;
    for (const BoundaryFace& face : boundary_) {
      Tet tet = tets_[static_cast<std::size_t>(face.tet)];
      tet.v[face.slot] = p;
      tet.n[0] = tet.n[1] = tet.n[2] = tet.n[3] = -1;
      tet.n[face.slot] = face.outside;
      tet.stamp = 0U;
      tet.state = kNoConflict;
      if (!is_ghost(tet)) {
        ++created_finite;
      }
      created_.push_back(tet);
    }
    if (finite_count_ - removed_finite + created_finite > max_tetrahedra_) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }

    new_ids_.clear();
    for (std::size_t i = 0; i < created_.size(); ++i) {
      int32_t id = -1;
      if (i < cavity_.size()) {
        id = cavity_[i];
      } else {
        id = allocate();
        if (id < 0) {
          return PHX_MC_CAPACITY_EXCEEDED;
        }
      }
      new_ids_.push_back(id);
    }
    for (std::size_t i = created_.size(); i < cavity_.size(); ++i) {
      Tet& dead = tets_[static_cast<std::size_t>(cavity_[i])];
      dead.v[0] = dead.v[1] = dead.v[2] = dead.v[3] = kDead;
      free_.push_back(cavity_[i]);
    }
    for (std::size_t i = 0; i < created_.size(); ++i) {
      tets_[static_cast<std::size_t>(new_ids_[i])] = created_[i];
      const BoundaryFace& face = boundary_[i];
      tets_[static_cast<std::size_t>(face.outside)].n[face.back] = new_ids_[i];
    }
    if (!link_star(new_ids_, p)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    for (int32_t vertex : removed_) {
      alive[static_cast<std::size_t>(vertex)] = 0;
    }
    finite_count_ += created_finite - removed_finite;
    hint_ = new_ids_.back();
    return PHX_MC_OK;
  }

  const double* points_;
  const double* weights_;
  int64_t max_tetrahedra_;
  std::vector<Tet> tets_;
  std::vector<std::uint32_t> vertex_stamp_;
  std::vector<int32_t> free_;
  std::uint32_t stamp_value_ = 0;
  std::uint64_t walk_step_ = 0;
  int64_t finite_count_ = 0;
  int32_t hint_ = 0;
  std::vector<int32_t> cavity_;
  std::vector<int32_t> stack_;
  std::vector<BoundaryFace> boundary_;
  std::vector<int32_t> removed_;
  std::vector<Tet> created_;
  std::vector<int32_t> new_ids_;
  std::vector<LinkSlot> link_table_;
  std::uint32_t link_stamp_ = 0;
};

int32_t triangulate_3d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, int64_t max_tetrahedra, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || point_count > kMaxMeshPoints || max_tetrahedra < 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (point_count > 0 && (points == nullptr || (weighted && weights == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const double* active_weights = weighted ? weights : nullptr;
  int32_t status = validate_points(points, point_count, 3, active_weights);
  if (status != PHX_MC_OK) {
    return status;
  }
  std::vector<int32_t> vertex_map;
  const std::vector<int32_t> representatives =
      deduplicate_points(points, point_count, 3, active_weights, vertex_map);
  const std::vector<int32_t> order = brio_hilbert_order(points, 3, representatives);

  std::vector<char> alive(static_cast<std::size_t>(point_count), 0);
  for (int32_t vertex : representatives) {
    alive[static_cast<std::size_t>(vertex)] = 1;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  {
    Triangulation3D triangulation(points, active_weights, point_count, max_tetrahedra);
    status = triangulation.build(order, alive);
    if (status != PHX_MC_OK) {
      return status;
    }
    triangulation.collect_cells(result->cells);
  }
  for (int32_t& target : vertex_map) {
    if (target >= 0 && alive[static_cast<std::size_t>(target)] == 0) {
      target = -1;
    }
  }
  result->dimension = 3;
  result->input_point_count = point_count;
  result->points.assign(points, points + 3 * point_count);
  result->vertex_map = std::move(vertex_map);
  canonicalize_cells(*result);
  *mesh = result.release();
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_delaunay_3d(int64_t point_count, const double* points, int64_t max_tetrahedra,
                           phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] {
    return phx::mc::triangulate_3d(point_count, points, nullptr, false, max_tetrahedra, mesh);
  });
}

int32_t phx_mc_regular_3d(int64_t point_count, const double* points, const double* weights,
                          int64_t max_tetrahedra, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] {
    return phx::mc::triangulate_3d(point_count, points, weights, true, max_tetrahedra, mesh);
  });
}

}  // extern "C"
