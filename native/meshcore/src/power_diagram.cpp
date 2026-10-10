//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted power cells: exact clipping of power cells against a tetrahedral
// domain decomposition, combinatorial vertex reconciliation, connected
// restricted-cell components and conforming polyhedral assembly.
//
// Pieces.  For every tet T the sites whose cells meet T are found breadth
// first over the regular-triangulation adjacency, starting at the owner of the
// centroid of T (a greedy power-distance walk); a nonempty piece enqueues only
// the neighbors that share a bisector face with it inside T, because the
// restricted power diagram of a convex tet is a face-to-face partition whose
// dual graph is connected.  Each piece is the exact clip of T by the
// bisector halfspaces of its site (clip3d.cpp); the bisector of i towards j is
// the exact negation of the bisector of j towards i.
//
// Vertex identity.  A vertex of a piece (s, T) lies on bisector faces, each
// naming a neighbor site, and on faces of T.  Its identity is the set S of
// sites at equal power distance (s and the neighbors of its bisector faces)
// together with its carrier simplex: the vertices of T common to its incident
// tet faces (T itself, a face, an edge or a vertex).  Seen from any other
// piece the same point has the same identity, so the key welds all pieces
// without a distance tolerance. Equal sites are closed over regular adjacency
// using exact implicit-vertex sides, including tangent bisectors absent from
// the positive-area face list. Cospherical/site-carrier coincidences are admitted.
//
// Cells.  Pieces of one site joined through an unconstrained tet face are one
// cell per connected component; constrained faces (boundary, interfaces,
// internal sheets) always separate cells.  Fragments between one cell and one
// neighbor (another cell, or one boundary facet) are merged into loops by
// cancelling opposite edges when the union is a set of disjoint simple disks,
// and kept as separate faces otherwise.  Both sides of every interior face
// must produce the same boundary edges.  Vertices left on exactly two edges
// (collinear hanging vertices of merged fragments) are removed everywhere.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "bounded_memory.hpp"
#include "clip3d.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc {

struct PowerCellsResult {
  NativeVector<double> vertices;
  NativeVector<int32_t> vertex_carriers;
  NativeVector<int64_t> vertex_site_offsets{0};
  NativeVector<int32_t> vertex_sites;
  NativeVector<int64_t> face_offsets{0};
  NativeVector<int32_t> face_vertices;
  NativeVector<int32_t> face_cells;
  NativeVector<int32_t> face_facets;
  NativeVector<int32_t> cell_sites;
  NativeVector<int32_t> cell_regions;
  NativeVector<double> cell_volumes;
  NativeVector<double> cell_moments;
  NativeVector<double> cell_second_moments;
  NativeVector<int32_t> piece_sites;
  NativeVector<int32_t> piece_tets;
  NativeVector<int32_t> piece_cells;
  NativeVector<double> piece_volumes;
  int64_t counters[PHX_MC_POWER_CELLS_COUNTERS] = {};
  int64_t failure[PHX_MC_POWER_CELLS_FAILURE] = {};
  double phase_seconds[4] = {-1.0, -1.0, -1.0, -1.0};
  int64_t source_work_units = 0;
};

namespace {

enum Failure : int64_t {
  kNoFailure = 0,
  kDegenerateVertex = 1,
  kMissingPiece = 2,
  kInconsistentFace = 3,
  kPieceBudget = 4,
  kVertexBudget = 5,
  kWorkBudget = 6,
  kInvalidDomain = 7,
  kClipFailure = 8,
  kSeedFailure = 9,
  kImageSourceDomain = 10,
};

enum Counter : int {
  kClips = 0,
  kPieces = 1,
  kEmptyClips = 2,
  kWalkSteps = 3,
  kMergedLinks = 4,
  kUnmergedGroups = 5,
  kRemovedVertices = 6,
  kCells = 7,
  kFaces = 8,
  kVertices = 9,
};

constexpr int32_t kVertexCapacity = 1 << 14;

struct Input {
  int64_t site_count;
  const double* sites;
  const double* weights;
  const int64_t* neighbor_offsets;
  const int32_t* neighbors;
  const int8_t* split_sites;
  int64_t point_count;
  const double* points;
  int64_t tet_count;
  const int32_t* tets;
  const int32_t* tet_regions;
  const int32_t* tet_face_facets;
  int64_t max_pieces;
  int64_t max_vertices;
  int64_t work_limit;
  int8_t record_phases;
  const ExactPowerCoordinates* exact_coordinates = nullptr;
};

using Key = NativeVector<int32_t>;

struct KeyHash {
  std::size_t operator()(const Key& key) const noexcept {
    uint64_t hash = 1469598103934665603ULL;
    for (const int32_t value : key) {
      hash ^= static_cast<uint32_t>(value);
      hash *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(hash);
  }
};

enum class FragmentKind : int8_t { kBisector, kTetFace };

struct Fragment {
  int32_t piece;
  int32_t start;
  int32_t size;
  FragmentKind kind;
  int32_t ref;  // neighbor site (bisector) or local tet face (tet face)
};

struct Piece {
  int32_t site;
  int32_t tet;
  int32_t fragment_begin;
  int32_t fragment_end;
  double volume;
  double moment[3];
  double second;
};

using Edge = std::pair<int32_t, int32_t>;
using GroupKey = std::tuple<int32_t, int32_t, int32_t>;  // cell, other, facet

int32_t validate_layout(const Input& in) {
  if (in.site_count < 1 || in.point_count < 4 || in.tet_count < 1 || in.max_pieces < 1 ||
      in.max_vertices < 1 || in.work_limit < 1 || in.site_count >= INT32_MAX ||
      (in.record_phases != 0 && in.record_phases != 1) ||
      in.point_count >= INT32_MAX || in.tet_count > INT32_MAX / 4 ||
      in.max_pieces >= INT32_MAX || in.max_vertices >= INT32_MAX ||
      !addressable(in.site_count, 3, sizeof(double)) ||
      !addressable(in.point_count, 3, sizeof(double)) ||
      !addressable(in.tet_count, 4, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if ((in.sites == nullptr && in.exact_coordinates == nullptr) || in.weights == nullptr || in.neighbor_offsets == nullptr ||
      in.split_sites == nullptr || in.points == nullptr || in.tets == nullptr ||
      in.tet_regions == nullptr || in.tet_face_facets == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return PHX_MC_OK;
}

int32_t validate(const Input& in) {
  const int32_t layout = validate_layout(in);
  if (layout != PHX_MC_OK) return layout;
  for (int64_t site = 0; site < in.site_count; ++site) {
    native_execution_charge(0);
    if (!weight_in_domain(in.weights[site]) ||
        (in.split_sites[site] != 0 && in.split_sites[site] != 1)) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (in.exact_coordinates == nullptr && !coordinate_in_domain(in.sites[3 * site + axis])) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int64_t entry = 0; entry < 3 * in.point_count; ++entry) {
    native_execution_charge(0);
    if (!coordinate_in_domain(in.points[entry])) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (in.neighbor_offsets[0] != 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t site = 0; site < in.site_count; ++site) {
    native_execution_charge(0);
    if (in.neighbor_offsets[site + 1] < in.neighbor_offsets[site]) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  const int64_t total = in.neighbor_offsets[in.site_count];
  if (!addressable(total, 1, sizeof(int32_t)) || (total > 0 && in.neighbors == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t site = 0; site < in.site_count; ++site) {
    for (int64_t entry = in.neighbor_offsets[site]; entry < in.neighbor_offsets[site + 1];
         ++entry) {
      native_execution_charge(0);
      const int32_t other = in.neighbors[entry];
      if (other < 0 || other >= in.site_count || other == site ||
          (entry > in.neighbor_offsets[site] && in.neighbors[entry - 1] >= other)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int64_t site = 0; site < in.site_count; ++site) {
    for (int64_t entry = in.neighbor_offsets[site]; entry < in.neighbor_offsets[site + 1]; ++entry) {
      native_execution_charge(0);
      const int32_t other = in.neighbors[entry];
      if (!std::binary_search(in.neighbors + in.neighbor_offsets[other],
                              in.neighbors + in.neighbor_offsets[other + 1],
                              static_cast<int32_t>(site))) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int64_t entry = 0; entry < 4 * in.tet_count; ++entry) {
    native_execution_charge(0);
    if (in.tets[entry] < 0 || in.tets[entry] >= in.point_count || in.tet_face_facets[entry] < -1) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (int64_t tet = 0; tet < in.tet_count; ++tet) {
    native_execution_charge(0);
    if (in.tet_regions[tet] < 0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

class Builder {
 public:
  Builder(const Input& in, PowerCellsResult& out) : in_(in), out_(out) {}

  int32_t run() try {
    int32_t status = phase(0, [&] { return adjacency(); });
    if (status == PHX_MC_OK) {
      status = phase(1, [&] {
        const int32_t clipped = clip_all();
        return clipped == PHX_MC_OK ? reconcile_tet_faces() : clipped;
      });
    }
    if (status == PHX_MC_OK) {
      status = phase(2, [&] {
        split_.assign(in_.split_sites, in_.split_sites + in_.site_count);
        return components();
      });
    }
    if (status == PHX_MC_OK) {
      status = phase(3, [&] { return assemble(); });
    }
    native_execution_charge(0);
    return status;
  } catch (const ExecutionRefusal& refusal) {
    return fail(kWorkBudget, -1, -1, refusal.status, refusal.status);
  }

  // A bisector may coincide with a carrier face: the opposite nonempty piece
  // can belong to a different site. Match the actual reciprocal fragment,
  // not an assumed same-site piece in the neighboring tetrahedron.
  int32_t reconcile_tet_faces() {
    fragment_mates_.assign(fragments_.size(), -1);
    NativeMap<Key, int32_t> pending;
    Key key;
    for (int32_t index = 0; index < static_cast<int32_t>(fragments_.size()); ++index) {
      native_execution_charge(0);
      const Fragment& fragment = fragments_[index];
      if (fragment.kind != FragmentKind::kTetFace) {
        continue;
      }
      const Piece& piece = pieces_[fragment.piece];
      const int32_t neighbor = tet_neighbor_[4 * piece.tet + fragment.ref];
      if (neighbor < 0) {
        continue;
      }
      key.clear();
      key.push_back(std::min(piece.tet, neighbor));
      key.push_back(std::max(piece.tet, neighbor));
      key.insert(key.end(), fragment_vertices_.begin() + fragment.start,
                 fragment_vertices_.begin() + fragment.start + fragment.size);
      std::sort(key.begin() + 2, key.end(), [](int32_t a, int32_t b) {
        native_execution_charge(0);
        return a < b;
      });
      auto [found, inserted] = pending.try_emplace(key, index);
      if (!inserted) {
        const int32_t other = found->second;
        if (other < 0 || pieces_[fragments_[other].piece].tet == piece.tet) {
          return fail(kInconsistentFace, piece.site, piece.tet, PHX_MC_OK,
                      PHX_MC_DEGENERATE_INPUT);
        }
        fragment_mates_[index] = other;
        fragment_mates_[other] = index;
        found->second = -1;
      }
    }
    for (const auto& [key, index] : pending) {
      native_execution_charge(0);
      if (index >= 0) {
        const Piece& piece = pieces_[fragments_[index].piece];
        return fail(kMissingPiece, piece.site, piece.tet, PHX_MC_OK,
                    PHX_MC_DEGENERATE_INPUT);
      }
    }
    return PHX_MC_OK;
  }

 private:
  const Input& in_;
  PowerCellsResult& out_;
  TetrahedronClipper clipper_;
  ClippedPolytope polytope_;
  NativeVector<int32_t> tet_neighbor_;
  NativeVector<Piece> pieces_;
  NativeVector<Fragment> fragments_;
  NativeVector<int32_t> fragment_vertices_;
  NativeUnorderedMap<Key, int32_t, KeyHash> keys_;
  NativeVector<double> coordinates_;
  NativeVector<int32_t> carriers_;
  NativeUnorderedMap<int64_t, int32_t> piece_index_;
  NativeVector<int8_t> split_;
  NativeVector<int32_t> piece_cell_;
  int32_t cell_count_ = 0;
  NativeVector<int64_t> identity_stamp_;
  int64_t identity_epoch_ = 0;
  NativeVector<int32_t> fragment_mates_;
  NativeVector<int32_t> piece_vertex_stamp_;
  Key identity_key_;
  NativeVector<int32_t> vertex_map_;

  template <class Function>
  int32_t phase(int index, Function&& function) {
    if (in_.record_phases == 0) {
      return function();
    }
    const auto started = std::chrono::steady_clock::now();
    const int32_t status = function();
    out_.phase_seconds[index] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return status;
  }

  int32_t fail(Failure reason, int64_t site, int64_t tet, int32_t clip_status, int32_t status) {
    out_.failure[0] = reason;
    out_.failure[1] = site;
    out_.failure[2] = tet;
    out_.failure[3] = clip_status;
    return status;
  }

  const double* point(int32_t index) const { return in_.points + 3 * static_cast<int64_t>(index); }
  const double* site(int32_t index) const {
    return in_.sites == nullptr ? nullptr : in_.sites + 3 * static_cast<int64_t>(index);
  }
  const ExactPowerCoordinates* exact_site(int32_t index) const {
    return in_.exact_coordinates == nullptr ? nullptr : in_.exact_coordinates + index;
  }
  double relative_site_coordinate(int32_t index, int axis, double coordinate) const {
    const auto* source = exact_site(index);
    return source == nullptr ? coordinate - site(index)[axis]
                             : (Expansion(coordinate) - source->exact[axis]).estimate();
  }

  bool over_budget() const {
    return out_.counters[kClips] + out_.counters[kWalkSteps] > in_.work_limit;
  }

  int64_t piece_key(int32_t site_index, int32_t tet) const {
    return static_cast<int64_t>(site_index) * in_.tet_count + tet;
  }

  // Face adjacency of the decomposition: every face is shared by at most two
  // tets, boundary faces are constrained, and the two sides of a face agree
  // on its facet; an unconstrained face never separates regions.
  int32_t adjacency() {
    tet_neighbor_.assign(static_cast<std::size_t>(4 * in_.tet_count), -1);
    NativeMap<std::array<int32_t, 3>, int32_t> open;
    for (int32_t tet = 0; tet < in_.tet_count; ++tet) {
      native_execution_charge(0);
      for (int k = 0; k < 4; ++k) {
        std::array<int32_t, 3> face{};
        int count = 0;
        for (int v = 0; v < 4; ++v) {
          if (v != k) {
            face[count++] = in_.tets[4 * tet + v];
          }
        }
        std::sort(face.begin(), face.end());
        const int32_t slot = 4 * tet + k;
        auto [found, inserted] = open.try_emplace(face, slot);
        if (inserted) {
          continue;
        }
        const int32_t other = found->second;
        if (other < 0 || in_.tet_face_facets[other] != in_.tet_face_facets[slot] ||
            (in_.tet_face_facets[slot] < 0 &&
             in_.tet_regions[other / 4] != in_.tet_regions[tet])) {
          return fail(kInvalidDomain, -1, tet, PHX_MC_OK, PHX_MC_INVALID_INPUT);
        }
        tet_neighbor_[slot] = other / 4;
        tet_neighbor_[other] = tet;
        found->second = -1;
      }
    }
    for (int64_t slot = 0; slot < 4 * in_.tet_count; ++slot) {
      native_execution_charge(0);
      if (tet_neighbor_[slot] < 0 && in_.tet_face_facets[slot] < 0) {
        return fail(kInvalidDomain, -1, slot / 4, PHX_MC_OK, PHX_MC_INVALID_INPUT);
      }
    }
    return PHX_MC_OK;
  }


  // Greedy descent of the power distance to x over the adjacency; the
  // distance strictly decreases, so the walk terminates.
  int32_t walk(int32_t start, const double* x, int32_t& owner) {
    int32_t current = start;
    while (true) {
      native_execution_charge(0);
      int32_t next = -1;
      int32_t best = current;
      for (int64_t e = in_.neighbor_offsets[current]; e < in_.neighbor_offsets[current + 1]; ++e) {
        native_execution_charge(0);
        const int32_t candidate = in_.neighbors[e];
        if (TetrahedronClipper::point_power_side(
                x, site(candidate), in_.weights[candidate], site(best), in_.weights[best],
                exact_site(candidate), exact_site(best)) > 0) {
          best = candidate;
          next = candidate;
        }
      }
      if (next < 0) {
        break;
      }
      native_execution_charge(1);
      ++out_.counters[kWalkSteps];
      if (over_budget()) {
        return fail(kWorkBudget, current, -1, PHX_MC_OK, PHX_MC_CAPACITY_EXCEEDED);
      }
      current = next;
    }
    owner = current;
    return PHX_MC_OK;
  }

  int32_t clip_all() {
    int32_t start = -1;
    for (int32_t s = 0; s < in_.site_count; ++s) {
      native_execution_charge(0);
      if (in_.neighbor_offsets[s + 1] > in_.neighbor_offsets[s]) {
        start = s;
        break;
      }
    }
    if (start < 0) {
      if (in_.site_count == 1) {
        start = 0;
      } else {
        return fail(kSeedFailure, -1, 0, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
      }
    }
    NativeVector<int32_t> stamp(static_cast<std::size_t>(in_.site_count), -1);
    NativeDeque<int32_t> queue;
    for (int32_t tet = 0; tet < in_.tet_count; ++tet) {
      native_execution_charge(0);
      double centroid[3] = {0.0, 0.0, 0.0};
      for (int v = 0; v < 4; ++v) {
        for (int axis = 0; axis < 3; ++axis) {
          centroid[axis] += 0.25 * point(in_.tets[4 * tet + v])[axis];
        }
      }
      int32_t owner = start;
      int32_t status = walk(start, centroid, owner);
      if (status != PHX_MC_OK) {
        return status;
      }
      start = owner;
      queue.clear();
      queue.push_back(owner);
      stamp[owner] = tet;
      bool found = false;
      while (!queue.empty()) {
        native_execution_charge(0);
        const int32_t s = queue.front();
        queue.pop_front();
        bool nonempty = false;
        status = clip_piece(s, tet, nonempty);
        if (status != PHX_MC_OK) {
          return status;
        }
        if (nonempty) {
          found = true;
          const Piece& piece = pieces_.back();
          for (int32_t f = piece.fragment_begin; f < piece.fragment_end; ++f) {
            native_execution_charge(0);
            const Fragment& fragment = fragments_[f];
            if (fragment.kind == FragmentKind::kBisector && stamp[fragment.ref] != tet) {
              stamp[fragment.ref] = tet;
              queue.push_back(fragment.ref);
            }
          }
        } else if (!found) {
          // A rounded walk may stop next to the owner: widen until a piece exists.
          for (int64_t e = in_.neighbor_offsets[s]; e < in_.neighbor_offsets[s + 1]; ++e) {
            native_execution_charge(0);
            const int32_t other = in_.neighbors[e];
            if (stamp[other] != tet) {
              stamp[other] = tet;
              queue.push_back(other);
            }
          }
        }
      }
      if (!found) {
        return fail(kSeedFailure, owner, tet, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
      }
    }
    out_.counters[kPieces] = static_cast<int64_t>(pieces_.size());
    return PHX_MC_OK;
  }

  int32_t clip_piece(int32_t s, int32_t tet, bool& nonempty) {
    nonempty = false;
    const int64_t first = in_.neighbor_offsets[s];
    const int32_t count = static_cast<int32_t>(in_.neighbor_offsets[s + 1] - first);
    double corners[12];
    for (int v = 0; v < 4; ++v) {
      for (int axis = 0; axis < 3; ++axis) {
        corners[3 * v + axis] = point(in_.tets[4 * tet + v])[axis];
      }
    }
    if (!native_execution_spend(1)) {
      return fail(kWorkBudget, s, tet, PHX_MC_OK,
                  native_execution_status(PHX_MC_CAPACITY_EXCEEDED));
    }
    ++out_.counters[kClips];
    if (over_budget()) {
      return fail(kWorkBudget, s, tet, PHX_MC_OK, PHX_MC_CAPACITY_EXCEEDED);
    }
    const int32_t status = clipper_.clip_power(
        corners, in_.sites, in_.weights, s, in_.neighbors + first, count,
        kVertexCapacity, polytope_, in_.exact_coordinates);
    if (status != PHX_MC_OK) {
      return fail(kClipFailure, s, tet, status, status);
    }
    // Exact clipping owns emptiness. Rounded fan moments can vanish for a
    // positive thin piece whose source intersections round to one point.
    if (polytope_.empty()) {
      ++out_.counters[kEmptyClips];
      return PHX_MC_OK;
    }
    if (static_cast<int64_t>(pieces_.size()) >= in_.max_pieces) {
      return fail(kPieceBudget, s, tet, PHX_MC_OK, PHX_MC_CAPACITY_EXCEEDED);
    }
    nonempty = true;
    return record(s, tet, first);
  }

  // Welds the vertices of the current polytope by their identity keys and
  // stores its faces as fragments on global vertices.
  int32_t record(int32_t s, int32_t tet, int64_t first_neighbor) {
    const ClippedPolytope& poly = polytope_;
    const int vertex_count = static_cast<int>(poly.vertices.size() / 3);
    const int face_count = static_cast<int>(poly.face_labels.size());
    if (fragments_.size() + poly.face_labels.size() >= INT32_MAX ||
        fragment_vertices_.size() + poly.face_vertices.size() >= INT32_MAX) {
      return fail(kVertexBudget, s, tet, PHX_MC_OK, PHX_MC_CAPACITY_EXCEEDED);
    }
    NativeVector<int32_t> & global = vertex_map_;
    global.resize(static_cast<std::size_t>(vertex_count));
    if (identity_stamp_.empty()) {
      identity_stamp_.assign(static_cast<std::size_t>(in_.site_count), 0);
    }
    Key& key = identity_key_;
    for (int v = 0; v < vertex_count; ++v) {
      native_execution_charge(0);
      key.clear();
      key.push_back(s);
      const int64_t epoch = ++identity_epoch_;
      identity_stamp_[s] = epoch;
      const double* p = site(s);
      for (std::size_t cursor = 0; cursor < key.size(); ++cursor) {
        native_execution_charge(0);
        const int32_t equal_site = key[cursor];
        for (int64_t e = in_.neighbor_offsets[equal_site];
             e < in_.neighbor_offsets[equal_site + 1]; ++e) {
          native_execution_charge(0);
          const int32_t other = in_.neighbors[e];
          if (identity_stamp_[other] == epoch) {
            continue;
          }
          identity_stamp_[other] = epoch;
          native_execution_charge(1);
          ++out_.counters[kWalkSteps];
          if (over_budget()) {
            return fail(kWorkBudget, s, tet, PHX_MC_OK, PHX_MC_CAPACITY_EXCEEDED);
          }
          if (clipper_.vertex_power_side(v, p, in_.weights[s], site(other), in_.weights[other],
                                        exact_site(s), exact_site(other)) == 0) {
            key.push_back(other);
          }
        }
      }
      std::sort(key.begin(), key.end(), [](int32_t a, int32_t b) {
        native_execution_charge(0);
        return a < b;
      });
      key.push_back(-1);
      const std::size_t carrier_begin = key.size();
      for (int k = 0; k < 4; ++k) {
        if (clipper_.vertex_base_side(v, k) != 0) {
          key.push_back(in_.tets[4 * tet + k]);
        }
      }
      std::sort(key.begin() + carrier_begin, key.end());
      if (key.size() == carrier_begin) {
        return fail(kDegenerateVertex, s, tet, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
      }
      auto [found, inserted] = keys_.try_emplace(key, static_cast<int32_t>(keys_.size()));
      if (inserted) {
        if (static_cast<int64_t>(keys_.size()) > in_.max_vertices) {
          return fail(kVertexBudget, s, tet, PHX_MC_OK, PHX_MC_CAPACITY_EXCEEDED);
        }
        coordinates_.insert(coordinates_.end(), poly.vertices.begin() + 3 * v,
                            poly.vertices.begin() + 3 * v + 3);
        for (std::size_t k = 0; k < 4; ++k) {
          carriers_.push_back(carrier_begin + k < key.size() ? key[carrier_begin + k] : -1);
        }
      }
      if (piece_vertex_stamp_.size() < keys_.size()) {
        piece_vertex_stamp_.resize(keys_.size(), 0);
      }
      const int32_t stamp = static_cast<int32_t>(pieces_.size()) + 1;
      if (piece_vertex_stamp_[found->second] == stamp) {
        return fail(kDegenerateVertex, s, tet, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
      }
      piece_vertex_stamp_[found->second] = stamp;
      global[v] = found->second;
    }
    Piece piece{};
    piece.site = s;
    piece.tet = tet;
    piece.fragment_begin = static_cast<int32_t>(fragments_.size());
    piece.volume = poly.volume;
    for (int axis = 0; axis < 3; ++axis) {
      piece.moment[axis] = poly.moment[axis];
    }
    piece.second = second_moment(s);
    const int32_t index = static_cast<int32_t>(pieces_.size());
    for (int f = 0; f < face_count; ++f) {
      native_execution_charge(0);
      const int32_t label = poly.face_labels[f];
      Fragment fragment{};
      fragment.piece = index;
      fragment.start = static_cast<int32_t>(fragment_vertices_.size());
      fragment.size = poly.face_offsets[f + 1] - poly.face_offsets[f];
      fragment.kind = label < 0 ? FragmentKind::kTetFace : FragmentKind::kBisector;
      fragment.ref = label < 0 ? -1 - label : in_.neighbors[first_neighbor + label];
      if (label >= 0) {
        for (int k = 0; k < 4; ++k) {
          bool incident = true;
          for (int32_t e = poly.face_offsets[f]; e < poly.face_offsets[f + 1]; ++e) {
            native_execution_charge(0);
            incident = incident && clipper_.vertex_base_side(poly.face_vertices[e], k) == 0;
          }
          if (incident) {
            fragment.kind = FragmentKind::kTetFace;
            fragment.ref = k;
            break;
          }
        }
      }
      for (int32_t e = poly.face_offsets[f]; e < poly.face_offsets[f + 1]; ++e) {
        native_execution_charge(0);
        fragment_vertices_.push_back(global[poly.face_vertices[e]]);
      }
      fragments_.push_back(fragment);
    }
    piece.fragment_end = static_cast<int32_t>(fragments_.size());
    pieces_.push_back(piece);
    piece_index_.emplace(piece_key(s, tet), index);
    return PHX_MC_OK;
  }

  // Integral of |x - p_s|^2 over the current polytope by fan tetrahedra
  // (o, x_0, x_i, x_{i+1}) from the vertex average o:
  // integral over a tetrahedron = V / 20 (sum |v_k|^2 + |sum v_k|^2).
  double second_moment(int32_t s) const {
    const ClippedPolytope& poly = polytope_;
    const std::size_t count = poly.vertices.size() / 3;
    double o[3] = {0.0, 0.0, 0.0};
    for (std::size_t v = 0; v < count; ++v) {
      native_execution_charge(0);
      for (int axis = 0; axis < 3; ++axis) {
        o[axis] += poly.vertices[3 * v + axis];
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      o[axis] = relative_site_coordinate(s, axis, o[axis] / static_cast<double>(count));
    }
    double total = 0.0;
    for (std::size_t f = 0; f + 1 < poly.face_offsets.size(); ++f) {
      native_execution_charge(0);
      const int32_t begin = poly.face_offsets[f];
      const int32_t end = poly.face_offsets[f + 1];
      double x0[3];
      for (int axis = 0; axis < 3; ++axis) {
        x0[axis] = relative_site_coordinate(s, axis, poly.vertices[3 * poly.face_vertices[begin] + axis]);
      }
      for (int32_t e = begin + 1; e + 1 < end; ++e) {
        native_execution_charge(0);
        double b[3];
        double c[3];
        for (int axis = 0; axis < 3; ++axis) {
          b[axis] = relative_site_coordinate(s, axis, poly.vertices[3 * poly.face_vertices[e] + axis]);
          c[axis] = relative_site_coordinate(s, axis, poly.vertices[3 * poly.face_vertices[e + 1] + axis]);
        }
        const double u[3] = {x0[0] - o[0], x0[1] - o[1], x0[2] - o[2]};
        const double v[3] = {b[0] - o[0], b[1] - o[1], b[2] - o[2]};
        const double w[3] = {c[0] - o[0], c[1] - o[1], c[2] - o[2]};
        const double volume = (u[0] * (v[1] * w[2] - v[2] * w[1]) -
                               u[1] * (v[0] * w[2] - v[2] * w[0]) +
                               u[2] * (v[0] * w[1] - v[1] * w[0])) /
                              6.0;
        double squares = 0.0;
        double sum[3] = {0.0, 0.0, 0.0};
        for (const double* vertex : {static_cast<const double*>(o), static_cast<const double*>(x0),
                                     static_cast<const double*>(b), static_cast<const double*>(c)}) {
          for (int axis = 0; axis < 3; ++axis) {
            squares += vertex[axis] * vertex[axis];
            sum[axis] += vertex[axis];
          }
        }
        total += volume / 20.0 * (squares + sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
      }
    }
    return total;
  }

  int32_t find_piece(int32_t s, int32_t tet) const {
    const auto found = piece_index_.find(piece_key(s, tet));
    return found == piece_index_.end() ? -1 : found->second;
  }

  static int32_t root(NativeVector<int32_t>& parent, int32_t x) {
    while (parent[x] != x) {
      native_execution_charge(0);
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  // Connected components of the pieces of every unsplit site through
  // unconstrained tet faces.  A site whose component would meet itself across
  // a constrained face (an internal sheet ending inside the cell) is split.
  int32_t components() {
    const int32_t count = static_cast<int32_t>(pieces_.size());
    while (true) {
      native_execution_charge(0);
      NativeVector<int32_t> parent(static_cast<std::size_t>(count));
      std::iota(parent.begin(), parent.end(), 0);
      int64_t links = 0;
      for (const Fragment& fragment : fragments_) {
        native_execution_charge(0);
        const Piece& piece = pieces_[fragment.piece];
        if (fragment.kind != FragmentKind::kTetFace || split_[piece.site] != 0) {
          continue;
        }
        const int32_t index = static_cast<int32_t>(&fragment - fragments_.data());
        const int32_t slot = 4 * piece.tet + fragment.ref;
        const int32_t neighbor = tet_neighbor_[slot];
        if (neighbor < 0 || in_.tet_face_facets[slot] >= 0) {
          continue;
        }
        const int32_t other = fragments_[fragment_mates_[index]].piece;
        if (pieces_[other].site != piece.site) {
          continue;
        }
        const int32_t a = root(parent, fragment.piece);
        const int32_t b = root(parent, other);
        if (fragment.piece < other) {
          ++links;
        }
        if (a != b) {
          parent[std::max(a, b)] = std::min(a, b);
        }
      }
      bool resplit = false;
      for (const Fragment& fragment : fragments_) {
        native_execution_charge(0);
        const Piece& piece = pieces_[fragment.piece];
        if (fragment.kind != FragmentKind::kTetFace || split_[piece.site] != 0) {
          continue;
        }
        const int32_t slot = 4 * piece.tet + fragment.ref;
        const int32_t neighbor = tet_neighbor_[slot];
        if (neighbor < 0 || in_.tet_face_facets[slot] < 0) {
          continue;
        }
        const int32_t index = static_cast<int32_t>(&fragment - fragments_.data());
        const int32_t other = fragments_[fragment_mates_[index]].piece;
        if (root(parent, fragment.piece) == root(parent, other)) {
          split_[piece.site] = 1;
          resplit = true;
        }
      }
      if (resplit) {
        continue;
      }
      out_.counters[kMergedLinks] = links;
      // Cells ordered by (site, first piece): deterministic from the inputs.
      NativeVector<int32_t> roots;
      for (int32_t p = 0; p < count; ++p) {
        native_execution_charge(0);
        if (root(parent, p) == p) {
          roots.push_back(p);
        }
      }
      std::sort(roots.begin(), roots.end(), [&](int32_t x, int32_t y) {
        native_execution_charge(0);
        return std::make_pair(pieces_[x].site, x) < std::make_pair(pieces_[y].site, y);
      });
      NativeVector<int32_t> cell_of_root(static_cast<std::size_t>(count), -1);
      for (std::size_t k = 0; k < roots.size(); ++k) {
        native_execution_charge(0);
        cell_of_root[roots[k]] = static_cast<int32_t>(k);
      }
      cell_count_ = static_cast<int32_t>(roots.size());
      piece_cell_.assign(static_cast<std::size_t>(count), -1);
      out_.cell_sites.assign(roots.size(), -1);
      out_.cell_regions.assign(roots.size(), -1);
      out_.cell_volumes.assign(roots.size(), 0.0);
      out_.cell_moments.assign(3 * roots.size(), 0.0);
      out_.cell_second_moments.assign(roots.size(), 0.0);
      for (int32_t p = 0; p < count; ++p) {
        native_execution_charge(0);
        const int32_t cell = cell_of_root[root(parent, p)];
        const Piece& piece = pieces_[p];
        piece_cell_[p] = cell;
        out_.cell_sites[cell] = piece.site;
        out_.cell_regions[cell] = in_.tet_regions[piece.tet];
        out_.cell_volumes[cell] += piece.volume;
        for (int axis = 0; axis < 3; ++axis) {
          out_.cell_moments[3 * cell + axis] += piece.moment[axis];
        }
        out_.cell_second_moments[cell] += piece.second;
      }
      return PHX_MC_OK;
    }
  }

  // Remaining directed boundary edges of a fragment group after cancelling
  // every edge against its reverse, sorted.
  NativeVector<Edge> boundary_edges(const NativeVector<int32_t>& group) const {
    NativeMap<Edge, int32_t> count;
    for (const int32_t index : group) {
      native_execution_charge(0);
      const Fragment& fragment = fragments_[index];
      for (int32_t k = 0; k < fragment.size; ++k) {
        native_execution_charge(0);
        const int32_t a = fragment_vertices_[fragment.start + k];
        const int32_t b = fragment_vertices_[fragment.start + (k + 1) % fragment.size];
        auto reverse = count.find({b, a});
        if (reverse != count.end() && reverse->second > 0) {
          --reverse->second;
        } else {
          ++count[{a, b}];
        }
      }
    }
    NativeVector<Edge> edges;
    for (const auto& [edge, multiplicity] : count) {
      for (int32_t k = 0; k < multiplicity; ++k) {
        native_execution_charge(0);
        edges.push_back(edge);
      }
    }
    return edges;
  }

  void newell(const int32_t* loop, int32_t size, double* normal) const {
    for (int32_t k = 0; k < size; ++k) {
      native_execution_charge(0);
      const double* a = coordinates_.data() + 3 * static_cast<int64_t>(loop[k]);
      const double* b = coordinates_.data() + 3 * static_cast<int64_t>(loop[(k + 1) % size]);
      normal[0] += (a[1] - b[1]) * (a[2] + b[2]);
      normal[1] += (a[2] - b[2]) * (a[0] + b[0]);
      normal[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
  }

  // Simple loops of a boundary edge set, or false when a vertex has several
  // outgoing edges (a pinch) or a loop runs against the group orientation
  // (a hole).
  bool chain(const NativeVector<Edge>& edges, const NativeVector<int32_t>& group,
             NativeVector<NativeVector<int32_t>>& loops) const {
    NativeMap<int32_t, int32_t> next;
    for (const Edge& edge : edges) {
      native_execution_charge(0);
      if (!next.emplace(edge.first, edge.second).second) {
        return false;
      }
    }
    NativeMap<int32_t, bool> used;
    for (const auto& [start, unused] : next) {
      native_execution_charge(0);
      if (used[start]) {
        continue;
      }
      NativeVector<int32_t> loop;
      int32_t current = start;
      while (!used[current]) {
        native_execution_charge(0);
        used[current] = true;
        loop.push_back(current);
        const auto found = next.find(current);
        if (found == next.end()) {
          return false;
        }
        current = found->second;
      }
      if (current != start || loop.size() < 3) {
        return false;
      }
      loops.push_back(std::move(loop));
    }
    if (loops.size() > 1) {
      double total[3] = {0.0, 0.0, 0.0};
      for (const int32_t index : group) {
        native_execution_charge(0);
        newell(fragment_vertices_.data() + fragments_[index].start, fragments_[index].size,
               total);
      }
      for (const auto& loop : loops) {
        native_execution_charge(0);
        double normal[3] = {0.0, 0.0, 0.0};
        newell(loop.data(), static_cast<int32_t>(loop.size()), normal);
        if (normal[0] * total[0] + normal[1] * total[1] + normal[2] * total[2] <= 0.0) {
          return false;
        }
      }
    }
    return true;
  }

  int32_t assemble() {
    NativeMap<GroupKey, NativeVector<int32_t>> groups;
    for (int32_t index = 0; index < static_cast<int32_t>(fragments_.size()); ++index) {
      native_execution_charge(0);
      const Fragment& fragment = fragments_[index];
      const Piece& piece = pieces_[fragment.piece];
      const int32_t cell = piece_cell_[fragment.piece];
      int32_t other = 0;
      int32_t facet = -1;
      if (fragment.kind == FragmentKind::kBisector) {
        const int32_t q = find_piece(fragment.ref, piece.tet);
        if (q < 0) {
          return fail(kMissingPiece, piece.site, piece.tet, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
        }
        other = piece_cell_[q];
      } else {
        const int32_t slot = 4 * piece.tet + fragment.ref;
        const int32_t neighbor = tet_neighbor_[slot];
        facet = in_.tet_face_facets[slot];
        if (neighbor < 0) {
          other = -1 - facet;
        } else {
          const int32_t q = fragments_[fragment_mates_[index]].piece;
          other = piece_cell_[q];
          if (other == cell) {
            continue;
          }
        }
      }
      groups[{cell, other, facet}].push_back(index);
    }
    NativeVector<NativeVector<int32_t>> loops;
    NativeVector<std::array<int32_t, 3>> loop_owners;
    for (const auto& [key, group] : groups) {
      native_execution_charge(0);
      const auto [cell, other, facet] = key;
      if (other >= 0 && other < cell) {
        continue;
      }
      NativeVector<Edge> edges = boundary_edges(group);
      if (other >= 0) {
        const auto reverse = groups.find({other, cell, facet});
        if (reverse == groups.end()) {
          return fail(kMissingPiece, out_.cell_sites[cell], -1, PHX_MC_OK,
                      PHX_MC_DEGENERATE_INPUT);
        }
        NativeVector<Edge> mirrored = boundary_edges(reverse->second);
        for (Edge& edge : mirrored) {
          native_execution_charge(0);
          std::swap(edge.first, edge.second);
        }
        std::sort(mirrored.begin(), mirrored.end(), [](const Edge& a, const Edge& b) {
          native_execution_charge(0);
          return a < b;
        });
        if (mirrored != edges) {
          return fail(kInconsistentFace, out_.cell_sites[cell], -1, PHX_MC_OK,
                      PHX_MC_DEGENERATE_INPUT);
        }
      }
      NativeVector<NativeVector<int32_t>> merged;
      if (!chain(edges, group, merged)) {
        ++out_.counters[kUnmergedGroups];
        merged.clear();
        for (const int32_t index : group) {
          native_execution_charge(0);
          const Fragment& fragment = fragments_[index];
          merged.emplace_back(fragment_vertices_.begin() + fragment.start,
                              fragment_vertices_.begin() + fragment.start + fragment.size);
        }
      }
      for (auto& loop : merged) {
        native_execution_charge(0);
        loops.push_back(std::move(loop));
        loop_owners.push_back({cell, other, facet});
      }
    }
    return publish(loops, loop_owners);
  }

  // Removes vertices on exactly two edges (never shrinking a loop below a
  // triangle), compacts the used vertices and writes the output arrays.
  int32_t publish(NativeVector<NativeVector<int32_t>>& loops,
                  const NativeVector<std::array<int32_t, 3>>& owners) {
    const std::size_t vertex_total = keys_.size();
    NativeVector<NativeVector<int32_t>> adjacent(vertex_total);
    for (const auto& loop : loops) {
      for (std::size_t k = 0; k < loop.size(); ++k) {
        native_execution_charge(0);
        const int32_t a = loop[k];
        const int32_t b = loop[(k + 1) % loop.size()];
        adjacent[a].push_back(b);
        adjacent[b].push_back(a);
      }
    }
    NativeVector<uint8_t> removable(vertex_total, 0);
    for (std::size_t v = 0; v < vertex_total; ++v) {
      native_execution_charge(0);
      auto& row = adjacent[v];
      std::sort(row.begin(), row.end(), [](int32_t a, int32_t b) {
        native_execution_charge(0);
        return a < b;
      });
      row.erase(std::unique(row.begin(), row.end(), [](int32_t a, int32_t b) {
        native_execution_charge(0);
        return a == b;
      }), row.end());
      if (row.size() == 2) {
        bool collinear = true;
        for (int axis = 0; axis < 3; ++axis) {
          const int next = (axis + 1) % 3;
          const double a[2] = {coordinates_[3 * row[0] + axis], coordinates_[3 * row[0] + next]};
          const double b[2] = {coordinates_[3 * v + axis], coordinates_[3 * v + next]};
          const double c[2] = {coordinates_[3 * row[1] + axis], coordinates_[3 * row[1] + next]};
          collinear = collinear && orient2d(a, b, c) == 0;
        }
        removable[v] = collinear ? 1 : 0;
      }
    }
    bool changed = true;
    while (changed) {
      native_execution_charge(0);
      changed = false;
      for (const auto& loop : loops) {
        std::size_t kept = 0;
        for (const int32_t v : loop) {
          native_execution_charge(0);
          kept += removable[v] == 0 ? 1 : 0;
        }
        if (kept < 3) {
          for (const int32_t v : loop) {
            native_execution_charge(0);
            changed = changed || removable[v] != 0;
            removable[v] = 0;
          }
        }
      }
    }
    NativeVector<int32_t> index(vertex_total, -1);
    for (const auto& loop : loops) {
      for (const int32_t v : loop) {
        native_execution_charge(0);
        if (removable[v] == 0) {
          index[v] = 0;
        }
      }
    }
    NativeVector<const Key*> ordered_keys(vertex_total, nullptr);
    for (const auto& [key, vertex] : keys_) {
      native_execution_charge(0);
      ordered_keys[vertex] = &key;
    }
    int32_t used = 0;
    for (std::size_t v = 0; v < vertex_total; ++v) {
      native_execution_charge(0);
      if (index[v] == 0) {
        index[v] = used++;
        out_.vertices.insert(out_.vertices.end(), coordinates_.begin() + 3 * v,
                             coordinates_.begin() + 3 * v + 3);
        out_.vertex_carriers.insert(out_.vertex_carriers.end(), carriers_.begin() + 4 * v,
                                    carriers_.begin() + 4 * v + 4);
        const Key& key = *ordered_keys[v];
        const auto separator = std::find(key.begin(), key.end(), -1);
        out_.vertex_sites.insert(out_.vertex_sites.end(), key.begin(), separator);
        out_.vertex_site_offsets.push_back(static_cast<int64_t>(out_.vertex_sites.size()));
      }
      out_.counters[kRemovedVertices] += removable[v];
    }
    for (std::size_t f = 0; f < loops.size(); ++f) {
      for (const int32_t v : loops[f]) {
        native_execution_charge(0);
        if (removable[v] == 0) {
          out_.face_vertices.push_back(index[v]);
        }
      }
      out_.face_offsets.push_back(static_cast<int64_t>(out_.face_vertices.size()));
      out_.face_cells.push_back(owners[f][0]);
      out_.face_cells.push_back(owners[f][1]);
      out_.face_facets.push_back(owners[f][2]);
    }
    for (std::size_t p = 0; p < pieces_.size(); ++p) {
      native_execution_charge(0);
      out_.piece_sites.push_back(pieces_[p].site);
      out_.piece_tets.push_back(pieces_[p].tet);
      out_.piece_cells.push_back(piece_cell_[p]);
      out_.piece_volumes.push_back(pieces_[p].volume);
    }
    out_.counters[kCells] = cell_count_;
    out_.counters[kFaces] = static_cast<int64_t>(loops.size());
    out_.counters[kVertices] = used;
    return PHX_MC_OK;
  }
};

}  // namespace
}  // namespace phx::mc

struct phx_mc_power_cells : phx::mc::NativeAllocatedObject {
  phx::mc::PowerCellsResult result;
};

extern "C" {

int32_t phx_mc_restricted_power_cells(int64_t site_count, const double* sites,
                                      const double* weights, const int64_t* neighbor_offsets,
                                      const int32_t* neighbors, const int8_t* split_sites,
                                      int64_t point_count, const double* points, int64_t tet_count,
                                      const int32_t* tets, const int32_t* tet_regions,
                                      const int32_t* tet_face_facets, int64_t max_pieces,
                                      int64_t max_vertices, int64_t work_limit, int8_t record_phases,
                                      phx_mc_power_cells** result) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *result = nullptr;
    if (work_limit < 0) return PHX_MC_INVALID_ARGUMENT;
    phx::mc::NativeExecutionScope execution(
        static_cast<uint64_t>(work_limit), std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(), std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<double>::infinity());
    const phx::mc::Input input{site_count, sites,      weights,       neighbor_offsets,
                               neighbors,  split_sites, point_count,  points,
                               tet_count,  tets,        tet_regions,  tet_face_facets,
                               max_pieces, max_vertices, work_limit, record_phases};
    int32_t status = phx::mc::validate_layout(input);
    if (status != PHX_MC_OK) {
      return status;
    }
    auto handle = phx::mc::make_native_unique<phx_mc_power_cells>();
    try {
      status = phx::mc::validate(input);
      if (status == PHX_MC_OK) {
        phx::mc::Builder builder(input, handle->result);
        status = builder.run();
      }
    } catch (const phx::mc::ExecutionRefusal& refusal) {
      handle->result.failure[0] = phx::mc::kWorkBudget;
      handle->result.failure[1] = -1;
      handle->result.failure[2] = -1;
      handle->result.failure[3] = refusal.status;
      status = refusal.status;
    }
    if (status == PHX_MC_OK || handle->result.failure[0] != phx::mc::kNoFailure) {
      *result = handle.release();
    }
    return status;
  });
}

// Exact image coordinates are sums, not rounded surrogate site triples.
// Image-axis site IDs remain in all construction witnesses; original ownership
// and authored action provenance are retained by the immutable source owner.
int32_t phx_mc_restricted_power_cells_exact(
    int64_t original_site_count, const double* original_sites,
    const double* original_weights, int64_t image_count,
    const int32_t* image_site_owners, const int64_t* image_coordinate_offsets,
    int64_t component_count, const double* image_coordinate_components,
    const int64_t* neighbor_offsets, const int32_t* neighbors, const int8_t* split_sites,
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    const int32_t* tet_regions, const int32_t* tet_face_facets,
    int64_t max_pieces, int64_t max_vertices, int64_t work_limit, int8_t record_phases,
    phx_mc_power_cells** result) {
  return phx::mc::guarded([&]() -> int32_t {
    using namespace phx::mc;
    if (result == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *result = nullptr;
    if (original_site_count < 1 || original_site_count >= INT32_MAX ||
        image_count < 1 || image_count >= INT32_MAX || component_count < 0 ||
        work_limit < 1 || original_sites == nullptr || original_weights == nullptr ||
        image_site_owners == nullptr || image_coordinate_offsets == nullptr ||
        (component_count != 0 && image_coordinate_components == nullptr) ||
        !addressable(original_site_count, 3, sizeof(double)) ||
        !addressable(image_count, 3, sizeof(int64_t)) ||
        !addressable(component_count, 1, sizeof(double))) return PHX_MC_INVALID_ARGUMENT;
    NativeExecutionScope execution(
        static_cast<uint64_t>(work_limit), std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(), std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<double>::infinity());
    int64_t source_work_units = 0;
    const auto source_visit = [&]() {
      native_execution_charge(1);
      ++source_work_units;
    };
    const auto source_domain_refusal = [&](int64_t coordinate, int64_t component) {
      auto refused = make_native_unique<phx_mc_power_cells>();
      refused->result.source_work_units = source_work_units;
      refused->result.failure[0] = kImageSourceDomain;
      refused->result.failure[1] = coordinate;
      refused->result.failure[2] = component;
      refused->result.failure[3] = PHX_MC_INVALID_INPUT;
      *result = refused.release();
      return PHX_MC_INVALID_INPUT;
    };
    // Validate the entire source envelope before allocating a construction or
    // mutating a clip workspace. Coordinate polynomials have degree <= six;
    // components must share the original coordinate quantum 2^-172 and the
    // absolute coordinate envelope, including cancellation-prone expansions.
    for (int64_t site = 0; site < original_site_count; ++site) {
      source_visit();
      if (!weight_in_domain(original_weights[site])) return PHX_MC_INVALID_INPUT;
      for (int axis = 0; axis < 3; ++axis)
        if (!coordinate_in_domain(original_sites[3 * site + axis])) return PHX_MC_INVALID_INPUT;
    }
    if (image_coordinate_offsets[0] != 0 ||
        image_coordinate_offsets[3 * image_count] != component_count)
      return PHX_MC_INVALID_ARGUMENT;
    for (int64_t image = 0; image < image_count; ++image) {
      source_visit();
      if (image_site_owners[image] < 0 || image_site_owners[image] >= original_site_count)
        return PHX_MC_INVALID_ARGUMENT;
      for (int axis = 0; axis < 3; ++axis) {
        const int64_t coordinate = 3 * image + axis;
        const int64_t first = image_coordinate_offsets[coordinate];
        const int64_t last = image_coordinate_offsets[coordinate + 1];
        if (first < 0 || last < first || last > component_count)
          return PHX_MC_INVALID_ARGUMENT;
        Expansion absolute_sum;
        double preceding = 0.0;
        for (int64_t entry = first; entry < last; ++entry) {
          source_visit();
          const double value = image_coordinate_components[entry];
          const double magnitude = std::abs(value);
          const double quantum = std::ldexp(value, 172);
          if (!std::isfinite(value) || value == 0.0 ||
              magnitude > kCoordinateMaxMagnitude ||
              !std::isfinite(quantum) || quantum != std::trunc(quantum))
            return source_domain_refusal(coordinate, entry);
          if (entry != first &&
              preceding > (magnitude - std::nextafter(magnitude, 0.0)) * 0.5)
            return source_domain_refusal(coordinate, entry);
          preceding = magnitude;
          absolute_sum = absolute_sum + Expansion(magnitude);
          if ((absolute_sum - Expansion(kCoordinateMaxMagnitude)).sign() > 0)
            return source_domain_refusal(coordinate, entry);
        }
      }
    }
    NativeVector<Approx> approximate(static_cast<std::size_t>(3 * image_count));
    NativeVector<Expansion> exact(static_cast<std::size_t>(3 * image_count));
    NativeVector<ExactPowerCoordinates> coordinates(static_cast<std::size_t>(image_count));
    NativeVector<double> weights(static_cast<std::size_t>(image_count));
    for (int64_t image = 0; image < image_count; ++image) {
      source_visit();
      weights[image] = original_weights[image_site_owners[image]];
      coordinates[image] = {approximate.data() + 3 * image, exact.data() + 3 * image};
      for (int axis = 0; axis < 3; ++axis) {
        const int64_t coordinate = 3 * image + axis;
        Approx filtered = Approx::exact(0.0);
        Expansion expanded;
        for (int64_t entry = image_coordinate_offsets[coordinate];
             entry < image_coordinate_offsets[coordinate + 1]; ++entry) {
          source_visit();
          const double value = image_coordinate_components[entry];
          filtered = filtered + Approx::exact(value);
          expanded = expanded + Expansion(value);
        }
        approximate[coordinate] = filtered;
        exact[coordinate] = std::move(expanded);
      }
    }
    auto handle = make_native_unique<phx_mc_power_cells>();
    handle->result.source_work_units = source_work_units;
    if (source_work_units >= work_limit) {
      handle->result.failure[0] = kWorkBudget;
      handle->result.failure[1] = -1;
      handle->result.failure[2] = -1;
      handle->result.failure[3] = PHX_MC_CAPACITY_EXCEEDED;
      *result = handle.release();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    const Input input{image_count, nullptr, weights.data(), neighbor_offsets, neighbors,
                      split_sites, point_count, points, tet_count, tets, tet_regions,
                      tet_face_facets, max_pieces, max_vertices,
                      work_limit - source_work_units, record_phases,
                      coordinates.data()};
    int32_t status = validate_layout(input);
    if (status != PHX_MC_OK) return status;
    try {
      status = validate(input);
      if (status == PHX_MC_OK) {
        Builder builder(input, handle->result);
        status = builder.run();
      }
    } catch (const ExecutionRefusal& refusal) {
      handle->result.failure[0] = kWorkBudget;
      handle->result.failure[1] = -1;
      handle->result.failure[2] = -1;
      handle->result.failure[3] = refusal.status;
      status = refusal.status;
    }
    if (status == PHX_MC_OK || handle->result.failure[0] != kNoFailure)
      *result = handle.release();
    return status;
  });
}

int64_t phx_mc_power_cells_source_work_units(const phx_mc_power_cells* result) {
  return result->result.source_work_units;
}

void phx_mc_power_cells_sizes(const phx_mc_power_cells* result, int64_t* sizes) {
  const phx::mc::PowerCellsResult& r = result->result;
  sizes[0] = static_cast<int64_t>(r.vertices.size() / 3);
  sizes[1] = static_cast<int64_t>(r.face_facets.size());
  sizes[2] = static_cast<int64_t>(r.face_vertices.size());
  sizes[3] = static_cast<int64_t>(r.cell_sites.size());
  sizes[4] = static_cast<int64_t>(r.piece_sites.size());
}

int64_t phx_mc_power_cells_site_witness_size(const phx_mc_power_cells* result) {
  return static_cast<int64_t>(result->result.vertex_sites.size());
}

void phx_mc_power_cells_export_vertex_sites(
    const phx_mc_power_cells* result, int64_t* offsets, int32_t* sites) {
  std::copy(result->result.vertex_site_offsets.begin(), result->result.vertex_site_offsets.end(), offsets);
  std::copy(result->result.vertex_sites.begin(), result->result.vertex_sites.end(), sites);
}

void phx_mc_power_cells_counters(const phx_mc_power_cells* result, int64_t* counters) {
  std::copy(result->result.counters, result->result.counters + PHX_MC_POWER_CELLS_COUNTERS,
            counters);
}

void phx_mc_power_cells_failure(const phx_mc_power_cells* result, int64_t* failure) {
  std::copy(result->result.failure, result->result.failure + PHX_MC_POWER_CELLS_FAILURE,
            failure);
}

void phx_mc_power_cells_export(const phx_mc_power_cells* result, double* vertices,
                               int64_t* face_offsets, int32_t* face_vertices, int32_t* face_cells,
                               int32_t* face_facets, int32_t* cell_sites, int32_t* cell_regions,
                               double* cell_volumes, double* cell_moments,
                               double* cell_second_moments, int32_t* piece_sites,
                               int32_t* piece_tets, int32_t* piece_cells, double* piece_volumes) {
  const phx::mc::PowerCellsResult& r = result->result;
  std::copy(r.vertices.begin(), r.vertices.end(), vertices);
  std::copy(r.face_offsets.begin(), r.face_offsets.end(), face_offsets);
  std::copy(r.face_vertices.begin(), r.face_vertices.end(), face_vertices);
  std::copy(r.face_cells.begin(), r.face_cells.end(), face_cells);
  std::copy(r.face_facets.begin(), r.face_facets.end(), face_facets);
  std::copy(r.cell_sites.begin(), r.cell_sites.end(), cell_sites);
  std::copy(r.cell_regions.begin(), r.cell_regions.end(), cell_regions);
  std::copy(r.cell_volumes.begin(), r.cell_volumes.end(), cell_volumes);
  std::copy(r.cell_moments.begin(), r.cell_moments.end(), cell_moments);
  std::copy(r.cell_second_moments.begin(), r.cell_second_moments.end(), cell_second_moments);
  std::copy(r.piece_sites.begin(), r.piece_sites.end(), piece_sites);
  std::copy(r.piece_tets.begin(), r.piece_tets.end(), piece_tets);
  std::copy(r.piece_cells.begin(), r.piece_cells.end(), piece_cells);
  std::copy(r.piece_volumes.begin(), r.piece_volumes.end(), piece_volumes);
}

void phx_mc_power_cells_export_vertex_carriers(const phx_mc_power_cells* result,
                                              int32_t* vertex_carriers) {
  std::copy(result->result.vertex_carriers.begin(), result->result.vertex_carriers.end(),
            vertex_carriers);
}

void phx_mc_power_cells_phase_seconds(const phx_mc_power_cells* result, double* seconds) {
  std::copy(result->result.phase_seconds, result->result.phase_seconds + 4, seconds);
}

void phx_mc_power_cells_free(phx_mc_power_cells* result) { phx::mc::destroy_native_object(result); }

}  // extern "C"
