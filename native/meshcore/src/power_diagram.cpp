//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted power cells: exact clipping of power cells against a tetrahedral
// domain decomposition, combinatorial vertex reconciliation, connected
// restricted-cell components and conforming polyhedral assembly.
//
// Pieces. Complete requests discover sites meeting each tet breadth first over
// regular-triangulation adjacency. Selected requests clip selected sites and
// their actual face neighbors as bounded support, without recursive expansion.
// Each piece is the exact clip of T by the bisector halfspaces of its site
// (clip3d.cpp); the bisector of i towards j is the exact negation of j towards i.
// Only selected cells are published; support components identify remote faces.
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
#include <set>
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
#include "clip3d.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "topology_keys.hpp"
#include "geometry3.hpp"

namespace phx::mc {

struct PowerFailure {
  int64_t reason = 0;
  int64_t site = -1;
  int64_t tet = -1;
  int32_t clip_status = PHX_MC_OK;
};

struct PowerCellsResult {
  std::vector<double> vertices;
  std::vector<int32_t> vertex_carriers;
  std::vector<int64_t> vertex_site_offsets;
  std::vector<int32_t> vertex_sites;
  std::vector<int64_t> face_offsets;
  std::vector<int32_t> face_vertices;
  std::vector<int32_t> face_cells;
  std::vector<int32_t> face_facets;
  std::vector<int32_t> face_remote_sites;
  std::vector<int32_t> face_remote_tets;
  std::vector<int32_t> cell_sites;
  std::vector<int32_t> cell_regions;
  std::vector<double> cell_volumes;
  std::vector<double> cell_moments;
  std::vector<double> cell_second_moments;
  std::vector<int32_t> piece_sites;
  std::vector<int32_t> piece_tets;
  std::vector<int32_t> piece_cells;
  std::vector<double> piece_volumes;
  std::vector<double> piece_moments;
  PowerFailure failure;
};

namespace {

enum Failure : int64_t {
  kNoFailure = 0,
  kDegenerateVertex = 1,
  kMissingPiece = 2,
  kInconsistentFace = 3,
  kInvalidDomain = 4,
  kClipFailure = 5,
  kSeedFailure = 6,
  kImageSourceDomain = 7,
  kOutputCountOverflow = 8,
};
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
  const ExactPowerCoordinates* exact_coordinates = nullptr;
  int64_t selected_site_count = -1;
  const int32_t* selected_sites = nullptr;
};

bool add_count(int64_t& total, int64_t count, int64_t width = 1) {
  if (count < 0 || width < 0 || (width != 0 && count > (INT64_MAX - total) / width))
    return false;
  total += count * width;
  return true;
}

using Key = std::vector<int32_t>;

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
using GroupKey = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>;

int32_t validate_layout(const Input& in) {
  if (in.site_count < 1 || in.point_count < 4 || in.tet_count < 1 ||
      in.site_count >= INT32_MAX || in.point_count >= INT32_MAX ||
      in.tet_count > INT32_MAX / 4 ||
      !addressable(in.site_count, 3, sizeof(double)) ||
      !addressable(in.site_count, 1, sizeof(double)) ||
      !addressable(in.site_count, 1, sizeof(int8_t)) ||
      !addressable(in.site_count + 1, 1, sizeof(int64_t)) ||
      !addressable(in.point_count, 3, sizeof(double)) ||
      !addressable(in.tet_count, 4, sizeof(int32_t)) ||
      !addressable(in.tet_count, 4, sizeof(int32_t)) ||
      !addressable(in.tet_count, 1, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if ((in.sites == nullptr && in.exact_coordinates == nullptr) || in.weights == nullptr ||
      in.neighbor_offsets == nullptr || in.split_sites == nullptr || in.points == nullptr ||
      in.tets == nullptr || in.tet_regions == nullptr || in.tet_face_facets == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return PHX_MC_OK;
}

int32_t validate(const Input& in) {
  const int32_t layout = validate_layout(in);
  if (layout != PHX_MC_OK) return layout;
  for (int64_t site = 0; site < in.site_count; ++site) {
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
    if (!coordinate_in_domain(in.points[entry])) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (in.neighbor_offsets[0] != 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t site = 0; site < in.site_count; ++site) {
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
      const int32_t other = in.neighbors[entry];
      if (other < 0 || other >= in.site_count || other == site ||
          (entry > in.neighbor_offsets[site] && in.neighbors[entry - 1] >= other)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int64_t site = 0; site < in.site_count; ++site) {
    for (int64_t entry = in.neighbor_offsets[site]; entry < in.neighbor_offsets[site + 1]; ++entry) {
      const int32_t other = in.neighbors[entry];
      if (!std::binary_search(in.neighbors + in.neighbor_offsets[other],
                              in.neighbors + in.neighbor_offsets[other + 1],
                              static_cast<int32_t>(site))) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int64_t entry = 0; entry < 4 * in.tet_count; ++entry) {
    if (in.tets[entry] < 0 || in.tets[entry] >= in.point_count || in.tet_face_facets[entry] < -1) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (int64_t tet = 0; tet < in.tet_count; ++tet) {
    if (in.tet_regions[tet] < 0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

class Builder {
 public:
  Builder(const Input& in, PowerCellsResult& out) : in_(in), out_(out) {}

  int32_t run() {
    int32_t status = adjacency();
    if (status == PHX_MC_OK) {
      const int32_t clipped = clip_all();
      status = clipped == PHX_MC_OK ? reconcile_tet_faces() : clipped;
    }
    if (status == PHX_MC_OK) {
      split_.assign(in_.split_sites, in_.split_sites + in_.site_count);
      status = components();
    }
    if (status == PHX_MC_OK) status = assemble();
    return status;
  }

  // Match reciprocal decomposition-face fragments, including a bisector that
  // coincides with a carrier face owned by a different site's piece.
  int32_t reconcile_tet_faces() {
    fragment_mates_.assign(fragments_.size(), -1);
    std::map<Key, int32_t> pending;
    Key key;
    for (int32_t index = 0; index < static_cast<int32_t>(fragments_.size()); ++index) {
      const Fragment& fragment = fragments_[index];
      if (fragment.kind != FragmentKind::kTetFace) continue;
      const Piece& piece = pieces_[fragment.piece];
      const int32_t neighbor = tet_neighbor_[4 * piece.tet + fragment.ref];
      if (neighbor < 0) continue;
      key.clear();
      key.push_back(std::min(piece.tet, neighbor));
      key.push_back(std::max(piece.tet, neighbor));
      key.insert(key.end(), fragment_vertices_.begin() + fragment.start,
                 fragment_vertices_.begin() + fragment.start + fragment.size);
      std::sort(key.begin() + 2, key.end());
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
      if (index >= 0 && in_.selected_site_count < 0) {
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
  std::vector<int32_t> tet_neighbor_;
  std::vector<Piece> pieces_;
  std::vector<Fragment> fragments_;
  std::vector<int32_t> fragment_vertices_;
  std::unordered_map<Key, int32_t, KeyHash> keys_;
  std::vector<double> coordinates_;
  std::vector<int32_t> carriers_;
  std::unordered_map<int64_t, int32_t> piece_index_;
  std::vector<int8_t> split_;
  std::vector<int32_t> piece_cell_;
  std::vector<int32_t> cell_roots_;
  std::vector<int32_t> output_cells_;
  int32_t cell_count_ = 0;
  std::vector<int64_t> identity_stamp_;
  int64_t identity_epoch_ = 0;
  std::vector<int32_t> fragment_mates_;
  std::vector<int32_t> piece_vertex_stamp_;
  Key identity_key_;
  std::vector<int32_t> vertex_map_;


  int32_t fail(Failure reason, int64_t site, int64_t tet, int32_t clip_status, int32_t status) {
    out_.failure.reason = reason;
    out_.failure.site = site;
    out_.failure.tet = tet;
    out_.failure.clip_status = clip_status;
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


  int64_t piece_key(int32_t site_index, int32_t tet) const {
    return static_cast<int64_t>(site_index) * in_.tet_count + tet;
  }

  // Face adjacency of the decomposition: every face is shared by at most two
  // tets, boundary faces are constrained, and the two sides of a face agree
  // on its facet; an unconstrained face never separates regions.
  int32_t adjacency() {
    tet_neighbor_.assign(static_cast<std::size_t>(4 * in_.tet_count), -1);
    std::map<std::array<int32_t, 3>, int32_t> open;
    for (int32_t tet = 0; tet < in_.tet_count; ++tet) {
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
      int32_t next = -1;
      int32_t best = current;
      for (int64_t e = in_.neighbor_offsets[current]; e < in_.neighbor_offsets[current + 1]; ++e) {
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
      current = next;
    }
    owner = current;
    return PHX_MC_OK;
  }

  int32_t clip_all() {
    if (in_.selected_site_count >= 0) {
      for (int32_t tet = 0; tet < in_.tet_count; ++tet) {
        for (int64_t i = 0; i < in_.selected_site_count; ++i) {
          bool nonempty = false;
          const int32_t status = clip_piece(in_.selected_sites[i], tet, nonempty);
          if (status != PHX_MC_OK) return status;
        }
      }
      // Seed only sites actually touching a selected piece, including a
      // bisector coincident with a carrier face. Explore the same support
      // site across its tet faces to recover its complete components, but
      // never recursively add its bisector neighbors.
      const int32_t reconciled = reconcile_tet_faces();
      if (reconciled != PHX_MC_OK) return reconciled;
      std::vector<const Key*> vertex_keys(keys_.size(), nullptr);
      for (const auto& [key, vertex] : keys_) vertex_keys[vertex] = &key;
      std::set<int64_t> queued;
      std::deque<std::pair<int32_t, int32_t>> queue;
      const auto enqueue = [&](int32_t site, int32_t tet) {
        if (site < 0 || tet < 0 ||
            std::binary_search(in_.selected_sites,
                               in_.selected_sites + in_.selected_site_count, site))
          return;
        if (queued.insert(piece_key(site, tet)).second) queue.emplace_back(site, tet);
      };
      const int32_t selected_fragments = static_cast<int32_t>(fragments_.size());
      for (int32_t i = 0; i < selected_fragments; ++i) {
        const Fragment& fragment = fragments_[i];
        const Piece& piece = pieces_[fragment.piece];
        if (fragment.kind == FragmentKind::kBisector) {
          enqueue(fragment.ref, piece.tet);
        } else if (fragment_mates_[i] < 0) {
          const int32_t neighbor = tet_neighbor_[4 * piece.tet + fragment.ref];
          if (neighbor >= 0)
            enqueue(remote_carrier_site(fragment, neighbor, vertex_keys), neighbor);
        }
      }
      while (!queue.empty()) {
        const auto [site, tet] = queue.front();
        queue.pop_front();
        bool nonempty = false;
        const int32_t status = clip_piece(site, tet, nonempty);
        if (status != PHX_MC_OK) return status;
        if (!nonempty) continue;
        const Piece& piece = pieces_.back();
        for (int32_t f = piece.fragment_begin; f < piece.fragment_end; ++f) {
          const Fragment& fragment = fragments_[f];
          if (fragment.kind == FragmentKind::kTetFace)
            enqueue(site, tet_neighbor_[4 * tet + fragment.ref]);
        }
      }
      return PHX_MC_OK;
    }
    int32_t start = -1;
    for (int32_t s = 0; s < in_.site_count; ++s) {
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
    std::vector<int32_t> stamp(static_cast<std::size_t>(in_.site_count), -1);
    std::deque<int32_t> queue;
    for (int32_t tet = 0; tet < in_.tet_count; ++tet) {
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
            const Fragment& fragment = fragments_[f];
            if (fragment.kind == FragmentKind::kBisector && stamp[fragment.ref] != tet) {
              stamp[fragment.ref] = tet;
              queue.push_back(fragment.ref);
            }
          }
        } else if (!found) {
          // A rounded walk may stop next to the owner: widen until a piece exists.
          for (int64_t e = in_.neighbor_offsets[s]; e < in_.neighbor_offsets[s + 1]; ++e) {
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
    const int32_t status = clipper_.clip_power(
        corners, in_.sites, in_.weights, s, count == 0 ? nullptr : in_.neighbors + first,
        count, polytope_,
        in_.exact_coordinates);
    if (status != PHX_MC_OK) {
      return fail(kClipFailure, s, tet, status, status);
    }
    // Exact nonemptiness does not guarantee a usable binary64 carrier. Refuse
    // positive exact pieces whose rounded vertices have no positive measure.
    if (polytope_.empty()) return PHX_MC_OK;
    if (!(polytope_.volume > 0.0) || !std::isfinite(polytope_.volume)) {
      return fail(kDegenerateVertex, s, tet, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
    }
    if (pieces_.size() >= static_cast<std::size_t>(INT32_MAX)) {
      return fail(kOutputCountOverflow, s, tet, PHX_MC_OK, PHX_MC_INDEX_RANGE_ERROR);
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
      return fail(kOutputCountOverflow, s, tet, PHX_MC_OK, PHX_MC_INDEX_RANGE_ERROR);
    }
    std::vector<int32_t> & global = vertex_map_;
    global.resize(static_cast<std::size_t>(vertex_count));
    if (identity_stamp_.empty()) {
      identity_stamp_.assign(static_cast<std::size_t>(in_.site_count), 0);
    }
    Key& key = identity_key_;
    for (int v = 0; v < vertex_count; ++v) {
      key.clear();
      key.push_back(s);
      const int64_t epoch = ++identity_epoch_;
      identity_stamp_[s] = epoch;
      const double* p = site(s);
      for (std::size_t cursor = 0; cursor < key.size(); ++cursor) {
        const int32_t equal_site = key[cursor];
        for (int64_t e = in_.neighbor_offsets[equal_site];
             e < in_.neighbor_offsets[equal_site + 1]; ++e) {
          const int32_t other = in_.neighbors[e];
          if (identity_stamp_[other] == epoch) {
            continue;
          }
          identity_stamp_[other] = epoch;
          const int side = clipper_.vertex_power_side(
              v, p, in_.weights[s], site(other), in_.weights[other], exact_site(s),
              exact_site(other));
          if (side < -1 || side > 1) {
            return fail(kClipFailure, s, tet, PHX_MC_INTERNAL_ERROR, PHX_MC_INTERNAL_ERROR);
          }
          if (side == 0) {
            key.push_back(other);
          }
        }
      }
      std::sort(key.begin(), key.end(), [](int32_t a, int32_t b) {
        return a < b;
      });
      key.push_back(-1);
      const std::size_t carrier_begin = key.size();
      for (int k = 0; k < 4; ++k) {
        const int side = clipper_.vertex_base_side(v, k);
        if (side < 0 || side > 1) {
          return fail(kClipFailure, s, tet, PHX_MC_INTERNAL_ERROR, PHX_MC_INTERNAL_ERROR);
        }
        if (side != 0) {
          key.push_back(in_.tets[4 * tet + k]);
        }
      }
      std::sort(key.begin() + carrier_begin, key.end());
      if (key.size() == carrier_begin) {
        return fail(kDegenerateVertex, s, tet, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
      }
      if (keys_.find(key) == keys_.end() && keys_.size() >= static_cast<std::size_t>(INT32_MAX)) {
        return PHX_MC_INDEX_RANGE_ERROR;
      }
      auto [found, inserted] = keys_.try_emplace(key, static_cast<int32_t>(keys_.size()));
      if (inserted) {

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
    for (int axis = 0; axis < 3; ++axis) piece.moment[axis] = poly.moment[axis];
    piece.second = second_moment(s);
    const int32_t index = static_cast<int32_t>(pieces_.size());
    for (int f = 0; f < face_count; ++f) {
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
            const int side = clipper_.vertex_base_side(poly.face_vertices[e], k);
            if (side < 0 || side > 1) {
              return fail(kClipFailure, s, tet, PHX_MC_INTERNAL_ERROR, PHX_MC_INTERNAL_ERROR);
            }
            incident = incident && side == 0;
          }
          if (incident) {
            fragment.kind = FragmentKind::kTetFace;
            fragment.ref = k;
            break;
          }
        }
      }
      for (int32_t e = poly.face_offsets[f]; e < poly.face_offsets[f + 1]; ++e) {
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
      for (int axis = 0; axis < 3; ++axis) {
        o[axis] += poly.vertices[3 * v + axis];
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      o[axis] = relative_site_coordinate(s, axis, o[axis] / static_cast<double>(count));
    }
    double total = 0.0;
    for (std::size_t f = 0; f + 1 < poly.face_offsets.size(); ++f) {
      const int32_t begin = poly.face_offsets[f];
      const int32_t end = poly.face_offsets[f + 1];
      double x0[3];
      for (int axis = 0; axis < 3; ++axis) {
        x0[axis] = relative_site_coordinate(s, axis, poly.vertices[3 * poly.face_vertices[begin] + axis]);
      }
      for (int32_t e = begin + 1; e + 1 < end; ++e) {
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


  // Connected components of the pieces of every unsplit site through
  // unconstrained tet faces.  A site whose component would meet itself across
  // a constrained face (an internal sheet ending inside the cell) is split.
  int32_t components() {
    const int32_t count = static_cast<int32_t>(pieces_.size());
    while (true) {
      std::vector<int32_t> parent(static_cast<std::size_t>(count));
      std::iota(parent.begin(), parent.end(), 0);
      for (const Fragment& fragment : fragments_) {
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
        if (fragment_mates_[index] < 0) continue;
        const int32_t other = fragments_[fragment_mates_[index]].piece;
        if (pieces_[other].site != piece.site) {
          continue;
        }
        const int32_t a = topology::find_root(parent, fragment.piece);
        const int32_t b = topology::find_root(parent, other);
        if (a != b) {
          parent[std::max(a, b)] = std::min(a, b);
        }
      }
      bool resplit = false;
      for (const Fragment& fragment : fragments_) {
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
        if (fragment_mates_[index] < 0) continue;
        const int32_t other = fragments_[fragment_mates_[index]].piece;
        if (topology::find_root(parent, fragment.piece) ==
            topology::find_root(parent, other)) {
          split_[piece.site] = 1;
          resplit = true;
        }
      }
      if (resplit) {
        continue;
      }
      // Cells ordered by (site, first piece): deterministic from the inputs.
      std::vector<int32_t> roots;
      for (int32_t p = 0; p < count; ++p) {
        if (topology::find_root(parent, p) == p) {
          roots.push_back(p);
        }
      }
      std::sort(roots.begin(), roots.end(), [&](int32_t x, int32_t y) {
        return std::make_pair(pieces_[x].site, x) < std::make_pair(pieces_[y].site, y);
      });
      std::vector<int32_t> cell_of_root(static_cast<std::size_t>(count), -1);
      for (std::size_t k = 0; k < roots.size(); ++k) {
        cell_of_root[roots[k]] = static_cast<int32_t>(k);
      }
      cell_count_ = static_cast<int32_t>(roots.size());
      piece_cell_.assign(static_cast<std::size_t>(count), -1);
      cell_roots_ = std::move(roots);
      for (int32_t p = 0; p < count; ++p) {
        const int32_t cell = cell_of_root[
            topology::find_root(parent, p)];
        piece_cell_[p] = cell;
      }
      output_cells_.assign(cell_roots_.size(), -1);
      int32_t output_count = 0;
      for (std::size_t cell = 0; cell < cell_roots_.size(); ++cell) {
        const int32_t site = pieces_[cell_roots_[cell]].site;
        if (in_.selected_site_count < 0 ||
            std::binary_search(in_.selected_sites, in_.selected_sites + in_.selected_site_count,
                               site))
          output_cells_[cell] = output_count++;
      }
      cell_count_ = output_count;
      return PHX_MC_OK;
    }
  }

  // Remaining directed boundary edges of a fragment group after cancelling
  // every edge against its reverse, sorted.
  std::vector<Edge> boundary_edges(const std::vector<int32_t>& group) const {
    std::map<Edge, int32_t> count;
    for (const int32_t index : group) {
      const Fragment& fragment = fragments_[index];
      for (int32_t k = 0; k < fragment.size; ++k) {
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
    std::vector<Edge> edges;
    for (const auto& [edge, multiplicity] : count) {
      for (int32_t k = 0; k < multiplicity; ++k) {
        edges.push_back(edge);
      }
    }
    return edges;
  }

  void newell(const int32_t* loop, int32_t size, double* normal) const {
    for (int32_t k = 0; k < size; ++k) {
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
  bool chain(const std::vector<Edge>& edges, const std::vector<int32_t>& group,
             std::vector<std::vector<int32_t>>& loops) const {
    std::map<int32_t, int32_t> next;
    for (const Edge& edge : edges) {
      if (!next.emplace(edge.first, edge.second).second) {
        return false;
      }
    }
    std::map<int32_t, bool> used;
    for (const auto& [start, unused] : next) {
      if (used[start]) {
        continue;
      }
      std::vector<int32_t> loop;
      int32_t current = start;
      while (!used[current]) {
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
        newell(fragment_vertices_.data() + fragments_[index].start, fragments_[index].size,
               total);
      }
      for (const auto& loop : loops) {
        double normal[3] = {0.0, 0.0, 0.0};
        newell(loop.data(), static_cast<int32_t>(loop.size()), normal);
        if (normal[0] * total[0] + normal[1] * total[1] + normal[2] * total[2] <= 0.0) {
          return false;
        }
      }
    }
    return true;
  }

  // Equal-power witnesses common to the carrier face identify the possible
  // owners across it; compare power at the opposite corner of the other tet.
  int32_t remote_carrier_site(const Fragment& fragment, int32_t neighbor,
                              const std::vector<const Key*>& vertex_keys) const {
    const Piece& piece = pieces_[fragment.piece];
    const Key& first = *vertex_keys[fragment_vertices_[fragment.start]];
    const auto end = std::find(first.begin(), first.end(), -1);
    const int32_t* tet = in_.tets + 4 * static_cast<int64_t>(piece.tet);
    int32_t opposite = -1;
    for (int k = 0; k < 4; ++k) {
      const int32_t vertex = in_.tets[4 * static_cast<int64_t>(neighbor) + k];
      bool on_face = false;
      for (int j = 0; j < 4; ++j)
        if (j != fragment.ref && tet[j] == vertex) on_face = true;
      if (!on_face) opposite = vertex;
    }
    if (opposite < 0) return -1;
    int32_t best = piece.site;
    for (auto it = first.begin(); it != end; ++it) {
      const int32_t candidate = *it;
      bool common = true;
      for (int32_t k = 1; k < fragment.size && common; ++k) {
        const Key& key = *vertex_keys[fragment_vertices_[fragment.start + k]];
        common = std::binary_search(key.begin(), std::find(key.begin(), key.end(), -1),
                                    candidate);
      }
      if (common && TetrahedronClipper::point_power_side(
              point(opposite), site(candidate), in_.weights[candidate],
              site(best), in_.weights[best], exact_site(candidate), exact_site(best)) > 0)
        best = candidate;
    }
    return best;
  }

  int32_t assemble() {
    std::map<GroupKey, std::vector<int32_t>> groups;
    std::vector<const Key*> vertex_keys(keys_.size(), nullptr);
    for (const auto& [key, vertex] : keys_) vertex_keys[vertex] = &key;
    for (int32_t index = 0; index < static_cast<int32_t>(fragments_.size()); ++index) {
      const Fragment& fragment = fragments_[index];
      const Piece& piece = pieces_[fragment.piece];
      const int32_t cell = piece_cell_[fragment.piece];
      int32_t other = 0;
      int32_t facet = -1;
      int32_t remote_site = -1;
      int32_t remote_tet = -1;
      if (fragment.kind == FragmentKind::kBisector) {
        const int32_t q = find_piece(fragment.ref, piece.tet);
        if (q < 0) {
          if (in_.selected_site_count < 0 ||
              std::binary_search(in_.selected_sites,
                                 in_.selected_sites + in_.selected_site_count, fragment.ref))
            return fail(kMissingPiece, piece.site, piece.tet, PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
          other = -1;
          remote_site = fragment.ref;
          remote_tet = piece.tet;
        } else {
          other = piece_cell_[q];
        }
      } else {
        const int32_t slot = 4 * piece.tet + fragment.ref;
        const int32_t neighbor = tet_neighbor_[slot];
        facet = in_.tet_face_facets[slot];
        if (neighbor < 0) {
          other = -1 - facet;
        } else {
          if (fragment_mates_[index] < 0) {
            remote_site = remote_carrier_site(fragment, neighbor, vertex_keys);
            if (remote_site < 0 || remote_site == piece.site ||
                std::binary_search(in_.selected_sites,
                                   in_.selected_sites + in_.selected_site_count, remote_site))
              return fail(kMissingPiece, piece.site, piece.tet, PHX_MC_OK,
                          PHX_MC_DEGENERATE_INPUT);
            other = -1;
            remote_tet = neighbor;
          } else {
            const int32_t q = fragments_[fragment_mates_[index]].piece;
            other = piece_cell_[q];
            if (other == cell) continue;
          }
        }
      }
      groups[{cell, other, facet, remote_site, remote_tet}].push_back(index);
    }
    std::vector<std::vector<int32_t>> loops;
    std::vector<std::array<int32_t, 5>> loop_owners;
    for (const auto& [key, group] : groups) {
      const auto [cell, other, facet, remote_site, remote_tet] = key;
      if (output_cells_[cell] < 0) continue;
      if (other >= 0 && output_cells_[other] >= 0 && other < cell) {
        continue;
      }
      std::vector<Edge> edges = boundary_edges(group);
      if (other >= 0) {
        const auto reverse = groups.find({other, cell, facet, -1, -1});
        if (reverse == groups.end()) {
          return fail(kMissingPiece, pieces_[cell_roots_[cell]].site, -1, PHX_MC_OK,
                      PHX_MC_DEGENERATE_INPUT);
        }
        std::vector<Edge> mirrored = boundary_edges(reverse->second);
        for (Edge& edge : mirrored) {
          std::swap(edge.first, edge.second);
        }
        std::sort(mirrored.begin(), mirrored.end(), [](const Edge& a, const Edge& b) {
          return a < b;
        });
        if (mirrored != edges) {
          return fail(kInconsistentFace, pieces_[cell_roots_[cell]].site, -1, PHX_MC_OK,
                      PHX_MC_DEGENERATE_INPUT);
        }
      }
      std::vector<std::vector<int32_t>> merged;
      if (!chain(edges, group, merged)) {
        merged.clear();
        for (const int32_t index : group) {
          const Fragment& fragment = fragments_[index];
          merged.emplace_back(fragment_vertices_.begin() + fragment.start,
                              fragment_vertices_.begin() + fragment.start + fragment.size);
        }
      }
      for (auto& loop : merged) {
        int32_t output_other = other >= 0 ? output_cells_[other] : other;
        int32_t output_remote_site = remote_site;
        int32_t output_remote_tet = remote_tet;
        if (other >= 0 && output_other < 0) {
          // Choose a fragment incident to this loop, not merely any fragment
          // in the component group: separate loops may use different tets.
          int32_t witness = -1;
          for (const int32_t index : group) {
            const Fragment& fragment = fragments_[index];
            for (int32_t k = 0; k < fragment.size; ++k) {
              if (fragment_vertices_[fragment.start + k] == loop[0] &&
                  fragment_vertices_[fragment.start + (k + 1) % fragment.size] == loop[1]) {
                witness = index;
                break;
              }
            }
            if (witness >= 0) break;
          }
          if (witness < 0)
            return fail(kInconsistentFace, pieces_[cell_roots_[cell]].site, -1,
                        PHX_MC_OK, PHX_MC_DEGENERATE_INPUT);
          const Fragment& fragment = fragments_[witness];
          const Piece& piece = pieces_[fragment.piece];
          const int32_t remote_piece = fragment.kind == FragmentKind::kBisector
              ? find_piece(fragment.ref, piece.tet)
              : fragments_[fragment_mates_[witness]].piece;
          output_remote_site = pieces_[remote_piece].site;
          output_remote_tet = pieces_[remote_piece].tet;
          output_other = -1;
        }
        loops.push_back(std::move(loop));
        loop_owners.push_back({output_cells_[cell], output_other, facet,
                               output_remote_site, output_remote_tet});
      }
    }
    return publish(loops, loop_owners);
  }

  int32_t reserve_output(const std::vector<std::vector<int32_t>>& loops,
                         const std::vector<uint8_t>& removable,
                         const std::vector<int32_t>& index,
                         const std::vector<const Key*>& ordered_keys) {
    int64_t vertex_count = 0;
    int64_t witness_count = 0;
    int64_t face_entries = 0;
    const int64_t face_count = static_cast<int64_t>(loops.size());
    for (std::size_t vertex = 0; vertex < index.size(); ++vertex) {
      if (index[vertex] < 0) continue;
      const Key& key = *ordered_keys[vertex];
      const auto separator = std::find(key.begin(), key.end(), -1);
      if (!add_count(vertex_count, 1) ||
          !add_count(witness_count, static_cast<int64_t>(separator - key.begin()))) {
        return PHX_MC_INDEX_RANGE_ERROR;
      }
    }
    for (const auto& loop : loops) {
      int64_t corners = 0;
      for (const int32_t vertex : loop) corners += removable[vertex] == 0 ? 1 : 0;
      if (!add_count(face_entries, corners)) return PHX_MC_INDEX_RANGE_ERROR;
    }
    if (!addressable(vertex_count, 3, sizeof(double)) ||
        !addressable(vertex_count, 4, sizeof(int32_t)) ||
        !addressable(vertex_count + 1, 1, sizeof(int64_t)) ||
        !addressable(witness_count, 1, sizeof(int32_t)) ||
        !addressable(face_count + 1, 1, sizeof(int64_t)) ||
        !addressable(face_entries, 1, sizeof(int32_t)) ||
        !addressable(face_count, 2, sizeof(int32_t)) ||
        !addressable(face_count, 1, sizeof(int32_t)) ||
        !addressable(cell_count_, 1, sizeof(int32_t)) ||
        !addressable(cell_count_, 3, sizeof(double)) ||
        !addressable(pieces_.size(), 1, sizeof(int32_t)) ||
        !addressable(pieces_.size(), 3, sizeof(double))) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    out_.vertices.reserve(static_cast<std::size_t>(3 * vertex_count));
    out_.vertex_carriers.reserve(static_cast<std::size_t>(4 * vertex_count));
    out_.vertex_site_offsets.reserve(static_cast<std::size_t>(vertex_count + 1));
    out_.vertex_sites.reserve(static_cast<std::size_t>(witness_count));
    out_.face_offsets.reserve(static_cast<std::size_t>(face_count + 1));
    out_.face_vertices.reserve(static_cast<std::size_t>(face_entries));
    out_.face_cells.reserve(static_cast<std::size_t>(2 * face_count));
    out_.face_facets.reserve(static_cast<std::size_t>(face_count));
    out_.piece_sites.reserve(pieces_.size());
    out_.piece_tets.reserve(pieces_.size());
    out_.piece_cells.reserve(pieces_.size());
    out_.piece_volumes.reserve(pieces_.size());
    out_.piece_moments.reserve(3 * pieces_.size());
    out_.vertex_site_offsets.push_back(0);
    out_.face_offsets.push_back(0);
    out_.cell_sites.assign(static_cast<std::size_t>(cell_count_), -1);
    out_.cell_regions.assign(static_cast<std::size_t>(cell_count_), -1);
    out_.cell_volumes.assign(static_cast<std::size_t>(cell_count_), 0.0);
    out_.cell_moments.assign(3 * static_cast<std::size_t>(cell_count_), 0.0);
    out_.cell_second_moments.assign(static_cast<std::size_t>(cell_count_), 0.0);
    for (std::size_t p = 0; p < pieces_.size(); ++p) {
      const int32_t cell = output_cells_[piece_cell_[p]];
      if (cell < 0) continue;
      const Piece& piece = pieces_[p];
      out_.cell_sites[cell] = piece.site;
      out_.cell_regions[cell] = in_.tet_regions[piece.tet];
      out_.cell_volumes[cell] += piece.volume;
      for (int axis = 0; axis < 3; ++axis)
        out_.cell_moments[3 * cell + axis] += piece.moment[axis];
      out_.cell_second_moments[cell] += piece.second;
    }
    return PHX_MC_OK;
  }

  // Removes vertices on exactly two edges (never shrinking a loop below a
  // triangle), compacts the used vertices and writes the output arrays.
  int32_t publish(std::vector<std::vector<int32_t>>& loops,
                  const std::vector<std::array<int32_t, 5>>& owners) {
    const std::size_t vertex_total = keys_.size();
    std::vector<std::vector<int32_t>> adjacent(vertex_total);
    for (const auto& loop : loops) {
      for (std::size_t k = 0; k < loop.size(); ++k) {
        const int32_t a = loop[k];
        const int32_t b = loop[(k + 1) % loop.size()];
        adjacent[a].push_back(b);
        adjacent[b].push_back(a);
      }
    }
    std::vector<uint8_t> removable(vertex_total, 0);
    for (std::size_t v = 0; v < vertex_total; ++v) {
      auto& row = adjacent[v];
      std::sort(row.begin(), row.end(), [](int32_t a, int32_t b) {
        return a < b;
      });
      row.erase(std::unique(row.begin(), row.end(), [](int32_t a, int32_t b) {
        return a == b;
      }), row.end());
      if (row.size() == 2) {
        removable[v] = geometry::collinear(coordinates_.data() + 3 * row[0],
                                           coordinates_.data() + 3 * v,
                                           coordinates_.data() + 3 * row[1], 0) ? 1 : 0;
      }
    }
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& loop : loops) {
        std::size_t kept = 0;
        for (const int32_t v : loop) {
          kept += removable[v] == 0 ? 1 : 0;
        }
        if (kept < 3) {
          for (const int32_t v : loop) {
            changed = changed || removable[v] != 0;
            removable[v] = 0;
          }
        }
      }
    }
    std::vector<int32_t> index(vertex_total, -1);
    for (const auto& loop : loops) {
      for (const int32_t v : loop) {
        if (removable[v] == 0) {
          index[v] = 0;
        }
      }
    }
    std::vector<const Key*> ordered_keys(vertex_total, nullptr);
    for (const auto& [key, vertex] : keys_) {
      ordered_keys[vertex] = &key;
    }
    const int32_t reserved = reserve_output(loops, removable, index, ordered_keys);
    if (reserved != PHX_MC_OK) return reserved;
    int32_t used = 0;
    for (std::size_t v = 0; v < vertex_total; ++v) {
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
    }
    for (std::size_t f = 0; f < loops.size(); ++f) {
      for (const int32_t v : loops[f]) {
        if (removable[v] == 0) {
          out_.face_vertices.push_back(index[v]);
        }
      }
      out_.face_offsets.push_back(static_cast<int64_t>(out_.face_vertices.size()));
      out_.face_cells.push_back(owners[f][0]);
      out_.face_cells.push_back(owners[f][1]);
      out_.face_facets.push_back(owners[f][2]);
      out_.face_remote_sites.push_back(owners[f][3]);
      out_.face_remote_tets.push_back(owners[f][4]);
    }
    for (std::size_t p = 0; p < pieces_.size(); ++p) {
      if (output_cells_[piece_cell_[p]] < 0) continue;
      out_.piece_sites.push_back(pieces_[p].site);
      out_.piece_tets.push_back(pieces_[p].tet);
      out_.piece_cells.push_back(output_cells_[piece_cell_[p]]);
      out_.piece_volumes.push_back(pieces_[p].volume);
      out_.piece_moments.insert(out_.piece_moments.end(), pieces_[p].moment,
                                pieces_[p].moment + 3);
    }
    return PHX_MC_OK;
  }
};

}  // namespace
}  // namespace phx::mc

struct phx_mc_power_cells {
  phx::mc::PowerCellsResult result;
};

namespace phx::mc {
namespace {


struct ExactImageCoordinates {
  std::vector<Approx> approximate;
  std::vector<Expansion> exact;
  std::vector<ExactPowerCoordinates> coordinates;
  std::vector<double> weights;

  explicit ExactImageCoordinates(int64_t count)
      : approximate(static_cast<std::size_t>(3 * count)),
        exact(static_cast<std::size_t>(3 * count)),
        coordinates(static_cast<std::size_t>(count)),
        weights(static_cast<std::size_t>(count)) {}
};

int32_t validate_exact_image_layout(const phx_mc_restricted_power_cells_exact_request& in) {
  if (in.original_site_count < 1 || in.original_site_count >= INT32_MAX ||
      in.image_count < 1 || in.image_count >= INT32_MAX || in.component_count < 0 ||
      in.original_sites == nullptr || in.original_weights == nullptr ||
      in.image_site_owners == nullptr || in.image_coordinate_offsets == nullptr ||
      in.failed_image_coordinate == nullptr || in.failed_component == nullptr ||
      (in.component_count != 0 && in.image_coordinate_components == nullptr) ||
      in.selected_site_count < -1 || in.selected_site_count > in.image_count ||
      (in.selected_site_count > 0 && in.selected_sites == nullptr) ||
      !addressable(in.original_site_count, 3, sizeof(double)) ||
      !addressable(in.original_site_count, 1, sizeof(double)) ||
      !addressable(in.image_count, 1, sizeof(int32_t)) ||
      !addressable(3 * in.image_count + 1, 1, sizeof(int64_t)) ||
      !addressable(in.component_count, 1, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return PHX_MC_OK;
}
int32_t refuse_image_source(const phx_mc_restricted_power_cells_exact_request& in,
                            int64_t coordinate, int64_t component) {
  *in.failed_image_coordinate = coordinate;
  *in.failed_component = component;
  return PHX_MC_INVALID_INPUT;
}

// Degree-six coordinate polynomials require the same 2^-172 quantum and
// absolute envelope as the authored sites, even for cancelling expansions.
int32_t validate_image_coordinate(const phx_mc_restricted_power_cells_exact_request& in,
                                  int64_t coordinate) {
  const int64_t first = in.image_coordinate_offsets[coordinate];
  const int64_t last = in.image_coordinate_offsets[coordinate + 1];
  if (first < 0 || last < first || last > in.component_count) return PHX_MC_INVALID_ARGUMENT;
  Expansion absolute_sum;
  double preceding = 0.0;
  for (int64_t entry = first; entry < last; ++entry) {
    const double value = in.image_coordinate_components[entry];
    const double magnitude = std::abs(value);
    const double quantum = std::ldexp(value, 172);
    if (!std::isfinite(value) || value == 0.0 ||
        magnitude > kCoordinateMaxMagnitude || !std::isfinite(quantum) ||
        quantum != std::trunc(quantum)) return refuse_image_source(in, coordinate, entry);
    if (entry != first &&
        preceding > (magnitude - std::nextafter(magnitude, 0.0)) * 0.5) {
      return refuse_image_source(in, coordinate, entry);
    }
    preceding = magnitude;
    absolute_sum = absolute_sum + Expansion(magnitude);
    if ((absolute_sum - Expansion(kCoordinateMaxMagnitude)).sign() > 0) {
      return refuse_image_source(in, coordinate, entry);
    }
  }
  return PHX_MC_OK;
}

int32_t validate_exact_image_source(const phx_mc_restricted_power_cells_exact_request& in) {
  for (int64_t site = 0; site < in.original_site_count; ++site) {
    if (!weight_in_domain(in.original_weights[site])) return PHX_MC_INVALID_INPUT;
    for (int axis = 0; axis < 3; ++axis)
      if (!coordinate_in_domain(in.original_sites[3 * site + axis])) return PHX_MC_INVALID_INPUT;
  }
  if (in.image_coordinate_offsets[0] != 0 ||
      in.image_coordinate_offsets[3 * in.image_count] != in.component_count) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t image = 0; image < in.image_count; ++image) {
    if (in.image_site_owners[image] < 0 ||
        in.image_site_owners[image] >= in.original_site_count) return PHX_MC_INVALID_ARGUMENT;
    for (int axis = 0; axis < 3; ++axis) {
      const int64_t coordinate = 3 * image + axis;
      const int32_t status = validate_image_coordinate(in, coordinate);
      if (status != PHX_MC_OK) return status;
    }
  }
  return PHX_MC_OK;
}

void prepare_exact_image_coordinates(const phx_mc_restricted_power_cells_exact_request& in,
                                    ExactImageCoordinates& out) {
  for (int64_t image = 0; image < in.image_count; ++image) {
    out.weights[image] = in.original_weights[in.image_site_owners[image]];
    out.coordinates[image] = {out.approximate.data() + 3 * image, out.exact.data() + 3 * image};
    for (int axis = 0; axis < 3; ++axis) {
      const int64_t coordinate = 3 * image + axis;
      Approx filtered = Approx::exact(0.0);
      Expansion expanded;
      for (int64_t entry = in.image_coordinate_offsets[coordinate];
           entry < in.image_coordinate_offsets[coordinate + 1]; ++entry) {
        const double value = in.image_coordinate_components[entry];
        filtered = filtered + Approx::exact(value);
        expanded = expanded + Expansion(value);
      }
      out.approximate[coordinate] = filtered;
      out.exact[coordinate] = std::move(expanded);
    }
  }
}

int32_t execute_power_builder(const Input& input, PowerCellsResult& out) {
  const int32_t status = validate(input);
  if (status != PHX_MC_OK) return status;
  Builder builder(input, out);
  return builder.run();
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_restricted_power_cells(
    const phx_mc_restricted_power_cells_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    if (request->result == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *request->result = nullptr;
    const phx::mc::Input input{
        request->site_count, request->sites, request->weights, request->neighbor_offsets,
        request->neighbors, request->split_sites, request->point_count, request->points,
        request->tet_count, request->tets, request->tet_regions, request->tet_face_facets,
        nullptr};
    int32_t status = phx::mc::validate_layout(input);
    if (status != PHX_MC_OK) return status;
    auto result = std::make_unique<phx_mc_power_cells>();
    status = phx::mc::execute_power_builder(input, result->result);
    if (status == PHX_MC_OK || result->result.failure.reason != phx::mc::kNoFailure) {
      *request->result = result.release();
    }
    return status;
  });
}

// Exact image coordinates are sums, not rounded surrogate site triples.
// Image-axis site IDs remain in all construction witnesses; original ownership
// and authored action provenance are retained by the immutable source owner.
int32_t phx_mc_restricted_power_cells_exact(
    const phx_mc_restricted_power_cells_exact_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    using namespace phx::mc;
    const auto& in = *request;
    if (in.result == nullptr || in.failed_image_coordinate == nullptr ||
        in.failed_component == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *in.result = nullptr;
    *in.failed_image_coordinate = -1;
    *in.failed_component = -1;
    int32_t status = validate_exact_image_layout(in);
    if (status != PHX_MC_OK) return status;
    std::vector<int32_t> selected;
    if (in.selected_site_count > 0) {
      selected.assign(in.selected_sites, in.selected_sites + in.selected_site_count);
      std::sort(selected.begin(), selected.end());
      if (selected.front() < 0 || selected.back() >= in.image_count ||
          std::adjacent_find(selected.begin(), selected.end()) != selected.end())
        return PHX_MC_INVALID_INPUT;
    }
    status = validate_exact_image_source(in);
    if (status != PHX_MC_OK) return status;
    ExactImageCoordinates images(in.image_count);
    prepare_exact_image_coordinates(in, images);
    const Input input{
        in.image_count, nullptr, images.weights.data(), in.neighbor_offsets, in.neighbors,
        in.split_sites, in.point_count, in.points, in.tet_count, in.tets,
        in.tet_regions, in.tet_face_facets, images.coordinates.data(),
        in.selected_site_count == in.image_count ? -1 : in.selected_site_count,
        selected.data()};
    status = validate_layout(input);
    if (status != PHX_MC_OK) return status;
    auto result = std::make_unique<phx_mc_power_cells>();
    status = execute_power_builder(input, result->result);
    if (status == PHX_MC_OK || result->result.failure.reason != kNoFailure) {
      *in.result = result.release();
    }
    return status;
  });
}

int32_t phx_mc_power_cells_sizes(const phx_mc_power_cells* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& r = result->result;
  sizes[0] = static_cast<int64_t>(r.vertices.size() / 3);
  sizes[1] = static_cast<int64_t>(r.face_facets.size());
  sizes[2] = static_cast<int64_t>(r.face_vertices.size());
  sizes[3] = static_cast<int64_t>(r.cell_sites.size());
  sizes[4] = static_cast<int64_t>(r.piece_sites.size());
  return PHX_MC_OK;
}
int32_t phx_mc_power_cells_failure(const phx_mc_power_cells* result, int64_t* reason,
                                   int64_t* site, int64_t* tet, int32_t* clip_status) {
  if (result == nullptr || reason == nullptr || site == nullptr || tet == nullptr ||
      clip_status == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *reason = result->result.failure.reason;
  *site = result->result.failure.site;
  *tet = result->result.failure.tet;
  *clip_status = result->result.failure.clip_status;
  return PHX_MC_OK;
}
int32_t phx_mc_power_cells_export_face_remotes(
    const phx_mc_power_cells* result, int64_t face_capacity,
    int32_t* remote_sites, int32_t* remote_tets) {
  if (result == nullptr || face_capacity < 0 ||
      !phx::mc::addressable(face_capacity, 1, sizeof(int32_t)))
    return PHX_MC_INVALID_ARGUMENT;
  const auto& r = result->result;
  const int64_t faces = static_cast<int64_t>(r.face_remote_sites.size());
  if (face_capacity < faces) return PHX_MC_CAPACITY_EXCEEDED;
  if (faces > 0 && (remote_sites == nullptr || remote_tets == nullptr))
    return PHX_MC_INVALID_ARGUMENT;
  if (faces > 0) {
    std::copy(r.face_remote_sites.begin(), r.face_remote_sites.end(), remote_sites);
    std::copy(r.face_remote_tets.begin(), r.face_remote_tets.end(), remote_tets);
  }
  return PHX_MC_OK;
}

int64_t phx_mc_power_cells_site_witness_size(const phx_mc_power_cells* result) {
  return result == nullptr ? -1 : static_cast<int64_t>(result->result.vertex_sites.size());
}

int32_t phx_mc_power_cells_export_vertex_sites(
    const phx_mc_power_cells* result, int64_t vertex_offset_capacity,
    int64_t vertex_capacity, int64_t witness_capacity, int64_t* offsets, int32_t* sites) {
  if (result == nullptr || vertex_offset_capacity < 0 || vertex_capacity < 0 ||
      witness_capacity < 0 ||
      !phx::mc::addressable(vertex_offset_capacity, 1, sizeof(int64_t)) ||
      !phx::mc::addressable(witness_capacity, 1, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& r = result->result;
  const int64_t vertices = static_cast<int64_t>(r.vertex_site_offsets.size()) - 1;
  const int64_t witnesses = static_cast<int64_t>(r.vertex_sites.size());
  if (vertex_offset_capacity < static_cast<int64_t>(r.vertex_site_offsets.size()) ||
      vertex_capacity < vertices || witness_capacity < witnesses) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  if (offsets == nullptr || (witnesses > 0 && sites == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::copy(r.vertex_site_offsets.begin(), r.vertex_site_offsets.end(), offsets);
  if (witnesses > 0) std::copy(r.vertex_sites.begin(), r.vertex_sites.end(), sites);
  return PHX_MC_OK;
}

int32_t phx_mc_power_cells_export(
    const phx_mc_power_cells* result, int64_t vertex_capacity, int64_t face_offset_capacity,
    int64_t face_capacity, int64_t face_entry_capacity, int64_t cell_capacity,
    int64_t piece_capacity, double* vertices, int64_t* face_offsets, int32_t* face_vertices,
    int32_t* face_cells, int32_t* face_facets, int32_t* cell_sites, int32_t* cell_regions,
    double* cell_volumes, double* cell_moments, double* cell_second_moments,
    int32_t* piece_sites, int32_t* piece_tets, int32_t* piece_cells,
    double* piece_volumes, double* piece_moments) {
  if (result == nullptr || vertex_capacity < 0 || face_offset_capacity < 0 ||
      face_capacity < 0 || face_entry_capacity < 0 || cell_capacity < 0 ||
      piece_capacity < 0 ||
      !phx::mc::addressable(vertex_capacity, 3, sizeof(double)) ||
      !phx::mc::addressable(face_offset_capacity, 1, sizeof(int64_t)) ||
      !phx::mc::addressable(face_capacity, 2, sizeof(int32_t)) ||
      !phx::mc::addressable(face_capacity, 1, sizeof(int32_t)) ||
      !phx::mc::addressable(face_entry_capacity, 1, sizeof(int32_t)) ||
      !phx::mc::addressable(cell_capacity, 1, sizeof(int32_t)) ||
      !phx::mc::addressable(cell_capacity, 3, sizeof(double)) ||
      !phx::mc::addressable(piece_capacity, 1, sizeof(int32_t)) ||
      !phx::mc::addressable(piece_capacity, 3, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& r = result->result;
  const int64_t vertices_count = static_cast<int64_t>(r.vertices.size() / 3);
  const int64_t faces_count = static_cast<int64_t>(r.face_facets.size());
  const int64_t entries_count = static_cast<int64_t>(r.face_vertices.size());
  const int64_t cells_count = static_cast<int64_t>(r.cell_sites.size());
  const int64_t pieces_count = static_cast<int64_t>(r.piece_sites.size());
  if (vertex_capacity < vertices_count ||
      face_offset_capacity < static_cast<int64_t>(r.face_offsets.size()) ||
      face_capacity < faces_count || face_entry_capacity < entries_count ||
      cell_capacity < cells_count || piece_capacity < pieces_count) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  if ((vertices_count > 0 && vertices == nullptr) || face_offsets == nullptr ||
      (entries_count > 0 && face_vertices == nullptr) ||
      (faces_count > 0 && (face_cells == nullptr || face_facets == nullptr)) ||
      (cells_count > 0 && (cell_sites == nullptr || cell_regions == nullptr ||
                           cell_volumes == nullptr || cell_moments == nullptr ||
                           cell_second_moments == nullptr)) ||
      (pieces_count > 0 && (piece_sites == nullptr || piece_tets == nullptr ||
                            piece_cells == nullptr || piece_volumes == nullptr ||
                            piece_moments == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (vertices_count > 0) std::copy(r.vertices.begin(), r.vertices.end(), vertices);
  std::copy(r.face_offsets.begin(), r.face_offsets.end(), face_offsets);
  if (entries_count > 0) std::copy(r.face_vertices.begin(), r.face_vertices.end(), face_vertices);
  if (faces_count > 0) {
    std::copy(r.face_cells.begin(), r.face_cells.end(), face_cells);
    std::copy(r.face_facets.begin(), r.face_facets.end(), face_facets);
  }
  if (cells_count > 0) {
    std::copy(r.cell_sites.begin(), r.cell_sites.end(), cell_sites);
    std::copy(r.cell_regions.begin(), r.cell_regions.end(), cell_regions);
    std::copy(r.cell_volumes.begin(), r.cell_volumes.end(), cell_volumes);
    std::copy(r.cell_moments.begin(), r.cell_moments.end(), cell_moments);
    std::copy(r.cell_second_moments.begin(), r.cell_second_moments.end(), cell_second_moments);
  }
  if (pieces_count > 0) {
    std::copy(r.piece_sites.begin(), r.piece_sites.end(), piece_sites);
    std::copy(r.piece_tets.begin(), r.piece_tets.end(), piece_tets);
    std::copy(r.piece_cells.begin(), r.piece_cells.end(), piece_cells);
    std::copy(r.piece_volumes.begin(), r.piece_volumes.end(), piece_volumes);
    std::copy(r.piece_moments.begin(), r.piece_moments.end(), piece_moments);
  }
  return PHX_MC_OK;
}

int32_t phx_mc_power_cells_export_vertex_carriers(const phx_mc_power_cells* result,
                                                 int64_t vertex_capacity,
                                                 int32_t* vertex_carriers) {
  if (result == nullptr || vertex_capacity < 0 ||
      !phx::mc::addressable(vertex_capacity, 4, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& carriers = result->result.vertex_carriers;
  const int64_t vertices = static_cast<int64_t>(carriers.size() / 4);
  if (vertex_capacity < vertices) return PHX_MC_CAPACITY_EXCEEDED;
  if (vertices > 0 && vertex_carriers == nullptr) return PHX_MC_INVALID_ARGUMENT;
  if (!carriers.empty()) std::copy(carriers.begin(), carriers.end(), vertex_carriers);
  return PHX_MC_OK;
}

void phx_mc_power_cells_free(phx_mc_power_cells* result) { delete result; }

}  // extern "C"
