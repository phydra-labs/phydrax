//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral quality improvement: 2-3 face removal, edge removal by
// optimal ring triangulation (Shewchuk's edge removal; 3-2 and 4-4 flips are
// its three- and four-cell cases), exact facet/segment relocation, directional
// collapse, bounded multiface reconnection and protected weighted exudation.
// The default objective is minimum physical dihedral; accepted improvement
// strictly raises that objective on the replaced cavity. Charts at or below
// the owning relative determinant floor use the exact construction objective
// and, when no reconnection exists, one certified interior edge-star vertex.
// Tensor metric owners rank proposals independently and use the same
// geometry/topology transaction. Exact source strata, region labels and
// protected points survive every stage.
#include "improve3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "capi_guard.hpp"
#include "exact_line.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "tet_mesh.hpp"

namespace phx::mc {
namespace {

using Cell = std::array<int32_t, 4>;

// Smallest strict gain (degrees) that counts as an improvement.
constexpr double kImprovement = 1e-9;
constexpr int kMaxRing = 7;

enum Counter : int {
  kFaceRemovals = 0,
  kEdgeRemovals,
  kRelocations,
  kVertexRemovals,
  kPasses,
  kAttempts,
  kSlivers,
  kWork,
  kInsertions,
  kMultifaceRemovals,
};

double quality(const TetMesh& mesh, const int32_t* v) {
  const double* p[4];
  mesh.corners(v, p);
  return geometry::minimum_dihedral(p);
}

double finite_quality(const TetMesh& mesh, std::span<const int32_t> tets) {
  double worst = 180.0;
  for (int32_t t : tets) {
    if (mesh.finite(t)) {
      worst = std::min(worst, quality(mesh, mesh.tet(t).v));
    }
  }
  return worst;
}

double proposal_quality(const TetMesh& mesh, std::span<const Cell> cells) {
  double worst = 180.0;
  for (const Cell& cell : cells) {
    worst = std::min(worst, quality(mesh, cell.data()));
  }
  return worst;
}

bool representable(const double* p) {
  return coordinate_in_domain(p[0]) && coordinate_in_domain(p[1]) && coordinate_in_domain(p[2]);
}

// Vertices (other than v) of the constrained faces incident to v.
void facet_neighbors(const TetMesh& mesh, int32_t v, std::span<const int32_t> star,
                     NativeVector<int32_t>& neighbors) {
  neighbors.clear();
  for (int32_t t : star) {
    const Tetrahedron& cell = mesh.tet(t);
    for (int k = 0; k < 4; ++k) {
      if (cell.v[k] == v || mesh.complex().constraint(t, k) == kNoConstraint) {
        continue;
      }
      for (int j = 0; j < 4; ++j) {
        if (j != k && cell.v[j] != v && cell.v[j] >= 0) {
          neighbors.push_back(cell.v[j]);
        }
      }
    }
  }
  std::sort(neighbors.begin(), neighbors.end());
  neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
}

// Every incident constrained plane must be retained, including folded
// interfaces whose source IDs happen to agree.
bool on_facet_plane(TetMesh& mesh, int32_t v, std::span<const int32_t> star,
                    const double* position, NativeVector<int32_t>& neighbors) {
  facet_neighbors(mesh, v, star, neighbors);
  for (int32_t t : star) {
    if (!mesh.finite(t)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      if (mesh.tet(t).v[k] == v || mesh.complex().constraint(t, k) == kNoConstraint) {
        continue;
      }
      int32_t f[3];
      oriented_facet(mesh.tet(t).v, k, f);
      if (!mesh.spend(4)) {
        return false;
      }
      // A conforming face may span several original source triangles. Its
      // actual represented plane, scientific source and region-side marks are
      // retained; positive star cells below preserve the oriented planar fan
      // with the same outer link, hence exactly the same source patch union.
      if (orient3d(mesh.point(f[0]), mesh.point(f[1]), mesh.point(f[2]), position) != 0) {
        return false;
      }
    }
  }
  return true;
}

// Smallest dihedral angle of the star with v at `position`, or -1 when a
// finite cell would not be positively oriented.  Coordinates are restored.
double star_quality_at(TetMesh& mesh, int32_t v, std::span<const int32_t> star,
                       const double* position) {
  double* p = mesh.mutable_point(v);
  struct RestoreCoordinates {
    double* point;
    double saved[3];
    ~RestoreCoordinates() { std::copy_n(saved, 3, point); }
  } restore{p, {p[0], p[1], p[2]}};
  std::copy_n(position, 3, p);
  double worst = 180.0;
  for (int32_t t : star) {
    native_execution_charge(0);
    if (!mesh.finite(t)) {
      continue;
    }
    if (!mesh.positive(mesh.tet(t).v)) {
      worst = -1.0;
      break;
    }
    worst = std::min(worst, quality(mesh, mesh.tet(t).v));
  }
  return worst;
}

bool remove_vertex_above(TetMesh& mesh, int32_t vertex, bool require_improvement,
                         double competing_quality, int32_t* accepted_neighbor);

bool smooth_vertex(TetMesh& mesh, int32_t v, bool& removed, int32_t& accepted_neighbor) {
  removed = false;
  const int8_t dimension = mesh.dimension(v);
  NativeVector<int32_t> star;
  NativeVector<int32_t> neighbors;
  if (dimension < 1 || mesh.protection(v) > 0.0 ||
      (dimension < 3 && mesh.policy() == BoundaryPolicy::kFixed) || !mesh.vertex_star(v, star)) {
    return false;
  }
  if (!native_execution_cavity(star.size())) return false;
  if (dimension == 1) {
    int32_t ends[2] = {-1, -1};
    int32_t source = -1;
    int count = 0;
    for (const auto& [edge, identity] : mesh.segments()) {
      if (!mesh.spend(1)) return false;
      const int32_t a = edge_low(edge), b = edge_high(edge);
      if (a != v && b != v) continue;
      if (count == 2 || (source >= 0 && source != identity)) return false;
      source = identity;
      ends[count++] = a == v ? b : a;
    }
    if (count != 2) return false;
    const double* first = mesh.point(ends[0]);
    const double* second = mesh.point(ends[1]);
    if (!collinear3d(first, second, mesh.point(v))) return false;
    // Exact ancestry uses the authored carrier. Bounded ancestry retains its
    // represented carrier; relocation_witness admits it against the original
    // source bound before any coordinates or witness are committed.
    const double* carrier_first = first;
    const double* carrier_second = second;
    for (const auto& segment : mesh.original_segments()) {
      if (!mesh.spend(1)) return false;
      if (segment[2] != source) continue;
      const double* a = mesh.original_point(segment[0]);
      const double* b = mesh.original_point(segment[1]);
      if (collinear3d(a, b, first) && collinear3d(a, b, second)) {
        carrier_first = a;
        carrier_second = b;
        break;
      }
    }
    const double baseline = finite_quality(mesh, star);
    double best_quality = baseline;
    std::array<double, 3> best{};
    static constexpr double fractions[] = {0.25, 0.5, 0.75};
    for (const double* endpoint : {first, second}) {
      for (double fraction : fractions) {
        if (!mesh.spend(static_cast<int64_t>(4 * star.size()))) return false;
        double desired[3], candidate[3], coordinate = 0.0;
        for (int k = 0; k < 3; ++k) {
          desired[k] = mesh.point(v)[k] + fraction * (endpoint[k] - mesh.point(v)[k]);
        }
        if (!construct_exact_line_point(
                carrier_first, carrier_second, first, second, desired, candidate, &coordinate,
                [](void* context, int64_t units) {
                  return static_cast<TetMesh*>(context)->spend(units);
                }, &mesh)) {
          if (mesh.work_exhausted()) return false;
          continue;
        }
        if (!on_facet_plane(mesh, v, star, candidate, neighbors)) continue;
        SourceWitness witness;
        if (mesh.relocation_witness(v, candidate, witness) != Insertion::kOk) {
          if (mesh.work_exhausted()) return false;
          continue;
        }
        const double q = star_quality_at(mesh, v, star, candidate);
        if (mesh.work_exhausted()) return false;
        if (q > best_quality + kImprovement) {
          best_quality = q;
          std::copy_n(candidate, 3, best.begin());
        }
      }
    }
    return best_quality > baseline + kImprovement && try_relocate(mesh, v, best.data(), true);
  }
  const double* p = mesh.point(v);
  const double origin[3] = {p[0], p[1], p[2]};
  if (dimension == 2) {
    facet_neighbors(mesh, v, star, neighbors);
  } else {
    for (int32_t t : star) {
      for (int32_t w : mesh.tet(t).v) {
        if (w != v && w >= 0) {
          neighbors.push_back(w);
        }
      }
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  }
  if (neighbors.empty()) {
    return false;
  }
  double centroid[3] = {0.0, 0.0, 0.0};
  bool shared[3] = {true, true, true};
  for (int32_t w : neighbors) {
    for (int i = 0; i < 3; ++i) {
      centroid[i] += mesh.point(w)[i];
      shared[i] = shared[i] && mesh.point(w)[i] == origin[i];
    }
  }
  for (int i = 0; i < 3; ++i) {
    centroid[i] /= static_cast<double>(neighbors.size());
  }
  // Candidate targets: the link centroid and, for interior vertices, the apex
  // position that would make the worst incident cell regular.
  NativeVector<std::array<double, 3>> targets;
  targets.push_back({centroid[0], centroid[1], centroid[2]});
  if (dimension == 3) {
    int32_t worst = -1;
    double worst_quality = 181.0;
    for (int32_t t : star) {
      const double q = quality(mesh, mesh.tet(t).v);
      if (q < worst_quality) {
        worst_quality = q;
        worst = t;
      }
    }
    const Tetrahedron& cell = mesh.tet(worst);
    const int slot = vertex_slot(cell, v);
    int32_t f[3];
    oriented_facet(cell.v, slot, f);
    const double* a = mesh.point(f[0]);
    const double* b = mesh.point(f[1]);
    const double* c = mesh.point(f[2]);
    double u[3], w[3], n[3];
    geometry::difference(b, a, u);
    geometry::difference(c, a, w);
    geometry::cross(u, w, n);
    const double norm = std::sqrt(geometry::dot(n, n));
    const double edge = (std::sqrt(geometry::squared_distance(a, b)) +
                         std::sqrt(geometry::squared_distance(b, c)) +
                         std::sqrt(geometry::squared_distance(c, a))) /
                        3.0;
    if (norm > 0.0) {
      const double height = std::sqrt(2.0 / 3.0) * edge / norm;
      targets.push_back({(a[0] + b[0] + c[0]) / 3.0 + height * n[0],
                         (a[1] + b[1] + c[1]) / 3.0 + height * n[1],
                         (a[2] + b[2] + c[2]) / 3.0 + height * n[2]});
    }
  }
  if (!mesh.spend(static_cast<int64_t>(8 * star.size()))) return false;
  const double baseline = finite_quality(mesh, star);
  double best_quality = baseline;
  double best[3] = {origin[0], origin[1], origin[2]};
  static constexpr double kSteps[4] = {1.0, 0.5, 0.25, 0.125};
  for (const auto& target : targets) {
    for (double step : kSteps) {
      native_execution_charge(0);
      double candidate[3];
      for (int i = 0; i < 3; ++i) {
        candidate[i] = origin[i] + step * (target[i] - origin[i]);
        if (dimension == 2 && shared[i]) {
          candidate[i] = origin[i];
        }
      }
      if (!representable(candidate) ||
          (dimension == 2 && !on_facet_plane(mesh, v, star, candidate, neighbors))) {
        continue;
      }
      const double q = star_quality_at(mesh, v, star, candidate);
      if (mesh.work_exhausted()) return false;
      if (q > best_quality + kImprovement) {
        best_quality = q;
        std::copy_n(candidate, 3, best);
      }
    }
  }
  if (dimension == 3 &&
      remove_vertex_above(mesh, v, true, best_quality, &accepted_neighbor)) {
    removed = true;
    return true;
  }
  if (mesh.work_exhausted()) return false;
  return best_quality > baseline + kImprovement && try_relocate(mesh, v, best, true);
}

bool remove_cell_face_cluster(TetMesh& mesh, int32_t cell) {
  if (!native_execution_cavity(1)) return false;
  NativeVector<int32_t> cavity{cell};
  const int32_t region = mesh.complex().region(cell);
  std::size_t begin = 0;
  for (int depth = 0; depth < 2; ++depth) {
    const std::size_t end = cavity.size();
    for (std::size_t cursor = begin; cursor < end; ++cursor) {
      const int32_t current = cavity[cursor];
      for (int slot = 0; slot < 4; ++slot) {
        native_execution_charge(0);
        const int32_t other = mesh.tet(current).n[slot];
        if (mesh.complex().constraint(current, slot) == kNoConstraint && mesh.finite(other) &&
            mesh.complex().region(other) == region &&
            std::find(cavity.begin(), cavity.end(), other) == cavity.end()) {
          if (!native_execution_cavity(cavity.size() + 1)) return false;
          cavity.push_back(other);
        }
      }
    }
    begin = end;
  }
  if (cavity.size() < 2) return false;
  NativeVector<int32_t> apices;
  for (int32_t entry : cavity) {
    apices.insert(apices.end(), mesh.tet(entry).v, mesh.tet(entry).v + 4);
  }
  std::sort(apices.begin(), apices.end());
  apices.erase(std::unique(apices.begin(), apices.end()), apices.end());
  for (int32_t apex : apices) {
    if (try_multiface_removal(mesh, cavity, apex, true)) return true;
    if (mesh.broken() || mesh.work_exhausted()) return false;
  }
  return false;
}

struct ConstructionAssessment {
  Approx orientation;
  double score;
  bool below_floor;
  double floor_margin;
};

// `corners[k]` is the position of `vertices[k]`; the chart is the canonical
// vertex-identity order that publication and its independent audit retain.
ConstructionAssessment assess_chart(const int32_t* vertices, const double* const* corners,
                                    double minimum_relative_determinant) {
  int perm[4];
  canonical_permutation(vertices, 4, perm);
  const double* p[4];
  for (int k = 0; k < 4; ++k) p[k] = corners[perm[k]];
  // The physical measure owner expands the determinant along Cartesian x.
  // An exact cyclic axis permutation reuses the orientation owner's z-row
  // expansion with that same arithmetic; no recentering or scalar clamping.
  double axes[4][3];
  for (int corner = 0; corner < 4; ++corner) {
    axes[corner][0] = p[corner][1];
    axes[corner][1] = p[corner][2];
    axes[corner][2] = p[corner][0];
  }
  native_execution_primitive_query();
  const Approx orientation = -orient3d_approx(axes[1], axes[2], axes[3], axes[0]);
  double score = orientation.value <= 0.0 ? -1.0
                 : orientation.bound == 0.0 ? std::numeric_limits<double>::infinity()
                                           : orientation.value / orientation.bound;
  bool below_floor = false;
  double floor_margin = 0.0;
  if (minimum_relative_determinant > 0.0) {
    double floor_score;
    below_floor = canonical_relative_floor(
        vertices, corners, minimum_relative_determinant, &floor_score) <= 0;
    score = std::min(score, floor_score);
    floor_margin = floor_score - 1.0;
  }
  return {orientation, score, below_floor, floor_margin};
}

ConstructionAssessment assess_construction(const TetMesh& mesh, const int32_t* vertices,
                                           double minimum_relative_determinant) {
  const double* corners[4];
  mesh.corners(vertices, corners);
  return assess_chart(vertices, corners, minimum_relative_determinant);
}

// Whether every proposed cell is above the owning floor (0 decides nothing).
bool above_floor(const TetMesh& mesh, std::span<const Cell> cells, double floor) {
  if (!(floor > 0.0)) return true;
  for (const Cell& cell : cells) {
    if (assess_construction(mesh, cell.data(), floor).below_floor) return false;
  }
  return true;
}

// A below-floor chart spanning two nearly coplanar constrained faces at a
// hinge has its whole interior dihedral angle at that hinge. Every
// reconnection keeps one chart over that plane, so the repair replaces the
// star of one of its unprotected edges by a new interior vertex. Candidates
// move from the edge midpoint toward the star ring centroid; a candidate is
// committed only when every child chart is certified positive and above the
// owning floor, and the best smallest dihedral angle is chosen.
enum class FloorInsertion { kApplied, kRefused, kCapacity };

FloorInsertion insert_floor_vertex(TetMesh& mesh, const Cell& cell,
                                   double minimum_relative_determinant,
                                   std::array<int32_t, 5>& affected_vertices) {
  if (!(minimum_relative_determinant > 0.0)) return FloorInsertion::kRefused;
  NativeVector<int32_t> star;
  NativeVector<int32_t> ring;
  static constexpr double kSteps[5] = {1.0, 0.5, 0.25, 0.125, 0.0625};
  const int32_t created = static_cast<int32_t>(mesh.vertex_count());
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      const int32_t a = cell[static_cast<std::size_t>(i)], b = cell[static_cast<std::size_t>(j)];
      if (mesh.segment_source(a, b) >= 0 || !mesh.edge_ring(a, b, star, ring)) {
        if (mesh.work_exhausted()) return FloorInsertion::kRefused;
        continue;
      }
      if (ring.empty() || std::find(ring.begin(), ring.end(), -1) != ring.end() ||
          !mesh.spend(static_cast<int64_t>(16 * star.size()))) {
        continue;
      }
      double midpoint[3], centroid[3] = {0.0, 0.0, 0.0};
      for (int axis = 0; axis < 3; ++axis) {
        midpoint[axis] = 0.5 * (mesh.point(a)[axis] + mesh.point(b)[axis]);
      }
      for (int32_t r : ring) {
        for (int axis = 0; axis < 3; ++axis) centroid[axis] += mesh.point(r)[axis];
      }
      for (int axis = 0; axis < 3; ++axis) centroid[axis] /= static_cast<double>(ring.size());
      double best_quality = -1.0;
      double best[3];
      for (double step : kSteps) {
        double position[3];
        for (int axis = 0; axis < 3; ++axis) {
          position[axis] = midpoint[axis] + step * (centroid[axis] - midpoint[axis]);
        }
        if (!representable(position)) continue;
        // Protected balls of the star vertices refuse the insertion itself;
        // rank only admissible candidates.
        bool protected_ball = false;
        for (int32_t t : star) {
          for (int32_t vertex : mesh.tet(t).v) {
            const double radius = mesh.protection(vertex);
            protected_ball = protected_ball ||
                (radius > 0.0 &&
                 geometry::squared_distance(mesh.point(vertex), position) < radius * radius);
          }
        }
        if (protected_ball) continue;
        double worst = 180.0;
        for (int32_t t : star) {
          for (int32_t endpoint : {a, b}) {
            native_execution_charge(0);
            int32_t child[4];
            const double* corners[4];
            std::copy_n(mesh.tet(t).v, 4, child);
            const int slot = vertex_slot(mesh.tet(t), endpoint);
            child[slot] = created;
            mesh.corners(mesh.tet(t).v, corners);
            corners[slot] = position;
            const ConstructionAssessment assessment =
                assess_chart(child, corners, minimum_relative_determinant);
            if (assessment.below_floor || assessment.orientation.certified_sign() != 1 ||
                orient3d(corners[0], corners[1], corners[2], corners[3]) <= 0) {
              worst = -1.0;
              break;
            }
            worst = std::min(worst, geometry::minimum_dihedral(corners));
          }
          if (worst < 0.0) break;
        }
        if (worst > best_quality) {
          best_quality = worst;
          std::copy_n(position, 3, best);
        }
      }
      if (best_quality < 0.0) continue;
      int32_t inserted = -1;
      const double size = 0.5 * (mesh.size(a) + mesh.size(b));
      switch (mesh.insert_edge_star_vertex(a, b, best, size, inserted)) {
        case Insertion::kOk:
          affected_vertices[4] = inserted;
          return FloorInsertion::kApplied;
        case Insertion::kCapacity:
          return FloorInsertion::kCapacity;
        default:
          if (mesh.broken() || mesh.work_exhausted()) return FloorInsertion::kRefused;
          break;
      }
    }
  }
  return FloorInsertion::kRefused;
}

bool improve_cell(TetMesh& mesh, int32_t t, int64_t* counters, bool construction_uncertain,
                  bool below_floor, double minimum_relative_determinant,
                  std::array<int32_t, 5>& affected_vertices, bool& capacity) {
  const Cell v = {mesh.tet(t).v[0], mesh.tet(t).v[1], mesh.tet(t).v[2], mesh.tet(t).v[3]};
  std::copy(v.begin(), v.end(), affected_vertices.begin());
  affected_vertices[4] = -1;
  for (int k = 0; k < 4; ++k) {
    if (try_face_removal(mesh, t, k, true)) {
      ++counters[kFaceRemovals];
      return true;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (try_edge_removal(mesh, v[i], v[j], true, std::nullopt,
                           std::numeric_limits<double>::infinity(), 0.25,
                           construction_uncertain, minimum_relative_determinant)) {
        ++counters[kEdgeRemovals];
        return true;
      }
    }
  }
  if (remove_cell_face_cluster(mesh, t)) {
    ++counters[kMultifaceRemovals];
    return true;
  }
  for (int32_t vertex : v) {
    bool removed = false;
    if (smooth_vertex(mesh, vertex, removed, affected_vertices[4])) {
      ++counters[removed ? kVertexRemovals : kRelocations];
      return true;
    }
  }
  for (int32_t vertex : v) {
    if (try_remove_vertex(mesh, vertex, true, &affected_vertices[4])) {
      ++counters[kVertexRemovals];
      return true;
    }
  }
  if (below_floor) {
    switch (insert_floor_vertex(mesh, v, minimum_relative_determinant, affected_vertices)) {
      case FloorInsertion::kApplied:
        ++counters[kInsertions];
        return true;
      case FloorInsertion::kCapacity:
        capacity = true;
        return false;
      case FloorInsertion::kRefused:
        break;
    }
  }
  return false;
}

struct Candidate {
  double quality;
  int32_t slot;
  int32_t v[4];  // sorted
  bool construction_uncertain = false;
  double construction_margin = 0.0;
  bool below_floor = false;
  bool operator<(const Candidate& other) const {
    const bool finite = std::isfinite(quality), other_finite = std::isfinite(other.quality);
    if (finite != other_finite) return !finite;
    if (finite && quality != other.quality) {
      return quality < other.quality;
    }
    return std::lexicographical_compare(v, v + 4, other.v, other.v + 4);
  }
};

bool current(const TetMesh& mesh, const Candidate& entry) {
  if (!mesh.complex().live(entry.slot) || !mesh.finite(entry.slot)) {
    return false;
  }
  int32_t v[4];
  std::copy_n(mesh.tet(entry.slot).v, 4, v);
  std::sort(v, v + 4);
  return std::equal(v, v + 4, entry.v);
}

bool collect_slivers(TetMesh& mesh, double target, double minimum_relative_determinant,
                     NativeVector<Candidate>& slivers) try {
  slivers.clear();
  const auto slots = static_cast<int32_t>(mesh.complex().tets.size());
  for (int32_t t = 0; t < slots; ++t) {
    native_execution_charge(0);
    if (!mesh.complex().live(t) || !mesh.finite(t)) {
      continue;
    }
    const double q = quality(mesh, mesh.tet(t).v);
    const ConstructionAssessment assessment = assess_construction(
        mesh, mesh.tet(t).v, minimum_relative_determinant);
    const bool uncertain = assessment.orientation.certified_sign() != 1 || assessment.below_floor;
    if (uncertain || !std::isfinite(q) || q < target) {
      Candidate entry{q, t, {0, 0, 0, 0}, uncertain,
                      assessment.below_floor ? assessment.floor_margin
                          : assessment.orientation.value - assessment.orientation.bound,
                      assessment.below_floor};
      std::copy_n(mesh.tet(t).v, 4, entry.v);
      std::sort(entry.v, entry.v + 4);
      slivers.push_back(entry);
    }
  }
  std::sort(slivers.begin(), slivers.end(), [&](const Candidate& a, const Candidate& b) {
    native_execution_charge(0);
    return a < b;
  });
  return true;
} catch (const ExecutionRefusal&) {
  return false;
} catch (const std::bad_alloc&) {
  return false;
}

struct ConstructionFrontier {
  static constexpr std::size_t kAbsent = std::numeric_limits<std::size_t>::max();
  NativeVector<std::size_t> pending;
  NativeVector<int32_t> star;
  NativeVector<int32_t> cells;
  NativeVector<Candidate> affected;

  void begin(const TetMesh& mesh, const NativeVector<Candidate>& queue) {
    pending.assign(mesh.complex().tets.size(), kAbsent);
    for (std::size_t index = 0; index < queue.size(); ++index) {
      native_execution_charge(0);
      pending[static_cast<std::size_t>(queue[index].slot)] = index;
    }
  }

  bool append(TetMesh& mesh, const std::array<int32_t, 5>& vertices,
              double minimum_relative_determinant, NativeVector<Candidate>& queue) {
    cells.clear();
    for (int32_t vertex : vertices) {
      native_execution_charge(0);
      if (vertex < 0 || !mesh.live_vertex(vertex)) continue;
      if (!mesh.vertex_star(vertex, star)) {
        if (mesh.work_exhausted()) return false;
        continue;
      }
      cells.insert(cells.end(), star.begin(), star.end());
    }
    std::sort(cells.begin(), cells.end(), [](int32_t a, int32_t b) {
      native_execution_charge(0);
      return a < b;
    });
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    affected.clear();
    for (int32_t slot : cells) {
      native_execution_charge(0);
      if (!mesh.complex().live(slot) || !mesh.finite(slot)) continue;
      const ConstructionAssessment assessment = assess_construction(
          mesh, mesh.tet(slot).v, minimum_relative_determinant);
      if (assessment.orientation.certified_sign() == 1 && !assessment.below_floor) continue;
      Candidate entry{quality(mesh, mesh.tet(slot).v), slot, {0, 0, 0, 0}, true,
                      assessment.below_floor ? assessment.floor_margin
                          : assessment.orientation.value - assessment.orientation.bound,
                      assessment.below_floor};
      std::copy_n(mesh.tet(slot).v, 4, entry.v);
      std::sort(entry.v, entry.v + 4);
      affected.push_back(entry);
    }
    // Scheduling follows original vertex identity, never reusable slot order.
    std::sort(affected.begin(), affected.end(), [](const Candidate& a, const Candidate& b) {
      native_execution_charge(0);
      return std::lexicographical_compare(a.v, a.v + 4, b.v, b.v + 4);
    });
    pending.resize(mesh.complex().tets.size(), kAbsent);
    for (const Candidate& entry : affected) {
      native_execution_charge(0);
      std::size_t& index = pending[static_cast<std::size_t>(entry.slot)];
      if (index != kAbsent &&
          std::equal(entry.v, entry.v + 4, queue[index].v)) {
        queue[index] = entry;
      } else {
        index = queue.size();
        queue.push_back(entry);
      }
    }
    return true;
  }
};

bool within_size_targets(const TetMesh& mesh, const Cell& vertices, double ratio_bound) {
  const double* p[4];
  mesh.corners(vertices.data(), p);
  double center[3];
  if (!geometry::tetrahedron_circumcenter(p[0], p[1], p[2], p[3], center)) {
    return false;
  }
  const double radius2 = geometry::squared_distance(center, p[0]);
  double shortest2 = std::numeric_limits<double>::infinity();
  double target = 0.0;
  for (int i = 0; i < 4; ++i) {
    target += 0.25 * mesh.size(vertices[static_cast<std::size_t>(i)]);
    for (int j = i + 1; j < 4; ++j) {
      shortest2 = std::min(shortest2, geometry::squared_distance(p[i], p[j]));
    }
  }
  return std::isfinite(radius2) && radius2 <= ratio_bound * ratio_bound * shortest2 &&
         (target == 0.0 || radius2 <= target * target);
}

bool regular_proposal(TetMesh& mesh, std::span<const Cell> cells,
                      std::span<const double> weights, double ratio_bound,
                      double max_weight_fraction) {
  if (!mesh.spend(static_cast<int64_t>(8 * cells.size() * cells.size()))) {
    return false;
  }
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (!within_size_targets(mesh, cells[i], ratio_bound)) {
      return false;
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        const int32_t x = cells[i][static_cast<std::size_t>(a)];
        const int32_t y = cells[i][static_cast<std::size_t>(b)];
        const double bound = max_weight_fraction *
                             geometry::squared_distance(mesh.point(x), mesh.point(y));
        if (weights[static_cast<std::size_t>(x)] > bound ||
            weights[static_cast<std::size_t>(y)] > bound) {
          return false;
        }
      }
    }
    for (std::size_t j = i + 1; j < cells.size(); ++j) {
      int count = 0;
      int32_t opposite = -1;
      for (int32_t v : cells[j]) {
        if (std::find(cells[i].begin(), cells[i].end(), v) == cells[i].end()) {
          opposite = v;
        } else {
          ++count;
        }
      }
      if (count != 3) {
        continue;
      }
      const double* p[4];
      mesh.corners(cells[i].data(), p);
      if (power3d(p[0], p[1], p[2], p[3], mesh.point(opposite),
                   weights[static_cast<std::size_t>(cells[i][0])],
                   weights[static_cast<std::size_t>(cells[i][1])],
                   weights[static_cast<std::size_t>(cells[i][2])],
                   weights[static_cast<std::size_t>(cells[i][3])],
                   weights[static_cast<std::size_t>(opposite)]) > 0) {
        return false;
      }
    }
  }
  return true;
}

bool exude_vertex(TetMesh& mesh, int32_t vertex, const ExudeOptions& options,
                  std::span<double> weights, int64_t* counters) {
  if (mesh.dimension(vertex) != 3 || mesh.protection(vertex) > 0.0) {
    return false;
  }
  NativeVector<int32_t> star;
  if (!mesh.vertex_star(vertex, star)) {
    return false;
  }
  double shortest2 = std::numeric_limits<double>::infinity();
  for (int32_t t : star) {
    for (int32_t v : mesh.tet(t).v) {
      if (v != vertex && v >= 0) {
        shortest2 = std::min(shortest2,
                             geometry::squared_distance(mesh.point(vertex), mesh.point(v)));
      }
    }
  }
  const double bound = options.max_weight_fraction * shortest2;
  for (int exponent = -12; exponent <= 0; ++exponent) {
    const double trial = std::ldexp(bound, exponent);
    if (!(trial > weights[static_cast<std::size_t>(vertex)]) || !weight_in_domain(trial)) {
      continue;
    }
    for (int32_t t : star) {
      if (!mesh.spend(16)) {
        return false;
      }
      ++counters[0];
      const Tetrahedron old = mesh.tet(t);
      const int slot = vertex_slot(old, vertex);
      const int32_t other = old.n[slot];
      if (!mesh.finite(other) || mesh.complex().constraint(t, slot) != kNoConstraint) {
        continue;
      }
      const Tetrahedron neighbor = mesh.tet(other);
      const int back = neighbor_slot(neighbor, t);
      const double* p[4];
      mesh.corners(other, p);
      if (power3d(p[0], p[1], p[2], p[3], mesh.point(vertex),
                   weights[static_cast<std::size_t>(neighbor.v[0])],
                   weights[static_cast<std::size_t>(neighbor.v[1])],
                   weights[static_cast<std::size_t>(neighbor.v[2])],
                   weights[static_cast<std::size_t>(neighbor.v[3])], trial) <= 0) {
        continue;
      }
      const double new_edge_bound =
          options.max_weight_fraction *
          geometry::squared_distance(mesh.point(vertex), mesh.point(neighbor.v[back]));
      bool size_valid = trial <= new_edge_bound &&
                        weights[static_cast<std::size_t>(neighbor.v[back])] <= new_edge_bound;
      for (int k = 0; k < 4; ++k) {
        if (k == back) {
          continue;
        }
        Cell proposed = {neighbor.v[0], neighbor.v[1], neighbor.v[2], neighbor.v[3]};
        proposed[static_cast<std::size_t>(k)] = vertex;
        size_valid = size_valid && within_size_targets(mesh, proposed, options.radius_edge_bound) &&
                     above_floor(mesh, std::span<const Cell>(&proposed, 1),
                                 options.minimum_relative_determinant);
      }
      if (size_valid && try_face_removal(mesh, t, slot, true)) {
        weights[static_cast<std::size_t>(vertex)] = trial;
        ++counters[1];
        return true;
      }
    }
    // Regular exudation needs both directions of bistellar reconnection:
    // a sliver commonly disappears in a 3-2 edge removal, not a 2-3 flip.
    NativeVector<int32_t> neighbors;
    for (int32_t t : star) {
      for (int32_t v : mesh.tet(t).v) {
        if (v != vertex && v >= 0) {
          neighbors.push_back(v);
        }
      }
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    const double saved_weight = weights[static_cast<std::size_t>(vertex)];
    weights[static_cast<std::size_t>(vertex)] = trial;
    for (int32_t other : neighbors) {
      ++counters[0];
      if (try_edge_removal(mesh, vertex, other, true, weights, options.radius_edge_bound,
                            options.max_weight_fraction, false,
                            options.minimum_relative_determinant)) {
        ++counters[1];
        return true;
      }
    }
    weights[static_cast<std::size_t>(vertex)] = saved_weight;
  }
  return false;
}

}  // namespace

bool try_face_removal(TetMesh& mesh, int32_t t, int slot, bool require_improvement) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (t < 0 || static_cast<std::size_t>(t) >= mesh.complex().tets.size() ||
      !mesh.complex().live(t) || slot < 0 || slot > 3 || !mesh.finite(t) ||
      mesh.complex().constraint(t, slot) != kNoConstraint) {
    return false;
  }
  const Tetrahedron first = mesh.tet(t);
  const int32_t u = first.n[slot];
  if (!mesh.finite(u)) {
    return false;
  }
  const Tetrahedron second = mesh.tet(u);
  const int back = neighbor_slot(second, t);
  NativeVector<Cell> proposed;
  for (int j = 0; j < 4; ++j) {
    if (j == back) {
      continue;
    }
    Cell cell = {second.v[0], second.v[1], second.v[2], second.v[3]};
    cell[static_cast<std::size_t>(j)] = first.v[slot];
    if (!mesh.positive(cell.data())) {
      return false;
    }
    proposed.push_back(cell);
  }
  if (!mesh.spend(6)) {
    return false;
  }
  if (require_improvement) {
    const double before = std::min(quality(mesh, first.v), quality(mesh, second.v));
    if (!(proposal_quality(mesh, proposed) > before + kImprovement)) {
      return false;
    }
  }
  const int32_t removed[] = {t, u};
  return mesh.replace(removed, proposed) == EditStatus::kOk;
}

bool try_edge_removal(TetMesh& mesh, int32_t a, int32_t b, bool require_improvement,
                      std::optional<std::span<const double>> regular_weights, double radius_edge_bound,
                      double max_weight_fraction, bool construction_objective,
                      double minimum_relative_determinant) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (!mesh.live_vertex(a) || !mesh.live_vertex(b) || a == b || mesh.segment_source(a, b) >= 0) {
    return false;
  }
  if (regular_weights &&
      regular_weights->size() != static_cast<std::size_t>(mesh.vertex_count())) {
    return false;
  }
  NativeVector<int32_t> tets;
  NativeVector<int32_t> ring;
  if (!mesh.edge_ring(a, b, tets, ring)) {
    return false;
  }
  const int n = static_cast<int>(ring.size());
  if (n < 3 || n > kMaxRing) {
    return false;
  }
  if (!mesh.spend(static_cast<int64_t>(n * n * n))) return false;
  for (int i = 0; i < n; ++i) {
    // tets[i] = (a, b, ring[i], ring[i + 1]); its facet opposite ring[i] is
    // the ring face (a, b, ring[i + 1]).
    if (ring[static_cast<std::size_t>(i)] < 0 ||
        mesh.complex().constraint(tets[static_cast<std::size_t>(i)],
                                  vertex_slot(mesh.tet(tets[static_cast<std::size_t>(i)]),
                                              ring[static_cast<std::size_t>(i)])) !=
            kNoConstraint) {
      return false;
    }
  }
  // Triangle (i, k, j) of the ring polygon, i < k < j, yields the cells
  // (r_k, r_i, r_j, a) and (r_i, r_k, r_j, b); -1 when either is invalid.
  const auto r = [&](int index) { return ring[static_cast<std::size_t>(index)]; };
  const auto cells_of = [&](int i, int k, int j, Cell& above, Cell& below) {
    above = {r(k), r(i), r(j), a};
    below = {r(i), r(k), r(j), b};
  };
  const auto construction_quality = [&](const int32_t* vertices) {
    return assess_construction(mesh, vertices, minimum_relative_determinant).score;
  };
  const auto triangle = [&](int i, int k, int j) {
    Cell above;
    Cell below;
    cells_of(i, k, j, above, below);
    if (!mesh.positive(above.data()) || !mesh.positive(below.data())) {
      return -1.0;
    }
    if (construction_objective) {
      return std::min(construction_quality(above.data()), construction_quality(below.data()));
    }
    return std::min(quality(mesh, above.data()), quality(mesh, below.data()));
  };
  double best[kMaxRing][kMaxRing];
  int choice[kMaxRing][kMaxRing];
  for (int length = 2; length < n; ++length) {
    for (int i = 0; i + length < n; ++i) {
      const int j = i + length;
      best[i][j] = -1.0;
      choice[i][j] = -1;
      for (int k = i + 1; k < j; ++k) {
        native_execution_charge(0);
        const double identity = construction_objective ? std::numeric_limits<double>::infinity() : 180.0;
        const double left = k - i >= 2 ? best[i][k] : identity;
        const double right = j - k >= 2 ? best[k][j] : identity;
        const double value = std::min({left, right, triangle(i, k, j)});
        if (value > best[i][j]) {
          best[i][j] = value;
          choice[i][j] = k;
        }
      }
    }
  }
  const double result = best[0][n - 1];
  if (result < 0.0 ||
      (require_improvement && !construction_objective &&
       !(result > finite_quality(mesh, tets) + kImprovement))) {
    return false;
  }
  if (construction_objective) {
    double previous = std::numeric_limits<double>::infinity();
    for (int32_t cell : tets) previous = std::min(previous, construction_quality(mesh.tet(cell).v));
    if (!(result > previous)) return false;
  }
  NativeVector<Cell> proposed;
  NativeVector<std::array<int, 2>> pending = {{0, n - 1}};
  while (!pending.empty()) {
    const auto [i, j] = pending.back();
    pending.pop_back();
    const int k = choice[i][j];
    Cell above;
    Cell below;
    cells_of(i, k, j, above, below);
    proposed.push_back(above);
    proposed.push_back(below);
    if (k - i >= 2) {
      pending.push_back({i, k});
    }
    if (j - k >= 2) {
      pending.push_back({k, j});
    }
  }
  if (regular_weights &&
      (!regular_proposal(mesh, proposed, *regular_weights, radius_edge_bound, max_weight_fraction) ||
       !above_floor(mesh, proposed, minimum_relative_determinant))) {
    return false;
  }
  return mesh.replace(tets, proposed) == EditStatus::kOk;
}

Insertion try_split_edge(TetMesh& mesh, int32_t a, int32_t b, const double* position,
                         double target_size, int32_t& inserted_vertex) {
  const MemoryScope memory_scope(mesh.memory_owner());
  return mesh.split_edge(a, b, position, target_size, inserted_vertex);
}

bool try_multiface_removal(TetMesh& mesh, std::span<const int32_t> cavity,
                          int32_t apex, bool require_improvement) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (cavity.size() < 2 || !mesh.live_vertex(apex) ||
      !mesh.spend(static_cast<int64_t>(8 * cavity.size()))) {
    return false;
  }
  NativeVector<int32_t> sorted(cavity.begin(), cavity.end());
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    return false;
  }
  bool contains_apex = false;
  NativeVector<Cell> cells;
  for (int32_t t : sorted) {
    if (t < 0 || static_cast<std::size_t>(t) >= mesh.complex().tets.size() ||
        !mesh.complex().live(t) || !mesh.finite(t)) {
      return false;
    }
    const Tetrahedron& old = mesh.tet(t);
    contains_apex = contains_apex || vertex_slot(old, apex) >= 0;
    for (int k = 0; k < 4; ++k) {
      if (std::binary_search(sorted.begin(), sorted.end(), old.n[k])) {
        continue;
      }
      int32_t f[3];
      oriented_facet(old.v, k, f);
      if (std::find(f, f + 3, apex) != f + 3) {
        continue;
      }
      Cell cone = {old.v[0], old.v[1], old.v[2], old.v[3]};
      cone[static_cast<std::size_t>(k)] = apex;
      if (!mesh.positive(cone.data())) {
        return false;
      }
      cells.push_back(cone);
    }
  }
  if (!contains_apex || cells.empty() ||
      (require_improvement &&
       !(proposal_quality(mesh, cells) > finite_quality(mesh, sorted) + kImprovement))) {
    return false;
  }
  return reconnect(mesh, sorted, cells) == EditStatus::kOk;
}

EditStatus reconnect(TetMesh& mesh, std::span<const int32_t> removed,
                      std::span<const Cell> proposed) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (removed.empty() || proposed.empty()) {
    return EditStatus::kNotLive;
  }
  if (!mesh.spend(static_cast<int64_t>(8 * (removed.size() + proposed.size())))) {
    return EditStatus::kBufferLimit;
  }
  NativeSet<int32_t> old_vertices, new_vertices;
  NativeSet<std::uint64_t> old_segments, new_edges;
  for (int32_t t : removed) {
    if (t < 0 || static_cast<std::size_t>(t) >= mesh.complex().tets.size() ||
        !mesh.complex().live(t) || !mesh.finite(t)) {
      return EditStatus::kNotLive;
    }
    const int32_t* v = mesh.tet(t).v;
    old_vertices.insert(v, v + 4);
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        if (mesh.segment_source(v[i], v[j]) >= 0) {
          old_segments.insert(undirected_edge(v[i], v[j]));
        }
      }
    }
  }
  for (const Cell& cell : proposed) {
    new_vertices.insert(cell.begin(), cell.end());
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        new_edges.insert(undirected_edge(cell[static_cast<std::size_t>(i)],
                                        cell[static_cast<std::size_t>(j)]));
      }
    }
  }
  if (old_vertices != new_vertices ||
      !std::includes(new_edges.begin(), new_edges.end(), old_segments.begin(), old_segments.end())) {
    return EditStatus::kConstraintViolation;
  }
  return mesh.replace(removed, proposed);
}

bool try_relocate(TetMesh& mesh, int32_t vertex, const double* position, bool require_improvement) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (!mesh.live_vertex(vertex) || mesh.dimension(vertex) < 1 || !geometry::finite3(position) ||
      mesh.protection(vertex) > 0.0 || !representable(position) ||
      (mesh.dimension(vertex) < 3 && mesh.policy() == BoundaryPolicy::kFixed)) {
    return false;
  }
  NativeVector<int32_t> star;
  NativeVector<int32_t> neighbors;
  if (!mesh.vertex_star(vertex, star)) {
    return false;
  }
  if (!native_execution_cavity(star.size())) return false;
  for (int32_t t : star) {
    if (!mesh.spend(4)) return false;
    for (int32_t w : mesh.tet(t).v) {
      if (w >= 0 && w != vertex && mesh.protection(w) > 0.0 &&
          geometry::squared_distance(mesh.point(w), position) <
              mesh.protection(w) * mesh.protection(w)) {
        return false;
      }
    }
  }
  if (mesh.dimension(vertex) < 3 && !on_facet_plane(mesh, vertex, star, position, neighbors)) {
    return false;
  }
  if (mesh.dimension(vertex) == 1) {
    int32_t neighbors_on_curve[2] = {-1, -1};
    int32_t scientific_source = -1;
    int count = 0;
    for (const auto& [edge, source] : mesh.segments()) {
      if (!mesh.spend(1)) {
        return false;
      }
      const int32_t a = edge_low(edge);
      const int32_t b = edge_high(edge);
      if (a != vertex && b != vertex) {
        continue;
      }
      if (count == 2 || (scientific_source >= 0 && scientific_source != source)) {
        return false;
      }
      scientific_source = source;
      neighbors_on_curve[count++] = a == vertex ? b : a;
    }
    if (count != 2 || !mesh.spend(6)) {
      return false;
    }
    const double* first = mesh.point(neighbors_on_curve[0]);
    const double* second = mesh.point(neighbors_on_curve[1]);
    if (!collinear3d(first, second, mesh.point(vertex)) ||
        !collinear3d(first, second, position) ||
        std::equal(position, position + 3, first) ||
        std::equal(position, position + 3, second)) {
      return false;
    }
    for (int k = 0; k < 3; ++k) {
      if (position[k] < std::min(first[k], second[k]) ||
          position[k] > std::max(first[k], second[k])) {
        return false;
      }
    }
  }
  if (!mesh.spend(static_cast<int64_t>(4 * star.size()))) {
    return false;
  }
  const double before = finite_quality(mesh, star);
  const double after = star_quality_at(mesh, vertex, star, position);
  if (after < 0.0 || (require_improvement && !(after > before + kImprovement))) {
    return false;
  }
  // A moved boundary vertex keeps a witness its source row admits.
  SourceWitness witness;
  if (mesh.relocation_witness(vertex, position, witness) != Insertion::kOk) {
    return false;
  }
  std::copy_n(position, 3, mesh.mutable_point(vertex));
  mesh.set_witness(vertex, witness);
  mesh.mark_accepted_state();
  return true;
}

bool try_relocate_vertices(TetMesh& mesh, std::span<const int32_t> vertices,
                           const double* positions, double radius_edge_bound,
                           double minimum_dihedral_degrees) {
  const MemoryScope memory_scope(mesh.memory_owner());
  NativeVector<int32_t> star, neighbors, cavity;
  const auto proposed_point = [&](int32_t vertex) -> const double* {
    const auto found = std::lower_bound(vertices.begin(), vertices.end(), vertex);
    return found != vertices.end() && *found == vertex
               ? positions + 3 * (found - vertices.begin())
               : mesh.point(vertex);
  };
  for (std::size_t row = 0; row < vertices.size(); ++row) {
    const int32_t vertex = vertices[row];
    const double* position = positions + 3 * row;
    if ((row != 0 && vertices[row - 1] >= vertex) || !mesh.live_vertex(vertex) ||
        mesh.dimension(vertex) < 1 || mesh.protection(vertex) > 0.0 ||
        !geometry::finite3(position) || !representable(position) ||
        (mesh.dimension(vertex) < 3 && mesh.policy() == BoundaryPolicy::kFixed) ||
        !mesh.vertex_star(vertex, star)) {
      return false;
    }
    if (mesh.dimension(vertex) < 3 &&
        !on_facet_plane(mesh, vertex, star, position, neighbors)) {
      return false;
    }
    if (mesh.dimension(vertex) == 1) {
      int32_t endpoints[2] = {-1, -1};
      int32_t source = -1;
      int count = 0;
      for (const auto& [edge, identity] : mesh.segments()) {
        if (!mesh.spend(1)) {
          return false;
        }
        const int32_t first = edge_low(edge), second = edge_high(edge);
        if (first != vertex && second != vertex) {
          continue;
        }
        if (count == 2 || (source >= 0 && source != identity)) {
          return false;
        }
        source = identity;
        endpoints[count++] = first == vertex ? second : first;
      }
      if (count != 2 || !mesh.spend(6)) {
        return false;
      }
      const double* first = mesh.point(endpoints[0]);
      const double* second = mesh.point(endpoints[1]);
      if (!collinear3d(first, second, mesh.point(vertex)) ||
          !collinear3d(first, second, position)) {
        return false;
      }
      int axis = 0;
      while (axis < 3 && first[axis] == second[axis]) {
        ++axis;
      }
      if (axis == 3 || !(std::min(first[axis], second[axis]) < position[axis] &&
                        position[axis] < std::max(first[axis], second[axis]))) {
        return false;
      }
    }
    if (!mesh.spend(static_cast<int64_t>(star.size()))) {
      return false;
    }
    cavity.insert(cavity.end(), star.begin(), star.end());
  }
  std::sort(cavity.begin(), cavity.end());
  cavity.erase(std::unique(cavity.begin(), cavity.end()), cavity.end());
  if (!native_execution_cavity(cavity.size())) {
    throw ExecutionRefusal{native_execution_status(PHX_MC_CAPACITY_EXCEEDED)};
  }
  for (int32_t cell : cavity) {
    if (!mesh.finite(cell)) {
      continue;
    }
    if (!mesh.spend(8)) {
      return false;
    }
    const int32_t* v = mesh.tet(cell).v;
    const double* p[4] = {proposed_point(v[0]), proposed_point(v[1]),
                          proposed_point(v[2]), proposed_point(v[3])};
    if (orient3d(p[0], p[1], p[2], p[3]) <= 0) {
      return false;
    }
    if (radius_edge_bound > 0.0) {
      double center[3];
      if (!geometry::tetrahedron_circumcenter(p[0], p[1], p[2], p[3], center) ||
          std::sqrt(geometry::squared_distance(center, p[0])) / geometry::shortest_edge(p) >
              radius_edge_bound) {
        return false;
      }
    }
    if (minimum_dihedral_degrees > 0.0 &&
        geometry::minimum_dihedral(p) < minimum_dihedral_degrees) {
      return false;
    }
    for (int slot = 0; slot < 4; ++slot) {
      if (mesh.complex().constraint(cell, slot) == kNoConstraint) {
        continue;
      }
      int32_t f[3];
      oriented_facet(v, slot, f);
      const double* old[3] = {mesh.point(f[0]), mesh.point(f[1]), mesh.point(f[2])};
      const double* next[3] = {proposed_point(f[0]), proposed_point(f[1]), proposed_point(f[2])};
      int axis = 0;
      double maximum = 0.0;
      for (int candidate = 0; candidate < 3; ++candidate) {
        const int x = (candidate + 1) % 3, y = (candidate + 2) % 3;
        const double area = std::abs((old[1][x] - old[0][x]) * (old[2][y] - old[0][y]) -
                                     (old[1][y] - old[0][y]) * (old[2][x] - old[0][x]));
        if (area > maximum) {
          maximum = area;
          axis = candidate;
        }
      }
      const int x = (axis + 1) % 3, y = (axis + 2) % 3;
      const double a[2] = {old[0][x], old[0][y]}, b[2] = {old[1][x], old[1][y]};
      const double c[2] = {old[2][x], old[2][y]};
      const double d[2] = {next[0][x], next[0][y]}, e[2] = {next[1][x], next[1][y]};
      const double g[2] = {next[2][x], next[2][y]};
      if (!mesh.spend(2) || orient2d_exact(a, b, c).sign() * orient2d_exact(d, e, g).sign() <= 0) {
        return false;
      }
    }
  }
  for (const auto& [edge, source] : mesh.segments()) {
    const int32_t first = edge_low(edge), second = edge_high(edge);
    if (!std::binary_search(vertices.begin(), vertices.end(), first) &&
        !std::binary_search(vertices.begin(), vertices.end(), second)) {
      continue;
    }
    if (!mesh.spend(4)) {
      return false;
    }
    const double* old_first = mesh.point(first);
    const double* old_second = mesh.point(second);
    const double* next_first = proposed_point(first);
    const double* next_second = proposed_point(second);
    int axis = 0;
    while (axis < 3 && old_first[axis] == old_second[axis]) {
      ++axis;
    }
    if (axis == 3 || !collinear3d(old_first, old_second, next_first) ||
        !collinear3d(old_first, old_second, next_second) ||
        ((old_first[axis] < old_second[axis]) != (next_first[axis] < next_second[axis])) ||
        next_first[axis] == next_second[axis]) {
      return false;
    }
  }
  // Every allocation, exact predicate, witness and work refusal precedes the
  // first coordinate write. The commit is allocation-free and cannot
  // partially fail.
  NativeVector<SourceWitness> witnesses(vertices.size());
  for (std::size_t row = 0; row < vertices.size(); ++row) {
    if (mesh.relocation_witness(vertices[row], positions + 3 * row, witnesses[row]) !=
        Insertion::kOk) {
      return false;
    }
  }
  if (!mesh.spend(0)) {
    return false;
  }
  for (std::size_t row = 0; row < vertices.size(); ++row) {
    std::copy_n(positions + 3 * row, 3, mesh.mutable_point(vertices[row]));
    mesh.set_witness(vertices[row], witnesses[row]);
  }
  if (!vertices.empty()) {
    mesh.mark_accepted_state();
  }
  return !vertices.empty();
}

namespace {
bool prepare_collapse(TetMesh& mesh, int32_t remove, int32_t keep,
                       bool require_improvement, const double* retained_quality,
                       NativeVector<int32_t>& star, NativeVector<int32_t>& other_star,
                       NativeVector<Cell>& cells) {
  if (!mesh.live_vertex(remove) || !mesh.live_vertex(keep) || remove == keep ||
      mesh.dimension(remove) < 1 || mesh.protection(remove) > 0.0 ||
      (mesh.dimension(remove) < 3 && mesh.policy() == BoundaryPolicy::kFixed)) {
    return false;
  }
  other_star.clear();
  cells.clear();
  if ((retained_quality == nullptr && !mesh.vertex_star(remove, star)) ||
      !mesh.vertex_star(keep, other_star) ||
      !mesh.spend(static_cast<int64_t>(16 * (star.size() + other_star.size())))) {
    return false;
  }
  // Include every simplex (empty, vertex, edge, triangle) of each link.
  using Simplex = std::array<int32_t, 3>;
  NativeSet<Simplex> left, right, edge_link;
  const auto add_link = [](const Tetrahedron& cell, int32_t a, int32_t b,
                           NativeSet<Simplex>& link) {
    int32_t vertices[3];
    int n = 0;
    for (int32_t v : cell.v) {
      if (v != a && v != b) {
        vertices[n++] = v;
      }
    }
    for (int mask = 0; mask < (1 << n); ++mask) {
      Simplex simplex = {kDeadVertex, kDeadVertex, kDeadVertex};
      int count = 0;
      for (int i = 0; i < n; ++i) {
        if ((mask & (1 << i)) != 0) {
          simplex[static_cast<std::size_t>(count++)] = vertices[i];
        }
      }
      std::sort(simplex.begin(), simplex.end());
      link.insert(simplex);
    }
  };
  // Retain the selected proposal in the original owner until its single commit.
  bool edge = false;
  for (int32_t t : star) {
    if (!mesh.finite(t) && mesh.dimension(remove) == 3) {
      return false;
    }
    const Tetrahedron& cell = mesh.tet(t);
    add_link(cell, remove, kDeadVertex, left);
    if (vertex_slot(cell, keep) >= 0) {
      edge = true;
      add_link(cell, remove, keep, edge_link);
    } else {
      Cell collapsed = {cell.v[0], cell.v[1], cell.v[2], cell.v[3]};
      collapsed[static_cast<std::size_t>(vertex_slot(cell, remove))] = keep;
      if (retained_quality == nullptr && !is_ghost(cell) && !mesh.positive(collapsed.data())) {
        return false;
      }
      cells.push_back(collapsed);
    }
  }
  for (int32_t t : other_star) {
    add_link(mesh.tet(t), keep, kDeadVertex, right);
  }
  NativeSet<Simplex> common;
  std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                        std::inserter(common, common.end()));
  if (!edge || common != edge_link || cells.empty()) {
    return false;
  }
  if (require_improvement) {
    double after = retained_quality == nullptr ? 180.0 : *retained_quality;
    if (retained_quality == nullptr) {
      for (const Cell& cell : cells) {
        if (std::find(cell.begin(), cell.end(), kGhostVertex) == cell.end()) {
          after = std::min(after, quality(mesh, cell.data()));
        }
      }
    }
    if (!(after > finite_quality(mesh, star) + kImprovement)) return false;
  }
  return true;
}
}  // namespace

bool try_collapse_edge(TetMesh& mesh, int32_t remove, int32_t keep,
                       bool require_improvement) {
  const MemoryScope memory_scope(mesh.memory_owner());
  NativeVector<int32_t> star, other_star;
  NativeVector<Cell> cells;
  if (!prepare_collapse(mesh, remove, keep, require_improvement, nullptr,
                        star, other_star, cells)) return false;
  EditStatus status;
  if (mesh.dimension(remove) < 3) {
    status = mesh.collapse_boundary_edge(remove, keep, star, other_star);
  } else {
    auto prepared = mesh.prepare_replacement(star, cells);
    if (prepared.status() != EditStatus::kOk) return false;
    status = prepared.commit();
  }
  if (status != EditStatus::kOk) {
    return false;
  }
  mesh.retire_vertex(remove);
  return true;
}

namespace {
bool remove_vertex_above(TetMesh& mesh, int32_t vertex, bool require_improvement,
                         double competing_quality, int32_t* accepted_neighbor) {
  const MemoryScope memory_scope(mesh.memory_owner());
  NativeVector<int32_t> star;
  if (!mesh.live_vertex(vertex) || mesh.dimension(vertex) != 3 ||
      mesh.protection(vertex) > 0.0 || !mesh.vertex_star(vertex, star)) {
    return false;
  }
  NativeVector<int32_t> link;
  for (int32_t t : star) {
    native_execution_charge(0);
    if (!mesh.finite(t)) {
      return false;
    }
    for (int32_t w : mesh.tet(t).v) {
      if (w != vertex) {
        link.push_back(w);
      }
    }
  }
  std::sort(link.begin(), link.end());
  link.erase(std::unique(link.begin(), link.end()), link.end());
  const double lower_bound = require_improvement
      ? std::max(competing_quality, finite_quality(mesh, star))
      : -std::numeric_limits<double>::infinity();
  struct CandidateRemoval { int32_t target; double quality; std::size_t cells; };
  NativeVector<CandidateRemoval> candidates;
  if (!mesh.spend(static_cast<int64_t>(link.size() * star.size()))) return false;
  NativeVector<Cell> cells;
  for (int32_t u : link) {
    native_execution_charge(0);
    cells.clear();
    bool valid = true;
    for (int32_t t : star) {
      native_execution_charge(0);
      const Tetrahedron& cell = mesh.tet(t);
      if (vertex_slot(cell, u) >= 0) {
        continue;
      }
      Cell collapsed = {cell.v[0], cell.v[1], cell.v[2], cell.v[3]};
      collapsed[static_cast<std::size_t>(vertex_slot(cell, vertex))] = u;
      if (!mesh.positive(collapsed.data())) {
        valid = false;
        break;
      }
      cells.push_back(collapsed);
    }
    if (valid && !cells.empty()) {
      const double q = proposal_quality(mesh, cells);
      if (!require_improvement || q > lower_bound + kImprovement) {
        candidates.push_back({u, q, cells.size()});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    native_execution_charge(0);
    return a.quality != b.quality ? a.quality > b.quality : a.target < b.target;
  });
  NativeVector<int32_t> other_star;
  for (const auto& candidate : candidates) {
    // Quality losers never incur link/source admission. Optional search can
    // skip a cavity shape outside the original cap without attempting it.
    if (active_execution_scope != nullptr &&
        star.size() + candidate.cells > active_execution_scope->remaining()[2]) continue;
    if (!prepare_collapse(mesh, vertex, candidate.target, false, &candidate.quality,
                          star, other_star, cells)) {
      if (mesh.work_exhausted()) return false;
      continue;
    }
    auto prepared = mesh.prepare_replacement(star, cells);
    if (prepared.status() != EditStatus::kOk) continue;
    if (prepared.commit() != EditStatus::kOk) return false;
    mesh.retire_vertex(vertex);
    if (accepted_neighbor != nullptr) *accepted_neighbor = candidate.target;
    return true;
  }
  return false;
}
}  // namespace

bool try_remove_vertex(TetMesh& mesh, int32_t vertex, bool require_improvement,
                       int32_t* accepted_neighbor) {
  return remove_vertex_above(mesh, vertex, require_improvement,
                            -std::numeric_limits<double>::infinity(), accepted_neighbor);
}

int32_t improve_mesh(TetMesh& mesh, const ImproveOptions& options, int64_t* counters) {
  const MemoryScope memory_scope(mesh.memory_owner());
  const int64_t initial_work = mesh.work();
  std::fill_n(counters, PHX_MC_TET_MESH_IMPROVE_COUNTERS, 0);
  struct PartialWork {
    const TetMesh& mesh;
    int64_t initial;
    int64_t& output;
    ~PartialWork() { output = mesh.work() - initial; }
  } work_evidence{mesh, initial_work, counters[kWork]};
  int32_t status = PHX_MC_OK;
  bool exhausted = false;
  NativeVector<Candidate> slivers;
  ConstructionFrontier frontier;
  try {
  for (int32_t pass = 0; pass < options.max_passes && status == PHX_MC_OK; ++pass) {
    if (!collect_slivers(mesh, options.min_dihedral_degrees,
                         options.minimum_relative_determinant, slivers)) {
      status = PHX_MC_CAPACITY_EXCEEDED;
      break;
    }
    if (slivers.empty()) {
      break;
    }
    ++counters[kPasses];
    bool progress = false;
    // Give each original ordinary sliver star one source/quality-admitted
    // removal opportunity before reconnections can replace its live link.
    // Construction/floor repairs retain their existing ordered objective.
    {
      NativeSet<int32_t> removal_visited;
      for (const Candidate& entry : slivers) {
        native_execution_charge(0);
        if (entry.construction_uncertain || entry.below_floor || !current(mesh, entry)) continue;
        if (!mesh.spend(1)) {
          status = PHX_MC_CAPACITY_EXCEEDED;
          break;
        }
        ++counters[kAttempts];
        for (int32_t vertex : entry.v) {
          if (mesh.dimension(vertex) != 3 || mesh.protection(vertex) > 0.0 ||
              !removal_visited.insert(vertex).second) continue;
          if (try_remove_vertex(mesh, vertex, true, nullptr)) {
            ++counters[kVertexRemovals];
            progress = true;
          }
          if (mesh.broken()) return PHX_MC_INTERNAL_ERROR;
          if (mesh.work_exhausted()) {
            status = PHX_MC_CAPACITY_EXCEEDED;
            break;
          }
        }
        if (status != PHX_MC_OK) break;
      }
    }
    if (status != PHX_MC_OK) break;
    if (progress && !collect_slivers(mesh, options.min_dihedral_degrees,
                                    options.minimum_relative_determinant, slivers)) {
      status = PHX_MC_CAPACITY_EXCEEDED;
      break;
    }
    frontier.begin(mesh, slivers);
    for (std::size_t cursor = 0; cursor < slivers.size(); ++cursor) {
      native_execution_charge(0);
      const Candidate entry = slivers[cursor];
      std::size_t& pending = frontier.pending[static_cast<std::size_t>(entry.slot)];
      if (pending != cursor) continue;
      pending = ConstructionFrontier::kAbsent;
      if (!current(mesh, entry)) continue;
      if (!mesh.spend(1)) {
        status = PHX_MC_CAPACITY_EXCEEDED;
        break;
      }
      ++counters[kAttempts];
      std::array<int32_t, 5> affected_vertices;
      bool capacity = false;
      const bool changed = improve_cell(mesh, entry.slot, counters, entry.construction_uncertain,
                                        entry.below_floor, options.minimum_relative_determinant,
                                        affected_vertices, capacity);
      progress = changed || progress;
      if (mesh.broken()) {
        return PHX_MC_INTERNAL_ERROR;
      }
      // The vertex or slot allowance, not the geometry, refused the repair.
      if (mesh.work_exhausted() || capacity) {
        status = PHX_MC_CAPACITY_EXCEEDED;
        break;
      }
      // A neighboring accepted cavity can make an earlier refusal feasible.
      // Revisit only actual affected construction charts in this same pass.
      if (changed && !frontier.append(
              mesh, affected_vertices, options.minimum_relative_determinant, slivers)) {
        status = PHX_MC_CAPACITY_EXCEEDED;
        break;
      }
    }
    if (!progress) {
      exhausted = true;
      break;
    }
  }
  } catch (const ExecutionRefusal& refusal) {
    status = refusal.status;
  } catch (const std::bad_alloc&) {
    status = PHX_MC_CAPACITY_EXCEEDED;
  }
  if (!collect_slivers(mesh, options.min_dihedral_degrees,
                       options.minimum_relative_determinant, slivers)) status = PHX_MC_CAPACITY_EXCEEDED;
  auto& unmet = mesh.unmet();
  unmet.clear();
  const Unmet reason = exhausted ? Unmet::kNoImprovement : Unmet::kBudget;
  for (const Candidate& entry : slivers) {
    UnmetRecord record{};
    std::copy_n(mesh.tet(entry.slot).v, 4, record.v);
    canonicalize_cell(record.v, nullptr, 4);
    record.criterion = entry.below_floor ? PHX_MC_TET_MESH_CRITERION_VALIDITY
                       : entry.construction_uncertain ? PHX_MC_TET_MESH_CRITERION_CONSTRUCTION
                                                      : PHX_MC_TET_MESH_CRITERION_DIHEDRAL;
    // A below-floor chart is represented exactly; it remains for the run's
    // actual budget or no-improvement reason, not as a construction refusal.
    record.reason = static_cast<int32_t>(
        entry.construction_uncertain && !entry.below_floor ? Unmet::kNonrepresentable : reason);
    record.value = entry.construction_uncertain ? entry.construction_margin : entry.quality;
    unmet.push_back(record);
  }
  std::sort(unmet.begin(), unmet.end(), [](const UnmetRecord& x, const UnmetRecord& y) {
    return std::lexicographical_compare(x.v, x.v + 4, y.v, y.v + 4);
  });
  counters[kSlivers] = static_cast<int64_t>(std::count_if(
      slivers.begin(), slivers.end(), [&](const Candidate& entry) {
        return !std::isfinite(entry.quality) || entry.quality < options.min_dihedral_degrees;
      }));
  if (status == PHX_MC_OK && !unmet.empty()) {
    status = PHX_MC_REFINEMENT_LIMIT;
  }
  return status;
}

int32_t exude_mesh(TetMesh& mesh, const ExudeOptions& options,
                   NativeVector<double>& weights, int64_t* counters) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (!std::isfinite(options.max_weight_fraction) || options.max_weight_fraction <= 0.0 ||
      options.max_weight_fraction > 0.25 || !std::isfinite(options.radius_edge_bound) ||
      options.radius_edge_bound < 1.0 || options.min_dihedral_degrees < 0.0 ||
      !std::isfinite(options.min_dihedral_degrees) || options.max_passes < 0 ||
      !(options.minimum_relative_determinant >= 0.0) ||
      !(options.minimum_relative_determinant < 1.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::fill_n(counters, 6, int64_t{0});
  if (weights.get_allocator().owner() != mesh.memory_owner()) {
    weights = NativeVector<double>(NativeAllocator<double>(mesh.memory_owner()));
  }
  weights.assign(static_cast<std::size_t>(mesh.vertex_count()), 0.0);
  const int64_t initial_work = mesh.work();
  NativeVector<Candidate> slivers;
  bool stalled = false;
  bool capacity = false;
  for (int32_t pass = 0; pass < options.max_passes; ++pass) {
    if (!collect_slivers(mesh, options.min_dihedral_degrees, options.minimum_relative_determinant,
                         slivers)) {
      capacity = true;
      break;
    }
    if (slivers.empty()) {
      break;
    }
    ++counters[3];
    bool progress = false;
    for (const Candidate& entry : slivers) {
      if (!current(mesh, entry)) {
        continue;
      }
      const Cell vertices = {mesh.tet(entry.slot).v[0], mesh.tet(entry.slot).v[1],
                             mesh.tet(entry.slot).v[2], mesh.tet(entry.slot).v[3]};
      for (int32_t vertex : vertices) {
        if (exude_vertex(mesh, vertex, options, weights, counters)) {
          progress = true;
          break;
        }
      }
      if (mesh.broken() || mesh.work_exhausted()) {
        break;
      }
    }
    if (!progress || mesh.broken() || mesh.work_exhausted()) {
      stalled = !progress;
      break;
    }
  }
  capacity = !collect_slivers(mesh, options.min_dihedral_degrees,
                              options.minimum_relative_determinant, slivers) || capacity;
  mesh.unmet().clear();
  const Unmet reason = stalled ? Unmet::kNoImprovement : Unmet::kBudget;
  for (const Candidate& entry : slivers) {
    UnmetRecord record{};
    std::copy_n(mesh.tet(entry.slot).v, 4, record.v);
    canonicalize_cell(record.v, nullptr, 4);
    record.criterion = entry.below_floor ? PHX_MC_TET_MESH_CRITERION_VALIDITY
                       : entry.construction_uncertain ? PHX_MC_TET_MESH_CRITERION_CONSTRUCTION
                                                      : PHX_MC_TET_MESH_CRITERION_DIHEDRAL;
    record.reason = static_cast<int32_t>(
        entry.construction_uncertain && !entry.below_floor ? Unmet::kNonrepresentable : reason);
    record.value = entry.construction_uncertain ? entry.construction_margin : entry.quality;
    mesh.unmet().push_back(record);
  }
  std::sort(mesh.unmet().begin(), mesh.unmet().end(), [](const auto& a, const auto& b) {
    return std::lexicographical_compare(a.v, a.v + 4, b.v, b.v + 4);
  });
  counters[2] = static_cast<int64_t>(std::count_if(weights.begin(), weights.end(),
                                                [](double w) { return w > 0.0; }));
  counters[4] = static_cast<int64_t>(std::count_if(
      slivers.begin(), slivers.end(), [&](const Candidate& entry) {
        return !std::isfinite(entry.quality) || entry.quality < options.min_dihedral_degrees;
      }));
  counters[5] = mesh.work() - initial_work;
  return mesh.broken() ? PHX_MC_INTERNAL_ERROR
         : capacity || mesh.work_exhausted() ? PHX_MC_CAPACITY_EXCEEDED
         : slivers.empty() ? PHX_MC_OK : PHX_MC_REFINEMENT_LIMIT;
}

}  // namespace phx::mc

namespace {


int32_t ready(phx_mc_tet_mesh* handle, int32_t* applied) {
  if (handle == nullptr || applied == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *applied = 0;
  return handle->mesh->broken() ? PHX_MC_INTERNAL_ERROR : PHX_MC_OK;
}

int32_t finish(phx_mc_tet_mesh* handle, bool done, int32_t* applied) {
  *applied = done ? 1 : 0;
  return handle->mesh->broken() ? PHX_MC_INTERNAL_ERROR
         : handle->mesh->work_exhausted() ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
}

int32_t resolve_cells(phx::mc::TetMesh& mesh, int64_t cell_count, const int32_t* cells,
                       phx::mc::NativeVector<int32_t>& cavity) {
  cavity.reserve(static_cast<std::size_t>(cell_count));
  phx::mc::NativeVector<int32_t> star;
  for (int64_t row = 0; row < cell_count; ++row) {
    const int32_t* vertices = cells + 4 * row;
    if (!mesh.live_vertex(vertices[0]) || !mesh.vertex_star(vertices[0], star)) {
      return mesh.work_exhausted() ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_INVALID_INPUT;
    }
    int32_t resolved = -1;
    std::array<int32_t, 4> requested = {vertices[0], vertices[1], vertices[2], vertices[3]};
    std::sort(requested.begin(), requested.end());
    for (int32_t t : star) {
      if (!mesh.spend(4)) {
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      if (!mesh.finite(t)) {
        continue;
      }
      std::array<int32_t, 4> actual = {mesh.tet(t).v[0], mesh.tet(t).v[1],
                                      mesh.tet(t).v[2], mesh.tet(t).v[3]};
      std::sort(actual.begin(), actual.end());
      if (actual == requested) {
        resolved = t;
        break;
      }
    }
    if (resolved < 0) {
      return PHX_MC_INVALID_INPUT;
    }
    cavity.push_back(resolved);
  }
  return PHX_MC_OK;
}

std::array<int64_t, 2> insertion_counts(const phx::mc::TetMesh& mesh) {
  std::array<int64_t, 2> counts{};
  for (int32_t t : mesh.cavity()) {
    counts[0] += !phx::mc::is_ghost(mesh.tet(t));
  }
  for (const phx::mc::BoundaryFacet& face : mesh.cavity_boundary()) {
    phx::mc::Tetrahedron cone = mesh.tet(face.tet);
    cone.v[face.slot] = static_cast<int32_t>(mesh.vertex_count());
    counts[1] += !phx::mc::is_ghost(cone);
  }
  return counts;
}

bool insertion_rows(phx::mc::TetMesh& mesh, int32_t* removed, int32_t* proposed,
                    const int32_t* expected_removed = nullptr,
                    const int32_t* expected_proposed = nullptr) {
  std::size_t old_row = 0;
  std::size_t new_row = 0;
  for (int32_t t : mesh.cavity()) {
    const phx::mc::Tetrahedron& cell = mesh.tet(t);
    if (phx::mc::is_ghost(cell)) {
      continue;
    }
    if (expected_removed != nullptr) {
      if (!std::equal(cell.v, cell.v + 4, expected_removed + 4 * old_row)) {
        return false;
      }
    } else {
      std::copy_n(cell.v, 4, removed + 4 * old_row);
    }
    ++old_row;
  }
  for (const phx::mc::BoundaryFacet& face : mesh.cavity_boundary()) {
    phx::mc::Tetrahedron cone = mesh.tet(face.tet);
    cone.v[face.slot] = static_cast<int32_t>(mesh.vertex_count());
    if (phx::mc::is_ghost(cone)) {
      continue;
    }
    if (expected_proposed != nullptr) {
      if (!std::equal(cone.v, cone.v + 4, expected_proposed + 4 * new_row)) {
        return false;
      }
    } else {
      std::copy_n(cone.v, 4, proposed + 4 * new_row);
    }
    ++new_row;
  }
  return true;
}

int32_t insertion_status(const phx::mc::TetMesh& mesh, phx::mc::Insertion status) {
  return mesh.broken() || status == phx::mc::Insertion::kInternal
             ? PHX_MC_INTERNAL_ERROR
             : mesh.work_exhausted() || status == phx::mc::Insertion::kCapacity
                   ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_tet_mesh_improve(phx_mc_tet_mesh* handle, double min_dihedral_degrees,
                                double minimum_relative_determinant, int32_t max_passes,
                                int64_t work_limit, int64_t* counters) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::MeasuredTetStage timing(mesh, phx::mc::TetExecutionStage::kImprovement);
    if (counters == nullptr || !(min_dihedral_degrees >= 0.0) ||
        !(min_dihedral_degrees < 70.5) || !(minimum_relative_determinant >= 0.0) ||
        !(minimum_relative_determinant < 1.0) || max_passes < 0 || work_limit < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (mesh.broken()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::ImproveOptions options;
    options.min_dihedral_degrees = min_dihedral_degrees;
    options.minimum_relative_determinant = minimum_relative_determinant;
    options.max_passes = max_passes;
    const int32_t status = phx::mc::improve_mesh(mesh, options, counters);
    return mesh.broken() ? PHX_MC_INTERNAL_ERROR : status;
  });
}

int32_t phx_mc_tet_mesh_flip_face(phx_mc_tet_mesh* handle, const int32_t* face,
                                  int32_t require_improvement, int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK || face == nullptr) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    int32_t t = -1;
    int slot = -1;
    const bool done = mesh.live_vertex(face[0]) && mesh.live_vertex(face[1]) &&
                      mesh.live_vertex(face[2]) &&
                      mesh.find_face(face[0], face[1], face[2], t, slot) &&
                      phx::mc::try_face_removal(mesh, t, slot, require_improvement != 0);
    return finish(handle, done, applied);
  });
}

int32_t phx_mc_tet_mesh_remove_edge(phx_mc_tet_mesh* handle, int32_t a, int32_t b,
                                    int32_t require_improvement, int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK) {
      return status;
    }
    const phx::mc::MemoryScope memory_scope(handle->mesh->memory_owner());
    const bool done =
        phx::mc::try_edge_removal(*handle->mesh, a, b, require_improvement != 0);
    return finish(handle, done, applied);
  });
}

int32_t phx_mc_tet_mesh_relocate(phx_mc_tet_mesh* handle, int32_t vertex, const double* position,
                                 int32_t require_improvement, int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK || position == nullptr) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::MemoryScope memory_scope(handle->mesh->memory_owner());
    const bool done =
        phx::mc::try_relocate(*handle->mesh, vertex, position, require_improvement != 0);
    return finish(handle, done, applied);
  });
}

int32_t phx_mc_tet_mesh_relocate_vertices(phx_mc_tet_mesh* handle, int64_t count,
                                         const int32_t* vertices, const double* positions,
                                         double radius_edge_bound, double minimum_dihedral_degrees,
                                         int64_t work_limit, int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK || count < 0 || count > std::numeric_limits<int32_t>::max() ||
        (count > 0 && (vertices == nullptr || positions == nullptr)) ||
        !std::isfinite(radius_edge_bound) || radius_edge_bound < 0.0 ||
        !std::isfinite(minimum_dihedral_degrees) || minimum_dihedral_degrees < 0.0 ||
        minimum_dihedral_degrees >= 180.0 || work_limit < 0) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    const bool done = phx::mc::try_relocate_vertices(
        mesh, {vertices, static_cast<std::size_t>(count)}, positions,
        radius_edge_bound, minimum_dihedral_degrees);
    return finish(handle, done, applied);
  });
}

int32_t phx_mc_tet_mesh_remove_vertex(phx_mc_tet_mesh* handle, int32_t vertex,
                                      int32_t require_improvement, int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK) {
      return status;
    }
    const phx::mc::MemoryScope memory_scope(handle->mesh->memory_owner());
    const bool done =
        phx::mc::try_remove_vertex(*handle->mesh, vertex, require_improvement != 0, nullptr);
    return finish(handle, done, applied);
  });
}

int32_t phx_mc_tet_mesh_construct_edge_split(phx_mc_tet_mesh* handle, int32_t a, int32_t b,
                                            double preferred_fraction, int64_t work_limit,
                                            double* position, double* parameter,
                                            int32_t* constructed, int8_t* witness_stratum,
                                            int32_t* witness_entity, double* witness_parameters,
                                            double* witness_deviation) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, constructed);
    if (status != PHX_MC_OK || position == nullptr || parameter == nullptr || work_limit < 0 ||
        witness_stratum == nullptr || witness_entity == nullptr ||
        witness_parameters == nullptr || witness_deviation == nullptr ||
        !std::isfinite(preferred_fraction) || preferred_fraction < 0.0 ||
        preferred_fraction >= 1.0) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    *parameter = 0.0;
    *witness_stratum = static_cast<int8_t>(phx::mc::SourceStratum::kNone);
    *witness_entity = -1;
    std::fill_n(witness_parameters, 2, 0.0);
    *witness_deviation = 0.0;
    phx::mc::SourceWitness witness;
    if (!mesh.live_vertex(a) || !mesh.live_vertex(b) || a == b || mesh.edge_tet(a, b) < 0) {
      return mesh.work_exhausted() ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
    }
    *constructed = mesh.construct_edge_point(a, b, position, parameter, preferred_fraction) ? 1 : 0;
    if (*constructed != 0) {
      const auto admitted = mesh.edge_construction_witness(a, b, position, witness);
      if (admitted == phx::mc::Insertion::kCapacity) {
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      *constructed = admitted == phx::mc::Insertion::kOk ? 1 : 0;
    }
    if (*constructed == 0 && !mesh.work_exhausted()) {
      const double fraction = preferred_fraction == 0.0 ? 0.5 : preferred_fraction;
      const auto admitted = mesh.construct_source_split(a, b, fraction, position, witness);
      if (admitted == phx::mc::Insertion::kCapacity) {
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      if (admitted == phx::mc::Insertion::kOk) {
        *constructed = 1;
        *parameter = fraction;
      }
    }
    if (*constructed != 0) {
      *witness_stratum = static_cast<int8_t>(witness.stratum);
      *witness_entity = witness.entity;
      std::copy_n(witness.parameters, 2, witness_parameters);
      *witness_deviation = witness.deviation;
    }
    return mesh.work_exhausted() ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_construct_curve_points(
    phx_mc_tet_mesh* handle, int64_t count, const int32_t* vertices,
    const int32_t* neighbors, const double* desired_positions, int64_t work_limit,
    double* positions, double* coordinates, int32_t* constructed) {
  return phx::mc::guarded([&]() -> int32_t {
    int32_t marker = 0;
    const int32_t status = ready(handle, &marker);
    if (status != PHX_MC_OK || count < 0 || count > std::numeric_limits<int32_t>::max() ||
        work_limit < 0 || (count > 0 &&
        (vertices == nullptr || neighbors == nullptr || desired_positions == nullptr ||
         positions == nullptr || coordinates == nullptr || constructed == nullptr))) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    for (int64_t row = 0; row < count; ++row) {
      constructed[row] = 0;
      coordinates[row] = invalid;
      std::fill_n(positions + 3 * row, 3, invalid);
      if (!mesh.spend(8)) {
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      const int32_t vertex = vertices[row], first = neighbors[2 * row], second = neighbors[2 * row + 1];
      const double* desired = desired_positions + 3 * row;
      if (!mesh.live_vertex(vertex) || mesh.dimension(vertex) != 1 ||
          !mesh.live_vertex(first) || !mesh.live_vertex(second) ||
          first == second || first == vertex || second == vertex ||
          !phx::mc::geometry::finite3(desired)) {
        continue;
      }
      const int32_t source = mesh.segment_source(vertex, first);
      if (source < 0 || mesh.segment_source(vertex, second) != source ||
          !phx::mc::collinear3d(mesh.point(first), mesh.point(second), mesh.point(vertex))) {
        continue;
      }
      const double* original_first = nullptr;
      const double* original_second = nullptr;
      for (const auto& segment : mesh.original_segments()) {
        if (!mesh.spend(1)) return PHX_MC_CAPACITY_EXCEEDED;
        if (segment[2] != source) continue;
        const double* a = mesh.original_point(segment[0]);
        const double* b = mesh.original_point(segment[1]);
        if (phx::mc::collinear3d(a, b, mesh.point(vertex)) &&
            phx::mc::collinear3d(a, b, mesh.point(first)) &&
            phx::mc::collinear3d(a, b, mesh.point(second))) {
          original_first = a;
          original_second = b;
          break;
        }
      }
      if (original_first != nullptr) {
        constructed[row] = phx::mc::construct_exact_line_point(
            original_first, original_second, mesh.point(first), mesh.point(second),
            desired, positions + 3 * row, coordinates + row,
            [](void* context, std::int64_t units) {
              return static_cast<phx::mc::TetMesh*>(context)->spend(units);
            }, &mesh) ? 1 : 0;
        if (constructed[row] &&
            (!phx::mc::collinear3d(original_first, original_second, positions + 3 * row) ||
             !phx::mc::collinear3d(mesh.point(first), mesh.point(second), positions + 3 * row))) {
          return PHX_MC_INTERNAL_ERROR;
        }
      }
      if (!constructed[row]) {
        coordinates[row] = invalid;
        std::fill_n(positions + 3 * row, 3, invalid);
      }
      if (mesh.work_exhausted()) {
        return PHX_MC_CAPACITY_EXCEEDED;
      }
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_split_edge(phx_mc_tet_mesh* handle, int32_t a, int32_t b,
                                  const double* position, double target_size,
                                  double source_fraction, int64_t work_limit,
                                  int32_t* inserted_vertex) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || position == nullptr || inserted_vertex == nullptr ||
        work_limit < 0 || !std::isfinite(target_size) || target_size < 0.0 ||
        !std::isfinite(source_fraction) || source_fraction < 0.0 || source_fraction >= 1.0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *inserted_vertex = -1;
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    if (mesh.broken()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::Insertion outcome =
        phx::mc::try_split_edge(mesh, a, b, position, target_size, *inserted_vertex);
    if (outcome == phx::mc::Insertion::kRefused && source_fraction > 0.0 &&
        !mesh.work_exhausted()) {
      double carrier[3];
      phx::mc::SourceWitness witness;
      outcome = mesh.construct_source_split(a, b, source_fraction, carrier, witness);
      if (outcome == phx::mc::Insertion::kOk) {
        outcome = std::equal(carrier, carrier + 3, position)
                      ? mesh.split_edge_bounded(a, b, carrier, witness, target_size,
                                                *inserted_vertex)
                      : phx::mc::Insertion::kRefused;
      }
    }
    const bool capacity = mesh.work_exhausted() || outcome == phx::mc::Insertion::kCapacity;
    return mesh.broken() || outcome == phx::mc::Insertion::kInternal
               ? PHX_MC_INTERNAL_ERROR
               : capacity ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_split_edge_relocate(phx_mc_tet_mesh* handle, int32_t a, int32_t b,
                                           const double* split_position,
                                           const double* final_position, double target_size,
                                           int64_t work_limit, int32_t* inserted_vertex) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || split_position == nullptr || final_position == nullptr ||
        inserted_vertex == nullptr || work_limit < 0 ||
        !std::isfinite(target_size) || target_size < 0.0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *inserted_vertex = -1;
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    if (mesh.broken()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    const phx::mc::Insertion outcome = mesh.split_edge_relocate(
        a, b, split_position, final_position, target_size, *inserted_vertex);
    const bool capacity = mesh.work_exhausted() || outcome == phx::mc::Insertion::kCapacity;
    return mesh.broken() || outcome == phx::mc::Insertion::kInternal
               ? PHX_MC_INTERNAL_ERROR
               : capacity ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_inspect_edge_insertion(
    phx_mc_tet_mesh* handle, int32_t first, int32_t second, const double* split_position,
    const double* final_position, double source_fraction, int64_t maximum_cavity_cells, int64_t work_limit,
    int32_t* removed_tetrahedra, int32_t* proposed_tetrahedra, int64_t* counts) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr || split_position == nullptr ||
        final_position == nullptr || counts == nullptr || maximum_cavity_cells < 0 ||
        maximum_cavity_cells > INT64_MAX / 4 || work_limit < 0 ||
        !std::isfinite(source_fraction) || source_fraction < 0.0 || source_fraction >= 1.0 ||
        (maximum_cavity_cells > 0 && (removed_tetrahedra == nullptr || proposed_tetrahedra == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::fill_n(counts, 4, int64_t{0});
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    if (mesh.broken()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::InsertKind kind = phx::mc::InsertKind::kInterior;
    const phx::mc::Insertion prepared = mesh.prepare_edge_insertion(
        first, second, split_position, final_position, source_fraction,
        static_cast<std::size_t>(maximum_cavity_cells), kind);
    if (prepared != phx::mc::Insertion::kOk) {
      mesh.abandon();
      return insertion_status(mesh, prepared);
    }
    const auto finite = insertion_counts(mesh);
    if (finite[0] == 0 || finite[1] == 0) {
      mesh.abandon();
      return PHX_MC_OK;
    }
    if (!mesh.spend(4 * (finite[0] + finite[1]))) {
      mesh.abandon();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    insertion_rows(mesh, removed_tetrahedra, proposed_tetrahedra);
    counts[0] = finite[0];
    counts[1] = finite[1];
    counts[2] = static_cast<int32_t>(kind);
    counts[3] = mesh.accepted_generation();
    mesh.abandon();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_commit_edge_insertion(
    phx_mc_tet_mesh* handle, int32_t first, int32_t second, const double* split_position,
    const double* final_position, double source_fraction, int64_t maximum_cavity_cells, int32_t insertion_kind,
    int64_t accepted_generation, int64_t removed_count, const int32_t* removed_tetrahedra,
    int64_t proposed_count, const int32_t* proposed_tetrahedra, double target_size,
    int64_t work_limit, int32_t* inserted_vertex) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr || split_position == nullptr ||
        final_position == nullptr || inserted_vertex == nullptr ||
        maximum_cavity_cells < 0 || maximum_cavity_cells > INT64_MAX / 4 ||
        removed_count <= 0 || proposed_count <= 0 || removed_count > maximum_cavity_cells ||
        proposed_count > maximum_cavity_cells - removed_count ||
        !std::isfinite(source_fraction) || source_fraction < 0.0 || source_fraction >= 1.0 ||
        removed_tetrahedra == nullptr || proposed_tetrahedra == nullptr ||
        insertion_kind < 0 || insertion_kind > 2 || accepted_generation < 0 ||
        !std::isfinite(target_size) || target_size < 0.0 || work_limit < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *inserted_vertex = -1;
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    if (mesh.broken()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    if (mesh.accepted_generation() != accepted_generation) {
      return PHX_MC_OK;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::InsertKind kind = phx::mc::InsertKind::kInterior;
    const phx::mc::Insertion prepared = mesh.prepare_edge_insertion(
        first, second, split_position, final_position, source_fraction,
        static_cast<std::size_t>(maximum_cavity_cells), kind);
    if (prepared != phx::mc::Insertion::kOk) {
      mesh.abandon();
      return insertion_status(mesh, prepared);
    }
    const auto finite = insertion_counts(mesh);
    if (finite[0] != removed_count || finite[1] != proposed_count ||
        static_cast<int32_t>(kind) != insertion_kind) {
      mesh.abandon();
      return PHX_MC_OK;
    }
    if (!mesh.spend(4 * (removed_count + proposed_count))) {
      mesh.abandon();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (!insertion_rows(mesh, nullptr, nullptr, removed_tetrahedra, proposed_tetrahedra)) {
      mesh.abandon();
      return PHX_MC_OK;
    }
    const int32_t pending = static_cast<int32_t>(mesh.vertex_count());
    const phx::mc::Insertion committed = mesh.commit(kind, target_size);
    if (committed == phx::mc::Insertion::kOk) {
      *inserted_vertex = pending;
    }
    mesh.abandon();
    return insertion_status(mesh, committed);
  });
}

int32_t phx_mc_tet_mesh_collapse_edge(phx_mc_tet_mesh* handle, int32_t remove, int32_t keep,
                                     int32_t require_improvement, int64_t work_limit,
                                     int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK || work_limit < 0) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    const bool done = phx::mc::try_collapse_edge(mesh, remove, keep, require_improvement != 0);
    const bool capacity = mesh.work_exhausted();
    *applied = done ? 1 : 0;
    return mesh.broken() ? PHX_MC_INTERNAL_ERROR
                        : capacity ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_remove_multiface(phx_mc_tet_mesh* handle, int64_t cell_count,
                                        const int32_t* cells, int32_t apex,
                                        int32_t require_improvement, int64_t work_limit,
                                        int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK || cells == nullptr || cell_count < 2 || work_limit < 0) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    if (cell_count > static_cast<int64_t>(mesh.complex().tets.size())) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::NativeVector<int32_t> cavity;
    const int32_t resolved = resolve_cells(mesh, cell_count, cells, cavity);
    if (resolved != PHX_MC_OK) {
      return resolved;
    }
    const bool done =
        phx::mc::try_multiface_removal(mesh, cavity, apex, require_improvement != 0);
    const bool capacity = mesh.work_exhausted();
    *applied = done ? 1 : 0;
    return mesh.broken() ? PHX_MC_INTERNAL_ERROR
                        : capacity ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_reconnect(phx_mc_tet_mesh* handle, int64_t removed_count,
                                 const int32_t* removed_tetrahedra, int64_t proposed_count,
                                 const int32_t* proposed_tetrahedra, int64_t work_limit,
                                 int32_t* applied) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = ready(handle, applied);
    if (status != PHX_MC_OK || removed_tetrahedra == nullptr || proposed_tetrahedra == nullptr ||
        removed_count < 1 || proposed_count < 1 || work_limit < 0) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    if (removed_count > static_cast<int64_t>(mesh.complex().tets.size()) ||
        proposed_count > std::numeric_limits<int32_t>::max()) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    if (proposed_count > mesh.max_tetrahedra() || !mesh.spend(4 * proposed_count)) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    phx::mc::NativeVector<int32_t> removed;
    const int32_t resolved = resolve_cells(mesh, removed_count, removed_tetrahedra, removed);
    if (resolved != PHX_MC_OK) {
      return resolved;
    }
    phx::mc::NativeVector<std::array<int32_t, 4>> proposed;
    proposed.reserve(static_cast<std::size_t>(proposed_count));
    for (int64_t i = 0; i < proposed_count; ++i) {
      const int32_t* v = proposed_tetrahedra + 4 * i;
      proposed.push_back({v[0], v[1], v[2], v[3]});
    }
    const phx::mc::EditStatus outcome = phx::mc::reconnect(mesh, removed, proposed);
    *applied = outcome == phx::mc::EditStatus::kOk ? 1 : 0;
    return mesh.broken() || outcome == phx::mc::EditStatus::kInternal
               ? PHX_MC_INTERNAL_ERROR
               : mesh.work_exhausted() || outcome == phx::mc::EditStatus::kCellLimit ||
                         outcome == phx::mc::EditStatus::kSlotLimit ||
                         outcome == phx::mc::EditStatus::kBufferLimit
                     ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_exude(phx_mc_tet_mesh* handle, double min_dihedral_degrees,
                             double max_weight_fraction, double radius_edge_bound,
                             double minimum_relative_determinant, int32_t max_passes,
                             int64_t work_limit, double* weights, int64_t* counters) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::MeasuredTetStage timing(mesh, phx::mc::TetExecutionStage::kExudation);
    if (weights == nullptr || counters == nullptr || work_limit < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::fill_n(weights, static_cast<std::size_t>(mesh.vertex_count()), 0.0);
    std::fill_n(counters, 6, int64_t{0});
    if (mesh.broken()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::ExudeOptions options;
    options.min_dihedral_degrees = min_dihedral_degrees;
    options.max_weight_fraction = max_weight_fraction;
    options.radius_edge_bound = radius_edge_bound;
    options.minimum_relative_determinant = minimum_relative_determinant;
    options.max_passes = max_passes;
    phx::mc::NativeVector<double> accepted_weights;
    const int64_t initial_work = mesh.work();
    int32_t status = PHX_MC_OK;
    try {
      status = phx::mc::exude_mesh(mesh, options, accepted_weights, counters);
    } catch (const std::bad_alloc&) {
      std::copy(accepted_weights.begin(), accepted_weights.end(), weights);
      counters[2] = static_cast<int64_t>(std::count_if(
          accepted_weights.begin(), accepted_weights.end(), [](double weight) { return weight > 0.0; }));
      counters[5] = mesh.work() - initial_work;
      throw;  // Preserve the owning allocator's refusal status/message.
    }
    std::copy(accepted_weights.begin(), accepted_weights.end(), weights);
    return status;
  });
}

}  // extern "C"
