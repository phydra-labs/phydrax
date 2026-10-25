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
// geometry/topology transaction. Exudation tries each vertex's increasing
// trial weights in the original order but evaluates weight-independent
// flip/ring work once per visit, locates the sign change of each exact
// (weight-affine) power test by bisection, and reuses a complete refused
// visit until an accepted flip changes data it reads. Exact source strata,
// region labels and protected points survive every stage.
#include "improve3d.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "capi_guard.hpp"
#include "counter_records.hpp"
#include "exact_line.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "tet_mesh.hpp"
#include "topology_keys.hpp"

namespace phx::mc {
namespace {

using Cell = std::array<int32_t, 4>;

// Smallest strict gain (degrees) that counts as an improvement.
constexpr double kImprovement = 1e-9;
constexpr int kMaxRing = 7;
constexpr std::size_t kCoupledVertices = kMaxRing + 2;
constexpr std::size_t kCoupledVariables = 3 * kCoupledVertices;
using AffectedVertices = std::array<int32_t, kCoupledVertices>;


double quality(const TetMesh& mesh, const int32_t* v) {
  const double* p[4];
  mesh.corners(v, p);
  return geometry::minimum_dihedral(p);
}

double finite_quality(const TetMesh& mesh, std::span<const int32_t> tets,
                      int32_t* worst_cell = nullptr) {
  double worst = 180.0;
  for (int32_t t : tets) {
    if (mesh.finite(t)) {
      const double value = quality(mesh, mesh.tet(t).v);
      if (value < worst) {
        worst = value;
        if (worst_cell != nullptr) *worst_cell = t;
      }
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
                       const double* position,
                       double rejection_floor = -std::numeric_limits<double>::infinity(),
                       int32_t* critical_cell = nullptr) {
  double* p = mesh.mutable_point(v);
  struct RestoreCoordinates {
    double* point;
    double saved[3];
    ~RestoreCoordinates() { std::copy_n(saved, 3, point); }
  } restore{p, {p[0], p[1], p[2]}};
  std::copy_n(position, 3, p);
  double worst = 180.0;
  // Testing the last active/rejecting cone first changes evaluation order
  // only. Every nonrejected score still visits the complete incident star.
  const int32_t first = critical_cell == nullptr ? -1 : *critical_cell;
  if (std::isfinite(rejection_floor) && first >= 0) {
    if (!mesh.spend(TetWorkCost::kVisit) || !mesh.positive(mesh.tet(first).v)) return -1.0;
    worst = quality(mesh, mesh.tet(first).v);
    if (worst <= rejection_floor) return rejection_floor;
  }
  for (int32_t t : star) {
    if (std::isfinite(rejection_floor) && t == first) continue;
    native_execution_charge(0);
    if (std::isfinite(rejection_floor) && !mesh.spend(TetWorkCost::kVisit)) return -1.0;
    if (!mesh.finite(t)) {
      continue;
    }
    if (!mesh.positive(mesh.tet(t).v)) {
      worst = -1.0;
      break;
    }
    const double value = quality(mesh, mesh.tet(t).v);
    if (value < worst) {
      worst = value;
      if (critical_cell != nullptr) *critical_cell = t;
    }
    // This value is only a rejection bound, never an accepted trial's score.
    // A single cone at/below the branch's strict acceptance floor proves
    // that evaluating its remaining cones cannot change that branch.
    if (worst <= rejection_floor) return rejection_floor;
  }
  return worst;
}

// Completed proposal arithmetic is reusable while its local stars are
// unchanged in this improvement invocation. Transaction and resource refusals
// never enter the proof bank; uncertain source coordinates are excluded.
using EdgeRemovalFailureKey =
    std::tuple<int32_t, int32_t, bool, bool, uint64_t, uint64_t, uint64_t>;

struct SmoothingSelection {
  double baseline;
  double quality;
  std::array<double, 3> position;
};

bool exact_star_witnesses(const TetMesh& mesh, std::span<const int32_t> star) {
  for (int32_t t : star) {
    for (int32_t v : mesh.tet(t).v) {
      if (v >= 0 && mesh.witness(v).deviation > 0.0) return false;
    }
  }
  return true;
}

struct ImprovementProofs {
  int64_t generation = -1;
  NativeSet<EdgeRemovalFailureKey> proved;
  NativeSet<std::tuple<int32_t, bool, uint64_t>> vertex_proved;
  NativeMap<int32_t, SmoothingSelection> smoothing;

  void synchronize(const TetMesh& mesh) {
    if (generation != mesh.accepted_generation()) {
      proved.clear();
      vertex_proved.clear();
      smoothing.clear();
      generation = mesh.accepted_generation();
    }
  }

  bool advance(TetMesh& mesh, std::span<const int32_t> affected_cells) {
    // The frontier's covering-anchor stars contain every changed surviving
    // or new cell, including every live vertex of the edited cavity. A changed
    // edge ring, vertex link, coordinate or constrained plane has its endpoint
    // in this conservative dirty set. Unrelated complete geometric negatives
    // remain exact proofs, not fresh work charged again after a remote edit.
    if (!mesh.spend(static_cast<int64_t>(affected_cells.size()))) return false;
    NativeSet<int32_t> dirty;
    for (int32_t cell : affected_cells) {
      for (int32_t vertex : mesh.tet(cell).v) {
        if (vertex >= 0) dirty.insert(vertex);
      }
    }
    std::erase_if(proved, [&](const EdgeRemovalFailureKey& entry) {
      return dirty.contains(std::get<0>(entry)) || dirty.contains(std::get<1>(entry));
    });
    std::erase_if(vertex_proved, [&](const auto& entry) {
      return dirty.contains(std::get<0>(entry));
    });
    std::erase_if(smoothing, [&](const auto& entry) { return dirty.contains(entry.first); });
    generation = mesh.accepted_generation();
    return true;
  }

  EdgeRemovalFailureKey key(const TetMesh& mesh, int32_t a, int32_t b,
                           bool require_improvement, bool construction_objective,
                           double radius_edge_bound, double max_weight_fraction,
                           double minimum_relative_determinant) {
    synchronize(mesh);
    // Endpoint direction and scalar bit patterns preserve the exact
    // proposal arithmetic; a mathematically equivalent order is not a hit.
    return {a, b, require_improvement, construction_objective,
            std::bit_cast<uint64_t>(radius_edge_bound),
            std::bit_cast<uint64_t>(max_weight_fraction),
            std::bit_cast<uint64_t>(minimum_relative_determinant)};
  }
};

bool remove_vertex_above(TetMesh& mesh, int32_t vertex, bool require_improvement,
                         double competing_quality, int32_t* accepted_neighbor,
                         ImprovementProofs* failures = nullptr);

// A movable curve vertex has two incident segments of one scientific source.
bool curve_interval(TetMesh& mesh, int32_t vertex, int32_t* endpoints, int32_t& source) {
  int count = 0;
  for (const auto& [edge, identity] : mesh.segments()) {
    if (!mesh.spend(1)) return false;
    const int32_t a = edge_low(edge), b = edge_high(edge);
    if (a != vertex && b != vertex) continue;
    if (count == 2 || (source >= 0 && source != identity)) return false;
    source = identity;
    endpoints[count++] = a == vertex ? b : a;
  }
  return count == 2;
}

struct CurveCarrier {
  const double* first;
  const double* second;
};

// Exact ancestry uses the authored line. A bounded represented line remains
// subject to relocation_witness's source-row admission before any commit.
bool prepare_curve_carrier(TetMesh& mesh, int32_t source, const double* first,
                           const double* second, CurveCarrier& carrier) {
  for (const auto& segment : mesh.original_segments()) {
    if (!mesh.spend(1)) return false;
    if (segment[2] != source) continue;
    const double* a = mesh.original_point(segment[0]);
    const double* b = mesh.original_point(segment[1]);
    if (geometry::collinear(a, b, first) && geometry::collinear(a, b, second)) {
      carrier.first = a;
      carrier.second = b;
      break;
    }
  }
  return true;
}

bool smooth_segment_vertex(TetMesh& mesh, int32_t v, std::span<const int32_t> star,
                           NativeVector<int32_t>& neighbors) {
    int32_t ends[2] = {-1, -1};
    int32_t source = -1;
    if (!curve_interval(mesh, v, ends, source)) return false;
    const double* first = mesh.point(ends[0]);
    const double* second = mesh.point(ends[1]);
    if (!geometry::collinear(first, second, mesh.point(v))) return false;
    CurveCarrier carrier{first, second};
    if (!prepare_curve_carrier(mesh, source, first, second, carrier)) return false;
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
                carrier.first, carrier.second, first, second, desired, candidate, &coordinate,
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

struct SmoothingTargets {
  double origin[3];
  bool shared[3] = {true, true, true};
  NativeVector<std::array<double, 3>> targets;
};

bool prepare_smoothing_chart(TetMesh& mesh, int32_t v, int8_t dimension,
                             std::span<const int32_t> star,
                             NativeVector<int32_t>& neighbors, const double* origin,
                             bool* shared) {
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
  for (int32_t w : neighbors) {
    for (int i = 0; i < 3; ++i) {
      shared[i] = shared[i] && mesh.point(w)[i] == origin[i];
    }
  }
  return true;
}

bool prepare_smoothing_targets(TetMesh& mesh, int32_t v, int8_t dimension,
                               std::span<const int32_t> star,
                               NativeVector<int32_t>& neighbors, SmoothingTargets& prepared) {
  if (!prepare_smoothing_chart(mesh, v, dimension, star, neighbors,
                                prepared.origin, prepared.shared)) return false;
  auto& targets = prepared.targets;
  double centroid[3] = {0.0, 0.0, 0.0};
  for (int32_t w : neighbors) {
    for (int axis = 0; axis < 3; ++axis) centroid[axis] += mesh.point(w)[axis];
  }
  for (int i = 0; i < 3; ++i) {
    centroid[i] /= static_cast<double>(neighbors.size());
  }
  // Candidate targets: the link centroid and, for interior vertices, the apex
  // position that would make the worst incident cell regular.
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
  return true;
}

// Maximin star optimization is a coordinate proposal, never a transaction.
// Nelder-Mead handles the nonsmooth minimum of incident dihedrals without
// differentiating the active-cell selector. It stops at the requested angle
// or binary64 spatial convergence; every evaluation uses the owning ledger.
bool optimize_star_target(TetMesh& mesh, int32_t vertex, int8_t dimension,
                          std::span<const int32_t> star, const bool* shared,
                          NativeVector<int32_t>& neighbors, double target, int32_t critical_cell,
                          double& best_quality, double* best) {
  if (best_quality >= target) return true;
  struct Trial {
    std::array<double, 3> position;
    double quality;
  };
  std::array<int, 3> axes{};
  std::size_t count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (dimension == 3 || !shared[axis]) axes[count++] = axis;
  }
  if (count == 0) return true;
  const bool axis_plane = dimension == 2 && count < 3;
  double shortest = std::numeric_limits<double>::infinity();
  for (int32_t neighbor : neighbors) {
    shortest = std::min(shortest,
        std::sqrt(geometry::squared_distance(mesh.point(vertex), mesh.point(neighbor))));
  }
  if (!(shortest > 0.0) || !std::isfinite(shortest)) return true;
  const auto evaluate = [&](const std::array<double, 3>& position,
                            double rejection_floor = -std::numeric_limits<double>::infinity()) {
    if (!std::isfinite(rejection_floor) &&
        !mesh.spend(static_cast<int64_t>(star.size()))) return -1.0;
    if (!geometry::point_in_domain(position.data()) ||
        (dimension == 2 && !axis_plane &&
         !on_facet_plane(mesh, vertex, star, position.data(), neighbors))) return -1.0;
    return star_quality_at(mesh, vertex, star, position.data(), rejection_floor, &critical_cell);
  };
  std::array<Trial, 4> simplex{};
  simplex[0] = {{best[0], best[1], best[2]}, best_quality};
  for (std::size_t index = 0; index < count; ++index) {
    simplex[index + 1].position = simplex[0].position;
    simplex[index + 1].position[axes[index]] += 0.25 * shortest;
    simplex[index + 1].quality = evaluate(simplex[index + 1].position);
  }
  const double resolution = std::sqrt(std::numeric_limits<double>::epsilon()) * shortest;
  while (!mesh.work_exhausted() && native_execution_refusal() == PHX_MC_OK) {
    // At most four entries; insertion sorting avoids a general introsort's
    // larger-range machinery and gives tied trials a canonical order.
    for (std::size_t index = 1; index <= count; ++index) {
      std::size_t cursor = index;
      while (cursor > 0 &&
             (simplex[cursor].quality > simplex[cursor - 1].quality ||
              (simplex[cursor].quality == simplex[cursor - 1].quality &&
               simplex[cursor].position < simplex[cursor - 1].position))) {
        std::swap(simplex[cursor], simplex[cursor - 1]);
        --cursor;
      }
    }
    if (simplex[0].quality > best_quality + kImprovement) {
      best_quality = simplex[0].quality;
      std::copy(simplex[0].position.begin(), simplex[0].position.end(), best);
    }
    if (best_quality >= target) break;
    double diameter = 0.0;
    std::array<double, 3> centroid{};
    for (std::size_t index = 0; index < count; ++index) {
      for (int axis = 0; axis < 3; ++axis) {
        centroid[axis] += simplex[index].position[axis] / static_cast<double>(count);
      }
    }
    for (std::size_t index = 1; index <= count; ++index) {
      diameter = std::max(diameter, std::sqrt(geometry::squared_distance(
          simplex[0].position.data(), simplex[index].position.data())));
    }
    if (diameter <= resolution) break;
    const auto along = [&](double factor, double rejection_floor) {
      Trial trial{};
      for (int axis = 0; axis < 3; ++axis) {
        trial.position[axis] = centroid[axis] +
            factor * (centroid[axis] - simplex[count].position[axis]);
        if (dimension == 2 && shared[axis]) trial.position[axis] = best[axis];
      }
      trial.quality = evaluate(trial.position, rejection_floor);
      return trial;
    };
    const Trial reflection = along(1.0, simplex[count].quality);
    if (reflection.quality > simplex[0].quality) {
      const Trial expansion = along(2.0, reflection.quality);
      simplex[count] = expansion.quality > reflection.quality ? expansion : reflection;
    } else if (reflection.quality > simplex[count - 1].quality) {
      simplex[count] = reflection;
    } else {
      const bool outside = reflection.quality > simplex[count].quality;
      const Trial contraction = along(outside ? 0.5 : -0.5,
          outside ? reflection.quality : simplex[count].quality);
      if (contraction.quality > (outside ? reflection.quality : simplex[count].quality)) {
        simplex[count] = contraction;
      } else {
        for (std::size_t index = 1; index <= count; ++index) {
          for (int axis = 0; axis < 3; ++axis) {
            simplex[index].position[axis] = simplex[0].position[axis] +
                0.5 * (simplex[index].position[axis] - simplex[0].position[axis]);
          }
          simplex[index].quality = evaluate(simplex[index].position);
        }
      }
    }
  }
  return !mesh.work_exhausted() && native_execution_refusal() == PHX_MC_OK;
}

bool select_smoothing_target(TetMesh& mesh, int32_t v, int8_t dimension,
                             std::span<const int32_t> star,
                             NativeVector<int32_t>& neighbors, double requested_quality,
                             SmoothingSelection& selected) {
  const double* p = mesh.point(v);
  SmoothingTargets prepared{.origin = {p[0], p[1], p[2]}, .targets = {}};
  if (!prepare_smoothing_targets(mesh, v, dimension, star, neighbors, prepared)) return false;
  const double* origin = prepared.origin;
  const bool* shared = prepared.shared;
  const auto& targets = prepared.targets;
  if (!mesh.spend(static_cast<int64_t>(8 * star.size()))) return false;
  int32_t critical_cell = -1;
  const double baseline = finite_quality(mesh, star, &critical_cell);
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
      if (!geometry::point_in_domain(candidate) ||
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
  if (!optimize_star_target(mesh, v, dimension, star, shared, neighbors, requested_quality,
                            critical_cell, best_quality, best)) return false;
  if (mesh.work_exhausted() || native_execution_refusal() != PHX_MC_OK) return false;
  selected = {baseline, best_quality, {best[0], best[1], best[2]}};
  return true;
}

// A cell owns at most four moving vertices. Optimize their complete incident
// union jointly: an improving simultaneous step need not admit any individual
// coordinate step. This small native inner kernel cannot cross the host
// boundary per trial (the owning work ledger and exact transaction remain
// native). It is not an additional sweep or an optimizer service.
struct CoupledCavity {
  TetMesh& mesh;
  NativeVector<int32_t> vertices;
  NativeVector<int32_t> cells;
  std::array<double, kCoupledVariables> origin{};
  std::array<int, kCoupledVariables> variables{};
  std::size_t count = 0;
  double scale = std::numeric_limits<double>::infinity();

  bool prepare(std::span<const int32_t> candidates) {
    if (std::count_if(candidates.begin(), candidates.end(), [&](int32_t vertex) {
          return mesh.dimension(vertex) >= 2 && mesh.protection(vertex) == 0.0 &&
                 (mesh.dimension(vertex) == 3 || mesh.policy() != BoundaryPolicy::kFixed);
        }) < 2) return false;
    NativeVector<int32_t> star, neighbors;
    NativeVector<int32_t> sorted(candidates.begin(), candidates.end());
    std::sort(sorted.begin(), sorted.end());
    for (int32_t vertex : sorted) {
      const int8_t dimension = mesh.dimension(vertex);
      if (dimension < 2 || mesh.protection(vertex) > 0.0 ||
          (dimension < 3 && mesh.policy() == BoundaryPolicy::kFixed)) continue;
      if (!mesh.vertex_star(vertex, star)) return false;
      bool shared[3] = {true, true, true};
      neighbors.clear();
      if (!prepare_smoothing_chart(mesh, vertex, dimension, star, neighbors,
                                    mesh.point(vertex), shared)) continue;
      // Axis-aligned facet coordinates are exactly preserved by retaining
      // their fixed components. Other facets use the existing source-aware
      // single-vertex constructor, not a rounded tangent-plane surrogate.
      if (dimension == 2 && !shared[0] && !shared[1] && !shared[2]) continue;
      const std::size_t row = vertices.size();
      vertices.push_back(vertex);
      for (int axis = 0; axis < 3; ++axis) {
        origin[3 * row + axis] = mesh.point(vertex)[axis];
        if (dimension == 3 || !shared[axis]) {
          variables[count++] = static_cast<int>(3 * row + axis);
        }
      }
      for (int32_t neighbor : neighbors) {
        scale = std::min(scale,
            std::sqrt(geometry::squared_distance(mesh.point(vertex), mesh.point(neighbor))));
      }
      cells.insert(cells.end(), star.begin(), star.end());
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return vertices.size() > 1 && count > 0 && scale > 0.0 && std::isfinite(scale) &&
           native_execution_cavity(cells.size());
  }

  struct Trial {
    std::array<double, kCoupledVariables> x{};
    std::array<double, kCoupledVariables> gradient{};
    double energy = 0.0;
    double quality = 180.0;
  };

  std::array<double, kCoupledVariables> positions(const Trial& trial) const {
    auto result = origin;
    for (std::size_t i = 0; i < count; ++i) {
      result[variables[i]] += scale * trial.x[i];
    }
    return result;
  }

  bool evaluate(Trial& trial, double threshold) {
    const auto proposed = positions(trial);
    for (std::size_t row = 0; row < vertices.size(); ++row) {
      if (!geometry::finite3(proposed.data() + 3 * row) ||
          !geometry::point_in_domain(proposed.data() + 3 * row)) return false;
    }
    trial.energy = 0.0;
    double maximum_cosine = -1.0;
    trial.gradient.fill(0.0);
    std::array<double, kCoupledVariables> coordinate_gradient{};
    // Consistent outward face order; every pair defines an internal dihedral.
    static constexpr int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    for (std::size_t index = 0; index < cells.size(); ++index) {
      const int32_t cell = cells[index];
      if (!mesh.finite(cell)) continue;
      if (!mesh.spend(TetWorkCost::kVisit)) return false;
      const double* p[4];
      int rows[4];
      for (int corner = 0; corner < 4; ++corner) {
        const int32_t vertex = mesh.tet(cell).v[corner];
        const auto found = std::lower_bound(vertices.begin(), vertices.end(), vertex);
        rows[corner] = found != vertices.end() && *found == vertex
            ? static_cast<int>(found - vertices.begin()) : -1;
        p[corner] = rows[corner] < 0 ? mesh.point(vertex) : proposed.data() + 3 * rows[corner];
      }
      if (orient3d(p[0], p[1], p[2], p[3]) <= 0) return false;
      double normals[4][3], unit[4][3], lengths[4], adjoint[4][3]{};
      for (int face = 0; face < 4; ++face) {
        double u[3], v[3];
        geometry::difference(p[faces[face][1]], p[faces[face][0]], u);
        geometry::difference(p[faces[face][2]], p[faces[face][0]], v);
        geometry::cross(u, v, normals[face]);
        lengths[face] = std::sqrt(geometry::dot(normals[face], normals[face]));
        if (!(lengths[face] > 0.0) || !std::isfinite(lengths[face])) return false;
        for (int axis = 0; axis < 3; ++axis) {
          unit[face][axis] = normals[face][axis] / lengths[face];
        }
      }
      for (int first = 0; first < 4; ++first) {
        for (int second = first + 1; second < 4; ++second) {
          const double cosine = -geometry::dot(unit[first], unit[second]);
          maximum_cosine = std::max(maximum_cosine, cosine);
          const double violation = cosine - threshold;
          if (!(violation > 0.0)) continue;
          trial.energy += violation * violation;
          for (int axis = 0; axis < 3; ++axis) {
            adjoint[first][axis] -= 2.0 * violation *
                (unit[second][axis] + cosine * unit[first][axis]) / lengths[first];
            adjoint[second][axis] -= 2.0 * violation *
                (unit[first][axis] + cosine * unit[second][axis]) / lengths[second];
          }
        }
      }
      for (int face = 0; face < 4; ++face) {
        const int a = faces[face][0], b = faces[face][1], c = faces[face][2];
        double u[3], v[3], gb[3], gc[3];
        geometry::difference(p[b], p[a], u);
        geometry::difference(p[c], p[a], v);
        geometry::cross(v, adjoint[face], gb);
        geometry::cross(adjoint[face], u, gc);
        for (int axis = 0; axis < 3; ++axis) {
          if (rows[a] >= 0) coordinate_gradient[3 * rows[a] + axis] -= gb[axis] + gc[axis];
          if (rows[b] >= 0) coordinate_gradient[3 * rows[b] + axis] += gb[axis];
          if (rows[c] >= 0) coordinate_gradient[3 * rows[c] + axis] += gc[axis];
        }
      }
    }
    trial.quality = std::acos(std::clamp(maximum_cosine, -1.0, 1.0)) *
                    (180.0 / std::acos(-1.0));
    for (std::size_t i = 0; i < count; ++i) {
      trial.gradient[i] = scale * coordinate_gradient[variables[i]];
    }
    return std::isfinite(trial.energy) && std::isfinite(trial.quality) &&
        std::all_of(trial.gradient.begin(), trial.gradient.end(),
                    [](double value) { return std::isfinite(value); });
  }
};

bool optimize_cavity_target(CoupledCavity& cavity, double target,
                            CoupledCavity::Trial& best, double& initial_quality) {
  TetMesh& mesh = cavity.mesh;
  // A one-ulp stricter computational aim avoids accepting a rounded cosine as
  // evidence at the physical threshold. The transaction checks real angles.
  const double threshold = std::nextafter(std::cos(target * (std::acos(-1.0) / 180.0)),
                                         -std::numeric_limits<double>::infinity());
  auto current = best;
  if (!cavity.evaluate(current, threshold)) return false;
  initial_quality = current.quality;
  best = current;
  NativeVector<double> inverse(cavity.count * cavity.count, 0.0);
  const auto reset = [&]() {
    std::fill(inverse.begin(), inverse.end(), 0.0);
    for (std::size_t i = 0; i < cavity.count; ++i) {
      inverse[i * cavity.count + i] = 1.0;
    }
  };
  reset();
  // The same binary64 shortest-edge spatial resolution as the existing
  // maximin simplex proposal; this only ends an unresolved private solve.
  // It never weakens the requested angle or the exact commit admission.
  const double resolution = std::sqrt(std::numeric_limits<double>::epsilon());
  // This is one coupled proposal. All iterations decrease its private merit;
  // no intermediate relocation, topology publication or frontier visit occurs.
  while (current.quality < target && !mesh.work_exhausted() &&
         native_execution_refusal() == PHX_MC_OK) {
    if (!mesh.spend(static_cast<int64_t>(cavity.count))) return false;
    std::array<double, kCoupledVariables> direction{};
    double slope = 0.0, largest = 0.0;
    for (std::size_t i = 0; i < cavity.count; ++i) {
      for (std::size_t j = 0; j < cavity.count; ++j) {
        direction[i] -= inverse[i * cavity.count + j] * current.gradient[j];
      }
      slope += direction[i] * current.gradient[i];
      largest = std::max(largest, std::abs(direction[i]));
    }
    if (!(slope < 0.0) || !std::isfinite(largest)) break;
    // Keep the first line-search trial within one local shortest edge. This
    // is a step bound, not a quota or a new pass allowance.
    double step = largest > 1.0 ? 1.0 / largest : 1.0;
    CoupledCavity::Trial next;
    bool accepted = false;
    // Near zero squared merit, spatial convergence alone cannot decide the
    // hard angle: retain full binary64 steps until feasibility or arithmetic
    // stagnation. Away from zero, the established spatial resolution ends a
    // stationary unsuccessful solve without refining numerical noise.
    const double step_resolution = current.energy <= std::numeric_limits<double>::epsilon()
        ? std::numeric_limits<double>::epsilon() : resolution;
    while (step * largest > step_resolution) {
      next = current;
      bool changed = false;
      for (std::size_t i = 0; i < cavity.count; ++i) {
        next.x[i] += step * direction[i];
        changed = changed || next.x[i] != current.x[i];
      }
      if (!changed) break;
      if (cavity.evaluate(next, threshold) &&
          next.energy <= current.energy + 1e-4 * step * slope) {
        accepted = true;
        break;
      }
      if (mesh.work_exhausted() || native_execution_refusal() != PHX_MC_OK) return false;
      step *= 0.5;
    }
    if (!accepted) break;
    if (next.quality > best.quality) best = next;
    if (!mesh.spend(static_cast<int64_t>(cavity.count))) return false;
    std::array<double, kCoupledVariables> s{}, y{}, hy{};
    double sy = 0.0, yhy = 0.0;
    for (std::size_t i = 0; i < cavity.count; ++i) {
      s[i] = next.x[i] - current.x[i];
      y[i] = next.gradient[i] - current.gradient[i];
      sy += s[i] * y[i];
    }
    for (std::size_t i = 0; i < cavity.count; ++i) {
      for (std::size_t j = 0; j < cavity.count; ++j) {
        hy[i] += inverse[i * cavity.count + j] * y[j];
      }
      yhy += y[i] * hy[i];
    }
    if (sy > 0.0 && std::isfinite(1.0 / sy) && std::isfinite(yhy / sy)) {
      for (std::size_t i = 0; i < cavity.count; ++i) {
        for (std::size_t j = 0; j < cavity.count; ++j) {
          inverse[i * cavity.count + j] += ((sy + yhy) / sy) * (s[i] * s[j] / sy) -
                           (hy[i] * s[j] + s[i] * hy[j]) / sy;
        }
      }
    } else {
      reset();
    }
    current = next;
  }
  return !mesh.work_exhausted() && native_execution_refusal() == PHX_MC_OK;
}

bool coupled_ring_candidates(TetMesh& mesh, const Cell& cell, const CoupledCavity& cavity,
                              const CoupledCavity::Trial& best,
                              NativeVector<int32_t>& candidates) {
  if (!mesh.spend(TetWorkCost::kVisit)) return false;
  const auto proposed = cavity.positions(best);
  const double* p[4];
  for (int corner = 0; corner < 4; ++corner) {
    const auto found = std::lower_bound(cavity.vertices.begin(), cavity.vertices.end(), cell[corner]);
    p[corner] = found != cavity.vertices.end() && *found == cell[corner]
        ? proposed.data() + 3 * (found - cavity.vertices.begin()) : mesh.point(cell[corner]);
  }
  static constexpr int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  double unit[4][3];
  for (int face = 0; face < 4; ++face) {
    double u[3], v[3];
    geometry::difference(p[faces[face][1]], p[faces[face][0]], u);
    geometry::difference(p[faces[face][2]], p[faces[face][0]], v);
    geometry::cross(u, v, unit[face]);
    const double length = std::sqrt(geometry::dot(unit[face], unit[face]));
    if (!(length > 0.0) || !std::isfinite(length)) return false;
    for (double& value : unit[face]) value /= length;
  }
  int first_face = 0, second_face = 1;
  double maximum = -2.0;
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      const double cosine = -geometry::dot(unit[first], unit[second]);
      if (cosine > maximum) {
        maximum = cosine;
        first_face = first;
        second_face = second;
      }
    }
  }
  int32_t edge[2];
  std::size_t endpoint = 0;
  for (int corner = 0; corner < 4; ++corner) {
    if (corner != first_face && corner != second_face) edge[endpoint++] = cell[corner];
  }
  NativeVector<int32_t> ring_cells, ring;
  if (!mesh.edge_ring(edge[0], edge[1], ring_cells, ring) ||
      ring.size() < 3 || ring.size() > kMaxRing) return false;
  // A boundary-edge ring contains the ghost closure vertex. It belongs to
  // the topological ring, not the physical coordinate/degree-of-freedom bank.
  // Remove it at this conversion boundary, before either eligibility lookup
  // below or CoupledCavity::prepare can index per-vertex storage.
  candidates.clear();
  candidates.reserve(ring.size() + 2);
  for (int32_t vertex : ring) {
    if (vertex != kGhostVertex) candidates.push_back(vertex);
  }
  candidates.push_back(edge[0]);
  candidates.push_back(edge[1]);
  std::sort(candidates.begin(), candidates.end());
  // Expanding a variable set must actually add an eligible degree of freedom.
  return std::any_of(candidates.begin(), candidates.end(), [&](int32_t vertex) {
    return !std::binary_search(cavity.vertices.begin(), cavity.vertices.end(), vertex) &&
        mesh.dimension(vertex) >= 2 && mesh.protection(vertex) == 0.0 &&
        (mesh.dimension(vertex) == 3 || mesh.policy() != BoundaryPolicy::kFixed);
  });
}

bool optimize_coupled_cell(TetMesh& mesh, const Cell& cell, double target,
                           AffectedVertices& affected_vertices,
                           TetMeshImproveCounters& counters) {
  CoupledCavity cavity{.mesh = mesh, .vertices = {}, .cells = {}};
  if (!cavity.prepare(cell)) return false;
  CoupledCavity::Trial best;
  double baseline = 0.0;
  if (!optimize_cavity_target(cavity, target, best, baseline)) return false;
  CoupledCavity expanded{.mesh = mesh, .vertices = {}, .cells = {}};
  NativeVector<int32_t> ring;
  CoupledCavity::Trial expanded_best;
  const CoupledCavity* selected = &cavity;
  const CoupledCavity::Trial* selection = &best;
  if (best.quality < target && coupled_ring_candidates(mesh, cell, cavity, best, ring) &&
      expanded.prepare(ring)) {
    // Expand the same private proposal, not the published mesh or its frontier.
    // Existing coordinates retain their candidate displacement in the new
    // shortest-edge chart; newly admitted ring vertices start at their origin.
    const auto proposed = cavity.positions(best);
    for (std::size_t i = 0; i < expanded.count; ++i) {
      const int coordinate = expanded.variables[i];
      const int32_t vertex = expanded.vertices[static_cast<std::size_t>(coordinate / 3)];
      const auto found = std::lower_bound(cavity.vertices.begin(), cavity.vertices.end(), vertex);
      if (found == cavity.vertices.end() || *found != vertex) continue;
      const std::size_t old = 3 * (found - cavity.vertices.begin()) + coordinate % 3;
      expanded_best.x[i] = (proposed[old] - expanded.origin[coordinate]) / expanded.scale;
    }
    double initial_quality = 0.0;
    if (!optimize_cavity_target(expanded, target, expanded_best, initial_quality)) return false;
    // Only a complete feasibility solution supersedes the original candidate.
    // An unsuccessful larger cavity cannot substitute its unrelated local
    // minimum for the owning cell's already-earned strict gain.
    if (expanded_best.quality >= target) {
      if (!mesh.spend(static_cast<int64_t>(expanded.cells.size()))) return false;
      baseline = finite_quality(mesh, expanded.cells);
      selected = &expanded;
      selection = &expanded_best;
    }
  }
  if (mesh.work_exhausted() || native_execution_refusal() != PHX_MC_OK ||
      !(selection->quality > baseline + kImprovement)) return false;
  const auto positions = selected->positions(*selection);
  const double admission_floor = selected == &expanded
      ? std::max(target, baseline + kImprovement) : baseline + kImprovement;
  if (!try_relocate_vertices(mesh, selected->vertices, positions.data(), 0.0,
                              admission_floor)) return false;
  for (std::size_t row = 0; row < selected->vertices.size(); ++row) {
    affected_vertices[row] = selected->vertices[row];
  }
  counters.relocations += static_cast<int64_t>(selected->vertices.size());
  return true;
}

bool smooth_vertex(TetMesh& mesh, int32_t v, bool& removed, int32_t& accepted_neighbor,
                   ImprovementProofs& failures, double requested_quality) {
  removed = false;
  const int8_t dimension = mesh.dimension(v);
  NativeVector<int32_t> star;
  NativeVector<int32_t> neighbors;
  if (dimension < 1 || mesh.protection(v) > 0.0 ||
      (dimension < 3 && mesh.policy() == BoundaryPolicy::kFixed) || !mesh.vertex_star(v, star)) {
    return false;
  }
  if (!native_execution_cavity(star.size())) return false;
  if (dimension == 1) return smooth_segment_vertex(mesh, v, star, neighbors);
  // A fresh improving retirement needs no coordinate optimization. Failed
  // retirement geometry is reusable by the same generation's proof bank.
  if (dimension == 3 &&
      remove_vertex_above(mesh, v, true, -std::numeric_limits<double>::infinity(),
                          &accepted_neighbor, &failures)) {
    removed = true;
    return true;
  }
  failures.synchronize(mesh);
  SmoothingSelection selected{};
  const SmoothingSelection* selection = &selected;
  const auto found = failures.smoothing.find(v);
  if (found != failures.smoothing.end()) {
    selection = &found->second;
  } else {
    if (!select_smoothing_target(mesh, v, dimension, star, neighbors, requested_quality,
                                 selected)) return false;
    // Selection retains the exact candidate traversal and float bit patterns.
    // Each removal/relocation admission is still retried below.
    if (exact_star_witnesses(mesh, star)) failures.smoothing.emplace(v, selected);
  }
  if (mesh.work_exhausted()) return false;
  return selection->quality > selection->baseline + kImprovement &&
         try_relocate(mesh, v, selection->position.data(), true);
}

bool remove_cell_face_cluster(TetMesh& mesh, int32_t cell, int32_t& accepted_apex) {
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
  // Extract the unchanged cavity boundary once. Rebuilding and sorting it
  // independently for every apex obscures identical geometry work; one
  // below-baseline cone already rules out that entire candidate.
  if (!mesh.spend(static_cast<int64_t>(8 * cavity.size()))) return false;
  std::sort(cavity.begin(), cavity.end());
  NativeVector<std::pair<Cell, int>> boundary;
  for (int32_t entry : cavity) {
    const Tetrahedron& tet = mesh.tet(entry);
    for (int slot = 0; slot < 4; ++slot) {
      if (!std::binary_search(cavity.begin(), cavity.end(), tet.n[slot])) {
        boundary.push_back({{tet.v[0], tet.v[1], tet.v[2], tet.v[3]}, slot});
      }
    }
  }
  const double baseline = finite_quality(mesh, cavity);
  NativeVector<Cell> proposal;
  proposal.reserve(boundary.size());
  for (int32_t apex : apices) {
    proposal.clear();
    bool improving = true;
    for (const auto& [cell, slot] : boundary) {
      if (!mesh.spend(TetWorkCost::kVisit)) return false;
      if (std::find(cell.begin(), cell.end(), apex) != cell.end() && cell[slot] != apex) {
        continue;
      }
      Cell cone = cell;
      cone[slot] = apex;
      if (!mesh.positive(cone.data()) ||
          !(quality(mesh, cone.data()) > baseline + kImprovement)) {
        improving = false;
        break;
      }
      proposal.push_back(cone);
    }
    if (improving && !proposal.empty() &&
        reconnect(mesh, cavity, proposal) == EditStatus::kOk) {
      accepted_apex = apex;
      return true;
    }
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

// Along this ray every cone determinant is affine. Intersect their positive
// intervals rather than imposing a minimum sampling step on a narrow kernel.
// Expansion estimates locate a candidate only; the caller still certifies
// every rounded child against exact positivity and the unchanged chart floor.
double star_visibility_step(TetMesh& mesh, std::span<const int32_t> star,
                            int32_t a, int32_t b, const double* midpoint,
                            const double* centroid) {
  double lower = 0.0, upper = 1.0;
  for (int32_t t : star) {
    if (!mesh.spend(TetWorkCost::kVisit)) return 0.0;
    for (int32_t endpoint : {a, b}) {
      const double* corners[4];
      mesh.corners(mesh.tet(t).v, corners);
      const int slot = vertex_slot(mesh.tet(t), endpoint);
      corners[slot] = midpoint;
      const Expansion first = orient3d_exact(corners[0], corners[1], corners[2], corners[3]);
      corners[slot] = centroid;
      const Expansion last = orient3d_exact(corners[0], corners[1], corners[2], corners[3]);
      if (first.sign() > 0 && last.sign() > 0) continue;
      if (first.sign() <= 0 && last.sign() <= 0) return 0.0;
      const double root = first.estimate() / (first - last).estimate();
      if (!std::isfinite(root)) return 0.0;
      if (first.sign() <= 0) lower = std::max(lower, root);
      else upper = std::min(upper, root);
      if (!(lower < upper)) return 0.0;
    }
  }
  return lower + 0.5 * (upper - lower);
}

FloorInsertion insert_floor_vertex(TetMesh& mesh, const Cell& cell,
                                   double minimum_relative_determinant,
                                   AffectedVertices& affected_vertices) {
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
      for (std::size_t trial = 0; trial <= std::size(kSteps); ++trial) {
        double step;
        if (trial < std::size(kSteps)) {
          step = kSteps[trial];
        } else {
          if (best_quality >= 0.0) break;
          step = star_visibility_step(mesh, star, a, b, midpoint, centroid);
          if (!(step > 0.0 && step < 1.0)) break;
        }
        double position[3];
        for (int axis = 0; axis < 3; ++axis) {
          position[axis] = midpoint[axis] + step * (centroid[axis] - midpoint[axis]);
        }
        if (!geometry::point_in_domain(position)) continue;
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


bool try_edge_removal_proposal(
    TetMesh& mesh, int32_t a, int32_t b, bool require_improvement,
    std::optional<std::span<const double>> regular_weights, double radius_edge_bound,
    double max_weight_fraction, bool construction_objective,
    double minimum_relative_determinant, ImprovementProofs* failures);


bool improve_cell(TetMesh& mesh, int32_t t, TetMeshImproveCounters& counters, bool construction_uncertain,
                  bool below_floor, double minimum_relative_determinant, double requested_quality,
                  AffectedVertices& affected_vertices, bool& capacity,
                  ImprovementProofs& failures) {
  const Cell v = {mesh.tet(t).v[0], mesh.tet(t).v[1], mesh.tet(t).v[2], mesh.tet(t).v[3]};
  // These are covering anchors, not the old cell's corners: every changed
  // surviving/new cell meets an anchor. Unchanged neighboring stars need not
  // be gathered, reassessed or requeued as if their geometry had changed.
  affected_vertices.fill(-1);
  const auto optimize_vertices = [&]() {
    if (optimize_coupled_cell(mesh, v, requested_quality, affected_vertices, counters)) return true;
    if (mesh.work_exhausted()) return false;
    bool progress = false;
    std::size_t anchors = 0;
    // Complete this cell's coordinate proposal over its four actual vertices,
    // rather than spending another requested pass for each partial relocation.
    for (int32_t vertex : v) {
      bool removed = false;
      int32_t surviving = -1;
      if (smooth_vertex(mesh, vertex, removed, surviving, failures, requested_quality)) {
        affected_vertices[anchors++] = removed ? surviving : vertex;
        ++(removed ? counters.vertex_removals : counters.relocations);
        progress = true;
        if (removed || quality(mesh, mesh.tet(t).v) >= requested_quality) return true;
      }
      if (mesh.work_exhausted()) return progress;
    }
    return progress;
  };
  for (int k = 0; k < 4; ++k) {
    if (try_face_removal(mesh, t, k, true)) {
      affected_vertices[0] = v[k];
      ++counters.face_removals;
      return true;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (try_edge_removal_proposal(mesh, v[i], v[j], true, std::nullopt,
                                    std::numeric_limits<double>::infinity(), 0.25,
                                    construction_uncertain, minimum_relative_determinant,
                                    &failures)) {
        affected_vertices[0] = v[i];
        affected_vertices[1] = v[j];
        ++counters.edge_removals;
        return true;
      }
    }
  }
  if (remove_cell_face_cluster(mesh, t, affected_vertices[4])) {
    ++counters.multiface_removals;
    return true;
  }
  if (optimize_vertices()) return true;
  if (below_floor) {
    switch (insert_floor_vertex(mesh, v, minimum_relative_determinant, affected_vertices)) {
      case FloorInsertion::kApplied:
        ++counters.vertex_insertions;
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
  // One ordinary angle visit per topological cell per requested pass.
  // New connectivity is a new child; relocation is not another hidden sweep.
  NativeSet<Cell> processed;
  NativeVector<int32_t> star;
  NativeVector<int32_t> cells;
  NativeVector<Candidate> affected;

  void begin(const TetMesh& mesh, const NativeVector<Candidate>& queue) {
    pending.assign(mesh.complex().tets.size(), kAbsent);
    processed.clear();
    for (std::size_t index = 0; index < queue.size(); ++index) {
      native_execution_charge(0);
      pending[static_cast<std::size_t>(queue[index].slot)] = index;
    }
  }

  bool append(TetMesh& mesh, const AffectedVertices& vertices,
              double minimum_relative_determinant, double requested_quality,
              NativeVector<Candidate>& queue, ImprovementProofs& proofs) {
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
    if (!proofs.advance(mesh, cells)) return false;
    affected.clear();
    for (int32_t slot : cells) {
      native_execution_charge(0);
      if (!mesh.complex().live(slot) || !mesh.finite(slot)) continue;
      const ConstructionAssessment assessment = assess_construction(
          mesh, mesh.tet(slot).v, minimum_relative_determinant);
      const double q = quality(mesh, mesh.tet(slot).v);
      const bool uncertain = assessment.orientation.certified_sign() != 1 || assessment.below_floor;
      if (!uncertain && std::isfinite(q) && q >= requested_quality) continue;
      Candidate entry{q, slot, {0, 0, 0, 0}, uncertain,
                      assessment.below_floor ? assessment.floor_margin
                          : assessment.orientation.value - assessment.orientation.bound,
                      assessment.below_floor};
      std::copy_n(mesh.tet(slot).v, 4, entry.v);
      std::sort(entry.v, entry.v + 4);
      if (!uncertain && processed.contains(Cell{entry.v[0], entry.v[1], entry.v[2], entry.v[3]})) {
        continue;
      }
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

// Regularity of a weighted proposal at the current weights. `varying` is the
// vertex whose trial weight a caller raises between calls (-1: none). Size,
// shape and weight-bound refusals, and power refusals that never read the
// varying weight, hold at every larger trial weight: a weight bound
// max_weight_fraction * |xy|^2 only tightens as the weight grows.
enum class Regularity : uint8_t { kRegular, kRefusedAtWeight, kRefused };

Regularity regular_proposal(TetMesh& mesh, std::span<const Cell> cells,
                            std::span<const double> weights, double ratio_bound,
                            double max_weight_fraction, int32_t varying) {
  if (!mesh.spend(static_cast<int64_t>(8 * cells.size() * cells.size()))) {
    return Regularity::kRefused;
  }
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (!within_size_targets(mesh, cells[i], ratio_bound)) {
      return Regularity::kRefused;
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        const int32_t x = cells[i][static_cast<std::size_t>(a)];
        const int32_t y = cells[i][static_cast<std::size_t>(b)];
        const double bound = max_weight_fraction *
                             geometry::squared_distance(mesh.point(x), mesh.point(y));
        if (weights[static_cast<std::size_t>(x)] > bound ||
            weights[static_cast<std::size_t>(y)] > bound) {
          return Regularity::kRefused;
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
        const bool reads_varying =
            opposite == varying ||
            std::find(cells[i].begin(), cells[i].end(), varying) != cells[i].end();
        return reads_varying ? Regularity::kRefusedAtWeight : Regularity::kRefused;
      }
    }
  }
  return Regularity::kRegular;
}

// The 2-3 flip of t's unconstrained finite face opposite `slot`: three
// positive children of t and its neighbor, admitted when the smallest
// dihedral strictly improves (if required). Never reads weights.
bool face_removal_proposal(TetMesh& mesh, int32_t t, int slot, bool require_improvement,
                           std::array<int32_t, 2>& removed, std::array<Cell, 3>& proposed) {
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
  if (back < 0) {
    return false;
  }
  std::size_t count = 0;
  for (int j = 0; j < 4; ++j) {
    if (j == back) {
      continue;
    }
    Cell cell = {second.v[0], second.v[1], second.v[2], second.v[3]};
    cell[static_cast<std::size_t>(j)] = first.v[slot];
    if (!mesh.positive(cell.data())) {
      return false;
    }
    proposed[count++] = cell;
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
  removed = {t, u};
  return true;
}

// Weight-independent edge removal on the gathered ring of (a, b): ring
// admission, the optimal ring triangulation and its objective test. On
// success appends the 2n - 4 proposed cells; a completed geometric negative
// enters `failures` (when given) exactly as an unplanned proposal would.
bool plan_edge_removal(TetMesh& mesh, int32_t a, int32_t b, std::span<const int32_t> tets,
                       std::span<const int32_t> ring, bool require_improvement,
                       bool construction_objective, double minimum_relative_determinant,
                       ImprovementProofs* failures, const EdgeRemovalFailureKey& failure_key,
                       NativeVector<Cell>& proposed) {
  const int n = static_cast<int>(ring.size());
  if (n < 3 || n > kMaxRing) {
    return false;
  }
  // Preserve the existing full n^3 work charge, factorized into n^2
  // bookkeeping and the geometric triangle tests. Only a mathematically
  // dominated subpolygon can omit its triangle test and that test's work.
  const int64_t full_work = static_cast<int64_t>(n * n * n);
  const int64_t bookkeeping = static_cast<int64_t>(n * n);
  const int64_t triangle_count = static_cast<int64_t>(n * (n - 1) * (n - 2) / 6);
  const int64_t geometric_work = full_work - bookkeeping;
  if (!mesh.spend(bookkeeping)) return false;
  for (int i = 0; i < n; ++i) {
    // tets[i] = (a, b, ring[i], ring[i + 1]); its facet opposite ring[i] is
    // the ring face (a, b, ring[i + 1]).
    if (ring[static_cast<std::size_t>(i)] < 0 ||
        mesh.complex().constraint(tets[static_cast<std::size_t>(i)],
                                  vertex_slot(mesh.tet(tets[static_cast<std::size_t>(i)]),
                                              ring[static_cast<std::size_t>(i)])) !=
            kNoConstraint) {
      // Constraint admission is unchanged, including its original work.
      mesh.spend(geometric_work);
      return false;
    }
  }
  // Bounded source witnesses may leave source orientation unresolved.
  // Only exact represented-coordinate legality can certify a negative DP.
  if (failures != nullptr) {
    if (mesh.witness(a).deviation > 0.0 || mesh.witness(b).deviation > 0.0) {
      failures = nullptr;
    } else {
      for (int32_t vertex : ring) {
        if (mesh.witness(vertex).deviation > 0.0) {
          failures = nullptr;
          break;
        }
      }
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
  const double threshold = require_improvement && !construction_objective
      ? finite_quality(mesh, tets) + kImprovement : -std::numeric_limits<double>::infinity();
  int64_t triangle_index = 0;
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
        const int64_t evaluation_work = geometric_work / triangle_count +
            (triangle_index++ < geometric_work % triangle_count ? 1 : 0);
        // An accepted triangulation's minimum must exceed the threshold on
        // BOTH subpolygons. Its root triangle cannot rescue a dominated child.
        if (left < 0.0 || right < 0.0 || left <= threshold || right <= threshold) continue;
        if (!mesh.spend(evaluation_work)) return false;
        const double value = std::min({left, right, triangle(i, k, j)});
        if (value > threshold && value > best[i][j]) {
          best[i][j] = value;
          choice[i][j] = k;
        }
      }
    }
  }
  const double result = best[0][n - 1];
  if (result < 0.0 || !(result > threshold)) {
    if (failures != nullptr) failures->proved.insert(failure_key);
    return false;
  }
  if (construction_objective) {
    double previous = std::numeric_limits<double>::infinity();
    for (int32_t cell : tets) previous = std::min(previous, construction_quality(mesh.tet(cell).v));
    if (!(result > previous)) {
      return false;
    }
  }
  // Bounded LIFO of pending ring sub-polygons; at most n - 2 are ever open.
  std::array<std::array<int, 2>, kMaxRing> pending{};
  std::size_t open = 0;
  pending[open++] = {0, n - 1};
  while (open > 0) {
    const auto [i, j] = pending[--open];
    const int k = choice[i][j];
    Cell above;
    Cell below;
    cells_of(i, k, j, above, below);
    proposed.push_back(above);
    proposed.push_back(below);
    if (k - i >= 2) {
      pending[open++] = {i, k};
    }
    if (j - k >= 2) {
      pending[open++] = {k, j};
    }
  }
  return true;
}

// Exudation trial weights bound * 2^e, e = -12..0, in increasing order.
constexpr std::size_t kTrialWeights = 13;

// One star cell's 2-3 exudation trial across a vertex's increasing trial
// weights. Its geometry, size/floor admission and improving proposal never
// read the trial weight. The new-edge bound holds on a prefix of the
// weights. The exact power test is the sign of an affine function of the
// trial weight (every lift entry shifts by it, exactly), so it is
// nonpositive between two nonpositive weights and otherwise positive on a
// suffix of the weights that bisection finds.
struct FaceTrial {
  enum class State : uint8_t { kUnvisited, kRefused, kTracking };
  State state = State::kUnvisited;
  int8_t slot = -1;
  int8_t back = -1;
  int8_t shape = -1;  // size targets and owning floor: -1 unknown, 0 refused, 1 admitted
  bool proposed = false;
  int32_t other = -1;
  int32_t first_positive = 0;  // first weight index with a positive power test
  int32_t last = -1;           // last weight index within the new-edge bound
  std::array<int32_t, 2> removed{};
  std::array<Cell, 3> cells{};
};

// One neighbor's edge removal. Ring, triangulation and improvement never
// read weights; the plan is made once per vertex visit and only its
// regularity and transaction are tried per weight.
struct EdgeTrial {
  enum class State : uint8_t { kUnplanned, kRefused, kPlanned };
  State state = State::kUnplanned;
  int8_t floor = -1;  // proposal above the owning floor: -1 unknown, 0 no, 1 yes
  std::size_t tets_begin = 0;
  std::size_t tets_count = 0;
  std::size_t cells_begin = 0;
  std::size_t cells_count = 0;
};

// A complete refused visit is reused while no accepted flip has changed the
// data it read (see mark_exuded): the same trials are counted, nothing is
// recomputed.
struct VertexRefusal {
  int64_t dirty = 0;    // epoch of the last flip that changed data the visit reads
  int64_t refused = 0;  // epoch of the recorded complete refusal (0: none)
  int64_t trials = 0;   // trials that refusal counted
};

// Reusable scratch of one exude_mesh invocation; capacity grows to the
// largest visited star and is never reallocated per trial.
struct ExudeScratch {
  NativeVector<int32_t> star;
  NativeVector<int32_t> neighbors;
  NativeVector<FaceTrial> faces;
  NativeVector<EdgeTrial> edges;
  NativeVector<int32_t> tets;
  NativeVector<int32_t> ring;
  NativeVector<Cell> cells;
  NativeVector<VertexRefusal> refusals;
  int64_t epoch = 1;
  bool reusable = false;
};

// Whether `order` lists the vertices of `cell` as an even permutation.
bool even_order(const int32_t* cell, const int32_t* order) {
  int position[4];
  for (int k = 0; k < 4; ++k) {
    position[k] = static_cast<int>(std::find(cell, cell + 4, order[k]) - cell);
  }
  int inversions = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      inversions += position[i] > position[j] ? 1 : 0;
    }
  }
  return inversions % 2 == 0;
}

// TetMesh::edge_ring of (a, b) started at the first cell of a's gathered
// star that contains b (the cell edge_tet selects from the same star walk),
// so the star is not walked again. Appends the ring cells and fills `ring`;
// charges the same one unit per ring cell.
bool star_edge_ring(TetMesh& mesh, std::span<const int32_t> star, int32_t a, int32_t b,
                    NativeVector<int32_t>& tets, NativeVector<int32_t>& ring) {
  ring.clear();
  const auto found = std::find_if(star.begin(), star.end(), [&](int32_t t) {
    return vertex_slot(mesh.tet(t), b) >= 0;
  });
  if (found == star.end()) {
    return false;
  }
  const std::size_t begin = tets.size();
  const int32_t start = *found;
  int32_t t = start;
  int32_t entry = -1;
  for (;;) {
    const Tetrahedron& current = mesh.tet(t);
    int others[2];
    int count = 0;
    for (int k = 0; k < 4; ++k) {
      if (current.v[k] != a && current.v[k] != b) {
        others[count++] = k;
      }
    }
    int32_t first = current.v[others[0]];
    int32_t second = current.v[others[1]];
    const int32_t order[4] = {a, b, first, second};
    if (!even_order(current.v, order)) {
      std::swap(first, second);
    }
    if (entry >= 0 && first != entry) {
      tets.resize(begin);
      return false;
    }
    tets.push_back(t);
    ring.push_back(first);
    // Next around the edge: across the facet (a, b, second).
    const int32_t next = current.n[vertex_slot(current, first)];
    if (next == start) {
      break;
    }
    if (tets.size() - begin > mesh.complex().tets.size()) {
      tets.resize(begin);
      return false;
    }
    entry = second;
    t = next;
  }
  return mesh.spend(static_cast<int64_t>(tets.size() - begin));
}

// First visit of the 2-3 trial across star cell t's face opposite `vertex`,
// at weight `index`; the caller charged it. Records the weight interval of
// the bound tests and the first weight with a positive power test. Each
// additional power evaluation is charged as one more trial.
bool track_face(TetMesh& mesh, int32_t vertex, int32_t t, std::span<const double> weights,
                std::span<const double> trials, int32_t index, double max_weight_fraction,
                FaceTrial& face) {
  face.state = FaceTrial::State::kRefused;
  const Tetrahedron& old = mesh.tet(t);
  const int slot = vertex_slot(old, vertex);
  const int32_t other = old.n[slot];
  if (!mesh.finite(other) || mesh.complex().constraint(t, slot) != kNoConstraint) {
    return false;
  }
  const Tetrahedron& neighbor = mesh.tet(other);
  const int back = neighbor_slot(neighbor, t);
  const int32_t apex = neighbor.v[back];
  const double new_edge_bound =
      max_weight_fraction * geometry::squared_distance(mesh.point(vertex), mesh.point(apex));
  if (!(weights[static_cast<std::size_t>(apex)] <= new_edge_bound)) {
    return false;
  }
  auto last = static_cast<int32_t>(trials.size()) - 1;
  while (last >= index && !(trials[static_cast<std::size_t>(last)] <= new_edge_bound)) {
    --last;
  }
  if (last < index) {
    return false;
  }
  const double* p[4];
  mesh.corners(other, p);
  const auto encroached = [&](int32_t at) {
    return power3d(p[0], p[1], p[2], p[3], mesh.point(vertex),
                   weights[static_cast<std::size_t>(neighbor.v[0])],
                   weights[static_cast<std::size_t>(neighbor.v[1])],
                   weights[static_cast<std::size_t>(neighbor.v[2])],
                   weights[static_cast<std::size_t>(neighbor.v[3])],
                   trials[static_cast<std::size_t>(at)]) > 0;
  };
  int32_t first_positive = index;
  if (!encroached(index)) {
    if (last == index || !mesh.spend(16) || !encroached(last)) {
      return false;
    }
    int32_t below = index;
    first_positive = last;
    while (first_positive - below > 1) {
      const int32_t middle = below + (first_positive - below) / 2;
      if (!mesh.spend(16)) {
        return false;
      }
      (encroached(middle) ? first_positive : below) = middle;
    }
  }
  face.state = FaceTrial::State::kTracking;
  face.slot = static_cast<int8_t>(slot);
  face.back = static_cast<int8_t>(back);
  face.other = other;
  face.first_positive = first_positive;
  face.last = last;
  return true;
}

// Weight-independent admission of a tracked 2-3 trial, evaluated once:
// the children's size targets and owning floor, then the improving flip.
bool admit_face(TetMesh& mesh, int32_t vertex, int32_t t, const ExudeOptions& options,
                FaceTrial& face) {
  if (face.shape < 0) {
    const Tetrahedron& neighbor = mesh.tet(face.other);
    bool admitted = true;
    for (int k = 0; k < 4 && admitted; ++k) {
      if (k == face.back) {
        continue;
      }
      Cell proposed = {neighbor.v[0], neighbor.v[1], neighbor.v[2], neighbor.v[3]};
      proposed[static_cast<std::size_t>(k)] = vertex;
      admitted = within_size_targets(mesh, proposed, options.radius_edge_bound) &&
                 above_floor(mesh, std::span<const Cell>(&proposed, 1),
                             options.minimum_relative_determinant);
    }
    face.shape = admitted ? 1 : 0;
  }
  if (face.shape == 0) {
    return false;
  }
  if (!face.proposed) {
    if (!face_removal_proposal(mesh, t, face.slot, true, face.removed, face.cells)) {
      return false;
    }
    face.proposed = true;
  }
  return true;
}

// Plans the removal of edge (vertex, other) from vertex's gathered star.
void plan_exude_edge(TetMesh& mesh, int32_t vertex, int32_t other,
                     double minimum_relative_determinant, ExudeScratch& scratch,
                     EdgeTrial& edge) {
  edge.state = EdgeTrial::State::kRefused;
  if (!mesh.live_vertex(other) || other == vertex || mesh.segment_source(vertex, other) >= 0) {
    return;
  }
  const std::size_t tets_begin = scratch.tets.size();
  if (!star_edge_ring(mesh, scratch.star, vertex, other, scratch.tets, scratch.ring)) {
    return;
  }
  const std::size_t cells_begin = scratch.cells.size();
  const std::span<const int32_t> tets(scratch.tets.data() + tets_begin,
                                      scratch.tets.size() - tets_begin);
  if (!plan_edge_removal(mesh, vertex, other, tets, scratch.ring, true, false,
                         minimum_relative_determinant, nullptr, EdgeRemovalFailureKey{},
                         scratch.cells)) {
    scratch.tets.resize(tets_begin);
    return;
  }
  edge.state = EdgeTrial::State::kPlanned;
  edge.tets_begin = tets_begin;
  edge.tets_count = tets.size();
  edge.cells_begin = cells_begin;
  edge.cells_count = scratch.cells.size() - cells_begin;
}

// One vertex visit. Trials keep their order (per increasing weight: each
// star cell's 2-3 flip, then each neighbor's weighted edge removal) and the
// same first admitted trial is applied. Weight-independent work runs once
// per visit; trials it decides are counted without recomputation.
// `scratch.reusable` reports a complete refusal no transaction refusal entered.
bool exude_vertex(TetMesh& mesh, int32_t vertex, const ExudeOptions& options,
                  std::span<double> weights, TetMeshExudeCounters& counters,
                  ExudeScratch& scratch) {
  scratch.reusable = false;
  if (mesh.dimension(vertex) != 3 || mesh.protection(vertex) > 0.0) {
    return false;
  }
  NativeVector<int32_t>& star = scratch.star;
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
  std::array<double, kTrialWeights> trials{};
  std::size_t count = 0;
  for (int exponent = -12; exponent <= 0; ++exponent) {
    const double trial = std::ldexp(bound, exponent);
    if (trial > weights[static_cast<std::size_t>(vertex)] && weight_in_domain(trial)) {
      trials[count++] = trial;
    }
  }
  const std::span<const double> ordered(trials.data(), count);
  scratch.faces.assign(star.size(), FaceTrial{});
  scratch.neighbors.clear();
  scratch.edges.clear();
  scratch.tets.clear();
  scratch.cells.clear();
  bool planned = false;
  bool transaction_refused = false;
  for (std::size_t position = 0; position < count; ++position) {
    const auto index = static_cast<int32_t>(position);
    const double trial = trials[position];
    for (std::size_t i = 0; i < star.size(); ++i) {
      if (mesh.work_exhausted()) {
        return false;
      }
      FaceTrial& face = scratch.faces[i];
      if (face.state == FaceTrial::State::kUnvisited) {
        if (!mesh.spend(16)) {
          return false;
        }
        ++counters.trials;
        if (!track_face(mesh, vertex, star[i], weights, ordered, index,
                        options.max_weight_fraction, face)) {
          continue;
        }
      } else {
        ++counters.trials;
        if (face.state == FaceTrial::State::kRefused) {
          continue;
        }
        if (index > face.last) {
          face.state = FaceTrial::State::kRefused;
          continue;
        }
      }
      if (index < face.first_positive) {
        continue;
      }
      if (!admit_face(mesh, vertex, star[i], options, face)) {
        face.state = FaceTrial::State::kRefused;
        continue;
      }
      if (mesh.replace(face.removed, face.cells) == EditStatus::kOk) {
        weights[static_cast<std::size_t>(vertex)] = trial;
        ++counters.flips;
        return true;
      }
      transaction_refused = true;
    }
    // Regular exudation needs both directions of bistellar reconnection:
    // a sliver commonly disappears in a 3-2 edge removal, not a 2-3 flip.
    if (!planned) {
      for (int32_t t : star) {
        for (int32_t v : mesh.tet(t).v) {
          if (v != vertex && v >= 0) {
            scratch.neighbors.push_back(v);
          }
        }
      }
      std::sort(scratch.neighbors.begin(), scratch.neighbors.end());
      scratch.neighbors.erase(std::unique(scratch.neighbors.begin(), scratch.neighbors.end()),
                              scratch.neighbors.end());
      scratch.edges.assign(scratch.neighbors.size(), EdgeTrial{});
      planned = true;
    }
    double& weight = weights[static_cast<std::size_t>(vertex)];
    const double saved_weight = weight;
    weight = trial;
    for (std::size_t i = 0; i < scratch.neighbors.size() && !mesh.work_exhausted(); ++i) {
      ++counters.trials;
      EdgeTrial& edge = scratch.edges[i];
      if (edge.state == EdgeTrial::State::kUnplanned) {
        plan_exude_edge(mesh, vertex, scratch.neighbors[i], options.minimum_relative_determinant,
                        scratch, edge);
      }
      if (edge.state != EdgeTrial::State::kPlanned) {
        continue;
      }
      const std::span<const Cell> cells(scratch.cells.data() + edge.cells_begin,
                                        edge.cells_count);
      const Regularity regularity = regular_proposal(
          mesh, cells, weights, options.radius_edge_bound, options.max_weight_fraction, vertex);
      if (regularity != Regularity::kRegular) {
        if (regularity == Regularity::kRefused) {
          edge.state = EdgeTrial::State::kRefused;
        }
        continue;
      }
      if (edge.floor < 0) {
        edge.floor = above_floor(mesh, cells, options.minimum_relative_determinant) ? 1 : 0;
      }
      if (edge.floor == 0) {
        edge.state = EdgeTrial::State::kRefused;
        continue;
      }
      const std::span<const int32_t> tets(scratch.tets.data() + edge.tets_begin,
                                          edge.tets_count);
      if (mesh.replace(tets, cells) == EditStatus::kOk) {
        ++counters.flips;
        return true;
      }
      transaction_refused = true;
    }
    weight = saved_weight;
  }
  scratch.reusable = !transaction_refused;
  return false;
}

// After the accepted flip of `vertex`, marks every vertex whose visit reads
// changed data: vertices of the created cells and of their neighbors (new
// stars, start cells and cells across replaced faces; created and removed
// cells span the same vertices), and vertices of the post-flip star of
// `vertex` and of the cells across its link faces (they read its new
// weight). One visit is charged per created cell; the star walk charges
// its own. An incomplete marking forgets every recorded refusal.
void mark_exuded(TetMesh& mesh, int32_t vertex, ExudeScratch& scratch) {
  ++scratch.epoch;
  const auto mark = [&](int32_t cell) {
    if (cell < 0) return;
    for (int32_t v : mesh.tet(cell).v) {
      if (v >= 0) scratch.refusals[static_cast<std::size_t>(v)].dirty = scratch.epoch;
    }
  };
  bool complete = true;
  for (int32_t cell : mesh.created()) {
    if (!mesh.spend(TetWorkCost::kVisit)) {
      complete = false;
      break;
    }
    mark(cell);
    for (int32_t next : mesh.tet(cell).n) mark(next);
  }
  if (complete && mesh.vertex_star(vertex, scratch.star)) {
    for (int32_t cell : scratch.star) {
      mark(cell);
      mark(mesh.tet(cell).n[vertex_slot(mesh.tet(cell), vertex)]);
    }
  } else {
    for (VertexRefusal& refusal : scratch.refusals) refusal.refused = 0;
  }
}

bool relocation_outside_protection(TetMesh& mesh, int32_t vertex, const double* position,
                                   std::span<const int32_t> star) {
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
  return true;
}

bool relocation_on_curve(TetMesh& mesh, int32_t vertex, const double* position) {
    int32_t neighbors_on_curve[2] = {-1, -1};
    int32_t scientific_source = -1;
    if (!curve_interval(mesh, vertex, neighbors_on_curve, scientific_source) || !mesh.spend(6)) {
      return false;
    }
    const double* first = mesh.point(neighbors_on_curve[0]);
    const double* second = mesh.point(neighbors_on_curve[1]);
    if (!geometry::collinear(first, second, mesh.point(vertex)) ||
        !geometry::collinear(first, second, position) ||
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
  return true;
}

struct RelocationPositions {
  const TetMesh& mesh;
  std::span<const int32_t> vertices;
  const double* positions;

  const double* operator()(int32_t vertex) const {
    const auto found = std::lower_bound(vertices.begin(), vertices.end(), vertex);
    return found != vertices.end() && *found == vertex
               ? positions + 3 * (found - vertices.begin())
               : mesh.point(vertex);
  }
};

bool coordinated_relocation_on_curve(TetMesh& mesh, int32_t vertex, const double* position) {
      int32_t endpoints[2] = {-1, -1};
      int32_t source = -1;
      if (!curve_interval(mesh, vertex, endpoints, source) || !mesh.spend(6)) {
        return false;
      }
      const double* first = mesh.point(endpoints[0]);
      const double* second = mesh.point(endpoints[1]);
      if (!geometry::collinear(first, second, mesh.point(vertex)) ||
          !geometry::collinear(first, second, position)) {
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
  return true;
}

bool prepare_relocation_cavity(TetMesh& mesh, const RelocationPositions& proposed_point,
                               NativeVector<int32_t>& star, NativeVector<int32_t>& neighbors,
                               NativeVector<int32_t>& cavity) {
  const auto vertices = proposed_point.vertices;
  const double* positions = proposed_point.positions;
  for (std::size_t row = 0; row < vertices.size(); ++row) {
    const int32_t vertex = vertices[row];
    const double* position = positions + 3 * row;
    if ((row != 0 && vertices[row - 1] >= vertex) || !mesh.live_vertex(vertex) ||
        mesh.dimension(vertex) < 1 || mesh.protection(vertex) > 0.0 ||
        !geometry::finite3(position) || !geometry::point_in_domain(position) ||
        (mesh.dimension(vertex) < 3 && mesh.policy() == BoundaryPolicy::kFixed) ||
        !mesh.vertex_star(vertex, star)) {
      return false;
    }
    if (!relocation_outside_protection(mesh, vertex, position, star)) return false;
    if (mesh.dimension(vertex) < 3 &&
        !on_facet_plane(mesh, vertex, star, position, neighbors)) {
      return false;
    }
    if (mesh.dimension(vertex) == 1 &&
        !coordinated_relocation_on_curve(mesh, vertex, position)) return false;
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
  return true;
}

bool check_relocation_facets(TetMesh& mesh, const RelocationPositions& proposed_point,
                             int32_t cell, const int32_t* v) {
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
  return true;
}

bool check_relocation_cells(TetMesh& mesh, const RelocationPositions& proposed_point,
                            std::span<const int32_t> cavity, double radius_edge_bound,
                            double minimum_dihedral_degrees) {
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
    if (!check_relocation_facets(mesh, proposed_point, cell, v)) return false;
  }
  return true;
}

bool check_relocation_segments(TetMesh& mesh, const RelocationPositions& proposed_point) {
  const auto vertices = proposed_point.vertices;
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
    if (axis == 3 || !geometry::collinear(old_first, old_second, next_first) ||
        !geometry::collinear(old_first, old_second, next_second) ||
        ((old_first[axis] < old_second[axis]) != (next_first[axis] < next_second[axis])) ||
        next_first[axis] == next_second[axis]) {
      return false;
    }
  }
  return true;
}

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

bool remove_vertex_above(TetMesh& mesh, int32_t vertex, bool require_improvement,
                         double competing_quality, int32_t* accepted_neighbor,
                         ImprovementProofs* failures) {
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
  // Only a completed empty quality shortlist is reusable. Link, source,
  // transaction and resource refusals below do not certify a negative.
  if (failures != nullptr) {
    failures->synchronize(mesh);
    if (!exact_star_witnesses(mesh, star)) failures = nullptr;
  }
  const auto failure_key =
      std::make_tuple(vertex, require_improvement, std::bit_cast<uint64_t>(lower_bound));
  if (failures != nullptr && failures->vertex_proved.contains(failure_key)) return false;
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
  if (candidates.empty() && !mesh.work_exhausted() && failures != nullptr) {
    failures->vertex_proved.insert(failure_key);
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

int32_t remove_ordinary_sliver_vertices(TetMesh& mesh, std::span<const Candidate> slivers,
                                       TetMeshImproveCounters& counters, bool& progress,
                                       ImprovementProofs& failures) {
  int32_t status = PHX_MC_OK;
    // Give each original ordinary sliver star one source/quality-admitted
    // removal opportunity before reconnections can replace its live link.
    // Construction/floor repairs retain their existing ordered objective.
      NativeSet<int32_t> removal_visited;
      for (const Candidate& entry : slivers) {
        native_execution_charge(0);
        if (entry.construction_uncertain || entry.below_floor || !current(mesh, entry)) continue;
        if (!mesh.spend(1)) {
          status = PHX_MC_CAPACITY_EXCEEDED;
          break;
        }
        ++counters.attempts;
        for (int32_t vertex : entry.v) {
          if (mesh.dimension(vertex) != 3 || mesh.protection(vertex) > 0.0 ||
              !removal_visited.insert(vertex).second) continue;
          if (remove_vertex_above(mesh, vertex, true,
                                  -std::numeric_limits<double>::infinity(), nullptr, &failures)) {
            ++counters.vertex_removals;
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
  return status;
}

int32_t reconnect_sliver_frontier(TetMesh& mesh, const ImproveOptions& options,
                                 TetMeshImproveCounters& counters, bool& progress,
                                 NativeVector<Candidate>& slivers, ConstructionFrontier& frontier,
                                 ImprovementProofs& failures) {
  int32_t status = PHX_MC_OK;
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
      // A neighboring relocation can satisfy a queued cell without changing
      // its vertex tuple. Connectivity identity alone is not an unmet proof.
      const ConstructionAssessment assessment = assess_construction(
          mesh, mesh.tet(entry.slot).v, options.minimum_relative_determinant);
      const bool uncertain = assessment.orientation.certified_sign() != 1 || assessment.below_floor;
      if (!uncertain &&
          quality(mesh, mesh.tet(entry.slot).v) >= options.min_dihedral_degrees) continue;
      if (!uncertain &&
          !frontier.processed.insert(Cell{entry.v[0], entry.v[1], entry.v[2], entry.v[3]}).second) {
        continue;
      }
      ++counters.attempts;
      AffectedVertices affected_vertices;
      bool capacity = false;
      const bool changed = improve_cell(mesh, entry.slot, counters, uncertain,
                                        assessment.below_floor, options.minimum_relative_determinant,
                                        options.min_dihedral_degrees, affected_vertices, capacity,
                                        failures);
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
      // Revisit only actual affected unmet charts in this same cavity worklist;
      // a replaced angle candidate must not disappear before its children
      // meet the requested objective. This is not another mesh-wide pass.
      if (changed && !frontier.append(
              mesh, affected_vertices, options.minimum_relative_determinant,
              options.min_dihedral_degrees, slivers, failures)) {
        status = PHX_MC_CAPACITY_EXCEEDED;
        break;
      }
    }
  return status;
}

int32_t record_improvement_unmet(TetMesh& mesh, const ImproveOptions& options,
                                TetMeshImproveCounters& counters, bool exhausted,
                                NativeVector<Candidate>& slivers, int32_t status) {
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
  counters.remaining_slivers = static_cast<int64_t>(std::count_if(
      slivers.begin(), slivers.end(), [&](const Candidate& entry) {
        return !std::isfinite(entry.quality) || entry.quality < options.min_dihedral_degrees;
      }));
  if (status == PHX_MC_OK && !unmet.empty()) {
    status = PHX_MC_REFINEMENT_LIMIT;
  }
  return status;
}

}  // namespace

bool try_face_removal(TetMesh& mesh, int32_t t, int slot, bool require_improvement) {
  const MemoryScope memory_scope(mesh.memory_owner());
  std::array<int32_t, 2> removed{};
  std::array<Cell, 3> proposed{};
  return face_removal_proposal(mesh, t, slot, require_improvement, removed, proposed) &&
         mesh.replace(removed, proposed) == EditStatus::kOk;
}

namespace {

bool try_edge_removal_proposal(
    TetMesh& mesh, int32_t a, int32_t b, bool require_improvement,
    std::optional<std::span<const double>> regular_weights, double radius_edge_bound,
    double max_weight_fraction, bool construction_objective,
    double minimum_relative_determinant, ImprovementProofs* failures) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (!mesh.live_vertex(a) || !mesh.live_vertex(b) || a == b || mesh.segment_source(a, b) >= 0) {
    return false;
  }
  if (regular_weights &&
      regular_weights->size() != static_cast<std::size_t>(mesh.vertex_count())) {
    return false;
  }
  // Weighted and construction-objective proposals do not participate in
  // the ordinary geometric proof bank.
  if (regular_weights || construction_objective) failures = nullptr;
  EdgeRemovalFailureKey failure_key{};
  if (failures != nullptr) {
    failure_key = failures->key(mesh, a, b, require_improvement, construction_objective,
                                radius_edge_bound, max_weight_fraction,
                                minimum_relative_determinant);
    if (failures->proved.contains(failure_key)) return false;
  }
  NativeVector<int32_t> tets;
  NativeVector<int32_t> ring;
  if (!mesh.edge_ring(a, b, tets, ring)) {
    return false;
  }
  NativeVector<Cell> proposed;
  if (!plan_edge_removal(mesh, a, b, tets, ring, require_improvement, construction_objective,
                         minimum_relative_determinant, failures, failure_key, proposed)) {
    return false;
  }
  if (regular_weights &&
      (regular_proposal(mesh, proposed, *regular_weights, radius_edge_bound, max_weight_fraction,
                        -1) != Regularity::kRegular ||
       !above_floor(mesh, proposed, minimum_relative_determinant))) {
    return false;
  }
  return mesh.replace(tets, proposed) == EditStatus::kOk;
}

}  // namespace

bool try_edge_removal(TetMesh& mesh, int32_t a, int32_t b, bool require_improvement,
                      std::optional<std::span<const double>> regular_weights,
                      double radius_edge_bound, double max_weight_fraction,
                      bool construction_objective, double minimum_relative_determinant) {
  return try_edge_removal_proposal(mesh, a, b, require_improvement, regular_weights,
                                   radius_edge_bound, max_weight_fraction,
                                   construction_objective, minimum_relative_determinant, nullptr);
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
          old_segments.insert(topology::undirected_edge(v[i], v[j]));
        }
      }
    }
  }
  for (const Cell& cell : proposed) {
    new_vertices.insert(cell.begin(), cell.end());
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        new_edges.insert(topology::undirected_edge(cell[static_cast<std::size_t>(i)], cell[static_cast<std::size_t>(j)]));
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
      mesh.protection(vertex) > 0.0 || !geometry::point_in_domain(position) ||
      (mesh.dimension(vertex) < 3 && mesh.policy() == BoundaryPolicy::kFixed)) {
    return false;
  }
  NativeVector<int32_t> star;
  NativeVector<int32_t> neighbors;
  if (!mesh.vertex_star(vertex, star)) {
    return false;
  }
  if (!native_execution_cavity(star.size())) return false;
  if (!relocation_outside_protection(mesh, vertex, position, star)) return false;
  if (mesh.dimension(vertex) < 3 && !on_facet_plane(mesh, vertex, star, position, neighbors)) {
    return false;
  }
  if (mesh.dimension(vertex) == 1 && !relocation_on_curve(mesh, vertex, position)) return false;
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
  const RelocationPositions proposed_point{mesh, vertices, positions};
  if (!prepare_relocation_cavity(mesh, proposed_point, star, neighbors, cavity) ||
      !check_relocation_cells(mesh, proposed_point, cavity, radius_edge_bound,
                              minimum_dihedral_degrees) ||
      !check_relocation_segments(mesh, proposed_point)) return false;
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


bool try_collapse_edge(TetMesh& mesh, int32_t remove, int32_t keep,
                       bool require_improvement) {
  const MemoryScope memory_scope(mesh.memory_owner());
  mesh.cancel_prepared_edit();
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

bool try_collapse_curved_edge(TetMesh& mesh, int32_t remove, int32_t keep,
                              bool require_improvement, const CurvedCollapseClaims* claims,
                              CurvedCollapseInspection* inspection) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (claims != nullptr) {
    bool applied = false;
    if (mesh.reuse_curved_collapse(remove, keep, require_improvement, *claims, applied)) {
      if (applied) mesh.retire_vertex(remove);
      return applied;
    }
  }
  mesh.cancel_prepared_edit();
  NativeVector<int32_t> star, other_star;
  NativeVector<Cell> cells;
  if (!mesh.live_vertex(remove) || mesh.dimension(remove) != 2 ||
      !prepare_collapse(mesh, remove, keep, require_improvement, nullptr, star, other_star,
                        cells)) {
    return false;
  }
  if (mesh.collapse_curved_boundary_edge(remove, keep, star, other_star, claims, inspection) !=
      EditStatus::kOk) {
    return false;
  }
  if (inspection != nullptr) mesh.bind_curved_collapse(require_improvement);
  if (inspection == nullptr) mesh.retire_vertex(remove);
  return true;
}


bool try_remove_vertex(TetMesh& mesh, int32_t vertex, bool require_improvement,
                       int32_t* accepted_neighbor) {
  return remove_vertex_above(mesh, vertex, require_improvement,
                            -std::numeric_limits<double>::infinity(), accepted_neighbor);
}

int32_t improve_mesh(TetMesh& mesh, const ImproveOptions& options, TetMeshImproveCounters& counters) {
  const MemoryScope memory_scope(mesh.memory_owner());
  const int64_t initial_work = mesh.work();
  counters = {};
  struct PartialWork {
    const TetMesh& mesh;
    int64_t initial;
    int64_t& output;
    ~PartialWork() { output = mesh.work() - initial; }
  } work_evidence{mesh, initial_work, counters.work_units};
  int32_t status = PHX_MC_OK;
  bool exhausted = false;
  NativeVector<Candidate> slivers;
  ConstructionFrontier frontier;
  ImprovementProofs proofs;
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
    ++counters.passes;
    bool progress = false;
    status = remove_ordinary_sliver_vertices(mesh, slivers, counters, progress, proofs);
    if (status == PHX_MC_INTERNAL_ERROR) return status;
    if (status != PHX_MC_OK) break;
    if (progress && !collect_slivers(mesh, options.min_dihedral_degrees,
                                    options.minimum_relative_determinant, slivers)) {
      status = PHX_MC_CAPACITY_EXCEEDED;
      break;
    }
    status = reconnect_sliver_frontier(
        mesh, options, counters, progress, slivers, frontier, proofs);
    if (status == PHX_MC_INTERNAL_ERROR) return status;
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
  return record_improvement_unmet(mesh, options, counters, exhausted, slivers, status);
}

int32_t exude_mesh(TetMesh& mesh, const ExudeOptions& options,
                   NativeVector<double>& weights, TetMeshExudeCounters& counters) {
  const MemoryScope memory_scope(mesh.memory_owner());
  if (!std::isfinite(options.max_weight_fraction) || options.max_weight_fraction <= 0.0 ||
      options.max_weight_fraction > 0.25 || !std::isfinite(options.radius_edge_bound) ||
      options.radius_edge_bound < 1.0 || options.min_dihedral_degrees < 0.0 ||
      !std::isfinite(options.min_dihedral_degrees) || options.max_passes < 0 ||
      !(options.minimum_relative_determinant >= 0.0) ||
      !(options.minimum_relative_determinant < 1.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  counters = {};
  if (weights.get_allocator().owner() != mesh.memory_owner()) {
    weights = NativeVector<double>(NativeAllocator<double>(mesh.memory_owner()));
  }
  weights.assign(static_cast<std::size_t>(mesh.vertex_count()), 0.0);
  const int64_t initial_work = mesh.work();
  NativeVector<Candidate> slivers;
  ExudeScratch scratch;
  scratch.refusals.assign(static_cast<std::size_t>(mesh.vertex_count()), VertexRefusal{});
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
    ++counters.passes;
    bool progress = false;
    for (const Candidate& entry : slivers) {
      if (!current(mesh, entry)) {
        continue;
      }
      const Cell vertices = {mesh.tet(entry.slot).v[0], mesh.tet(entry.slot).v[1],
                             mesh.tet(entry.slot).v[2], mesh.tet(entry.slot).v[3]};
      for (int32_t vertex : vertices) {
        VertexRefusal& known = scratch.refusals[static_cast<std::size_t>(vertex)];
        if (known.refused > 0 && known.dirty <= known.refused && !mesh.work_exhausted()) {
          // Every datum the recorded visit read is unchanged: the same
          // trials refuse again, so they are counted, not recomputed.
          counters.trials += known.trials;
          continue;
        }
        const int64_t trials = counters.trials;
        if (exude_vertex(mesh, vertex, options, weights, counters, scratch)) {
          progress = true;
          mark_exuded(mesh, vertex, scratch);
          break;
        }
        if (scratch.reusable && !mesh.work_exhausted() && !mesh.broken()) {
          known.refused = scratch.epoch;
          known.trials = counters.trials - trials;
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
  counters.weighted_vertices = static_cast<int64_t>(std::count_if(weights.begin(), weights.end(),
                                                [](double w) { return w > 0.0; }));
  counters.remaining_slivers = static_cast<int64_t>(std::count_if(
      slivers.begin(), slivers.end(), [&](const Candidate& entry) {
        return !std::isfinite(entry.quality) || entry.quality < options.min_dihedral_degrees;
      }));
  counters.work_units = mesh.work() - initial_work;
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
    phx::mc::TetMeshImproveCounters state;
    phx::mc::CounterPublisher publisher{state, counters};
    const int32_t status = phx::mc::improve_mesh(mesh, options, state);
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
      } else if (!mesh.work_exhausted() &&
                 mesh.construct_interior_edge_point(a, b, fraction, position)) {
        // No exact or source-row carrier: an unconstrained edge offers the
        // rounded interior carrier, which carries no source witness.
        witness = phx::mc::SourceWitness{};
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
          !phx::mc::geometry::collinear(mesh.point(first), mesh.point(second), mesh.point(vertex))) {
        continue;
      }
      const double* original_first = nullptr;
      const double* original_second = nullptr;
      for (const auto& segment : mesh.original_segments()) {
        if (!mesh.spend(1)) return PHX_MC_CAPACITY_EXCEEDED;
        if (segment[2] != source) continue;
        const double* a = mesh.original_point(segment[0]);
        const double* b = mesh.original_point(segment[1]);
        if (phx::mc::geometry::collinear(a, b, mesh.point(vertex)) &&
            phx::mc::geometry::collinear(a, b, mesh.point(first)) &&
            phx::mc::geometry::collinear(a, b, mesh.point(second))) {
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
            (!phx::mc::geometry::collinear(original_first, original_second, positions + 3 * row) ||
             !phx::mc::geometry::collinear(mesh.point(first), mesh.point(second), positions + 3 * row))) {
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
    phx::mc::EdgeInsertionCounts state;
    phx::mc::CounterPublisher publisher{state, counts};
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
    state.removed = finite[0];
    state.proposed = finite[1];
    state.kind = static_cast<int32_t>(kind);
    state.generation = mesh.accepted_generation();
    const auto retained = mesh.retain_edge_insertion(
        first, second, split_position, final_position, source_fraction,
        static_cast<std::size_t>(maximum_cavity_cells), kind);
    if (retained != phx::mc::Insertion::kOk) {
      mesh.abandon();
      state = {};
      return insertion_status(mesh, retained);
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_inspect_projected_edge_insertion(
    phx_mc_tet_mesh* handle, int32_t first, int32_t second, const double* split_position,
    const double* final_position, double source_fraction, int32_t source,
    int64_t maximum_cavity_cells, int64_t work_limit, int32_t* removed_tetrahedra,
    int32_t* proposed_tetrahedra, int32_t* projected_faces, int64_t* counts) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr || split_position == nullptr ||
        final_position == nullptr || counts == nullptr || source < 0 ||
        maximum_cavity_cells < 0 || maximum_cavity_cells > INT64_MAX / 12 ||
        !phx::mc::addressable(maximum_cavity_cells, 12, sizeof(int32_t)) ||
        work_limit < 0 || !std::isfinite(source_fraction) ||
        source_fraction <= 0.0 || source_fraction >= 1.0 ||
        (maximum_cavity_cells > 0 && (removed_tetrahedra == nullptr ||
         proposed_tetrahedra == nullptr || projected_faces == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::fill_n(counts, 5, int64_t{0});
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    if (mesh.broken()) return PHX_MC_INTERNAL_ERROR;
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::InsertKind kind = phx::mc::InsertKind::kInterior;
    const auto prepared = mesh.prepare_projected_edge_insertion(
        first, second, split_position, final_position, source_fraction,
        static_cast<std::size_t>(maximum_cavity_cells), source, kind);
    if (prepared != phx::mc::Insertion::kOk) {
      mesh.abandon();
      return insertion_status(mesh, prepared);
    }
    const auto finite = insertion_counts(mesh);
    const auto faces = mesh.projected_insertion_faces();
    if (finite[0] <= 0 || finite[1] <= 0 ||
        finite[0] + finite[1] > maximum_cavity_cells ||
        faces.size() > static_cast<std::size_t>(4 * maximum_cavity_cells)) {
      mesh.abandon();
      return PHX_MC_INTERNAL_ERROR;
    }
    if (!mesh.spend(4 * (finite[0] + finite[1]) + 3 * static_cast<int64_t>(faces.size()))) {
      mesh.abandon();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    insertion_rows(mesh, removed_tetrahedra, proposed_tetrahedra);
    for (std::size_t row = 0; row < faces.size(); ++row) {
      std::copy_n(faces[row].data(), 3, projected_faces + 3 * row);
    }
    counts[0] = finite[0];
    counts[1] = finite[1];
    counts[2] = static_cast<int32_t>(kind);
    counts[3] = mesh.accepted_generation();
    counts[4] = static_cast<int64_t>(faces.size());
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_commit_edge_insertion(
    const phx_mc_tet_mesh_commit_edge_insertion_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  auto handle = request->mesh;
  auto first = request->first;
  auto second = request->second;
  auto split_position = request->split_position;
  auto final_position = request->final_position;
  auto source_fraction = request->source_fraction;
  auto maximum_cavity_cells = request->maximum_cavity_cells;
  auto insertion_kind = request->insertion_kind;
  auto accepted_generation = request->accepted_generation;
  auto removed_count = request->removed_count;
  auto removed_tetrahedra = request->removed_tetrahedra;
  auto proposed_count = request->proposed_count;
  auto proposed_tetrahedra = request->proposed_tetrahedra;
  auto target_size = request->target_size;
  auto work_limit = request->work_limit;
  auto inserted_vertex = request->inserted_vertex;
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr || split_position == nullptr ||
        final_position == nullptr || inserted_vertex == nullptr ||
        maximum_cavity_cells < 0 || maximum_cavity_cells > INT64_MAX / 4 ||
        removed_count <= 0 || proposed_count <= 0 || removed_count > maximum_cavity_cells ||
        proposed_count > maximum_cavity_cells - removed_count ||
        !std::isfinite(source_fraction) || source_fraction < 0.0 || source_fraction >= 1.0 ||
        removed_tetrahedra == nullptr || proposed_tetrahedra == nullptr ||
        insertion_kind < 0 || insertion_kind > 2 || accepted_generation < 0 ||
        request->claim_count < 0 ||
        (request->claim_count > 0 && (request->projected_source < 0 ||
         request->claim_digest == nullptr || request->claim_faces == nullptr ||
         request->claim_coordinates == nullptr || request->claim_deviations == nullptr)) ||
        !std::isfinite(target_size) || target_size < 0.0 || work_limit < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    // PHX_MC_OK reports execution without an error, not an applied edit.
    // A stale or mismatched plan retains -1; only a committed vertex changes it.
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
    const bool projected = request->claim_count > 0;
    const bool reused = mesh.reuse_edge_insertion(
        first, second, split_position, final_position, source_fraction,
        static_cast<std::size_t>(maximum_cavity_cells), kind,
        projected ? request->projected_source : -1);
    const phx::mc::Insertion prepared = reused ? phx::mc::Insertion::kOk
        : projected ? mesh.prepare_projected_edge_insertion(
              first, second, split_position, final_position, source_fraction,
              static_cast<std::size_t>(maximum_cavity_cells), request->projected_source, kind)
        : mesh.prepare_edge_insertion(
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
    if (projected) {
      const phx::mc::CurvedCollapseClaims claims{
          request->projected_source, accepted_generation, request->claim_digest,
          request->claim_count, request->claim_faces, request->claim_coordinates,
          request->claim_deviations};
      const auto admitted = mesh.authenticate_projected_insertion(claims);
      if (admitted != phx::mc::Insertion::kOk) {
        mesh.abandon();
        return insertion_status(mesh, admitted);
      }
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

int32_t phx_mc_tet_mesh_collapse_curved_edge(
    const phx_mc_tet_mesh_curved_collapse_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const auto& in = *request;
    const int32_t status = ready(in.mesh, in.applied);
    if (status != PHX_MC_OK || in.work_limit < 0 || in.digest == nullptr ||
        in.claim_count <= 0 || in.claim_faces == nullptr || in.claim_coordinates == nullptr ||
        in.claim_deviations == nullptr || in.accepted_generation < 0 || in.source < 0 ||
        (in.require_improvement != 0 && in.require_improvement != 1) ||
        !phx::mc::addressable(in.claim_count, 9, sizeof(double))) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *in.mesh->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    for (int64_t i = 0; i < 3 * in.claim_count; ++i) {
      if (in.claim_faces[i] < 0 || in.claim_faces[i] >= mesh.vertex_count()) {
        return PHX_MC_INVALID_ARGUMENT;
      }
    }
    phx::mc::TetWorkBudgetWindow budget(mesh, in.work_limit);
    phx::mc::CurvedCollapseClaims claims;
    claims.source = in.source;
    claims.generation = in.accepted_generation;
    claims.digest = in.digest;
    claims.count = in.claim_count;
    claims.faces = in.claim_faces;
    claims.coordinates = in.claim_coordinates;
    claims.deviations = in.claim_deviations;
    const bool done = phx::mc::try_collapse_curved_edge(
        mesh, in.remove, in.keep, in.require_improvement != 0, &claims, nullptr);
    const bool capacity = mesh.work_exhausted();
    *in.applied = done ? 1 : 0;
    return mesh.broken() ? PHX_MC_INTERNAL_ERROR
                        : capacity ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_inspect_curved_collapse(phx_mc_tet_mesh* handle, int32_t remove,
                                                int32_t keep, int32_t require_improvement,
                                                int64_t work_limit, int64_t capacity,
                                                int64_t* face_count, int32_t* source,
                                                int64_t* accepted_generation, int32_t* faces) {
  return phx::mc::guarded([&]() -> int32_t {
    if (face_count == nullptr || source == nullptr || accepted_generation == nullptr ||
        faces == nullptr || capacity < 1 || work_limit < 0 ||
        (require_improvement != 0 && require_improvement != 1) ||
        !phx::mc::addressable(capacity, 3, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *face_count = 0;
    *source = -1;
    *accepted_generation = -1;
    if (handle == nullptr || handle->mesh == nullptr) return PHX_MC_INVALID_ARGUMENT;
    phx::mc::TetMesh& mesh = *handle->mesh;
    if (mesh.broken()) return PHX_MC_INTERNAL_ERROR;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    phx::mc::CurvedCollapseInspection inspection;
    const bool admitted = phx::mc::try_collapse_curved_edge(
        mesh, remove, keep, require_improvement != 0, nullptr, &inspection);
    if (mesh.broken()) return PHX_MC_INTERNAL_ERROR;
    if (mesh.work_exhausted()) return PHX_MC_CAPACITY_EXCEEDED;
    if (!admitted) return PHX_MC_OK;
    const auto count = static_cast<int64_t>(inspection.faces.size());
    if (count > capacity) {
      mesh.cancel_prepared_edit();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    for (std::size_t i = 0; i < inspection.faces.size(); ++i) {
      std::copy_n(inspection.faces[i].data(), 3, faces + 3 * i);
    }
    *face_count = count;
    *source = inspection.source;
    *accepted_generation = inspection.generation;
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_curved_collapse_cavity(
    phx_mc_tet_mesh* handle, int32_t remove, int32_t keep, int64_t generation,
    int64_t removed_capacity, int64_t proposed_capacity, int64_t work_limit,
    int32_t* removed, int32_t* proposed, int64_t* counts) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr || counts == nullptr ||
        removed_capacity < 0 || proposed_capacity < 0 || work_limit < 0 ||
        !phx::mc::addressable(removed_capacity, 4, sizeof(int32_t)) ||
        !phx::mc::addressable(proposed_capacity, 4, sizeof(int32_t)) ||
        (removed_capacity > 0 && removed == nullptr) ||
        (proposed_capacity > 0 && proposed == nullptr)) return PHX_MC_INVALID_ARGUMENT;
    counts[0] = counts[1] = 0;
    auto& mesh = *handle->mesh;
    if (mesh.broken()) return PHX_MC_INTERNAL_ERROR;
    if (!mesh.retained_curved_collapse(remove, keep, generation)) return PHX_MC_INVALID_INPUT;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    int64_t old_count = 0, new_count = 0;
    for (int32_t cell : mesh.retained_removed()) old_count += mesh.finite(cell);
    for (const auto& cell : mesh.retained_proposed()) new_count += !phx::mc::is_ghost(cell);
    if (!mesh.spend(static_cast<int64_t>(
            mesh.retained_removed().size() + mesh.retained_proposed().size()))) {
      mesh.cancel_prepared_edit();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (removed_capacity == 0 && proposed_capacity == 0) {
      counts[0] = old_count;
      counts[1] = new_count;
      return PHX_MC_OK;
    }
    if (old_count > removed_capacity || new_count > proposed_capacity ||
        !mesh.spend(4 * (old_count + new_count))) {
      mesh.cancel_prepared_edit();
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    int64_t row = 0;
    for (int32_t cell : mesh.retained_removed()) {
      if (mesh.finite(cell)) std::copy_n(mesh.tet(cell).v, 4, removed + 4 * row++);
    }
    row = 0;
    for (const auto& cell : mesh.retained_proposed()) {
      if (!phx::mc::is_ghost(cell)) std::copy_n(cell.v, 4, proposed + 4 * row++);
    }
    counts[0] = old_count;
    counts[1] = new_count;
    return PHX_MC_OK;
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
    phx::mc::TetMeshExudeCounters state;
    phx::mc::CounterPublisher publisher{state, counters};
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
      status = phx::mc::exude_mesh(mesh, options, accepted_weights, state);
    } catch (const std::bad_alloc&) {
      std::copy(accepted_weights.begin(), accepted_weights.end(), weights);
      state.weighted_vertices = static_cast<int64_t>(std::count_if(
          accepted_weights.begin(), accepted_weights.end(), [](double weight) { return weight > 0.0; }));
      state.work_units = mesh.work() - initial_work;
      throw;  // Preserve the owning allocator's refusal status/message.
    }
    std::copy(accepted_weights.begin(), accepted_weights.end(), weights);
    return status;
  });
}

}  // extern "C"
