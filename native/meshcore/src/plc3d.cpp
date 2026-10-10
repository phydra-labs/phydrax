//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// PLC validation, boundary recovery and region classification (plc3d.hpp).
//
// Phases, each refusing with evidence before the next starts:
//   1. validation: exact point domain and distinctness, exactly planar simple
//      polygons (triangulated by the exact 2D CDT), coplanar facet groups,
//      the PLC edge table, exact pairwise contacts and oriented chain closure
//      of every region;
//   2. protection radii of acute input vertices (feature_protection.cpp);
//   3. recovery rounds on the exact Delaunay tetrahedralization of all points
//      and eight enclosing corners: (a) segment recovery -- every subsegment
//      of every PLC edge becomes a Delaunay edge by exact on-segment splits
//      (fixed boundary: refused); (b) facet recovery -- the missing parts of
//      every coplanar group are recovered component by component: the cavity
//      is the set of tetrahedra whose interior meets the component's relative
//      interior (an edge crossing its interior or an internal edge of it, or a
//      face lying in it), each side is refilled by constrained-Delaunay gift
//      wrapping over exact contact tests, and both fills are committed
//      through one validated CavityEdit; (c) subsegments lost by a fill are
//      split (conforming) or refused (fixed).  A conforming facet that no gift
//      wrap can fill receives an exactly coplanar interior Steiner point and the
//      round restarts from the enlarged point set;
//   4. classification by flooding over unconstrained facets from both sides of
//      every constrained facet (declared incidence), the enclosing corners
//      (void) and seeds; conflicts are refused;
//   5. publication of the domain tetrahedra, oriented subfacets with their
//      source facets, subsegments with their PLC edges and ancestry.
//
// The comments of recover_component state why the cavity sides are closed
// polyhedra; the CavityEdit validation (oriented boundary filled exactly once,
// interior facets paired, positive orientations) is the certificate that the
// committed fill covers the cavity exactly once.
#include "plc3d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <new>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "cavity.hpp"
#include "intersections.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"
#include "triangulation3d.hpp"

namespace phx::mc {
namespace {

constexpr int32_t kVoidRegion = -3;
constexpr int32_t kCorners = 8;

using Triple = std::array<int32_t, 3>;

class PhaseTimer {
 public:
  PhaseTimer(bool enabled, PlcRecovery& result, PlcPhase phase)
      : enabled_(enabled), result_(result), phase_(phase) {
    if (enabled_) {
      ++result_.phase_invocations[phase_];
      start_ = std::chrono::steady_clock::now();
    }
  }
  ~PhaseTimer() { stop(); }
  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;
  void stop() {
    if (enabled_) {
      result_.phase_nanoseconds[phase_] +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start_).count();
      enabled_ = false;
    }
  }
 private:
  bool enabled_;
  PlcRecovery& result_;
  PlcPhase phase_;
  std::chrono::steady_clock::time_point start_{};
};


uint64_t edge_key(int32_t a, int32_t b) {
  if (a > b) {
    std::swap(a, b);
  }
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

Triple sorted_triple(const int32_t* v) {
  Triple key{v[0], v[1], v[2]};
  std::sort(key.begin(), key.end());
  return key;
}

// Orientation-aware key: the rotation starting at the smallest vertex.
Triple oriented_key(const int32_t* v) {
  const FacetKey key = FacetKey::of(v);
  return {key.a, key.b, key.c};
}

Triple reversed_key(const int32_t* v) {
  const FacetKey key = FacetKey::of(v).reversed();
  return {key.a, key.b, key.c};
}

bool legal_contact(int8_t kind) {
  return kind == PHX_MC_DISJOINT || kind == PHX_MC_SHARED_VERTEX || kind == PHX_MC_SHARED_EDGE;
}

struct Box {
  double lower[3];
  double upper[3];
};

bool overlap(const Box& a, const Box& b) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a.upper[axis] < b.lower[axis] || b.upper[axis] < a.lower[axis]) {
      return false;
    }
  }
  return true;
}

// Axis whose coordinate-plane projection of a nondegenerate triangle keeps an
// exactly nonzero orientation; the largest approximate normal component first.
int projection_axis(const double* a, const double* b, const double* c) {
  const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double normal[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                            u[0] * w[1] - u[1] * w[0]};
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3,
            [&](int x, int y) { return std::fabs(normal[x]) > std::fabs(normal[y]); });
  for (int axis : order) {
    const int i = (axis + 1) % 3;
    const int j = (axis + 2) % 3;
    const double pa[2] = {a[i], a[j]};
    const double pb[2] = {b[i], b[j]};
    const double pc[2] = {c[i], c[j]};
    if (orient2d(pa, pb, pc) != 0) {
      return axis;
    }
  }
  return -1;
}

// Exact closed-segment intersection test of two coplanar-or-skew segments that
// share no endpoint id; true when they meet.
bool segments_meet(const double* a, const double* b, const double* c, const double* d) {
  if (orient3d(a, b, c, d) != 0) {
    return false;
  }
  const bool c_on_line = collinear3d(a, b, c);
  const bool d_on_line = collinear3d(a, b, d);
  if (c_on_line && d_on_line) {
    int axis = 0;
    for (int k = 0; k < 3; ++k) {
      if (a[k] != b[k]) {
        axis = k;
        break;
      }
    }
    const double lo1 = std::min(a[axis], b[axis]);
    const double hi1 = std::max(a[axis], b[axis]);
    const double lo2 = std::min(c[axis], d[axis]);
    const double hi2 = std::max(c[axis], d[axis]);
    return !(hi1 < lo2 || hi2 < lo1);
  }
  const int axis = projection_axis(a, b, c_on_line ? d : c);
  const int i = (axis + 1) % 3;
  const int j = (axis + 2) % 3;
  const double pa[2] = {a[i], a[j]};
  const double pb[2] = {b[i], b[j]};
  const double pc[2] = {c[i], c[j]};
  const double pd[2] = {d[i], d[j]};
  const int o1 = orient2d(pa, pb, pc);
  const int o2 = orient2d(pa, pb, pd);
  const int o3 = orient2d(pc, pd, pa);
  const int o4 = orient2d(pc, pd, pb);
  const auto within = [&](const double* p, const double* q, const double* r) {
    return std::min(p[0], q[0]) <= r[0] && r[0] <= std::max(p[0], q[0]) &&
           std::min(p[1], q[1]) <= r[1] && r[1] <= std::max(p[1], q[1]);
  };
  if (o1 * o2 < 0 && o3 * o4 < 0) {
    return true;
  }
  return (o1 == 0 && within(pa, pb, pc)) || (o2 == 0 && within(pa, pb, pd)) ||
         (o3 == 0 && within(pc, pd, pa)) || (o4 == 0 && within(pc, pd, pb));
}

// Exact 2D CDT of one planar polygon region: boundary loop plus free interior
// points, projected along `axis`.  Returns triangles in input indices with the
// orientation of the loop, or false when the loop is not a simple polygon.
bool triangulate_planar(std::span<const double* const> corners, int64_t loop_count, int axis,
                        NativeVector<Triple>& triangles, int32_t& status, int64_t max_work,
                        int64_t& used_work, int32_t& refusal) {
  const int i = (axis + 1) % 3;
  const int j = (axis + 2) % 3;
  const int64_t count = static_cast<int64_t>(corners.size());
  NativeVector<double> planar(static_cast<std::size_t>(2 * count));
  for (int64_t k = 0; k < count; ++k) {
    planar[static_cast<std::size_t>(2 * k)] = corners[static_cast<std::size_t>(k)][i];
    planar[static_cast<std::size_t>(2 * k + 1)] = corners[static_cast<std::size_t>(k)][j];
  }
  NativeVector<int32_t> loop(static_cast<std::size_t>(2 * loop_count));
  for (int64_t k = 0; k < loop_count; ++k) {
    loop[static_cast<std::size_t>(2 * k)] = static_cast<int32_t>(k);
    loop[static_cast<std::size_t>(2 * k + 1)] = static_cast<int32_t>((k + 1) % loop_count);
  }
  phx_mc_mesh* raw = nullptr;
  uint64_t work_evidence[9] = {};
  uint64_t memory_evidence[6] = {};
  const MemoryOwner owner = scratch_memory_owner();
  const uint64_t remaining_bytes = owner
      ? owner->limit_bytes() - owner->live_bytes()
      : static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  const int64_t scratch_limit = static_cast<int64_t>(std::min<uint64_t>(
      remaining_bytes, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
  status = phx_mc_constrained_delaunay_2d(count, planar.data(), loop_count, loop.data(), 0,
                                          nullptr, 0, 0.0, std::numeric_limits<double>::infinity(),
                                          0, 4 * count + 8, 4 * count + 8, max_work,
                                          scratch_limit, work_evidence, memory_evidence, &raw);
  const std::unique_ptr<phx_mc_mesh, void (*)(phx_mc_mesh*)> mesh(raw, phx_mc_mesh_free);
  used_work = static_cast<int64_t>(work_evidence[0]);
  refusal = memory_evidence[5] ? kPlcScratchByteBudget
            : work_evidence[7] ? kPlcWorkBudget : kPlcTetrahedronBudget;
  if (status != PHX_MC_OK) {
    return false;
  }
  // Distinct inputs keep distinct vertices; invert the vertex map.
  NativeVector<int32_t> original(static_cast<std::size_t>(mesh->point_count()), -1);
  for (int64_t k = 0; k < count; ++k) {
    const int32_t vertex = mesh->vertex_map[static_cast<std::size_t>(k)];
    if (vertex < 0 || original[static_cast<std::size_t>(vertex)] >= 0) {
      status = PHX_MC_INVALID_INPUT;
      return false;
    }
    original[static_cast<std::size_t>(vertex)] = static_cast<int32_t>(k);
  }
  // Loop orientation in the projection: the sign at its lexicographically
  // smallest vertex, which is strictly convex.
  int64_t lowest = 0;
  for (int64_t k = 1; k < loop_count; ++k) {
    const double* p = &planar[static_cast<std::size_t>(2 * k)];
    const double* q = &planar[static_cast<std::size_t>(2 * lowest)];
    if (p[0] < q[0] || (p[0] == q[0] && p[1] < q[1])) {
      lowest = k;
    }
  }
  const double* before = &planar[static_cast<std::size_t>(2 * ((lowest + loop_count - 1) % loop_count))];
  const double* after = &planar[static_cast<std::size_t>(2 * ((lowest + 1) % loop_count))];
  const int sense = orient2d(before, &planar[static_cast<std::size_t>(2 * lowest)], after);
  if (sense == 0) {
    status = PHX_MC_INVALID_INPUT;
    return false;
  }
  triangles.clear();
  for (int64_t t = 0; t < mesh->cell_count(); ++t) {
    Triple triangle{};
    for (int r = 0; r < 3; ++r) {
      triangle[static_cast<std::size_t>(r)] =
          original[static_cast<std::size_t>(mesh->cells[static_cast<std::size_t>(3 * t + r)])];
      if (triangle[static_cast<std::size_t>(r)] < 0) {
        status = PHX_MC_INVALID_INPUT;
        return false;
      }
    }
    if (sense < 0) {
      std::swap(triangle[1], triangle[2]);
    }
    triangles.push_back(triangle);
  }
  return true;
}

class Recoverer {
 public:
  Recoverer(const PlcInput& input, PlcRecovery& output) : in_(input), out_(output) {}

  int32_t run(bool source_only = false) try {
    out_.measurement_enabled = in_.measure_phases;
    PhaseTimer validation(in_.measure_phases, out_, kPlcValidationPhase);
    int32_t status = validate_points(
        in_.points, in_.point_count, 3, nullptr,
        [](void* context) { static_cast<Recoverer*>(context)->charge_visit(); }, this);
    if (status != PHX_MC_OK) {
      return status;
    }
    n_ = static_cast<int32_t>(in_.point_count);
    for (auto step : {&Recoverer::check_distinct, &Recoverer::triangulate_polygons,
                      &Recoverer::check_facets, &Recoverer::check_tolerances,
                      &Recoverer::build_groups,
                      &Recoverer::build_edges, &Recoverer::check_contacts,
                      &Recoverer::check_closure}) {
      status = (this->*step)();
      if (status != PHX_MC_OK) {
        finish_counters();
        return status;
      }
    }
    validation.stop();
    if (source_only) {
      // Source preparation uses the same exact validation, polygon CDT,
      // coplanar grouping and canonical edge order, but never fills a solid.
      out_.points.assign(in_.points, in_.points + 3 * static_cast<int64_t>(n_));
      out_.plc_edges = edges_;
      out_.input_triangles = triangles_;
      out_.input_polygons = tri_polygon_;
      finish_counters();
      return PHX_MC_OK;
    }
    status = recover();
    if (status == PHX_MC_OK) {
      PhaseTimer classification(in_.measure_phases, out_, kPlcClassificationPhase);
      status = classify();
    }
    if (status != PHX_MC_OK) {
      finish_counters();
      return status;
    }
    {
      PhaseTimer publication(in_.measure_phases, out_, kPlcPublicationPhase);
      publish();
    }
    finish_counters();
    return PHX_MC_OK;
  } catch (const ExecutionRefusal& refusal) {
    finish_counters();
    return fail(refusal.status, kPlcWorkBudget, kPlcNoEntity, -1);
  } catch (const std::bad_alloc&) {
    finish_counters();
    return fail(native_execution_status(PHX_MC_CAPACITY_EXCEEDED),
                kPlcScratchByteBudget, kPlcNoEntity, -1);
  }

 private:
  // ------------------------------------------------------------ utilities
  int32_t fail(int32_t status, int32_t reason, int32_t first_kind, int64_t first,
               int32_t second_kind = kPlcNoEntity, int64_t second = -1) {
    out_.failure = PlcFailure{reason, first_kind, first, second_kind, second};
    return status;
  }

  int32_t work_failure() {
    return fail(PHX_MC_CAPACITY_EXCEEDED, kPlcWorkBudget, kPlcNoEntity, -1);
  }

  int64_t work() const {
    return work_ + (triangulation_ ? triangulation_->work() : 0);
  }

  bool spend(int64_t units) {
    if (work_exhausted_ || units < 0 || work() > in_.work_limit - units ||
        !native_execution_spend(units)) {
      work_exhausted_ = true;
      return false;
    }
    work_ += units;
    return true;
  }

  void charge_visit() {
    native_execution_charge(0);
  }

  const double* point(int32_t v) const { return points_.data() + 3 * static_cast<int64_t>(v); }
  int32_t vertex_count() const { return static_cast<int32_t>(points_.size() / 3); }

  void finish_counters() {
    out_.counters[kPlcWork] = work();
    out_.counters[kPlcInputTriangles] = static_cast<int64_t>(tri_facet_.size());
    out_.counters[kPlcFacetGroups] = static_cast<int64_t>(group_facet_.size());
    out_.counters[kPlcEdges] = static_cast<int64_t>(edges_.size() / 2);
    if (triangulation_) {
      peak_bytes_ = std::max(peak_bytes_, triangulation_->retained_bytes());
    }
    out_.counters[kPlcPeakBytes] = static_cast<int64_t>(
        out_.memory_owner ? out_.memory_owner->peak_bytes() : peak_bytes_);
  }

  // ------------------------------------------------------------ validation
  int32_t check_distinct() {
    native_execution_charge(0);
    NativeVector<int32_t> order(static_cast<std::size_t>(n_));
    std::iota(order.begin(), order.end(), 0);
    const double* p = in_.points;
    std::sort(order.begin(), order.end(), [&](int32_t x, int32_t y) {
      charge_visit();
      return std::lexicographical_compare(p + 3 * x, p + 3 * x + 3, p + 3 * y, p + 3 * y + 3);
    });
    for (std::size_t k = 1; k < order.size(); ++k) {
      charge_visit();
      if (std::equal(p + 3 * order[k - 1], p + 3 * order[k - 1] + 3, p + 3 * order[k])) {
        return fail(PHX_MC_INVALID_INPUT, kPlcDuplicatePoint, kPlcPoint,
                    std::min(order[k - 1], order[k]), kPlcPoint, std::max(order[k - 1], order[k]));
      }
    }
    return PHX_MC_OK;
  }

  int32_t triangulate_polygons() {
    const double* p = in_.points;
    NativeVector<Triple> local;
    for (int64_t polygon = 0; polygon < in_.polygon_count; ++polygon) {
      charge_visit();
      const int32_t* loop = in_.polygon_vertices + in_.polygon_offsets[polygon];
      const int64_t size = in_.polygon_offsets[polygon + 1] - in_.polygon_offsets[polygon];
      const auto invalid = [&] {
        return fail(PHX_MC_INVALID_INPUT, kPlcInvalidPolygon, kPlcPolygon, polygon);
      };
      if (!spend(size)) return work_failure();
      NativeVector<int32_t> distinct(loop, loop + size);
      std::sort(distinct.begin(), distinct.end(), [&](int32_t a, int32_t b) {
        charge_visit();
        return a < b;
      });
      if (size < 3 || std::adjacent_find(distinct.begin(), distinct.end()) != distinct.end()) {
        return invalid();
      }
      int64_t third = 2;
      while (third < size && collinear3d(p + 3 * loop[0], p + 3 * loop[1], p + 3 * loop[third])) {
        ++third;
        charge_visit();
      }
      if (third == size) {
        return invalid();
      }
      const double* a = p + 3 * loop[0];
      const double* b = p + 3 * loop[1];
      const double* c = p + 3 * loop[third];
      for (int64_t k = 0; k < size; ++k) {
        charge_visit();
        if (orient3d(a, b, c, p + 3 * loop[k]) != 0) {
          return invalid();
        }
      }
      if (size == 3) {
        add_triangle({loop[0], loop[1], loop[2]}, polygon);
        continue;
      }
      NativeVector<const double*> corners;
      for (int64_t k = 0; k < size; ++k) {
        corners.push_back(p + 3 * loop[k]);
      }
      int32_t status = PHX_MC_OK;
      int64_t used_work = 0;
      int32_t refusal = kPlcNoFailure;
      const bool valid = triangulate_planar(
          corners, size, projection_axis(a, b, c), local, status,
          std::max<int64_t>(in_.work_limit - work(), 0), used_work, refusal);
      if (!spend(used_work)) {
        return work_failure();
      }
      if (!valid || static_cast<int64_t>(local.size()) != size - 2) {
        if (status == PHX_MC_CAPACITY_EXCEEDED) {
          return fail(status, refusal, kPlcPolygon, polygon);
        }
        return invalid();
      }
      for (const Triple& triangle : local) {
        add_triangle({loop[triangle[0]], loop[triangle[1]], loop[triangle[2]]}, polygon);
      }
    }
    return PHX_MC_OK;
  }

  void add_triangle(const Triple& triangle, int64_t polygon) {
    triangles_.insert(triangles_.end(), triangle.begin(), triangle.end());
    tri_polygon_.push_back(static_cast<int32_t>(polygon));
    tri_facet_.push_back(in_.polygon_facets[polygon]);
  }

  int32_t check_facets() {
    if (in_.facet_count == 0) {
      return fail(PHX_MC_INVALID_INPUT, kPlcInvalidFacet, kPlcNoEntity, -1);
    }
    NativeVector<int64_t> uses(static_cast<std::size_t>(in_.facet_count), 0);
    for (int64_t polygon = 0; polygon < in_.polygon_count; ++polygon) {
      ++uses[static_cast<std::size_t>(in_.polygon_facets[polygon])];
    }
    for (int64_t facet = 0; facet < in_.facet_count; ++facet) {
      if (uses[static_cast<std::size_t>(facet)] == 0 ||
          (in_.facet_regions[2 * facet] < 0 && in_.facet_regions[2 * facet + 1] < 0)) {
        return fail(PHX_MC_INVALID_INPUT, kPlcInvalidFacet, kPlcFacet, facet);
      }
    }
    return PHX_MC_OK;
  }

  int32_t check_tolerances() {
    const auto admissible = [](double value) { return std::isfinite(value) && value >= 0.0; };
    for (int64_t facet = 0; in_.facet_tolerances != nullptr && facet < in_.facet_count; ++facet) {
      if (!admissible(in_.facet_tolerances[facet])) {
        return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcFacet, facet);
      }
    }
    for (int64_t s = 0; in_.segment_tolerances != nullptr && s < in_.segment_count; ++s) {
      if (!admissible(in_.segment_tolerances[s])) {
        return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcEdge, s);
      }
    }
    return PHX_MC_OK;
  }

  double facet_tolerance(int32_t facet) const {
    return in_.facet_tolerances == nullptr ? 0.0
                                           : in_.facet_tolerances[static_cast<std::size_t>(facet)];
  }

  // A point of PLC edge e lies on its explicit segment and on every incident
  // facet; the smallest declared bound governs it.
  double edge_tolerance(int32_t e) const {
    double tolerance = std::numeric_limits<double>::infinity();
    if (e < in_.segment_count) {
      tolerance = in_.segment_tolerances == nullptr
                      ? 0.0
                      : in_.segment_tolerances[static_cast<std::size_t>(e)];
    }
    const auto range = std::equal_range(
        edge_uses_.begin(), edge_uses_.end(),
        std::pair<uint64_t, int32_t>{edge_key(edges_[2 * static_cast<std::size_t>(e)],
                                              edges_[2 * static_cast<std::size_t>(e) + 1]),
                                     -1},
        [](const auto& x, const auto& y) { return x.first < y.first; });
    for (auto use = range.first; use != range.second; ++use) {
      tolerance = std::min(tolerance, facet_tolerance(tri_facet_[static_cast<std::size_t>(use->second)]));
    }
    return std::isfinite(tolerance) ? tolerance : 0.0;
  }

  bool exact_vertex(int32_t v) const {
    return witnesses_[static_cast<std::size_t>(v)].deviation == 0.0;
  }

  int32_t find_root(std::span<int32_t> parent, int32_t x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
      parent[static_cast<std::size_t>(x)] =
          parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
      x = parent[static_cast<std::size_t>(x)];
    }
    return x;
  }

  // Triangles of one facet sharing an edge used by exactly those two, in
  // opposite directions and coplanar, belong to one planar group.
  int32_t build_groups() {
    const int32_t count = static_cast<int32_t>(tri_facet_.size());
    for (int32_t t = 0; t < count; ++t) {
      charge_visit();
      for (int r = 0; r < 3; ++r) {
        edge_uses_.push_back({edge_key(tri(t)[r], tri(t)[(r + 1) % 3]), t});
      }
    }
    std::sort(edge_uses_.begin(), edge_uses_.end(), [&](const auto& a, const auto& b) {
      charge_visit();
      return a < b;
    });
    NativeVector<int32_t> parent(static_cast<std::size_t>(count));
    std::iota(parent.begin(), parent.end(), 0);
    for (std::size_t k = 0; k < edge_uses_.size();) {
      charge_visit();
      std::size_t end = k;
      while (end < edge_uses_.size() && edge_uses_[end].first == edge_uses_[k].first) {
        ++end;
      }
      if (end - k == 2) {
        const int32_t s = edge_uses_[k].second;
        const int32_t t = edge_uses_[k + 1].second;
        if (tri_facet_[static_cast<std::size_t>(s)] == tri_facet_[static_cast<std::size_t>(t)] &&
            coplanar_and_opposed(s, t)) {
          parent[static_cast<std::size_t>(find_root(parent, s))] = find_root(parent, t);
        }
      }
      k = end;
    }
    NativeVector<int32_t> group_of_root(static_cast<std::size_t>(count), -1);
    tri_group_.resize(static_cast<std::size_t>(count));
    for (int32_t t = 0; t < count; ++t) {
      charge_visit();
      const int32_t root = find_root(parent, t);
      int32_t& group = group_of_root[static_cast<std::size_t>(root)];
      if (group < 0) {
        group = static_cast<int32_t>(group_facet_.size());
        group_facet_.push_back(tri_facet_[static_cast<std::size_t>(t)]);
      }
      tri_group_[static_cast<std::size_t>(t)] = group;
    }
    return PHX_MC_OK;
  }

  const int32_t* tri(int32_t t) const { return triangles_.data() + 3 * static_cast<int64_t>(t); }

  bool coplanar_and_opposed(int32_t s, int32_t t) const {
    const int32_t* a = tri(s);
    const int32_t* b = tri(t);
    int32_t apex = -1;
    for (int r = 0; r < 3; ++r) {
      if (a[0] != b[r] && a[1] != b[r] && a[2] != b[r]) {
        apex = b[r];
      }
    }
    if (orient3d(in_.points + 3 * a[0], in_.points + 3 * a[1], in_.points + 3 * a[2],
                 in_.points + 3 * apex) != 0) {
      return false;
    }
    for (int r = 0; r < 3; ++r) {
      for (int q = 0; q < 3; ++q) {
        if (a[r] == b[(q + 1) % 3] && a[(r + 1) % 3] == b[q]) {
          return true;
        }
      }
    }
    return false;
  }

  bool internal_edge(uint64_t key) const {
    const auto range = std::equal_range(
        edge_uses_.begin(), edge_uses_.end(), std::pair<uint64_t, int32_t>{key, -1},
        [](const auto& x, const auto& y) { return x.first < y.first; });
    return range.second - range.first == 2 &&
           tri_group_[static_cast<std::size_t>(range.first->second)] ==
               tri_group_[static_cast<std::size_t>((range.first + 1)->second)];
  }

  int32_t add_edge(int32_t a, int32_t b) {
    const int32_t id = static_cast<int32_t>(edges_.size() / 2);
    edges_.push_back(a);
    edges_.push_back(b);
    edge_ids_.emplace(edge_key(a, b), id);
    return id;
  }

  int32_t build_edges() {
    for (int64_t s = 0; s < in_.segment_count; ++s) {
      const int32_t a = in_.segments[2 * s];
      const int32_t b = in_.segments[2 * s + 1];
      const auto existing = edge_ids_.find(edge_key(a, b));
      if (a == b || existing != edge_ids_.end()) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcEdge,
                    existing != edge_ids_.end() ? existing->second : s, kPlcEdge, s);
      }
      add_edge(a, b);
    }
    for (int32_t t = 0; t < static_cast<int32_t>(tri_facet_.size()); ++t) {
      for (int r = 0; r < 3; ++r) {
        const int32_t a = tri(t)[r];
        const int32_t b = tri(t)[(r + 1) % 3];
        const uint64_t key = edge_key(a, b);
        if (!internal_edge(key) && edge_ids_.find(key) == edge_ids_.end()) {
          add_edge(std::min(a, b), std::max(a, b));
        }
      }
    }
    return PHX_MC_OK;
  }

  // Exact pairwise legality of triangles, explicit segments and free points:
  // they may meet only at shared vertices and shared edges.
  int32_t check_contacts() {
    enum Kind : int8_t { kTriangle, kSegment, kFree };
    struct Item {
      Box box;
      Kind kind;
      int32_t index;
    };
    NativeVector<char> referenced(static_cast<std::size_t>(n_), 0);
    for (int32_t v : triangles_) {
      referenced[static_cast<std::size_t>(v)] = 1;
    }
    for (int64_t k = 0; k < 2 * in_.segment_count; ++k) {
      referenced[static_cast<std::size_t>(in_.segments[k])] = 1;
    }
    NativeVector<Item> items;
    const auto box_of = [&](const int32_t* v, int count) {
      Box box{};
      for (int axis = 0; axis < 3; ++axis) {
        box.lower[axis] = box.upper[axis] = in_.points[3 * v[0] + axis];
        for (int r = 1; r < count; ++r) {
          box.lower[axis] = std::min(box.lower[axis], in_.points[3 * v[r] + axis]);
          box.upper[axis] = std::max(box.upper[axis], in_.points[3 * v[r] + axis]);
        }
      }
      return box;
    };
    for (int32_t t = 0; t < static_cast<int32_t>(tri_facet_.size()); ++t) {
      items.push_back({box_of(tri(t), 3), kTriangle, t});
    }
    for (int32_t s = 0; s < static_cast<int32_t>(in_.segment_count); ++s) {
      items.push_back({box_of(in_.segments + 2 * s, 2), kSegment, s});
    }
    for (int32_t v = 0; v < n_; ++v) {
      if (referenced[static_cast<std::size_t>(v)] == 0) {
        free_points_.push_back(v);
        items.push_back({box_of(&v, 1), kFree, v});
      }
    }
    std::sort(items.begin(), items.end(), [](const Item& x, const Item& y) {
      return x.box.lower[0] < y.box.lower[0] ||
             (x.box.lower[0] == y.box.lower[0] &&
              (x.kind < y.kind || (x.kind == y.kind && x.index < y.index)));
    });
    const auto kind_of = [&](const Item& item) -> std::pair<int32_t, int64_t> {
      switch (item.kind) {
        case kTriangle:
          return {kPlcPolygon, tri_polygon_[static_cast<std::size_t>(item.index)]};
        case kSegment:
          return {kPlcEdge, item.index};
        case kFree:
          break;
      }
      return {kPlcPoint, item.index};
    };
    Contact contact;
    for (std::size_t i = 0; i < items.size(); ++i) {
      for (std::size_t j = i + 1; j < items.size(); ++j) {
        if (items[j].box.lower[0] > items[i].box.upper[0]) {
          break;
        }
        if (!overlap(items[i].box, items[j].box)) {
          continue;
        }
        if (!spend(1)) {
          return work_failure();
        }
        ++out_.counters[kPlcContactTests];
        const Item& x = items[i].kind <= items[j].kind ? items[i] : items[j];
        const Item& y = items[i].kind <= items[j].kind ? items[j] : items[i];
        if (!legal_pair(x.kind, x.index, y.kind, y.index, contact)) {
          const auto [first_kind, first] = kind_of(x);
          const auto [second_kind, second] = kind_of(y);
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, first_kind, first,
                      second_kind, second);
        }
      }
    }
    return PHX_MC_OK;
  }

  bool legal_pair(int8_t first_kind, int32_t first, int8_t second_kind, int32_t second,
                  Contact& contact) const {
    const double* p = in_.points;
    const auto ids = [](const int32_t* v, int count, int64_t* out) {
      for (int r = 0; r < count; ++r) {
        out[r] = v[r];
      }
    };
    int64_t first_ids[3];
    int64_t second_ids[3];
    if (first_kind == 0) {  // triangle
      const int32_t* t = tri(first);
      const double* const triangle[3] = {p + 3 * t[0], p + 3 * t[1], p + 3 * t[2]};
      ids(t, 3, first_ids);
      if (second_kind == 0) {
        const int32_t* u = tri(second);
        const double* const other[3] = {p + 3 * u[0], p + 3 * u[1], p + 3 * u[2]};
        ids(u, 3, second_ids);
        return intersect_triangles(triangle, other, first_ids, second_ids, contact) ==
                   PHX_MC_OK &&
               legal_contact(contact.kind);
      }
      if (second_kind == 1) {
        const int32_t* s = in_.segments + 2 * second;
        const double* const segment[2] = {p + 3 * s[0], p + 3 * s[1]};
        ids(s, 2, second_ids);
        return intersect_segment_triangle(segment, triangle, second_ids, first_ids, contact) ==
                   PHX_MC_OK &&
               legal_contact(contact.kind);
      }
      int side = 0;
      return locate_on_triangle(p + 3 * second, triangle, side) == kFeatureNone;
    }
    if (first_kind == 1) {  // segment
      const int32_t* s = in_.segments + 2 * first;
      if (second_kind == 1) {
        const int32_t* u = in_.segments + 2 * second;
        const bool shares = s[0] == u[0] || s[0] == u[1] || s[1] == u[0] || s[1] == u[1];
        if (!shares) {
          return !segments_meet(p + 3 * s[0], p + 3 * s[1], p + 3 * u[0], p + 3 * u[1]);
        }
        // Sharing one endpoint: illegal only when collinear and overlapping.
        const int32_t common = (s[0] == u[0] || s[0] == u[1]) ? s[0] : s[1];
        const int32_t x = s[0] == common ? s[1] : s[0];
        const int32_t y = u[0] == common ? u[1] : u[0];
        const double* c = p + 3 * common;
        if (!collinear3d(c, p + 3 * x, p + 3 * y)) {
          return true;
        }
        for (int axis = 0; axis < 3; ++axis) {
          const double dx = p[3 * x + axis] - c[axis];
          const double dy = p[3 * y + axis] - c[axis];
          if (dx != 0.0 || dy != 0.0) {
            return (dx > 0.0) != (dy > 0.0);
          }
        }
        return false;
      }
      const double* q = p + 3 * second;
      return !(collinear3d(p + 3 * s[0], p + 3 * s[1], q) &&
               segments_meet(p + 3 * s[0], p + 3 * s[1], q, q));
    }
    return true;  // two distinct free points
  }

  // Every region's oriented facet chain is closed: facets bound their positive
  // region with reversed orientation and their negative region directly.
  int32_t check_closure() {
    struct Entry {
      int32_t region;
      uint64_t key;
      int32_t coefficient;
    };
    NativeVector<Entry> entries;
    for (int32_t t = 0; t < static_cast<int32_t>(tri_facet_.size()); ++t) {
      charge_visit();
      const int32_t facet = tri_facet_[static_cast<std::size_t>(t)];
      const int32_t positive = in_.facet_regions[2 * facet];
      const int32_t negative = in_.facet_regions[2 * facet + 1];
      if (positive == negative) {
        continue;
      }
      for (int r = 0; r < 3; ++r) {
        const int32_t a = tri(t)[r];
        const int32_t b = tri(t)[(r + 1) % 3];
        const int32_t sense = a < b ? 1 : -1;
        if (positive >= 0) {
          entries.push_back({positive, edge_key(a, b), -sense});
        }
        if (negative >= 0) {
          entries.push_back({negative, edge_key(a, b), sense});
        }
      }
    }
    std::sort(entries.begin(), entries.end(), [&](const Entry& x, const Entry& y) {
      charge_visit();
      return x.region < y.region || (x.region == y.region && x.key < y.key);
    });
    for (std::size_t k = 0; k < entries.size();) {
      charge_visit();
      std::size_t end = k;
      int32_t sum = 0;
      while (end < entries.size() && entries[end].region == entries[k].region &&
             entries[end].key == entries[k].key) {
        sum += entries[end].coefficient;
        ++end;
      }
      if (sum != 0) {
        const auto edge = edge_ids_.find(entries[k].key);
        return fail(PHX_MC_INVALID_INPUT, kPlcOpenBoundary, kPlcRegion, entries[k].region,
                    kPlcEdge, edge == edge_ids_.end() ? -1 : edge->second);
      }
      k = end;
    }
    return PHX_MC_OK;
  }

  // ------------------------------------------------------------ recovery
  int32_t recover() {
    PhaseTimer preparation(in_.measure_phases, out_, kPlcPreparationPhase);
    int64_t protection_work = 0;
    protection_ = protection_radii(in_.points, n_, edges_, triangles_, protection_work,
                                  std::max<int64_t>(in_.work_limit - work(), 0));
    if (!spend(protection_work) || protection_.size() != static_cast<std::size_t>(n_)) {
      return work_failure();
    }
    for (double radius : protection_) {
      out_.counters[kPlcProtectedVertices] += radius > 0.0 ? 1 : 0;
    }
    points_.assign(in_.points, in_.points + 3 * static_cast<int64_t>(n_));
    add_corners();
    vertex_dimension_.assign(static_cast<std::size_t>(vertex_count()), 0);
    witnesses_.assign(static_cast<std::size_t>(vertex_count()), SourceWitness{});
    edge_points_.assign(edges_.size() / 2, {});
    tri_interior_.assign(tri_facet_.size(), {});
    preparation.stop();
    for (int64_t round = 1;; ++round) {
      out_.counters[kPlcRounds] = round;
      int32_t status;
      {
        PhaseTimer initial(in_.measure_phases, out_, kPlcPreparationPhase);
        status = build_triangulation();
      }
      PhaseTimer boundary(in_.measure_phases && status == PHX_MC_OK, out_, kPlcBoundaryRecoveryPhase);
      bool restart = false;
      if (status == PHX_MC_OK) {
        status = recover_segments(restart);
      }
      if (status == PHX_MC_OK && !restart) {
        status = subdivide_facets();
      }
      if (status == PHX_MC_OK && !restart) {
        status = recover_groups(restart);
      }
      if (status == PHX_MC_OK && !restart) {
        status = recheck_segments(restart);
      }
      if (status != PHX_MC_OK || !restart) {
        return status;
      }
    }
  }

  // Eight corners of a cube about the input bounding box whose distance from
  // every input point exceeds the input diameter, so every PLC entity is
  // interior to the triangulation and every input diametral ball excludes them.
  void add_corners() {
    double lower[3];
    double upper[3];
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = upper[axis] = in_.points[axis];
      for (int32_t v = 1; v < n_; ++v) {
        lower[axis] = std::min(lower[axis], in_.points[3 * v + axis]);
        upper[axis] = std::max(upper[axis], in_.points[3 * v + axis]);
      }
    }
    double reach = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      reach = std::max(reach, upper[axis] - lower[axis]);
    }
    reach = std::exp2(std::ceil(std::log2(std::max(reach, 0x1p-60))) + 2.0);
    for (int corner = 0; corner < kCorners; ++corner) {
      for (int axis = 0; axis < 3; ++axis) {
        const double center = 0.5 * (lower[axis] + upper[axis]);
        points_.push_back((corner >> axis) & 1 ? center + reach : center - reach);
      }
    }
  }

  int32_t budget_failure(int32_t status) {
    if (status != PHX_MC_CAPACITY_EXCEEDED) {
      return status;
    }
    return triangulation_ && triangulation_->refusal() == Refusal::kWork
               ? work_failure()
               : fail(status, kPlcTetrahedronBudget, kPlcNoEntity, -1);
  }

  int32_t build_triangulation() {
    if (triangulation_) {
      work_ += triangulation_->work();
      peak_bytes_ = std::max(peak_bytes_, triangulation_->retained_bytes());
    }
    InsertionLimits limits;
    limits.max_tetrahedra = in_.max_tetrahedra;
    limits.work_limit = std::max<int64_t>(in_.work_limit - work_, 0);
    triangulation_ = make_native_unique<Triangulation3D>(
        points_.data(), nullptr, vertex_count(), limits, kMaxTetrahedronSlots, out_.memory_owner);
    alive_.assign(static_cast<std::size_t>(vertex_count()), 1);
    NativeVector<int32_t> all(static_cast<std::size_t>(vertex_count()));
    std::iota(all.begin(), all.end(), 0);
    const int32_t status =
        triangulation_->build(brio_hilbert_order(points_.data(), 3, all), alive_);
    if (status != PHX_MC_OK) {
      return budget_failure(status);
    }
    vertex_tet_.assign(static_cast<std::size_t>(vertex_count()), -1);
    reconnected_ = false;
    const TetrahedralComplex& complex = triangulation_->complex();
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] != kDeadVertex && !is_ghost(tet)) {
        for (int32_t v : tet.v) {
          vertex_tet_[static_cast<std::size_t>(v)] = static_cast<int32_t>(t);
        }
      }
    }
    return PHX_MC_OK;
  }

  // Tetrahedra containing vertex v (its star), finite and ghost.
  std::span<const int32_t> star(int32_t v) {
    const TetrahedralComplex& complex = triangulation_->complex();
    int32_t start = vertex_tet_[static_cast<std::size_t>(v)];
    if (start < 0 || !complex.live(start) || vertex_slot(complex.tets[start], v) < 0) {
      // Only reachable before any cavity edit, when the walk hint is valid.
      start = triangulation_->locate(point(v));
      vertex_tet_[static_cast<std::size_t>(v)] = start;
    }
    star_.clear();
    ++star_stamp_;
    if (star_mark_.size() < complex.tets.size()) {
      star_mark_.resize(complex.tets.size(), 0U);
    }
    star_.push_back(start);
    star_mark_[static_cast<std::size_t>(start)] = star_stamp_;
    for (std::size_t k = 0; k < star_.size(); ++k) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(star_[k])];
      const int apex = vertex_slot(tet, v);
      for (int s = 0; s < 4; ++s) {
        const int32_t next = tet.n[s];
        if (s != apex && star_mark_[static_cast<std::size_t>(next)] != star_stamp_) {
          star_mark_[static_cast<std::size_t>(next)] = star_stamp_;
          star_.push_back(next);
        }
      }
    }
    spend(static_cast<int64_t>(star_.size()));
    return star_;
  }

  bool has_edge(int32_t a, int32_t b) {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t t : star(a)) {
      if (vertex_slot(complex.tets[static_cast<std::size_t>(t)], b) >= 0) {
        return true;
      }
    }
    return false;
  }

  bool find_face(const int32_t* f, int32_t& t, int& slot) {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t candidate : star(f[0])) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(candidate)];
      if (vertex_slot(tet, f[1]) >= 0 && vertex_slot(tet, f[2]) >= 0) {
        for (int k = 0; k < 4; ++k) {
          if (tet.v[k] != f[0] && tet.v[k] != f[1] && tet.v[k] != f[2]) {
            t = candidate;
            slot = k;
            return true;
          }
        }
      }
    }
    return false;
  }

  // Vertices of PLC edge e from its first to its second endpoint.
  NativeVector<int32_t> chain(int32_t e) const {
    NativeVector<int32_t> vertices{edges_[2 * static_cast<std::size_t>(e)]};
    const NativeVector<int32_t>& inner = edge_points_[static_cast<std::size_t>(e)];
    vertices.insert(vertices.end(), inner.begin(), inner.end());
    vertices.push_back(edges_[2 * static_cast<std::size_t>(e) + 1]);
    return vertices;
  }

  // Dominant axis of PLC edge e; its chain is ordered along it.
  int edge_axis(int32_t e) const {
    const double* first = point(edges_[2 * static_cast<std::size_t>(e)]);
    const double* second = point(edges_[2 * static_cast<std::size_t>(e) + 1]);
    int axis = 0;
    for (int k = 0; k < 3; ++k) {
      if (std::fabs(second[k] - first[k]) > std::fabs(second[axis] - first[axis])) {
        axis = k;
      }
    }
    return axis;
  }

  // Locator of a chain vertex on PLC edge e (input endpoints are exact).
  double edge_parameter(int32_t e, int32_t w) const {
    if (w == edges_[2 * static_cast<std::size_t>(e)]) return 0.0;
    if (w == edges_[2 * static_cast<std::size_t>(e) + 1]) return 1.0;
    return witnesses_[static_cast<std::size_t>(w)].parameters[0];
  }

  // Split point of subsegment (u, v) of PLC edge e: on the protecting sphere
  // or shell of a protected input apex, otherwise near the midpoint. An exact
  // collinear binary64 point is preferred; otherwise, under a positive edge
  // tolerance, the correctly rounded carrier of the exact source-line witness
  // at the same parameter, strictly inside (u, v) along the edge's axis and
  // accepted only within the declared bound.
  int32_t construct_split(int32_t e, int32_t u, int32_t v, double* p, SourceWitness& witness) {
    const auto apex = [&](int32_t w) {
      return w < n_ ? protection_[static_cast<std::size_t>(w)] : 0.0;
    };
    double window = 0.0;
    const double fraction = split_target(point(u), point(v), apex(u), apex(v), window);
    const double* corners[2] = {point(edges_[2 * static_cast<std::size_t>(e)]),
                                point(edges_[2 * static_cast<std::size_t>(e) + 1])};
    const double first = edge_parameter(e, u);
    witness = SourceWitness{SourceStratum::kSegment, e,
                            {first + fraction * (edge_parameter(e, v) - first), 0.0}, 0.0};
    if (exact_vertex(u) && exact_vertex(v) &&
        exact_segment_point(point(u), point(v), fraction, window, p)) {
      source_locator(corners, SourceStratum::kSegment, p, witness.parameters);
      return PHX_MC_OK;
    }
    const double tolerance = edge_tolerance(e);
    if (tolerance == 0.0) {
      return fail(PHX_MC_INVALID_INPUT, kPlcNonrepresentable, kPlcEdge, e);
    }
    if (!spend(16)) {
      return work_failure();
    }
    const int axis = edge_axis(e);
    const double low = std::min(point(u)[axis], point(v)[axis]);
    const double high = std::max(point(u)[axis], point(v)[axis]);
    double deviation = 0.0;
    if (!source_carrier(corners, SourceStratum::kSegment, witness.parameters, p, deviation) ||
        !(low < p[axis] && p[axis] < high)) {
      return fail(PHX_MC_INVALID_INPUT, kPlcNonrepresentable, kPlcEdge, e);
    }
    witness.deviation = deviation;
    if (deviation > tolerance) {
      out_.source_refusal[0] = deviation;
      out_.source_refusal[1] = tolerance;
      return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcEdge, e);
    }
    return PHX_MC_OK;
  }

  // Inserts the constructed split point of subsegment (u, v) of PLC edge e.
  int32_t split(int32_t e, int32_t u, int32_t v, bool insert) {
    double p[3];
    SourceWitness witness;
    const int32_t constructed = construct_split(e, u, v, p, witness);
    if (constructed != PHX_MC_OK) {
      return constructed;
    }
    const int32_t id = append_vertex(p, 1, witness);
    if (id < 0) {
      return fail(PHX_MC_CAPACITY_EXCEEDED, kPlcVertexBudget, kPlcEdge, e);
    }
    NativeVector<int32_t>& inner = edge_points_[static_cast<std::size_t>(e)];
    const int32_t first = edges_[2 * static_cast<std::size_t>(e)];
    const int32_t second = edges_[2 * static_cast<std::size_t>(e) + 1];
    const int axis = edge_axis(e);
    const bool ascending = point(second)[axis] > point(first)[axis];
    const auto before = [&](int32_t x, int32_t y) {
      return ascending ? point(x)[axis] < point(y)[axis] : point(x)[axis] > point(y)[axis];
    };
    inner.insert(std::upper_bound(inner.begin(), inner.end(), id, before), id);
    ++out_.counters[kPlcSegmentSteiner];
    if (!insert) {
      return PHX_MC_OK;
    }
    triangulation_->rebind(points_.data(), vertex_count());
    alive_.push_back(1);
    int32_t duplicate = -1;
    const int32_t status = triangulation_->insert(id, alive_, &duplicate);
    if (status != PHX_MC_OK) {
      return budget_failure(status);
    }
    if (duplicate >= 0) {
      return PHX_MC_INTERNAL_ERROR;
    }
    vertex_tet_.push_back(-1);
    return PHX_MC_OK;
  }

  int32_t append_vertex(const double* p, int8_t dimension, const SourceWitness& witness) {
    if (static_cast<int64_t>(vertex_count()) - kCorners + 1 > in_.max_vertices) {
      return -1;
    }
    const int32_t old_count = vertex_count();
    const std::size_t next_count = static_cast<std::size_t>(old_count) + 1;
    // Every potentially allocating companion growth precedes coordinate
    // relocation; the borrowed triangulation span is rebound without allocation.
    if (vertex_dimension_.capacity() < next_count) {
      vertex_dimension_.reserve(std::max(
          next_count, vertex_dimension_.capacity() + vertex_dimension_.capacity() / 2 + 1));
    }
    if (witnesses_.capacity() < next_count) {
      witnesses_.reserve(std::max(next_count, witnesses_.capacity() + witnesses_.capacity() / 2 + 1));
    }
    if (triangulation_) {
      if (alive_.capacity() < next_count) {
        alive_.reserve(std::max(
            next_count, alive_.capacity() + alive_.capacity() / 2 + 1));
      }
      if (vertex_tet_.capacity() < next_count) {
        vertex_tet_.reserve(std::max(
            next_count, vertex_tet_.capacity() + vertex_tet_.capacity() / 2 + 1));
      }
      triangulation_->reserve_vertices(static_cast<int64_t>(next_count));
    }
    if (points_.capacity() < points_.size() + 3) {
      points_.reserve(std::max(
          points_.size() + 3, points_.capacity() + points_.capacity() / 2 + 3));
    }
    if (triangulation_) {
      triangulation_->rebind(points_.data(), old_count);
    }
    points_.insert(points_.end(), p, p + 3);
    vertex_dimension_.push_back(dimension);
    witnesses_.push_back(witness);
    if (triangulation_) {
      triangulation_->rebind(points_.data(), static_cast<int64_t>(next_count));
    }
    return vertex_count() - 1;
  }

  // Recover every subsegment by constrained cavity reconnection first.
  // Conforming subdivision is a second construction strategy, not a
  // prerequisite for represented segments that have no binary64 split point.
  // A segment between coplanar input triangles that neither reconnects in its
  // own cavity nor splits exactly is deferred: it is an interior edge of the
  // planar facet component recovered from both of its sides, whose fill
  // creates it (recheck_segments still requires it afterwards).
  // Delaunay insertion of a split point is valid only before any
  // reconnection; afterwards the split is recorded and the round restarts
  // from the rebuilt Delaunay triangulation of all points.
  int32_t recover_segments(bool& restart) {
    deferred_segments_.clear();
    for (;;) {
      protected_segments_.clear();
      NativeVector<std::array<int32_t, 3>> missing;
      for (int32_t e = 0; e < static_cast<int32_t>(edges_.size() / 2); ++e) {
        const NativeVector<int32_t> vertices = chain(e);
        for (std::size_t k = 0; k + 1 < vertices.size(); ++k) {
          if (has_edge(vertices[k], vertices[k + 1])) {
            protected_segments_.push_back({vertices[k], vertices[k + 1]});
          } else if (deferred_segments_.count(edge_key(vertices[k], vertices[k + 1])) == 0) {
            missing.push_back({e, vertices[k], vertices[k + 1]});
          }
        }
      }
      if (work_exhausted_ || work() > in_.work_limit) {
        return work_failure();
      }
      if (missing.empty()) {
        return PHX_MC_OK;
      }
      for (const auto& [e, u, v] : missing) {
        if (has_edge(u, v)) {
          continue;
        }
        int32_t status = recover_edge(e, u, v);
        if (status == PHX_MC_OK) {
          continue;
        }
        if (status != PHX_MC_CONSTRAINT_INTERSECTION || out_.failure.reason != kPlcFixedSegment) {
          return status;
        }
        double split_at[3];
        SourceWitness witness;
        int32_t constructed = PHX_MC_INVALID_INPUT;
        if (in_.policy == BoundaryPolicy::kConforming) {
          constructed = construct_split(e, u, v, split_at, witness);
          if (constructed == PHX_MC_CAPACITY_EXCEEDED) {
            return constructed;
          }
        }
        const bool subdivide = constructed == PHX_MC_OK;
        const bool defer = !subdivide && coplanar_incidence(e);
        if (in_.policy == BoundaryPolicy::kFixed && !defer) {
          return status;
        }
        protected_segments_.pop_back();
        recovery_edge_ = {-1, -1};
        out_.failure = PlcFailure{};
        out_.source_refusal[0] = 0.0;
        out_.source_refusal[1] = 0.0;
        if (defer) {
          deferred_segments_.insert(edge_key(u, v));
          continue;
        }
        if (reconnected_) {
          restart = true;
          return split(e, u, v, false);
        }
        status = split(e, u, v, true);
        if (status != PHX_MC_OK) {
          return status;
        }
      }
    }
  }

  // Whether two input triangles bounding PLC edge e are coplanar, so that
  // the edge is interior to a planar component of their subfacets.
  bool coplanar_incidence(int32_t e) const {
    const auto range = std::equal_range(
        edge_uses_.begin(), edge_uses_.end(),
        std::pair<uint64_t, int32_t>{edge_key(edges_[2 * static_cast<std::size_t>(e)],
                                              edges_[2 * static_cast<std::size_t>(e) + 1]),
                                     -1},
        [](const auto& x, const auto& y) { return x.first < y.first; });
    for (auto first = range.first; first != range.second; ++first) {
      for (auto second = first + 1; second != range.second; ++second) {
        if (coplanar_triangles(first->second, second->second)) {
          return true;
        }
      }
    }
    return false;
  }

  bool coplanar_triangles(int32_t s, int32_t t) const {
    const int32_t* a = tri(s);
    for (int r = 0; r < 3; ++r) {
      if (orient3d(in_.points + 3 * a[0], in_.points + 3 * a[1], in_.points + 3 * a[2],
                   in_.points + 3 * tri(t)[r]) != 0) {
        return false;
      }
    }
    return true;
  }

  // Subfacets: each input triangle, or the exact 2D CDT of its boundary with
  // the Steiner points of its PLC edges plus its interior Steiner points.
  int32_t subdivide_facets() {
    sub_.clear();
    sub_group_.clear();
    sub_origin_.clear();
    sub_index_.clear();
    NativeVector<Triple> local;
    for (int32_t t = 0; t < static_cast<int32_t>(tri_facet_.size()); ++t) {
      NativeVector<int32_t> boundary;
      for (int r = 0; r < 3; ++r) {
        const int32_t a = tri(t)[r];
        const int32_t b = tri(t)[(r + 1) % 3];
        boundary.push_back(a);
        const auto edge = edge_ids_.find(edge_key(a, b));
        if (edge != edge_ids_.end()) {
          const NativeVector<int32_t>& inner = edge_points_[static_cast<std::size_t>(edge->second)];
          if (edges_[2 * static_cast<std::size_t>(edge->second)] == a) {
            boundary.insert(boundary.end(), inner.begin(), inner.end());
          } else {
            boundary.insert(boundary.end(), inner.rbegin(), inner.rend());
          }
        }
      }
      const NativeVector<int32_t>& interior = tri_interior_[static_cast<std::size_t>(t)];
      if (boundary.size() == 3 && interior.empty()) {
        add_sub({tri(t)[0], tri(t)[1], tri(t)[2]}, t);
        continue;
      }
      NativeVector<int32_t> all = boundary;
      all.insert(all.end(), interior.begin(), interior.end());
      NativeVector<const double*> corners;
      for (int32_t v : all) {
        corners.push_back(point(v));
      }
      int32_t status = PHX_MC_OK;
      const int axis = projection_axis(point(tri(t)[0]), point(tri(t)[1]), point(tri(t)[2]));
      if (!spend(static_cast<int64_t>(all.size()))) {
        return work_failure();
      }
      int64_t used_work = 0;
      int32_t refusal = kPlcNoFailure;
      const bool valid = triangulate_planar(
          corners, static_cast<int64_t>(boundary.size()), axis, local, status,
          std::max<int64_t>(in_.work_limit - work(), 0), used_work, refusal);
      if (!spend(used_work)) {
        return work_failure();
      }
      if (!valid) {
        if (status == PHX_MC_CAPACITY_EXCEEDED) {
          return fail(status, refusal, kPlcPolygon, tri_polygon_[static_cast<std::size_t>(t)]);
        }
        // Exact points of an exactly planar polygon always triangulate.
        // Bounded carriers may fold its projection: a source-deviation refusal.
        double deviation = 0.0;
        for (int32_t v : all) {
          deviation = std::max(deviation, witnesses_[static_cast<std::size_t>(v)].deviation);
        }
        if (deviation == 0.0) {
          return PHX_MC_INTERNAL_ERROR;
        }
        out_.source_refusal[0] = deviation;
        out_.source_refusal[1] = facet_tolerance(tri_facet_[static_cast<std::size_t>(t)]);
        return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcPolygon,
                    tri_polygon_[static_cast<std::size_t>(t)]);
      }
      for (const Triple& triangle : local) {
        add_sub({all[static_cast<std::size_t>(triangle[0])],
                 all[static_cast<std::size_t>(triangle[1])],
                 all[static_cast<std::size_t>(triangle[2])]},
                t);
      }
    }
    return PHX_MC_OK;
  }

  void add_sub(const Triple& triangle, int32_t origin) {
    sub_index_.emplace(sorted_triple(triangle.data()), static_cast<int32_t>(sub_.size()));
    sub_.push_back(triangle);
    sub_group_.push_back(tri_group_[static_cast<std::size_t>(origin)]);
    sub_origin_.push_back(origin);
  }

  int32_t sub_facet(int32_t s) const {
    return tri_facet_[static_cast<std::size_t>(sub_origin_[static_cast<std::size_t>(s)])];
  }

  int32_t recover_groups(bool& restart) {
    TetrahedralComplex& complex = triangulation_->complex();
    complex.enable_labels();
    NativeVector<NativeVector<int32_t>> members(group_facet_.size());
    NativeVector<std::pair<uint64_t, int32_t>> uses;
    for (int32_t s = 0; s < static_cast<int32_t>(sub_.size()); ++s) {
      const Triple& t = sub_[static_cast<std::size_t>(s)];
      members[static_cast<std::size_t>(sub_group_[static_cast<std::size_t>(s)])].push_back(s);
      for (int r = 0; r < 3; ++r) {
        uses.push_back({edge_key(t[r], t[(r + 1) % 3]), s});
      }
    }
    std::sort(uses.begin(), uses.end());
    for (int32_t g = 0; g < static_cast<int32_t>(members.size()); ++g) {
      NativeVector<int32_t> missing;
      for (int32_t s : members[static_cast<std::size_t>(g)]) {
        int32_t t = -1;
        int slot = -1;
        if (!find_face(sub_[static_cast<std::size_t>(s)].data(), t, slot)) {
          missing.push_back(s);
          continue;
        }
        const int32_t existing = complex.constraint(t, slot);
        if (existing == kNoConstraint) {
          complex.set_facet_constraint(t, slot, sub_facet(s));
          triangulation_->count_constrained_facet();
        } else if (existing != sub_facet(s)) {
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcFacet, existing,
                      kPlcFacet, sub_facet(s));
        }
      }
      join_deferred_neighbors(missing, uses);
      for (const NativeVector<int32_t>& component : components(missing)) {
        bool failed = false;
        const int32_t status = recover_component(component, failed);
        if (status != PHX_MC_OK) {
          return status;
        }
        if (!failed) {
          continue;
        }
        const int32_t facet = group_facet_[static_cast<std::size_t>(g)];
        if (in_.policy == BoundaryPolicy::kFixed) {
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet, kPlcFacet, facet);
        }
        return add_facet_steiner(component, facet, restart);
      }
    }
    return PHX_MC_OK;
  }

  // A deferred segment that is still absent is an interior edge of the
  // planar component of the missing subfacets on both of its sides: missing
  // coplanar subfacets of other groups across it join the component, which
  // keeps each subfacet's own facet label.
  void join_deferred_neighbors(NativeVector<int32_t>& missing,
                               std::span<const std::pair<uint64_t, int32_t>> uses) {
    if (deferred_segments_.empty()) {
      return;
    }
    for (std::size_t k = 0; k < missing.size(); ++k) {
      const Triple g = sub_[static_cast<std::size_t>(missing[k])];
      for (int r = 0; r < 3; ++r) {
        const int32_t a = g[r];
        const int32_t b = g[(r + 1) % 3];
        const uint64_t key = edge_key(a, b);
        if (deferred_segments_.count(key) == 0 || has_edge(a, b)) {
          continue;
        }
        const auto range = std::equal_range(
            uses.begin(), uses.end(), std::pair<uint64_t, int32_t>{key, -1},
            [](const auto& x, const auto& y) { return x.first < y.first; });
        for (auto use = range.first; use != range.second; ++use) {
          const Triple& t = sub_[static_cast<std::size_t>(use->second)];
          const int32_t apex = t[0] != a && t[0] != b ? t[0] : t[1] != a && t[1] != b ? t[1] : t[2];
          if (std::find(missing.begin(), missing.end(), use->second) == missing.end() &&
              orient3d(point(g[0]), point(g[1]), point(g[2]), point(apex)) == 0) {
            missing.push_back(use->second);
          }
        }
      }
    }
  }

  // Missing subfacets connected through shared edges into exactly planar
  // components (bounded carriers make neighbouring subfacets non-coplanar;
  // each planar piece is then recovered on its own plane).
  NativeVector<NativeVector<int32_t>> components(std::span<const int32_t> missing) const {
    NativeVector<int32_t> parent(missing.size());
    std::iota(parent.begin(), parent.end(), 0);
    NativeVector<std::pair<uint64_t, int32_t>> uses;
    for (int32_t k = 0; k < static_cast<int32_t>(missing.size()); ++k) {
      const Triple& t = sub_[static_cast<std::size_t>(missing[static_cast<std::size_t>(k)])];
      for (int r = 0; r < 3; ++r) {
        uses.push_back({edge_key(t[r], t[(r + 1) % 3]), k});
      }
    }
    std::sort(uses.begin(), uses.end());
    NativeVector<int32_t> scratch = parent;
    for (std::size_t k = 1; k < uses.size(); ++k) {
      if (uses[k].first == uses[k - 1].first &&
          coplanar_subfacets(missing[static_cast<std::size_t>(uses[k].second)],
                             missing[static_cast<std::size_t>(uses[k - 1].second)])) {
        int32_t a = uses[k].second;
        int32_t b = uses[k - 1].second;
        while (scratch[static_cast<std::size_t>(a)] != a) {
          a = scratch[static_cast<std::size_t>(a)];
        }
        while (scratch[static_cast<std::size_t>(b)] != b) {
          b = scratch[static_cast<std::size_t>(b)];
        }
        scratch[static_cast<std::size_t>(a)] = b;
      }
    }
    NativeMap<int32_t, NativeVector<int32_t>> grouped;
    for (int32_t k = 0; k < static_cast<int32_t>(missing.size()); ++k) {
      int32_t root = k;
      while (scratch[static_cast<std::size_t>(root)] != root) {
        root = scratch[static_cast<std::size_t>(root)];
      }
      grouped[root].push_back(missing[static_cast<std::size_t>(k)]);
    }
    NativeVector<NativeVector<int32_t>> result;
    for (auto& [root, list] : grouped) {
      result.push_back(std::move(list));
    }
    return result;
  }

  bool coplanar_subfacets(int32_t s, int32_t t) const {
    const Triple& a = sub_[static_cast<std::size_t>(s)];
    for (int32_t v : sub_[static_cast<std::size_t>(t)]) {
      if (orient3d(point(a[0]), point(a[1]), point(a[2]), point(v)) != 0) {
        return false;
      }
    }
    return true;
  }

  // Whether p lies strictly inside subfacet s in the projection of its input
  // triangle, where the subfacet CDT is built.
  bool inside_projection(int32_t s, const double* p) const {
    const Triple& t = sub_[static_cast<std::size_t>(s)];
    const int32_t* input = tri(sub_origin_[static_cast<std::size_t>(s)]);
    const int axis = projection_axis(point(input[0]), point(input[1]), point(input[2]));
    const int x = (axis + 1) % 3;
    const int y = (axis + 2) % 3;
    const double q[2] = {p[x], p[y]};
    int sign = 0;
    for (int r = 0; r < 3; ++r) {
      const double a[2] = {point(t[r])[x], point(t[r])[y]};
      const double b[2] = {point(t[(r + 1) % 3])[x], point(t[(r + 1) % 3])[y]};
      const int side = orient2d(a, b, q);
      if (side == 0 || (sign != 0 && side != sign)) {
        return false;
      }
      sign = side;
    }
    return true;
  }

  int32_t add_facet_steiner(std::span<const int32_t> component, int32_t facet,
                            bool& restart) {
    int32_t largest = component[0];
    double best = -1.0;
    for (int32_t s : component) {
      const Triple& t = sub_[static_cast<std::size_t>(s)];
      const double* a = point(t[0]);
      const double* b = point(t[1]);
      const double* c = point(t[2]);
      const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      const double w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
      const double n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                           u[0] * w[1] - u[1] * w[0]};
      const double area = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
      if (area > best) {
        best = area;
        largest = s;
      }
    }
    const Triple& t = sub_[static_cast<std::size_t>(largest)];
    const int32_t origin = sub_origin_[static_cast<std::size_t>(largest)];
    const int32_t* input = tri(origin);
    const double* corners[3] = {point(input[0]), point(input[1]), point(input[2])};
    SourceWitness witness{SourceStratum::kFacet, origin, {0.0, 0.0}, 0.0};
    double p[3];
    if (exact_vertex(t[0]) && exact_vertex(t[1]) && exact_vertex(t[2]) &&
        exact_triangle_point(point(t[0]), point(t[1]), point(t[2]), p)) {
      source_locator(corners, SourceStratum::kFacet, p, witness.parameters);
    } else {
      // The carrier of the input-triangle witness at the subfacet centroid.
      const double tolerance = facet_tolerance(facet);
      if (tolerance == 0.0) {
        return fail(PHX_MC_INVALID_INPUT, kPlcNonrepresentable, kPlcFacet, facet);
      }
      if (!spend(24)) {
        return work_failure();
      }
      double centroid[3];
      for (int k = 0; k < 3; ++k) {
        centroid[k] = (point(t[0])[k] + point(t[1])[k] + point(t[2])[k]) / 3.0;
      }
      source_locator(corners, SourceStratum::kFacet, centroid, witness.parameters);
      if (!source_carrier(corners, SourceStratum::kFacet, witness.parameters, p,
                          witness.deviation) ||
          !inside_projection(largest, p)) {
        return fail(PHX_MC_INVALID_INPUT, kPlcNonrepresentable, kPlcFacet, facet);
      }
      if (witness.deviation > tolerance) {
        out_.source_refusal[0] = witness.deviation;
        out_.source_refusal[1] = tolerance;
        return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcFacet, facet);
      }
    }
    const int32_t id = append_vertex(p, 2, witness);
    if (id < 0) {
      return fail(PHX_MC_CAPACITY_EXCEEDED, kPlcVertexBudget, kPlcFacet, facet);
    }
    tri_interior_[static_cast<std::size_t>(origin)].push_back(id);
    ++out_.counters[kPlcFacetSteiner];
    restart = true;
    return PHX_MC_OK;
  }

  // Subsegments lost by a facet fill are split (conforming) or refused.
  int32_t recheck_segments(bool& restart) {
    for (int32_t e = 0; e < static_cast<int32_t>(edges_.size() / 2); ++e) {
      const NativeVector<int32_t> vertices = chain(e);
      for (std::size_t k = 0; k + 1 < vertices.size(); ++k) {
        if (has_edge(vertices[k], vertices[k + 1])) {
          continue;
        }
        if (in_.policy == BoundaryPolicy::kFixed) {
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, e);
        }
        restart = true;
        const int32_t status = split(e, vertices[k], vertices[k + 1], false);
        if (status != PHX_MC_OK) {
          return status;
        }
      }
    }
    return PHX_MC_OK;
  }

  // ---------------------------------------------------- facet cavities
  struct Face {
    Triple v;
    bool open;
    bool obstacle;
    Box box;
  };

  struct Side {
    NativeVector<Face> faces;
    NativeMap<Triple, int32_t> index;  // oriented key -> face
    NativeVector<int32_t> vertices;
    NativeVector<std::array<int32_t, 4>> tets;
  };

  Box box_of(const int32_t* v, int count) const {
    Box box{};
    for (int axis = 0; axis < 3; ++axis) {
      box.lower[axis] = box.upper[axis] = point(v[0])[axis];
      for (int r = 1; r < count; ++r) {
        box.lower[axis] = std::min(box.lower[axis], point(v[r])[axis]);
        box.upper[axis] = std::max(box.upper[axis], point(v[r])[axis]);
      }
    }
    return box;
  }

  void add_face(Side& side, const Triple& v, bool obstacle) {
    side.index.emplace(oriented_key(v.data()), static_cast<int32_t>(side.faces.size()));
    side.faces.push_back({v, true, obstacle, box_of(v.data(), 3)});
  }

  void reset_front(Side& side, std::size_t initial_faces) {
    while (side.faces.size() > initial_faces) {
      side.index.erase(oriented_key(side.faces.back().v.data()));
      side.faces.pop_back();
    }
    for (Face& face : side.faces) {
      face.open = true;
    }
    side.tets.clear();
  }

  bool triangle_contact_legal(const Triple& x, const Triple& y) {
    const double* const first[3] = {point(x[0]), point(x[1]), point(x[2])};
    const double* const second[3] = {point(y[0]), point(y[1]), point(y[2])};
    const int64_t first_ids[3] = {x[0], x[1], x[2]};
    const int64_t second_ids[3] = {y[0], y[1], y[2]};
    ++out_.counters[kPlcContactTests];
    Contact contact;
    return intersect_triangles(first, second, first_ids, second_ids, contact) == PHX_MC_OK &&
           legal_contact(contact.kind);
  }

  bool segment_face_legal(int32_t u, int32_t v, const Triple& face) {
    const double* const segment[2] = {point(u), point(v)};
    const double* const triangle[3] = {point(face[0]), point(face[1]), point(face[2])};
    const int64_t segment_ids[2] = {u, v};
    const int64_t triangle_ids[3] = {face[0], face[1], face[2]};
    if (!spend(1)) {
      return false;
    }
    ++out_.counters[kPlcContactTests];
    Contact contact;
    return intersect_segment_triangle(segment, triangle, segment_ids, triangle_ids, contact) ==
               PHX_MC_OK &&
           legal_contact(contact.kind);
  }

  // Whether tetrahedron (f, w) may join the fill of `side`.  A rejection by
  // contact with an obstacle face reports that face's key in `blocker`.
  bool admissible(const Side& side, const Triple& f, int32_t w, Triple* blocker = nullptr) {
    const std::array<Triple, 3> faces{Triple{f[1], w, f[2]}, Triple{f[0], f[2], w},
                                      Triple{f[0], w, f[1]}};
    for (const Triple& face : faces) {
      const auto same = side.index.find(oriented_key(face.data()));
      if (same != side.index.end()) {
        if (!side.faces[static_cast<std::size_t>(same->second)].open) {
          return false;
        }
      } else if (side.index.count(reversed_key(face.data())) != 0) {
        return false;
      }
    }
    const int32_t tet[4] = {f[0], f[1], f[2], w};
    const Box box = box_of(tet, 4);
    const Triple own = sorted_triple(f.data());
    for (const Face& other : side.faces) {
      if (!(other.open || other.obstacle) || !overlap(box, other.box)) {
        continue;
      }
      const Triple key = sorted_triple(other.v.data());
      if (key == own) {
        continue;
      }
      for (const Triple& face : faces) {
        if (key != sorted_triple(face.data()) && !triangle_contact_legal(face, other.v)) {
          if (blocker != nullptr && other.obstacle) {
            *blocker = key;
          }
          return false;
        }
      }
    }
    // A constrained curve must not pass through any face of a new cell.
    // Combined with vertex exclusion below, this forces its trace to be
    // mesh edges, rather than merely preserving its endpoint coordinates.
    for (const auto& segment : protected_segments_) {
      if (!overlap(box, box_of(segment.data(), 2))) {
        continue;
      }
      if (!segment_face_legal(segment[0], segment[1], f)) {
        return false;
      }
      for (const Triple& face : faces) {
        if (!segment_face_legal(segment[0], segment[1], face)) {
          return false;
        }
      }
    }
    for (int32_t u : side.vertices) {
      if (u == f[0] || u == f[1] || u == f[2] || u == w) {
        continue;
      }
      if (orient3d(point(f[0]), point(f[1]), point(f[2]), point(u)) >= 0 &&
          orient3d(point(f[1]), point(w), point(f[2]), point(u)) >= 0 &&
          orient3d(point(f[0]), point(f[2]), point(w), point(u)) >= 0 &&
          orient3d(point(f[0]), point(w), point(f[1]), point(u)) >= 0) {
        return false;
      }
    }
    return true;
  }

  // Adds every one-sided tetrahedron across an unconstrained cavity facet,
  // or with `blocked_only` across those facets in `blockers_` only; false
  // when there is none.
  template <class SideOf>
  bool grow_cavity(NativeVector<int32_t>& cavity, SideOf& side_of, bool blocked_only = false) {
    const TetrahedralComplex& complex = triangulation_->complex();
    NativeVector<int32_t> added;
    for (int32_t t : cavity) {
      for (int k = 0; k < 4; ++k) {
        const int32_t next = complex.tets[static_cast<std::size_t>(t)].n[k];
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(next)];
        if (is_ghost(tet) || complex.constraint(t, k) != kNoConstraint ||
            std::binary_search(cavity.begin(), cavity.end(), next)) {
          continue;
        }
        if (blocked_only) {
          Triple face{};
          oriented_facet(complex.tets[static_cast<std::size_t>(t)].v, k, face.data());
          if (std::find(blockers_.begin(), blockers_.end(), sorted_triple(face.data())) ==
              blockers_.end()) {
            continue;
          }
        }
        bool positive = false;
        bool negative = false;
        for (int32_t v : tet.v) {
          const int value = side_of(v);
          positive = positive || value > 0;
          negative = negative || value < 0;
        }
        if (!(positive && negative)) {
          // Only the first eligible cavity neighbor discovers this cell.
          // This bounds the unique touched extent without another ledger.
          bool discovered = false;
          for (int j = 0; j < 4; ++j) {
            const int32_t previous = tet.n[j];
            if (previous >= t || complex.constraint(next, j) != kNoConstraint ||
                !std::binary_search(cavity.begin(), cavity.end(), previous)) {
              continue;
            }
            if (blocked_only) {
              Triple face{};
              oriented_facet(tet.v, j, face.data());
              if (std::find(blockers_.begin(), blockers_.end(), sorted_triple(face.data())) ==
                  blockers_.end()) {
                continue;
              }
            }
            discovered = true;
            break;
          }
          if (discovered) {
            continue;
          }
          if (!native_execution_cavity(cavity.size() + added.size() + 1)) {
            return false;
          }
          added.push_back(next);
          out_.counters[kPlcLargestCavity] =
              std::max(out_.counters[kPlcLargestCavity],
                       static_cast<int64_t>(cavity.size() + added.size()));
        }
      }
    }
    std::sort(added.begin(), added.end());
    // Discovery above is unique; sorting preserves the original merge order.
    if (added.empty() || !spend(4 * static_cast<int64_t>(added.size()))) {
      return false;
    }
    const std::size_t middle = cavity.size();
    cavity.resize(middle + added.size());
    std::size_t old = middle;
    std::size_t extra = added.size();
    std::size_t destination = cavity.size();
    while (extra > 0) {
      --destination;
      if (old > 0 && added[extra - 1] < cavity[old - 1]) {
        cavity[destination] = cavity[--old];
      } else {
        cavity[destination] = added[--extra];
      }
    }
    return true;
  }

  // A point strictly on the positive side of every facet of `side` (its
  // kernel), searched between the component and the side's off-plane
  // vertices and verified exactly; false when no candidate verifies.
  bool kernel_point(const Side& side, std::span<const int32_t> component,
                    NativeVector<double>& cone) {
    const Triple& base = sub_[static_cast<std::size_t>(component[0])];
    NativeVector<int32_t> apexes;
    for (int32_t v : side.vertices) {
      if (orient3d(point(base[0]), point(base[1]), point(base[2]), point(v)) != 0) {
        apexes.push_back(v);
      }
    }
    const auto inside = [&](const double* q) {
      if (!spend(static_cast<int64_t>(side.faces.size() + tri_facet_.size()))) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!coordinate_in_domain(q[axis])) {
          return false;
        }
      }
      for (const Face& face : side.faces) {
        if (orient3d(point(face.v[0]), point(face.v[1]), point(face.v[2]), q) <= 0) {
          return false;
        }
      }
      for (int32_t t = 0; t < static_cast<int32_t>(tri_facet_.size()); ++t) {
        const double* const triangle[3] = {point(tri(t)[0]), point(tri(t)[1]), point(tri(t)[2])};
        int source_side = 0;
        if (locate_on_triangle(q, triangle, source_side) != kFeatureNone) {
          return false;
        }
      }
      return true;
    };
    for (double fraction : {0.5, 0.25, 0.125, 0.0625, 0.03125}) {
      for (int32_t s : component) {
        const Triple& g = sub_[static_cast<std::size_t>(s)];
        double center[3];
        for (int axis = 0; axis < 3; ++axis) {
          center[axis] = (point(g[0])[axis] + point(g[1])[axis] + point(g[2])[axis]) / 3.0;
        }
        for (int32_t apex : apexes) {
          double q[3];
          for (int axis = 0; axis < 3; ++axis) {
            q[axis] = center[axis] + fraction * (point(apex)[axis] - center[axis]);
          }
          if (inside(q)) {
            cone.assign(q, q + 3);
            return true;
          }
          if (work_exhausted_ || work() > in_.work_limit) {
            return false;
          }
        }
      }
    }
    return false;
  }

  // Removes Steiner points appended for an edit that did not commit.
  void discard(std::span<const int32_t> appended) {
    for (std::size_t k = 0; k < appended.size(); ++k) {
      points_.resize(points_.size() - 3);
      vertex_dimension_.pop_back();
      witnesses_.pop_back();
      alive_.pop_back();
      vertex_tet_.pop_back();
    }
    if (!appended.empty()) {
      triangulation_->rebind(points_.data(), vertex_count());
    }
  }

  // Constrained-Delaunay gift wrapping: every open face takes, among the side
  // vertices strictly on its positive side, the admissible one first in the
  // exact (symbolically perturbed) circumsphere order.  A trapped front
  // leaves in `blockers_` the obstacle faces that refused its candidates,
  // and the trapped face itself when it is an obstacle.
  bool gift_wrap_greedy(Side& side) {
    blockers_.clear();
    NativeVector<int32_t> queue(side.faces.size());
    std::iota(queue.begin(), queue.end(), 0);
    std::reverse(queue.begin(), queue.end());
    NativeVector<int32_t> remaining;
    NativeVector<Triple> refused;
    while (!queue.empty()) {
      const int32_t index = queue.back();
      queue.pop_back();
      if (!side.faces[static_cast<std::size_t>(index)].open) {
        continue;
      }
      const Triple f = side.faces[static_cast<std::size_t>(index)].v;
      remaining.clear();
      for (int32_t w : side.vertices) {
        if (w != f[0] && w != f[1] && w != f[2] &&
            orient3d(point(f[0]), point(f[1]), point(f[2]), point(w)) > 0) {
          remaining.push_back(w);
        }
      }
      int32_t chosen = -1;
      refused.clear();
      while (!remaining.empty()) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < remaining.size(); ++k) {
          const int32_t b = remaining[best];
          const int32_t w = remaining[k];
          const bool b_endpoint = b == recovery_edge_[0] || b == recovery_edge_[1];
          const bool w_endpoint = w == recovery_edge_[0] || w == recovery_edge_[1];
          if ((!b_endpoint && w_endpoint) ||
              (b_endpoint == w_endpoint &&
               insphere_sos(point(f[0]), point(f[1]), point(f[2]), point(b), point(w), f[0],
                            f[1], f[2], b, w) > 0)) {
            best = k;
          }
        }
        ++out_.counters[kPlcCandidates];
        if (!spend(static_cast<int64_t>(remaining.size() + side.faces.size()))) {
          return false;
        }
        Triple blocker{-1, -1, -1};
        if (admissible(side, f, remaining[best], &blocker)) {
          chosen = remaining[best];
          break;
        }
        if (blocker[0] >= 0) {
          refused.push_back(blocker);
        }
        remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(best));
      }
      if (chosen < 0) {
        blockers_.swap(refused);
        if (side.faces[static_cast<std::size_t>(index)].obstacle) {
          blockers_.push_back(sorted_triple(f.data()));
        }
        return false;
      }
      side.tets.push_back({f[0], f[1], f[2], chosen});
      side.faces[static_cast<std::size_t>(index)].open = false;
      const std::array<Triple, 3> faces{Triple{f[1], chosen, f[2]}, Triple{f[0], f[2], chosen},
                                        Triple{f[0], chosen, f[1]}};
      for (const Triple& face : faces) {
        const auto same = side.index.find(oriented_key(face.data()));
        if (same != side.index.end()) {
          side.faces[static_cast<std::size_t>(same->second)].open = false;
        } else {
          add_face(side, Triple{face[0], face[2], face[1]}, false);
          queue.push_back(static_cast<int32_t>(side.faces.size() - 1));
        }
      }
    }
    return true;
  }

  // Greedy wrapping, then the exhaustive search from the reset front.
  bool gift_wrap(Side& side) {
    const std::size_t initial_faces = side.faces.size();
    if (gift_wrap_greedy(side)) {
      return true;
    }
    reset_front(side, initial_faces);
    return gift_wrap_search(side);
  }

  // Immutable constraints can be non-Delaunay.  If greedy circumsphere
  // ordering traps a front, search alternative legal ears from the reset
  // front, undoing each rejected front exactly.  Work exhaustion is sticky
  // and distinguishes an unfinished search from a proved absence of a
  // vertex-only fill.
  bool gift_wrap_search(Side& side) {
    const std::size_t initial_faces = side.faces.size();
    const auto reset = [&]() { reset_front(side, initial_faces); };
    if (work_exhausted_) {
      return false;
    }
    struct Frame {
      int32_t face;
      NativeVector<int32_t> choices;
      std::size_t face_count = 0;
      std::array<int32_t, 4> closed{};
      int closed_count = 0;
      bool active = false;
    };
    NativeVector<Frame> stack;
    const auto push = [&]() {
      for (int32_t i = 0; i < static_cast<int32_t>(side.faces.size()); ++i) {
        if (!side.faces[static_cast<std::size_t>(i)].open) {
          continue;
        }
        const Triple& f = side.faces[static_cast<std::size_t>(i)].v;
        Frame frame;
        frame.face = i;
        for (int32_t w : side.vertices) {
          if (orient3d(point(f[0]), point(f[1]), point(f[2]), point(w)) > 0) {
            frame.choices.push_back(w);
          }
        }
        stack.push_back(std::move(frame));
        return true;
      }
      return false;
    };
    if (!push()) {
      return true;
    }
    while (!stack.empty()) {
      Frame& frame = stack.back();
      if (frame.active) {
        while (side.faces.size() > frame.face_count) {
          side.index.erase(oriented_key(side.faces.back().v.data()));
          side.faces.pop_back();
        }
        for (int k = 0; k < frame.closed_count; ++k) {
          side.faces[static_cast<std::size_t>(frame.closed[k])].open = true;
        }
        side.tets.pop_back();
        frame.active = false;
      }
      if (frame.choices.empty()) {
        stack.pop_back();
        continue;
      }
      const Triple f = side.faces[static_cast<std::size_t>(frame.face)].v;
      std::size_t best = 0;
      for (std::size_t k = 1; k < frame.choices.size(); ++k) {
        const int32_t b = frame.choices[best], w = frame.choices[k];
        if (insphere_sos(point(f[0]), point(f[1]), point(f[2]), point(b), point(w),
                         f[0], f[1], f[2], b, w) > 0) {
          best = k;
        }
      }
      const int32_t chosen = frame.choices[best];
      frame.choices.erase(frame.choices.begin() + static_cast<std::ptrdiff_t>(best));
      ++out_.counters[kPlcCandidates];
      if (!spend(static_cast<int64_t>(side.faces.size() + side.vertices.size()))) {
        reset();
        return false;
      }
      if (!admissible(side, f, chosen)) {
        continue;
      }
      frame.face_count = side.faces.size();
      frame.closed_count = 1;
      frame.closed[0] = frame.face;
      frame.active = true;
      side.faces[static_cast<std::size_t>(frame.face)].open = false;
      side.tets.push_back({f[0], f[1], f[2], chosen});
      const std::array<Triple, 3> faces{Triple{f[1], chosen, f[2]},
                                      Triple{f[0], f[2], chosen},
                                      Triple{f[0], chosen, f[1]}};
      for (const Triple& face : faces) {
        const auto same = side.index.find(oriented_key(face.data()));
        if (same == side.index.end()) {
          add_face(side, Triple{face[0], face[2], face[1]}, false);
        } else {
          frame.closed[frame.closed_count++] = same->second;
          side.faces[static_cast<std::size_t>(same->second)].open = false;
        }
      }
      if (!push()) {
        return true;
      }
    }
    reset();
    return false;
  }

  bool edge_kernel_point(const Side& side, double* q) {
    const auto inside = [&]() {
      if (!spend(static_cast<int64_t>(side.faces.size() + tri_facet_.size()))) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!coordinate_in_domain(q[axis])) {
          return false;
        }
      }
      for (const Face& face : side.faces) {
        if (orient3d(point(face.v[0]), point(face.v[1]), point(face.v[2]), q) <= 0) {
          return false;
        }
      }
      // Interior insertion is permitted by the fixed policy; insertion on
      // any original PLC face is not, even before that face is recovered.
      for (int32_t t = 0; t < static_cast<int32_t>(tri_facet_.size()); ++t) {
        const double* const triangle[3] = {point(tri(t)[0]), point(tri(t)[1]), point(tri(t)[2])};
        int side_of = 0;
        if (locate_on_triangle(q, triangle, side_of) != kFeatureNone) {
          return false;
        }
      }
      return true;
    };
    std::fill_n(q, 3, 0.0);
    for (int32_t w : side.vertices) {
      for (int axis = 0; axis < 3; ++axis) {
        q[axis] += point(w)[axis] / static_cast<double>(side.vertices.size());
      }
    }
    if (inside()) {
      return true;
    }
    for (double fraction : {0.5, 0.25, 0.125, 0.0625}) {
      for (const Face& face : side.faces) {
        for (int32_t w : side.vertices) {
          for (int axis = 0; axis < 3; ++axis) {
            const double center = (point(face.v[0])[axis] + point(face.v[1])[axis] +
                                   point(face.v[2])[axis]) / 3.0;
            q[axis] = center + fraction * (point(w)[axis] - center);
          }
          if (inside()) {
            return true;
          }
          if (work_exhausted_) {
            return false;
          }
        }
      }
    }
    return false;
  }

  // Recover an immutable PLC edge by replacing its intersected star, without
  // changing the edge or any already constrained facet.  Exact segment/face
  // contacts select the cavity; the same contacts prohibit a candidate fill
  // from cutting the edge.  CavityEdit verifies the oriented boundary and
  // all surviving facet constraints before committing the replacement.
  int32_t recover_edge(int32_t e, int32_t u, int32_t v) {
    recovery_edge_ = {u, v};
    protected_segments_.push_back({u, v});
    TetrahedralComplex& complex = triangulation_->complex();
    complex.enable_labels();
    NativeVector<int32_t> cavity;
    const int32_t edge[2] = {u, v};
    const Box edge_box = box_of(edge, 2);
    for (int32_t t = 0; t < static_cast<int32_t>(complex.tets.size()); ++t) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(t)];
      if (!complex.live(t) || is_ghost(tet) || !overlap(edge_box, box_of(tet.v, 4))) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        Triple face{};
        oriented_facet(tet.v, k, face.data());
        if (!segment_face_legal(u, v, face)) {
          if (!native_execution_cavity(cavity.size() + 1)) {
            return fail(native_execution_status(PHX_MC_CAPACITY_EXCEEDED),
                        kPlcTetrahedronBudget, kPlcEdge, e);
          }
          cavity.push_back(t);
          out_.counters[kPlcLargestCavity] =
              std::max(out_.counters[kPlcLargestCavity], static_cast<int64_t>(cavity.size()));
          break;
        }
      }
    }
    if (work_exhausted_) {
      return work_failure();
    }
    if (cavity.empty()) {
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, e);
    }
    Side side;
    NativeMap<Triple, int32_t> labels;
    NativeVector<int32_t> appended;
    for (;;) {
      side = Side{};
      labels.clear();
      for (int32_t t : cavity) {
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(t)];
        side.vertices.insert(side.vertices.end(), tet.v, tet.v + 4);
        for (int k = 0; k < 4; ++k) {
          const bool inside = std::binary_search(cavity.begin(), cavity.end(), tet.n[k]);
          const int32_t constraint = complex.constraint(t, k);
          if (inside && (constraint == kNoConstraint || tet.n[k] < t)) {
            continue;
          }
          Triple face{};
          oriented_facet(tet.v, k, face.data());
          add_face(side, face, true);
          if (constraint != kNoConstraint) {
            labels.emplace(sorted_triple(face.data()), constraint);
          }
          if (inside) {
            add_face(side, Triple{face[0], face[2], face[1]}, true);
          }
        }
      }
      std::sort(side.vertices.begin(), side.vertices.end());
      side.vertices.erase(std::unique(side.vertices.begin(), side.vertices.end()),
                          side.vertices.end());
      const std::size_t initial_faces = side.faces.size();
      if (gift_wrap_greedy(side)) {
        break;
      }
      if (work_exhausted_) {
        return work_failure();
      }
      reset_front(side, initial_faces);
      double q[3];
      if (edge_kernel_point(side, q)) {
        const int32_t id = append_vertex(q, 3, SourceWitness{});
        if (id < 0) {
          return fail(PHX_MC_CAPACITY_EXCEEDED, kPlcVertexBudget, kPlcEdge, e);
        }
        appended.push_back(id);
        alive_.push_back(1);
        vertex_tet_.push_back(-1);
        triangulation_->rebind(points_.data(), vertex_count());
        side.vertices.push_back(id);
        if (gift_wrap_greedy(side)) {
          break;
        }
        discard(appended);
        appended.clear();
      }
      // Enlarged reconnection cavities cost superlinear exact contact work.
      // A subdividable segment admitted by the original source owner takes the
      // conforming split, and a segment between coplanar input triangles is
      // recovered with their planar facet component (recover_segments).
      // Otherwise the segment may be recoverable only after the surrounding
      // tetrahedra are reconnected as well.  Grow across unconstrained facets
      // only; every prior PLC facet stays an obstacle in the enlarged cavity.
      double split_at[3];
      SourceWitness split_witness;
      int32_t split_status = PHX_MC_INVALID_INPUT;
      if (in_.policy == BoundaryPolicy::kConforming) {
        const PlcFailure previous_failure = out_.failure;
        const double previous_deviation = out_.source_refusal[0];
        const double previous_bound = out_.source_refusal[1];
        split_status = construct_split(e, u, v, split_at, split_witness);
        if (split_status == PHX_MC_CAPACITY_EXCEEDED) {
          return split_status;
        }
        // This is an alternative-construction probe, not a failed recovery:
        // an unsplittable edge may still be recovered by cavity reconnection.
        out_.failure = previous_failure;
        out_.source_refusal[0] = previous_deviation;
        out_.source_refusal[1] = previous_bound;
      }
      if (split_status == PHX_MC_OK ||
          coplanar_incidence(e)) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, e);
      }
      auto one_side = [](int32_t) { return 1; };
      if (!grow_cavity(cavity, one_side)) {
        if (!work_exhausted_ && native_execution_status(PHX_MC_OK) != PHX_MC_OK) {
          return fail(native_execution_status(PHX_MC_CAPACITY_EXCEEDED),
                      kPlcTetrahedronBudget, kPlcEdge, e);
        }
        return work_exhausted_ ? work_failure()
                              : fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment,
                                     kPlcEdge, e);
      }
    }
    bool recovered = false;
    for (const auto& tet : side.tets) {
      recovered = recovered ||
                  (std::find(tet.begin(), tet.end(), u) != tet.end() &&
                   std::find(tet.begin(), tet.end(), v) != tet.end());
    }
    if (!recovered) {
      discard(appended);
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, e);
    }
    CavityEdit edit(complex);
    edit.begin(EditLimits{std::numeric_limits<std::size_t>::max(),
                          std::numeric_limits<std::size_t>::max(), in_.max_tetrahedra});
    for (int32_t t : cavity) {
      if (edit.remove(t) != EditStatus::kOk) {
        discard(appended);
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    for (const auto& tet : side.tets) {
      int32_t constraints[4];
      for (int k = 0; k < 4; ++k) {
        Triple face{};
        oriented_facet(tet.data(), k, face.data());
        const auto label = labels.find(sorted_triple(face.data()));
        constraints[k] = label == labels.end() ? kNoConstraint : label->second;
      }
      if (edit.add(tet.data(), constraints, kNoRegion) != EditStatus::kOk) {
        discard(appended);
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    EditStatus status = edit.validate([&](const int32_t* tet) {
      return orient3d(point(tet[0]), point(tet[1]), point(tet[2]), point(tet[3])) > 0;
    });
    if (status == EditStatus::kOk) {
      status = edit.commit();
    }
    if (status != EditStatus::kOk) {
      edit.rollback();
      discard(appended);
      if (status == EditStatus::kCellLimit || status == EditStatus::kSlotLimit) {
        return fail(PHX_MC_CAPACITY_EXCEEDED, kPlcTetrahedronBudget, kPlcEdge, e);
      }
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, e);
    }
    reconnected_ = true;
    for (int32_t t : edit.created()) {
      for (int32_t w : complex.tets[static_cast<std::size_t>(t)].v) {
        vertex_tet_[static_cast<std::size_t>(w)] = t;
      }
    }
    ++out_.counters[kPlcCavities];
    out_.counters[kPlcInteriorSteiner] += static_cast<int64_t>(appended.size());
    out_.counters[kPlcLargestCavity] =
        std::max(out_.counters[kPlcLargestCavity], static_cast<int64_t>(cavity.size()));
    peak_bytes_ = std::max(peak_bytes_, triangulation_->retained_bytes() + edit.retained_bytes());
    recovery_edge_ = {-1, -1};
    return PHX_MC_OK;
  }

  // Recovers one connected set of missing subfacets of a planar group.  Its
  // boundary edges are mesh edges (PLC subsegments or edges of present
  // subfacets), so no tetrahedron's interior meets that boundary: the cavity
  // (tetrahedra whose interior meets the component's relative interior) meets
  // the group plane only inside the component, every cavity boundary facet
  // lies in one closed half-space, and the component splits the cavity into
  // two closed sides.  A cavity tetrahedron that crosses the plane has a plane
  // section whose vertices are edge crossings strictly inside the component
  // (in a subfacet interior or on an internal subfacet edge); one on a single
  // side has a facet lying in the component.
  int32_t recover_component(std::span<const int32_t> component, bool& failed) {
    TetrahedralComplex& complex = triangulation_->complex();
    const Triple& base = sub_[static_cast<std::size_t>(component[0])];
    const double* pa = point(base[0]);
    const double* pb = point(base[1]);
    const double* pc = point(base[2]);
    NativeVector<int32_t> members_vertices;
    NativeMap<uint64_t, int32_t> edge_count;
    NativeMap<Triple, int32_t> keys;
    for (int32_t s : component) {
      const Triple& t = sub_[static_cast<std::size_t>(s)];
      members_vertices.insert(members_vertices.end(), t.begin(), t.end());
      keys.emplace(sorted_triple(t.data()), s);
      for (int r = 0; r < 3; ++r) {
        ++edge_count[edge_key(t[r], t[(r + 1) % 3])];
      }
    }
    std::sort(members_vertices.begin(), members_vertices.end());
    members_vertices.erase(std::unique(members_vertices.begin(), members_vertices.end()),
                           members_vertices.end());
    const auto member = [&](int32_t v) {
      return std::binary_search(members_vertices.begin(), members_vertices.end(), v);
    };
    NativeUnorderedMap<int32_t, int> sign;
    const auto side_of = [&](int32_t v) {
      const auto found = sign.find(v);
      if (found != sign.end()) {
        return found->second;
      }
      const int value = orient3d(pa, pb, pc, point(v));
      sign.emplace(v, value);
      return value;
    };
    const auto in_cavity = [&](int32_t t) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(t)];
      int signs[4];
      int zeros = 0;
      bool positive = false;
      bool negative = false;
      for (int k = 0; k < 4; ++k) {
        signs[k] = side_of(tet.v[k]);
        zeros += signs[k] == 0 ? 1 : 0;
        positive = positive || signs[k] > 0;
        negative = negative || signs[k] < 0;
      }
      Contact contact;
      if (positive && negative) {
        for (int i = 0; i < 4; ++i) {
          for (int j = i + 1; j < 4; ++j) {
            if (signs[i] * signs[j] >= 0) {
              continue;
            }
            const double* const segment[2] = {point(tet.v[i]), point(tet.v[j])};
            const int64_t segment_ids[2] = {tet.v[i], tet.v[j]};
            for (int32_t s : component) {
              const Triple& g = sub_[static_cast<std::size_t>(s)];
              const double* const triangle[3] = {point(g[0]), point(g[1]), point(g[2])};
              const int64_t triangle_ids[3] = {g[0], g[1], g[2]};
              ++out_.counters[kPlcContactTests];
              if (intersect_segment_triangle(segment, triangle, segment_ids, triangle_ids,
                                             contact) != PHX_MC_OK) {
                continue;
              }
              if (contact.kind == PHX_MC_CROSSING) {
                return true;
              }
              const int8_t feature = contact.points[0].second_feature;
              if (contact.kind == PHX_MC_TOUCHING && feature >= 3 && feature < 6 &&
                  contact.points[0].first_feature == kFeatureInterior) {
                const int opposite = feature - 3;
                if (edge_count[edge_key(g[(opposite + 1) % 3], g[(opposite + 2) % 3])] == 2) {
                  return true;
                }
              }
            }
          }
        }
        return false;
      }
      // A mesh edge lying in the plane through the component's interior
      // (the other diagonal of a planar quadrilateral) bounds tetrahedra that
      // meet the component in that edge only; its whole star is replaced.
      for (int i = 0; i < 4 && zeros >= 2; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          if (signs[i] != 0 || signs[j] != 0 || !member(tet.v[i]) || !member(tet.v[j])) {
            continue;
          }
          const double* const segment[2] = {point(tet.v[i]), point(tet.v[j])};
          const int64_t segment_ids[2] = {tet.v[i], tet.v[j]};
          for (int32_t s : component) {
            const Triple& g = sub_[static_cast<std::size_t>(s)];
            const double* const triangle[3] = {point(g[0]), point(g[1]), point(g[2])};
            const int64_t triangle_ids[3] = {g[0], g[1], g[2]};
            ++out_.counters[kPlcContactTests];
            if (intersect_segment_triangle(segment, triangle, segment_ids, triangle_ids,
                                           contact) == PHX_MC_OK &&
                contact.kind == PHX_MC_COPLANAR_OVERLAP) {
              return true;
            }
          }
        }
      }
      if (zeros != 3) {
        return false;
      }
      int32_t face[3];
      int r = 0;
      for (int k = 0; k < 4; ++k) {
        if (signs[k] == 0) {
          face[r++] = tet.v[k];
        }
      }
      if (!member(face[0]) || !member(face[1]) || !member(face[2])) {
        return false;
      }
      for (int32_t s : component) {
        const double* const first[3] = {point(face[0]), point(face[1]), point(face[2])};
        const Triple& g = sub_[static_cast<std::size_t>(s)];
        const double* const second[3] = {point(g[0]), point(g[1]), point(g[2])};
        const int64_t first_ids[3] = {face[0], face[1], face[2]};
        const int64_t second_ids[3] = {g[0], g[1], g[2]};
        ++out_.counters[kPlcContactTests];
        if (intersect_triangles(first, second, first_ids, second_ids, contact) == PHX_MC_OK &&
            (contact.kind == PHX_MC_COPLANAR_OVERLAP || contact.kind == PHX_MC_COINCIDENT)) {
          return true;
        }
      }
      return false;
    };
    // Breadth-first search from the stars of the component vertices,
    // expanding only through cavity tetrahedra.
    NativeUnorderedSet<int32_t> examined;
    NativeVector<int32_t> cavity;
    NativeVector<int32_t> frontier;
    for (int32_t v : members_vertices) {
      for (int32_t t : star(v)) {
        if (!is_ghost(complex.tets[static_cast<std::size_t>(t)]) && examined.insert(t).second) {
          frontier.push_back(t);
        }
      }
    }
    std::sort(frontier.begin(), frontier.end());
    while (!frontier.empty()) {
      const int32_t t = frontier.back();
      frontier.pop_back();
      if (!spend(4)) {
        return work_failure();
      }
      if (!in_cavity(t)) {
        continue;
      }
      if (!native_execution_cavity(cavity.size() + 1)) {
        return fail(native_execution_status(PHX_MC_CAPACITY_EXCEEDED),
                    kPlcTetrahedronBudget, kPlcFacet, sub_facet(component[0]));
      }
      cavity.push_back(t);
      out_.counters[kPlcLargestCavity] =
          std::max(out_.counters[kPlcLargestCavity], static_cast<int64_t>(cavity.size()));
      for (int32_t next : complex.tets[static_cast<std::size_t>(t)].n) {
        if (!is_ghost(complex.tets[static_cast<std::size_t>(next)]) &&
            examined.insert(next).second) {
          frontier.push_back(next);
        }
      }
    }
    std::sort(cavity.begin(), cavity.end());
    ++out_.counters[kPlcCavities];
    out_.counters[kPlcLargestCavity] =
        std::max(out_.counters[kPlcLargestCavity], static_cast<int64_t>(cavity.size()));
    // Cavity sides: boundary facets, constrained facets inside the cavity
    // (both orientations, as obstacles of their side) and the component.  A
    // boundary facet takes the side of its off-plane vertices, or of its
    // tetrahedron's apex when it lies in the plane outside the component.
    // When a side admits no fill (the facet's chosen triangulation disagrees
    // with a degenerate, e.g. coplanar, neighborhood outside the cavity) the
    // cavity grows by one layer of one-sided tetrahedra across unconstrained
    // facets and both sides are wrapped again.
    Side sides[2];
    NativeVector<std::pair<Triple, int32_t>> inner_constraints;
    NativeVector<int32_t> cavity_vertices;
    NativeVector<double> cones[2];
    for (;;) {
      sides[0] = Side{};
      sides[1] = Side{};
      cones[0].clear();
      cones[1].clear();
      inner_constraints.clear();
      cavity_vertices.clear();
      for (int32_t t : cavity) {
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(t)];
        cavity_vertices.insert(cavity_vertices.end(), tet.v, tet.v + 4);
        for (int k = 0; k < 4; ++k) {
          const int32_t next = tet.n[k];
          const bool inside = std::binary_search(cavity.begin(), cavity.end(), next);
          const int32_t constraint = complex.constraint(t, k);
          if (inside && next < t) {
            continue;
          }
          Triple face{};
          oriented_facet(tet.v, k, face.data());
          // An inner face in the group plane outside the component (an
          // enlarged cavity reaches both sides there) separates the sides:
          // each side keeps it, one orientation each, as a fixed boundary.
          const bool planar = inside && side_of(face[0]) == 0 && side_of(face[1]) == 0 &&
                              side_of(face[2]) == 0 &&
                              std::all_of(component.begin(), component.end(), [&](int32_t s) {
                                return triangle_contact_legal(face, sub_[static_cast<std::size_t>(s)]);
                              });
          if (inside && constraint == kNoConstraint && !planar) {
            continue;
          }
          int side = 0;
          for (int32_t v : face) {
            side = side != 0 ? side : side_of(v);
          }
          side = side != 0 ? side : side_of(tet.v[k]);
          if (side == 0) {
            return PHX_MC_INTERNAL_ERROR;
          }
          Side& target = sides[side > 0 ? 0 : 1];
          add_face(target, face, true);
          if (inside) {
            add_face(planar ? sides[side > 0 ? 1 : 0] : target,
                     Triple{face[0], face[2], face[1]}, true);
            if (constraint != kNoConstraint) {
              inner_constraints.push_back({sorted_triple(face.data()), constraint});
            }
          }
        }
      }
      for (int32_t s : component) {
        const Triple& g = sub_[static_cast<std::size_t>(s)];
        add_face(sides[0], g, true);
        add_face(sides[1], Triple{g[0], g[2], g[1]}, true);
      }
      std::sort(cavity_vertices.begin(), cavity_vertices.end());
      cavity_vertices.erase(std::unique(cavity_vertices.begin(), cavity_vertices.end()),
                            cavity_vertices.end());
      for (int32_t v : cavity_vertices) {
        const int value = side_of(v);
        if (value >= 0) {
          sides[0].vertices.push_back(v);
        }
        if (value <= 0) {
          sides[1].vertices.push_back(v);
        }
      }
      // A side whose greedy fill is trapped by unconstrained cavity boundary
      // faces first takes the tetrahedra behind those faces (TetGen-style
      // local enlargement), which costs no exhaustive search.  A side that
      // gift wrapping cannot fill (a polyhedron that needs a Steiner point,
      // like Schönhardt's) is filled by the cone from an interior Steiner
      // point strictly inside every boundary facet's half-space -- a verified
      // point of the side's kernel.  The fixed policy permits it: the point
      // is interior, not on the boundary.
      bool filled = true;
      bool enlarged = false;
      for (int s = 0; s < 2 && filled; ++s) {
        const std::size_t initial_faces = sides[s].faces.size();
        if (gift_wrap_greedy(sides[s])) {
          continue;
        }
        reset_front(sides[s], initial_faces);
        if (work_exhausted_) {
          return work_failure();
        }
        if (grow_cavity(cavity, side_of, true)) {
          enlarged = true;
          filled = false;
          break;
        }
        if (work_exhausted_) {
          return work_failure();
        }
        if (native_execution_status(PHX_MC_OK) != PHX_MC_OK) {
          return fail(native_execution_status(PHX_MC_CAPACITY_EXCEEDED),
                      kPlcTetrahedronBudget, kPlcFacet, sub_facet(component[0]));
        }
        if (gift_wrap_search(sides[s])) {
          continue;
        }
        filled = kernel_point(sides[s], component, cones[s]);
      }
      if (filled) {
        break;
      }
      if (work_exhausted_ || work() > in_.work_limit) {
        return work_failure();
      }
      if (!enlarged && !grow_cavity(cavity, side_of)) {
        if (!work_exhausted_ && native_execution_status(PHX_MC_OK) != PHX_MC_OK) {
          return fail(native_execution_status(PHX_MC_CAPACITY_EXCEEDED),
                      kPlcTetrahedronBudget, kPlcFacet, sub_facet(component[0]));
        }
        if (work_exhausted_) {
          return work_failure();
        }
        failed = true;
        return PHX_MC_OK;
      }
      out_.counters[kPlcLargestCavity] =
          std::max(out_.counters[kPlcLargestCavity], static_cast<int64_t>(cavity.size()));
    }
    // One transactional edit replaces the cavity by both fills.
    NativeMap<Triple, int32_t> labels;
    for (const auto& [key, id] : inner_constraints) {
      labels.emplace(key, id);
    }
    for (const auto& [key, s] : keys) {
      labels.emplace(key, sub_facet(s));
    }
    NativeVector<int32_t> appended;
    for (int s = 0; s < 2; ++s) {
      if (cones[s].empty()) {
        continue;
      }
      const int32_t id = append_vertex(cones[s].data(), 3, SourceWitness{});
      if (id < 0) {
        discard(appended);
        return fail(PHX_MC_CAPACITY_EXCEEDED, kPlcVertexBudget, kPlcFacet,
                    sub_facet(component[0]));
      }
      appended.push_back(id);
      alive_.push_back(1);
      vertex_tet_.push_back(-1);
      triangulation_->rebind(points_.data(), vertex_count());
      sides[s].vertices.push_back(id);
      // Wrapping with the kernel candidate, rather than blindly coning the
      // boundary, preserves interior PLC points and curves as well.
      if (!gift_wrap(sides[s])) {
        discard(appended);
        if (work_exhausted_) {
          return work_failure();
        }
        failed = true;
        return PHX_MC_OK;
      }
      cavity_vertices.push_back(id);
    }
    CavityEdit edit(complex);
    edit.begin(EditLimits{std::numeric_limits<std::size_t>::max(),
                          std::numeric_limits<std::size_t>::max(), in_.max_tetrahedra});
    for (int32_t t : cavity) {
      if (edit.remove(t) != EditStatus::kOk) {
        discard(appended);
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    for (const Side& side : sides) {
      for (const auto& tet : side.tets) {
        int32_t constraints[4];
        for (int k = 0; k < 4; ++k) {
          int32_t face[3];
          oriented_facet(tet.data(), k, face);
          const auto label = labels.find(sorted_triple(face));
          constraints[k] = label == labels.end() ? kNoConstraint : label->second;
        }
        if (edit.add(tet.data(), constraints, kNoRegion) != EditStatus::kOk) {
          discard(appended);
          return PHX_MC_INTERNAL_ERROR;
        }
      }
    }
    EditStatus status = edit.validate([&](const int32_t* v) {
      return orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) > 0;
    });
    if (status == EditStatus::kOk) {
      status = edit.commit();
    }
    if (status != EditStatus::kOk) {
      edit.rollback();
      discard(appended);
      if (status == EditStatus::kCellLimit || status == EditStatus::kSlotLimit) {
        return fail(PHX_MC_CAPACITY_EXCEEDED, kPlcTetrahedronBudget, kPlcNoEntity, -1);
      }
      failed = true;
      return PHX_MC_OK;
    }
    reconnected_ = true;
    peak_bytes_ = std::max(peak_bytes_, triangulation_->retained_bytes() + edit.retained_bytes());
    for (int32_t t : edit.created()) {
      if (static_cast<std::size_t>(t) >= star_mark_.size()) {
        star_mark_.resize(complex.tets.size(), 0U);
      }
      for (int32_t v : complex.tets[static_cast<std::size_t>(t)].v) {
        if (v >= 0) {
          vertex_tet_[static_cast<std::size_t>(v)] = t;
        }
      }
    }
    for (int32_t v : cavity_vertices) {
      const int32_t t = vertex_tet_[static_cast<std::size_t>(v)];
      if (!complex.live(t) || vertex_slot(complex.tets[static_cast<std::size_t>(t)], v) < 0) {
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    for (std::size_t k = 0; k < keys.size(); ++k) {
      triangulation_->count_constrained_facet();
    }
    out_.counters[kPlcInteriorSteiner] += static_cast<int64_t>(appended.size());
    return PHX_MC_OK;
  }

  // ------------------------------------------------------------ classification
  int32_t classify() {
    TetrahedralComplex& complex = triangulation_->complex();
    complex.enable_labels();
    complex.clear_regions();
    struct Seed {
      int32_t tet;
      int32_t label;
    };
    NativeVector<Seed> seeds;
    const auto label_of = [](int32_t region) { return region < 0 ? kVoidRegion : region; };
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t facet = complex.constraint(static_cast<int32_t>(t), k);
        if (facet == kNoConstraint) {
          continue;
        }
        int32_t face[3];
        oriented_facet(tet.v, k, face);
        const auto sub = sub_index_.find(sorted_triple(face));
        if (sub == sub_index_.end()) {
          return PHX_MC_INTERNAL_ERROR;
        }
        const bool positive =
            oriented_key(face) == oriented_key(sub_[static_cast<std::size_t>(sub->second)].data());
        seeds.push_back({static_cast<int32_t>(t),
                         label_of(in_.facet_regions[2 * facet + (positive ? 0 : 1)])});
      }
      for (int32_t v : tet.v) {
        if (v >= n_ && v < n_ + kCorners) {
          seeds.push_back({static_cast<int32_t>(t), kVoidRegion});
          break;
        }
      }
    }
    for (int64_t s = 0; s < in_.seed_count; ++s) {
      native_execution_charge(0, 1);
      const int32_t t = seed_tet(in_.seeds + 3 * s);
      if (t < 0) {
        return fail(PHX_MC_INVALID_INPUT, kPlcInvalidSeed, kPlcSeed, s);
      }
      seeds.push_back({t, label_of(in_.seed_regions[s])});
    }
    NativeVector<int32_t> stack;
    for (const Seed& seed : seeds) {
      if (!spend(1)) {
        return work_failure();
      }
      const int32_t existing = complex.region(seed.tet);
      if (existing != kNoRegion) {
        if (existing != seed.label) {
          return conflict(existing, seed.label);
        }
        continue;
      }
      complex.set_region(seed.tet, seed.label);
      stack.push_back(seed.tet);
      while (!stack.empty()) {
        const int32_t t = stack.back();
        stack.pop_back();
        if (!spend(4)) {
          return work_failure();
        }
        for (int k = 0; k < 4; ++k) {
          const int32_t next = complex.tets[static_cast<std::size_t>(t)].n[k];
          if (complex.constraint(t, k) != kNoConstraint ||
              is_ghost(complex.tets[static_cast<std::size_t>(next)])) {
            continue;
          }
          const int32_t label = complex.region(next);
          if (label == kNoRegion) {
            complex.set_region(next, seed.label);
            stack.push_back(next);
          } else if (label != seed.label) {
            return conflict(label, seed.label);
          }
        }
      }
    }
    return check_embedded();
  }

  int32_t conflict(int32_t first, int32_t second) {
    if (first == kVoidRegion || second == kVoidRegion) {
      return fail(PHX_MC_INVALID_INPUT, kPlcRegionLeak, kPlcRegion,
                  first == kVoidRegion ? second : first);
    }
    return fail(PHX_MC_INVALID_INPUT, kPlcInconsistentRegions, kPlcRegion,
                std::min(first, second), kPlcRegion, std::max(first, second));
  }

  // The finite tetrahedron containing a seed, or -1 when the seed lies on a
  // constrained facet or outside the enclosing corners.
  int32_t seed_tet(const double* p) {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) {
        continue;
      }
      int signs[4];
      bool inside = true;
      for (int k = 0; k < 4 && inside; ++k) {
        signs[k] = triangulation_->orient_with(tet, k, p);
        inside = signs[k] >= 0;
      }
      if (!inside) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        if (signs[k] == 0 && complex.constraint(static_cast<int32_t>(t), k) != kNoConstraint) {
          return -1;
        }
      }
      return static_cast<int32_t>(t);
    }
    return -1;
  }

  bool domain_vertex(int32_t v) {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t t : star(v)) {
      if (!is_ghost(complex.tets[static_cast<std::size_t>(t)]) && complex.region(t) >= 0) {
        return true;
      }
    }
    return false;
  }

  // Explicit segments and free points must lie in the closure of a region.
  int32_t check_embedded() {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t e = 0; e < static_cast<int32_t>(in_.segment_count); ++e) {
      const NativeVector<int32_t> vertices = chain(e);
      for (std::size_t k = 0; k + 1 < vertices.size(); ++k) {
        bool inside = false;
        for (int32_t t : star(vertices[k])) {
          inside = inside ||
                   (vertex_slot(complex.tets[static_cast<std::size_t>(t)], vertices[k + 1]) >= 0 &&
                    !is_ghost(complex.tets[static_cast<std::size_t>(t)]) &&
                    complex.region(t) >= 0);
        }
        if (!inside) {
          return fail(PHX_MC_INVALID_INPUT, kPlcOutsideDomain, kPlcEdge, e);
        }
      }
    }
    for (int32_t v : free_points_) {
      if (!domain_vertex(v)) {
        return fail(PHX_MC_INVALID_INPUT, kPlcOutsideDomain, kPlcPoint, v);
      }
    }
    return PHX_MC_OK;
  }

  // ------------------------------------------------------------ publication
  int32_t renumber(int32_t v) const { return v < n_ ? v : v - kCorners; }

  void publish() {
    const TetrahedralComplex& complex = triangulation_->complex();
    out_.points.assign(points_.begin(), points_.begin() + 3 * static_cast<int64_t>(n_));
    out_.points.insert(out_.points.end(), points_.begin() + 3 * static_cast<int64_t>(n_ + kCorners),
                       points_.end());
    out_.vertex_dimension.assign(vertex_dimension_.begin(), vertex_dimension_.begin() + n_);
    out_.vertex_dimension.insert(out_.vertex_dimension.end(),
                                 vertex_dimension_.begin() + n_ + kCorners,
                                 vertex_dimension_.end());
    out_.protection.assign(out_.points.size() / 3, 0.0);
    std::copy(protection_.begin(), protection_.end(), out_.protection.begin());
    NativeMap<Triple, int32_t> faces;
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet) ||
          complex.region(static_cast<int32_t>(t)) < 0) {
        continue;
      }
      for (int32_t v : tet.v) {
        out_.tets.push_back(renumber(v));
      }
      out_.tet_regions.push_back(complex.region(static_cast<int32_t>(t)));
      for (int k = 0; k < 4; ++k) {
        const int32_t facet = complex.constraint(static_cast<int32_t>(t), k);
        if (facet == kNoConstraint) {
          continue;
        }
        int32_t face[3];
        oriented_facet(tet.v, k, face);
        const Triple key = sorted_triple(face);
        if (faces.count(key) == 0) {
          faces.emplace(key, facet);
          const Triple& oriented =
              sub_[static_cast<std::size_t>(sub_index_.at(key))];
          for (int32_t v : oriented) {
            out_.faces.push_back(renumber(v));
          }
          out_.face_sources.push_back(facet);
        }
      }
    }
    for (int32_t e = 0; e < static_cast<int32_t>(edges_.size() / 2); ++e) {
      const NativeVector<int32_t> vertices = chain(e);
      for (std::size_t k = 0; k + 1 < vertices.size(); ++k) {
        out_.segments.push_back(renumber(vertices[k]));
        out_.segments.push_back(renumber(vertices[k + 1]));
        out_.segment_sources.push_back(e);
      }
    }
    out_.plc_edges = edges_;
    out_.input_triangles = triangles_;
    out_.input_polygons = tri_polygon_;
    out_.witnesses.assign(witnesses_.begin(), witnesses_.begin() + n_);
    out_.witnesses.insert(out_.witnesses.end(), witnesses_.begin() + n_ + kCorners,
                          witnesses_.end());
  }

  const PlcInput& in_;
  PlcRecovery& out_;
  int32_t n_ = 0;
  int64_t work_ = 0;
  std::array<int32_t, 2> recovery_edge_{-1, -1};
  bool work_exhausted_ = false;
  // Whether a cavity edit replaced Delaunay tetrahedra of this round.
  bool reconnected_ = false;
  NativeVector<std::array<int32_t, 2>> protected_segments_;
  // Subsegments recovered with their planar facet component (edge keys).
  NativeUnorderedSet<uint64_t> deferred_segments_;
  // Obstacle faces that refused the trapped face of the last greedy wrap.
  NativeVector<Triple> blockers_;
  std::size_t peak_bytes_ = 0;
  // Input triangulation of the polygons.
  NativeVector<int32_t> triangles_;
  NativeVector<int32_t> tri_polygon_;
  NativeVector<int32_t> tri_facet_;
  NativeVector<int32_t> tri_group_;
  NativeVector<int32_t> group_facet_;
  NativeVector<std::pair<uint64_t, int32_t>> edge_uses_;
  NativeVector<int32_t> free_points_;
  // PLC edges and their Steiner points.
  NativeVector<int32_t> edges_;
  NativeUnorderedMap<uint64_t, int32_t> edge_ids_;
  NativeVector<NativeVector<int32_t>> edge_points_;
  NativeVector<NativeVector<int32_t>> tri_interior_;
  NativeVector<double> protection_;
  // Construction state.
  NativeVector<double> points_;
  NativeVector<int8_t> vertex_dimension_;
  NativeVector<SourceWitness> witnesses_;  // per point, corners included
  NativeUniquePtr<Triangulation3D> triangulation_;
  NativeVector<char> alive_;
  NativeVector<int32_t> vertex_tet_;
  NativeVector<int32_t> star_;
  NativeVector<std::uint32_t> star_mark_;
  std::uint32_t star_stamp_ = 0;
  // Subfacets of the current round.
  NativeVector<Triple> sub_;
  NativeVector<int32_t> sub_group_;
  NativeVector<int32_t> sub_origin_;
  NativeMap<Triple, int32_t> sub_index_;
};

}  // namespace

static int32_t execute_plc(const PlcInput& input, PlcRecovery& result, bool source_only) {
  std::optional<MemoryBudgetWindow> budget;
  MemoryOwner owner = scratch_memory_owner();
  if (owner || input.max_scratch_bytes != std::numeric_limits<std::size_t>::max()) {
    budget.emplace(input.max_scratch_bytes, owner);
    owner = budget->owner();
  }
  MemoryScope memory(owner);
  result = PlcRecovery(owner);
  int32_t status;
  try {
    status = Recoverer(input, result).run(source_only);
  } catch (const std::bad_alloc&) {
    result.failure = PlcFailure{kPlcScratchByteBudget, kPlcNoEntity, -1, kPlcNoEntity, -1};
    status = PHX_MC_CAPACITY_EXCEEDED;
  }
  if (budget) {
    result.memory_measured = true;
    result.memory_snapshot = budget->evidence();
    result.counters[kPlcPeakBytes] = static_cast<int64_t>(result.memory_snapshot[2]);
  }
  return status;
}

int32_t recover_plc(const PlcInput& input, PlcRecovery& result) {
  return execute_plc(input, result, false);
}

int32_t prepare_plc_source(const PlcInput& input, PlcRecovery& result) {
  return execute_plc(input, result, true);
}

}  // namespace phx::mc

struct phx_mc_plc3d : phx::mc::NativeAllocatedObject {
  phx::mc::PlcRecovery recovery;
};

namespace {

int32_t check_arguments(const phx::mc::PlcInput& in) {
  using phx::mc::addressable;
  if (in.point_count < 0 || in.point_count > phx::mc::kMaxMeshPoints - 8 ||
      in.polygon_count < 0 || in.facet_count < 0 || in.segment_count < 0 ||
      in.seed_count < 0 || !addressable(in.point_count, 3, sizeof(double)) ||
      !addressable(in.facet_count, 2, sizeof(int32_t)) ||
      !addressable(in.segment_count, 2, sizeof(int32_t)) ||
      !addressable(in.seed_count, 3, sizeof(double)) || in.max_vertices < in.point_count ||
      in.max_vertices > phx::mc::kMaxMeshPoints - 8 || in.max_tetrahedra < 1 ||
      in.work_limit < 0 || (in.point_count > 0 && in.points == nullptr) ||
      in.polygon_offsets == nullptr || (in.polygon_count > 0 && in.polygon_facets == nullptr) ||
      (in.facet_count > 0 && in.facet_regions == nullptr) ||
      (in.segment_count > 0 && in.segments == nullptr) ||
      (in.seed_count > 0 && (in.seeds == nullptr || in.seed_regions == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (in.polygon_offsets[0] != 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t p = 0; p < in.polygon_count; ++p) {
    phx::mc::native_execution_charge(0);
    if (in.polygon_offsets[p + 1] < in.polygon_offsets[p] || in.polygon_facets[p] < 0 ||
        in.polygon_facets[p] >= in.facet_count) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  const int64_t loops = in.polygon_offsets[in.polygon_count];
  if (loops > 0 && in.polygon_vertices == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t k = 0; k < loops; ++k) {
    phx::mc::native_execution_charge(0);
    if (in.polygon_vertices[k] < 0 || in.polygon_vertices[k] >= in.point_count) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  for (int64_t k = 0; k < 2 * in.segment_count; ++k) {
    phx::mc::native_execution_charge(0);
    if (in.segments[k] < 0 || in.segments[k] >= in.point_count) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  for (int64_t k = 0; k < 2 * in.facet_count; ++k) {
    phx::mc::native_execution_charge(0);
    if (in.facet_regions[k] < -1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  for (int64_t k = 0; k < in.seed_count; ++k) {
    phx::mc::native_execution_charge(0);
    if (in.seed_regions[k] < -1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  return phx::mc::validate_points(in.seeds, in.seed_count, 3, nullptr);
}

}  // namespace

extern "C" {

int32_t phx_mc_plc3d_recover(int64_t point_count, const double* points, int64_t polygon_count,
                             const int64_t* polygon_offsets, const int32_t* polygon_vertices,
                             const int32_t* polygon_facets, int64_t facet_count,
                             const int32_t* facet_regions, int64_t segment_count,
                             const int32_t* segments, int64_t seed_count, const double* seeds,
                             const int32_t* seed_regions, int32_t boundary_policy,
                             const double* facet_tolerances, const double* segment_tolerances,
                             int64_t max_vertices, int64_t max_tetrahedra, int64_t work_limit,
                             uint64_t maximum_scratch_bytes, int32_t measure_phases,
                             phx_mc_plc3d** result) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *result = nullptr;
    if (measure_phases != 0 && measure_phases != 1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (maximum_scratch_bytes > std::numeric_limits<std::size_t>::max()) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (boundary_policy != PHX_MC_PLC3D_FIXED && boundary_policy != PHX_MC_PLC3D_CONFORMING) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::PlcInput input;
    input.point_count = point_count;
    input.points = points;
    input.polygon_count = polygon_count;
    input.polygon_offsets = polygon_offsets;
    input.polygon_vertices = polygon_vertices;
    input.polygon_facets = polygon_facets;
    input.facet_count = facet_count;
    input.facet_regions = facet_regions;
    input.segment_count = segment_count;
    input.segments = segments;
    input.seed_count = seed_count;
    input.seeds = seeds;
    input.seed_regions = seed_regions;
    input.policy = static_cast<phx::mc::BoundaryPolicy>(boundary_policy);
    input.facet_tolerances = facet_tolerances;
    input.segment_tolerances = segment_tolerances;
    input.max_vertices = max_vertices;
    input.max_tetrahedra = max_tetrahedra;
    input.work_limit = work_limit;
    input.measure_phases = measure_phases != 0;
    input.max_scratch_bytes = static_cast<std::size_t>(maximum_scratch_bytes);
    int32_t status = check_arguments(input);
    if (status != PHX_MC_OK) {
      return status;
    }
    // The fixed-size diagnostic carrier is control-plane state, not a
    // numerical scratch buffer. It must survive any numerical allocation
    // refusal, including a zero-byte cap, without spending that cap.
    phx::mc::NativeUniquePtr<phx_mc_plc3d> handle;
    {
      phx::mc::MemoryScope diagnostics(phx::mc::MemoryOwner{});
      handle = phx::mc::make_native_unique<phx_mc_plc3d>();
    }
    status = phx::mc::recover_plc(input, handle->recovery);
    if (status == PHX_MC_OK || handle->recovery.failure.reason != phx::mc::kPlcNoFailure ||
        handle->recovery.measurement_enabled) {
      *result = handle.release();
    }
    return status;
  });
}

int32_t phx_mc_plc3d_source_constraints(
    int64_t point_count, const double* points, int64_t polygon_count,
    const int64_t* polygon_offsets, const int32_t* polygon_vertices,
    const int32_t* polygon_facets, int64_t facet_count, const int32_t* facet_regions,
    int64_t segment_count, const int32_t* segments, int64_t work_limit,
    uint64_t maximum_scratch_bytes, int32_t measure_phases, phx_mc_plc3d** result) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *result = nullptr;
    if ((measure_phases != 0 && measure_phases != 1) ||
        maximum_scratch_bytes > std::numeric_limits<std::size_t>::max()) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::PlcInput input;
    input.point_count = point_count;
    input.points = points;
    input.polygon_count = polygon_count;
    input.polygon_offsets = polygon_offsets;
    input.polygon_vertices = polygon_vertices;
    input.polygon_facets = polygon_facets;
    input.facet_count = facet_count;
    input.facet_regions = facet_regions;
    input.segment_count = segment_count;
    input.segments = segments;
    input.max_vertices = point_count;
    input.max_tetrahedra = 1;  // Not consumed by source-only preparation.
    input.work_limit = work_limit;
    input.max_scratch_bytes = static_cast<std::size_t>(maximum_scratch_bytes);
    input.measure_phases = measure_phases != 0;
    const int32_t valid = check_arguments(input);
    if (valid != PHX_MC_OK) {
      return valid;
    }
    phx::mc::NativeUniquePtr<phx_mc_plc3d> handle;
    {
      phx::mc::MemoryScope diagnostics(phx::mc::MemoryOwner{});
      handle = phx::mc::make_native_unique<phx_mc_plc3d>();
    }
    const int32_t status = phx::mc::prepare_plc_source(input, handle->recovery);
    if (status == PHX_MC_OK || handle->recovery.failure.reason != phx::mc::kPlcNoFailure ||
        handle->recovery.measurement_enabled) {
      *result = handle.release();
    }
    return status;
  });
}

void phx_mc_plc3d_sizes(const phx_mc_plc3d* result, int64_t* sizes) {
  const phx::mc::PlcRecovery& r = result->recovery;
  sizes[0] = static_cast<int64_t>(r.points.size() / 3);
  sizes[1] = static_cast<int64_t>(r.tet_regions.size());
  sizes[2] = static_cast<int64_t>(r.face_sources.size());
  sizes[3] = static_cast<int64_t>(r.segment_sources.size());
  sizes[4] = static_cast<int64_t>(r.plc_edges.size() / 2);
  sizes[5] = static_cast<int64_t>(r.input_polygons.size());
}

void phx_mc_plc3d_counters(const phx_mc_plc3d* result, int64_t* counters) {
  std::copy_n(result->recovery.counters, phx::mc::kPlcCounterCount, counters);
}

extern "C" int32_t phx_mc_plc3d_phase_times(const phx_mc_plc3d* result, int64_t* nanoseconds,
                                int64_t* invocations, int32_t* enabled) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || nanoseconds == nullptr || invocations == nullptr || enabled == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::PlcRecovery& recovery = result->recovery;
    std::copy_n(recovery.phase_nanoseconds, phx::mc::kPlcPhaseCount, nanoseconds);
    std::copy_n(recovery.phase_invocations, phx::mc::kPlcPhaseCount, invocations);
    *enabled = recovery.measurement_enabled ? 1 : 0;
    return PHX_MC_OK;
  });
}

void phx_mc_plc3d_failure(const phx_mc_plc3d* result, int64_t* failure) {
  const phx::mc::PlcFailure& f = result->recovery.failure;
  failure[0] = f.reason;
  failure[1] = f.first_kind;
  failure[2] = f.first;
  failure[3] = f.second_kind;
  failure[4] = f.second;
}

void phx_mc_plc3d_export(const phx_mc_plc3d* result, double* points, int32_t* tets,
                         int32_t* tet_regions, int32_t* faces, int32_t* face_sources,
                         int32_t* segments, int32_t* segment_sources, double* protection,
                         int32_t* plc_edges, int8_t* vertex_dimension, int32_t* input_triangles,
                         int32_t* input_polygons) {
  const phx::mc::PlcRecovery& r = result->recovery;
  std::copy(r.points.begin(), r.points.end(), points);
  std::copy(r.tets.begin(), r.tets.end(), tets);
  std::copy(r.tet_regions.begin(), r.tet_regions.end(), tet_regions);
  std::copy(r.faces.begin(), r.faces.end(), faces);
  std::copy(r.face_sources.begin(), r.face_sources.end(), face_sources);
  std::copy(r.segments.begin(), r.segments.end(), segments);
  std::copy(r.segment_sources.begin(), r.segment_sources.end(), segment_sources);
  std::copy(r.protection.begin(), r.protection.end(), protection);
  std::copy(r.plc_edges.begin(), r.plc_edges.end(), plc_edges);
  std::copy(r.vertex_dimension.begin(), r.vertex_dimension.end(), vertex_dimension);
  std::copy(r.input_triangles.begin(), r.input_triangles.end(), input_triangles);
  std::copy(r.input_polygons.begin(), r.input_polygons.end(), input_polygons);
}

void phx_mc_plc3d_source_witnesses(const phx_mc_plc3d* result, int8_t* strata,
                                   int32_t* entities, double* parameters, double* deviations,
                                   double* refusal) {
  const phx::mc::PlcRecovery& r = result->recovery;
  for (std::size_t v = 0; v < r.witnesses.size(); ++v) {
    strata[v] = static_cast<int8_t>(r.witnesses[v].stratum);
    entities[v] = r.witnesses[v].entity;
    parameters[2 * v] = r.witnesses[v].parameters[0];
    parameters[2 * v + 1] = r.witnesses[v].parameters[1];
    deviations[v] = r.witnesses[v].deviation;
  }
  refusal[0] = r.source_refusal[0];
  refusal[1] = r.source_refusal[1];
}

int32_t phx_mc_plc3d_memory_evidence(const phx_mc_plc3d* result, uint64_t* evidence,
                                    int32_t* enabled) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || evidence == nullptr || enabled == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *enabled = result->recovery.memory_measured ? 1 : 0;
    if (result->recovery.memory_measured) {
      const auto& snapshot = result->recovery.memory_snapshot;
      std::copy(snapshot.begin(), snapshot.end(), evidence);
    }
    return PHX_MC_OK;
  });
}

void phx_mc_plc3d_free(phx_mc_plc3d* result) { phx::mc::destroy_native_object(result); }

}  // extern "C"
