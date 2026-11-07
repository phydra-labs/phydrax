//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact PLC edge-cavity broad-phase regression: adjacency candidates must
// match an exhaustive exact segment/facet reference for crossings, endpoint
// touches, and coplanar overlap, and remain correct after an accepted edit.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <span>
#include <map>
#include <utility>
#include <vector>
#include <set>

#include "cavity.hpp"
#include "check.hpp"
#include "fp_hazards.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"
#include "plc3d.hpp"
#include "predicates.hpp"
#include "triangulation3d.hpp"

namespace {
std::atomic<bool> refuse_allocations{false};
std::atomic<std::size_t> refused_allocation_attempts{0};
}

void* operator new(std::size_t size) {
  if (refuse_allocations.load(std::memory_order_relaxed)) {
    refused_allocation_attempts.fetch_add(1, std::memory_order_relaxed);
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {

using namespace phx::mc;

bool legal_contact(int8_t kind) {
  return kind == PHX_MC_DISJOINT || kind == PHX_MC_SHARED_VERTEX ||
         kind == PHX_MC_SHARED_EDGE;
}

std::array<int32_t, 3> sorted_face(const int32_t* vertices) {
  std::array<int32_t, 3> face{vertices[0], vertices[1], vertices[2]};
  std::sort(face.begin(), face.end());
  return face;
}

TetrahedralComplex bipyramid(const double* points) {
  TetrahedralComplex complex;
  complex.tets.push_back(Tetrahedron{{0, 1, 2, 3}, {-1, -1, -1, -1}, 0, 0});
  complex.tets.push_back(Tetrahedron{{0, 2, 1, 4}, {-1, -1, -1, -1}, 0, 0});
  PHX_CHECK(orient3d(points, points + 3, points + 6, points + 9) > 0);
  PHX_CHECK(orient3d(points, points + 6, points + 3, points + 12) > 0);
  complex.tets[0].n[3] = 1;
  complex.tets[1].n[3] = 0;
  complex.finite_count = 2;

  std::vector<int32_t> ghosts;
  for (int32_t t = 0; t < 2; ++t) {
    for (int slot = 0; slot < 4; ++slot) {
      if (complex.tets[static_cast<std::size_t>(t)].n[slot] >= 0) continue;
      int32_t face[3];
      oriented_facet(complex.tets[static_cast<std::size_t>(t)].v, slot, face);
      const int32_t ghost_id = static_cast<int32_t>(complex.tets.size());
      complex.tets.push_back(Tetrahedron{{kGhostVertex, face[0], face[1], face[2]},
                                         {t, -1, -1, -1}, 0, 0});
      complex.tets[static_cast<std::size_t>(t)].n[slot] = ghost_id;
      ghosts.push_back(ghost_id);
    }
  }

  std::map<std::array<int32_t, 3>, std::pair<int32_t, int>> boundary_edges;
  for (int32_t id : ghosts) {
    Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
    for (int slot = 1; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(tet.v, slot, face);
      const auto key = sorted_face(face);
      const auto found = boundary_edges.find(key);
      if (found == boundary_edges.end()) {
        boundary_edges.emplace(key, std::pair{id, slot});
      } else {
        const auto [other, back] = found->second;
        tet.n[slot] = other;
        complex.tets[static_cast<std::size_t>(other)].n[back] = id;
        boundary_edges.erase(found);
      }
    }
  }
  PHX_CHECK(boundary_edges.empty());
  PHX_CHECK(complex.audit());
  return complex;
}

std::vector<int32_t> labels_of(const TetrahedralComplex& complex) {
  std::vector<int32_t> labels;
  if (!complex.labeled()) return labels;
  labels.reserve(5 * complex.tets.size());
  for (std::size_t t = 0; t < complex.tets.size(); ++t) {
    labels.push_back(complex.region(static_cast<int32_t>(t)));
    for (int k = 0; k < 4; ++k) {
      labels.push_back(complex.constraint(static_cast<int32_t>(t), k));
    }
  }
  return labels;
}

bool same_complex(const TetrahedralComplex& complex,
                  const std::vector<Tetrahedron>& tets,
                  const std::vector<int32_t>& free_slots,
                  const std::vector<int32_t>& labels, int64_t finite_count) {
  if (complex.tets.size() != tets.size() || complex.free_slots != free_slots ||
      labels_of(complex) != labels || complex.finite_count != finite_count) return false;
  for (std::size_t i = 0; i < tets.size(); ++i) {
    const Tetrahedron& actual = complex.tets[i];
    const Tetrahedron& expected = tets[i];
    for (int k = 0; k < 4; ++k) {
      if (actual.v[k] != expected.v[k] || actual.n[k] != expected.n[k]) return false;
    }
    if (actual.stamp != expected.stamp || actual.state != expected.state) return false;
  }
  return true;
}

bool segment_face_is_legal(const double* points, int32_t first, int32_t second,
                           const Tetrahedron& tet, int slot) {
  int32_t face[3];
  oriented_facet(tet.v, slot, face);
  const double* segment[2] = {points + 3 * static_cast<std::size_t>(first),
                              points + 3 * static_cast<std::size_t>(second)};
  const double* triangle[3] = {points + 3 * static_cast<std::size_t>(face[0]),
                               points + 3 * static_cast<std::size_t>(face[1]),
                               points + 3 * static_cast<std::size_t>(face[2])};
  const int64_t segment_ids[2] = {first, second};
  const int64_t triangle_ids[3] = {face[0], face[1], face[2]};
  Contact contact;
  return intersect_segment_triangle(segment, triangle, segment_ids, triangle_ids, contact) ==
             PHX_MC_OK &&
         legal_contact(contact.kind);
}

std::vector<int32_t> exhaustive_cavity(const TetrahedralComplex& complex,
                                       const double* points, int32_t first,
                                       int32_t second) {
  std::vector<int32_t> cavity;
  for (std::size_t t = 0; t < complex.tets.size(); ++t) {
    const Tetrahedron& tet = complex.tets[t];
    if (!complex.live(static_cast<int32_t>(t)) || is_ghost(tet)) continue;
    for (int slot = 0; slot < 4; ++slot) {
      if (!segment_face_is_legal(points, first, second, tet, slot)) {
        cavity.push_back(static_cast<int32_t>(t));
        break;
      }
    }
  }
  return cavity;
}

void compare_with_exhaustive(EdgeCavityScratch& scratch,
                             const TetrahedralComplex& complex, const double* points,
                             int32_t first, int32_t second, int32_t start) {
  const std::vector<int32_t> expected = exhaustive_cavity(complex, points, first, second);
  std::vector<int32_t> actual;
  PHX_CHECK(collect_edge_cavity(complex, points, first, second, start, scratch, actual) ==
            PHX_MC_OK);
  PHX_CHECK(actual == expected);
}

void test_edge_candidates_include_touching_and_coplanar_constraints() {
  // The shared facet is z=0. Point 5 lies in its strict interior; its exact
  // binary coordinates let the contact kernels distinguish touching from
  // disjointness without a floating tolerance.
  const double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.25, 0.25, 1.0, 0.25, 0.25, -1.0,
                           0.25, 0.25, 0.0};
  TetrahedralComplex complex = bipyramid(points);
  EdgeCavityScratch scratch;

  // A proper crossing, a non-shared endpoint touch at the facet interior, a
  // coplanar segment overlapping the facet, and a legal shared-edge contact.
  compare_with_exhaustive(scratch, complex, points, 3, 4, 0);
  compare_with_exhaustive(scratch, complex, points, 3, 5, 0);
  compare_with_exhaustive(scratch, complex, points, 0, 5, 0);
  compare_with_exhaustive(scratch, complex, points, 0, 1, 0);
  PHX_CHECK(exhaustive_cavity(complex, points, 3, 4).size() == 2);
  PHX_CHECK(exhaustive_cavity(complex, points, 3, 5).size() == 2);
  PHX_CHECK(exhaustive_cavity(complex, points, 0, 5).size() == 2);
  PHX_CHECK(exhaustive_cavity(complex, points, 0, 1).empty());
}

void test_adjacency_candidates_follow_accepted_cavity_edit() {
  const double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.25, 0.25, 1.0, 0.25, 0.25, -1.0};
  TetrahedralComplex complex = bipyramid(points);
  complex.enable_labels();
  complex.set_region(0, 5);
  complex.set_region(1, 5);
  complex.set_facet_constraint(0, 0, 42);
  PHX_CHECK(complex.audit());
  EdgeCavityScratch scratch;
  compare_with_exhaustive(scratch, complex, points, 3, 4, 0);

  int32_t face[3];
  oriented_facet(complex.tets[0].v, 3, face);
  std::array<std::array<int32_t, 4>, 3> proposed{};
  for (int k = 0; k < 3; ++k) {
    auto& cell = proposed[static_cast<std::size_t>(k)];
    cell = {face[k], face[(k + 1) % 3], 3, 4};
    if (orient3d(points + 3 * cell[0], points + 3 * cell[1], points + 3 * cell[2],
                 points + 3 * cell[3]) < 0) {
      std::swap(cell[0], cell[1]);
    }
  }
  CavityEdit edit(complex);
  edit.begin();
  PHX_CHECK(edit.remove(0) == EditStatus::kOk);
  PHX_CHECK(edit.remove(1) == EditStatus::kOk);
  for (const auto& cell : proposed) PHX_CHECK(edit.add(cell.data(), nullptr, 5) == EditStatus::kOk);
  PHX_CHECK(edit.validate([&](const int32_t* v) {
    return orient3d(points + 3 * v[0], points + 3 * v[1], points + 3 * v[2], points + 3 * v[3]) > 0;
  }) == EditStatus::kOk);
  const std::vector<Tetrahedron> before_tets = complex.tets;
  const std::vector<int32_t> before_free = complex.free_slots;
  const std::vector<int32_t> before_labels = labels_of(complex);
  const int64_t before_finite = complex.finite_count;
  refused_allocation_attempts.store(0, std::memory_order_relaxed);
  refuse_allocations.store(true, std::memory_order_relaxed);
  bool allocation_refused = false;
  try {
    static_cast<void>(edit.commit());
  } catch (const std::bad_alloc&) {
    allocation_refused = true;
  }
  refuse_allocations.store(false, std::memory_order_relaxed);
  PHX_CHECK(allocation_refused);
  PHX_CHECK(refused_allocation_attempts.load(std::memory_order_relaxed) > 0);
  PHX_CHECK(same_complex(complex, before_tets, before_free, before_labels, before_finite));
  PHX_CHECK(edit.commit() == EditStatus::kOk);
  PHX_CHECK(complex.finite_count == 3);
  PHX_CHECK(complex.audit());
  int marked_facets = 0;
  for (std::size_t t = 0; t < complex.tets.size(); ++t) {
    for (int k = 0; k < 4; ++k) marked_facets += complex.constraint(static_cast<int32_t>(t), k) == 42;
  }
  PHX_CHECK(marked_facets == 2);
  const int32_t start = edit.created()[0];
  compare_with_exhaustive(scratch, complex, points, 3, 4, start);
  PHX_CHECK(exhaustive_cavity(complex, points, 3, 4).empty());
  const std::array<int32_t, 3> flipped_ids{
      edit.created()[0], edit.created()[1], edit.created()[2]};
  const std::array<std::array<int32_t, 4>, 2> original{{
      {0, 1, 2, 3}, {0, 2, 1, 4}}};
  edit.begin();
  for (int32_t id : flipped_ids) PHX_CHECK(edit.remove(id) == EditStatus::kOk);
  for (const auto& cell : original) {
    PHX_CHECK(edit.add(cell.data(), nullptr, 5) == EditStatus::kOk);
  }
  PHX_CHECK(edit.validate([&](const int32_t* v) {
    return orient3d(points + 3 * v[0], points + 3 * v[1], points + 3 * v[2],
                    points + 3 * v[3]) > 0;
  }) == EditStatus::kOk);
  complex.reserve(0, 1);
  refused_allocation_attempts.store(0, std::memory_order_relaxed);
  refuse_allocations.store(true, std::memory_order_relaxed);
  EditStatus reversed = EditStatus::kInternal;
  bool commit_threw = false;
  try {
    reversed = edit.commit();
  } catch (...) {
    commit_threw = true;
  }
  refuse_allocations.store(false, std::memory_order_relaxed);
  PHX_CHECK(!commit_threw);
  PHX_CHECK(reversed == EditStatus::kOk);
  PHX_CHECK(refused_allocation_attempts.load(std::memory_order_relaxed) == 0);
  PHX_CHECK(complex.finite_count == 2);
  PHX_CHECK(complex.audit());
  marked_facets = 0;
  for (std::size_t t = 0; t < complex.tets.size(); ++t) {
    for (int k = 0; k < 4; ++k) marked_facets += complex.constraint(static_cast<int32_t>(t), k) == 42;
  }
  PHX_CHECK(marked_facets == 2);
  compare_with_exhaustive(scratch, complex, points, 3, 4, edit.created()[0]);
}

void test_source_witness_carriers_report_exact_deviation() {
  const double a[3] = {0.1, 0.2, 0.3};
  const double b[3] = {0.4, 0.5, 0.6};
  const double c[3] = {0.2, 0.7, 0.9};
  const double* segment[2] = {a, b};
  const double segment_parameters[2] = {0.375, 0.0};
  Expansion exact_segment[3];
  PHX_CHECK(source_parameters_valid(SourceStratum::kSegment, segment_parameters));
  PHX_CHECK(source_point(segment, SourceStratum::kSegment, segment_parameters, exact_segment));
  double segment_carrier[3], segment_deviation = -1.0;
  PHX_CHECK(source_carrier(segment, SourceStratum::kSegment, segment_parameters,
                           segment_carrier, segment_deviation));
  PHX_CHECK(segment_carrier[0] == nearest_double(exact_segment[0]));
  PHX_CHECK(source_deviation(segment, SourceStratum::kSegment, segment_parameters,
                             segment_carrier) == segment_deviation);
  PHX_CHECK(segment_deviation >= 0.0);

  const double* facet[3] = {a, b, c};
  const double facet_parameters[2] = {0.25, 0.375};
  Expansion exact_facet[3];
  PHX_CHECK(source_parameters_valid(SourceStratum::kFacet, facet_parameters));
  PHX_CHECK(source_point(facet, SourceStratum::kFacet, facet_parameters, exact_facet));
  double facet_carrier[3], facet_deviation = -1.0;
  PHX_CHECK(source_carrier(facet, SourceStratum::kFacet, facet_parameters,
                           facet_carrier, facet_deviation));
  PHX_CHECK(facet_carrier[2] == nearest_double(exact_facet[2]));
  PHX_CHECK(source_deviation(facet, SourceStratum::kFacet, facet_parameters,
                             facet_carrier) == facet_deviation);
  PHX_CHECK(facet_deviation >= 0.0);

  const double tiny_parameter[2] = {0x1p-501, 0.0};
  PHX_CHECK(!source_parameters_valid(SourceStratum::kSegment, tiny_parameter));
  PHX_CHECK(!source_parameters_valid(SourceStratum::kNone, segment_parameters));
  // Plane witnesses use the owning mesh's declared affine chart, not these
  // three simplex corners, even when their parameters happen to be bounded.
  PHX_CHECK(!source_parameters_valid(SourceStratum::kPlane, facet_parameters));
  PHX_CHECK(!source_point(facet, SourceStratum::kPlane, facet_parameters, exact_facet));
  PHX_CHECK(!on_source_entity(facet, SourceStratum::kPlane, a));
}

void test_incremental_triangulation_owns_closed_adjacency() {
  const double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0, 0.125, 0.125, 0.125};
  const std::array<int32_t, 5> order{0, 1, 2, 3, 4};
  std::array<char, 5> alive{1, 1, 1, 1, 1};
  Triangulation3D triangulation(points, nullptr, 5);
  PHX_CHECK(triangulation.build(order, alive) == PHX_MC_OK);
  const TetrahedralComplex& complex = triangulation.complex();
  PHX_CHECK(complex.audit());
  PHX_CHECK(complex.finite_count == 4);
  PHX_CHECK(alive[4] != 0);
  for (const Tetrahedron& tet : complex.tets) {
    if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
    PHX_CHECK(orient3d(points + 3 * tet.v[0], points + 3 * tet.v[1],
                       points + 3 * tet.v[2], points + 3 * tet.v[3]) > 0);
  }
  PHX_CHECK(triangulation.locate(points + 12) >= 0);
}

void test_protection_index_keeps_nonincident_segment_candidates() {
  const double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0,
                           -1.0, 0.0, 0.0, 0.4, 0.4, -0.5, 0.4, 0.4, 0.5};
  const std::array<int32_t, 8> edges{0, 1, 0, 2, 0, 3, 4, 5};
  const std::vector<double> radii =
      protection_radii(points, 6, edges, std::span<const int32_t>{});
  const double component[3] = {0.4, 0.4, 0.0};
  const double expected = 0.5 * std::sqrt(
      component[0] * component[0] + component[1] * component[1] +
      component[2] * component[2]);
  PHX_CHECK(radii.size() == 6);
  PHX_CHECK(radii[0] == expected);
  PHX_CHECK(radii[1] == 0.0 && radii[2] == 0.0 && radii[3] == 0.0);
  PHX_CHECK(radii[4] == 0.0 && radii[5] == 0.0);
}

void test_source_preparation_triangulates_and_compiles_plc_edges() {
  const std::array<double, 24> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
      0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
  const std::array<int64_t, 7> offsets{0, 4, 8, 12, 16, 20, 24};
  const std::array<int32_t, 24> vertices{
      0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4,
      2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5};
  const std::array<int32_t, 6> polygon_facets{0, 1, 2, 3, 4, 5};
  const std::array<int32_t, 12> facet_regions{
      -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0};
  PlcInput input;
  input.point_count = 8;
  input.points = points.data();
  input.polygon_count = 6;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 6;
  input.facet_regions = facet_regions.data();
  input.policy = BoundaryPolicy::kConforming;

  PlcRecovery source;
  PHX_CHECK(prepare_plc_source(input, source) == PHX_MC_OK);
  PHX_CHECK(source.points.size() == points.size());
  PHX_CHECK(source.input_triangles.size() == 36);
  PHX_CHECK(source.input_polygons.size() == 12);
  PHX_CHECK(source.plc_edges.size() == 24);
  PHX_CHECK(source.vertex_dimension.size() == 8);
  PHX_CHECK(source.witnesses.size() == 8);
  PHX_CHECK(source.tets.empty() && source.faces.empty() && source.segments.empty());
}

void test_conforming_nonrepresentable_segment_uses_vertex_cavity() {
  const double height = std::sqrt(3.0) / 2.0;
  const double center_y = height / 3.0;
  // The base face is Delaunay, so the unconstrained triangulation omits the
  // apex-to-apex edge. The apex x coordinates are adjacent doubles, leaving
  // no representable point strictly inside that exact source segment.
  const std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, height, 0.0,
      0.5, center_y, 1.0, std::nextafter(0.5, 1.0), center_y, -1.0};
  const std::array<int64_t, 7> offsets{0, 3, 6, 9, 12, 15, 18};
  const std::array<int32_t, 18> triangles{
      0, 1, 3, 1, 2, 3, 2, 0, 3,
      0, 2, 4, 2, 1, 4, 1, 0, 4};
  const std::array<int32_t, 6> polygon_facets{0, 1, 2, 3, 4, 5};
  const std::array<int32_t, 12> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
  const std::array<int32_t, 2> segments{3, 4};
  PlcInput input;
  input.point_count = 5;
  input.points = points.data();
  input.polygon_count = 6;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 6;
  input.facet_regions = facet_regions.data();
  input.segment_count = 1;
  input.segments = segments.data();
  input.policy = BoundaryPolicy::kConforming;

  PlcRecovery prepared;
  const int32_t prepare_status = prepare_plc_source(input, prepared);
  if (prepare_status != PHX_MC_OK) {
    std::fprintf(stderr, "nonrepresentable segment fixture preparation failed: status=%d "
                         "reason=%d first=(%d,%lld)\n",
                 prepare_status, prepared.failure.reason, prepared.failure.first_kind,
                 static_cast<long long>(prepared.failure.first));
    PHX_CHECK(prepare_status == PHX_MC_OK);
    return;
  }
  // No binary64 point lies strictly inside the source segment.
  PHX_CHECK(std::nextafter(points[9], 1.0) == points[12]);

  const std::array<int32_t, 5> order{0, 1, 2, 3, 4};
  std::array<char, 5> alive{1, 1, 1, 1, 1};
  Triangulation3D unconstrained(points.data(), nullptr, 5);
  const int32_t build_status = unconstrained.build(order, alive);
  if (build_status != PHX_MC_OK) {
    std::fprintf(stderr, "nonrepresentable segment Delaunay precondition failed: status=%d\n",
                 build_status);
    PHX_CHECK(build_status == PHX_MC_OK);
    return;
  }
  bool unconstrained_segment = false;
  Expansion expected_volume;
  for (const Tetrahedron& tet : unconstrained.complex().tets) {
    if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
    bool first = false;
    bool second = false;
    for (int32_t vertex : tet.v) {
      first = first || vertex == 3;
      second = second || vertex == 4;
    }
    unconstrained_segment = unconstrained_segment || (first && second);
    expected_volume = expected_volume + orient3d_exact(
        points.data() + 3 * tet.v[0], points.data() + 3 * tet.v[1],
        points.data() + 3 * tet.v[2], points.data() + 3 * tet.v[3]);
  }
  if (unconstrained_segment) {
    std::fprintf(stderr, "nonrepresentable segment fixture already contains its source edge\n");
    PHX_CHECK(!unconstrained_segment);
    return;
  }

  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  if (status != PHX_MC_OK) {
    std::fprintf(stderr,
                 "conforming nonrepresentable segment cavity recovery failed: status=%d "
                 "reason=%d first=(%d,%lld) second=(%d,%lld)\n",
                 status, result.failure.reason, result.failure.first_kind,
                 static_cast<long long>(result.failure.first), result.failure.second_kind,
                 static_cast<long long>(result.failure.second));
    PHX_CHECK(status == PHX_MC_OK);
    return;
  }
  if (result.points != std::vector<double>(points.begin(), points.end())) {
    PHX_CHECK(result.points == std::vector<double>(points.begin(), points.end()));
    return;
  }
  if (result.tet_regions.size() != 3 ||
      result.tets.size() != 4 * result.tet_regions.size()) {
    std::fprintf(stderr, "vertex-only segment cavity changed cell count: cells=%zu tets=%zu\n",
                 result.tet_regions.size(), result.tets.size() / 4);
    PHX_CHECK(result.tet_regions.size() == 3);
    PHX_CHECK(result.tets.size() == 4 * result.tet_regions.size());
    return;
  }
  PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                        [](int32_t region) { return region == 1; }));
  PHX_CHECK(result.vertex_dimension.size() == points.size() / 3);
  PHX_CHECK(result.witnesses.size() == points.size() / 3);

  std::map<std::array<int32_t, 3>, int32_t> incidence;
  Expansion recovered_volume;
  bool recovered_segment = false;
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    const int32_t* tet = result.tets.data() + 4 * cell;
    const Expansion volume = orient3d_exact(
        result.points.data() + 3 * tet[0], result.points.data() + 3 * tet[1],
        result.points.data() + 3 * tet[2], result.points.data() + 3 * tet[3]);
    PHX_CHECK(volume.sign() > 0);
    recovered_volume = recovered_volume + volume;
    bool first = false;
    bool second = false;
    for (int slot = 0; slot < 4; ++slot) {
      first = first || tet[slot] == 3;
      second = second || tet[slot] == 4;
      int32_t face[3];
      oriented_facet(tet, slot, face);
      ++incidence[sorted_face(face)];
    }
    recovered_segment = recovered_segment || (first && second);
  }
  PHX_CHECK(recovered_segment);
  PHX_CHECK((recovered_volume - expected_volume).sign() == 0);

  std::map<std::array<int32_t, 3>, int32_t> source_faces;
  for (std::size_t row = 0; row < 6; ++row) {
    const std::array<int32_t, 3> face = sorted_face(triangles.data() + 3 * row);
    source_faces.emplace(face, static_cast<int32_t>(row));
    PHX_CHECK(incidence[face] == 1);
  }
  for (const auto& [face, count] : incidence) {
    PHX_CHECK(count == 1 || count == 2);
    if (count == 1) PHX_CHECK(source_faces.contains(face));
  }
  if (result.faces.size() != 18 || result.face_sources.size() != 6) {
    std::fprintf(stderr, "vertex-only segment cavity changed source-face bank: "
                         "faces=%zu sources=%zu\n",
                 result.faces.size() / 3, result.face_sources.size());
    PHX_CHECK(result.faces.size() == 18);
    PHX_CHECK(result.face_sources.size() == 6);
    return;
  }
  PHX_CHECK(result.face_orientations == std::vector<int8_t>(6, 1));
  std::array<int32_t, 6> source_face_counts{};
  for (std::size_t row = 0; row < result.face_sources.size(); ++row) {
    const int32_t source = result.face_sources[row];
    if (source < 0 || source >= static_cast<int32_t>(source_face_counts.size())) {
      PHX_CHECK(source >= 0 && source < static_cast<int32_t>(source_face_counts.size()));
      return;
    }
    ++source_face_counts[static_cast<std::size_t>(source)];
    const std::array<int32_t, 3> face{
        result.faces[3 * row], result.faces[3 * row + 1], result.faces[3 * row + 2]};
    const int32_t* authored = triangles.data() + 3 * static_cast<std::size_t>(source);
    const auto source_face = source_faces.find(sorted_face(face.data()));
    PHX_CHECK(source_face != source_faces.end());
    if (source_face != source_faces.end()) {
      PHX_CHECK(source == source_face->second);
      bool follows_authored_orientation = false;
      for (int rotation = 0; rotation < 3; ++rotation) {
        follows_authored_orientation = follows_authored_orientation ||
            (face[0] == authored[rotation] &&
             face[1] == authored[(rotation + 1) % 3] &&
             face[2] == authored[(rotation + 2) % 3]);
      }
      PHX_CHECK(follows_authored_orientation);
    }
  }
  if (result.segments.size() != 2 * result.segment_sources.size()) {
    std::fprintf(stderr, "vertex-only segment cavity changed source-segment bank: "
                         "segments=%zu sources=%zu\n",
                 result.segments.size() / 2, result.segment_sources.size());
    PHX_CHECK(result.segments.size() == 2 * result.segment_sources.size());
    return;
  }
  std::size_t source_segment_count = 0;
  bool source_segment = false;
  for (std::size_t row = 0; row < result.segment_sources.size(); ++row) {
    const int32_t first = result.segments[2 * row];
    const int32_t second = result.segments[2 * row + 1];
    if (!((first == 3 && second == 4) || (first == 4 && second == 3))) continue;
    source_segment = true;
    ++source_segment_count;
    const int32_t source = result.segment_sources[row];
    PHX_CHECK(source >= 0 && source < static_cast<int32_t>(result.plc_edges.size() / 2));
    if (source >= 0 && source < static_cast<int32_t>(result.plc_edges.size() / 2)) {
      const int32_t source_first = result.plc_edges[2 * source];
      const int32_t source_second = result.plc_edges[2 * source + 1];
      PHX_CHECK((source_first == 3 && source_second == 4) ||
                (source_first == 4 && source_second == 3));
    }
  }
  PHX_CHECK(source_segment_count == 1);
  PHX_CHECK(source_segment);
}

void test_unsplittable_boundary_loop_edge_recovers_under_fixed_and_conforming() {
  // The exact binary64 endpoint deltas are (1501199875790165, 750599937895083)/2^54,
  // with coprime odd numerators. Since both varying coordinates stay in [1/4, 1/2),
  // any binary64 point on the line has an integer segment parameter; no strict
  // interior carrier exists.
  const std::array<double, 15> points{
      0.0, 0.25, 0.25,
      0.0, 17.0 / 64.0, 5.0 / 16.0,
      0.0, 1.0 / 3.0, 7.0 / 24.0,
      0.0, 5.0 / 16.0, 15.0 / 64.0,
      1.0, 0.3, 0.3};
  std::array<int32_t, 18> triangles{
      0, 1, 2, 0, 2, 3, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4};
  double interior[3]{};
  for (std::size_t vertex = 0; vertex < points.size() / 3; ++vertex) {
    for (int axis = 0; axis < 3; ++axis) {
      interior[axis] += points[3 * vertex + static_cast<std::size_t>(axis)] / 5.0;
    }
  }
  for (std::size_t row = 0; row < triangles.size() / 3; ++row) {
    int32_t* face = triangles.data() + 3 * row;
    if (orient3d(points.data() + 3 * face[0], points.data() + 3 * face[1],
                 points.data() + 3 * face[2], interior) > 0) {
      std::swap(face[1], face[2]);
    }
  }
  const std::array<int64_t, 7> offsets{0, 3, 6, 9, 12, 15, 18};
  const std::array<int32_t, 6> polygon_facets{0, 1, 2, 3, 4, 5};
  const std::array<int32_t, 12> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
  PlcInput input;
  input.point_count = 5;
  input.points = points.data();
  input.polygon_count = 6;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 6;
  input.facet_regions = facet_regions.data();
  input.policy = BoundaryPolicy::kConforming;

  PlcRecovery prepared;
  const int32_t prepare_status = prepare_plc_source(input, prepared);
  if (prepare_status != PHX_MC_OK) {
    std::fprintf(stderr, "unsplittable boundary-edge preparation failed: status=%d\n",
                 prepare_status);
    PHX_CHECK(prepare_status == PHX_MC_OK);
    return;
  }

  const std::array<int32_t, 5> order{0, 1, 2, 3, 4};
  std::array<char, 5> alive{1, 1, 1, 1, 1};
  Triangulation3D unconstrained(points.data(), nullptr, 5);
  const int32_t build_status = unconstrained.build(order, alive);
  if (build_status != PHX_MC_OK) {
    PHX_CHECK(build_status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(unconstrained.complex().audit());
  bool initial_edge = false;
  Expansion initial_volume;
  for (std::size_t row = 0; row < unconstrained.complex().tets.size(); ++row) {
    const int32_t cell = static_cast<int32_t>(row);
    const Tetrahedron& tet = unconstrained.complex().tets[row];
    if (!unconstrained.complex().live(cell) || is_ghost(tet)) continue;
    bool has_first = false;
    bool has_second = false;
    for (int32_t vertex : tet.v) {
      has_first = has_first || vertex == 0;
      has_second = has_second || vertex == 2;
    }
    initial_edge = initial_edge || (has_first && has_second);
    initial_volume = initial_volume + orient3d_exact(
        points.data() + 3 * tet.v[0], points.data() + 3 * tet.v[1],
        points.data() + 3 * tet.v[2], points.data() + 3 * tet.v[3]);
  }
  PHX_CHECK(!initial_edge);
  if (initial_edge) return;
  TetrahedralComplex& complex = unconstrained.complex();
  if (complex.finite_count != 2) {
    PHX_CHECK(complex.finite_count == 2);
    return;
  }
  std::array<std::array<int32_t, 4>, 2> original_tets{};
  std::array<int32_t, 2> original_cells{-1, -1};
  std::array<int32_t, 2> old_hull_ghosts{-1, -1};
  const std::array<int32_t, 3> old_base_faces[2]{{0, 1, 3}, {1, 2, 3}};
  std::size_t finite_row = 0;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const int32_t cell = static_cast<int32_t>(row);
    const Tetrahedron& tet = complex.tets[row];
    if (!complex.live(cell) || is_ghost(tet)) continue;
    if (finite_row >= original_tets.size()) {
      PHX_CHECK(finite_row < original_tets.size());
      return;
    }
    std::copy_n(tet.v, 4, original_tets[finite_row].begin());
    original_cells[finite_row] = cell;
    for (int slot = 0; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(tet.v, slot, face);
      for (int base = 0; base < 2; ++base) {
        if (sorted_face(face) == old_base_faces[base]) {
          old_hull_ghosts[static_cast<std::size_t>(base)] = tet.n[slot];
        }
      }
    }
    ++finite_row;
  }
  PHX_CHECK(finite_row == 2);
  if (finite_row != 2) return;
  for (int32_t ghost : old_hull_ghosts) {
    PHX_CHECK(complex.live(ghost));
    if (!complex.live(ghost)) return;
    PHX_CHECK(is_ghost(complex.tets[static_cast<std::size_t>(ghost)]));
  }

  std::array<std::array<int32_t, 4>, 2> retessellated_tets{{
      {0, 1, 2, 4}, {0, 2, 3, 4}}};
  for (auto& tet : retessellated_tets) {
    if (orient3d(points.data() + 3 * tet[0], points.data() + 3 * tet[1],
                 points.data() + 3 * tet[2], points.data() + 3 * tet[3]) < 0) {
      std::swap(tet[0], tet[1]);
    }
  }
  const auto ghost_shell = [&](const std::array<int32_t, 4>& finite) {
    const auto apex = std::find(finite.begin(), finite.end(), 4);
    const int slot = static_cast<int>(apex - finite.begin());
    std::array<int32_t, 4> shell = finite;
    shell[static_cast<std::size_t>(slot)] = kGhostVertex;
    const int first = slot == 0 ? 1 : 0;
    const int second = slot <= 1 ? 2 : 1;
    std::swap(shell[static_cast<std::size_t>(first)],
              shell[static_cast<std::size_t>(second)]);
    return shell;
  };
  CavityEdit hull_flip(complex);
  hull_flip.begin();
  for (int32_t cell : original_cells) {
    PHX_CHECK(hull_flip.remove(cell) == EditStatus::kOk);
  }
  for (int32_t ghost : old_hull_ghosts) {
    PHX_CHECK(hull_flip.remove(ghost) == EditStatus::kOk);
  }
  for (const auto& tet : retessellated_tets) {
    PHX_CHECK(hull_flip.add(tet.data(), nullptr, kNoRegion) == EditStatus::kOk);
  }
  for (const auto& tet : retessellated_tets) {
    const auto shell = ghost_shell(tet);
    PHX_CHECK(hull_flip.add(shell.data(), nullptr, kNoRegion) == EditStatus::kOk);
  }
  const auto positive_tet = [&](const int32_t* tet) {
    return orient3d(points.data() + 3 * tet[0], points.data() + 3 * tet[1],
                    points.data() + 3 * tet[2], points.data() + 3 * tet[3]) > 0;
  };
  EditStatus hull_status = hull_flip.validate(positive_tet);
  if (hull_status == EditStatus::kOk) hull_status = hull_flip.commit();
  if (hull_status != EditStatus::kOk) {
    std::fprintf(stderr, "exact ghost hull replacement failed: status=%d\n",
                 static_cast<int>(hull_status));
    PHX_CHECK(hull_status == EditStatus::kOk);
    hull_flip.rollback();
    return;
  }
  PHX_CHECK(complex.finite_count == 2);
  PHX_CHECK(complex.audit());
  if (hull_flip.created().size() != 4) {
    PHX_CHECK(hull_flip.created().size() == 4);
    return;
  }
  const std::array<int32_t, 4> flipped_cells{
      hull_flip.created()[0], hull_flip.created()[1],
      hull_flip.created()[2], hull_flip.created()[3]};
  PHX_CHECK(unconstrained.set_hint(flipped_cells[0]));
  const std::vector<Tetrahedron> before_rollback = complex.tets;
  const std::vector<int32_t> before_free = complex.free_slots;
  const std::vector<int32_t> before_labels = labels_of(complex);
  const int64_t before_finite = complex.finite_count;

  const auto reject_incomplete_hull_patch = [&](const auto& ghost_shells) {
    CavityEdit rejected_hull_flip(complex);
    rejected_hull_flip.begin();
    for (int32_t cell : flipped_cells) {
      PHX_CHECK(rejected_hull_flip.remove(cell) == EditStatus::kOk);
    }
    for (const auto& tet : original_tets) {
      PHX_CHECK(rejected_hull_flip.add(tet.data(), nullptr, kNoRegion) ==
                EditStatus::kOk);
    }
    for (const auto& shell : ghost_shells) {
      PHX_CHECK(rejected_hull_flip.add(shell.data(), nullptr, kNoRegion) ==
                EditStatus::kOk);
    }
    const EditStatus status = rejected_hull_flip.validate(positive_tet);
    PHX_CHECK(status != EditStatus::kOk);
    rejected_hull_flip.rollback();
    PHX_CHECK(same_complex(complex, before_rollback, before_free, before_labels,
                           before_finite));
    PHX_CHECK(complex.audit());
  };
  const std::vector<std::array<int32_t, 4>> correct_hull_shells{
      ghost_shell(original_tets[0]), ghost_shell(original_tets[1])};
  std::vector<std::array<int32_t, 4>> missing_subface_shells = correct_hull_shells;
  missing_subface_shells.pop_back();
  reject_incomplete_hull_patch(missing_subface_shells);

  std::vector<std::array<int32_t, 4>> reversed_subface_shells = correct_hull_shells;
  auto& reversed_shell = reversed_subface_shells.front();
  int first_non_ghost = -1;
  int second_non_ghost = -1;
  for (int slot = 0; slot < 4; ++slot) {
    if (reversed_shell[static_cast<std::size_t>(slot)] == kGhostVertex) continue;
    if (first_non_ghost < 0) first_non_ghost = slot;
    else second_non_ghost = slot;
  }
  PHX_CHECK(first_non_ghost >= 0 && second_non_ghost >= 0);
  std::swap(reversed_shell[static_cast<std::size_t>(first_non_ghost)],
            reversed_shell[static_cast<std::size_t>(second_non_ghost)]);
  reject_incomplete_hull_patch(reversed_subface_shells);

  std::vector<std::array<int32_t, 4>> extra_subface_shells = correct_hull_shells;
  extra_subface_shells.push_back(ghost_shell(retessellated_tets[0]));
  reject_incomplete_hull_patch(extra_subface_shells);
  PHX_CHECK(unconstrained.locate(points.data()) >= 0);


  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  if (status != PHX_MC_OK) {
    std::fprintf(stderr,
                 "unsplittable boundary-loop edge recovery failed: status=%d "
                 "reason=%d first=(%d,%lld)\n",
                 status, result.failure.reason, result.failure.first_kind,
                 static_cast<long long>(result.failure.first));
    PHX_CHECK(status == PHX_MC_OK);
    return;
  }
  if (result.points != std::vector<double>(points.begin(), points.end())) {
    PHX_CHECK(result.points == std::vector<double>(points.begin(), points.end()));
    return;
  }
  if (result.tets.size() != 8 || result.tet_regions.size() != 2) {
    PHX_CHECK(result.tets.size() == 8);
    PHX_CHECK(result.tet_regions.size() == 2);
    return;
  }
  PHX_CHECK(result.tet_regions[0] == 1 && result.tet_regions[1] == 1);
  if (result.face_sources.size() != 6 || result.faces.size() != 18 ||
      result.face_orientations.size() != 6) {
    PHX_CHECK(result.face_sources.size() == 6);
    PHX_CHECK(result.faces.size() == 18);
    PHX_CHECK(result.face_orientations.size() == 6);
    return;
  }
  PHX_CHECK(result.face_orientations == std::vector<int8_t>(6, 1));
  if (result.segments.size() != 2 * result.segment_sources.size()) {
    PHX_CHECK(result.segments.size() == 2 * result.segment_sources.size());
    return;
  }
  std::array<int32_t, 6> source_face_counts{};
  for (std::size_t row = 0; row < result.face_sources.size(); ++row) {
    const int32_t source = result.face_sources[row];
    if (source < 0 || source >= static_cast<int32_t>(source_face_counts.size())) {
      PHX_CHECK(source >= 0 && source < static_cast<int32_t>(source_face_counts.size()));
      return;
    }
    ++source_face_counts[static_cast<std::size_t>(source)];
    const std::array<int32_t, 3> face{
        result.faces[3 * row], result.faces[3 * row + 1], result.faces[3 * row + 2]};
    PHX_CHECK(sorted_face(face.data()) ==
              sorted_face(result.input_triangles.data() + 3 * source));
  }
  PHX_CHECK(std::all_of(source_face_counts.begin(), source_face_counts.end(),
                        [](int32_t count) { return count == 1; }));
  bool recovered_edge = false;
  Expansion recovered_volume;
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    const int32_t* tet = result.tets.data() + 4 * cell;
    const Expansion volume = orient3d_exact(
        result.points.data() + 3 * tet[0], result.points.data() + 3 * tet[1],
        result.points.data() + 3 * tet[2], result.points.data() + 3 * tet[3]);
    PHX_CHECK(volume.sign() > 0);
    recovered_volume = recovered_volume + volume;
    bool has_first = false;
    bool has_second = false;
    for (int slot = 0; slot < 4; ++slot) {
      has_first = has_first || tet[slot] == 0;
      has_second = has_second || tet[slot] == 2;
    }
    recovered_edge = recovered_edge || (has_first && has_second);
  }
  PHX_CHECK(recovered_edge);
  PHX_CHECK((recovered_volume - initial_volume).sign() == 0);
  std::size_t recovered_source_edge_rows = 0;
  for (std::size_t row = 0; row < result.segment_sources.size(); ++row) {
    const int32_t first = result.segments[2 * row];
    const int32_t second = result.segments[2 * row + 1];
    if (!((first == 0 && second == 2) || (first == 2 && second == 0))) continue;
    ++recovered_source_edge_rows;
    const int32_t source = result.segment_sources[row];
    PHX_CHECK(source >= 0 && source < static_cast<int32_t>(result.plc_edges.size() / 2));
    if (source >= 0 && source < static_cast<int32_t>(result.plc_edges.size() / 2)) {
      const int32_t source_first = result.plc_edges[2 * source];
      const int32_t source_second = result.plc_edges[2 * source + 1];
      PHX_CHECK((source_first == 0 && source_second == 2) ||
                (source_first == 2 && source_second == 0));
    }
  }
  PHX_CHECK(recovered_source_edge_rows == 1);
  input.policy = BoundaryPolicy::kFixed;
  PlcRecovery fixed_result;
  const int32_t fixed_status = recover_plc(input, fixed_result);
  if (fixed_status != PHX_MC_OK) {
    std::fprintf(stderr,
                 "fixed unsplittable boundary-loop edge recovery failed: status=%d "
                 "reason=%d first=(%d,%lld)\n",
                 fixed_status, fixed_result.failure.reason,
                 fixed_result.failure.first_kind,
                 static_cast<long long>(fixed_result.failure.first));
    PHX_CHECK(fixed_status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(fixed_result.points == std::vector<double>(points.begin(), points.end()));
  PHX_CHECK(fixed_result.tets.size() == 8);
  PHX_CHECK(fixed_result.tet_regions == std::vector<int32_t>({1, 1}));
  PHX_CHECK(fixed_result.faces.size() == 18);
  PHX_CHECK(fixed_result.face_sources.size() == 6);
  PHX_CHECK(fixed_result.face_orientations == std::vector<int8_t>(6, 1));
  std::array<int32_t, 6> fixed_face_counts{};
  for (std::size_t row = 0; row < fixed_result.face_sources.size(); ++row) {
    const int32_t source = fixed_result.face_sources[row];
    PHX_CHECK(source >= 0 && source < static_cast<int32_t>(fixed_face_counts.size()));
    if (source < 0 || source >= static_cast<int32_t>(fixed_face_counts.size())) return;
    ++fixed_face_counts[static_cast<std::size_t>(source)];
    PHX_CHECK(sorted_face(fixed_result.faces.data() + 3 * row) ==
              sorted_face(fixed_result.input_triangles.data() + 3 * source));
  }
  PHX_CHECK(std::all_of(fixed_face_counts.begin(), fixed_face_counts.end(),
                        [](int32_t count) { return count == 1; }));
  bool fixed_edge_recovered = false;
  Expansion fixed_volume;
  for (std::size_t cell = 0; cell < fixed_result.tet_regions.size(); ++cell) {
    const int32_t* tet = fixed_result.tets.data() + 4 * cell;
    const Expansion volume = orient3d_exact(
        fixed_result.points.data() + 3 * tet[0],
        fixed_result.points.data() + 3 * tet[1],
        fixed_result.points.data() + 3 * tet[2],
        fixed_result.points.data() + 3 * tet[3]);
    PHX_CHECK(volume.sign() > 0);
    PHX_CHECK(fixed_result.tet_regions[cell] == 1);
    fixed_volume = fixed_volume + volume;
    bool has_first = false;
    bool has_second = false;
    for (int slot = 0; slot < 4; ++slot) {
      has_first = has_first || tet[slot] == 0;
      has_second = has_second || tet[slot] == 2;
    }
    fixed_edge_recovered = fixed_edge_recovered || (has_first && has_second);
  }
  PHX_CHECK(fixed_edge_recovered);
  PHX_CHECK((fixed_volume - initial_volume).sign() == 0);
  PHX_CHECK(fixed_result.segments.size() ==
            2 * fixed_result.segment_sources.size());
}

// Scientific contract of a conforming recovery: positive cells of one
// region, every source edge row published as a chain of tetrahedral edges
// from its first to its last endpoint, every published witnessed vertex
// exactly on its source entity (deviation zero; witnessed positions that are
// not binary64 points are suppressed), and every facet present.
void check_conforming_recovery(const PlcRecovery& result, std::size_t input_points,
                               std::size_t facet_count) {
  PHX_CHECK(result.witnesses.size() == result.points.size() / 3);
  PHX_CHECK(result.segments.size() == 2 * result.segment_sources.size());
  if (result.witnesses.size() != result.points.size() / 3 ||
      result.segments.size() != 2 * result.segment_sources.size()) {
    return;
  }
  const auto vertex_point = [&](int32_t vertex) {
    return result.points.data() + 3 * static_cast<std::size_t>(vertex);
  };
  for (std::size_t vertex = input_points; vertex < result.witnesses.size(); ++vertex) {
    const PlcRecoveryWitness& witness = result.witnesses[vertex];
    if (witness.stratum == SourceStratum::kNone) continue;
    PHX_CHECK(witness.deviation == 0.0);
    const int32_t* corners_rows = witness.stratum == SourceStratum::kSegment
        ? result.plc_edges.data() + 2 * static_cast<std::size_t>(witness.entity)
        : result.input_triangles.data() + 3 * static_cast<std::size_t>(witness.entity);
    const double* corners[3] = {vertex_point(corners_rows[0]), vertex_point(corners_rows[1]),
                                witness.stratum == SourceStratum::kFacet
                                    ? vertex_point(corners_rows[2]) : nullptr};
    PHX_CHECK(on_source_entity(corners, witness.stratum,
                               vertex_point(static_cast<int32_t>(vertex))));
  }
  std::set<std::pair<int32_t, int32_t>> tet_edges;
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    const int32_t* tet = result.tets.data() + 4 * cell;
    PHX_CHECK(orient3d(vertex_point(tet[0]), vertex_point(tet[1]), vertex_point(tet[2]),
                       vertex_point(tet[3])) > 0);
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        tet_edges.insert({std::min(tet[i], tet[j]), std::max(tet[i], tet[j])});
      }
    }
  }
  std::map<int32_t, std::vector<std::pair<int32_t, int32_t>>> chains;
  for (std::size_t row = 0; row < result.segment_sources.size(); ++row) {
    const int32_t source = result.segment_sources[row];
    PHX_CHECK(source >= 0 && source < static_cast<int32_t>(result.plc_edges.size() / 2));
    const int32_t first = result.segments[2 * row];
    const int32_t second = result.segments[2 * row + 1];
    PHX_CHECK(tet_edges.contains({std::min(first, second), std::max(first, second)}));
    chains[source].push_back({first, second});
  }
  for (const auto& [source, pieces] : chains) {
    const int32_t last = result.plc_edges[2 * static_cast<std::size_t>(source) + 1];
    int32_t at = result.plc_edges[2 * static_cast<std::size_t>(source)];
    std::vector<char> used(pieces.size(), 0);
    std::size_t walked = 0;
    while (at != last && walked < pieces.size()) {
      std::size_t next = pieces.size();
      for (std::size_t row = 0; row < pieces.size() && next == pieces.size(); ++row) {
        if (used[row] == 0 && (pieces[row].first == at || pieces[row].second == at)) next = row;
      }
      if (next == pieces.size()) break;
      used[next] = 1;
      at = pieces[next].first == at ? pieces[next].second : pieces[next].first;
      ++walked;
    }
    PHX_CHECK(at == last);
    PHX_CHECK(walked == pieces.size());
  }
  PHX_CHECK(result.face_sources.size() == result.faces.size() / 3);
  PHX_CHECK(result.face_orientations.size() == result.face_sources.size());
  std::vector<int32_t> facet_counts(facet_count, 0);
  for (std::size_t row = 0; row < result.face_sources.size(); ++row) {
    const int32_t facet = result.face_sources[row];
    PHX_CHECK(facet >= 0 && facet < static_cast<int32_t>(facet_count));
    PHX_CHECK(result.face_orientations[row] == 1);
    if (facet >= 0 && facet < static_cast<int32_t>(facet_count)) {
      ++facet_counts[static_cast<std::size_t>(facet)];
    }
  }
  PHX_CHECK(std::all_of(facet_counts.begin(), facet_counts.end(),
                        [](int32_t count) { return count > 0; }));
}

void test_conforming_missing_boundary_edges_recover_with_exact_witnesses() {
  // Neither AB nor AC is an edge of the Delaunay tetrahedralization of the
  // six points (F encroaches AB). Both must be recovered as conforming
  // chains whose published vertices lie exactly on their sources.
  const std::array<double, 18> points{
      0.0, 0.25, 0.25,
      0.0, 17.0 / 64.0, 5.0 / 16.0,
      0.0, 1.0 / 3.0, 7.0 / 24.0,
      0.0, 5.0 / 16.0, 15.0 / 64.0,
      1.0, 0.3, 0.3,
      0.0, 0.25, 9.0 / 32.0};
  std::array<int32_t, 24> triangles{
      0, 1, 2, 0, 2, 3, 0, 5, 1,
      0, 5, 4, 5, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4};
  double interior[3]{};
  for (std::size_t vertex = 0; vertex < points.size() / 3; ++vertex) {
    for (int axis = 0; axis < 3; ++axis) {
      interior[axis] += points[3 * vertex + static_cast<std::size_t>(axis)] / 6.0;
    }
  }
  for (std::size_t row = 0; row < triangles.size() / 3; ++row) {
    int32_t* face = triangles.data() + 3 * row;
    if (orient3d(points.data() + 3 * face[0], points.data() + 3 * face[1],
                 points.data() + 3 * face[2], interior) > 0) {
      std::swap(face[1], face[2]);
    }
  }
  const std::array<int64_t, 9> offsets{0, 3, 6, 9, 12, 15, 18, 21, 24};
  const std::array<int32_t, 8> polygon_facets{0, 1, 2, 3, 4, 5, 6, 7};
  const std::array<int32_t, 16> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
  PlcInput input;
  input.point_count = 6;
  input.points = points.data();
  input.polygon_count = 8;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 8;
  input.facet_regions = facet_regions.data();
  input.policy = BoundaryPolicy::kConforming;

  const std::array<int32_t, 6> order{0, 1, 2, 3, 4, 5};
  std::array<char, 6> alive{1, 1, 1, 1, 1, 1};
  Triangulation3D initial(points.data(), nullptr, 6);
  const int32_t build_status = initial.build(order, alive);
  if (build_status != PHX_MC_OK) {
    PHX_CHECK(build_status == PHX_MC_OK);
    return;
  }
  const auto has_tet_edge = [](const TetrahedralComplex& complex,
                               int32_t first, int32_t second) {
    for (std::size_t row = 0; row < complex.tets.size(); ++row) {
      const Tetrahedron& tet = complex.tets[row];
      if (!complex.live(static_cast<int32_t>(row)) || is_ghost(tet)) continue;
      if (std::find(tet.v, tet.v + 4, first) != tet.v + 4 &&
          std::find(tet.v, tet.v + 4, second) != tet.v + 4) {
        return true;
      }
    }
    return false;
  };
  PHX_CHECK(!has_tet_edge(initial.complex(), 0, 1));
  PHX_CHECK(!has_tet_edge(initial.complex(), 0, 2));
  Expansion initial_volume;
  for (std::size_t row = 0; row < initial.complex().tets.size(); ++row) {
    const Tetrahedron& tet = initial.complex().tets[row];
    if (!initial.complex().live(static_cast<int32_t>(row)) || is_ghost(tet)) continue;
    initial_volume = initial_volume + orient3d_exact(
        points.data() + 3 * tet.v[0], points.data() + 3 * tet.v[1],
        points.data() + 3 * tet.v[2], points.data() + 3 * tet.v[3]);
  }

  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  if (status != PHX_MC_OK) {
    std::fprintf(stderr, "missing boundary edge recovery failed: status=%d reason=%d first=(%d,%lld)\n",
                 status, result.failure.reason, result.failure.first_kind,
                 static_cast<long long>(result.failure.first));
    PHX_CHECK(status == PHX_MC_OK);
    return;
  }
  check_conforming_recovery(result, 6, 8);
  std::set<int32_t> recovered_rows(result.segment_sources.begin(), result.segment_sources.end());
  for (std::size_t row = 0; row < result.plc_edges.size() / 2; ++row) {
    PHX_CHECK(recovered_rows.contains(static_cast<int32_t>(row)));
  }
  Expansion recovered_volume;
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    const int32_t* tet = result.tets.data() + 4 * cell;
    PHX_CHECK(result.tet_regions[cell] == 1);
    recovered_volume = recovered_volume + orient3d_exact(
        result.points.data() + 3 * tet[0], result.points.data() + 3 * tet[1],
        result.points.data() + 3 * tet[2], result.points.data() + 3 * tet[3]);
  }
  PHX_CHECK((recovered_volume - initial_volume).sign() == 0);

  // Selective protection either preserves the complete input triangle and
  // its edges, or reports a fixed-boundary recovery refusal; it must never
  // silently publish a split of that source facet.
  const std::array<std::uint8_t, 8> protected_facets{1, 0, 0, 0, 0, 0, 0, 0};
  input.protected_facets = protected_facets.data();
  PlcRecovery selective;
  const int32_t protected_status = recover_plc(input, selective);
  if (protected_status == PHX_MC_OK) {
    std::size_t count = 0;
    auto original = std::array<int32_t, 3>{triangles[0], triangles[1], triangles[2]};
    std::sort(original.begin(), original.end());
    for (std::size_t row = 0; row < selective.face_sources.size(); ++row) {
      if (selective.face_sources[row] != 0) continue;
      ++count;
      auto face = std::array<int32_t, 3>{
          selective.faces[3 * row], selective.faces[3 * row + 1], selective.faces[3 * row + 2]};
      std::sort(face.begin(), face.end());
      PHX_CHECK(face == original);
    }
    PHX_CHECK(count == 1);
  } else {
    PHX_CHECK(selective.failure.reason == phx::mc::kPlcFixedFacet ||
              selective.failure.reason == phx::mc::kPlcFixedSegment);
  }
}

// Recovers a closed triangle surface, one facet per triangle oriented with
// the single region 1 behind it.
int32_t recover_closed_surface(const std::vector<double>& points, std::vector<int32_t> triangles,
                               BoundaryPolicy policy, PlcRecovery& result) {
  const std::size_t count = points.size() / 3;
  double interior[3]{};
  for (std::size_t vertex = 0; vertex < count; ++vertex) {
    for (int axis = 0; axis < 3; ++axis) {
      interior[axis] += points[3 * vertex + static_cast<std::size_t>(axis)] / static_cast<double>(count);
    }
  }
  const std::size_t faces = triangles.size() / 3;
  for (std::size_t row = 0; row < faces; ++row) {
    int32_t* face = triangles.data() + 3 * row;
    if (orient3d(points.data() + 3 * face[0], points.data() + 3 * face[1],
                 points.data() + 3 * face[2], interior) > 0) {
      std::swap(face[1], face[2]);
    }
  }
  std::vector<int64_t> offsets(faces + 1);
  std::vector<int32_t> facets(faces);
  std::vector<int32_t> regions(2 * faces);
  for (std::size_t row = 0; row <= faces; ++row) offsets[row] = static_cast<int64_t>(3 * row);
  for (std::size_t row = 0; row < faces; ++row) {
    facets[row] = static_cast<int32_t>(row);
    regions[2 * row] = -1;
    regions[2 * row + 1] = 1;
  }
  PlcInput input;
  input.point_count = static_cast<int64_t>(count);
  input.points = points.data();
  input.polygon_count = static_cast<int64_t>(faces);
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = facets.data();
  input.facet_count = static_cast<int64_t>(faces);
  input.facet_regions = regions.data();
  input.policy = policy;
  return recover_plc(input, result);
}

// FIXED keeps the boundary: every vertex beyond the input is a volume
// point and every source edge row is one published segment.
void check_fixed_boundary(const PlcRecovery& result, std::size_t input_points) {
  for (std::size_t vertex = input_points; vertex < result.witnesses.size(); ++vertex) {
    PHX_CHECK(result.witnesses[vertex].stratum == SourceStratum::kNone);
  }
  for (int32_t vertex : result.faces) PHX_CHECK(vertex < static_cast<int32_t>(input_points));
  PHX_CHECK(result.segments.size() == result.plc_edges.size());
}

// Schoenhardt prism (lateral quads split along their reflex diagonals): no
// tetrahedralization of its six vertices exists, so recovery needs a split
// of a boundary edge. Top triangle twisted by 30 or 20 degrees (binary64
// coordinates of cos/sin, fixed bit patterns).
std::vector<double> schoenhardt_points(bool thirty_degrees) {
  std::vector<double> points{
      0x1.0000000000000p+0,  0x0.0p+0,               0x0.0p+0,
      -0x1.ffffffffffffdp-2, 0x1.bb67ae8584cabp-1,   0x0.0p+0,
      -0x1.0000000000004p-1, -0x1.bb67ae8584ca8p-1,  0x0.0p+0};
  if (thirty_degrees) {
    points.insert(points.end(), {0x1.bb67ae8584cabp-1, 0x1.fffffffffffffp-2, 1.0,
                                 -0x1.bb67ae8584ca9p-1, 0x1.0000000000003p-1, 1.0,
                                 -0x1.a79394c9e8a0ap-53, -1.0, 1.0});
  } else {
    points.insert(points.end(), {0x1.e11f642522d1cp-1, 0x1.5e3a8748a0bf5p-2, 1.0,
                                 -0x1.8836fa2cf5038p-1, 0x1.491b7523c161ep-1, 1.0,
                                 -0x1.63a1a7e0b73a8p-3, -0x1.f838b8c811c16p-1, 1.0});
  }
  return points;
}

std::vector<int32_t> schoenhardt_triangles() {
  std::vector<int32_t> triangles{0, 1, 2, 3, 4, 5};
  for (int32_t k = 0; k < 3; ++k) {
    const int32_t a = k;
    const int32_t b = (k + 1) % 3;
    triangles.insert(triangles.end(), {a, b, 3 + b, a, 3 + b, 3 + a});
  }
  return triangles;
}

void test_witnessed_segment_split_is_published_exactly() {
  // With a 30 degree twist the conforming recovery keeps one representable
  // boundary split; FIXED removes it again into the volume.
  const std::vector<double> points = schoenhardt_points(true);
  PlcRecovery conforming;
  int32_t status = recover_closed_surface(points, schoenhardt_triangles(),
                                          BoundaryPolicy::kConforming, conforming);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) return;
  check_conforming_recovery(conforming, 6, 8);
  std::size_t segment_points = 0;
  for (std::size_t vertex = 6; vertex < conforming.witnesses.size(); ++vertex) {
    if (conforming.witnesses[vertex].stratum != SourceStratum::kSegment) continue;
    ++segment_points;
    const PlcRecoveryWitness& witness = conforming.witnesses[vertex];
    const int32_t* edge = conforming.plc_edges.data() + 2 * static_cast<std::size_t>(witness.entity);
    Expansion exact[3];
    segment_witness_position(points.data() + 3 * edge[0], points.data() + 3 * edge[1],
                             witness.parameters[0], exact);
    for (int axis = 0; axis < 3; ++axis) {
      PHX_CHECK((exact[axis] - Expansion(conforming.points[3 * vertex + static_cast<std::size_t>(axis)]))
                    .sign() == 0);
    }
  }
  PHX_CHECK(segment_points > 0);
  PlcRecovery fixed;
  status = recover_closed_surface(points, schoenhardt_triangles(), BoundaryPolicy::kFixed, fixed);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) return;
  check_conforming_recovery(fixed, 6, 8);
  check_fixed_boundary(fixed, 6);
  PHX_CHECK(fixed.points.size() / 3 > 6);
}

void test_nonrepresentable_split_is_suppressed_into_the_volume() {
  // With a 20 degree twist the needed split point is not a binary64 point;
  // it is suppressed under both policies and a volume point remains.
  const std::vector<double> points = schoenhardt_points(false);
  for (BoundaryPolicy policy : {BoundaryPolicy::kConforming, BoundaryPolicy::kFixed}) {
    PlcRecovery result;
    const int32_t status = recover_closed_surface(points, schoenhardt_triangles(), policy, result);
    PHX_CHECK(status == PHX_MC_OK);
    if (status != PHX_MC_OK) continue;
    check_conforming_recovery(result, 6, 8);
    check_fixed_boundary(result, 6);
    PHX_CHECK(result.points.size() / 3 > 6);
  }
}

void test_bent_exterior_crease_edges_recover_under_both_policies() {
  // A hexagonal base in the plane x + y + 3z = 1 whose rounded vertices make
  // every fan diagonal a crease bent by rounding only, under an apex. The
  // creases are missing from the Delaunay tetrahedralization and are not
  // recoverable by in-plane flips.
  const std::vector<double> points{
      0x1.0888888888888p+0,  0x1.0000000000000p-1, -0x1.6c16c16c16c15p-3,
      0x1.888888888888ap-1,  0x1.3b1ee411cd706p-1, -0x1.04df3b7872a15p-3,
      0x1.ddddddddddde0p-3,  0x1.3b1ee411cd706p-1, 0x1.9cde17ce9080bp-5,
      -0x1.1111111111110p-5, 0x1.0000000000000p-1, 0x1.6c16c16c16c15p-3,
      0x1.dddddddddddd6p-3,  0x1.89c237dc651f5p-2, 0x1.04df3b7872a15p-3,
      0x1.888888888888ap-1,  0x1.89c237dc651f5p-2, -0x1.9cde17ce90818p-5,
      0x1.3dbfe0c377b3bp-1,  0x1.3dbfe0c377b3cp-1, 0x1.727f4494ce361p-2};
  std::vector<int32_t> triangles;
  for (int32_t k = 1; k < 5; ++k) triangles.insert(triangles.end(), {0, k, k + 1});
  for (int32_t k = 0; k < 6; ++k) triangles.insert(triangles.end(), {6, k, (k + 1) % 6});
  bool bent = false;
  for (int32_t k = 1; k < 4; ++k) {
    bent = bent || orient3d(points.data(), points.data() + 3 * k, points.data() + 3 * (k + 1),
                            points.data() + 3 * (k + 2)) != 0;
  }
  PHX_CHECK(bent);
  for (BoundaryPolicy policy : {BoundaryPolicy::kConforming, BoundaryPolicy::kFixed}) {
    PlcRecovery result;
    const int32_t status = recover_closed_surface(points, triangles, policy, result);
    PHX_CHECK(status == PHX_MC_OK);
    if (status != PHX_MC_OK) continue;
    check_conforming_recovery(result, 7, 10);
    if (policy == BoundaryPolicy::kFixed) check_fixed_boundary(result, 7);
  }
}
void test_conforming_representable_segment_prefers_vertex_cavity() {
  const double height = std::sqrt(3.0) / 2.0;
  const double center_y = height / 3.0;
  const std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, height, 0.0,
      0.5, center_y, 1.0, 0.625, center_y, -1.0};
  const std::array<int64_t, 7> offsets{0, 3, 6, 9, 12, 15, 18};
  const std::array<int32_t, 18> triangles{
      0, 1, 3, 1, 2, 3, 2, 0, 3, 0, 2, 4, 2, 1, 4, 1, 0, 4};
  const std::array<int32_t, 6> polygon_facets{0, 1, 2, 3, 4, 5};
  const std::array<int32_t, 12> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
  const std::array<int32_t, 2> segment{3, 4};
  PlcInput input;
  input.point_count = 5;
  input.points = points.data();
  input.polygon_count = 6;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 6;
  input.facet_regions = facet_regions.data();
  input.segment_count = 1;
  input.segments = segment.data();
  input.policy = BoundaryPolicy::kConforming;

  PlcRecovery prepared;
  PHX_CHECK(prepare_plc_source(input, prepared) == PHX_MC_OK);

  const std::array<int32_t, 5> order{0, 1, 2, 3, 4};
  std::array<char, 5> alive{1, 1, 1, 1, 1};
  Triangulation3D unconstrained(points.data(), nullptr, 5);
  PHX_CHECK(unconstrained.build(order, alive) == PHX_MC_OK);
  Expansion expected_volume;
  bool unconstrained_segment = false;
  for (const Tetrahedron& tet : unconstrained.complex().tets) {
    if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
    bool has_first = false;
    bool has_second = false;
    for (int32_t vertex : tet.v) {
      has_first = has_first || vertex == 3;
      has_second = has_second || vertex == 4;
    }
    unconstrained_segment = unconstrained_segment || (has_first && has_second);
    expected_volume = expected_volume + orient3d_exact(
        points.data() + 3 * tet.v[0], points.data() + 3 * tet.v[1],
        points.data() + 3 * tet.v[2], points.data() + 3 * tet.v[3]);
  }
  PHX_CHECK(!unconstrained_segment);

  PlcRecovery result;
  PHX_CHECK(recover_plc(input, result) == PHX_MC_OK);
  PHX_CHECK(result.points == std::vector<double>(points.begin(), points.end()));
  PHX_CHECK(result.tet_regions.size() == 3);
  PHX_CHECK(result.tets.size() == 4 * result.tet_regions.size());
  if (result.tets.size() != 4 * result.tet_regions.size()) return;
  PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                        [](std::int32_t region) { return region == 1; }));
  Expansion recovered_volume;
  for (std::size_t row = 0; row < result.tet_regions.size(); ++row) {
    const std::int32_t* tet = result.tets.data() + 4 * row;
    const Expansion volume = orient3d_exact(
        result.points.data() + 3 * tet[0], result.points.data() + 3 * tet[1],
        result.points.data() + 3 * tet[2], result.points.data() + 3 * tet[3]);
    PHX_CHECK(volume.sign() > 0);
    recovered_volume = recovered_volume + volume;
  }
  PHX_CHECK((recovered_volume - expected_volume).sign() == 0);
  PHX_CHECK(result.faces.size() == 18 && result.face_sources.size() == 6);
  PHX_CHECK(result.face_orientations == std::vector<int8_t>(6, 1));
  bool source_segment = false;
  for (std::size_t row = 0; row < result.segment_sources.size(); ++row) {
    const int32_t source = result.segment_sources[row];
    if (source < 0 ||
        source >= static_cast<int32_t>(result.plc_edges.size() / 2)) {
      continue;
    }
    const int32_t source_first = result.plc_edges[2 * source];
    const int32_t source_second = result.plc_edges[2 * source + 1];
    source_segment =
        source_segment ||
        ((source_first == 3 && source_second == 4) ||
         (source_first == 4 && source_second == 3));
  }
  PHX_CHECK(source_segment);
}


void test_incremental_point_slot_reindexes_corners_and_rolls_back() {
  const std::array<double, 36> original_points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 2.0, -2.0,
      -2.0, -2.0, 2.0, 2.0, -2.0, 2.0, -2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  const std::array<int32_t, 12> order{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<char> alive(12, 1);
  Triangulation3D triangulation(original_points.data(), nullptr, 12);
  const int32_t build_status = triangulation.build(order, alive);
  if (build_status != PHX_MC_OK) {
    std::fprintf(stderr, "corner-slot triangulation setup failed: status=%d\n",
                 build_status);
    PHX_CHECK(build_status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(triangulation.complex().audit());

  const std::vector<Tetrahedron> original_tets = triangulation.complex().tets;
  const auto same_tet_bank = [&](const std::vector<Tetrahedron>& expected) -> bool {
    const auto& actual = triangulation.complex().tets;
    if (actual.size() != expected.size()) return false;
    for (std::size_t row = 0; row < actual.size(); ++row) {
      for (int slot = 0; slot < 4; ++slot) {
        if (actual[row].v[slot] != expected[row].v[slot] ||
            actual[row].n[slot] != expected[row].n[slot]) return false;
      }
      if (actual[row].stamp != expected[row].stamp ||
          actual[row].state != expected[row].state) return false;
    }
    return true;
  };
  PHX_CHECK(!triangulation.insert_point_slot(4, 11));
  PHX_CHECK(same_tet_bank(original_tets));
  PHX_CHECK(triangulation.complex().audit());

  std::array<double, 39> inserted_points{};
  std::copy_n(original_points.data(), 12, inserted_points.data());
  inserted_points[12] = 0.2;
  inserted_points[13] = 0.15;
  inserted_points[14] = 0.1;
  std::copy(original_points.begin() + 12, original_points.end(),
            inserted_points.begin() + 15);
  std::vector<char> inserted_alive = alive;
  inserted_alive.insert(inserted_alive.begin() + 4, static_cast<char>(1));
  if (!triangulation.insert_point_slot(4, 12)) {
    PHX_CHECK(false);
    return;
  }
  triangulation.rebind(inserted_points.data(), 13);
  auto& complex = triangulation.complex();
  PHX_CHECK(complex.tets.size() == original_tets.size());
  for (std::size_t row = 0; row < original_tets.size(); ++row) {
    const Tetrahedron& before = original_tets[row];
    const Tetrahedron& shifted = complex.tets[row];
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t expected =
          before.v[slot] >= 4 ? before.v[slot] + 1 : before.v[slot];
      PHX_CHECK(shifted.v[slot] == expected);
      PHX_CHECK(shifted.n[slot] == before.n[slot]);
    }
  }
  PHX_CHECK(complex.audit());

  const int32_t insert_status = triangulation.insert(4, inserted_alive);
  if (insert_status != PHX_MC_OK) {
    std::fprintf(stderr, "corner-slot point insertion failed: status=%d\n",
                 insert_status);
    PHX_CHECK(insert_status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(complex.audit());
  bool inserted_vertex_is_live = false;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const int32_t cell = static_cast<int32_t>(row);
    const Tetrahedron& tet = complex.tets[row];
    if (!complex.live(cell) || is_ghost(tet)) continue;
    inserted_vertex_is_live =
        inserted_vertex_is_live || vertex_slot(tet, 4) >= 0;
  }
  PHX_CHECK(inserted_vertex_is_live);
  for (int corner = 0; corner < 8; ++corner) {
    const double* actual = triangulation.point(5 + corner);
    const double* expected = original_points.data() + 3 * (4 + corner);
    for (int axis = 0; axis < 3; ++axis) PHX_CHECK(actual[axis] == expected[axis]);
  }

  const std::vector<Tetrahedron> before_rollback = complex.tets;
  const std::vector<int32_t> before_free_slots = complex.free_slots;
  const int64_t before_finite_count = complex.finite_count;
  std::vector<int32_t> before_constraints;
  std::vector<int32_t> before_regions;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    before_regions.push_back(complex.region(static_cast<int32_t>(row)));
    for (int slot = 0; slot < 4; ++slot) {
      before_constraints.push_back(
          complex.constraint(static_cast<int32_t>(row), slot));
    }
  }
  int32_t cell = -1;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const int32_t candidate = static_cast<int32_t>(row);
    if (complex.live(candidate) && !is_ghost(complex.tets[row])) {
      cell = candidate;
      break;
    }
  }
  if (cell < 0) {
    PHX_CHECK(cell >= 0);
    return;
  }
  const Tetrahedron proposal = complex.tets[static_cast<std::size_t>(cell)];
  CavityEdit rejected_edit(complex);
  rejected_edit.begin();
  PHX_CHECK(rejected_edit.remove(cell) == EditStatus::kOk);
  PHX_CHECK(rejected_edit.add(
      proposal.v, nullptr, complex.region(cell)) == EditStatus::kOk);
  const EditStatus rejected =
      rejected_edit.validate([](const int32_t*) { return false; });
  PHX_CHECK(rejected != EditStatus::kOk);
  rejected_edit.rollback();
  PHX_CHECK(complex.audit());
  PHX_CHECK(complex.finite_count == before_finite_count);
  PHX_CHECK(complex.free_slots == before_free_slots);
  PHX_CHECK(same_tet_bank(before_rollback));
  std::size_t constraint_row = 0;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    PHX_CHECK(complex.region(static_cast<int32_t>(row)) == before_regions[row]);
    for (int slot = 0; slot < 4; ++slot) {
      PHX_CHECK(complex.constraint(static_cast<int32_t>(row), slot) ==
                before_constraints[constraint_row++]);
    }
  }
}

void test_external_cavity_replacement_reseeds_incremental_locate() {
  const std::array<double, 18> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.25, 0.25, 1.0, 0.25, 0.25, -1.0, 0.2, 0.2, 0.1};
  Triangulation3D triangulation(points.data(), nullptr, 5);
  TetrahedralComplex& complex = triangulation.complex();
  complex = bipyramid(points.data());
  PHX_CHECK(complex.audit());

  int32_t face[3];
  oriented_facet(complex.tets[0].v, 3, face);
  std::array<std::array<int32_t, 4>, 3> flipped{};
  for (int k = 0; k < 3; ++k) {
    auto& tet = flipped[static_cast<std::size_t>(k)];
    tet = {face[k], face[(k + 1) % 3], 3, 4};
    if (orient3d(points.data() + 3 * tet[0], points.data() + 3 * tet[1],
                 points.data() + 3 * tet[2], points.data() + 3 * tet[3]) < 0) {
      std::swap(tet[0], tet[1]);
    }
  }
  CavityEdit expand(complex);
  expand.begin();
  PHX_CHECK(expand.remove(0) == EditStatus::kOk);
  PHX_CHECK(expand.remove(1) == EditStatus::kOk);
  for (const auto& tet : flipped) {
    PHX_CHECK(expand.add(tet.data(), nullptr, kNoRegion) == EditStatus::kOk);
  }
  if (expand.validate([&](const int32_t* vertices) {
        return orient3d(points.data() + 3 * vertices[0],
                        points.data() + 3 * vertices[1],
                        points.data() + 3 * vertices[2],
                        points.data() + 3 * vertices[3]) > 0;
      }) != EditStatus::kOk ||
      expand.commit() != EditStatus::kOk) {
    PHX_CHECK(false);
    return;
  }
  if (expand.created().size() != 3) {
    PHX_CHECK(expand.created().size() == 3);
    return;
  }
  const std::array<int32_t, 3> flipped_ids{
      expand.created()[0], expand.created()[1], expand.created()[2]};
  PHX_CHECK(complex.audit());

  const std::array<std::array<int32_t, 4>, 2> original{{
      {0, 1, 2, 3}, {0, 2, 1, 4}}};
  const int32_t stale_hint = flipped_ids[2];
  PHX_CHECK(triangulation.set_hint(stale_hint));
  CavityEdit contract(complex);
  contract.begin();
  for (int32_t cell : flipped_ids) PHX_CHECK(contract.remove(cell) == EditStatus::kOk);
  for (const auto& tet : original) {
    PHX_CHECK(contract.add(tet.data(), nullptr, kNoRegion) == EditStatus::kOk);
  }
  if (contract.validate([&](const int32_t* vertices) {
        return orient3d(points.data() + 3 * vertices[0],
                        points.data() + 3 * vertices[1],
                        points.data() + 3 * vertices[2],
                        points.data() + 3 * vertices[3]) > 0;
      }) != EditStatus::kOk ||
      contract.commit() != EditStatus::kOk) {
    PHX_CHECK(false);
    return;
  }
  PHX_CHECK(!complex.live(stale_hint));
  PHX_CHECK(complex.audit());

  PHX_CHECK(triangulation.insert_point_slot(5, 5));
  triangulation.rebind(points.data(), 6);
  std::array<char, 6> alive{1, 1, 1, 1, 1, 1};
  const int32_t insert_status = triangulation.insert(5, alive);
  if (insert_status != PHX_MC_OK) {
    std::fprintf(stderr, "post-cavity incremental insertion failed: status=%d\n",
                 insert_status);
    PHX_CHECK(insert_status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(complex.audit());
  PHX_CHECK(triangulation.locate(points.data() + 15) >= 0);
  bool inserted = false;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const Tetrahedron& tet = complex.tets[row];
    if (complex.live(static_cast<int32_t>(row)) && !is_ghost(tet)) {
      inserted = inserted || vertex_slot(tet, 5) >= 0;
    }
  }
  PHX_CHECK(inserted);
  const int32_t located = triangulation.locate(points.data() + 15);
  if (located < 0 || !triangulation.set_hint(located)) {
    PHX_CHECK(located >= 0);
    return;
  }
  const Tetrahedron valid_tet = complex.tets[static_cast<std::size_t>(located)];
  complex.tets[static_cast<std::size_t>(located)].v[1] = kDeadVertex;
  PHX_CHECK(triangulation.locate(points.data() + 15) < 0);
  complex.tets[static_cast<std::size_t>(located)] = valid_tet;
  PHX_CHECK(complex.audit());

  const std::vector<Tetrahedron> before_tets = complex.tets;
  const std::vector<int32_t> before_free = complex.free_slots;
  const std::vector<int32_t> before_labels = labels_of(complex);
  const int64_t before_finite = complex.finite_count;
  int32_t cell = -1;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const int32_t candidate = static_cast<int32_t>(row);
    if (complex.live(candidate) && !is_ghost(complex.tets[row])) {
      cell = candidate;
      break;
    }
  }
  if (cell < 0) {
    PHX_CHECK(cell >= 0);
    return;
  }
  const Tetrahedron proposal = complex.tets[static_cast<std::size_t>(cell)];
  CavityEdit rejected(complex);
  rejected.begin();
  PHX_CHECK(rejected.remove(cell) == EditStatus::kOk);
  PHX_CHECK(rejected.add(proposal.v, nullptr, kNoRegion) == EditStatus::kOk);
  PHX_CHECK(rejected.validate([](const int32_t*) { return false; }) !=
            EditStatus::kOk);
  rejected.rollback();
  PHX_CHECK(same_complex(complex, before_tets, before_free, before_labels,
                         before_finite));
  PHX_CHECK(complex.audit());
  PHX_CHECK(triangulation.locate(points.data() + 15) >= 0);
}

// A constrained face around the short edge ab (between the cells abcd and
// abec of its three-cell ring) blocks the cavity of a point next to ab, but
// the cavity reaches the other side around ab: the cone would erase the
// constrained face (its star link fails after the topology write). The
// insertion is refused before any write and leaves the complex unchanged.
void test_insertion_refuses_a_cavity_around_a_constrained_face() {
  const std::array<double, 18> points{
      0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0,
      -0.5, 0.85, 0.0, -0.5, -0.85, 0.0, 0.001, 0.0005, 0.0};
  const std::array<int32_t, 5> order{0, 1, 2, 3, 4};
  std::array<char, 6> alive{1, 1, 1, 1, 1, 1};
  Triangulation3D triangulation(points.data(), nullptr, 6);
  PHX_CHECK(triangulation.build(order, alive) == PHX_MC_OK);
  TetrahedralComplex& complex = triangulation.complex();
  complex.enable_labels();
  int sides = 0;
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const auto cell = static_cast<int32_t>(row);
    const Tetrahedron& tet = complex.tets[row];
    if (!complex.live(cell) || is_ghost(tet) || vertex_slot(tet, 0) < 0 ||
        vertex_slot(tet, 1) < 0 || vertex_slot(tet, 2) < 0) {
      continue;
    }
    for (int slot = 0; slot < 4; ++slot) {
      if (tet.v[slot] > 2) complex.set_facet_constraint(cell, slot, 0);
    }
    ++sides;
  }
  PHX_CHECK(sides == 2);
  PHX_CHECK(complex.finite_count == 3);
  PHX_CHECK(complex.audit());
  const std::vector<Tetrahedron> before_tets = complex.tets;
  const std::vector<int32_t> before_free = complex.free_slots;
  const std::vector<int32_t> before_labels = labels_of(complex);
  const int32_t status = triangulation.insert(5, alive);
  PHX_CHECK(status == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK(complex.audit());
  PHX_CHECK(complex.tets.size() == before_tets.size() && complex.free_slots == before_free &&
            labels_of(complex) == before_labels && complex.finite_count == 3);
  for (std::size_t row = 0; row < before_tets.size(); ++row) {
    for (int k = 0; k < 4; ++k) {
      PHX_CHECK(complex.tets[row].v[k] == before_tets[row].v[k] &&
                complex.tets[row].n[k] == before_tets[row].n[k]);
    }
  }
  PHX_CHECK(alive[5] != 0);
}

void test_source_preparation_refuses_nonplanar_and_self_intersecting_polygons() {
  const std::array<int64_t, 2> offsets{0, 4};
  const std::array<int32_t, 4> loop{0, 1, 2, 3};
  const std::array<int32_t, 1> polygon_facets{0};
  const std::array<int32_t, 2> facet_regions{-1, 0};
  PlcInput input;
  input.point_count = 4;
  input.polygon_count = 1;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = loop.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 1;
  input.facet_regions = facet_regions.data();
  PlcRecovery source;

  const std::array<double, 12> nonplanar{
      0, 0, 0, 1, 0, 0, 1, 1, 0.125, 0, 1, 0};
  input.points = nonplanar.data();
  PHX_CHECK(prepare_plc_source(input, source) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(source.failure.reason == kPlcInvalidPolygon);

  const std::array<double, 12> bowtie{
      0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0};
  input.points = bowtie.data();
  PHX_CHECK(prepare_plc_source(input, source) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(source.failure.reason == kPlcInvalidPolygon);
}


void test_recover_plc_publishes_a_single_region_tetrahedron() {
  const std::array<double, 12> points{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::array<int64_t, 5> offsets{0, 3, 6, 9, 12};
  const std::array<int32_t, 12> vertices{
      1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1};
  const std::array<int32_t, 4> polygon_facets{0, 1, 2, 3};
  const std::array<int32_t, 8> facet_regions{-1, 1, -1, 1, -1, 1, -1, 1};
  PlcInput input;
  input.point_count = 4;
  input.points = points.data();
  input.polygon_count = 4;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 4;
  input.facet_regions = facet_regions.data();

  PlcRecovery result;
  PHX_CHECK(recover_plc(input, result) == PHX_MC_OK);
  PHX_CHECK(result.points.size() == points.size());
  PHX_CHECK(result.tets.size() == 4);
  PHX_CHECK(result.tet_regions == std::vector<int32_t>{1});
  PHX_CHECK(result.faces.size() == 12);
  PHX_CHECK(result.face_sources.size() == 4);
  PHX_CHECK(result.face_orientations == std::vector<int8_t>(4, 1));
  PHX_CHECK(result.segments.size() == 12);
  PHX_CHECK(result.segment_sources.size() == 6);
  PHX_CHECK(result.protection.size() == 4);
  PHX_CHECK(result.witnesses.size() == 4);
}

// A unit-square prism whose top facet lies on z = 1 + 0.1 x + 0.3 y: its six
// correctly rounded top vertices are not exactly coplanar, so only the
// declared plane groups the six top triangles (their five internal diagonals
// stop being PLC edges). Each vertex is admitted by its rounding box alone;
// a vertex lifted by 1e-6 is refused unless the tolerance covers it.
void test_declared_facet_plane_groups_rounded_triangles() {
  std::array<double, 30> points{
      0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
      0, 0, 0x1.0p+0, 1, 0, 0x1.199999999999ap+0, 1, 1, 0x1.6666666666666p+0,
      0, 1, 0x1.4cccccccccccdp+0, 0.5, 0.5, 0x1.3333333333333p+0,
      0x1.3333333333333p-2, 0x1.6666666666666p-1, 0x1.3d70a3d70a3d7p+0};
  const std::array<int64_t, 12> offsets{0, 4, 8, 12, 16, 20, 23, 26, 29, 32, 35, 38};
  const std::array<int32_t, 38> vertices{0, 3, 2, 1, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6,
                                         3, 0, 4, 7, 4, 5, 8, 5, 6, 8, 6, 7, 9, 7, 4, 9,
                                         6, 9, 8, 4, 8, 9};
  const std::array<int32_t, 11> polygon_facets{0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5};
  const std::array<int32_t, 12> facet_regions{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0};
  std::array<double, 54> planes{};
  planes.fill(std::numeric_limits<double>::quiet_NaN());
  const std::array<double, 9> top{0, 0, 1, 1, 0, 0.1, 0, 1, 0.3};
  std::copy(top.begin(), top.end(), planes.begin() + 45);
  PlcInput input;
  input.point_count = 10;
  input.points = points.data();
  input.polygon_count = 11;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 6;
  input.facet_regions = facet_regions.data();
  input.policy = BoundaryPolicy::kFixed;

  PlcRecovery undeclared;
  PHX_CHECK(prepare_plc_source(input, undeclared) == PHX_MC_OK);
  PHX_CHECK(undeclared.plc_edges.size() == 2 * 17);
  input.facet_planes = planes.data();
  PlcRecovery declared;
  PHX_CHECK(prepare_plc_source(input, declared) == PHX_MC_OK);
  PHX_CHECK(declared.plc_edges.size() == 2 * 12);
  PlcRecovery recovered;
  PHX_CHECK(recover_plc(input, recovered) == PHX_MC_OK);
  PHX_CHECK(recovered.points.size() == points.size());
  PHX_CHECK(recovered.segments.size() == 2 * 12);

  points[29] += 1e-6;
  PlcRecovery refused;
  PHX_CHECK(recover_plc(input, refused) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(refused.failure.reason == kPlcSourceDeviation);
  PHX_CHECK(refused.failure.first_kind == kPlcFacet && refused.failure.first == 5);
  PHX_CHECK(refused.failure.second_kind == kPlcPoint && refused.failure.second == 9);
  PHX_CHECK(refused.source_refusal.certified_deviation > 9.5e-7 &&
            refused.source_refusal.certified_deviation < 9.6e-7);
  const std::array<double, 6> tolerances{0, 0, 0, 0, 0, 1e-5};
  input.facet_plane_tolerances = tolerances.data();
  PlcRecovery tolerated;
  PHX_CHECK(recover_plc(input, tolerated) == PHX_MC_OK);

  planes[52] = 0.0;
  planes[51] = 0.0;
  planes[53] = 0.0;  // second axis (0, 0, 0): no plane
  PlcRecovery invalid;
  PHX_CHECK(prepare_plc_source(input, invalid) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(invalid.failure.reason == kPlcSourceDeviation && invalid.failure.first == 5);
}

void test_recover_fixed_triangulated_planar_facets() {
  const std::array<double, 24> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
      0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
  const std::array<int64_t, 13> offsets{
      0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36};
  const std::array<int32_t, 36> triangles{
      0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6,
      0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3,
      0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5};
  const std::array<int32_t, 12> triangle_facets{
      0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  const std::array<int32_t, 12> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
  PlcInput input;
  input.point_count = 8;
  input.points = points.data();
  input.polygon_count = 12;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = triangle_facets.data();
  input.facet_count = 6;
  input.facet_regions = facet_regions.data();
  input.policy = BoundaryPolicy::kFixed;

  PlcRecovery result;
  PHX_CHECK(recover_plc(input, result) == PHX_MC_OK);
  PHX_CHECK(result.points.size() == points.size());
  PHX_CHECK(!result.tets.empty());
  PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                        [](int32_t region) { return region == 1; }));
  PHX_CHECK(result.faces.size() >= 36 && result.faces.size() % 3 == 0);
  PHX_CHECK(result.face_sources.size() == result.faces.size() / 3);
  std::array<int32_t, 6> source_face_counts{};
  for (std::size_t face = 0; face < result.face_sources.size(); ++face) {
    const int32_t source = result.face_sources[face];
    PHX_CHECK(source >= 0 && source < static_cast<int32_t>(source_face_counts.size()));
    if (source >= 0 && source < static_cast<int32_t>(source_face_counts.size())) {
      ++source_face_counts[static_cast<std::size_t>(source)];
    }
    PHX_CHECK(result.face_orientations[face] == 1);
  }
  for (int32_t count : source_face_counts) PHX_CHECK(count > 0);
}

void test_recover_fixed_warped_facet_edge_without_steiner() {
  const std::array<double, 24> points{
      0.020000000035049353, 0.019999999999999997, 0.019999999999999997,
      0.98, 0.02, 0.020000000158269752,
      0.9799999999649507, 0.98, 0.019999999999999997,
      0.020000000226534618, 0.9799999999999992, 0.02000000000000082,
      0.019999999999999997, 0.020000000035049353, 0.98,
      0.9799999999999992, 0.020000000226534618, 0.9799999999999992,
      0.9800000000000019, 0.9799999999649525, 0.9800000000000019,
      0.0200000000350475, 0.9800000000019, 0.9800000000019};
  const std::array<int64_t, 13> offsets{
      0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36};
  const std::array<int32_t, 36> triangles{
      5, 1, 0, 4, 0, 3, 7, 6, 4, 5, 6, 1,
      7, 4, 3, 6, 2, 1, 6, 5, 4, 2, 3, 0,
      1, 2, 0, 4, 5, 0, 6, 7, 2, 7, 3, 2};
  const std::array<int32_t, 12> triangle_facets{
      2, 5, 1, 3, 5, 3, 1, 0, 0, 2, 4, 4};
  const std::array<int32_t, 12> facet_regions{
      0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1};
  const std::array<int32_t, 24> segments{
      0, 1, 1, 2, 2, 3, 0, 3, 4, 5, 5, 6,
      6, 7, 4, 7, 0, 4, 1, 5, 2, 6, 3, 7};
  PlcInput input;
  input.point_count = 8;
  input.points = points.data();
  input.polygon_count = 12;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = triangle_facets.data();
  input.facet_count = 6;
  input.facet_regions = facet_regions.data();
  input.segment_count = 12;
  input.segments = segments.data();
  input.policy = BoundaryPolicy::kFixed;

  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) return;
  PHX_CHECK(result.points == std::vector<double>(points.begin(), points.end()));
  PHX_CHECK(result.tets.size() >= 4 && result.tets.size() % 4 == 0);
  PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                        [](int32_t region) { return region == 0; }));

  const auto has_facet_triangle = [&](std::array<int32_t, 3> expected) {
    std::sort(expected.begin(), expected.end());
    for (std::size_t face = 0; face < result.face_sources.size(); ++face) {
      if (result.face_sources[face] != 1) continue;
      std::array<int32_t, 3> actual{
          result.faces[3 * face], result.faces[3 * face + 1], result.faces[3 * face + 2]};
      std::sort(actual.begin(), actual.end());
      if (actual == expected) return true;
    }
    return false;
  };
  PHX_CHECK(has_facet_triangle({7, 6, 4}));
  PHX_CHECK(has_facet_triangle({6, 5, 4}));

  bool recovered_crease = false;
  for (std::size_t segment = 0; segment < result.segment_sources.size(); ++segment) {
    if (result.segment_sources[segment] != 14) continue;
    const int32_t first = result.segments[2 * segment];
    const int32_t second = result.segments[2 * segment + 1];
    recovered_crease = (first == 4 && second == 6) || (first == 6 && second == 4);
  }
  PHX_CHECK(recovered_crease);
}

void test_recover_fixed_non_delaunay_internal_facet_without_split() {
  const std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.25, 0.25, 1.0, 0.25, 0.25, -0.25};
  const std::array<int64_t, 8> offsets{0, 3, 6, 9, 12, 15, 18, 21};
  const std::array<int32_t, 21> triangles{
      1, 2, 3, 0, 3, 2, 0, 1, 3, 2, 1, 4,
      0, 4, 1, 0, 2, 4, 0, 1, 2};
  const std::array<int32_t, 7> triangle_facets{0, 1, 2, 3, 4, 5, 6};
  const std::array<int32_t, 14> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 2, -1, 2, -1, 2, 1, 2};

  const std::array<int32_t, 5> order{0, 1, 2, 3, 4};
  std::array<char, 5> alive{1, 1, 1, 1, 1};
  Triangulation3D unconstrained(points.data(), nullptr, 5);
  PHX_CHECK(unconstrained.build(order, alive) == PHX_MC_OK);
  bool has_source_facet = false;
  for (const Tetrahedron& tet : unconstrained.complex().tets) {
    if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
    for (int slot = 0; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(tet.v, slot, face);
      has_source_facet = has_source_facet || sorted_face(face) ==
          std::array<int32_t, 3>{0, 1, 2};
    }
  }
  PHX_CHECK(!has_source_facet);

  PlcInput input;
  input.point_count = 5;
  input.points = points.data();
  input.polygon_count = 7;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = triangle_facets.data();
  input.facet_count = 7;
  input.facet_regions = facet_regions.data();
  input.policy = BoundaryPolicy::kFixed;

  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  if (status != PHX_MC_OK) {
    std::fprintf(stderr,
                 "fixed non-Delaunay facet recovery failed: status=%d reason=%d "
                 "first=(%d,%lld) second=(%d,%lld) source_refusal=(%.17g,%.17g)\n",
                 status, result.failure.reason, result.failure.first_kind,
                 static_cast<long long>(result.failure.first), result.failure.second_kind,
                 static_cast<long long>(result.failure.second),
                 result.source_refusal.certified_deviation,
                 result.source_refusal.declared_bound);
    PHX_CHECK(status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(result.points == std::vector<double>(points.begin(), points.end()));
  PHX_CHECK(result.tets.size() % 4 == 0);
  PHX_CHECK(result.tet_regions.size() == result.tets.size() / 4);
  PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                        [](int32_t region) { return region == 1 || region == 2; }));

  std::map<std::array<int32_t, 3>, int32_t> incidence;
  Expansion total_volume;
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    const int32_t* tet = result.tets.data() + 4 * cell;
    const double* a = result.points.data() + 3 * tet[0];
    const double* b = result.points.data() + 3 * tet[1];
    const double* c = result.points.data() + 3 * tet[2];
    const double* d = result.points.data() + 3 * tet[3];
    const Expansion volume = orient3d_exact(a, b, c, d);
    PHX_CHECK(volume.sign() > 0);
    total_volume = total_volume + volume;
    for (int slot = 0; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(tet, slot, face);
      ++incidence[sorted_face(face)];
    }
  }
  PHX_CHECK((total_volume - Expansion(5.0 / 4.0)).sign() == 0);
  for (const auto& [face, count] : incidence) {
    (void)face;
    PHX_CHECK(count == 1 || count == 2);
  }
  const std::array<int32_t, 3> interface_key{0, 1, 2};
  PHX_CHECK(incidence[interface_key] == 2);

  std::array<int32_t, 7> source_face_counts{};
  std::int32_t interface_count = 0;
  for (std::size_t face = 0; face < result.face_sources.size(); ++face) {
    const int32_t source = result.face_sources[face];
    PHX_CHECK(source >= 0 && source < static_cast<int32_t>(source_face_counts.size()));
    if (source >= 0 && source < static_cast<int32_t>(source_face_counts.size())) {
      ++source_face_counts[static_cast<std::size_t>(source)];
    }
    PHX_CHECK(result.face_orientations[face] == 1);
    if (source != 6) continue;
    ++interface_count;
    // Same oriented triangle as the source (any cyclic rotation).
    const int32_t expected[3] = {0, 1, 2};
    PHX_CHECK(FacetKey::of(result.faces.data() + 3 * face) == FacetKey::of(expected));
  }
  for (int32_t count : source_face_counts) PHX_CHECK(count == 1);
  PHX_CHECK(interface_count == 1);
}

void test_recover_plc_failure_preserves_previous_publication() {
  const std::array<double, 12> points{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::array<int64_t, 5> offsets{0, 3, 6, 9, 12};
  const std::array<int32_t, 12> vertices{
      1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1};
  const std::array<int32_t, 4> polygon_facets{0, 1, 2, 3};
  const std::array<int32_t, 8> facet_regions{-1, 1, -1, 1, -1, 1, -1, 1};
  const std::array<double, 3> outside_seed{4, 4, 4};
  const std::array<int32_t, 1> seed_regions{1};
  PlcInput input;
  input.point_count = 4;
  input.points = points.data();
  input.polygon_count = 4;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 4;
  input.facet_regions = facet_regions.data();
  input.seed_count = 1;
  input.seeds = outside_seed.data();
  input.seed_regions = seed_regions.data();

  PlcRecovery result;
  result.points = {11.0, 12.0, 13.0};
  result.tets = {17, 18, 19, 20};
  const int32_t status = recover_plc(input, result);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(result.points == std::vector<double>({11.0, 12.0, 13.0}));
  PHX_CHECK(result.tets == std::vector<int32_t>({17, 18, 19, 20}));
  PHX_CHECK(result.failure.reason == kPlcInvalidSeed);
}


void test_recover_plc_classifies_two_regions_across_a_shared_facet() {
  const double height = std::sqrt(2.0 / 3.0);
  const double third = std::sqrt(3.0) / 2.0;
  const std::array<double, 15> points{
      0, 0, 0, 1, 0, 0, 0.5, third, 0,
      0.5, third / 3.0, height, 0.5, third / 3.0, -height};
  const std::array<int64_t, 8> offsets{0, 3, 6, 9, 12, 15, 18, 21};
  const std::array<int32_t, 21> vertices{
      1, 2, 3, 0, 3, 2, 0, 1, 3, 2, 1, 4,
      0, 4, 1, 0, 2, 4, 0, 1, 2};
  const std::array<int32_t, 7> polygon_facets{0, 1, 2, 3, 4, 5, 6};
  const std::array<int32_t, 14> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 2, -1, 2, -1, 2, 1, 2};
  PlcInput input;
  input.point_count = 5;
  input.points = points.data();
  input.polygon_count = 7;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 7;
  input.facet_regions = facet_regions.data();

  PlcRecovery result;
  PHX_CHECK(recover_plc(input, result) == PHX_MC_OK);
  PHX_CHECK(result.tets.size() >= 8 && result.tets.size() % 4 == 0);
  std::vector<int32_t> regions = result.tet_regions;
  std::sort(regions.begin(), regions.end());
  regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
  PHX_CHECK(regions == std::vector<int32_t>({1, 2}));
  PHX_CHECK(result.faces.size() >= 21 && result.faces.size() % 3 == 0);
  std::vector<int32_t> face_sources = result.face_sources;
  std::sort(face_sources.begin(), face_sources.end());
  face_sources.erase(std::unique(face_sources.begin(), face_sources.end()), face_sources.end());
  PHX_CHECK(face_sources == std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6}));
}

void test_recover_plc_handles_sliver_geometry_and_explicit_fixed_edge() {
  const std::array<double, 12> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5, 0.5, 0x1p-20};
  const std::array<int64_t, 5> offsets{0, 3, 6, 9, 12};
  const std::array<int32_t, 12> vertices{
      1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1};
  const std::array<int32_t, 4> polygon_facets{0, 1, 2, 3};
  const std::array<int32_t, 8> facet_regions{-1, 1, -1, 1, -1, 1, -1, 1};
  const std::array<int32_t, 2> segment{0, 1};
  PlcInput input;
  input.point_count = 4;
  input.points = points.data();
  input.polygon_count = 4;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 4;
  input.facet_regions = facet_regions.data();
  input.segment_count = 1;
  input.segments = segment.data();
  input.policy = BoundaryPolicy::kFixed;

  PlcRecovery result;
  PHX_CHECK(recover_plc(input, result) == PHX_MC_OK);
  PHX_CHECK(result.tets.size() == 4);
  PHX_CHECK(result.segments.size() == 12);
  PHX_CHECK(result.segment_sources.size() == 6);
  PHX_CHECK(std::any_of(result.protection.begin(), result.protection.end(),
                        [](double radius) { return radius > 0.0; }));
}

void test_recover_plc_nonconvex_prism() {
  const std::array<double, 36> points{
      0, 0, 0, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0,
      0, 0, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 0, 2, 1};
  const std::array<int64_t, 9> offsets{0, 6, 12, 16, 20, 24, 28, 32, 36};
  const std::array<int32_t, 36> vertices{
      5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11,
      0, 1, 7, 6, 1, 2, 8, 7, 2, 3, 9, 8,
      3, 4, 10, 9, 4, 5, 11, 10, 5, 0, 6, 11};
  const std::array<int32_t, 8> polygon_facets{0, 1, 2, 3, 4, 5, 6, 7};
  const std::array<int32_t, 16> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
  PlcInput input;
  input.point_count = 12;
  input.points = points.data();
  input.polygon_count = 8;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 8;
  input.facet_regions = facet_regions.data();

  PlcRecovery result;
  PHX_CHECK(recover_plc(input, result) == PHX_MC_OK);
  PHX_CHECK(!result.tets.empty());
  PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                        [](int32_t region) { return region == 1; }));
  PHX_CHECK(result.faces.size() % 3 == 0);
  const std::size_t face_count = result.faces.size() / 3;
  PHX_CHECK(result.face_sources.size() == face_count);
  PHX_CHECK(result.face_orientations.size() == face_count);
  std::array<int32_t, 8> faces_per_facet{};
  for (std::size_t face = 0; face < face_count; ++face) {
    const int32_t facet = result.face_sources[face];
    PHX_CHECK(facet >= 0 && facet < static_cast<int32_t>(faces_per_facet.size()));
    PHX_CHECK(result.face_orientations[face] == 1);
    if (facet >= 0 && facet < static_cast<int32_t>(faces_per_facet.size())) {
      ++faces_per_facet[static_cast<std::size_t>(facet)];
    }
  }
  for (const int32_t count : faces_per_facet) PHX_CHECK(count > 0);
  for (std::size_t tet = 0; tet < result.tets.size() / 4; ++tet) {
    double center[3] = {0, 0, 0};
    for (int corner = 0; corner < 4; ++corner) {
      const int32_t vertex = result.tets[4 * tet + static_cast<std::size_t>(corner)];
      for (int axis = 0; axis < 3; ++axis) {
        center[axis] += result.points[3 * static_cast<std::size_t>(vertex) + axis] / 4.0;
      }
    }
    PHX_CHECK(!(center[0] > 1.0 && center[1] > 1.0));
  }
}

void test_recover_plc_corner_scaffold_tracks_inserted_points_and_region_interface() {
  const std::array<double, 15> points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.25, 0.25, 1.0, 0.25, 0.25, -0.25};
  const std::array<int64_t, 8> offsets{0, 3, 6, 9, 12, 15, 18, 21};
  const std::array<int32_t, 21> triangles{
      1, 2, 3, 0, 3, 2, 0, 1, 3, 2, 1, 4, 0, 4, 1, 0, 2, 4, 0, 1, 2};
  const std::array<int32_t, 7> polygon_facets{0, 1, 2, 3, 4, 5, 6};
  const std::array<int32_t, 14> facet_regions{
      -1, 1, -1, 1, -1, 1, -1, 2, -1, 2, -1, 2, 1, 2};
  const std::array<double, 6> seeds{
      0.25, 0.25, 0.25, 0.25, 0.25, -0.0625};
  const std::array<int32_t, 2> seed_regions{1, 2};
  PlcInput input;
  input.point_count = 5;
  input.points = points.data();
  input.polygon_count = 7;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 7;
  input.facet_regions = facet_regions.data();
  input.seed_count = 2;
  input.seeds = seeds.data();
  input.seed_regions = seed_regions.data();
  input.policy = BoundaryPolicy::kConforming;

  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  if (status != PHX_MC_OK) {
    std::fprintf(stderr,
                 "corner-scaffold region regression failed: status=%d reason=%d "
                 "first=(%d,%lld) second=(%d,%lld)\n",
                 status, result.failure.reason, result.failure.first_kind,
                 static_cast<long long>(result.failure.first), result.failure.second_kind,
                 static_cast<long long>(result.failure.second));
  }
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) return;

  PHX_CHECK(result.tets.size() == 4 * result.tet_regions.size());
  PHX_CHECK(result.vertex_dimension.size() == result.points.size() / 3);
  PHX_CHECK(result.witnesses.size() == result.points.size() / 3);

  std::array<int32_t, 7> source_face_counts{};
  std::map<std::array<int32_t, 3>, int32_t> face_region_masks;
  std::array<Expansion, 2> region_six_volumes{};
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    const int32_t* tet = result.tets.data() + 4 * cell;
    const int32_t region = result.tet_regions[cell];
    if (region != 1 && region != 2) {
      PHX_CHECK(false);
      continue;
    }
    const Expansion six_volume = orient3d_exact(
        result.points.data() + 3 * static_cast<std::size_t>(tet[0]),
        result.points.data() + 3 * static_cast<std::size_t>(tet[1]),
        result.points.data() + 3 * static_cast<std::size_t>(tet[2]),
        result.points.data() + 3 * static_cast<std::size_t>(tet[3]));
    PHX_CHECK(six_volume.sign() > 0);
    region_six_volumes[static_cast<std::size_t>(region - 1)] =
        region_six_volumes[static_cast<std::size_t>(region - 1)] + six_volume;
    for (int slot = 0; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(tet, slot, face);
      face_region_masks[sorted_face(face)] |= 1 << (region - 1);
    }
  }
  PHX_CHECK((region_six_volumes[0] - Expansion(1.0)).sign() == 0);
  PHX_CHECK((region_six_volumes[1] - Expansion(0.25)).sign() == 0);
  PHX_CHECK(result.face_sources.size() == result.face_orientations.size());
  for (std::size_t face = 0; face < result.face_sources.size(); ++face) {
    const int32_t source = result.face_sources[face];
    PHX_CHECK(source >= 0 && source < static_cast<int32_t>(source_face_counts.size()));
    PHX_CHECK(result.face_orientations[face] == 1);
    if (source < 0 || source >= static_cast<int32_t>(source_face_counts.size())) continue;
    ++source_face_counts[static_cast<std::size_t>(source)];
    const auto key = sorted_face(result.faces.data() + 3 * face);
    const auto found = face_region_masks.find(key);
    PHX_CHECK(found != face_region_masks.end());
    if (found == face_region_masks.end()) continue;
    const int32_t expected_mask = source == 6 ? 3 : (source < 3 ? 1 : 2);
    PHX_CHECK(found->second == expected_mask);
  }
  for (int32_t count : source_face_counts) PHX_CHECK(count > 0);
}

void test_recover_plc_removes_a_closed_cavity() {
  const std::array<double, 48> points{
      0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 0,
      0, 0, 4, 4, 0, 4, 0, 4, 4, 4, 4, 4,
      1, 1, 1, 3, 1, 1, 1, 3, 1, 3, 3, 1,
      1, 1, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3};
  const std::array<int64_t, 13> offsets{
      0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48};
  const std::array<int32_t, 48> vertices{
      0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4,
      2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5,
      8, 9, 11, 10, 12, 14, 15, 13, 8, 12, 13, 9,
      10, 11, 15, 14, 8, 10, 14, 12, 9, 13, 15, 11};
  const std::array<int32_t, 12> polygon_facets{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::array<int32_t, 24> facet_regions{};
  for (std::size_t facet = 0; facet < polygon_facets.size(); ++facet) {
    facet_regions[2 * facet] = -1;
    facet_regions[2 * facet + 1] = 1;
  }
  PlcInput input;
  input.point_count = 16;
  input.points = points.data();
  input.polygon_count = 12;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = polygon_facets.data();
  input.facet_count = 12;
  input.facet_regions = facet_regions.data();

  PlcRecovery result;
  PHX_CHECK(recover_plc(input, result) == PHX_MC_OK);
  PHX_CHECK(!result.tets.empty());
  PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                        [](int32_t region) { return region == 1; }));
  for (std::size_t tet = 0; tet < result.tets.size() / 4; ++tet) {
    double center[3] = {0, 0, 0};
    for (int corner = 0; corner < 4; ++corner) {
      const int32_t vertex = result.tets[4 * tet + static_cast<std::size_t>(corner)];
      for (int axis = 0; axis < 3; ++axis) {
        center[axis] += result.points[3 * static_cast<std::size_t>(vertex) + axis] / 4.0;
      }
    }
    PHX_CHECK(!(center[0] > 1.0 && center[0] < 3.0 && center[1] > 1.0 &&
                center[1] < 3.0 && center[2] > 1.0 && center[2] < 3.0));
  }
}

void check_grouped_material_interface_with_cavity(bool embedded_segment) {
  // Unchanged native_family_matrix._cavity_plc: two unit boxes, a shared
  // material interface and an exact-dyadic half-unit cavity in the right box.
  // The interface is planar but is not initially a mesh surface. Grouping
  // polygons must not bypass the facet recovery needed to construct it.
  const std::array<double, 66> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
      0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
      2, 0, 0, 2, 1, 0, 2, 0, 1, 2, 1, 1,
      1.25, .25, .25, 1.75, .25, .25, 1.25, .75, .25, 1.75, .75, .25,
      1.25, .25, .75, 1.75, .25, .75, 1.25, .75, .75, 1.75, .75, .75,
      .25, .25, .25, .75, .75, .75};
  const std::array<int32_t, 68> vertices{
      0, 2, 3, 1, 1, 3, 9, 8,
      4, 5, 7, 6, 5, 10, 11, 7,
      0, 1, 5, 4, 1, 8, 10, 5,
      2, 6, 7, 3, 3, 7, 11, 9,
      0, 4, 6, 2, 8, 9, 11, 10,
      1, 3, 7, 5,
      12, 14, 15, 13, 16, 17, 19, 18, 12, 13, 17, 16,
      14, 18, 19, 15, 12, 16, 18, 14, 13, 15, 19, 17};
  const std::array<int32_t, 17> facets{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2,
                                       3, 3, 3, 3, 3, 3};
  const std::array<int32_t, 8> regions{-1, 0, -1, 1, 1, 0, 1, -1};
  const std::array<int32_t, 2> segment{20, 21};
  std::array<int64_t, 18> offsets{};
  for (std::size_t row = 0; row < offsets.size(); ++row) offsets[row] = 4 * row;
  PlcInput input;
  input.point_count = embedded_segment ? 22 : 20;
  input.points = points.data();
  input.polygon_count = 17;
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = vertices.data();
  input.polygon_facets = facets.data();
  input.facet_count = 4;
  input.facet_regions = regions.data();
  input.segment_count = embedded_segment ? 1 : 0;
  input.segments = embedded_segment ? segment.data() : nullptr;
  input.policy = BoundaryPolicy::kConforming;
  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  if (status != PHX_MC_OK) {
    std::fprintf(stderr, "grouped material cavity recovery: status=%d reason=%d first=(%d,%lld)\n",
                 status, result.failure.reason, result.failure.first_kind,
                 static_cast<long long>(result.failure.first));
  }
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) return;
  PHX_CHECK(!result.tets.empty());
  std::array<Expansion, 2> six_volumes;
  std::map<std::array<int32_t, 3>, int> face_masks;
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    const int32_t region = result.tet_regions[cell];
    PHX_CHECK(region == 0 || region == 1);
    if (region != 0 && region != 1) return;
    const int32_t* tet = result.tets.data() + 4 * cell;
    const Expansion volume = orient3d_exact(
        result.points.data() + 3 * tet[0], result.points.data() + 3 * tet[1],
        result.points.data() + 3 * tet[2], result.points.data() + 3 * tet[3]);
    PHX_CHECK(volume.sign() > 0);
    six_volumes[region] = six_volumes[region] + volume;
    for (int slot = 0; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(tet, slot, face);
      face_masks[sorted_face(face)] |= 1 << region;
    }
  }
  PHX_CHECK((six_volumes[0] - Expansion(6.0)).sign() == 0);
  PHX_CHECK((six_volumes[1] - Expansion(5.25)).sign() == 0);
  std::array<Expansion, 17> twice_areas;
  for (std::size_t row = 0; row < result.face_sources.size(); ++row) {
    const int32_t* face = result.faces.data() + 3 * row;
    const int32_t facet = result.face_sources[row];
    PHX_CHECK(facet >= 0 && facet < 4);
    PHX_CHECK(face_masks[sorted_face(face)] == (facet == 0 ? 1 : facet == 2 ? 3 : 2));
    int matches = 0;
    for (std::size_t polygon = 0; polygon < facets.size(); ++polygon) {
      if (facets[polygon] != facet) continue;
      const double* first = points.data() + 3 * vertices[4 * polygon];
      const double* second = points.data() + 3 * vertices[4 * polygon + 1];
      const double* third = points.data() + 3 * vertices[4 * polygon + 2];
      int axis = 0;
      while (first[axis] != second[axis] || first[axis] != third[axis]) ++axis;
      bool contained = true;
      for (int corner = 0; corner < 3; ++corner) {
        const double* point = result.points.data() + 3 * face[corner];
        contained = contained && point[axis] == first[axis];
        for (int direction = 0; direction < 3; ++direction) {
          double lower = first[direction], upper = first[direction];
          for (int vertex = 1; vertex < 4; ++vertex) {
            const double coordinate = points[3 * vertices[4 * polygon + vertex] + direction];
            lower = std::min(lower, coordinate);
            upper = std::max(upper, coordinate);
          }
          contained = contained && lower <= point[direction] && point[direction] <= upper;
        }
      }
      if (!contained) continue;
      ++matches;
      const int x = (axis + 1) % 3, y = (axis + 2) % 3;
      Expansion area;
      for (int corner = 0; corner < 3; ++corner) {
        const double* a = result.points.data() + 3 * face[corner];
        const double* b = result.points.data() + 3 * face[(corner + 1) % 3];
        area = area + Expansion::product(a[x], b[y]) - Expansion::product(a[y], b[x]);
      }
      PHX_CHECK(area.sign() != 0);
      twice_areas[polygon] = twice_areas[polygon] + (area.sign() < 0 ? -area : area);
    }
    PHX_CHECK(matches == 1);
  }
  for (std::size_t polygon = 0; polygon < facets.size(); ++polygon) {
    PHX_CHECK((twice_areas[polygon] - Expansion(polygon < 11 ? 2.0 : 0.5)).sign() == 0);
  }
  if (embedded_segment) {
    double total_x_length = 0.0;
    std::map<int32_t, int> degree;
    for (std::size_t row = 0; row < result.segment_sources.size(); ++row) {
      const int32_t source = result.segment_sources[row];
      if (source < 0 || 2 * static_cast<std::size_t>(source) + 1 >= result.plc_edges.size()) continue;
      const int32_t a = result.plc_edges[2 * static_cast<std::size_t>(source)];
      const int32_t b = result.plc_edges[2 * static_cast<std::size_t>(source) + 1];
      if (!((a == 20 && b == 21) || (a == 21 && b == 20))) continue;
      const int32_t u = result.segments[2 * row], v = result.segments[2 * row + 1];
      ++degree[u];
      ++degree[v];
      for (int32_t vertex : {u, v}) {
        const double* p = result.points.data() + 3 * vertex;
        PHX_CHECK(std::fabs(p[0] - p[1]) < 1e-12);
        PHX_CHECK(std::fabs(p[0] - p[2]) < 1e-12);
        PHX_CHECK(p[0] >= .25 && p[0] <= .75);
      }
      total_x_length += std::fabs(result.points[3 * u] - result.points[3 * v]);
    }
    PHX_CHECK(std::fabs(total_x_length - .5) < 1e-12);
    PHX_CHECK(degree[20] == 1 && degree[21] == 1);
    for (const auto& [vertex, count] : degree) {
      if (vertex != 20 && vertex != 21) PHX_CHECK(count == 2);
    }
  }
}

void test_grouped_material_interface_with_cavity_recovers_missing_planar_surface() {
  check_grouped_material_interface_with_cavity(false);
  // Exact authored PLC with protected interior segment. Its interface split
  // used to legalize against an artificial corner before inserting the new
  // source point, cycling between two triangulations.
  check_grouped_material_interface_with_cavity(true);
}

// Point 1455 of a 1552-point CAD PLC under FIXED: its ear-clipped wedge has
// no constrained Delaunay fill and no cavity vertex sees every wall, but a
// Steiner-free fill exists; the exhaustive search must find it. Vertices are
// relabelled 0..13; seven are witnessed split points with exact positions.
void test_suppression_wedge_without_delaunay_fill_is_searched() {
  constexpr int32_t kPlaceholder = std::numeric_limits<int32_t>::min() / 2;
  const std::array<double, 42> points{
      -0x1.1999999999998p+0, -0x1.6666666666664p-1, 0x1.bc52a869d13cfp+0,
      -0x1.1999999999998p+0, -0x1.3ee9e7ee94666p-1, 0x1.c6f1726f1c333p+0,
      -0x1.ed218e7d83891p-1, -0x1.333333333333p-2, 0x1.0cccccccccccep+1,
      -0x1.c09dc1ffdf5bp-1, -0x1.e7a1ea66251c8p-2, 0x1.0cccccccccccep+1,
      -0x1.f4ca60e969affp-1, -0x1.6cc31d3f40596p-3, 0x1.0cccccccccccep+1,
      -0x1.ff814acc7e089p-1, -0x1.89e6425e51188p-2, 0x1.01f66ae769103p+1,
      -0x1.a9c3fa71f4f7ap-1, -0x1.999999999998p-4, 0x1.1ad780719b708p+1,
      -0x1.1b461bb45155p-1, -0x1.eba02ab4e65f5p-2, 0x1.2220da22b5454p+1,
      -0x1.42eb7e2f660ep-1, -0x1.2679d7fe90b02p-1, 0x1.198f92899670fp+1,
      -0x1.1069bb721c59cp+0, -0x1.139060590573ap-1, 0x1.dc9e3639e3ac9p+0,
      -0x1.0a45bd91db2c5p+0, -0x1.0075e5475b2cbp-1, 0x1.e873511060528p+0,
      -0x1.0c3f75ef25046p+0, -0x1.ffd2211d1c739p-2, 0x1.e671de1f1f499p+0,
      -0x1.07fb18f19eff6p+0, -0x1.ea12f2a8e590dp-2, 0x1.eda55b599fabap+0,
      -0x1.0b83bc3d7f7a4p+0, -0x1.ff5ae71152c85p-2, 0x1.e729f80422e74p+0};
  const std::array<int32_t, 7> witnessed{5, 6, 7, 8, 9, 10, 11};
  const std::array<SourceStratum, 7> strata{
      SourceStratum::kSegment, SourceStratum::kSegment, SourceStratum::kSegment, SourceStratum::kSegment, SourceStratum::kSegment, SourceStratum::kFacet, SourceStratum::kSegment};
  const std::array<double, 63> corners{
      -0x1.1999999999998p+0, -0x1.3ee9e7ee94666p-1, 0x1.c6f1726f1c333p+0,
      -0x1.ed218e7d83891p-1, -0x1.333333333333p-2, 0x1.0cccccccccccep+1,
      0.0, 0.0, 0.0,
      -0x1.ed218e7d83891p-1, -0x1.333333333333p-2, 0x1.0cccccccccccep+1,
      -0x1.6666666666664p-1, 0x1.99999999999cp-4, 0x1.28e234166a143p+1,
      0.0, 0.0, 0.0,
      -0x1.6666666666664p-1, -0x1.5206911b4cc5dp-1, 0x1.11e4c21f9335p+1,
      -0x1.a04ba20478876p-2, -0x1.333333333333p-2, 0x1.325cf225d7559p+1,
      0.0, 0.0, 0.0,
      -0x1.6666666666664p-1, -0x1.5206911b4cc5dp-1, 0x1.11e4c21f9335p+1,
      -0x1.a04ba20478876p-2, -0x1.333333333333p-2, 0x1.325cf225d7559p+1,
      0.0, 0.0, 0.0,
      -0x1.1999999999998p+0, -0x1.3ee9e7ee94666p-1, 0x1.c6f1726f1c333p+0,
      -0x1.ed218e7d83891p-1, -0x1.333333333333p-2, 0x1.0cccccccccccep+1,
      0.0, 0.0, 0.0,
      -0x1.1999999999998p+0, -0x1.6666666666664p-1, 0x1.bc52a869d13cfp+0,
      -0x1.ed218e7d83891p-1, -0x1.333333333333p-2, 0x1.0cccccccccccep+1,
      -0x1.1999999999998p+0, -0x1.3ee9e7ee94666p-1, 0x1.c6f1726f1c333p+0,
      -0x1.1999999999998p+0, -0x1.3ee9e7ee94666p-1, 0x1.c6f1726f1c333p+0,
      -0x1.ed218e7d83891p-1, -0x1.333333333333p-2, 0x1.0cccccccccccep+1,
      0.0, 0.0, 0.0};
  const std::array<double, 14> parameters{
      0x1.79bd5fc7dda52p-1, 0x0p+0,
      0x1p-1, 0x0p+0,
      0x1p-1, 0x0p+0,
      0x1.e39bbcb4daeep-3, 0x0p+0,
      0x1.0c85407044b5bp-2, 0x0p+0,
      0x1.cp-2, 0x1.4p-2,
      0x1.8642a038225aep-2, 0x0p+0};
  const std::array<std::array<int32_t, 4>, 24> walls{{
      {0, 5, 2, kPlaceholder},
      {9, 10, 1, kPlaceholder},
      {11, 10, 9, kPlaceholder},
      {5, 10, 11, kPlaceholder},
      {5, 0, 10, kPlaceholder},
      {0, kPlaceholder, 2, 3},
      {8, 12, kPlaceholder, 7},
      {kPlaceholder, 6, 7, 12},
      {3, 8, kPlaceholder, 7},
      {3, kPlaceholder, 8, 12},
      {kPlaceholder, 6, 12, 5},
      {3, 6, 7, kPlaceholder},
      {12, kPlaceholder, 10, 3},
      {kPlaceholder, 11, 5, 12},
      {5, 6, 4, kPlaceholder},
      {2, 6, 3, kPlaceholder},
      {3, 0, kPlaceholder, 10},
      {10, kPlaceholder, 12, 13},
      {kPlaceholder, 11, 12, 13},
      {4, 6, 2, kPlaceholder},
      {5, 4, 2, kPlaceholder},
      {10, 1, kPlaceholder, 13},
      {9, 11, kPlaceholder, 13},
      {9, 1, 13, kPlaceholder}}};
  const std::array<std::array<int32_t, 4>, 24> marks{{
      {-1, -1, -1, 558},
      {-1, -1, -1, 558},
      {-1, -1, -1, 558},
      {-1, -1, -1, 558},
      {-1, -1, -1, 558},
      {-1, 559, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, 585},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1},
      {-1, -1, -1, -1}}};
  detail::SuppressionCavity cavity;
  cavity.points = points;
  cavity.witnessed = witnessed;
  cavity.witness_strata = strata;
  cavity.witness_corners = corners;
  cavity.witness_parameters = parameters;
  cavity.walls = walls;
  cavity.wall_marks = marks;
  cavity.placeholder = kPlaceholder;
  detail::SuppressionCavityFill fill;
  PHX_CHECK(detail::fill_suppression_cavity(cavity, fill) == PHX_MC_OK);
  PHX_CHECK(!fill.delaunay);
  PHX_CHECK(fill.filled);
  PHX_CHECK(fill.nodes > 0);
  if (!fill.filled) return;
  // Every wall side is closed by exactly one cell; every other cell face is
  // shared with exactly one cell seeing it reversed.
  std::map<FacetKey, int> sides;
  for (const auto& cell : fill.cells) {
    PHX_CHECK(std::find(cell.begin(), cell.end(), kPlaceholder) == cell.end());
    for (int slot = 0; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(cell.data(), slot, face);
      ++sides[FacetKey::of(face)];
    }
  }
  std::set<FacetKey> wall_sides;
  for (const auto& wall : walls) {
    const int slot = static_cast<int>(std::find(wall.begin(), wall.end(), kPlaceholder) - wall.begin());
    int32_t face[3];
    oriented_facet(wall.data(), slot, face);
    wall_sides.insert(FacetKey::of(face));
    PHX_CHECK(sides[FacetKey::of(face)] == 1);
  }
  for (const auto& [side, count] : sides) {
    PHX_CHECK(count == 1);
    if (!wall_sides.contains(side)) PHX_CHECK(sides.contains(side.reversed()));
  }
}

// A 26-triangle patch of the scale-2 torus reconstruction envelope (nearly
// flat offset faces), as an internal sheet of region 0 inside a box hull.
// Under FIXED the split point of the segment (15, 21) is suppressed last;
// vertices 15, 21, 20, 22 are coplanar up to an orientation of ~4e-22, so
// the star of the split point holds a sliver over a subface. Its wedge has
// no Steiner-free fill (the exhaustive search fails) and its kernel is
// pinched to the sliver's plane, so no binary64 cone point exists either;
// the point is removed only once the wedge is merged with the cells beyond
// its link faces. Before that the recovery refused FIXED_SEGMENT.
void test_suppression_merges_a_sliver_wedge_with_its_neighbours() {
  const std::vector<double> points{
      -0x1.321f529b7e48dp-2, 0x1.64ab155864700p-9, -0x1.9f044e4d6d58ep-4,
      -0x1.321f529b7e48dp-2, 0x1.44e4bff842afcp-8, -0x1.9f6d8f547cb34p-4,
      -0x1.1da47153d0345p-2, 0x1.64ab155864700p-9, -0x1.8b6cffe486f86p-4,
      -0x1.2ea4bfb8f9dcap-2, 0x1.90fa434d4d438p-8, -0x1.9aac371f2c0a0p-4,
      -0x1.1da47153d0345p-2, 0x1.3ff164bc0b59bp-7, -0x1.8bbdaebc7f731p-4,
      -0x1.1da47153d0345p-2, 0x1.74437725edd60p-6, -0x1.8d0ed94a2f94fp-4,
      -0x1.10a41fc60deb2p-2, -0x1.2c4e63cec09e0p-8, -0x1.74953c7234962p-4,
      -0x1.0929900c221fdp-2, 0x1.64ab155864700p-9, -0x1.62b433c85e1cdp-4,
      -0x1.0ec37ed33496ep-2, 0x1.64ab155864700p-9, -0x1.6d12b8a6cf44ep-4,
      -0x1.0ec37ed33496ep-2, 0x1.1aa48ab4c6656p-6, -0x1.6d12b8a6cf44ep-4,
      -0x1.0929900c221fdp-2, 0x1.74437725edd60p-6, -0x1.628a696721414p-4,
      -0x1.0f0ee6494c799p-2, 0x1.74437725edd60p-6, -0x1.6e40567f2ecfap-4,
      -0x1.0929900c221fdp-2, 0x1.417243acd1c5cp-5, -0x1.64ee3e9c504d6p-4,
      -0x1.015ee75c92b0dp-2, -0x1.1b18b1cfd4ba0p-6, -0x1.56aafd8a8568cp-4,
      -0x1.e95d5d88e816cp-3, -0x1.1b18b1cfd4ba0p-6, -0x1.21510e2381767p-4,
      -0x1.0574b7b52e89ap-2, 0x1.64ab155864700p-9, -0x1.56aafd8a8568cp-4,
      -0x1.02b3867e68436p-2, -0x1.677031e86de53p-7, -0x1.56aafd8a8568cp-4,
      -0x1.e95d5d88e816cp-3, -0x1.df29744103290p-10, -0x1.176176e794470p-4,
      -0x1.e95d5d88e816cp-3, 0x1.64ab155864700p-9, -0x1.1392d9c79b6fbp-4,
      -0x1.04f3d3509a726p-2, 0x1.74437725edd60p-6, -0x1.56aafd8a8568cp-4,
      -0x1.04f3d3509a726p-2, 0x1.bfc4b98e1d95ep-8, -0x1.56aafd8a8568cp-4,
      -0x1.e95d5d88e816cp-3, 0x1.38f5f1b6b472bp-6, -0x1.1392d9c79b6fbp-4,
      -0x1.e95d5d88e816cp-3, 0x1.74437725edd60p-6, -0x1.15966b59ebcccp-4,
      -0x1.03a2bd51870c3p-2, 0x1.5df8c5d0678f0p-5, -0x1.56aafd8a8568cp-4,
      -0x1.04787c2539657p-2, 0x1.bf54b594797cap-6, -0x1.56aafd8a8568cp-4,
      -0x1.e95d5d88e816cp-3, 0x1.31b0aedaec2bep-5, -0x1.1ae383e68ac86p-4,
      -0x1.e95d5d88e816cp-3, 0x1.5df8c5d0678f0p-5, -0x1.1d31d8f539a98p-4,
      -0x1.e3597df5eb8f7p-3, 0x1.64ab155864700p-9, -0x1.04bf786bcd16ep-4,
      -0x1.df9945c0d3287p-3, -0x1.99efe71e5a8f4p-7, -0x1.04bf786bcd16ep-4,
      -0x1.dd706f156eafdp-3, 0x1.64ab155864700p-9, -0x1.d9cb3709b4922p-5,
      -0x1.d75168e158446p-3, 0x1.79ea0fcf1641ap-7, -0x1.c14f1e395ae47p-5,
      -0x1.af239e16ec059p-3, -0x1.1925a596238cep-3, -0x1.56aafd8a8568cp-4,
      -0x1.9771d86a2fc4cp-3, -0x1.1925a596238cep-3, -0x1.31cbeca73419bp-4,
      -0x1.ba125575f3db9p-3, -0x1.e05fc60d8ec80p-4, -0x1.56aafd8a8568cp-4,
      -0x1.b5a8494b81df2p-3, -0x1.0e6653e8197e5p-3, -0x1.56aafd8a8568cp-4,
      -0x1.a55a6d67c1e75p-3, -0x1.fc30f008b30d0p-4, -0x1.2090a266f15bfp-4,
      -0x1.9771d86a2fc4cp-3, -0x1.f3995d29258fap-4, -0x1.04bf786bcd16ep-4,
      -0x1.abecb9b1ddd94p-3, -0x1.e05fc60d8ec80p-4, -0x1.2db53afb293fcp-4,
      -0x1.9e36039523d1fp-3, -0x1.e05fc60d8ec80p-4, -0x1.04bf786bcd16ep-4,
      -0x1.855338a25fc2cp-3, -0x1.070705ce538aep-3, -0x1.04bf786bcd16ep-4,
      -0x1.6e7c15dad39bep-3, 0x1.64ab155864700p-9, -0x1.6ca7e7babe988p-3,
      -0x1.6e7c15dad39bep-3, 0x1.74437725edd60p-6, -0x1.6a3af6febb231p-3,
      -0x1.4586534b7772ep-3, -0x1.23c7f85239018p-7, -0x1.4f2c8902b3584p-3,
      -0x1.4586534b7772ep-3, 0x1.64ab155864700p-9, -0x1.445f7bcd67b41p-3,
      -0x1.4dc7a8e49a5e7p-3, 0x1.74437725edd60p-6, -0x1.4f2c8902b3584p-3,
      -0x1.4df6ad2ca323fp-3, 0x1.30c0a81c904d6p-6, -0x1.4f2c8902b3584p-3,
      -0x1.4586534b7772ep-3, 0x1.f45c51d48132cp-8, -0x1.451c52c970e07p-3,
      -0x1.4586534b7772ep-3, 0x1.0f2944f072564p-5, -0x1.4f2c8902b3584p-3,
      -0x1.408b17954aecfp-3, -0x1.e67da83ce114ep-7, -0x1.4f2c8902b3584p-3,
      -0x1.38bcb8a748624p-3, -0x1.6997b95cb86a0p-7, -0x1.4262ee5e8447ap-3,
      -0x1.2b6881200b218p-3, 0x1.64ab155864700p-9, -0x1.2636c673572f4p-3,
      -0x1.28ea30b684b4bp-3, 0x1.74437725edd60p-6, -0x1.2636c673572f4p-3,
      -0x1.29bfa1850a6bep-3, 0x1.0acaf0de74c60p-6, -0x1.2636c673572f4p-3,
      -0x1.1c9090bc1b49ep-3, 0x1.74437725edd60p-6, -0x1.17dd188567644p-3,
      -0x1.c763460d6e7e7p-2, -0x1.798ea1a81a302p-3, -0x1.10f8a60720728p-2,
      -0x1.c763460d6e7e7p-2, -0x1.798ea1a81a302p-3, 0x1.f9610c74592c0p-4,
      -0x1.c763460d6e7e7p-2, 0x1.a68a73b273dabp-3, -0x1.10f8a60720728p-2,
      -0x1.c763460d6e7e7p-2, 0x1.a68a73b273dabp-3, 0x1.f9610c74592c0p-4,
      -0x1.ac1ec529c9fa7p-5, -0x1.798ea1a81a302p-3, -0x1.10f8a60720728p-2,
      -0x1.ac1ec529c9fa7p-5, -0x1.798ea1a81a302p-3, 0x1.f9610c74592c0p-4,
      -0x1.ac1ec529c9fa7p-5, 0x1.a68a73b273dabp-3, -0x1.10f8a60720728p-2,
      -0x1.ac1ec529c9fa7p-5, 0x1.a68a73b273dabp-3, 0x1.f9610c74592c0p-4,  };
  const std::vector<int32_t> triangles{
      0, 3, 1, 6, 7, 8, 4, 11, 5, 2, 8, 9,
      13, 18, 16, 16, 18, 15, 15, 21, 22, 15, 22, 20,
      15, 18, 21, 20, 22, 19, 10, 23, 12, 19, 26, 24,
      19, 22, 25, 14, 27, 17, 14, 28, 27, 27, 29, 30,
      31, 36, 35, 33, 34, 35, 35, 38, 37, 32, 39, 36,
      40, 44, 45, 41, 47, 44, 42, 49, 48, 43, 51, 52,
      43, 46, 51, 50, 52, 53, 54, 55, 57, 54, 57, 56,
      58, 60, 61, 58, 61, 59, 54, 58, 59, 54, 59, 55,
      56, 57, 61, 56, 61, 60, 54, 56, 60, 54, 60, 58,
      55, 59, 61, 55, 61, 57,  };
  const std::size_t point_count = points.size() / 3;
  const std::size_t faces = triangles.size() / 3;
  constexpr std::size_t kSheetFaces = 26;
  std::vector<int64_t> offsets(faces + 1);
  std::vector<int32_t> facets(faces);
  std::vector<int32_t> regions(2 * faces);
  std::set<std::pair<int32_t, int32_t>> edge_set;
  for (std::size_t row = 0; row <= faces; ++row) offsets[row] = static_cast<int64_t>(3 * row);
  for (std::size_t row = 0; row < faces; ++row) {
    facets[row] = static_cast<int32_t>(row);
    regions[2 * row] = row < kSheetFaces ? 0 : -1;
    regions[2 * row + 1] = 0;
    for (int k = 0; k < 3; ++k) {
      const int32_t a = triangles[3 * row + static_cast<std::size_t>(k)];
      const int32_t b = triangles[3 * row + static_cast<std::size_t>((k + 1) % 3)];
      edge_set.insert({std::min(a, b), std::max(a, b)});
    }
  }
  std::vector<int32_t> segments;
  for (const auto& [a, b] : edge_set) segments.insert(segments.end(), {a, b});
  PlcInput input;
  input.point_count = static_cast<int64_t>(point_count);
  input.points = points.data();
  input.polygon_count = static_cast<int64_t>(faces);
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = facets.data();
  input.facet_count = static_cast<int64_t>(faces);
  input.facet_regions = regions.data();
  input.segment_count = static_cast<int64_t>(edge_set.size());
  input.segments = segments.data();
  input.policy = BoundaryPolicy::kFixed;
  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) return;
  check_conforming_recovery(result, point_count, faces);
  check_fixed_boundary(result, point_count);
  PHX_CHECK(result.suppression_search.searches > 0);
  PHX_CHECK(result.suppression_search.merges > 0);
  PHX_CHECK(result.suppression_search.max_merge_layers >= 1);
  PHX_CHECK(result.suppression_search.exhausted == 0);
}

// A capped patch of the original lobed-r4 reconstruction surface, whose full
// recovery refused STEINER_SUPPRESSION at its PLC edge (36, 88): the 63
// source triangles within 0.04 of that edge, closed by a fan to one apex
// below them, every facet void outside and region 0 inside, CONFORMING.
// Here vertices 9, 16, 17 are the source vertices 36, 88, 89. A cluster of
// source vertices lies within ~3e-3 of the edge length of 88, nearly on the
// line of the edge (needle facets such as (9, 17, 16), height ~1e-3 of its
// length). Recovery splits the segments through the cluster and places
// volume Steiner points of planar fan flips and earlier suppressions a few
// 1e-6 of an edge length from their subfaces. The last non-binary64 split
// point of the segment (32, 37) (source (129, 167); 129 lies within 1.1e-4
// of the line of (36, 88)) then had a wedge that no fill covers while those
// points stay wall vertices, and growth that never enclosed a vertex
// exhausted its region: recovery refused STEINER_SUPPRESSION, as the full
// surface did. Merged cavities now enclose volume Steiner points, which
// leave with the split point.
void test_suppression_merges_enclose_volume_steiner_points() {
  const std::vector<double> points{
      -0x1.e72d6ad559595p-4, -0x1.34bee8af41e50p-5, -0x1.0e00f058de8edp-4,
      -0x1.e3b3727ee20bdp-4, -0x1.d79f1875dacf5p-4, 0x1.95c560de94dc0p-7,
      -0x1.cfb7267fa5e90p-4, -0x1.33ab786198fb6p-4, -0x1.9a4d5fb895b58p-6,
      -0x1.b3c75ccd15e37p-4, -0x1.7ba7226c1bd40p-4, -0x1.5d1e03f1508c0p-5,
      -0x1.b2541123dbe23p-4, -0x1.510e8dbd63023p-4, -0x1.b24f2d4ec22f6p-5,
      -0x1.b15ef4f01e56cp-4, -0x1.a54a9ae7171a4p-4, -0x1.09d712fb59ff5p-5,
      -0x1.a9b68108af55ap-4, -0x1.9da226ffa8192p-4, -0x1.a1140d1869164p-5,
      -0x1.a87b610cb61e0p-4, -0x1.34bee8af41e50p-5, -0x1.8fd6b00d23270p-4,
      -0x1.9f0cd01466728p-4, -0x1.2e7768404b5a8p-3, 0x1.95c560de94dc0p-7,
      -0x1.87bb7c7523108p-4, -0x1.e44c930515297p-4, -0x1.5d1e03f1508c0p-5,
      -0x1.87bb7c7523108p-4, -0x1.7ba7226c1bd40p-4, -0x1.1cd937e38f78bp-4,
      -0x1.65246fab7d64ap-4, -0x1.591015a276282p-4, -0x1.8fd6b00d23270p-4,
      -0x1.60462218c6894p-4, -0x1.2e7768404b5a8p-3, -0x1.0e334f38977d8p-5,
      -0x1.31faed35a0d72p-4, -0x1.25e6932c999aap-4, -0x1.0daee7710de79p-3,
      -0x1.2dc84a5bdca1ep-4, -0x1.7221a63de593fp-3, -0x1.526f3f7d875dap-6,
      -0x1.4e6728a07c7b7p-5, -0x1.2e17854880532p-3, -0x1.8fd6b00d23270p-4,
      -0x1.4e429e003901bp-5, -0x1.2e7768404b5a8p-3, -0x1.8f292f6daed52p-4,
      -0x1.4d761060d9761p-5, -0x1.2e7768404b5a8p-3, -0x1.8fd6b00d23270p-4,
      -0x1.4d5914b92a453p-5, -0x1.2e5b0a4254e0bp-3, -0x1.900f6c09101aap-4,
      -0x1.4d56b2f380654p-5, -0x1.2e932dccd75c5p-3, -0x1.8f9f24f40b236p-4,
      -0x1.4d3ea76738c05p-5, -0x1.2e7768404b5a8p-3, -0x1.9002356017582p-4,
      -0x1.4d3e64c470c5ep-5, -0x1.2e8d1a4113748p-3, -0x1.8fd6b00d23270p-4,
      -0x1.4d2e01b33f873p-5, -0x1.2e89017cc724dp-3, -0x1.8ff9e2861abbap-4,
      -0x1.4ce79cc1505e0p-5, -0x1.897669dc7a90fp-3, -0x1.068aae3917a2cp-5,
      -0x1.4ce79cc1505e0p-5, -0x1.2ec4e12d960ecp-3, -0x1.8f3bbe328dbe8p-4,
      -0x1.4ce79cc1505e0p-5, -0x1.2e9981c8be09bp-3, -0x1.8fd6b00d23270p-4,
      -0x1.4ce79cc1505e0p-5, -0x1.2e8ca75d1f59ap-3, -0x1.90012e46cb255p-4,
      -0x1.4ce79cc1505e0p-5, -0x1.2e7768404b5a8p-3, -0x1.901a2088325d8p-4,
      -0x1.4ce79cc1505e0p-5, -0x1.2e281878bd351p-3, -0x1.90754f9c3f71ep-4,
      -0x1.4ce79cc1505e0p-5, -0x1.531d7291685bep-4, -0x1.244a572375483p-3,
      -0x1.4c7ce6c4fe61dp-5, -0x1.2e9215bf5fd99p-3, -0x1.900c0b0b4c251p-4,
      -0x1.4bc47832ba017p-5, -0x1.2e7768404b5a8p-3, -0x1.906842546e554p-4,
      -0x1.4bb27eb891740p-5, -0x1.2ec4afc27b150p-3, -0x1.8fd6b00d23270p-4,
      -0x1.82a76b0bc2b50p-8, -0x1.9f1b3f4a88cb4p-3, -0x1.5d1e03f1508c0p-5,
      -0x1.228baa6f197c0p-8, -0x1.7ba7226c1bd40p-4, -0x1.388f2f10cf044p-3,
      0x1.09f8f16402f52p-7, -0x1.abe5a00e31f4fp-3, -0x1.904786fff532cp-5,
      0x1.23d81c29e0f81p-7, -0x1.91fffe9ab2670p-4, -0x1.43bb9d281a4dcp-3,
      0x1.d69efd9e95900p-7, -0x1.9f1b3f4a88cb4p-3, -0x1.016bc997c15f2p-4,
      0x1.d69efd9e95900p-7, -0x1.c45682923182bp-4, -0x1.388f2f10cf044p-3,
      0x1.973daee30c7ecp-6, -0x1.899d794810946p-3, -0x1.8fd6b00d23270p-4,
      0x1.9fc0bc80dbfbcp-6, -0x1.2e7768404b5a8p-3, -0x1.2201075a9cddcp-3,
      0x1.2cb20755e9523p-5, -0x1.7158ad4ef7cfbp-3, -0x1.0acc9d153e08bp-3,
      0x1.bd540fdbf4922p-6, -0x1.38397fcd86f38p-4, -0x1.f8460b9172c32p-6,
  };
  const std::vector<int32_t> triangles{
      0, 4, 2, 0, 7, 4, 1, 2, 5, 1, 5, 8,
      2, 3, 5, 2, 4, 3, 3, 4, 6, 3, 6, 5,
      4, 7, 10, 4, 10, 6, 5, 6, 9, 5, 9, 8,
      6, 10, 17, 6, 17, 9, 7, 11, 10, 8, 9, 12,
      8, 12, 14, 9, 16, 12, 9, 17, 16, 10, 11, 15,
      10, 15, 17, 11, 13, 18, 11, 18, 15, 12, 16, 19,
      12, 19, 14, 13, 28, 18, 13, 29, 28, 14, 19, 23,
      15, 18, 17, 16, 17, 19, 17, 18, 20, 17, 20, 22,
      17, 21, 19, 17, 22, 21, 18, 27, 20, 18, 28, 27,
      19, 21, 25, 19, 24, 23, 19, 25, 24, 20, 27, 22,
      21, 22, 25, 22, 26, 25, 22, 27, 26, 23, 24, 33,
      24, 25, 33, 25, 26, 30, 25, 30, 32, 25, 32, 37,
      25, 35, 33, 25, 37, 35, 26, 27, 30, 27, 28, 34,
      27, 31, 30, 27, 34, 36, 27, 36, 38, 27, 38, 31,
      28, 29, 34, 30, 31, 41, 30, 41, 32, 31, 38, 40,
      31, 40, 41, 32, 39, 37, 32, 41, 39, 0, 2, 42,
      7, 0, 42, 2, 1, 42, 1, 8, 42, 11, 7, 42,
      8, 14, 42, 13, 11, 42, 29, 13, 42, 14, 23, 42,
      23, 33, 42, 33, 35, 42, 35, 37, 42, 36, 34, 42,
      38, 36, 42, 34, 29, 42, 40, 38, 42, 41, 40, 42,
      37, 39, 42, 39, 41, 42,
  };
  const std::size_t point_count = points.size() / 3;
  const std::size_t faces = triangles.size() / 3;
  std::vector<int64_t> offsets(faces + 1);
  std::vector<int32_t> facets(faces);
  std::vector<int32_t> regions(2 * faces);
  for (std::size_t row = 0; row <= faces; ++row) offsets[row] = static_cast<int64_t>(3 * row);
  for (std::size_t row = 0; row < faces; ++row) {
    facets[row] = static_cast<int32_t>(row);
    regions[2 * row] = -1;
    regions[2 * row + 1] = 0;
  }
  PlcInput input;
  input.point_count = static_cast<int64_t>(point_count);
  input.points = points.data();
  input.polygon_count = static_cast<int64_t>(faces);
  input.polygon_offsets = offsets.data();
  input.polygon_vertices = triangles.data();
  input.polygon_facets = facets.data();
  input.facet_count = static_cast<int64_t>(faces);
  input.facet_regions = regions.data();
  input.policy = BoundaryPolicy::kConforming;
  PlcRecovery result;
  const int32_t status = recover_plc(input, result);
  PHX_CHECK(status == PHX_MC_OK);
  if (status != PHX_MC_OK) return;
  check_conforming_recovery(result, point_count, faces);
  PHX_CHECK(result.suppression_search.merges > 0);
  // The region-0 cells fill the closed surface: their volumes sum to its
  // signed volume.
  double enclosed = 0.0;
  for (std::size_t row = 0; row < faces; ++row) {
    const double* a = points.data() + 3 * static_cast<std::size_t>(triangles[3 * row]);
    const double* b = points.data() + 3 * static_cast<std::size_t>(triangles[3 * row + 1]);
    const double* c = points.data() + 3 * static_cast<std::size_t>(triangles[3 * row + 2]);
    enclosed += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                 a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
  }
  double meshed = 0.0;
  for (std::size_t cell = 0; cell < result.tet_regions.size(); ++cell) {
    PHX_CHECK(result.tet_regions[cell] == 0);
    const double* v[4];
    for (int k = 0; k < 4; ++k) {
      v[k] = result.points.data() + 3 * static_cast<std::size_t>(result.tets[4 * cell + k]);
    }
    double u[3][3];
    for (int k = 0; k < 3; ++k) {
      for (int axis = 0; axis < 3; ++axis) u[k][axis] = v[k + 1][axis] - v[0][axis];
    }
    meshed += (u[0][0] * (u[1][1] * u[2][2] - u[1][2] * u[2][1]) -
               u[0][1] * (u[1][0] * u[2][2] - u[1][2] * u[2][0]) +
               u[0][2] * (u[1][0] * u[2][1] - u[1][1] * u[2][0])) / 6.0;
  }
  PHX_CHECK(enclosed > 0.0);
  PHX_CHECK(std::fabs(meshed - enclosed) <= 1e-12);
}

// The distance envelope of the unit tetrahedron at offset 1.5 has grid
// vertices such as (-1.5, 0.5, 0.5) at exactly the offset from a face. Its
// marching-tetrahedra surface used to carry clusters of ~10 vertices within
// 4e-14 of each such vertex, with needle facets whose split points no dyadic
// parameter reached (the reconstruction envelopes failed NONREPRESENTABLE at
// such facets). With the kernel's separated field the surface is a closed,
// well-spaced triangle PLC with one facet per triangle; its recovery splits
// many segments and facets, inserts most of the split points into star-shaped
// parts of non-Delaunay conflict cavities, and must succeed under both
// boundary policies.
void test_offset_tie_envelope_recovers_under_both_policies() {
  constexpr double source[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr int64_t source_faces[] = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  void* handle = nullptr;
  std::array<double, 4> bounds{};
  const phx_mc_surface_envelope_create_request request{
      .vertices = source,
      .vertex_count = 4,
      .faces = source_faces,
      .face_count = 4,
      .offset = 1.5,
      .spacing = 0.5,
      .certificate_spacing = 0.4,
      .result = &handle,
      .bounds = bounds.data(),
  };
  if (phx_mc_surface_envelope_create(&request) != PHX_MC_OK || handle == nullptr) {
    PHX_CHECK(false);
    return;
  }
  std::array<int64_t, 4> sizes{};
  PHX_CHECK(phx_mc_surface_envelope_sizes(handle, sizes.data()) == PHX_MC_OK);
  std::vector<double> points(static_cast<std::size_t>(3 * sizes[0]));
  std::vector<int64_t> triangles(static_cast<std::size_t>(3 * sizes[1]));
  std::vector<double> volume_points(static_cast<std::size_t>(3 * sizes[2]));
  std::vector<int64_t> cells(static_cast<std::size_t>(4 * sizes[3]));
  const int32_t arrays_status = phx_mc_surface_envelope_arrays(
      handle, sizes[0], sizes[1], sizes[2], sizes[3], points.data(), triangles.data(),
      volume_points.data(), cells.data());
  phx_mc_surface_envelope_free(handle);
  PHX_CHECK(arrays_status == PHX_MC_OK);
  if (arrays_status != PHX_MC_OK) return;
  const std::size_t triangle_count = static_cast<std::size_t>(sizes[1]);
  std::vector<int64_t> offsets(triangle_count + 1);
  std::vector<int32_t> vertices(3 * triangle_count);
  std::vector<int32_t> polygon_facets(triangle_count);
  std::vector<int32_t> facet_regions(2 * triangle_count);
  for (std::size_t triangle = 0; triangle < triangle_count; ++triangle) {
    offsets[triangle + 1] = static_cast<int64_t>(3 * (triangle + 1));
    for (int corner = 0; corner < 3; ++corner) {
      vertices[3 * triangle + static_cast<std::size_t>(corner)] =
          static_cast<int32_t>(triangles[3 * triangle + static_cast<std::size_t>(corner)]);
    }
    polygon_facets[triangle] = static_cast<int32_t>(triangle);
    facet_regions[2 * triangle] = -1;
    facet_regions[2 * triangle + 1] = 1;
  }
  for (BoundaryPolicy policy : {BoundaryPolicy::kFixed, BoundaryPolicy::kConforming}) {
    PlcInput input;
    input.point_count = sizes[0];
    input.points = points.data();
    input.polygon_count = static_cast<int64_t>(triangle_count);
    input.polygon_offsets = offsets.data();
    input.polygon_vertices = vertices.data();
    input.polygon_facets = polygon_facets.data();
    input.facet_count = static_cast<int64_t>(triangle_count);
    input.facet_regions = facet_regions.data();
    input.policy = policy;
    PlcRecovery result;
    const int32_t status = recover_plc(input, result);
    if (status != PHX_MC_OK) {
      std::fprintf(stderr, "offset-tie envelope recovery failed: policy=%d status=%d\n",
                   static_cast<int>(policy), status);
      PHX_CHECK(status == PHX_MC_OK);
      continue;
    }
    PHX_CHECK(!result.tets.empty());
    PHX_CHECK(std::all_of(result.tet_regions.begin(), result.tet_regions.end(),
                          [](int32_t region) { return region == 1; }));
    // Recovery keeps the source points as the prefix of its point bank.
    PHX_CHECK(result.points.size() >= points.size() &&
              std::equal(points.begin(), points.end(), result.points.begin()));
    std::vector<char> covered(triangle_count, 0);
    for (int32_t facet : result.face_sources) {
      PHX_CHECK(facet >= 0 && static_cast<std::size_t>(facet) < triangle_count);
      if (facet >= 0 && static_cast<std::size_t>(facet) < triangle_count) {
        covered[static_cast<std::size_t>(facet)] = 1;
      }
    }
    PHX_CHECK(std::all_of(covered.begin(), covered.end(), [](char used) { return used != 0; }));
  }
}

void test_affine_recovery_c_api_guards() {
  const std::array<double, 12> points{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::array<int64_t, 5> offsets{0, 3, 6, 9, 12};
  const std::array<int32_t, 12> vertices{1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1};
  const std::array<int32_t, 4> facets{0, 1, 2, 3};
  const std::array<int32_t, 8> regions{-1, 1, -1, 1, -1, 1, -1, 1};
  const std::array<double, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::array<double, 9> reflected{-1, 0, 0, 0, 2, 0, 0, 0, 1};
  const std::array<double, 9> singular{1, 0, 0, 0, 1, 0, 0, 0, 0};
  auto nonfinite = identity;
  phx_mc_plc3d_result* result = nullptr;
  phx_mc_plc3d_prepare_source_request request{};
  request.point_count = 4;
  request.points = points.data();
  request.polygon_count = 4;
  request.polygon_vertex_count = 12;
  request.polygon_offsets = offsets.data();
  request.polygon_vertices = vertices.data();
  request.polygon_facets = facets.data();
  request.facet_count = 4;
  request.facet_regions = regions.data();
  request.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  request.result = &result;

  PHX_CHECK(phx_mc_plc3d_recover_with_affine_metric(nullptr, identity.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  auto invalid = [&](const phx_mc_plc3d_prepare_source_request& bad, const double* matrix,
                     int32_t expected) {
    result = reinterpret_cast<phx_mc_plc3d_result*>(static_cast<uintptr_t>(1));
    PHX_CHECK(phx_mc_plc3d_recover_with_affine_metric(&bad, matrix) == expected);
    PHX_CHECK(result == nullptr);
  };
  invalid(request, nullptr, PHX_MC_INVALID_ARGUMENT);
  invalid(request, singular.data(), PHX_MC_INVALID_ARGUMENT);
  nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
  invalid(request, nonfinite.data(), PHX_MC_INVALID_ARGUMENT);
  nonfinite[0] = std::numeric_limits<double>::infinity();
  invalid(request, nonfinite.data(), PHX_MC_INVALID_ARGUMENT);
  auto bad = request;
  bad.points = nullptr;
  invalid(bad, identity.data(), PHX_MC_INVALID_ARGUMENT);
  bad = request;
  bad.polygon_vertices = nullptr;
  invalid(bad, identity.data(), PHX_MC_INVALID_ARGUMENT);
  bad = request;
  bad.polygon_vertex_count = -1;
  invalid(bad, identity.data(), PHX_MC_INVALID_ARGUMENT);
  bad = request;
  bad.polygon_vertex_count = 11;
  invalid(bad, identity.data(), PHX_MC_INVALID_ARGUMENT);
  bad = request;
  bad.point_count = std::numeric_limits<int32_t>::max();
  invalid(bad, identity.data(), PHX_MC_INDEX_RANGE_ERROR);
  const std::array<int32_t, 4> bad_facets{4, 1, 2, 3};
  bad = request;
  bad.polygon_facets = bad_facets.data();
  result = nullptr;
  PHX_CHECK(phx_mc_plc3d_recover_with_affine_metric(&bad, identity.data()) ==
            PHX_MC_INVALID_INPUT);
  phx_mc_plc3d_free(result);
  bad = request;
  bad.result = nullptr;
  PHX_CHECK(phx_mc_plc3d_recover_with_affine_metric(&bad, identity.data()) ==
            PHX_MC_INVALID_ARGUMENT);

  for (const double* matrix : {identity.data(), reflected.data()}) {
    result = nullptr;
    PHX_CHECK(phx_mc_plc3d_recover_with_affine_metric(&request, matrix) == PHX_MC_OK);
    PHX_CHECK(result != nullptr);
    if (result != nullptr) {
      std::array<int64_t, PHX_MC_PLC3D_SIZES> sizes{};
      PHX_CHECK(phx_mc_plc3d_sizes(result, sizes.data()) == PHX_MC_OK);
      PHX_CHECK(sizes[0] == 4 && sizes[1] == 1 && sizes[2] == 4);
      phx_mc_plc3d_free(result);
    }
  }
  result = nullptr;
  PHX_CHECK(phx_mc_plc3d_recover(&request) == PHX_MC_OK);
  PHX_CHECK(result != nullptr);
  if (result != nullptr) {
    std::array<int64_t, PHX_MC_PLC3D_SIZES> sizes{};
    PHX_CHECK(phx_mc_plc3d_sizes(result, sizes.data()) == PHX_MC_OK);
    PHX_CHECK(sizes[0] == 4 && sizes[1] == 1 && sizes[2] == 4);
    phx_mc_plc3d_free(result);
  }
}

// Every published bank of one PLC3D C API call, compared exactly across modes.
struct Plc3dOutcome {
  int32_t status = -1;
  bool published = false;
  std::array<int64_t, PHX_MC_PLC3D_SIZES> sizes{};
  std::vector<double> points;
  std::vector<int32_t> tetrahedra;
  std::vector<int32_t> tetrahedron_regions;
  std::vector<int32_t> faces;
  std::vector<int32_t> face_sources;
  std::vector<int8_t> face_orientations;
  std::vector<int32_t> segments;
  std::vector<int32_t> segment_sources;
  std::vector<double> protection;
  std::vector<int8_t> vertex_dimensions;
  std::vector<int32_t> plc_edges;
  std::vector<int32_t> input_triangles;
  std::vector<int32_t> input_polygons;
  std::vector<int8_t> strata;
  std::vector<int32_t> entities;
  std::vector<double> parameters;
  std::vector<double> deviations;
  std::array<int32_t, 3> failure_kinds{};
  std::array<int64_t, 2> failure_entities{};
  std::array<double, 2> failure_bounds{};

  bool operator==(const Plc3dOutcome&) const = default;
};

enum class Plc3dEntry { kPrepareSource, kRecover, kRecoverWithAffineMetric, kRecoverProtected };

Plc3dOutcome plc3d_outcome(Plc3dEntry entry, const std::vector<double>& points,
                           const std::vector<int64_t>& offsets,
                           const std::vector<int32_t>& vertices) {
  const int64_t polygon_count = static_cast<int64_t>(offsets.size()) - 1;
  std::vector<int32_t> facets(static_cast<std::size_t>(polygon_count));
  std::vector<int32_t> regions(static_cast<std::size_t>(2 * polygon_count));
  for (int64_t facet = 0; facet < polygon_count; ++facet) {
    facets[static_cast<std::size_t>(facet)] = static_cast<int32_t>(facet);
    regions[static_cast<std::size_t>(2 * facet)] = -1;
    regions[static_cast<std::size_t>(2 * facet + 1)] = 1;
  }
  const std::vector<uint8_t> protected_facets(static_cast<std::size_t>(polygon_count), 1);
  constexpr std::array<double, 9> metric{-1, 0, 0, 0, 2, 0, 0, 0, 1};
  phx_mc_plc3d_result* result = nullptr;
  phx_mc_plc3d_prepare_source_request request{};
  request.point_count = static_cast<int64_t>(points.size() / 3);
  request.points = points.data();
  request.polygon_count = polygon_count;
  request.polygon_vertex_count = static_cast<int64_t>(vertices.size());
  request.polygon_offsets = offsets.data();
  request.polygon_vertices = vertices.data();
  request.polygon_facets = facets.data();
  request.facet_count = polygon_count;
  request.facet_regions = regions.data();
  request.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  request.result = &result;
  Plc3dOutcome outcome;
  switch (entry) {
    case Plc3dEntry::kPrepareSource:
      outcome.status = phx_mc_plc3d_prepare_source(&request);
      break;
    case Plc3dEntry::kRecover:
      outcome.status = phx_mc_plc3d_recover(&request);
      break;
    case Plc3dEntry::kRecoverWithAffineMetric:
      outcome.status = phx_mc_plc3d_recover_with_affine_metric(&request, metric.data());
      break;
    case Plc3dEntry::kRecoverProtected:
      outcome.status =
          phx_mc_plc3d_recover_protected(&request, protected_facets.data(), nullptr);
      break;
  }
  outcome.published = result != nullptr;
  if (result == nullptr) return outcome;
  PHX_CHECK(phx_mc_plc3d_sizes(result, outcome.sizes.data()) == PHX_MC_OK);
  const auto rows = [&](int row, int width) {
    return static_cast<std::size_t>(outcome.sizes[static_cast<std::size_t>(row)] * width);
  };
  outcome.points.resize(rows(0, 3));
  outcome.tetrahedra.resize(rows(1, 4));
  outcome.tetrahedron_regions.resize(rows(1, 1));
  outcome.faces.resize(rows(2, 3));
  outcome.face_sources.resize(rows(2, 1));
  outcome.face_orientations.resize(rows(2, 1));
  outcome.segments.resize(rows(3, 2));
  outcome.segment_sources.resize(rows(3, 1));
  outcome.protection.resize(rows(0, 1));
  outcome.vertex_dimensions.resize(rows(0, 1));
  outcome.plc_edges.resize(rows(4, 2));
  outcome.input_triangles.resize(rows(5, 3));
  outcome.input_polygons.resize(rows(5, 1));
  PHX_CHECK(phx_mc_plc3d_export(
                result, outcome.sizes[0], outcome.sizes[1], outcome.sizes[2], outcome.sizes[3],
                outcome.sizes[4], outcome.sizes[5], outcome.points.data(),
                outcome.tetrahedra.data(), outcome.tetrahedron_regions.data(),
                outcome.faces.data(), outcome.face_sources.data(),
                outcome.face_orientations.data(), outcome.segments.data(),
                outcome.segment_sources.data(), outcome.protection.data(),
                outcome.vertex_dimensions.data(), outcome.plc_edges.data(),
                outcome.input_triangles.data(), outcome.input_polygons.data()) == PHX_MC_OK);
  outcome.strata.resize(rows(0, 1));
  outcome.entities.resize(rows(0, 1));
  outcome.parameters.resize(rows(0, 2));
  outcome.deviations.resize(rows(0, 1));
  PHX_CHECK(phx_mc_plc3d_source_witnesses(result, outcome.sizes[0], outcome.strata.data(),
                                          outcome.entities.data(), outcome.parameters.data(),
                                          outcome.deviations.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_plc3d_failure(result, &outcome.failure_kinds[0], &outcome.failure_kinds[1],
                                 &outcome.failure_entities[0], &outcome.failure_kinds[2],
                                 &outcome.failure_entities[1], &outcome.failure_bounds[0],
                                 &outcome.failure_bounds[1]) == PHX_MC_OK);
  phx_mc_plc3d_free(result);
  return outcome;
}

// The unit corner tetrahedron and the exactly cospherical octahedron give their
// round-to-nearest banks through every PLC3D entry in every caller mode; a
// subnormal coordinate lies outside the exact coordinate domain.
void test_plc3d_entries_under_floating_point_hazards() {
  const std::vector<double> corner{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<int64_t> corner_offsets{0, 3, 6, 9, 12};
  const std::vector<int32_t> corner_vertices{1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1};
  const std::vector<double> octahedron{1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
  const std::vector<int64_t> octahedron_offsets{0, 3, 6, 9, 12, 15, 18, 21, 24};
  const std::vector<int32_t> octahedron_vertices{0, 2, 4, 1, 4, 2, 0, 4, 3, 1, 3, 4,
                                                 0, 5, 2, 1, 2, 5, 0, 3, 5, 1, 5, 3};
  std::vector<double> subnormal = corner;
  subnormal[9] = 0x1p-1060;
  for (const Plc3dEntry entry : {Plc3dEntry::kPrepareSource, Plc3dEntry::kRecover,
                                 Plc3dEntry::kRecoverWithAffineMetric,
                                 Plc3dEntry::kRecoverProtected}) {
    const bool recovers = entry != Plc3dEntry::kPrepareSource;
    const Plc3dOutcome corner_reference =
        plc3d_outcome(entry, corner, corner_offsets, corner_vertices);
    PHX_CHECK(corner_reference.status == PHX_MC_OK && corner_reference.published);
    PHX_CHECK(corner_reference.sizes[0] == 4);
    PHX_CHECK(corner_reference.sizes[1] == (recovers ? 1 : 0));
    PHX_CHECK(corner_reference.points == corner);
    const Plc3dOutcome octahedron_reference =
        plc3d_outcome(entry, octahedron, octahedron_offsets, octahedron_vertices);
    PHX_CHECK(octahedron_reference.status == PHX_MC_OK && octahedron_reference.published);
    PHX_CHECK(octahedron_reference.sizes[0] == 6);
    PHX_CHECK(octahedron_reference.sizes[1] == (recovers ? 4 : 0));
    PHX_CHECK(octahedron_reference.points == octahedron);
    const Plc3dOutcome subnormal_reference =
        plc3d_outcome(entry, subnormal, corner_offsets, corner_vertices);
    PHX_CHECK(subnormal_reference.status == PHX_MC_RANGE_ERROR);
    PHX_CHECK(!subnormal_reference.published);
    phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
      PHX_CHECK(plc3d_outcome(entry, corner, corner_offsets, corner_vertices) ==
                corner_reference);
      PHX_CHECK(plc3d_outcome(entry, octahedron, octahedron_offsets, octahedron_vertices) ==
                octahedron_reference);
      PHX_CHECK(plc3d_outcome(entry, subnormal, corner_offsets, corner_vertices) ==
                subnormal_reference);
    });
  }
}

}  // namespace


int main() {
  test_affine_recovery_c_api_guards();
  test_plc3d_entries_under_floating_point_hazards();
  test_edge_candidates_include_touching_and_coplanar_constraints();
  test_adjacency_candidates_follow_accepted_cavity_edit();
  test_source_witness_carriers_report_exact_deviation();
  test_incremental_triangulation_owns_closed_adjacency();
  test_protection_index_keeps_nonincident_segment_candidates();
  test_source_preparation_triangulates_and_compiles_plc_edges();
  test_source_preparation_refuses_nonplanar_and_self_intersecting_polygons();
  test_conforming_nonrepresentable_segment_uses_vertex_cavity();
  test_unsplittable_boundary_loop_edge_recovers_under_fixed_and_conforming();
  test_conforming_missing_boundary_edges_recover_with_exact_witnesses();
  test_witnessed_segment_split_is_published_exactly();
  test_nonrepresentable_split_is_suppressed_into_the_volume();
  test_bent_exterior_crease_edges_recover_under_both_policies();
  test_conforming_representable_segment_prefers_vertex_cavity();
  test_incremental_point_slot_reindexes_corners_and_rolls_back();
  test_external_cavity_replacement_reseeds_incremental_locate();
  test_insertion_refuses_a_cavity_around_a_constrained_face();
  test_recover_plc_publishes_a_single_region_tetrahedron();
  test_declared_facet_plane_groups_rounded_triangles();
  test_recover_fixed_triangulated_planar_facets();
  test_recover_fixed_warped_facet_edge_without_steiner();
  test_recover_fixed_non_delaunay_internal_facet_without_split();
  test_recover_plc_failure_preserves_previous_publication();
  test_recover_plc_classifies_two_regions_across_a_shared_facet();
  test_recover_plc_handles_sliver_geometry_and_explicit_fixed_edge();
  test_recover_plc_nonconvex_prism();
  test_recover_plc_removes_a_closed_cavity();
  test_grouped_material_interface_with_cavity_recovers_missing_planar_surface();
  test_recover_plc_corner_scaffold_tracks_inserted_points_and_region_interface();
  test_offset_tie_envelope_recovers_under_both_policies();
  test_suppression_wedge_without_delaunay_fill_is_searched();
  test_suppression_merges_a_sliver_wedge_with_its_neighbours();
  test_suppression_merges_enclose_volume_steiner_points();
  return phx::mc::test::finish("test_plc3d");
}
