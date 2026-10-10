//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Transactional cavity edits and the prepared incremental 3D triangulation:
// oriented boundary and reciprocal adjacency, constrained-facet preservation,
// refusal without change (validation, buffers, cells, slots, work, refused
// allocations), slot reuse, and prepared construction equal to the point-set
// entry point.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#include "cavity.hpp"
#include "check.hpp"
#include "intersections.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"
#include "triangulation3d.hpp"

namespace {

using namespace phx::mc;

// While set, every global allocation of this process is refused.
std::atomic<bool> refuse_allocations{false};

void* allocate_bytes(std::size_t size) {
  if (refuse_allocations.load(std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size == 0 ? 1 : size)) {
    return memory;
  }
  throw std::bad_alloc();
}

std::vector<double> random_points(int count, std::uint64_t seed) {
  std::vector<double> points(static_cast<std::size_t>(3 * count));
  for (double& value : points) {
    seed = splitmix64(seed);
    value = static_cast<double>(seed >> 11) * 0x1p-53;
  }
  return points;
}

// A triangulation of `points` built by the shared incremental state.
struct Built {
  std::vector<double> points;
  std::vector<char> alive;
  Triangulation3D triangulation;

  explicit Built(std::vector<double> input, int64_t slot_limit = kMaxTetrahedronSlots)
      : points(std::move(input)),
        triangulation(points.data(), nullptr, static_cast<int64_t>(points.size() / 3),
                      InsertionLimits{}, slot_limit) {
    std::vector<int32_t> order(points.size() / 3);
    for (std::size_t k = 0; k < order.size(); ++k) {
      order[k] = static_cast<int32_t>(k);
    }
    alive.assign(order.size(), 1);
    PHX_CHECK(triangulation.build(order, alive) == PHX_MC_OK);
  }
  TetrahedralComplex& complex() { return triangulation.complex(); }
  const double* point(int32_t v) const { return points.data() + 3 * static_cast<int64_t>(v); }
  bool positive(const int32_t* v) const {
    return orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) > 0;
  }
};

// Everything observable about a complex, for bit-identity after a refusal.
struct Snapshot {
  std::vector<std::array<int32_t, 8>> slots;
  std::vector<int32_t> free_slots;
  std::vector<int32_t> labels;
  int64_t finite = 0;

  static Snapshot of(const TetrahedralComplex& complex) {
    Snapshot snapshot;
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      snapshot.slots.push_back({tet.v[0], tet.v[1], tet.v[2], tet.v[3], tet.n[0], tet.n[1],
                                tet.n[2], tet.n[3]});
      for (int k = 0; k < 4; ++k) {
        snapshot.labels.push_back(complex.constraint(static_cast<int32_t>(t), k));
      }
      snapshot.labels.push_back(complex.region(static_cast<int32_t>(t)));
    }
    snapshot.free_slots.assign(complex.free_slots.begin(), complex.free_slots.end());
    snapshot.finite = complex.finite_count;
    return snapshot;
  }
  bool operator==(const Snapshot& other) const {
    return slots == other.slots && free_slots == other.free_slots && labels == other.labels &&
           finite == other.finite;
  }
};

// Canonical sorted finite cells of a complex.
std::vector<std::array<int32_t, 4>> cells_of(const TetrahedralComplex& complex) {
  std::vector<std::array<int32_t, 4>> cells;
  for (const Tetrahedron& tet : complex.tets) {
    if (tet.v[0] != kDeadVertex && !is_ghost(tet)) {
      std::array<int32_t, 4> cell = {tet.v[0], tet.v[1], tet.v[2], tet.v[3]};
      canonicalize_cell(cell.data(), nullptr, 4);
      cells.push_back(cell);
    }
  }
  std::sort(cells.begin(), cells.end());
  return cells;
}

// A 2-3 flip across facet `slot` of finite tetrahedron t: the three
// tetrahedra around the segment joining the two apexes, each oriented
// positively when possible.
struct Flip {
  int32_t first;
  int32_t second;
  std::array<std::array<int32_t, 4>, 3> proposal;
  int32_t apex[2];
  int32_t facet[3];
};

bool make_flip(const Built& built, int32_t t, int slot, Flip& flip) {
  const TetrahedralComplex& complex = built.triangulation.complex();
  const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(t)];
  const int32_t other = tet.n[slot];
  const Tetrahedron& neighbor = complex.tets[static_cast<std::size_t>(other)];
  if (is_ghost(tet) || is_ghost(neighbor) || other < t) {
    return false;
  }
  const int back = neighbor_slot(neighbor, t);
  flip.first = t;
  flip.second = other;
  flip.apex[0] = tet.v[slot];
  flip.apex[1] = neighbor.v[back];
  oriented_facet(tet.v, slot, flip.facet);
  for (int r = 0; r < 3; ++r) {
    std::array<int32_t, 4> cell = {flip.facet[r], flip.facet[(r + 1) % 3], flip.apex[0],
                                   flip.apex[1]};
    if (!built.positive(cell.data())) {
      std::swap(cell[0], cell[1]);
    }
    flip.proposal[static_cast<std::size_t>(r)] = cell;
  }
  return true;
}

EditStatus stage_flip(CavityEdit& edit, const Built& built, const Flip& flip,
                      const EditLimits& limits = EditLimits{}) {
  edit.begin(limits);
  EditStatus status = edit.remove(flip.first);
  if (status == EditStatus::kOk) {
    status = edit.remove(flip.second);
  }
  for (const auto& cell : flip.proposal) {
    if (status == EditStatus::kOk) {
      status = edit.add(cell.data(), nullptr, kNoRegion);
    }
  }
  if (status != EditStatus::kOk) {
    return status;
  }
  return edit.validate([&](const int32_t* v) { return built.positive(v); });
}

// The flip is legal exactly when the apex segment crosses the shared facet's
// interior; validation must agree with the exact segment/triangle kernel.
void test_flip_validation_matches_geometry() {
  Built built(random_points(60, 7));
  CavityEdit edit(built.complex());
  int legal = 0;
  int illegal = 0;
  int mismatches = 0;
  const Snapshot before = Snapshot::of(built.complex());
  const std::size_t slots = built.complex().tets.size();
  for (std::size_t t = 0; t < slots; ++t) {
    for (int slot = 0; slot < 4; ++slot) {
      Flip flip{};
      if (!make_flip(built, static_cast<int32_t>(t), slot, flip)) {
        continue;
      }
      const double* const segment[2] = {built.point(flip.apex[0]), built.point(flip.apex[1])};
      const double* const triangle[3] = {built.point(flip.facet[0]), built.point(flip.facet[1]),
                                         built.point(flip.facet[2])};
      Contact contact;
      intersect_segment_triangle(segment, triangle, nullptr, nullptr, contact);
      const bool crossing = contact.kind == PHX_MC_CROSSING;
      const EditStatus status = stage_flip(edit, built, flip);
      edit.rollback();
      mismatches += (status == EditStatus::kOk) != crossing ? 1 : 0;
      legal += crossing ? 1 : 0;
      illegal += crossing ? 0 : 1;
    }
  }
  PHX_CHECK(mismatches == 0);
  PHX_CHECK(legal > 0 && illegal > 0);
  // Staging and rolling back never writes.
  PHX_CHECK(Snapshot::of(built.complex()) == before);
}

bool find_legal_flip(Built& built, CavityEdit& edit, Flip& flip) {
  for (std::size_t t = 0; t < built.complex().tets.size(); ++t) {
    for (int slot = 0; slot < 4; ++slot) {
      if (built.complex().live(static_cast<int32_t>(t)) &&
          make_flip(built, static_cast<int32_t>(t), slot, flip) &&
          stage_flip(edit, built, flip) == EditStatus::kOk) {
        return true;
      }
    }
  }
  return false;
}

void test_flip_commit_revert_and_slot_reuse() {
  Built built(random_points(40, 11));
  TetrahedralComplex& complex = built.complex();
  CavityEdit edit(complex);
  const auto original = cells_of(complex);
  const int64_t finite = complex.finite_count;
  Flip flip{};
  PHX_CHECK(find_legal_flip(built, edit, flip));
  const std::array<int32_t, 4> first = {
      complex.tets[static_cast<std::size_t>(flip.first)].v[0],
      complex.tets[static_cast<std::size_t>(flip.first)].v[1],
      complex.tets[static_cast<std::size_t>(flip.first)].v[2],
      complex.tets[static_cast<std::size_t>(flip.first)].v[3]};
  const std::array<int32_t, 4> second = {
      complex.tets[static_cast<std::size_t>(flip.second)].v[0],
      complex.tets[static_cast<std::size_t>(flip.second)].v[1],
      complex.tets[static_cast<std::size_t>(flip.second)].v[2],
      complex.tets[static_cast<std::size_t>(flip.second)].v[3]};
  PHX_CHECK(edit.commit() == EditStatus::kOk);
  PHX_CHECK(complex.audit());
  PHX_CHECK(complex.finite_count == finite + 1);
  // Removed slots are reused first.
  PHX_CHECK(edit.created()[0] == flip.first && edit.created()[1] == flip.second);
  const std::vector<int32_t> three(edit.created().begin(), edit.created().end());

  // 3-2 flip back: the original pair fills the same oriented boundary.
  edit.begin(EditLimits{});
  for (int32_t t : three) {
    PHX_CHECK(edit.remove(t) == EditStatus::kOk);
  }
  PHX_CHECK(edit.add(first.data(), nullptr, kNoRegion) == EditStatus::kOk);
  PHX_CHECK(edit.add(second.data(), nullptr, kNoRegion) == EditStatus::kOk);
  PHX_CHECK(edit.validate([&](const int32_t* v) { return built.positive(v); }) ==
            EditStatus::kOk);
  PHX_CHECK(edit.commit() == EditStatus::kOk);
  PHX_CHECK(complex.audit());
  PHX_CHECK(cells_of(complex) == original);
  PHX_CHECK(complex.free_slots.size() == 1 && complex.free_slots.back() == three[2]);

  // The next flip reuses the released slot instead of growing the storage.
  const std::size_t size = complex.tets.size();
  PHX_CHECK(find_legal_flip(built, edit, flip));
  PHX_CHECK(edit.commit() == EditStatus::kOk);
  PHX_CHECK(complex.tets.size() == size && complex.free_slots.empty());
  PHX_CHECK(complex.audit());
}

void test_refusals_leave_the_complex_unchanged() {
  Built built(random_points(40, 13));
  TetrahedralComplex& complex = built.complex();
  CavityEdit edit(complex);
  Flip flip{};
  PHX_CHECK(find_legal_flip(built, edit, flip));
  edit.rollback();
  const Snapshot before = Snapshot::of(complex);
  auto positive = [&](const int32_t* v) { return built.positive(v); };

  // Inverted proposal.
  Flip inverted = flip;
  std::swap(inverted.proposal[0][0], inverted.proposal[0][1]);
  PHX_CHECK(stage_flip(edit, built, inverted) == EditStatus::kInverted);
  // Incomplete fill of the oriented boundary.
  edit.begin(EditLimits{});
  edit.remove(flip.first);
  edit.remove(flip.second);
  edit.add(flip.proposal[0].data(), nullptr, kNoRegion);
  edit.add(flip.proposal[1].data(), nullptr, kNoRegion);
  PHX_CHECK(edit.validate(positive) == EditStatus::kBoundaryMismatch);
  // Repeated and dead slots.
  edit.begin(EditLimits{});
  edit.remove(flip.first);
  edit.remove(flip.first);
  PHX_CHECK(edit.validate(positive) == EditStatus::kDuplicate);
  edit.begin(EditLimits{});
  PHX_CHECK(edit.remove(static_cast<int32_t>(complex.tets.size())) == EditStatus::kNotLive);
  // Buffer bounds.
  PHX_CHECK(stage_flip(edit, built, flip, EditLimits{1, 8, INT64_MAX}) ==
            EditStatus::kBufferLimit);
  PHX_CHECK(stage_flip(edit, built, flip, EditLimits{8, 2, INT64_MAX}) ==
            EditStatus::kBufferLimit);
  // Cell limit at commit time.
  PHX_CHECK(stage_flip(edit, built, flip, EditLimits{8, 8, complex.finite_count}) ==
            EditStatus::kOk);
  PHX_CHECK(edit.commit() == EditStatus::kCellLimit);
  PHX_CHECK(Snapshot::of(complex) == before);
  PHX_CHECK(complex.audit());
}

void test_slot_limit_is_a_refusal() {
  std::vector<double> points = random_points(30, 17);
  Built probe(points);
  // A complex allowed exactly its current slots cannot grow by one.
  Built built(points, static_cast<int64_t>(probe.complex().tets.size()));
  TetrahedralComplex& complex = built.complex();
  PHX_CHECK(complex.free_slots.empty());
  CavityEdit edit(complex);
  Flip flip{};
  PHX_CHECK(find_legal_flip(built, edit, flip));
  const Snapshot before = Snapshot::of(complex);
  PHX_CHECK(edit.commit() == EditStatus::kSlotLimit);
  PHX_CHECK(Snapshot::of(complex) == before);
}

void test_constrained_facets_survive() {
  Built built(random_points(40, 19));
  TetrahedralComplex& complex = built.complex();
  complex.enable_labels();
  CavityEdit edit(complex);
  Flip flip{};
  PHX_CHECK(find_legal_flip(built, edit, flip));
  edit.rollback();
  auto positive = [&](const int32_t* v) { return built.positive(v); };

  // The flipped facet itself is constrained: the flip would remove it.
  const int shared = neighbor_slot(complex.tets[static_cast<std::size_t>(flip.first)],
                                   flip.second);
  complex.set_facet_constraint(flip.first, shared, 5);
  const Snapshot constrained = Snapshot::of(complex);
  PHX_CHECK(stage_flip(edit, built, flip) == EditStatus::kConstraintViolation);
  PHX_CHECK(Snapshot::of(complex) == constrained);
  complex.set_facet_constraint(flip.first, shared, kNoConstraint);

  // A declared facet absent from the proposal refuses the edit.
  edit.begin(EditLimits{});
  edit.remove(flip.first);
  edit.remove(flip.second);
  for (const auto& cell : flip.proposal) {
    edit.add(cell.data(), nullptr, kNoRegion);
  }
  edit.preserve(flip.facet[0], flip.facet[1], flip.facet[2], kNoConstraint);
  PHX_CHECK(edit.validate(positive) == EditStatus::kConstraintViolation);

  // A constrained boundary facet carries over to the tetrahedron filling it,
  // on both sides; the new interior facets are unconstrained.
  const int outer = (shared + 1) % 4;
  complex.set_facet_constraint(flip.first, outer, 9);
  PHX_CHECK(stage_flip(edit, built, flip) == EditStatus::kOk);
  PHX_CHECK(edit.commit() == EditStatus::kOk);
  PHX_CHECK(complex.audit());
  int carried = 0;
  for (int32_t t : edit.created()) {
    for (int k = 0; k < 4; ++k) {
      carried += complex.constraint(t, k) == 9 ? 1 : 0;
    }
  }
  PHX_CHECK(carried == 1);
}

void test_refused_allocation_leaves_the_complex_unchanged() {
  Built built(random_points(30, 23));
  TetrahedralComplex& complex = built.complex();
  complex.tets.shrink_to_fit();
  CavityEdit edit(complex);
  Flip flip{};
  PHX_CHECK(find_legal_flip(built, edit, flip));
  const Snapshot before = Snapshot::of(complex);
  bool refused = false;
  refuse_allocations.store(true);
  try {
    edit.commit();
  } catch (const std::bad_alloc&) {
    refused = true;
  }
  refuse_allocations.store(false);
  PHX_CHECK(refused);
  PHX_CHECK(Snapshot::of(complex) == before);
  PHX_CHECK(complex.audit());
}

// ------------------------------------------------------------ prepared ABI

struct Mesh {
  std::vector<int32_t> cells;
  std::vector<int32_t> vertex_map;
  std::vector<int32_t> constraints;
  std::vector<int32_t> regions;
  std::vector<double> points;
};

Mesh read(phx_mc_mesh* mesh) {
  Mesh result;
  const int64_t cells = phx_mc_mesh_cell_count(mesh);
  result.cells.resize(static_cast<std::size_t>(4 * cells));
  result.constraints.resize(static_cast<std::size_t>(4 * cells));
  result.regions.resize(static_cast<std::size_t>(cells));
  result.vertex_map.resize(static_cast<std::size_t>(phx_mc_mesh_input_point_count(mesh)));
  result.points.resize(static_cast<std::size_t>(3 * phx_mc_mesh_point_count(mesh)));
  phx_mc_mesh_copy_cells(mesh, result.cells.data());
  phx_mc_mesh_copy_cell_constraints(mesh, result.constraints.data());
  phx_mc_mesh_copy_cell_regions(mesh, result.regions.data());
  phx_mc_mesh_copy_vertex_map(mesh, result.vertex_map.data());
  phx_mc_mesh_copy_points(mesh, result.points.data());
  phx_mc_mesh_free(mesh);
  return result;
}

Mesh finalize(const phx_mc_triangulation_3d* handle) {
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_triangulation_3d_finalize(handle, &mesh) == PHX_MC_OK);
  return read(mesh);
}

phx_mc_triangulation_3d* create(const std::vector<double>& points, int64_t count,
                                int64_t max_cavity = INT64_MAX / 2) {
  phx_mc_triangulation_3d* handle = nullptr;
  PHX_CHECK(phx_mc_triangulation_3d_create(count, points.data(), 1000000, INT64_MAX / 2,
                                           max_cavity, &handle) == PHX_MC_OK);
  return handle;
}

std::vector<int64_t> statistics(const phx_mc_triangulation_3d* handle) {
  std::vector<int64_t> values(PHX_MC_TRIANGULATION_3D_STATISTICS);
  PHX_CHECK(phx_mc_triangulation_3d_statistics(handle, values.data()) == PHX_MC_OK);
  return values;
}

// Batches with exact duplicates (within a batch and of earlier vertices) give
// the point-set triangulation of the concatenated points.
void test_prepared_batches_match_the_point_set() {
  std::vector<double> points = random_points(3000, 29);
  std::copy_n(points.begin() + 30, 9, points.begin() + 3 * 2500);   // earlier duplicates
  std::copy_n(points.begin() + 3 * 2600, 3, points.begin() + 3 * 2601);  // batch duplicate
  const int64_t count = static_cast<int64_t>(points.size() / 3);
  phx_mc_mesh* reference = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(count, points.data(), INT64_MAX / 2, &reference) == PHX_MC_OK);
  const Mesh expected = read(reference);

  phx_mc_triangulation_3d* handle = create(points, 500);
  std::vector<int32_t> vertices(static_cast<std::size_t>(count));
  std::vector<int32_t> status(static_cast<std::size_t>(count), -7);
  const int64_t batches[3][2] = {{500, 2000}, {2000, 2001}, {2001, count}};
  for (const auto& batch : batches) {
    PHX_CHECK(phx_mc_triangulation_3d_insert(
                  handle, batch[1] - batch[0], points.data() + 3 * batch[0], INT64_MAX,
                  vertices.data() + batch[0], status.data() + batch[0]) == PHX_MC_OK);
  }
  PHX_CHECK(std::all_of(status.begin() + 500, status.end(),
                        [](int32_t s) { return s == PHX_MC_OK; }));
  PHX_CHECK(vertices[2500] == 10 && vertices[2602] == 2602 && vertices[2601] == 2600);
  const Mesh prepared = finalize(handle);
  PHX_CHECK(prepared.cells == expected.cells);
  PHX_CHECK(prepared.vertex_map == expected.vertex_map);
  PHX_CHECK(prepared.points == points);
  const std::vector<int64_t> values = statistics(handle);
  PHX_CHECK(values[PHX_MC_TRIANGULATION_3D_VERTEX_IDS] == count);
  PHX_CHECK(values[PHX_MC_TRIANGULATION_3D_FINITE_CELLS] ==
            static_cast<int64_t>(expected.cells.size() / 4));
  PHX_CHECK(values[PHX_MC_TRIANGULATION_3D_REFUSED] == 0);
  PHX_CHECK(values[PHX_MC_TRIANGULATION_3D_PEAK_BYTES] >=
            values[PHX_MC_TRIANGULATION_3D_RETAINED_BYTES]);
  phx_mc_triangulation_3d_free(handle);
}

void test_prepared_limits_refuse_without_change() {
  const std::vector<double> points = random_points(400, 31);
  // Work limit: the remaining points are refused once it is exhausted.
  phx_mc_triangulation_3d* handle = create(points, 100);
  const Mesh initial = finalize(handle);
  std::vector<int32_t> vertices(300);
  std::vector<int32_t> status(300);
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, 300, points.data() + 300, 0, vertices.data(),
                                           status.data()) == PHX_MC_OK);
  PHX_CHECK(std::all_of(status.begin(), status.end(),
                        [](int32_t s) { return s == PHX_MC_CAPACITY_EXCEEDED; }));
  PHX_CHECK(std::all_of(vertices.begin(), vertices.end(), [](int32_t v) { return v == -1; }));
  const Mesh after = finalize(handle);
  PHX_CHECK(after.cells == initial.cells);
  PHX_CHECK(statistics(handle)[PHX_MC_TRIANGULATION_3D_VERTEX_IDS] == 400);
  phx_mc_triangulation_3d_free(handle);

  // Cavity bound: refused insertions leave no trace, so the result is the
  // point-set triangulation of the accepted points (ids map monotonically).
  handle = create(points, 100, 16);
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, 300, points.data() + 300, INT64_MAX,
                                           vertices.data(), status.data()) == PHX_MC_OK);
  int refused = 0;
  std::vector<double> accepted(points.begin(), points.begin() + 300);
  std::vector<int32_t> ids(100);
  for (int32_t k = 0; k < 100; ++k) {
    ids[static_cast<std::size_t>(k)] = k;
  }
  for (int i = 0; i < 300; ++i) {
    if (status[static_cast<std::size_t>(i)] == PHX_MC_CAPACITY_EXCEEDED) {
      ++refused;
      PHX_CHECK(vertices[static_cast<std::size_t>(i)] == -1);
    } else {
      PHX_CHECK(status[static_cast<std::size_t>(i)] == PHX_MC_OK);
      PHX_CHECK(vertices[static_cast<std::size_t>(i)] == 100 + i);
      accepted.insert(accepted.end(), points.begin() + 3 * (100 + i),
                      points.begin() + 3 * (101 + i));
      ids.push_back(100 + i);
    }
  }
  PHX_CHECK(refused > 0 && refused < 300);
  PHX_CHECK(statistics(handle)[PHX_MC_TRIANGULATION_3D_REFUSED] == refused);
  phx_mc_mesh* subset = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(static_cast<int64_t>(ids.size()), accepted.data(), INT64_MAX / 2,
                               &subset) == PHX_MC_OK);
  Mesh expected = read(subset);
  for (int32_t& v : expected.cells) {
    v = ids[static_cast<std::size_t>(v)];
  }
  PHX_CHECK(finalize(handle).cells == expected.cells);
  phx_mc_triangulation_3d_free(handle);

  // Batch validation and id capacity refuse the whole call.
  handle = nullptr;
  PHX_CHECK(phx_mc_triangulation_3d_create(100, points.data(), 101, INT64_MAX / 2, 64,
                                           &handle) == PHX_MC_OK);
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, 2, points.data() + 300, INT64_MAX,
                                           vertices.data(),
                                           status.data()) == PHX_MC_CAPACITY_EXCEEDED);
  std::vector<double> bad(points.begin() + 300, points.begin() + 303);
  bad[1] = 0x1p130;
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, 1, bad.data(), INT64_MAX, vertices.data(),
                                           status.data()) == PHX_MC_RANGE_ERROR);
  PHX_CHECK(statistics(handle)[PHX_MC_TRIANGULATION_3D_VERTEX_IDS] == 100);
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, -1, bad.data(), INT64_MAX, vertices.data(),
                                           status.data()) == PHX_MC_INVALID_ARGUMENT);
  phx_mc_triangulation_3d_free(handle);
  PHX_CHECK(phx_mc_triangulation_3d_create(3, points.data(), 10, 100, 64, &handle) ==
            PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(handle == nullptr);
}

// Cube lattice with its mid-plane z = 1 constrained: the plane survives every
// insertion, regions separated by it stay consistent, points on it are refused
// unchanged, and the triangulation stays constrained Delaunay.
void test_prepared_constrained_plane_and_regions() {
  std::vector<double> points;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        points.insert(points.end(), {double(i), double(j), double(k)});
      }
    }
  }
  phx_mc_triangulation_3d* handle = create(points, 27);
  const Mesh initial = finalize(handle);
  std::vector<int32_t> facets;
  for (std::size_t c = 0; c < initial.cells.size() / 4; ++c) {
    const int32_t* v = initial.cells.data() + 4 * c;
    for (int s = 0; s < 4; ++s) {
      int32_t facet[3];
      oriented_facet(v, s, facet);
      bool plane = true;
      for (int32_t vertex : facet) {
        plane = plane && points[static_cast<std::size_t>(3 * vertex + 2)] == 1.0;
      }
      // Both sides of each plane facet are listed; marking is idempotent.
      if (plane) {
        facets.insert(facets.end(), facet, facet + 3);
      }
    }
  }
  const int64_t facet_count = static_cast<int64_t>(facets.size() / 3);
  std::vector<int32_t> ids(static_cast<std::size_t>(facet_count), 4);
  std::vector<int32_t> status(static_cast<std::size_t>(facet_count));
  PHX_CHECK(phx_mc_triangulation_3d_constrain_facets(handle, facet_count, facets.data(),
                                                     ids.data(), status.data()) == PHX_MC_OK);
  PHX_CHECK(std::all_of(status.begin(), status.end(), [](int32_t s) { return s == PHX_MC_OK; }));
  const int32_t absent[3] = {0, 1, 26};
  const int32_t other_id = 5;
  int32_t item = -1;
  PHX_CHECK(phx_mc_triangulation_3d_constrain_facets(handle, 1, absent, &other_id, &item) ==
            PHX_MC_OK);
  PHX_CHECK(item == PHX_MC_INVALID_INPUT);

  const double seeds[6] = {0.31, 0.57, 0.23, 0.31, 0.57, 1.77};
  const int32_t labels[2] = {0, 1};
  int32_t seed_status[2] = {-1, -1};
  PHX_CHECK(phx_mc_triangulation_3d_label_regions(handle, 2, seeds, labels, seed_status) ==
            PHX_MC_OK);
  PHX_CHECK(seed_status[0] == PHX_MC_OK && seed_status[1] == PHX_MC_OK);

  // A point on the constrained plane is refused without change.
  const Mesh before = finalize(handle);
  const double on_plane[3] = {0.3, 0.6, 1.0};
  int32_t vertex = 0;
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, 1, on_plane, INT64_MAX, &vertex, &item) ==
            PHX_MC_OK);
  PHX_CHECK(item == PHX_MC_CONSTRAINT_INTERSECTION && vertex == -1);
  PHX_CHECK(finalize(handle).cells == before.cells);

  std::vector<double> extra = random_points(300, 37);
  for (double& x : extra) {
    x *= 2.0;
  }
  std::vector<int32_t> vertices(300);
  std::vector<int32_t> inserted(300);
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, 300, extra.data(), INT64_MAX, vertices.data(),
                                           inserted.data()) == PHX_MC_OK);
  PHX_CHECK(std::all_of(inserted.begin(), inserted.end(),
                        [](int32_t s) { return s == PHX_MC_OK; }));
  const Mesh mesh = finalize(handle);
  std::vector<double> all = points;
  all.push_back(0.3);
  all.push_back(0.6);
  all.push_back(1.0);
  all.insert(all.end(), extra.begin(), extra.end());
  PHX_CHECK(mesh.points == all);
  auto at = [&](int32_t v) { return all.data() + 3 * static_cast<int64_t>(v); };
  double constrained_area = 0.0;
  int wrong_region = 0;
  int non_delaunay = 0;
  std::vector<std::array<int32_t, 5>> facet_list;  // sorted facet, cell, slot
  const std::size_t cells = mesh.cells.size() / 4;
  for (std::size_t c = 0; c < cells; ++c) {
    const int32_t* v = mesh.cells.data() + 4 * c;
    PHX_CHECK(orient3d(at(v[0]), at(v[1]), at(v[2]), at(v[3])) > 0);
    const double z = (at(v[0])[2] + at(v[1])[2] + at(v[2])[2] + at(v[3])[2]) / 4.0;
    wrong_region += mesh.regions[c] != (z < 1.0 ? 0 : 1) ? 1 : 0;
    for (int s = 0; s < 4; ++s) {
      std::array<int32_t, 5> entry{};
      int count = 0;
      for (int r = 0; r < 4; ++r) {
        if (r != s) {
          entry[static_cast<std::size_t>(count++)] = v[r];
        }
      }
      std::sort(entry.begin(), entry.begin() + 3);
      entry[3] = static_cast<int32_t>(c);
      entry[4] = s;
      facet_list.push_back(entry);
      if (mesh.constraints[4 * c + static_cast<std::size_t>(s)] == 4) {
        const double* p = at(entry[0]);
        const double* q = at(entry[1]);
        const double* r = at(entry[2]);
        PHX_CHECK(p[2] == 1.0 && q[2] == 1.0 && r[2] == 1.0);
        constrained_area +=
            0.5 * std::fabs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
      }
    }
  }
  // Both sides of the plane carry the constraint: twice the 2 x 2 square.
  PHX_CHECK(std::fabs(constrained_area - 8.0) < 1e-12);
  PHX_CHECK(wrong_region == 0);
  std::sort(facet_list.begin(), facet_list.end());
  for (std::size_t i = 0; i + 1 < facet_list.size(); ++i) {
    const auto& a = facet_list[i];
    const auto& b = facet_list[i + 1];
    if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2]) {
      continue;
    }
    if (mesh.constraints[4 * static_cast<std::size_t>(a[3]) + static_cast<std::size_t>(a[4])] >=
        0) {
      continue;
    }
    const int32_t* first = mesh.cells.data() + 4 * a[3];
    const int32_t* second = mesh.cells.data() + 4 * b[3];
    const int32_t apex = second[b[4]];
    non_delaunay += insphere_sos(at(first[0]), at(first[1]), at(first[2]), at(first[3]),
                                 at(apex), first[0], first[1], first[2], first[3], apex) > 0
                        ? 1
                        : 0;
  }
  PHX_CHECK(non_delaunay == 0);

  // Location on the constrained plane, at a vertex and outside the hull.
  const double queries[9] = {0.3, 0.6, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0};
  int32_t located[12];
  int8_t kinds[3] = {9, 9, 9};
  int32_t query_status[3];
  PHX_CHECK(phx_mc_triangulation_3d_locate(handle, 3, queries, located, kinds, query_status) ==
            PHX_MC_OK);
  PHX_CHECK(kinds[0] == 1 && kinds[1] == 3 && kinds[2] == -1 && located[8] == -1);
  phx_mc_triangulation_3d_free(handle);
}

void test_prepared_refused_allocation() {
  const std::vector<double> points = random_points(600, 41);
  phx_mc_triangulation_3d* handle = create(points, 300);
  std::vector<int32_t> vertices(300, 5);
  std::vector<int32_t> status(300, 5);
  refuse_allocations.store(true);
  const int32_t call = phx_mc_triangulation_3d_insert(handle, 300, points.data() + 900, INT64_MAX,
                                                      vertices.data(), status.data());
  refuse_allocations.store(false);
  PHX_CHECK(call == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(statistics(handle)[PHX_MC_TRIANGULATION_3D_VERTEX_IDS] == 300);
  PHX_CHECK(phx_mc_triangulation_3d_insert(handle, 300, points.data() + 900, INT64_MAX,
                                           vertices.data(), status.data()) == PHX_MC_OK);
  phx_mc_mesh* reference = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(600, points.data(), INT64_MAX / 2, &reference) == PHX_MC_OK);
  PHX_CHECK(finalize(handle).cells == read(reference).cells);
  phx_mc_triangulation_3d_free(handle);
}

}  // namespace

void* operator new(std::size_t size) { return allocate_bytes(size); }
void* operator new[](std::size_t size) { return allocate_bytes(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

int main() {
  test_flip_validation_matches_geometry();
  test_flip_commit_revert_and_slot_reuse();
  test_refusals_leave_the_complex_unchanged();
  test_slot_limit_is_a_refusal();
  test_constrained_facets_survive();
  test_refused_allocation_leaves_the_complex_unchanged();
  test_prepared_batches_match_the_point_set();
  test_prepared_limits_refuse_without_change();
  test_prepared_constrained_plane_and_regions();
  test_prepared_refused_allocation();
  return phx::mc::test::finish("test_cavity");
}
