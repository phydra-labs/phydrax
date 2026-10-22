//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Delegated curved-source boundary collapse: declared curved face sources admit
// non-coplanar folds only within one declared source; a nonmutating inspection
// reports the validated candidate; its commit requires host claims bound to the
// exact new faces, their bitwise coordinates, the declared digest, the current
// accepted generation and the declared tolerance, and retains them as delegated
// claims distinct from planar PLC rows.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <set>
#include <limits>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "tet_mesh.hpp"
#include "tet_mesh_fixtures.hpp"

namespace {

using phx::mc::test::Domain;

constexpr int kRing = 6;
constexpr int32_t kTop = 0;
constexpr int32_t kBottom = 1;
constexpr int32_t kCenter = 2;
constexpr int32_t kFirstRing = 3;
constexpr double kTolerance = 0.5;

void orient(const std::vector<double>& points, int32_t* cell) {
  const auto at = [&](int32_t v) { return points.data() + 3 * v; };
  if (phx::mc::orient3d(at(cell[0]), at(cell[1]), at(cell[2]), at(cell[3])) < 0) {
    std::swap(cell[0], cell[1]);
  }
}

// Hexagonal bicone: a raised top vertex and a lowered bottom vertex over an
// equatorial ring, closed by one interior center vertex. Top and bottom fans
// are two curved sources meeting along the ring.
Domain bicone(bool ring_segments) {
  Domain domain;
  domain.points = {0.0, 0.0, 0.25, 0.0, 0.0, -1.0, 0.0, 0.0, -0.25};
  for (int i = 0; i < kRing; ++i) {
    const double angle = 2.0 * M_PI * i / kRing;
    domain.points.insert(domain.points.end(), {std::cos(angle), std::sin(angle), 0.0});
  }
  for (int i = 0; i < kRing; ++i) {
    const int32_t a = kFirstRing + i, b = kFirstRing + (i + 1) % kRing;
    for (int32_t apex : {kTop, kBottom}) {
      int32_t cell[4] = {kCenter, apex, a, b};
      orient(domain.points, cell);
      domain.tets.insert(domain.tets.end(), cell, cell + 4);
      domain.regions.push_back(0);
      domain.faces.insert(domain.faces.end(), {apex, a, b});
      domain.face_sources.push_back(apex == kTop ? 0 : 1);
    }
    if (ring_segments) {
      domain.segments.insert(domain.segments.end(), {a, b});
      domain.segment_sources.push_back(0);
    }
  }
  return domain;
}

std::array<uint8_t, 64> digests() {
  std::array<uint8_t, 64> value{};
  for (std::size_t i = 0; i < value.size(); ++i) value[i] = static_cast<uint8_t>(i + 1);
  return value;
}

phx_mc_tet_mesh* create(const Domain& domain, bool declare, int32_t* status,
                        const std::vector<double>* claims_override = nullptr) {
  static const int32_t sources[2] = {0, 1};
  static const double tolerances[2] = {kTolerance, kTolerance};
  static const auto digest = digests();
  std::vector<double> claims(domain.face_sources.size(), 0.125);
  if (claims_override != nullptr) claims = *claims_override;
  phx_mc_tet_mesh* mesh = nullptr;
  const phx_mc_tet_mesh_create_request request{
      .point_count = static_cast<int64_t>(domain.points.size() / 3),
      .points = domain.points.data(),
      .tet_count = static_cast<int64_t>(domain.regions.size()),
      .tets = domain.tets.data(),
      .tet_regions = domain.regions.data(),
      .face_count = static_cast<int64_t>(domain.face_sources.size()),
      .faces = domain.faces.data(),
      .face_sources = domain.face_sources.data(),
      .segment_count = static_cast<int64_t>(domain.segment_sources.size()),
      .segments = domain.segments.data(),
      .segment_sources = domain.segment_sources.data(),
      .protection_radii = nullptr,
      .source_point_count = 0,
      .source_points = nullptr,
      .source_face_count = 0,
      .source_faces = nullptr,
      .source_face_group_rows = nullptr,
      .source_face_tolerances = nullptr,
      .source_segment_count = 0,
      .source_segments = nullptr,
      .source_segment_group_rows = nullptr,
      .source_segment_tolerances = nullptr,
      .witness_strata = nullptr,
      .witness_entities = nullptr,
      .witness_parameters = nullptr,
      .curved_source_count = declare ? 2 : 0,
      .curved_sources = declare ? sources : nullptr,
      .curved_tolerances = declare ? tolerances : nullptr,
      .curved_digests = declare ? digest.data() : nullptr,
      .face_claims = declare ? claims.data() : nullptr,
      .boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING,
      .max_vertices = 64,
      .max_tetrahedra = 256,
      .max_scratch_bytes = INT64_MAX,
      .mesh = &mesh,
  };
  *status = phx_mc_tet_mesh_create(&request);
  return mesh;
}

struct Inspected {
  int64_t count = 0;
  int32_t source = -1;
  int64_t generation = -1;
  std::vector<int32_t> faces;
};

Inspected inspect(phx_mc_tet_mesh* mesh, int32_t remove, int32_t keep) {
  Inspected result;
  result.faces.assign(3 * 64, -1);
  PHX_CHECK(phx_mc_tet_mesh_inspect_curved_collapse(mesh, remove, keep, 0, INT64_MAX, 64,
                                                     &result.count, &result.source,
                                                     &result.generation,
                                                     result.faces.data()) == PHX_MC_OK);
  result.faces.resize(static_cast<std::size_t>(3 * result.count));
  return result;
}

std::vector<double> coordinates(const Domain& domain, const std::vector<int32_t>& faces) {
  std::vector<double> values;
  for (int32_t v : faces) {
    values.insert(values.end(), domain.points.begin() + 3 * v, domain.points.begin() + 3 * v + 3);
  }
  return values;
}

int32_t commit(phx_mc_tet_mesh* mesh, int32_t remove, int32_t keep, const Inspected& plan,
               int64_t generation, const uint8_t* digest, const std::vector<int32_t>& faces,
               const std::vector<double>& coords, const std::vector<double>& deviations,
               int64_t work_limit = INT64_MAX, int32_t expected_status = PHX_MC_OK) {
  int32_t applied = -1;
  const phx_mc_tet_mesh_curved_collapse_request request{
      .mesh = mesh,
      .remove = remove,
      .keep = keep,
      .source = plan.source,
      .accepted_generation = generation,
      .digest = digest,
      .claim_count = static_cast<int64_t>(faces.size() / 3),
      .claim_faces = faces.data(),
      .claim_coordinates = coords.data(),
      .claim_deviations = deviations.data(),
      .require_improvement = 0,
      .work_limit = work_limit,
      .applied = &applied,
  };
  PHX_CHECK(phx_mc_tet_mesh_collapse_curved_edge(&request) == expected_status);
  return applied;
}

int64_t claim_count(const phx_mc_tet_mesh* mesh) {
  int64_t count = -1;
  PHX_CHECK(phx_mc_tet_mesh_curved_claims(mesh, 0, &count, nullptr, nullptr, nullptr, nullptr,
                                          nullptr) == PHX_MC_OK);
  return count;
}

void test_undeclared_curved_folds_are_refused() {
  int32_t status = PHX_MC_OK;
  phx_mc_tet_mesh* mesh = create(bicone(true), false, &status);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
}

void test_fold_between_two_curved_sources_is_refused() {
  int32_t status = PHX_MC_OK;
  phx_mc_tet_mesh* mesh = create(bicone(false), true, &status);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh == nullptr);
}

void test_declared_claims_are_complete_finite_and_bounded() {
  const Domain domain = bicone(true);
  for (double bad : {std::numeric_limits<double>::quiet_NaN(), 2.0 * kTolerance, -1.0,
                     std::numeric_limits<double>::infinity()}) {
    std::vector<double> claims(domain.face_sources.size(), 0.125);
    claims[3] = bad;
    int32_t status = PHX_MC_OK;
    phx_mc_tet_mesh* mesh = create(domain, true, &status, &claims);
    PHX_CHECK(status == PHX_MC_INVALID_INPUT);
    PHX_CHECK(mesh == nullptr);
  }
}

void test_inspected_curved_collapse_commits_only_exact_claims() {
  const Domain domain = bicone(true);
  int32_t status = PHX_MC_OK;
  phx_mc_tet_mesh* mesh = create(domain, true, &status);
  PHX_CHECK(status == PHX_MC_OK);
  if (mesh == nullptr) return;
  PHX_CHECK(claim_count(mesh) == 2 * kRing);
  const auto before = phx::mc::test::export_mesh(mesh);
  // Feature (segment) vertices and the interior are never curved removals.
  PHX_CHECK(inspect(mesh, kFirstRing, kFirstRing + 1).count == 0);
  PHX_CHECK(inspect(mesh, kCenter, kTop).count == 0);
  const Inspected plan = inspect(mesh, kTop, kFirstRing);
  PHX_CHECK(plan.count == kRing - 2);
  PHX_CHECK(plan.source == 0);
  int64_t generation = -1;
  PHX_CHECK(phx_mc_tet_mesh_accepted_generation(mesh, &generation) == PHX_MC_OK);
  PHX_CHECK(plan.generation == generation);
  // Inspection never mutates the accepted state.
  const auto inspected = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(inspected.tets == before.tets && inspected.faces == before.faces);
  const auto digest = digests();
  const std::vector<double> coords = coordinates(domain, plan.faces);
  const std::vector<double> good(static_cast<std::size_t>(plan.count), 0.25);
  // Stale generation, wrong digest, perturbed coordinates, missing face,
  // out-of-tolerance, NaN and negative claims are all refused unchanged.
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation + 1, digest.data(), plan.faces,
                   coords, good) == 0);
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data() + 32, plan.faces,
                   coords, good) == 0);
  std::vector<double> perturbed = coords;
  perturbed[0] = std::nextafter(perturbed[0], 2.0);
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data(), plan.faces,
                   perturbed, good) == 0);
  auto reordered = plan.faces;
  std::swap(reordered[1], reordered[2]);
  // Row permutations are legal only with the correspondingly ordered exact
  // corner coordinates, never by ignoring the delegated coordinate binding.
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data(), reordered,
                   coords, good) == 0);
  auto duplicate = plan.faces;
  std::copy_n(duplicate.begin(), 3, duplicate.begin() + 3);
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data(), duplicate,
                   coordinates(domain, duplicate), good) == 0);
  const std::vector<int32_t> fewer(plan.faces.begin(), plan.faces.end() - 3);
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data(), fewer,
                   coordinates(domain, fewer),
                   std::vector<double>(static_cast<std::size_t>(plan.count - 1), 0.25)) == 0);
  for (double bad : {2.0 * kTolerance, std::numeric_limits<double>::quiet_NaN(), -0.5}) {
    std::vector<double> deviations = good;
    deviations.back() = bad;
    PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data(), plan.faces,
                     coords, deviations) == 0);
  }
  const auto refused = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(refused.tets == before.tets && refused.faces == before.faces);
  PHX_CHECK(claim_count(mesh) == 2 * kRing);
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data(), plan.faces, coords,
                   good) == 1);
  const auto after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(handle_audit(mesh));
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK(after.dimension[kTop] < 0);
  PHX_CHECK(after.face_sources.size() == before.face_sources.size() - 2);
  // Delegated claims are retained with their generation; the source-evidence
  // ancestry of every face (including committed curved faces) still resolves.
  const int64_t count = claim_count(mesh);
  PHX_CHECK(count == 2 * kRing + plan.count);
  std::vector<int32_t> rows(static_cast<std::size_t>(count)), sources(rows.size());
  std::vector<double> values(9 * rows.size()), deviations(rows.size());
  std::vector<int64_t> generations(rows.size());
  int64_t filled = -1;
  PHX_CHECK(phx_mc_tet_mesh_curved_claims(mesh, count, &filled, rows.data(), sources.data(),
                                          values.data(), deviations.data(),
                                          generations.data()) == PHX_MC_OK);
  PHX_CHECK(filled == count);
  for (int64_t i = 2 * kRing; i < count; ++i) {
    PHX_CHECK(sources[static_cast<std::size_t>(i)] == 0);
    PHX_CHECK(deviations[static_cast<std::size_t>(i)] == 0.25);
    PHX_CHECK(generations[static_cast<std::size_t>(i)] == generation);
  }
  int64_t ancestors = -1;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(mesh, 0, &ancestors, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, nullptr,
                                            nullptr) == PHX_MC_OK);
  PHX_CHECK(ancestors >= static_cast<int64_t>(after.face_sources.size()));
  // The committed state advanced; the consumed plan cannot be replayed.
  PHX_CHECK(commit(mesh, kTop, kFirstRing, plan, generation, digest.data(), plan.faces, coords,
                   good) == 0);
  phx_mc_tet_mesh_free(mesh);
}

// Include the public source CSR and the actual witness row/fraction readout,
// not merely the exported mesh's source labels.
std::vector<uint64_t> source_readout(phx_mc_tet_mesh* handle) {
  const auto mesh = phx::mc::test::export_mesh(handle);
  int64_t count = 0;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &count, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  std::vector<int64_t> offsets(mesh.face_sources.size() + 1);
  std::vector<int32_t> rows(static_cast<std::size_t>(count)), segments(mesh.segment_sources.size() + 1);
  std::vector<int8_t> strata(mesh.dimension.size());
  std::vector<int32_t> entities(strata.size());
  std::vector<double> parameters(2 * strata.size()), deviations(strata.size());
  double bounds[2];
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, count, &count, bounds, offsets.data(),
      rows.data(), segments.data(), strata.data(), entities.data(), parameters.data(),
      deviations.data()) == PHX_MC_OK);
  std::vector<uint64_t> result;
  const auto append = [&](const auto& values) {
    for (const auto value : values) result.push_back(static_cast<uint64_t>(value));
  };
  append(offsets); append(rows); append(segments); append(strata); append(entities);
  for (double value : bounds) result.push_back(std::bit_cast<uint64_t>(value));
  for (double value : parameters) result.push_back(std::bit_cast<uint64_t>(value));
  for (double value : deviations) result.push_back(std::bit_cast<uint64_t>(value));
  for (int32_t source : {0, 1}) {
    result.push_back(std::bit_cast<uint64_t>(handle->mesh->curved_tolerance(source)));
    const auto* digest = handle->mesh->curved_digest(source);
    PHX_CHECK(digest != nullptr);
    for (int k = 0; k < 32; ++k) result.push_back(digest[k]);
  }
  for (const auto& claim : handle->mesh->curved_claims()) {
    result.insert(result.end(), {static_cast<uint64_t>(claim.row), static_cast<uint64_t>(claim.source),
        std::bit_cast<uint64_t>(claim.deviation), static_cast<uint64_t>(claim.generation)});
  }
  for (const auto& face : handle->mesh->original_faces()) {
    for (int32_t value : face) result.push_back(static_cast<uint64_t>(value));
    for (int k = 0; k < 3; ++k) {
      for (int axis = 0; axis < 3; ++axis) {
        result.push_back(std::bit_cast<uint64_t>(handle->mesh->original_point(face[k])[axis]));
      }
    }
  }
  return result;
}

int32_t rebuild_curved(phx_mc_tet_mesh* handle, const Domain& domain,
                        const std::array<uint8_t, 64>& digest,
                        const std::array<double, 2>& tolerances) {
  const int32_t sources[2] = {0, 1};
  const std::vector<double> claims(domain.face_sources.size(), 0.125);
  const phx::mc::CurvedSourceDeclaration curved{
      2, sources, tolerances.data(), digest.data(), claims.data()};
  return handle->mesh->build(static_cast<int64_t>(domain.points.size() / 3), domain.points.data(),
      static_cast<int64_t>(domain.regions.size()), domain.tets.data(), domain.regions.data(),
      static_cast<int64_t>(domain.face_sources.size()), domain.faces.data(), domain.face_sources.data(),
      static_cast<int64_t>(domain.segment_sources.size()), domain.segments.data(),
      domain.segment_sources.data(), nullptr, nullptr, &curved);
}

void test_retained_collapse_matches_fresh_and_invalidates_every_prepare() {
  const Domain domain = bicone(true);
  const auto digest = digests();
  std::vector<uint64_t> expected_source;
  phx::mc::test::Exported expected;
  int64_t reused_work = 0;
  for (int mode = 0; mode < 6; ++mode) {
    int32_t status;
    auto* handle = create(domain, true, &status);
    PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
    const auto before = phx::mc::test::export_mesh(handle);
    const auto old_source = source_readout(handle);
    const auto plan = inspect(handle, kTop, kFirstRing);
    const auto coords = coordinates(domain, plan.faces);
    const std::vector<double> deviations(static_cast<std::size_t>(plan.count), 0.25);
    PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
    PHX_CHECK(source_readout(handle) == old_source);
    if (mode == 1) {
      handle->mesh->cancel_prepared_edit();
      const auto released = handle->mesh->memory_evidence()[1];
      handle->mesh->cancel_prepared_edit();
      PHX_CHECK(handle->mesh->memory_evidence()[1] == released);
    }
    if (mode == 2) {
      // An unsuccessful ordinary preparation must invalidate, not keep a
      // warm curved owner merely because it did not reach edit_.begin().
      int32_t applied = -1;
      PHX_CHECK(phx_mc_tet_mesh_collapse_edge(handle, -1, kFirstRing, 0, INT64_MAX,
                                              &applied) == PHX_MC_OK && applied == 0);
    }
    if (mode == 3) {
      auto empty = handle->mesh->prepare_replacement({}, {});
      PHX_CHECK(empty.status() != phx::mc::EditStatus::kOk);
    }
    if (mode == 4) {
      // Another inspected candidate owns the same edit, not a second cache.
      PHX_CHECK(inspect(handle, kTop, kFirstRing + 1).count == plan.count);
    }
    if (mode == 5) {
      PHX_CHECK(rebuild_curved(handle, domain, digest,
          {std::numeric_limits<double>::quiet_NaN(), kTolerance}) == PHX_MC_INVALID_INPUT);
      PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
      PHX_CHECK(source_readout(handle) == old_source);
    }
    if (mode == 0) {
      const phx::mc::CurvedCollapseClaims claims{plan.source, plan.generation, digest.data(),
          plan.count, plan.faces.data(), coords.data(), deviations.data()};
      bool applied = false;
      PHX_CHECK(!handle->mesh->reuse_curved_collapse(kTop, kFirstRing, true, claims, applied));
      PHX_CHECK(!applied);
    }
    const int64_t initial_work = handle->mesh->work();
    PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, digest.data(),
                     plan.faces, coords, deviations) == 1);
    const int64_t work = handle->mesh->work() - initial_work;
    if (mode == 0) {
      reused_work = work;
      PHX_CHECK(work == phx::mc::TetWorkCost::kFacet * plan.count);
      expected = phx::mc::test::export_mesh(handle);
      expected_source = source_readout(handle);
      int32_t fresh_status;
      auto* fresh = create(domain, true, &fresh_status);
      PHX_CHECK(fresh_status == PHX_MC_OK && fresh != nullptr);
      const int64_t fresh_work = fresh->mesh->work();
      PHX_CHECK(commit(fresh, kTop, kFirstRing, plan, plan.generation, digest.data(),
                       plan.faces, coords, deviations) == 1);
      PHX_CHECK(fresh->mesh->work() - fresh_work > reused_work);
      PHX_CHECK(phx::mc::test::export_mesh(fresh) == expected);
      PHX_CHECK(source_readout(fresh) == expected_source);
      phx_mc_tet_mesh_free(fresh);
    } else {
      PHX_CHECK(work > reused_work);
      PHX_CHECK(phx::mc::test::export_mesh(handle) == expected);
      PHX_CHECK(source_readout(handle) == expected_source);
    }
    PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, digest.data(),
                     plan.faces, coords, deviations) == 0);
    phx_mc_tet_mesh_free(handle);
  }
}

void test_retained_collapse_work_and_scratch_boundaries_are_atomic() {
  const Domain domain = bicone(true);
  const auto digest = digests();
  std::size_t scratch = 0;
  for (int mode = 0; mode < 5; ++mode) {
    int32_t status;
    auto* handle = create(domain, true, &status);
    PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
    const auto before = phx::mc::test::export_mesh(handle);
    const auto plan = inspect(handle, kTop, kFirstRing);
    const auto coords = coordinates(domain, plan.faces);
    const std::vector<double> deviations(static_cast<std::size_t>(plan.count), 0.25);
    const int64_t exact_work = phx::mc::TetWorkCost::kFacet * plan.count;
    const bool refused = mode == 1 || mode == 4;
    {
      const std::size_t allowance = mode == 3 ? scratch : mode == 4 ? scratch - 1 : 1 << 24;
      phx::mc::MemoryBudgetWindow window(allowance, handle->mesh->memory_owner());
      PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, digest.data(),
          plan.faces, coords, deviations, mode < 2 ? exact_work - mode : INT64_MAX,
          refused ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK) == (refused ? 0 : 1));
      if (mode == 2) scratch = static_cast<std::size_t>(window.evidence()[2]);
    }
    if (refused) {
      PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
      PHX_CHECK(claim_count(handle) == 2 * kRing);
      PHX_CHECK(!handle->mesh->broken());
      // Both work refusal and an allocation exception retire the owner once;
      // a later legitimate cache miss can still validate the unchanged input.
      PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, digest.data(),
                       plan.faces, coords, deviations) == 1);
    }
    phx_mc_tet_mesh_free(handle);
  }
}

void test_retained_collapse_stales_after_an_accepted_edit() {
  const Domain domain = bicone(true);
  int32_t status;
  auto* handle = create(domain, true, &status);
  PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
  const auto plan = inspect(handle, kTop, kFirstRing);
  const double midpoint[3] = {0.0, 0.0, -0.625};
  int32_t inserted = -1;
  PHX_CHECK(phx_mc_tet_mesh_split_edge(handle, kCenter, kBottom, midpoint, 0.0, 0.5,
                                       INT64_MAX, &inserted) == PHX_MC_OK && inserted >= 0);
  const auto accepted = phx::mc::test::export_mesh(handle);
  const auto digest = digests();
  PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, digest.data(),
      plan.faces, coordinates(domain, plan.faces),
      std::vector<double>(static_cast<std::size_t>(plan.count), 0.25)) == 0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == accepted);
  phx_mc_tet_mesh_free(handle);
}

void test_retained_collapse_refused_readout_and_changed_source_cannot_reuse() {
  const Domain domain = bicone(true);
  int32_t status;
  auto* handle = create(domain, true, &status);
  PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
  const auto before = phx::mc::test::export_mesh(handle);
  const auto old_source = source_readout(handle);
  const auto plan = inspect(handle, kTop, kFirstRing);
  int64_t count = -1, generation = -1;
  int32_t source = -1, one_face[3]{};
  PHX_CHECK(phx_mc_tet_mesh_inspect_curved_collapse(handle, kTop, kFirstRing, 0,
      INT64_MAX, 1, &count, &source, &generation, one_face) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(count == 0 && source == -1 && generation == -1);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(source_readout(handle) == old_source);
  auto digest = digests();
  digest[0] ^= 0x80;
  PHX_CHECK(rebuild_curved(handle, domain, digest, {kTolerance, kTolerance}) == PHX_MC_OK);
  const auto rebound = source_readout(handle);
  PHX_CHECK(rebound != old_source);
  const auto old_digest = digests();
  PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, old_digest.data(),
      plan.faces, coordinates(domain, plan.faces),
      std::vector<double>(static_cast<std::size_t>(plan.count), 0.25)) == 0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(source_readout(handle) == rebound);
  const auto current = inspect(handle, kTop, kFirstRing);
  PHX_CHECK(commit(handle, kTop, kFirstRing, current, current.generation, digest.data(),
      current.faces, coordinates(domain, current.faces),
      std::vector<double>(static_cast<std::size_t>(current.count), 0.25)) == 1);
  phx_mc_tet_mesh_free(handle);
}

void test_successive_same_source_collapses_preserve_raw_and_outward_claim_rows() {
  Domain domain = bicone(true);
  std::array<int32_t, 2> steiner{};
  for (int patch = 0; patch < 2; ++patch) {
    const int edge = 3 * patch;
    const int32_t a = kTop, b = kFirstRing + edge;
    const int32_t c = kFirstRing + (edge + 1) % kRing;
    const int32_t inserted = static_cast<int32_t>(domain.points.size() / 3);
    steiner[patch] = inserted;
    // Author a genuinely curved three-face patch, rather than subdividing an
    // existing plane. Coarsening an inverse planar subdivision would append a
    // delegated row overlapping the original source row's positive-area owner.
    domain.points.insert(domain.points.end(), {
        0.25 * (domain.points[3 * b] + domain.points[3 * c]),
        0.25 * (domain.points[3 * b + 1] + domain.points[3 * c + 1]), 0.2});
    const std::array<std::array<int32_t, 3>, 3> faces{{
        {a, b, inserted}, {b, c, inserted}, {c, a, inserted}}};
    for (std::size_t child = 0; child < faces.size(); ++child) {
      const auto& face = faces[child];
      int32_t cell[4] = {kCenter, face[0], face[1], face[2]};
      orient(domain.points, cell);
      if (child == 0) {
        std::copy_n(cell, 4, domain.tets.begin() + 4 * (2 * edge));
        std::copy(face.begin(), face.end(), domain.faces.begin() + 3 * (2 * edge));
      } else {
        domain.tets.insert(domain.tets.end(), cell, cell + 4);
        domain.regions.push_back(0);
        domain.faces.insert(domain.faces.end(), face.begin(), face.end());
        domain.face_sources.push_back(0);
      }
    }
  }
  const auto digest = digests();
  const auto compare_internal = [](phx::mc::TetMesh& left, phx::mc::TetMesh& right) {
    const auto& a = left.complex();
    const auto& b = right.complex();
    PHX_CHECK(a.tets.size() == b.tets.size() && a.finite_count == b.finite_count);
    PHX_CHECK(std::equal(a.free_slots.begin(), a.free_slots.end(),
                         b.free_slots.begin(), b.free_slots.end()));
    for (std::size_t slot = 0; slot < a.tets.size(); ++slot) {
      const int32_t t = static_cast<int32_t>(slot);
      PHX_CHECK(a.live(t) == b.live(t));
      if (!a.live(t)) continue;
      PHX_CHECK(std::equal(a.tets[slot].v, a.tets[slot].v + 4, b.tets[slot].v));
      PHX_CHECK(std::equal(a.tets[slot].n, a.tets[slot].n + 4, b.tets[slot].n));
      PHX_CHECK(a.region(t) == b.region(t));
      for (int k = 0; k < 4; ++k) PHX_CHECK(a.constraint(t, k) == b.constraint(t, k));
    }
    PHX_CHECK(left.vertex_count() == right.vertex_count());
    for (int32_t v = 0; v < left.vertex_count(); ++v) {
      PHX_CHECK(left.live_vertex(v) == right.live_vertex(v));
      if (!left.live_vertex(v)) continue;
      // The first visited slot is the private incident locator; the complete
      // order also exposes native slot/neighbor drift hidden by sorted export.
      std::vector<int32_t> left_star, right_star;
      PHX_CHECK(left.vertex_star(v, left_star) && right.vertex_star(v, right_star));
      PHX_CHECK(left_star == right_star);
    }
  };
  for (bool outward : {false, true}) {
    std::array<phx_mc_tet_mesh*, 2> handles{};
    for (auto& handle : handles) {
      int32_t status;
      handle = create(domain, true, &status);
      PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
      PHX_CHECK(handle_audit(handle));
      for (int32_t vertex : steiner) {
        PHX_CHECK(handle->mesh->dimension(vertex) == 2 && handle->mesh->protection(vertex) == 0.0);
      }
    }
    PHX_CHECK(source_readout(handles[0]) == source_readout(handles[1]));
    compare_internal(*handles[0]->mesh, *handles[1]->mesh);
    for (int32_t removed : steiner) {
      const int64_t retained_start = handles[0]->mesh->work();
      auto plan = inspect(handles[0], removed, kTop);
      const int64_t inspection_work = handles[0]->mesh->work() - retained_start;
      const int64_t fresh_start = handles[1]->mesh->work();
      const auto fresh_plan = inspect(handles[1], removed, kTop);
      PHX_CHECK(handles[1]->mesh->work() - fresh_start == inspection_work);
      PHX_CHECK(plan.count > 0 && plan.source == 0 && fresh_plan.source == 0);
      PHX_CHECK(plan.faces == fresh_plan.faces && plan.generation == fresh_plan.generation);
      if (outward) {
        // This is the actual Python adapter's public convention. Do not
        // silently flip these rows back: their source-point order is evidence.
        for (int64_t i = 0; i < plan.count; ++i) std::swap(plan.faces[3 * i + 1], plan.faces[3 * i + 2]);
      }
      std::vector<double> coords;
      for (int32_t v : plan.faces) {
        const double* p = handles[0]->mesh->point(v);
        coords.insert(coords.end(), p, p + 3);
      }
      const std::vector<double> deviations(static_cast<std::size_t>(plan.count), 0.25);
      handles[1]->mesh->cancel_prepared_edit();
      const int64_t before_retained = handles[0]->mesh->work();
      PHX_CHECK(commit(handles[0], removed, kTop, plan, plan.generation, digest.data(),
                       plan.faces, coords, deviations) == 1);
      const int64_t retained_work = handles[0]->mesh->work() - before_retained;
      const int64_t before_fresh = handles[1]->mesh->work();
      PHX_CHECK(commit(handles[1], removed, kTop, plan, plan.generation, digest.data(),
                       plan.faces, coords, deviations) == 1);
      PHX_CHECK(retained_work == phx::mc::TetWorkCost::kFacet * plan.count);
      PHX_CHECK(handles[1]->mesh->work() - before_fresh == inspection_work + retained_work);
      PHX_CHECK(phx::mc::test::export_mesh(handles[0]) == phx::mc::test::export_mesh(handles[1]));
      PHX_CHECK(source_readout(handles[0]) == source_readout(handles[1]));
      compare_internal(*handles[0]->mesh, *handles[1]->mesh);
      for (auto* handle : handles) {
        const auto claims = handle->mesh->curved_claims();
        for (int64_t i = 0; i < plan.count; ++i) {
          const auto& claim = claims[claims.size() - static_cast<std::size_t>(plan.count) + i];
          PHX_CHECK(claim.source == 0 && claim.generation == plan.generation);
          const auto& face = handle->mesh->original_faces()[claim.row];
          for (int k = 0; k < 3; ++k) {
            for (int axis = 0; axis < 3; ++axis) {
              PHX_CHECK(std::bit_cast<uint64_t>(handle->mesh->original_point(face[k])[axis]) ==
                        std::bit_cast<uint64_t>(coords[9 * i + 3 * k + axis]));
            }
          }
        }
        PHX_CHECK(handle_audit(handle));
      }
    }
    for (auto* handle : handles) phx_mc_tet_mesh_free(handle);
  }
}

void test_fresh_malformed_claims_refuse_before_staging_work_and_scratch() {
  const Domain domain = bicone(true);
  const auto digest = digests();
  int32_t status;
  auto* donor = create(domain, true, &status);
  PHX_CHECK(status == PHX_MC_OK && donor != nullptr);
  std::vector<int32_t> star;
  PHX_CHECK(donor->mesh->vertex_star(kTop, star));
  const int64_t inspection_start = donor->mesh->work();
  const auto plan = inspect(donor, kTop, kFirstRing);
  const int64_t inspection_work = donor->mesh->work() - inspection_start;
  const int64_t prefix_work = inspection_work -
      phx::mc::TetWorkCost::kCellValidation * static_cast<int64_t>(star.size());
  PHX_CHECK(prefix_work > 0 && plan.count > 2);
  phx_mc_tet_mesh_free(donor);
  const auto good_coordinates = coordinates(domain, plan.faces);
  const std::vector<double> good_deviations(static_cast<std::size_t>(plan.count), 0.25);

  // A genuinely fresh valid commit earns each claim exactly once, in addition
  // to the same preparation and validated-cell work as an inspection.
  auto* fresh = create(domain, true, &status);
  PHX_CHECK(status == PHX_MC_OK && fresh != nullptr);
  const int64_t fresh_start = fresh->mesh->work();
  PHX_CHECK(commit(fresh, kTop, kFirstRing, plan, plan.generation, digest.data(),
      plan.faces, good_coordinates, good_deviations) == 1);
  PHX_CHECK(fresh->mesh->work() - fresh_start ==
      inspection_work + phx::mc::TetWorkCost::kFacet * plan.count);
  phx_mc_tet_mesh_free(fresh);

  // None of these target handles is inspected: every attempt is a real cache
  // miss. The independent inspection's known cell-validation charge separates
  // preparation from staging, so this allowance admits the offending claim
  // but cannot admit staging/validation before that claim is rejected.
  for (int malformed = 0; malformed < 6; ++malformed) {
    auto bad_faces = plan.faces;
    auto bad_deviations = good_deviations;
    int64_t checked_rows = 1;
    if (malformed == 0) bad_deviations[0] = -0.25;
    if (malformed == 1) {
      std::copy_n(bad_faces.begin(), 3, bad_faces.begin() + 3);
      checked_rows = 2;
    }
    if (malformed == 3) bad_deviations[0] = std::numeric_limits<double>::quiet_NaN();
    if (malformed == 4) {
      bad_faces.resize(bad_faces.size() - 3);
      bad_deviations.resize(bad_deviations.size() - 1);
      checked_rows = 0;
    }
    if (malformed == 5) bad_faces[0] = kBottom;
    auto bad_coordinates = coordinates(domain, bad_faces);
    if (malformed == 2) bad_coordinates[0] = std::nextafter(bad_coordinates[0], 2.0);
    const int64_t refusal_work = prefix_work + phx::mc::TetWorkCost::kFacet * checked_rows;
    PHX_CHECK(refusal_work < inspection_work);
    std::size_t refusal_scratch = 0;
    for (int boundary = 0; boundary < 4; ++boundary) {
      auto* handle = create(domain, true, &status);
      PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
      const auto before = phx::mc::test::export_mesh(handle);
      const auto before_source = source_readout(handle);
      const int64_t initial_work = handle->mesh->work();
      const bool scratch_refusal = boundary == 3;
      {
        const std::size_t allowance = boundary == 2 ? refusal_scratch
            : boundary == 3 ? refusal_scratch - 1 : 1 << 24;
        phx::mc::MemoryBudgetWindow window(allowance, handle->mesh->memory_owner());
        PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, digest.data(),
            bad_faces, bad_coordinates, bad_deviations,
            boundary == 0 ? refusal_work : INT64_MAX,
            scratch_refusal ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK) == 0);
        if (boundary == 1) {
          refusal_scratch = static_cast<std::size_t>(window.evidence()[2]);
          PHX_CHECK(refusal_scratch > 0);
        }
        if (boundary == 2) PHX_CHECK(window.evidence()[5] == 0);
        if (scratch_refusal) PHX_CHECK(window.evidence()[5] > 0);
      }
      if (!scratch_refusal) {
        PHX_CHECK(handle->mesh->work() - initial_work == refusal_work);
      }
      PHX_CHECK(!handle->mesh->broken());
      PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
      PHX_CHECK(source_readout(handle) == before_source);
      PHX_CHECK(commit(handle, kTop, kFirstRing, plan, plan.generation, digest.data(),
          plan.faces, good_coordinates, good_deviations) == 1);
      phx_mc_tet_mesh_free(handle);
    }
  }
}

void test_boundary_point_lookup_keeps_first_closed_row() {
  Domain domain = bicone(false);
  std::fill(domain.face_sources.begin(), domain.face_sources.end(), 0);
  int32_t status = PHX_MC_OK;
  phx_mc_tet_mesh* handle = create(domain, true, &status);
  PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
  if (handle == nullptr) return;
  // Each boundary vertex belongs to several closed boxes/triangles. The
  // indexed search must retain all boundary contacts and the exhaustive
  // search's lowest-row witness, not its spatial traversal's first hit.
  for (int32_t vertex = 0; vertex < kFirstRing + kRing; ++vertex) {
    if (vertex == kCenter) continue;
    int32_t expected = -1;
    for (std::size_t row = 0; row < domain.face_sources.size(); ++row) {
      const int32_t* face = domain.faces.data() + 3 * row;
      if (std::find(face, face + 3, vertex) != face + 3) {
        expected = static_cast<int32_t>(row);
        break;
      }
    }
    const auto& witness = handle->mesh->witness(vertex);
    PHX_CHECK(expected >= 0);
    PHX_CHECK(witness.stratum == phx::mc::SourceStratum::kFacet);
    PHX_CHECK(witness.entity == expected && witness.deviation == 0.0);
  }
  phx_mc_tet_mesh_free(handle);
}

void test_legacy_create_is_unchanged_by_absent_declaration() {
  // Planar PLC faces never need claims; a zero declaration is the legacy path.
  Domain box = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 21), 0.0);
  int32_t status = PHX_MC_OK;
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(box, PHX_MC_TET_MESH_BOUNDARY_CONFORMING,
                                                     nullptr, 1 << 20, 1 << 23, &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr && claim_count(mesh) == 0);
  phx_mc_tet_mesh_free(mesh);
}

// Rounded interior edge carriers: admitted only for an edge without a
// segment; a constrained face through the edge still refuses an off-plane
// carrier, and a committed body split carries no source witness.
void test_rounded_interior_carrier_is_unconstrained_only() {
  const Domain box =
      phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 21), 0.0);
  int32_t status = PHX_MC_OK;
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      box, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 1 << 20, 1 << 23, &status);
  PHX_CHECK(status == PHX_MC_OK && handle != nullptr);
  if (handle == nullptr) return;
  phx::mc::TetMesh& mesh = *handle->mesh;
  const auto before = phx::mc::test::export_mesh(handle);
  const auto at = [&](int32_t v) { return mesh.point(v); };
  const auto collinear = [&](int32_t a, int32_t b, const double* p) {
    return phx::mc::geometry::collinear(at(a), at(b), p);
  };
  // (2) A segment edge never admits a non-collinear witness.
  bool segment_checked = false;
  for (std::size_t s = 0; s + 1 < before.segments.size() && !segment_checked; s += 2) {
    const int32_t a = before.segments[s], b = before.segments[s + 1];
    double p[3];
    for (int k = 0; k < 3; ++k) p[k] = 0.5 * (at(a)[k] + at(b)[k]);
    int axis = 0;
    for (int k = 1; k < 3; ++k) {
      if (std::abs(at(a)[k] - at(b)[k]) > std::abs(at(a)[axis] - at(b)[axis])) axis = k;
    }
    const int off = (axis + 1) % 3;
    p[off] += p[off] < 0.5 ? 0x1p-20 : -0x1p-20;
    if (collinear(a, b, p)) continue;
    int32_t source = -1;
    bool on_edge = false;
    PHX_CHECK(!mesh.split_positions_admitted(a, b, p, p, source, on_edge));
    PHX_CHECK(!mesh.construct_interior_edge_point(a, b, 0.375, p));
    segment_checked = true;
  }
  PHX_CHECK(segment_checked);
  // (1) A non-segment edge of a constrained face refuses an off-plane carrier.
  std::set<std::uint64_t> segments;
  for (std::size_t s = 0; s + 1 < before.segments.size(); s += 2) {
    segments.insert(phx::mc::topology::undirected_edge(before.segments[s],
                                                        before.segments[s + 1]));
  }
  bool face_checked = false;
  for (std::size_t f = 0; f < before.face_sources.size() && !face_checked; ++f) {
    const int32_t* face = before.faces.data() + 3 * f;
    for (int e = 0; e < 3 && !face_checked; ++e) {
      const int32_t a = face[e], b = face[(e + 1) % 3];
      if (segments.count(phx::mc::topology::undirected_edge(a, b)) != 0) continue;
      int normal = -1;
      for (int k = 0; k < 3; ++k) {
        if (at(face[0])[k] == at(face[1])[k] && at(face[1])[k] == at(face[2])[k]) normal = k;
      }
      if (normal < 0) continue;
      double p[3];
      for (int k = 0; k < 3; ++k) p[k] = 0.5 * (at(a)[k] + at(b)[k]);
      p[normal] += p[normal] < 0.5 ? 0x1p-20 : -0x1p-20;
      int32_t source = -1;
      bool on_edge = true;
      PHX_CHECK(mesh.split_positions_admitted(a, b, p, p, source, on_edge));
      PHX_CHECK(!on_edge && source < 0);
      phx::mc::InsertKind kind = phx::mc::InsertKind::kInterior;
      PHX_CHECK(mesh.prepare_edge_insertion(a, b, p, p, 0.0, 512, kind) ==
                phx::mc::Insertion::kRefused);
      mesh.abandon();
      face_checked = true;
    }
  }
  PHX_CHECK(face_checked);
  const auto unchanged = phx::mc::test::export_mesh(handle);
  PHX_CHECK(unchanged == before);
  // (3) A body edge splits at its off-line rounded carrier without a witness.
  bool body_split = false;
  for (std::size_t t = 0; t < before.regions.size() && !body_split; ++t) {
    const int32_t* cell = before.tets.data() + 4 * t;
    for (int i = 0; i < 4 && !body_split; ++i) {
      for (int j = i + 1; j < 4 && !body_split; ++j) {
        const int32_t a = cell[i], b = cell[j];
        if (before.dimension[a] != 3 || before.dimension[b] != 3) continue;
        for (double fraction : {0.3, 0.37, 0.41, 0.59, 0.63}) {
          double p[3];
          if (!mesh.construct_interior_edge_point(a, b, fraction, p) || collinear(a, b, p)) {
            continue;
          }
          phx::mc::InsertKind kind = phx::mc::InsertKind::kInterior;
          if (mesh.prepare_edge_insertion(a, b, p, p, fraction, 512, kind) !=
              phx::mc::Insertion::kOk) {
            mesh.abandon();
            continue;
          }
          const auto vertex = static_cast<int32_t>(mesh.vertex_count());
          PHX_CHECK(kind == phx::mc::InsertKind::kInterior);
          PHX_CHECK(mesh.commit(kind, 0.0) == phx::mc::Insertion::kOk);
          mesh.abandon();
          PHX_CHECK(mesh.witness(vertex).stratum == phx::mc::SourceStratum::kNone);
          PHX_CHECK(mesh.dimension(vertex) == 3);
          body_split = true;
          break;
        }
      }
    }
  }
  PHX_CHECK(body_split);
  PHX_CHECK(handle_audit(handle));
  PHX_CHECK(phx::mc::test::all_positive(phx::mc::test::export_mesh(handle)));
  int64_t ancestors = -1;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &ancestors, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, nullptr,
                                            nullptr) == PHX_MC_OK);
  PHX_CHECK(ancestors > 0);
  phx_mc_tet_mesh_free(handle);
}

}  // namespace

int main() {
  test_undeclared_curved_folds_are_refused();
  test_fold_between_two_curved_sources_is_refused();
  test_declared_claims_are_complete_finite_and_bounded();
  test_inspected_curved_collapse_commits_only_exact_claims();
  test_retained_collapse_matches_fresh_and_invalidates_every_prepare();
  test_retained_collapse_work_and_scratch_boundaries_are_atomic();
  test_retained_collapse_stales_after_an_accepted_edit();
  test_retained_collapse_refused_readout_and_changed_source_cannot_reuse();
  test_fresh_malformed_claims_refuse_before_staging_work_and_scratch();
  test_successive_same_source_collapses_preserve_raw_and_outward_claim_rows();
  test_boundary_point_lookup_keeps_first_closed_row();
  test_legacy_create_is_unchanged_by_absent_declaration();
  test_rounded_interior_carrier_is_unconstrained_only();
  return phx::mc::test::finish("test_curved_collapse");
}
