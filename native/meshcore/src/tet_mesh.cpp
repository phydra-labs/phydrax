//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI of the constrained tetrahedral mesh: creation from host arrays,
// quality evidence and canonical export (see tet_mesh.hpp).
#include "tet_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "capi_guard.hpp"
#include "counter_records.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"

namespace phx::mc {
namespace {

bool valid_creation_limits(const phx_mc_tet_mesh_create_request& in) {
  return in.point_count >= 4 && in.point_count <= kMaxMeshPoints && in.tet_count >= 1 &&
         in.face_count >= 0 && in.segment_count >= 0 && in.max_vertices >= in.point_count &&
         in.max_vertices <= kMaxMeshPoints && in.max_tetrahedra >= in.tet_count &&
         in.max_tetrahedra <= kMaxTetrahedronSlots / 2 && in.max_scratch_bytes >= 0 &&
         static_cast<std::uint64_t>(in.max_scratch_bytes) <=
             std::numeric_limits<std::size_t>::max();
}

bool valid_creation_buffers(const phx_mc_tet_mesh_create_request& in) {
  return addressable(in.point_count, 3, sizeof(double)) &&
         addressable(in.tet_count, 4, sizeof(int32_t)) &&
         addressable(in.face_count, 3, sizeof(int32_t)) &&
         addressable(in.segment_count, 2, sizeof(int32_t)) &&
         in.points != nullptr && in.tets != nullptr && in.tet_regions != nullptr &&
         (in.face_count == 0 || (in.faces != nullptr && in.face_sources != nullptr)) &&
         (in.segment_count == 0 ||
          (in.segments != nullptr && in.segment_sources != nullptr));
}

bool valid_source_buffers(const SourceComplex& source, int64_t point_count) {
  return source.face_count >= 0 && source.segment_count >= 0 && source.point_count >= 0 &&
         addressable(source.point_count, 3, sizeof(double)) &&
         addressable(source.face_count, 3, sizeof(int32_t)) &&
         addressable(source.segment_count, 2, sizeof(int32_t)) &&
         addressable(point_count, 2, sizeof(double)) &&
         (source.face_count == 0 || (source.faces != nullptr && source.face_sources != nullptr)) &&
         (source.segment_count == 0 ||
          (source.segments != nullptr && source.segment_sources != nullptr)) &&
         (source.witness_strata == nullptr) == (source.witness_entities == nullptr) &&
         (source.witness_strata == nullptr) == (source.witness_parameters == nullptr);
}

int32_t create(const phx_mc_tet_mesh_create_request& in, const SourceComplex* source,
               const CurvedSourceDeclaration* curved) {
  if (in.mesh == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *in.mesh = nullptr;
  if (!valid_creation_limits(in) || !valid_creation_buffers(in) ||
      (source != nullptr && !valid_source_buffers(*source, in.point_count))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  BoundaryPolicy boundary = BoundaryPolicy::kFixed;
  switch (in.boundary_policy) {
    case PHX_MC_TET_MESH_BOUNDARY_FIXED:
      boundary = BoundaryPolicy::kFixed;
      break;
    case PHX_MC_TET_MESH_BOUNDARY_CONFORMING:
      boundary = BoundaryPolicy::kConforming;
      break;
    default:
      return PHX_MC_INVALID_ARGUMENT;
  }
  const MemoryBudgetWindow budget(static_cast<std::size_t>(in.max_scratch_bytes));
  const MemoryScope memory_scope(budget.owner());
  // Coordinate refusals need no retained mesh allocation.
  const int32_t valid = validate_points(in.points, in.point_count, 3, nullptr);
  if (valid != PHX_MC_OK) return valid;
  auto handle = make_native_unique<phx_mc_tet_mesh>();
  handle->mesh =
      make_native_unique<TetMesh>(boundary, in.max_vertices, in.max_tetrahedra, budget.owner());
  const int32_t status = handle->mesh->build(
      in.point_count, in.points, in.tet_count, in.tets, in.tet_regions, in.face_count, in.faces,
      in.face_sources, in.segment_count, in.segments, in.segment_sources, in.protection_radii, source,
      curved);
  if (status != PHX_MC_OK) return status;
  *in.mesh = handle.release();
  return PHX_MC_OK;
}

SourceComplex source_from_request(const phx_mc_tet_mesh_create_request& in) {
  SourceComplex source;
  source.point_count = in.source_point_count;
  source.points = in.source_points;
  source.face_count = in.source_face_count;
  source.faces = in.source_faces;
  source.face_sources = in.source_face_group_rows;
  source.face_tolerances = in.source_face_tolerances;
  source.segment_count = in.source_segment_count;
  source.segments = in.source_segments;
  source.segment_sources = in.source_segment_group_rows;
  source.segment_tolerances = in.source_segment_tolerances;
  source.witness_strata = in.witness_strata;
  source.witness_entities = in.witness_entities;
  source.witness_parameters = in.witness_parameters;
  return source;
}

int32_t check(const phx_mc_tet_mesh* handle) {
  if (handle == nullptr || handle->mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return handle->mesh->broken() ? PHX_MC_INTERNAL_ERROR : PHX_MC_OK;
}

MemoryOwner memory_owner_of(const phx_mc_tet_mesh* handle) {
  return handle != nullptr && handle->mesh ? handle->mesh->memory_owner() : MemoryOwner{};
}

// Shared by the owning quality report and borrowed proposal tables.
double shape(const double* const* p, double& minimum, double& maximum) {
  geometry::dihedral_extremes(p, minimum, maximum);
  double center[3];
  if (!geometry::tetrahedron_circumcenter(p[0], p[1], p[2], p[3], center)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(geometry::squared_distance(center, p[0])) / geometry::shortest_edge(p);
}

struct ProposalWorkLimit {};

struct ProposalWork {
  TetMesh& mesh;
  int64_t remaining;

  bool spend() {
    if (remaining == 0 || !mesh.spend(1)) {
      return false;
    }
    --remaining;
    return true;
  }

  void operator()() {
    if (!spend()) throw ProposalWorkLimit{};
  }
};

int32_t proposal_shape(phx_mc_tet_mesh* handle, int64_t point_count, const double* points,
                       const int64_t* vertex_ids, int64_t tet_count, const int32_t* tets,
                       double minimum_relative_determinant, int64_t work_limit,
                       double* values) {
  const int32_t status = check(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  if (point_count < 0 || point_count > kMaxMeshPoints || tet_count < 0 || work_limit < 0 ||
      !addressable(point_count, 3, sizeof(double)) ||
      !addressable(tet_count, 4, sizeof(int32_t)) || values == nullptr ||
      !(minimum_relative_determinant >= 0.0) || !(minimum_relative_determinant < 1.0) ||
      (point_count > 0 && (points == nullptr || vertex_ids == nullptr)) ||
      (tet_count > 0 && tets == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  TetMesh& mesh = *handle->mesh;
  const MemoryScope memory_scope(mesh.memory_owner());
  ProposalWork work{mesh, work_limit};
  try {
    const int32_t valid = validate_points(points, point_count, 3, nullptr,
                                         [&work] { work(); });
    if (valid != PHX_MC_OK) {
      return valid;
    }
    double worst_ratio = 0.0;
    double minimum = 180.0;
    int64_t below_floor = 0;
    for (int64_t row = 0; row < tet_count; ++row) {
      const double* p[4];
      int64_t identities[4];
      for (int corner = 0; corner < 4; ++corner) {
        if (!work.spend()) {
          return PHX_MC_CAPACITY_EXCEEDED;
        }
        const int32_t vertex = tets[4 * row + corner];
        if (vertex < 0 || vertex >= point_count) {
          return PHX_MC_INVALID_INPUT;
        }
        p[corner] = points + 3 * static_cast<std::size_t>(vertex);
        identities[corner] = vertex_ids[vertex];
        // The chart is decided by authoritative identities, never local rows.
        if (identities[corner] < 0 ||
            std::find(identities, identities + corner, identities[corner]) !=
                identities + corner) {
          return PHX_MC_INVALID_INPUT;
        }
      }
      // One root shape evaluation, charged before the canonical constructions.
      if (!work.spend()) {
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      double low = 0.0;
      double high = 0.0;
      const double ratio = shape(p, low, high);
      // A failed/nonrepresentable construction must never disappear in min/max.
      if (std::isnan(ratio) || !std::isfinite(low)) {
        worst_ratio = std::numeric_limits<double>::infinity();
        minimum = 0.0;
      } else {
        worst_ratio = std::max(worst_ratio, ratio);
        minimum = std::min(minimum, low);
      }
      if (minimum_relative_determinant > 0.0) {
        // The owning floor action on the canonical chart of the cell's
        // authoritative vertex identities, exactly as publication charts it.
        if (!work.spend()) {
          return PHX_MC_CAPACITY_EXCEEDED;
        }
        double score = 0.0;
        below_floor += canonical_relative_floor(
            identities, p, minimum_relative_determinant, &score) <= 0;
      }
    }
    values[0] = worst_ratio;
    values[1] = minimum;
    values[2] = static_cast<double>(below_floor);
    return PHX_MC_OK;
  } catch (const ProposalWorkLimit&) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
}

int32_t quality(const phx_mc_tet_mesh* handle, double sliver_degrees, double* values,
                int64_t* histogram, int64_t* slivers) {
  int32_t status = check(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  if (values == nullptr || histogram == nullptr || slivers == nullptr ||
      !std::isfinite(sliver_degrees)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const TetMesh& mesh = *handle->mesh;
  const MemoryScope memory_scope(mesh.memory_owner());
  double minimum = 180.0;
  double maximum = 0.0;
  double worst_ratio = 0.0;
  double ratio_total = 0.0;
  double smallest_volume = std::numeric_limits<double>::infinity();
  double volume = 0.0;
  double quality_total = 0.0;
  int64_t count = 0;
  *slivers = 0;
  std::fill_n(histogram, PHX_MC_TET_MESH_QUALITY_BINS, 0);
  for (std::size_t s = 0; s < mesh.complex().tets.size(); ++s) {
    const auto t = static_cast<int32_t>(s);
    if (!mesh.complex().live(t) || !mesh.finite(t)) {
      continue;
    }
    const double* p[4];
    mesh.corners(t, p);
    double low = 0.0;
    double high = 0.0;
    const double ratio = shape(p, low, high);
    minimum = std::min(minimum, low);
    maximum = std::max(maximum, high);
    quality_total += low;
    worst_ratio = std::max(worst_ratio, ratio);
    ratio_total += ratio;
    const double measure = geometry::signed_volume(p);
    smallest_volume = std::min(smallest_volume, measure);
    volume += measure;
    const auto bin = static_cast<int64_t>(std::floor(low / 5.0));
    ++histogram[std::clamp<int64_t>(bin, 0, PHX_MC_TET_MESH_QUALITY_BINS - 1)];
    *slivers += low < sliver_degrees ? 1 : 0;
    ++count;
  }
  const double cells = static_cast<double>(std::max<int64_t>(count, 1));
  const double summary[PHX_MC_TET_MESH_QUALITY_VALUES] = {
      minimum, maximum, worst_ratio, ratio_total / cells, smallest_volume, volume,
      quality_total / cells};
  std::copy_n(summary, PHX_MC_TET_MESH_QUALITY_VALUES, values);
  return PHX_MC_OK;
}

int32_t counts(const phx_mc_tet_mesh* handle, int64_t* values) {
  int32_t status = check(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  if (values == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const TetMesh& mesh = *handle->mesh;
  const MemoryScope memory_scope(mesh.memory_owner());
  NativeVector<std::array<int32_t, 4>> faces;
  mesh.collect_faces(faces);
  TetMeshCounts state;
  CounterPublisher publisher{state, values};
  state.vertices = mesh.vertex_count();
  state.tetrahedra = mesh.complex().finite_count;
  state.faces = static_cast<int64_t>(faces.size());
  state.segments = static_cast<int64_t>(mesh.segments().size());
  state.unmet_records = static_cast<int64_t>(mesh.unmet().size());
  state.work_units = mesh.work();
  return PHX_MC_OK;
}

int32_t export_arrays(const phx_mc_tet_mesh* handle, double* points, int32_t* tets,
                      int32_t* regions, int32_t* faces, int32_t* face_sources, int32_t* segments,
                      int32_t* segment_sources, int8_t* dimension, double* sizes) {
  int32_t status = check(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  const TetMesh& mesh = *handle->mesh;
  const MemoryScope memory_scope(mesh.memory_owner());
  NativeVector<std::array<int32_t, 5>> cells;
  NativeVector<std::array<int32_t, 4>> face_rows;
  mesh.collect_cells(cells);
  mesh.collect_faces(face_rows);
  NativeVector<std::array<int32_t, 3>> segment_rows;
  segment_rows.reserve(mesh.segments().size());
  for (const auto& entry : mesh.segments()) {
    segment_rows.push_back({edge_low(entry.first), edge_high(entry.first), entry.second});
  }
  std::sort(segment_rows.begin(), segment_rows.end());
  if (points == nullptr || dimension == nullptr || sizes == nullptr ||
      (!cells.empty() && (tets == nullptr || regions == nullptr)) ||
      (!face_rows.empty() && (faces == nullptr || face_sources == nullptr)) ||
      (!segment_rows.empty() && (segments == nullptr || segment_sources == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t vertices = mesh.vertex_count();
  for (int64_t v = 0; v < vertices; ++v) {
    const auto id = static_cast<int32_t>(v);
    std::copy_n(mesh.point(id), 3, points + 3 * v);
    dimension[v] = mesh.dimension(id);
    sizes[v] = mesh.size(id);
  }
  for (std::size_t i = 0; i < cells.size(); ++i) {
    std::copy_n(cells[i].begin(), 4, tets + 4 * i);
    regions[i] = cells[i][4];
  }
  for (std::size_t i = 0; i < face_rows.size(); ++i) {
    std::copy_n(face_rows[i].begin(), 3, faces + 3 * i);
    face_sources[i] = face_rows[i][3];
  }
  for (std::size_t i = 0; i < segment_rows.size(); ++i) {
    segments[2 * i] = segment_rows[i][0];
    segments[2 * i + 1] = segment_rows[i][1];
    segment_sources[i] = segment_rows[i][2];
  }
  return PHX_MC_OK;
}

int32_t unmet(const phx_mc_tet_mesh* handle, int32_t* tets, int32_t* codes, double* values) {
  int32_t status = check(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  const TetMesh& mesh = *handle->mesh;
  const MemoryScope memory_scope(mesh.memory_owner());
  const std::span<const UnmetRecord> records = mesh.unmet();
  if (!records.empty() && (tets == nullptr || codes == nullptr || values == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (std::size_t i = 0; i < records.size(); ++i) {
    std::copy_n(records[i].v, 4, tets + 4 * i);
    codes[2 * i] = records[i].criterion;
    codes[2 * i + 1] = records[i].reason;
    values[i] = records[i].value;
  }
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_tet_mesh_create(const phx_mc_tet_mesh_create_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const auto& in = *request;
    // Absent source rows use the initial exact faces and segments.
    const bool declared = in.source_points != nullptr;
    if ((declared && in.source_point_count < 1) ||
        (!declared && (in.source_point_count != 0 || in.source_faces != nullptr ||
                       in.source_segments != nullptr || in.source_face_count != 0 ||
                       in.source_segment_count != 0 || in.source_face_tolerances != nullptr ||
                       in.source_segment_tolerances != nullptr || in.witness_strata != nullptr ||
                       in.witness_entities != nullptr || in.witness_parameters != nullptr))) {
      if (in.mesh != nullptr) *in.mesh = nullptr;
      return PHX_MC_INVALID_ARGUMENT;
    }
    // Curved claims bind the own source complex only.
    const bool curved = in.curved_source_count != 0;
    if (in.curved_source_count < 0 ||
        (curved && (declared || in.curved_sources == nullptr ||
                    in.curved_tolerances == nullptr || in.curved_digests == nullptr ||
                    (in.face_count > 0 && in.face_claims == nullptr) ||
                    !phx::mc::addressable(in.curved_source_count, 32, sizeof(uint8_t)))) ||
        (!curved && (in.curved_sources != nullptr || in.curved_tolerances != nullptr ||
                     in.curved_digests != nullptr || in.face_claims != nullptr))) {
      if (in.mesh != nullptr) *in.mesh = nullptr;
      return PHX_MC_INVALID_ARGUMENT;
    }
    const auto source = phx::mc::source_from_request(in);
    phx::mc::CurvedSourceDeclaration declaration;
    declaration.source_count = in.curved_source_count;
    declaration.sources = in.curved_sources;
    declaration.tolerances = in.curved_tolerances;
    declaration.digests = in.curved_digests;
    declaration.face_claims = in.face_claims;
    return phx::mc::create(in, declared ? &source : nullptr, curved ? &declaration : nullptr);
  });
}

int32_t phx_mc_tet_mesh_accepted_generation(const phx_mc_tet_mesh* handle,
                                            int64_t* generation) {
  return phx::mc::guarded([&]() -> int32_t {
    if (generation == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *generation = -1;
    const int32_t status = phx::mc::check(handle);
    if (status != PHX_MC_OK) return status;
    *generation = handle->mesh->accepted_generation();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_curved_claims(const phx_mc_tet_mesh* handle, int64_t capacity,
                                      int64_t* count, int32_t* rows, int32_t* sources,
                                      double* coordinates, double* deviations,
                                      int64_t* generations) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = phx::mc::check(handle);
    if (status != PHX_MC_OK || count == nullptr || capacity < 0) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::TetMesh& mesh = *handle->mesh;
    const auto claims = mesh.curved_claims();
    *count = static_cast<int64_t>(claims.size());
    if (rows == nullptr && sources == nullptr && coordinates == nullptr &&
        deviations == nullptr && generations == nullptr) {
      return PHX_MC_OK;
    }
    if (rows == nullptr || sources == nullptr || coordinates == nullptr ||
        deviations == nullptr || generations == nullptr || capacity < *count) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const auto faces = mesh.original_faces();
    for (std::size_t i = 0; i < claims.size(); ++i) {
      const auto& face = faces[static_cast<std::size_t>(claims[i].row)];
      rows[i] = claims[i].row;
      sources[i] = claims[i].source;
      for (int k = 0; k < 3; ++k) {
        std::copy_n(mesh.original_point(face[static_cast<std::size_t>(k)]), 3,
                    coordinates + 9 * i + 3 * k);
      }
      deviations[i] = claims[i].deviation;
      generations[i] = claims[i].generation;
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_quality(const phx_mc_tet_mesh* mesh, double sliver_degrees,
                                double* values, int64_t* histogram, int64_t* slivers) {
  return phx::mc::guarded([&] {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(mesh));
    return phx::mc::quality(mesh, sliver_degrees, values, histogram, slivers);
  });
}

int32_t phx_mc_tet_mesh_proposal_shape(phx_mc_tet_mesh* mesh, int64_t point_count,
                                      const double* points, const int64_t* vertex_ids,
                                      int64_t tet_count, const int32_t* tets,
                                      double minimum_relative_determinant,
                                      int64_t work_limit, double* values) {
  return phx::mc::guarded([&] {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(mesh));
    return phx::mc::proposal_shape(mesh, point_count, points, vertex_ids, tet_count, tets,
                                   minimum_relative_determinant, work_limit, values);
  });
}

int32_t phx_mc_tet_mesh_counts(const phx_mc_tet_mesh* mesh, int64_t* counts) {
  return phx::mc::guarded([&] {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(mesh));
    return phx::mc::counts(mesh, counts);
  });
}

int32_t phx_mc_tet_mesh_work_units(const phx_mc_tet_mesh* handle, int64_t* work) {
  return phx::mc::guarded([&]() -> int32_t {
    if (work == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *work = 0;
    const int32_t status = phx::mc::check(handle);
    if (status != PHX_MC_OK) return status;
    *work = handle->mesh->work();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_set_work_limit(phx_mc_tet_mesh* mesh, int64_t remaining_work_units) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(mesh));
    const int32_t status = phx::mc::check(mesh);
    if (status != PHX_MC_OK) {
      return status;
    }
    if (remaining_work_units < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    mesh->mesh->set_work_limit(remaining_work_units);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_export(const phx_mc_tet_mesh* mesh, double* points, int32_t* tets,
                               int32_t* tet_regions, int32_t* faces, int32_t* face_sources,
                               int32_t* segments, int32_t* segment_sources,
                               int8_t* vertex_dimension, double* vertex_sizes) {
  return phx::mc::guarded([&] {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(mesh));
    return phx::mc::export_arrays(mesh, points, tets, tet_regions, faces, face_sources, segments,
                                  segment_sources, vertex_dimension, vertex_sizes);
  });
}

int32_t phx_mc_tet_mesh_unmet(const phx_mc_tet_mesh* mesh, int32_t* tets, int32_t* codes,
                              double* values) {
  return phx::mc::guarded([&] {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(mesh));
    return phx::mc::unmet(mesh, tets, codes, values);
  });
}

int32_t phx_mc_tet_mesh_source_evidence(const phx_mc_tet_mesh* handle, int64_t capacity,
                                        int64_t* count, double* values,
                                        int64_t* face_ancestor_offsets,
                                        int32_t* face_ancestor_rows, int32_t* segment_ancestors,
                                        int8_t* witness_strata, int32_t* witness_entities,
                                        double* witness_parameters,
                                        double* witness_deviations) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(handle));
    const int32_t status = phx::mc::check(handle);
    if (status != PHX_MC_OK || capacity < 0 || count == nullptr) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::TetMesh& mesh = *handle->mesh;
    phx::mc::NativeVector<std::array<int32_t, 4>> faces;
    mesh.collect_faces(faces);
    phx::mc::NativeVector<int64_t> offsets(1, 0);
    phx::mc::NativeVector<int32_t> rows, contributors;
    // The count and fill calls, and repeated evidence of an unchanged mesh,
    // reuse one exact ancestry CSR; any differing face input recomputes it.
    if (!mesh.cached_face_ancestry(faces, offsets, rows)) {
      offsets.assign(1, 0);
      rows.clear();
      for (const auto& face : faces) {
        if (!mesh.face_ancestry(face.data(), face[3], contributors)) {
          return PHX_MC_INVALID_INPUT;
        }
        rows.insert(rows.end(), contributors.begin(), contributors.end());
        offsets.push_back(static_cast<int64_t>(rows.size()));
      }
      mesh.store_face_ancestry(faces, offsets, rows);
    }
    *count = static_cast<int64_t>(rows.size());
    if (values == nullptr && face_ancestor_offsets == nullptr && face_ancestor_rows == nullptr &&
        segment_ancestors == nullptr && witness_strata == nullptr && witness_entities == nullptr &&
        witness_parameters == nullptr && witness_deviations == nullptr) return PHX_MC_OK;
    phx::mc::NativeVector<std::array<int32_t, 3>> segments;
    segments.reserve(mesh.segments().size());
    for (const auto& [edge, source] : mesh.segments()) {
      segments.push_back({phx::mc::edge_low(edge), phx::mc::edge_high(edge), source});
    }
    std::sort(segments.begin(), segments.end());
    if (capacity < *count || values == nullptr || face_ancestor_offsets == nullptr ||
        (!rows.empty() && face_ancestor_rows == nullptr) ||
        (!segments.empty() && segment_ancestors == nullptr) || witness_strata == nullptr ||
        witness_entities == nullptr || witness_parameters == nullptr ||
        witness_deviations == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    values[0] = mesh.requested_source_tolerance();
    values[1] = mesh.achieved_source_error_bound();
    std::copy(offsets.begin(), offsets.end(), face_ancestor_offsets);
    std::copy(rows.begin(), rows.end(), face_ancestor_rows);
    for (std::size_t i = 0; i < segments.size(); ++i) {
      segment_ancestors[i] =
          mesh.original_segment_ancestor(segments[i][0], segments[i][1], segments[i][2]);
    }
    for (int32_t v = 0; v < mesh.vertex_count(); ++v) {
      const phx::mc::SourceWitness& witness = mesh.witness(v);
      const auto row = static_cast<std::size_t>(v);
      witness_strata[row] = static_cast<int8_t>(witness.stratum);
      witness_entities[row] = witness.entity;
      witness_parameters[2 * row] = witness.parameters[0];
      witness_parameters[2 * row + 1] = witness.parameters[1];
      witness_deviations[row] = witness.deviation;
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_source_refusals(const phx_mc_tet_mesh* handle, int64_t capacity,
                                        int64_t* count, int32_t* edges, int8_t* strata,
                                        int32_t* entities, double* values) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = phx::mc::check(handle);
    if (status != PHX_MC_OK || count == nullptr || capacity < 0) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    const auto refusals = handle->mesh->source_refusals();
    *count = static_cast<int64_t>(refusals.size());
    if (edges == nullptr && strata == nullptr && entities == nullptr && values == nullptr) {
      return PHX_MC_OK;
    }
    if (edges == nullptr || strata == nullptr || entities == nullptr || values == nullptr ||
        capacity < *count) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < refusals.size(); ++i) {
      edges[2 * i] = refusals[i].v[0];
      edges[2 * i + 1] = refusals[i].v[1];
      strata[i] = static_cast<int8_t>(refusals[i].stratum);
      entities[i] = refusals[i].entity;
      values[2 * i] = refusals[i].deviation;
      values[2 * i + 1] = refusals[i].tolerance;
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_protect_vertices(phx_mc_tet_mesh* handle, int64_t count,
                                        const int32_t* vertices) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(handle));
    const int32_t status = phx::mc::check(handle);
    if (status != PHX_MC_OK || count < 0 || (count > 0 && vertices == nullptr)) {
      return status != PHX_MC_OK ? status : PHX_MC_INVALID_ARGUMENT;
    }
    const bool accepted = handle->mesh->protect_vertices(count, vertices);
    return accepted ? PHX_MC_OK
                    : handle->mesh->work_exhausted() ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_INVALID_INPUT;
  });
}

int32_t phx_mc_tet_mesh_measure_execution(phx_mc_tet_mesh* handle, int32_t enabled) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(handle));
    if (handle == nullptr || handle->mesh == nullptr || (enabled != 0 && enabled != 1)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    handle->mesh->measure_execution(enabled != 0);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_execution_times(const phx_mc_tet_mesh* handle, double* seconds,
                                       int32_t* measured) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::MemoryScope memory_scope(
        phx::mc::memory_owner_of(handle));
    // Failure timing remains readable even when scientific mesh operations are
    // refused because the handle is broken. This does not clear its failure.
    if (handle == nullptr || handle->mesh == nullptr || seconds == nullptr || measured == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const auto& values = handle->mesh->execution_seconds();
    const auto& flags = handle->mesh->execution_measured();
    std::copy(values.begin(), values.end(), seconds);
    std::copy(flags.begin(), flags.end(), measured);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_set_memory_limit(phx_mc_tet_mesh* handle, int64_t max_scratch_bytes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr || max_scratch_bytes < 0 ||
        static_cast<std::uint64_t>(max_scratch_bytes) >
            std::numeric_limits<std::size_t>::max()) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::MemoryScope memory_scope(handle->mesh->memory_owner());
    return handle->mesh->set_memory_limit(static_cast<std::size_t>(max_scratch_bytes))
               ? PHX_MC_OK : PHX_MC_CAPACITY_EXCEEDED;
  });
}

int32_t phx_mc_tet_mesh_memory_evidence(const phx_mc_tet_mesh* handle, std::uint64_t* values) {
  return phx::mc::guarded([&]() -> int32_t {
    // Allocation failures and broken-premise outcomes remain diagnosable.
    if (handle == nullptr || handle->mesh == nullptr || values == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::MemoryScope memory_scope(handle->mesh->memory_owner());
    const auto evidence = handle->mesh->memory_evidence();
    const auto state = phx::mc::ManagedMemory::from_slots(evidence.data());
    phx::mc::CounterPublisher publisher{state, values};
    return PHX_MC_OK;
  });
}

void phx_mc_tet_mesh_free(phx_mc_tet_mesh* mesh) {
  const phx::mc::MemoryScope memory_scope(
      phx::mc::memory_owner_of(mesh));
  phx::mc::destroy_native_object(mesh);
}

}  // extern "C"
