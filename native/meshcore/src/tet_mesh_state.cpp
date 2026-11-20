//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Atomic construction of persistent tetrahedral state, including its closed
// ghost hull, source banks, labels, and stable logical identities.
#include "tet_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <vector>

#include "facet_coverage.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc {
namespace {

// A cell face keyed by its two largest vertices, in the bucket of its
// smallest; `parity` is that of the permutation taking the face as oriented
// in its cell (oriented_facet) to increasing order. A face shared by two
// cells on opposite sides appears with opposite parities.
struct FaceRecord {
  std::int32_t b;
  std::int32_t c;
  std::int32_t cell;
  std::int16_t slot;
  std::int16_t parity;
};

int sorted_face_of(const Tetrahedron& tet, int slot, std::int32_t* key) {
  oriented_facet(tet.v, slot, key);
  const int parity = ((key[0] > key[1]) + (key[0] > key[2]) + (key[1] > key[2])) & 1;
  if (key[0] > key[1]) std::swap(key[0], key[1]);
  if (key[1] > key[2]) std::swap(key[1], key[2]);
  if (key[0] > key[1]) std::swap(key[0], key[1]);
  return parity;
}

// The faces of cells [0, cell_count) in buckets by smallest vertex
// (`bucket[v]` .. `bucket[v + 1]`), each sorted by (b, c, cell, slot).
void bucket_faces(const std::vector<Tetrahedron>& tets, std::size_t cell_count,
                  std::size_t vertex_count, std::vector<FaceRecord>& records,
                  std::vector<std::uint32_t>& bucket) {
  bucket.assign(vertex_count + 1, 0);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    const Tetrahedron& tet = tets[cell];
    // The smallest vertex of a face is the smallest of the other three corners.
    for (int slot = 0; slot < 4; ++slot) {
      std::int32_t smallest = std::numeric_limits<std::int32_t>::max();
      for (int corner = 0; corner < 4; ++corner) {
        if (corner != slot) smallest = std::min(smallest, tet.v[corner]);
      }
      ++bucket[static_cast<std::size_t>(smallest) + 1];
    }
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) bucket[vertex + 1] += bucket[vertex];
  std::vector<std::uint32_t> cursor(bucket.begin(), bucket.end() - 1);
  records.resize(4 * cell_count);
  std::int32_t key[3];
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for (int slot = 0; slot < 4; ++slot) {
      const int parity = sorted_face_of(tets[cell], slot, key);
      records[cursor[static_cast<std::size_t>(key[0])]++] =
          FaceRecord{key[1], key[2], static_cast<std::int32_t>(cell),
                     static_cast<std::int16_t>(slot), static_cast<std::int16_t>(parity)};
    }
  }
  // Records arrive in (cell, slot) order, so a stable sort by (b, c) leaves
  // each run in occurrence order. Buckets are small (vertex degree).
  const auto before = [](const FaceRecord& x, const FaceRecord& y) {
    return x.b < y.b || (x.b == y.b && x.c < y.c);
  };
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    FaceRecord* first = records.data() + bucket[vertex];
    FaceRecord* last = records.data() + bucket[vertex + 1];
    if (last - first > 64) {
      // Larger buckets order by the full key; (cell, slot) is the occurrence order.
      std::sort(first, last, [](const FaceRecord& x, const FaceRecord& y) {
        if (x.b != y.b) return x.b < y.b;
        if (x.c != y.c) return x.c < y.c;
        return x.cell != y.cell ? x.cell < y.cell : x.slot < y.slot;
      });
      continue;
    }
    for (FaceRecord* current = first + 1; current < last; ++current) {
      const FaceRecord value = *current;
      FaceRecord* hole = current;
      while (hole > first && before(value, hole[-1])) {
        *hole = hole[-1];
        --hole;
      }
      *hole = value;
    }
  }
}

// Links the ghost cells [cell_count, tets.size()) across their faces through
// the boundary edges: a ghost face over the directed edge (x, y) meets the
// pending ghost face over (y, x), in ghost order. False when a directed edge
// repeats while pending or a ghost face stays unmatched.
bool link_ghosts(std::vector<Tetrahedron>& tets, std::size_t cell_count,
                 std::size_t vertex_count) {
  struct EdgeRecord {
    std::int32_t high;
    std::int32_t ghost;
    std::int32_t slot;
    std::int32_t forward;
  };
  std::vector<std::uint32_t> bucket(vertex_count + 1, 0);
  const auto edge_of = [&](std::size_t ghost, int slot, std::int32_t* edge) {
    std::int32_t face[3];
    oriented_facet(tets[ghost].v, slot, face);
    edge[0] = face[1];
    edge[1] = face[2];
  };
  std::int32_t edge[2];
  for (std::size_t ghost = cell_count; ghost < tets.size(); ++ghost) {
    for (int slot = 1; slot < 4; ++slot) {
      edge_of(ghost, slot, edge);
      ++bucket[static_cast<std::size_t>(std::min(edge[0], edge[1])) + 1];
    }
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) bucket[vertex + 1] += bucket[vertex];
  std::vector<std::uint32_t> cursor(bucket.begin(), bucket.end() - 1);
  std::vector<EdgeRecord> records(3 * (tets.size() - cell_count));
  for (std::size_t ghost = cell_count; ghost < tets.size(); ++ghost) {
    for (int slot = 1; slot < 4; ++slot) {
      edge_of(ghost, slot, edge);
      const bool forward = edge[0] < edge[1];
      records[cursor[static_cast<std::size_t>(forward ? edge[0] : edge[1])]++] =
          EdgeRecord{forward ? edge[1] : edge[0], static_cast<std::int32_t>(ghost), slot,
                     forward ? 1 : 0};
    }
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    // Fill order is ghost order; a stable sort by the far end keeps it.
    std::stable_sort(records.begin() + bucket[vertex], records.begin() + bucket[vertex + 1],
                     [](const EdgeRecord& x, const EdgeRecord& y) { return x.high < y.high; });
    std::uint32_t row = bucket[vertex];
    while (row < bucket[vertex + 1]) {
      std::int64_t pending[2]{-1, -1};
      std::uint32_t last = row;
      for (; last < bucket[vertex + 1] && records[last].high == records[row].high; ++last) {
        const EdgeRecord& current = records[last];
        const int direction = current.forward;
        if (pending[1 - direction] >= 0) {
          const EdgeRecord& other = records[static_cast<std::size_t>(pending[1 - direction])];
          tets[static_cast<std::size_t>(current.ghost)].n[current.slot] = other.ghost;
          tets[static_cast<std::size_t>(other.ghost)].n[other.slot] = current.ghost;
          pending[1 - direction] = -1;
        } else if (pending[direction] >= 0) {
          return false;
        } else {
          pending[direction] = last;
        }
      }
      if (pending[0] >= 0 || pending[1] >= 0) return false;
      row = last;
    }
  }
  return true;
}

// A cell face: (cell, slot).
using FaceSlot = std::pair<std::int32_t, int>;

// Pairs the faces of the cells [0, cell_count) by their vertices (each face
// of at most two cells, on opposite sides). `boundary` receives the boundary
// faces in face key order (the ghost order), `face_cells` each constrained
// face row's first occurrence in (cell, slot) order; every boundary face is
// a constrained one and none is constrained twice.
int32_t pair_faces(std::vector<Tetrahedron>& tets, std::size_t cell_count,
                   std::size_t vertex_count, std::span<const std::int32_t> faces,
                   std::vector<FaceSlot>& boundary, std::vector<FaceSlot>& face_cells) {
  // Cell faces bucketed by their smallest vertex, each bucket sorted by the
  // other two vertices, then by occurrence (cell row, face slot): a face is a
  // run of equal keys, its first record its first occurrence.
  std::vector<FaceRecord> records;
  std::vector<std::uint32_t> bucket;
  bucket_faces(tets, cell_count, vertex_count, records, bucket);
  std::vector<std::uint32_t> boundary_records;
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    std::uint32_t row = bucket[vertex];
    const std::uint32_t end = bucket[vertex + 1];
    while (row < end) {
      std::uint32_t last = row + 1;
      while (last < end && records[last].b == records[row].b &&
             records[last].c == records[row].c) {
        ++last;
      }
      if (last - row > 2) return PHX_MC_INVALID_INPUT;
      if (last - row == 1) {
        boundary_records.push_back(row);
      } else {
        const FaceRecord& first = records[row];
        const FaceRecord& second = records[row + 1];
        if (first.parity == second.parity) return PHX_MC_INVALID_INPUT;
        tets[static_cast<std::size_t>(first.cell)].n[first.slot] = second.cell;
        tets[static_cast<std::size_t>(second.cell)].n[second.slot] = first.cell;
      }
      row = last;
    }
  }
  if (boundary_records.empty()) return PHX_MC_INVALID_INPUT;
  // The first face record of each constrained face row; a face is claimed once.
  std::vector<std::uint8_t> claimed(records.size(), 0);
  for (std::size_t row = 0; row < face_cells.size(); ++row) {
    std::int32_t key[3];
    std::copy_n(faces.data() + 3 * row, 3, key);
    std::sort(key, key + 3);
    if (key[0] < 0 || static_cast<std::size_t>(key[2]) >= vertex_count) return PHX_MC_INVALID_INPUT;
    const auto begin = records.begin() + bucket[static_cast<std::size_t>(key[0])];
    const auto end = records.begin() + bucket[static_cast<std::size_t>(key[0]) + 1];
    const auto found = std::lower_bound(begin, end, key, [](const FaceRecord& record,
                                                            const std::int32_t* wanted) {
      return record.b < wanted[1] || (record.b == wanted[1] && record.c < wanted[2]);
    });
    if (found == end || found->b != key[1] || found->c != key[2]) return PHX_MC_INVALID_INPUT;
    const std::size_t record = static_cast<std::size_t>(found - records.begin());
    if (claimed[record] != 0) return PHX_MC_INVALID_INPUT;
    claimed[record] = 1;
    face_cells[row] = {found->cell, found->slot};
  }
  boundary.reserve(boundary_records.size());
  for (std::uint32_t record : boundary_records) {
    if (claimed[record] == 0) return PHX_MC_INVALID_INPUT;
    boundary.emplace_back(records[record].cell, records[record].slot);
  }
  return PHX_MC_OK;
}

// pair_faces for cells whose adjacency `neighbors` (per cell and slot, -1 on
// the boundary) a PLC recovery published: the same boundary order and first
// occurrences, without pairing every face. A constrained face on the
// boundary is found among the boundary faces, an interior one among the
// cells around its smallest vertex.
int32_t adopt_adjacency(std::vector<Tetrahedron>& tets, std::size_t cell_count,
                        std::size_t vertex_count, std::span<const std::int32_t> faces,
                        std::span<const std::int32_t> neighbors,
                        std::vector<FaceSlot>& boundary, std::vector<FaceSlot>& face_cells) {
  if (neighbors.size() != 4 * cell_count) return PHX_MC_INVALID_ARGUMENT;
  struct BoundaryFace {
    std::array<std::int32_t, 3> key;
    FaceSlot face;
  };
  std::vector<BoundaryFace> keyed;
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for (int slot = 0; slot < 4; ++slot) {
      const std::int32_t across = neighbors[4 * cell + static_cast<std::size_t>(slot)];
      if (across < -1 || across >= static_cast<std::int32_t>(cell_count)) {
        return PHX_MC_INVALID_INPUT;
      }
      tets[cell].n[slot] = across;
      if (across >= 0) continue;
      BoundaryFace entry{{}, {static_cast<std::int32_t>(cell), slot}};
      sorted_face_of(tets[cell], slot, entry.key.data());
      keyed.push_back(entry);
    }
  }
  if (keyed.empty()) return PHX_MC_INVALID_INPUT;
  std::sort(keyed.begin(), keyed.end(),
            [](const BoundaryFace& x, const BoundaryFace& y) { return x.key < y.key; });
  std::vector<std::uint8_t> claimed(keyed.size(), 0);
  std::vector<std::size_t> interior;
  for (std::size_t row = 0; row < face_cells.size(); ++row) {
    std::array<std::int32_t, 3> key;
    std::copy_n(faces.data() + 3 * row, 3, key.begin());
    std::sort(key.begin(), key.end());
    if (key[0] < 0 || static_cast<std::size_t>(key[2]) >= vertex_count) return PHX_MC_INVALID_INPUT;
    const auto found = std::lower_bound(
        keyed.begin(), keyed.end(), key,
        [](const BoundaryFace& entry, const std::array<std::int32_t, 3>& wanted) {
          return entry.key < wanted;
        });
    if (found == keyed.end() || found->key != key) {
      interior.push_back(row);
      continue;
    }
    const std::size_t at = static_cast<std::size_t>(found - keyed.begin());
    if (claimed[at] != 0) return PHX_MC_INVALID_INPUT;
    claimed[at] = 1;
    face_cells[row] = found->face;
  }
  boundary.reserve(keyed.size());
  for (std::size_t at = 0; at < keyed.size(); ++at) {
    if (claimed[at] == 0) return PHX_MC_INVALID_INPUT;
    boundary.push_back(keyed[at].face);
  }
  if (interior.empty()) return PHX_MC_OK;
  // The cells around each interior face's smallest vertex, in cell order
  // (a counting sort), so the first holding the face is its first occurrence.
  std::vector<std::uint32_t> starts(vertex_count + 1, 0U);
  std::vector<std::uint8_t> wanted(vertex_count, 0);
  for (std::size_t row : interior) {
    const std::int32_t* face = faces.data() + 3 * row;
    wanted[static_cast<std::size_t>(std::min({face[0], face[1], face[2]}))] = 1;
  }
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for (std::int32_t vertex : tets[cell].v) {
      if (wanted[static_cast<std::size_t>(vertex)] != 0) ++starts[static_cast<std::size_t>(vertex) + 1];
    }
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) starts[vertex + 1] += starts[vertex];
  std::vector<std::uint32_t> around(starts.back());
  {
    std::vector<std::uint32_t> cursor(starts.begin(), starts.end() - 1);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
      for (std::int32_t vertex : tets[cell].v) {
        if (wanted[static_cast<std::size_t>(vertex)] != 0) {
          around[cursor[static_cast<std::size_t>(vertex)]++] = static_cast<std::uint32_t>(cell);
        }
      }
    }
  }
  std::vector<FaceSlot> interior_faces;
  interior_faces.reserve(interior.size());
  for (std::size_t row : interior) {
    const std::int32_t* face = faces.data() + 3 * row;
    const std::int32_t smallest = std::min({face[0], face[1], face[2]});
    face_cells[row] = {-1, -1};
    for (std::uint32_t at = starts[static_cast<std::size_t>(smallest)];
         at < starts[static_cast<std::size_t>(smallest) + 1]; ++at) {
      const Tetrahedron& tet = tets[around[at]];
      int outside = -1;
      int held = 0;
      for (int corner = 0; corner < 4; ++corner) {
        if (std::find(face, face + 3, tet.v[corner]) != face + 3) {
          ++held;
        } else {
          outside = corner;
        }
      }
      if (held == 3) {
        face_cells[row] = {static_cast<std::int32_t>(around[at]), outside};
        break;
      }
    }
    if (face_cells[row].first < 0) return PHX_MC_INVALID_INPUT;
    interior_faces.push_back(face_cells[row]);
  }
  std::sort(interior_faces.begin(), interior_faces.end());
  if (std::adjacent_find(interior_faces.begin(), interior_faces.end()) != interior_faces.end()) {
    return PHX_MC_INVALID_INPUT;
  }
  return PHX_MC_OK;
}

FacetCoverageBounds facet_bounds(const double* a, const double* b, const double* c) {
  FacetCoverageBounds result{};
  for (int axis = 0; axis < 3; ++axis) {
    result.lower[axis] = std::min({a[axis], b[axis], c[axis]});
    result.upper[axis] = std::max({a[axis], b[axis], c[axis]});
  }
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3;
    const int v = (axis + 2) % 3;
    const double pa[2]{a[u], a[v]};
    const double pb[2]{b[u], b[v]};
    const double pc[2]{c[u], c[v]};
    if (orient2d(pa, pb, pc) != 0) {
      result.nondegenerate = true;
      break;
    }
  }
  return result;
}

bool addressable_count(std::size_t count, std::size_t width) {
  return width == 0 || count <= std::numeric_limits<std::size_t>::max() / width;
}

std::uint64_t edge_key(std::int32_t a, std::int32_t b) {
  const std::uint32_t low = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t high = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(low) << 32) | high;
}


}  // namespace

int32_t TetMesh::set_affine_metric(std::span<const double> linear) {
  clear_implicit_facet_refusal();
  if (linear.size() != 9 ||
      !std::all_of(linear.begin(), linear.end(), [](double x) { return std::isfinite(x); })) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::array<double, 9> next;
  std::copy(linear.begin(), linear.end(), next.begin());
  if (next == metric_) return PHX_MC_OK;
  if (!points_.empty()) return PHX_MC_INVALID_ARGUMENT;
  const double a = next[0], b = next[1], c = next[2];
  const double d = next[3], e = next[4], f = next[5];
  const double g = next[6], h = next[7], i = next[8];
  const double det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
  if (det == 0 || !std::isfinite(det)) return PHX_MC_INVALID_ARGUMENT;
  const std::array<double, 9> adj{
      e*i-f*h, c*h-b*i, b*f-c*e, f*g-d*i, a*i-c*g,
      c*d-a*f, d*h-e*g, b*g-a*h, a*e-b*d};
  std::array<double, 9> inverse;
  for (std::size_t k = 0; k < 9; ++k) {
    inverse[k] = adj[k] / det;
    if (!std::isfinite(inverse[k])) return PHX_MC_INVALID_ARGUMENT;
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) return PHX_MC_INVALID_ARGUMENT;
  metric_ = next;
  metric_inverse_ = inverse;
  metric_active_ = next != std::array<double, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  metric_orientation_ = det < 0 ? -1 : 1;
  unmet_records_.clear();
  ++generation_;
  return PHX_MC_OK;
}

int32_t TetMesh::build(std::span<const double> points,
                       std::span<const std::int32_t> tetrahedra,
                       std::span<const std::int32_t> regions,
                       std::span<const std::int32_t> faces,
                       std::span<const std::int32_t> face_sources,
                       std::span<const std::int8_t> face_orientations,
                       std::span<const std::int32_t> segments,
                       std::span<const std::int32_t> segment_sources,
                       std::span<const double> protection_radii,
                       const SourceComplexView* source,
                       std::span<const std::int8_t> protected_vertices,
                       const RecoveredTopology* recovered) {
  clear_implicit_facet_refusal();
  if (implicit_source_) return PHX_MC_INVALID_ARGUMENT;
  try {
    plane_refusal_.reset();
    TetMesh staged(policy_);
    staged.metric_ = metric_;
    staged.metric_inverse_ = metric_inverse_;
    staged.metric_active_ = metric_active_;
    staged.metric_orientation_ = metric_orientation_;
    staged.identity_bound_ = identity_bound_;
    staged.source_id_ = source_id_;
    staged.source_revision_ = source_revision_;
    staged.source_map_id_ = source_map_id_;
    staged.frame_id_ = frame_id_;
    staged.pending_implicit_source_ = pending_implicit_source_;
    const std::int32_t status = staged.build_state(
        points, tetrahedra, regions, faces, face_sources, face_orientations,
        segments, segment_sources, protection_radii, source, protected_vertices, recovered);
    if (status != PHX_MC_OK) {
      plane_refusal_ = staged.plane_refusal_;
      return status;
    }
    staged.generation_ = 0;
    // The flip test reads the mesh metric; the recovery's proof holds under
    // the identity metric only.
    staged.delaunay_generation_ = recovered != nullptr && recovered->delaunay &&
                                          !staged.metric_active_
                                      ? 0
                                      : std::numeric_limits<std::uint64_t>::max();
    staged.swap_state(*this);
    unmet_records_.clear();
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_ALLOCATION_FAILURE;
  } catch (const std::length_error&) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
}

int32_t TetMesh::build_state(std::span<const double> points,
                             std::span<const std::int32_t> tetrahedra,
                             std::span<const std::int32_t> regions,
                             std::span<const std::int32_t> faces,
                             std::span<const std::int32_t> face_sources,
                             std::span<const std::int8_t> face_orientations,
                             std::span<const std::int32_t> segments,
                             std::span<const std::int32_t> segment_sources,
                             std::span<const double> protection_radii,
                             const SourceComplexView* source,
                             std::span<const std::int8_t> protected_vertices,
                             const RecoveredTopology* recovered) {
  if (points.empty() || points.size() % 3 != 0 || tetrahedra.empty() ||
      tetrahedra.size() % 4 != 0 || faces.size() % 3 != 0 ||
      segments.size() % 2 != 0 ||
      (recovered != nullptr && recovered->self_covered && source != nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const std::size_t vertex_count = points.size() / 3;
  const std::size_t cell_count = tetrahedra.size() / 4;
  const std::size_t face_count = faces.size() / 3;
  const std::size_t segment_count = segments.size() / 2;
  if (vertex_count < 4 || vertex_count >= static_cast<std::size_t>(kMaxTetrahedronSlots) ||
      cell_count >= static_cast<std::size_t>(kMaxTetrahedronSlots) ||
      face_sources.size() != face_count || segment_sources.size() != segment_count ||
      (!face_orientations.empty() && face_orientations.size() != face_count) ||
      (!regions.empty() && regions.size() != cell_count) ||
      (!protection_radii.empty() && protection_radii.size() != vertex_count) ||
      (!protected_vertices.empty() && protected_vertices.size() != vertex_count)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!addressable_count(vertex_count, 3) || !addressable_count(cell_count, 4) ||
      !addressable_count(face_count, 3) || !addressable_count(segment_count, 2) ||
      cell_count > std::numeric_limits<std::uint32_t>::max() / 4) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const std::int32_t coordinate_status = validate_points(
      points.data(), static_cast<std::int64_t>(vertex_count), 3, nullptr);
  if (coordinate_status != PHX_MC_OK) return coordinate_status;

  points_.assign(points.begin(), points.end());
  vertex_index_.assign_identity(vertex_count);
  vertex_ids_.resize(vertex_count);
  vertex_live_.assign(vertex_count, 1);
  vertex_dimension_.assign(vertex_count, -1);
  protection_radii_.assign(vertex_count, 0.0);
  if (!protection_radii.empty()) {
    std::copy(protection_radii.begin(), protection_radii.end(), protection_radii_.begin());
  }
  protected_vertices_.assign(vertex_count, 0);
  if (!protected_vertices.empty()) {
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      if (protected_vertices[vertex] != 0 && protected_vertices[vertex] != 1) {
        return PHX_MC_INVALID_INPUT;
      }
      protected_vertices_[vertex] = static_cast<std::uint8_t>(protected_vertices[vertex]);
    }
  }
  witnesses_.assign(vertex_count, SourceWitness{});
  vertex_has_edge_parameter_.assign(vertex_count, 0);
  vertex_edge_endpoints_.assign(vertex_count, {});
  vertex_edge_parameters_.assign(vertex_count, 0.0);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    vertex_ids_[vertex] = static_cast<std::uint64_t>(vertex);
    if (!std::isfinite(protection_radii_[vertex]) || protection_radii_[vertex] < 0.0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  next_vertex_id_ = static_cast<std::uint64_t>(vertex_count);

  if (source == nullptr) {
    source_points_.assign(points.begin(), points.end());
    source_faces_.assign(faces.begin(), faces.end());
    source_face_groups_.assign(face_sources.begin(), face_sources.end());
    source_face_ids_.assign(face_sources.begin(), face_sources.end());
    source_face_tolerances_.assign(face_count, 0.0);
    source_segments_.assign(segments.begin(), segments.end());
    source_segment_groups_.assign(segment_sources.begin(), segment_sources.end());
    source_segment_ids_.assign(segment_sources.begin(), segment_sources.end());
    source_segment_tolerances_.assign(segment_count, 0.0);
  } else {
    if (source->points.size() % 3 != 0 || source->faces.size() % 3 != 0 ||
        source->segments.size() % 2 != 0 ||
        source->face_group_rows.size() != source->faces.size() / 3 ||
        source->segment_group_rows.size() != source->segments.size() / 2 ||
        (!source->face_ids.empty() && source->face_ids.size() != source->faces.size() / 3) ||
        (!source->segment_ids.empty() && source->segment_ids.size() != source->segments.size() / 2) ||
        (!source->face_tolerances.empty() &&
         source->face_tolerances.size() != source->faces.size() / 3) ||
        (!source->segment_tolerances.empty() &&
         source->segment_tolerances.size() != source->segments.size() / 2) ||
        (!source->witnesses.empty() && source->witnesses.size() != vertex_count)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const std::int32_t source_status = validate_points(
        source->points.data(), static_cast<std::int64_t>(source->points.size() / 3), 3, nullptr);
    if (source_status != PHX_MC_OK) return source_status;
    source_points_.assign(source->points.begin(), source->points.end());
    source_faces_.assign(source->faces.begin(), source->faces.end());
    source_face_groups_.assign(source->face_group_rows.begin(), source->face_group_rows.end());
    if (source->face_ids.empty()) {
      source_face_ids_.assign(source->face_group_rows.begin(), source->face_group_rows.end());
    } else {
      source_face_ids_.assign(source->face_ids.begin(), source->face_ids.end());
    }
    source_face_tolerances_.assign(source->face_group_rows.size(), 0.0);
    if (!source->face_tolerances.empty()) {
      std::copy(source->face_tolerances.begin(), source->face_tolerances.end(),
                source_face_tolerances_.begin());
    }
    source_segments_.assign(source->segments.begin(), source->segments.end());
    source_segment_groups_.assign(source->segment_group_rows.begin(),
                                  source->segment_group_rows.end());
    if (source->segment_ids.empty()) {
      source_segment_ids_.assign(source->segment_group_rows.begin(), source->segment_group_rows.end());
    } else {
      source_segment_ids_.assign(source->segment_ids.begin(), source->segment_ids.end());
    }
    source_segment_tolerances_.assign(source->segment_group_rows.size(), 0.0);
    if (!source->segment_tolerances.empty()) {
      std::copy(source->segment_tolerances.begin(), source->segment_tolerances.end(),
                source_segment_tolerances_.begin());
    }
    if (!source->witnesses.empty()) {
      std::copy(source->witnesses.begin(), source->witnesses.end(), witnesses_.begin());
    }
    if ((!source->face_planes.empty() &&
         source->face_planes.size() != 9 * source->face_group_rows.size()) ||
        (!source->face_plane_tolerances.empty() &&
         source->face_plane_tolerances.size() != source->face_group_rows.size())) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }

  const std::size_t source_vertex_count = source_points_.size() / 3;
  for (std::int32_t vertex : source_faces_) {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= source_vertex_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (std::int32_t vertex : source_segments_) {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= source_vertex_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (double tolerance : source_face_tolerances_) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) return PHX_MC_INVALID_INPUT;
  }
  for (double tolerance : source_segment_tolerances_) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) return PHX_MC_INVALID_INPUT;
  }
  if (const std::int32_t declared =
          source == nullptr
              ? install_declared_planes({}, {})
              : install_declared_planes(source->face_planes, source->face_plane_tolerances);
      declared != PHX_MC_OK) {
    return declared;
  }
  for (std::size_t row = 0; row < face_count; ++row) {
    const std::int32_t source_group = face_sources[row];
    if (source_group < 0 ||
        (!face_orientations.empty() && face_orientations[row] != 1 &&
         face_orientations[row] != -1)) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (std::size_t row = 0; row < segment_count; ++row) {
    if (segment_sources[row] < 0 || segments[2 * row] == segments[2 * row + 1] ||
        segments[2 * row] < 0 || segments[2 * row + 1] < 0 ||
        static_cast<std::size_t>(segments[2 * row]) >= vertex_count ||
        static_cast<std::size_t>(segments[2 * row + 1]) >= vertex_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }

  complex_.tets.reserve(cell_count + face_count + 8);
  complex_.tets.resize(cell_count);
  cell_ids_.reserve(cell_count + face_count + 8);
  cell_ids_.resize(cell_count);
  cell_index_.assign_identity(cell_count);
  for (std::size_t row = 0; row < cell_count; ++row) {
    std::int32_t vertices[4];
    std::copy_n(tetrahedra.data() + 4 * row, 4, vertices);
    for (int corner = 0; corner < 4; ++corner) {
      if (vertices[corner] < 0 || static_cast<std::size_t>(vertices[corner]) >= vertex_count) {
        return PHX_MC_INVALID_INPUT;
      }
      for (int other = 0; other < corner; ++other) {
        if (vertices[corner] == vertices[other]) return PHX_MC_INVALID_INPUT;
      }
      vertex_dimension_[static_cast<std::size_t>(vertices[corner])] = 3;
    }
    const double* corners[4];
    const SourceWitness* corner_witnesses[4];
    for (int corner = 0; corner < 4; ++corner) {
      corners[corner] = points_.data() + 3 * static_cast<std::size_t>(vertices[corner]);
      corner_witnesses[corner] = &witnesses_[static_cast<std::size_t>(vertices[corner])];
    }
    if (!carriers_positive(corners, corner_witnesses)) return PHX_MC_INVALID_INPUT;
    complex_.tets[row] = Tetrahedron{{vertices[0], vertices[1], vertices[2], vertices[3]},
                                     {-1, -1, -1, -1}, 0, 0};
    cell_ids_[row] = static_cast<std::uint64_t>(row);
  }
  complex_.finite_count = static_cast<std::int64_t>(cell_count);
  next_cell_id_ = static_cast<std::uint64_t>(cell_count);

  // The boundary faces in ghost order and each constrained face row's cell face.
  std::vector<FaceSlot> boundary_faces;
  std::vector<FaceSlot> face_cells(face_count);
  const std::int32_t paired =
      recovered != nullptr && !recovered->neighbors.empty()
          ? adopt_adjacency(complex_.tets, cell_count, vertex_count, faces, recovered->neighbors,
                            boundary_faces, face_cells)
          : pair_faces(complex_.tets, cell_count, vertex_count, faces, boundary_faces,
                       face_cells);
  if (paired != PHX_MC_OK) return paired;
  facets_.reserve(face_count);
  facet_index_.reserve(face_count);
  for (std::size_t row = 0; row < face_count; ++row) {
    const std::int32_t* vertices = faces.data() + 3 * row;
    for (int corner = 0; corner < 3; ++corner) {
      vertex_dimension_[static_cast<std::size_t>(vertices[corner])] =
          std::min<std::int8_t>(vertex_dimension_[static_cast<std::size_t>(vertices[corner])], 2);
      if (policy_ == BoundaryPolicy::kFixed) {
        protected_vertices_[static_cast<std::size_t>(vertices[corner])] = 1;
      }
    }
    const std::int8_t orientation = face_orientations.empty() ? 1 : face_orientations[row];
    TetSnapshotFacet face;
    face.id = next_facet_id_++;
    for (int k = 0; k < 3; ++k) face.vertices[k] = static_cast<std::uint64_t>(vertices[k]);
    facet_index_.mark_used(face.id);
    face.source = face_sources[row];
    face.orientation = orientation;
    face.protected_facet = policy_ == BoundaryPolicy::kFixed;
    facets_.push_back(face);
  }

  if (segment_count != 0) {
    // A segment must be a cell edge (found among its first endpoint's cells,
    // listed by a counting sort), and each is constrained once.
    std::vector<std::uint32_t> starts(vertex_count + 1, 0U);
    std::vector<std::uint8_t> first_endpoint(vertex_count, 0);
    for (std::size_t row = 0; row < segment_count; ++row) {
      first_endpoint[static_cast<std::size_t>(segments[2 * row])] = 1;
    }
    for (std::size_t row = 0; row < cell_count; ++row) {
      for (std::int32_t vertex : complex_.tets[row].v) {
        if (first_endpoint[static_cast<std::size_t>(vertex)] != 0) {
          ++starts[static_cast<std::size_t>(vertex) + 1];
        }
      }
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      starts[vertex + 1] += starts[vertex];
    }
    std::vector<std::uint32_t> around(starts.back());
    {
      std::vector<std::uint32_t> cursor(starts.begin(), starts.end() - 1);
      for (std::size_t row = 0; row < cell_count; ++row) {
        for (std::int32_t vertex : complex_.tets[row].v) {
          if (first_endpoint[static_cast<std::size_t>(vertex)] != 0) {
            around[cursor[static_cast<std::size_t>(vertex)]++] = static_cast<std::uint32_t>(row);
          }
        }
      }
    }
    std::vector<std::uint64_t> constrained(segment_count);
    for (std::size_t row = 0; row < segment_count; ++row) {
      const std::int32_t a = segments[2 * row];
      const std::int32_t b = segments[2 * row + 1];
      bool edge = false;
      for (std::uint32_t at = starts[static_cast<std::size_t>(a)];
           !edge && at < starts[static_cast<std::size_t>(a) + 1]; ++at) {
        const Tetrahedron& tet = complex_.tets[around[at]];
        edge = std::find(tet.v, tet.v + 4, b) != tet.v + 4;
      }
      if (!edge) return PHX_MC_INVALID_INPUT;
      constrained[row] = edge_key(a, b);
    }
    std::sort(constrained.begin(), constrained.end());
    if (std::adjacent_find(constrained.begin(), constrained.end()) != constrained.end()) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  segments_.reserve(segment_count);
  segment_index_.reserve(segment_count);
  for (std::size_t row = 0; row < segment_count; ++row) {
    const std::int32_t a = segments[2 * row];
    const std::int32_t b = segments[2 * row + 1];
    TetSnapshotSegment segment;
    segment.id = next_segment_id_++;
    segment.vertices = {static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b)};
    segment.source = segment_sources[row];
    segment_index_.mark_used(segment.id);
    segment.protected_segment = policy_ == BoundaryPolicy::kFixed;
    segments_.push_back(segment);
    vertex_dimension_[static_cast<std::size_t>(a)] =
        std::min<std::int8_t>(vertex_dimension_[static_cast<std::size_t>(a)], 1);
    vertex_dimension_[static_cast<std::size_t>(b)] =
        std::min<std::int8_t>(vertex_dimension_[static_cast<std::size_t>(b)], 1);
    if (policy_ == BoundaryPolicy::kFixed) {
      protected_vertices_[static_cast<std::size_t>(a)] = 1;
      protected_vertices_[static_cast<std::size_t>(b)] = 1;
    }
  }

  // One ghost cell per boundary face, in face key order, closing the hull.
  if (cell_count + boundary_faces.size() >= static_cast<std::size_t>(kMaxTetrahedronSlots)) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  complex_.tets.resize(cell_count + boundary_faces.size());
  for (std::size_t index = 0; index < boundary_faces.size(); ++index) {
    const auto [owner, slot] = boundary_faces[index];
    const std::int32_t ghost_slot = static_cast<std::int32_t>(cell_count + index);
    Tetrahedron& cell = complex_.tets[static_cast<std::size_t>(owner)];
    std::int32_t oriented[3];
    oriented_facet(cell.v, slot, oriented);
    complex_.tets[static_cast<std::size_t>(ghost_slot)] =
        Tetrahedron{{kGhostVertex, oriented[0], oriented[1], oriented[2]},
                    {owner, -1, -1, -1}, 0, 0};
    cell.n[slot] = ghost_slot;
  }
  cell_ids_.resize(complex_.tets.size(), std::numeric_limits<std::uint64_t>::max());
  if (!link_ghosts(complex_.tets, cell_count, vertex_count)) return PHX_MC_INVALID_INPUT;
  complex_.enable_labels();
  if (!regions.empty()) {
    for (std::size_t row = 0; row < cell_count; ++row) {
      complex_.set_region(static_cast<std::int32_t>(row), regions[row]);
    }
  }
  for (std::size_t row = 0; row < face_count; ++row) {
    complex_.set_facet_constraint(face_cells[row].first, face_cells[row].second,
                                  face_sources[row]);
  }
  if (pending_implicit_source_) {
    auto pending = std::move(pending_implicit_source_);
    const int32_t status = bind_implicit_source(std::move(pending));
    if (status != PHX_MC_OK) return status;
  }

  // Every boundary vertex of a constrained facet on a declared plane must be
  // admitted on it; the first that is not (facet order, then corner order)
  // refuses the declaration with its record as evidence.
  for (std::size_t row = 0; row < face_count; ++row) {
    if (implicit_source_ &&
        std::binary_search(implicit_source_->groups.begin(), implicit_source_->groups.end(),
                           facets_[row].source)) continue;
    const auto group = declared_groups_.find(facets_[row].source);
    if (group == declared_groups_.end()) continue;
    const std::int32_t plane_row = group->second.rows.front();
    for (std::uint64_t vertex : facets_[row].vertices) {
      const double* carrier = points_.data() + 3 * vertex;
      if (plane_admits(plane_row, carrier)) continue;
      TetPlaneRefusal refusal;
      refusal.vertex = static_cast<std::int64_t>(vertex);
      if (!plane_record(plane_row, carrier, refusal.record)) return PHX_MC_INTERNAL_ERROR;
      refusal.tolerance = source_face_plane_tolerances_[static_cast<std::size_t>(plane_row)];
      plane_refusal_ = refusal;
      return PHX_MC_INVALID_INPUT;
    }
  }
  // A recovered mesh that is its own source bank is covered by construction
  // (RecoveredTopology::self_covered).
  if ((recovered == nullptr || !recovered->self_covered) && !constraints_covered()) {
    return PHX_MC_INVALID_INPUT;
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (vertex_dimension_[vertex] < 0) return PHX_MC_INVALID_INPUT;
    TetEditVertex candidate;
    candidate.id = vertex_ids_[vertex];
    std::copy_n(points_.data() + 3 * vertex, 3, candidate.carrier.begin());
    candidate.witness = witnesses_[vertex];
    candidate.has_edge_parameter = vertex_has_edge_parameter_[vertex] != 0;
    candidate.edge_endpoints = vertex_edge_endpoints_[vertex];
    candidate.edge_parameter = vertex_edge_parameters_[vertex];
    candidate.protection_radius = protection_radii_[vertex];
    candidate.protected_vertex = protected_vertices_[vertex] != 0;
    if (!source_carrier_valid(candidate)) return PHX_MC_INVALID_INPUT;
  }
  // The adjacency, ghost hull and identities above are consistent by
  // construction (every face pairing checked its reversed orientation).
  return PHX_MC_OK;
}

// Whether every constrained face and segment is covered by its source group
// (source_face_valid, source_segment_valid), each in row order.
bool TetMesh::constraints_covered() const {
  const std::size_t face_count = facets_.size();
  const std::size_t segment_count = segments_.size();
  std::vector<std::pair<std::int32_t, std::uint32_t>> grouped(source_face_groups_.size());
  for (std::size_t row = 0; row < grouped.size(); ++row) {
    grouped[row] = {source_face_groups_[row], static_cast<std::uint32_t>(row)};
  }
  std::sort(grouped.begin(), grouped.end());
  using GroupRows = std::span<const std::pair<std::int32_t, std::uint32_t>>;
  const auto implicit_group = [&](std::int32_t source) {
    return implicit_source_ &&
           std::binary_search(implicit_source_->groups.begin(), implicit_source_->groups.end(),
                              source);
  };
  const auto prepare_group = [&](GroupRows rows, PreparedFacetGroup& group) {
    const std::int32_t source = rows.front().first;
    const bool implicit = implicit_group(source);
    group.rows.clear();
    group.primitives.clear();
    group.indices.clear();
    group.bounds.clear();
    group.prepare_declared = !implicit && declared_groups_.contains(source);
    group.prepare_bounds = !implicit && !group.prepare_declared;
    for (const auto& entry : rows) {
      const std::size_t row = entry.second;
      const auto* face = source_faces_.data() + 3 * row;
      group.rows.push_back(row);
      group.primitives.push_back({face[0], face[1], face[2], static_cast<std::int32_t>(row)});
      group.indices.push_back(static_cast<std::int32_t>(group.indices.size()));
      if (group.prepare_bounds) {
        group.bounds.push_back(facet_bounds(
            source_points_.data() + 3 * static_cast<std::size_t>(face[0]),
            source_points_.data() + 3 * static_cast<std::size_t>(face[1]),
            source_points_.data() + 3 * static_cast<std::size_t>(face[2])));
      }
    }
    if (group.prepare_declared) {
      const auto declared = declared_groups_.find(source);
      if (declared != declared_groups_.end()) prepare_declared_group(declared->second, group);
    } else if (group.prepare_bounds) {
      group.coverage_index.build(group.primitives, group.indices, group.bounds);
    }
  };
  // Per distinct group: its rows in `grouped` and its up-front preparation
  // (-1: the scratch group).
  std::vector<std::int32_t> group_ids;
  std::vector<std::uint32_t> group_starts;
  std::vector<std::int32_t> group_prepared;
  std::size_t prepared_count = 0;
  for (std::size_t at = 0; at < grouped.size();) {
    std::size_t end = at + 1;
    while (end < grouped.size() && grouped[end].first == grouped[at].first) ++end;
    const std::int32_t source = grouped[at].first;
    const bool scratch = end - at == 1 && !implicit_group(source) &&
                         !declared_groups_.contains(source);
    group_ids.push_back(source);
    group_starts.push_back(static_cast<std::uint32_t>(at));
    group_prepared.push_back(scratch ? -1 : static_cast<std::int32_t>(prepared_count++));
    at = end;
  }
  group_starts.push_back(static_cast<std::uint32_t>(grouped.size()));
  // Coverage indices view their group's own banks, so every group is placed
  // before any is prepared.
  std::vector<PreparedFacetGroup> prepared_groups(prepared_count);
  const auto rows_of = [&](std::size_t group) {
    return GroupRows(grouped).subspan(group_starts[group],
                                      group_starts[group + 1] - group_starts[group]);
  };
  for (std::size_t group = 0; group < group_ids.size(); ++group) {
    if (group_prepared[group] >= 0) {
      prepare_group(rows_of(group),
                    prepared_groups[static_cast<std::size_t>(group_prepared[group])]);
    }
  }
  PreparedFacetGroup scratch_group;
  const PreparedFacetGroup no_source_rows;
  for (std::size_t row = 0; row < face_count; ++row) {
    const TetSnapshotFacet& face = facets_[row];
    TetEditFacet candidate;
    candidate.id = face.id;
    candidate.vertices = face.vertices;
    candidate.source = face.source;
    candidate.orientation = face.orientation;
    candidate.protected_facet = face.protected_facet;
    const auto found = std::lower_bound(group_ids.begin(), group_ids.end(), face.source);
    const PreparedFacetGroup* group = &no_source_rows;
    if (found != group_ids.end() && *found == face.source) {
      const auto index = static_cast<std::size_t>(found - group_ids.begin());
      if (group_prepared[index] < 0) {
        prepare_group(rows_of(index), scratch_group);
        group = &scratch_group;
      } else {
        group = &prepared_groups[static_cast<std::size_t>(group_prepared[index])];
      }
    }
    if (!source_face_valid(candidate, nullptr, group)) return false;
  }
  // Source segment row order determines the first successful exact witness.
  std::vector<std::pair<std::int32_t, std::uint32_t>> segment_grouped(
      source_segment_groups_.size());
  for (std::size_t row = 0; row < segment_grouped.size(); ++row) {
    segment_grouped[row] = {source_segment_groups_[row], static_cast<std::uint32_t>(row)};
  }
  std::sort(segment_grouped.begin(), segment_grouped.end());
  std::vector<std::size_t> segment_rows;
  for (std::size_t row = 0; row < segment_count; ++row) {
    const TetSnapshotSegment& segment = segments_[row];
    TetEditSegment candidate;
    candidate.id = segment.id;
    candidate.vertices = segment.vertices;
    candidate.source = segment.source;
    candidate.protected_segment = segment.protected_segment;
    const auto begin = std::lower_bound(
        segment_grouped.begin(), segment_grouped.end(),
        std::pair<std::int32_t, std::uint32_t>{segment.source, 0U});
    segment_rows.clear();
    for (auto entry = begin; entry != segment_grouped.end() && entry->first == segment.source;
         ++entry) {
      segment_rows.push_back(entry->second);
    }
    if (!source_segment_valid(candidate, nullptr, &segment_rows)) return false;
  }
  return true;
}

int32_t TetMesh::install_vertex_sizes(std::span<const double> sizes) {
  if (sizes.empty()) return PHX_MC_OK;
  std::size_t live = 0;
  for (std::uint8_t flag : vertex_live_) live += flag != 0;
  if (sizes.size() != live) return PHX_MC_INVALID_ARGUMENT;
  for (double size : sizes) {
    if (!std::isfinite(size) || size < 0.0) return PHX_MC_INVALID_INPUT;
  }
  try {
    std::vector<double> installed(vertex_live_.size(), 0.0);
    std::size_t row = 0;
    for (std::size_t slot = 0; slot < vertex_live_.size(); ++slot) {
      if (vertex_live_[slot] != 0) installed[slot] = sizes[row++];
    }
    vertex_sizes_.swap(installed);
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_ALLOCATION_FAILURE;
  }
}

void TetMesh::copy_state_to(TetMesh& destination) const {
  destination.metric_ = metric_;
  destination.metric_inverse_ = metric_inverse_;
  destination.metric_active_ = metric_active_;
  destination.metric_orientation_ = metric_orientation_;
  destination.complex_ = complex_;
  destination.points_ = points_;
  destination.vertex_ids_ = vertex_ids_;
  destination.vertex_live_ = vertex_live_;
  destination.vertex_dimension_ = vertex_dimension_;
  destination.protection_radii_ = protection_radii_;
  destination.protected_vertices_ = protected_vertices_;
  destination.witnesses_ = witnesses_;
  destination.vertex_has_edge_parameter_ = vertex_has_edge_parameter_;
  destination.vertex_edge_endpoints_ = vertex_edge_endpoints_;
  destination.vertex_edge_parameters_ = vertex_edge_parameters_;
  destination.cell_ids_ = cell_ids_;
  destination.vertex_ancestry_ = vertex_ancestry_;
  destination.cell_ancestry_ = cell_ancestry_;
  destination.facets_ = facets_;
  destination.segments_ = segments_;
  destination.source_points_ = source_points_;
  destination.source_faces_ = source_faces_;
  destination.source_face_groups_ = source_face_groups_;
  destination.source_face_ids_ = source_face_ids_;
  destination.source_face_tolerances_ = source_face_tolerances_;
  destination.source_segments_ = source_segments_;
  destination.source_segment_groups_ = source_segment_groups_;
  destination.source_segment_ids_ = source_segment_ids_;
  destination.source_segment_tolerances_ = source_segment_tolerances_;
  destination.source_face_planes_ = source_face_planes_;
  destination.source_face_plane_tolerances_ = source_face_plane_tolerances_;
  destination.declared_groups_ = declared_groups_;
  destination.implicit_source_ = implicit_source_;
  destination.pending_implicit_source_ = pending_implicit_source_;
  destination.periodic_orbits_ = periodic_orbits_;
  destination.quotient_cells_ = quotient_cells_;
  destination.periodic_entity_maps_ = periodic_entity_maps_;
  destination.vertex_sizes_ = vertex_sizes_;
  destination.vertex_index_ = vertex_index_;
  destination.cell_index_ = cell_index_;
  destination.used_periodic_orbit_ids_ = used_periodic_orbit_ids_;
  destination.used_quotient_cell_ids_ = used_quotient_cell_ids_;
  destination.used_periodic_lifted_ids_ = used_periodic_lifted_ids_;
  destination.used_periodic_quotient_ids_ = used_periodic_quotient_ids_;
  destination.facet_index_ = facet_index_;
  destination.segment_index_ = segment_index_;
  destination.generation_ = generation_;
  destination.delaunay_generation_ = delaunay_generation_;
  destination.next_vertex_id_ = next_vertex_id_;
  destination.next_cell_id_ = next_cell_id_;
  destination.next_facet_id_ = next_facet_id_;
  destination.next_segment_id_ = next_segment_id_;
  destination.identity_bound_ = identity_bound_;
  destination.source_id_ = source_id_;
  destination.source_revision_ = source_revision_;
  destination.source_map_id_ = source_map_id_;
  destination.frame_id_ = frame_id_;
  destination.periodic_identity_ = periodic_identity_;
}

void TetMesh::swap_state(TetMesh& other) noexcept {
  using std::swap;
  swap(metric_, other.metric_);
  swap(metric_inverse_, other.metric_inverse_);
  swap(metric_active_, other.metric_active_);
  swap(metric_orientation_, other.metric_orientation_);
  swap(complex_, other.complex_);
  swap(points_, other.points_);
  swap(vertex_ids_, other.vertex_ids_);
  swap(vertex_live_, other.vertex_live_);
  swap(vertex_dimension_, other.vertex_dimension_);
  swap(protection_radii_, other.protection_radii_);
  swap(protected_vertices_, other.protected_vertices_);
  swap(witnesses_, other.witnesses_);
  swap(vertex_has_edge_parameter_, other.vertex_has_edge_parameter_);
  swap(vertex_edge_endpoints_, other.vertex_edge_endpoints_);
  swap(vertex_edge_parameters_, other.vertex_edge_parameters_);
  swap(cell_ids_, other.cell_ids_);
  swap(vertex_ancestry_, other.vertex_ancestry_);
  swap(cell_ancestry_, other.cell_ancestry_);
  swap(facets_, other.facets_);
  swap(segments_, other.segments_);
  swap(source_points_, other.source_points_);
  swap(source_faces_, other.source_faces_);
  swap(source_face_groups_, other.source_face_groups_);
  swap(source_face_ids_, other.source_face_ids_);
  swap(source_face_tolerances_, other.source_face_tolerances_);
  swap(source_segments_, other.source_segments_);
  swap(source_segment_groups_, other.source_segment_groups_);
  swap(source_segment_tolerances_, other.source_segment_tolerances_);
  swap(source_face_planes_, other.source_face_planes_);
  swap(source_face_plane_tolerances_, other.source_face_plane_tolerances_);
  swap(declared_groups_, other.declared_groups_);
  swap(implicit_source_, other.implicit_source_);
  swap(pending_implicit_source_, other.pending_implicit_source_);
  swap(source_segment_ids_, other.source_segment_ids_);
  swap(periodic_orbits_, other.periodic_orbits_);
  swap(quotient_cells_, other.quotient_cells_);
  swap(periodic_entity_maps_, other.periodic_entity_maps_);
  swap(vertex_sizes_, other.vertex_sizes_);
  swap(vertex_index_, other.vertex_index_);
  swap(cell_index_, other.cell_index_);
  swap(used_periodic_orbit_ids_, other.used_periodic_orbit_ids_);
  swap(used_quotient_cell_ids_, other.used_quotient_cell_ids_);
  swap(used_periodic_lifted_ids_, other.used_periodic_lifted_ids_);
  swap(used_periodic_quotient_ids_, other.used_periodic_quotient_ids_);
  swap(facet_index_, other.facet_index_);
  swap(segment_index_, other.segment_index_);
  swap(generation_, other.generation_);
  swap(delaunay_generation_, other.delaunay_generation_);
  swap(next_vertex_id_, other.next_vertex_id_);
  swap(next_cell_id_, other.next_cell_id_);
  swap(next_facet_id_, other.next_facet_id_);
  swap(next_segment_id_, other.next_segment_id_);
  swap(identity_bound_, other.identity_bound_);
  swap(source_id_, other.source_id_);
  swap(source_revision_, other.source_revision_);
  swap(source_map_id_, other.source_map_id_);
  swap(frame_id_, other.frame_id_);
  swap(periodic_identity_, other.periodic_identity_);
}

bool TetMesh::audit_state() const {
  if (!complex_.audit() || points_.size() != 3 * vertex_ids_.size() ||
      vertex_live_.size() != vertex_ids_.size() ||
      vertex_dimension_.size() != vertex_ids_.size() ||
      protection_radii_.size() != vertex_ids_.size() ||
      protected_vertices_.size() != vertex_ids_.size() ||
      vertex_has_edge_parameter_.size() != vertex_ids_.size() ||
      vertex_edge_endpoints_.size() != vertex_ids_.size() ||
      vertex_edge_parameters_.size() != vertex_ids_.size() ||
      witnesses_.size() != vertex_ids_.size() || cell_ids_.size() != complex_.tets.size() ||
      source_faces_.size() % 3 != 0 || source_segments_.size() % 2 != 0 ||
      source_face_groups_.size() != source_faces_.size() / 3 ||
      source_face_ids_.size() != source_faces_.size() / 3 ||
      source_face_tolerances_.size() != source_faces_.size() / 3 ||
      source_segment_groups_.size() != source_segments_.size() / 2 ||
      source_segment_ids_.size() != source_segments_.size() / 2 ||
      source_segment_tolerances_.size() != source_segments_.size() / 2 ||
      source_face_planes_.size() != 3 * source_faces_.size() ||
      source_face_plane_tolerances_.size() != source_faces_.size() / 3 ||
      !std::all_of(vertex_ancestry_.begin(), vertex_ancestry_.end(), [&](const auto& item) {
        if (!vertex_index_.used(item.first) || item.second.empty()) return false;
        return std::all_of(item.second.begin(), item.second.end(), [&](const TetVertexAncestry& ancestor) {
          return vertex_index_.used(ancestor.source_vertex_id) &&
                 std::isfinite(ancestor.weight);
        });
      }) ||
      !std::all_of(cell_ancestry_.begin(), cell_ancestry_.end(), [&](const auto& item) {
        if (!cell_index_.used(item.first) || !item.second.ancestor_cell_ids ||
            item.second.edit_kind < 0 ||
            item.second.edit_kind > static_cast<std::int32_t>(TetEditKind::kRetriangulate)) {
          return false;
        }
        return std::all_of(item.second.ancestor_cell_ids->begin(),
                           item.second.ancestor_cell_ids->end(), [&](std::uint64_t ancestor) {
          return cell_index_.used(ancestor);
        });
      })) {
    return false;
  }
  std::size_t live_vertices = 0;
  for (std::size_t slot = 0; slot < vertex_ids_.size(); ++slot) {
    if (vertex_live_[slot] == 0) continue;
    ++live_vertices;
    if (vertex_index_.slot(vertex_ids_[slot]) != static_cast<std::int32_t>(slot)) return false;
  }
  std::size_t live_cells = 0;
  for (std::size_t slot = 0; slot < complex_.tets.size(); ++slot) {
    const std::int32_t local = static_cast<std::int32_t>(slot);
    if (!complex_.live(local)) continue;
    const Tetrahedron& cell = complex_.tets[slot];
    for (std::int32_t vertex : cell.v) {
      if (vertex != kGhostVertex && !live_vertex(vertex)) return false;
    }
    if (is_ghost(cell)) continue;
    ++live_cells;
    if (cell_index_.slot(cell_ids_[slot]) != local) return false;
  }
  if (live_vertices != vertex_index_.live_count() || live_cells != cell_index_.live_count()) {
    return false;
  }
  for (const TetSnapshotFacet& facet : facets_) {
    if (!facet_index_.used(facet.id)) return false;
    for (std::uint64_t vertex : facet.vertices) if (vertex_slot(vertex) < 0) return false;
  }
  for (const TetSnapshotSegment& segment : segments_) {
    if (!segment_index_.used(segment.id)) return false;
    if (vertex_slot(segment.vertices[0]) < 0 || vertex_slot(segment.vertices[1]) < 0) return false;
  }
  return true;
}

}  // namespace phx::mc
