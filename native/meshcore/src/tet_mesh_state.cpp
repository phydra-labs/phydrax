//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "tet_mesh.hpp"

namespace phx::mc {

TetMesh::TetMesh(BoundaryPolicy policy, int64_t max_vertices, int64_t max_tetrahedra,
                 std::size_t max_allocation_bytes)
    : TetMesh(policy, max_vertices, max_tetrahedra, select_memory_owner(max_allocation_bytes)) {}

TetMesh::TetMesh(BoundaryPolicy policy, int64_t max_vertices, int64_t max_tetrahedra,
                 MemoryOwner owner)
    : memory_owner_(checked_memory_owner(std::move(owner))),
      policy_(policy),
      max_vertices_(max_vertices),
      max_tetrahedra_(max_tetrahedra),
      complex_(kMaxTetrahedronSlots, memory_owner_),
      edit_(complex_) {}

int32_t TetMesh::build(int64_t point_count, const double* points, int64_t tet_count,
                       const int32_t* tets, const int32_t* regions, int64_t face_count,
                       const int32_t* faces, const int32_t* face_sources, int64_t segment_count,
                       const int32_t* segments, const int32_t* segment_sources,
                       const double* protection, const SourceComplex* source) {
  const MemoryScope memory_scope(memory_owner_);
  try {
    TetMesh staged(policy_, max_vertices_, max_tetrahedra_, memory_owner_);
    const int32_t status = staged.build_state(point_count, points, tet_count, tets, regions,
                                              face_count, faces, face_sources, segment_count,
                                              segments, segment_sources, protection, source);
    if (status != PHX_MC_OK) {
      return status;
    }
    if (!native_execution_spend(TetWorkCost::kCheckpoint)) return PHX_MC_CAPACITY_EXCEEDED;
    // Publish only after every input/state allocation and predicate succeeds.
    // Moving identically owned allocators does not allocate.
    complex_ = std::move(staged.complex_);
    // Boundary-witness preparation traverses the staged complex. Its marks
    // and generation must move together so later cavities cannot reuse them.
    stamp_ = staged.stamp_;
    points_ = std::move(staged.points_);
    original_points_ = std::move(staged.original_points_);
    original_faces_ = std::move(staged.original_faces_);
    original_segments_ = std::move(staged.original_segments_);
    face_tolerance_ = std::move(staged.face_tolerance_);
    segment_tolerance_ = std::move(staged.segment_tolerance_);
    face_rows_ = std::move(staged.face_rows_);
    segment_rows_ = std::move(staged.segment_rows_);
    witnesses_ = std::move(staged.witnesses_);
    source_refusals_.clear();
    dimension_ = std::move(staged.dimension_);
    protection_ = std::move(staged.protection_);
    sizes_ = std::move(staged.sizes_);
    vertex_tet_ = std::move(staged.vertex_tet_);
    vertex_stamp_ = std::move(staged.vertex_stamp_);
    segments_ = std::move(staged.segments_);
    records_ = std::move(staged.records_);
    slot_reason_ = std::move(staged.slot_reason_);
    unmet_.clear();
    cavity_.clear();
    stack_.clear();
    scratch_star_.clear();
    boundary_.clear();
    crossed_.clear();
    boundary_edges_.clear();
    new_faces_.clear();
    edit_.begin(EditLimits{});
    transaction_open_ = false;
    pending_ = kNoPending;
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
}

int32_t TetMesh::build_state(int64_t point_count, const double* points, int64_t tet_count,
                             const int32_t* tets, const int32_t* regions, int64_t face_count,
                             const int32_t* faces, const int32_t* face_sources,
                             int64_t segment_count, const int32_t* segments,
                             const int32_t* segment_sources, const double* protection,
                             const SourceComplex* source) {
  const MemoryScope memory_scope(memory_owner_);
  const auto n = static_cast<std::size_t>(point_count);
  native_execution_charge(tet_count);
  points_.assign(points, points + 3 * n);
  dimension_.assign(n, -1);
  sizes_.assign(n, 0.0);
  vertex_tet_.assign(n, -1);
  vertex_stamp_.assign(n + 1, 0U);
  if (protection != nullptr) {
    protection_.assign(protection, protection + n);
  } else {
    protection_.assign(n, 0.0);
  }
  for (double radius : protection_) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    if (!std::isfinite(radius) || radius < 0.0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  int32_t status = build_cells(point_count, tet_count, tets, regions);
  if (status == PHX_MC_OK) {
    status = build_constraints(point_count, face_count, faces, face_sources);
  }
  if (status == PHX_MC_OK) {
    SourceComplex own;
    own.face_count = face_count;
    own.faces = faces;
    own.face_sources = face_sources;
    own.segment_count = segment_count;
    own.segments = segments;
    own.segment_sources = segment_sources;
    status = build_source(point_count, source != nullptr ? *source : own);
  }
  if (status == PHX_MC_OK) {
    status = build_segments(point_count, segment_count, segments, segment_sources);
  }
  if (status == PHX_MC_OK) {
    classify_vertices();
    status = complete_boundary_witnesses();
    if (status == PHX_MC_OK) {
      status = verify_ancestry(point_count, face_count, faces, face_sources, segment_count,
                               segments, segment_sources);
    }
  }
  slot_reason_.assign(complex_.tets.size(), Unmet::kNone);
  records_.clear();
  records_.shrink_to_fit();
  return status;
}

void TetMesh::set_work_limit(int64_t budget) {
  work_exhausted_ = false;
  work_limit_ = work_ > std::numeric_limits<int64_t>::max() - budget
                    ? std::numeric_limits<int64_t>::max()
                    : work_ + budget;
}

bool TetMesh::spend(int64_t units) {
  if (work_exhausted_ || units < 0 || work_ > work_limit_ - units ||
      !native_execution_spend(units)) {
    work_exhausted_ = true;
    return false;
  }
  work_ += units;
  return true;
}

bool TetMesh::protect_vertices(int64_t count, const int32_t* vertices) {
  if (count < 0 || (count > 0 && vertices == nullptr) || !spend(count) || !spend(count)) {
    return false;
  }
  for (int64_t i = 0; i < count; ++i) {
    if (!live_vertex(vertices[i])) {
      return false;
    }
  }
  for (int64_t i = 0; i < count; ++i) {
    dimension_[static_cast<std::size_t>(vertices[i])] = 0;
  }
  if (count > 0) {
    mark_accepted_state();
  }
  return true;
}

int TetMesh::orient_with(int32_t t, int slot, const double* p) const {
  const MemoryScope memory_scope(memory_owner_);
  if (p == pending_point_ && pending_ != kNoPending) {
    int32_t vertices[4];
    std::copy_n(tet(t).v, 4, vertices);
    vertices[slot] = pending_;
    return source_orientation(vertices);
  }
  const double* q[4];
  for (int k = 0; k < 4; ++k) {
    q[k] = k == slot ? p : point(tet(t).v[k]);
  }
  return orient3d(q[0], q[1], q[2], q[3]);
}

template <class Allocator>
bool TetMesh::vertex_star(int32_t a, std::vector<int32_t, Allocator>& star) {
  const MemoryScope memory_scope(memory_owner_);
  star.clear();
  if (!live_vertex(a)) {
    return false;
  }
  const int32_t start = vertex_tet_[static_cast<std::size_t>(a)];
  if (start < 0 || !complex_.live(start) || vertex_slot(tet(start), a) < 0) {
    return false;
  }
  if (!spend(TetWorkCost::kVisit)) return false;
  next_stamp();
  mark(start);
  star.push_back(start);
  for (std::size_t i = 0; i < star.size(); ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const Tetrahedron& current = tet(star[i]);
    const int apex = vertex_slot(current, a);
    for (int k = 0; k < 4; ++k) {
      const int32_t next = current.n[k];
      if (k != apex && !marked(next)) {
        if (!spend(TetWorkCost::kVisit)) return false;
        mark(next);
        star.push_back(next);
      }
    }
  }
  return true;
}

int32_t TetMesh::edge_tet(int32_t a, int32_t b) {
  const MemoryScope memory_scope(memory_owner_);
  if (!vertex_star(a, scratch_star_)) {
    return -1;
  }
  for (int32_t t : scratch_star_) {
    if (vertex_slot(tet(t), b) >= 0) {
      return t;
    }
  }
  return -1;
}

bool TetMesh::find_face(int32_t a, int32_t b, int32_t c, int32_t& t, int& slot) {
  const MemoryScope memory_scope(memory_owner_);
  if (!vertex_star(a, scratch_star_)) {
    return false;
  }
  for (int32_t s : scratch_star_) {
    const Tetrahedron& current = tet(s);
    if (is_ghost(current) || vertex_slot(current, b) < 0 || vertex_slot(current, c) < 0) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      if (current.v[k] != a && current.v[k] != b && current.v[k] != c) {
        t = s;
        slot = k;
        return true;
      }
    }
  }
  return false;
}

template <class TetAllocator, class RingAllocator>
bool TetMesh::edge_ring(int32_t a, int32_t b, std::vector<int32_t, TetAllocator>& tets,
                        std::vector<int32_t, RingAllocator>& ring) {
  const MemoryScope memory_scope(memory_owner_);
  tets.clear();
  ring.clear();
  const int32_t start = edge_tet(a, b);
  if (start < 0) {
    return false;
  }
  int32_t t = start;
  int32_t entry = -1;
  for (;;) {
    const Tetrahedron& current = tet(t);
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
    if (!even_permutation(current.v, order)) {
      std::swap(first, second);
    }
    if (entry >= 0 && first != entry) {
      return false;
    }
    tets.push_back(t);
    ring.push_back(first);
    // Next around the edge: across the facet (a, b, second).
    const int32_t next = current.n[vertex_slot(current, first)];
    if (next == start) {
      break;
    }
    if (tets.size() > complex_.tets.size()) {
      return false;
    }
    entry = second;
    t = next;
  }
  return spend(static_cast<int64_t>(tets.size()));
}

MemoryOwner TetMesh::checked_memory_owner(MemoryOwner owner) {
  if (!owner) {
    throw std::invalid_argument("native.tet_mesh.memory_owner must be nonnull; got null");
  }
  return owner;
}

MemoryOwner TetMesh::select_memory_owner(std::size_t bytes) {
  MemoryOwner owner = scratch_memory_owner();
  if (owner &&
      (bytes == std::numeric_limits<std::size_t>::max() || owner->limit_bytes() == bytes)) {
    return owner;
  }
  return std::make_shared<BoundedMemoryResource>(bytes);
}

int32_t TetMesh::build_cells(int64_t point_count, int64_t tet_count, const int32_t* tets,
                             const int32_t* regions) {
  const MemoryScope memory_scope(memory_owner_);
  complex_.tets.clear();
  complex_.tets.reserve(static_cast<std::size_t>(2 * tet_count + 8));
  for (int64_t i = 0; i < tet_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t* v = tets + 4 * i;
    for (int k = 0; k < 4; ++k) {
      if (v[k] < 0 || v[k] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
      for (int j = 0; j < k; ++j) {
        if (v[j] == v[k]) {
          return PHX_MC_INVALID_INPUT;
        }
      }
    }
    // Source witnesses are published after carrier topology is constructed.
    // verify_ancestry subsequently proves positivity on the authoritative S.
    if (regions[i] < 0 || orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) <= 0) {
      return PHX_MC_INVALID_INPUT;
    }
    complex_.tets.push_back(Tetrahedron{{v[0], v[1], v[2], v[3]}, {-1, -1, -1, -1}, 0U, 0});
  }
  complex_.finite_count = tet_count;
  records_.clear();
  records_.reserve(static_cast<std::size_t>(4 * tet_count));
  for (int64_t i = 0; i < tet_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t* v = tets + 4 * i;
    for (int k = 0; k < 4; ++k) {
      int32_t facet[3];
      oriented_facet(v, k, facet);
      records_.push_back({FaceKey::of(facet[0], facet[1], facet[2]), static_cast<int32_t>(i), k});
    }
  }
  std::sort(records_.begin(), records_.end(), [](const FacetRecord& a, const FacetRecord& b) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    return a < b;
  });
  NativeVector<int32_t> ghosts;
  for (std::size_t r = 0; r < records_.size();) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    std::size_t end = r + 1;
    while (end < records_.size() && records_[end].key == records_[r].key) {
      ++end;
    }
    if (end - r > 2) {
      return PHX_MC_INVALID_INPUT;
    }
    const FacetRecord& first = records_[r];
    Tetrahedron& t = complex_.tets[static_cast<std::size_t>(first.tet)];
    if (end - r == 2) {
      const FacetRecord& second = records_[r + 1];
      Tetrahedron& u = complex_.tets[static_cast<std::size_t>(second.tet)];
      int32_t f[3];
      int32_t g[3];
      oriented_facet(t.v, first.slot, f);
      oriented_facet(u.v, second.slot, g);
      if (!(FacetKey::of(f) == FacetKey::of(g).reversed())) {
        return PHX_MC_INVALID_INPUT;
      }
      t.n[first.slot] = second.tet;
      u.n[second.slot] = first.tet;
    } else {
      Tetrahedron shell = t;
      shell.v[first.slot] = kGhostVertex;
      // Odd permutation of the facet: the outside becomes the positive side.
      const int a = first.slot == 0 ? 1 : 0;
      const int b = first.slot <= 1 ? 2 : 1;
      std::swap(shell.v[a], shell.v[b]);
      shell.n[0] = shell.n[1] = shell.n[2] = shell.n[3] = -1;
      shell.n[first.slot] = first.tet;
      const auto ghost = static_cast<int32_t>(complex_.tets.size());
      complex_.tets.push_back(shell);
      complex_.tets[static_cast<std::size_t>(first.tet)].n[first.slot] = ghost;
      ghosts.push_back(ghost);
    }
    r = end;
  }
  return link_ghosts(ghosts, point_count, regions);
}

int32_t TetMesh::link_ghosts(std::span<const int32_t> ghosts, int64_t point_count,
                             const int32_t* regions) {
  const MemoryScope memory_scope(memory_owner_);
  struct GhostFacet {
    std::uint64_t edge;
    int32_t forward;  // 1 when the facet reads (G, low, high)
    int32_t ghost;
    int32_t slot;
    bool operator<(const GhostFacet& other) const {
      if (edge != other.edge) {
        return edge < other.edge;
      }
      return forward != other.forward ? forward < other.forward : ghost < other.ghost;
    }
  };
  NativeVector<GhostFacet> entries;
  entries.reserve(3 * ghosts.size());
  for (int32_t g : ghosts) {
    const Tetrahedron& shell = tet(g);
    const int apex = vertex_slot(shell, kGhostVertex);
    for (int s = 0; s < 4; ++s) {
      if (s == apex) {
        continue;
      }
      int32_t facet[3];
      oriented_facet(shell.v, s, facet);
      const int at = static_cast<int>(std::find(facet, facet + 3, kGhostVertex) - facet);
      const int32_t p = facet[(at + 1) % 3];
      const int32_t q = facet[(at + 2) % 3];
      entries.push_back({topology::undirected_edge(p, q), p < q ? 1 : 0, g, s});
    }
  }
  std::sort(entries.begin(), entries.end());
  for (std::size_t r = 0; r < entries.size();) {
    std::size_t middle = r;
    while (middle < entries.size() && entries[middle].edge == entries[r].edge &&
           entries[middle].forward == 0) {
      ++middle;
    }
    std::size_t end = middle;
    while (end < entries.size() && entries[end].edge == entries[r].edge) {
      ++end;
    }
    if (middle - r != end - middle) {
      return PHX_MC_INVALID_INPUT;
    }
    for (std::size_t i = 0; i < middle - r; ++i) {
      const GhostFacet& x = entries[r + i];
      const GhostFacet& y = entries[middle + i];
      complex_.tets[static_cast<std::size_t>(x.ghost)].n[x.slot] = y.ghost;
      complex_.tets[static_cast<std::size_t>(y.ghost)].n[y.slot] = x.ghost;
    }
    r = end;
  }
  complex_.enable_labels();
  for (int64_t i = 0; i < complex_.finite_count; ++i) {
    complex_.set_region(static_cast<int32_t>(i), regions[i]);
    for (int32_t v : complex_.tets[static_cast<std::size_t>(i)].v) {
      vertex_tet_[static_cast<std::size_t>(v)] = static_cast<int32_t>(i);
    }
  }
  (void)point_count;
  return complex_.audit() ? PHX_MC_OK : PHX_MC_INVALID_INPUT;
}

int32_t TetMesh::build_constraints(int64_t point_count, int64_t face_count, const int32_t* faces,
                                   const int32_t* sources) {
  const MemoryScope memory_scope(memory_owner_);
  for (int64_t i = 0; i < face_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t* f = faces + 3 * i;
    if (sources[i] < 0) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int k = 0; k < 3; ++k) {
      if (f[k] < 0 || f[k] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    const FacetRecord* record = find_record(FaceKey::of(f[0], f[1], f[2]));
    if (record == nullptr) {
      return PHX_MC_INVALID_INPUT;
    }
    const int32_t existing = complex_.constraint(record->tet, record->slot);
    if (existing == kNoConstraint) {
      complex_.set_facet_constraint(record->tet, record->slot, sources[i]);
    } else if (existing != sources[i]) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  // The domain boundary and every region interface are constrained.
  for (int64_t i = 0; i < complex_.finite_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const auto t = static_cast<int32_t>(i);
    for (int k = 0; k < 4; ++k) {
      const int32_t other = tet(t).n[k];
      const bool open = is_ghost(tet(other)) || complex_.region(other) != complex_.region(t);
      if (open && complex_.constraint(t, k) == kNoConstraint) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  return PHX_MC_OK;
}

int32_t TetMesh::build_segments(int64_t point_count, int64_t segment_count, const int32_t* segments,
                                const int32_t* sources) {
  const MemoryScope memory_scope(memory_owner_);
  NativeVector<std::uint64_t> edges;
  edges.reserve(static_cast<std::size_t>(6 * complex_.finite_count));
  for (int64_t i = 0; i < complex_.finite_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t* v = complex_.tets[static_cast<std::size_t>(i)].v;
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        edges.push_back(topology::undirected_edge(v[a], v[b]));
      }
    }
  }
  std::sort(edges.begin(), edges.end(), [](uint64_t a, uint64_t b) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    return a < b;
  });
  edges.erase(std::unique(edges.begin(), edges.end(),
                          [](uint64_t a, uint64_t b) {
                            native_execution_charge(TetWorkCost::kCheckpoint);
                            return a == b;
                          }),
              edges.end());
  segments_.clear();
  segments_.reserve(static_cast<std::size_t>(segment_count));
  for (int64_t i = 0; i < segment_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t a = segments[2 * i];
    const int32_t b = segments[2 * i + 1];
    if (a < 0 || b < 0 || a >= point_count || b >= point_count || a == b || sources[i] < 0) {
      return PHX_MC_INVALID_INPUT;
    }
    const std::uint64_t key = topology::undirected_edge(a, b);
    if (!std::binary_search(edges.begin(), edges.end(), key)) {
      return PHX_MC_INVALID_INPUT;
    }
    const auto [found, inserted] = segments_.emplace(key, sources[i]);
    if (!inserted && found->second != sources[i]) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  // A face edge that is not a segment lies inside one planar facet: exactly
  // two coplanar faces of one source.
  struct FaceEdge {
    std::uint64_t edge;
    int32_t source;
    int32_t apex;
    bool operator<(const FaceEdge& other) const {
      if (edge != other.edge) {
        return edge < other.edge;
      }
      return source != other.source ? source < other.source : apex < other.apex;
    }
  };
  NativeVector<FaceEdge> face_edges;
  for (const FacetRecord& record : records_) {
    const int32_t source = complex_.constraint(record.tet, record.slot);
    const int32_t other = tet(record.tet).n[record.slot];
    if (source == kNoConstraint || (!is_ghost(tet(other)) && other < record.tet)) {
      continue;
    }
    const int32_t* f = record.key.v;
    face_edges.push_back({topology::undirected_edge(f[0], f[1]), source, f[2]});
    face_edges.push_back({topology::undirected_edge(f[1], f[2]), source, f[0]});
    face_edges.push_back({topology::undirected_edge(f[0], f[2]), source, f[1]});
  }
  std::sort(face_edges.begin(), face_edges.end());
  for (std::size_t r = 0; r < face_edges.size();) {
    std::size_t end = r + 1;
    while (end < face_edges.size() && face_edges[end].edge == face_edges[r].edge) {
      ++end;
    }
    const std::uint64_t key = face_edges[r].edge;
    if (segments_.count(key) == 0) {
      // Exactly planar, or folded only by a certified bounded witness of
      // one of its four vertices; source identity always agrees.
      const auto bounded = [&](int32_t v) {
        return witnesses_[static_cast<std::size_t>(v)].deviation > 0.0;
      };
      const bool paired = end - r == 2 && face_edges[r].source == face_edges[r + 1].source;
      const bool planar =
          paired && (orient3d(point(edge_low(key)), point(edge_high(key)),
                              point(face_edges[r].apex), point(face_edges[r + 1].apex)) == 0 ||
                     bounded(edge_low(key)) || bounded(edge_high(key)) ||
                     bounded(face_edges[r].apex) || bounded(face_edges[r + 1].apex));
      if (!planar) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    r = end;
  }
  return PHX_MC_OK;
}

void TetMesh::classify_vertices() {
  const MemoryScope memory_scope(memory_owner_);
  for (int64_t i = 0; i < complex_.finite_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    for (int32_t v : complex_.tets[static_cast<std::size_t>(i)].v) {
      dimension_[static_cast<std::size_t>(v)] = 3;
    }
  }
  for (const FacetRecord& record : records_) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    if (complex_.constraint(record.tet, record.slot) != kNoConstraint) {
      for (int32_t v : record.key.v) {
        dimension_[static_cast<std::size_t>(v)] = 2;
      }
    }
  }
  NativeVector<int32_t> degree(dimension_.size(), 0);
  NativeVector<int32_t> neighbor(2 * dimension_.size(), -1);
  NativeVector<int32_t> source(2 * dimension_.size(), -1);
  NativeVector<std::uint64_t> keys;
  keys.reserve(segments_.size());
  for (const auto& entry : segments_) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    keys.push_back(entry.first);
  }
  std::sort(keys.begin(), keys.end());
  for (std::uint64_t key : keys) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t ends[2] = {edge_low(key), edge_high(key)};
    for (int e = 0; e < 2; ++e) {
      const auto v = static_cast<std::size_t>(ends[e]);
      if (degree[v] < 2) {
        neighbor[2 * v + static_cast<std::size_t>(degree[v])] = ends[1 - e];
        source[2 * v + static_cast<std::size_t>(degree[v])] = segments_.at(key);
      }
      ++degree[v];
    }
  }
  for (std::size_t v = 0; v < dimension_.size(); ++v) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    if (degree[v] == 0) {
      continue;
    }
    // A bounded segment witness folds the represented curve within its
    // certified deviation; it does not make a corner.
    const SourceWitness& witness = witnesses_[v];
    const bool smooth = degree[v] == 2 && source[2 * v] == source[2 * v + 1] &&
                        ((witness.deviation > 0.0 && witness.stratum == SourceStratum::kSegment) ||
                         geometry::collinear(point(neighbor[2 * v]), point(static_cast<int32_t>(v)),
                                             point(neighbor[2 * v + 1])));
    dimension_[v] = smooth ? 1 : 0;
  }
}

void TetMesh::next_stamp() {
  if (++stamp_ == 0) {
    for (Tetrahedron& t : complex_.tets) {
      t.stamp = 0U;
    }
    stamp_ = 1;
  }
}

void TetMesh::after_commit() {
  const MemoryScope memory_scope(memory_owner_);
  mark_accepted_state();
  slot_reason_.resize(complex_.tets.size(), Unmet::kNone);
  for (int32_t id : edit_.created()) {
    slot_reason_[static_cast<std::size_t>(id)] = Unmet::kNone;
    const Tetrahedron& current = tet(id);
    for (int32_t v : current.v) {
      if (v >= 0) {
        int32_t& incident = vertex_tet_[static_cast<std::size_t>(v)];
        if (incident < 0 || !complex_.live(incident) || vertex_slot(tet(incident), v) < 0 ||
            (is_ghost(tet(incident)) && !is_ghost(current))) {
          incident = id;
        }
      }
    }
  }
}

bool TetMesh::even_permutation(const int32_t* reference, const int32_t* order) {
  int position[4];
  for (int k = 0; k < 4; ++k) {
    position[k] = vertex_slot(
        Tetrahedron{
            {reference[0], reference[1], reference[2], reference[3]}, {-1, -1, -1, -1}, 0U, 0},
        order[k]);
  }
  int inversions = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      inversions += position[i] > position[j] ? 1 : 0;
    }
  }
  return inversions % 2 == 0;
}

template bool TetMesh::vertex_star<std::allocator<int32_t>>(
    int32_t, std::vector<int32_t, std::allocator<int32_t>>&);

template bool TetMesh::edge_ring<std::allocator<int32_t>, std::allocator<int32_t>>(
    int32_t, int32_t, std::vector<int32_t, std::allocator<int32_t>>&,
    std::vector<int32_t, std::allocator<int32_t>>&);

template bool TetMesh::vertex_star<NativeAllocator<int32_t>>(
    int32_t, std::vector<int32_t, NativeAllocator<int32_t>>&);

template bool TetMesh::edge_ring<NativeAllocator<int32_t>, NativeAllocator<int32_t>>(
    int32_t, int32_t, std::vector<int32_t, NativeAllocator<int32_t>>&,
    std::vector<int32_t, NativeAllocator<int32_t>>&);

}  // namespace phx::mc
