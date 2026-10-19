//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "tet_mesh.hpp"

namespace phx::mc {

int32_t TetMesh::locate(const double* p, int32_t start, int32_t& blocked_tet, int& blocked_slot) {
  const MemoryScope memory_scope(memory_owner_);
  blocked_tet = -1;
  blocked_slot = -1;
  int32_t t = start;
  int32_t previous = -1;
  for (;;) {
    if (!spend(TetWorkCost::kVisit)) {
      return -1;
    }
    const Tetrahedron& current = tet(t);
    const int first = static_cast<int>(splitmix64(walk_step_++) & 3U);
    int32_t next = -1;
    int crossing = -1;
    for (int i = 0; i < 4; ++i) {
      const int k = (first + i) & 3;
      if (current.n[k] == previous) {
        continue;
      }
      if (orient_with(t, k, p) < 0) {
        next = current.n[k];
        crossing = k;
        break;
      }
    }
    if (next < 0) {
      return t;
    }
    if (complex_.constraint(t, crossing) != kNoConstraint || is_ghost(tet(next))) {
      blocked_tet = t;
      blocked_slot = crossing;
      return -1;
    }
    previous = t;
    t = next;
  }
}

bool TetMesh::split_positions_admitted(int32_t a, int32_t b, const double* witness,
                                       const double* position, int32_t& source,
                                       bool& on_edge) const {
  if (!live_vertex(a) || !live_vertex(b) || a == b || !geometry::finite3(witness) ||
      !geometry::finite3(position) || !coordinate_in_domain(witness[0]) ||
      !coordinate_in_domain(witness[1]) || !coordinate_in_domain(witness[2]) ||
      !coordinate_in_domain(position[0]) || !coordinate_in_domain(position[1]) ||
      !coordinate_in_domain(position[2]) || !geometry::collinear(point(a), point(b), witness)) {
    return false;
  }
  int axis = 0;
  for (int k = 1; k < 3; ++k) {
    if (std::abs(point(a)[k] - point(b)[k]) > std::abs(point(a)[axis] - point(b)[axis])) {
      axis = k;
    }
  }
  const double low = std::min(point(a)[axis], point(b)[axis]);
  const double high = std::max(point(a)[axis], point(b)[axis]);
  if (!(low < witness[axis] && witness[axis] < high)) {
    return false;
  }
  on_edge = witness == position || geometry::collinear(point(a), point(b), position);
  source = segment_source(a, b);
  return !(source >= 0 && !on_edge) &&
         !(on_edge && !(low < position[axis] && position[axis] < high));
}

Insertion TetMesh::prepare_edge_insertion(int32_t a, int32_t b, const double* witness,
                                          const double* position, double source_fraction,
                                          std::size_t cell_limit, InsertKind& kind) try {
  const MemoryScope memory_scope(memory_owner_);
  int32_t source = -1;
  SourceWitness construction;
  bool bounded = false;
  Insertion status = admit_edge_construction(a, b, witness, position, source_fraction, source,
                                             bounded, construction);
  if (status != Insertion::kOk) return status;
  if (vertex_count() >= max_vertices_ || cell_limit == 0) {
    return Insertion::kCapacity;
  }
  NativeVector<int32_t> star;
  NativeVector<int32_t> ring;
  if (!edge_ring(a, b, star, ring)) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  int32_t seed = -1;
  status = select_edge_insertion_seed(a, b, position, source, bounded, star, kind, seed);
  if (status != Insertion::kOk) return status;
  int32_t blocked = -1;
  int blocked_slot = -1;
  const int32_t origin = bounded ? seed : locate(position, seed, blocked, blocked_slot);
  if (origin < 0) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  status = prepare(position, origin, kind == InsertKind::kSubsegment ? a : -1,
                   kind == InsertKind::kSubsegment ? b : -1, cell_limit,
                   bounded ? &construction : nullptr);
  if (status != Insertion::kOk) {
    abandon();
    return bounded && status == Insertion::kRefused
               ? prepare_bounded_edge_star(a, b, position, source, construction, cell_limit)
               : status;
  }
  return admit_cavity_protection(position);

} catch (const std::bad_alloc&) {
  abandon();
  return Insertion::kCapacity;
}

Insertion TetMesh::prepare(const double* p, int32_t origin, int32_t split_a, int32_t split_b,
                           std::size_t cell_limit, const SourceWitness* construction) try {
  const MemoryScope memory_scope(memory_owner_);
  pending_edge_star_ = false;
  pending_ = static_cast<int32_t>(vertex_count());
  pending_source_ = construction == nullptr ? SourceWitness{} : *construction;
  pending_source_authored_ = construction != nullptr;
  std::copy_n(p, 3, pending_point_);
  split_a_ = split_a;
  split_b_ = split_b;
  preparation_cell_limit_ = cell_limit;
  if (coincident(origin, p)) {
    return Insertion::kDuplicate;
  }
  Insertion status = grow(origin);
  if (status == Insertion::kOk) {
    status = shrink(origin);
  }
  if (status == Insertion::kOk) {
    status = collect();
  }
  return status;
} catch (const std::bad_alloc&) {
  pending_ = kNoPending;
  return Insertion::kCapacity;
}

Insertion TetMesh::commit(InsertKind kind, double size) try {
  if (pending_edge_star_) {
    pending_edge_star_ = false;
    int32_t inserted = -1;
    return split_edge_star(split_a_, split_b_, pending_point_, size,
                           segment_source(split_a_, split_b_), true, &pending_source_, inserted);
  }
  const MemoryScope memory_scope(memory_owner_);
  EditTransaction transaction(*this, crossed_);
  if (vertex_count() >= max_vertices_) {
    return Insertion::kCapacity;
  }
  const int32_t p = pending_;
  SourceWitness witness;
  const Insertion admitted = pending_witness(kind, witness);
  if (admitted != Insertion::kOk) {
    return admitted;
  }
  pending_source_ = witness;
  NativeUnorderedMap<std::uint64_t, int32_t> staged_segments{
      NativeAllocator<std::pair<const std::uint64_t, int32_t>>(memory_owner_)};
  if (kind == InsertKind::kSubsegment) {
    const int32_t source = segment_source(split_a_, split_b_);
    if (source < 0) {
      return Insertion::kRefused;
    }
    staged_segments.emplace(topology::undirected_edge(split_a_, p), source);
    staged_segments.emplace(topology::undirected_edge(p, split_b_), source);
    segments_.reserve(segments_.size() + 2);
  }
  // Every buffer the vertex needs is reserved before anything is written.
  reserve_for(points_, points_.size() + 3);
  reserve_for(dimension_, dimension_.size() + 1);
  reserve_for(witnesses_, witnesses_.size() + 1);
  reserve_for(protection_, protection_.size() + 1);
  reserve_for(sizes_, sizes_.size() + 1);
  reserve_for(vertex_tet_, vertex_tet_.size() + 1);
  reserve_for(vertex_stamp_, vertex_stamp_.size() + 1);
  reserve_for(slot_reason_, complex_.tets.size() + boundary_.size());
  edit_.begin(EditLimits{std::numeric_limits<std::size_t>::max(),
                         std::numeric_limits<std::size_t>::max(), max_tetrahedra_});
  for (int32_t t : cavity_) {
    if (edit_.remove(t) != EditStatus::kOk) {
      return Insertion::kInternal;
    }
  }
  transaction.retire_constraints();
  EditStatus status = EditStatus::kOk;
  for (const BoundaryFacet& face : boundary_) {
    Tetrahedron cone = tet(face.tet);
    cone.v[face.slot] = p;
    int32_t constraints[4];
    for (int j = 0; j < 4; ++j) {
      constraints[j] = j == face.slot ? complex_.constraint(face.tet, face.slot)
                                      : new_face_source(cone.v, face.slot, j);
    }
    status = edit_.add(cone.v, constraints, complex_.region(face.tet));
    if (status != EditStatus::kOk) {
      break;
    }
  }
  if (status == EditStatus::kOk) {
    status = edit_.validate([&](const int32_t* v) { return positive(v); });
  }
  if (status == EditStatus::kOk) {
    status = edit_.commit();
  }
  if (status != EditStatus::kOk) {
    edit_.rollback();
    if (status == EditStatus::kInternal) {
      broken_ = true;
      return Insertion::kInternal;
    }
    return status == EditStatus::kCellLimit || status == EditStatus::kSlotLimit ||
                   status == EditStatus::kBufferLimit
               ? Insertion::kCapacity
               : Insertion::kRefused;
  }
  transaction.release();
  points_.insert(points_.end(), pending_point_, pending_point_ + 3);
  const int8_t dim = kind == InsertKind::kSubsegment                        ? 1
                     : (kind == InsertKind::kSubfacet || !crossed_.empty()) ? 2
                                                                            : 3;
  dimension_.push_back(dim);
  witnesses_.push_back(witness);
  protection_.push_back(0.0);
  sizes_.push_back(size);
  vertex_tet_.push_back(-1);
  vertex_stamp_.push_back(0U);
  pending_ = kNoPending;
  if (kind == InsertKind::kSubsegment) {
    segments_.erase(topology::undirected_edge(split_a_, split_b_));
    segments_.insert(staged_segments.extract(topology::undirected_edge(split_a_, p)));
    segments_.insert(staged_segments.extract(topology::undirected_edge(p, split_b_)));
  }
  largest_cavity_ = std::max(largest_cavity_, cavity_.size());
  after_commit();
  return Insertion::kOk;
} catch (const std::bad_alloc&) {
  return Insertion::kCapacity;
}

Insertion TetMesh::split_edge(int32_t a, int32_t b, const double* position, double size,
                              int32_t& inserted) {
  return split_edge_relocate(a, b, position, position, size, inserted);
}

Insertion TetMesh::split_edge_relocate(int32_t a, int32_t b, const double* split_position,
                                       const double* position, double size, int32_t& inserted) {
  inserted = -1;
  int32_t source = -1;
  bool on_edge = false;
  if (!std::isfinite(size) || size < 0.0 ||
      !split_positions_admitted(a, b, split_position, position, source, on_edge)) {
    return Insertion::kRefused;
  }
  return split_edge_star(a, b, position, size, source, on_edge, nullptr, inserted);
}

Insertion TetMesh::insert_edge_star_vertex(int32_t a, int32_t b, const double* position,
                                           double size, int32_t& inserted) {
  inserted = -1;
  if (!live_vertex(a) || !live_vertex(b) || a == b || segment_source(a, b) >= 0 ||
      !std::isfinite(size) || size < 0.0 || !geometry::finite3(position) ||
      !coordinate_in_domain(position[0]) || !coordinate_in_domain(position[1]) ||
      !coordinate_in_domain(position[2])) {
    return Insertion::kRefused;
  }
  return split_edge_star(a, b, position, size, -1, false, nullptr, inserted);
}

Insertion TetMesh::split_edge_bounded(int32_t a, int32_t b, const double* position,
                                      const SourceWitness& witness, double size,
                                      int32_t& inserted) {
  inserted = -1;
  if (!live_vertex(a) || !live_vertex(b) || a == b || !std::isfinite(size) || size < 0.0 ||
      witness.stratum == SourceStratum::kNone || witness.entity < 0 ||
      static_cast<std::size_t>(witness.entity) >= (witness.stratum == SourceStratum::kSegment
                                                       ? original_segments_.size()
                                                       : original_faces_.size()) ||
      !geometry::finite3(position)) {
    return Insertion::kRefused;
  }
  const double* corners[3];
  row_corners(witness.stratum, static_cast<std::size_t>(witness.entity), corners);
  const double deviation = source_deviation(corners, witness.stratum, witness.parameters, position);
  const int32_t segment = segment_source(a, b);
  const auto& identity = witness.stratum == SourceStratum::kSegment
                             ? original_segments_[static_cast<std::size_t>(witness.entity)][2]
                             : original_faces_[static_cast<std::size_t>(witness.entity)][3];
  if (deviation < 0.0 || deviation != witness.deviation ||
      (witness.stratum == SourceStratum::kSegment) != (segment >= 0) ||
      (segment >= 0 && identity != segment) ||
      (deviation > 0.0 && !rounded_carrier(witness.stratum, witness.entity, witness, position))) {
    return Insertion::kRefused;
  }
  if (deviation > row_tolerance(witness.stratum, witness.entity)) {
    return Insertion::kDeviation;
  }
  NativeVector<int32_t> star;
  NativeVector<int32_t> ring;
  if (!edge_ring(a, b, star, ring)) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  const Insertion admitted = admit_split_facets(a, b, witness, star);
  if (admitted != Insertion::kOk) return admitted;
  for (int32_t seed : star) {
    if (!finite(seed)) continue;
    const int32_t proposal = static_cast<int32_t>(vertex_count());
    Insertion result =
        prepare(position, seed, a, b, std::numeric_limits<std::size_t>::max(), &witness);
    if (result == Insertion::kOk) {
      result = commit(segment >= 0 ? InsertKind::kSubsegment : InsertKind::kSubfacet, size);
    }
    if (result == Insertion::kOk) {
      inserted = proposal;
      return result;
    }
    abandon();
    if (result != Insertion::kRefused) return result;
    break;
  }
  return split_edge_star(a, b, position, size, segment, true, &witness, inserted);
}

Insertion TetMesh::admit_edge_construction(int32_t a, int32_t b, const double* witness,
                                           const double* position, double source_fraction,
                                           int32_t& source, bool& bounded,
                                           SourceWitness& construction) {
  bool on_edge = false;
  if (!split_positions_admitted(a, b, witness, position, source, on_edge)) {
    if (!(source_fraction > 0.0) || !std::equal(witness, witness + 3, position)) {
      return Insertion::kRefused;
    }
    double carrier[3];
    const Insertion status = construct_source_split(a, b, source_fraction, carrier, construction);
    if (status != Insertion::kOk) {
      return status;
    }
    if (!std::equal(carrier, carrier + 3, witness)) {
      return Insertion::kRefused;
    }
    bounded = true;
    source = segment_source(a, b);
    on_edge = true;
  }
  return Insertion::kOk;
}

Insertion TetMesh::select_edge_insertion_seed(int32_t a, int32_t b, const double* position,
                                              int32_t source, bool bounded,
                                              std::span<const int32_t> star, InsertKind& kind,
                                              int32_t& seed) {
  seed = -1;
  kind = source >= 0 ? InsertKind::kSubsegment : InsertKind::kInterior;
  for (int32_t t : star) {
    if (is_ghost(tet(t))) {
      continue;
    }
    if (seed < 0) {
      seed = t;
    }
    for (int k = 0; k < 4; ++k) {
      if (tet(t).v[k] != a && tet(t).v[k] != b && complex_.constraint(t, k) != kNoConstraint) {
        if (!spend(TetWorkCost::kFacet)) {
          return Insertion::kCapacity;
        }
        if ((!bounded && orient_with(t, k, position) != 0) || policy_ == BoundaryPolicy::kFixed) {
          return Insertion::kRefused;
        }
        if (kind == InsertKind::kInterior) {
          kind = InsertKind::kSubfacet;
        }
      }
    }
  }
  if (seed < 0 || (kind == InsertKind::kSubsegment && policy_ == BoundaryPolicy::kFixed)) {
    return Insertion::kRefused;
  }
  return Insertion::kOk;
}

Insertion TetMesh::admit_cavity_protection(const double* position) {
  for (int32_t t : cavity_) {
    if (!spend(TetWorkCost::kFacet)) {
      abandon();
      return Insertion::kCapacity;
    }
    for (int32_t vertex : tet(t).v) {
      if (vertex >= 0 && protection(vertex) > 0.0 &&
          geometry::squared_distance(point(vertex), position) <
              protection(vertex) * protection(vertex)) {
        abandon();
        return Insertion::kRefused;
      }
    }
  }
  return Insertion::kOk;
}

Insertion TetMesh::edge_star_preparation(int32_t a, int32_t b, const double* position,
                                         int32_t source, bool on_edge, const SourceWitness* bounded,
                                         NativeVector<int32_t>& star,
                                         NativeVector<Crossed>& retired, bool& constrained,
                                         SourceWitness& witness) {
  NativeVector<int32_t> ring;
  if (!edge_ring(a, b, star, ring)) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  constrained = source >= 0;
  for (int32_t t : star) {
    for (int32_t v : tet(t).v) {
      if (v >= 0 && protection(v) > 0.0 &&
          geometry::squared_distance(point(v), position) < protection(v) * protection(v)) {
        return Insertion::kRefused;
      }
    }
    for (int k = 0; k < 4; ++k) {
      if (tet(t).v[k] != a && tet(t).v[k] != b && complex_.constraint(t, k) != kNoConstraint) {
        constrained = true;
        if (t < tet(t).n[k]) {
          retired.push_back({t, k, complex_.constraint(t, k)});
        }
      }
    }
  }
  if (constrained && policy_ == BoundaryPolicy::kFixed) {
    return Insertion::kRefused;
  }
  if (!on_edge && !split_surface_preserved(a, b, star, position)) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  if (vertex_count() >= max_vertices_ ||
      !spend(static_cast<int64_t>(TetWorkCost::kCell * star.size()))) {
    return Insertion::kCapacity;
  }
  if (bounded != nullptr) {
    witness = *bounded;
    return Insertion::kOk;
  }
  return star_witness(a, b, star, position, source, constrained, witness);
}

Insertion TetMesh::prepare_bounded_edge_star(int32_t a, int32_t b, const double* position,
                                             int32_t source, const SourceWitness& bounded,
                                             std::size_t cell_limit) {
  NativeVector<int32_t> star;
  NativeVector<Crossed> retired;
  bool constrained = false;
  SourceWitness witness;
  const Insertion admitted = edge_star_preparation(a, b, position, source, true, &bounded, star,
                                                   retired, constrained, witness);
  if (admitted != Insertion::kOk) {
    return admitted;
  }
  std::size_t finite_cells = 0;
  for (int32_t t : star) {
    finite_cells += finite(t) ? 1 : 0;
  }
  if (finite_cells > cell_limit / 3) {
    return Insertion::kCapacity;
  }
  boundary_.clear();
  for (int32_t t : star) {
    const Tetrahedron& old = tet(t);
    for (int endpoint : {a, b}) {
      const int slot = vertex_slot(old, endpoint);
      const int32_t neighbor = old.n[slot];
      boundary_.push_back({t, slot, neighbor, neighbor_slot(tet(neighbor), t)});
    }
  }
  cavity_ = std::move(star);
  crossed_ = std::move(retired);
  pending_ = static_cast<int32_t>(vertex_count());
  pending_source_ = witness;
  pending_source_authored_ = true;
  split_a_ = a;
  split_b_ = b;
  std::copy_n(position, 3, pending_point_);
  pending_edge_star_ = true;
  return Insertion::kOk;
}

Insertion TetMesh::split_edge_star(int32_t a, int32_t b, const double* position, double size,
                                   int32_t source, bool on_edge, const SourceWitness* bounded,
                                   int32_t& inserted) try {
  const MemoryScope memory_scope(memory_owner_);
  NativeVector<int32_t> star;
  NativeVector<Crossed> retired;
  bool constrained = false;
  SourceWitness witness;
  const Insertion admitted = edge_star_preparation(a, b, position, source, on_edge, bounded, star,
                                                   retired, constrained, witness);
  if (admitted != Insertion::kOk) {
    return admitted;
  }
  EditTransaction transaction(*this, retired);
  NativeUnorderedMap<std::uint64_t, int32_t> staged_segments{
      NativeAllocator<std::pair<const std::uint64_t, int32_t>>(memory_owner_)};
  const int32_t p = static_cast<int32_t>(vertex_count());
  if (source >= 0) {
    staged_segments.emplace(topology::undirected_edge(a, p), source);
    staged_segments.emplace(topology::undirected_edge(p, b), source);
    segments_.reserve(segments_.size() + 2);
  }
  reserve_for(points_, points_.size() + 3);
  reserve_for(dimension_, dimension_.size() + 1);
  reserve_for(witnesses_, witnesses_.size() + 1);
  reserve_for(protection_, protection_.size() + 1);
  reserve_for(sizes_, sizes_.size() + 1);
  reserve_for(vertex_tet_, vertex_tet_.size() + 1);
  reserve_for(vertex_stamp_, vertex_stamp_.size() + 1);
  reserve_for(slot_reason_, complex_.tets.size() + 2 * star.size());
  pending_ = p;
  pending_source_ = witness;
  std::copy_n(position, 3, pending_point_);
  edit_.begin(EditLimits{star.size(), 2 * star.size(), max_tetrahedra_});
  EditStatus status = EditStatus::kOk;
  for (int32_t t : star) {
    status = edit_.remove(t);
    if (status != EditStatus::kOk) {
      break;
    }
    const Tetrahedron& old = tet(t);
    for (int endpoint : {a, b}) {
      int32_t child[4] = {old.v[0], old.v[1], old.v[2], old.v[3]};
      const int slot = vertex_slot(old, endpoint);
      child[slot] = p;
      int32_t marks[4];
      for (int k = 0; k < 4; ++k) {
        marks[k] = old.v[k] == (endpoint == a ? b : a) ? kNoConstraint : complex_.constraint(t, k);
      }
      status = edit_.add(child, marks, complex_.region(t));
      if (status != EditStatus::kOk) {
        break;
      }
    }
    if (status != EditStatus::kOk) {
      break;
    }
  }
  transaction.retire_constraints();
  if (status == EditStatus::kOk) {
    status = edit_.validate([&](const int32_t* v) { return positive(v); });
  }
  if (status == EditStatus::kOk) {
    status = edit_.commit();
  }
  if (status != EditStatus::kOk) {
    edit_.rollback();
    broken_ = broken_ || status == EditStatus::kInternal;
    return status == EditStatus::kCellLimit || status == EditStatus::kSlotLimit ||
                   status == EditStatus::kBufferLimit
               ? Insertion::kCapacity
           : status == EditStatus::kInternal ? Insertion::kInternal
                                             : Insertion::kRefused;
  }
  transaction.release();
  points_.insert(points_.end(), pending_point_, pending_point_ + 3);
  dimension_.push_back(source >= 0 ? 1 : constrained ? 2 : 3);
  witnesses_.push_back(witness);
  protection_.push_back(0.0);
  sizes_.push_back(size);
  vertex_tet_.push_back(-1);
  vertex_stamp_.push_back(0U);
  if (source >= 0) {
    segments_.erase(topology::undirected_edge(a, b));
    segments_.insert(staged_segments.extract(topology::undirected_edge(a, p)));
    segments_.insert(staged_segments.extract(topology::undirected_edge(p, b)));
  }
  pending_ = kNoPending;
  largest_cavity_ = std::max(largest_cavity_, star.size());
  after_commit();
  inserted = p;
  return Insertion::kOk;
} catch (const std::bad_alloc&) {
  pending_ = kNoPending;
  return Insertion::kCapacity;
}

int32_t TetMesh::star_apex(int32_t t, int k, int32_t a, int32_t b) const {
  for (int r = 0; r < 4; ++r) {
    const int32_t v = tet(t).v[r];
    if (r != k && v != a && v != b) {
      return v;
    }
  }
  return -1;
}

bool TetMesh::split_surface_preserved(int32_t a, int32_t b, std::span<const int32_t> star,
                                      const double* position) {
  struct Plane {
    std::array<int32_t, 3> vertices;
    int32_t source;
    int axis;
    int sign;
  };
  struct Proof {
    Expansion area_delta;
    int64_t edge_delta = 0;
  };
  using Stratum = std::array<int32_t, 3>;  // represented plane, oriented regions
  NativeVector<Plane> planes;
  NativeMap<Stratum, Proof> patches;
  const auto area = [](const double* const* p, int axis) {
    const int x = (axis + 1) % 3, y = (axis + 2) % 3;
    const double u[2] = {p[0][x], p[0][y]};
    const double v[2] = {p[1][x], p[1][y]};
    const double w[2] = {p[2][x], p[2][y]};
    return orient2d_exact(u, v, w);
  };
  for (int32_t t : star) {
    if (!finite(t)) {
      continue;
    }
    for (int slot = 0; slot < 4; ++slot) {
      if (!spend(TetWorkCost::kCell)) {
        return false;
      }
      const int32_t other = tet(t).n[slot];
      const int32_t source = complex_.constraint(t, slot);
      const int32_t mine = complex_.region(t), theirs = complex_.region(other);
      if (tet(t).v[slot] == a || tet(t).v[slot] == b || (finite(other) && other < t) ||
          (source == kNoConstraint && finite(other) && mine == theirs)) {
        continue;
      }
      int32_t f[3];
      oriented_facet(tet(t).v, slot, f);
      const double* p[3] = {point(f[0]), point(f[1]), point(f[2])};
      if (orient3d(p[0], p[1], p[2], position) != 0) {
        return false;
      }
      std::size_t plane = 0;
      for (; plane < planes.size(); ++plane) {
        if (!spend(TetWorkCost::kFacet)) {
          return false;
        }
        const Plane& represented = planes[plane];
        const auto& v = represented.vertices;
        if (represented.source == source &&
            orient3d(point(v[0]), point(v[1]), point(v[2]), p[0]) == 0 &&
            orient3d(point(v[0]), point(v[1]), point(v[2]), p[1]) == 0 &&
            orient3d(point(v[0]), point(v[1]), point(v[2]), p[2]) == 0) {
          break;
        }
      }
      if (plane == planes.size()) {
        int axis = 0, sign = 0;
        for (; axis < 3; ++axis) {
          sign = area(p, axis).sign();
          if (sign != 0) {
            break;
          }
        }
        if (sign == 0) {
          return false;
        }
        planes.push_back({{f[0], f[1], f[2]}, source, axis, sign});
      }
      const Plane& represented = planes[plane];
      const Expansion before = area(p, represented.axis);
      const int sign = before.sign();
      if (sign == 0) {
        return false;
      }
      const int64_t orientation = sign == represented.sign ? 1 : -1;
      const Stratum stratum = orientation > 0 ? Stratum{static_cast<int32_t>(plane), mine, theirs}
                                              : Stratum{static_cast<int32_t>(plane), theirs, mine};
      Proof& proof = patches[stratum];
      Expansion delta = before;
      for (int endpoint : {a, b}) {
        const double* child[3] = {p[0], p[1], p[2]};
        for (int i = 0; i < 3; ++i) {
          if (f[i] == endpoint) {
            child[i] = position;
          }
        }
        const Expansion after = area(child, represented.axis);
        if (after.sign() != sign) {
          return false;
        }
        delta = delta - after;
      }
      proof.area_delta = orientation > 0 ? proof.area_delta + delta : proof.area_delta - delta;
      for (int i = 0; i < 3; ++i) {
        if (f[i] == a && f[(i + 1) % 3] == b) {
          proof.edge_delta += orientation;
        } else if (f[i] == b && f[(i + 1) % 3] == a) {
          proof.edge_delta -= orientation;
        }
      }
    }
  }
  for (const auto& [stratum, proof] : patches) {
    if (proof.edge_delta != 0 || !proof.area_delta.is_zero()) {
      return false;
    }
  }
  // Child marks and regions inherit each old side unchanged. CavityEdit
  // validates the identical outer shell and all reciprocal finite/ghost
  // pairings; no source identity or triangle ancestry is synthesized here.
  return true;
}

bool TetMesh::coincident(int32_t t, const double* p) const {
  for (int32_t v : tet(t).v) {
    if (v >= 0) {
      const double* q = point(v);
      if (q[0] == p[0] && q[1] == p[1] && q[2] == p[2]) {
        return true;
      }
    }
  }
  return false;
}

bool TetMesh::conflicts(int32_t t, const double* p) const {
  const MemoryScope memory_scope(memory_owner_);
  const int32_t* v = tet(t).v;
  bool exact_source = p == pending_point_ && pending_source_.deviation > 0.0;
  for (int k = 0; k < 4; ++k) exact_source = exact_source || witness(v[k]).deviation > 0.0;
  if (!exact_source) return insphere(point(v[0]), point(v[1]), point(v[2]), point(v[3]), p) > 0;
  Expansion coordinates[5][3];
  for (int k = 0; k < 4; ++k) {
    if (!source_position(v[k], coordinates[k])) return false;
  }
  if (p == pending_point_) {
    if (!source_position(pending_, coordinates[4])) return false;
  } else {
    for (int axis = 0; axis < 3; ++axis) coordinates[4][axis] = Expansion(p[axis]);
  }
  return insphere_expansion(coordinates[0], coordinates[1], coordinates[2], coordinates[3],
                            coordinates[4]) > 0;
}

bool TetMesh::contains(int32_t t, const double* p) const {
  const MemoryScope memory_scope(memory_owner_);
  for (int k = 0; k < 4; ++k) {
    if (orient_with(t, k, p) < 0) {
      return false;
    }
  }
  return true;
}

Insertion TetMesh::grow(int32_t origin) {
  const MemoryScope memory_scope(memory_owner_);
  const double* p = pending_point_;
  const bool crossable = policy_ == BoundaryPolicy::kConforming;
  next_stamp();
  cavity_.clear();
  stack_.clear();
  std::size_t finite_count = is_ghost(tet(origin)) ? 0 : 1;
  if (finite_count > preparation_cell_limit_ || !native_execution_cavity(1)) {
    return Insertion::kCapacity;
  }
  mark(origin);
  cavity_.push_back(origin);
  stack_.push_back(origin);
  while (!stack_.empty()) {
    const int32_t t = stack_.back();
    stack_.pop_back();
    if (is_ghost(tet(t))) {
      continue;
    }
    if (!spend(TetWorkCost::kFacet)) {
      return Insertion::kCapacity;
    }
    for (int k = 0; k < 4; ++k) {
      const int32_t u = tet(t).n[k];
      if (marked(u)) {
        continue;
      }
      bool take = false;
      if (complex_.constraint(t, k) != kNoConstraint) {
        if (crossable && orient_with(t, k, p) == 0 && pending_source_authored_) {
          int32_t face[3];
          oriented_facet(tet(t).v, k, face);
          if (!facet_witness_admitted(face, complex_.constraint(t, k), pending_source_)) {
            continue;
          }
        }
        take = crossable && orient_with(t, k, p) == 0 && (is_ghost(tet(u)) || conflicts(u, p));
      } else {
        take = !is_ghost(tet(u)) && conflicts(u, p);
      }
      if (take) {
        if (!is_ghost(tet(u)) && ++finite_count > preparation_cell_limit_) {
          return Insertion::kCapacity;
        }
        if (!native_execution_cavity(cavity_.size() + 1)) return Insertion::kCapacity;
        mark(u);
        cavity_.push_back(u);
        stack_.push_back(u);
      }
    }
  }
  return Insertion::kOk;
}

Insertion TetMesh::shrink(int32_t origin) {
  const MemoryScope memory_scope(memory_owner_);
  const double* p = pending_point_;
  bool changed = true;
  while (changed) {
    changed = false;
    if (!spend(static_cast<int64_t>(cavity_.size()))) {
      return Insertion::kCapacity;
    }
    for (int32_t t : cavity_) {
      if (!marked(t)) {
        continue;
      }
      const Tetrahedron& current = tet(t);
      bool drop = false;
      if (is_ghost(current)) {
        drop = !marked(current.n[vertex_slot(current, kGhostVertex)]);
      } else {
        for (int k = 0; k < 4 && !drop; ++k) {
          drop = !marked(current.n[k]) && orient_with(t, k, p) <= 0;
        }
      }
      if (drop) {
        if (t == origin || (!is_ghost(current) && contains(t, p))) {
          return Insertion::kRefused;
        }
        unmark(t);
        changed = true;
      }
    }
  }
  std::size_t kept = 0;
  for (int32_t t : cavity_) {
    if (marked(t)) {
      cavity_[kept++] = t;
    }
  }
  cavity_.resize(kept);
  return Insertion::kOk;
}

Insertion TetMesh::collect() {
  const MemoryScope memory_scope(memory_owner_);
  const double* p = pending_point_;
  boundary_.clear();
  crossed_.clear();
  std::size_t finite_removed = 0;
  std::size_t finite_proposed = 0;
  for (int32_t t : cavity_) {
    const Tetrahedron& current = tet(t);
    if (!is_ghost(current)) {
      ++finite_removed;
    }
    for (int k = 0; k < 4; ++k) {
      const int32_t u = current.n[k];
      if (!marked(u)) {
        boundary_.push_back({t, k, u, neighbor_slot(tet(u), t)});
        Tetrahedron cone = current;
        cone.v[k] = pending_;
        if (!is_ghost(cone) && ++finite_proposed > preparation_cell_limit_) {
          return Insertion::kCapacity;
        }
      } else if (t < u && complex_.constraint(t, k) != kNoConstraint) {
        crossed_.push_back({t, k, complex_.constraint(t, k)});
      }
    }
  }
  if (finite_removed > preparation_cell_limit_ ||
      finite_proposed > preparation_cell_limit_ - finite_removed) {
    return Insertion::kCapacity;
  }
  // Every cavity vertex and segment must survive on the boundary.
  ++vertex_mark_;
  boundary_edges_.clear();
  for (const BoundaryFacet& face : boundary_) {
    const Tetrahedron& current = tet(face.tet);
    int32_t f[3];
    oriented_facet(current.v, face.slot, f);
    for (int r = 0; r < 3; ++r) {
      if (f[r] >= 0) {
        vertex_stamp_[static_cast<std::size_t>(f[r])] = vertex_mark_;
        if (f[(r + 1) % 3] >= 0) {
          boundary_edges_.push_back(topology::undirected_edge(f[r], f[(r + 1) % 3]));
        }
      }
    }
  }
  std::sort(boundary_edges_.begin(), boundary_edges_.end());
  const std::uint64_t split = split_a_ >= 0 ? topology::undirected_edge(split_a_, split_b_) : 0;
  for (int32_t t : cavity_) {
    const int32_t* v = tet(t).v;
    for (int a = 0; a < 4; ++a) {
      if (v[a] >= 0 && vertex_stamp_[static_cast<std::size_t>(v[a])] != vertex_mark_) {
        return Insertion::kRefused;
      }
      for (int b = a + 1; b < 4; ++b) {
        if (v[a] < 0 || v[b] < 0) {
          continue;
        }
        const std::uint64_t key = topology::undirected_edge(v[a], v[b]);
        if ((split_a_ < 0 || key != split) && segments_.count(key) != 0 &&
            !std::binary_search(boundary_edges_.begin(), boundary_edges_.end(), key)) {
          return Insertion::kRefused;
        }
      }
    }
  }
  // Each edge of a crossed face not through p bounds the split region of
  // its source once; the face (p, edge) inherits that source.
  new_faces_.clear();
  for (const Crossed& face : crossed_) {
    int32_t f[3];
    oriented_facet(tet(face.tet).v, face.slot, f);
    for (int r = 0; r < 3; ++r) {
      const int32_t a = f[r];
      const int32_t b = f[(r + 1) % 3];
      if (!geometry::collinear(point(a), point(b), p)) {
        new_faces_.push_back({topology::undirected_edge(a, b), face.source});
      }
    }
  }
  std::sort(new_faces_.begin(), new_faces_.end());
  std::size_t kept = 0;
  for (std::size_t r = 0; r < new_faces_.size();) {
    std::size_t end = r + 1;
    while (end < new_faces_.size() && new_faces_[end].first == new_faces_[r].first) {
      ++end;
    }
    if (end - r == 1) {
      new_faces_[kept++] = new_faces_[r];
    } else if (end - r > 2 || new_faces_[r].second != new_faces_[r + 1].second) {
      return Insertion::kRefused;
    }
    r = end;
  }
  new_faces_.resize(kept);
  return Insertion::kOk;
}

int32_t TetMesh::new_face_source(const int32_t* v, int apex, int j) const {
  int32_t edge[2];
  int count = 0;
  for (int r = 0; r < 4; ++r) {
    if (r != apex && r != j) {
      edge[count++] = v[r];
    }
  }
  if (edge[0] < 0 || edge[1] < 0) {
    return kNoConstraint;
  }
  const std::pair<std::uint64_t, int32_t> probe{topology::undirected_edge(edge[0], edge[1]),
                                                std::numeric_limits<int32_t>::min()};
  const auto found = std::lower_bound(new_faces_.begin(), new_faces_.end(), probe);
  return found != new_faces_.end() && found->first == probe.first ? found->second : kNoConstraint;
}

}  // namespace phx::mc
