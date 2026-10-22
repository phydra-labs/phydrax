//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "tet_mesh.hpp"

namespace phx::mc {

EditStatus TetMesh::collapse_boundary_edge(int32_t remove, int32_t keep,
                                           std::span<const int32_t> star,
                                           std::span<const int32_t> keep_star) {
  const MemoryScope memory_scope(memory_owner_);
  cancel_prepared_edit();
  if (policy_ != BoundaryPolicy::kConforming || !live_vertex(remove) || !live_vertex(keep) ||
      remove == keep || dimension(remove) < 1 || dimension(remove) > 2 ||
      dimension(keep) > dimension(remove) || protection(remove) > 0.0) {
    return EditStatus::kConstraintViolation;
  }
  if (!spend(static_cast<int64_t>(TetWorkCost::kCell * (star.size() + keep_star.size())))) {
    return EditStatus::kBufferLimit;
  }
  BoundaryCurve curve;
  EditStatus status = collect_boundary_curve(remove, keep, star, curve);
  if (status != EditStatus::kOk) return status;
  NativeUnorderedMap<std::uint64_t, int32_t> staged_segments{
      NativeAllocator<std::pair<const std::uint64_t, int32_t>>(memory_owner_)};
  status = admit_boundary_curve(remove, keep, curve, staged_segments);
  if (status != EditStatus::kOk) return status;
  // Local proof keys never replace persistent source ancestry identities.
  BoundaryCollapseProof proof;
  status = collect_removed_boundary(remove, keep, star, proof);
  if (status != EditStatus::kOk) return status;
  status = collect_kept_boundary(keep, keep_star, proof);
  if (status != EditStatus::kOk) return status;
  for (const auto& [stratum, surface] : proof.surfaces) {
    status = certify_boundary_surface(surface);
    if (status != EditStatus::kOk) return status;
  }
  EditTransaction transaction(*this, proof.retired);
  status = stage_boundary_collapse(remove, keep, star, curve.other, proof);
  if (status != EditStatus::kOk) return status;
  if (!spend(static_cast<int64_t>(TetWorkCost::kCellValidation * star.size()))) {
    return EditStatus::kBufferLimit;
  }
  transaction.retire_constraints();
  status = edit_.validate([&](const int32_t* v) { return positive(v); });
  if (status == EditStatus::kOk) {
    status = edit_.commit();
  }
  if (status != EditStatus::kOk) {
    broken_ = broken_ || status == EditStatus::kInternal;
    return status;
  }
  transaction.release();
  if (curve.other >= 0) {
    segments_.erase(topology::undirected_edge(remove, keep));
    segments_.erase(topology::undirected_edge(remove, curve.other));
    segments_.insert(staged_segments.extract(topology::undirected_edge(keep, curve.other)));
  }
  largest_cavity_ = std::max(largest_cavity_, star.size());
  after_commit();
  return EditStatus::kOk;
}

EditStatus TetMesh::collect_boundary_curve(int32_t remove, int32_t keep,
                                           std::span<const int32_t> star,
                                           BoundaryCurve& curve) {
  // Every constrained incident segment belongs to this complete vertex star,
  // including its ghost cells. Reuse the local neighbor scratch rather than
  // walking unrelated segments throughout the mesh for every collapse.
  scratch_ring_.clear();
  for (int32_t t : star) {
    for (int32_t vertex : tet(t).v) {
      if (vertex >= 0 && vertex != remove) scratch_ring_.push_back(vertex);
    }
  }
  std::sort(scratch_ring_.begin(), scratch_ring_.end());
  scratch_ring_.erase(std::unique(scratch_ring_.begin(), scratch_ring_.end()),
                      scratch_ring_.end());
  for (int32_t other : scratch_ring_) {
    if (!spend(TetWorkCost::kVisit)) return EditStatus::kBufferLimit;
    const int32_t source = segment_source(remove, other);
    if (source < 0) continue;
    if (dimension(remove) != 1) {
      return EditStatus::kConstraintViolation;
    }
    if (curve.degree > 0 && source != curve.source) {
      return EditStatus::kConstraintViolation;
    }
    curve.source = source;
    ++curve.degree;
    if (other == keep) {
      curve.keep = true;
    } else {
      curve.other = other;
    }
  }
  return EditStatus::kOk;
}

EditStatus TetMesh::admit_boundary_curve(
    int32_t remove, int32_t keep, const BoundaryCurve& curve,
    NativeUnorderedMap<std::uint64_t, int32_t>& staged_segments) {
  if (dimension(remove) == 1) {
    if (!spend(TetWorkCost::kCell)) {
      return EditStatus::kBufferLimit;
    }
    if (curve.degree != 2 || !curve.keep || curve.other < 0 ||
        segment_source(keep, curve.other) >= 0 ||
        !geometry::collinear(point(keep), point(curve.other), point(remove))) {
      return EditStatus::kConstraintViolation;
    }
    bool strict = false;
    for (int axis = 0; axis < 3; ++axis) {
      if (point(keep)[axis] != point(curve.other)[axis]) {
        strict = std::min(point(keep)[axis], point(curve.other)[axis]) < point(remove)[axis] &&
                 point(remove)[axis] < std::max(point(keep)[axis], point(curve.other)[axis]);
        break;
      }
    }
    if (!strict) {
      return EditStatus::kConstraintViolation;
    }
    staged_segments.emplace(topology::undirected_edge(keep, curve.other), curve.source);
    segments_.reserve(segments_.size() + 1);
  }
  return EditStatus::kOk;
}

Expansion TetMesh::boundary_projected_area(const int32_t* f, int axis) const {
  const int x = (axis + 1) % 3, y = (axis + 2) % 3;
  const double a[2] = {point(f[0])[x], point(f[0])[y]};
  const double b[2] = {point(f[1])[x], point(f[1])[y]};
  const double c[2] = {point(f[2])[x], point(f[2])[y]};
  return orient2d_exact(a, b, c);
}

bool TetMesh::describe_boundary_face(int32_t t, int slot, const int32_t* f,
                                     BoundaryStratum& stratum, int& axis, int& reference_sign,
                                     bool add_plane, BoundaryCollapseProof& proof) {
  auto& planes = proof.planes;
  if (!spend(TetWorkCost::kCell)) {
    return false;
  }
  const int32_t source = complex_.constraint(t, slot);
  std::size_t plane = 0;
  for (; plane < planes.size(); ++plane) {
    if (!spend(TetWorkCost::kFacet)) {
      return false;
    }
    const auto& represented = planes[plane];
    if (represented.source != source) {
      continue;
    }
    const auto& v = represented.vertices;
    if (orient3d(point(v[0]), point(v[1]), point(v[2]), point(f[0])) == 0 &&
        orient3d(point(v[0]), point(v[1]), point(v[2]), point(f[1])) == 0 &&
        orient3d(point(v[0]), point(v[1]), point(v[2]), point(f[2])) == 0) {
      break;
    }
  }
  if (plane == planes.size()) {
    if (!add_plane) {
      stratum = {-1, kNoRegion, kNoRegion};
      return true;  // unrelated keep-star patch cannot enter the intersection
    }
    reference_sign = 0;
    for (axis = 0; axis < 3; ++axis) {
      reference_sign = boundary_projected_area(f, axis).sign();
      if (reference_sign != 0) {
        break;
      }
    }
    if (reference_sign == 0) {
      return false;
    }
    planes.push_back({{f[0], f[1], f[2]}, source, axis, reference_sign});
  } else {
    axis = planes[plane].axis;
    reference_sign = planes[plane].reference_sign;
  }
  const int sign = boundary_projected_area(f, axis).sign();
  if (sign == 0) {
    return false;
  }
  const int32_t mine = complex_.region(t);
  const int32_t theirs = complex_.region(tet(t).n[slot]);
  const int32_t identity = static_cast<int32_t>(plane);
  stratum = sign == reference_sign ? BoundaryStratum{identity, mine, theirs}
                                   : BoundaryStratum{identity, theirs, mine};
  return true;
}

void TetMesh::add_boundary_link(const int32_t* f, int32_t a, int32_t b,
                                NativeSet<BoundaryLinkSimplex>& link) {
  int32_t v[2];
  int n = 0;
  for (int i = 0; i < 3; ++i) {
    if (f[i] != a && f[i] != b) {
      v[n++] = f[i];
    }
  }
  for (int mask = 0; mask < (1 << n); ++mask) {
    BoundaryLinkSimplex simplex{kDeadVertex, kDeadVertex};
    int count = 0;
    for (int i = 0; i < n; ++i) {
      if ((mask & (1 << i)) != 0) {
        simplex[static_cast<std::size_t>(count++)] = v[i];
      }
    }
    std::sort(simplex.begin(), simplex.end());
    link.insert(simplex);
  }
}

void TetMesh::add_boundary_chain(const int32_t* f, int64_t coefficient,
                                 NativeMap<std::uint64_t, int64_t>& boundary) {
  for (int i = 0; i < 3; ++i) {
    const int32_t a = f[i], b = f[(i + 1) % 3];
    if (a != b) {
      boundary[topology::undirected_edge(a, b)] += a < b ? coefficient : -coefficient;
    }
  }
}

EditStatus TetMesh::certify_boundary_chain(
    const NativeMap<std::uint64_t, int64_t>& boundary_delta) {
  struct Line {
    int32_t a, b;
    int axis;
    NativeMap<double, int64_t> events;
  };
  NativeVector<Line> lines;
  for (const auto& [edge, coefficient] : boundary_delta) {
    if (coefficient == 0) {
      continue;
    }
    const int32_t a = edge_low(edge), b = edge_high(edge);
    std::size_t line = 0;
    for (; line < lines.size(); ++line) {
      if (!spend(TetWorkCost::kCollinearityPair)) {
        return EditStatus::kBufferLimit;
      }
      if (geometry::collinear(point(lines[line].a), point(lines[line].b), point(a)) &&
          geometry::collinear(point(lines[line].a), point(lines[line].b), point(b))) {
        break;
      }
    }
    if (line == lines.size()) {
      int axis = 0;
      while (axis < 3 && point(a)[axis] == point(b)[axis]) {
        ++axis;
      }
      if (axis == 3) {
        return EditStatus::kConstraintViolation;
      }
      lines.push_back(Line{a, b, axis, {}});
    }
    const int axis = lines[line].axis;
    lines[line].events[point(a)[axis]] += coefficient;
    lines[line].events[point(b)[axis]] -= coefficient;
  }
  for (const Line& line : lines) {
    for (const auto& [coordinate, coefficient] : line.events) {
      if (coefficient != 0) {
        return EditStatus::kConstraintViolation;
      }
    }
  }
  return EditStatus::kOk;
}

EditStatus TetMesh::certify_boundary_surface(const BoundarySurfaceProof& proof) {
  if (!proof.area_delta.is_zero()) {
    return EditStatus::kConstraintViolation;
  }
  NativeSet<BoundaryLinkSimplex> common;
  std::set_intersection(proof.left.begin(), proof.left.end(), proof.right.begin(),
                        proof.right.end(), std::inserter(common, common.end()));
  if (common != proof.edge) {
    return EditStatus::kConstraintViolation;
  }
  return certify_boundary_chain(proof.boundary_delta);
}

bool TetMesh::check_boundary_side(const int32_t* f, int32_t region,
                                  BoundaryFaceProof& proof) const {
  const int sign = boundary_projected_area(f, proof.axis).sign();
  if (sign == 0) {
    return false;
  }
  const int side = sign == proof.reference_sign ? 0 : 1;
  return region == proof.stratum[static_cast<std::size_t>(side + 1)] && ++proof.sides[side] == 1;
}

EditStatus TetMesh::stage_boundary_collapse(int32_t remove, int32_t keep,
                                            std::span<const int32_t> star, int32_t curve_other,
                                            BoundaryCollapseProof& proof_state) {
  auto& faces = proof_state.faces;
  reserve_for(slot_reason_, complex_.tets.size() + star.size());
  edit_.begin(EditLimits{star.size(), star.size(), max_tetrahedra_});
  bool curve_represented = dimension(remove) != 1;
  std::size_t finite_survivors = 0;
  for (int32_t t : star) {
    EditStatus status = edit_.remove(t);
    if (status != EditStatus::kOk) {
      return status;
    }
    const Tetrahedron& old = tet(t);
    const int remove_slot = vertex_slot(old, remove);
    if (remove_slot < 0) {
      return EditStatus::kNotLive;
    }
    // The shell is unchanged. Account for a source triangle whose mapped
    // second side belongs to an untouched shell neighbor.
    int32_t shell[3];
    oriented_facet(old.v, remove_slot, shell);
    const auto shell_proof = FaceKey::of(shell[0], shell[1], shell[2]);
    const auto found_shell = faces.find(shell_proof);
    if (found_shell != faces.end()) {
      const int32_t outside = old.n[remove_slot];
      const int back = neighbor_slot(tet(outside), t);
      if (back < 0) {
        return EditStatus::kNonManifold;
      }
      oriented_facet(tet(outside).v, back, shell);
      if (!check_boundary_side(shell, complex_.region(outside), found_shell->second)) {
        return EditStatus::kConstraintViolation;
      }
    }
    if (vertex_slot(old, keep) >= 0) {
      continue;
    }
    int32_t v[4] = {old.v[0], old.v[1], old.v[2], old.v[3]};
    v[remove_slot] = keep;
    int32_t marks[4] = {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint};
    for (int slot = 0; slot < 4; ++slot) {
      int32_t f[3];
      oriented_facet(v, slot, f);
      const auto found = faces.find(FaceKey::of(f[0], f[1], f[2]));
      if (found != faces.end()) {
        if (!check_boundary_side(f, complex_.region(t), found->second)) {
          return EditStatus::kConstraintViolation;
        }
        marks[slot] = found->second.source;
      }
    }
    if (curve_other >= 0 && finite(t) && std::find(v, v + 4, curve_other) != v + 4) {
      curve_represented = true;
    }
    status = edit_.add(v, marks, complex_.region(t));
    if (status != EditStatus::kOk) {
      return status;
    }
    finite_survivors += finite(t) ? 1 : 0;
  }
  if (finite_survivors == 0 || !curve_represented) {
    return EditStatus::kConstraintViolation;
  }
  for (const auto& [key, proof] : faces) {
    if (proof.sides[0] != 1 || proof.sides[1] != 1) {
      return EditStatus::kConstraintViolation;
    }
    edit_.preserve(key.v[0], key.v[1], key.v[2], proof.source);
  }
  return EditStatus::kOk;
}

EditStatus TetMesh::collect_removed_boundary(int32_t remove, int32_t keep,
                                             std::span<const int32_t> star,
                                             BoundaryCollapseProof& proof_state) {
  auto& surfaces = proof_state.surfaces;
  auto& faces = proof_state.faces;
  auto& retired = proof_state.retired;
  for (int32_t t : star) {
    if (!finite(t)) {
      continue;
    }
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t source = complex_.constraint(t, slot);
      const int32_t other = tet(t).n[slot];
      if (tet(t).v[slot] == remove || source == kNoConstraint || (finite(other) && other < t)) {
        continue;
      }
      int32_t f[3];
      oriented_facet(tet(t).v, slot, f);
      BoundaryStratum stratum;
      int axis, reference_sign;
      if (!describe_boundary_face(t, slot, f, stratum, axis, reference_sign, true, proof_state) ||
          orient3d(point(f[0]), point(f[1]), point(f[2]), point(keep)) != 0) {
        return work_exhausted_ ? EditStatus::kBufferLimit : EditStatus::kConstraintViolation;
      }
      BoundarySurfaceProof& proof = surfaces[stratum];
      add_boundary_link(f, remove, kDeadVertex, proof.left);
      const bool edge_face = std::find(f, f + 3, keep) != f + 3;
      if (edge_face) {
        add_boundary_link(f, remove, keep, proof.edge);
      }
      const Expansion before = boundary_projected_area(f, axis);
      const int64_t orientation = before.sign() == reference_sign ? 1 : -1;
      add_boundary_chain(f, orientation, proof.boundary_delta);
      for (int32_t& v : f) {
        if (v == remove) {
          v = keep;
        }
      }
      const Expansion after = boundary_projected_area(f, axis);
      if ((!edge_face && after.sign() != before.sign()) || (edge_face && !after.is_zero())) {
        return EditStatus::kConstraintViolation;
      }
      if (!edge_face) {
        bool bounded = false;
        for (int32_t vertex : f) bounded = bounded || witness(vertex).deviation != 0.0;
        if (bounded) {
          NativeVector<int32_t> contributors;
          if (!face_ancestry(f, source, contributors)) return EditStatus::kConstraintViolation;
        }
        add_boundary_chain(f, -orientation, proof.boundary_delta);
      }
      proof.area_delta = before.sign() == reference_sign ? proof.area_delta + before - after
                                                         : proof.area_delta + after - before;
      if (!edge_face && !faces
                             .emplace(FaceKey::of(f[0], f[1], f[2]),
                                      BoundaryFaceProof{stratum, source, axis, reference_sign})
                             .second) {
        return EditStatus::kConstraintViolation;
      }
      retired.push_back({t, slot, source});
    }
  }
  if (surfaces.empty()) {
    return EditStatus::kConstraintViolation;
  }
  return EditStatus::kOk;
}

EditStatus TetMesh::collect_kept_boundary(int32_t keep, std::span<const int32_t> keep_star,
                                          BoundaryCollapseProof& proof_state) {
  auto& surfaces = proof_state.surfaces;
  for (int32_t t : keep_star) {
    if (!finite(t)) {
      continue;
    }
    for (int slot = 0; slot < 4; ++slot) {
      if (tet(t).v[slot] == keep || complex_.constraint(t, slot) == kNoConstraint ||
          (finite(tet(t).n[slot]) && tet(t).n[slot] < t)) {
        continue;
      }
      int32_t f[3];
      oriented_facet(tet(t).v, slot, f);
      BoundaryStratum stratum;
      int axis, reference_sign;
      if (!describe_boundary_face(t, slot, f, stratum, axis, reference_sign, false, proof_state)) {
        return work_exhausted_ ? EditStatus::kBufferLimit : EditStatus::kConstraintViolation;
      }
      const auto found = surfaces.find(stratum);
      if (found != surfaces.end()) {
        add_boundary_link(f, keep, kDeadVertex, found->second.right);
      }
    }
  }
  return EditStatus::kOk;
}
namespace {

// One rotation of a's cyclic order equals b.
bool same_cycle(const int32_t* a, const int32_t* b) {
  for (int shift = 0; shift < 3; ++shift) {
    if (a[0] == b[shift] && a[1] == b[(shift + 1) % 3] && a[2] == b[(shift + 2) % 3]) {
      return true;
    }
  }
  return false;
}

bool same_bits(const double* a, const double* b) {
  for (int axis = 0; axis < 3; ++axis) {
    if (std::bit_cast<std::uint64_t>(a[axis]) != std::bit_cast<std::uint64_t>(b[axis])) {
      return false;
    }
  }
  return true;
}

// Native candidate orientation belongs to the retained material-side proof.
// Delegated rows keep the legacy unordered face identity plus bitwise ordered
// coordinates: public adapters may reverse inspection rows to point outward.
// The fresh map supplies uniqueness marks without edit-staging allocation.
template <class Lookup>
EditStatus validate_curved_face_claims(TetMesh& mesh, const CurvedCollapseClaims& claims,
                                       double tolerance, std::size_t count,
                                       const Lookup& lookup) {
  if (claims.count != static_cast<int64_t>(count) || claims.count <= 0 ||
      claims.faces == nullptr || claims.coordinates == nullptr || claims.deviations == nullptr) {
    return EditStatus::kConstraintViolation;
  }
  for (int64_t i = 0; i < claims.count; ++i) {
    if (!mesh.spend(TetWorkCost::kFacet)) return EditStatus::kBufferLimit;
    const int32_t* face = claims.faces + 3 * i;
    if (!lookup(FaceKey::of(face[0], face[1], face[2]), i) ||
        !std::isfinite(claims.deviations[i]) || claims.deviations[i] < 0.0 ||
        claims.deviations[i] > tolerance) {
      return EditStatus::kConstraintViolation;
    }
    for (int k = 0; k < 3; ++k) {
      if (!mesh.live_vertex(face[k]) ||
          !same_bits(mesh.point(face[k]), claims.coordinates + 9 * i + 3 * k)) {
        return EditStatus::kConstraintViolation;
      }
    }
  }
  return EditStatus::kOk;
}

}  // namespace

EditStatus TetMesh::collapse_curved_boundary_edge(
    int32_t remove, int32_t keep, std::span<const int32_t> star,
    std::span<const int32_t> keep_star, const CurvedCollapseClaims* claims,
    CurvedCollapseInspection* inspection) {
  const MemoryScope memory_scope(memory_owner_);
  cancel_prepared_edit();
  if ((claims == nullptr) == (inspection == nullptr) ||
      policy_ != BoundaryPolicy::kConforming || !live_vertex(remove) || !live_vertex(keep) ||
      remove == keep || dimension(remove) != 2 || dimension(keep) > 2 ||
      protection(remove) > 0.0) {
    return EditStatus::kConstraintViolation;
  }
  int32_t patch = -1;
  double tolerance = 0.0;
  if (claims != nullptr) {
    patch = claims->source;
    tolerance = curved_tolerance(patch);
    const std::uint8_t* digest = curved_digest(patch);
    if (std::isnan(tolerance) || digest == nullptr || claims->digest == nullptr ||
        !std::equal(digest, digest + 32, claims->digest) ||
        claims->generation != accepted_generation_ || claims->count <= 0 ||
        claims->faces == nullptr || claims->coordinates == nullptr ||
        claims->deviations == nullptr) {
      return EditStatus::kConstraintViolation;
    }
  }
  if (!spend(static_cast<int64_t>(TetWorkCost::kCell * (star.size() + keep_star.size())))) {
    return EditStatus::kBufferLimit;
  }
  // A curved surface vertex carries no segment: feature curves never move.
  BoundaryCurve curve;
  EditStatus status = collect_boundary_curve(remove, keep, star, curve);
  if (status != EditStatus::kOk) return status;
  if (curve.degree != 0) return EditStatus::kConstraintViolation;

  struct CurvedFace {
    std::array<int32_t, 3> oriented;
    int32_t mine;
    int32_t theirs;
    int sides[2] = {0, 0};
    int64_t claim = -1;
  };
  NativeMap<FaceKey, CurvedFace> faces;
  NativeSet<BoundaryLinkSimplex> left, right, edge;
  NativeMap<std::uint64_t, int64_t> chain;
  NativeVector<Crossed> retired;
  bool stratified = false;
  int32_t stratum_mine = kNoRegion, stratum_theirs = kNoRegion;
  // Orient one curved-source face of `t` by the patch's single material
  // stratum; false for another source or stratum.
  const auto orient_face = [&](int32_t t, int slot, int32_t* f, int32_t& mine,
                               int32_t& theirs, bool establish) {
    oriented_facet(tet(t).v, slot, f);
    mine = complex_.region(t);
    theirs = complex_.region(tet(t).n[slot]);
    if (!stratified) {
      if (!establish) return false;
      stratified = true;
      stratum_mine = mine;
      stratum_theirs = theirs;
      return true;
    }
    if (mine == stratum_mine && theirs == stratum_theirs) return true;
    if (mine == stratum_theirs && theirs == stratum_mine) {
      std::swap(f[1], f[2]);
      std::swap(mine, theirs);
      return true;
    }
    return false;
  };
  for (int32_t t : star) {
    if (!finite(t)) continue;
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t source = complex_.constraint(t, slot);
      const int32_t other = tet(t).n[slot];
      if (tet(t).v[slot] == remove || source == kNoConstraint ||
          (finite(other) && other < t)) {
        continue;
      }
      if (!spend(TetWorkCost::kFacet)) return EditStatus::kBufferLimit;
      int32_t f[3], mine, theirs;
      if (patch < 0) patch = source;
      if (source != patch || std::isnan(curved_tolerance(patch)) ||
          !orient_face(t, slot, f, mine, theirs, true)) {
        return EditStatus::kConstraintViolation;
      }
      add_boundary_link(f, remove, kDeadVertex, left);
      const bool edge_face = std::find(f, f + 3, keep) != f + 3;
      if (edge_face) add_boundary_link(f, remove, keep, edge);
      add_boundary_chain(f, 1, chain);
      int32_t g[3] = {f[0], f[1], f[2]};
      for (int32_t& v : g) {
        if (v == remove) v = keep;
      }
      add_boundary_chain(g, -1, chain);
      if (!edge_face) {
        if (geometry::collinear(point(g[0]), point(g[1]), point(g[2])) ||
            !faces.emplace(FaceKey::of(g[0], g[1], g[2]),
                           CurvedFace{{g[0], g[1], g[2]}, mine, theirs}).second) {
          return EditStatus::kConstraintViolation;
        }
      }
      retired.push_back({t, slot, source});
    }
  }
  if (faces.empty()) return EditStatus::kConstraintViolation;
  for (int32_t t : keep_star) {
    if (!finite(t)) continue;
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t other = tet(t).n[slot];
      if (tet(t).v[slot] == keep || complex_.constraint(t, slot) != patch ||
          (finite(other) && other < t)) {
        continue;
      }
      if (!spend(TetWorkCost::kFacet)) return EditStatus::kBufferLimit;
      int32_t f[3], mine, theirs;
      // An unrelated stratum of the same source cannot enter the link.
      if (orient_face(t, slot, f, mine, theirs, false)) {
        add_boundary_link(f, keep, kDeadVertex, right);
      }
    }
  }
  // Surface link condition and exact preservation of the oriented patch
  // boundary chain (the old and new fans share one boundary cycle).
  NativeSet<BoundaryLinkSimplex> common;
  std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                        std::inserter(common, common.end()));
  if (common != edge) return EditStatus::kConstraintViolation;
  for (const auto& [key, coefficient] : chain) {
    if (coefficient != 0) return EditStatus::kConstraintViolation;
  }
  // Preserve fresh-refusal ordering: every host row is authenticated before
  // edit_.begin(), staging reservations or cell-validation work are admitted.
  if (claims != nullptr) {
    status = validate_curved_face_claims(*this, *claims, tolerance, faces.size(),
        [&](const FaceKey& key, int64_t claim) {
          const auto found = faces.find(key);
          if (found == faces.end() || found->second.claim >= 0) return false;
          found->second.claim = claim;
          return true;
        });
    if (status != EditStatus::kOk) return status;
  }
  const auto check_side = [](const int32_t* g, int32_t region, CurvedFace& face) {
    const int side = same_cycle(face.oriented.data(), g) ? 0 : 1;
    return region == (side == 0 ? face.mine : face.theirs) && ++face.sides[side] == 1;
  };
  EditTransaction transaction(*this, retired);
  reserve_for(slot_reason_, complex_.tets.size() + star.size());
  edit_.begin(EditLimits{star.size(), star.size(), max_tetrahedra_});
  std::size_t finite_survivors = 0;
  for (int32_t t : star) {
    status = edit_.remove(t);
    if (status != EditStatus::kOk) return status;
    const Tetrahedron& old = tet(t);
    const int remove_slot = vertex_slot(old, remove);
    if (remove_slot < 0) return EditStatus::kNotLive;
    // A new face may coincide with an unchanged shell face whose second
    // side is an untouched neighbor of the star.
    int32_t shell[3];
    oriented_facet(old.v, remove_slot, shell);
    const auto found_shell = faces.find(FaceKey::of(shell[0], shell[1], shell[2]));
    if (found_shell != faces.end()) {
      const int32_t outside = old.n[remove_slot];
      const int back = neighbor_slot(tet(outside), t);
      if (back < 0) return EditStatus::kNonManifold;
      oriented_facet(tet(outside).v, back, shell);
      if (!check_side(shell, complex_.region(outside), found_shell->second)) {
        return EditStatus::kConstraintViolation;
      }
    }
    if (vertex_slot(old, keep) >= 0) continue;
    int32_t v[4] = {old.v[0], old.v[1], old.v[2], old.v[3]};
    v[remove_slot] = keep;
    int32_t marks[4] = {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint};
    for (int slot = 0; slot < 4; ++slot) {
      int32_t f[3];
      oriented_facet(v, slot, f);
      const auto found = faces.find(FaceKey::of(f[0], f[1], f[2]));
      if (found != faces.end()) {
        if (!check_side(f, complex_.region(t), found->second)) {
          return EditStatus::kConstraintViolation;
        }
        marks[slot] = patch;
      }
    }
    status = edit_.add(v, marks, complex_.region(t));
    if (status != EditStatus::kOk) return status;
    finite_survivors += finite(t) ? 1 : 0;
  }
  if (finite_survivors == 0) return EditStatus::kConstraintViolation;
  for (const auto& [key, face] : faces) {
    if (face.sides[0] != 1 || face.sides[1] != 1) return EditStatus::kConstraintViolation;
    edit_.preserve(key.v[0], key.v[1], key.v[2], patch);
  }
  const std::size_t added = faces.size();
  if (inspection != nullptr) inspection->faces.reserve(added);
  PreparedReplacement prepared;
  prepared.binding_.operation = PreparedBinding::Operation::kCollapse;
  prepared.binding_.first = remove;
  prepared.binding_.second = keep;
  std::copy_n(point(remove), 3, prepared.binding_.endpoints.begin());
  std::copy_n(point(keep), 3, prepared.binding_.endpoints.begin() + 3);
  prepared.binding_.source = patch;
  prepared.binding_.tolerance = curved_tolerance(patch);
  std::copy_n(curved_digest(patch), 32, prepared.binding_.digest.begin());
  prepared.binding_.cavity_size = star.size();
  prepared.binding_.faces.reserve(added);
  for (const auto& [key, face] : faces) prepared.binding_.faces.push_back(face.oriented);
  if (!spend(static_cast<int64_t>(TetWorkCost::kCellValidation * star.size()))) {
    return EditStatus::kBufferLimit;
  }
  transaction.retire_constraints();
  status = edit_.validate([&](const int32_t* cell) { return positive(cell); });
  if (status != EditStatus::kOk) {
    broken_ = broken_ || status == EditStatus::kInternal;
    return status;
  }
  // Restore accepted marks before the owner can escape through the FFI.
  // Only edit_'s already-validated cavity and its one token remain staged.
  transaction.retain();
  prepared.binding_.retired = std::move(retired);
  prepared.owner_ = this;
  prepared.generation_ = accepted_generation_;
  prepared.revision_ = edit_.staging_revision();
  prepared.status_ = EditStatus::kOk;
  if (inspection != nullptr) {
    inspection->source = patch;
    inspection->generation = accepted_generation_;
    inspection->faces.assign(prepared.binding_.faces.begin(), prepared.binding_.faces.end());
    retained_edit_ = std::move(prepared);
    return EditStatus::kOk;
  }
  return commit_curved_collapse(prepared, *claims, true);
}

bool TetMesh::reuse_curved_collapse(int32_t remove, int32_t keep,
                                    bool require_improvement,
                                    const CurvedCollapseClaims& claims, bool& applied) {
  if (!retained_edit_.current() ||
      retained_edit_.binding_.operation != PreparedBinding::Operation::kCollapse ||
      retained_edit_.binding_.first != remove || retained_edit_.binding_.second != keep ||
      retained_edit_.binding_.require_improvement != require_improvement) {
    return false;
  }
  auto prepared = std::move(retained_edit_);
  applied = commit_curved_collapse(prepared, claims, false) == EditStatus::kOk;
  return true;
}

EditStatus TetMesh::commit_curved_collapse(PreparedReplacement& prepared,
                                           const CurvedCollapseClaims& claims,
                                           bool claims_validated) {
  const MemoryScope memory_scope(memory_owner_);
  const auto& binding = prepared.binding_;
  const int32_t patch = binding.source;
  const double tolerance = curved_tolerance(patch);
  const auto* digest = curved_digest(patch);
  if (!live_vertex(binding.first) || !live_vertex(binding.second) ||
      !same_bits(point(binding.first), binding.endpoints.data()) ||
      !same_bits(point(binding.second), binding.endpoints.data() + 3)) {
    return EditStatus::kConstraintViolation;
  }
  if (!prepared.current() || claims.source != patch ||
      claims.generation != accepted_generation_ || claims.digest == nullptr ||
      digest == nullptr || !std::equal(digest, digest + 32, claims.digest) ||
      !std::equal(digest, digest + 32, binding.digest.begin()) ||
      std::bit_cast<uint64_t>(tolerance) != std::bit_cast<uint64_t>(binding.tolerance) ||
      claims.count != static_cast<int64_t>(binding.faces.size()) ||
      claims.faces == nullptr || claims.coordinates == nullptr || claims.deviations == nullptr) {
    return EditStatus::kConstraintViolation;
  }
  if (!claims_validated) {
    NativeVector<uint8_t> seen(binding.faces.size(), 0);
    const EditStatus validation = validate_curved_face_claims(
        *this, claims, tolerance, binding.faces.size(),
        [&](const FaceKey& key, int64_t) {
          const auto found = std::lower_bound(binding.faces.begin(), binding.faces.end(), key,
              [](const auto& candidate, const FaceKey& value) {
                return FaceKey::of(candidate[0], candidate[1], candidate[2]) < value;
              });
          if (found == binding.faces.end() ||
              !(FaceKey::of((*found)[0], (*found)[1], (*found)[2]) == key) ||
              seen[static_cast<std::size_t>(found - binding.faces.begin())]++) return false;
          return true;
        });
    if (validation != EditStatus::kOk) return validation;
  }
  const std::size_t added = binding.faces.size();
  original_points_.reserve(original_points_.size() + 9 * added);
  original_faces_.reserve(original_faces_.size() + added);
  face_tolerance_.reserve(face_tolerance_.size() + added);
  face_rows_.reserve(face_rows_.size() + added);
  claimed_rows_.reserve(claimed_rows_.size() + added);
  curved_claims_.reserve(curved_claims_.size() + added);
  EditTransaction transaction(*this, binding.retired);
  transaction.retire_constraints();
  const EditStatus status =
      native_execution_cavity(edit_.removed().size() + edit_.proposal().size())
          ? edit_.commit() : EditStatus::kBufferLimit;
  if (status != EditStatus::kOk) {
    broken_ = broken_ || status == EditStatus::kInternal;
    return status;
  }
  prepared.owner_ = nullptr;
  transaction.release();
  // Committed faces become delegated-claim rows of their curved source; a
  // claim stays labeled with its binding, never an authored or planar row.
  for (int64_t i = 0; i < claims.count; ++i) {
    const int32_t* claimed = claims.faces + 3 * i;
    const auto base = static_cast<int32_t>(original_points_.size() / 3);
    for (int k = 0; k < 3; ++k) {
      const double* p = point(claimed[k]);
      original_points_.insert(original_points_.end(), p, p + 3);
    }
    const auto row = static_cast<int32_t>(original_faces_.size());
    original_faces_.push_back({base, base + 1, base + 2, patch});
    face_tolerance_.push_back(0.0);
    const std::pair<int32_t, int32_t> entry{patch, row};
    face_rows_.insert(std::upper_bound(face_rows_.begin(), face_rows_.end(), entry), entry);
    claimed_rows_.insert(std::upper_bound(claimed_rows_.begin(), claimed_rows_.end(), entry),
                         entry);
    curved_claims_.push_back({row, patch, claims.deviations[i], claims.generation});
  }
  release_face_ancestry();
  largest_cavity_ = std::max(largest_cavity_, binding.cavity_size);
  after_commit();
  return EditStatus::kOk;
}

}  // namespace phx::mc
