//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "tet_mesh.hpp"

namespace phx::mc {

double TetMesh::achieved_source_error_bound() const {
  double bound = 0.0;
  for (std::size_t v = 0; v < witnesses_.size(); ++v) {
    if (dimension_[v] >= 0) {
      bound = std::max(bound, witnesses_[v].deviation);
    }
  }
  return bound;
}

double TetMesh::requested_source_tolerance() const {
  double bound = 0.0;
  for (double value : face_tolerance_) bound = std::max(bound, value);
  for (double value : segment_tolerance_) bound = std::max(bound, value);
  return bound;
}

template <class Allocator>
void TetMesh::collect_faces(std::vector<std::array<int32_t, 4>, Allocator>& faces) const {
  const MemoryScope memory_scope(memory_owner_);
  faces.clear();
  for (std::size_t s = 0; s < complex_.tets.size(); ++s) {
    const auto t = static_cast<int32_t>(s);
    const Tetrahedron& current = tet(t);
    if (current.v[0] == kDeadVertex || is_ghost(current)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      const int32_t source = complex_.constraint(t, k);
      if (source == kNoConstraint) {
        continue;
      }
      const int32_t other = current.n[k];
      const bool ghost = is_ghost(tet(other));
      const int32_t mine = complex_.region(t);
      const int32_t theirs = ghost ? kNoRegion : complex_.region(other);
      // Each face once: from its finite side, from the lower region, or
      // from the lower slot within one region.
      if (!ghost && (theirs < mine || (theirs == mine && other < t))) {
        continue;
      }
      int32_t facet[3];
      oriented_facet(current.v, k, facet);
      // oriented_facet faces into t; flip for the outward normal.
      std::swap(facet[1], facet[2]);
      if (!ghost && theirs == mine) {
        const FaceKey key = FaceKey::of(facet[0], facet[1], facet[2]);
        facet[0] = key.v[0];
        facet[1] = key.v[1];
        facet[2] = key.v[2];
      }
      canonicalize_cell(facet, nullptr, 3);
      faces.push_back({facet[0], facet[1], facet[2], source});
    }
  }
  std::sort(faces.begin(), faces.end());
}

template <class Allocator>
void TetMesh::collect_cells(std::vector<std::array<int32_t, 5>, Allocator>& cells) const {
  const MemoryScope memory_scope(memory_owner_);
  cells.clear();
  for (std::size_t s = 0; s < complex_.tets.size(); ++s) {
    const Tetrahedron& current = complex_.tets[s];
    if (current.v[0] == kDeadVertex || is_ghost(current)) {
      continue;
    }
    int32_t v[4] = {current.v[0], current.v[1], current.v[2], current.v[3]};
    canonicalize_cell(v, nullptr, 4);
    cells.push_back({v[0], v[1], v[2], v[3], complex_.region(static_cast<int32_t>(s))});
  }
  std::sort(cells.begin(), cells.end());
}

template void TetMesh::collect_faces<std::allocator<std::array<int32_t, 4>>>(
    std::vector<std::array<int32_t, 4>, std::allocator<std::array<int32_t, 4>>>&) const;

template void TetMesh::collect_faces<NativeAllocator<std::array<int32_t, 4>>>(
    std::vector<std::array<int32_t, 4>, NativeAllocator<std::array<int32_t, 4>>>&) const;

template void TetMesh::collect_cells<std::allocator<std::array<int32_t, 5>>>(
    std::vector<std::array<int32_t, 5>, std::allocator<std::array<int32_t, 5>>>&) const;

template void TetMesh::collect_cells<NativeAllocator<std::array<int32_t, 5>>>(
    std::vector<std::array<int32_t, 5>, NativeAllocator<std::array<int32_t, 5>>>&) const;

}  // namespace phx::mc
