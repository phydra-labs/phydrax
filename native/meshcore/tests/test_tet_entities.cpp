// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Native packed polyhedral entity extraction and cell vertex counting.
#include <array>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

// Two cubes sharing face (1, 3, 7, 5):
// Cell 0 vertices 0..7; Cell 1 vertices 1, 8, 3, 9, 5, 10, 7, 11.
void test_two_cubes() {
  const int64_t vertex_count = 12;
  const int64_t cell_count = 2;
  const std::array<int64_t, 2> cell_face_counts{6, 6};
  const std::array<int64_t, 12> loop_lengths{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  // Outward loops for cube 0 and cube 1 (shared face reversed on cube 1).
  const std::vector<int32_t> loop_values{
      // Cube 0
      0, 2, 3, 1,
      4, 5, 7, 6,
      0, 1, 5, 4,
      2, 6, 7, 3,
      0, 4, 6, 2,
      1, 3, 7, 5,
      // Cube 1 (x in [1, 2], shared face is 1-5-7-3)
      1, 3, 9, 8,
      5, 10, 11, 7,
      1, 8, 10, 5,
      3, 7, 11, 9,
      1, 5, 7, 3,
      8, 9, 11, 10,
  };
  const int64_t loop_count = static_cast<int64_t>(loop_lengths.size());
  const int64_t entry_count = static_cast<int64_t>(loop_values.size());

  std::array<int64_t, 2> counts{};
  PHX_CHECK(phx_mc_packed_cell_vertex_counts(
                vertex_count, cell_count, cell_face_counts.data(), loop_count,
                loop_lengths.data(), entry_count, loop_values.data(), counts.data()) == PHX_MC_OK);
  PHX_CHECK(counts[0] == 8 && counts[1] == 8);

  std::vector<int32_t> loop_faces(static_cast<std::size_t>(loop_count), -1);
  std::vector<int32_t> loop_signs(static_cast<std::size_t>(loop_count), 0);
  int64_t face_count = -1;
  std::vector<int32_t> face_owner_loops(static_cast<std::size_t>(loop_count), -1);
  std::vector<int32_t> face_neighbor_loops(static_cast<std::size_t>(loop_count), -1);
  std::vector<int32_t> face_vertex_offsets(static_cast<std::size_t>(loop_count + 1), -1);
  std::vector<int32_t> face_vertex_values(static_cast<std::size_t>(entry_count), -1);
  std::vector<int32_t> cell_vertex_offsets(static_cast<std::size_t>(cell_count + 1), -1);
  std::vector<int32_t> cell_vertex_values(static_cast<std::size_t>(entry_count), -1);
  int64_t edge_count = -1;
  std::vector<int32_t> edges(static_cast<std::size_t>(2 * entry_count), -1);
  std::vector<int32_t> edge_first_sides(static_cast<std::size_t>(entry_count), -1);
  std::vector<int32_t> face_edge_values(static_cast<std::size_t>(entry_count), -1);
  std::vector<int32_t> face_order(static_cast<std::size_t>(loop_count), -1);
  std::vector<int32_t> edge_order(static_cast<std::size_t>(entry_count), -1);

  PHX_CHECK(phx_mc_packed_polyhedral_mesh_entities(
                vertex_count, cell_count, cell_face_counts.data(), loop_count,
                loop_lengths.data(), entry_count, loop_values.data(),
                loop_faces.data(), loop_signs.data(), &face_count,
                face_owner_loops.data(), face_neighbor_loops.data(),
                face_vertex_offsets.data(), face_vertex_values.data(),
                cell_vertex_offsets.data(), cell_vertex_values.data(),
                &edge_count, edges.data(), edge_first_sides.data(),
                face_edge_values.data(), face_order.data(), edge_order.data()) == PHX_MC_OK);
  // Two glued cubes: 11 distinct faces, 20 distinct edges, 8 vertices per cell.
  PHX_CHECK(face_count == 11);
  PHX_CHECK(edge_count == 20);
  PHX_CHECK(cell_vertex_offsets[0] == 0 && cell_vertex_offsets[1] == 8 &&
            cell_vertex_offsets[2] == 16);
  // Shared face is loop 5 (cube 0) and loop 10 (cube 1) with opposite signs.
  PHX_CHECK(loop_faces[5] == loop_faces[10]);
  PHX_CHECK(loop_signs[5] == -loop_signs[10]);
  const int32_t shared = loop_faces[5];
  PHX_CHECK(face_owner_loops[shared] == 5 && face_neighbor_loops[shared] == 10);
  // edge_order is strictly lexicographically increasing.
  for (int64_t i = 1; i < edge_count; ++i) {
    const int32_t prev = edge_order[i - 1];
    const int32_t curr = edge_order[i];
    PHX_CHECK(edges[2 * prev] < edges[2 * curr] ||
              (edges[2 * prev] == edges[2 * curr] && edges[2 * prev + 1] < edges[2 * curr + 1]));
  }
}

void test_rejects_malformed_packed_polyhedra() {
  // Single tetrahedron with one flipped face fails closed-shell edge pairing.
  const std::array<int64_t, 1> cell_face_counts{4};
  const std::array<int64_t, 4> loop_lengths{3, 3, 3, 3};
  std::vector<int32_t> loops{1, 2, 3,  0, 3, 2,  0, 1, 3,  0, 1, 2};  // last face flipped
  std::vector<int32_t> loop_faces(4), loop_signs(4), owner(4), neighbor(4), foff(5), fval(12),
      coff(2), cval(12), edges(24), efirst(12), fedge(12), forder(4), eorder(12);
  int64_t fcount = 0, ecount = 0;
  PHX_CHECK(phx_mc_packed_polyhedral_mesh_entities(
                4, 1, cell_face_counts.data(), 4, loop_lengths.data(), 12, loops.data(),
                loop_faces.data(), loop_signs.data(), &fcount, owner.data(), neighbor.data(),
                foff.data(), fval.data(), coff.data(), cval.data(), &ecount, edges.data(),
                efirst.data(), fedge.data(), forder.data(), eorder.data()) == PHX_MC_INVALID_INPUT);

  // Fixing orientation of the 4th face succeeds with 4 faces and 6 edges.
  loops[9] = 0;
  loops[10] = 2;
  loops[11] = 1;
  PHX_CHECK(phx_mc_packed_polyhedral_mesh_entities(
                4, 1, cell_face_counts.data(), 4, loop_lengths.data(), 12, loops.data(),
                loop_faces.data(), loop_signs.data(), &fcount, owner.data(), neighbor.data(),
                foff.data(), fval.data(), coff.data(), cval.data(), &ecount, edges.data(),
                efirst.data(), fedge.data(), forder.data(), eorder.data()) == PHX_MC_OK);
  PHX_CHECK(fcount == 4 && ecount == 6);

  // Repeated vertex inside a loop is rejected.
  std::vector<int32_t> dup = loops;
  dup[2] = dup[1];
  PHX_CHECK(phx_mc_packed_polyhedral_mesh_entities(
                4, 1, cell_face_counts.data(), 4, loop_lengths.data(), 12, dup.data(),
                loop_faces.data(), loop_signs.data(), &fcount, owner.data(), neighbor.data(),
                foff.data(), fval.data(), coff.data(), cval.data(), &ecount, edges.data(),
                efirst.data(), fedge.data(), forder.data(), eorder.data()) == PHX_MC_INVALID_INPUT);

  // Out-of-range vertex is rejected by vertex counter too.
  std::vector<int32_t> oob = loops;
  oob[0] = 4;
  std::array<int64_t, 1> counts{};
  PHX_CHECK(phx_mc_packed_cell_vertex_counts(
                4, 1, cell_face_counts.data(), 4, loop_lengths.data(), 12, oob.data(),
                counts.data()) == PHX_MC_INVALID_INPUT);
}

}  // namespace

int main() {
  test_two_cubes();
  test_rejects_malformed_packed_polyhedra();
  return phx::mc::test::finish("test_tet_entities");
}
