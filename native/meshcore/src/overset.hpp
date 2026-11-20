//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Overset assembly of three-dimensional affine-tetrahedron and trilinear-
// hexahedron parts in one call: hole map against the other parts' walls,
// cut cells, fringe layers, donor search and certification after
// construction. The contract of every input and output is the C entry
// phx_mc_overset_assemble (phydrax_meshcore.h); kernels assume validated
// arguments.
#pragma once

#include <cstdint>

namespace phx::mc::overset {

// Stats slots (phx_mc_overset_assemble documents them).
inline constexpr int kStatCount = 16;

struct Input {
  int32_t part_count;
  const int64_t* vertex_offsets;  // part_count + 1
  const double* vertices;         // (V, 3)
  const int64_t* cell_offsets;    // part_count + 1
  const int32_t* cells;           // (C, 8); tetrahedra end with four -1
  const int64_t* cell_ids;        // (C)
  const uint8_t* cell_excluded;   // (C)
  const int64_t* face_offsets;    // part_count + 1
  const int32_t* faces;           // (F, 4); triangles end with -1
  const uint8_t* face_roles;      // (F): 0 other exterior, 1 wall, 2 receptor boundary
  const int32_t* priorities;      // part_count
  const int32_t* name_ranks;      // part_count
  int32_t fringe_layers;
  int32_t donor_priority;  // 0 finest donor, 1 part priority
  int32_t fringe_donors;   // fringe cells donate when no active cell does
  int32_t prove;
};

struct Output {
  int8_t* vertex_status;        // (V)
  int32_t* fringe_layer;        // (V)
  uint8_t* wall_contact;        // (V)
  uint8_t* protected_conflict;  // (V)
  int8_t* cell_status;          // (C)
  uint8_t* ambiguous;           // (C)
  int32_t* donor_status;        // (V)
  int64_t* donor_cell;          // (V)
  double* donor_reference;      // (V, 3)
  double* donor_residual;       // (V)
  double* donor_measure;        // (V)
  int32_t* candidate_parts;     // (V)
  double* stats;                // (kStatCount)
};

void assemble(const Input& input, const Output& output);

}  // namespace phx::mc::overset
