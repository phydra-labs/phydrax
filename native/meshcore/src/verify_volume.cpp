//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Certificate of a tetrahedral mesh of the solid bounded by a closed, outward
// oriented, embedded triangle surface, checked after construction.
//
// Checked here, exactly: every cell is positively oriented (orient3d on the
// stored binary64 coordinates); every face is shared by at most two cells, and
// two cells sharing a face induce opposite orientations on it; the faces of
// exactly one cell are, with their outward orientation, exactly the triangles
// of the given surface (as oriented triangles, each once).
//
// Why that suffices: positive orientation and the opposite-orientation pairing
// make the cells a locally injective, orientation-preserving piecewise-linear
// map of a pseudomanifold whose boundary is the given surface. If the surface
// is embedded (a premise owned by the surface's own certificate), the covering
// degree of every point off the surface is the winding number of the surface,
// which is 1 inside and 0 outside; a locally injective map of degree 1 covers
// the inside exactly once and nothing outside. Hence the cells tile the solid
// without overlap or gap.
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {

// Outward faces of a positively oriented cell (v0, v1, v2, v3): the face
// opposite local vertex k, ordered so that (face, k) is an odd permutation.
constexpr std::array<std::array<int, 3>, 4> kOutwardFaces{
    {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}}};

struct FaceRecord {
  std::array<int32_t, 3> sorted;  // vertex indices in increasing order
  int64_t owner;                  // cell index, or -1 - surface triangle index
  bool even;                      // the oriented triangle is an even permutation of `sorted`
};

FaceRecord make_record(int32_t a, int32_t b, int32_t c, int64_t owner) {
  std::array<int32_t, 3> v{a, b, c};
  // Parity of the sort: count transpositions of a three-element sort network.
  bool even = true;
  if (v[0] > v[1]) {
    std::swap(v[0], v[1]);
    even = !even;
  }
  if (v[1] > v[2]) {
    std::swap(v[1], v[2]);
    even = !even;
  }
  if (v[0] > v[1]) {
    std::swap(v[0], v[1]);
    even = !even;
  }
  return {v, owner, even};
}

bool same_face(const FaceRecord& first, const FaceRecord& second) {
  return first.sorted == second.sorted;
}

bool face_less(const FaceRecord& first, const FaceRecord& second) {
  if (first.sorted != second.sorted) return first.sorted < second.sorted;
  return first.owner < second.owner;
}

void note(int64_t& count, int64_t& first, int64_t index) {
  if (count == 0) first = index;
  ++count;
}

}  // namespace

extern "C" int32_t phx_mc_verify_volume(int64_t point_count, const double* points,
                                        int64_t cell_count, const int32_t* cells,
                                        int64_t surface_count, const int32_t* surface,
                                        phx_mc_volume_verdict* verdict) {
  return phx::mc::guarded([&]() -> int32_t {
    if (verdict == nullptr || point_count < 0 || cell_count < 0 || surface_count < 0 ||
        (point_count > 0 && points == nullptr) || (cell_count > 0 && cells == nullptr) ||
        (surface_count > 0 && surface == nullptr) ||
        !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(cell_count, 4, sizeof(int32_t)) ||
        !phx::mc::addressable(surface_count, 3, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t k = 0; k < 4 * cell_count; ++k) {
      if (cells[k] < 0 || cells[k] >= point_count) return PHX_MC_INVALID_INPUT;
    }
    for (int64_t k = 0; k < 3 * surface_count; ++k) {
      if (surface[k] < 0 || surface[k] >= point_count) return PHX_MC_INVALID_INPUT;
    }
    phx_mc_volume_verdict result{};
    result.first_nonpositive_cell = -1;
    result.first_nonmanifold_cell = -1;
    result.first_unmatched_cell = -1;
    result.first_unmatched_surface_triangle = -1;

    for (int64_t cell = 0; cell < cell_count; ++cell) {
      const int32_t* v = cells + 4 * cell;
      const int sign = phx::mc::orient3d(points + 3 * v[0], points + 3 * v[1],
                                         points + 3 * v[2], points + 3 * v[3]);
      if (sign <= 0) note(result.nonpositive_cells, result.first_nonpositive_cell, cell);
    }

    std::vector<FaceRecord> faces;
    faces.reserve(static_cast<std::size_t>(4 * cell_count + surface_count));
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      const int32_t* v = cells + 4 * cell;
      for (const auto& local : kOutwardFaces) {
        faces.push_back(make_record(v[local[0]], v[local[1]], v[local[2]], cell));
      }
    }
    for (int64_t triangle = 0; triangle < surface_count; ++triangle) {
      const int32_t* v = surface + 3 * triangle;
      faces.push_back(make_record(v[0], v[1], v[2], -1 - triangle));
    }
    std::sort(faces.begin(), faces.end(), face_less);

    std::size_t begin = 0;
    while (begin < faces.size()) {
      std::size_t end = begin + 1;
      while (end < faces.size() && same_face(faces[begin], faces[end])) ++end;
      // Within a group, surface records (owner < 0) precede cell records.
      std::size_t first_cell = begin;
      while (first_cell < end && faces[first_cell].owner < 0) ++first_cell;
      const std::size_t surface_records = first_cell - begin;
      const std::size_t cell_records = end - first_cell;
      if (surface_records > 1) {
        for (std::size_t k = begin + 1; k < first_cell; ++k) {
          note(result.unmatched_surface_triangles, result.first_unmatched_surface_triangle,
               -1 - faces[k].owner);
        }
      }
      if (cell_records > 2) {
        for (std::size_t k = first_cell; k < end; ++k) {
          note(result.nonmanifold_faces, result.first_nonmanifold_cell, faces[k].owner);
        }
      } else if (cell_records == 2) {
        if (faces[first_cell].even == faces[first_cell + 1].even) {
          note(result.nonmanifold_faces, result.first_nonmanifold_cell, faces[first_cell].owner);
        }
        if (surface_records > 0) {
          note(result.unmatched_surface_triangles, result.first_unmatched_surface_triangle,
               -1 - faces[begin].owner);
        }
      } else if (cell_records == 1) {
        ++result.boundary_faces;
        const bool matched = surface_records > 0 && faces[begin].even == faces[first_cell].even;
        if (!matched) {
          note(result.unmatched_boundary_faces, result.first_unmatched_cell,
               faces[first_cell].owner);
          if (surface_records > 0) {
            note(result.unmatched_surface_triangles, result.first_unmatched_surface_triangle,
                 -1 - faces[begin].owner);
          }
        }
      } else if (surface_records > 0) {
        note(result.unmatched_surface_triangles, result.first_unmatched_surface_triangle,
             -1 - faces[begin].owner);
      }
      begin = end;
    }
    result.certified = result.nonpositive_cells == 0 && result.nonmanifold_faces == 0 &&
                               result.unmatched_boundary_faces == 0 &&
                               result.unmatched_surface_triangles == 0
                           ? 1
                           : 0;
    *verdict = result;
    return PHX_MC_OK;
  });
}
