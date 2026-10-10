//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Four genuine trilinear hexes partition a positively oriented affine tetra.
// This is its barycentric dual, not a change of the tetrahedron's family tag.
#include <algorithm>
#include <array>
#include <limits>
#include "bounded_memory.hpp"
#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

extern "C" PHX_MC_API int32_t phx_mc_tetrahedron_dual_hexes(
    int64_t tetrahedron_count, int64_t vertex_count, const int64_t* nodes,
    int64_t hex_capacity, int64_t* hexes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!phx::mc::addressable(tetrahedron_count, 32, sizeof(int64_t)) ||
        vertex_count < 0 || hex_capacity < 0 ||
        (tetrahedron_count != 0 && (nodes == nullptr || hexes == nullptr)))
      return PHX_MC_INVALID_ARGUMENT;
    if (hex_capacity < 4 * tetrahedron_count) return PHX_MC_CAPACITY_EXCEEDED;
    for (int64_t t = 0; t < tetrahedron_count; ++t) {
      const int64_t* p = nodes + 15 * t;
      for (int i = 0; i < 15; ++i) {
        if (p[i] < 0 || p[i] >= vertex_count) return PHX_MC_INVALID_INPUT;
        for (int j = 0; j < i; ++j)
          if (p[i] == p[j]) return PHX_MC_INVALID_INPUT;
      }
    }
    // nodes: v0..v3, e01,e02,e03,e12,e13,e23,
    //        f012,f013,f023,f123, tetrahedron_center.
    // Each star uses a positive ordering of its three outgoing edge directions.
    constexpr int template_nodes[4][8] = {
        {0, 4, 10, 5, 6, 11, 14, 12},
        {1, 4, 11, 8, 7, 10, 14, 13},
        {2, 5, 10, 7, 9, 12, 14, 13},
        {3, 6, 12, 9, 8, 11, 14, 13}};
    for (int64_t t = 0; t < tetrahedron_count; ++t)
      for (int h = 0; h < 4; ++h)
        for (int corner = 0; corner < 8; ++corner)
          hexes[32 * t + 8 * h + corner] =
              nodes[15 * t + template_nodes[h][corner]];
    return PHX_MC_OK;
  });
}

// Independent 1:8 grid template. The 27 supplied shared-grid node IDs are
// ordered (x*3+y)*3+z, x,y,z in {0,1,2}. Neighboring parents MUST share
// the nine nodes of their common face; no hanging-face template is accepted.
extern "C" PHX_MC_API int32_t phx_mc_subdivide_hex_grid(
    int64_t parent_count, int64_t vertex_count, const int64_t* nodes,
    int64_t hex_capacity, int64_t* hexes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!phx::mc::addressable(parent_count, 64, sizeof(int64_t)) ||
        vertex_count < 0 || hex_capacity < 0 ||
        (parent_count != 0 && (nodes == nullptr || hexes == nullptr)))
      return PHX_MC_INVALID_ARGUMENT;
    if (hex_capacity < 8 * parent_count) return PHX_MC_CAPACITY_EXCEEDED;
    for (int64_t parent = 0; parent < parent_count; ++parent) {
      const int64_t* p = nodes + 27 * parent;
      for (int i = 0; i < 27; ++i) {
        if (p[i] < 0 || p[i] >= vertex_count) return PHX_MC_INVALID_INPUT;
        for (int j = 0; j < i; ++j)
          if (p[i] == p[j]) return PHX_MC_INVALID_INPUT;
      }
    }
    constexpr int corners[8][3] = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
                                  {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}};
    for (int64_t parent = 0; parent < parent_count; ++parent)
      for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
          for (int z = 0; z < 2; ++z) {
            const int child = (x * 2 + y) * 2 + z;
            for (int corner = 0; corner < 8; ++corner) {
              const int node = ((x + corners[corner][0]) * 3 +
                                y + corners[corner][1]) * 3 + z + corners[corner][2];
              hexes[64 * parent + 8 * child + corner] = nodes[27 * parent + node];
            }
          }
    return PHX_MC_OK;
  });
}

namespace {
struct CutEdgeKey {
  int64_t first, second, row;
  bool operator<(const CutEdgeKey& other) const {
    return first != other.first ? first < other.first : second < other.second;
  }
};
struct CutCornerLink {
  std::array<int64_t, 10> edges{};
  std::array<int64_t, 10> faces{};
  std::array<std::array<int64_t, 2>, 10> pairs{};
  int edge_count = 0, face_count = 0;
};
}

extern "C" PHX_MC_API int32_t phx_mc_polyhedron_corner_hexes(
    int64_t nv, int64_t ne, int64_t nf, int64_t nc,
    int64_t face_entries, int64_t cell_face_entries, int64_t corner_entries,
    const int64_t* edges, const int64_t* face_offsets, const int64_t* face_vertices,
    const int64_t* cell_offsets, const int64_t* cell_faces,
    const int64_t* vertex_offsets, const int64_t* cell_vertices,
    const int8_t* corner_orientations, int64_t vertex_count,
    int64_t hex_capacity, int64_t scratch_limit, int64_t work_limit, int64_t* hexes,
    int64_t* parent_cells, int64_t* hex_count, int64_t* witness, int64_t* work_units) {
  return phx::mc::guarded([&]() -> int32_t {
    using phx::mc::NativeVector;
    if (nv <= 0 || ne < 0 || nf <= 0 || nc <= 0 || work_limit <= 0 || scratch_limit <= 0 ||
        nv > std::numeric_limits<int64_t>::max() - ne ||
        nv + ne > std::numeric_limits<int64_t>::max() - nf ||
        nv + ne + nf > std::numeric_limits<int64_t>::max() - nc ||
        vertex_count != nv + ne + nf + nc ||
        !phx::mc::addressable(ne, 2, sizeof(int64_t)) ||
        !phx::mc::addressable(face_entries, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(cell_face_entries, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(corner_entries, 8, sizeof(int64_t)) ||
        !phx::mc::addressable(nf + 1, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(nc + 1, 1, sizeof(int64_t)) ||
        edges == nullptr || face_offsets == nullptr || face_vertices == nullptr ||
        cell_offsets == nullptr || cell_faces == nullptr || vertex_offsets == nullptr ||
        cell_vertices == nullptr || corner_orientations == nullptr || hexes == nullptr ||
        parent_cells == nullptr || hex_count == nullptr || witness == nullptr || work_units == nullptr)
      return PHX_MC_INVALID_ARGUMENT;
    *hex_count = 0;
    *work_units = 0;
    std::fill(witness, witness + 4, -1);
    if (hex_capacity < corner_entries) return PHX_MC_CAPACITY_EXCEEDED;
    if (scratch_limit < 16384 || ne > (scratch_limit - 16384) / static_cast<int64_t>(sizeof(CutEdgeKey)))
      return PHX_MC_CAPACITY_EXCEEDED;
    phx::mc::MemoryBudgetWindow memory(static_cast<std::size_t>(scratch_limit));
    phx::mc::MemoryScope memory_scope(memory.owner());
    if (face_offsets[0] || face_offsets[nf] != face_entries ||
        cell_offsets[0] || cell_offsets[nc] != cell_face_entries ||
        vertex_offsets[0] || vertex_offsets[nc] != corner_entries)
      return PHX_MC_INVALID_INPUT;
    auto charge = [&](int64_t amount) {
      if (amount < 0 || amount > work_limit - *work_units)
        throw phx::mc::ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
      phx::mc::native_execution_charge(amount);
      *work_units += amount;
    };
    for (int64_t face = 0; face < nf; ++face) {
      charge(1);
      if (face_offsets[face] < 0 || face_offsets[face + 1] < face_offsets[face] ||
          face_offsets[face + 1] > face_entries ||
          face_offsets[face + 1] - face_offsets[face] < 3 ||
          face_offsets[face + 1] - face_offsets[face] > 16)
        return PHX_MC_INVALID_INPUT;
      for (int64_t row = face_offsets[face]; row < face_offsets[face + 1]; ++row) {
        charge(1);
        if (face_vertices[row] < 0 || face_vertices[row] >= nv) return PHX_MC_INVALID_INPUT;
        for (int64_t prior = face_offsets[face]; prior < row; ++prior) {
          charge(1);
          if (face_vertices[prior] == face_vertices[row]) return PHX_MC_INVALID_INPUT;
        }
      }
    }
    for (int64_t cell = 0; cell < nc; ++cell) {
      charge(1);
      if (cell_offsets[cell] < 0 || cell_offsets[cell + 1] < cell_offsets[cell] ||
          cell_offsets[cell + 1] > cell_face_entries ||
          cell_offsets[cell + 1] - cell_offsets[cell] < 4 ||
          cell_offsets[cell + 1] - cell_offsets[cell] > 10 ||
          vertex_offsets[cell] < 0 || vertex_offsets[cell + 1] < vertex_offsets[cell] ||
          vertex_offsets[cell + 1] > corner_entries ||
          vertex_offsets[cell + 1] - vertex_offsets[cell] < 4 ||
          vertex_offsets[cell + 1] - vertex_offsets[cell] > 16)
        return PHX_MC_INVALID_INPUT;
    }
    NativeVector<CutEdgeKey> lookup(static_cast<std::size_t>(ne));
    for (int64_t row = 0; row < ne; ++row) {
      charge(1);
      const int64_t first = std::min(edges[2 * row], edges[2 * row + 1]);
      const int64_t second = std::max(edges[2 * row], edges[2 * row + 1]);
      if (first < 0 || first == second || second >= nv) return PHX_MC_INVALID_INPUT;
      lookup[row] = {first, second, row};
    }
    std::sort(lookup.begin(), lookup.end(), [&](const CutEdgeKey& a, const CutEdgeKey& b) {
      charge(1);
      return a < b;
    });
    for (int64_t row = 1; row < ne; ++row)
      if (lookup[row].first == lookup[row - 1].first && lookup[row].second == lookup[row - 1].second)
        return PHX_MC_INVALID_INPUT;
    auto edge_row = [&](int64_t first, int64_t second) -> int64_t {
      CutEdgeKey key{std::min(first, second), std::max(first, second), 0};
      auto found = std::lower_bound(lookup.begin(), lookup.end(), key,
          [&](const CutEdgeKey& a, const CutEdgeKey& b) { charge(1); return a < b; });
      return found == lookup.end() || found->first != key.first || found->second != key.second ? -1 : found->row;
    };
    int64_t output = 0;
    for (int64_t cell = 0; cell < nc; ++cell) {
      std::array<CutCornerLink, 16> links{};
      const int64_t start = vertex_offsets[cell], end = vertex_offsets[cell + 1];
      auto local_vertex = [&](int64_t vertex) -> int {
        for (int64_t row = start; row < end; ++row) {
          charge(1);
          if (cell_vertices[row] == vertex) return static_cast<int>(row - start);
        }
        return -1;
      };
      for (int64_t row = start; row < end; ++row) {
        charge(1);
        if (cell_vertices[row] < 0 || cell_vertices[row] >= nv ||
            (corner_orientations[row] != -1 && corner_orientations[row] != 1))
          return PHX_MC_INVALID_INPUT;
        for (int64_t previous = start; previous < row; ++previous) {
          charge(1);
          if (cell_vertices[row] == cell_vertices[previous]) return PHX_MC_INVALID_INPUT;
        }
      }
      for (int64_t slot = cell_offsets[cell]; slot < cell_offsets[cell + 1]; ++slot) {
        const int64_t face = cell_faces[slot];
        charge(1);
        if (face < 0 || face >= nf) return PHX_MC_INVALID_INPUT;
        for (int64_t prior = cell_offsets[cell]; prior < slot; ++prior) {
          charge(1);
          if (cell_faces[prior] == face) return PHX_MC_INVALID_INPUT;
        }
        const int64_t first = face_offsets[face], last = face_offsets[face + 1];
        for (int64_t row = first; row < last; ++row) {
          const int64_t vertex = face_vertices[row];
          const int local = local_vertex(vertex);
          if (local < 0) return PHX_MC_INVALID_INPUT;
          auto& link = links[local];
          if (link.face_count == 10) return PHX_MC_INVALID_INPUT;
          const int64_t before = face_vertices[row == first ? last - 1 : row - 1];
          const int64_t after = face_vertices[row + 1 == last ? first : row + 1];
          if (local_vertex(before) < 0 || local_vertex(after) < 0) return PHX_MC_INVALID_INPUT;
          const int64_t a = edge_row(vertex, before), b = edge_row(vertex, after);
          if (a < 0 || b < 0 || a == b) return PHX_MC_INVALID_INPUT;
          link.faces[link.face_count] = face;
          link.pairs[link.face_count++] = {std::min(a, b), std::max(a, b)};
          for (int64_t edge : {a, b}) {
            auto found = std::find(link.edges.begin(), link.edges.begin() + link.edge_count, edge);
            if (found == link.edges.begin() + link.edge_count) {
              if (link.edge_count == 10) return PHX_MC_INVALID_INPUT;
              link.edges[link.edge_count++] = edge;
            }
          }
        }
      }
      for (int64_t row = start; row < end; ++row) {
        auto& link = links[row - start];
        if (link.edge_count != 3 || link.face_count != 3) {
          witness[0] = cell; witness[1] = cell_vertices[row];
          witness[2] = link.edge_count; witness[3] = link.face_count;
          return PHX_MC_REFINEMENT_LIMIT;
        }
        std::sort(link.edges.begin(), link.edges.begin() + 3);
        if (corner_orientations[row] < 0) std::swap(link.edges[1], link.edges[2]);
        const int64_t a = link.edges[0], b = link.edges[1], c = link.edges[2];
        auto face_node = [&](int64_t first, int64_t second) -> int64_t {
          const std::array<int64_t, 2> pair{std::min(first, second), std::max(first, second)};
          int matches = 0;
          int64_t node = -1;
          for (int slot = 0; slot < 3; ++slot) {
            charge(1);
            if (link.pairs[slot] == pair) { ++matches; node = nv + ne + link.faces[slot]; }
          }
          return matches == 1 ? node : -1;
        };
        const int64_t ab = face_node(a, b), ac = face_node(a, c), bc = face_node(b, c);
        if (ab < 0 || ac < 0 || bc < 0) return PHX_MC_INVALID_INPUT;
        const std::array<int64_t, 8> nodes{
            cell_vertices[row], nv + a, ab, nv + b, nv + c, ac, nv + ne + nf + cell, bc};
        std::copy(nodes.begin(), nodes.end(), hexes + 8 * output);
        parent_cells[output++] = cell;
      }
    }
    *hex_count = output;
    return PHX_MC_OK;
  });
}
