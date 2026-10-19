//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Four trilinear hexes form the barycentric dual of a positively oriented
// affine tetrahedron.
#include <algorithm>
#include <array>
#include <limits>

#include "bounded_memory.hpp"
#include "capi_guard.hpp"
#include "phydrax_meshcore.h"
#include "topology_keys.hpp"

extern "C" PHX_MC_API int32_t phx_mc_tetrahedron_dual_hexes(int64_t tetrahedron_count,
                                                            int64_t vertex_count,
                                                            const int64_t* nodes,
                                                            int64_t hex_capacity, int64_t* hexes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!phx::mc::addressable(tetrahedron_count, 32, sizeof(int64_t)) || vertex_count < 0 ||
        hex_capacity < 0 || (tetrahedron_count != 0 && (nodes == nullptr || hexes == nullptr)))
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
    constexpr int template_nodes[4][8] = {{0, 4, 10, 5, 6, 11, 14, 12},
                                          {1, 4, 11, 8, 7, 10, 14, 13},
                                          {2, 5, 10, 7, 9, 12, 14, 13},
                                          {3, 6, 12, 9, 8, 11, 14, 13}};
    for (int64_t t = 0; t < tetrahedron_count; ++t)
      for (int h = 0; h < 4; ++h)
        for (int corner = 0; corner < 8; ++corner)
          hexes[32 * t + 8 * h + corner] = nodes[15 * t + template_nodes[h][corner]];
    return PHX_MC_OK;
  });
}

// Independent 1:8 grid template. The 27 supplied shared-grid node IDs are
// ordered (x*3+y)*3+z, x,y,z in {0,1,2}. Neighboring parents MUST share
// the nine nodes of their common face; no hanging-face template is accepted.
extern "C" PHX_MC_API int32_t phx_mc_subdivide_hex_grid(int64_t parent_count, int64_t vertex_count,
                                                        const int64_t* nodes, int64_t hex_capacity,
                                                        int64_t* hexes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!phx::mc::addressable(parent_count, 64, sizeof(int64_t)) || vertex_count < 0 ||
        hex_capacity < 0 || (parent_count != 0 && (nodes == nullptr || hexes == nullptr)))
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
    constexpr int corners[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                   {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    for (int64_t parent = 0; parent < parent_count; ++parent)
      for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
          for (int z = 0; z < 2; ++z) {
            const int child = (x * 2 + y) * 2 + z;
            for (int corner = 0; corner < 8; ++corner) {
              const int node = ((x + corners[corner][0]) * 3 + y + corners[corner][1]) * 3 + z +
                               corners[corner][2];
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

struct CornerHexWork {
  const phx_mc_polyhedron_corner_hexes_request& request;
  void operator()(int64_t amount) const {
    if (amount < 0 || amount > request.work_limit - *request.work_units)
      throw phx::mc::ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
    phx::mc::native_execution_charge(amount);
    *request.work_units += amount;
  }
};

int32_t validate_corner_hex_request(const phx_mc_polyhedron_corner_hexes_request& request) {
  if (request.nv <= 0 || request.ne < 0 || request.nf <= 0 || request.nc <= 0 ||
      request.work_limit <= 0 || request.scratch_limit <= 0 ||
      request.nv > std::numeric_limits<int64_t>::max() - request.ne ||
      request.nv + request.ne > std::numeric_limits<int64_t>::max() - request.nf ||
      request.nv + request.ne + request.nf > std::numeric_limits<int64_t>::max() - request.nc ||
      request.vertex_count != request.nv + request.ne + request.nf + request.nc ||
      !phx::mc::addressable(request.ne, 2, sizeof(int64_t)) ||
      !phx::mc::addressable(request.face_entries, 1, sizeof(int64_t)) ||
      !phx::mc::addressable(request.cell_face_entries, 1, sizeof(int64_t)) ||
      !phx::mc::addressable(request.corner_entries, 8, sizeof(int64_t)) ||
      !phx::mc::addressable(request.nf + 1, 1, sizeof(int64_t)) ||
      !phx::mc::addressable(request.nc + 1, 1, sizeof(int64_t)) || request.edges == nullptr ||
      request.face_offsets == nullptr || request.face_vertices == nullptr ||
      request.cell_offsets == nullptr || request.cell_faces == nullptr ||
      request.vertex_offsets == nullptr || request.cell_vertices == nullptr ||
      request.corner_orientations == nullptr || request.hexes == nullptr ||
      request.parent_cells == nullptr || request.hex_count == nullptr ||
      request.witness == nullptr || request.work_units == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  return PHX_MC_OK;
}

int32_t validate_corner_hex_incidence(const phx_mc_polyhedron_corner_hexes_request& request,
                                      const CornerHexWork& charge) {
  for (int64_t face = 0; face < request.nf; ++face) {
    charge(1);
    if (request.face_offsets[face] < 0 ||
        request.face_offsets[face + 1] < request.face_offsets[face] ||
        request.face_offsets[face + 1] > request.face_entries ||
        request.face_offsets[face + 1] - request.face_offsets[face] < 3 ||
        request.face_offsets[face + 1] - request.face_offsets[face] > 16)
      return PHX_MC_INVALID_INPUT;
    for (int64_t row = request.face_offsets[face]; row < request.face_offsets[face + 1]; ++row) {
      charge(1);
      if (request.face_vertices[row] < 0 || request.face_vertices[row] >= request.nv)
        return PHX_MC_INVALID_INPUT;
      for (int64_t prior = request.face_offsets[face]; prior < row; ++prior) {
        charge(1);
        if (request.face_vertices[prior] == request.face_vertices[row]) return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int64_t cell = 0; cell < request.nc; ++cell) {
    charge(1);
    if (request.cell_offsets[cell] < 0 ||
        request.cell_offsets[cell + 1] < request.cell_offsets[cell] ||
        request.cell_offsets[cell + 1] > request.cell_face_entries ||
        request.cell_offsets[cell + 1] - request.cell_offsets[cell] < 4 ||
        request.cell_offsets[cell + 1] - request.cell_offsets[cell] > 10 ||
        request.vertex_offsets[cell] < 0 ||
        request.vertex_offsets[cell + 1] < request.vertex_offsets[cell] ||
        request.vertex_offsets[cell + 1] > request.corner_entries ||
        request.vertex_offsets[cell + 1] - request.vertex_offsets[cell] < 4 ||
        request.vertex_offsets[cell + 1] - request.vertex_offsets[cell] > 16)
      return PHX_MC_INVALID_INPUT;
  }
  return PHX_MC_OK;
}

int32_t prepare_cut_edge_lookup(const phx_mc_polyhedron_corner_hexes_request& request,
                                const CornerHexWork& charge,
                                phx::mc::NativeVector<CutEdgeKey>& lookup) {
  for (int64_t row = 0; row < request.ne; ++row) {
    charge(1);
    const auto endpoints = phx::mc::topology::undirected_edge_vertices(request.edges[2 * row],
                                                                       request.edges[2 * row + 1]);
    const int64_t first = endpoints[0], second = endpoints[1];
    if (first < 0 || first == second || second >= request.nv) return PHX_MC_INVALID_INPUT;
    lookup[row] = {first, second, row};
  }
  std::sort(lookup.begin(), lookup.end(), [&](const CutEdgeKey& a, const CutEdgeKey& b) {
    charge(1);
    return a < b;
  });
  for (int64_t row = 1; row < request.ne; ++row)
    if (lookup[row].first == lookup[row - 1].first && lookup[row].second == lookup[row - 1].second)
      return PHX_MC_INVALID_INPUT;
  return PHX_MC_OK;
}

int64_t find_cut_edge(const phx::mc::NativeVector<CutEdgeKey>& lookup, const CornerHexWork& charge,
                      int64_t first, int64_t second) {
  const auto endpoints = phx::mc::topology::undirected_edge_vertices(first, second);
  CutEdgeKey key{endpoints[0], endpoints[1], 0};
  auto found = std::lower_bound(lookup.begin(), lookup.end(), key,
                                [&](const CutEdgeKey& a, const CutEdgeKey& b) {
                                  charge(1);
                                  return a < b;
                                });
  return found == lookup.end() || found->first != key.first || found->second != key.second
             ? -1
             : found->row;
}

int32_t build_corner_links(const phx_mc_polyhedron_corner_hexes_request& request,
                           const phx::mc::NativeVector<CutEdgeKey>& lookup,
                           const CornerHexWork& charge, int64_t cell, int64_t start, int64_t end,
                           std::array<CutCornerLink, 16>& links) {
  auto local_vertex = [&](int64_t vertex) -> int {
    for (int64_t row = start; row < end; ++row) {
      charge(1);
      if (request.cell_vertices[row] == vertex) return static_cast<int>(row - start);
    }
    return -1;
  };
  for (int64_t row = start; row < end; ++row) {
    charge(1);
    if (request.cell_vertices[row] < 0 || request.cell_vertices[row] >= request.nv ||
        (request.corner_orientations[row] != -1 && request.corner_orientations[row] != 1))
      return PHX_MC_INVALID_INPUT;
    for (int64_t previous = start; previous < row; ++previous) {
      charge(1);
      if (request.cell_vertices[row] == request.cell_vertices[previous])
        return PHX_MC_INVALID_INPUT;
    }
  }
  for (int64_t slot = request.cell_offsets[cell]; slot < request.cell_offsets[cell + 1]; ++slot) {
    const int64_t face = request.cell_faces[slot];
    charge(1);
    if (face < 0 || face >= request.nf) return PHX_MC_INVALID_INPUT;
    for (int64_t prior = request.cell_offsets[cell]; prior < slot; ++prior) {
      charge(1);
      if (request.cell_faces[prior] == face) return PHX_MC_INVALID_INPUT;
    }
    const int64_t first = request.face_offsets[face], last = request.face_offsets[face + 1];
    for (int64_t row = first; row < last; ++row) {
      const int64_t vertex = request.face_vertices[row];
      const int local = local_vertex(vertex);
      if (local < 0) return PHX_MC_INVALID_INPUT;
      auto& link = links[local];
      if (link.face_count == 10) return PHX_MC_INVALID_INPUT;
      const int64_t before = request.face_vertices[row == first ? last - 1 : row - 1];
      const int64_t after = request.face_vertices[row + 1 == last ? first : row + 1];
      if (local_vertex(before) < 0 || local_vertex(after) < 0) return PHX_MC_INVALID_INPUT;
      const int64_t a = find_cut_edge(lookup, charge, vertex, before),
                    b = find_cut_edge(lookup, charge, vertex, after);
      if (a < 0 || b < 0 || a == b) return PHX_MC_INVALID_INPUT;
      link.faces[link.face_count] = face;
      link.pairs[link.face_count++] = phx::mc::topology::undirected_edge_vertices(a, b);
      for (int64_t edge : {a, b}) {
        auto found = std::find(link.edges.begin(), link.edges.begin() + link.edge_count, edge);
        if (found == link.edges.begin() + link.edge_count) {
          if (link.edge_count == 10) return PHX_MC_INVALID_INPUT;
          link.edges[link.edge_count++] = edge;
        }
      }
    }
  }
  return PHX_MC_OK;
}

int32_t emit_corner_hexes(const phx_mc_polyhedron_corner_hexes_request& request,
                          const CornerHexWork& charge, int64_t cell, int64_t start, int64_t end,
                          std::array<CutCornerLink, 16>& links, int64_t& output) {
  for (int64_t row = start; row < end; ++row) {
    auto& link = links[row - start];
    if (link.edge_count != 3 || link.face_count != 3) {
      request.witness[0] = cell;
      request.witness[1] = request.cell_vertices[row];
      request.witness[2] = link.edge_count;
      request.witness[3] = link.face_count;
      return PHX_MC_REFINEMENT_LIMIT;
    }
    std::sort(link.edges.begin(), link.edges.begin() + 3);
    if (request.corner_orientations[row] < 0) std::swap(link.edges[1], link.edges[2]);
    const int64_t a = link.edges[0], b = link.edges[1], c = link.edges[2];
    auto face_node = [&](int64_t first, int64_t second) -> int64_t {
      const auto pair = phx::mc::topology::undirected_edge_vertices(first, second);
      int matches = 0;
      int64_t node = -1;
      for (int slot = 0; slot < 3; ++slot) {
        charge(1);
        if (link.pairs[slot] == pair) {
          ++matches;
          node = request.nv + request.ne + link.faces[slot];
        }
      }
      return matches == 1 ? node : -1;
    };
    const int64_t ab = face_node(a, b), ac = face_node(a, c), bc = face_node(b, c);
    if (ab < 0 || ac < 0 || bc < 0) return PHX_MC_INVALID_INPUT;
    const std::array<int64_t, 8> nodes{request.cell_vertices[row],
                                       request.nv + a,
                                       ab,
                                       request.nv + b,
                                       request.nv + c,
                                       ac,
                                       request.nv + request.ne + request.nf + cell,
                                       bc};
    std::copy(nodes.begin(), nodes.end(), request.hexes + 8 * output);
    request.parent_cells[output++] = cell;
  }
  return PHX_MC_OK;
}
}  // namespace

extern "C" PHX_MC_API int32_t
phx_mc_polyhedron_corner_hexes(const phx_mc_polyhedron_corner_hexes_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    int32_t status = validate_corner_hex_request(*request);
    if (status != PHX_MC_OK) return status;
    *request->hex_count = 0;
    *request->work_units = 0;
    std::fill(request->witness, request->witness + 4, -1);
    if (request->hex_capacity < request->corner_entries) return PHX_MC_CAPACITY_EXCEEDED;
    if (request->scratch_limit < 16384 ||
        request->ne > (request->scratch_limit - 16384) / static_cast<int64_t>(sizeof(CutEdgeKey)))
      return PHX_MC_CAPACITY_EXCEEDED;
    phx::mc::MemoryBudgetWindow memory(static_cast<std::size_t>(request->scratch_limit));
    phx::mc::MemoryScope memory_scope(memory.owner());
    if (request->face_offsets[0] || request->face_offsets[request->nf] != request->face_entries ||
        request->cell_offsets[0] ||
        request->cell_offsets[request->nc] != request->cell_face_entries ||
        request->vertex_offsets[0] ||
        request->vertex_offsets[request->nc] != request->corner_entries)
      return PHX_MC_INVALID_INPUT;
    CornerHexWork charge{*request};
    status = validate_corner_hex_incidence(*request, charge);
    if (status != PHX_MC_OK) return status;
    phx::mc::NativeVector<CutEdgeKey> lookup(static_cast<std::size_t>(request->ne));
    status = prepare_cut_edge_lookup(*request, charge, lookup);
    if (status != PHX_MC_OK) return status;
    int64_t output = 0;
    for (int64_t cell = 0; cell < request->nc; ++cell) {
      std::array<CutCornerLink, 16> links{};
      const int64_t start = request->vertex_offsets[cell], end = request->vertex_offsets[cell + 1];
      status = build_corner_links(*request, lookup, charge, cell, start, end, links);
      if (status != PHX_MC_OK) return status;
      status = emit_corner_hexes(*request, charge, cell, start, end, links, output);
      if (status != PHX_MC_OK) return status;
    }
    *request->hex_count = output;
    return PHX_MC_OK;
  });
}
