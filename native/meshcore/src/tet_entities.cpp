//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// phx_mc_tet_mesh_entities: the sorted unique edges and faces of a tetrahedral
// mesh with every cell's routes into them, by a least-significant-first radix
// sort of the sorted vertex rows (one stable counting pass per vertex
// column). The result is the lexicographic order of distinct sorted vertex
// rows, so it equals the generic NumPy construction exactly.
//
// phx_mc_polyhedral_mesh_entities: the same radix grouping for standard
// tetrahedron, pyramid, prism and hexahedron cells, numbered by first
// appearance as the generic polyhedral connectivity numbers them.
//
// phx_mc_incidence_order: the canonical (lower, upper, sign) row order of an
// incidence over dense entity ranks, by three stable counting passes.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr int kEdgeLocal[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
constexpr int kFaceLocal[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
// Local edge number of the local vertex pair (first < second).
constexpr int kEdgeIndex[4][4] = {{-1, 0, 1, 2}, {-1, -1, 3, 4}, {-1, -1, -1, 5}, {-1, -1, -1, -1}};

template <std::size_t Width>
struct Row {
  std::array<int32_t, Width> vertex;  // increasing
  int32_t slot;                       // cell * locals + local entity
};

// Lexicographic order of the rows (values in [0, key_count)) by stable
// counting passes, last column first.
template <std::size_t Width>
void radix_sort(int64_t key_count, std::vector<Row<Width>>& rows) {
  std::vector<Row<Width>> scratch(rows.size());
  std::vector<int64_t> cursor(static_cast<std::size_t>(key_count) + 1);
  for (std::size_t column = Width; column-- > 0;) {
    std::fill(cursor.begin(), cursor.end(), 0);
    for (const Row<Width>& row : rows) ++cursor[static_cast<std::size_t>(row.vertex[column]) + 1];
    for (int64_t vertex = 0; vertex < key_count; ++vertex) cursor[vertex + 1] += cursor[vertex];
    for (const Row<Width>& row : rows) {
      scratch[static_cast<std::size_t>(cursor[static_cast<std::size_t>(row.vertex[column])]++)] =
          row;
    }
    rows.swap(scratch);
  }
}

}  // namespace

extern "C" int32_t phx_mc_tet_mesh_entities(int64_t vertex_count, int64_t cell_count,
                                            const int32_t* cells, int32_t* edges,
                                            int64_t* edge_count, int32_t* faces,
                                            int64_t* face_count, int32_t* cell_edges,
                                            int32_t* cell_faces, int32_t* face_edges,
                                            int32_t* vertex_edge_order,
                                            int32_t* edge_face_order,
                                            int32_t* face_cell_order) {
  return phx::mc::guarded([&]() -> int32_t {
    if (vertex_count < 0 || cell_count < 0 || edge_count == nullptr || face_count == nullptr ||
        vertex_count > std::numeric_limits<int32_t>::max() ||
        cell_count > std::numeric_limits<int32_t>::max() / 12 ||
        (cell_count > 0 &&
         (cells == nullptr || edges == nullptr || faces == nullptr || cell_edges == nullptr ||
          cell_faces == nullptr || face_edges == nullptr || vertex_edge_order == nullptr ||
          edge_face_order == nullptr || face_cell_order == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t item = 0; item < 4 * cell_count; ++item) {
      if (cells[item] < 0 || cells[item] >= vertex_count) return PHX_MC_INVALID_INPUT;
    }

    std::vector<Row<2>> edge_items(static_cast<std::size_t>(6 * cell_count));
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      const int32_t* row = cells + 4 * cell;
      for (int local = 0; local < 6; ++local) {
        const int32_t u = row[kEdgeLocal[local][0]];
        const int32_t v = row[kEdgeLocal[local][1]];
        edge_items[static_cast<std::size_t>(6 * cell + local)] =
            Row<2>{{std::min(u, v), std::max(u, v)}, static_cast<int32_t>(6 * cell + local)};
      }
    }
    radix_sort(vertex_count, edge_items);
    int64_t edge_total = 0;
    for (std::size_t index = 0; index < edge_items.size(); ++index) {
      const Row<2>& item = edge_items[index];
      if (index == 0 || item.vertex != edge_items[index - 1].vertex) {
        edges[2 * edge_total] = item.vertex[0];
        edges[2 * edge_total + 1] = item.vertex[1];
        ++edge_total;
      }
      cell_edges[item.slot] = static_cast<int32_t>(edge_total - 1);
    }
    edge_items = std::vector<Row<2>>();

    std::vector<Row<3>> face_items(static_cast<std::size_t>(4 * cell_count));
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      const int32_t* row = cells + 4 * cell;
      for (int local = 0; local < 4; ++local) {
        const int32_t u = row[kFaceLocal[local][0]];
        const int32_t v = row[kFaceLocal[local][1]];
        const int32_t w = row[kFaceLocal[local][2]];
        const int32_t low = std::min(std::min(u, v), w);
        const int32_t high = std::max(std::max(u, v), w);
        face_items[static_cast<std::size_t>(4 * cell + local)] =
            Row<3>{{low, u ^ v ^ w ^ low ^ high, high}, static_cast<int32_t>(4 * cell + local)};
      }
    }
    radix_sort(vertex_count, face_items);
    int64_t face_total = 0;
    for (std::size_t index = 0; index < face_items.size(); ++index) {
      const Row<3>& item = face_items[index];
      if (index == 0 || item.vertex != face_items[index - 1].vertex) {
        faces[3 * face_total] = item.vertex[0];
        faces[3 * face_total + 1] = item.vertex[1];
        faces[3 * face_total + 2] = item.vertex[2];
        // The edges opposite each face vertex in turn are edges of the cell
        // owning this slot: its local vertices in increasing value order.
        const int64_t cell = item.slot / 4;
        const int32_t* row = cells + 4 * cell;
        int local[3] = {kFaceLocal[item.slot % 4][0], kFaceLocal[item.slot % 4][1],
                        kFaceLocal[item.slot % 4][2]};
        for (int i = 1; i < 3; ++i) {
          for (int j = i; j > 0 && row[local[j - 1]] > row[local[j]]; --j) {
            std::swap(local[j - 1], local[j]);
          }
        }
        const int opposite[3][2] = {{local[1], local[2]}, {local[0], local[2]}, {local[0], local[1]}};
        for (int k = 0; k < 3; ++k) {
          const int first = std::min(opposite[k][0], opposite[k][1]);
          const int second = std::max(opposite[k][0], opposite[k][1]);
          face_edges[3 * face_total + k] = cell_edges[6 * cell + kEdgeIndex[first][second]];
        }
        ++face_total;
      }
      cell_faces[item.slot] = static_cast<int32_t>(face_total - 1);
    }
    // Incidence orders: each degree's face routes, flat in simplex-major
    // order (opposite each local vertex in turn), stably counting-sorted by
    // face row.
    const auto counting_order = [](int64_t key_count, int64_t count, const auto& key_of,
                                   int32_t* order) {
      std::vector<int64_t> cursor(static_cast<std::size_t>(key_count) + 1, 0);
      for (int64_t item = 0; item < count; ++item) ++cursor[key_of(item) + 1];
      for (int64_t key = 0; key < key_count; ++key) cursor[key + 1] += cursor[key];
      for (int64_t item = 0; item < count; ++item) {
        order[cursor[key_of(item)]++] = static_cast<int32_t>(item);
      }
    };
    counting_order(
        vertex_count, 2 * edge_total,
        [&](int64_t item) -> int64_t { return edges[2 * (item / 2) + 1 - item % 2]; },
        vertex_edge_order);
    counting_order(
        edge_total, 3 * face_total,
        [&](int64_t item) -> int64_t { return face_edges[item]; }, edge_face_order);
    // The face opposite sorted vertex r of a cell omits local vertex
    // rank[r] (a stable sort of the cell's vertices); in combination order
    // that face is number 3 - rank[r].
    std::vector<int32_t> cell_face_routes(static_cast<std::size_t>(4 * cell_count));
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      const int32_t* row = cells + 4 * cell;
      int rank[4] = {0, 1, 2, 3};
      for (int i = 1; i < 4; ++i) {
        for (int j = i; j > 0 && row[rank[j - 1]] > row[rank[j]]; --j) {
          std::swap(rank[j - 1], rank[j]);
        }
      }
      for (int r = 0; r < 4; ++r) {
        cell_face_routes[static_cast<std::size_t>(4 * cell + r)] =
            cell_faces[4 * cell + 3 - rank[r]];
      }
    }
    counting_order(
        face_total, 4 * cell_count,
        [&](int64_t item) -> int64_t { return cell_face_routes[static_cast<std::size_t>(item)]; },
        face_cell_order);
    *edge_count = edge_total;
    *face_count = face_total;
    return PHX_MC_OK;
  });
}

namespace {

// Outward face loops of the standard cell kinds, as the Python
// _POLYHEDRAL_FACE_ROUTES: tetrahedron, pyramid, prism, hexahedron.
struct KindLoops {
  int arity;
  int face_count;
  int size[6];
  int vertex[6][4];
};
constexpr KindLoops kKindLoops[4] = {
    {4, 4, {3, 3, 3, 3}, {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}}},
    {5, 5, {4, 3, 3, 3, 3}, {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}}},
    {6, 5, {3, 3, 4, 4, 4}, {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}}},
    {8,
     6,
     {4, 4, 4, 4, 4, 4},
     {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}}},
};

}  // namespace

extern "C" int32_t phx_mc_polyhedral_mesh_entities(
    int64_t vertex_count, int64_t cell_count, const int32_t* cell_kinds, const int32_t* cells,
    int32_t* loop_faces, int32_t* loop_signs, int64_t* face_count, int32_t* face_owner_loops,
    int32_t* face_neighbor_loops, int32_t* face_vertex_offsets, int32_t* face_vertex_values,
    int64_t* edge_count, int32_t* edges, int32_t* edge_first_sides, int32_t* face_edge_values,
    int32_t* face_order, int32_t* edge_order) {
  return phx::mc::guarded([&]() -> int32_t {
    if (vertex_count < 0 || cell_count < 0 || face_count == nullptr || edge_count == nullptr ||
        vertex_count >= std::numeric_limits<int32_t>::max() ||
        cell_count > std::numeric_limits<int32_t>::max() / 24 ||
        (cell_count > 0 &&
         (cell_kinds == nullptr || cells == nullptr || loop_faces == nullptr ||
          loop_signs == nullptr || face_owner_loops == nullptr || face_neighbor_loops == nullptr ||
          face_vertex_offsets == nullptr || face_vertex_values == nullptr || edges == nullptr ||
          edge_first_sides == nullptr || face_edge_values == nullptr || face_order == nullptr ||
          edge_order == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int64_t loop_total = 0;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      if (cell_kinds[cell] < 0 || cell_kinds[cell] > 3) return PHX_MC_INVALID_ARGUMENT;
      loop_total += kKindLoops[cell_kinds[cell]].face_count;
    }
    // A canonical loop starts at its smallest vertex and runs toward the
    // smaller neighbor; triangles pad with vertex_count, so no triangle
    // equals a quadrilateral.
    std::vector<Row<4>> loops(static_cast<std::size_t>(loop_total));
    std::vector<int8_t> loop_size(static_cast<std::size_t>(loop_total));
    {
      int64_t loop = 0;
      const int32_t* row = cells;
      for (int64_t cell = 0; cell < cell_count; ++cell) {
        const KindLoops& kind = kKindLoops[cell_kinds[cell]];
        for (int i = 0; i < kind.arity; ++i) {
          if (row[i] < 0 || row[i] >= vertex_count) return PHX_MC_INVALID_INPUT;
          for (int j = 0; j < i; ++j) {
            if (row[i] == row[j]) return PHX_MC_INVALID_INPUT;
          }
        }
        for (int face = 0; face < kind.face_count; ++face, ++loop) {
          const int n = kind.size[face];
          int32_t value[4];
          int first = 0;
          for (int p = 0; p < n; ++p) {
            value[p] = row[kind.vertex[face][p]];
            if (value[p] < value[first]) first = p;
          }
          const int sign = value[(first + 1) % n] < value[(first + n - 1) % n] ? 1 : -1;
          Row<4>& item = loops[static_cast<std::size_t>(loop)];
          for (int p = 0; p < n; ++p) item.vertex[p] = value[(first + sign * p + n) % n];
          if (n == 3) item.vertex[3] = static_cast<int32_t>(vertex_count);
          item.slot = static_cast<int32_t>(loop);
          loop_size[static_cast<std::size_t>(loop)] = static_cast<int8_t>(n);
          loop_signs[loop] = sign;
        }
        row += kind.arity;
      }
    }
    const std::vector<Row<4>> canonical = loops;
    radix_sort(vertex_count + 1, loops);
    // Faces number distinct loops by first appearance: a stable sort leaves
    // each run's smallest loop first. A third use or a repeated orientation
    // is a defect the generic construction reports.
    std::vector<int32_t> run_first(static_cast<std::size_t>(loop_total));
    std::vector<int32_t> second(static_cast<std::size_t>(loop_total), -1);
    for (std::size_t start = 0; start < loops.size();) {
      std::size_t stop = start + 1;
      while (stop < loops.size() && loops[stop].vertex == loops[start].vertex) ++stop;
      const int32_t first = loops[start].slot;
      if (stop - start > 2) return PHX_MC_INVALID_INPUT;
      if (stop - start == 2) {
        second[static_cast<std::size_t>(first)] = loops[start + 1].slot;
        if (loop_signs[first] == loop_signs[loops[start + 1].slot]) return PHX_MC_INVALID_INPUT;
      }
      for (std::size_t item = start; item < stop; ++item) {
        run_first[static_cast<std::size_t>(loops[item].slot)] = first;
      }
      start = stop;
    }
    loops = std::vector<Row<4>>();
    int64_t face_total = 0;
    int64_t entry_total = 0;
    face_vertex_offsets[0] = 0;
    for (int64_t loop = 0; loop < loop_total; ++loop) {
      const int32_t first = run_first[static_cast<std::size_t>(loop)];
      if (first == loop) {
        const Row<4>& item = canonical[static_cast<std::size_t>(loop)];
        const int n = loop_size[static_cast<std::size_t>(loop)];
        for (int p = 0; p < n; ++p) face_vertex_values[entry_total++] = item.vertex[p];
        face_owner_loops[face_total] = static_cast<int32_t>(loop);
        face_neighbor_loops[face_total] = second[static_cast<std::size_t>(loop)];
        loop_faces[loop] = static_cast<int32_t>(face_total++);
        face_vertex_offsets[face_total] = static_cast<int32_t>(entry_total);
      } else {
        loop_faces[loop] = loop_faces[first];
      }
    }
    // Edges number the distinct face sides by first appearance in face order.
    std::vector<Row<2>> sides(static_cast<std::size_t>(entry_total));
    for (int64_t face = 0; face < face_total; ++face) {
      const int32_t begin = face_vertex_offsets[face];
      const int32_t end = face_vertex_offsets[face + 1];
      for (int32_t entry = begin; entry < end; ++entry) {
        const int32_t u = face_vertex_values[entry];
        const int32_t v = face_vertex_values[entry + 1 < end ? entry + 1 : begin];
        sides[static_cast<std::size_t>(entry)] = Row<2>{{std::min(u, v), std::max(u, v)}, entry};
      }
    }
    radix_sort(vertex_count, sides);
    std::vector<int32_t> side_first(static_cast<std::size_t>(entry_total));
    std::vector<std::size_t> run_starts;  // sorted positions
    for (std::size_t start = 0; start < sides.size();) {
      std::size_t stop = start + 1;
      while (stop < sides.size() && sides[stop].vertex == sides[start].vertex) ++stop;
      run_starts.push_back(start);
      for (std::size_t item = start; item < stop; ++item) {
        side_first[static_cast<std::size_t>(sides[item].slot)] = sides[start].slot;
      }
      start = stop;
    }
    int64_t edge_total = 0;
    for (int64_t entry = 0; entry < entry_total; ++entry) {
      const int32_t first = side_first[static_cast<std::size_t>(entry)];
      if (first == entry) {
        edge_first_sides[edge_total] = static_cast<int32_t>(entry);
        face_edge_values[entry] = static_cast<int32_t>(edge_total++);
      } else {
        face_edge_values[entry] = face_edge_values[first];
      }
    }
    for (std::size_t run = 0; run < run_starts.size(); ++run) {
      const Row<2>& side = sides[run_starts[run]];
      const int32_t edge = face_edge_values[side.slot];
      edges[2 * edge] = side.vertex[0];
      edges[2 * edge + 1] = side.vertex[1];
      edge_order[run] = edge;
    }
    // Faces in the tuple order of their increasing vertex rows: values shift
    // by one so that a triangle's zero pad precedes every fourth vertex.
    std::vector<Row<4>> sets(static_cast<std::size_t>(face_total));
    for (int64_t face = 0; face < face_total; ++face) {
      const int32_t begin = face_vertex_offsets[face];
      const int n = face_vertex_offsets[face + 1] - begin;
      Row<4>& item = sets[static_cast<std::size_t>(face)];
      for (int p = 0; p < n; ++p) {
        const int32_t value = face_vertex_values[begin + p] + 1;
        int q = p;
        for (; q > 0 && item.vertex[q - 1] > value; --q) item.vertex[q] = item.vertex[q - 1];
        item.vertex[q] = value;
      }
      if (n == 3) item.vertex[3] = 0;
      item.slot = static_cast<int32_t>(face);
    }
    radix_sort(vertex_count + 1, sets);
    for (std::size_t index = 0; index < sets.size(); ++index) {
      if (index > 0 && sets[index].vertex == sets[index - 1].vertex) return PHX_MC_INVALID_INPUT;
      face_order[index] = sets[index].slot;
    }
    *face_count = face_total;
    *edge_count = edge_total;
    return PHX_MC_OK;
  });
}

extern "C" int32_t phx_mc_incidence_order(int64_t lower_count, int64_t upper_count,
                                          int64_t count, const int32_t* lower,
                                          const int32_t* upper, const int8_t* signs,
                                          int32_t* order) {
  return phx::mc::guarded([&]() -> int32_t {
    if (lower_count < 0 || upper_count < 0 || count < 0 ||
        count > std::numeric_limits<int32_t>::max() ||
        (count > 0 && (lower == nullptr || upper == nullptr || signs == nullptr ||
                       order == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t item = 0; item < count; ++item) {
      if (lower[item] < 0 || lower[item] >= lower_count || upper[item] < 0 ||
          upper[item] >= upper_count || (signs[item] != -1 && signs[item] != 1)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    // Stable counting passes, least significant key first: sign, upper, lower.
    std::vector<int32_t> current(static_cast<std::size_t>(count));
    std::vector<int32_t> next(static_cast<std::size_t>(count));
    {
      int64_t negatives = 0;
      for (int64_t item = 0; item < count; ++item) negatives += signs[item] < 0;
      int64_t cursor[2] = {0, negatives};
      for (int64_t item = 0; item < count; ++item) {
        current[static_cast<std::size_t>(cursor[signs[item] > 0]++)] = static_cast<int32_t>(item);
      }
    }
    const auto pass = [&](const int32_t* key, int64_t key_count) {
      std::vector<int64_t> cursor(static_cast<std::size_t>(key_count) + 1, 0);
      for (int64_t item = 0; item < count; ++item) ++cursor[static_cast<std::size_t>(key[item]) + 1];
      for (int64_t value = 0; value < key_count; ++value) cursor[value + 1] += cursor[value];
      for (const int32_t item : current) {
        next[static_cast<std::size_t>(cursor[static_cast<std::size_t>(key[item])]++)] = item;
      }
      current.swap(next);
    };
    pass(upper, upper_count);
    pass(lower, lower_count);
    std::copy(current.begin(), current.end(), order);
    return PHX_MC_OK;
  });
}
