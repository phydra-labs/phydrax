//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// phx_mc_tet_mesh_entities: the sorted unique edges and faces of a tetrahedral
// mesh with every cell's routes into them, by a least-significant-first radix
// sort of the sorted vertex rows (one stable counting pass per vertex
// column). The result is the lexicographic order of distinct sorted vertex
// rows, so it equals the generic NumPy construction exactly.
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

// Lexicographic order of the rows by stable counting passes, last column first.
template <std::size_t Width>
void radix_sort(int64_t vertex_count, std::vector<Row<Width>>& rows) {
  std::vector<Row<Width>> scratch(rows.size());
  std::vector<int64_t> cursor(static_cast<std::size_t>(vertex_count) + 1);
  for (std::size_t column = Width; column-- > 0;) {
    std::fill(cursor.begin(), cursor.end(), 0);
    for (const Row<Width>& row : rows) ++cursor[static_cast<std::size_t>(row.vertex[column]) + 1];
    for (int64_t vertex = 0; vertex < vertex_count; ++vertex) cursor[vertex + 1] += cursor[vertex];
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
