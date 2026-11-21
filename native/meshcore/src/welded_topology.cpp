// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// phx_mc_welded_topology: the combinatorial counts of phydrax's welded-topology
// audit (phydrax.meshing._audit_topology), identical to its host route.
//
// Rows are vertex keys padded with negative entries; a row's key is its
// multiset of nonnegative entries. Rows are grouped by key through a counting
// sort on the least entry and a comparison sort of each (small) bucket by the
// remaining sorted entries and the row index, so every group lists its
// occurrences in index order, which is the order the host route pairs them
// in. Star components use one union-find over (entity, cell) incidences with
// a counting-sorted table per least vertex key.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr std::int64_t kPad = std::numeric_limits<std::int64_t>::max();

struct UnknownIncidence {};

// Sorted key rows (negative entries become kPad, as the host route's padding
// sentinel), grouped: `order` lists row indices by (key, index) and `starts`
// delimits the groups of equal keys.
struct Grouping {
  std::int64_t width = 0;
  std::vector<std::int64_t> keys;
  std::vector<std::int64_t> order;
  std::vector<std::int64_t> starts;

  const std::int64_t* key(std::int64_t row) const { return keys.data() + row * width; }
};

Grouping group_rows(const std::int64_t* rows, std::int64_t count, std::int64_t width,
                    std::int64_t key_count) {
  Grouping grouping;
  grouping.width = width;
  grouping.keys.resize(static_cast<std::size_t>(count * width));
  for (std::int64_t row = 0; row < count; ++row) {
    std::int64_t* key = grouping.keys.data() + row * width;
    for (std::int64_t column = 0; column < width; ++column) {
      const std::int64_t value = rows[row * width + column];
      key[column] = value < 0 ? kPad : value;
    }
    std::sort(key, key + width);
  }
  // Bucket key_count holds rows without a nonnegative entry.
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(key_count + 2), 0);
  const auto bucket = [&](std::int64_t row) {
    const std::int64_t first = grouping.key(row)[0];
    return first == kPad ? key_count : first;
  };
  for (std::int64_t row = 0; row < count; ++row) ++offsets[bucket(row) + 1];
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  grouping.order.resize(static_cast<std::size_t>(count));
  {
    std::vector<std::int64_t> cursor(offsets.begin(), offsets.end() - 1);
    for (std::int64_t row = 0; row < count; ++row) grouping.order[cursor[bucket(row)]++] = row;
  }
  const auto less = [&](std::int64_t first, std::int64_t second) {
    const std::int64_t* a = grouping.key(first);
    const std::int64_t* b = grouping.key(second);
    for (std::int64_t column = 1; column < width; ++column) {
      if (a[column] != b[column]) return a[column] < b[column];
    }
    return first < second;
  };
  for (std::int64_t value = 0; value <= key_count; ++value) {
    // Counting placement keeps index order; only multi-row buckets need sorting.
    if (offsets[value + 1] - offsets[value] > 1) {
      std::sort(grouping.order.begin() + offsets[value], grouping.order.begin() + offsets[value + 1],
                less);
    }
  }
  for (std::int64_t position = 0; position < count; ++position) {
    if (position == 0 ||
        !std::equal(grouping.key(grouping.order[position]),
                    grouping.key(grouping.order[position]) + width,
                    grouping.key(grouping.order[position - 1]))) {
      grouping.starts.push_back(position);
    }
  }
  grouping.starts.push_back(count);
  return grouping;
}

// (first key, second key, cell) incidences, counting-sorted by first key and
// sorted and unique within each first key; a union-find over them.
struct Incidences {
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> second;
  std::vector<std::int64_t> cell;
  std::vector<std::int64_t> parent;

  Incidences(const std::vector<std::int64_t>& first_keys, const std::vector<std::int64_t>& second_keys,
             const std::vector<std::int64_t>& cells, std::int64_t key_count) {
    const std::size_t count = first_keys.size();
    offsets.assign(static_cast<std::size_t>(key_count + 1), 0);
    for (const std::int64_t key : first_keys) ++offsets[key + 1];
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<std::int64_t> cursor(offsets.begin(), offsets.end() - 1);
    std::vector<std::pair<std::int64_t, std::int64_t>> placed(count);
    for (std::size_t index = 0; index < count; ++index) {
      placed[cursor[first_keys[index]]++] = {second_keys[index], cells[index]};
    }
    std::int64_t write = 0;
    for (std::int64_t key = 0; key < key_count; ++key) {
      const auto begin = placed.begin() + offsets[key];
      const auto end = placed.begin() + offsets[key + 1];
      std::sort(begin, end);
      const std::int64_t start = write;
      for (auto item = begin; item != end; ++item) {
        if (write > start && placed[write - 1] == *item) continue;
        placed[write++] = *item;
      }
      offsets[key] = start;
    }
    offsets[key_count] = write;
    second.resize(static_cast<std::size_t>(write));
    cell.resize(static_cast<std::size_t>(write));
    for (std::int64_t index = 0; index < write; ++index) {
      second[index] = placed[index].first;
      cell[index] = placed[index].second;
    }
    parent.resize(static_cast<std::size_t>(write));
    std::iota(parent.begin(), parent.end(), std::int64_t{0});
  }

  std::int64_t find(std::int64_t first_key, std::int64_t second_key, std::int64_t owner) const {
    if (first_key < 0 || second_key < 0 ||
        first_key + 1 >= static_cast<std::int64_t>(offsets.size())) {
      throw UnknownIncidence{};
    }
    std::int64_t low = offsets[first_key];
    std::int64_t high = offsets[first_key + 1];
    while (low < high) {
      const std::int64_t middle = low + (high - low) / 2;
      if (second[middle] < second_key || (second[middle] == second_key && cell[middle] < owner)) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    if (low == offsets[first_key + 1] || second[low] != second_key || cell[low] != owner) {
      throw UnknownIncidence{};
    }
    return low;
  }

  std::int64_t root(std::int64_t node) {
    while (parent[node] != node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  }

  void unite(std::int64_t first, std::int64_t second_node) {
    const std::int64_t a = root(first);
    const std::int64_t b = root(second_node);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }

  // Entities (first, second key) whose incidences span several components.
  template <class InScope>
  std::int64_t split_entities(InScope&& in_scope) {
    std::int64_t split = 0;
    const std::int64_t key_count = static_cast<std::int64_t>(offsets.size()) - 1;
    for (std::int64_t key = 0; key < key_count; ++key) {
      std::int64_t start = offsets[key];
      while (start < offsets[key + 1]) {
        std::int64_t end = start + 1;
        while (end < offsets[key + 1] && second[end] == second[start]) ++end;
        if (in_scope(key, second[start])) {
          const std::int64_t base = root(start);
          for (std::int64_t node = start + 1; node < end; ++node) {
            if (root(node) != base) {
              ++split;
              break;
            }
          }
        }
        start = end;
      }
    }
    return split;
  }
};

struct Input {
  std::int32_t dimension;
  std::int64_t key_count;
  std::int64_t cell_count, audited_cell_count, cell_width;
  const std::int64_t* cells;
  std::int64_t facet_count, audited_facet_count, facet_width, audited_facet_width;
  const std::int64_t* facets;
  const std::int64_t* facet_cells;
  const std::int8_t* facet_signs;
  std::int64_t edge_count;
  const std::int64_t* edges;
  const std::int64_t* edge_cells;
  std::vector<char> scope;  // empty: every key is in scope
};

std::int64_t loop_length(const std::int64_t* row, std::int64_t width) {
  return std::count_if(row, row + width, [](std::int64_t value) { return value >= 0; });
}

// Successor and predecessor of the first least vertex of a padded loop.
std::pair<std::int64_t, std::int64_t> loop_neighbors(const std::int64_t* row, std::int64_t width) {
  const std::int64_t length = loop_length(row, width);
  // A loop without vertices reads its first (padding) entry, as the host's
  // modulo by zero gives index 0.
  if (length == 0) return {row[0], row[0]};
  std::int64_t position = 0;
  for (std::int64_t column = 1; column < width; ++column) {
    const std::int64_t value = row[column] >= 0 ? row[column] : kPad;
    const std::int64_t best = row[position] >= 0 ? row[position] : kPad;
    if (value < best) position = column;
  }
  return {row[(position + 1) % length], row[(position - 1 + length) % length]};
}

// Number of audited occurrences of every group and the inconsistent pairs.
void facet_counts(const Input& input, const Grouping& grouping, std::int64_t* counts,
                  std::int8_t* boundary, std::int64_t& open_groups) {
  std::int64_t nonmanifold = 0;
  std::int64_t inconsistent = 0;
  open_groups = 0;
  const std::int64_t groups = static_cast<std::int64_t>(grouping.starts.size()) - 1;
  for (std::int64_t group = 0; group < groups; ++group) {
    // Audited occurrences precede supplementary ones in index order.
    std::int64_t audited = 0;
    for (std::int64_t position = grouping.starts[group]; position < grouping.starts[group + 1];
         ++position) {
      if (grouping.order[position] < input.audited_facet_count) ++audited;
    }
    if (audited == 0) continue;
    const std::int64_t first = grouping.order[grouping.starts[group]];
    if (audited > 2) ++nonmanifold;
    if (audited == 1) {
      ++open_groups;
      boundary[first] = 1;
    }
    if (audited != 2) continue;
    const std::int64_t second = grouping.order[grouping.starts[group] + 1];
    const std::int64_t* a = input.facets + first * input.facet_width;
    const std::int64_t* b = input.facets + second * input.facet_width;
    bool bad = false;
    switch (input.dimension) {
      case 1:
        bad = input.facet_signs[first] == input.facet_signs[second];
        break;
      case 2:
        bad = (a[0] < a[1]) == (b[0] < b[1]);
        break;
      default:
        bad = loop_neighbors(a, input.facet_width).first !=
              loop_neighbors(b, input.facet_width).second;
        break;
    }
    if (bad) ++inconsistent;
  }
  counts[2] = nonmanifold;
  counts[3] = inconsistent;
}

// Ridges of the audited boundary facets that are not shared by exactly two.
std::int64_t open_ridges(const Input& input, const std::int8_t* boundary) {
  const std::int64_t width = input.audited_facet_width;
  if (width == 1) return 0;
  std::vector<std::int64_t> first;
  std::vector<std::int64_t> second;
  for (std::int64_t facet = 0; facet < input.audited_facet_count; ++facet) {
    if (boundary[facet] == 0) continue;
    const std::int64_t* row = input.facets + facet * input.facet_width;
    if (width == 2) {
      first.push_back(row[0]);
      first.push_back(row[1]);
      continue;
    }
    const std::int64_t length = loop_length(row, input.facet_width);
    for (std::int64_t column = 0; column < length; ++column) {
      const std::int64_t a = row[column];
      const std::int64_t b = row[(column + 1) % length];
      first.push_back(std::min(a, b));
      second.push_back(std::max(a, b));
    }
  }
  std::vector<std::int64_t> order(first.size());
  std::iota(order.begin(), order.end(), std::int64_t{0});
  const bool pairs = width != 2;
  const auto key = [&](std::int64_t index) {
    return std::pair<std::int64_t, std::int64_t>{first[index], pairs ? second[index] : 0};
  };
  std::sort(order.begin(), order.end(),
            [&](std::int64_t a, std::int64_t b) { return key(a) < key(b); });
  std::int64_t unmatched = 0;
  for (std::size_t start = 0; start < order.size();) {
    std::size_t end = start + 1;
    while (end < order.size() && key(order[end]) == key(order[start])) ++end;
    if (end - start != 2) ++unmatched;
    start = end;
  }
  return unmatched;
}

// Collapsed and duplicate audited cells.
void cell_counts(const Input& input, std::int64_t* counts) {
  std::int64_t collapsed = 0;
  const Grouping grouping =
      group_rows(input.cells, input.audited_cell_count, input.cell_width, input.key_count);
  for (std::int64_t cell = 0; cell < input.audited_cell_count; ++cell) {
    const std::int64_t* key = grouping.key(cell);
    for (std::int64_t column = 1; column < input.cell_width; ++column) {
      if (key[column] != kPad && key[column] == key[column - 1]) {
        ++collapsed;
        break;
      }
    }
  }
  counts[0] = collapsed;
  counts[1] = input.audited_cell_count - (static_cast<std::int64_t>(grouping.starts.size()) - 1);
}

bool in_scope(const Input& input, std::int64_t key) {
  return input.scope.empty() || (key >= 0 && input.scope[key] != 0);
}

// Consecutive occurrence pairs of every star facet group.
template <class Visit>
void for_each_link(const Grouping& grouping, Visit&& visit) {
  const std::int64_t groups = static_cast<std::int64_t>(grouping.starts.size()) - 1;
  for (std::int64_t group = 0; group < groups; ++group) {
    for (std::int64_t position = grouping.starts[group] + 1; position < grouping.starts[group + 1];
         ++position) {
      visit(grouping.order[position - 1], grouping.order[position]);
    }
  }
}

std::int64_t vertex_stars(const Input& input, const Grouping& grouping) {
  std::vector<std::int64_t> vertices;
  std::vector<std::int64_t> zeros;
  std::vector<std::int64_t> owners;
  for (std::int64_t cell = 0; cell < input.cell_count; ++cell) {
    for (std::int64_t column = 0; column < input.cell_width; ++column) {
      const std::int64_t vertex = input.cells[cell * input.cell_width + column];
      if (vertex < 0 || !in_scope(input, vertex)) continue;
      vertices.push_back(vertex);
      zeros.push_back(0);
      owners.push_back(cell);
    }
  }
  if (vertices.empty()) return 0;
  Incidences incidences(vertices, zeros, owners, input.key_count);
  for_each_link(grouping, [&](std::int64_t first, std::int64_t second) {
    const std::int64_t* row = input.facets + first * input.facet_width;
    for (std::int64_t column = 0; column < input.facet_width; ++column) {
      const std::int64_t vertex = row[column];
      if (vertex < 0 || !in_scope(input, vertex)) continue;
      incidences.unite(incidences.find(vertex, 0, input.facet_cells[first]),
                       incidences.find(vertex, 0, input.facet_cells[second]));
    }
  });
  return incidences.split_entities([](std::int64_t, std::int64_t) { return true; });
}

std::int64_t edge_stars(const Input& input, const Grouping& grouping) {
  std::vector<std::int64_t> lows;
  std::vector<std::int64_t> highs;
  std::vector<std::int64_t> owners;
  for (std::int64_t edge = 0; edge < input.edge_count; ++edge) {
    const std::int64_t a = std::min(input.edges[2 * edge], input.edges[2 * edge + 1]);
    const std::int64_t b = std::max(input.edges[2 * edge], input.edges[2 * edge + 1]);
    if (!in_scope(input, a) && !in_scope(input, b)) continue;
    lows.push_back(a);
    highs.push_back(b);
    owners.push_back(input.edge_cells[edge]);
  }
  if (lows.empty()) return 0;
  Incidences incidences(lows, highs, owners, input.key_count);
  for_each_link(grouping, [&](std::int64_t first, std::int64_t second) {
    const std::int64_t* row = input.facets + first * input.facet_width;
    const std::int64_t length = loop_length(row, input.facet_width);
    for (std::int64_t column = 0; column < length; ++column) {
      const std::int64_t a = std::min(row[column], row[(column + 1) % length]);
      const std::int64_t b = std::max(row[column], row[(column + 1) % length]);
      if (!in_scope(input, a) && !in_scope(input, b)) continue;
      incidences.unite(incidences.find(a, b, input.facet_cells[first]),
                       incidences.find(a, b, input.facet_cells[second]));
    }
  });
  return incidences.split_entities([](std::int64_t, std::int64_t) { return true; });
}

bool keys_bounded(const std::int64_t* values, std::int64_t count, std::int64_t key_count) {
  return std::all_of(values, values + count,
                     [&](std::int64_t value) { return value < key_count; });
}

}  // namespace

extern "C" int32_t phx_mc_welded_topology(
    int32_t dimension, int32_t embedded, int32_t check_manifold, int32_t check_watertight,
    int64_t key_count, int64_t cell_count, int64_t audited_cell_count, int64_t cell_width,
    const int64_t* cells, int64_t facet_count, int64_t audited_facet_count, int64_t facet_width,
    int64_t audited_facet_width, const int64_t* facets, const int64_t* facet_cells,
    const int8_t* facet_signs, int64_t edge_count, const int64_t* edges,
    const int64_t* edge_cells, int64_t scope_count, const int64_t* scope, int64_t* counts,
    int8_t* boundary) {
  return phx::mc::guarded([&]() -> int32_t {
    if (dimension < 1 || dimension > 3 || key_count < 0 || cell_count < 0 ||
        audited_cell_count < 0 || audited_cell_count > cell_count || cell_width < 1 ||
        facet_count < 0 || audited_facet_count < 0 || audited_facet_count > facet_count ||
        facet_width < 1 || audited_facet_width < 1 || audited_facet_width > facet_width ||
        (dimension == 2 && facet_width < 2) ||
        edge_count < 0 || scope_count < -1 || counts == nullptr ||
        key_count >= std::numeric_limits<std::int64_t>::max() - 2 ||
        !phx::mc::addressable(cell_count, cell_width, sizeof(int64_t)) ||
        !phx::mc::addressable(facet_count, facet_width, sizeof(int64_t)) ||
        !phx::mc::addressable(edge_count, 2, sizeof(int64_t)) ||
        (cell_count > 0 && cells == nullptr) ||
        (facet_count > 0 &&
         (facets == nullptr || facet_cells == nullptr || facet_signs == nullptr)) ||
        (audited_facet_count > 0 && boundary == nullptr) ||
        (edge_count > 0 && (edges == nullptr || edge_cells == nullptr)) ||
        (scope_count > 0 && scope == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!keys_bounded(cells, cell_count * cell_width, key_count) ||
        !keys_bounded(facets, facet_count * facet_width, key_count) ||
        !keys_bounded(edges, edge_count * 2, key_count) ||
        std::any_of(edges, edges + edge_count * 2, [](int64_t key) { return key < 0; }) ||
        !keys_bounded(scope, std::max<int64_t>(scope_count, 0), key_count)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    Input input{dimension, key_count, cell_count, audited_cell_count, cell_width, cells,
                facet_count, audited_facet_count, facet_width, audited_facet_width, facets,
                facet_cells, facet_signs, edge_count, edges, edge_cells, {}};
    if (scope_count >= 0) {
      input.scope.assign(static_cast<std::size_t>(key_count), 0);
      for (int64_t index = 0; index < scope_count; ++index) {
        if (scope[index] < 0) return PHX_MC_INVALID_ARGUMENT;
        input.scope[scope[index]] = 1;
      }
    }
    std::fill(counts, counts + 7, int64_t{0});
    std::fill(boundary, boundary + audited_facet_count, int8_t{0});
    const Grouping grouping = group_rows(facets, facet_count, facet_width, key_count);
    std::int64_t open_groups = 0;
    facet_counts(input, grouping, counts, boundary, open_groups);
    if (check_watertight != 0) counts[4] = embedded != 0 ? open_groups : open_ridges(input, boundary);
    cell_counts(input, counts);
    if (check_manifold != 0 && dimension >= 2) {
      try {
        counts[5] = vertex_stars(input, grouping);
        if (dimension == 3) counts[6] = edge_stars(input, grouping);
      } catch (const UnknownIncidence&) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    return PHX_MC_OK;
  });
}
