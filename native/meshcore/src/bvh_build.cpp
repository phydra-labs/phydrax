//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Native construction of the level-ordered BVH topology of phydrax's packed
// BVH (phydrax/_bvh.py, prepare_bvh): the item order and, per node in
// breadth-first level order, its contiguous item range and its children.
// Every builder is a function of the item boxes alone (ties break by item
// index), so the topology is deterministic and equals the host reference's.
//
// MEDIAN: a dividing node splits at the middle of its range after ordering its
// items by (center on the longest box axis, item); the first longest axis wins
// ties. Only the partition at the middle decides the children, and a child
// that divides again is reordered by its own key, so a dividing node is
// partitioned (nth_element under the same total order) and only children that
// stay leaves are sorted.
// SAH: per dividing node and axis, item centers fall in `bins` equal bins over
// the node's center extent; the cheapest plane (sum over both sides of the
// half boundary measure times the item count, first plane and first axis on
// ties) sends bins above it right. A node whose centers share one bin on every
// axis splits at its middle position. Each side keeps ascending item order.
// MORTON: Karras (2012) radix tree over the items sorted by (code, item),
// collapsed to leaves of at most leaf_size items; the split search of each
// internal node runs the host reference's synchronous halving schedule.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

namespace {

using phx::mc::addressable;

struct Levels {
  std::vector<int64_t> start, stop, left, right, level_offsets{0};
};

// Level-synchronous top-down build: `split(begin, end)` reorders order[begin,
// end) and returns the middle.
template <class Split>
void split_levels(int64_t count, int64_t leaf_size, Split&& split, Levels& out) {
  std::vector<int64_t> level_start{0}, level_stop{count};
  int64_t node_count = 1;
  while (!level_start.empty()) {
    std::vector<int64_t> next_start, next_stop;
    for (std::size_t k = 0; k < level_start.size(); ++k) {
      const int64_t begin = level_start[k], end = level_stop[k];
      out.start.push_back(begin);
      out.stop.push_back(end);
      if (end - begin > leaf_size) {
        out.left.push_back(node_count);
        out.right.push_back(node_count + 1);
        node_count += 2;
        const int64_t middle = split(begin, end);
        next_start.push_back(begin);
        next_stop.push_back(middle);
        next_start.push_back(middle);
        next_stop.push_back(end);
      } else {
        out.left.push_back(-1);
        out.right.push_back(-1);
      }
    }
    out.level_offsets.push_back(static_cast<int64_t>(out.start.size()));
    level_start.swap(next_start);
    level_stop.swap(next_stop);
  }
}

void median(int64_t count, int32_t dimension, const double* lower, const double* upper,
            int64_t leaf_size, int64_t* order, Levels& out) {
  std::vector<double> centers(static_cast<std::size_t>(count * dimension));
  for (int64_t i = 0; i < count * dimension; ++i) centers[i] = 0.5 * (lower[i] + upper[i]);
  split_levels(count, leaf_size, [&](int64_t begin, int64_t end) {
    int32_t axis = 0;
    double longest = -std::numeric_limits<double>::infinity();
    for (int32_t a = 0; a < dimension; ++a) {
      double low = std::numeric_limits<double>::infinity();
      double high = -std::numeric_limits<double>::infinity();
      for (int64_t k = begin; k < end; ++k) {
        low = std::min(low, lower[order[k] * dimension + a]);
        high = std::max(high, upper[order[k] * dimension + a]);
      }
      if (high - low > longest) {
        longest = high - low;
        axis = a;
      }
    }
    const auto less = [&](int64_t x, int64_t y) {
      const double cx = centers[x * dimension + axis], cy = centers[y * dimension + axis];
      return cx < cy || (cx == cy && x < y);
    };
    const int64_t middle = begin + (end - begin) / 2;
    std::nth_element(order + begin, order + middle, order + end, less);
    if (middle - begin <= leaf_size) std::sort(order + begin, order + middle, less);
    if (end - middle <= leaf_size) std::sort(order + middle, order + end, less);
    return middle;
  }, out);
}

double half_area(const double* extent, int32_t dimension) {
  if (dimension == 1) return extent[0];
  double area = 0.0;
  for (int32_t first = 0; first < dimension; ++first)
    for (int32_t second = first + 1; second < dimension; ++second)
      area = area + extent[first] * extent[second];
  return area;
}

void sah(int64_t count, int32_t dimension, const double* lower, const double* upper,
         int64_t leaf_size, int32_t bins, int64_t* order, Levels& out) {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  std::vector<double> centers(static_cast<std::size_t>(count * dimension));
  for (int64_t i = 0; i < count * dimension; ++i) centers[i] = 0.5 * (lower[i] + upper[i]);
  std::vector<int64_t> item_bins, bin_count(bins);
  std::vector<double> bin_min(static_cast<std::size_t>(bins) * dimension);
  std::vector<double> bin_max(static_cast<std::size_t>(bins) * dimension);
  std::vector<double> prefix_min(bin_min.size()), prefix_max(bin_min.size());
  std::vector<double> suffix_min(bin_min.size()), suffix_max(bin_min.size());
  std::vector<double> extent(dimension);
  std::vector<int64_t> scratch;
  split_levels(count, leaf_size, [&](int64_t begin, int64_t end) {
    const int64_t n = end - begin;
    item_bins.assign(static_cast<std::size_t>(n * dimension), 0);
    double best_cost = kInfinity;
    int32_t best_axis = 0;
    int64_t best_plane = 0;
    for (int32_t axis = 0; axis < dimension; ++axis) {
      double center_min = kInfinity, center_max = -kInfinity;
      for (int64_t k = begin; k < end; ++k) {
        center_min = std::min(center_min, centers[order[k] * dimension + axis]);
        center_max = std::max(center_max, centers[order[k] * dimension + axis]);
      }
      const double span = center_max - center_min;
      const double scale = span > 0.0 ? static_cast<double>(bins) / span : 0.0;
      std::fill(bin_count.begin(), bin_count.end(), 0);
      std::fill(bin_min.begin(), bin_min.end(), kInfinity);
      std::fill(bin_max.begin(), bin_max.end(), -kInfinity);
      for (int64_t k = begin; k < end; ++k) {
        const int64_t item = order[k];
        const double position =
            std::floor((centers[item * dimension + axis] - center_min) * scale);
        const int64_t bin = std::clamp<int64_t>(static_cast<int64_t>(position), 0, bins - 1);
        item_bins[static_cast<std::size_t>((k - begin) * dimension + axis)] = bin;
        ++bin_count[bin];
        for (int32_t a = 0; a < dimension; ++a) {
          bin_min[bin * dimension + a] = std::min(bin_min[bin * dimension + a], lower[item * dimension + a]);
          bin_max[bin * dimension + a] = std::max(bin_max[bin * dimension + a], upper[item * dimension + a]);
        }
      }
      for (int32_t a = 0; a < dimension; ++a) {
        prefix_min[a] = bin_min[a];
        prefix_max[a] = bin_max[a];
        const int64_t last = (bins - 1) * dimension + a;
        suffix_min[last] = bin_min[last];
        suffix_max[last] = bin_max[last];
      }
      for (int32_t b = 1; b < bins; ++b)
        for (int32_t a = 0; a < dimension; ++a) {
          prefix_min[b * dimension + a] = std::min(prefix_min[(b - 1) * dimension + a], bin_min[b * dimension + a]);
          prefix_max[b * dimension + a] = std::max(prefix_max[(b - 1) * dimension + a], bin_max[b * dimension + a]);
        }
      for (int32_t b = bins - 2; b >= 0; --b)
        for (int32_t a = 0; a < dimension; ++a) {
          suffix_min[b * dimension + a] = std::min(suffix_min[(b + 1) * dimension + a], bin_min[b * dimension + a]);
          suffix_max[b * dimension + a] = std::max(suffix_max[(b + 1) * dimension + a], bin_max[b * dimension + a]);
        }
      double axis_cost = kInfinity;
      int64_t plane = 0;
      int64_t left_count = 0;
      for (int32_t p = 0; p + 1 < bins; ++p) {
        left_count += bin_count[p];
        const int64_t right_count = n - left_count;
        double cost = kInfinity;
        if (left_count > 0 && right_count > 0) {
          for (int32_t a = 0; a < dimension; ++a) extent[a] = prefix_max[p * dimension + a] - prefix_min[p * dimension + a];
          const double left_area = half_area(extent.data(), dimension);
          for (int32_t a = 0; a < dimension; ++a)
            extent[a] = suffix_max[(p + 1) * dimension + a] - suffix_min[(p + 1) * dimension + a];
          cost = left_area * static_cast<double>(left_count) +
                 half_area(extent.data(), dimension) * static_cast<double>(right_count);
        }
        if (cost < axis_cost) {
          axis_cost = cost;
          plane = p;
        }
      }
      if (axis_cost < best_cost) {
        best_cost = axis_cost;
        best_axis = axis;
        best_plane = plane;
      }
    }
    const bool separable = std::isfinite(best_cost);
    // Items stay in ascending order within each side (a stable partition of a
    // segment kept in item order).
    scratch.assign(order + begin, order + end);
    int64_t write = begin;
    for (int pass = 0; pass < 2; ++pass)
      for (int64_t k = 0; k < n; ++k) {
        const bool right = separable ? item_bins[static_cast<std::size_t>(k * dimension + best_axis)] > best_plane
                                     : k >= n / 2;
        if (right == (pass == 1)) order[write++] = scratch[static_cast<std::size_t>(k)];
      }
    int64_t lefts = 0;
    for (int64_t k = 0; k < n; ++k)
      lefts += separable ? (item_bins[static_cast<std::size_t>(k * dimension + best_axis)] > best_plane ? 0 : 1)
                         : (k >= n / 2 ? 0 : 1);
    return begin + lefts;
  }, out);
}

int bit_length(uint64_t value) { return value == 0 ? 0 : 64 - std::countl_zero(value); }

void morton(int64_t count, const uint64_t* codes, int64_t leaf_size, int64_t* order, Levels& out) {
  for (int64_t i = 0; i < count; ++i) order[i] = i;
  std::sort(order, order + count, [&](int64_t x, int64_t y) {
    return codes[x] < codes[y] || (codes[x] == codes[y] && x < y);
  });
  std::vector<uint64_t> keys(static_cast<std::size_t>(count));
  for (int64_t i = 0; i < count; ++i) keys[i] = codes[order[i]];
  const int64_t internal = count - 1;
  std::vector<int64_t> range_start(static_cast<std::size_t>(2 * count - 1));
  std::vector<int64_t> range_stop(range_start.size());
  std::vector<int64_t> tree_left(range_start.size(), -1), tree_right(range_start.size(), -1);
  for (int64_t i = 0; i < count; ++i) {
    range_start[internal + i] = i;
    range_stop[internal + i] = i + 1;
  }
  const auto prefix = [&](int64_t node, int64_t other) -> int64_t {
    if (other < 0 || other >= count) return -1;
    const uint64_t difference = keys[node] ^ keys[other];
    if (difference == 0)
      return 128 - bit_length(static_cast<uint64_t>(node) ^ static_cast<uint64_t>(other));
    return 64 - bit_length(difference);
  };
  std::vector<int64_t> direction(static_cast<std::size_t>(std::max<int64_t>(internal, 0)));
  std::vector<int64_t> length(direction.size()), node_prefix(direction.size());
  int64_t longest = 0;
  for (int64_t i = 0; i < internal; ++i) {
    const int64_t d = prefix(i, i + 1) > prefix(i, i - 1) ? 1 : -1;
    const int64_t minimum = prefix(i, i - d);
    int64_t bound = 2;
    while (prefix(i, i + bound * d) > minimum) bound *= 2;
    int64_t l = 0;
    for (int64_t step = bound / 2; step >= 1; step /= 2)
      if (prefix(i, i + (l + step) * d) > minimum) l += step;
    direction[i] = d;
    length[i] = l;
    node_prefix[i] = prefix(i, i + l * d);
    longest = std::max(longest, l);
  }
  // The host's split search runs synchronously over all nodes: divisors 2, 4,
  // ... until every step ceil(length / divisor) is at most 1.
  std::vector<int64_t> divisors;
  for (int64_t divisor = 2;; divisor *= 2) {
    divisors.push_back(divisor);
    if ((longest + divisor - 1) / divisor <= 1) break;
  }
  for (int64_t i = 0; i < internal; ++i) {
    const int64_t d = direction[i], l = length[i];
    int64_t split = 0;
    for (const int64_t divisor : divisors) {
      const int64_t step = (l + divisor - 1) / divisor;
      if (prefix(i, i + (split + step) * d) > node_prefix[i]) split += step;
    }
    const int64_t gamma = i + split * d + std::min<int64_t>(d, 0);
    const int64_t other = i + l * d;
    const int64_t first = std::min(i, other), last = std::max(i, other);
    range_start[i] = first;
    range_stop[i] = last + 1;
    tree_left[i] = first == gamma ? internal + gamma : gamma;
    tree_right[i] = last == gamma + 1 ? internal + gamma + 1 : gamma + 1;
  }
  std::vector<int64_t> frontier{0};
  int64_t node_count = 1;
  while (!frontier.empty()) {
    std::vector<int64_t> next;
    for (const int64_t node : frontier) {
      const int64_t begin = range_start[node], end = range_stop[node];
      out.start.push_back(begin);
      out.stop.push_back(end);
      if (end - begin > leaf_size) {
        out.left.push_back(node_count);
        out.right.push_back(node_count + 1);
        node_count += 2;
        next.push_back(tree_left[node]);
        next.push_back(tree_right[node]);
      } else {
        out.left.push_back(-1);
        out.right.push_back(-1);
      }
    }
    out.level_offsets.push_back(static_cast<int64_t>(out.start.size()));
    frontier.swap(next);
  }
}

}  // namespace

extern "C" {

int32_t phx_mc_bvh_build(int32_t kind, int64_t item_count, int32_t dimension,
                         const double* lower, const double* upper, const uint64_t* codes,
                         int64_t leaf_size, int32_t sah_bins, int64_t* order, int64_t* start,
                         int64_t* stop, int64_t* left, int64_t* right, int64_t* level_offsets,
                         int64_t* node_count, int64_t* level_count) {
  if (item_count <= 0 || dimension <= 0 || leaf_size <= 0 ||
      !addressable(item_count, dimension, sizeof(double)) ||
      !addressable(2 * item_count, 1, sizeof(int64_t)) || order == nullptr || start == nullptr ||
      stop == nullptr || left == nullptr || right == nullptr || level_offsets == nullptr ||
      node_count == nullptr || level_count == nullptr ||
      (kind == PHX_MC_BVH_BUILD_MORTON ? codes == nullptr : (lower == nullptr || upper == nullptr)) ||
      (kind == PHX_MC_BVH_BUILD_SAH && sah_bins < 2)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  try {
    Levels levels;
    switch (kind) {
      case PHX_MC_BVH_BUILD_MEDIAN:
        for (int64_t i = 0; i < item_count; ++i) order[i] = i;
        median(item_count, dimension, lower, upper, leaf_size, order, levels);
        break;
      case PHX_MC_BVH_BUILD_SAH:
        for (int64_t i = 0; i < item_count; ++i) order[i] = i;
        sah(item_count, dimension, lower, upper, leaf_size, sah_bins, order, levels);
        break;
      case PHX_MC_BVH_BUILD_MORTON:
        morton(item_count, codes, leaf_size, order, levels);
        break;
      default:
        return PHX_MC_INVALID_ARGUMENT;
    }
    const std::size_t nodes = levels.start.size();
    std::copy(levels.start.begin(), levels.start.end(), start);
    std::copy(levels.stop.begin(), levels.stop.end(), stop);
    std::copy(levels.left.begin(), levels.left.end(), left);
    std::copy(levels.right.begin(), levels.right.end(), right);
    std::copy(levels.level_offsets.begin(), levels.level_offsets.end(), level_offsets);
    *node_count = static_cast<int64_t>(nodes);
    *level_count = static_cast<int64_t>(levels.level_offsets.size()) - 1;
  } catch (const std::bad_alloc&) {
    return PHX_MC_ALLOCATION_FAILURE;
  }
  return PHX_MC_OK;
}

}  // extern "C"
