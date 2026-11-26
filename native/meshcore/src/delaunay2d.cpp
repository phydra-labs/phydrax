//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: 2D Delaunay and regular triangulations (incremental Bowyer-Watson
// with ghost triangles and symbolic perturbation).
//
// Unweighted 2D Delaunay uses a lean self-contained fast path without
// constraint, region, live-bit, or vertex-incident tables, with a small capped
// thread-local scratch buffer (point_count <= 4096) to avoid per-call heap
// churn. Weighted regular triangulations remain on Triangulation2D.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) && \
    !defined(PHX_MC_DISABLE_NEON)
#define PHX_MC_DELAUNAY2D_NEON 1
#include <arm_neon.h>
#else
#define PHX_MC_DELAUNAY2D_NEON 0
#endif

#if defined(_MSC_VER)
#define PHX_MC_NOINLINE __declspec(noinline)
#define PHX_MC_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define PHX_MC_NOINLINE [[gnu::noinline]]
#define PHX_MC_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define PHX_MC_NOINLINE
#define PHX_MC_ALWAYS_INLINE inline
#endif

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"
#include "triangulation2d.hpp"

namespace phx::mc {

namespace {

// Upper bound on point counts served by the thread-local scratch cache.
// Larger inputs always allocate and release call-local storage so large
// buffers are never retained per thread.
inline constexpr std::size_t kSmallScratchMaxPoints = 4096;

// 32-byte aligned triangle slot combining vertices, neighbors, and cavity
// epoch mark in a single power-of-two cache-aligned record:
//   * zero cache-line straddling (2 slots per 64B line, 4 per 128B line)
//   * single shift addressing (slot << 5) instead of multiply-by-24
//   * checking mark and reading v/n touch one cache line instead of two
struct alignas(32) FastTriangle2D {
  int32_t v[3];
  std::uint32_t mark;
  int32_t n[3];
  int32_t pad;
};
static_assert(sizeof(FastTriangle2D) == 32);

struct FastBoundaryEdge {
  int32_t u;
  int32_t w;
  int32_t outside;
  int32_t back;
};

struct UnweightedScratch {
  std::vector<SpatialKey> keys;
  std::vector<SpatialKey> keys_scratch;
  std::vector<int32_t> order;
  std::vector<double> local_points;
  std::vector<int32_t> vertex_map;
  std::vector<FastTriangle2D> triangles;
  // Reused as per-vertex cone link table during Bowyer-Watson and as
  // first-vertex group offsets during canonicalization (non-overlapping phases).
  std::vector<std::uint32_t> vertex_aux;
  std::vector<int32_t> stack;
  std::vector<int32_t> cavity;
  std::vector<int32_t> created;
  std::vector<FastBoundaryEdge> boundary;
  std::vector<int32_t> free_slots;
  std::size_t cavity_cap = 0;
  std::uint32_t marks_epoch = 0;

  void ensure_cavity_capacity(std::size_t min_cap) {
    if (cavity_cap >= min_cap) return;
    cavity_cap = min_cap;
    stack.resize(3 * cavity_cap + 4);
    cavity.resize(cavity_cap);
    created.resize(3 * cavity_cap);
    boundary.resize(3 * cavity_cap);
  }
};

UnweightedScratch& small_unweighted_scratch() {
  thread_local UnweightedScratch scratch;
  return scratch;
}

// Validates all 2D input coordinates and computes their bounding box in one
// pass. Non-finite values always fail coordinate_in_domain; the cold fallback
// calls validate_points to preserve exact NONFINITE_INPUT vs RANGE_ERROR
// precedence.
int32_t validate_and_bounds_2d(const double* points, std::size_t count, double& min_x,
                               double& max_x, double& min_y, double& max_y) {
#if PHX_MC_DELAUNAY2D_NEON
  const float64x2_t vzero = vdupq_n_f64(0.0);
  const float64x2_t vmin_mag = vdupq_n_f64(kCoordinateMinMagnitude);
  const float64x2_t vmax_mag = vdupq_n_f64(kCoordinateMaxMagnitude);
  const float64x2_t p0 = vld1q_f64(points);
  float64x2_t vlo = p0;
  float64x2_t vhi = p0;
  const float64x2_t mag0 = vabsq_f64(p0);
  uint64x2_t ok = vorrq_u64(vceqq_f64(mag0, vzero),
                            vandq_u64(vcgeq_f64(mag0, vmin_mag), vcleq_f64(mag0, vmax_mag)));
  for (std::size_t i = 1; i < count; ++i) {
    const float64x2_t p = vld1q_f64(points + 2 * i);
    const float64x2_t mag = vabsq_f64(p);
    const uint64x2_t valid =
        vorrq_u64(vceqq_f64(mag, vzero),
                  vandq_u64(vcgeq_f64(mag, vmin_mag), vcleq_f64(mag, vmax_mag)));
    ok = vandq_u64(ok, valid);
    vlo = vbslq_f64(vcltq_f64(p, vlo), p, vlo);
    vhi = vbslq_f64(vcgtq_f64(p, vhi), p, vhi);
  }
  if ((vgetq_lane_u64(ok, 0) & vgetq_lane_u64(ok, 1)) != ~std::uint64_t{0}) [[unlikely]] {
    return validate_points(points, static_cast<int64_t>(count), 2, nullptr);
  }
  min_x = vgetq_lane_f64(vlo, 0);
  min_y = vgetq_lane_f64(vlo, 1);
  max_x = vgetq_lane_f64(vhi, 0);
  max_y = vgetq_lane_f64(vhi, 1);
  return PHX_MC_OK;
#else
  double lx = points[0];
  double ux = lx;
  double ly = points[1];
  double uy = ly;
  bool bad = !coordinate_in_domain(lx) || !coordinate_in_domain(ly);
  for (std::size_t i = 1; i < count; ++i) {
    const double x = points[2 * i];
    const double y = points[2 * i + 1];
    if (!coordinate_in_domain(x) || !coordinate_in_domain(y)) bad = true;
    if (x < lx) lx = x;
    if (x > ux) ux = x;
    if (y < ly) ly = y;
    if (y > uy) uy = y;
  }
  if (bad) [[unlikely]] {
    return validate_points(points, static_cast<int64_t>(count), 2, nullptr);
  }
  min_x = lx;
  max_x = ux;
  min_y = ly;
  max_y = uy;
  return PHX_MC_OK;
#endif
}

inline void insertion_sort_keys(SpatialKey* data, std::size_t begin, std::size_t end) {
  for (std::size_t at = begin + 1; at < end; ++at) {
    const SpatialKey item = data[at];
    std::size_t to = at;
    while (to > begin && data[to - 1].key > item.key) {
      data[to] = data[to - 1];
      --to;
    }
    data[to] = item;
  }
}

// Single-pass top-bit bucket sort for small point sets (<= 4096). All keys
// share the prefix above bit_width(min_key ^ max_key), so the shifted bucket
// index is monotonic in key.
template <int BucketBits>
void bucket_sort_keys(std::vector<SpatialKey>& keys, std::vector<SpatialKey>& scratch,
                      std::size_t count, std::uint64_t min_key, std::uint64_t max_key) {
  constexpr std::size_t kBuckets = std::size_t{1} << BucketBits;
  constexpr std::uint64_t kMask = kBuckets - 1;
  const int top_bit = std::bit_width(min_key ^ max_key);
  const int shift = top_bit > BucketBits ? top_bit - BucketBits : 0;
  std::array<std::uint32_t, kBuckets> counts{};
  const SpatialKey* const src = keys.data();
  for (std::size_t i = 0; i < count; ++i) {
    ++counts[(src[i].key >> shift) & kMask];
  }
  std::uint32_t total = 0;
  for (std::size_t b = 0; b < kBuckets; ++b) {
    const std::uint32_t c = counts[b];
    counts[b] = total;
    total += c;
  }
  SpatialKey* const dst = scratch.data();
  for (std::size_t i = 0; i < count; ++i) {
    dst[counts[(src[i].key >> shift) & kMask]++] = src[i];
  }
  if (shift > 0) {
    std::size_t begin = 0;
    for (std::size_t b = 0; b < kBuckets; ++b) {
      const std::size_t end = counts[b];
      const std::size_t len = end - begin;
      if (len > 1) {
        if (len <= 32) {
          insertion_sort_keys(dst, begin, end);
        } else {
          std::sort(dst + begin, dst + end,
                    [](const SpatialKey& a, const SpatialKey& b_item) { return a.key < b_item.key; });
        }
      }
      begin = end;
    }
  }
  keys.swap(scratch);
}

// Multi-pass 11-bit LSD radix sort on the top differing bits for large inputs
// (> 4096), followed by tie-break sorting on any remaining lower bits.
void radix_sort_keys_large(std::vector<SpatialKey>& keys, std::vector<SpatialKey>& scratch,
                           std::size_t count, std::uint64_t min_key, std::uint64_t max_key) {
  constexpr int kDigitBits = 11;
  constexpr std::size_t kBuckets = std::size_t{1} << kDigitBits;
  constexpr std::uint64_t kMask = kBuckets - 1;
  const int top_bit = std::bit_width(min_key ^ max_key);
  const int separated = std::min(top_bit, static_cast<int>(std::bit_width(count)) + 2);
  const int digits = (separated + kDigitBits - 1) / kDigitBits;
  const int base = std::max(0, top_bit - kDigitBits * digits);

  std::vector<std::uint32_t> histogram(static_cast<std::size_t>(digits) * kBuckets, 0U);
  const SpatialKey* const initial = keys.data();
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t shifted = initial[i].key >> base;
    for (int d = 0; d < digits; ++d) {
      const auto bucket = static_cast<std::size_t>((shifted >> (kDigitBits * d)) & kMask);
      ++histogram[static_cast<std::size_t>(d) * kBuckets + bucket];
    }
  }
  for (int d = 0; d < digits; ++d) {
    std::uint32_t* const offsets = histogram.data() + static_cast<std::size_t>(d) * kBuckets;
    const int shift = base + kDigitBits * d;
    if (offsets[(keys[0].key >> shift) & kMask] == count) continue;
    std::uint32_t total = 0;
    for (std::size_t b = 0; b < kBuckets; ++b) {
      const std::uint32_t c = offsets[b];
      offsets[b] = total;
      total += c;
    }
    const SpatialKey* const src = keys.data();
    SpatialKey* const dst = scratch.data();
    for (std::size_t i = 0; i < count; ++i) {
      dst[offsets[(src[i].key >> shift) & kMask]++] = src[i];
    }
    keys.swap(scratch);
  }
  if (base > 0) {
    SpatialKey* const data = keys.data();
    std::size_t begin = 0;
    while (begin < count) {
      std::size_t end = begin + 1;
      const std::uint64_t prefix = data[begin].key >> base;
      while (end < count && (data[end].key >> base) == prefix) ++end;
      const std::size_t len = end - begin;
      if (len > 1) {
        if (len <= 32) {
          insertion_sort_keys(data, begin, end);
        } else {
          std::sort(data + begin, data + end,
                    [](const SpatialKey& a, const SpatialKey& b_item) { return a.key < b_item.key; });
        }
      }
      begin = end;
    }
  }
}

void sort_spatial_keys(std::vector<SpatialKey>& keys, std::vector<SpatialKey>& scratch,
                       std::size_t count, std::uint64_t min_key, std::uint64_t max_key) {
  if (count < 2 || min_key == max_key) return;
  if (count <= 64) {
    std::sort(keys.data(), keys.data() + count,
              [](const SpatialKey& a, const SpatialKey& b) { return a.key < b.key; });
    return;
  }
  if (scratch.size() < count) scratch.resize(count);
  if (count <= 512) {
    bucket_sort_keys<8>(keys, scratch, count, min_key, max_key);
  } else if (count <= 2048) {
    bucket_sort_keys<10>(keys, scratch, count, min_key, max_key);
  } else if (count <= 4096) {
    bucket_sort_keys<11>(keys, scratch, count, min_key, max_key);
  } else {
    radix_sort_keys_large(keys, scratch, count, min_key, max_key);
  }
}

void compute_hilbert_keys_2d(const double* points, std::size_t count, double min_x, double max_x,
                             double min_y, double max_y, SpatialKey* keys_ptr,
                             std::uint64_t& min_key, std::uint64_t& max_key) {
  constexpr double kCells = static_cast<double>((std::uint64_t{1} << 31) - 1);
  const double extent_x = max_x - min_x;
  const double extent_y = max_y - min_y;
  min_key = std::numeric_limits<std::uint64_t>::max();
  max_key = 0;
#if PHX_MC_DELAUNAY2D_NEON
  const float64x2_t vmin = {min_x, min_y};
  const float64x2_t vext = {extent_x, extent_y};
  const float64x2_t vzero = vdupq_n_f64(0.0);
  const float64x2_t vcells = vdupq_n_f64(kCells);
  for (std::size_t i = 0; i < count; ++i) {
    const float64x2_t p = vld1q_f64(points + 2 * i);
    const float64x2_t scaled = vmulq_f64(vdivq_f64(vsubq_f64(p, vmin), vext), vcells);
    const float64x2_t clamped = vminq_f64(vmaxq_f64(scaled, vzero), vcells);
    const uint64x2_t q = vcvtq_u64_f64(clamped);
    const std::uint64_t key =
        planar_hilbert_key(static_cast<std::uint32_t>(vgetq_lane_u64(q, 0)),
                           static_cast<std::uint32_t>(vgetq_lane_u64(q, 1)));
    keys_ptr[i] = {key, static_cast<int32_t>(i), 0U};
    if (key < min_key) min_key = key;
    if (key > max_key) max_key = key;
  }
#else
  for (std::size_t i = 0; i < count; ++i) {
    const double sx = (points[2 * i] - min_x) / extent_x * kCells;
    const double sy = (points[2 * i + 1] - min_y) / extent_y * kCells;
    const auto qx = static_cast<std::uint32_t>(std::clamp(sx, 0.0, kCells));
    const auto qy = static_cast<std::uint32_t>(std::clamp(sy, 0.0, kCells));
    const std::uint64_t key = planar_hilbert_key(qx, qy);
    keys_ptr[i] = {key, static_cast<int32_t>(i), 0U};
    if (key < min_key) min_key = key;
    if (key > max_key) max_key = key;
  }
#endif
}

std::size_t deduplicate_sorted_keys_2d(const double* points, std::size_t count,
                                       SpatialKey* keys_ptr, std::vector<int32_t>& vertex_map,
                                       bool& has_duplicates) {
  has_duplicates = false;
  std::size_t written = 0;
  std::size_t start = 0;
  while (start < count) {
    std::size_t stop = start + 1;
    const std::uint64_t run_key = keys_ptr[start].key;
    while (stop < count && keys_ptr[stop].key == run_key) ++stop;
    if (stop == start + 1) [[likely]] {
      if (written != start) {
        keys_ptr[written] = keys_ptr[start];
      }
      ++written;
      start = stop;
      continue;
    }
    std::sort(keys_ptr + start, keys_ptr + stop,
              [points](const SpatialKey& left, const SpatialKey& right) {
                const double ax = points[2 * static_cast<std::size_t>(left.index)];
                const double bx = points[2 * static_cast<std::size_t>(right.index)];
                if (ax != bx) return ax < bx;
                const double ay = points[2 * static_cast<std::size_t>(left.index) + 1];
                const double by = points[2 * static_cast<std::size_t>(right.index) + 1];
                if (ay != by) return ay < by;
                return left.index < right.index;
              });
    const std::size_t run_written = written;
    std::size_t first = start;
    while (first < stop) {
      const int32_t representative = keys_ptr[first].index;
      const double rx = points[2 * static_cast<std::size_t>(representative)];
      const double ry = points[2 * static_cast<std::size_t>(representative) + 1];
      std::size_t last = first + 1;
      while (last < stop) {
        const int32_t candidate = keys_ptr[last].index;
        if (points[2 * static_cast<std::size_t>(candidate)] != rx ||
            points[2 * static_cast<std::size_t>(candidate) + 1] != ry) {
          break;
        }
        ++last;
      }
      if (last > first + 1) {
        if (!has_duplicates) {
          has_duplicates = true;
          vertex_map.resize(count);
          std::iota(vertex_map.begin(), vertex_map.end(), int32_t{0});
        }
        for (std::size_t k = first + 1; k < last; ++k) {
          vertex_map[static_cast<std::size_t>(keys_ptr[k].index)] = representative;
        }
      }
      keys_ptr[written++] = keys_ptr[first];
      first = last;
    }
    if (written - run_written > 1) {
      std::sort(keys_ptr + run_written, keys_ptr + written,
                [](const SpatialKey& left, const SpatialKey& right) {
                  return left.index < right.index;
                });
    }
    start = stop;
  }
  return written;
}

// Computes 2D Hilbert keys, deduplicates identical input points (smallest
// index wins), builds the exact BRIO+Hilbert insertion order, and gathers the
// distinct coordinates once in insertion order.
int32_t prepare_unweighted_points(const double* points, std::size_t count, double min_x,
                                  double max_x, double min_y, double max_y,
                                  UnweightedScratch& scratch, std::size_t& distinct_count,
                                  bool& has_duplicates) {
  if (scratch.keys.size() < count) scratch.keys.resize(count);
  std::uint64_t min_key = 0;
  std::uint64_t max_key = 0;
  compute_hilbert_keys_2d(points, count, min_x, max_x, min_y, max_y, scratch.keys.data(), min_key,
                          max_key);
  sort_spatial_keys(scratch.keys, scratch.keys_scratch, count, min_key, max_key);
  SpatialKey* const keys_ptr = scratch.keys.data();

  const std::size_t written =
      deduplicate_sorted_keys_2d(points, count, keys_ptr, scratch.vertex_map, has_duplicates);
  distinct_count = written;
  if (written < 3) {
    return PHX_MC_DEGENERATE_INPUT;
  }

  const int rounds = brio_round_count(written);
  std::array<std::size_t, 64> starts{};
  for (std::size_t k = 0; k < written; ++k) {
    const std::uint32_t round = brio_round(keys_ptr[k].index, rounds);
    keys_ptr[k].round = round;
    ++starts[round + 1];
  }
  for (int r = 0; r < rounds; ++r) {
    starts[static_cast<std::size_t>(r + 1)] += starts[static_cast<std::size_t>(r)];
  }
  if (scratch.order.size() < written) scratch.order.resize(written);
  int32_t* const order = scratch.order.data();
  for (std::size_t k = 0; k < written; ++k) {
    order[starts[keys_ptr[k].round]++] = keys_ptr[k].index;
  }

  if (scratch.local_points.size() < 2 * written) scratch.local_points.resize(2 * written);
  double* const local_points = scratch.local_points.data();
  for (std::size_t k = 0; k < written; ++k) {
    const std::size_t src = 2 * static_cast<std::size_t>(order[k]);
    local_points[2 * k] = points[src];
    local_points[2 * k + 1] = points[src + 1];
  }
  return PHX_MC_OK;
}

// Cold helpers kept out-of-line so hot walk and cavity loops avoid function
// calls and register spilling on their common paths.
PHX_MC_NOINLINE unsigned exit_edge_exact(const FastTriangle2D& tri, const double* coords,
                                           const double* q, unsigned uncertain) {
  unsigned extra = 0;
  for (int k = 0; k < 3; ++k) {
    if (((uncertain >> k) & 1U) != 0 &&
        orient2d(coords + 2 * static_cast<std::size_t>(tri.v[next3(k)]),
                 coords + 2 * static_cast<std::size_t>(tri.v[prev3(k)]), q) < 0) {
      extra |= 1U << k;
    }
  }
  return extra;
}

PHX_MC_NOINLINE bool conflict_sos_cold(const FastTriangle2D& tri, const double* coords,
                                         const double* q, const int32_t* identities, int32_t p) {
  return incircle_sos(coords + 2 * static_cast<std::size_t>(tri.v[0]),
                      coords + 2 * static_cast<std::size_t>(tri.v[1]),
                      coords + 2 * static_cast<std::size_t>(tri.v[2]), q,
                      identities[static_cast<std::size_t>(tri.v[0])],
                      identities[static_cast<std::size_t>(tri.v[1])],
                      identities[static_cast<std::size_t>(tri.v[2])],
                      identities[static_cast<std::size_t>(p)]) > 0;
}

PHX_MC_NOINLINE bool conflict_ghost_cold(const FastTriangle2D* slots, const FastTriangle2D& tri,
                                           const double* coords, const double* q,
                                           const int32_t* identities, int32_t p) {
  const int side = orient2d_filtered(coords + 2 * static_cast<std::size_t>(tri.v[0]),
                                     coords + 2 * static_cast<std::size_t>(tri.v[1]), q);
  if (side != 0) {
    return side > 0;
  }
  const FastTriangle2D& fin = slots[tri.n[2]];
  const double* a = coords + 2 * static_cast<std::size_t>(fin.v[0]);
  const double* b = coords + 2 * static_cast<std::size_t>(fin.v[1]);
  const double* c = coords + 2 * static_cast<std::size_t>(fin.v[2]);
  const int sign = incircle_stage(a, b, c, q);
  if (sign != 0) return sign > 0;
  return conflict_sos_cold(fin, coords, q, identities, p);
}

PHX_MC_NOINLINE int32_t allocate_slot_slow(UnweightedScratch& scratch, int32_t& tri_count,
                                             FastTriangle2D*& slots, std::size_t& tri_cap) {
  if (!scratch.free_slots.empty()) {
    const int32_t slot = scratch.free_slots.back();
    scratch.free_slots.pop_back();
    return slot;
  }
  if (static_cast<int64_t>(tri_count) >= kMaxMeshPoints) {
    throw std::length_error("triangulation2d slot index exceeds int32 range");
  }
  const std::size_t next_size = std::min<std::size_t>(
      static_cast<std::size_t>(kMaxMeshPoints), scratch.triangles.size() * 2 + 8);
  scratch.triangles.resize(next_size);
  slots = scratch.triangles.data();
  tri_cap = next_size;
  return tri_count++;
}

PHX_MC_NOINLINE void grow_cavity_buffers(UnweightedScratch& scratch, std::size_t& cavity_cap,
                                           int32_t*& stack, int32_t*& cavity,
                                           FastBoundaryEdge*& boundary, int32_t*& created) {
  scratch.ensure_cavity_capacity(scratch.cavity_cap * 2);
  cavity_cap = scratch.cavity_cap;
  stack = scratch.stack.data();
  cavity = scratch.cavity.data();
  boundary = scratch.boundary.data();
  created = scratch.created.data();
}

PHX_MC_NOINLINE void reset_marks_epoch(UnweightedScratch& scratch) {
  for (FastTriangle2D& tri : scratch.triangles) {
    tri.mark = 0U;
  }
  scratch.marks_epoch = 0U;
}

PHX_MC_NOINLINE void release_excess_cavity(UnweightedScratch& scratch, FastTriangle2D* slots,
                                             const int32_t* cavity, std::size_t excess) {
  for (std::size_t c = 0; c < excess; ++c) {
    const int32_t dead = cavity[c];
    slots[dead].v[2] = kInfiniteVertex;
    scratch.free_slots.push_back(dead);
  }
}

#if PHX_MC_DELAUNAY2D_NEON
// Evaluates exact IEEE-754 binary64 Shewchuk stage-A incircle filter using
// 128-bit NEON lanes with -ffp-contract=off.
PHX_MC_ALWAYS_INLINE bool finite_conflict_fast(const FastTriangle2D& tri,
                                                        const double* __restrict coords,
                                                        float64x2_t vq, const double* q,
                                                        const int32_t* identities, int32_t p) {
  const float64x2_t va = vsubq_f64(vld1q_f64(coords + 2 * static_cast<std::size_t>(tri.v[0])), vq);
  const float64x2_t vb = vsubq_f64(vld1q_f64(coords + 2 * static_cast<std::size_t>(tri.v[1])), vq);
  const float64x2_t vc = vsubq_f64(vld1q_f64(coords + 2 * static_cast<std::size_t>(tri.v[2])), vq);

  const float64x2_t sq_a = vmulq_f64(va, va);
  const float64x2_t sq_b = vmulq_f64(vb, vb);
  const float64x2_t sq_c = vmulq_f64(vc, vc);
  const float64x2_t lift01 = vpaddq_f64(sq_a, sq_b);
  const double clift = vaddvq_f64(sq_c);

  const float64x2_t bc = vmulq_f64(vb, vextq_f64(vc, vc, 1));
  const float64x2_t ca = vmulq_f64(vc, vextq_f64(va, va, 1));
  const float64x2_t ab = vmulq_f64(va, vextq_f64(vb, vb, 1));

  const float64x2_t left01 = vuzp1q_f64(bc, ca);
  const float64x2_t right01 = vuzp2q_f64(bc, ca);
  const float64x2_t det01 = vmulq_f64(lift01, vsubq_f64(left01, right01));
  const float64x2_t perm01 =
      vmulq_f64(vaddq_f64(vabsq_f64(left01), vabsq_f64(right01)), lift01);

  const double ab_diff = vgetq_lane_f64(ab, 0) - vgetq_lane_f64(ab, 1);
  const double ab_abs = vaddvq_f64(vabsq_f64(ab));
  const double det = vaddvq_f64(det01) + clift * ab_diff;
  const double permanent = vaddvq_f64(perm01) + ab_abs * clift;

  constexpr double kEpsilon = 0x1p-53;
  const double errbound = (10.0 + 96.0 * kEpsilon) * kEpsilon * permanent;
  if (std::fabs(det) > errbound) [[likely]] {
    return det > 0.0;
  }
  return conflict_sos_cold(tri, coords, q, identities, p);
}
#endif

// Stochastic visibility walk matching Triangulation2D::walk bit-for-bit.
// Evaluates splitmix64 only when 2+ edges are negative (when 0 or 1 edge is
// negative, the exit edge does not depend on `first` while `step` still
// advances identically).
PHX_MC_ALWAYS_INLINE int32_t walk_2d(const FastTriangle2D* __restrict slots,
                                              const double* __restrict coords, int32_t start,
                                              const double* __restrict q,
                                              std::uint64_t& walk_step) {
  int32_t t = start;
  if (slots[t].v[2] == kInfiniteVertex) [[unlikely]] {
    t = slots[t].n[2];
  }
  int32_t previous = -1;
  std::uint64_t step = walk_step;
  constexpr double kEpsilon = 0x1p-53;
  constexpr double kBound = (3.0 + 16.0 * kEpsilon) * kEpsilon;
  // Packed 2-bit table for kExit[first][negative] (16 bits per `first`):
  //   first=0: {0,0,1,0,2,0,1,0} = 0x1210
  //   first=1: {0,0,1,1,2,2,1,1} = 0x5A50
  //   first=2: {0,0,1,0,2,2,2,2} = 0xAA10
  constexpr std::uint64_t kPackedExit = 0xAA105A501210ULL;
#if PHX_MC_DELAUNAY2D_NEON
  const float64x2_t vq = vld1q_f64(q);
  const float64x2_t vkbound = vdupq_n_f64(kBound);
  const uint64x2_t vbits01 = {1ULL, 2ULL};
#else
  const double qx = q[0];
  const double qy = q[1];
#endif
  for (;;) {
    const FastTriangle2D& tri = slots[t];
    const int32_t v0 = tri.v[0];
    const int32_t v1 = tri.v[1];
    const int32_t v2 = tri.v[2];
    const int32_t n0 = tri.n[0];
    const int32_t n1 = tri.n[1];
    const int32_t n2 = tri.n[2];
    const std::uint64_t cur_step = step++;
    const unsigned valid = static_cast<unsigned>(n0 != previous) |
                           (static_cast<unsigned>(n1 != previous) << 1) |
                           (static_cast<unsigned>(n2 != previous) << 2);
    unsigned negative = 0;
    unsigned uncertain = 0;
#if PHX_MC_DELAUNAY2D_NEON
    const float64x2_t va = vsubq_f64(vld1q_f64(coords + 2 * static_cast<std::size_t>(v0)), vq);
    const float64x2_t vb = vsubq_f64(vld1q_f64(coords + 2 * static_cast<std::size_t>(v1)), vq);
    const float64x2_t vc = vsubq_f64(vld1q_f64(coords + 2 * static_cast<std::size_t>(v2)), vq);
    const float64x2_t bc = vmulq_f64(vb, vextq_f64(vc, vc, 1));
    const float64x2_t ca = vmulq_f64(vc, vextq_f64(va, va, 1));
    const float64x2_t ab = vmulq_f64(va, vextq_f64(vb, vb, 1));

    const float64x2_t left01 = vuzp1q_f64(bc, ca);
    const float64x2_t right01 = vuzp2q_f64(bc, ca);
    const float64x2_t det01 = vsubq_f64(left01, right01);
    const float64x2_t eb01 = vmulq_f64(vaddq_f64(vabsq_f64(left01), vabsq_f64(right01)), vkbound);
    const uint64x2_t neg01 = vandq_u64(vcgtq_f64(vnegq_f64(det01), eb01), vbits01);
    const uint64x2_t unc01 = vandq_u64(vcleq_f64(vabsq_f64(det01), eb01), vbits01);
    const uint64x2_t masks01 = vpaddq_u64(neg01, unc01);

    const double l2 = vgetq_lane_f64(ab, 0);
    const double r2 = vgetq_lane_f64(ab, 1);
    const double det2 = l2 - r2;
    const double eb2 = kBound * (std::fabs(l2) + std::fabs(r2));

    negative = static_cast<unsigned>(vgetq_lane_u64(masks01, 0)) |
               (static_cast<unsigned>(-det2 > eb2) << 2);
    uncertain = static_cast<unsigned>(vgetq_lane_u64(masks01, 1)) |
                (static_cast<unsigned>(std::fabs(det2) <= eb2) << 2);
#else
    const double* a = coords + 2 * static_cast<std::size_t>(v0);
    const double* b = coords + 2 * static_cast<std::size_t>(v1);
    const double* c = coords + 2 * static_cast<std::size_t>(v2);
    const double ax = a[0] - qx, ay = a[1] - qy;
    const double bx = b[0] - qx, by = b[1] - qy;
    const double cx = c[0] - qx, cy = c[1] - qy;
    const double left[3] = {bx * cy, cx * ay, ax * by};
    const double right[3] = {by * cx, cy * ax, ay * bx};
    for (int k = 0; k < 3; ++k) {
      const double det = left[k] - right[k];
      const double errbound = kBound * (std::fabs(left[k]) + std::fabs(right[k]));
      negative |= static_cast<unsigned>(-det > errbound) << k;
      uncertain |= static_cast<unsigned>(std::fabs(det) <= errbound) << k;
    }
#endif
    uncertain &= valid;
    if (uncertain != 0) [[unlikely]] {
      negative |= exit_edge_exact(tri, coords, q, uncertain);
    }
    negative &= valid;
    if (negative == 0) {
      walk_step = step;
      return t;
    }
    int crossed;
    if ((negative & (negative - 1U)) == 0) [[likely]] {
      crossed = std::countr_zero(negative);
    } else {
      const unsigned first = static_cast<unsigned>(splitmix64(cur_step) % 3U);
      crossed = static_cast<int>((kPackedExit >> (16U * first + 2U * negative)) & 3U);
    }
    previous = t;
    t = crossed == 0 ? n0 : crossed == 1 ? n1 : n2;
    if (slots[t].v[2] == kInfiniteVertex) [[unlikely]] {
      walk_step = step;
      return t;
    }
  }
}

PHX_MC_NOINLINE void normalize_ghost_slot(FastTriangle2D* slots, int32_t gslot) noexcept {
  FastTriangle2D& tri = slots[gslot];
  const int32_t v0 = tri.v[0], v1 = tri.v[1], v2 = tri.v[2];
  const int32_t n0 = tri.n[0], n1 = tri.n[1], n2 = tri.n[2];
  if (v0 == kInfiniteVertex) {
    tri.v[0] = v1;
    tri.v[1] = v2;
    tri.v[2] = v0;
    tri.n[0] = n1;
    tri.n[1] = n2;
    tri.n[2] = n0;
  } else {
    tri.v[0] = v2;
    tri.v[1] = v0;
    tri.v[2] = v1;
    tri.n[0] = n2;
    tri.n[1] = n0;
    tri.n[2] = n1;
  }
}

void initialize_fast_simplex(FastTriangle2D* slots, int32_t a, int32_t b, int32_t c) noexcept {
  // Slot 0: finite (a, b, c). Slots 1..3: ghosts across a->b, b->c, c->a.
  // Existing marks stay <= marks_epoch, so no overwrite is needed.
  slots[0].v[0] = a;
  slots[0].v[1] = b;
  slots[0].v[2] = c;
  slots[0].n[0] = 2;
  slots[0].n[1] = 3;
  slots[0].n[2] = 1;
  slots[1].v[0] = b;
  slots[1].v[1] = a;
  slots[1].v[2] = kInfiniteVertex;
  slots[1].n[0] = 3;
  slots[1].n[1] = 2;
  slots[1].n[2] = 0;
  slots[2].v[0] = c;
  slots[2].v[1] = b;
  slots[2].v[2] = kInfiniteVertex;
  slots[2].n[0] = 1;
  slots[2].n[1] = 3;
  slots[2].n[2] = 0;
  slots[3].v[0] = a;
  slots[3].v[1] = c;
  slots[3].v[2] = kInfiniteVertex;
  slots[3].n[0] = 2;
  slots[3].n[1] = 1;
  slots[3].n[2] = 0;
}

PHX_MC_ALWAYS_INLINE void expand_cavity_2d(
    UnweightedScratch& scratch, FastTriangle2D* __restrict slots, const double* __restrict coords,
    const int32_t* __restrict identities, int32_t start, int32_t p, const double* q,
    std::uint32_t inside, std::uint32_t outside, std::size_t& cavity_cap, int32_t*& stack,
    int32_t*& cavity, FastBoundaryEdge*& boundary, int32_t*& created, std::size_t& removed,
    std::size_t& edges, int64_t& finite_removed) {
  std::size_t top = 1;
  removed = 0;
  edges = 0;
  finite_removed = 0;
  stack[0] = start;
  slots[start].mark = inside;
#if PHX_MC_DELAUNAY2D_NEON
  const float64x2_t vq = vld1q_f64(q);
#endif
  while (top != 0) {
    if (removed == cavity_cap) [[unlikely]] {
      grow_cavity_buffers(scratch, cavity_cap, stack, cavity, boundary, created);
    }
    const int32_t t = stack[--top];
    cavity[removed++] = t;
    const FastTriangle2D& cur = slots[t];
    const int32_t v0 = cur.v[0], v1 = cur.v[1], v2 = cur.v[2];
    const int32_t n0 = cur.n[0], n1 = cur.n[1], n2 = cur.n[2];
    finite_removed += v2 != kInfiniteVertex ? 1 : 0;

    const auto visit_neighbor = [&](int32_t nb, int32_t u, int32_t w) {
      FastTriangle2D& nbtri = slots[nb];
      const std::uint32_t mark = nbtri.mark;
      if (mark == inside) return;
      if (mark != outside) {
        bool in_conflict;
        if (nbtri.v[2] == kInfiniteVertex) [[unlikely]] {
          in_conflict = conflict_ghost_cold(slots, nbtri, coords, q, identities, p);
        } else {
#if PHX_MC_DELAUNAY2D_NEON
          in_conflict = finite_conflict_fast(nbtri, coords, vq, q, identities, p);
#else
          const int sign = incircle_stage(
              coords + 2 * static_cast<std::size_t>(nbtri.v[0]),
              coords + 2 * static_cast<std::size_t>(nbtri.v[1]),
              coords + 2 * static_cast<std::size_t>(nbtri.v[2]), q);
          in_conflict =
              sign != 0 ? (sign > 0) : conflict_sos_cold(nbtri, coords, q, identities, p);
#endif
        }
        if (in_conflict) {
          nbtri.mark = inside;
          stack[top++] = nb;
          return;
        }
        nbtri.mark = outside;
      }
      const int back = nbtri.n[0] == t ? 0 : nbtri.n[1] == t ? 1 : 2;
      boundary[edges++] = {u, w, nb, back};
    };

    visit_neighbor(n0, v1, v2);
    visit_neighbor(n1, v2, v0);
    visit_neighbor(n2, v0, v1);
  }
}

PHX_MC_ALWAYS_INLINE void link_cavity_cone_2d(
    UnweightedScratch& scratch, FastTriangle2D*& slots, std::size_t& tri_cap,
    std::uint32_t* __restrict vertex_slots, const int32_t* cavity, const FastBoundaryEdge* boundary,
    int32_t* created, std::size_t removed, std::size_t edges, int32_t p, int64_t finite_removed,
    int32_t& tri_count, int64_t& finite_count, int32_t& hint) {
  const std::size_t reused = std::min(removed, edges);
  if (removed > reused) [[unlikely]] {
    release_excess_cavity(scratch, slots, cavity, removed - reused);
  }
  const bool has_free = !scratch.free_slots.empty();
  int32_t ghost0 = -1;
  int32_t ghost1 = -1;
  int ghost_count = 0;
  for (std::size_t e = 0; e < edges; ++e) {
    const FastBoundaryEdge& edge = boundary[e];
    int32_t t;
    if (e < reused) {
      t = cavity[removed - 1 - e];
    } else if (!has_free && static_cast<std::size_t>(tri_count) < tri_cap) [[likely]] {
      t = tri_count++;
    } else {
      t = allocate_slot_slow(scratch, tri_count, slots, tri_cap);
    }
    created[e] = t;
    FastTriangle2D& tri = slots[t];
    tri.v[0] = edge.u;
    tri.v[1] = edge.w;
    tri.v[2] = p;
    tri.n[2] = edge.outside;
    slots[edge.outside].n[edge.back] = t;
    vertex_slots[static_cast<std::size_t>(edge.u + 1)] = static_cast<std::uint32_t>(t);
    if ((edge.u | edge.w) < 0) [[unlikely]] {
      if ((ghost_count++ & 1) == 0) {
        ghost0 = t;
      } else {
        ghost1 = t;
      }
    } else {
      hint = t;
    }
  }
  finite_count += static_cast<int64_t>(edges) - ghost_count - finite_removed;
  for (std::size_t e = 0; e < edges; ++e) {
    const int32_t t = created[e];
    const auto other = static_cast<int32_t>(
        vertex_slots[static_cast<std::size_t>(boundary[e].w + 1)]);
    slots[t].n[0] = other;
    slots[other].n[1] = t;
  }
  if (ghost_count != 0) [[unlikely]] {
    normalize_ghost_slot(slots, ghost0);
    if (ghost_count > 1) normalize_ghost_slot(slots, ghost1);
  }
}

// Lean unweighted 2D Bowyer-Watson builder operating directly on
// UnweightedScratch. Matches Triangulation2D's exact walk, predicate calls,
// symbolic perturbation, slot reuse order, and ghost convention.
int32_t build_fast_delaunay_2d(UnweightedScratch& scratch, std::size_t count,
                               int32_t& out_tri_count, int64_t& out_finite_count) {
  const double* __restrict coords = scratch.local_points.data();
  const int32_t* __restrict identities = scratch.order.data();

  const int32_t a = 0;
  int32_t b = 1;
  std::size_t third = 2;
  int side = 0;
  for (; third < count; ++third) {
    side = orient2d(coords, coords + 2, coords + 2 * third);
    if (side != 0) break;
  }
  if (third == count) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  int32_t c = static_cast<int32_t>(third);
  if (side < 0) {
    std::swap(b, c);
  }

  const std::size_t initial_slots = 2 * count + 4;
  if (scratch.triangles.size() < initial_slots) {
    scratch.triangles.resize(initial_slots);
  }
  const std::uint64_t epoch_budget =
      2 * std::min<std::size_t>(count, std::size_t{1000000000}) + 8;
  if (scratch.marks_epoch > std::numeric_limits<std::uint32_t>::max() - epoch_budget)
      [[unlikely]] {
    reset_marks_epoch(scratch);
  }
  if (scratch.vertex_aux.size() < count + 1) {
    scratch.vertex_aux.resize(count + 1);
  }
  scratch.free_slots.clear();
  scratch.ensure_cavity_capacity(64);

  FastTriangle2D* slots = scratch.triangles.data();
  std::uint32_t* __restrict vertex_slots = scratch.vertex_aux.data();
  int32_t* stack = scratch.stack.data();
  int32_t* cavity = scratch.cavity.data();
  FastBoundaryEdge* boundary = scratch.boundary.data();
  int32_t* created = scratch.created.data();
  std::size_t cavity_cap = scratch.cavity_cap;
  std::size_t tri_cap = scratch.triangles.size();

  initialize_fast_simplex(slots, a, b, c);
  int32_t tri_count = 4;
  int64_t finite_count = 1;
  int32_t hint = 0;
  std::uint64_t walk_step = 0;

  for (std::size_t idx = 2; idx < count; ++idx) {
    if (idx == third) [[unlikely]] continue;
    const auto p = static_cast<int32_t>(idx);
    const double* const q = coords + 2 * idx;
    const int32_t start = walk_2d(slots, coords, hint, q, walk_step);

    if (scratch.marks_epoch >= std::numeric_limits<std::uint32_t>::max() - 4U) [[unlikely]] {
      reset_marks_epoch(scratch);
    }
    scratch.marks_epoch += 2U;
    const std::uint32_t inside = scratch.marks_epoch;
    const std::uint32_t outside = inside + 1U;

    std::size_t removed = 0;
    std::size_t edges = 0;
    int64_t finite_removed = 0;
    expand_cavity_2d(scratch, slots, coords, identities, start, p, q, inside, outside, cavity_cap,
                     stack, cavity, boundary, created, removed, edges, finite_removed);
    link_cavity_cone_2d(scratch, slots, tri_cap, vertex_slots, cavity, boundary, created, removed,
                        edges, p, finite_removed, tri_count, finite_count, hint);
  }

  out_tri_count = tri_count;
  out_finite_count = finite_count;
  return PHX_MC_OK;
}

// Rotates every live finite triangle to its smallest input vertex and writes
// them in canonical lexicographic order directly into `cells_out`. Performs no
// heap allocations once `scratch.vertex_aux` is sized, keeping caller output
// writes exception-safe and atomic.
void emit_canonical_triangles(FastTriangle2D* slots, int32_t tri_count, const int32_t* identities,
                              std::size_t point_count, UnweightedScratch& scratch,
                              int32_t* cells_out) {
  const std::size_t offset_len = point_count + 1;
  std::uint32_t* const offsets = scratch.vertex_aux.data();
  std::memset(offsets, 0, offset_len * sizeof(std::uint32_t));

  int32_t largest_first = 0;
  for (int32_t t = 0; t < tri_count; ++t) {
    FastTriangle2D& tri = slots[t];
    if (tri.v[2] < 0) continue;
    const int32_t a = identities[static_cast<std::size_t>(tri.v[0])];
    const int32_t b = identities[static_cast<std::size_t>(tri.v[1])];
    const int32_t c = identities[static_cast<std::size_t>(tri.v[2])];
    const bool second = (b < a) & (b < c);
    const bool third = (c < a) & (c < b);
    const int32_t r0 = second ? b : third ? c : a;
    const int32_t r1 = second ? c : third ? a : b;
    const int32_t r2 = second ? a : third ? b : c;
    tri.v[0] = r0;
    tri.v[1] = r1;
    tri.v[2] = r2;
    ++offsets[static_cast<std::size_t>(r0) + 1];
    if (r0 > largest_first) largest_first = r0;
  }

  const std::size_t groups = static_cast<std::size_t>(largest_first) + 1;
  std::uint32_t running = 0;
  for (std::size_t v = 0; v <= groups; ++v) {
    running += offsets[v];
    offsets[v] = running;
  }

  for (int32_t t = 0; t < tri_count; ++t) {
    const FastTriangle2D& tri = slots[t];
    if (tri.v[2] < 0) continue;
    const std::size_t dst = offsets[static_cast<std::size_t>(tri.v[0])]++;
    int32_t* const out = cells_out + 3 * dst;
    out[0] = tri.v[0];
    out[1] = tri.v[1];
    out[2] = tri.v[2];
  }

  // offsets[v] now marks the end of first-vertex group v in cells_out.
  std::size_t begin = 0;
  for (std::size_t vertex = 0; vertex < groups; ++vertex) {
    const std::size_t end = offsets[vertex];
    const std::size_t len = end - begin;
    if (len == 2) {
      int32_t* const p0 = cells_out + 3 * begin;
      int32_t* const p1 = p0 + 3;
      const std::uint64_t k0 =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(p0[1])) << 32) |
          static_cast<std::uint32_t>(p0[2]);
      const std::uint64_t k1 =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(p1[1])) << 32) |
          static_cast<std::uint32_t>(p1[2]);
      if (k1 < k0) {
        std::swap(p0[1], p1[1]);
        std::swap(p0[2], p1[2]);
      }
    } else if (len > 2 && len <= 32) {
      for (std::size_t at = begin + 1; at < end; ++at) {
        const int32_t b = cells_out[3 * at + 1];
        const int32_t c = cells_out[3 * at + 2];
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(b)) << 32) |
            static_cast<std::uint32_t>(c);
        std::size_t to = at;
        while (to > begin) {
          const int32_t pb = cells_out[3 * (to - 1) + 1];
          const int32_t pc = cells_out[3 * (to - 1) + 2];
          const std::uint64_t pkey =
              (static_cast<std::uint64_t>(static_cast<std::uint32_t>(pb)) << 32) |
              static_cast<std::uint32_t>(pc);
          if (pkey <= key) break;
          cells_out[3 * to + 1] = pb;
          cells_out[3 * to + 2] = pc;
          --to;
        }
        cells_out[3 * to + 1] = b;
        cells_out[3 * to + 2] = c;
      }
    } else if (len > 32) [[unlikely]] {
      // Reuse scratch.keys (len <= distinct_count - 1 < point_count <= keys.size()).
      SpatialKey* const hub = scratch.keys.data();
      for (std::size_t i = 0; i < len; ++i) {
        const int32_t* const row = cells_out + 3 * (begin + i);
        hub[i].key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row[1])) << 32) |
                     static_cast<std::uint32_t>(row[2]);
      }
      std::sort(hub, hub + len,
                [](const SpatialKey& a, const SpatialKey& b_item) { return a.key < b_item.key; });
      for (std::size_t i = 0; i < len; ++i) {
        int32_t* const row = cells_out + 3 * (begin + i);
        row[1] = static_cast<int32_t>(hub[i].key >> 32);
        row[2] = static_cast<int32_t>(hub[i].key & 0xFFFFFFFFULL);
      }
    }
    begin = end;
  }
}

void emit_raw_triangles(const FastTriangle2D* slots, int32_t tri_count, const int32_t* identities,
                        int32_t* cells_out) {
  int32_t* out = cells_out;
  for (int32_t t = 0; t < tri_count; ++t) {
    const FastTriangle2D& tri = slots[t];
    if (tri.v[2] < 0) continue;
    out[0] = identities[static_cast<std::size_t>(tri.v[0])];
    out[1] = identities[static_cast<std::size_t>(tri.v[1])];
    out[2] = identities[static_cast<std::size_t>(tri.v[2])];
    out += 3;
  }
}

int32_t run_unweighted_2d(int64_t point_count, const double* points, UnweightedScratch& scratch,
                          int32_t& tri_count, int64_t& finite_count, bool& has_duplicates) {
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  const auto count = static_cast<std::size_t>(point_count);
  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;
  int32_t status = validate_and_bounds_2d(points, count, min_x, max_x, min_y, max_y);
  if (status != PHX_MC_OK) {
    return status;
  }
  if (count < 3 || max_x == min_x || max_y == min_y) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  std::size_t distinct_count = 0;
  status = prepare_unweighted_points(points, count, min_x, max_x, min_y, max_y, scratch,
                                     distinct_count, has_duplicates);
  if (status != PHX_MC_OK) {
    return status;
  }
  return build_fast_delaunay_2d(scratch, distinct_count, tri_count, finite_count);
}

int32_t triangulate_2d_unweighted(int64_t point_count, const double* points, bool canonical,
                                  phx_mc_mesh** mesh) {
  const auto count = static_cast<std::size_t>(point_count);
  UnweightedScratch local_scratch;
  UnweightedScratch& scratch =
      count <= kSmallScratchMaxPoints ? small_unweighted_scratch() : local_scratch;
  int32_t tri_count = 0;
  int64_t finite_count = 0;
  bool has_duplicates = false;
  const int32_t status =
      run_unweighted_2d(point_count, points, scratch, tri_count, finite_count, has_duplicates);
  if (status != PHX_MC_OK) {
    return status;
  }
  if (canonical && scratch.vertex_aux.size() < count + 1) {
    scratch.vertex_aux.resize(count + 1);
  }
  auto result = std::make_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  result->points.assign(points, points + 2 * count);
  if (has_duplicates) {
    if (count > kSmallScratchMaxPoints) {
      result->vertex_map = std::move(scratch.vertex_map);
    } else {
      result->vertex_map.assign(scratch.vertex_map.data(), scratch.vertex_map.data() + count);
    }
  } else {
    result->vertex_map.resize(count);
    std::iota(result->vertex_map.begin(), result->vertex_map.end(), int32_t{0});
  }
  result->cells.resize(static_cast<std::size_t>(3 * finite_count));
  if (canonical) {
    emit_canonical_triangles(scratch.triangles.data(), tri_count, scratch.order.data(), count,
                             scratch, result->cells.data());
  } else {
    emit_raw_triangles(scratch.triangles.data(), tri_count, scratch.order.data(),
                       result->cells.data());
  }
  *mesh = result.release();
  return PHX_MC_OK;
}

int32_t triangulate_2d_weighted(int64_t point_count, const double* points, const double* weights,
                                bool canonical, phx_mc_mesh** mesh) {
  int32_t status = validate_points(points, point_count, 2, weights);
  if (status != PHX_MC_OK) {
    return status;
  }
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  const auto [prepared, order] =
      deduplicated_insertion_order(points, point_count, 2, weights, result->vertex_map);
  if (prepared != PHX_MC_OK) return prepared;
  // Vertex k of the triangulation is input point order[k], stored in
  // insertion order so that walks and cavities touch nearby memory; its
  // input index orders the symbolic perturbation, so the triangulation is
  // that of the input points.
  const std::size_t count = order.size();
  std::vector<double> local_points(2 * count);
  std::vector<double> local_weights(count);
  for (std::size_t k = 0; k < count; ++k) {
    const double* point = points + 2 * static_cast<std::size_t>(order[k]);
    local_points[2 * k] = point[0];
    local_points[2 * k + 1] = point[1];
    local_weights[k] = weights[order[k]];
  }
  std::vector<int32_t> local_order(count);
  std::iota(local_order.begin(), local_order.end(), 0);
  Triangulation2D triangulation;
  triangulation.reset(local_points.data(), static_cast<int64_t>(count), local_weights.data());
  triangulation.identities = order.data();
  status = build_ordered_delaunay_2d(triangulation, local_order);
  if (status != PHX_MC_OK) {
    return status;
  }
  std::vector<char> input_alive(static_cast<std::size_t>(point_count), 0);
  for (std::size_t k = 0; k < count; ++k) {
    input_alive[static_cast<std::size_t>(order[k])] = triangulation.vertex_triangle[k] >= 0;
  }
  for (int32_t& target : result->vertex_map) {
    if (target >= 0 && input_alive[static_cast<std::size_t>(target)] == 0) target = -1;
  }
  result->points.assign(points, points + 2 * point_count);
  triangulation.finite_cells(result->cells);  // input indices (identities)
  if (canonical) {
    const int32_t canonicalized = canonicalize_cells(*result, 1);
    if (canonicalized != PHX_MC_OK) return canonicalized;
  }
  *mesh = result.release();
  return PHX_MC_OK;
}

}  // namespace

int32_t triangulate_2d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || !addressable(point_count, 2, sizeof(double)) ||
      !addressable(point_count, 1, sizeof(int32_t)) ||
      !addressable(point_count, 1, sizeof(double)) ||
      (point_count > 0 && points == nullptr) ||
      (weighted && point_count > 0 && weights == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (point_count > kMaxMeshPoints) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  if (weighted) {
    return triangulate_2d_weighted(point_count, points, weights, canonical, mesh);
  }
  return triangulate_2d_unweighted(point_count, points, canonical, mesh);
}

}  // namespace phx::mc

extern "C" {

int32_t phx_mc_delaunay_2d(int64_t point_count, const double* points, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::triangulate_2d(point_count, points, nullptr, false, true, mesh);
  });
}

int32_t phx_mc_delaunay_2d_into(int64_t point_count, const double* points, int64_t cell_capacity,
                                int32_t* cells, int32_t* vertex_map, int64_t* cell_count) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!phx::mc::addressable(cell_capacity, 3, sizeof(int32_t)) || cell_count == nullptr ||
        (cell_capacity != 0 && cells == nullptr) || (point_count > 0 && vertex_map == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (point_count < 0 || !phx::mc::addressable(point_count, 2, sizeof(double)) ||
        !phx::mc::addressable(point_count, 1, sizeof(int32_t)) ||
        !phx::mc::addressable(point_count, 1, sizeof(double)) ||
        (point_count > 0 && points == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (point_count > phx::mc::kMaxMeshPoints) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    const auto count = static_cast<std::size_t>(point_count);
    phx::mc::UnweightedScratch local_scratch;
    phx::mc::UnweightedScratch& scratch =
        count <= phx::mc::kSmallScratchMaxPoints ? phx::mc::small_unweighted_scratch()
                                                 : local_scratch;
    int32_t tri_count = 0;
    int64_t finite_count = 0;
    bool has_duplicates = false;
    const int32_t status = phx::mc::run_unweighted_2d(point_count, points, scratch, tri_count,
                                                      finite_count, has_duplicates);
    if (status != PHX_MC_OK) {
      return status;
    }
    if (finite_count > cell_capacity) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (scratch.vertex_aux.size() < count + 1) {
      scratch.vertex_aux.resize(count + 1);
    }
    phx::mc::emit_canonical_triangles(scratch.triangles.data(), tri_count, scratch.order.data(),
                                      count, scratch, cells);
    if (has_duplicates) {
      std::memcpy(vertex_map, scratch.vertex_map.data(), count * sizeof(int32_t));
    } else {
      for (int32_t i = 0; i < static_cast<int32_t>(count); ++i) {
        vertex_map[i] = i;
      }
    }
    *cell_count = finite_count;
    return PHX_MC_OK;
  });
}

int32_t phx_mc_regular_2d(int64_t point_count, const double* points, const double* weights,
                          phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::triangulate_2d(point_count, points, weights, true, true, mesh);
  });
}

}  // extern "C"
