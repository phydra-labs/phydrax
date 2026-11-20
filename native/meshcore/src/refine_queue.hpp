// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace phx::mc {

// Fixed row-major bucket order, then the caller's strict priority order.
// Reintroduced entries can return to any earlier bucket; buckets are not
// permanently closed. The caller's validity predicate discards stale entries.
// Use one bucket unless the estimated element count exceeds 100 times the
// input point count; measured small rings did not benefit from subdivision.
// Given a finite initial queue, an insertion cap, and the caller invariant
// that new work is enqueued only by a successful insertion (or finitely many
// times while processing it), the queue drains or the cap is reached: between
// insertions every pop removes one of finitely many entries. No monotone
// worst-key assumption is needed.
//
// One bucket is a std::vector binary heap (std::push_heap/std::pop_heap, as
// std::priority_queue). The 256 spatial buckets are heaps of the given arity
// whose arrays come from one queue-owned arena: capacities are 4 * 2^c
// entries, a grown-out array returns to the free list of its class, and the
// memory is released with the queue (a few allocations per call instead of
// several per used bucket). A heap of another shape pops the same maximum,
// so the pop sequence is the same only when Worse is a strict total order on
// distinct entries or equivalent entries are interchangeable for the caller;
// callers choosing Arity != 2 assert that.
template <class Entry, class Worse, int Arity = 2>
class SpatialRefinementQueue {
  static_assert(std::is_trivially_copyable_v<Entry>);
  static_assert(Arity >= 2);

 public:
  SpatialRefinementQueue(double xmin, double ymin, double xmax, double ymax,
                         double estimated_elements, int64_t input_points)
      : xmin_(xmin), ymin_(ymin), dx_(xmax - xmin), dy_(ymax - ymin),
        buckets_(dx_ > 0.0 && dy_ > 0.0 &&
                         estimated_elements > 100.0 * static_cast<double>(input_points)
                     ? 16
                     : 1),
        current_(buckets_ * buckets_) {}
  SpatialRefinementQueue(const SpatialRefinementQueue&) = delete;
  SpatialRefinementQueue& operator=(const SpatialRefinementQueue&) = delete;

  bool empty() const { return current_ == buckets_ * buckets_; }
  bool spatial() const { return buckets_ != 1; }
  const Entry& top() const {
    return spatial() ? entries_[current_].data[0] : single_.front();
  }
  void pop() {
    if (!spatial()) {
      std::pop_heap(single_.begin(), single_.end(), Worse{});
      single_.pop_back();
      if (single_.empty()) current_ = 1;
      return;
    }
    Bucket& bucket = entries_[current_];
    if constexpr (Arity == 2) {
      std::pop_heap(bucket.data, bucket.data + bucket.size, Worse{});
      --bucket.size;
    } else {
      --bucket.size;
      if (bucket.size != 0) sift_down(bucket.data, bucket.size, bucket.data[bucket.size]);
    }
    while (current_ < buckets_ * buckets_ && entries_[current_].size == 0) ++current_;
  }

  template <class IsCurrent>
  bool pop_current(Entry& result, IsCurrent&& is_current) {
    while (!empty()) {
      Entry entry = top();
      pop();
      if (is_current(entry)) {
        result = entry;
        return true;
      }
    }
    return false;
  }
  void push(const Entry& entry, double x = 0.0, double y = 0.0) {
    if (!spatial()) {
      single_.push_back(entry);
      std::push_heap(single_.begin(), single_.end(), Worse{});
      current_ = 0;
      return;
    }
    const double u = std::clamp((x - xmin_) / dx_, 0.0, 1.0);
    const double v = std::clamp((y - ymin_) / dy_, 0.0, 1.0);
    const int ix = std::min(static_cast<int>(u * buckets_), buckets_ - 1);
    const int iy = std::min(static_cast<int>(v * buckets_), buckets_ - 1);
    const int index = iy * buckets_ + ix;
    Bucket& bucket = entries_[index];
    if (bucket.size == bucket.capacity) grow(bucket);
    if constexpr (Arity == 2) {
      bucket.data[bucket.size++] = entry;
      std::push_heap(bucket.data, bucket.data + bucket.size, Worse{});
    } else {
      sift_up(bucket.data, bucket.size++, entry);
    }
    current_ = std::min(current_, index);
  }

 private:
  // Places entry at hole (the new last position) or above it.
  static void sift_up(Entry* data, std::size_t hole, const Entry& entry) {
    const Worse worse{};
    while (hole != 0) {
      const std::size_t parent = (hole - 1) / Arity;
      if (!worse(data[parent], entry)) break;
      data[hole] = data[parent];
      hole = parent;
    }
    data[hole] = entry;
  }

  // Refills the root of the size-entry heap with the former last entry
  // (Floyd: the hole descends along the larger children to a leaf, where
  // entry, usually small, rises again). Child selection is a plain select so
  // that it compiles without a data-dependent branch.
  static void sift_down(Entry* data, std::size_t size, const Entry entry) {
    const Worse worse{};
    std::size_t hole = 0;
    for (;;) {
      const std::size_t first = Arity * hole + 1;
      if (first >= size) break;
      const std::size_t last = std::min(first + Arity, size);
      std::size_t best = first;
      for (std::size_t child = first + 1; child < last; ++child) {
        best = worse(data[best], data[child]) ? child : best;
      }
      data[hole] = data[best];
      hole = best;
    }
    sift_up(data, hole, entry);
  }

  static constexpr std::size_t kSmallest = 4;
  static constexpr std::size_t kFirstChunk = 4096 / sizeof(Entry) * sizeof(Entry);

  struct Bucket {
    Entry* data = nullptr;
    std::size_t size = 0;
    std::size_t capacity = 0;
    int size_class = -1;
  };

  // Free blocks of one class form a singly linked list through their storage.
  static_assert(kSmallest * sizeof(Entry) >= sizeof(void*));

  void grow(Bucket& bucket) {
    const int size_class = bucket.size_class + 1;
    const std::size_t capacity = kSmallest << size_class;
    Entry* data = take(size_class, capacity);
    if (bucket.size != 0) std::memcpy(data, bucket.data, bucket.size * sizeof(Entry));
    if (bucket.data != nullptr) give(bucket.size_class, bucket.data);
    bucket.data = data;
    bucket.capacity = capacity;
    bucket.size_class = size_class;
  }

  Entry* take(int size_class, std::size_t capacity) {
    if (static_cast<std::size_t>(size_class) < free_.size() &&
        free_[static_cast<std::size_t>(size_class)] != nullptr) {
      void* block = free_[static_cast<std::size_t>(size_class)];
      std::memcpy(&free_[static_cast<std::size_t>(size_class)], block, sizeof(void*));
      return static_cast<Entry*>(block);
    }
    const std::size_t bytes = capacity * sizeof(Entry);
    if (bytes > chunk_left_) {
      const std::size_t chunk = std::max(bytes, chunks_.empty() ? kFirstChunk : 2 * last_chunk_);
      chunks_.push_back(std::make_unique_for_overwrite<std::byte[]>(chunk));
      chunk_next_ = chunks_.back().get();
      chunk_left_ = chunk;
      last_chunk_ = chunk;
    }
    std::byte* block = chunk_next_;
    chunk_next_ += bytes;
    chunk_left_ -= bytes;
    return reinterpret_cast<Entry*>(block);
  }

  void give(int size_class, Entry* data) {
    if (static_cast<std::size_t>(size_class) >= free_.size()) {
      free_.resize(static_cast<std::size_t>(size_class) + 1, nullptr);
    }
    void* head = free_[static_cast<std::size_t>(size_class)];
    std::memcpy(static_cast<void*>(data), &head, sizeof(void*));
    free_[static_cast<std::size_t>(size_class)] = data;
  }

  double xmin_, ymin_, dx_, dy_;
  int buckets_;
  int current_;
  std::array<Bucket, 256> entries_{};
  std::vector<Entry> single_;
  std::vector<std::unique_ptr<std::byte[]>> chunks_;
  std::vector<void*> free_;
  std::byte* chunk_next_ = nullptr;
  std::size_t chunk_left_ = 0;
  std::size_t last_chunk_ = 0;
};

}  // namespace phx::mc
