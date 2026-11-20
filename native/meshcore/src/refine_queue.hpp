// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
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
template <class Entry, class Worse>
class SpatialRefinementQueue {
 public:
  SpatialRefinementQueue(double xmin, double ymin, double xmax, double ymax,
                         double estimated_elements, int64_t input_points)
      : xmin_(xmin), ymin_(ymin), dx_(xmax - xmin), dy_(ymax - ymin),
        buckets_(dx_ > 0.0 && dy_ > 0.0 &&
                         estimated_elements > 100.0 * static_cast<double>(input_points)
                     ? 16
                     : 1),
        current_(buckets_ * buckets_) {}

  bool empty() const { return current_ == buckets_ * buckets_; }
  bool spatial() const { return buckets_ != 1; }
  const Entry& top() const { return entries_[current_].top(); }
  void pop() {
    entries_[current_].pop();
    while (current_ < buckets_ * buckets_ && entries_[current_].empty()) ++current_;
  }

  template <class IsCurrent>
  bool pop_current(Entry& result, IsCurrent&& is_current) {
    while (!empty()) {
      Entry entry = top();
      pop();
      if (is_current(entry)) {
        result = std::move(entry);
        return true;
      }
    }
    return false;
  }
  void push(const Entry& entry, double x = 0.0, double y = 0.0) {
    int bucket = 0;
    if (spatial()) {
      const double u = std::clamp((x - xmin_) / dx_, 0.0, 1.0);
      const double v = std::clamp((y - ymin_) / dy_, 0.0, 1.0);
      const int ix = std::min(static_cast<int>(u * buckets_), buckets_ - 1);
      const int iy = std::min(static_cast<int>(v * buckets_), buckets_ - 1);
      bucket = iy * buckets_ + ix;
    }
    entries_[bucket].push(entry);
    current_ = std::min(current_, bucket);
  }

 private:
  double xmin_, ymin_, dx_, dy_;
  int buckets_;
  int current_;
  std::array<std::priority_queue<Entry, std::vector<Entry>, Worse>, 256> entries_;
};

}  // namespace phx::mc
