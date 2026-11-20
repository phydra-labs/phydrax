//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Open-addressing map from 64-bit keys to int64 values.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace phx::mc {

// Keys are any value but ~0; the table grows at half load and never erases.
class KeyMap {
 public:
  explicit KeyMap(std::size_t expected) { allocate(expected); }
  // The value of key, inserted as -1 when absent.
  int64_t& slot(std::uint64_t key) {
    if (2 * (size_ + 1) > keys_.size()) grow();
    std::size_t index = hash(key) & (keys_.size() - 1);
    while (keys_[index] != kEmpty && keys_[index] != key) index = (index + 1) & (keys_.size() - 1);
    if (keys_[index] == kEmpty) {
      keys_[index] = key;
      ++size_;
    }
    return values_[index];
  }
  int64_t find(std::uint64_t key) const {
    std::size_t index = hash(key) & (keys_.size() - 1);
    while (keys_[index] != kEmpty) {
      if (keys_[index] == key) return values_[index];
      index = (index + 1) & (keys_.size() - 1);
    }
    return -1;
  }

 private:
  static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
  static std::size_t hash(std::uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return static_cast<std::size_t>(key);
  }
  void allocate(std::size_t expected) {
    std::size_t capacity = 16;
    while (capacity < 2 * expected + 16) capacity *= 2;
    keys_.assign(capacity, kEmpty);
    values_.assign(capacity, -1);
    size_ = 0;
  }
  void grow() {
    std::vector<std::uint64_t> keys = std::move(keys_);
    std::vector<int64_t> values = std::move(values_);
    allocate(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i)
      if (keys[i] != kEmpty) slot(keys[i]) = values[i];
  }
  std::vector<std::uint64_t> keys_;
  std::vector<int64_t> values_;
  std::size_t size_ = 0;
};

}  // namespace phx::mc
