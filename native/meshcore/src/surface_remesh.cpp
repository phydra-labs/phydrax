// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Source-agnostic certified surface remeshing; contract in surface_remesh.hpp.
#include "surface_remesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include "exact_scalar.hpp"
#include "geometry3.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"
#include "scratch_faces.hpp"

namespace phx::mc::remesh {
bool legal_triangle_contact(const std::array<Point, 3>& first, const Triangle& first_ids,
                            const std::array<Point, 3>& second, const Triangle& second_ids) {
  const double* a[3] = {first[0].data(), first[1].data(), first[2].data()};
  const double* b[3] = {second[0].data(), second[1].data(), second[2].data()};
  if (first_ids[0] != first_ids[1] && first_ids[1] != first_ids[2] &&
      first_ids[2] != first_ids[0] && second_ids[0] != second_ids[1] &&
      second_ids[1] != second_ids[2] && second_ids[2] != second_ids[0]) {
    int shared = 0, others[3], other_count = 0;
    bool matching = true;
    for (int j = 0; j < 3; ++j) {
      int match = -1;
      for (int i = 0; i < 3; ++i)
        if (first_ids[i] == second_ids[j]) match = i;
      if (match < 0) others[other_count++] = j;
      else {
        ++shared;
        matching = matching && phx::mc::geometry::same_point(a[match], b[j]);
      }
    }
    // Two triangles in different planes which share two vertices meet only on
    // their common edge. The exact nonzero determinant also proves both are
    // nondegenerate; coplanar or out-of-domain cases use the full predicate.
    if (shared == 2 && matching && phx::mc::geometry::point_in_domain(a[0]) &&
        phx::mc::geometry::point_in_domain(a[1]) && phx::mc::geometry::point_in_domain(a[2]) &&
        phx::mc::geometry::point_in_domain(b[others[0]]) &&
        phx::mc::orient3d(a[0], a[1], a[2], b[others[0]]) != 0)
      return true;
    // With one shared vertex, two strictly equal plane-side signs show that
    // the rest of the second triangle cannot touch the first triangle's plane.
    if (shared == 1 && matching && phx::mc::geometry::point_in_domain(a[0]) &&
        phx::mc::geometry::point_in_domain(a[1]) && phx::mc::geometry::point_in_domain(a[2]) &&
        phx::mc::geometry::point_in_domain(b[others[0]]) &&
        phx::mc::geometry::point_in_domain(b[others[1]])) {
      const int first_side = phx::mc::orient3d(a[0], a[1], a[2], b[others[0]]);
      if (first_side != 0 &&
          first_side == phx::mc::orient3d(a[0], a[1], a[2], b[others[1]]) &&
          !phx::mc::geometry::collinear(b[0], b[1], b[2]))
        return true;
    }
  }
  phx::mc::Contact contact;
  return phx::mc::intersect_triangles(a, b, first_ids.data(), second_ids.data(), contact) ==
             PHX_MC_OK &&
         (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
          contact.kind == PHX_MC_SHARED_EDGE);
}

namespace {
using V = Point;
using F = Triangle;
// Large spatial-block banks own their pages and return them to the OS at the
// end of the remesh. This does not alter the process-wide allocator.
template <class T>
class SpatialPageAllocator {
 public:
  using value_type = T;
  SpatialPageAllocator() noexcept = default;
  template <class U> SpatialPageAllocator(const SpatialPageAllocator<U>&) noexcept {}
  T* allocate(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_alloc();
    if (count == 0) return nullptr;
#if defined(_WIN32)
    return static_cast<T*>(::operator new(count * sizeof(T)));
#else
    void* const memory = mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory == MAP_FAILED) throw std::bad_alloc();
    return static_cast<T*>(memory);
#endif
  }
  void deallocate(T* memory, size_t count) noexcept {
    if (memory == nullptr) return;
#if defined(_WIN32)
    ::operator delete(memory);
#else
    (void)munmap(memory, count * sizeof(T));
#endif
  }
};
template <class T, class U>
bool operator==(const SpatialPageAllocator<T>&, const SpatialPageAllocator<U>&) noexcept {
  return true;
}

using phx::mc::geometry::cross;
using phx::mc::geometry::difference;
using phx::mc::geometry::dot;
using phx::mc::geometry::next_up;
using phx::mc::geometry::point_in_domain;

constexpr uint32_t kScoreLevels = 1u << 20;
constexpr size_t kBucketBlockRows = 8;

constexpr double kPi = 3.141592653589793;
constexpr double kTurnSlack = 1.0e-9;  // radians

V add(const V& a, const V& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V mul(const V& a, double t) { return {a[0] * t, a[1] * t, a[2] * t}; }
double length(const V& a) { return std::hypot(a[0], a[1], a[2]); }

uint64_t cell_key(int64_t i, int64_t j, int64_t k) {
  constexpr int64_t bias = int64_t{1} << 20;
  return (static_cast<uint64_t>(i + bias) << 42) ^ (static_cast<uint64_t>(j + bias) << 21) ^
         static_cast<uint64_t>(k + bias);
}
// The caller's packed double bank is borrowed read-only. Only moved vertices
// acquire owned coordinates; appended vertices are owned separately.
class PointStore {
 public:
  void own(std::vector<V>&& points) { owned_ = std::move(points); }
  void borrow(const double* points, size_t count) {
    borrowed_ = points;
    base_count_ = count;
    moved_ = std::make_unique<ScratchRows<V, 512>>();
    slots_.assign(count, kMissing);
  }
  size_t size() const { return borrowed_ ? base_count_ + appended_.size() : owned_.size(); }
  V operator[](size_t index) const {
    if (!borrowed_) return owned_[index];
    if (index >= base_count_) return appended_[index - base_count_];
    const uint64_t slot = wide_ ? wide_slots_[index] : slots_[index];
    const uint64_t missing = wide_ ? std::numeric_limits<uint64_t>::max() : kMissing;
    if (slot != missing) return moved_->get(static_cast<size_t>(slot));
    return {borrowed_[3 * index], borrowed_[3 * index + 1], borrowed_[3 * index + 2]};
  }
  void set(size_t index, const V& value) {
    if (!borrowed_) { owned_[index] = value; return; }
    if (index >= base_count_) { appended_[index - base_count_] = value; return; }
    uint64_t slot = wide_ ? wide_slots_[index] : slots_[index];
    uint64_t missing = wide_ ? std::numeric_limits<uint64_t>::max() : kMissing;
    if (std::memcmp(value.data(), borrowed_ + 3 * index, 3 * sizeof(double)) == 0) {
      if (slot != missing) {
        free_slots_.push_back(slot);
        if (wide_) wide_slots_[index] = missing;
        else slots_[index] = kMissing;
      }
      return;
    }
    if (slot == missing) {
      if (!wide_ && free_slots_.empty() && moved_->size() >= kMissing) {
        wide_slots_.reserve(slots_.size());
        for (uint32_t entry : slots_)
          wide_slots_.push_back(entry == kMissing ? std::numeric_limits<uint64_t>::max() : entry);
        std::vector<uint32_t>().swap(slots_);
        wide_ = true;
      }
      if (free_slots_.empty()) {
        slot = moved_->size();
        moved_->push(value);
      } else {
        slot = free_slots_.back();
        free_slots_.pop_back();
        moved_->set(static_cast<size_t>(slot), value);
      }
      if (wide_) wide_slots_[index] = slot;
      else slots_[index] = static_cast<uint32_t>(slot);
      return;
    }
    moved_->set(static_cast<size_t>(slot), value);
  }
  void push_back(const V& value) {
    if (borrowed_) appended_.push_back(value);
    else owned_.push_back(value);
  }
  void pop_back() {
    if (borrowed_) appended_.pop_back();
    else owned_.pop_back();
  }
 private:
  static constexpr uint32_t kMissing = std::numeric_limits<uint32_t>::max();
  const double* borrowed_ = nullptr;
  size_t base_count_ = 0;
  std::vector<V> owned_, appended_;
  std::unique_ptr<ScratchRows<V, 512>> moved_;
  std::vector<uint32_t> slots_;
  std::vector<uint64_t> wide_slots_;
  std::vector<uint64_t> free_slots_;
  bool wide_ = false;
};



// Original compact faces are immutable throughout remeshing: edits kill old
// rows and append new rows. Borrow them instead of widening a second full mesh.
class TriangleStore {
 public:
  void own(std::vector<F>&& faces) { owned = std::move(faces); }
  void reset(std::vector<F>&& faces) {
    appended.clear();
    compact = nullptr;
    base_count = 0;
    narrow = false;
    owned = std::move(faces);
  }
  void borrow(const int32_t* faces, size_t count) {
    compact = faces;
    base_count = count;
    narrow = true;
  }
  size_t size() const { return base_count + (narrow ? appended.size() : owned.size()); }
  F operator[](size_t row) const {
    if (row < base_count)
      return {compact[3 * row], compact[3 * row + 1], compact[3 * row + 2]};
    if (narrow) {
      const auto face = appended.get(row - base_count);
      return {face[0], face[1], face[2]};
    }
    return owned[row - base_count];
  }
  void push_back(const F& face) {
    if (narrow && std::all_of(face.begin(), face.end(), [](int64_t v) {
          return v >= 0 && v <= std::numeric_limits<int32_t>::max();
        })) {
      appended.push({static_cast<int32_t>(face[0]), static_cast<int32_t>(face[1]),
                     static_cast<int32_t>(face[2])});
      return;
    }
    if (narrow) {
      owned.reserve(appended.size() + 1);
      for (size_t i = 0; i < appended.size(); ++i) {
        const auto row = appended.get(i);
        owned.push_back({row[0], row[1], row[2]});
      }
      appended.clear();
      narrow = false;
    }
    owned.push_back(face);
  }

 private:
  const int32_t* compact = nullptr;
  size_t base_count = 0;
  bool narrow = false;
  ScratchFaces appended;
  std::vector<F> owned;
};

// Compact-input evidence is privately paged; the caller's coordinates and
// source arrays remain borrowed and unchanged. Repacking takes ordinary
// ownership of the small survivor evidence bank.
class EvidenceStore {
 public:
  void page() { scratch_ = std::make_unique<ScratchRows<VertexEvidence, 512>>(); }
  void reserve(size_t count) { if (!scratch_) owned_.reserve(count); }
  void push_back(const VertexEvidence& value) {
    if (scratch_) scratch_->push(value);
    else owned_.push_back(value);
  }
  void pop_back() {
    if (scratch_) scratch_->pop();
    else owned_.pop_back();
  }
  VertexEvidence operator[](size_t index) const {
    return scratch_ ? scratch_->get(index) : owned_[index];
  }
  void set(size_t index, const VertexEvidence& value) {
    if (scratch_) scratch_->set(index, value);
    else owned_[index] = value;
  }
  void own(std::vector<VertexEvidence>&& values) {
    scratch_.reset();
    owned_ = std::move(values);
  }
 private:
  std::vector<VertexEvidence> owned_;
  std::unique_ptr<ScratchRows<VertexEvidence, 512>> scratch_;
};
// The open cone is common on finely resolved carriers. Materialize only
// blocks containing another exact floor; no quantization or rounding changes.
class FloorStore {
 public:
  void reserve(size_t count) { blocks_.reserve(count / kBlock + (count % kBlock != 0)); }
  void push_back(double value) {
    const size_t block = size_ / kBlock, offset = size_ % kBlock;
    if (block == blocks_.size()) blocks_.emplace_back();
    if (value != kOpenCone && !blocks_[block]) {
      blocks_[block].reset(new double[kBlock]);
      std::fill_n(blocks_[block].get(), kBlock, kOpenCone);
    }
    if (blocks_[block]) blocks_[block][offset] = value;
    ++size_;
  }
  double operator[](size_t row) const {
    const auto& block = blocks_[row / kBlock];
    return block ? block[row % kBlock] : kOpenCone;
  }
 private:
  static constexpr size_t kBlock = 4096;
  size_t size_ = 0;
  std::vector<std::unique_ptr<double[]>> blocks_;
};
// Stable row-indexed banks grow by chunks instead of copying all historical
// rows on capacity expansion.
template <class T, T Initial>
class ChunkedBank {
 public:
  size_t size() const { return size_; }
  void resize(size_t count) {
    const size_t needed = count / kBlock + (count % kBlock != 0);
    while (blocks_.size() < needed) {
      std::unique_ptr<T[]> block(new T[kBlock]);
      std::fill_n(block.get(), kBlock, Initial);
      blocks_.push_back(std::move(block));
    }
    size_ = count;
  }
  T& operator[](size_t row) { return blocks_[row / kBlock][row % kBlock]; }
  T operator[](size_t row) const { return blocks_[row / kBlock][row % kBlock]; }
  void clear() { blocks_.clear(); size_ = 0; }
 private:
  static constexpr size_t kBlock = 1u << 16;
  size_t size_ = 0;
  std::vector<std::unique_ptr<T[]>> blocks_;
};

// The frequently queried liveness bits are separate from the 21-bit scores:
// broad-phase scans can reject dead historical faces from a compact bitmap.
class FaceState {
  using Row = std::array<uint8_t, 3>;
  static constexpr uint8_t kAlive = 1u << 5, kStuck = 1u << 6;
  static constexpr uint8_t kScoreHighMask = kAlive - 1;
  static_assert(kScoreLevels < (1u << 21), "face scores require more than 21 bits");
 public:
  void page() { scratch_ = std::make_unique<ScratchRows<Row, 512>>(); }
  void assign(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / 3)
      throw std::length_error("face state size overflow");
    values_.clear();
    if (scratch_) {
      scratch_->clear();
      for (size_t row = 0; row < count; ++row) scratch_->push({});
    } else {
      values_.resize(count * 3);
    }
    alive_.assign(count / 64 + (count % 64 != 0), ~uint64_t{0});
    rows_ = count;
  }
  void push() {
    if (rows_ >= std::numeric_limits<size_t>::max() / 3)
      throw std::length_error("face state size overflow");
    if (scratch_) scratch_->push({});
    else {
      const size_t offset = values_.size();
      values_.resize(offset + 3);
      values_[offset] = values_[offset + 1] = values_[offset + 2] = 0;
    }
    if (rows_ % 64 == 0) alive_.push_back(~uint64_t{0});
    else alive_[rows_ / 64] |= uint64_t{1} << (rows_ % 64);
    ++rows_;
  }
  bool alive(size_t row) const { return (alive_[row / 64] & (uint64_t{1} << (row % 64))) != 0; }
  bool stuck(size_t row) const { return (get(row)[2] & kStuck) != 0; }
  uint32_t score(size_t row) const {
    const Row value = get(row);
    return static_cast<uint32_t>(value[0]) | (static_cast<uint32_t>(value[1]) << 8) |
           (static_cast<uint32_t>(value[2] & kScoreHighMask) << 16);
  }
  void set_alive(size_t row, bool value) {
    const uint64_t mask = uint64_t{1} << (row % 64);
    if (value) alive_[row / 64] |= mask;
    else alive_[row / 64] &= ~mask;
  }
  void set_stuck(size_t row, bool value) {
    Row item = get(row);
    if (value) item[2] |= kStuck;
    else item[2] &= ~kStuck;
    set(row, item);
  }
  void set_score(size_t row, uint32_t score) {
    Row item = get(row);
    item[0] = static_cast<uint8_t>(score);
    item[1] = static_cast<uint8_t>(score >> 8);
    item[2] = static_cast<uint8_t>((item[2] & ~kScoreHighMask) |
                                   ((score >> 16) & kScoreHighMask));
    set(row, item);
  }
 private:
  Row get(size_t row) const {
    if (scratch_) return scratch_->get(row);
    const size_t offset = row * 3;
    return {values_[offset], values_[offset + 1], values_[offset + 2]};
  }
  void set(size_t row, const Row& item) {
    if (scratch_) scratch_->set(row, item);
    else {
      const size_t offset = row * 3;
      values_[offset] = item[0];
      values_[offset + 1] = item[1];
      values_[offset + 2] = item[2];
    }
  }
  ChunkedBank<uint8_t, 0> values_;
  std::unique_ptr<ScratchRows<Row, 512>> scratch_;
  std::vector<uint64_t> alive_;
  size_t rows_ = 0;
};

// Append-ordered edited incidence with recyclable nodes. Original incidence
// stays in the borrowed carrier CSR; dead edited rows are unlinked on access.
class EditedStars {
  template <class I> struct Node { I face, next; };
  template <class I> static constexpr I missing = std::numeric_limits<I>::max();
 public:
  void initialize(size_t count) {
    heads_.assign(count, missing<uint32_t>);
  }
  void add_vertex() {
    if (wide_) {
      wide_heads_.push_back(missing<uint64_t>);
    } else {
      heads_.push_back(missing<uint32_t>);
    }
  }
  void pop_vertex() {
    if (wide_) wide_heads_.pop_back();
    else heads_.pop_back();
  }
  void append(size_t vertex, int64_t face) {
    if (!wide_ && (static_cast<uint64_t>(face) >= missing<uint32_t> ||
                   (free_ == missing<uint32_t> && nodes_.size() >= missing<uint32_t>)))
      widen();
    if (wide_)
      append_impl(wide_heads_, wide_nodes_, wide_free_, vertex, static_cast<uint64_t>(face));
    else
      append_impl(heads_, nodes_, free_, vertex, static_cast<uint32_t>(face));
  }
  void collect(size_t vertex, const FaceState& alive, std::vector<int64_t>& out) {
    if (wide_) collect_impl(wide_heads_, wide_nodes_, wide_free_, vertex, alive, out);
    else collect_impl(heads_, nodes_, free_, vertex, alive, out);
  }
  void clear(size_t vertex) {
    if (wide_) clear_impl(wide_heads_, wide_nodes_, wide_free_, vertex);
    else clear_impl(heads_, nodes_, free_, vertex);
  }
 private:
  template <class I>
  static void append_impl(std::vector<I>& heads,
                          std::vector<Node<I>>& nodes, I& free, size_t vertex, I face) {
    I index;
    if (free == missing<I>) {
      index = static_cast<I>(nodes.size());
      nodes.push_back({face, missing<I>});
    } else {
      index = free;
      free = nodes[index].next;
      nodes[index] = {face, missing<I>};
    }
    if (heads[vertex] == missing<I>) heads[vertex] = index;
    else {
      I last = heads[vertex];
      while (nodes[last].next != missing<I>) last = nodes[last].next;
      nodes[last].next = index;
    }
  }
  template <class I>
  static void collect_impl(std::vector<I>& heads,
                           std::vector<Node<I>>& nodes, I& free, size_t vertex,
                           const FaceState& alive, std::vector<int64_t>& out) {
    I previous = missing<I>, current = heads[vertex];
    while (current != missing<I>) {
      const I next = nodes[current].next;
      if (alive.alive(static_cast<size_t>(nodes[current].face))) {
        out.push_back(static_cast<int64_t>(nodes[current].face));
        previous = current;
      } else {
        if (previous == missing<I>) heads[vertex] = next;
        else nodes[previous].next = next;
        nodes[current].next = free;
        free = current;
      }
      current = next;
    }
  }
  template <class I>
  static void clear_impl(std::vector<I>& heads,
                         std::vector<Node<I>>& nodes, I& free, size_t vertex) {
    I current = heads[vertex];
    while (current != missing<I>) {
      const I next = nodes[current].next;
      nodes[current].next = free;
      free = current;
      current = next;
    }
    heads[vertex] = missing<I>;
  }
  void widen() {
    const auto convert = [](uint32_t i) -> uint64_t {
      return i == missing<uint32_t> ? missing<uint64_t> : i;
    };
    wide_heads_.reserve(heads_.size());
    wide_nodes_.reserve(nodes_.size());
    for (uint32_t i : heads_) wide_heads_.push_back(convert(i));
    for (const auto& node : nodes_) wide_nodes_.push_back({node.face, convert(node.next)});
    wide_free_ = convert(free_);
    std::vector<uint32_t>().swap(heads_);
    std::vector<Node<uint32_t>>().swap(nodes_);
    wide_ = true;
  }
  std::vector<uint32_t> heads_;
  std::vector<Node<uint32_t>> nodes_;
  uint32_t free_ = missing<uint32_t>;
  std::vector<uint64_t> wide_heads_;
  std::vector<Node<uint64_t>> wide_nodes_;
  uint64_t wide_free_ = missing<uint64_t>;
  bool wide_ = false;
};


// One entry per face, ordered exactly by (quantized score, face ID).
class ScoreQueue {
 public:
  explicit ScoreQueue(const FaceState& scores) : scores_(scores) {}
  bool empty() const { return heap_.empty(); }
  int64_t front() const { return heap_.front(); }
  void insert(int64_t id) {
    const size_t face = static_cast<size_t>(id);
    if (!wide_ && (face >= kMissing || heap_.size() >= kMissing)) widen();
    if (wide_) {
      if (face >= wide_position_.size()) wide_position_.resize(face + 1);
    } else if (face >= position_.size()) {
      position_.resize(face + 1);
    }
    if (position(face) >= 0) return;
    set_position(face, static_cast<int64_t>(heap_.size()));
    heap_.push_back(id);
    rise(heap_.size() - 1);
  }
  void erase(int64_t id) {
    const size_t face = static_cast<size_t>(id);
    const int64_t current = position(face);
    if (current < 0) return;
    const size_t at = static_cast<size_t>(current);
    swap_entries(at, heap_.size() - 1);
    heap_.pop_back();
    set_position(face, -1);
    if (at < heap_.size()) {
      if (at > 0 && less(heap_[at], heap_[(at - 1) / 2])) rise(at);
      else sink(at);
    }
  }
 private:
  static constexpr uint32_t kMissing = std::numeric_limits<uint32_t>::max();
  int64_t position(size_t face) const {
    if (wide_) return face < wide_position_.size() ? wide_position_[face] : -1;
    if (face >= position_.size() || position_[face] == kMissing) return -1;
    return position_[face];
  }
  void set_position(size_t face, int64_t value) {
    if (wide_) wide_position_[face] = value;
    else position_[face] = value < 0 ? kMissing : static_cast<uint32_t>(value);
  }
  void widen() {
    wide_position_.resize(position_.size());
    for (size_t i = 0; i < position_.size(); ++i) {
      const uint32_t value = position_[i];
      wide_position_[i] = value == kMissing ? -1 : static_cast<int64_t>(value);
    }
    position_.clear();
    wide_ = true;
  }
  bool less(int64_t a, int64_t b) const {
    const uint32_t x = scores_.score(static_cast<size_t>(a)), y = scores_.score(static_cast<size_t>(b));
    return x < y || (x == y && a < b);
  }
  void swap_entries(size_t a, size_t b) {
    std::swap(heap_[a], heap_[b]);
    set_position(static_cast<size_t>(heap_[a]), static_cast<int64_t>(a));
    set_position(static_cast<size_t>(heap_[b]), static_cast<int64_t>(b));
  }
  void rise(size_t at) {
    while (at > 0) {
      const size_t parent = (at - 1) / 2;
      if (!less(heap_[at], heap_[parent])) break;
      swap_entries(at, parent);
      at = parent;
    }
  }
  void sink(size_t at) {
    while (at < heap_.size() / 2) {
      size_t child = 2 * at + 1;
      if (child + 1 < heap_.size() && less(heap_[child + 1], heap_[child])) ++child;
      if (!less(heap_[child], heap_[at])) break;
      swap_entries(at, child);
      at = child;
    }
  }
  const FaceState& scores_;
  std::vector<int64_t> heap_;
  ChunkedBank<int64_t, -1> wide_position_;
  ChunkedBank<uint32_t, std::numeric_limits<uint32_t>::max()> position_;
  bool wide_ = false;
};


class Remesher {
 public:
  Remesher(const TriangleAdmission& admission, const RemeshTargets& targets)
      : predicate(admission), targets(targets), h(targets.edge_length), queue(state) {}

  const TriangleAdmission& predicate;
  const RemeshTargets& targets;
  double h = 0;
  double solid_side = 1;  // +1 when the normals point away from the enclosed solid
  PointStore points;
  EvidenceStore evidence;
  TriangleStore triangles;
  FaceState state;
  FloorStore floors;                             // source normal cone floor per triangle
  std::unordered_map<int64_t, NormalCone> cones;  // lazily computed source cones
  std::vector<std::vector<int64_t>> star;
  // For borrowed carriers the original incidence is immutable. Keep it in a
  // compact CSR bank, and allocate mutable incidence only for edited stars.
  std::vector<size_t> base_star_offsets;
  std::vector<uint32_t> base_star_ids;
  std::vector<uint64_t> wide_star_ids;
  const int32_t* borrowed_star_offsets = nullptr;
  const int32_t* borrowed_star_rows = nullptr;
  size_t borrowed_star_vertices = 0, borrowed_star_faces = 0;
  EditedStars edited_star;
  bool compact_stars = false;
  struct BucketBlock {
    V lower{}, upper{};
    bool invalid = false;
  };
  struct BucketGroup {
    V lower{}, upper{};
    bool invalid = false;
  };
  struct Bucket {
    std::vector<uint32_t> ids;
    std::vector<int64_t> wide_ids;
    bool wide = false;
    size_t size() const { return wide ? wide_ids.size() : ids.size(); }
    int64_t get(size_t at) const { return wide ? wide_ids[at] : ids[at]; }
    void clear() { if (wide) wide_ids.clear(); else ids.clear(); }
    void push(int64_t id) {
      if (!wide && static_cast<uint64_t>(id) > std::numeric_limits<uint32_t>::max()) {
        wide_ids.reserve(ids.size() + 1);
        for (uint32_t row : ids) wide_ids.push_back(row);
        std::vector<uint32_t>().swap(ids);
        wide = true;
      }
      if (wide) wide_ids.push_back(id);
      else ids.push_back(static_cast<uint32_t>(id));
    }
    std::vector<BucketBlock, SpatialPageAllocator<BucketBlock>> blocks;
    std::vector<std::vector<BucketGroup>> levels;
    std::vector<std::vector<BucketGroup>> tail_levels;
    size_t indexed_blocks = 0;
    size_t reindex_at = 4096;
  };
  std::unordered_map<uint64_t, Bucket> buckets;
  bool bucket_index_ready = false;
  bool bucket_rebuilding = false;
  ScoreQueue queue;
  bool queue_enabled = false;
  int64_t splits = 0, collapses = 0, flips = 0, relocations = 0;


  bool admitted(const F& t) const {
    const size_t a = static_cast<size_t>(t[0]), b = static_cast<size_t>(t[1]),
                 c = static_cast<size_t>(t[2]);
    return predicate.admits(points[a], points[b], points[c],
                            {evidence[a], evidence[b], evidence[c]});
  }
  // Shape and size factor of a triangle in [0, 1].
  double shape_size(const V& a, const V& b, const V& c) const {
    std::array<double, 3> lengths{length(difference(b, c)), length(difference(c, a)),
                                  length(difference(a, b))};
    std::sort(lengths.begin(), lengths.end());
    const double twice_area = length(cross(difference(b, a), difference(c, a)));
    if (!(twice_area > 0) || !(lengths[0] > 0)) return 0;
    const double sine = std::min(1.0, twice_area / (lengths[1] * lengths[2]));
    const double minimum_angle = std::asin(sine) * 180.0 / std::acos(-1.0);
    const double shape = std::min(1.0, minimum_angle / targets.minimum_angle_degrees);
    const double size = std::min(1.0, (4.0 / 3.0) * h / lengths[2]);
    return shape * size;
  }
  V normal(const F& t) const {
    const V& a = points[static_cast<size_t>(t[0])];
    return cross(difference(points[static_cast<size_t>(t[1])], a),
                 difference(points[static_cast<size_t>(t[2])], a));
  }
  std::array<VertexEvidence, 3> corner_evidence(const F& t) const {
    return {evidence[static_cast<size_t>(t[0])], evidence[static_cast<size_t>(t[1])],
            evidence[static_cast<size_t>(t[2])]};
  }
  NormalCone cone_of(const F& t) const {
    return predicate.normal_cone(points[static_cast<size_t>(t[0])],
                                 points[static_cast<size_t>(t[1])],
                                 points[static_cast<size_t>(t[2])], corner_evidence(t));
  }
  double floor_of(const F& t) const {
    return predicate.normal_cone_floor(points[static_cast<size_t>(t[0])],
                                       points[static_cast<size_t>(t[1])],
                                       points[static_cast<size_t>(t[2])], corner_evidence(t));
  }
  // The created triangles of a staged edit, their cone floors, and their
  // cones, computed only when an edge turns beyond both floors.
  struct Staged {
    const std::vector<F>* created = nullptr;
    std::vector<double> floors;
    std::vector<std::optional<NormalCone>> cones;
  };
  // A triangle of a staged state: a live ID, or (id < 0) a created row.
  struct Ref {
    int64_t id = -1;
    size_t row = 0;
  };
  F triangle(const Ref& ref, const Staged& staged) const {
    return ref.id >= 0 ? triangles[static_cast<size_t>(ref.id)] : (*staged.created)[ref.row];
  }
  double cone_floor(const Ref& ref, const Staged& staged) const {
    return ref.id >= 0 ? floors[static_cast<size_t>(ref.id)] : staged.floors[ref.row];
  }
  const NormalCone& cone(const Ref& ref, Staged& staged) {
    if (ref.id >= 0) {
      auto found = cones.find(ref.id);
      if (found == cones.end()) found = cones.emplace(ref.id, cone_of(triangle(ref, staged))).first;
      return found->second;
    }
    auto& slot = staged.cones[ref.row];
    if (!slot) slot = cone_of(triangle(ref, staged));
    return *slot;
  }
  // The triangle holding directed edge q -> p (the neighbor across p -> q) in
  // the state where `removed` (sorted IDs) are dead and the staged triangles
  // are live.
  std::optional<Ref> across(int64_t p, int64_t q, const Staged& staged,
                            const std::vector<int64_t>& removed) {
    const auto holds = [&](const F& t) {
      for (int k = 0; k < 3; ++k)
        if (t[k] == q && t[(k + 1) % 3] == p) return true;
      return false;
    };
    if (staged.created != nullptr)
      for (size_t row = 0; row < staged.created->size(); ++row)
        if (holds((*staged.created)[row])) return Ref{-1, row};
    for (int64_t id : live_star(p)) {
      if (std::binary_search(removed.begin(), removed.end(), id)) continue;
      if (holds(triangles[static_cast<size_t>(id)])) return Ref{id, 0};
    }
    return std::nullopt;
  }
  // Quantized score of a triangle in that state: shape and size times the
  // fold factor (see surface_remesh.hpp); a degenerate normal on either side
  // leaves the edge to the degenerate triangle's own score. A turn within the
  // two cone floors is within the cones' separation, so cones are computed
  // only for edges turning beyond the floors or closing a convex wedge
  // sharper than the minimum angle.
  uint32_t score(const Ref& self, Staged& staged, const std::vector<int64_t>& removed) {
    const F t = triangle(self, staged);
    const double value = shape_size(points[static_cast<size_t>(t[0])],
                                    points[static_cast<size_t>(t[1])],
                                    points[static_cast<size_t>(t[2])]);
    if (!(value > 0)) return 0;
    const V own = normal(t);
    const double wedge_target = targets.minimum_angle_degrees * kPi / 180;
    double fold = 1;
    for (int k = 0; k < 3; ++k) {
      const auto neighbor = across(t[k], t[(k + 1) % 3], staged, removed);
      if (!neighbor) continue;
      const F& other_triangle = triangle(*neighbor, staged);
      const V other = normal(other_triangle);
      if (!(length(other) > 0)) continue;
      const double turn = std::atan2(length(cross(own, other)), dot(own.data(), other.data()));
      // The neighbor's far corner below t's plane, seen from the solid side,
      // makes the edge convex: the solid wedge there is pi - turn.
      int64_t far = other_triangle[0];
      for (int64_t v : other_triangle)
        if (v != t[k] && v != t[(k + 1) % 3]) far = v;
      const double height = solid_side * dot(own.data(),
                                             difference(points[static_cast<size_t>(far)],
                                                        points[static_cast<size_t>(t[k])])
                                                 .data());
      const bool sharp_convex = height < 0 && kPi - turn < wedge_target;
      // The slack absorbs the rounding of the angle evaluations.
      if (!sharp_convex &&
          turn <= cone_floor(self, staged) + cone_floor(*neighbor, staged) + kTurnSlack)
        continue;
      const NormalCone first = cone(self, staged);
      const NormalCone& second = cone(*neighbor, staged);
      const double allowed = cone_separation(first, second);
      if (allowed < kPi && turn > allowed + kTurnSlack)
        fold = std::min(fold, (kPi - turn) / (kPi - allowed));
      if (!sharp_convex) continue;
      // Admitted normals lie in their cones, so every admitted pair turns by
      // at least the cones' forced turn. A source forcing a wedge sharper
      // than the minimum angle keeps it; otherwise the wedge is held to it.
      if (forced_turn(first, second) < kPi - wedge_target)
        fold = std::min(fold, (kPi - turn) / wedge_target);
    }
    return quantize(value * std::max(fold, 0.0));
  }
  static uint32_t quantize(double value) {
    return static_cast<uint32_t>(
        std::clamp(std::floor(value * kScoreLevels), 0.0, static_cast<double>(kScoreLevels)));
  }
  // Rescores a live triangle in the current state and requeues it.
  void rescore(int64_t id) {
    const auto slot = static_cast<size_t>(id);
    if (queue_enabled) queue.erase(id);
    Staged none;
    state.set_score(slot, score(Ref{id, 0}, none, {}));
    if (queue_enabled && state.score(slot) < kScoreLevels && !state.stuck(slot)) queue.insert(id);
  }
  // Grid cells met by the triangle's bounding box. Two triangles in contact
  // have overlapping boxes, and a point of the overlap lies in a cell of both
  // ranges (floor is monotone), so exact ranges find every contact pair.
  std::array<int64_t, 6> bucket_range(const F& t) const {
    const auto [lo, hi] = face_box(t);
    std::array<int64_t, 6> range{};
    for (int d = 0; d < 3; ++d) {
      range[d] = static_cast<int64_t>(std::floor(lo[d] / h));
      range[3 + d] = static_cast<int64_t>(std::floor(hi[d] / h));
    }
    return range;
  }
  std::pair<V, V> face_box(const F& t) const {
    V lo = points[static_cast<size_t>(t[0])], hi = lo;
    for (int k = 1; k < 3; ++k) {
      const V point = points[static_cast<size_t>(t[k])];
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], point[d]);
        hi[d] = std::max(hi[d], point[d]);
      }
    }
    return {lo, hi};
  }
  bool degenerate(const F& t) const {
    return phx::mc::geometry::collinear(points[static_cast<size_t>(t[0])],
                                        points[static_cast<size_t>(t[1])],
                                        points[static_cast<size_t>(t[2])]);
  }
  static bool disjoint(const V& a, const V& b, const V& c, const V& d) {
    for (int axis = 0; axis < 3; ++axis)
      if (b[axis] < c[axis] || d[axis] < a[axis]) return true;
    return false;
  }
  uint32_t morton(const F& t) const {
    uint32_t coordinate[3]{};
    for (int d = 0; d < 3; ++d) {
      const double middle = (points[static_cast<size_t>(t[0])][d] / 3 +
                             points[static_cast<size_t>(t[1])][d] / 3 +
                             points[static_cast<size_t>(t[2])][d] / 3) / h;
      const double fraction = middle - std::floor(middle);
      coordinate[d] = static_cast<uint32_t>(std::clamp(fraction * 1024, 0.0, 1023.0));
    }
    uint32_t key = 0;
    for (unsigned bit = 0; bit < 10; ++bit)
      for (unsigned d = 0; d < 3; ++d) key |= ((coordinate[d] >> bit) & 1u) << (3 * bit + d);
    return key;
  }
  static void extend_group(BucketGroup& group, const BucketGroup& value) {
    for (int d = 0; d < 3; ++d) {
      group.lower[d] = std::min(group.lower[d], value.lower[d]);
      group.upper[d] = std::max(group.upper[d], value.upper[d]);
    }
    group.invalid = group.invalid || value.invalid;
  }
  // The append tail is indexed incrementally. Bounds only expand until the
  // next rebuild; incomplete final blocks are updated on every appended face.
  void update_tail(Bucket& bucket, size_t block_index) {
    const BucketBlock& block = bucket.blocks[block_index];
    BucketGroup value{block.lower, block.upper, block.invalid};
    size_t index = block_index - bucket.indexed_blocks;
    for (size_t level = 0;; ++level) {
      if (level == bucket.tail_levels.size()) {
        bucket.tail_levels.emplace_back();
        if (level != 0) {
          const auto& children = bucket.tail_levels[level - 1];
          value = children.front();
          for (size_t child = 1; child < children.size(); ++child)
            extend_group(value, children[child]);
        }
        bucket.tail_levels[level].push_back(value);
        break;
      }
      auto& groups = bucket.tail_levels[level];
      const size_t parent = index / 8;
      if (parent == groups.size()) groups.push_back(value);
      else extend_group(groups[parent], value);
      value = groups[parent];
      index = parent;
      if (groups.size() == 1 && level + 1 == bucket.tail_levels.size()) break;
    }
  }
  void index_bucket(Bucket& bucket) {
    bucket.levels.clear();
    bucket.tail_levels.clear();
    bucket.indexed_blocks = bucket.blocks.size();
    if (bucket.indexed_blocks != 0 && bucket.size() % kBucketBlockRows != 0)
      --bucket.indexed_blocks;
    size_t children = bucket.indexed_blocks;
    while (children != 0) {
      const size_t level = bucket.levels.size();
      bucket.levels.emplace_back();
      auto& groups = bucket.levels.back();
      groups.reserve(children / 8 + (children % 8 != 0));
      for (size_t first = 0; first < children; first += 8) {
        const size_t end = std::min(children, first + 8);
        BucketGroup group;
        if (level == 0) {
          group = {bucket.blocks[first].lower, bucket.blocks[first].upper,
                   bucket.blocks[first].invalid};
        } else {
          group = bucket.levels[level - 1][first];
        }
        for (size_t child = first + 1; child < end; ++child) {
          const BucketGroup next = level == 0
              ? BucketGroup{bucket.blocks[child].lower, bucket.blocks[child].upper,
                            bucket.blocks[child].invalid}
              : bucket.levels[level - 1][child];
          for (int d = 0; d < 3; ++d) {
            group.lower[d] = std::min(group.lower[d], next.lower[d]);
            group.upper[d] = std::max(group.upper[d], next.upper[d]);
          }
          group.invalid = group.invalid || next.invalid;
        }
        groups.push_back(group);
      }
      children = groups.size();
      if (children == 1) break;
    }
    if (bucket.indexed_blocks < bucket.blocks.size())
      update_tail(bucket, bucket.indexed_blocks);
  }
  void append_bucket(Bucket& bucket, int64_t id) {
    const size_t at = bucket.size();
    bucket.push(id);
    if (!bucket_index_ready) return;
    const F t = triangles[static_cast<size_t>(id)];
    const auto [lo, hi] = face_box(t);
    if (at % kBucketBlockRows == 0) {
      bucket.blocks.push_back({lo, hi, degenerate(t)});
    } else {
      auto& block = bucket.blocks.back();
      for (int d = 0; d < 3; ++d) {
        block.lower[d] = std::min(block.lower[d], lo[d]);
        block.upper[d] = std::max(block.upper[d], hi[d]);
      }
      block.invalid = block.invalid || degenerate(t);
    }
    if (!bucket_rebuilding) {
      update_tail(bucket, bucket.blocks.size() - 1);
      if (bucket.size() >= bucket.reindex_at) rebuild_bucket(bucket);
    }
  }
  void rebuild_bucket(Bucket& bucket) {
    std::vector<std::pair<uint32_t, int64_t>> order;
    order.reserve(bucket.size());
    for (size_t at = 0; at < bucket.size(); ++at) {
      const int64_t id = bucket.get(at);
      if (state.alive(static_cast<size_t>(id)))
        order.emplace_back(morton(triangles[static_cast<size_t>(id)]), id);
    }
    std::sort(order.begin(), order.end());
    bucket.clear();
    bucket.blocks.clear();
    const size_t size = order.size();
    const size_t extra = std::max(size_t{4096}, size / 8);
    bucket.reindex_at = size > std::numeric_limits<size_t>::max() - extra
                            ? std::numeric_limits<size_t>::max()
                            : size + extra;
    const size_t desired_blocks = bucket.reindex_at / kBucketBlockRows +
                                  (bucket.reindex_at % kBucketBlockRows != 0);
    const size_t current_blocks = size / kBucketBlockRows + (size % kBucketBlockRows != 0);
    bucket.blocks.reserve(desired_blocks <= bucket.blocks.max_size() ? desired_blocks : current_blocks);
    bucket_rebuilding = true;
    for (const auto& item : order) append_bucket(bucket, item.second);
    bucket_rebuilding = false;
    index_bucket(bucket);
  }
  void prepare_buckets() {
    bucket_index_ready = true;
    for (auto& [key, bucket] : buckets) rebuild_bucket(bucket);
  }
  // Blocks conservatively bound registered faces. Reindexing drops dead rows;
  // until then gathering ignores them.
  void gather_candidates(const F& t, const std::vector<int64_t>& removed,
                         std::vector<int64_t>& out) {
    out.clear();
    const auto range = bucket_range(t);
    const auto [lo, hi] = face_box(t);
    for (int64_t i = range[0]; i <= range[3]; ++i)
      for (int64_t j = range[1]; j <= range[4]; ++j)
        for (int64_t k = range[2]; k <= range[5]; ++k) {
          const auto found = buckets.find(cell_key(i, j, k));
          if (found == buckets.end()) continue;
          const auto& cell = found->second;
          const auto inspect = [&](size_t block_index) {
            const BucketBlock& block = cell.blocks[block_index];
            if (!block.invalid && disjoint(lo, hi, block.lower, block.upper)) return;
            const size_t begin = block_index * kBucketBlockRows;
            const size_t end = begin + std::min(kBucketBlockRows, cell.size() - begin);
            for (size_t at = begin; at < end; ++at) {
              const int64_t id = cell.get(at);
              if (!state.alive(static_cast<size_t>(id))) continue;
              if (std::binary_search(removed.begin(), removed.end(), id)) continue;
              const F candidate = triangles[static_cast<size_t>(id)];
              const auto [candidate_lo, candidate_hi] = face_box(candidate);
              if (disjoint(lo, hi, candidate_lo, candidate_hi) &&
                  (!block.invalid || !degenerate(candidate))) continue;
              out.push_back(id);
            }
          };
          const auto visit = [&](auto&& self, const auto& levels, size_t level, size_t index,
                                 size_t offset, size_t block_count) -> void {
            const BucketGroup& group = levels[level][index];
            if (!group.invalid && disjoint(lo, hi, group.lower, group.upper)) return;
            const size_t first = index * 8;
            const size_t count = level == 0 ? block_count : levels[level - 1].size();
            const size_t end = std::min(count, first + 8);
            for (size_t child = first; child < end; ++child) {
              if (level == 0) inspect(offset + child);
              else self(self, levels, level - 1, child, offset, block_count);
            }
          };
          if (!cell.levels.empty())
            visit(visit, cell.levels, cell.levels.size() - 1, 0, 0, cell.indexed_blocks);
          if (!cell.tail_levels.empty())
            visit(visit, cell.tail_levels, cell.tail_levels.size() - 1, 0, cell.indexed_blocks,
                  cell.blocks.size() - cell.indexed_blocks);
        }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
  }
  // Registers a triangle in the buckets and stars; its score depends on its
  // edge neighbors, so callers rescore once the edit's triangles are all in.
  void insert(int64_t id) {
    const F& t = triangles[static_cast<size_t>(id)];
    const auto range = bucket_range(t);
    for (int64_t i = range[0]; i <= range[3]; ++i)
      for (int64_t j = range[1]; j <= range[4]; ++j)
        for (int64_t k = range[2]; k <= range[5]; ++k)
          append_bucket(buckets[cell_key(i, j, k)], id);
    if (compact_stars) {
      for (int64_t v : t) edited_star.append(static_cast<size_t>(v), id);
    } else {
      for (int64_t v : t) star[static_cast<size_t>(v)].push_back(id);
    }
  }
  int64_t add_triangle(const F& t, double floor, const std::optional<NormalCone>& cone) {
    const int64_t id = static_cast<int64_t>(triangles.size());
    triangles.push_back(t);
    state.push();
    floors.push_back(floor);
    if (cone) cones.emplace(id, *cone);
    insert(id);
    return id;
  }
  void kill(int64_t id) {
    state.set_alive(static_cast<size_t>(id), false);
    if (queue_enabled) queue.erase(id);
  }
  std::vector<int64_t> live_star(int64_t v) {
    if (compact_stars) {
      std::vector<int64_t> incident;
      const size_t vertex = static_cast<size_t>(v);
      if (borrowed_star_offsets != nullptr && vertex < borrowed_star_vertices) {
        const size_t begin = static_cast<size_t>(borrowed_star_offsets[vertex]);
        const size_t end = static_cast<size_t>(borrowed_star_offsets[vertex + 1]);
        for (size_t i = begin; i < end; ++i) {
          const int64_t id = borrowed_star_rows[i];
          if (state.alive(static_cast<size_t>(id))) incident.push_back(id);
        }
      } else if (vertex + 1 < base_star_offsets.size()) {
        const size_t end = base_star_offsets[vertex + 1];
        for (size_t i = base_star_offsets[vertex]; i < end; ++i) {
          const int64_t id = static_cast<int64_t>(base_star_ids.empty() ? wide_star_ids[i]
                                                                        : base_star_ids[i]);
          if (state.alive(static_cast<size_t>(id))) incident.push_back(id);
        }
      }
      edited_star.collect(vertex, state, incident);
      return incident;
    }
    auto& incident = star[static_cast<size_t>(v)];
    incident.erase(std::remove_if(incident.begin(), incident.end(),
                                  [&](int64_t id) { return !state.alive(static_cast<size_t>(id)); }),
                   incident.end());
    return incident;
  }
  std::vector<int64_t> edge_triangles(int64_t a, int64_t b) {
    std::vector<int64_t> found;
    for (int64_t id : live_star(a)) {
      const F& t = triangles[static_cast<size_t>(id)];
      if (t[0] == b || t[1] == b || t[2] == b) found.push_back(id);
    }
    return found;
  }
  std::vector<int64_t> neighbors(int64_t v) {
    std::vector<int64_t> result;
    for (int64_t id : live_star(v))
      for (int64_t w : triangles[static_cast<size_t>(id)])
        if (w != v) result.push_back(w);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }
  V project(const V& x) const { return predicate.project(x); }


  struct Edit {
    std::vector<int64_t> removed;
    std::vector<std::pair<int64_t, V>> moved;
    bool adds_vertex = false;
    V added{};
    std::vector<int64_t> dropped;  // vertices whose stars the edit empties
    std::vector<F> created;
  };

  static std::array<int64_t, 3> sorted(const F& t) {
    std::array<int64_t, 3> key{t[0], t[1], t[2]};
    std::sort(key.begin(), key.end());
    return key;
  }

  // Stages the edit's positions, checks every admission condition, and either
  // applies it or restores the previous positions. `kImprove` edits must raise
  // every created score above the replaced patch's worst score; `kPreserve`
  // edits (regularization) must not lower it.
  enum class Admission { kImprove, kPreserve };
  bool apply_if_admissible(const Edit& edit, Admission admission = Admission::kImprove) {
    if (edit.created.empty() || edit.removed.empty()) return false;
    uint32_t removed_minimum = kScoreLevels;
    V reference{0, 0, 0};
    std::vector<std::array<int64_t, 3>> removed_keys;
    for (int64_t id : edit.removed) {
      if (!state.alive(static_cast<size_t>(id))) return false;
      removed_minimum = std::min(removed_minimum, state.score(static_cast<size_t>(id)));
      const F& t = triangles[static_cast<size_t>(id)];
      reference = add(reference, cross(difference(points[static_cast<size_t>(t[1])],
                                                  points[static_cast<size_t>(t[0])]),
                                       difference(points[static_cast<size_t>(t[2])],
                                                  points[static_cast<size_t>(t[0])])));
      removed_keys.push_back(sorted(t));
    }
    std::sort(removed_keys.begin(), removed_keys.end());
    std::vector<int64_t> removed_ids = edit.removed;
    std::sort(removed_ids.begin(), removed_ids.end());
    std::vector<std::pair<int64_t, V>> saved;
    std::vector<VertexEvidence> saved_evidence;
    for (const auto& [vertex, position] : edit.moved) {
      saved.emplace_back(vertex, points[static_cast<size_t>(vertex)]);
      saved_evidence.push_back(evidence[static_cast<size_t>(vertex)]);
    }
    bool pushed = false;
    auto restore = [&]() {
      for (size_t i = 0; i < saved.size(); ++i) {
        points.set(static_cast<size_t>(saved[i].first), saved[i].second);
        evidence.set(static_cast<size_t>(saved[i].first), saved_evidence[i]);
      }
      if (pushed) {
        points.pop_back();
        evidence.pop_back();
        if (compact_stars) edited_star.pop_vertex();
        else star.pop_back();
      }
      return false;
    };
    for (const auto& [vertex, position] : edit.moved) {
      if (!point_in_domain(position)) return restore();
      points.set(static_cast<size_t>(vertex), position);
      evidence.set(static_cast<size_t>(vertex), predicate.evaluate(position));
    }
    if (edit.adds_vertex) {
      points.push_back(edit.added);
      evidence.push_back(predicate.evaluate(edit.added));
      if (compact_stars) edited_star.add_vertex();
      else star.emplace_back();
      pushed = true;
      if (!point_in_domain(edit.added)) return restore();
    }
    std::vector<std::array<int64_t, 3>> created_keys;
    for (size_t row = 0; row < edit.created.size(); ++row) {
      const F& t = edit.created[row];
      if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) return restore();
      const V& a = points[static_cast<size_t>(t[0])];
      const V& b = points[static_cast<size_t>(t[1])];
      const V& c = points[static_cast<size_t>(t[2])];
      if (phx::mc::geometry::collinear(a, b, c, 0)) return restore();
      // Shape and size alone bound the score from above (fold <= 1): a
      // cheap refusal before the band, contact and source-cone checks.
      const uint32_t bound = quantize(shape_size(a, b, c));
      if (admission == Admission::kImprove ? bound <= removed_minimum : bound < removed_minimum)
        return restore();
      const V normal = cross(difference(b, a), difference(c, a));
      if (!(dot(normal.data(), reference.data()) > 0)) return restore();
      if (!admitted(t)) return restore();
      created_keys.push_back(sorted(t));
    }
    std::sort(created_keys.begin(), created_keys.end());
    if (std::adjacent_find(created_keys.begin(), created_keys.end()) != created_keys.end())
      return restore();
    // A created face may not duplicate a surviving face.
    for (const F& t : edit.created)
      for (int64_t id : live_star(t[0])) {
        if (std::binary_search(removed_ids.begin(), removed_ids.end(), id)) continue;
        if (sorted(triangles[static_cast<size_t>(id)]) == sorted(t)) return restore();
      }
    // Exact contact checks preserve disjoint, shared-vertex and shared-edge cases.
    std::vector<int64_t> candidates;
    for (size_t first = 0; first < edit.created.size(); ++first) {
      const F& t = edit.created[first];
      const std::array<V, 3> t_values{points[static_cast<size_t>(t[0])],
                                      points[static_cast<size_t>(t[1])],
                                      points[static_cast<size_t>(t[2])]};
      for (size_t second = first + 1; second < edit.created.size(); ++second) {
        const F& u = edit.created[second];
        const std::array<V, 3> u_values{points[static_cast<size_t>(u[0])],
                                        points[static_cast<size_t>(u[1])],
                                        points[static_cast<size_t>(u[2])]};
        if (!legal_triangle_contact(t_values, t, u_values, u))
          return restore();
      }
      gather_candidates(t, removed_ids, candidates);
      for (int64_t id : candidates) {
        const F& u = triangles[static_cast<size_t>(id)];
        const std::array<V, 3> u_values{points[static_cast<size_t>(u[0])],
                                        points[static_cast<size_t>(u[1])],
                                        points[static_cast<size_t>(u[2])]};
        if (!legal_triangle_contact(t_values, t, u_values, u))
          return restore();
      }
    }
    // Full scores last: they may need the created triangles' source cones.
    Staged staged;
    staged.created = &edit.created;
    staged.cones.resize(edit.created.size());
    for (const F& t : edit.created) staged.floors.push_back(floor_of(t));
    for (size_t row = 0; row < edit.created.size(); ++row) {
      const uint32_t created_score = score(Ref{-1, row}, staged, removed_ids);
      if (admission == Admission::kImprove ? created_score <= removed_minimum
                                           : created_score < removed_minimum)
        return restore();
    }
    // Surviving edge neighbors of the created patch change only their fold
    // factor. One that does not keep its score is held to the created
    // triangles' bound, so in the multiset argument it is a replaced element
    // whose successor still exceeds the patch's worst score.
    std::vector<int64_t> ring;
    for (const F& t : edit.created)
      for (int k = 0; k < 3; ++k) {
        const int64_t p = t[k], q = t[(k + 1) % 3];
        for (int64_t id : live_star(p)) {
          if (std::binary_search(removed_ids.begin(), removed_ids.end(), id)) continue;
          const F& u = triangles[static_cast<size_t>(id)];
          for (int j = 0; j < 3; ++j)
            if (u[j] == q && u[(j + 1) % 3] == p) ring.push_back(id);
        }
      }
    std::sort(ring.begin(), ring.end());
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
    for (int64_t id : ring) {
      const uint32_t ring_score = score(Ref{id, 0}, staged, removed_ids);
      if (ring_score >= state.score(static_cast<size_t>(id))) continue;
      if (admission == Admission::kImprove ? ring_score <= removed_minimum
                                           : ring_score < removed_minimum)
        return restore();
    }
    for (int64_t id : edit.removed) kill(id);
    for (int64_t vertex : edit.dropped) {
      if (compact_stars) edited_star.clear(static_cast<size_t>(vertex));
      else star[static_cast<size_t>(vertex)].clear();
    }
    std::vector<int64_t> touched;
    std::vector<int64_t> created_ids;
    for (size_t row = 0; row < edit.created.size(); ++row) {
      const F& t = edit.created[row];
      created_ids.push_back(add_triangle(t, staged.floors[row], staged.cones[row]));
      touched.insert(touched.end(), t.begin(), t.end());
    }
    for (int64_t id : created_ids) rescore(id);
    for (int64_t id : ring) rescore(id);
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (int64_t v : touched)
      for (int64_t id : live_star(v))
        if (state.stuck(static_cast<size_t>(id))) {
          state.set_stuck(static_cast<size_t>(id), false);
          if (queue_enabled && state.score(static_cast<size_t>(id)) < kScoreLevels)
            queue.insert(id);
        }
    return true;
  }

  bool split(int64_t a, int64_t b, const V& position) {
    const auto incident = edge_triangles(a, b);
    if (incident.size() != 2) return false;
    Edit edit;
    edit.removed = incident;
    edit.adds_vertex = true;
    edit.added = position;
    const int64_t m = static_cast<int64_t>(points.size());
    for (int64_t id : incident) {
      const F& t = triangles[static_cast<size_t>(id)];
      for (int i = 0; i < 3; ++i) {
        const int64_t p = t[i], q = t[(i + 1) % 3], r = t[(i + 2) % 3];
        if ((p == a && q == b) || (p == b && q == a)) {
          edit.created.push_back({p, m, r});
          edit.created.push_back({m, q, r});
        }
      }
    }
    if (!apply_if_admissible(edit)) return false;
    ++splits;
    return true;
  }

  // `kPreserve` flips are valence-driven: admitted only when they strictly
  // lower the squared deviation from valence six over the four patch vertices.
  bool flip(int64_t a, int64_t b, Admission admission = Admission::kImprove) {
    const auto incident = edge_triangles(a, b);
    if (incident.size() != 2) return false;
    int64_t c = -1, d = -1;
    for (int64_t id : incident) {
      const F& t = triangles[static_cast<size_t>(id)];
      for (int i = 0; i < 3; ++i) {
        if (t[i] == a && t[(i + 1) % 3] == b) c = t[(i + 2) % 3];
        if (t[i] == b && t[(i + 1) % 3] == a) d = t[(i + 2) % 3];
      }
    }
    if (c < 0 || d < 0 || c == d || !edge_triangles(c, d).empty()) return false;
    if (admission == Admission::kPreserve) {
      const int64_t va = static_cast<int64_t>(neighbors(a).size()),
                    vb = static_cast<int64_t>(neighbors(b).size()),
                    vc = static_cast<int64_t>(neighbors(c).size()),
                    vd = static_cast<int64_t>(neighbors(d).size());
      auto deviation = [](int64_t valence) { return (valence - 6) * (valence - 6); };
      if (va <= 3 || vb <= 3) return false;
      if (deviation(va - 1) + deviation(vb - 1) + deviation(vc + 1) + deviation(vd + 1) >=
          deviation(va) + deviation(vb) + deviation(vc) + deviation(vd))
        return false;
    }
    Edit edit;
    edit.removed = incident;
    edit.created = {{c, a, d}, {c, d, b}};
    if (!apply_if_admissible(edit, admission)) return false;
    ++flips;
    return true;
  }

  // Collapses u into v and places v at `position`.
  bool collapse(int64_t u, int64_t v, const V& position,
                Admission admission = Admission::kImprove) {
    const auto incident = edge_triangles(u, v);
    if (incident.size() != 2) return false;
    std::vector<int64_t> opposite;
    for (int64_t id : incident)
      for (int64_t w : triangles[static_cast<size_t>(id)])
        if (w != u && w != v) opposite.push_back(w);
    std::sort(opposite.begin(), opposite.end());
    const auto u_neighbors = neighbors(u), v_neighbors = neighbors(v);
    std::vector<int64_t> common;
    std::set_intersection(u_neighbors.begin(), u_neighbors.end(), v_neighbors.begin(),
                          v_neighbors.end(), std::back_inserter(common));
    if (common != opposite || opposite.size() != 2) return false;
    for (int64_t w : opposite)
      if (neighbors(w).size() <= 3) return false;
    Edit edit;
    std::vector<int64_t> patch = live_star(u);
    for (int64_t id : live_star(v)) patch.push_back(id);
    std::sort(patch.begin(), patch.end());
    patch.erase(std::unique(patch.begin(), patch.end()), patch.end());
    edit.removed = patch;
    for (int64_t id : patch) {
      F t = triangles[static_cast<size_t>(id)];
      const bool has_u = t[0] == u || t[1] == u || t[2] == u;
      const bool has_v = t[0] == v || t[1] == v || t[2] == v;
      if (has_u && has_v) continue;
      for (int64_t& w : t)
        if (w == u) w = v;
      edit.created.push_back(t);
    }
    edit.moved.emplace_back(v, position);
    edit.dropped = {u};
    if (!apply_if_admissible(edit, admission)) return false;
    ++collapses;
    return true;
  }

  bool relocate(int64_t v, const V& position, Admission admission = Admission::kImprove) {
    if (position == points[static_cast<size_t>(v)]) return false;
    Edit edit;
    edit.removed = live_star(v);
    for (int64_t id : edit.removed) edit.created.push_back(triangles[static_cast<size_t>(id)]);
    edit.moved.emplace_back(v, position);
    if (!apply_if_admissible(edit, admission)) return false;
    ++relocations;
    return true;
  }

  V star_centroid(int64_t v) {
    V sum{0, 0, 0};
    double weight = 0;
    for (int64_t id : live_star(v)) {
      const F& t = triangles[static_cast<size_t>(id)];
      const V& a = points[static_cast<size_t>(t[0])];
      const V& b = points[static_cast<size_t>(t[1])];
      const V& c = points[static_cast<size_t>(t[2])];
      const double area = length(cross(difference(b, a), difference(c, a)));
      sum = add(sum, mul(add(add(a, b), c), area / 3));
      weight += area;
    }
    return weight > 0 ? mul(sum, 1 / weight) : points[static_cast<size_t>(v)];
  }

  // With connectivity fixed, increasing the sorted incident-edge multiset
  // increases the global sorted edge multiset. The ordinary preserve admission
  // still checks every score, orientation, band and contact constraint.
  bool relocate_for_size(int64_t vertex, const V& candidate) {
    if (!point_in_domain(candidate)) return false;
    const V original = points[static_cast<size_t>(vertex)];
    const auto adjacent = neighbors(vertex);
    std::vector<double> before, after;
    before.reserve(adjacent.size());
    after.reserve(adjacent.size());
    for (int64_t neighbor : adjacent) {
      const V point = points[static_cast<size_t>(neighbor)];
      before.push_back(length(difference(original, point)));
      after.push_back(length(difference(candidate, point)));
    }
    std::sort(before.begin(), before.end());
    std::sort(after.begin(), after.end());
    if (!(before < after)) return false;
    return relocate(vertex, candidate, Admission::kPreserve);
  }

  // The equilateral position of t's corner k over the opposite edge, in t's
  // plane on the corner's side; nullopt for a degenerate t.
  std::optional<V> ideal_apex(const F& t, int k) const {
    const V& p = points[static_cast<size_t>(t[k])];
    const V& a = points[static_cast<size_t>(t[(k + 1) % 3])];
    const V& b = points[static_cast<size_t>(t[(k + 2) % 3])];
    const V edge = difference(b, a);
    const V middle = mul(add(a, b), 0.5);
    V up = cross(cross(edge, difference(p, a)), edge);
    const double norm = length(up);
    if (!(norm > 0) || !std::isfinite(norm)) return std::nullopt;
    if (dot(up.data(), difference(p, middle).data()) < 0) up = mul(up, -1);
    return add(middle, mul(up, 0.5 * std::sqrt(3.0) * length(edge) / norm));
  }

  std::optional<V> collapse_apex_goal(int64_t a, int64_t b) {
    std::vector<int64_t> patch = live_star(a);
    for (int64_t id : live_star(b)) patch.push_back(id);
    std::sort(patch.begin(), patch.end());
    patch.erase(std::unique(patch.begin(), patch.end()), patch.end());
    V total{};
    size_t count = 0;
    for (int64_t id : patch) {
      const F face = triangles[static_cast<size_t>(id)];
      int corner = -1, matches = 0;
      for (int k = 0; k < 3; ++k)
        if (face[k] == a || face[k] == b) { corner = k; ++matches; }
      if (matches != 1) continue;
      if (const auto apex = ideal_apex(face, corner)) {
        total = add(total, *apex);
        ++count;
      }
    }
    if (count == 0) return std::nullopt;
    return mul(total, 1.0 / static_cast<double>(count));
  }

  // Relocations of t's corners part of the way toward the star centroid
  // (fractions 1/2, ..., 2^-10) and toward the corner's ideal apex (1, ...,
  // 2^-10), projected and as is. A full move that lifts t can push another
  // star triangle below t's score where a shorter one does not. The list is
  // finite and every admitted relocation is an ordinary improving edit.
  bool relocate_partially(const F& t) {
    for (int k = 0; k < 3; ++k) {
      const int64_t v = t[k];
      const V origin = points[static_cast<size_t>(v)];
      std::vector<std::pair<V, double>> goals{{star_centroid(v), 0.5}};
      if (const auto apex = ideal_apex(t, k)) goals.emplace_back(*apex, 1.0);
      for (const auto& [goal, first] : goals) {
        const V step = difference(goal, origin);
        for (double fraction = first; fraction >= 0x1p-10; fraction *= 0.5) {
          const V position = add(origin, mul(step, fraction));
          if (relocate(v, project(position)) || relocate(v, position)) return true;
        }
      }
    }
    return false;
  }

  bool improve(int64_t id) {
    const F t = triangles[static_cast<size_t>(id)];
    std::array<std::pair<double, int>, 3> edges;  // (length, local edge opposite corner k)
    for (int k = 0; k < 3; ++k)
      edges[k] = {length(difference(points[static_cast<size_t>(t[(k + 1) % 3])],
                                    points[static_cast<size_t>(t[(k + 2) % 3])])),
                  k};
    std::sort(edges.begin(), edges.end());
    auto endpoints = [&](int k) {
      return std::pair<int64_t, int64_t>{t[(k + 1) % 3], t[(k + 2) % 3]};
    };
    auto midpoint = [&](int64_t a, int64_t b) {
      return mul(add(points[static_cast<size_t>(a)], points[static_cast<size_t>(b)]), 0.5);
    };
    const auto [la, lb] = endpoints(edges[2].second);
    if (edges[2].first > (4.0 / 3.0) * h) {
      const V middle = midpoint(la, lb);
      if (split(la, lb, project(middle)) || split(la, lb, middle)) return true;
    }
    // The longest edge is opposite the largest angle.
    for (int rank = 2; rank >= 0; --rank) {
      const auto [a, b] = endpoints(edges[rank].second);
      if (flip(a, b)) return true;
    }
    const auto [sa, sb] = endpoints(edges[0].second);
    if (collapse(sa, sb, project(midpoint(sa, sb))) ||
        collapse(sa, sb, points[static_cast<size_t>(sb)]) ||
        collapse(sb, sa, points[static_cast<size_t>(sa)]))
      return true;
    for (int64_t v : t) {
      const V centroid = star_centroid(v);
      if (relocate(v, project(centroid)) || relocate(v, centroid) ||
          relocate(v, project(points[static_cast<size_t>(v)])))
        return true;
    }
    if (relocate_partially(t)) return true;
    const V middle = midpoint(la, lb);
    return split(la, lb, project(middle)) || split(la, lb, middle);
  }

  // Worst-first quality pass; terminates because every admitted edit strictly
  // decreases the multiset of quantized scores.
  void improve_worst_first() {
    for (size_t id = 0; id < triangles.size(); ++id)
      if (state.alive(id) && state.stuck(id)) {
        state.set_stuck(id, false);
        if (state.score(id) < kScoreLevels) queue.insert(static_cast<int64_t>(id));
      }
    while (!queue.empty()) {
      const int64_t id = queue.front();
      queue.erase(id);
      if (!state.alive(static_cast<size_t>(id))) continue;
      if (!improve(id)) state.set_stuck(static_cast<size_t>(id), true);
    }
  }

  // Short-edge collapses (below 4/5 h) and valence-driven flips under the same
  // certified admission, never lowering a patch's worst score. Every collapse
  // removes a vertex and every flip keeps the vertex count while strictly
  // lowering the integer valence energy, so the pair (vertex count, valence
  // energy) decreases lexicographically and the sweeps terminate. A short
  // edge collapses to its projected midpoint or either endpoint; only when the
  // targets request admission-limited coarsening are the further collapse
  // positions and the size relocations tried, so every other caller keeps the
  // plain quality/regularization outputs.
  void regularize(bool size_only = false) {
    const double short_edge = 0.8 * h;
    size_t sweep = 0;
    bool changed = true;
    while (changed) {
      ++sweep;
      changed = false;
      // Snapshot the sweep's incidence, not a 3F array of 64-bit edge pairs.
      // Face rows are immutable even after they die, so this enumerates exactly
      // the original sorted unique (a,b) edge list while edits proceed.
      const size_t vertices = points.size(), rows = triangles.size();
      const bool borrowed_sweep = size_only && sweep == 1 &&
          borrowed_star_offsets != nullptr && vertices == borrowed_star_vertices &&
          rows == borrowed_star_faces && splits == 0 && collapses == 0 &&
          flips == 0 && relocations == 0;
      std::vector<size_t> offsets;
      std::vector<uint32_t> narrow_rows;
      std::vector<uint64_t> wide_rows;
      if (!borrowed_sweep) {
        offsets.assign(vertices + 1, 0);
        for (size_t id = 0; id < rows; ++id) {
          if (!state.alive(id)) continue;
          for (int64_t v : triangles[id]) ++offsets[static_cast<size_t>(v) + 1];
        }
        for (size_t v = 1; v < offsets.size(); ++v) offsets[v] += offsets[v - 1];
        if (rows <= std::numeric_limits<uint32_t>::max()) narrow_rows.resize(offsets.back());
        else wide_rows.resize(offsets.back());
        for (size_t id = 0; id < rows; ++id) {
          if (!state.alive(id)) continue;
          for (int64_t v : triangles[id]) {
            const size_t slot = offsets[static_cast<size_t>(v)]++;
            if (narrow_rows.empty()) wide_rows[slot] = id;
            else narrow_rows[slot] = static_cast<uint32_t>(id);
          }
        }
        for (size_t v = vertices; v > 0; --v) offsets[v] = offsets[v - 1];
        offsets[0] = 0;
      }
      std::vector<int64_t> neighbors;
      for (size_t vertex = 0; vertex < vertices; ++vertex) {
        const int64_t a = static_cast<int64_t>(vertex);
        neighbors.clear();
        const size_t begin = borrowed_sweep ? static_cast<size_t>(borrowed_star_offsets[vertex])
                                            : offsets[vertex];
        const size_t end = borrowed_sweep ? static_cast<size_t>(borrowed_star_offsets[vertex + 1])
                                          : offsets[vertex + 1];
        for (size_t slot = begin; slot < end; ++slot) {
          const size_t id = borrowed_sweep ? static_cast<size_t>(borrowed_star_rows[slot])
              : (narrow_rows.empty() ? wide_rows[slot] : narrow_rows[slot]);
          const F t = triangles[id];
          for (int k = 0; k < 3; ++k) {
            const int64_t first = t[k], second = t[(k + 1) % 3];
            if (std::min(first, second) == a) neighbors.push_back(std::max(first, second));
          }
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        for (int64_t b : neighbors) {
          if (edge_triangles(a, b).size() != 2) continue;
          const V& pa = points[static_cast<size_t>(a)];
          const V& pb = points[static_cast<size_t>(b)];
          if (length(difference(pa, pb)) < short_edge) {
            const V middle = mul(add(pa, pb), 0.5);
            const V keep_a = pa, keep_b = pb;
            if (collapse(a, b, project(middle), Admission::kPreserve) ||
                collapse(a, b, keep_b, Admission::kPreserve) ||
                collapse(b, a, keep_a, Admission::kPreserve)) {
              changed = true;
              continue;
            }
            if (!size_only && targets.admission_limited_coarsening) {
              if (collapse(a, b, project(add(mul(keep_a, 0.75), mul(keep_b, 0.25))),
                           Admission::kPreserve) ||
                  collapse(a, b, project(add(mul(keep_a, 0.25), mul(keep_b, 0.75))),
                           Admission::kPreserve)) {
                changed = true;
                continue;
              }
              const V first_goal = star_centroid(a), second_goal = star_centroid(b);
              const V goal = mul(add(first_goal, second_goal), 0.5);
              if (collapse(a, b, project(goal), Admission::kPreserve) ||
                  collapse(a, b, project(mul(add(middle, goal), 0.5)), Admission::kPreserve) ||
                  collapse(a, b, project(first_goal), Admission::kPreserve) ||
                  collapse(a, b, project(second_goal), Admission::kPreserve) ||
                  collapse(a, b, project(mul(add(middle, first_goal), 0.5)), Admission::kPreserve) ||
                  collapse(a, b, project(mul(add(middle, second_goal), 0.5)), Admission::kPreserve)) {
                changed = true;
                continue;
              }
              if (const auto apex = collapse_apex_goal(a, b)) {
                if (collapse(a, b, project(*apex), Admission::kPreserve) ||
                    collapse(a, b, project(mul(add(middle, *apex), 0.5)), Admission::kPreserve)) {
                  changed = true;
                  continue;
                }
              }
              if (relocate_for_size(a, project(first_goal)) ||
                  relocate_for_size(b, project(second_goal)) ||
                  relocate_for_size(a, project(mul(add(keep_a, first_goal), 0.5))) ||
                  relocate_for_size(b, project(mul(add(keep_b, second_goal), 0.5))) ||
                  relocate_for_size(a, project(add(mul(keep_a, 0.75), mul(first_goal, 0.25)))) ||
                  relocate_for_size(b, project(add(mul(keep_b, 0.75), mul(second_goal, 0.25))))) {
                changed = true;
                continue;
              }
            }
          }
          if (!size_only && flip(a, b, Admission::kPreserve)) changed = true;
        }
      }
    }
  }

  // The band-centering offset of a vertex at `position`: minus `fraction`
  // times the mean sagitta (centroid minus its projection) of the triangles
  // of `star` using `vertex`, whose corners `place` positions. Moving a vertex
  // by it shares the band between the vertices and the triangle interiors
  // instead of leaving the vertices on the represented surface and every
  // interior on one side of it.
  template <class Place>
  V centered(const V& position, const std::vector<F>& star, int64_t vertex, double fraction,
             const Place& place) const {
    V sum{0, 0, 0};
    int uses = 0;
    for (const F& t : star) {
      if (t[0] != vertex && t[1] != vertex && t[2] != vertex) continue;
      const V centroid = mul(add(add(place(t[0]), place(t[1])), place(t[2])), 1.0 / 3.0);
      sum = add(sum, difference(centroid, project(centroid)));
      ++uses;
    }
    return uses == 0 ? position
                     : add(position, mul(sum, -fraction / static_cast<double>(uses)));
  }

  // Collapses u into v at the projected edge midpoint and, in the same edit,
  // relaxes the 1-ring of the merged vertex toward the projected average of
  // its post-collapse neighbors (two damped Jacobi steps) and centers the
  // moved vertices in the band. One admission checks only the final state,
  // so the area freed by the removed vertex is spread over the ring instead of
  // concentrated in v's star. Centering fractions 1/2, 1/4 and 0 are tried.
  bool collapse_relaxed(int64_t u, int64_t v) {
    constexpr int kRelaxationSteps = 2;
    constexpr double kRelaxationWeight = 0.5;
    const auto incident = edge_triangles(u, v);
    if (incident.size() != 2) return false;
    std::vector<int64_t> opposite;
    for (int64_t id : incident)
      for (int64_t w : triangles[static_cast<size_t>(id)])
        if (w != u && w != v) opposite.push_back(w);
    std::sort(opposite.begin(), opposite.end());
    const auto u_neighbors = neighbors(u), v_neighbors = neighbors(v);
    std::vector<int64_t> common;
    std::set_intersection(u_neighbors.begin(), u_neighbors.end(), v_neighbors.begin(),
                          v_neighbors.end(), std::back_inserter(common));
    if (common != opposite || opposite.size() != 2) return false;
    for (int64_t w : opposite)
      if (neighbors(w).size() <= 3) return false;
    std::vector<int64_t> ring = u_neighbors;
    ring.insert(ring.end(), v_neighbors.begin(), v_neighbors.end());
    std::sort(ring.begin(), ring.end());
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
    ring.erase(std::remove_if(ring.begin(), ring.end(),
                              [&](int64_t w) { return w == u || w == v; }),
               ring.end());
    const auto row = [&](int64_t w) -> std::ptrdiff_t {
      const auto found = std::lower_bound(ring.begin(), ring.end(), w);
      return found != ring.end() && *found == w ? found - ring.begin() : -1;
    };
    std::vector<std::vector<int64_t>> adjacency(ring.size());
    for (size_t i = 0; i < ring.size(); ++i) {
      auto around = neighbors(ring[i]);
      for (int64_t& w : around)
        if (w == u) w = v;
      std::sort(around.begin(), around.end());
      around.erase(std::unique(around.begin(), around.end()), around.end());
      adjacency[i] = std::move(around);
    }
    const V merged = project(
        mul(add(points[static_cast<size_t>(u)], points[static_cast<size_t>(v)]), 0.5));
    std::vector<V> position(ring.size());
    for (size_t i = 0; i < ring.size(); ++i) position[i] = points[static_cast<size_t>(ring[i])];
    const auto current = [&](int64_t w) -> V {
      if (w == v) return merged;
      const auto i = row(w);
      return i >= 0 ? position[static_cast<size_t>(i)] : points[static_cast<size_t>(w)];
    };
    for (int step = 0; step < kRelaxationSteps; ++step) {
      std::vector<V> next(ring.size());
      for (size_t i = 0; i < ring.size(); ++i) {
        V sum{0, 0, 0};
        for (int64_t w : adjacency[i]) sum = add(sum, current(w));
        const V average = mul(sum, 1.0 / static_cast<double>(adjacency[i].size()));
        next[i] = project(add(mul(position[i], 1.0 - kRelaxationWeight),
                              mul(average, kRelaxationWeight)));
      }
      position = std::move(next);
    }
    std::vector<int64_t> patch = live_star(u);
    for (int64_t w : ring) {
      const auto around = live_star(w);
      patch.insert(patch.end(), around.begin(), around.end());
    }
    const auto v_star = live_star(v);
    patch.insert(patch.end(), v_star.begin(), v_star.end());
    std::sort(patch.begin(), patch.end());
    patch.erase(std::unique(patch.begin(), patch.end()), patch.end());
    std::vector<F> created;
    for (int64_t id : patch) {
      F t = triangles[static_cast<size_t>(id)];
      const bool has_u = t[0] == u || t[1] == u || t[2] == u;
      const bool has_v = t[0] == v || t[1] == v || t[2] == v;
      if (has_u && has_v) continue;
      for (int64_t& w : t)
        if (w == u) w = v;
      created.push_back(t);
    }
    for (double fraction : {0.5, 0.25, 0.0}) {
      Edit edit;
      edit.removed = patch;
      edit.created = created;
      edit.moved.emplace_back(v, centered(merged, created, v, fraction, current));
      for (size_t i = 0; i < ring.size(); ++i) {
        const V target = centered(position[i], created, ring[i], fraction, current);
        if (!(target == points[static_cast<size_t>(ring[i])]))
          edit.moved.emplace_back(ring[i], target);
      }
      edit.dropped = {u};
      if (!apply_if_admissible(edit, Admission::kPreserve)) continue;
      ++collapses;
      relocations += static_cast<int64_t>(edit.moved.size()) - 1;
      return true;
    }
    return false;
  }

  // Collapses u into v and, while that compound edit is refused, absorbs into
  // v its shortest staged neighbor across an edge below 4/5 h, admitting the
  // compound once per absorption: v at the projected centroid of the merged
  // vertices, centered in the band by 1/2, 1/4 or none of the sagitta. A
  // graded rosette of near-full-score triangles around a point refuses every
  // single collapse and relaxed collapse, because each re-fans the removed
  // vertex's area into triangles far below the rosette's worst score, while
  // merging the rosette's core into one vertex leaves a fan at or above that
  // score. Every absorption satisfies the edge-collapse link condition in the
  // staged state (two triangles on the edge, common neighbors exactly their
  // opposite corners, each of valence above three), so the staged carrier
  // stays a closed oriented two-manifold; the one preserve admission checks
  // the final state. Each step absorbs a new vertex, so an attempt is finite,
  // and an admitted attempt removes at least one vertex.
  bool collapse_absorbing(int64_t u, int64_t v) {
    const double short_edge = 0.8 * h;
    // Live IDs of the triangles touching v or an absorbed vertex (sorted), and
    // their staged corners with every absorbed vertex renamed v; triangles that
    // collapsed onto an edge are no longer staged.
    std::vector<int64_t> patch;
    std::vector<F> staged;
    std::vector<int64_t> absorbed;
    const auto in_patch = [&](int64_t id) {
      return std::binary_search(patch.begin(), patch.end(), id);
    };
    const auto has = [](const F& t, int64_t w) { return t[0] == w || t[1] == w || t[2] == w; };
    const auto add_star = [&](int64_t w) {
      for (int64_t id : live_star(w)) {
        if (in_patch(id)) continue;
        patch.insert(std::upper_bound(patch.begin(), patch.end(), id), id);
        staged.push_back(triangles[static_cast<size_t>(id)]);
      }
    };
    // Neighbors of w in the staged state: triangles outside the patch are
    // unchanged, staged triangles carry the renamed corners.
    const auto staged_neighbors = [&](int64_t w) {
      std::vector<int64_t> result;
      const auto collect = [&](const F& t) {
        if (!has(t, w)) return;
        for (int64_t x : t)
          if (x != w) result.push_back(x);
      };
      for (int64_t id : live_star(w))
        if (!in_patch(id)) collect(triangles[static_cast<size_t>(id)]);
      for (const F& t : staged) collect(t);
      std::sort(result.begin(), result.end());
      result.erase(std::unique(result.begin(), result.end()), result.end());
      return result;
    };
    // Absorbs w into v when the staged edge (w, v) passes the link condition.
    const auto absorb = [&](int64_t w) {
      std::vector<F> on_edge;
      for (int64_t id : live_star(w))
        if (!in_patch(id) && has(triangles[static_cast<size_t>(id)], v))
          on_edge.push_back(triangles[static_cast<size_t>(id)]);
      for (const F& t : staged)
        if (has(t, w) && has(t, v)) on_edge.push_back(t);
      if (on_edge.size() != 2) return false;
      std::vector<int64_t> opposite;
      for (const F& t : on_edge)
        for (int64_t x : t)
          if (x != w && x != v) opposite.push_back(x);
      std::sort(opposite.begin(), opposite.end());
      const auto w_neighbors = staged_neighbors(w), v_neighbors = staged_neighbors(v);
      std::vector<int64_t> common;
      std::set_intersection(w_neighbors.begin(), w_neighbors.end(), v_neighbors.begin(),
                            v_neighbors.end(), std::back_inserter(common));
      if (common != opposite || opposite.size() != 2) return false;
      for (int64_t x : opposite)
        if (staged_neighbors(x).size() <= 3) return false;
      add_star(w);
      std::vector<F> next;
      for (F t : staged) {
        if (has(t, w) && has(t, v)) continue;
        for (int64_t& x : t)
          if (x == w) x = v;
        next.push_back(t);
      }
      staged = std::move(next);
      absorbed.push_back(w);
      return true;
    };
    add_star(v);
    if (!absorb(u)) return false;
    while (true) {
      V sum = points[static_cast<size_t>(v)];
      for (int64_t w : absorbed) sum = add(sum, points[static_cast<size_t>(w)]);
      const V merged = project(mul(sum, 1.0 / static_cast<double>(absorbed.size() + 1)));
      const auto place = [&](int64_t w) {
        return w == v ? merged : points[static_cast<size_t>(w)];
      };
      for (double fraction : {0.5, 0.25, 0.0}) {
        Edit edit;
        edit.removed = patch;
        edit.created = staged;
        edit.moved.emplace_back(v, centered(merged, staged, v, fraction, place));
        edit.dropped = absorbed;
        if (!apply_if_admissible(edit, Admission::kPreserve)) continue;
        collapses += static_cast<int64_t>(absorbed.size());
        return true;
      }
      std::vector<std::pair<double, int64_t>> candidates;
      for (int64_t w : staged_neighbors(v)) {
        const double edge = length(difference(merged, points[static_cast<size_t>(w)]));
        if (edge < short_edge) candidates.emplace_back(edge, w);
      }
      std::sort(candidates.begin(), candidates.end());
      bool grew = false;
      for (const auto& [edge, w] : candidates)
        if (absorb(w)) {
          grew = true;
          break;
        }
      if (!grew) return false;
    }
  }

  // Admission-limited coarsening after the sweeps. A collapse sweep stalls
  // when every remaining short edge (below 4/5 h) is refused, usually by the
  // band, while most triangles still lie well inside it: the vertices sit on
  // the represented surface, every triangle interior on one side of it, and a
  // collapse concentrates the removed vertex's area in one star. Each round
  // centers the vertices around the short edges in the band (preserve
  // relocations by half, else a quarter, of the mean star sagitta), then tries
  // the short edges shortest first as relaxed, centered collapses, else as
  // absorbing collapses (see collapse_absorbing), at most one per 2-ring. A
  // round that collapses nothing ends the phase; every other round removes a
  // vertex, so the phase terminates.
  void coarsen_uniformly() {
    const double short_edge = 0.8 * h;
    while (true) {
      std::vector<std::tuple<double, int64_t, int64_t>> short_edges;
      for (size_t id = 0; id < triangles.size(); ++id) {
        if (!state.alive(id)) continue;
        const F t = triangles[id];
        for (int k = 0; k < 3; ++k) {
          const int64_t a = t[k], b = t[(k + 1) % 3];
          // A closed oriented carrier holds each edge once in each direction.
          if (a > b) continue;
          const double edge = length(difference(points[static_cast<size_t>(a)],
                                                points[static_cast<size_t>(b)]));
          if (edge < short_edge) short_edges.emplace_back(edge, a, b);
        }
      }
      if (short_edges.empty()) return;
      std::sort(short_edges.begin(), short_edges.end());
      std::vector<int64_t> active;
      for (const auto& [edge, a, b] : short_edges)
        for (int64_t end : {a, b}) {
          active.push_back(end);
          for (int64_t w : neighbors(end)) active.push_back(w);
        }
      std::sort(active.begin(), active.end());
      active.erase(std::unique(active.begin(), active.end()), active.end());
      const auto place = [&](int64_t w) { return points[static_cast<size_t>(w)]; };
      for (double fraction : {0.5, 0.25}) {
        bool moved = false;
        for (int64_t vertex : active) {
          std::vector<F> star;
          for (int64_t id : live_star(vertex)) star.push_back(triangles[static_cast<size_t>(id)]);
          if (star.empty()) continue;
          const V target =
              centered(points[static_cast<size_t>(vertex)], star, vertex, fraction, place);
          if (relocate(vertex, target, Admission::kPreserve)) moved = true;
        }
        if (moved) break;
      }
      std::vector<uint8_t> touched(points.size(), 0);
      bool collapsed = false;
      for (const auto& [edge, a, b] : short_edges) {
        if (touched[static_cast<size_t>(a)] || touched[static_cast<size_t>(b)]) continue;
        if (!(length(difference(points[static_cast<size_t>(a)],
                                points[static_cast<size_t>(b)])) < short_edge) ||
            !(collapse_relaxed(a, b) || collapse_absorbing(a, b)))
          continue;
        collapsed = true;
        touched[static_cast<size_t>(a)] = touched[static_cast<size_t>(b)] = 1;
        for (int64_t w : neighbors(b)) {
          touched[static_cast<size_t>(w)] = 1;
          for (int64_t x : neighbors(w)) touched[static_cast<size_t>(x)] = 1;
        }
      }
      if (!collapsed) return;
    }
  }
  // Once most history is dead, discard it before allocating the quality
  // queue. Sorting surviving old IDs gives monotone face and vertex
  // indirections, preserving every comparison order and first-use output.
  // This is called only after the collapse-only size pass, before any split
  // can introduce a vertex; survivors can nevertheless have moved.
  void repack_after_size_pass() {
    const size_t rows = triangles.size();
    size_t live = 0;
    for (size_t id = 0; id < rows; ++id) live += state.alive(id);
    if (live == 0 || live > (rows - live) / 16) return;
    std::vector<std::pair<size_t, F>> survivors;
    std::vector<int64_t> used_vertices;
    survivors.reserve(live);
    if (live <= used_vertices.max_size() / 3) used_vertices.reserve(3 * live);
    for (size_t id = 0; id < rows; ++id) {
      if (!state.alive(id)) continue;
      const F face = triangles[id];
      survivors.emplace_back(id, face);
      used_vertices.insert(used_vertices.end(), face.begin(), face.end());
    }
    std::sort(used_vertices.begin(), used_vertices.end());
    used_vertices.erase(std::unique(used_vertices.begin(), used_vertices.end()), used_vertices.end());
    std::vector<V> new_points;
    std::vector<VertexEvidence> new_evidence;
    new_points.reserve(used_vertices.size());
    new_evidence.reserve(used_vertices.size());
    for (int64_t old_vertex : used_vertices) {
      new_points.push_back(points[static_cast<size_t>(old_vertex)]);
      new_evidence.push_back(evidence[static_cast<size_t>(old_vertex)]);
    }
    std::vector<F> new_faces;
    new_faces.reserve(live);
    FaceState new_state;
    new_state.assign(live);
    FloorStore new_floors;
    new_floors.reserve(live);
    std::unordered_map<int64_t, NormalCone> new_cones;
    new_cones.reserve(std::min(live, cones.size()));
    for (size_t row = 0; row < survivors.size(); ++row) {
      const auto& [old_id, old_face] = survivors[row];
      F face = old_face;
      for (int64_t& vertex : face)
        vertex = static_cast<int64_t>(std::lower_bound(used_vertices.begin(), used_vertices.end(),
                                                       vertex) - used_vertices.begin());
      new_faces.push_back(face);
      new_state.set_score(row, state.score(old_id));
      new_state.set_stuck(row, state.stuck(old_id));
      new_floors.push_back(floors[old_id]);
      const auto cone = cones.find(static_cast<int64_t>(old_id));
      if (cone != cones.end()) new_cones.emplace(static_cast<int64_t>(row), cone->second);
    }
    triangles.reset(std::move(new_faces));
    PointStore new_store;
    new_store.own(std::move(new_points));
    points = std::move(new_store);
    evidence.own(std::move(new_evidence));
    state = std::move(new_state);
    floors = std::move(new_floors);
    cones = std::move(new_cones);
    std::vector<size_t>().swap(base_star_offsets);
    std::vector<uint32_t>().swap(base_star_ids);
    std::vector<uint64_t>().swap(wide_star_ids);
    borrowed_star_offsets = nullptr;
    borrowed_star_rows = nullptr;
    borrowed_star_vertices = borrowed_star_faces = 0;
    edited_star = EditedStars{};
    compact_stars = false;
    star.assign(points.size(), {});
    std::unordered_map<uint64_t, Bucket>().swap(buckets);
    bucket_index_ready = false;
    bucket_rebuilding = false;
    for (size_t id = 0; id < triangles.size(); ++id) insert(static_cast<int64_t>(id));
    prepare_buckets();
  }


  void run(std::vector<V>& input_points, std::vector<F>& input_triangles,
           const int32_t* borrowed = nullptr, size_t borrowed_count = 0,
           const double* borrowed_points = nullptr, size_t point_count = 0,
           const int32_t* borrowed_offsets = nullptr, const int32_t* borrowed_rows = nullptr) {
    if (borrowed_points != nullptr) points.borrow(borrowed_points, point_count);
    else points.own(std::move(input_points));
    if (borrowed != nullptr)
      triangles.borrow(borrowed, borrowed_count);
    else
      triangles.own(std::move(input_triangles));
    if (borrowed != nullptr) evidence.page();
    evidence.reserve(points.size());
    for (size_t v = 0; v < points.size(); ++v) evidence.push_back(predicate.evaluate(points[v]));
    compact_stars = borrowed != nullptr;
    if (compact_stars) {
      edited_star.initialize(points.size());
      borrowed_star_offsets = borrowed_offsets;
      borrowed_star_rows = borrowed_rows;
      borrowed_star_vertices = point_count;
      borrowed_star_faces = borrowed_count;
      if (borrowed_star_offsets == nullptr) {
        base_star_offsets.assign(points.size() + 1, 0);
        for (size_t id = 0; id < triangles.size(); ++id)
          for (int64_t v : triangles[id]) ++base_star_offsets[static_cast<size_t>(v) + 1];
        for (size_t v = 1; v < base_star_offsets.size(); ++v)
          base_star_offsets[v] += base_star_offsets[v - 1];
        if (triangles.size() <= std::numeric_limits<uint32_t>::max())
          base_star_ids.resize(base_star_offsets.back());
        else
          wide_star_ids.resize(base_star_offsets.back());
        for (size_t id = 0; id < triangles.size(); ++id)
          for (int64_t v : triangles[id]) {
            const size_t pos = base_star_offsets[static_cast<size_t>(v)]++;
            if (base_star_ids.empty()) wide_star_ids[pos] = id;
            else base_star_ids[pos] = static_cast<uint32_t>(id);
          }
        for (size_t v = points.size(); v > 0; --v)
          base_star_offsets[v] = base_star_offsets[v - 1];
        base_star_offsets[0] = 0;
      }
    } else {
      star.assign(points.size(), {});
    }
    // Edits keep the orientation and the embedding, so the sign of the
    // enclosed volume, and with it the solid side, is that of the input.
    double volume = 0;
    for (size_t id = 0; id < triangles.size(); ++id) {
      const F t = triangles[id];
      const V& a = points[static_cast<size_t>(t[0])];
      volume += dot(a.data(), cross(difference(points[static_cast<size_t>(t[1])], a),
                                    difference(points[static_cast<size_t>(t[2])], a))
                                  .data());
    }
    solid_side = volume < 0 ? -1 : 1;
    if (borrowed != nullptr) state.page();
    state.assign(triangles.size());
    floors.reserve(triangles.size());
    for (size_t id = 0; id < triangles.size(); ++id) {
      floors.push_back(floor_of(triangles[id]));
      if (compact_stars) {
        const auto range = bucket_range(triangles[id]);
        for (int64_t i = range[0]; i <= range[3]; ++i)
          for (int64_t j = range[1]; j <= range[4]; ++j)
            for (int64_t k = range[2]; k <= range[5]; ++k)
              append_bucket(buckets[cell_key(i, j, k)], static_cast<int64_t>(id));
      } else {
        insert(static_cast<int64_t>(id));
      }
    }
    prepare_buckets();
    for (size_t id = 0; id < triangles.size(); ++id) {
      rescore(static_cast<int64_t>(id));
    }
    // The collapse-only size sweep belongs to admission-limited coarsening.
    if (targets.admission_limited_coarsening) {
      regularize(true);
      repack_after_size_pass();
    }
    queue_enabled = true;
    for (size_t id = 0; id < triangles.size(); ++id)
      if (state.alive(id) && !state.stuck(id) && state.score(id) < kScoreLevels)
        queue.insert(static_cast<int64_t>(id));
    improve_worst_first();
    regularize();
    improve_worst_first();
    if (!targets.admission_limited_coarsening) return;
    coarsen_uniformly();
    improve_worst_first();
  }

  int64_t unmet() const {
    int64_t count = 0;
    for (size_t id = 0; id < triangles.size(); ++id)
      count += state.alive(id) && state.score(id) < kScoreLevels;
    return count;
  }

  // Surviving triangles over the vertices they use, in first-use order.
  void compact(std::vector<V>& output_points, std::vector<F>& output_triangles) const {
    std::vector<int64_t> remap(points.size(), -1);
    output_points.clear();
    output_triangles.clear();
    for (size_t id = 0; id < triangles.size(); ++id) {
      if (!state.alive(id)) continue;
      F t = triangles[id];
      for (int64_t& v : t) {
        if (remap[static_cast<size_t>(v)] < 0) {
          remap[static_cast<size_t>(v)] = static_cast<int64_t>(output_points.size());
          output_points.push_back(points[static_cast<size_t>(v)]);
        }
        v = remap[static_cast<size_t>(v)];
      }
      output_triangles.push_back(t);
    }
  }
};
}  // namespace

double cone_separation(const NormalCone& first, const NormalCone& second) {
  const double first_length = length(first.axis), second_length = length(second.axis);
  if (!(first.half_angle < kPi) || !(second.half_angle < kPi) || !(first_length > 0) ||
      !(second_length > 0) || !std::isfinite(first_length) || !std::isfinite(second_length))
    return kPi;
  const double between = std::atan2(length(cross(first.axis, second.axis)),
                                    dot(first.axis.data(), second.axis.data()));
  return std::min(kPi, between + std::max(first.half_angle, 0.0) +
                           std::max(second.half_angle, 0.0));
}

double forced_turn(const NormalCone& first, const NormalCone& second) {
  const double first_length = length(first.axis), second_length = length(second.axis);
  if (!(first.half_angle < kPi) || !(second.half_angle < kPi) || !(first_length > 0) ||
      !(second_length > 0) || !std::isfinite(first_length) || !std::isfinite(second_length))
    return 0;
  const double between = std::atan2(length(cross(first.axis, second.axis)),
                                    dot(first.axis.data(), second.axis.data()));
  return std::max(0.0, between - std::max(first.half_angle, 0.0) -
                           std::max(second.half_angle, 0.0));
}

namespace {

// An upper bound on sqrt(numerator / denominator) for exact positive dyadics:
// the directed upper quotient, then the correctly rounded square root raised
// until its exact square covers it.
double upper_sqrt_quotient(const Dyadic& numerator, const Dyadic& denominator) {
  const double square = exact::directed_quotient_bounds(numerator, denominator).upper;
  if (!std::isfinite(square)) return std::numeric_limits<double>::infinity();
  const Dyadic bound(square);
  double root = std::sqrt(square);
  while ((Dyadic(root) * Dyadic(root) - bound).sign() < 0) {
    root = next_up(root);
    if (!std::isfinite(root)) return root;
  }
  return root;
}

}  // namespace

// Every point of the triangle lies within the returned radius of a corner,
// plus the caller's absolute `roundoff`. The squared sides and the squared
// cross product are exact dyadics of the binary64 corners, so the obtuse
// classification and the radius are exact before one directed rounding: a
// floating cross product loses relative accuracy of about eps / theta at an
// apex angle theta and cannot bound a needle's circumradius. A zero cross
// product (degenerate triangle) or a nonfinite corner has no free radius.
double certified_free_radius(const Point& a, const Point& b, const Point& c,
                           double roundoff) {
  for (const Point* corner : {&a, &b, &c})
    for (int axis = 0; axis < 3; ++axis)
      if (!std::isfinite((*corner)[axis])) return std::numeric_limits<double>::infinity();
  std::array<Dyadic, 3> ab, ac, bc;
  for (int axis = 0; axis < 3; ++axis) {
    ab[axis] = Dyadic(b[axis]) - Dyadic(a[axis]);
    ac[axis] = Dyadic(c[axis]) - Dyadic(a[axis]);
    bc[axis] = Dyadic(c[axis]) - Dyadic(b[axis]);
  }
  const auto squared = [](const std::array<Dyadic, 3>& v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  };
  const std::array<Dyadic, 3> normal{ab[1] * ac[2] - ab[2] * ac[1],
                                     ab[2] * ac[0] - ab[0] * ac[2],
                                     ab[0] * ac[1] - ab[1] * ac[0]};
  const Dyadic twice_area_squared = squared(normal);
  if (twice_area_squared.sign() <= 0) return std::numeric_limits<double>::infinity();
  const std::array<Dyadic, 3> squares{squared(bc), squared(ac), squared(ab)};  // opposite a, b, c
  int obtuse = -1;
  for (int k = 0; k < 3; ++k)
    if ((squares[k] - squares[(k + 1) % 3] - squares[(k + 2) % 3]).sign() > 0) obtuse = k;
  double radius;
  if (obtuse < 0) {
    // Circumradius: R^2 = |bc|^2 |ca|^2 |ab|^2 / (4 |ab x ac|^2).
    radius = upper_sqrt_quotient(squares[0] * squares[1] * squares[2],
                                 twice_area_squared * Dyadic(4.0));
  } else {
    // Obtuse at corner `obtuse`: the bisectors of its two sides meet the
    // longest side at distance side / (2 cos(adjacent angle)) from that side's
    // ends; with S the squared sides, that distance squared is
    // S_other^2 S_obtuse / (S_obtuse + S_other - S_k)^2 for the angle at k.
    radius = 0;
    for (int k = 0; k < 3; ++k) {
      if (k == obtuse) continue;
      const int other = 3 - k - obtuse;
      const Dyadic twice_cosine_product = squares[obtuse] + squares[other] - squares[k];
      if (twice_cosine_product.sign() <= 0) return std::numeric_limits<double>::infinity();
      radius = std::max(radius, upper_sqrt_quotient(
                                    squares[other] * squares[other] * squares[obtuse],
                                    twice_cosine_product * twice_cosine_product));
    }
  }
  if (!std::isfinite(radius)) return radius;
  return next_up(radius + roundoff);
}

int32_t remesh_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                       const TriangleAdmission& admission, const RemeshTargets& targets,
                       RemeshCounters& counters) {
  if (!(targets.edge_length > 0) || !std::isfinite(targets.edge_length) ||
      !(targets.minimum_angle_degrees > 0) || !(targets.minimum_angle_degrees < 60))
    return PHX_MC_INVALID_ARGUMENT;
  if (triangles.empty()) return PHX_MC_INVALID_ARGUMENT;
  for (const Triangle& t : triangles)
    for (int64_t v : t)
      if (v < 0 || v >= static_cast<int64_t>(points.size())) return PHX_MC_INVALID_ARGUMENT;
  Remesher remesher(admission, targets);
  remesher.run(points, triangles);
  counters.splits = remesher.splits;
  counters.collapses = remesher.collapses;
  counters.flips = remesher.flips;
  counters.relocations = remesher.relocations;
  counters.unmet = remesher.unmet();
  remesher.compact(points, triangles);
  if (triangles.empty()) return PHX_MC_INTERNAL_ERROR;
  return PHX_MC_OK;
}

int32_t remesh_surface_borrowed_i32(std::vector<Point>& points, const double* input_points,
                                    size_t point_count, const int32_t* triangles,
                                    size_t triangle_count, const int32_t* vertex_face_offsets,
                                    const int32_t* vertex_faces, std::vector<Triangle>& output,
                                    const TriangleAdmission& admission,
                                    const RemeshTargets& targets, RemeshCounters& counters) {
  if (!(targets.edge_length > 0) || !std::isfinite(targets.edge_length) ||
      !(targets.minimum_angle_degrees > 0) || !(targets.minimum_angle_degrees < 60) ||
      input_points == nullptr || triangles == nullptr || triangle_count == 0 ||
      ((vertex_face_offsets == nullptr) != (vertex_faces == nullptr)))
    return PHX_MC_INVALID_ARGUMENT;
  for (size_t row = 0; row < triangle_count; ++row)
    for (int k = 0; k < 3; ++k)
      if (triangles[3 * row + k] < 0 ||
          static_cast<size_t>(triangles[3 * row + k]) >= point_count)
        return PHX_MC_INVALID_ARGUMENT;
  {
    Remesher remesher(admission, targets);
    remesher.run(points, output, triangles, triangle_count, input_points, point_count,
                 vertex_face_offsets, vertex_faces);
    counters.splits = remesher.splits;
    counters.collapses = remesher.collapses;
    counters.flips = remesher.flips;
    counters.relocations = remesher.relocations;
    counters.unmet = remesher.unmet();
    remesher.compact(points, output);
  }
  return output.empty() ? PHX_MC_INTERNAL_ERROR : PHX_MC_OK;
}

}  // namespace phx::mc::remesh
