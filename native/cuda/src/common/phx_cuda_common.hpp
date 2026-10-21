#pragma once
#include "phydrax_cuda.h"
#include <cuda_runtime_api.h>
#include <xla/ffi/api/ffi.h>
#include <atomic>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace phx::cuda {
using Status = phx_cuda_status;
void set_last_error(const std::string& message);
xla::ffi::Error ffi_error(const std::string& message);
xla::ffi::Error cuda_error(cudaError_t status, const char* operation);
int32_t current_device();
struct PartitionStats {
  int32_t device;
  int64_t partition;
  int64_t live;
  int64_t capacity;
  int64_t rebuilds;
};
class RegistryBase {
 public:
  virtual ~RegistryBase() = default;
  virtual int64_t release(int64_t partition, const int32_t* ids, int64_t count) = 0;
  virtual int64_t release_all() = 0;
  virtual std::vector<PartitionStats> stats() = 0;
};
// Allocate each component's registry with `new` and intentionally retain it.
// Explicit release/shutdown precedes CUDA/JAX teardown; static destructors do not.
template <class Entry> class Registry final : public RegistryBase {
  struct Slot { std::shared_ptr<Entry> entry; typename std::list<int32_t>::iterator lru; };
  struct Partition {
    int64_t capacity = 1;
    int64_t rebuilds = 0;
    std::list<int32_t> lru;
    std::unordered_map<int32_t, Slot> entries;
  };
  using Key = std::pair<int32_t, int64_t>;
  std::mutex mutex_;
  std::map<Key, Partition> partitions_;
  std::unordered_set<int32_t> live_ids_;
  std::atomic<uint32_t> next_id_{1};

  int32_t mint_locked() {
    for (uint64_t attempt = 0; attempt < static_cast<uint64_t>(std::numeric_limits<int32_t>::max()); ++attempt) {
      const uint32_t raw = next_id_.fetch_add(1, std::memory_order_relaxed);
      const int32_t id = static_cast<int32_t>(raw & 0x7fffffffU);
      if (id != 0 && live_ids_.insert(id).second) return id;
    }
    throw std::runtime_error("CUDA registry id space exhausted");
  }
  void erase_locked(Partition& partition, int32_t id, std::vector<std::shared_ptr<Entry>>& retired) {
    auto found = partition.entries.find(id);
    if (found == partition.entries.end()) return;
    retired.push_back(std::move(found->second.entry));
    partition.lru.erase(found->second.lru);
    partition.entries.erase(found);
    live_ids_.erase(id);
  }
  int32_t store_locked(int32_t device, int64_t key, int64_t capacity, int32_t old_id,
                       std::shared_ptr<Entry> entry, std::vector<std::shared_ptr<Entry>>& retired) {
    if (capacity < 1 || !entry) throw std::invalid_argument("CUDA registry requires capacity >= 1 and a non-null entry");
    auto& partition = partitions_[{device, key}];
    partition.capacity = capacity;
    const int32_t fresh = mint_locked();
    if (old_id != 0) erase_locked(partition, old_id, retired);
    partition.lru.push_front(fresh);
    partition.entries.emplace(fresh, Slot{std::move(entry), partition.lru.begin()});
    while (partition.entries.size() > static_cast<uint64_t>(capacity))
      erase_locked(partition, partition.lru.back(), retired);
    return fresh;
  }
 public:
  std::shared_ptr<Entry> find(int32_t device, int64_t partition, int32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto part = partitions_.find({device, partition});
    if (part == partitions_.end()) return nullptr;
    auto found = part->second.entries.find(id);
    if (found == part->second.entries.end()) return nullptr;
    part->second.lru.splice(part->second.lru.begin(), part->second.lru, found->second.lru);
    return found->second.entry;
  }
  int32_t find_device(int64_t partition, int32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : partitions_)
      if (item.first.second == partition && item.second.entries.count(id)) return item.first.first;
    return -1;
  }
  int32_t insert(int32_t device, int64_t partition, int64_t capacity, std::shared_ptr<Entry> entry) {
    std::vector<std::shared_ptr<Entry>> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    return store_locked(device, partition, capacity, 0, std::move(entry), retired);
  }
  int32_t rekey(int32_t device, int64_t partition, int64_t capacity, int32_t old_id, std::shared_ptr<Entry> entry) {
    std::vector<std::shared_ptr<Entry>> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    return store_locked(device, partition, capacity, old_id, std::move(entry), retired);
  }
  // Failed mutation retires only its memo, retaining partition work evidence.
  void erase(int32_t device, int64_t partition, int32_t id) {
    std::shared_ptr<Entry> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    auto part = partitions_.find({device, partition});
    if (part == partitions_.end()) return;
    auto found = part->second.entries.find(id);
    if (found == part->second.entries.end()) return;
    retired = std::move(found->second.entry);
    part->second.lru.erase(found->second.lru);
    part->second.entries.erase(found);
    live_ids_.erase(id);
  }
  void count_rebuild(int32_t device, int64_t partition) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++partitions_[{device, partition}].rebuilds;
  }
  int64_t release(int64_t partition, const int32_t* ids, int64_t count) override {
    if (count < 0 || (count > 0 && ids == nullptr)) throw std::invalid_argument("Invalid CUDA registry release ids");
    std::vector<std::shared_ptr<Entry>> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto item = partitions_.begin(); item != partitions_.end();) {
      if (item->first.second == partition) {
        for (int64_t i = 0; i < count; ++i) erase_locked(item->second, ids[i], retired);
        if (item->second.entries.empty()) { item = partitions_.erase(item); continue; }
      }
      ++item;
    }
    return static_cast<int64_t>(retired.size());
  }
  int64_t release_all() override {
    std::map<Key, Partition> retired;
    int64_t count;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      count = static_cast<int64_t>(live_ids_.size());
      retired.swap(partitions_);
      live_ids_.clear();
    }
    return count;
  }
  std::vector<PartitionStats> stats() override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PartitionStats> result;
    result.reserve(partitions_.size());
    for (const auto& item : partitions_)
      result.push_back({item.first.first, item.first.second,
        static_cast<int64_t>(item.second.entries.size()), item.second.capacity, item.second.rebuilds});
    return result;
  }
};
struct Handler { const char* name; XLA_FFI_Handler* handler; };
}  // namespace phx::cuda

phx::cuda::RegistryBase* component_registry();
int32_t component_shutdown();

#define PHX_CUDA_HANDLER_TABLE(...) \
  namespace { const ::phx::cuda::Handler phx_handlers[] = {__VA_ARGS__}; } \
  extern "C" PHX_CUDA_EXPORT int32_t phx_cuda_handler_count(void) { \
    return static_cast<int32_t>(sizeof(phx_handlers) / sizeof(phx_handlers[0])); } \
  extern "C" PHX_CUDA_EXPORT const char* phx_cuda_handler_name(int32_t index) { \
    return index >= 0 && index < phx_cuda_handler_count() ? phx_handlers[index].name : nullptr; } \
  extern "C" PHX_CUDA_EXPORT void* phx_cuda_handler(int32_t index) { \
    return index >= 0 && index < phx_cuda_handler_count() ? reinterpret_cast<void*>(phx_handlers[index].handler) : nullptr; }
