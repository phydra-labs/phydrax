#pragma once
#include "phx_cuda_common.hpp"

namespace phx::cuda {
// Budgets retain cleanup poison after the registry retires its last entry.
class TensorRegistry final : public RegistryBase {
  RegistryBase& registry_;
  void (*check_)(bool, int64_t);
 public:
  TensorRegistry(RegistryBase& registry, void (*check)(bool, int64_t))
      : registry_(registry), check_(check) {}
  int64_t release(int64_t partition, const int32_t* ids, int64_t count) override {
    const auto released = registry_.release(partition, ids, count);
    check_(false, partition);
    return released;
  }
  int64_t release_partition(int64_t partition) override {
    const auto released = registry_.release_partition(partition);
    check_(false, partition);
    return released;
  }
  int64_t release_all() override {
    const auto released = registry_.release_all();
    check_(true, 0);
    return released;
  }
  std::vector<PartitionStats> stats() override { return registry_.stats(); }
};
}  // namespace phx::cuda
