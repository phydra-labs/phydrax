#include "phx_cuda_common.hpp"
#include <exception>
#include <utility>
namespace {
thread_local std::string last_error;
template<class Function> int32_t guard(Function&& function) noexcept {
  try { function(); last_error.clear(); return PHX_CUDA_SUCCESS; }
  catch (const std::invalid_argument& error) { phx::cuda::set_last_error(error.what()); return PHX_CUDA_INVALID_ARGUMENT; }
  catch (const std::exception& error) { phx::cuda::set_last_error(error.what()); return PHX_CUDA_PROVIDER_ERROR; }
  catch (...) { phx::cuda::set_last_error("Unknown native CUDA failure"); return PHX_CUDA_PROVIDER_ERROR; }
}
}
namespace phx::cuda {
void set_last_error(const std::string& message) { last_error = message; }
xla::ffi::Error ffi_error(const std::string& message) {
  set_last_error(message);
  return xla::ffi::Error::Internal(message);
}
xla::ffi::Error cuda_error(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return xla::ffi::Error::Success();
  return ffi_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
int32_t current_device() {
  int device = -1;
  const cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) throw std::runtime_error(std::string("cudaGetDevice: ") + cudaGetErrorString(status));
  return device;
}
}
extern "C" {
const char* phx_cuda_last_error() { return last_error.c_str(); }
int32_t phx_cuda_registry_partitions() {
  int32_t count = -1;
  const auto status = guard([&] {
    auto* registry = component_registry();
    count = registry ? static_cast<int32_t>(registry->stats().size()) : 0;
  });
  return status == PHX_CUDA_SUCCESS ? count : -1;
}
int32_t phx_cuda_registry_partition(int32_t index, int32_t* device, int64_t* partition,
                                   int64_t* live, int64_t* capacity, int64_t* rebuilds) {
  return guard([&] {
    if (!device || !partition || !live || !capacity || !rebuilds || index < 0)
      throw std::invalid_argument("Invalid CUDA registry stats output or index");
    auto* registry = component_registry();
    if (!registry) throw std::invalid_argument("Component has no registry");
    const auto stats = registry->stats();
    if (static_cast<size_t>(index) >= stats.size()) throw std::invalid_argument("CUDA registry partition index out of range");
    const auto& item = stats[index];
    *device = item.device; *partition = item.partition; *live = item.live;
    *capacity = item.capacity; *rebuilds = item.rebuilds;
  });
}
int32_t phx_cuda_release(int64_t partition, const int32_t* ids, int64_t count, int64_t* released) {
  return guard([&] {
    if (!released || count < 0 || (count > 0 && !ids)) throw std::invalid_argument("Invalid CUDA release arguments");
    auto* registry = component_registry();
    *released = registry ? registry->release(partition, ids, count) : 0;
  });
}
int32_t phx_cuda_release_all(int64_t* released) {
  return guard([&] {
    if (!released) throw std::invalid_argument("CUDA release_all requires an output pointer");
    auto* registry = component_registry();
    *released = registry ? registry->release_all() : 0;
  });
}
int32_t phx_cuda_shutdown() {
  try { return component_shutdown(); }
  catch (const std::exception& error) { phx::cuda::set_last_error(error.what()); return PHX_CUDA_PROVIDER_ERROR; }
  catch (...) { phx::cuda::set_last_error("Unknown CUDA shutdown failure"); return PHX_CUDA_PROVIDER_ERROR; }
}
}
