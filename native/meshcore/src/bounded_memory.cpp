//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "bounded_memory.hpp"
#include "tet_mesh.hpp"

namespace phx::mc {
thread_local MemoryOwner active_memory_owner;
thread_local NativeExecutionScope* active_execution_scope = nullptr;

class NativeHostStorage final : public NativeAllocatedObject {
 public:
  NativeHostStorage() = default;
  ~NativeHostStorage() { native_object_owner()->release_external_storage(bound_); }
  NativeHostStorage(const NativeHostStorage&) = delete;
  NativeHostStorage& operator=(const NativeHostStorage&) = delete;
  bool resize(std::size_t requested) noexcept {
    if (!native_object_owner()->resize_external_storage(bound_, requested)) return false;
    bound_ = requested;
    return true;
  }
  const MemoryOwner& owner() const noexcept { return native_object_owner(); }
  std::size_t bound() const noexcept { return bound_; }
 private:
  std::size_t bound_ = 0;
};
}  // namespace phx::mc

extern "C" int32_t phx_mc_execution_begin(
    uint64_t work, uint64_t queries, uint64_t cavity, uint64_t scratch,
    double seconds, phx_mc_tet_mesh* memory_mesh, void** out) {
  if (out == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *out = nullptr;
  using Clock = std::chrono::steady_clock;
  const double available =
      std::chrono::duration<double>(Clock::time_point::max() - Clock::now()).count();
  if (scratch > std::numeric_limits<std::size_t>::max() || std::isnan(seconds) ||
      seconds < 0.0 || (std::isfinite(seconds) && seconds >= available)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (memory_mesh != nullptr && memory_mesh->mesh == nullptr) return PHX_MC_INVALID_ARGUMENT;
  if (memory_mesh != nullptr && phx::mc::active_execution_scope != nullptr &&
      memory_mesh->mesh->memory_owner() != phx::mc::scratch_memory_owner()) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  try {
    *out = new phx::mc::NativeExecutionScope(
        work, queries, cavity, static_cast<std::size_t>(scratch), seconds,
        memory_mesh == nullptr ? phx::mc::scratch_memory_owner()
                               : memory_mesh->mesh->memory_owner());
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_CAPACITY_EXCEEDED;
  } catch (...) {
    return PHX_MC_INTERNAL_ERROR;
  }
}

extern "C" int32_t phx_mc_execution_end(
    void* opaque, uint64_t* counters, uint64_t* memory, double* seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope ||
      counters == nullptr || memory == nullptr || seconds == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  scope->charge(0, 0);
  const auto work = scope->evidence();
  const auto bytes = scope->memory_evidence();
  std::copy(work.begin(), work.end(), counters);
  std::copy(bytes.begin(), bytes.end(), memory);
  *seconds = scope->seconds();
  const int32_t status = scope->status() == PHX_MC_OK && bytes[5] != 0
                             ? PHX_MC_CAPACITY_EXCEEDED : scope->status();
  delete scope;
  return status;
}

extern "C" int32_t phx_mc_execution_charge(void* opaque, uint64_t work, uint64_t queries) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope) return PHX_MC_INVALID_ARGUMENT;
  return scope->charge_external(work, queries) ? PHX_MC_OK : scope->status();
}

extern "C" int32_t phx_mc_execution_import_preparation(
    void* opaque, uint64_t work, uint64_t queries, double seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  const double duration_max = std::chrono::duration<double>(
      std::chrono::steady_clock::duration::max()).count();
  if (scope == nullptr || scope != phx::mc::active_execution_scope ||
      !std::isfinite(seconds) || seconds < 0.0 || seconds >= duration_max) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return scope->import_preparation(work, queries, seconds) ? PHX_MC_OK : (
      scope->status() == PHX_MC_OK ? PHX_MC_INVALID_ARGUMENT : scope->status());
}

extern "C" int32_t phx_mc_execution_prior_seconds(void* opaque, double* seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope || seconds == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *seconds = scope->prior_seconds();
  return PHX_MC_OK;
}

extern "C" int32_t phx_mc_execution_admit_work_bound(void* opaque, uint64_t maximum_work) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope) return PHX_MC_INVALID_ARGUMENT;
  return scope->admit(maximum_work) ? PHX_MC_OK : scope->status();
}

extern "C" int32_t phx_mc_execution_admit_cavity(void* opaque, uint64_t cells) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope ||
      cells > std::numeric_limits<std::size_t>::max()) return PHX_MC_INVALID_ARGUMENT;
  return scope->admit_cavity(static_cast<std::size_t>(cells)) ? PHX_MC_OK : scope->status();
}

extern "C" int32_t phx_mc_execution_remaining(
    void* opaque, uint64_t* remaining, double* wall_seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope ||
      remaining == nullptr || wall_seconds == nullptr) return PHX_MC_INVALID_ARGUMENT;
  scope->charge(0, 0);
  const auto allowance = scope->remaining();
  std::copy(allowance.begin(), allowance.end(), remaining);
  *wall_seconds = scope->remaining_wall_seconds();
  return scope->status();
}

extern "C" int32_t phx_mc_execution_allocate_host_array(
    void* opaque, uint64_t bytes, uint64_t alignment, void** out_owner, void** out_data) {
  if (out_owner != nullptr) *out_owner = nullptr;
  if (out_data != nullptr) *out_data = nullptr;
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope ||
      out_owner == nullptr || out_data == nullptr || out_owner == out_data ||
      bytes > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      alignment == 0 || alignment > std::numeric_limits<std::size_t>::max() ||
      (alignment & (alignment - 1)) != 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!scope->charge(0)) return scope->status();
  try {
    auto owner = phx::mc::make_native_unique<phx::mc::NativeHostBuffer>(
        static_cast<std::size_t>(bytes), static_cast<std::size_t>(alignment));
    *out_data = owner->data();
    *out_owner = owner.release();
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return phx::mc::native_execution_status(PHX_MC_CAPACITY_EXCEEDED);
  } catch (...) {
    return PHX_MC_INTERNAL_ERROR;
  }
}

extern "C" void phx_mc_execution_free_host_array(void* owner) {
  phx::mc::destroy_native_object(static_cast<phx::mc::NativeHostBuffer*>(owner));
}

extern "C" int32_t phx_mc_execution_reserve_host_storage(
    void* opaque, uint64_t bytes_upper, void** reservation) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope ||
      reservation == nullptr || bytes_upper > std::numeric_limits<std::size_t>::max()) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  auto* existing = static_cast<phx::mc::NativeHostStorage*>(*reservation);
  if (existing != nullptr && existing->owner() != phx::mc::scratch_memory_owner()) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (existing != nullptr && bytes_upper <= existing->bound()) {
    existing->resize(static_cast<std::size_t>(bytes_upper));
    return PHX_MC_OK;
  }
  if (!scope->charge(0)) return scope->status();
  try {
    if (existing != nullptr) {
      if (existing->resize(static_cast<std::size_t>(bytes_upper))) return PHX_MC_OK;
    } else {
      auto owner = phx::mc::make_native_unique<phx::mc::NativeHostStorage>();
      if (owner->resize(static_cast<std::size_t>(bytes_upper))) {
        *reservation = owner.release();
        return PHX_MC_OK;
      }
    }
    scope->refuse_external_storage();
    return PHX_MC_CAPACITY_EXCEEDED;
  } catch (const std::bad_alloc&) {
    scope->refuse_external_storage();
    return PHX_MC_CAPACITY_EXCEEDED;
  } catch (...) {
    return PHX_MC_INTERNAL_ERROR;
  }
}

extern "C" void phx_mc_execution_release_host_storage(void* reservation) {
  phx::mc::destroy_native_object(static_cast<phx::mc::NativeHostStorage*>(reservation));
}
