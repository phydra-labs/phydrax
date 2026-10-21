// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// AmgX's blocking legacy-stream C API is an explicit synchronization boundary.
#include "phx_cuda_common.hpp"
#include <amgx_c.h>
#include <cuda_runtime_api.h>
#include <xla/ffi/api/ffi.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ffi = xla::ffi;
namespace {
using I32 = ffi::Buffer<ffi::S32>;
using F64 = ffi::Buffer<ffi::F64>;
using RI32 = ffi::Result<I32>;
using RF64 = ffi::Result<F64>;
std::recursive_mutex& mutex() { static auto* value = new std::recursive_mutex; return *value; }
void cuda_check(cudaError_t rc) {
  if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
void amgx_check(AMGX_RC rc) {
  if (rc == AMGX_RC_OK) return;
  char text[4096] = {};
  AMGX_get_error_string(rc, text, sizeof(text));
  throw std::runtime_error(std::string("AmgX: ") + text);
}
bool initialized = false;
int64_t references = 0;
struct Resources {
  int device;
  AMGX_config_handle config = nullptr;
  AMGX_resources_handle handle = nullptr;
};
// Resources destruction frees process-global pools: retain every device resource
// until ALL hierarchies have been released, then destroy them during shutdown.
std::map<int, Resources>& resources() { static auto* value = new std::map<int, Resources>; return *value; }
AMGX_resources_handle acquire_resources(int device) {
  if (!initialized) { amgx_check(AMGX_initialize()); initialized = true; }
  auto found = resources().find(device);
  if (found != resources().end()) return found->second.handle;
  Resources item{device};
  amgx_check(AMGX_config_create(&item.config, "{\"config_version\":2}"));
  try { amgx_check(AMGX_resources_create(&item.handle, item.config, nullptr, 1, &device)); }
  catch (...) { AMGX_config_destroy(item.config); throw; }
  resources().emplace(device, item);
  return item.handle;
}
struct Entry {
  int device;
  int n = 0, nnz = 0;
  std::string config_text;
  std::vector<int32_t> offsets, columns;
  // Reuse host diagnostics staging across solves; no per-call allocation once
  // this hierarchy has seen the current multi-RHS extent.
  std::vector<int32_t> statuses, iteration_counts;
  std::vector<double> norms;
  AMGX_config_handle config = nullptr;
  AMGX_matrix_handle matrix = nullptr;
  AMGX_solver_handle solver = nullptr;
  AMGX_vector_handle rhs = nullptr, solution = nullptr;
  explicit Entry(int device_) : device(device_) { ++references; }
  ~Entry() {
    std::lock_guard<std::recursive_mutex> lock(mutex());
    int previous = 0; cudaGetDevice(&previous); cudaSetDevice(device);
    if (solution) AMGX_vector_destroy(solution);
    if (rhs) AMGX_vector_destroy(rhs);
    if (solver) AMGX_solver_destroy(solver);
    if (matrix) AMGX_matrix_destroy(matrix);
    if (config) AMGX_config_destroy(config);
    --references;
    cudaSetDevice(previous);
  }
};
phx::cuda::Registry<Entry>& registry() { static auto* value = new phx::cuda::Registry<Entry>; return *value; }
int device_id() { int value; cuda_check(cudaGetDevice(&value)); return value; }
int dimension(I32 offsets, I32 columns, F64 values) {
  const int64_t n = offsets.element_count() - 1;
  if (n < 1 || n > std::numeric_limits<int>::max() || columns.element_count() != values.element_count() ||
      values.element_count() > std::numeric_limits<int>::max()) throw std::invalid_argument("AmgX requires nonempty square int32 CSR storage.");
  return static_cast<int>(n);
}
std::vector<int32_t> copy_indices(I32 buffer) {
  std::vector<int32_t> result(buffer.element_count());
  cuda_check(cudaMemcpy(result.data(), buffer.typed_data(), result.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
  return result;
}
std::shared_ptr<Entry> setup(int device, I32 offsets, I32 columns, F64 values, std::string_view config) {
  auto entry = std::make_shared<Entry>(device);
  entry->n = dimension(offsets, columns, values);
  entry->nnz = static_cast<int>(values.element_count());
  entry->config_text = std::string(config);
  entry->offsets = copy_indices(offsets); entry->columns = copy_indices(columns);
  if (entry->offsets.front() != 0 || entry->offsets.back() != entry->nnz ||
      !std::is_sorted(entry->offsets.begin(), entry->offsets.end()) ||
      std::any_of(entry->columns.begin(), entry->columns.end(), [&](int c) { return c < 0 || c >= entry->n; }))
    throw std::invalid_argument("AmgX received invalid CSR indices.");
  auto resource = acquire_resources(device);
  amgx_check(AMGX_config_create(&entry->config, entry->config_text.c_str()));
  amgx_check(AMGX_matrix_create(&entry->matrix, resource, AMGX_mode_dDDI));
  amgx_check(AMGX_solver_create(&entry->solver, resource, AMGX_mode_dDDI, entry->config));
  amgx_check(AMGX_vector_create(&entry->rhs, resource, AMGX_mode_dDDI));
  amgx_check(AMGX_vector_create(&entry->solution, resource, AMGX_mode_dDDI));
  amgx_check(AMGX_matrix_upload_all(entry->matrix, entry->n, entry->nnz, 1, 1,
      offsets.typed_data(), columns.typed_data(), values.typed_data(), nullptr));
  amgx_check(AMGX_solver_setup(entry->solver, entry->matrix));
  amgx_check(AMGX_vector_bind(entry->rhs, entry->matrix));
  amgx_check(AMGX_vector_bind(entry->solution, entry->matrix));
  return entry;
}
int32_t read_id(I32 id) {
  if (id.element_count() != 1) throw std::invalid_argument("AmgX id must contain one int32.");
  int32_t value; cuda_check(cudaMemcpy(&value, id.typed_data(), sizeof(value), cudaMemcpyDeviceToHost)); return value;
}
void output_int(RI32 output, int32_t value) {
  if (output->element_count() != 1) throw std::invalid_argument("AmgX scalar result must contain one int32.");
  cuda_check(cudaMemcpy(output->typed_data(), &value, sizeof(value), cudaMemcpyHostToDevice));
}
void validate_attrs(int64_t partition, int64_t capacity) {
  if (partition <= 0 || capacity < 1) throw std::invalid_argument("AmgX partition and hierarchy cache capacity must be positive.");
}
template <typename F> ffi::Error invoke(cudaStream_t stream, F operation) {
  std::lock_guard<std::recursive_mutex> lock(mutex());
  try {
    cuda_check(cudaStreamSynchronize(stream));
    operation();
    // The legacy API does not accept the XLA stream. Wait for its default-stream
    // work too, then synchronize XLA before returning ownership of result buffers.
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaStreamSynchronize(stream));
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    phx::cuda::set_last_error(error.what());
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    phx::cuda::set_last_error(error.what());
    return ffi::Error::Internal(error.what());
  }
}
ffi::Error Setup(cudaStream_t stream, I32 offsets, I32 columns, F64 values, RI32 id,
                 int64_t partition, int64_t capacity, std::string_view config) {
  return invoke(stream, [&] {
    validate_attrs(partition, capacity);
    output_int(id, registry().insert(device_id(), partition, capacity, setup(device_id(), offsets, columns, values, config)));
  });
}
ffi::Error Refresh(cudaStream_t stream, I32 previous, I32 offsets, I32 columns, F64 values,
                   RI32 id, RI32 rebuilt, int64_t partition, int64_t capacity, std::string_view config) {
  return invoke(stream, [&] {
    validate_attrs(partition, capacity);
    const int device = device_id(), old_id = read_id(previous);
    auto entry = registry().find(device, partition, old_id);
    const int n = dimension(offsets, columns, values);
    const bool reusable = entry && entry->n == n && entry->nnz == values.element_count() &&
        entry->config_text == config && entry->offsets == copy_indices(offsets) && entry->columns == copy_indices(columns);
    if (reusable) {
      try {
        amgx_check(AMGX_matrix_replace_coefficients(entry->matrix, entry->n, entry->nnz, values.typed_data(), nullptr));
        amgx_check(AMGX_solver_resetup(entry->solver, entry->matrix));
        output_int(id, registry().rekey(device, partition, capacity, old_id, entry));
        output_int(rebuilt, 0);
      } catch (const std::exception& error) {
        // A failed in-place resetup cannot remain resident under the immutable
        // old ID. Drain provider work before destroying its hierarchy or
        // returning ownership of the refresh's coefficient buffers to XLA.
        const auto completion = cudaDeviceSynchronize();
        registry().erase(device, partition, old_id);
        if (completion != cudaSuccess)
          throw std::runtime_error(std::string(error.what()) +
              "; CUDA error completing failed AmgX refresh: " + cudaGetErrorString(completion));
        throw;
      }
    } else {
      entry = setup(device, offsets, columns, values, config);
      output_int(id, registry().insert(device, partition, capacity, entry));
      registry().count_rebuild(device, partition);
      output_int(rebuilt, 1);
    }
  });
}
ffi::Error Solve(cudaStream_t stream, I32 id, I32 offsets, I32 columns, F64 values, F64 rhs, F64 guess,
                 RF64 x, RI32 status, RI32 iterations, RF64 residual, RI32 rebuilt,
                 int64_t partition, int64_t capacity, std::string_view config) {
  return invoke(stream, [&] {
    validate_attrs(partition, capacity);
    const int device = device_id(), n = dimension(offsets, columns, values);
    if (rhs.dimensions().size() != 2 || rhs.dimensions()[1] != n || rhs.element_count() != guess.element_count() ||
        x->element_count() != rhs.element_count()) throw std::invalid_argument("AmgX RHS and initial guess must have shape [nrhs,n].");
    const int64_t count = rhs.dimensions()[0];
    if (status->element_count() != count || iterations->element_count() != count || residual->element_count() != count)
      throw std::invalid_argument("AmgX per-RHS result dimensions do not match.");
    auto entry = registry().find(device, partition, read_id(id));
    bool was_rebuilt = !entry;
    if (entry && (entry->n != n || entry->nnz != values.element_count() || entry->config_text != config))
      throw std::invalid_argument("AmgX resident id does not match its CSR/config operands.");
    if (!entry) {
      entry = setup(device, offsets, columns, values, config);
      registry().insert(device, partition, capacity, entry);
      registry().count_rebuild(device, partition);
    }
    auto& statuses = entry->statuses;
    auto& iteration_counts = entry->iteration_counts;
    auto& norms = entry->norms;
    statuses.resize(count); iteration_counts.resize(count); norms.resize(count);
    for (int64_t index = 0; index < count; ++index) {
      amgx_check(AMGX_vector_upload(entry->rhs, n, 1, rhs.typed_data() + index*n));
      amgx_check(AMGX_vector_upload(entry->solution, n, 1, guess.typed_data() + index*n));
      amgx_check(AMGX_solver_solve(entry->solver, entry->rhs, entry->solution));
      AMGX_SOLVE_STATUS result;
      amgx_check(AMGX_solver_get_status(entry->solver, &result));
      statuses[index] = static_cast<int32_t>(result);
      amgx_check(AMGX_solver_get_iterations_number(entry->solver, &iteration_counts[index]));
      amgx_check(AMGX_solver_calculate_residual_norm(entry->solver, entry->matrix, entry->rhs, entry->solution, &norms[index]));
      amgx_check(AMGX_vector_download(entry->solution, x->typed_data() + index*n));
    }
    cuda_check(cudaMemcpy(status->typed_data(), statuses.data(), count*sizeof(int32_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(iterations->typed_data(), iteration_counts.data(), count*sizeof(int32_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(residual->typed_data(), norms.data(), count*sizeof(double), cudaMemcpyHostToDevice));
    output_int(rebuilt, was_rebuilt ? 1 : 0);
  });
}
ffi::Error Release(cudaStream_t stream, I32 ids, RI32 released, int64_t partition) {
  return invoke(stream, [&] {
    auto host_ids = copy_indices(ids);
    output_int(released, static_cast<int32_t>(registry().release(partition, host_ids.data(), host_ids.size())));
  });
}
} // namespace

using phx::cuda::set_last_error;
phx::cuda::RegistryBase* component_registry() { return &registry(); }
int32_t component_shutdown() {
  std::lock_guard<std::recursive_mutex> lock(mutex());
  if (references != 0) { set_last_error("AmgX shutdown requires releasing all live hierarchies first."); return PHX_CUDA_PROVIDER_ERROR; }
  try {
    int previous = device_id();
    for (auto& pair : resources()) {
      cuda_check(cudaSetDevice(pair.first));
      amgx_check(AMGX_resources_destroy(pair.second.handle));
      amgx_check(AMGX_config_destroy(pair.second.config));
    }
    resources().clear();
    if (initialized) { amgx_check(AMGX_finalize()); initialized = false; }
    cuda_check(cudaSetDevice(previous));
    return PHX_CUDA_SUCCESS;
  } catch (const std::exception& error) { set_last_error(error.what()); return PHX_CUDA_PROVIDER_ERROR; }
}

extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_amgx_setup(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_amgx_refresh(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_amgx_solve(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_amgx_release(XLA_FFI_CallFrame*);

XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_amgx_setup, Setup,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<I32>().Arg<I32>().Arg<F64>()
        .Ret<I32>().Attr<int64_t>("partition").Attr<int64_t>("capacity").Attr<std::string_view>("config"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_amgx_refresh, Refresh,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<I32>().Arg<I32>().Arg<I32>().Arg<F64>()
        .Ret<I32>().Ret<I32>().Attr<int64_t>("partition").Attr<int64_t>("capacity").Attr<std::string_view>("config"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_amgx_solve, Solve,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<I32>().Arg<I32>().Arg<I32>().Arg<F64>().Arg<F64>().Arg<F64>()
        .Ret<F64>().Ret<I32>().Ret<I32>().Ret<F64>().Ret<I32>()
        .Attr<int64_t>("partition").Attr<int64_t>("capacity").Attr<std::string_view>("config"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_amgx_release, Release,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<I32>().Ret<I32>().Attr<int64_t>("partition"));
PHX_CUDA_HANDLER_TABLE({"phydrax_amgx_setup", phydrax_amgx_setup},
                       {"phydrax_amgx_refresh", phydrax_amgx_refresh},
                       {"phydrax_amgx_solve", phydrax_amgx_solve},
                       {"phydrax_amgx_release", phydrax_amgx_release})
