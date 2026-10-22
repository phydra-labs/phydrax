/*
 * Copyright (c) 2025 John Viljoen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* Phydrax adaptations of Spineax cudss/solver.cpp: public XLA FFI symbols,
 * real symmetric systems, partitioned bounded memo registry, explicit rebuild
 * evidence, pivot-diagonal query, and an owned C ABI lifecycle; no nanobind.
 * The source distribution has no inline notice; its MIT notice is retained above.
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "cuda_runtime_api.h"
#include "cudss.h"
#include "xla/ffi/api/ffi.h"
#include "phx_cuda_common.hpp"

namespace ffi = xla::ffi;
namespace {

#define CUDA_CHECK(call) do { \
  cudaError_t status_ = (call); \
  if (status_ != cudaSuccess) return ffi::Error::Internal( \
      std::string("cuDSS CUDA failure: ") + cudaGetErrorString(status_)); \
} while (0)
#define CUDSS_CHECK(call) do { \
  cudssStatus_t status_ = (call); \
  if (status_ != CUDSS_STATUS_SUCCESS) return ffi::Error::Internal( \
      std::string("cuDSS failure in ") + #call + " (status " + \
      std::to_string(static_cast<int>(status_)) + ")"); \
} while (0)

template <ffi::DataType T> constexpr cudssDataType_t cudss_dtype();
template <> constexpr cudssDataType_t cudss_dtype<ffi::F32>() { return CUDSS_R_32F; }
template <> constexpr cudssDataType_t cudss_dtype<ffi::F64>() { return CUDSS_R_64F; }

struct Options {
  int64_t partition, capacity, mtype, mview, reordering, memory;
  int64_t hybrid_device_limit;
  int64_t group = 0;
  std::array<int, 8> devices{};
  int device_count = 0;
  Options() = default;
  Options(int64_t p, int64_t c, int64_t t, int64_t v, int64_t r, int64_t m,
          int64_t limit, int64_t g, ffi::Span<const int64_t> indices)
      : partition(p), capacity(c), mtype(t), mview(v), reordering(r), memory(m),
        hybrid_device_limit(limit), group(g),
        device_count(static_cast<int>(indices.size())) {
    if (device_count <= 8)
      for (int i = 0; i < device_count; ++i) {
        devices[i] = indices[i] >= 0 && indices[i] <= INT32_MAX
                         ? static_cast<int>(indices[i]) : -1;
      }
  }
  ffi::Error validate() const {
    if (capacity < 1) return ffi::Error::InvalidArgument("cuDSS capacity must be positive");
    if (mtype != CUDSS_MTYPE_SYMMETRIC && mtype != CUDSS_MTYPE_SPD)
      return ffi::Error::InvalidArgument("cuDSS supports real symmetric or SPD matrices only");
    if (mview < CUDSS_MVIEW_FULL || mview > CUDSS_MVIEW_LOWER)
      return ffi::Error::InvalidArgument("cuDSS invalid matrix view");
    if (reordering < CUDSS_REORDERING_ALG_DEFAULT || reordering > CUDSS_REORDERING_ALG_NONE)
      return ffi::Error::InvalidArgument("cuDSS invalid reordering algorithm");
    if (memory != 0 && memory != 1)
      return ffi::Error::InvalidArgument("cuDSS memory must be 0 (device) or 1 (hybrid)");
    if (hybrid_device_limit < 0 || (memory == 0 && hybrid_device_limit != 0))
      return ffi::Error::InvalidArgument(
          "cuDSS hybrid device limit requires hybrid memory and nonnegative bytes");
    if (device_count != 0 && device_count != 2 && device_count != 4 && device_count != 8)
      return ffi::Error::InvalidArgument("cuDSS MG requires an ordered 2, 4, or 8 device reservation");
    if (device_count != 0 && (memory != 0 || group == 0))
      return ffi::Error::InvalidArgument("cuDSS MG requires device memory and an explicit group");
    for (int i = 0; i < device_count; ++i) {
      if (devices[i] < 0)
        return ffi::Error::InvalidArgument("cuDSS MG device indices must be nonnegative int32");
      for (int j = 0; j < i; ++j)
        if (devices[i] == devices[j])
          return ffi::Error::InvalidArgument("cuDSS MG device reservation contains duplicates");
    }
    return ffi::Error::Success();
  }
};

struct DeviceReservation {
  int64_t group;
  std::array<int, 8> devices;
  int count;
};

// One process owns each live vendor device set. Equal reservations may retain
// several factors; intersecting, unequal reservations cannot coexist.
static ffi::Error reserve_devices(const Options& options, int device,
                                  std::shared_ptr<DeviceReservation>* result) {
  static std::mutex mutex;
  static std::vector<std::weak_ptr<DeviceReservation>> reservations;
  const int count = options.device_count == 0 ? 1 : options.device_count;
  auto devices = options.devices;
  if (options.device_count == 0) devices[0] = device;
  if (options.device_count != 0 && device != devices[0])
    return ffi::Error::InvalidArgument("cuDSS MG must execute once on its initiating device");
  std::lock_guard<std::mutex> lock(mutex);
  for (auto it = reservations.begin(); it != reservations.end();) {
    auto live = it->lock();
    if (!live) { it = reservations.erase(it); continue; }
    bool intersects = false;
    for (int i = 0; i < count; ++i)
      for (int j = 0; j < live->count; ++j)
        intersects |= devices[i] == live->devices[j];
    if (intersects) {
      if (live->group != options.group || live->count != count ||
          !std::equal(devices.begin(), devices.begin() + count, live->devices.begin()))
        return ffi::Error::InvalidArgument("cuDSS device reservation overlaps a different live group");
      *result = std::move(live);
      return ffi::Error::Success();
    }
    ++it;
  }
  *result = std::make_shared<DeviceReservation>(DeviceReservation{options.group, devices, count});
  reservations.push_back(*result);
  return ffi::Error::Success();
}

static std::recursive_mutex& mg_phase_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

static ffi::Error require_mg_pointer(const Options& options, const void* pointer,
                                     int64_t count) {
  if (options.device_count == 0 || count == 0) return ffi::Error::Success();
  cudaPointerAttributes attributes{};
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != options.devices[0])
    return ffi::Error::InvalidArgument("cuDSS MG requires full initiating-device CSR/RHS/output buffers");
  return ffi::Error::Success();
}

struct FactorEntry {
  std::mutex operation_mu;
  std::shared_ptr<DeviceReservation> reservation;
  std::array<cudaStream_t, 8> mg_streams{};
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t A = nullptr, x_dummy = nullptr, b_dummy = nullptr;
  cudaEvent_t done = nullptr;
  uint32_t fp[2] = {0, 0};
  std::array<uint32_t, 8 * 6 * 2> analysis_memory_words{};
  int64_t n = 0, nnz = 0, batch = 1;
  int device = -1;
  cudssDataType_t dtype = CUDSS_R_64F;
  Options options{};
  bool factorized = false;

  ~FactorEntry() {
    std::unique_lock<std::recursive_mutex> group_lock(mg_phase_mutex(), std::defer_lock);
    if (options.device_count != 0) group_lock.lock();
    int previous = -1;
    cudaGetDevice(&previous);
    for (int i = 0; i < options.device_count; ++i)
      if (mg_streams[i] && cudaSetDevice(options.devices[i]) == cudaSuccess) {
        cudaStreamSynchronize(mg_streams[i]);
      }
    if (device >= 0) cudaSetDevice(device);
    // Completion must precede every cuDSS teardown, including LRU eviction.
    if (done) { cudaEventSynchronize(done); cudaEventDestroy(done); }
    if (A) cudssMatrixDestroy(A);
    if (b_dummy) cudssMatrixDestroy(b_dummy);
    if (x_dummy) cudssMatrixDestroy(x_dummy);
    if (data && handle) cudssDataDestroy(handle, data);
    if (config) cudssConfigDestroy(config);
    if (handle) cudssDestroy(handle);
    for (int i = 0; i < options.device_count; ++i)
      if (mg_streams[i]) {
        auto status = cudaSetDevice(options.devices[i]);
        // Vendor destruction can enqueue work outside the configured stream.
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        if (status == cudaSuccess) status = cudaStreamDestroy(mg_streams[i]);
        if (status != cudaSuccess)
          phx::cuda::set_cleanup_error("cuDSS MG teardown", cudaGetErrorString(status));
      }
    if (previous >= 0) {
      const auto status = cudaSetDevice(previous);
      if (status != cudaSuccess)
        phx::cuda::set_cleanup_error("cuDSS device restore", cudaGetErrorString(status));
    }
  }
};

phx::cuda::Registry<FactorEntry>& registry() {
  static auto* instance = new phx::cuda::Registry<FactorEntry>();
  return *instance;
}

struct PhaseLease {
  std::shared_ptr<FactorEntry> entry;
  std::unique_lock<std::recursive_mutex> group_lock;
  std::unique_lock<std::mutex> lock;
  int32_t id = 0;
  int device = -1;
};

static ffi::Error write_scalar(cudaStream_t stream,
                               ffi::ResultBuffer<ffi::S32>& output, int32_t value) {
  if (output->element_count() != 1)
    return ffi::Error::InvalidArgument("cuDSS scalar result must have one element");
  CUDA_CHECK(cudaMemcpyAsync(output->typed_data(), &value, sizeof(value),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return ffi::Error::Success();
}

template <ffi::DataType T>
static ffi::Error begin_phase(cudaStream_t stream, ffi::Buffer<ffi::S32>& token,
                              const Options& options, PhaseLease* lease) {
  if (auto err = options.validate(); err.failure()) return err;
  CUDA_CHECK(cudaGetDevice(&lease->device));
  if (options.device_count != 0) {
    if (lease->device != options.devices[0])
      return ffi::Error::InvalidArgument("cuDSS MG must execute once on its initiating device");
    lease->group_lock = std::unique_lock<std::recursive_mutex>(mg_phase_mutex());
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      return ffi::Error::InvalidArgument("cuDSS MG does not support CUDA graph capture");
  }
  const int64_t count = token.element_count();
  if (count < 1) return ffi::Error::InvalidArgument("cuDSS token must not be empty");
  int32_t single_id = 0;
  std::vector<int32_t> batch_ids(count > 1 ? count : 0);
  int32_t* ids = count == 1 ? &single_id : batch_ids.data();
  CUDA_CHECK(cudaMemcpyAsync(ids, token.typed_data(), count * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int64_t i = 1; i < count; ++i)
    if (ids[i] != ids[0]) return ffi::Error::InvalidArgument(
        "cuDSS batched token ids differ; analyze one block-diagonal batch instead");
  lease->id = ids[0];
  auto& r = registry();
  const int owner_device = r.find_device(options.partition, lease->id);
  if (owner_device >= 0 && owner_device != lease->device)
    return ffi::Error::InvalidArgument("cuDSS token device differs from the FFI stream device");
  while (true) {
    auto e = r.find(lease->device, options.partition, lease->id);
    if (!e) return ffi::Error::Success();
    std::unique_lock<std::mutex> lock(e->operation_mu);
    // A waiting operation may have consumed this immutable state's name.
    if (r.find(lease->device, options.partition, lease->id) != e) continue;
    if (e->device != lease->device)
      return ffi::Error::InvalidArgument("cuDSS token device differs from the FFI stream device");
    if (e->dtype != cudss_dtype<T>())
      return ffi::Error::InvalidArgument("cuDSS token dtype mismatch");
    if (count != 1 && e->batch != 1 && count != e->batch)
      return ffi::Error::InvalidArgument("cuDSS token batch size mismatch");
    const auto& o = e->options;
    if (o.mtype != options.mtype || o.mview != options.mview ||
        o.reordering != options.reordering || o.memory != options.memory ||
        o.hybrid_device_limit != options.hybrid_device_limit ||
        o.group != options.group || o.device_count != options.device_count ||
        o.devices != options.devices)
      return ffi::Error::InvalidArgument("cuDSS token configuration mismatch");
    CUDA_CHECK(cudaStreamWaitEvent(stream, e->done, 0));
    if (options.device_count == 0) CUDSS_CHECK(cudssSetStream(e->handle, stream));
    lease->entry = std::move(e);
    lease->lock = std::move(lock);
    return ffi::Error::Success();
  }
}

template <ffi::DataType T>
static ffi::Error validate_structure(const FactorEntry* e, cudaStream_t stream,
                                     ffi::Buffer<ffi::S32>& offsets,
                                     ffi::Buffer<ffi::S32>& columns,
                                     ffi::Buffer<ffi::U32>& fingerprint, ffi::Buffer<T>& values) {
  if (offsets.element_count() != e->n + 1 || columns.element_count() != e->nnz ||
      values.element_count() != e->nnz || fingerprint.element_count() != 2)
    return ffi::Error::InvalidArgument("cuDSS token CSR structure size mismatch");
  for (const auto& buffer : {std::pair<const void*, int64_t>(values.typed_data(), values.element_count()),
                             {offsets.typed_data(), offsets.element_count()},
                             {columns.typed_data(), columns.element_count()}})
    if (auto err = require_mg_pointer(e->options, buffer.first, buffer.second); err.failure()) return err;
  uint32_t fp[2];
  CUDA_CHECK(cudaMemcpyAsync(fp, fingerprint.typed_data(), sizeof(fp),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (fp[0] != e->fp[0] || fp[1] != e->fp[1])
    return ffi::Error::InvalidArgument("cuDSS CSR structure fingerprint mismatch; analyze the new pattern");
  return ffi::Error::Success();
}

template <ffi::DataType T>
static ffi::Error repoint(FactorEntry* e, cudaStream_t stream,
                          ffi::Buffer<ffi::S32>& offsets,
                          ffi::Buffer<ffi::S32>& columns,
                          ffi::Buffer<ffi::U32>& fingerprint, ffi::Buffer<T>& values) {
  if (auto err = validate_structure(e, stream, offsets, columns, fingerprint, values); err.failure())
    return err;
  CUDSS_CHECK(cudssMatrixSetCsrPointers(e->A, offsets.typed_data(), nullptr,
                                       columns.typed_data(), values.typed_data()));
  return ffi::Error::Success();
}

static ffi::Error execute(FactorEntry* e, cudaStream_t stream, cudssPhase_t phase,
                           cudssMatrix_t x = nullptr, cudssMatrix_t b = nullptr) {
  // Only the initiating XLA stream is available. Owned peer streams never
  // borrow JAX streams; establish readiness before synchronous MG execution.
  if (e->options.device_count != 0) CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto status = cudssExecute(e->handle, phase, e->config, e->data,
                                   e->A, x ? x : e->x_dummy, b ? b : e->b_dummy);
  if (e->options.device_count != 0) {
    cudaError_t completion = cudaSuccess;
    for (int i = 0; i < e->options.device_count; ++i) {
      auto error = cudaSetDevice(e->options.devices[i]);
      if (error == cudaSuccess) error = cudaStreamSynchronize(e->mg_streams[i]);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      if (completion == cudaSuccess) completion = error;
    }
    const auto restoration = cudaSetDevice(e->device);
    if (completion == cudaSuccess) completion = restoration;
    CUDSS_CHECK(status);
    CUDA_CHECK(completion);
  }
  // Failed phases may enqueue work and outlive this lease through waiting readers.
  const auto completion = cudaEventRecord(e->done, stream);
  if (status != CUDSS_STATUS_SUCCESS || completion != cudaSuccess)
    cudaStreamSynchronize(stream);
  // Keep the provider error when completion recording also fails.
  CUDSS_CHECK(status);
  CUDA_CHECK(completion);
  // Host return status alone does not report cuDSS device-side failures.
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int info = 0;
  size_t written = 0;
  CUDSS_CHECK(cudssDataGet(e->handle, e->data, CUDSS_DATA_INFO,
                          &info, sizeof(info), &written));
  if (written != sizeof(info) || info != 0)
    return ffi::Error::Internal("cuDSS phase failed device-side (info " +
                                std::to_string(info) + ")");
  return ffi::Error::Success();
}

template <ffi::DataType T>
static ffi::Error create_entry(cudaStream_t stream, ffi::Buffer<T>& values,
                               ffi::Buffer<ffi::S32>& offsets,
                               ffi::Buffer<ffi::S32>& columns,
                               ffi::Buffer<ffi::U32>& fingerprint,
                               int64_t batch, int device, const Options& options,
                               std::shared_ptr<FactorEntry>* output) {
  if (auto err = options.validate(); err.failure()) return err;
  std::unique_lock<std::recursive_mutex> group_lock(mg_phase_mutex(), std::defer_lock);
  if (options.device_count != 0) {
    group_lock.lock();
    if (batch != 1)
      return ffi::Error::InvalidArgument("cuDSS MG refuses matrix batches, not multiple RHS");
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      return ffi::Error::InvalidArgument("cuDSS MG does not support CUDA graph capture");
  }
  for (const auto& buffer : {std::pair<const void*, int64_t>(values.typed_data(), values.element_count()),
                             {offsets.typed_data(), offsets.element_count()},
                             {columns.typed_data(), columns.element_count()}})
    if (auto err = require_mg_pointer(options, buffer.first, buffer.second); err.failure()) return err;
  const int64_t n = offsets.element_count() - 1, nnz = columns.element_count();
  if (n < 1 || batch < 1 || n % batch != 0 || nnz % batch != 0 ||
      values.element_count() != nnz || fingerprint.element_count() != 2)
    return ffi::Error::InvalidArgument("cuDSS invalid expanded block CSR dimensions or fingerprint");
  auto e = std::make_shared<FactorEntry>();
  e->device = device;
  e->n = n; e->nnz = nnz; e->batch = batch;
  e->dtype = cudss_dtype<T>(); e->options = options;
  if (auto err = reserve_devices(options, device, &e->reservation); err.failure()) return err;
  CUDA_CHECK(cudaEventCreateWithFlags(&e->done, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(e->done, stream));
  CUDA_CHECK(cudaMemcpyAsync(e->fp, fingerprint.typed_data(), sizeof(e->fp),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (options.device_count == 0) {
    CUDSS_CHECK(cudssCreate(&e->handle));
    CUDSS_CHECK(cudssSetStream(e->handle, stream));
  } else {
    CUDSS_CHECK(cudssCreateMg(&e->handle, options.device_count, e->options.devices.data()));
    for (int i = 0; i < options.device_count; ++i) {
      CUDA_CHECK(cudaSetDevice(options.devices[i]));
      const auto status = cudaStreamCreateWithFlags(&e->mg_streams[i], cudaStreamNonBlocking);
      cudaSetDevice(device);
      CUDA_CHECK(status);
    }
    CUDSS_CHECK(cudssSetMgStreams(e->handle, e->mg_streams.data(), options.device_count));
  }
  CUDSS_CHECK(cudssConfigCreate(&e->config));
  if (options.device_count != 0) {
    CUDSS_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_DEVICE_COUNT,
                              &e->options.device_count, sizeof(e->options.device_count)));
    CUDSS_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_DEVICE_INDICES,
                              e->options.devices.data(), options.device_count * sizeof(int)));
  }
  const auto reordering = static_cast<cudssReorderingAlg_t>(options.reordering);
  CUDSS_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_REORDERING_ALG,
                            &reordering, sizeof(reordering)));
  const int hybrid = static_cast<int>(options.memory), ir_steps = 0;
  CUDSS_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &hybrid, sizeof(hybrid)));
  // Refinement belongs to the JAX layer, which carries the current CSR buffers.
  CUDSS_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_IR_N_STEPS, &ir_steps, sizeof(ir_steps)));
  CUDSS_CHECK(cudssDataCreate(e->handle, &e->data));
  CUDSS_CHECK(cudssMatrixCreateCsr(&e->A, n, n, nnz,
      offsets.typed_data(), nullptr, columns.typed_data(), values.typed_data(),
      CUDSS_R_32I, CUDSS_R_32I, e->dtype,
      static_cast<cudssMatrixType_t>(options.mtype),
      static_cast<cudssMatrixViewType_t>(options.mview), CUDSS_BASE_ZERO));
  // Analysis/numeric phases require dense descriptors but never read their data.
  CUDSS_CHECK(cudssMatrixCreateDn(&e->b_dummy, n, 1, n, values.typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&e->x_dummy, n, 1, n, values.typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR));
  if (auto err = execute(e.get(), stream, CUDSS_PHASE_ANALYSIS); err.failure()) return err;
  if (options.memory != 0 && options.hybrid_device_limit != 0) {
    int64_t minimum = 0;
    size_t written = 0;
    CUDSS_CHECK(cudssDataGet(e->handle, e->data,
        CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &minimum, sizeof(minimum), &written));
    if (written != sizeof(minimum) || minimum < 0)
      return ffi::Error::Internal("cuDSS returned an invalid hybrid device minimum");
    if (options.hybrid_device_limit < minimum)
      return ffi::Error(ffi::ErrorCode::kResourceExhausted,
          "cuDSS hybrid device limit is below the post-analysis device minimum");
    // This vendor limit covers factorization/solve, not analysis or host memory.
    CUDSS_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
        &options.hybrid_device_limit, sizeof(options.hybrid_device_limit)));
  }
  // Preserve per-calling-device estimates, including host attribution, without
  // assuming that host entries from several devices are additive.
  e->analysis_memory_words.fill(UINT32_MAX);
  const int count = options.device_count == 0 ? 1 : options.device_count;
  for (int i = 0; i < count; ++i) {
    const int owner = options.device_count == 0 ? device : options.devices[i];
    CUDA_CHECK(cudaSetDevice(owner));
    int64_t estimates[16]{};
    size_t written = 0;
    const auto status = cudssDataGet(e->handle, e->data, CUDSS_DATA_MEMORY_ESTIMATES,
                                    estimates, sizeof(estimates), &written);
    const auto restored = cudaSetDevice(device);
    CUDA_CHECK(restored);
    if (status == CUDSS_STATUS_NOT_SUPPORTED) continue;
    CUDSS_CHECK(status);
    if (written != sizeof(estimates))
      return ffi::Error::Internal("cuDSS returned incomplete memory estimates");
    for (int j = 0; j < 6; ++j) {
      if (estimates[j] < 0 || (options.memory == 0 && j >= 4)) continue;
      const auto bytes = static_cast<uint64_t>(estimates[j]);
      e->analysis_memory_words[(i * 6 + j) * 2] = static_cast<uint32_t>(bytes);
      e->analysis_memory_words[(i * 6 + j) * 2 + 1] = static_cast<uint32_t>(bytes >> 32);
    }
  }
  *output = std::move(e);
  return ffi::Error::Success();
}

#define OPTIONS_ARGUMENTS const int64_t partition, const int64_t capacity, \
  const int64_t mtype, const int64_t mview, const int64_t reordering, const int64_t memory, \
  const int64_t hybrid_device_limit, const int64_t group, ffi::Span<const int64_t> devices
#define OPTIONS_VALUE Options{partition, capacity, mtype, mview, reordering, memory, hybrid_device_limit, group, devices}

template <ffi::DataType T>
static ffi::Error analyze(cudaStream_t stream, ffi::Buffer<T> values,
    ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    ffi::Buffer<ffi::U32> fingerprint, ffi::ResultBuffer<ffi::S32> token,
    ffi::ResultBuffer<ffi::U32> memory_estimates,
    OPTIONS_ARGUMENTS, const int64_t batch) {
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  std::shared_ptr<FactorEntry> e;
  if (auto err = create_entry(stream, values, offsets, columns, fingerprint,
                              batch, device, OPTIONS_VALUE, &e); err.failure()) return err;
  const int count = e->options.device_count == 0 ? 1 : e->options.device_count;
  if (memory_estimates->element_count() != count * 6 * 2)
    return ffi::Error::InvalidArgument("cuDSS memory estimate output has the wrong shape");
  CUDA_CHECK(cudaMemcpyAsync(memory_estimates->typed_data(), e->analysis_memory_words.data(),
                             count * 6 * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
  return write_scalar(stream, token, registry().insert(device, partition, capacity, std::move(e)));
}

template <ffi::DataType T, bool Refactorize>
static ffi::Error numeric(cudaStream_t stream, ffi::Buffer<ffi::S32> token,
    ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    ffi::Buffer<ffi::U32> fingerprint, ffi::Buffer<T> values,
    ffi::ResultBuffer<ffi::S32> output, ffi::ResultBuffer<ffi::S32> rebuilt_output,
    OPTIONS_ARGUMENTS) {
  const auto options = OPTIONS_VALUE;
  PhaseLease lease;
  if (auto err = begin_phase<T>(stream, token, options, &lease); err.failure()) return err;
  auto& e = lease.entry;
  const bool rebuilt = !e;
  auto& r = registry();
  if (rebuilt) {
    if (auto err = create_entry(stream, values, offsets, columns, fingerprint,
                                1, lease.device, options, &e); err.failure()) return err;
  } else if (auto err = repoint(e.get(), stream, offsets, columns, fingerprint, values); err.failure()) {
    e->factorized = false;
    r.erase(lease.device, partition, lease.id);
    return err;
  }
  const auto phase = Refactorize && e->factorized ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
  e->factorized = false;
  if (auto err = execute(e.get(), stream, phase); err.failure()) {
    // Retire the old immutable name before releasing operation_mu. Readers
    // already waiting on this entry will recheck the registry and rebuild.
    if (!rebuilt) r.erase(lease.device, partition, lease.id);
    return err;
  }
  e->factorized = true;
  const auto id = rebuilt ? r.insert(lease.device, partition, capacity, e)
                          : r.rekey(lease.device, partition, capacity, lease.id, e);
  if (rebuilt) r.count_rebuild(lease.device, partition);
  if (auto err = write_scalar(stream, output, id); err.failure()) return err;
  return write_scalar(stream, rebuilt_output, rebuilt);
}

template <ffi::DataType T>
static ffi::Error factor_for_read(cudaStream_t stream, ffi::Buffer<ffi::S32>& token,
    ffi::Buffer<ffi::S32>& offsets, ffi::Buffer<ffi::S32>& columns,
    ffi::Buffer<ffi::U32>& fingerprint, ffi::Buffer<T>& values,
    const Options& options, PhaseLease* lease, bool* rebuilt) {
  if (auto err = begin_phase<T>(stream, token, options, lease); err.failure()) return err;
  auto& e = lease->entry;
  if (e) {
    // A resident read consumes the factors, not the caller's CSR storage.
    // Native refinement is disabled; keep the factorized descriptor unchanged.
    // Numeric phases rebind current operands, and rebuilds create a new A.
    if (auto err = validate_structure(e.get(), stream, offsets, columns, fingerprint, values); err.failure()) {
      registry().erase(lease->device, options.partition, lease->id);
      return err;
    }
  }
  *rebuilt = !e || !e->factorized;
  if (*rebuilt) {
    // A resident analysis entry remains independently valid; do not retain its
    // mutex lock when replacing the shared_ptr (insertion may evict it).
    if (lease->lock.owns_lock()) lease->lock.unlock();
    // Reads never resurrect the caller's immutable id: a rebuilt memo gets a
    // fresh name, leaving superseded tokens tied to their original operands.
    if (auto err = create_entry(stream, values, offsets, columns, fingerprint,
                                1, lease->device, options, &e); err.failure()) return err;
    if (auto err = execute(e.get(), stream, CUDSS_PHASE_FACTORIZATION); err.failure()) return err;
    e->factorized = true;
    lease->id = registry().insert(lease->device, options.partition, options.capacity, e);
    registry().count_rebuild(lease->device, options.partition);
  }
  return ffi::Error::Success();
}

struct DensePair {
  cudssMatrix_t b = nullptr, x = nullptr;
  ~DensePair() { if (b) cudssMatrixDestroy(b); if (x) cudssMatrixDestroy(x); }
};

template <ffi::DataType T>
static ffi::Error solve(cudaStream_t stream, ffi::Buffer<ffi::S32> token,
    ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    ffi::Buffer<ffi::U32> fingerprint, ffi::Buffer<T> values, ffi::Buffer<T> rhs,
    ffi::ResultBuffer<T> output, ffi::ResultBuffer<ffi::S32> rebuilt_output,
    OPTIONS_ARGUMENTS) {
  PhaseLease lease;
  bool rebuilt = false;
  if (auto err = factor_for_read(stream, token, offsets, columns, fingerprint,
                                 values, OPTIONS_VALUE, &lease, &rebuilt); err.failure()) return err;
  auto& e = lease.entry;
  if (rhs.element_count() == 0 || rhs.element_count() % e->n != 0 ||
      rhs.dimensions().size() == 0 || rhs.dimensions().back() != e->n ||
      output->element_count() != rhs.element_count())
    return ffi::Error::InvalidArgument("cuDSS rhs must have shape [..., N] and output must match");
  if (auto err = require_mg_pointer(e->options, rhs.typed_data(), rhs.element_count()); err.failure()) return err;
  if (auto err = require_mg_pointer(e->options, output->typed_data(), output->element_count()); err.failure()) return err;
  const int64_t nrhs = rhs.element_count() / e->n;
  DensePair dense;
  CUDSS_CHECK(cudssMatrixCreateDn(&dense.b, e->n, nrhs, e->n, rhs.typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&dense.x, e->n, nrhs, e->n, output->typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR));
  if (auto err = execute(e.get(), stream, CUDSS_PHASE_SOLVE, dense.x, dense.b); err.failure()) {
    e->factorized = false;
    registry().erase(lease.device, partition, lease.id);
    return err;
  }
  return write_scalar(stream, rebuilt_output, rebuilt);
}

template <ffi::DataType T>
static ffi::Error diagonal(cudaStream_t stream, ffi::Buffer<ffi::S32> token,
    ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    ffi::Buffer<ffi::U32> fingerprint, ffi::Buffer<T> values,
    ffi::ResultBuffer<T> output, ffi::ResultBuffer<ffi::S32> rebuilt_output,
    OPTIONS_ARGUMENTS) {
  PhaseLease lease;
  bool rebuilt = false;
  if (auto err = factor_for_read(stream, token, offsets, columns, fingerprint,
                                 values, OPTIONS_VALUE, &lease, &rebuilt); err.failure()) return err;
  auto& e = lease.entry;
  if (output->element_count() != e->n)
    return ffi::Error::InvalidArgument("cuDSS pivot diagonal output must have N elements");
  size_t written = 0;
  // cuDSS returns the pivot diagonal in input order (not reordering order).
  const auto status = cudssDataGet(e->handle, e->data, CUDSS_DATA_DIAG,
      output->typed_data(), e->n * sizeof(typename ffi::NativeType<T>), &written);
  CUDA_CHECK(cudaEventRecord(e->done, stream));
  CUDSS_CHECK(status);
  return write_scalar(stream, rebuilt_output, rebuilt);
}

static ffi::Error release(cudaStream_t stream, ffi::Buffer<ffi::S32> tokens,
                           ffi::ResultBuffer<ffi::S32> output, int64_t partition,
                           int64_t release_partition) {
  int32_t released = 0;
  if (release_partition == 1) {
    const auto completion = cudaStreamSynchronize(stream);
    released = static_cast<int32_t>(registry().release_partition(partition));
    CUDA_CHECK(completion);
  } else {
    if (release_partition != 0)
      return ffi::Error::InvalidArgument("cuDSS release mode must be zero or one");
    std::vector<int32_t> ids(tokens.element_count());
    if (!ids.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(ids.data(), tokens.typed_data(), ids.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    released = static_cast<int32_t>(
        registry().release(partition, ids.data(), ids.size()));
  }
  if (output->element_count() == 1) return write_scalar(stream, output, released);
  // broadcast_all batches the result as well as the operands. Release is one
  // deduplicated operation; every output lane receives that operation's count.
  std::vector<int32_t> counts(output->element_count(), released);
  if (!counts.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(output->typed_data(), counts.data(), counts.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return ffi::Error::Success();
}
}  // namespace

phx::cuda::RegistryBase* component_registry() { return &registry(); }
int32_t component_shutdown() { return PHX_CUDA_SUCCESS; }

#define COMMON_ATTRIBUTES \
  .Attr<int64_t>("partition").Attr<int64_t>("capacity") \
  .Attr<int64_t>("mtype").Attr<int64_t>("mview") \
  .Attr<int64_t>("reordering").Attr<int64_t>("memory") \
  .Attr<int64_t>("hybrid_device_limit").Attr<int64_t>("group") \
  .Attr<ffi::Span<const int64_t>>("devices")
#define STATE_ARGUMENTS(Type) \
  .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>() \
  .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::U32>>().Arg<ffi::Buffer<Type>>()
#define DEFINE_HANDLERS(Suffix, Type) \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_analyze_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_factorize_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_refactorize_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_solve_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_diagonal_##Suffix(XLA_FFI_CallFrame*); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_analyze_##Suffix, analyze<Type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() \
    .Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<ffi::S32>>() \
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::U32>>() \
    .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::U32>>() COMMON_ATTRIBUTES .Attr<int64_t>("batch")); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_factorize_##Suffix, (numeric<Type, false>), \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() STATE_ARGUMENTS(Type) \
    .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S32>>() COMMON_ATTRIBUTES); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_refactorize_##Suffix, (numeric<Type, true>), \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() STATE_ARGUMENTS(Type) \
    .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S32>>() COMMON_ATTRIBUTES); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_solve_##Suffix, solve<Type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() STATE_ARGUMENTS(Type) \
    .Arg<ffi::Buffer<Type>>().Ret<ffi::Buffer<Type>>() \
    .Ret<ffi::Buffer<ffi::S32>>() COMMON_ATTRIBUTES); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_diagonal_##Suffix, diagonal<Type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() STATE_ARGUMENTS(Type) \
    .Ret<ffi::Buffer<Type>>().Ret<ffi::Buffer<ffi::S32>>() COMMON_ATTRIBUTES)

extern "C" {
DEFINE_HANDLERS(f32, ffi::F32);
DEFINE_HANDLERS(f64, ffi::F64);
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_release(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_release, release,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S32>>()
    .Attr<int64_t>("partition").Attr<int64_t>("release_partition"));
}

PHX_CUDA_HANDLER_TABLE(
  {"phydrax_cudss_analyze_f32", phydrax_cudss_analyze_f32},
  {"phydrax_cudss_analyze_f64", phydrax_cudss_analyze_f64},
  {"phydrax_cudss_factorize_f32", phydrax_cudss_factorize_f32},
  {"phydrax_cudss_factorize_f64", phydrax_cudss_factorize_f64},
  {"phydrax_cudss_refactorize_f32", phydrax_cudss_refactorize_f32},
  {"phydrax_cudss_refactorize_f64", phydrax_cudss_refactorize_f64},
  {"phydrax_cudss_solve_f32", phydrax_cudss_solve_f32},
  {"phydrax_cudss_solve_f64", phydrax_cudss_solve_f64},
  {"phydrax_cudss_diagonal_f32", phydrax_cudss_diagonal_f32},
  {"phydrax_cudss_diagonal_f64", phydrax_cudss_diagonal_f64},
  {"phydrax_cudss_release", phydrax_cudss_release})
