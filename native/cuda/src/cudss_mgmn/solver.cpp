// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// cuDSS 0.8 MGMN: global descriptors, exclusively owned local row buffers.
// https://docs.nvidia.com/cuda/cudss/0.8.0/advanced_features.html#mgmn-mode-label
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "cuda_runtime_api.h"
#include "cudss.h"
#include "xla/ffi/api/ffi.h"
#include "phx_cuda_common.hpp"
#include "../common/collective_lease.hpp"
#include "sparse_action.hpp"

namespace ffi = xla::ffi;
namespace {
using phx::collective::CallGuard;

void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void mpi_check(int status) {
  if (status != MPI_SUCCESS) throw std::runtime_error("cuDSS MGMN MPI operation failed");
}
void vendor_check(cudssStatus_t status) {
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("cuDSS MGMN vendor status " + std::to_string(status));
}
// All recoverable preparation/phase errors become an all-rank refusal. Rank
// loss within a vendor collective remains owned by the lease's supervisor.
template <class Function>
void agreed(MPI_Comm comm, cudaStream_t stream, Function&& function) {
  std::string message;
  try { function(); } catch (const std::exception& error) { message = error.what(); }
  const auto completion = cudaStreamSynchronize(stream);
  if (message.empty() && completion != cudaSuccess) message = cudaGetErrorString(completion);
  int bad = !message.empty(), any_bad = 0;
  mpi_check(MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, comm));
  if (!any_bad) return;
  int rank = 0, first = std::numeric_limits<int>::max();
  mpi_check(MPI_Comm_rank(comm, &rank));
  int candidate = bad ? rank : first;
  mpi_check(MPI_Allreduce(&candidate, &first, 1, MPI_INT, MPI_MIN, comm));
  std::array<char, 1024> text{};
  if (rank == first) std::copy_n(message.data(), std::min(message.size(), text.size() - 1), text.data());
  mpi_check(MPI_Bcast(text.data(), text.size(), MPI_CHAR, first, comm));
  throw std::runtime_error(std::string("cuDSS MGMN all-rank refusal: ") + text.data());
}

template <ffi::DataType T> constexpr cudssDataType_t dtype();
template <> constexpr cudssDataType_t dtype<ffi::F32>() { return CUDSS_R_32F; }
template <> constexpr cudssDataType_t dtype<ffi::F64>() { return CUDSS_R_64F; }

template <ffi::DataType T>
auto scalar(cudaStream_t stream, const ffi::Buffer<T>& buffer) {
  if (buffer.element_count() != 1) throw std::invalid_argument("MGMN scalar operand must have one element");
  typename ffi::NativeType<T> result;
  cuda_check(cudaMemcpyAsync(&result, buffer.typed_data(), sizeof(result), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  return result;
}
template <ffi::DataType T, class Value>
void put(cudaStream_t stream, ffi::ResultBuffer<T>& output, const Value* data, size_t count) {
  if (output->element_count() != static_cast<int64_t>(count))
    throw std::invalid_argument("MGMN result buffer has the wrong extent");
  cuda_check(cudaMemcpyAsync(output->typed_data(), data, count * sizeof(Value), cudaMemcpyHostToDevice, stream));
  cuda_check(cudaStreamSynchronize(stream));
}

struct Options {
  int64_t n, partition, capacity, reordering;
  ffi::Span<const int64_t> starts;
  std::string_view preparation, structure, comm_path, comm_sha, comm_build;
  int64_t local_rows() const { return starts[partition + 1] - starts[partition]; }
  void validate(MPI_Comm comm) const {
    int rank = -1, size = 0;
    mpi_check(MPI_Comm_rank(comm, &rank)); mpi_check(MPI_Comm_size(comm, &size));
    if (n < 1 || n > INT32_MAX || starts.size() != static_cast<size_t>(size + 1) ||
        partition != rank || starts[0] != 0 || starts[size] != n || capacity < 1 ||
        preparation.empty() || structure.empty())
      throw std::invalid_argument("MGMN requires global int32 dimensions and exact ordered rank ownership");
    for (int i = 0; i < size; ++i)
      if (starts[i] >= starts[i + 1]) throw std::invalid_argument("MGMN rows must be contiguous, disjoint, nonempty and unpadded");
    if (reordering != CUDSS_REORDERING_ALG_DEFAULT && reordering != CUDSS_REORDERING_ALG_AMD &&
        reordering != CUDSS_REORDERING_ALG_NESTED_DISSECTION)
      throw std::invalid_argument("cuDSS 0.8 MGMN refuses COLAMD/BTF_COLAMD");
    if (comm_path != PHX_CUDSS_COMM_LAYER || comm_sha != PHX_CUDSS_COMM_SHA256 ||
        comm_build != PHX_CUDSS_COMM_BUILD_IDENTITY)
      throw std::invalid_argument("MGMN communication layer differs from the explicitly pinned site build");
  }
};
void agree_options(const Options& options, MPI_Comm comm, cudaStream_t stream) {
  std::string local = std::to_string(options.n) + ":" + std::to_string(options.capacity) +
      ":" + std::to_string(options.reordering);
  for (const auto value : {options.preparation, options.structure, options.comm_sha, options.comm_build})
    local += ":" + std::to_string(value.size()) + ":" + std::string(value);
  for (const auto row : options.starts) local += ":" + std::to_string(row);
  int rank = 0;
  mpi_check(MPI_Comm_rank(comm, &rank));
  int length = static_cast<int>(local.size());
  mpi_check(MPI_Bcast(&length, 1, MPI_INT, 0, comm));
  std::string global = rank == 0 ? local : std::string(length, '\0');
  mpi_check(MPI_Bcast(global.data(), length, MPI_CHAR, 0, comm));
  agreed(comm, stream, [&] {
    if (global != local) throw std::invalid_argument("MGMN global descriptors and solver configuration differ across ranks");
  });
}
void agree_extent(int64_t extent, MPI_Comm comm, cudaStream_t stream) {
  int64_t smallest = 0, largest = 0;
  mpi_check(MPI_Allreduce(&extent, &smallest, 1, MPI_INT64_T, MPI_MIN, comm));
  mpi_check(MPI_Allreduce(&extent, &largest, 1, MPI_INT64_T, MPI_MAX, comm));
  agreed(comm, stream, [&] {
    if (smallest != largest) throw std::invalid_argument("MGMN RHS/statistic column counts differ across ranks");
  });
}

// No destructor performs collective/vendor work. Only agreed phases and the
// shared lease's retirement callback destroy these resources.
struct Entry {
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t matrix = nullptr, dummy_x = nullptr, dummy_b = nullptr;
  MPI_Comm comm = MPI_COMM_NULL;
  int device = -1;
  bool factorized = false;
  cudssDataType_t value_type = CUDSS_R_64F;
  int64_t global_n = 0, first_row = 0, last_row = 0, reordering = 0;
  std::vector<int32_t> offsets, columns;
  std::vector<unsigned char> values;
  std::array<uint32_t, 12> memory{};
  bool matches(const Options& options) const {
    return global_n == options.n && first_row == options.starts[options.partition] &&
        last_row == options.starts[options.partition + 1] - 1 && reordering == options.reordering;
  }
};
void destroy(Entry& entry) {
  cuda_check(cudaSetDevice(entry.device));
  // Every phase and result copy already completed its stream before retirement.
  if (entry.dummy_b) { vendor_check(cudssMatrixDestroy(entry.dummy_b)); entry.dummy_b = nullptr; }
  if (entry.dummy_x) { vendor_check(cudssMatrixDestroy(entry.dummy_x)); entry.dummy_x = nullptr; }
  if (entry.matrix) { vendor_check(cudssMatrixDestroy(entry.matrix)); entry.matrix = nullptr; }
  if (entry.data) { vendor_check(cudssDataDestroy(entry.handle, entry.data)); entry.data = nullptr; }
  if (entry.config) { vendor_check(cudssConfigDestroy(entry.config)); entry.config = nullptr; }
  if (entry.handle) { vendor_check(cudssDestroy(entry.handle)); entry.handle = nullptr; }
}
struct Cache {
  std::map<int64_t, std::unique_ptr<Entry>> entries;
  MPI_Comm comm = MPI_COMM_NULL;
  int64_t next_id = 0;
};
using CacheKey = std::pair<int64_t, std::string>;
std::mutex& cache_mutex() { static auto* mutex = new std::mutex; return *mutex; }
std::map<CacheKey, std::shared_ptr<Cache>>& caches() {
  static auto* result = new std::map<CacheKey, std::shared_ptr<Cache>>;
  return *result;
}
std::shared_ptr<Cache> cache(CallGuard& guard, int64_t lease, std::string_view preparation) {
  CacheKey key{lease, std::string(preparation)};
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    const auto found = caches().find(key);
    if (found != caches().end()) return found->second;
  }
  auto result = std::make_shared<Cache>();
  mpi_check(MPI_Comm_dup(guard.communicator(), &result->comm));
  guard.on_retire("cudss_mgmn:" + std::string(preparation), [key, result](MPI_Comm comm) {
    // Retirement follows the union of numeric names. Communicator ownership
    // is independent of evictable memo membership on any particular rank.
    int size = 0;
    mpi_check(MPI_Comm_size(comm, &size));
    std::vector<int64_t> ids;
    for (const auto& item : result->entries) ids.push_back(item.first);
    const int count = static_cast<int>(ids.size());
    std::vector<int> counts(size), offsets(size);
    mpi_check(MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm));
    for (int i = 1; i < size; ++i) offsets[i] = offsets[i - 1] + counts[i - 1];
    std::vector<int64_t> all(offsets.back() + counts.back());
    mpi_check(MPI_Allgatherv(ids.data(), count, MPI_INT64_T, all.data(),
        counts.data(), offsets.data(), MPI_INT64_T, comm));
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    for (const auto id : all) {
      const auto found = result->entries.find(id);
      agreed(comm, nullptr, [&] { if (found != result->entries.end()) destroy(*found->second); });
    }
    result->entries.clear();
    mpi_check(MPI_Comm_free(&result->comm));
    std::lock_guard<std::mutex> lock(cache_mutex());
    caches().erase(key);
  });
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    caches().emplace(std::move(key), result);
  }
  return result;
}
void erase(Cache& cache, int64_t id, MPI_Comm comm, cudaStream_t stream) {
  const auto found = cache.entries.find(id);
  agreed(comm, stream, [&] { if (found != cache.entries.end()) destroy(*found->second); });
  if (found != cache.entries.end()) cache.entries.erase(found);
}

struct Snapshot {
  std::vector<int32_t> offsets, columns;
  std::vector<unsigned char> values;
};
template <ffi::DataType T>
Snapshot validate_buffers(cudaStream_t stream, const Options& options,
    ffi::Buffer<T> values, ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns) {
  if (values.dimensions().size() != 1 || offsets.dimensions().size() != 1 ||
      columns.dimensions().size() != 1 || values.element_count() != columns.element_count() ||
      values.element_count() > INT32_MAX || offsets.element_count() != options.local_rows() + 1)
    throw std::invalid_argument("MGMN requires one unbatched rank-local CSR with no padded rows");
  cudaStreamCaptureStatus capture;
  cuda_check(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone) throw std::invalid_argument("MGMN does not support CUDA graph capture");
  int device = -1; cuda_check(cudaGetDevice(&device));
  for (const auto& buffer : {std::pair<const void*, int64_t>{offsets.untyped_data(), offsets.element_count()},
      {columns.untyped_data(), columns.element_count()}, {values.untyped_data(), values.element_count()}}) {
    if (buffer.second == 0) continue;
    cudaPointerAttributes attributes{};
    cuda_check(cudaPointerGetAttributes(&attributes, buffer.first));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
      throw std::invalid_argument("MGMN numeric leaves must be buffers on the initiating rank's GPU");
  }
  Snapshot snapshot;
  snapshot.offsets.resize(offsets.element_count()); snapshot.columns.resize(columns.element_count());
  snapshot.values.resize(values.element_count() * sizeof(typename ffi::NativeType<T>));
  cuda_check(cudaMemcpyAsync(snapshot.offsets.data(), offsets.typed_data(), snapshot.offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  if (!snapshot.columns.empty()) {
    cuda_check(cudaMemcpyAsync(snapshot.columns.data(), columns.typed_data(), snapshot.columns.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(snapshot.values.data(), values.typed_data(), snapshot.values.size(), cudaMemcpyDeviceToHost, stream));
  }
  cuda_check(cudaStreamSynchronize(stream));
  if (snapshot.offsets.front() != 0 || snapshot.offsets.back() != values.element_count())
    throw std::invalid_argument("MGMN CSR offsets must start at zero and end at local nnz");
  for (int64_t row = 0; row < options.local_rows(); ++row) {
    const auto begin = snapshot.offsets[row], end = snapshot.offsets[row + 1];
    if (begin < 0 || end < begin || end > values.element_count()) throw std::invalid_argument("MGMN CSR offsets are not monotone");
    int32_t previous = -1;
    for (int32_t i = begin; i < end; ++i) {
      const auto column = snapshot.columns[i];
      typename ffi::NativeType<T> value;
      std::memcpy(&value, snapshot.values.data() + i * sizeof(value), sizeof(value));
      if (column <= previous || column >= options.n || !std::isfinite(value))
        throw std::invalid_argument("MGMN CSR entries must be finite with sorted unique global int32 columns");
      previous = column;
    }
  }
  return snapshot;
}
void execute(Entry& entry, MPI_Comm control, cudaStream_t stream, int phase,
    cudssMatrix_t x = nullptr, cudssMatrix_t b = nullptr) {
  agreed(control, stream, [&] { vendor_check(cudssSetStream(entry.handle, stream)); });
  // MGMN phases are synchronous, but completion is still explicitly owned.
  const auto status = cudssExecute(entry.handle, phase, entry.config, entry.data,
      entry.matrix, x ? x : entry.dummy_x, b ? b : entry.dummy_b);
  agreed(control, stream, [&] {
    cuda_check(cudaStreamSynchronize(stream));
    vendor_check(status);
    int info = 0; size_t written = 0;
    vendor_check(cudssDataGet(entry.handle, entry.data, CUDSS_DATA_INFO, &info, sizeof(info), &written));
    if (written != sizeof(info) || info != 0)
      throw std::runtime_error("cuDSS MGMN device-side phase failure, info " + std::to_string(info));
  });
}

template <ffi::DataType T>
std::unique_ptr<Entry> create(MPI_Comm control, MPI_Comm numerical, cudaStream_t stream, const Options& options,
    ffi::Buffer<T> values, ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    Snapshot snapshot) {
  auto entry = std::make_unique<Entry>();
  auto& e = *entry;
  e.comm = numerical;
  e.global_n = options.n; e.first_row = options.starts[options.partition];
  e.last_row = options.starts[options.partition + 1] - 1; e.reordering = options.reordering;
  int64_t local_nnz = values.element_count(), global_nnz = 0;
  mpi_check(MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, control));
  try {
    agreed(control, stream, [&] {
      cuda_check(cudaGetDevice(&e.device)); e.value_type = dtype<T>();
      vendor_check(cudssCreate(&e.handle));
      vendor_check(cudssSetStream(e.handle, stream));
      vendor_check(cudssSetCommLayer(e.handle, PHX_CUDSS_COMM_LAYER));
      vendor_check(cudssConfigCreate(&e.config));
      const int zero = 0;
      const auto reordering = static_cast<cudssReorderingAlg_t>(options.reordering);
      vendor_check(cudssConfigSet(e.config, CUDSS_CONFIG_REORDERING_ALG, &reordering, sizeof(reordering)));
      vendor_check(cudssConfigSet(e.config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &zero, sizeof(zero)));
      vendor_check(cudssConfigSet(e.config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &zero, sizeof(zero)));
      vendor_check(cudssConfigSet(e.config, CUDSS_CONFIG_IR_N_STEPS, &zero, sizeof(zero)));
      vendor_check(cudssDataCreate(e.handle, &e.data));
      vendor_check(cudssDataSet(e.handle, e.data, CUDSS_DATA_COMM_HOST, &e.comm, sizeof(e.comm)));
      vendor_check(cudssDataSet(e.handle, e.data, CUDSS_DATA_COMM_DEVICE, &e.comm, sizeof(e.comm)));
      vendor_check(cudssMatrixCreateCsr(&e.matrix, options.n, options.n, global_nnz,
          offsets.typed_data(), nullptr, columns.typed_data(), values.typed_data(),
          CUDSS_R_32I, CUDSS_R_32I, e.value_type, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
      vendor_check(cudssMatrixSetDistributionRow1d(e.matrix, options.starts[options.partition], options.starts[options.partition + 1] - 1));
      // cuDSS 0.8 cudssExecute explicitly requires global nrows/ncols and
      // LOCAL ld for distributed dense buffers, including true multiple RHS.
      vendor_check(cudssMatrixCreateDn(&e.dummy_b, options.n, 1, options.local_rows(), values.typed_data(), e.value_type, CUDSS_LAYOUT_COL_MAJOR));
      vendor_check(cudssMatrixCreateDn(&e.dummy_x, options.n, 1, options.local_rows(), values.typed_data(), e.value_type, CUDSS_LAYOUT_COL_MAJOR));
      vendor_check(cudssMatrixSetDistributionRow1d(e.dummy_b, options.starts[options.partition], options.starts[options.partition + 1] - 1));
      vendor_check(cudssMatrixSetDistributionRow1d(e.dummy_x, options.starts[options.partition], options.starts[options.partition + 1] - 1));
    });
    execute(e, control, stream, CUDSS_PHASE_ANALYSIS);
    agreed(control, stream, [&] {
      e.memory.fill(UINT32_MAX);
      int64_t estimates[16]{}; size_t written = 0;
      const auto status = cudssDataGet(e.handle, e.data, CUDSS_DATA_MEMORY_ESTIMATES, estimates, sizeof(estimates), &written);
      if (status != CUDSS_STATUS_NOT_SUPPORTED) {
        vendor_check(status);
        if (written != sizeof(estimates)) throw std::runtime_error("MGMN returned incomplete memory estimates");
        // Hybrid-only fields 4/5 are undefined in admitted device-only MGMN.
        for (int i = 0; i < 4; ++i) if (estimates[i] >= 0) {
          const auto bytes = static_cast<uint64_t>(estimates[i]);
          e.memory[2 * i] = static_cast<uint32_t>(bytes); e.memory[2 * i + 1] = static_cast<uint32_t>(bytes >> 32);
        }
      }
    });
    e.offsets = std::move(snapshot.offsets); e.columns = std::move(snapshot.columns); e.values = std::move(snapshot.values);
    return entry;
  } catch (...) {
    agreed(control, stream, [&] { destroy(e); });
    throw;
  }
}
int64_t insert(Cache& cache, std::unique_ptr<Entry> entry, const Options& options,
    MPI_Comm comm, cudaStream_t stream) {
  if (cache.next_id == INT64_MAX) throw std::runtime_error("MGMN numeric generation space exhausted");
  const auto id = ++cache.next_id;
  cache.entries.emplace(id, std::move(entry));
  int excess = cache.entries.size() > static_cast<uint64_t>(options.capacity), any_excess = 0;
  mpi_check(MPI_Allreduce(&excess, &any_excess, 1, MPI_INT, MPI_MAX, comm));
  while (any_excess) {
    int64_t oldest = cache.entries.empty() ? INT64_MAX : cache.entries.begin()->first, global_oldest = 0;
    mpi_check(MPI_Allreduce(&oldest, &global_oldest, 1, MPI_INT64_T, MPI_MIN, comm));
    erase(cache, global_oldest, comm, stream);
    excess = cache.entries.size() > static_cast<uint64_t>(options.capacity);
    mpi_check(MPI_Allreduce(&excess, &any_excess, 1, MPI_INT, MPI_MAX, comm));
  }
  return id;
}

#define OPTIONS_ARGS int64_t global_n, ffi::Span<const int64_t> row_starts, int64_t partition_index, \
    int64_t capacity, int64_t reordering, std::string_view preparation_id, std::string_view structure_id, \
    std::string_view communication_library_path, std::string_view communication_library_sha256, std::string_view communication_build_identity
#define OPTIONS Options{global_n, partition_index, capacity, reordering, row_starts, preparation_id, structure_id, \
    communication_library_path, communication_library_sha256, communication_build_identity}
#define GUARD_ARGS std::string_view collective_configuration_id, std::string_view collective_call_id

template <ffi::DataType T, int Phase>
ffi::Error lifecycle(cudaStream_t stream, ffi::Token, ffi::Buffer<ffi::S64> lease_buf,
    ffi::Buffer<ffi::S64> ticket_buf, ffi::Buffer<ffi::S64> id_buf,
    ffi::Buffer<T> values, ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    ffi::Result<ffi::Token>, ffi::ResultBuffer<ffi::S64> output,
    ffi::ResultBuffer<ffi::S32> rebuilt_output, ffi::ResultBuffer<ffi::U32> memory_output,
    OPTIONS_ARGS, GUARD_ARGS) {
  try {
    const auto lease_id = scalar(stream, lease_buf), ticket = scalar(stream, ticket_buf);
    CallGuard guard(lease_id, collective_configuration_id, collective_call_id, ticket);
    const auto comm = guard.communicator(); const auto options = OPTIONS;
    Snapshot snapshot; int64_t old_id = 0;
    agreed(comm, stream, [&] {
      options.validate(comm); old_id = scalar(stream, id_buf);
      snapshot = validate_buffers(stream, options, values, offsets, columns);
    });
    agree_options(options, comm, stream);
    auto owner = cache(guard, lease_id, preparation_id);
    const auto found = owner->entries.find(old_id);
    int missing = found == owner->entries.end(), rebuild = 0;
    if (!missing) missing = !found->second->matches(options) || found->second->value_type != dtype<T>();
    mpi_check(MPI_Allreduce(&missing, &rebuild, 1, MPI_INT, MPI_MAX, comm));
    std::unique_ptr<Entry> entry;
    if constexpr (Phase == 0) {
      entry = create(comm, owner->comm, stream, options, values, offsets, columns, std::move(snapshot));
    } else {
      if (rebuild) {
        erase(*owner, old_id, comm, stream);
        entry = create(comm, owner->comm, stream, options, values, offsets, columns, std::move(snapshot));
      } else {
        entry = std::move(found->second);
        owner->entries.erase(found);
      }
      try {
        agreed(comm, stream, [&] {
          if (!rebuild && (entry->offsets != snapshot.offsets || entry->columns != snapshot.columns))
            throw std::invalid_argument("MGMN binding cannot change the prepared CSR pattern");
          vendor_check(cudssMatrixSetCsrPointers(entry->matrix, offsets.typed_data(), nullptr, columns.typed_data(), values.typed_data()));
        });
        const int phase = Phase == 2 && entry->factorized ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
        entry->factorized = false;
        execute(*entry, comm, stream, phase);
        entry->factorized = true;
        if (!rebuild) entry->values = std::move(snapshot.values);
      } catch (...) { agreed(comm, stream, [&] { destroy(*entry); }); throw; }
    }
    const auto memory = entry->memory;
    const auto id = insert(*owner, std::move(entry), options, comm, stream);
    const int32_t count = Phase == 0 ? 0 : rebuild;
    agreed(comm, stream, [&] { put(stream, output, &id, 1); put(stream, rebuilt_output, &count, 1); put(stream, memory_output, memory.data(), memory.size()); });
    guard.complete(); return ffi::Error::Success();
  } catch (const std::exception& error) { return ffi::Error::Internal(error.what()); }
}

template <ffi::DataType T>
ffi::Error solve(cudaStream_t stream, ffi::Token, ffi::Buffer<ffi::S64> lease_buf,
    ffi::Buffer<ffi::S64> ticket_buf, ffi::Buffer<ffi::S64> id_buf,
    ffi::Buffer<T> values, ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    ffi::Buffer<T> rhs, ffi::Result<ffi::Token>, ffi::ResultBuffer<T> output,
    ffi::ResultBuffer<ffi::S32> rebuilt_output, ffi::ResultBuffer<ffi::S64> memo_output, OPTIONS_ARGS, GUARD_ARGS) {
  try {
    const auto lease_id = scalar(stream, lease_buf), ticket = scalar(stream, ticket_buf);
    CallGuard guard(lease_id, collective_configuration_id, collective_call_id, ticket);
    const auto comm = guard.communicator(); const auto options = OPTIONS;
    Snapshot snapshot; int64_t id = 0, nrhs = 0;
    agreed(comm, stream, [&] {
      options.validate(comm); id = scalar(stream, id_buf);
      snapshot = validate_buffers(stream, options, values, offsets, columns);
      if (rhs.dimensions().size() != 2 || rhs.dimensions()[1] != options.local_rows() ||
          rhs.dimensions()[0] < 1 || output->dimensions().size() != 2 ||
          output->dimensions()[0] != rhs.dimensions()[0] || output->dimensions()[1] != rhs.dimensions()[1])
        throw std::invalid_argument("MGMN RHS requires [nrhs,owned_rows], not a matrix batch or global vector");
      nrhs = rhs.dimensions()[0];
    });
    agree_options(options, comm, stream);
    agree_extent(nrhs, comm, stream);
    auto owner = cache(guard, lease_id, preparation_id);
    auto found = owner->entries.find(id);
    int miss = found == owner->entries.end();
    if (!miss) {
      const auto& entry = *found->second;
      miss = !entry.factorized || !entry.matches(options) || entry.value_type != dtype<T>() || entry.offsets != snapshot.offsets ||
          entry.columns != snapshot.columns || entry.values != snapshot.values;
    }
    int rebuild = 0; mpi_check(MPI_Allreduce(&miss, &rebuild, 1, MPI_INT, MPI_MAX, comm));
    Entry* entry = nullptr;
    if (rebuild) {
      auto fresh = create(comm, owner->comm, stream, options, values, offsets, columns, std::move(snapshot));
      try { execute(*fresh, comm, stream, CUDSS_PHASE_FACTORIZATION); fresh->factorized = true; }
      catch (...) { agreed(comm, stream, [&] { destroy(*fresh); }); throw; }
      // Never resurrect an immutable caller ID: rebuilt state receives a fresh
      // generation; old states continue to reconstruct from their own operands.
      id = insert(*owner, std::move(fresh), options, comm, stream);
      entry = owner->entries.at(id).get();
    } else entry = found->second.get();
    cudssMatrix_t b = nullptr, x = nullptr;
    try {
      agreed(comm, stream, [&] {
        vendor_check(cudssMatrixSetCsrPointers(entry->matrix, offsets.typed_data(), nullptr, columns.typed_data(), values.typed_data()));
        vendor_check(cudssMatrixCreateDn(&b, options.n, nrhs, options.local_rows(), rhs.typed_data(), dtype<T>(), CUDSS_LAYOUT_COL_MAJOR));
        vendor_check(cudssMatrixCreateDn(&x, options.n, nrhs, options.local_rows(), output->typed_data(), dtype<T>(), CUDSS_LAYOUT_COL_MAJOR));
        vendor_check(cudssMatrixSetDistributionRow1d(b, options.starts[options.partition], options.starts[options.partition + 1] - 1));
        vendor_check(cudssMatrixSetDistributionRow1d(x, options.starts[options.partition], options.starts[options.partition + 1] - 1));
      });
      execute(*entry, comm, stream, CUDSS_PHASE_SOLVE, x, b);
    } catch (...) {
      agreed(comm, stream, [&] { if (b) vendor_check(cudssMatrixDestroy(b)); if (x) vendor_check(cudssMatrixDestroy(x)); });
      throw;
    }
    agreed(comm, stream, [&] {
      vendor_check(cudssMatrixDestroy(b)); vendor_check(cudssMatrixDestroy(x));
      const int32_t count = rebuild; put(stream, rebuilt_output, &count, 1);
      put(stream, memo_output, &id, 1);
    });
    guard.complete(); return ffi::Error::Success();
  } catch (const std::exception& error) { return ffi::Error::Internal(error.what()); }
}

// Only O(nrhs) owned statistics cross the host boundary, never a full vector.
template <ffi::DataType T>
ffi::Error norm(cudaStream_t stream, ffi::Token, ffi::Buffer<ffi::S64> lease_buf,
    ffi::Buffer<ffi::S64> ticket_buf, ffi::Buffer<T> statistics,
    ffi::Result<ffi::Token>, ffi::ResultBuffer<T> output, int64_t require_success,
    double relative_tolerance, double absolute_tolerance, GUARD_ARGS) {
  try {
    CallGuard guard(scalar(stream, lease_buf), collective_configuration_id, collective_call_id, scalar(stream, ticket_buf));
    auto comm = guard.communicator();
    const int64_t count = statistics.element_count();
    std::vector<typename ffi::NativeType<T>> local, global;
    agreed(comm, stream, [&] {
      if (statistics.dimensions().size() != 2 || statistics.dimensions()[0] != 3 || count > INT32_MAX || count == 0)
        throw std::invalid_argument("MGMN norms require [3,nrhs] owned statistics");
      local.resize(count); global.resize(count);
      cuda_check(cudaMemcpyAsync(local.data(), statistics.typed_data(), count * sizeof(local[0]), cudaMemcpyDeviceToHost, stream));
    });
    agree_extent(count, comm, stream);
    const double settings[3] = {static_cast<double>(require_success), relative_tolerance, absolute_tolerance};
    double smallest[3], largest[3];
    mpi_check(MPI_Allreduce(settings, smallest, 3, MPI_DOUBLE, MPI_MIN, comm));
    mpi_check(MPI_Allreduce(settings, largest, 3, MPI_DOUBLE, MPI_MAX, comm));
    agreed(comm, stream, [&] {
      for (int i = 0; i < 3; ++i)
        if (!std::isfinite(settings[i]) || smallest[i] != largest[i] || settings[i] < 0)
          throw std::invalid_argument("MGMN residual/derivative acceptance configuration differs across ranks");
    });
    const auto type = T == ffi::F32 ? MPI_FLOAT : MPI_DOUBLE;
    const int columns = static_cast<int>(count / 3);
    mpi_check(MPI_Allreduce(local.data(), global.data(), 2 * columns, type, MPI_SUM, comm));
    mpi_check(MPI_Allreduce(local.data() + 2 * columns, global.data() + 2 * columns, columns, type, MPI_MIN, comm));
    agreed(comm, stream, [&] {
      if (require_success) for (int i = 0; i < columns; ++i) {
        if (global[2 * columns + i] != 1 || !std::isfinite(global[i]) ||
            !std::isfinite(global[columns + i]) ||
            std::sqrt(global[i]) > absolute_tolerance + relative_tolerance * std::sqrt(global[columns + i]))
          throw std::runtime_error("cuDSS MGMN implicit derivative failed its owned true-residual contract");
      }
    });
    agreed(comm, stream, [&] { put(stream, output, global.data(), global.size()); });
    guard.complete(); return ffi::Error::Success();
  } catch (const std::exception& error) { return ffi::Error::Internal(error.what()); }
}
ffi::Error release(cudaStream_t stream, ffi::Token, ffi::Buffer<ffi::S64> lease_buf,
    ffi::Buffer<ffi::S64> ticket_buf, ffi::Buffer<ffi::S64> id_buf,
    ffi::Result<ffi::Token>, ffi::ResultBuffer<ffi::S32> output,
    std::string_view preparation_id, GUARD_ARGS) {
  try {
    const auto lease_id = scalar(stream, lease_buf);
    CallGuard guard(lease_id, collective_configuration_id, collective_call_id, scalar(stream, ticket_buf));
    auto owner = cache(guard, lease_id, preparation_id);
    const auto id = scalar(stream, id_buf);
    int32_t local_count = owner->entries.count(id), count = 0;
    mpi_check(MPI_Allreduce(&local_count, &count, 1, MPI_INT32_T, MPI_MAX, guard.communicator()));
    erase(*owner, id, guard.communicator(), stream);
    agreed(guard.communicator(), stream, [&] { put(stream, output, &count, 1); });
    guard.complete(); return ffi::Error::Success();
  } catch (const std::exception& error) { return ffi::Error::Internal(error.what()); }
}
}  // namespace

// Generic local release/atexit must never trigger communicator operations.
phx::cuda::RegistryBase* component_registry() { return nullptr; }
int32_t component_shutdown() { return PHX_CUDA_SUCCESS; }
extern "C" {
PHX_CUDA_EXPORT const char* phx_cuda_cudss_mgmn_communication_library_path() { return PHX_CUDSS_COMM_LAYER; }
PHX_CUDA_EXPORT const char* phx_cuda_cudss_mgmn_communication_library_sha256() { return PHX_CUDSS_COMM_SHA256; }
PHX_CUDA_EXPORT const char* phx_cuda_cudss_mgmn_communication_build_identity() { return PHX_CUDSS_COMM_BUILD_IDENTITY; }
}
#define GUARD_ATTRIBUTES .Attr<std::string_view>("collective_configuration_id").Attr<std::string_view>("collective_call_id")
#define OPTION_ATTRIBUTES .Attr<int64_t>("global_n").Attr<ffi::Span<const int64_t>>("row_starts") \
  .Attr<int64_t>("partition_index").Attr<int64_t>("capacity").Attr<int64_t>("reordering") \
  .Attr<std::string_view>("preparation_id").Attr<std::string_view>("structure_id") \
  .Attr<std::string_view>("communication_library_path").Attr<std::string_view>("communication_library_sha256") \
  .Attr<std::string_view>("communication_build_identity") GUARD_ATTRIBUTES
#define PREFIX ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Token>() \
  .Arg<ffi::Buffer<ffi::S64>>().Arg<ffi::Buffer<ffi::S64>>()
#define STATE(Type) .Arg<ffi::Buffer<ffi::S64>>().Arg<ffi::Buffer<Type>>() \
  .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
#define DEFINE(Type, Suffix) \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_analyze_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_factorize_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_refactorize_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_solve_##Suffix(XLA_FFI_CallFrame*); \
  PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_norm_##Suffix(XLA_FFI_CallFrame*); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_analyze_##Suffix, (lifecycle<Type, 0>), PREFIX STATE(Type) \
    .Ret<ffi::Token>().Ret<ffi::Buffer<ffi::S64>>().Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::U32>>() OPTION_ATTRIBUTES); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_factorize_##Suffix, (lifecycle<Type, 1>), PREFIX STATE(Type) \
    .Ret<ffi::Token>().Ret<ffi::Buffer<ffi::S64>>().Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::U32>>() OPTION_ATTRIBUTES); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_refactorize_##Suffix, (lifecycle<Type, 2>), PREFIX STATE(Type) \
    .Ret<ffi::Token>().Ret<ffi::Buffer<ffi::S64>>().Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::U32>>() OPTION_ATTRIBUTES); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_solve_##Suffix, solve<Type>, PREFIX STATE(Type) \
    .Arg<ffi::Buffer<Type>>().Ret<ffi::Token>().Ret<ffi::Buffer<Type>>().Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S64>>() OPTION_ATTRIBUTES); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_norm_##Suffix, norm<Type>, PREFIX \
    .Arg<ffi::Buffer<Type>>().Ret<ffi::Token>().Ret<ffi::Buffer<Type>>() \
    .Attr<int64_t>("require_success").Attr<double>("relative_tolerance").Attr<double>("absolute_tolerance") GUARD_ATTRIBUTES)
extern "C" {
DEFINE(ffi::F32, f32);
DEFINE(ffi::F64, f64);
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_release(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_release, release, PREFIX
  .Arg<ffi::Buffer<ffi::S64>>().Ret<ffi::Token>().Ret<ffi::Buffer<ffi::S32>>()
  .Attr<std::string_view>("preparation_id") GUARD_ATTRIBUTES);
}
PHX_CUDA_HANDLER_TABLE(
  {"phydrax_cudss_mgmn_analyze_f32", phydrax_cudss_mgmn_analyze_f32},
  {"phydrax_cudss_mgmn_analyze_f64", phydrax_cudss_mgmn_analyze_f64},
  {"phydrax_cudss_mgmn_factorize_f32", phydrax_cudss_mgmn_factorize_f32},
  {"phydrax_cudss_mgmn_factorize_f64", phydrax_cudss_mgmn_factorize_f64},
  {"phydrax_cudss_mgmn_refactorize_f32", phydrax_cudss_mgmn_refactorize_f32},
  {"phydrax_cudss_mgmn_refactorize_f64", phydrax_cudss_mgmn_refactorize_f64},
  {"phydrax_cudss_mgmn_solve_f32", phydrax_cudss_mgmn_solve_f32},
  {"phydrax_cudss_mgmn_solve_f64", phydrax_cudss_mgmn_solve_f64},
  {"phydrax_cudss_mgmn_norm_f32", phydrax_cudss_mgmn_norm_f32},
  {"phydrax_cudss_mgmn_norm_f64", phydrax_cudss_mgmn_norm_f64},
  {"phydrax_cudss_mgmn_exchange_f32", phydrax_cudss_mgmn_exchange_f32},
  {"phydrax_cudss_mgmn_exchange_f64", phydrax_cudss_mgmn_exchange_f64},
  {"phydrax_cudss_mgmn_reduce_f32", phydrax_cudss_mgmn_reduce_f32},
  {"phydrax_cudss_mgmn_reduce_f64", phydrax_cudss_mgmn_reduce_f64},
  {"phydrax_cudss_mgmn_reduce_i32", phydrax_cudss_mgmn_reduce_i32},
  {"phydrax_cudss_mgmn_release", phydrax_cudss_mgmn_release})
