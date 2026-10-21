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
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
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
    return ffi::Error::Success();
  }
};

struct FactorEntry {
  std::mutex operation_mu;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t A = nullptr, x_dummy = nullptr, b_dummy = nullptr;
  cudaEvent_t done = nullptr;
  uint32_t fp[2] = {0, 0};
  int64_t n = 0, nnz = 0, batch = 1;
  int device = -1;
  cudssDataType_t dtype = CUDSS_R_64F;
  Options options{};
  bool factorized = false;

  ~FactorEntry() {
    int previous = -1;
    cudaGetDevice(&previous);
    if (device >= 0) cudaSetDevice(device);
    // Completion must precede every cuDSS teardown, including LRU eviction.
    if (done) { cudaEventSynchronize(done); cudaEventDestroy(done); }
    if (A) cudssMatrixDestroy(A);
    if (b_dummy) cudssMatrixDestroy(b_dummy);
    if (x_dummy) cudssMatrixDestroy(x_dummy);
    if (data && handle) cudssDataDestroy(handle, data);
    if (config) cudssConfigDestroy(config);
    if (handle) cudssDestroy(handle);
    if (previous >= 0) cudaSetDevice(previous);
  }
};

phx::cuda::Registry<FactorEntry>& registry() {
  static auto* instance = new phx::cuda::Registry<FactorEntry>();
  return *instance;
}

struct PhaseLease {
  std::shared_ptr<FactorEntry> entry;
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
        o.reordering != options.reordering || o.memory != options.memory)
      return ffi::Error::InvalidArgument("cuDSS token configuration mismatch");
    CUDA_CHECK(cudaStreamWaitEvent(stream, e->done, 0));
    CUDSS_CHECK(cudssSetStream(e->handle, stream));
    lease->entry = std::move(e);
    lease->lock = std::move(lock);
    return ffi::Error::Success();
  }
}

template <ffi::DataType T>
static ffi::Error repoint(FactorEntry* e, cudaStream_t stream,
                          ffi::Buffer<ffi::S32>& offsets,
                          ffi::Buffer<ffi::S32>& columns,
                          ffi::Buffer<ffi::U32>& fingerprint, ffi::Buffer<T>& values) {
  if (offsets.element_count() != e->n + 1 || columns.element_count() != e->nnz ||
      values.element_count() != e->nnz || fingerprint.element_count() != 2)
    return ffi::Error::InvalidArgument("cuDSS token CSR structure size mismatch");
  uint32_t fp[2];
  CUDA_CHECK(cudaMemcpyAsync(fp, fingerprint.typed_data(), sizeof(fp),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (fp[0] != e->fp[0] || fp[1] != e->fp[1])
    return ffi::Error::InvalidArgument("cuDSS CSR structure fingerprint mismatch; analyze the new pattern");
  CUDSS_CHECK(cudssMatrixSetCsrPointers(e->A, offsets.typed_data(), nullptr,
                                       columns.typed_data(), values.typed_data()));
  return ffi::Error::Success();
}

static ffi::Error execute(FactorEntry* e, cudaStream_t stream, cudssPhase_t phase) {
  const auto status = cudssExecute(e->handle, phase, e->config, e->data,
                                   e->A, e->x_dummy, e->b_dummy);
  // Failed phases may enqueue work and outlive this lease through waiting readers.
  const auto completion = cudaEventRecord(e->done, stream);
  if (status != CUDSS_STATUS_SUCCESS || completion != cudaSuccess)
    cudaStreamSynchronize(stream);
  // Keep the provider error when completion recording also fails.
  CUDSS_CHECK(status);
  CUDA_CHECK(completion);
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
  const int64_t n = offsets.element_count() - 1, nnz = columns.element_count();
  if (n < 1 || batch < 1 || n % batch != 0 || nnz % batch != 0 ||
      values.element_count() != nnz || fingerprint.element_count() != 2)
    return ffi::Error::InvalidArgument("cuDSS invalid expanded block CSR dimensions or fingerprint");
  auto e = std::make_shared<FactorEntry>();
  e->device = device;
  e->n = n; e->nnz = nnz; e->batch = batch;
  e->dtype = cudss_dtype<T>(); e->options = options;
  CUDA_CHECK(cudaEventCreateWithFlags(&e->done, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(e->done, stream));
  CUDA_CHECK(cudaMemcpyAsync(e->fp, fingerprint.typed_data(), sizeof(e->fp),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDSS_CHECK(cudssCreate(&e->handle));
  CUDSS_CHECK(cudssSetStream(e->handle, stream));
  CUDSS_CHECK(cudssConfigCreate(&e->config));
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
  *output = std::move(e);
  return ffi::Error::Success();
}

#define OPTIONS_ARGUMENTS const int64_t partition, const int64_t capacity, \
  const int64_t mtype, const int64_t mview, const int64_t reordering, const int64_t memory
#define OPTIONS_VALUE Options{partition, capacity, mtype, mview, reordering, memory}

template <ffi::DataType T>
static ffi::Error analyze(cudaStream_t stream, ffi::Buffer<T> values,
    ffi::Buffer<ffi::S32> offsets, ffi::Buffer<ffi::S32> columns,
    ffi::Buffer<ffi::U32> fingerprint, ffi::ResultBuffer<ffi::S32> token,
    OPTIONS_ARGUMENTS, const int64_t batch) {
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  std::shared_ptr<FactorEntry> e;
  if (auto err = create_entry(stream, values, offsets, columns, fingerprint,
                              batch, device, OPTIONS_VALUE, &e); err.failure()) return err;
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
    if (auto err = repoint(e.get(), stream, offsets, columns, fingerprint, values); err.failure()) return err;
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
    registry().insert(lease->device, options.partition, options.capacity, e);
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
  const int64_t nrhs = rhs.element_count() / e->n;
  DensePair dense;
  CUDSS_CHECK(cudssMatrixCreateDn(&dense.b, e->n, nrhs, e->n, rhs.typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&dense.x, e->n, nrhs, e->n, output->typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR));
  const auto status = cudssExecute(e->handle, CUDSS_PHASE_SOLVE,
                                   e->config, e->data, e->A, dense.x, dense.b);
  CUDA_CHECK(cudaEventRecord(e->done, stream));
  CUDSS_CHECK(status);
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
                           ffi::ResultBuffer<ffi::S32> output, int64_t partition) {
  std::vector<int32_t> ids(tokens.element_count());
  if (!ids.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(ids.data(), tokens.typed_data(), ids.size() * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  const auto released = static_cast<int32_t>(
      registry().release(partition, ids.data(), ids.size()));
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
  .Attr<int64_t>("reordering").Attr<int64_t>("memory")
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
    .Ret<ffi::Buffer<ffi::S32>>() COMMON_ATTRIBUTES .Attr<int64_t>("batch")); \
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
    .Arg<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S32>>().Attr<int64_t>("partition"));
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
