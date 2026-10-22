// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Site-GPU-aware MPI transports only coordinates named by the owned CSR pattern.
#include "sparse_action.hpp"
#include "../common/collective_lease.hpp"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ffi = xla::ffi;
namespace {

void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void mpi_check(int status) {
  if (status != MPI_SUCCESS) throw std::runtime_error("sparse coordinate MPI operation failed");
}

// No rank enters the next communication phase after another rank's local error.
// The lease's supervisor bounds failures inside MPI, including process loss.
template <class Function>
void phase(MPI_Comm comm, const char* name, Function&& function) {
  std::string message;
  try { function(); }
  catch (const std::exception& error) { message = error.what(); }
  int local = message.empty() ? 0 : 1, failed = 0;
  mpi_check(MPI_Allreduce(&local, &failed, 1, MPI_INT, MPI_MAX, comm));
  if (failed) throw std::runtime_error(std::string(name) + ": " +
      (message.empty() ? "failed on a peer" : message));
}

class StreamCompletion final {
 public:
  explicit StreamCompletion(cudaStream_t stream) : stream_(stream) {}
  ~StreamCompletion() { if (!completed_) cudaStreamSynchronize(stream_); }
  void finish() { cuda_check(cudaStreamSynchronize(stream_)); completed_ = true; }
 private:
  cudaStream_t stream_;
  bool completed_ = false;
};

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  void allocate(size_t count) {
    // A valid buffer for zero-count MPI calls avoids vendor-specific null handling.
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), std::max(size_t{1}, count) * sizeof(T)));
  }
  T* data() const { return data_; }
 private:
  T* data_ = nullptr;
};

template <class T>
__global__ void gather(const T* input, const int32_t* positions, T* output, int count, int rhs_count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = input[static_cast<int64_t>(positions[i / rhs_count]) * rhs_count + i % rhs_count];
}

template <class T>
__global__ void scatter_add(const T* input, const int32_t* positions, T* output, int count, int rhs_count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) atomicAdd(output + static_cast<int64_t>(positions[i / rhs_count]) * rhs_count + i % rhs_count, input[i]);
}

int blocks(int count) { return static_cast<int>((static_cast<int64_t>(count) + 255) / 256); }

struct Pattern {
  std::vector<int32_t> columns, pointers, requests, incoming, nnz_positions;
  std::vector<int> request_counts, request_offsets, incoming_counts, incoming_offsets;
  int incoming_size = 0;
};

int displacements(const std::vector<int>& counts, std::vector<int>* offsets) {
  int64_t size = 0;
  offsets->resize(counts.size());
  for (size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] < 0 || size + counts[i] > INT_MAX)
      throw std::invalid_argument("coordinate request capacity exceeds MPI int32 counts");
    (*offsets)[i] = static_cast<int>(size);
    size += counts[i];
  }
  return static_cast<int>(size);
}

void prepare_local(cudaStream_t stream, ffi::BufferR1<ffi::S32> indices,
                   ffi::BufferR1<ffi::S32> indptr, ffi::Span<const int64_t> starts,
                   int64_t partition, int ranks, int rank, int64_t input_size,
                   int64_t output_size, bool transpose, Pattern* pattern) {
  if (starts.size() != static_cast<size_t>(ranks + 1) || partition != rank ||
      starts[0] != 0 || starts[ranks] > INT_MAX)
    throw std::invalid_argument("CSR row ownership must match the ordered lease communicator");
  for (int i = 0; i < ranks; ++i)
    if (starts[i + 1] <= starts[i])
      throw std::invalid_argument("CSR owned ranges must form an exact positive partition");
  const int64_t rows = starts[rank + 1] - starts[rank];
  const int64_t nnz = indices.element_count();
  if (nnz > INT_MAX || indptr.element_count() != rows + 1 ||
      input_size != (transpose ? nnz : rows) || output_size != (transpose ? rows : nnz))
    throw std::invalid_argument("coordinate exchange buffers must be unpadded rank-local CSR shapes");
  pattern->columns.resize(nnz);
  pattern->pointers.resize(rows + 1);
  if (nnz) cuda_check(cudaMemcpyAsync(pattern->columns.data(), indices.typed_data(),
      nnz * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaMemcpyAsync(pattern->pointers.data(), indptr.typed_data(),
      (rows + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  if (pattern->pointers[0] != 0 || pattern->pointers.back() != nnz)
    throw std::invalid_argument("CSR indptr must begin at zero and end at local nnz");
  for (int64_t row = 0; row < rows; ++row) {
    const int32_t begin = pattern->pointers[row], end = pattern->pointers[row + 1];
    if (begin < 0 || end < begin || end > nnz)
      throw std::invalid_argument("CSR indptr is not monotone");
    for (int32_t i = begin; i < end; ++i)
      if (pattern->columns[i] < 0 || pattern->columns[i] >= starts[ranks] ||
          (i != begin && pattern->columns[i] <= pattern->columns[i - 1]))
        throw std::invalid_argument("CSR columns must be sorted unique global int32 coordinates");
  }
  pattern->requests = pattern->columns;
  std::sort(pattern->requests.begin(), pattern->requests.end());
  pattern->requests.erase(std::unique(pattern->requests.begin(), pattern->requests.end()), pattern->requests.end());
  pattern->request_counts.assign(ranks, 0);
  for (int32_t column : pattern->requests) {
    const int owner = std::upper_bound(starts.begin(), starts.end(), column) - starts.begin() - 1;
    ++pattern->request_counts[owner];
  }
  displacements(pattern->request_counts, &pattern->request_offsets);
  pattern->incoming_counts.resize(ranks);
  pattern->nnz_positions.resize(nnz);
  for (int64_t i = 0; i < nnz; ++i)
    pattern->nnz_positions[i] = std::lower_bound(pattern->requests.begin(), pattern->requests.end(),
        pattern->columns[i]) - pattern->requests.begin();
}

// Layout/structure identities must agree independently of rank-local capacities.
void agree_pattern(MPI_Comm comm, ffi::Span<const int64_t> starts, std::string_view identity,
                   int64_t rhs_count, int64_t array_rank) {
  uint64_t hash = 14695981039346656037ULL;
  for (int64_t start : starts)
    for (int shift = 0; shift < 64; shift += 8) {
      hash ^= static_cast<uint64_t>(start) >> shift & 255U;
      hash *= 1099511628211ULL;
    }
  for (int64_t extent : {rhs_count, array_rank})
    for (int shift = 0; shift < 64; shift += 8) {
      hash ^= static_cast<uint64_t>(extent) >> shift & 255U;
      hash *= 1099511628211ULL;
    }
  for (unsigned char byte : identity) { hash ^= byte; hash *= 1099511628211ULL; }
  uint64_t low = 0, high = 0;
  mpi_check(MPI_Allreduce(&hash, &low, 1, MPI_UINT64_T, MPI_MIN, comm));
  mpi_check(MPI_Allreduce(&hash, &high, 1, MPI_UINT64_T, MPI_MAX, comm));
  if (low != high) throw std::invalid_argument("CSR global structure, row ownership, or true RHS shape differs across ranks");
}

template <ffi::DataType Type>
ffi::Error exchange(cudaStream_t stream, ffi::Token,
                    ffi::BufferR0<ffi::S64> lease, ffi::BufferR0<ffi::S64> ticket,
                    ffi::BufferR1<ffi::S32> indices, ffi::BufferR1<ffi::S32> indptr,
                    ffi::Buffer<Type> input, ffi::Result<ffi::Token>,
                    ffi::Result<ffi::Buffer<Type>> output, ffi::Span<const int64_t> starts,
                    int64_t partition, std::string_view structure_id, int64_t transpose,
                    std::string_view configuration_id, std::string_view call_id) {
  using T = ffi::NativeType<Type>;
  try {
    int64_t lease_id = 0, invocation = 0;
    StreamCompletion scalar_completion(stream);
    cuda_check(cudaMemcpyAsync(&lease_id, lease.typed_data(), sizeof(lease_id), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(&invocation, ticket.typed_data(), sizeof(invocation), cudaMemcpyDeviceToHost, stream));
    scalar_completion.finish();
    phx::collective::CallGuard guard(lease_id, configuration_id, call_id, invocation);
    StreamCompletion completion(stream);
    MPI_Comm comm = guard.communicator();
    int rank = -1, ranks = 0;
    mpi_check(MPI_Comm_rank(comm, &rank));
    mpi_check(MPI_Comm_size(comm, &ranks));
    Pattern pattern;
    int rhs_count = 1;
    phase(comm, "CSR coordinate preparation", [&] {
      if ((transpose != 0 && transpose != 1) || structure_id.empty())
        throw std::invalid_argument("coordinate exchange requires explicit operation and structure identities");
      const auto input_shape = input.dimensions(), output_shape = output->dimensions();
      if (input_shape.size() < 1 || input_shape.size() > 2 || output_shape.size() != input_shape.size() ||
          input.element_count() > INT_MAX || output->element_count() > INT_MAX ||
          (input_shape.size() == 2 && (input_shape[1] <= 0 || input_shape[1] > INT_MAX ||
                                      input_shape[1] != output_shape[1])))
        throw std::invalid_argument("coordinate exchange requires matching vector or true RHS-block buffers");
      rhs_count = input_shape.size() == 2 ? static_cast<int>(input_shape[1]) : 1;
      prepare_local(stream, indices, indptr, starts, partition, ranks, rank,
                    input_shape[0], output_shape[0], transpose != 0, &pattern);
    });
    agree_pattern(comm, starts, structure_id, rhs_count, input.dimensions().size());
    mpi_check(MPI_Alltoall(pattern.request_counts.data(), 1, MPI_INT,
                          pattern.incoming_counts.data(), 1, MPI_INT, comm));
    phase(comm, "CSR request allocation", [&] {
      pattern.incoming_size = displacements(pattern.incoming_counts, &pattern.incoming_offsets);
      pattern.incoming.resize(pattern.incoming_size);
    });
    mpi_check(MPI_Alltoallv(pattern.requests.data(), pattern.request_counts.data(),
        pattern.request_offsets.data(), MPI_INT32_T, pattern.incoming.data(),
        pattern.incoming_counts.data(), pattern.incoming_offsets.data(), MPI_INT32_T, comm));
    DeviceBuffer<int32_t> nnz_positions, owned_positions;
    DeviceBuffer<T> requested_values, incoming_values;
    const int nnz = static_cast<int>(indices.element_count());
    const int requested = static_cast<int>(pattern.requests.size());
    phase(comm, "CSR GPU packing", [&] {
      if (static_cast<int64_t>(pattern.incoming_size) * rhs_count > INT_MAX ||
          static_cast<int64_t>(requested) * rhs_count > INT_MAX)
        throw std::invalid_argument("true RHS coordinate messages exceed MPI int32 capacity");
      for (size_t peer = 0; peer < pattern.request_counts.size(); ++peer) {
        pattern.request_counts[peer] *= rhs_count;
        pattern.request_offsets[peer] *= rhs_count;
        pattern.incoming_counts[peer] *= rhs_count;
        pattern.incoming_offsets[peer] *= rhs_count;
      }
      for (auto& column : pattern.incoming) {
        if (column < starts[rank] || column >= starts[rank + 1])
          throw std::invalid_argument("peer requested a coordinate outside this rank's owned interval");
        column -= starts[rank];
      }
      nnz_positions.allocate(nnz);
      owned_positions.allocate(pattern.incoming_size);
      requested_values.allocate(static_cast<size_t>(requested) * rhs_count);
      incoming_values.allocate(static_cast<size_t>(pattern.incoming_size) * rhs_count);
      if (nnz) cuda_check(cudaMemcpyAsync(nnz_positions.data(), pattern.nnz_positions.data(),
          nnz * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
      if (pattern.incoming_size) cuda_check(cudaMemcpyAsync(owned_positions.data(), pattern.incoming.data(),
          pattern.incoming_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
      if (transpose) {
        cuda_check(cudaMemsetAsync(requested_values.data(), 0, static_cast<size_t>(requested) * rhs_count * sizeof(T), stream));
        if (nnz) scatter_add<<<blocks(nnz * rhs_count), 256, 0, stream>>>(input.typed_data(),
            nnz_positions.data(), requested_values.data(), nnz * rhs_count, rhs_count);
      } else if (pattern.incoming_size) {
        gather<<<blocks(pattern.incoming_size * rhs_count), 256, 0, stream>>>(input.typed_data(),
            owned_positions.data(), incoming_values.data(), pattern.incoming_size * rhs_count, rhs_count);
      }
      cuda_check(cudaGetLastError());
      cuda_check(cudaStreamSynchronize(stream));
    });
    const MPI_Datatype datatype = Type == ffi::F32 ? MPI_FLOAT : MPI_DOUBLE;
    if (transpose) {
      mpi_check(MPI_Alltoallv(requested_values.data(), pattern.request_counts.data(),
          pattern.request_offsets.data(), datatype, incoming_values.data(),
          pattern.incoming_counts.data(), pattern.incoming_offsets.data(), datatype, comm));
    } else {
      mpi_check(MPI_Alltoallv(incoming_values.data(), pattern.incoming_counts.data(),
          pattern.incoming_offsets.data(), datatype, requested_values.data(),
          pattern.request_counts.data(), pattern.request_offsets.data(), datatype, comm));
    }
    phase(comm, "CSR GPU unpacking", [&] {
      if (transpose) {
        cuda_check(cudaMemsetAsync(output->typed_data(), 0, output->element_count() * sizeof(T), stream));
        if (pattern.incoming_size) scatter_add<<<blocks(pattern.incoming_size * rhs_count), 256, 0, stream>>>(
            incoming_values.data(), owned_positions.data(), output->typed_data(), pattern.incoming_size * rhs_count, rhs_count);
      } else if (nnz) {
        gather<<<blocks(nnz * rhs_count), 256, 0, stream>>>(requested_values.data(),
            nnz_positions.data(), output->typed_data(), nnz * rhs_count, rhs_count);
      }
      cuda_check(cudaGetLastError());
      completion.finish();
    });
    guard.complete();
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(std::string("distributed CSR coordinate exchange: ") + error.what());
  }
}

template <ffi::DataType Type>
ffi::Error reduce(cudaStream_t stream, ffi::Token,
                  ffi::BufferR0<ffi::S64> lease, ffi::BufferR0<ffi::S64> ticket,
                  ffi::Buffer<Type> input, ffi::Result<ffi::Token>,
                  ffi::Result<ffi::Buffer<Type>> output, int64_t operation,
                  std::string_view configuration_id, std::string_view call_id) {
  try {
    int64_t lease_id = 0, invocation = 0;
    StreamCompletion scalar_completion(stream);
    cuda_check(cudaMemcpyAsync(&lease_id, lease.typed_data(), sizeof(lease_id), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(&invocation, ticket.typed_data(), sizeof(invocation), cudaMemcpyDeviceToHost, stream));
    scalar_completion.finish();
    phx::collective::CallGuard guard(lease_id, configuration_id, call_id, invocation);
    StreamCompletion completion(stream);
    MPI_Comm comm = guard.communicator();
    phase(comm, "distributed reduction admission", [&] {
      if ((operation != 0 && operation != 1) || input.dimensions().size() > 1 ||
          input.element_count() > INT_MAX || input.element_count() != output->element_count() ||
          input.dimensions().size() != output->dimensions().size())
        throw std::invalid_argument("distributed reduction requires matching scalar or RHS-vector buffers");
      cuda_check(cudaStreamSynchronize(stream));
    });
    int64_t metadata[3] = {static_cast<int64_t>(input.element_count()),
                          static_cast<int64_t>(input.dimensions().size()), operation};
    int64_t low[3] = {}, high[3] = {};
    mpi_check(MPI_Allreduce(metadata, low, 3, MPI_INT64_T, MPI_MIN, comm));
    mpi_check(MPI_Allreduce(metadata, high, 3, MPI_INT64_T, MPI_MAX, comm));
    if (!std::equal(low, low + 3, high))
      throw std::invalid_argument("distributed reduction shape or operation differs across ranks");
    const MPI_Datatype datatype = Type == ffi::F32 ? MPI_FLOAT :
        (Type == ffi::F64 ? MPI_DOUBLE : MPI_INT32_T);
    mpi_check(MPI_Allreduce(input.typed_data(), output->typed_data(),
        static_cast<int>(input.element_count()), datatype,
        operation == 0 ? MPI_SUM : MPI_MAX, comm));
    phase(comm, "distributed reduction completion", [&] { completion.finish(); });
    guard.complete();
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(std::string("distributed scalar reduction: ") + error.what());
  }
}
}  // namespace

#define PHX_CSR_EXCHANGE(Suffix, Type) \
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_exchange_##Suffix(XLA_FFI_CallFrame*); \
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_exchange_##Suffix, exchange<Type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() \
      .Arg<ffi::Token>().Arg<ffi::BufferR0<ffi::S64>>().Arg<ffi::BufferR0<ffi::S64>>() \
      .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::Buffer<Type>>() \
      .Ret<ffi::Token>().Ret<ffi::Buffer<Type>>() \
      .Attr<ffi::Span<const int64_t>>("row_starts").Attr<int64_t>("partition_index") \
      .Attr<std::string_view>("structure_id").Attr<int64_t>("transpose") \
      .Attr<std::string_view>("collective_configuration_id").Attr<std::string_view>("collective_call_id"))
PHX_CSR_EXCHANGE(f32, ffi::F32);
PHX_CSR_EXCHANGE(f64, ffi::F64);

#define PHX_CSR_REDUCE(Suffix, Type) \
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_reduce_##Suffix(XLA_FFI_CallFrame*); \
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudss_mgmn_reduce_##Suffix, reduce<Type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() \
      .Arg<ffi::Token>().Arg<ffi::BufferR0<ffi::S64>>().Arg<ffi::BufferR0<ffi::S64>>() \
      .Arg<ffi::Buffer<Type>>().Ret<ffi::Token>().Ret<ffi::Buffer<Type>>() \
      .Attr<int64_t>("operation") \
      .Attr<std::string_view>("collective_configuration_id").Attr<std::string_view>("collective_call_id"))
PHX_CSR_REDUCE(f32, ffi::F32);
PHX_CSR_REDUCE(f64, ffi::F64);
PHX_CSR_REDUCE(i32, ffi::S32);
