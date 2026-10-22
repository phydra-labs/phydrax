#include "phx_cuda_common.hpp"
#include "collective_lease.hpp"
#include <mpi-ext.h>
#include <cudecomp.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <string_view>
#include <utility>

static_assert(CUDECOMP_MAJOR == 0 && CUDECOMP_MINOR == 7 && CUDECOMP_PATCH == 0,
              "The direct3D slab provider requires cuDecomp 0.7.0");
#if !defined(OMPI_MAJOR_VERSION) || OMPI_MAJOR_VERSION != 5 || !OMPI_HAVE_MPI_EXT_CUDA
#error "The qualified cuDecomp slab profile requires the site Open MPI 5 CUDA extension"
#endif

namespace ffi = xla::ffi;
namespace {
class MpiFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void check_mpi(int status, const char* operation) {
  if (status != MPI_SUCCESS)
    throw MpiFailure(std::string(operation) + " failed; the collective lease supervisor owns rank-loss termination");
}
void check_vendor(cudecompResult_t status, const char* operation) {
  if (status == CUDECOMP_RESULT_MPI_ERROR)
    throw MpiFailure(std::string(operation) + " reported MPI failure; no further status collective is permitted");
  if (status != CUDECOMP_RESULT_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with cuDecomp status " + std::to_string(status));
}
// Agree local failures before entering the next vendor/MPI collective phase.
// These collectives run only under CallGuard's bounded process-owner supervisor.
template <class Function>
void phase(MPI_Comm comm, const char* name, Function&& function) {
  std::string message;
  try { function(); }
  catch (const MpiFailure&) { throw; }
  catch (const std::exception& error) { message = error.what(); }
  catch (...) { message = "unknown local failure"; }
  int local = !message.empty(), any = 0;
  check_mpi(MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, comm), name);
  if (any)
    throw std::runtime_error(std::string(name) + ": " + (message.empty() ? "another rank failed" : message));
}
void check_environment() {
  for (const char* name : {"CUDECOMP_ENABLE_CUMEM", "CUDECOMP_ENABLE_NCCL_UBR",
                           "CUDECOMP_ENABLE_CUDA_GRAPHS", "CUDECOMP_ENABLE_PERFORMANCE_REPORT"}) {
    const char* value = std::getenv(name);
    if (value && std::strcmp(value, "0") != 0)
      throw std::invalid_argument(std::string(name) + " must be unset or exactly 0 for the cuDecomp slab provider");
  }
  Dl_info library{};
  if (!dladdr(reinterpret_cast<void*>(&cudecompInit), &library) || !library.dli_fname ||
      std::filesystem::canonical(library.dli_fname) != std::filesystem::canonical(PHX_CUDECOMP_LIBRARY))
    throw std::runtime_error("loaded cuDecomp differs from the explicitly supplied site library");
}
int64_t multiply(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
    throw std::invalid_argument("cuDecomp shape or workspace byte count overflows int64");
  return a * b;
}
class DeviceSelection final {
 public:
  void select(int device) {
    previous_ = phx::cuda::current_device();
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
  }
  void restore() {
    if (previous_ >= 0) {
      check_cuda(cudaSetDevice(previous_), "restore cuDecomp host CUDA device");
      previous_ = -1;
    }
  }
  ~DeviceSelection() { if (previous_ >= 0) cudaSetDevice(previous_); }
 private:
  int previous_ = -1;
};
struct Preparation {
  int32_t device = -1, precision_bits = 0;
  std::array<int64_t, 3> shape{};
  int64_t maximum_scratch_bytes = 0, workspace_bytes = 0;
  cudecompHandle_t handle = nullptr;
  cudecompGridDesc_t descriptor = nullptr;
  void* workspace = nullptr;
  cudecompPencilInfo_t y{}, x{};
  bool ready = false;
  // No destructor invokes a collective. The owning lease retires this recipe.
  void retire(MPI_Comm comm) {
    DeviceSelection selection;
    phase(comm, "cuDecomp retirement device", [&] { selection.select(device); });
    phase(comm, "cuDecomp retirement synchronization", [&] { check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); });
    phase(comm, "cuDecomp workspace retirement", [&] {
      if (workspace) { check_vendor(cudecompFree(handle, descriptor, workspace), "cudecompFree"); workspace = nullptr; }
    });
    phase(comm, "cuDecomp descriptor retirement", [&] {
      if (descriptor) { check_vendor(cudecompGridDescDestroy(handle, descriptor), "cudecompGridDescDestroy"); descriptor = nullptr; }
    });
    phase(comm, "cuDecomp handle retirement", [&] {
      if (handle) { check_vendor(cudecompFinalize(handle), "cudecompFinalize"); handle = nullptr; }
    });
    phase(comm, "cuDecomp retirement device restoration", [&] { selection.restore(); });
    ready = false;
  }
};
using Key = std::pair<int64_t, std::string>;
struct KeyLess {
  using is_transparent = void;
  template <class Left, class Right>
  bool operator()(const std::pair<int64_t, Left>& a, const std::pair<int64_t, Right>& b) const {
    return a.first < b.first ||
        (a.first == b.first && std::string_view(a.second) < std::string_view(b.second));
  }
};
class GroupRegistry final : public phx::cuda::RegistryBase {
 public:
  std::mutex mutex;
  std::map<Key, std::shared_ptr<Preparation>, KeyLess> entries;
  int64_t release(int64_t lease, const int32_t*, int64_t) override {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& item : entries)
      if (item.first.first == lease)
        throw std::runtime_error("cuDecomp group resources require collective lease retirement, not local release");
    return 0;
  }
  int64_t release_partition(int64_t lease) override {
    return release(lease, nullptr, 0);
  }
  int64_t release_all() override {
    std::lock_guard<std::mutex> lock(mutex);
    if (!entries.empty()) throw std::runtime_error("retire live cuDecomp collective leases before release_all or shutdown");
    return 0;
  }
  std::vector<phx::cuda::PartitionStats> stats() override {
    std::lock_guard<std::mutex> lock(mutex);
    std::map<std::pair<int32_t, int64_t>, int64_t> counts;
    for (const auto& item : entries) ++counts[{item.second->device, item.first.first}];
    std::vector<phx::cuda::PartitionStats> result;
    result.reserve(counts.size());
    for (const auto& item : counts)
      result.push_back({item.first.first, item.first.second, item.second, item.second, 0});
    return result;
  }
};
GroupRegistry& registry() { static auto* value = new GroupRegistry; return *value; }
std::shared_ptr<Preparation> find(int64_t lease_id, std::string_view preparation_id) {
  auto& owner = registry();
  std::lock_guard<std::mutex> lock(owner.mutex);
  auto found = owner.entries.find(std::pair<int64_t, std::string_view>{lease_id, preparation_id});
  return found == owner.entries.end() ? nullptr : found->second;
}
void check_one_gpu_per_rank(MPI_Comm comm, const cudaUUID_t& uuid) {
  MPI_Comm local = MPI_COMM_NULL;
  check_mpi(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local), "MPI_Comm_split_type");
  // Allocation and CUDA queries have already succeeded on every rank before
  // entering this node-local collective phase.
  int count = 0;
  check_mpi(MPI_Comm_size(local, &count), "MPI_Comm_size");
  std::vector<cudaUUID_t> devices;
  phase(comm, "cuDecomp GPU membership storage", [&] { devices.resize(count); });
  check_mpi(MPI_Allgather(&uuid, sizeof(uuid), MPI_BYTE, devices.data(), sizeof(uuid), MPI_BYTE, local), "GPU UUID agreement");
  bool duplicate = false;
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < i; ++j)
      duplicate = duplicate || std::memcmp(&devices[i], &devices[j], sizeof(uuid)) == 0;
  check_mpi(MPI_Comm_free(&local), "MPI_Comm_free GPU membership");
  phase(comm, "cuDecomp one GPU per rank", [&] {
    if (duplicate) throw std::invalid_argument("cuDecomp requires one distinct CUDA device per rank on each node");
  });
}
void check_pencil(const cudecompPencilInfo_t& pencil, const std::array<int64_t, 3>& shape,
                  int ranks, int rank, bool y) {
  const std::array<int64_t, 3> expected = {shape[2], y ? shape[1] : shape[1] / ranks,
                                        y ? shape[0] / ranks : shape[0]};
  const std::array<int64_t, 3> lo = {0, y ? 0 : rank * (shape[1] / ranks),
                                   y ? rank * (shape[0] / ranks) : 0};
  int64_t size = 1;
  for (int i = 0; i < 3; ++i) {
    if (pencil.order[i] != 2 - i || pencil.shape[i] != expected[i] ||
        pencil.lo[i] != lo[i] || pencil.hi[i] != lo[i] + expected[i] - 1 ||
        pencil.halo_extents[i] != 0 || pencil.padding[i] != 0)
      throw std::runtime_error("cuDecomp pencil metadata does not match the exact direct3D slab mapping");
    size = multiply(size, expected[i]);
  }
  if (pencil.size != size) throw std::runtime_error("cuDecomp pencil size disagrees with its exact shape");
}
void metadata(const Preparation& preparation, int64_t* output) {
  output[0] = preparation.workspace_bytes;
  int index = 1;
  for (const auto* pencil : {&preparation.y, &preparation.x})
    for (const auto* field : {pencil->shape, pencil->lo, pencil->hi, pencil->order})
      for (int i = 0; i < 3; ++i) output[index++] = field[i];
}
void prepare(int64_t lease_id, int64_t ticket, const char* configuration_id, const char* call_id,
             const char* preparation_id,
             const int64_t* shape, int32_t precision_bits, int32_t device_ordinal,
             int64_t maximum_scratch_bytes, int64_t* output) {
  phx::collective::CallGuard guard(lease_id, configuration_id ? configuration_id : "",
      call_id ? call_id : "", ticket);
  const MPI_Comm comm = guard.communicator();
  DeviceSelection selection;
  int rank = 0, ranks = 0;
  check_mpi(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");
  check_mpi(MPI_Comm_size(comm, &ranks), "MPI_Comm_size");
  int device = -1;
  cudaDeviceProp properties{};
  std::array<int64_t, 5> recipe{};
  phase(comm, "cuDecomp preparation validation", [&] {
    if (!configuration_id || !*configuration_id || !call_id || !*call_id ||
        !preparation_id || !*preparation_id || !shape || !output)
      throw std::invalid_argument("cuDecomp preparation requires configuration, call, preparation, shape and metadata outputs");
    if (precision_bits != 64 && precision_bits != 128)
      throw std::invalid_argument("cuDecomp supports only complex64 and complex128");
    if (maximum_scratch_bytes < 0) throw std::invalid_argument("cuDecomp scratch budget must be nonnegative");
    check_environment();
    // The shared lease's require_initialized() already admitted this site MPI
    // through MPIX_Query_cuda_support() before creating the guarded communicator.
    int64_t size = 1;
    for (int i = 0; i < 3; ++i) {
      if (shape[i] <= 0 || shape[i] > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("cuDecomp global dimensions must be positive int32 values");
      size = multiply(size, shape[i]);
      recipe[i] = shape[i];
    }
    multiply(size, precision_bits / 8);
    if (shape[0] % ranks || shape[1] % ranks)
      throw std::invalid_argument("cuDecomp exact slabs require nx and ny divisible by the execution group size");
    recipe[3] = precision_bits;
    recipe[4] = maximum_scratch_bytes;
    if (device_ordinal < 0) throw std::invalid_argument("cuDecomp preparation requires an explicit local JAX device ordinal");
    selection.select(device_ordinal);
    device = device_ordinal;
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  });
  std::array<int64_t, 5> minimum{}, maximum{};
  check_mpi(MPI_Allreduce(recipe.data(), minimum.data(), recipe.size(), MPI_INT64_T, MPI_MIN, comm), "cuDecomp recipe minimum");
  check_mpi(MPI_Allreduce(recipe.data(), maximum.data(), recipe.size(), MPI_INT64_T, MPI_MAX, comm), "cuDecomp recipe maximum");
  phase(comm, "cuDecomp recipe agreement", [&] {
    if (minimum != maximum) throw std::invalid_argument("cuDecomp preparation recipes disagree across ranks");
  });
  check_one_gpu_per_rank(comm, properties.uuid);
  const Key key{lease_id, preparation_id};
  auto preparation = find(key.first, key.second);
  int existing = preparation ? 1 : 0, all_existing = 0, any_existing = 0;
  check_mpi(MPI_Allreduce(&existing, &all_existing, 1, MPI_INT, MPI_MIN, comm), "cuDecomp existing recipe minimum");
  check_mpi(MPI_Allreduce(&existing, &any_existing, 1, MPI_INT, MPI_MAX, comm), "cuDecomp existing recipe maximum");
  phase(comm, "cuDecomp existing preparation agreement", [&] {
    if (all_existing != any_existing) throw std::runtime_error("cuDecomp preparation ownership disagrees across ranks");
    if (preparation && (!preparation->ready || preparation->device != device ||
        preparation->precision_bits != precision_bits || preparation->maximum_scratch_bytes != maximum_scratch_bytes ||
        !std::equal(preparation->shape.begin(), preparation->shape.end(), shape)))
      throw std::invalid_argument("cuDecomp preparation identity was reused for a different or incomplete recipe");
  });
  if (!preparation) {
    phase(comm, "cuDecomp resource ownership", [&] {
      preparation = std::make_shared<Preparation>();
      preparation->device = device;
      preparation->precision_bits = precision_bits;
      preparation->maximum_scratch_bytes = maximum_scratch_bytes;
      std::copy_n(shape, 3, preparation->shape.begin());
      guard.on_retire("cudecomp:" + key.second, [key, preparation](MPI_Comm retire_comm) {
        preparation->retire(retire_comm);
        auto& owner = registry();
        std::lock_guard<std::mutex> lock(owner.mutex);
        owner.entries.erase(key);
      });
      auto& owner = registry();
      std::lock_guard<std::mutex> lock(owner.mutex);
      if (!owner.entries.emplace(key, preparation).second) throw std::runtime_error("duplicate cuDecomp group preparation");
    });
    phase(comm, "cuDecomp initialization", [&] { check_vendor(cudecompInit(&preparation->handle, comm), "cudecompInit"); });
    cudecompGridDescConfig_t config{};
    phase(comm, "cuDecomp descriptor configuration", [&] {
      check_vendor(cudecompGridDescConfigSetDefaults(&config), "cudecompGridDescConfigSetDefaults");
      for (int i = 0; i < 3; ++i) {
        config.gdims[i] = static_cast<int32_t>(shape[i]);
        for (int j = 0; j < 3; ++j) config.transpose_mem_order[i][j] = 2 - j;
      }
      config.pdims[0] = ranks;
      config.pdims[1] = 1;
      config.rank_order = CUDECOMP_RANK_ORDER_ROW_MAJOR;
      config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_A2A;
      config.halo_comm_backend = CUDECOMP_HALO_COMM_MPI;
    });
    phase(comm, "cuDecomp descriptor creation", [&] {
      check_vendor(cudecompGridDescCreate(preparation->handle, &preparation->descriptor, &config, nullptr), "cudecompGridDescCreate");
    });
    phase(comm, "cuDecomp pencil and workspace queries", [&] {
      check_vendor(cudecompGetPencilInfo(preparation->handle, preparation->descriptor, &preparation->y, 1, nullptr, nullptr), "cudecompGetPencilInfo(Y)");
      check_vendor(cudecompGetPencilInfo(preparation->handle, preparation->descriptor, &preparation->x, 0, nullptr, nullptr), "cudecompGetPencilInfo(X)");
      check_pencil(preparation->y, preparation->shape, ranks, rank, true);
      check_pencil(preparation->x, preparation->shape, ranks, rank, false);
      int64_t elements = 0;
      check_vendor(cudecompGetTransposeWorkspaceSize(preparation->handle, preparation->descriptor, &elements), "cudecompGetTransposeWorkspaceSize");
      preparation->workspace_bytes = multiply(elements, precision_bits / 8);
      if (preparation->workspace_bytes <= 0 || preparation->workspace_bytes > maximum_scratch_bytes ||
          static_cast<uint64_t>(preparation->workspace_bytes) > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("cuDecomp scalar transpose workspace exceeds the external scratch budget or address space");
    });
    phase(comm, "cuDecomp explicit workspace allocation", [&] {
      // With MPI-only backends and CUMEM/UBR disabled, the pinned implementation
      // selects ordinary cudaMalloc. Never pass the null AUTO-workspace sentinel.
      check_vendor(cudecompMalloc(preparation->handle, preparation->descriptor, &preparation->workspace,
                                  static_cast<size_t>(preparation->workspace_bytes)), "cudecompMalloc");
      if (!preparation->workspace) throw std::runtime_error("cuDecomp explicit workspace is null");
      check_cuda(cudaDeviceSynchronize(), "cuDecomp preparation cudaDeviceSynchronize");
    });
    preparation->ready = true;
  }
  metadata(*preparation, output);
  phase(comm, "cuDecomp host device restoration", [&] { selection.restore(); });
  guard.complete();
}

// Constructed after CallGuard: even exception unwinding waits for this stream
// before the guard can release admission or trigger its failure supervisor.
class StreamCompletion final {
 public:
  explicit StreamCompletion(cudaStream_t stream) : stream_(stream) {}
  ~StreamCompletion() { if (!finished_) cudaStreamSynchronize(stream_); }
  void finish() {
    check_cuda(cudaStreamSynchronize(stream_), "cuDecomp cudaStreamSynchronize");
    finished_ = true;
  }
 private:
  cudaStream_t stream_;
  bool finished_ = false;
};
template <ffi::DataType Type>
ffi::Error transpose(cudaStream_t stream, ffi::Token, ffi::BufferR0<ffi::S64> lease,
                     ffi::BufferR0<ffi::S64> ticket, ffi::BufferR4<Type> input,
                     ffi::Result<ffi::Token>, ffi::ResultBufferR4<Type> output,
                     std::string_view preparation_id, int64_t direction,
                     std::string_view configuration_id, std::string_view call_id) {
  try {
    StreamCompletion scalar_completion(stream);
    int64_t lease_id = 0, invocation = 0;
    check_cuda(cudaMemcpyAsync(&lease_id, lease.typed_data(), sizeof(lease_id), cudaMemcpyDeviceToHost, stream), "copy collective lease");
    check_cuda(cudaMemcpyAsync(&invocation, ticket.typed_data(), sizeof(invocation), cudaMemcpyDeviceToHost, stream), "copy collective ticket");
    scalar_completion.finish();
    phx::collective::CallGuard guard(lease_id, configuration_id, call_id, invocation);
    StreamCompletion completion(stream);
    const MPI_Comm comm = guard.communicator();
    auto preparation = find(lease_id, preparation_id);
    int64_t lanes = 0;
    phase(comm, "cuDecomp execution validation", [&] {
      if (!preparation || !preparation->ready || !preparation->handle || !preparation->descriptor || !preparation->workspace)
        throw std::invalid_argument("cuDecomp requires an existing live group-owned preparation");
      if (direction != 0 && direction != 1) throw std::invalid_argument("cuDecomp direction must be 0 (YToX) or 1 (XToY)");
      constexpr int32_t bits = Type == ffi::C64 ? 64 : 128;
      if (preparation->precision_bits != bits || preparation->device != phx::cuda::current_device())
        throw std::invalid_argument("cuDecomp preparation precision or CUDA device mismatch");
      cudaPointerAttributes source_attributes{}, target_attributes{};
      check_cuda(cudaPointerGetAttributes(&source_attributes, input.typed_data()), "cuDecomp input device");
      check_cuda(cudaPointerGetAttributes(&target_attributes, output->typed_data()), "cuDecomp output device");
      if (source_attributes.type != cudaMemoryTypeDevice || target_attributes.type != cudaMemoryTypeDevice ||
          source_attributes.device != preparation->device || target_attributes.device != preparation->device)
        throw std::invalid_argument("cuDecomp input/output must reside on the prepared CUDA device");
      cudaStreamCaptureStatus capture;
      check_cuda(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
      if (capture != cudaStreamCaptureStatusNone) throw std::invalid_argument("cuDecomp slab execution does not support CUDA Graph capture");
      const auto& source = direction == 0 ? preparation->y : preparation->x;
      const auto& target = direction == 0 ? preparation->x : preparation->y;
      lanes = input.dimensions()[0];
      if (lanes <= 0 || output->dimensions()[0] != lanes)
        throw std::invalid_argument("cuDecomp q-front input/output lane counts must agree and be positive");
      for (int i = 0; i < 3; ++i)
        if (input.dimensions()[i + 1] != source.shape[2 - i] || output->dimensions()[i + 1] != target.shape[2 - i])
          throw std::invalid_argument("cuDecomp input/output must have the exact q-front direct3D slab shape");
      multiply(multiply(lanes, source.size), bits / 8);
      multiply(multiply(lanes, target.size), bits / 8);
    });
    const std::array<int64_t, 3> operation = {lanes, direction, Type == ffi::C64 ? 64 : 128};
    std::array<int64_t, 3> minimum{}, maximum{};
    check_mpi(MPI_Allreduce(operation.data(), minimum.data(), operation.size(), MPI_INT64_T, MPI_MIN, comm), "cuDecomp operation minimum");
    check_mpi(MPI_Allreduce(operation.data(), maximum.data(), operation.size(), MPI_INT64_T, MPI_MAX, comm), "cuDecomp operation maximum");
    phase(comm, "cuDecomp operation agreement", [&] {
      if (minimum != maximum)
        throw std::invalid_argument("cuDecomp dynamic q lane count, direction or precision disagrees across ranks");
    });
    const auto dtype = Type == ffi::C64 ? CUDECOMP_FLOAT_COMPLEX : CUDECOMP_DOUBLE_COMPLEX;
    const int64_t source_size = direction == 0 ? preparation->y.size : preparation->x.size;
    const int64_t target_size = direction == 0 ? preparation->x.size : preparation->y.size;
    for (int64_t q = 0; q < lanes; ++q) {
      phase(comm, "cuDecomp scalar transpose", [&] {
        auto operation = direction == 0 ? cudecompTransposeYToX : cudecompTransposeXToY;
        check_vendor(operation(preparation->handle, preparation->descriptor,
            input.typed_data() + q * source_size, output->typed_data() + q * target_size,
            preparation->workspace, dtype, nullptr, nullptr, nullptr, nullptr, stream), "cuDecomp slab transpose");
      });
    }
    completion.finish();
    guard.complete();
    return ffi::Error::Success();
  } catch (const std::exception& error) { return phx::cuda::ffi_error(error.what()); }
  catch (...) { return phx::cuda::ffi_error("Unknown cuDecomp transpose failure"); }
}
}  // namespace

phx::cuda::RegistryBase* component_registry() { return &registry(); }
int32_t component_shutdown() { registry().release_all(); return PHX_CUDA_SUCCESS; }
extern "C" PHX_CUDA_EXPORT const char* phx_cuda_cudecomp_build_identity() { return PHX_CUDECOMP_BUILD_IDENTITY; }
extern "C" PHX_CUDA_EXPORT int32_t phx_cuda_cudecomp_prepare(
    int64_t lease_id, int64_t ticket, const char* configuration_id, const char* call_id,
    const char* preparation_id,
    const int64_t* shape3, int32_t precision_bits, int32_t device_ordinal,
    int64_t maximum_scratch_bytes, int64_t* metadata25) {
  try {
    prepare(lease_id, ticket, configuration_id, call_id, preparation_id, shape3, precision_bits,
            device_ordinal, maximum_scratch_bytes, metadata25);
    return PHX_CUDA_SUCCESS;
  } catch (const std::invalid_argument& error) {
    phx::cuda::set_last_error(error.what()); return PHX_CUDA_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    phx::cuda::set_last_error(error.what()); return PHX_CUDA_PROVIDER_ERROR;
  } catch (...) {
    phx::cuda::set_last_error("Unknown cuDecomp preparation failure"); return PHX_CUDA_PROVIDER_ERROR;
  }
}

#define PHX_CUDECOMP_TRANSPOSE(Suffix, Type) \
  extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudecomp_transpose_##Suffix(XLA_FFI_CallFrame*); \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cudecomp_transpose_##Suffix, transpose<Type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() \
      .Arg<ffi::Token>().Arg<ffi::BufferR0<ffi::S64>>().Arg<ffi::BufferR0<ffi::S64>>() \
      .Arg<ffi::BufferR4<Type>>().Ret<ffi::Token>().Ret<ffi::BufferR4<Type>>() \
      .Attr<std::string_view>("preparation_id").Attr<int64_t>("direction") \
      .Attr<std::string_view>("collective_configuration_id").Attr<std::string_view>("collective_call_id"))
PHX_CUDECOMP_TRANSPOSE(c64, ffi::C64);
PHX_CUDECOMP_TRANSPOSE(c128, ffi::C128);
PHX_CUDA_HANDLER_TABLE(
  {"phydrax_cudecomp_transpose_c64", phydrax_cudecomp_transpose_c64},
  {"phydrax_cudecomp_transpose_c128", phydrax_cudecomp_transpose_c128})
