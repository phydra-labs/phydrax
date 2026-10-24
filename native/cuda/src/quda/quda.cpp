// Copyright (c) 2026 PHYDRA, Inc. All rights reserved.
// Device-resident public QUDA C API adaptation through the public XLA FFI.
#include "phx_cuda_common.hpp"
#include "quda_kernels.hpp"
#include <quda.h>
#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>
#include <array>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif
#include <stdexcept>
#include <string>

namespace ffi = xla::ffi;
namespace quda {
// Declared in QUDA's internal tune_quda.h: "cpu_arch=...,gpu_arch=...,cuda_version=...".
// QUDA aborts on a tune cache written by a build with another value.
const std::string get_quda_hash();
#ifdef _WIN32
cudaError_t release_constant_arguments();
#endif
}  // namespace quda
namespace {
std::mutex quda_mutex;
bool initialized = false;
int initialized_device = -1;

void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

class DeviceAllocation {
 public:
  void* data = nullptr;
  explicit DeviceAllocation(size_t bytes) { check_cuda(cudaMalloc(&data, bytes), "QUDA temporary allocation"); }
  ~DeviceAllocation() { if (data) cudaFree(data); }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
};

void prepare_cache() {
#ifdef _WIN32
  // Use the Unicode CRT environment shared with the statically linked QUDA.
  if (_wgetenv(L"QUDA_RESOURCE_PATH") != nullptr) return;
#else
  if (std::getenv("QUDA_RESOURCE_PATH") != nullptr) return;
#endif
  std::filesystem::path root;
#ifdef _WIN32
  const wchar_t* profile = _wgetenv(L"USERPROFILE");
  if (profile == nullptr || *profile == L'\0')
    throw std::runtime_error("QUDA tune cache requires USERPROFILE");
  root = std::filesystem::path(profile) / L"AppData" / L"Local";
  if (!root.is_absolute()) throw std::runtime_error("QUDA tune cache requires an absolute USERPROFILE");
#else
  const char* xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg != nullptr && *xdg != '\0') {
    root = xdg;
  } else {
    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') root = home;
    else {
      const passwd* entry = getpwuid(getuid());
      if (entry == nullptr) throw std::runtime_error("QUDA tune cache requires HOME or XDG_CACHE_HOME");
      root = entry->pw_dir;
    }
    root /= ".cache";
  }
#endif
  // The Phydrax build hash covers sources only; QUDA's hash adds the compiled
  // architectures and compiler version that its tune cache must match.
  std::string variant = quda::get_quda_hash();
  for (char& character : variant) {
    const bool safe = (character >= '0' && character <= '9') || (character >= 'a' && character <= 'z') ||
                      (character >= 'A' && character <= 'Z') || character == '.' || character == '-';
    if (!safe) character = '_';
  }
  const auto path = root / "phydrax" / "quda" / phx_cuda_build_hash() / variant;
  std::filesystem::create_directories(path);
#ifdef _WIN32
  if (_wputenv_s(L"QUDA_RESOURCE_PATH", path.c_str()) != 0)
#else
  if (setenv("QUDA_RESOURCE_PATH", path.c_str(), 0) != 0)
#endif
    throw std::runtime_error("Unable to set QUDA_RESOURCE_PATH for the native tune cache");
}

void initialize() {
  int device;
  check_cuda(cudaGetDevice(&device), "QUDA current CUDA device");
  if (initialized && device != initialized_device)
    throw std::invalid_argument("QUDA supports one CUDA device per process; initialized on device "
      + std::to_string(initialized_device) + ", refusing device " + std::to_string(device));
  if (!initialized) {
    prepare_cache();
    setVerbosityQuda(QUDA_SILENT, "", stdout);
    initQuda(device);
    initialized_device = device;
    initialized = true;
  }
}

struct Geometry {
  std::array<int, 4> extent;
  int64_t volume = 1;
  Geometry(int64_t x, int64_t y, int64_t z, int64_t t) {
    const std::array<int64_t, 4> input{x, y, z, t};
    for (int axis = 0; axis < 4; ++axis) {
      if (input[axis] < 2 || input[axis] % 2 || input[axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument("QUDA requires periodic four-dimensional even extents");
      if (volume > std::numeric_limits<int>::max() / input[axis])
        throw std::invalid_argument("QUDA lattice exceeds the supported 32-bit site indexing envelope");
      extent[axis] = static_cast<int>(input[axis]);
      volume *= input[axis];
    }
  }
};

template <ffi::DataType dtype> constexpr QudaPrecision precision() {
  return dtype == ffi::C128 ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
}

template <ffi::DataType dtype>
QudaGaugeParam load_gauge(ffi::Buffer<dtype> links, const Geometry& geometry) {
  const auto dims = links.dimensions();
  if (dims.size() != 5 || dims[0] != 4 || dims[1] != 2 || dims[2] != geometry.volume / 2 || dims[3] != 3 || dims[4] != 3)
    throw std::invalid_argument("QUDA gauge links must have shape (4,2,volume/2,3,3)");
  auto param = newQudaGaugeParam();
  for (int axis = 0; axis < 4; ++axis) param.X[axis] = geometry.extent[axis];
  param.location = QUDA_CUDA_FIELD_LOCATION;
  param.anisotropy = 1.0;
  param.tadpole_coeff = 1.0;
  param.scale = 1.0;
  param.type = QUDA_WILSON_LINKS;
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;
  param.cpu_prec = precision<dtype>();
  param.cuda_prec = param.cuda_prec_sloppy = param.cuda_prec_refinement_sloppy =
    param.cuda_prec_precondition = param.cuda_prec_eigensolver = precision<dtype>();
  param.reconstruct = param.reconstruct_sloppy = param.reconstruct_refinement_sloppy =
    param.reconstruct_precondition = param.reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  param.ga_pad = 0;
  param.use_resident_gauge = 0;
  param.make_resident_gauge = 1;
  std::array<void*, 4> pointers;
  for (int axis = 0; axis < 4; ++axis) pointers[axis] = links.typed_data() + axis * geometry.volume * 9;
  // QUDA owns just one resident field. Reload on EVERY call; no correctness
  // dependency on prior calls, CSE, rematerialization, or gauge pointer identity.
  loadGaugeQuda(pointers.data(), &param);
  return param;
}

// QUDA executes on private streams, not the XLA stream. A synchronized boundary
// is intentional: wait for XLA producers before loading DEVICE links, and wait
// for all QUDA work before returning device outputs to XLA. The process mutex
// spans both barriers and all resident-gauge accesses.
template <class Work>
ffi::Error serialized(cudaStream_t stream, Work&& work) {
  std::lock_guard<std::mutex> lock(quda_mutex);
  try {
    check_cuda(cudaStreamSynchronize(stream), "QUDA XLA input stream synchronization");
    initialize();
    work();
    check_cuda(cudaDeviceSynchronize(), "QUDA completion synchronization");
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    phx::cuda::set_last_error(error.what());
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return phx::cuda::ffi_error(error.what());
  }
}

template <ffi::DataType dtype, ffi::DataType real_dtype>
ffi::Error gauge_action(cudaStream_t stream, ffi::Buffer<dtype> links, ffi::Buffer<ffi::PRED> mask,
  ffi::ResultBuffer<real_dtype> out, int64_t x, int64_t y, int64_t z, int64_t t) {
  return serialized(stream, [&] {
    const Geometry geometry(x, y, z, t);
    if (mask.element_count() != geometry.volume * 6 || out->element_count() != 1)
      throw std::invalid_argument("QUDA action requires owned_mask shape (volume,6) and a scalar result");
    DeviceAllocation invalid(sizeof(int));
    check_cuda(cudaMemset(invalid.data, 0, sizeof(int)), "QUDA mask validation initialization");
    phx::cuda::quda_kernels::launch_any_unowned(mask.typed_data(), mask.element_count(), static_cast<int*>(invalid.data));
    int host_invalid = 0;
    check_cuda(cudaMemcpy(&host_invalid, invalid.data, sizeof(int), cudaMemcpyDeviceToHost), "QUDA ownership validation");
    if (host_invalid) throw std::invalid_argument("QUDA gauge action requires a fully owned periodic lattice");
    load_gauge(links, geometry);
    double plaquette[3];
    plaqQuda(plaquette);
    // QUDA's public reduction returns three host scalars, never a host gauge.
    using Real = ffi::NativeType<real_dtype>;
    const Real action = static_cast<Real>(6.0 * geometry.volume * (1.0 - plaquette[0]));
    check_cuda(cudaMemcpy(out->typed_data(), &action, sizeof(Real), cudaMemcpyHostToDevice), "QUDA action scalar result");
  });
}

template <ffi::DataType dtype, class Real>
ffi::Error gauge_force(cudaStream_t stream, ffi::Buffer<dtype> links, ffi::ResultBuffer<dtype> out,
  int64_t x, int64_t y, int64_t z, int64_t t) {
  return serialized(stream, [&] {
    const Geometry geometry(x, y, z, t);
    if (out->element_count() != links.element_count()) throw std::invalid_argument("QUDA force shape must match links");
    auto param = load_gauge(links, geometry);
    DeviceAllocation momentum(4 * geometry.volume * 10 * sizeof(Real));
    std::array<void*, 4> momentum_ptrs;
    for (int axis = 0; axis < 4; ++axis)
      momentum_ptrs[axis] = static_cast<Real*>(momentum.data) + axis * geometry.volume * 10;
    int paths[4][6][3];
    int* path_ptrs[4][6];
    int** direction_ptrs[4];
    for (int mu = 0; mu < 4; ++mu) {
      int path = 0;
      for (int nu = 0; nu < 4; ++nu) if (nu != mu) {
        // Both orientations of each plaquette with the differentiated link
        // removed, in QUDA's forward=0..3/backward=7..4 convention.
        paths[mu][path][0] = nu;
        paths[mu][path][1] = 7 - mu;
        paths[mu][path][2] = 7 - nu;
        ++path;
        paths[mu][path][0] = 7 - nu;
        paths[mu][path][1] = 7 - mu;
        paths[mu][path][2] = nu;
        ++path;
      }
      for (int k = 0; k < 6; ++k) path_ptrs[mu][k] = paths[mu][k];
      direction_ptrs[mu] = path_ptrs[mu];
    }
    int lengths[6] = {3, 3, 3, 3, 3, 3};
    double coefficients[6] = {1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3};
    param.use_resident_gauge = 1;
    param.make_resident_gauge = 0;
    param.use_resident_mom = 0;
    param.make_resident_mom = 0;
    param.overwrite_mom = 1;
    param.return_result_mom = 1;
    computeGaugeForceQuda(momentum_ptrs.data(), nullptr, direction_ptrs, lengths, coefficients, 6, 3, 1.0, &param);
    check_cuda(cudaDeviceSynchronize(), "QUDA momentum completion synchronization");
    phx::cuda::quda_kernels::launch_decode_momentum(
      static_cast<Real*>(momentum.data), reinterpret_cast<Real*>(out->typed_data()), 4 * geometry.volume);
    check_cuda(cudaGetLastError(), "QUDA momentum decoding");
  });
}

template <ffi::DataType dtype>
ffi::Error wilson_dslash(cudaStream_t stream, ffi::Buffer<dtype> links, ffi::Buffer<dtype> source,
  ffi::ResultBuffer<dtype> out, int64_t x, int64_t y, int64_t z, int64_t t, int64_t dagger) {
  return serialized(stream, [&] {
    const Geometry geometry(x, y, z, t);
    const auto dims = source.dimensions();
    if (dims.size() != 5 || dims[0] < 1 || dims[1] != 2 || dims[2] != geometry.volume / 2 || dims[3] != 4 || dims[4] != 3
        || out->element_count() != source.element_count() || (dagger != 0 && dagger != 1))
      throw std::invalid_argument("QUDA spinors require (rhs,2,volume/2,4,3) and dagger=0 or 1");
    load_gauge(links, geometry);
    auto param = newQudaInvertParam();
    param.dslash_type = QUDA_WILSON_DSLASH;
    param.inv_type = QUDA_BICGSTAB_INVERTER;
    param.mass = 0.0;
    param.kappa = 0.125;
    param.m5 = 0.0;
    param.Ls = 1;
    param.solution_type = QUDA_MAT_SOLUTION;
    param.solve_type = QUDA_DIRECT_PC_SOLVE;
    param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    param.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;
    param.mass_normalization = QUDA_ASYMMETRIC_MASS_NORMALIZATION;
    param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
    param.tol = 1e-10;
    param.maxiter = 1;
    param.reliable_delta = 0.1;
    param.cpu_prec = param.cuda_prec = param.cuda_prec_sloppy = param.cuda_prec_refinement_sloppy =
      param.cuda_prec_precondition = param.cuda_prec_eigensolver = precision<dtype>();
    param.input_location = param.output_location = QUDA_CUDA_FIELD_LOCATION;
    param.dirac_order = QUDA_DIRAC_ORDER;
    param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    param.verbosity = QUDA_SILENT;
    const int64_t parity_size = geometry.volume / 2 * 12;
    for (int64_t rhs = 0; rhs < dims[0]; ++rhs)
      for (int parity = 0; parity < 2; ++parity)
        dslashQuda(out->typed_data() + (rhs * 2 + parity) * parity_size,
          source.typed_data() + (rhs * 2 + 1 - parity) * parity_size, &param, static_cast<QudaParity>(parity));
  });
}
}  // namespace

phx::cuda::RegistryBase* component_registry() { return nullptr; }
int32_t component_shutdown() {
  std::lock_guard<std::mutex> lock(quda_mutex);
  if (!initialized) return PHX_CUDA_SUCCESS;
  try {
    int previous;
    check_cuda(cudaGetDevice(&previous), "QUDA shutdown device query");
    check_cuda(cudaSetDevice(initialized_device), "QUDA shutdown device selection");
    check_cuda(cudaDeviceSynchronize(), "QUDA shutdown synchronization");
#ifdef _WIN32
    check_cuda(quda::release_constant_arguments(), "QUDA constant argument event teardown");
#endif
    endQuda();
    initialized = false;
    initialized_device = -1;
    check_cuda(cudaSetDevice(previous), "QUDA shutdown device restoration");
    return PHX_CUDA_SUCCESS;
  } catch (const std::exception& error) {
    phx::cuda::set_last_error(error.what());
    return PHX_CUDA_PROVIDER_ERROR;
  }
}

#define PHX_QUDA_GEOMETRY .Attr<int64_t>("x").Attr<int64_t>("y").Attr<int64_t>("z").Attr<int64_t>("t")
#define PHX_QUDA_ACTION(name, complex_type, real_type) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(name, (gauge_action<complex_type, real_type>), \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<complex_type>>() \
      .Arg<ffi::Buffer<ffi::PRED>>().Ret<ffi::Buffer<real_type>>() PHX_QUDA_GEOMETRY)
#define PHX_QUDA_FORCE(name, complex_type, real_type) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(name, (gauge_force<complex_type, real_type>), \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<complex_type>>() \
      .Ret<ffi::Buffer<complex_type>>() PHX_QUDA_GEOMETRY)
#define PHX_QUDA_DSLASH(name, complex_type) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(name, wilson_dslash<complex_type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<complex_type>>() \
      .Arg<ffi::Buffer<complex_type>>().Ret<ffi::Buffer<complex_type>>() PHX_QUDA_GEOMETRY .Attr<int64_t>("dagger"))
extern "C" {
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_quda_gauge_action_c64(XLA_FFI_CallFrame*);
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_quda_gauge_action_c128(XLA_FFI_CallFrame*);
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_quda_gauge_force_c64(XLA_FFI_CallFrame*);
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_quda_gauge_force_c128(XLA_FFI_CallFrame*);
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_quda_wilson_dslash_c64(XLA_FFI_CallFrame*);
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_quda_wilson_dslash_c128(XLA_FFI_CallFrame*);
PHX_QUDA_ACTION(phydrax_quda_gauge_action_c64, ffi::C64, ffi::F32);
PHX_QUDA_ACTION(phydrax_quda_gauge_action_c128, ffi::C128, ffi::F64);
PHX_QUDA_FORCE(phydrax_quda_gauge_force_c64, ffi::C64, float);
PHX_QUDA_FORCE(phydrax_quda_gauge_force_c128, ffi::C128, double);
PHX_QUDA_DSLASH(phydrax_quda_wilson_dslash_c64, ffi::C64);
PHX_QUDA_DSLASH(phydrax_quda_wilson_dslash_c128, ffi::C128);
}
PHX_CUDA_HANDLER_TABLE(
  {"phydrax_quda_gauge_action_c64", phydrax_quda_gauge_action_c64},
  {"phydrax_quda_gauge_action_c128", phydrax_quda_gauge_action_c128},
  {"phydrax_quda_gauge_force_c64", phydrax_quda_gauge_force_c64},
  {"phydrax_quda_gauge_force_c128", phydrax_quda_gauge_force_c128},
  {"phydrax_quda_wilson_dslash_c64", phydrax_quda_wilson_dslash_c64},
  {"phydrax_quda_wilson_dslash_c128", phydrax_quda_wilson_dslash_c128})
