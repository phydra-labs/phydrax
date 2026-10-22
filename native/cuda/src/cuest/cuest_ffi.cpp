// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// This adapter is original integration code; NVIDIA headers are not vendored.
// NVIDIA cuEST licensed-deliverable notice (required by its C API headers):
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
// LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
// SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE. IT IS
// PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
// NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
// DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
// NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
// LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
// SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
// DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
// WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
// ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
// OF THESE LICENSED DELIVERABLES.
//
// U.S. Government End Users. These Licensed Deliverables are a
// "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
// 1995), consisting of "commercial computer software" and "commercial
// computer software documentation" as such terms are used in 48
// C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
// only as a commercial end item. Consistent with 48 C.F.R.12.212 and
// 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
// U.S. Government End Users acquire the Licensed Deliverables with
// only those rights set forth herein.
#include "phx_cuda_common.hpp"
#include <cuest.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <tuple>
#include <utility>

namespace ffi = xla::ffi;
namespace {
using I32 = ffi::Buffer<ffi::S32>;
using I64 = ffi::Buffer<ffi::S64>;
using F64 = ffi::Buffer<ffi::F64>;
using RF64 = ffi::Result<F64>;
using RI32 = ffi::Result<I32>;
std::recursive_mutex& mutex() { static auto* m = new std::recursive_mutex; return *m; }
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(s)); }
void vendor(cuestStatus_t s, const char* operation) {
  if (s != CUEST_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": cuEST status " + std::to_string(s));
}
#define CUEST(call) vendor((call), #call)
size_t add(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a) throw std::invalid_argument("cuEST workspace size overflow");
  return a + b;
}
size_t bytes(int64_t n, size_t width) {
  if (n < 0 || static_cast<uint64_t>(n) > std::numeric_limits<size_t>::max()/width)
    throw std::invalid_argument("cuEST buffer size overflow");
  return static_cast<size_t>(n)*width;
}
struct Settings {
  int64_t representation, relative, inversion, variable, limit, capacity;
  double cutoff, screening;
  void validate() const {
    if ((representation != 0 && representation != 1) || (relative != 0 && relative != 1) ||
        (inversion != 0 && inversion != 1) || variable <= 0 || limit <= 0 || capacity <= 0 ||
        !std::isfinite(cutoff) || cutoff < 0 || !std::isfinite(screening) || screening < 0)
      throw std::invalid_argument("Invalid cuEST representation, fitting, screening, workspace or cache settings");
  }
  bool operator==(const Settings& b) const {
    return representation == b.representation && relative == b.relative && inversion == b.inversion &&
      variable == b.variable && cutoff == b.cutoff && screening == b.screening;
  }
};
// Only queried/user-owned storage is byte-accounted. Opaque vendor metadata,
// private CUDA allocations and JIT memory have no public size query.
struct Budget {
  std::mutex mutex;
  const int64_t limit, capacity;
  size_t used = 0;
  bool cleanup_failed = false;
  Budget(int64_t l, int64_t c) : limit(l), capacity(c) {}
  void reserve(size_t n) {
    std::lock_guard<std::mutex> lock(mutex);
    if (cleanup_failed) throw std::runtime_error("cuEST partition cleanup failed");
    if (used > static_cast<uint64_t>(limit) || n > static_cast<uint64_t>(limit) - used)
      throw std::runtime_error("cuEST partition owned storage limit exceeded (host + device, including live leases)");
    used += n;
  }
  void release(size_t n) {
    std::lock_guard<std::mutex> lock(mutex);
    used -= n;
  }
  void poison(const std::string& message) {
    { std::lock_guard<std::mutex> lock(mutex); cleanup_failed = true; }
    phx::cuda::set_last_error(message);
  }
};
// Counting the allocator's actual request also accounts vector growth and
// construction temporaries before allocation, rather than estimating capacity.
template<class T> struct HostAllocator {
  using value_type = T;
  std::shared_ptr<Budget> budget;
  explicit HostAllocator(std::shared_ptr<Budget> b) : budget(std::move(b)) {}
  template<class U> HostAllocator(const HostAllocator<U>& b) : budget(b.budget) {}
  T* allocate(size_t n) {
    const size_t size = bytes(n, sizeof(T));
    budget->reserve(size);
    try { return std::allocator<T>{}.allocate(n); }
    catch (...) { budget->release(size); throw; }
  }
  void deallocate(T* p, size_t n) noexcept {
    std::allocator<T>{}.deallocate(p, n);
    budget->release(n*sizeof(T));
  }
  template<class U> bool operator==(const HostAllocator<U>& b) const { return budget == b.budget; }
  template<class U> bool operator!=(const HostAllocator<U>& b) const { return !(*this == b); }
};
template<class T> using HostVector = std::vector<T, HostAllocator<T>>;
struct Parameters {
  cuestParametersType_t type;
  void* value = nullptr;
  std::shared_ptr<Budget> budget;
  Parameters(cuestParametersType_t t, std::shared_ptr<Budget> b) : type(t), budget(std::move(b)) {
    CUEST(cuestParametersCreate(type, &value));
  }
  ~Parameters() {
    if (value && cuestParametersDestroy(type, value) != CUEST_STATUS_SUCCESS)
      budget->poison("cuEST parameter cleanup failed");
  }
  Parameters(const Parameters&) = delete;
  template<class T> void set(int attr, T v) { CUEST(cuestParametersConfigure(type, value, attr, &v, sizeof(v))); }
  void abandon() noexcept { value = nullptr; }
};
// Every opaque object is destroyed before its user-owned persistent workspace.
struct Object {
  void* value = nullptr;
  cuestStatus_t (*destroy)(void*);
  std::shared_ptr<Budget> budget;
  Object(cuestStatus_t (*d)(void*), std::shared_ptr<Budget> b) : destroy(d), budget(std::move(b)) {}
  bool clear() {
    if (!value) return true;
    const auto status = destroy(value);
    value = nullptr;
    if (status == CUEST_STATUS_SUCCESS) return true;
    budget->poison("cuEST object cleanup failed: status " + std::to_string(status));
    return false;
  }
  void abandon() noexcept { value = nullptr; }
  ~Object() { clear(); }
  Object(const Object&) = delete;
  Object(Object&& other) noexcept
      : value(std::exchange(other.value, nullptr)), destroy(other.destroy), budget(std::move(other.budget)) {}
};
struct Entry;
struct Workspace {
  Entry* owner;
  cuestWorkspace_t value{};
  explicit Workspace(Entry* e) : owner(e) {}
  ~Workspace();
  bool release();
  void resize(cuestWorkspaceDescriptor_t d);
  Workspace(const Workspace&) = delete;
};
struct Packed {
  HostVector<int32_t> shells;
  HostVector<double> exponents, coefficients;
  HostVector<int32_t> centers;
  HostVector<uint64_t> counts;
  int64_t nao = 0;
  uint64_t max_l = 0;
  explicit Packed(const std::shared_ptr<Budget>& b)
      : shells(HostAllocator<int32_t>(b)), exponents(HostAllocator<double>(b)),
        coefficients(HostAllocator<double>(b)), centers(HostAllocator<int32_t>(b)),
        counts(HostAllocator<uint64_t>(b)) {}
};
struct Input {
  Packed primary, auxiliary;
  HostVector<double> positions;
  int64_t natom = 0;
  explicit Input(const std::shared_ptr<Budget>& b)
      : primary(b), auxiliary(b), positions(HostAllocator<double>(b)) {}
};
template<class T, class B> void stage(HostVector<T>& out, B b, cudaStream_t stream) {
  out.resize(static_cast<size_t>(b.element_count()));
  if (!out.empty()) check(cudaMemcpyAsync(out.data(), b.typed_data(), bytes(b.element_count(), sizeof(T)), cudaMemcpyDeviceToHost, stream));
}
void shape(F64 b, int64_t n, int64_t m, const char* name) {
  auto d = b.dimensions();
  if (d.size() != 2 || d[0] != n || d[1] != m) throw std::invalid_argument(std::string("Invalid cuEST ") + name + " shape");
}
void inspect(Packed& p, int64_t natom, bool pure) {
  p.centers.clear(); p.counts.clear(); p.nao = 0; p.max_l = 0;
  int32_t previous = -1;
  HostVector<bool> angular{HostAllocator<bool>(p.shells.get_allocator().budget)};
  auto contiguous = [&] {
    if (std::find(angular.begin(), angular.end(), false) != angular.end())
      throw std::invalid_argument("cuEST requires all angular momenta 0..L at each basis center");
  };
  const size_t shell_count = p.shells.size()/4;
  p.centers.reserve(shell_count); p.counts.reserve(shell_count);
  for (size_t i = 0; i < p.shells.size(); i += 4) {
    int32_t center = p.shells[i], l = p.shells[i+1], n = p.shells[i+2], offset = p.shells[i+3];
    if (center < previous || center < 0 || center >= natom || l < 0 || n <= 0 || offset < 0 ||
        static_cast<size_t>(offset) > p.exponents.size() || static_cast<size_t>(n) > p.exponents.size()-offset)
      throw std::invalid_argument("Invalid packed cuEST shell metadata");
    if (center != previous) {
      contiguous(); angular.clear(); p.centers.push_back(center); p.counts.push_back(0); previous = center;
    }
    ++p.counts.back();
    if (static_cast<size_t>(l) >= angular.size()) angular.resize(static_cast<size_t>(l)+1, false);
    angular[l] = true;
    p.max_l = std::max(p.max_l, static_cast<uint64_t>(l));
    const int64_t width = pure ? 2LL*l+1 : (static_cast<int64_t>(l)+1)*(static_cast<int64_t>(l)+2)/2;
    if (width > std::numeric_limits<int64_t>::max()-p.nao) throw std::invalid_argument("cuEST AO count overflow");
    p.nao += width;
  }
  contiguous();
  for (double x : p.exponents) if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("cuEST exponents must be positive finite FP64");
  for (double x : p.coefficients) if (!std::isfinite(x)) throw std::invalid_argument("cuEST coefficients must be finite FP64");
}
struct Entry {
  int device;
  int64_t partition;
  int32_t id = 0;
  std::array<int64_t, 4> key;
  Settings settings;
  std::shared_ptr<Budget> budget;
  Input input;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  Object handle{cuestDestroy, budget};
  Workspace primary_ws{this}, auxiliary_ws{this}, pair_ws{this}, oe_ws{this}, df_ws{this}, temporary{this}, scratch{this};
  Object primary{cuestAOBasisDestroy, budget}, auxiliary{cuestAOBasisDestroy, budget}, pair{cuestAOPairListDestroy, budget};
  Object oe{cuestOEIntPlanDestroy, budget}, df{cuestDFIntPlanDestroy, budget};
  HostVector<double> center_xyz;
  HostVector<double> density_staging;
  std::optional<Parameters> compute_parameters[18];
  Parameters& parameters(cuestParametersType_t type) {
    auto& p = compute_parameters[static_cast<size_t>(type)];
    if (!p) {
      p.emplace(type, budget);
      switch (type) {
        case CUEST_POTENTIALCOMPUTE_PARAMETERS:
          p->set(CUEST_POTENTIALCOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE, CUEST_FFLOAT_USAGE_MODE_OFF); break;
        case CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS:
          p->set(CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE, CUEST_FFLOAT_USAGE_MODE_OFF); break;
        case CUEST_DFCOULOMBCOMPUTE_PARAMETERS:
          p->set(CUEST_DFCOULOMBCOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE, CUEST_FFLOAT_USAGE_MODE_OFF); break;
        case CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS:
          p->set(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE, CUEST_FFLOAT_USAGE_MODE_OFF);
          // RECOMPUTE preserves the plan without large host/device caches.
          p->set(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS_MEMORY_POLICY, CUEST_DFSYMMETRICDERIVATIVECOMPUTE_MEMORY_POLICY_RECOMPUTE); break;
        default: break;
      }
    }
    return *p;
  }
  Entry(int d, int64_t p, std::array<int64_t, 4> k, const Settings& s, std::shared_ptr<Budget> b)
      : device(d), partition(p), key(k), settings(s), budget(std::move(b)),
        input(budget), center_xyz(HostAllocator<double>(budget)),
        density_staging(HostAllocator<double>(budget)) {}
  ~Entry() {
    int previous = device;
    const auto got_device = cudaGetDevice(&previous);
    const bool selected =
        got_device == cudaSuccess && cudaSetDevice(device) == cudaSuccess;
    const bool drained =
        selected && (!stream || cudaStreamSynchronize(stream) == cudaSuccess);
    if (!drained) {
      for (auto& p : compute_parameters) {
        if (p) p->abandon();
        p.reset();
      }
      for (Object* o : {&df, &oe, &pair, &auxiliary, &primary, &handle})
        o->abandon();
      for (Workspace* w : {&scratch, &temporary, &df_ws, &oe_ws, &pair_ws,
                           &auxiliary_ws, &primary_ws})
        w->value = {};
      blas = nullptr;
      stream = nullptr;
      budget->poison(
          "cuEST teardown abandoned device resources after ownership or drain failure");
      if (selected && got_device == cudaSuccess &&
          cudaSetDevice(previous) != cudaSuccess)
        budget->poison("cuEST device restore failed");
      return;
    }
    bool safe = true;
    for (auto& p : compute_parameters) p.reset();
    // A failed vendor teardown may leave a live object using its workspace.
    // Abandon that storage, retaining its charge and poisoning this partition.
    for (Object* o : {&df, &oe, &pair, &auxiliary, &primary, &handle})
      if (!o->clear()) safe = false;
    for (Workspace* w : {&scratch, &temporary, &df_ws, &oe_ws, &pair_ws,
                         &auxiliary_ws, &primary_ws}) {
      if (safe) w->release();
      else w->value = {};
    }
    if (!safe)
      budget->poison("cuEST teardown could not safely release owned workspace");
    if (blas && cublasDestroy(blas) != CUBLAS_STATUS_SUCCESS)
      budget->poison("cuEST cuBLAS cleanup failed");
    if (stream && cudaStreamDestroy(stream) != cudaSuccess)
      budget->poison("cuEST stream cleanup failed");
    if (got_device == cudaSuccess && cudaSetDevice(previous) != cudaSuccess)
      budget->poison("cuEST device restore failed");
  }
  void sync() { check(cudaStreamSynchronize(stream)); }
  double* scratch_data() { return reinterpret_cast<double*>(scratch.value.deviceBuffer); }
};
struct StreamDrain {
  cudaStream_t stream;
  std::shared_ptr<Budget> budget;
  bool drained = false;
  ~StreamDrain() {
    if (!drained && cudaStreamSynchronize(stream) != cudaSuccess)
      budget->poison("cuEST construction/staging stream drain failed");
  }
};
phx::cuda::Registry<Entry>& registry() {
  static auto* instance = new phx::cuda::Registry<Entry>();
  return *instance;
}
using ContentKey = std::tuple<int, int64_t, std::array<int64_t, 4>>;
struct IndexItem {
  int32_t id;
  std::weak_ptr<Entry> entry;
};
struct Index {
  // Calls hold mutex(); releases may destroy entries independently. Budgets
  // therefore synchronize their own accounting and outlive registry leases.
  std::map<ContentKey, IndexItem> items;
  std::map<std::pair<int, int64_t>, std::shared_ptr<Budget>> budgets;
  void prune() {
    for (auto it = items.begin(); it != items.end();) {
      if (it->second.entry.expired() ||
          registry().find_device(std::get<1>(it->first), it->second.id) != std::get<0>(it->first))
        it = items.erase(it);
      else ++it;
    }
    for (auto it = budgets.begin(); it != budgets.end();) {
      const auto b = it->second;
      std::lock_guard<std::mutex> guard(b->mutex);
      if (b.use_count() == 2 && b->used == 0 && !b->cleanup_failed) it = budgets.erase(it);
      else ++it;
    }
  }
};
Index& index() { static auto* instance = new Index; return *instance; }
Workspace::~Workspace() { release(); }
bool Workspace::release() {
  bool ok = true;
  if (value.deviceBuffer) {
    if (cudaFree(reinterpret_cast<void*>(value.deviceBuffer)) == cudaSuccess)
      owner->budget->release(value.deviceBufferSizeInBytes);
    else { owner->budget->poison("cuEST cudaFree workspace failed"); ok = false; }
  }
  if (value.hostBuffer) {
    if (cudaFreeHost(reinterpret_cast<void*>(value.hostBuffer)) == cudaSuccess)
      owner->budget->release(value.hostBufferSizeInBytes);
    else { owner->budget->poison("cuEST cudaFreeHost workspace failed"); ok = false; }
  }
  value = {};
  return ok;
}
void Workspace::resize(cuestWorkspaceDescriptor_t d) {
  if (d.hostBufferSizeInBytes <= value.hostBufferSizeInBytes && d.deviceBufferSizeInBytes <= value.deviceBufferSizeInBytes) return;
  owner->sync();
  if (!release()) throw std::runtime_error("cuEST workspace resize cleanup failed");
  const size_t required = add(d.hostBufferSizeInBytes, d.deviceBufferSizeInBytes);
  owner->budget->reserve(required);
  void* host = nullptr;
  if (d.hostBufferSizeInBytes) {
    const auto status = cudaMallocHost(&host, d.hostBufferSizeInBytes);
    if (status != cudaSuccess) { owner->budget->release(required); check(status); }
  }
  value.hostBuffer = reinterpret_cast<uintptr_t>(host);
  value.hostBufferSizeInBytes = d.hostBufferSizeInBytes;
  void* device = nullptr;
  if (d.deviceBufferSizeInBytes) {
    const auto status = cudaMalloc(&device, d.deviceBufferSizeInBytes);
    if (status != cudaSuccess) { owner->budget->release(d.deviceBufferSizeInBytes); check(status); }
  }
  value.deviceBuffer = reinterpret_cast<uintptr_t>(device);
  value.deviceBufferSizeInBytes = d.deviceBufferSizeInBytes;
}
void make_basis(Entry& e, const Packed& p, bool pure, Workspace& persistent, Object& basis) {
  Parameters sp(CUEST_AOSHELL_PARAMETERS, e.budget), bp(CUEST_AOBASIS_PARAMETERS, e.budget);
  HostVector<Object> shells{HostAllocator<Object>(e.budget)};
  HostVector<cuestAOShell_t> raw{HostAllocator<cuestAOShell_t>(e.budget)};
  shells.reserve(p.shells.size()/4); raw.reserve(p.shells.size()/4);
  StreamDrain drain{e.stream, e.budget};
  for (size_t i = 0; i < p.shells.size(); i += 4) {
    shells.emplace_back(cuestAOShellDestroy, e.budget);
    auto& s = shells.back();
    CUEST(cuestAOShellCreate(e.handle.value, pure, p.shells[i+1], p.shells[i+2],
      p.exponents.data()+p.shells[i+3], p.coefficients.data()+p.shells[i+3], sp.value, &s.value));
    raw.push_back(s.value);
  }
  cuestWorkspaceDescriptor_t pd{}, td{};
  CUEST(cuestAOBasisCreateWorkspaceQuery(e.handle.value, p.counts.size(), p.counts.data(), raw.data(), bp.value, &pd, &td, nullptr));
  persistent.resize(pd); e.temporary.resize(td);
  CUEST(cuestAOBasisCreate(e.handle.value, p.counts.size(), p.counts.data(), raw.data(), bp.value, &persistent.value, &e.temporary.value, &basis.value));
  e.sync();
  drain.drained = true;
}
void initialize(Entry& e) {
  check(cudaStreamCreateWithFlags(&e.stream, cudaStreamNonBlocking));
  check(cublasCreate(&e.blas)); check(cublasSetStream(e.blas, e.stream));
  check(cublasSetMathMode(e.blas, CUBLAS_PEDANTIC_MATH));
  Parameters hp(CUEST_HANDLE_PARAMETERS, e.budget);
  hp.set(CUEST_HANDLE_PARAMETERS_CUDASTREAM, e.stream);
  hp.set(CUEST_HANDLE_PARAMETERS_CUBLAS, e.blas);
  const uint64_t max_l = std::max<uint64_t>(10, std::max(e.input.primary.max_l, e.input.auxiliary.max_l));
  hp.set(CUEST_HANDLE_PARAMETERS_MAX_L_SOLID_HARMONIC, max_l);
  CUEST(cuestCreate(hp.value, &e.handle.value));
  CUEST(cuestSetMathMode(e.handle.value, CUEST_NATIVE_FP64_MATH_MODE));
  make_basis(e, e.input.primary, e.settings.representation != 0, e.primary_ws, e.primary);
  e.center_xyz.reserve(e.input.primary.centers.size()*3);
  for (int32_t center : e.input.primary.centers)
    e.center_xyz.insert(e.center_xyz.end(), e.input.positions.begin()+3*center, e.input.positions.begin()+3*center+3);
  Parameters pp(CUEST_AOPAIRLIST_PARAMETERS, e.budget), op(CUEST_OEINTPLAN_PARAMETERS, e.budget);
  cuestWorkspaceDescriptor_t pd{}, td{};
  CUEST(cuestAOPairListCreateWorkspaceQuery(e.handle.value, e.primary.value, e.input.primary.centers.size(), e.center_xyz.data(), e.settings.screening, pp.value, &pd, &td, nullptr));
  e.pair_ws.resize(pd); e.temporary.resize(td);
  CUEST(cuestAOPairListCreate(e.handle.value, e.primary.value, e.input.primary.centers.size(), e.center_xyz.data(), e.settings.screening, pp.value, &e.pair_ws.value, &e.temporary.value, &e.pair.value));
  e.sync();
  CUEST(cuestOEIntPlanCreateWorkspaceQuery(e.handle.value, e.primary.value, e.pair.value, op.value, &pd, &td, nullptr));
  e.oe_ws.resize(pd); e.temporary.resize(td);
  CUEST(cuestOEIntPlanCreate(e.handle.value, e.primary.value, e.pair.value, op.value, &e.oe_ws.value, &e.temporary.value, &e.oe.value));
  e.sync();
}
void ensure_df(Entry& e) {
  if (e.df.value) return;
  if (e.input.primary.centers != e.input.auxiliary.centers)
    throw std::invalid_argument("cuEST DF requires identical primary and auxiliary basis center sets");
  make_basis(e, e.input.auxiliary, true, e.auxiliary_ws, e.auxiliary);
  Parameters p(CUEST_DFINTPLAN_PARAMETERS, e.budget);
  p.set(CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF, e.settings.cutoff);
  p.set(CUEST_DFINTPLAN_PARAMETERS_FITTING_RELATIVE_CONDITIONING, static_cast<int32_t>(e.settings.relative));
  p.set(CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM, static_cast<cuestDFIntPlanParametersFittingAlgorithm_t>(e.settings.inversion));
  p.set(CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION, 1.0);
  p.set(CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, 0.0);
  p.set(CUEST_DFINTPLAN_PARAMETERS_THREE_INDEX_INTEGRAL_DIRECT, int32_t{0});
  cuestWorkspaceDescriptor_t pd{}, td{};
  CUEST(cuestDFIntPlanCreateWorkspaceQuery(e.handle.value, e.primary.value, e.auxiliary.value, e.pair.value, p.value, &pd, &td, nullptr));
  e.df_ws.resize(pd); e.temporary.resize(td);
  CUEST(cuestDFIntPlanCreate(e.handle.value, e.primary.value, e.auxiliary.value, e.pair.value, p.value, &e.df_ws.value, &e.temporary.value, &e.df.value));
  e.sync();
}
void packed_shape(I32 shell, F64 exp, F64 coef) {
  auto d = shell.dimensions();
  if (d.size() != 2 || d[0] == 0 || d[1] != 4 || exp.dimensions().size() != 1 ||
      coef.dimensions().size() != 1 || exp.element_count() != coef.element_count())
    throw std::invalid_argument("cuEST packed basis requires shells(ns,4) and matching flat primitive arrays");
}
std::shared_ptr<Entry> acquire(cudaStream_t stream, I64 binding_key, I32 ps, F64 pe, F64 pc,
                              I32 as, F64 ae, F64 ac, F64 xyz, F64 charges, Settings s,
                              int64_t partition) {
  s.validate(); packed_shape(ps, pe, pc); packed_shape(as, ae, ac);
  const auto kd = binding_key.dimensions();
  if (kd.size() != 1 || kd[0] != 4)
    throw std::invalid_argument("cuEST binding_key must have shape (4,) and dtype int64");
  auto d = xyz.dimensions();
  if (d.size() != 2 || d[0] <= 0 || d[1] != 3 || charges.dimensions().size() != 1 || charges.element_count() != d[0])
    throw std::invalid_argument("cuEST requires positions(natom,3) and charges(natom)");
  int device; check(cudaGetDevice(&device));
  std::array<int64_t, 4> key{};
  // The certificate is made from actual contents at explicit Python binding.
  // A warm acquire reads only these 32 bytes, never the full dynamic recipe.
  check(cudaMemcpyAsync(key.data(), binding_key.typed_data(), sizeof(key), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  auto& idx = index();
  const ContentKey content{device, partition, key};
  auto found = idx.items.find(content);
  if (found != idx.items.end()) {
    auto entry = found->second.entry.lock();
    if (entry && registry().find(device, partition, found->second.id) == entry) {
      if (!(entry->settings == s) || entry->settings.limit != s.limit || entry->settings.capacity != s.capacity)
        throw std::invalid_argument("cuEST binding settings differ from the certified partition");
      if (entry->input.natom != d[0] ||
          entry->input.primary.shells.size() != static_cast<size_t>(ps.element_count()) ||
          entry->input.primary.exponents.size() != static_cast<size_t>(pe.element_count()) ||
          entry->input.auxiliary.shells.size() != static_cast<size_t>(as.element_count()) ||
          entry->input.auxiliary.exponents.size() != static_cast<size_t>(ae.element_count()))
        throw std::invalid_argument("cuEST recipe shapes differ from the certified binding");
      { std::lock_guard<std::mutex> guard(entry->budget->mutex);
        if (entry->budget->cleanup_failed) throw std::runtime_error("cuEST partition cleanup failed"); }
      return entry;
    }
    idx.items.erase(found);
  }
  idx.prune();
  const auto budget_key = std::make_pair(device, partition);
  auto b = idx.budgets.find(budget_key);
  if (b == idx.budgets.end())
    b = idx.budgets.emplace(budget_key, std::make_shared<Budget>(s.limit, s.capacity)).first;
  const auto budget = b->second;
  if (budget->limit != s.limit || budget->capacity != s.capacity)
    throw std::invalid_argument("cuEST partition workspace/capacity policy is immutable while memos or leases remain live");
  budget->reserve(0);
  auto result = std::allocate_shared<Entry>(HostAllocator<Entry>(budget), device, partition, key, s, budget);
  // Canonical insertion performs capacity eviction before staging/allocation;
  // the shared budget still charges any retired Entry held by another lease.
  result->id = registry().insert(device, partition, s.capacity, result);
  try {
    registry().count_rebuild(device, partition);
    Input in(budget); in.natom = d[0];
    // Drain before the staged host buffers are destroyed on any failure.
    StreamDrain drain{stream, budget};
    stage(in.primary.shells, ps, stream); stage(in.primary.exponents, pe, stream); stage(in.primary.coefficients, pc, stream);
    stage(in.auxiliary.shells, as, stream); stage(in.auxiliary.exponents, ae, stream); stage(in.auxiliary.coefficients, ac, stream);
    stage(in.positions, xyz, stream); check(cudaStreamSynchronize(stream));
    drain.drained = true;
    inspect(in.primary, in.natom, s.representation != 0); inspect(in.auxiliary, in.natom, true);
    for (double x : in.positions) if (!std::isfinite(x)) throw std::invalid_argument("cuEST positions must be finite bohr coordinates");
    // Transfer ownership of every staged vector without a second recipe copy.
    result->input = std::move(in);
    initialize(*result);
    budget->reserve(0);
    idx.items.emplace(content, IndexItem{result->id, result});
    return result;
  } catch (...) {
    // Staging can fail after a previous asynchronous copy was submitted.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      budget->poison("cuEST recipe staging stream drain failed");
    registry().erase(device, partition, result->id);
    throw;
  }
}
void evict(const std::shared_ptr<Entry>& e) {
  if (e) registry().erase(e->device, e->partition, e->id);
}
void scalar(RI32 out, int32_t v, cudaStream_t stream) {
  if (out->element_count() != 1) throw std::invalid_argument("cuEST status must be scalar int32");
  check(cudaMemcpyAsync(out->typed_data(), &v, sizeof(v), cudaMemcpyHostToDevice, stream));
  check(cudaStreamSynchronize(stream));
}
void zero(F64 out, cudaStream_t stream) { check(cudaMemsetAsync(out.typed_data(), 0, bytes(out.element_count(), sizeof(double)), stream)); }
void scale(Entry& e, int64_t n, double a, double* x) {
  while (n) { const int chunk = static_cast<int>(std::min<int64_t>(n, std::numeric_limits<int>::max())); check(cublasDscal(e.blas, chunk, &a, x, 1)); x += chunk; n -= chunk; }
}
void axpy(Entry& e, int64_t n, double a, const double* x, double* y) {
  while (n) { const int chunk = static_cast<int>(std::min<int64_t>(n, std::numeric_limits<int>::max())); check(cublasDaxpy(e.blas, chunk, &a, x, 1, y, 1)); x += chunk; y += chunk; n -= chunk; }
}
void scatter(Entry& e, const double* compact, double* full, double factor = 1) {
  if (static_cast<int64_t>(e.input.primary.centers.size()) == e.input.natom) {
    axpy(e, 3*e.input.natom, factor, compact, full); return;
  }
  for (size_t i = 0; i < e.input.primary.centers.size(); ++i)
    axpy(e, 3, factor, compact+3*i, full+3*e.input.primary.centers[i]);
}
void allocate_scratch(Entry& e, bool gradient) {
  // One-electron values need only negative charges; gradients also need a
  // compact basis gradient and a full-system charge gradient.
  const size_t n = bytes(e.input.natom, (gradient ? 7 : 1)*sizeof(double));
  e.scratch.resize({0, n});
}
const double* negative_charges(Entry& e, F64 q) {
  double* neg = e.scratch_data();
  check(cudaMemcpyAsync(neg, q.typed_data(), bytes(e.input.natom, sizeof(double)), cudaMemcpyDeviceToDevice, e.stream));
  scale(e, e.input.natom, -1, neg); return neg;
}
uint64_t density_shape(Entry& e, F64 density, F64 factors) {
  const int64_t n = e.input.primary.nao;
  shape(density, n, n, "density");
  auto d = factors.dimensions();
  if (d.size() != 2 || d[1] != n) throw std::invalid_argument("cuEST factors must have shape (rank,nao), reconstructing spin-summed D=F.T@F");
  if (d[0] == 0) {
    // No fake occupied orbital is passed to vendor APIs. Verify the algebraic
    // zero channel rather than accepting a contradictory nonzero density.
    auto& staging = e.density_staging;
    stage(staging, density, e.stream); e.sync();
    if (std::any_of(staging.begin(), staging.end(), [](double x) { return x != 0; }))
      throw std::invalid_argument("cuEST rank-zero factors require exactly zero density");
  }
  return static_cast<uint64_t>(d[0]);
}
// A separate synchronization boundary makes memo resources independent of XLA
// stream identity. No CUDA graph capture is supported by this host-query ABI.
template<class F> ffi::Error invoke(cudaStream_t xla_stream, RI32 status, F f) {
  std::lock_guard<std::recursive_mutex> lock(mutex());
  std::shared_ptr<Entry> entry;
  try {
    check(cudaStreamSynchronize(xla_stream));
    f(entry);
    if (entry) { entry->sync(); entry->budget->reserve(0); }
    scalar(status, 0, xla_stream); return ffi::Error::Success();
  } catch (const std::exception& error) {
    if (entry) {
      if (cudaStreamSynchronize(entry->stream) != cudaSuccess)
        entry->budget->poison("cuEST failed operation stream drain failed");
      evict(entry);
    }
    phx::cuda::set_last_error(error.what());
    // Failure is an FFI error, not a successful result containing partial arrays.
    // The scalar is advisory only; Python must not consume failed results.
    if (status->element_count() == 1) { int32_t code = 1; cudaMemcpyAsync(status->typed_data(), &code, sizeof(code), cudaMemcpyHostToDevice, xla_stream); cudaStreamSynchronize(xla_stream); }
    if (dynamic_cast<const std::invalid_argument*>(&error)) return ffi::Error::InvalidArgument(error.what());
    return ffi::Error::Internal(error.what());
  }
}
#define INPUTS cudaStream_t stream, I64 binding_key, I32 ps, F64 pe, F64 pc, I32 as, F64 ae, F64 ac, F64 xyz, F64 charges
#define ATTRIBUTES int64_t representation, double fitting_cutoff, int64_t relative_conditioning, int64_t inversion, double screening, int64_t variable_workspace_bytes, int64_t workspace_limit_bytes, int64_t cache_capacity, int64_t partition
#define ACQUIRE acquire(stream, binding_key, ps, pe, pc, as, ae, ac, xyz, charges, Settings{representation, relative_conditioning, inversion, variable_workspace_bytes, workspace_limit_bytes, cache_capacity, fitting_cutoff, screening}, partition)
ffi::Error BindGeometry(INPUTS, RI32 status, ATTRIBUTES) {
  return invoke(stream, status, [&](std::shared_ptr<Entry>& ptr) { ptr = ACQUIRE; });
}
ffi::Error OneElectron(INPUTS, RF64 s, RF64 t, RF64 v, RI32 status, ATTRIBUTES) {
  return invoke(stream, status, [&](std::shared_ptr<Entry>& ptr) {
    ptr = ACQUIRE; Entry& e = *ptr;
    const int64_t n = e.input.primary.nao;
    shape(*s, n, n, "overlap result"); shape(*t, n, n, "kinetic result"); shape(*v, n, n, "potential result");
    allocate_scratch(e, false); const double* q = negative_charges(e, charges);
    auto& sp = e.parameters(CUEST_OVERLAPCOMPUTE_PARAMETERS);
    auto& tp = e.parameters(CUEST_KINETICCOMPUTE_PARAMETERS);
    auto& vp = e.parameters(CUEST_POTENTIALCOMPUTE_PARAMETERS);
    cuestWorkspaceDescriptor_t td{};
    CUEST(cuestOverlapComputeWorkspaceQuery(e.handle.value, e.oe.value, sp.value, &td, s->typed_data())); e.temporary.resize(td);
    CUEST(cuestOverlapCompute(e.handle.value, e.oe.value, sp.value, &e.temporary.value, s->typed_data())); e.sync();
    CUEST(cuestKineticComputeWorkspaceQuery(e.handle.value, e.oe.value, tp.value, &td, t->typed_data())); e.temporary.resize(td);
    CUEST(cuestKineticCompute(e.handle.value, e.oe.value, tp.value, &e.temporary.value, t->typed_data())); e.sync();
    CUEST(cuestPotentialComputeWorkspaceQuery(e.handle.value, e.oe.value, vp.value, &td, e.input.natom, xyz.typed_data(), q, v->typed_data())); e.temporary.resize(td);
    CUEST(cuestPotentialCompute(e.handle.value, e.oe.value, vp.value, &e.temporary.value, e.input.natom, xyz.typed_data(), q, v->typed_data()));
  });
}
ffi::Error JK(INPUTS, F64 density, F64 factors, RF64 j, RF64 k, RI32 status, ATTRIBUTES) {
  return invoke(stream, status, [&](std::shared_ptr<Entry>& ptr) {
    ptr = ACQUIRE; Entry& e = *ptr;
    shape(*j, e.input.primary.nao, e.input.primary.nao, "Coulomb result"); shape(*k, e.input.primary.nao, e.input.primary.nao, "exchange result");
    const uint64_t rank = density_shape(e, density, factors);
    if (!rank) { zero(*j, e.stream); zero(*k, e.stream); return; }
    ensure_df(e);
    auto& jp = e.parameters(CUEST_DFCOULOMBCOMPUTE_PARAMETERS);
    auto& kp = e.parameters(CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS);
    const cuestWorkspaceDescriptor_t variable{0, static_cast<size_t>(e.settings.variable)};
    cuestWorkspaceDescriptor_t td{};
    CUEST(cuestDFCoulombComputeWorkspaceQuery(e.handle.value, e.df.value, jp.value, &td, density.typed_data(), j->typed_data())); e.temporary.resize(td);
    CUEST(cuestDFCoulombCompute(e.handle.value, e.df.value, jp.value, &e.temporary.value, density.typed_data(), j->typed_data())); e.sync();
    CUEST(cuestDFSymmetricExchangeComputeWorkspaceQuery(e.handle.value, e.df.value, kp.value, &variable, &td, rank, factors.typed_data(), k->typed_data())); e.temporary.resize(td);
    CUEST(cuestDFSymmetricExchangeCompute(e.handle.value, e.df.value, kp.value, &variable, &e.temporary.value, rank, factors.typed_data(), k->typed_data()));
  });
}
ffi::Error Gradient(INPUTS, F64 density, F64 factors, F64 weighted, RF64 one, RF64 two, RF64 pulay, RI32 status, ATTRIBUTES) {
  return invoke(stream, status, [&](std::shared_ptr<Entry>& ptr) {
    ptr = ACQUIRE; Entry& e = *ptr;
    shape(*one, e.input.natom, 3, "one-electron gradient"); shape(*two, e.input.natom, 3, "two-electron gradient"); shape(*pulay, e.input.natom, 3, "overlap-Pulay gradient");
    shape(weighted, e.input.primary.nao, e.input.primary.nao, "energy-weighted density");
    const uint64_t rank = density_shape(e, density, factors);
    allocate_scratch(e, true); const double* q = negative_charges(e, charges);
    double* compact = e.scratch_data()+e.input.natom;
    double* charge_gradient = compact+3*e.input.natom;
    zero(*one, e.stream); zero(*two, e.stream); zero(*pulay, e.stream);
    auto& tp = e.parameters(CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS);
    auto& vp = e.parameters(CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS);
    auto& sp = e.parameters(CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS);
    cuestWorkspaceDescriptor_t td{};
    CUEST(cuestKineticDerivativeComputeWorkspaceQuery(e.handle.value, e.oe.value, tp.value, &td, density.typed_data(), compact)); e.temporary.resize(td);
    CUEST(cuestKineticDerivativeCompute(e.handle.value, e.oe.value, tp.value, &e.temporary.value, density.typed_data(), compact));
    scatter(e, compact, one->typed_data()); e.sync();
    CUEST(cuestPotentialDerivativeComputeWorkspaceQuery(e.handle.value, e.oe.value, vp.value, &td, e.input.natom, xyz.typed_data(), q, density.typed_data(), compact, charge_gradient)); e.temporary.resize(td);
    CUEST(cuestPotentialDerivativeCompute(e.handle.value, e.oe.value, vp.value, &e.temporary.value, e.input.natom, xyz.typed_data(), q, density.typed_data(), compact, charge_gradient));
    scatter(e, compact, one->typed_data()); axpy(e, 3*e.input.natom, 1, charge_gradient, one->typed_data()); e.sync();
    CUEST(cuestOverlapDerivativeComputeWorkspaceQuery(e.handle.value, e.oe.value, sp.value, &td, weighted.typed_data(), compact)); e.temporary.resize(td);
    CUEST(cuestOverlapDerivativeCompute(e.handle.value, e.oe.value, sp.value, &e.temporary.value, weighted.typed_data(), compact));
    scatter(e, compact, pulay->typed_data(), -1); e.sync();
    if (!rank) return;
    ensure_df(e);
    auto& dp = e.parameters(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS);
    const cuestWorkspaceDescriptor_t variable{0, static_cast<size_t>(e.settings.variable)};
    CUEST(cuestDFSymmetricDerivativeComputeWorkspaceQuery(e.handle.value, e.df.value, dp.value, &variable, &td, .5, density.typed_data(), -.25, 1, &rank, factors.typed_data(), compact)); e.temporary.resize(td);
    CUEST(cuestDFSymmetricDerivativeCompute(e.handle.value, e.df.value, dp.value, &variable, &e.temporary.value, .5, density.typed_data(), -.25, 1, &rank, factors.typed_data(), compact));
    scatter(e, compact, two->typed_data());
  });
}
#undef INPUTS
#undef ATTRIBUTES
#undef ACQUIRE
}  // namespace

phx::cuda::RegistryBase* component_registry() { return &registry(); }
int32_t component_shutdown() {
  std::lock_guard<std::recursive_mutex> lock(mutex());
  registry().release_all();
  index().items.clear();
  index().prune();
  for (const auto& item : index().budgets) {
    std::lock_guard<std::mutex> guard(item.second->mutex);
    if (item.second->cleanup_failed) return PHX_CUDA_PROVIDER_ERROR;
  }
  return PHX_CUDA_SUCCESS;
}
#define BASE_BIND ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<I64>().Arg<I32>().Arg<F64>().Arg<F64>().Arg<I32>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>()
#define SETTINGS_BIND .Attr<int64_t>("representation").Attr<double>("fitting_cutoff").Attr<int64_t>("relative_conditioning").Attr<int64_t>("inversion").Attr<double>("screening").Attr<int64_t>("variable_workspace_bytes").Attr<int64_t>("workspace_limit_bytes").Attr<int64_t>("cache_capacity").Attr<int64_t>("partition")
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cuest_bind_geometry(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cuest_one_electron(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cuest_jk(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cuest_gradient(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cuest_bind_geometry, BindGeometry,
  BASE_BIND.Ret<I32>() SETTINGS_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cuest_one_electron, OneElectron,
  BASE_BIND.Ret<F64>().Ret<F64>().Ret<F64>().Ret<I32>() SETTINGS_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cuest_jk, JK,
  BASE_BIND.Arg<F64>().Arg<F64>().Ret<F64>().Ret<F64>().Ret<I32>() SETTINGS_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cuest_gradient, Gradient,
  BASE_BIND.Arg<F64>().Arg<F64>().Arg<F64>().Ret<F64>().Ret<F64>().Ret<F64>().Ret<I32>() SETTINGS_BIND);
PHX_CUDA_HANDLER_TABLE({"phydrax_cuest_bind_geometry", phydrax_cuest_bind_geometry},
                       {"phydrax_cuest_one_electron", phydrax_cuest_one_electron},
                       {"phydrax_cuest_jk", phydrax_cuest_jk},
                       {"phydrax_cuest_gradient", phydrax_cuest_gradient})
