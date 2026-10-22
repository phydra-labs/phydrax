#include "phx_cuda_common.hpp"
#include "tensor_cleanup.hpp"
#include "tensor_registry.hpp"
#include <cutensor.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace ffi = xla::ffi;
namespace {

void check(cutensorStatus_t status, const char* operation) {
  if (status != CUTENSOR_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " + cutensorGetErrorString(status));
}
void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
uint64_t add_bytes(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::invalid_argument("cuTENSOR workspace byte count overflow");
  return a + b;
}
uint64_t aligned_bytes(uint64_t bytes) {
  return add_bytes(bytes, 255) & ~uint64_t{255};
}

struct Tokens {
  std::string_view text;
  int64_t next() {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
      text.remove_prefix(1);
    if (text.empty()) throw std::invalid_argument("cuTENSOR recipe is incomplete");
    int64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr == text.data() ||
        (parsed.ptr != text.data() + text.size() &&
         !std::isspace(static_cast<unsigned char>(*parsed.ptr))))
      throw std::invalid_argument("cuTENSOR recipe requires whitespace-separated int64 tokens");
    text.remove_prefix(static_cast<size_t>(parsed.ptr - text.data()));
    return value;
  }
  void finish() {
    for (unsigned char c : text)
      if (!std::isspace(c)) throw std::invalid_argument("cuTENSOR recipe has trailing tokens");
  }
};

struct Tensor {
  std::vector<int64_t> extents;
  std::vector<int64_t> strides;
  std::vector<int32_t> modes;
  uint64_t bytes = 0;
  cutensorTensorDescriptor_t descriptor = nullptr;

  void layout(size_t element_bytes) {
    strides.resize(extents.size());
    int64_t count = 1;
    for (size_t axis = extents.size(); axis-- > 0;) {
      const int64_t extent = extents[axis];
      if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent)
        throw std::invalid_argument("cuTENSOR extents must be positive with representable packed strides");
      strides[axis] = count;
      count *= extent;
    }
    if (static_cast<uint64_t>(count) > std::numeric_limits<size_t>::max() / element_bytes)
      throw std::invalid_argument("cuTENSOR tensor byte count overflow");
    bytes = static_cast<uint64_t>(count) * element_bytes;
  }
  bool contains(int32_t mode) const {
    return std::find(modes.begin(), modes.end(), mode) != modes.end();
  }
};
Tensor parse_tensor(Tokens& tokens, size_t element_bytes) {
  Tensor tensor;
  const int64_t rank = tokens.next();
  if (rank < 0 || rank > 64)
    throw std::invalid_argument("cuTENSOR supports tensor ranks from zero through 64");
  tensor.extents.reserve(static_cast<size_t>(rank));
  tensor.modes.reserve(static_cast<size_t>(rank));
  for (int64_t axis = 0; axis < rank; ++axis) tensor.extents.push_back(tokens.next());
  for (int64_t axis = 0; axis < rank; ++axis) {
    const int64_t mode = tokens.next();
    if (mode < std::numeric_limits<int32_t>::min() || mode > std::numeric_limits<int32_t>::max() ||
        tensor.contains(static_cast<int32_t>(mode)))
      throw std::invalid_argument("cuTENSOR modes must be distinct int32 IDs within each tensor");
    tensor.modes.push_back(static_cast<int32_t>(mode));
  }
  tensor.layout(element_bytes);
  return tensor;
}

struct Precision {
  cudaDataType_t storage;
  cutensorComputeDescriptor_t compute;
  size_t bytes;
};
Precision precision(ffi::DataType type) {
  switch (type) {
    case ffi::F32: return {CUDA_R_32F, CUTENSOR_COMPUTE_DESC_32F, sizeof(float)};
    case ffi::F64: return {CUDA_R_64F, CUTENSOR_COMPUTE_DESC_64F, sizeof(double)};
    case ffi::C64: return {CUDA_C_32F, CUTENSOR_COMPUTE_DESC_32F, sizeof(std::complex<float>)};
    case ffi::C128: return {CUDA_C_64F, CUTENSOR_COMPUTE_DESC_64F, sizeof(std::complex<double>)};
    default: throw std::invalid_argument("cuTENSOR requires homogeneous float32/64 or complex64/128 buffers");
  }
}

// Vendor host metadata has no byte-size query; registry capacity bounds its
// count, not its unknown host bytes. The partition budget counts every retained
// device scratch allocation, including evicted Entries still leased by calls.
struct Budget {
  std::mutex mutex;
  const int64_t workspace_limit, cache_limit, capacity;
  uint64_t used = 0;
  bool cleanup_failed = false;
  Budget(int64_t workspace, int64_t cache, int64_t count)
      : workspace_limit(workspace), cache_limit(cache), capacity(count) {}
};
struct Stage {
  enum class Kind { Permute, Reduce, Contract } kind;
  size_t a, b, out;
  cutensorOperationDescriptor_t operation = nullptr;
  cutensorPlan_t plan = nullptr;
  uint64_t workspace = 0;
};
struct Entry {
  std::mutex operation_mu;
  int device = -1;
  std::string recipe;
  ffi::DataType dtype = ffi::F32;
  int64_t workspace_limit = 0, cache_limit = 0;
  size_t input_count = 0;
  cutensorHandle_t handle = nullptr;
  cutensorPlanPreference_t preference = nullptr;
  std::vector<Tensor> tensors;
  std::vector<Stage> stages;
  std::array<uint64_t, 5> offsets{};
  uint64_t intermediate_bytes = 0, workspace_bytes = 0, scratch_bytes = 0;
  std::shared_ptr<Budget> budget;
  // Storage is reusable, values are not: each call reruns every intermediate
  // producer with beta=0 before consuming it, using only that call's inputs.
  void* scratch = nullptr;
  uint64_t charged_bytes = 0;

  void cleanup_error(const char* operation, const char* detail) noexcept {
    if (budget) {
      std::lock_guard<std::mutex> guard(budget->mutex);
      budget->cleanup_failed = true;
    }
    phx::cuda::set_cleanup_error(operation, detail);
  }
  bool cuda_cleanup(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return true;
    cleanup_error(operation, cudaGetErrorString(status));
    return false;
  }
  bool vendor_cleanup(cutensorStatus_t status, const char* operation) noexcept {
    if (status == CUTENSOR_STATUS_SUCCESS) return true;
    cleanup_error(operation, cutensorGetErrorString(status));
    return false;
  }
  bool get_device(int& value) noexcept { return cuda_cleanup(cudaGetDevice(&value), "cudaGetDevice(cuTENSOR cleanup)"); }
  bool set_device(int value) noexcept { return cuda_cleanup(cudaSetDevice(value), "cudaSetDevice(cuTENSOR cleanup)"); }
  bool drain() noexcept { return cuda_cleanup(cudaDeviceSynchronize(), "cudaDeviceSynchronize(cuTENSOR cleanup)"); }
  ~Entry() noexcept {
    phx::cuda::TensorCleanup<Entry> cleanup(*this, device);
    if (!cleanup.ready()) return;
    for (auto& stage : stages) {
      if (stage.plan && !vendor_cleanup(cutensorDestroyPlan(stage.plan), "cutensorDestroyPlan")) return;
      if (stage.operation && !vendor_cleanup(cutensorDestroyOperationDescriptor(stage.operation), "cutensorDestroyOperationDescriptor")) return;
    }
    if (preference && !vendor_cleanup(cutensorDestroyPlanPreference(preference), "cutensorDestroyPlanPreference")) return;
    for (auto& tensor : tensors)
      if (tensor.descriptor && !vendor_cleanup(cutensorDestroyTensorDescriptor(tensor.descriptor), "cutensorDestroyTensorDescriptor")) return;
    if (handle && !vendor_cleanup(cutensorDestroy(handle), "cutensorDestroy")) return;
    // Opaque owners must be gone before their scratch and charge can retire.
    cleanup.release_scratch(scratch, charged_bytes,
        [this](void* pointer) noexcept { return cuda_cleanup(cudaFree(pointer), "cudaFree(cuTENSOR scratch)"); },
        [this](uint64_t bytes) noexcept {
          std::lock_guard<std::mutex> guard(budget->mutex);
          budget->used -= bytes;
        });
  }
};
phx::cuda::Registry<Entry>& registry() {
  static auto* instance = new phx::cuda::Registry<Entry>();
  return *instance;
}
struct IndexItem {
  int device;
  int64_t partition;
  int32_t id;
  std::weak_ptr<Entry> entry;
};
struct Index {
  std::mutex mutex;
  std::vector<IndexItem> items;
  std::map<std::pair<int, int64_t>, std::shared_ptr<Budget>> budgets;
  void prune_budgets() {
    for (auto item = budgets.begin(); item != budgets.end();) {
      const auto budget = item->second;
      std::lock_guard<std::mutex> guard(budget->mutex);
      if (budget.use_count() == 2 && budget->used == 0 && !budget->cleanup_failed) item = budgets.erase(item);
      else ++item;
    }
  }
};
Index& index() {
  static auto* instance = new Index();
  return *instance;
}

void tensor_context(std::ostream& out, size_t slot, const Tensor& tensor) {
  out << " tensor[" << slot << ":" << (slot < 2 ? "input" : slot == 2 ? "output" : "reduction") << "] extents=[";
  for (size_t axis = 0; axis < tensor.extents.size(); ++axis) {
    if (axis) out << ',';
    out << tensor.extents[axis];
  }
  out << "] strides=[";
  for (size_t axis = 0; axis < tensor.strides.size(); ++axis) {
    if (axis) out << ',';
    out << tensor.strides[axis];
  }
  out << "] modes=[";
  for (size_t axis = 0; axis < tensor.modes.size(); ++axis) {
    if (axis) out << ',';
    out << tensor.modes[axis];
  }
  out << ']';
}
void describe(Entry& entry, size_t tensor, Precision p) {
  auto& t = entry.tensors[tensor];
  // cuTENSOR's byte alignment must be a multiple of the storage element size.
  // Natural element alignment is pointer-independent and checked on every bind.
  const auto alignment = static_cast<uint32_t>(p.bytes);
  const auto status = cutensorCreateTensorDescriptor(entry.handle, &t.descriptor,
      static_cast<uint32_t>(t.extents.size()), t.extents.data(), t.strides.data(),
      p.storage, alignment);
  if (status != CUTENSOR_STATUS_SUCCESS) {
    std::ostringstream context;
    context << "cutensorCreateTensorDescriptor: " << cutensorGetErrorString(status)
            << "; device=" << entry.device << " cuda_dtype=" << static_cast<int>(p.storage)
            << " element_bytes=" << p.bytes << " alignment_bytes=" << alignment;
    tensor_context(context, tensor, t);
    throw std::runtime_error(context.str());
  }
}
void add_stage(Entry& entry, Stage::Kind kind, size_t a, size_t b, size_t out, Precision p) {
  entry.stages.push_back({kind, a, b, out});
  auto& stage = entry.stages.back();
  const auto& ta = entry.tensors[a];
  const auto& td = entry.tensors[out];
  try {
  if (kind == Stage::Kind::Permute) {
    check(cutensorCreatePermutation(entry.handle, &stage.operation,
          ta.descriptor, ta.modes.data(), CUTENSOR_OP_IDENTITY,
          td.descriptor, td.modes.data(), p.compute), "cutensorCreatePermutation");
  } else if (kind == Stage::Kind::Reduce) {
    check(cutensorCreateReduction(entry.handle, &stage.operation,
          ta.descriptor, ta.modes.data(), CUTENSOR_OP_IDENTITY,
          td.descriptor, td.modes.data(), CUTENSOR_OP_IDENTITY,
          td.descriptor, td.modes.data(), CUTENSOR_OP_ADD, p.compute), "cutensorCreateReduction");
  } else {
    const auto& tb = entry.tensors[b];
    check(cutensorCreateContraction(entry.handle, &stage.operation,
          ta.descriptor, ta.modes.data(), CUTENSOR_OP_IDENTITY,
          tb.descriptor, tb.modes.data(), CUTENSOR_OP_IDENTITY,
          td.descriptor, td.modes.data(), CUTENSOR_OP_IDENTITY,
          td.descriptor, td.modes.data(), p.compute), "cutensorCreateContraction");
  }
  const uint64_t available = static_cast<uint64_t>(entry.workspace_limit) - entry.intermediate_bytes;
  uint64_t minimum = 0, recommended = 0;
  check(cutensorEstimateWorkspaceSize(entry.handle, stage.operation, entry.preference,
        CUTENSOR_WORKSPACE_MIN, &minimum), "cutensorEstimateWorkspaceSize(minimum)");
  if (minimum > available)
    throw std::invalid_argument("cuTENSOR minimum workspace plus intermediates exceeds workspace_limit_bytes");
  check(cutensorEstimateWorkspaceSize(entry.handle, stage.operation, entry.preference,
        CUTENSOR_WORKSPACE_DEFAULT, &recommended), "cutensorEstimateWorkspaceSize(default)");
  check(cutensorCreatePlan(entry.handle, &stage.plan, stage.operation, entry.preference,
        std::min(available, std::max(minimum, recommended))), "cutensorCreatePlan");
  check(cutensorPlanGetAttribute(entry.handle, stage.plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &stage.workspace, sizeof(stage.workspace)), "cutensorPlanGetAttribute(required_workspace)");
  if (stage.workspace > available)
    throw std::runtime_error("cuTENSOR plan exceeds its workspace limit");
  entry.workspace_bytes = std::max(entry.workspace_bytes, stage.workspace);
  } catch (const std::runtime_error& failure) {
    std::ostringstream context;
    context << "cuTENSOR " << (kind == Stage::Kind::Permute ? "permutation" :
        kind == Stage::Kind::Reduce ? "reduction" : "contraction")
        << " stage=" << entry.stages.size() - 1 << " device=" << entry.device
        << " cuda_dtype=" << static_cast<int>(p.storage) << ':';
    tensor_context(context, a, ta);
    if (kind == Stage::Kind::Contract) tensor_context(context, b, entry.tensors[b]);
    tensor_context(context, out, td);
    context << "; " << failure.what();
    throw std::runtime_error(context.str());
  }
}

std::shared_ptr<Entry> create_entry(int device, std::string_view recipe, ffi::DataType dtype,
                                  int64_t workspace_limit, int64_t cache_limit, size_t inputs,
                                  const std::shared_ptr<Budget>& budget) {
  const Precision p = precision(dtype);
  auto entry = std::make_shared<Entry>();
  entry->device = device;
  entry->budget = budget;
  entry->recipe = recipe;
  entry->dtype = dtype;
  entry->workspace_limit = workspace_limit;
  entry->cache_limit = cache_limit;
  Tokens tokens{recipe};
  const int64_t count = tokens.next();
  if ((count != 1 && count != 2) || count != static_cast<int64_t>(inputs))
    throw std::invalid_argument("cuTENSOR requires one or two input buffers matching its recipe");
  entry->input_count = inputs;
  entry->tensors.reserve(5);
  entry->stages.reserve(3);
  for (size_t i = 0; i < inputs; ++i) entry->tensors.push_back(parse_tensor(tokens, p.bytes));
  // Fixed tensor slots: inputs at 0/1, result at 2, optional reductions at 3/4.
  if (inputs == 1) entry->tensors.emplace_back();
  entry->tensors.push_back(parse_tensor(tokens, p.bytes));
  if (tokens.next() != 0)
    throw std::invalid_argument("cuTENSOR executes individual unary/binary operations; recipe path must be empty");
  tokens.finish();
  std::map<int32_t, int64_t> extents;
  for (size_t i = 0; i < inputs; ++i) {
    const auto& t = entry->tensors[i];
    for (size_t axis = 0; axis < t.modes.size(); ++axis) {
      const auto [found, inserted] = extents.emplace(t.modes[axis], t.extents[axis]);
      if (!inserted && found->second != t.extents[axis])
        throw std::invalid_argument("cuTENSOR shared mode extents disagree");
    }
  }
  const auto& output = entry->tensors[2];
  for (size_t axis = 0; axis < output.modes.size(); ++axis) {
    const auto found = extents.find(output.modes[axis]);
    if (found == extents.end() || found->second != output.extents[axis])
      throw std::invalid_argument("cuTENSOR output modes and extents must be drawn from the inputs");
  }
  std::array<size_t, 2> operands{0, 1};
  if (inputs == 2) {
    for (size_t i = 0; i < 2; ++i) {
      Tensor reduced;
      const auto& source = entry->tensors[i];
      const auto& other = entry->tensors[1 - i];
      for (size_t axis = 0; axis < source.modes.size(); ++axis) {
        const int32_t mode = source.modes[axis];
        if (other.contains(mode) || output.contains(mode)) {
          reduced.modes.push_back(mode);
          reduced.extents.push_back(source.extents[axis]);
        }
      }
      if (reduced.modes.size() != source.modes.size()) {
        reduced.layout(p.bytes);
        operands[i] = entry->tensors.size();
        entry->offsets[operands[i]] = entry->intermediate_bytes;
        entry->intermediate_bytes = add_bytes(entry->intermediate_bytes, aligned_bytes(reduced.bytes));
        entry->tensors.push_back(std::move(reduced));
      }
    }
  }
  if (entry->intermediate_bytes > static_cast<uint64_t>(workspace_limit))
    throw std::invalid_argument("cuTENSOR reduction intermediates exceed workspace_limit_bytes");
  check(cutensorCreate(&entry->handle), "cutensorCreate");
  check(cutensorHandleResizePlanCache(entry->handle, 0), "cutensorHandleResizePlanCache");
  check(cutensorCreatePlanPreference(entry->handle, &entry->preference,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), "cutensorCreatePlanPreference");
  const cutensorCacheMode_t cache_mode = CUTENSOR_CACHE_MODE_NONE;
  check(cutensorPlanPreferenceSetAttribute(entry->handle, entry->preference,
        CUTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cache_mode, sizeof(cache_mode)),
        "cutensorPlanPreferenceSetAttribute(cache_mode)");
  for (size_t i = 0; i < entry->tensors.size(); ++i)
    if (i != 1 || inputs == 2) describe(*entry, i, p);
  if (inputs == 1) {
    const auto kind = entry->tensors[0].modes.size() == output.modes.size()
        ? Stage::Kind::Permute : Stage::Kind::Reduce;
    add_stage(*entry, kind, 0, 0, 2, p);
  } else {
    for (size_t i = 0; i < 2; ++i)
      if (operands[i] != i) add_stage(*entry, Stage::Kind::Reduce, i, 0, operands[i], p);
    add_stage(*entry, Stage::Kind::Contract, operands[0], operands[1], 2, p);
  }
  entry->scratch_bytes = add_bytes(entry->intermediate_bytes, entry->workspace_bytes);
  if (entry->scratch_bytes) {
    {
      std::lock_guard<std::mutex> guard(entry->budget->mutex);
      if (entry->scratch_bytes > static_cast<uint64_t>(cache_limit) - entry->budget->used)
        throw std::invalid_argument("cuTENSOR retained scratch exceeds partition cache_limit_bytes; release memos or allow overlap space");
      entry->budget->used += entry->scratch_bytes;
      entry->charged_bytes = entry->scratch_bytes;
    }
    check_cuda(cudaMalloc(&entry->scratch, static_cast<size_t>(entry->scratch_bytes)),
               "cudaMalloc(cuTENSOR retained scratch)");
  }
  return entry;
}

std::shared_ptr<Entry> memo(int device, int64_t partition, int64_t capacity,
                            std::string_view recipe, ffi::DataType dtype,
                            int64_t workspace_limit, int64_t cache_limit, size_t inputs) {
  auto& idx = index();
  std::lock_guard<std::mutex> guard(idx.mutex);
  auto& items = idx.items;
  // Weak indexing retains neither recipes nor descriptors after registry
  // eviction. Pruning does not change the registry's LRU order.
  items.erase(std::remove_if(items.begin(), items.end(), [](const IndexItem& item) {
    return item.entry.expired() || registry().find_device(item.partition, item.id) != item.device;
  }), items.end());
  idx.prune_budgets();
  const auto key = std::make_pair(device, partition);
  auto found_budget = idx.budgets.find(key);
  if (found_budget == idx.budgets.end()) {
    found_budget = idx.budgets.emplace(key,
        std::make_shared<Budget>(workspace_limit, cache_limit, capacity)).first;
  }
  const auto budget = found_budget->second;
  if (budget->workspace_limit != workspace_limit || budget->cache_limit != cache_limit ||
      budget->capacity != capacity)
    throw std::invalid_argument("cuTENSOR partition workspace/cache/capacity policy is immutable while memos or calls remain live");
  {
    std::lock_guard<std::mutex> budget_guard(budget->mutex);
    if (budget->cleanup_failed) throw std::runtime_error("cuTENSOR partition scratch cleanup failed");
  }
  for (auto& item : items) {
    if (item.device != device || item.partition != partition) continue;
    auto entry = item.entry.lock();
    if (entry && entry->recipe == recipe && entry->dtype == dtype &&
        entry->workspace_limit == workspace_limit && entry->cache_limit == cache_limit &&
        entry->input_count == inputs &&
        registry().find(device, partition, item.id) == entry) {
      return entry;
    }
  }
  auto entry = create_entry(device, recipe, dtype, workspace_limit, cache_limit, inputs, budget);
  const int32_t id = registry().insert(device, partition, capacity, entry);
  registry().count_rebuild(device, partition);
  items.push_back({device, partition, id, entry});
  return entry;
}

void validate_buffer(ffi::AnyBuffer buffer, const Tensor& tensor, ffi::DataType dtype) {
  const auto dims = buffer.dimensions();
  if (buffer.element_type() != dtype || dims.size() != tensor.extents.size() ||
      !std::equal(dims.begin(), dims.end(), tensor.extents.begin()))
    throw std::invalid_argument("cuTENSOR buffer dtype/shape differs from recipe");
  if (!buffer.untyped_data()) throw std::invalid_argument("cuTENSOR requires non-null tensor buffers");
  const size_t alignment = precision(dtype).bytes;
  if (reinterpret_cast<uintptr_t>(buffer.untyped_data()) % alignment != 0)
    throw std::invalid_argument("cuTENSOR buffer pointer violates its natural storage-element alignment");
}
bool overlaps(const void* a, uint64_t a_bytes, const void* b, uint64_t b_bytes) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x <= y ? y - x < a_bytes : x - y < b_bytes;
}
struct StreamDrain {
  cudaStream_t stream;
  const std::shared_ptr<Entry>& entry;
  bool drained = false;
  ~StreamDrain() noexcept { if (!drained) finish(); }
  cudaError_t finish() noexcept {
    drained = true;
    const auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      if (entry) entry->cuda_cleanup(status, "cudaStreamSynchronize(cuTENSOR)");
      else phx::cuda::set_cleanup_error("cudaStreamSynchronize(cuTENSOR)", cudaGetErrorString(status));
    }
    return status;
  }
};
struct Scalars {
  const float f_one = 1, f_zero = 0;
  const double d_one = 1, d_zero = 0;
  const std::complex<float> c_one{1, 0}, c_zero{0, 0};
  const std::complex<double> z_one{1, 0}, z_zero{0, 0};
  const void* get(ffi::DataType type, bool one) const {
    switch (type) {
      case ffi::F32: return one ? &f_one : &f_zero;
      case ffi::F64: return one ? &d_one : &d_zero;
      case ffi::C64: return one ? &c_one : &c_zero;
      case ffi::C128: return one ? &z_one : &z_zero;
      default: throw std::invalid_argument("cuTENSOR scalar dtype unsupported");
    }
  }
};

ffi::Error contract(cudaStream_t stream, ffi::RemainingArgs args, ffi::Result<ffi::AnyBuffer> result,
                    std::string_view recipe, int64_t workspace_limit_bytes,
                    int64_t cache_limit_bytes, int64_t partition, int64_t capacity) {
  std::shared_ptr<Entry> entry;
  std::unique_lock<std::mutex> operation;
  Scalars scalars;
  // Declared last so every exit drains before releasing scalars, descriptor/
  // scratch leases, or the per-entry serialization lock.
  StreamDrain drain{stream, entry};
  ffi::Error error = ffi::Error::Success();
  try {
    if (capacity < 1 || workspace_limit_bytes < 0 || cache_limit_bytes < 0)
      throw std::invalid_argument("cuTENSOR requires capacity >= 1 and nonnegative workspace/cache byte limits");
    if (args.size() != 1 && args.size() != 2)
      throw std::invalid_argument("cuTENSOR accepts one or two numeric inputs");
    const auto dtype = result->element_type();
    precision(dtype);
    std::array<std::optional<ffi::AnyBuffer>, 2> inputs;
    for (size_t i = 0; i < args.size(); ++i) {
      auto value = args.get<ffi::AnyBuffer>(i);
      if (!value) throw std::invalid_argument("cuTENSOR inputs must be numeric buffers");
      inputs[i] = *value;
    }
    entry = memo(phx::cuda::current_device(), partition, capacity, recipe, dtype,
                 workspace_limit_bytes, cache_limit_bytes, args.size());
    operation = std::unique_lock<std::mutex>(entry->operation_mu);
    validate_buffer(*result, entry->tensors[2], dtype);
    for (size_t i = 0; i < args.size(); ++i) {
      validate_buffer(*inputs[i], entry->tensors[i], dtype);
      if (overlaps(inputs[i]->untyped_data(), entry->tensors[i].bytes,
                   result->untyped_data(), entry->tensors[2].bytes))
        throw std::invalid_argument("cuTENSOR input and output buffers must not overlap");
    }
    std::array<void*, 5> pointers{};
    for (size_t i = 0; i < args.size(); ++i) pointers[i] = inputs[i]->untyped_data();
    pointers[2] = result->untyped_data();
    for (size_t i = 3; i < entry->tensors.size(); ++i)
      pointers[i] = static_cast<char*>(entry->scratch) + entry->offsets[i];
    void* workspace = entry->workspace_bytes
        ? static_cast<char*>(entry->scratch) + entry->intermediate_bytes : nullptr;
    const void* alpha = scalars.get(dtype, true);
    const void* beta = scalars.get(dtype, false);
    for (const auto& stage : entry->stages) {
      if (stage.kind == Stage::Kind::Permute) {
        check(cutensorPermute(entry->handle, stage.plan, alpha, pointers[stage.a],
              pointers[stage.out], stream), "cutensorPermute");
      } else if (stage.kind == Stage::Kind::Reduce) {
        check(cutensorReduce(entry->handle, stage.plan, alpha, pointers[stage.a], beta,
              pointers[stage.out], pointers[stage.out], workspace, stage.workspace, stream), "cutensorReduce");
      } else {
        check(cutensorContract(entry->handle, stage.plan, alpha, pointers[stage.a], pointers[stage.b],
              beta, pointers[stage.out], pointers[stage.out], workspace, stage.workspace, stream), "cutensorContract");
      }
    }
  } catch (const std::invalid_argument& failure) {
    error = ffi::Error::InvalidArgument(failure.what());
  } catch (const std::exception& failure) {
    error = ffi::Error::Internal(failure.what());
  } catch (...) {
    error = ffi::Error::Internal("Unknown cuTENSOR failure");
  }
  const cudaError_t drained = drain.finish();
  if (drained != cudaSuccess) {
    const std::string prior = error.failure() ? error.message() + "; " : "";
    error = ffi::Error::Internal(prior + "cudaStreamSynchronize(cuTENSOR): " + cudaGetErrorString(drained));
  }
  if (error.failure()) phx::cuda::set_last_error(error.message());
  return error;
}
}  // namespace

phx::cuda::RegistryBase* component_registry() {
  static auto* view = new phx::cuda::TensorRegistry(registry(), [](bool all, int64_t partition) {
    std::lock_guard<std::mutex> guard(index().mutex);
    for (const auto& item : index().budgets) {
      if (!all && item.first.second != partition) continue;
      std::lock_guard<std::mutex> budget_guard(item.second->mutex);
      if (item.second->cleanup_failed)
        phx::cuda::set_cleanup_error("cuTENSOR partition cleanup failed");
    }
  });
  return view;
}
int32_t component_shutdown() {
  std::lock_guard<std::mutex> guard(index().mutex);
  registry().release_all();
  index().items.clear();
  index().prune_budgets();
  for (const auto& item : index().budgets) {
    std::lock_guard<std::mutex> budget_guard(item.second->mutex);
    if (item.second->cleanup_failed) {
      phx::cuda::set_cleanup_error("cuTENSOR partition cleanup failed");
      return PHX_CUDA_PROVIDER_ERROR;
    }
  }
  return PHX_CUDA_SUCCESS;
}
extern "C" {
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cutensor_contract(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cutensor_contract, contract,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs()
    .Ret<ffi::AnyBuffer>().Attr<std::string_view>("recipe")
    .Attr<int64_t>("workspace_limit_bytes").Attr<int64_t>("cache_limit_bytes")
    .Attr<int64_t>("partition").Attr<int64_t>("capacity"));
}
PHX_CUDA_HANDLER_TABLE({"phydrax_cutensor_contract", phydrax_cutensor_contract})
