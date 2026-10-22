// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// cuTensorNet 2.14 network-centric API; no implicit optimizer in execution.
#include "phx_cuda_common.hpp"
#include "tensor_cleanup.hpp"
#include "tensor_registry.hpp"
#include <cutensornet.h>
#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ffi = xla::ffi;
static_assert(CUTENSORNET_VERSION == 21400, "cuTensorNet 2.14.0 headers are required");
namespace {
constexpr size_t kMaxRecipeBytes = 1024 * 1024;
constexpr int64_t kMaxInputs = 4096;
constexpr int64_t kMaxRank = 64;
void check_status(cutensornetStatus_t status, const char* operation) {
  if (status != CUTENSORNET_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " + cutensornetGetErrorString(status));
}
#define PHX_CUTN_CHECK(operation) check_status((operation), #operation)
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Tokens {
  std::string_view text;
  int64_t next() {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) throw std::invalid_argument("Truncated tensor recipe");
    text.remove_prefix(first);
    const auto end = text.find_first_of(" \t\r\n");
    const auto token = text.substr(0, end);
    int64_t value = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size())
      throw std::invalid_argument("Tensor recipe requires integer tokens");
    text.remove_prefix(token.size());
    return value;
  }
  void finish() const {
    if (text.find_first_not_of(" \t\r\n") != std::string_view::npos)
      throw std::invalid_argument("Trailing tensor recipe tokens");
  }
};
struct Tensor {
  std::vector<int64_t> extents, strides;
  std::vector<int32_t> modes;
};
Tensor tensor(Tokens& tokens, std::map<int32_t, int64_t>& dimensions, bool output) {
  const int64_t rank = tokens.next();
  if (rank < 0 || rank > kMaxRank) throw std::invalid_argument("Tensor rank must be in [0,64]");
  Tensor result;
  result.extents.resize(rank); result.strides.resize(rank); result.modes.resize(rank);
  for (auto& extent : result.extents) {
    extent = tokens.next();
    if (extent <= 0) throw std::invalid_argument("Tensor extents must be positive");
  }
  std::map<int32_t, bool> seen;
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t mode = tokens.next();
    if (mode < std::numeric_limits<int32_t>::min() || mode > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("Tensor modes must fit int32");
    result.modes[i] = static_cast<int32_t>(mode);
    if (!seen.emplace(result.modes[i], true).second)
      throw std::invalid_argument("Repeated modes must be normalized before cuTensorNet");
    const auto found = dimensions.find(result.modes[i]);
    if (output && found == dimensions.end()) throw std::invalid_argument("Output mode absent from inputs");
    if (found != dimensions.end() && found->second != result.extents[i])
      throw std::invalid_argument("Shared mode extents disagree");
    dimensions.emplace(result.modes[i], result.extents[i]);
  }
  int64_t stride = 1;
  for (int64_t i = rank; i-- > 0;) {
    result.strides[i] = stride;
    if (result.extents[i] > std::numeric_limits<int64_t>::max() / stride)
      throw std::invalid_argument("Tensor element count overflows int64");
    stride *= result.extents[i];
  }
  return result;
}
struct Recipe {
  std::vector<Tensor> inputs;
  Tensor output;
  std::vector<cutensornetNodePair_t> path;
};
Recipe parse(std::string_view text, bool planner) {
  if (text.empty() || text.size() > kMaxRecipeBytes)
    throw std::invalid_argument("Tensor recipe must contain at most 1 MiB");
  Tokens tokens{text};
  const int64_t count = tokens.next();
  if (count < 1 || count > kMaxInputs) throw std::invalid_argument("Input count must be in [1,4096]");
  Recipe result;
  result.inputs.reserve(count);
  std::map<int32_t, int64_t> dimensions;
  for (int64_t i = 0; i < count; ++i) result.inputs.push_back(tensor(tokens, dimensions, false));
  result.output = tensor(tokens, dimensions, true);
  const int64_t steps = tokens.next();
  if (steps < 0 || steps > count) throw std::invalid_argument("Invalid contraction path length");
  if (planner && steps != 0) throw std::invalid_argument("Planner requires an empty input path");
  if (!planner && count > 1 && steps != count - 1)
    throw std::invalid_argument("Fixed cuTensorNet path requires input_count-1 binary steps");
  if (!planner && count == 1 && steps > 1) throw std::invalid_argument("Invalid unary path");
  int64_t live = count;
  result.path.reserve(steps);
  for (int64_t i = 0; i < steps; ++i) {
    const int64_t arity = tokens.next();
    if (count == 1 && arity == 1 && tokens.next() == 0) continue;
    if (arity != 2 || live < 2) throw std::invalid_argument("cuTensorNet requires a binary path");
    const int64_t first = tokens.next(), second = tokens.next();
    if (first < 0 || second < 0 || first >= live || second >= live || first == second)
      throw std::invalid_argument("Contraction path index is outside the evolving live list");
    result.path.push_back({static_cast<int32_t>(first), static_cast<int32_t>(second)});
    --live;
  }
  tokens.finish();
  return result;
}
std::optional<Recipe> unit_axis_recipe(const Recipe& recipe) {
  const bool needs_unit = recipe.output.modes.empty() ||
      std::any_of(recipe.inputs.begin(), recipe.inputs.end(),
          [](const Tensor& input) { return input.modes.empty(); });
  if (!needs_unit) return std::nullopt;
  std::set<int32_t> used;
  for (const auto& input : recipe.inputs) used.insert(input.modes.begin(), input.modes.end());
  int64_t candidate = 0;
  while (candidate <= std::numeric_limits<int32_t>::max() &&
         used.find(static_cast<int32_t>(candidate)) != used.end()) ++candidate;
  if (candidate > std::numeric_limits<int32_t>::max())
    throw std::invalid_argument("cuTensorNet has no free private unit-axis label");
  Recipe lifted = recipe;
  const auto mode = static_cast<int32_t>(candidate);
  // Rank-zero inputs are not accepted by NetworkAppendTensor. A retained shared
  // unit axis lifts every descriptor without changing storage or contraction.
  for (auto& input : lifted.inputs) {
    input.extents.push_back(1); input.strides.push_back(1); input.modes.push_back(mode);
  }
  lifted.output.extents.push_back(1);
  lifted.output.strides.push_back(1);
  lifted.output.modes.push_back(mode);
  return lifted;
}
struct Type {
  cudaDataType_t storage;
  cutensornetComputeType_t compute;
};
Type type(ffi::DataType value) {
  switch (value) {
    case ffi::DataType::F32: return {CUDA_R_32F, CUTENSORNET_COMPUTE_32F};
    case ffi::DataType::F64: return {CUDA_R_64F, CUTENSORNET_COMPUTE_64F};
    case ffi::DataType::C64: return {CUDA_C_32F, CUTENSORNET_COMPUTE_32F};
    case ffi::DataType::C128: return {CUDA_C_64F, CUTENSORNET_COMPUTE_64F};
    default: throw std::invalid_argument("cuTensorNet supports only float32/64 and complex64/128");
  }
}
void validate(ffi::AnyBuffer buffer, const Tensor& expected, ffi::DataType dtype) {
  const auto shape = buffer.dimensions();
  if (buffer.element_type() != dtype || shape.size() != expected.extents.size() ||
      !std::equal(shape.begin(), shape.end(), expected.extents.begin()))
    throw std::invalid_argument("cuTensorNet buffer shape or homogeneous dtype disagrees with recipe");
}
struct Budget {
  const int64_t workspace_limit, cache_limit, capacity;
  int64_t used = 0;
  bool cleanup_failed = false;
  std::mutex mutex;
  Budget(int64_t workspace, int64_t cache, int64_t entries)
      : workspace_limit(workspace), cache_limit(cache), capacity(entries) {}
  void reserve(int64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    if (cleanup_failed) throw std::runtime_error("cuTensorNet partition scratch cleanup failed");
    if (bytes < 0 || bytes > cache_limit - used)
      throw std::invalid_argument("cuTensorNet retained device scratch exceeds partition cache_limit_bytes, including active evicted leases");
    used += bytes;
  }
  void release(int64_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    used -= bytes;
  }
};
struct Entry {
  int device;
  Recipe recipe;
  std::optional<Recipe> lifted_recipe;
  std::string recipe_text;
  cutensornetHandle_t handle = nullptr;
  cutensornetNetworkDescriptor_t network = nullptr;
  cutensornetContractionOptimizerInfo_t info = nullptr;
  cutensornetContractionOptimizerConfig_t config = nullptr;
  cutensornetWorkspaceDescriptor_t workspace = nullptr;
  std::vector<int64_t> tensor_ids;
  int64_t device_scratch = 0, host_scratch = 0;
  std::shared_ptr<Budget> budget;
  void* device_buffer = nullptr;
  std::unique_ptr<unsigned char[]> host_buffer;
  int64_t charged_bytes = 0;
  bool prepared = false;
  bool cleanup_started = false;
  std::atomic<bool> cleanup_failed{false};
  std::atomic<bool> invalid{false};
  std::mutex mutex;
  explicit Entry(int device_, Recipe recipe_)
      : device(device_), recipe(std::move(recipe_)), lifted_recipe(unit_axis_recipe(recipe)) {}
  const Recipe& descriptors() const { return lifted_recipe ? *lifted_recipe : recipe; }
  void cleanup_error(const char* operation, const char* detail) noexcept {
    cleanup_failed = true;
    if (budget) {
      std::lock_guard<std::mutex> lock(budget->mutex);
      budget->cleanup_failed = true;
    }
    phx::cuda::set_cleanup_error(operation, detail);
  }
  bool cuda_cleanup(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return true;
    cleanup_error(operation, cudaGetErrorString(status));
    return false;
  }
  bool vendor_cleanup(cutensornetStatus_t status, const char* operation) noexcept {
    if (status == CUTENSORNET_STATUS_SUCCESS) return true;
    cleanup_error(operation, cutensornetGetErrorString(status));
    return false;
  }
  bool get_device(int& value) noexcept { return cuda_cleanup(cudaGetDevice(&value), "cudaGetDevice(cuTensorNet cleanup)"); }
  bool set_device(int value) noexcept { return cuda_cleanup(cudaSetDevice(value), "cudaSetDevice(cuTensorNet cleanup)"); }
  bool drain() noexcept { return cuda_cleanup(cudaDeviceSynchronize(), "cudaDeviceSynchronize(cuTensorNet cleanup)"); }
  ~Entry() noexcept { close(); }
  void close() noexcept {
    if (cleanup_started) return;
    cleanup_started = true;
    phx::cuda::TensorCleanup<Entry> cleanup(*this, device);
    if (!cleanup.ready()) {
      // Device work may still reference host workspace; retain it as well.
      host_buffer.release();
      return;
    }
    if ((workspace && !vendor_cleanup(cutensornetDestroyWorkspaceDescriptor(workspace), "cutensornetDestroyWorkspaceDescriptor")) ||
        (config && !vendor_cleanup(cutensornetDestroyContractionOptimizerConfig(config), "cutensornetDestroyContractionOptimizerConfig")) ||
        (info && !vendor_cleanup(cutensornetDestroyContractionOptimizerInfo(info), "cutensornetDestroyContractionOptimizerInfo")) ||
        (network && !vendor_cleanup(cutensornetDestroyNetwork(network), "cutensornetDestroyNetwork")) ||
        (handle && !vendor_cleanup(cutensornetDestroy(handle), "cutensornetDestroy"))) {
      // An opaque owner may still reference either workspace. Abandon host
      // ownership so member destruction cannot free it; retain the device charge.
      // cleanup_started prevents retrying partially completed vendor teardown.
      host_buffer.release();
      return;
    }
    cleanup.release_scratch(device_buffer, charged_bytes,
        [this](void* pointer) noexcept { return cuda_cleanup(cudaFree(pointer), "cudaFree(cuTensorNet scratch)"); },
        [this](int64_t bytes) noexcept { budget->release(bytes); });
    // Host storage has no device budget charge and no surviving opaque owner.
    host_buffer.reset();
  }
};
phx::cuda::Registry<Entry>& registry() {
  static auto* value = new phx::cuda::Registry<Entry>;
  return *value;
}
struct MemoId {
  int device;
  int64_t partition, capacity;
  ffi::DataType dtype;
  int32_t id;
  std::weak_ptr<Entry> entry;
};
std::vector<MemoId>& index() {
  static auto* value = new std::vector<MemoId>;
  return *value;
}
void prune_index() {
  auto& items = index();
  items.erase(std::remove_if(items.begin(), items.end(), [](const MemoId& item) {
    return item.entry.expired() || registry().find_device(item.partition, item.id) != item.device;
  }), items.end());
}
std::mutex& index_mutex() { static auto* value = new std::mutex; return *value; }
using BudgetKey = std::pair<int, int64_t>;
std::map<BudgetKey, std::shared_ptr<Budget>>& budgets() {
  static auto* value = new std::map<BudgetKey, std::shared_ptr<Budget>>;
  return *value;
}
void prune_budgets() {
  for (auto it = budgets().begin(); it != budgets().end();) {
    const auto budget = it->second;
    std::lock_guard<std::mutex> lock(budget->mutex);
    if (budget.use_count() == 2 && budget->used == 0 && !budget->cleanup_failed)
      it = budgets().erase(it);
    else ++it;
  }
}
std::shared_ptr<Budget> partition_budget(int device, int64_t partition, int64_t workspace,
                                       int64_t cache, int64_t capacity) {
  prune_budgets();
  const BudgetKey key{device, partition};
  const auto found = budgets().find(key);
  auto value = found == budgets().end() ? nullptr : found->second;
  if (value) {
    std::lock_guard<std::mutex> lock(value->mutex);
    if (value->cleanup_failed) throw std::runtime_error("cuTensorNet partition scratch cleanup failed");
    if (value->workspace_limit != workspace || value->cache_limit != cache || value->capacity != capacity)
      throw std::invalid_argument("cuTensorNet partition workspace/cache/capacity policies are immutable while entries or active leases exist");
    return value;
  }
  value = std::make_shared<Budget>(workspace, cache, capacity);
  budgets()[key] = value;
  return value;
}
void tensor_status(cutensornetStatus_t status, const char* operation,
                   const Entry& entry, const Tensor& tensor, int64_t slot,
                   cudaDataType_t dtype) {
  if (status == CUTENSORNET_STATUS_SUCCESS) return;
  std::ostringstream context;
  context << operation << ": " << cutensornetGetErrorString(status)
          << "; device=" << entry.device << " tensor_slot=" << slot
          << " role=" << (slot < 0 ? "output" : "input")
          << " cuda_dtype=" << static_cast<int>(dtype)
          << " shared_unit_axis=" << entry.lifted_recipe.has_value() << " extents=[";
  for (size_t axis = 0; axis < tensor.extents.size(); ++axis) {
    if (axis) context << ',';
    context << tensor.extents[axis];
  }
  context << "] modes=[";
  for (size_t axis = 0; axis < tensor.modes.size(); ++axis) {
    if (axis) context << ',';
    context << tensor.modes[axis];
  }
  context << "] strides=[";
  for (size_t axis = 0; axis < tensor.strides.size(); ++axis) {
    if (axis) context << ',';
    context << tensor.strides[axis];
  }
  context << ']';
  throw std::runtime_error(context.str());
}
void create_network(Entry& entry, Type dtype) {
  PHX_CUTN_CHECK(cutensornetCreate(&entry.handle));
  PHX_CUTN_CHECK(cutensornetCreateNetwork(entry.handle, &entry.network));
  entry.tensor_ids.resize(entry.recipe.inputs.size());
  const auto& descriptors = entry.descriptors();
  const cutensornetTensorQualifiers_t qualifiers{0, 0, 0};
  for (size_t i = 0; i < entry.recipe.inputs.size(); ++i) {
    const auto& input = descriptors.inputs[i];
    tensor_status(cutensornetNetworkAppendTensor(entry.handle, entry.network,
        static_cast<int32_t>(input.modes.size()), input.extents.data(), input.modes.data(),
        &qualifiers, dtype.storage, &entry.tensor_ids[i]),
        "cutensornetNetworkAppendTensor", entry, input, i, dtype.storage);
  }
  const auto& output = descriptors.output;
  tensor_status(cutensornetNetworkSetOutputTensor(entry.handle, entry.network,
      static_cast<int32_t>(output.modes.size()), output.modes.data(), dtype.storage),
      "cutensornetNetworkSetOutputTensor", entry, output, -1, dtype.storage);
  PHX_CUTN_CHECK(cutensornetNetworkSetAttribute(entry.handle, entry.network,
      CUTENSORNET_NETWORK_COMPUTE_TYPE, &dtype.compute, sizeof(dtype.compute)));
  PHX_CUTN_CHECK(cutensornetCreateContractionOptimizerInfo(entry.handle, entry.network, &entry.info));
}
void check_unsliced(Entry& entry) {
  int64_t slices = 0;
  int32_t modes = -1;
  PHX_CUTN_CHECK(cutensornetContractionOptimizerInfoGetAttribute(entry.handle, entry.info,
      CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &slices, sizeof(slices)));
  PHX_CUTN_CHECK(cutensornetContractionOptimizerInfoGetAttribute(entry.handle, entry.info,
      CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES, &modes, sizeof(modes)));
  if (slices != 1 || modes != 0) throw std::runtime_error("cuTensorNet slicing is forbidden");
}
std::vector<cutensornetNodePair_t> read_path(Entry& entry) {
  std::vector<cutensornetNodePair_t> result(entry.recipe.inputs.size() - 1);
  cutensornetContractionPath_t path{static_cast<int32_t>(result.size()), result.data()};
  PHX_CUTN_CHECK(cutensornetContractionOptimizerInfoGetAttribute(entry.handle, entry.info,
      CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, &path, sizeof(path)));
  if (path.numContractions != static_cast<int32_t>(result.size()))
    throw std::runtime_error("cuTensorNet returned an unexpected path length");
  return result;
}
void inject_path(Entry& entry) {
  cutensornetContractionPath_t path{static_cast<int32_t>(entry.recipe.path.size()), entry.recipe.path.data()};
  PHX_CUTN_CHECK(cutensornetContractionOptimizerInfoSetAttribute(entry.handle, entry.info,
      CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, &path, sizeof(path)));
  const cutensornetSlicingConfig_t slicing{0, nullptr};
  PHX_CUTN_CHECK(cutensornetContractionOptimizerInfoSetAttribute(entry.handle, entry.info,
      CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG, &slicing, sizeof(slicing)));
  check_unsliced(entry);
  const auto imported = read_path(entry);
  for (size_t i = 0; i < imported.size(); ++i)
    if (imported[i].first != path.data[i].first || imported[i].second != path.data[i].second)
      throw std::runtime_error("cuTensorNet modified the supplied fixed path");
  PHX_CUTN_CHECK(cutensornetNetworkSetOptimizerInfo(entry.handle, entry.network, entry.info));
  PHX_CUTN_CHECK(cutensornetCreateWorkspaceDescriptor(entry.handle, &entry.workspace));
}
void bind(Entry& entry, ffi::RemainingArgs inputs, ffi::AnyBuffer output) {
  if (inputs.size() != entry.recipe.inputs.size()) throw std::invalid_argument("Incorrect input count");
  const auto dtype = output.element_type();
  validate(output, entry.recipe.output, dtype);
  for (size_t i = 0; i < entry.recipe.inputs.size(); ++i) {
    const auto input = inputs.get<ffi::AnyBuffer>(i);
    if (!input.has_value()) throw std::invalid_argument(input.error().message());
    validate(*input, entry.recipe.inputs[i], dtype);
    const auto& descriptor = entry.descriptors().inputs[i];
    tensor_status(cutensornetNetworkSetInputTensorMemory(entry.handle, entry.network, entry.tensor_ids[i],
        input->untyped_data(), descriptor.strides.data()),
        "cutensornetNetworkSetInputTensorMemory", entry, descriptor, i, type(dtype).storage);
  }
  const auto& descriptor = entry.descriptors().output;
  tensor_status(cutensornetNetworkSetOutputTensorMemory(entry.handle, entry.network,
      output.untyped_data(), descriptor.strides.data()),
      "cutensornetNetworkSetOutputTensorMemory", entry, descriptor, -1, type(dtype).storage);
}
void set_memory(Entry& entry, cutensornetMemspace_t space, cutensornetWorkspaceKind_t kind,
                void* pointer, int64_t bytes) {
  PHX_CUTN_CHECK(cutensornetWorkspaceSetMemory(entry.handle, entry.workspace, space, kind, pointer, bytes));
}
void prepare(Entry& entry, int64_t limit) {
  if (!entry.prepared) {
    PHX_CUTN_CHECK(cutensornetWorkspaceComputeContractionSizes(entry.handle, entry.network, entry.info, entry.workspace));
    PHX_CUTN_CHECK(cutensornetWorkspaceGetMemorySize(entry.handle, entry.workspace, CUTENSORNET_WORKSIZE_PREF_MIN,
        CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &entry.device_scratch));
    PHX_CUTN_CHECK(cutensornetWorkspaceGetMemorySize(entry.handle, entry.workspace, CUTENSORNET_WORKSIZE_PREF_MIN,
        CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, &entry.host_scratch));
  }
  if (entry.device_scratch < 0 || entry.host_scratch < 0 || entry.device_scratch > limit ||
      entry.host_scratch > limit - entry.device_scratch)
    throw std::invalid_argument("cuTensorNet actual minimum host+device scratch exceeds workspace_limit_bytes");
  if (!entry.prepared) {
    entry.host_buffer.reset(entry.host_scratch ? new unsigned char[entry.host_scratch] : nullptr);
    entry.budget->reserve(entry.device_scratch);
    entry.charged_bytes = entry.device_scratch;
    if (entry.device_scratch) cuda_check(cudaMalloc(&entry.device_buffer, entry.device_scratch));
    set_memory(entry, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, entry.device_buffer, entry.device_scratch);
    set_memory(entry, CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, entry.host_buffer.get(), entry.host_scratch);
    set_memory(entry, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE, nullptr, 0);
    set_memory(entry, CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_CACHE, nullptr, 0);
    PHX_CUTN_CHECK(cutensornetNetworkPrepareContraction(entry.handle, entry.network, entry.workspace));
    entry.prepared = true;
  }
}
void contract(Entry& entry, cudaStream_t stream) {
  try {
    PHX_CUTN_CHECK(cutensornetWorkspacePurgeCache(entry.handle, entry.workspace, CUTENSORNET_MEMSPACE_DEVICE));
    PHX_CUTN_CHECK(cutensornetWorkspacePurgeCache(entry.handle, entry.workspace, CUTENSORNET_MEMSPACE_HOST));
    PHX_CUTN_CHECK(cutensornetNetworkContract(entry.handle, entry.network, 0, entry.workspace, nullptr, stream));
    const auto status = cudaStreamSynchronize(stream);
    entry.cuda_cleanup(status, "cudaStreamSynchronize(cuTensorNet)");
    cuda_check(status);
    PHX_CUTN_CHECK(cutensornetWorkspacePurgeCache(entry.handle, entry.workspace, CUTENSORNET_MEMSPACE_DEVICE));
    PHX_CUTN_CHECK(cutensornetWorkspacePurgeCache(entry.handle, entry.workspace, CUTENSORNET_MEMSPACE_HOST));
  } catch (const std::exception& error) {
    const auto status = cudaStreamSynchronize(stream);
    entry.cuda_cleanup(status, "cudaStreamSynchronize(cuTensorNet error cleanup)");
    if (status != cudaSuccess)
      throw std::runtime_error(std::string(error.what()) + "; stream drain: " + cudaGetErrorString(status));
    throw;
  }
}
std::shared_ptr<Entry> memo(int device, int64_t partition, int64_t capacity,
                           int64_t workspace_limit, int64_t cache_limit,
                           std::string_view recipe, ffi::DataType dtype) {
  std::lock_guard<std::mutex> lock(index_mutex());
  auto budget = partition_budget(device, partition, workspace_limit, cache_limit, capacity);
  // Names and descriptor ownership remain solely in canonical registry entries.
  // The weak index follows canonical release/LRU eviction, without a second cache.
  prune_index();
  for (auto& item : index()) {
    if (item.device != device || item.partition != partition) continue;
    if (item.capacity != capacity)
      throw std::invalid_argument("cuTensorNet partition capacity is immutable while entries are resident");
    auto candidate = item.entry.lock();
    if (item.dtype != dtype || !candidate || candidate->recipe_text != recipe) continue;
    auto resident = registry().find(device, partition, item.id);
    if (resident && !resident->invalid.load()) return resident;
    registry().erase(device, partition, item.id);
    break;
  }
  auto entry = std::make_shared<Entry>(device, parse(recipe, false));
  entry->budget = std::move(budget);
  entry->recipe_text = std::string(recipe);
  create_network(*entry, type(dtype));
  inject_path(*entry);
  const int32_t id = registry().insert(device, partition, capacity, entry);
  registry().count_rebuild(device, partition);
  prune_index();
  index().push_back({device, partition, capacity, dtype, id, entry});
  return entry;
}
ffi::Error Contract(cudaStream_t stream, ffi::RemainingArgs inputs, ffi::Result<ffi::AnyBuffer> output,
                    std::string_view recipe, int64_t workspace_limit_bytes, int64_t cache_limit_bytes,
                    int64_t partition, int64_t capacity) {
  std::shared_ptr<Entry> entry;
  try {
    if (workspace_limit_bytes < 0 || cache_limit_bytes < 0 || partition <= 0 || capacity < 1 || capacity > 4096)
      throw std::invalid_argument("cuTensorNet requires nonnegative byte limits and positive partition/capacity <=4096");
    if (recipe.empty() || recipe.size() > kMaxRecipeBytes) throw std::invalid_argument("Invalid tensor recipe size");
    int device = 0; cuda_check(cudaGetDevice(&device));
    entry = memo(device, partition, capacity, workspace_limit_bytes, cache_limit_bytes,
                      recipe, output->element_type());
    std::lock_guard<std::mutex> lock(entry->mutex);
    // Scratch is reused under the entry lock. Its partition budget includes
    // evicted active leases; opaque vendor host descriptor bytes remain unknown.
    if (entry->invalid.load()) throw std::runtime_error("cuTensorNet memo was invalidated by a concurrent failure");
    try {
      bind(*entry, inputs, *output);
      prepare(*entry, workspace_limit_bytes);
      contract(*entry, stream);
    } catch (...) {
      // Retire mutable provider state after any failed invocation. Drain while
      // still owning the entry lock, before another caller can bind its pointers.
      entry->invalid.store(true);
      const auto drain = cudaStreamSynchronize(stream);
      entry->cuda_cleanup(drain, "cudaStreamSynchronize(cuTensorNet invocation cleanup)");
      if (drain != cudaSuccess)
        throw std::runtime_error(std::string("cuTensorNet failure; stream drain: ") + cudaGetErrorString(drain));
      throw;
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    const auto drain = cudaStreamSynchronize(stream);
    if (drain != cudaSuccess) {
      if (entry) entry->cuda_cleanup(drain, "cudaStreamSynchronize(cuTensorNet failure cleanup)");
      else phx::cuda::set_cleanup_error("cudaStreamSynchronize(cuTensorNet failure cleanup)", cudaGetErrorString(drain));
    }
    std::string message = error.what();
    if (drain != cudaSuccess) message += std::string("; stream drain: ") + cudaGetErrorString(drain);
    phx::cuda::set_last_error(message);
    if (dynamic_cast<const std::invalid_argument*>(&error)) return ffi::Error::InvalidArgument(message);
    return ffi::Error::Internal(message);
  }
}
void configure(Entry& entry, cutensornetContractionOptimizerConfigAttributes_t attr, int32_t value) {
  PHX_CUTN_CHECK(cutensornetContractionOptimizerConfigSetAttribute(entry.handle, entry.config, attr, &value, sizeof(value)));
}
void optimize(Entry& entry, int64_t workspace_limit, int32_t iterations, int32_t seed) {
  PHX_CUTN_CHECK(cutensornetCreateContractionOptimizerConfig(entry.handle, &entry.config));
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING, 1);
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, iterations);
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS, 1);
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, iterations);
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS, std::min(iterations, 500));
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS, 1);
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR, 1);
  configure(entry, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED, seed);
  const cutensornetSmartOption_t smart = CUTENSORNET_SMART_OPTION_DISABLED;
  PHX_CUTN_CHECK(cutensornetContractionOptimizerConfigSetAttribute(entry.handle, entry.config,
      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION, &smart, sizeof(smart)));
  const cutensornetMemoryModel_t memory = CUTENSORNET_MEMORY_MODEL_HEURISTIC;
  PHX_CUTN_CHECK(cutensornetContractionOptimizerConfigSetAttribute(entry.handle, entry.config,
      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL, &memory, sizeof(memory)));
  const cutensornetOptimizerCost_t objective = CUTENSORNET_OPTIMIZER_COST_FLOPS;
  PHX_CUTN_CHECK(cutensornetContractionOptimizerConfigSetAttribute(entry.handle, entry.config,
      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE, &objective, sizeof(objective)));
  PHX_CUTN_CHECK(cutensornetContractionOptimize(entry.handle, entry.network, entry.config,
      static_cast<uint64_t>(workspace_limit), entry.info));
  check_unsliced(entry);
}
}  // namespace

phx::cuda::RegistryBase* component_registry() {
  static auto* view = new phx::cuda::TensorRegistry(registry(), [](bool all, int64_t partition) {
    std::lock_guard<std::mutex> lock(index_mutex());
    for (const auto& item : budgets()) {
      if (!all && item.first.second != partition) continue;
      std::lock_guard<std::mutex> budget_lock(item.second->mutex);
      if (item.second->cleanup_failed)
        phx::cuda::set_cleanup_error("cuTensorNet partition cleanup failed");
    }
  });
  return view;
}
int32_t component_shutdown() {
  std::lock_guard<std::mutex> lock(index_mutex());
  index().clear();
  registry().release_all();
  prune_budgets();
  for (const auto& item : budgets()) {
    std::lock_guard<std::mutex> budget_lock(item.second->mutex);
    if (item.second->cleanup_failed) {
      phx::cuda::set_cleanup_error("cuTensorNet partition cleanup failed");
      return PHX_CUDA_PROVIDER_ERROR;
    }
  }
  return PHX_CUDA_SUCCESS;
}

// HOST shape-only planner. pair_capacity counts int32 elements; pair_count counts pairs.
// The caller owns a fresh subprocess, CUDA_VISIBLE_DEVICES="", and a wall deadline.
// Planner recipes prefix the numerical recipe with dtype 0=F32,1=F64,2=C64,3=C128.
// The only supported objective is FLOPS; no CUDA runtime APIs are called here.
extern "C" PHX_CUDA_EXPORT int32_t phydrax_cutensornet_plan(
    const char* recipe, int64_t workspace_limit_bytes, int64_t max_search_iterations,
    int64_t seed, int32_t* pair_path, int64_t pair_capacity, int64_t* pair_count,
    char* error, int64_t error_capacity) {
  if (pair_count) *pair_count = 0;
  if (error && error_capacity > 0) error[0] = '\0';
  try {
    if (!recipe || !pair_count || pair_capacity < 0 || (pair_capacity && !pair_path) ||
        workspace_limit_bytes < 0 || max_search_iterations < 1 || max_search_iterations > 10000 ||
        seed < 0 || seed > std::numeric_limits<int32_t>::max() || error_capacity < 0 ||
        (error_capacity && !error)) throw std::invalid_argument("Invalid cuTensorNet planner arguments");
    size_t length = 0;
    while (length <= kMaxRecipeBytes && recipe[length] != '\0') ++length;
    if (length > kMaxRecipeBytes) throw std::invalid_argument("Planner recipe exceeds 1 MiB");
    Tokens tokens{std::string_view(recipe, length)};
    const int64_t dtype_code = tokens.next();
    Type dtype;
    switch (dtype_code) {
      case 0: dtype = type(ffi::DataType::F32); break;
      case 1: dtype = type(ffi::DataType::F64); break;
      case 2: dtype = type(ffi::DataType::C64); break;
      case 3: dtype = type(ffi::DataType::C128); break;
      default: throw std::invalid_argument("Invalid planner dtype code");
    }
    Entry entry(-1, parse(tokens.text, true));
    const int64_t count = static_cast<int64_t>(entry.recipe.inputs.size()) - 1;
    if (pair_capacity < 2 * count) throw std::invalid_argument("Planner pair output capacity is too small");
    create_network(entry, dtype);
    optimize(entry, workspace_limit_bytes, static_cast<int32_t>(max_search_iterations), static_cast<int32_t>(seed));
    const auto path = read_path(entry);
    int64_t live = static_cast<int64_t>(entry.recipe.inputs.size());
    for (size_t i = 0; i < path.size(); ++i) {
      if (path[i].first < 0 || path[i].second < 0 || path[i].first >= live || path[i].second >= live ||
          path[i].first == path[i].second) throw std::runtime_error("Optimizer returned invalid live-list path");
      pair_path[2 * i] = path[i].first; pair_path[2 * i + 1] = path[i].second;
      --live;
    }
    entry.close();
    if (entry.cleanup_failed)
      throw std::runtime_error("cuTensorNet planner cleanup failed");
    *pair_count = count;
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_capacity > 0) {
      const size_t length = std::min(std::strlen(failure.what()), static_cast<size_t>(error_capacity - 1));
      std::memcpy(error, failure.what(), length); error[length] = '\0';
    }
    phx::cuda::set_last_error(failure.what());
    return dynamic_cast<const std::invalid_argument*>(&failure) ? 1 : 2;
  }
}

extern "C" {
PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cutensornet_contract(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(phydrax_cutensornet_contract, Contract,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().Ret<ffi::AnyBuffer>()
        .Attr<std::string_view>("recipe").Attr<int64_t>("workspace_limit_bytes")
        .Attr<int64_t>("cache_limit_bytes").Attr<int64_t>("partition").Attr<int64_t>("capacity"));
}
PHX_CUDA_HANDLER_TABLE({"phydrax_cutensornet_contract", phydrax_cutensornet_contract})
