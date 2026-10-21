#include "collective_lease.hpp"
#include <mpi-ext.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
thread_local std::string last_error;
void check(int status, const char* operation) {
  if (status != MPI_SUCCESS) {
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, text, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(text, length));
  }
}
void require_initialized() {
  int initialized = 0, finalized = 0, threading = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (!initialized || finalized) throw std::runtime_error("site MPI must already be initialized and not finalized");
  MPI_Query_thread(&threading);
  if (threading != MPI_THREAD_MULTIPLE)
    throw std::runtime_error("collective watchdog and XLA workers require MPI_THREAD_MULTIPLE");
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&MPI_Get_library_version), &info) || !info.dli_fname)
    throw std::runtime_error("cannot establish the loaded site MPI library identity");
  if (std::filesystem::canonical(info.dli_fname) != std::filesystem::canonical(PHX_COLLECTIVE_MPI_LIBRARY))
    throw std::runtime_error("loaded MPI differs from the site MPI used to build the collective bridge");
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
  if (MPIX_Query_cuda_support() != 1)
    throw std::runtime_error("initialized site MPI does not support CUDA buffers");
#else
  throw std::runtime_error("site MPI exposes no qualified CUDA-awareness query");
#endif
}

// Every blocking collective/vendor phase has a local deadline. On expiry the
// site MPI launcher is notified through MPI_Abort, not another collective. A
// second local deadline prevents a stuck MPI_Abort from preserving this rank.
class Supervisor final {
 public:
  Supervisor(MPI_Comm source, double seconds) : source_(source), seconds_(seconds), worker_([this] { run(); }) {}
  ~Supervisor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (armed_) deadline_ = std::chrono::steady_clock::now();
      else stop_ = true;
      condition_.notify_all();
    }
    worker_.join();
  }
  void arm() {
    std::lock_guard<std::mutex> lock(mutex_);
    deadline_ = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds_));
    armed_ = true;
    condition_.notify_all();
  }
  void disarm() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    condition_.notify_all();
  }
  void fail() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = true;
    deadline_ = std::chrono::steady_clock::now();
    condition_.notify_all();
  }
 private:
  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (!armed_) { condition_.wait(lock); continue; }
      const auto deadline = deadline_;
      if (condition_.wait_until(lock, deadline) != std::cv_status::timeout) continue;
      if (!armed_ || deadline != deadline_) continue;
      const MPI_Comm source = source_;
      lock.unlock();
      std::thread([source] { MPI_Abort(source, EXIT_FAILURE); }).detach();
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::_Exit(EXIT_FAILURE);
    }
  }
  MPI_Comm source_;
  double seconds_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_ = false, armed_ = false;
  std::chrono::steady_clock::time_point deadline_;
  std::thread worker_;
};

}  // namespace
namespace phx::collective {
struct CallFrame {
  uint64_t sequence;
  int64_t ticket;
  uint32_t phase;
  char configuration[129];
  char call[129];
};
struct LeaseState final {
  LeaseState(MPI_Comm source, double timeout, std::string configuration, int count)
      : owner(std::this_thread::get_id()), configuration_id(std::move(configuration)),
        agreement_records(size_t{4096} * count), call_records(count), supervisor(source, timeout) {}
  std::thread::id owner;
  std::string configuration_id;
  MPI_Comm control = MPI_COMM_NULL, numeric = MPI_COMM_NULL;
  std::mutex queue;
  std::atomic<bool> failed{false}, closed{false};
  int64_t next_ticket = 0, active_ticket = 0;
  uint64_t sequence = 0;
  std::string expected_schedule;
  size_t schedule_position = 0;
  std::vector<char> agreement_records;
  std::vector<CallFrame> call_records;
  std::map<std::string, std::function<void(MPI_Comm)>> retirement;
  Supervisor supervisor;
};
}  // namespace phx::collective
namespace {
using Lease = phx::collective::LeaseState;
std::mutex& registry_mutex() { static auto* value = new std::mutex; return *value; }
std::map<int64_t, std::shared_ptr<Lease>>& registry() {
  static auto* value = new std::map<int64_t, std::shared_ptr<Lease>>;
  return *value;
}
std::shared_ptr<Lease> find(int64_t id) {
  if (id <= 0) throw std::invalid_argument("collective lease identity is required");
  std::lock_guard<std::mutex> lock(registry_mutex());
  const auto found = registry().find(id);
  if (found == registry().end()) throw std::runtime_error("collective lease is closed, stale, or not local");
  return found->second;
}
void live(const Lease& lease) {
  if (lease.closed) throw std::runtime_error("collective lease is closed");
  if (lease.failed) throw std::runtime_error("collective lease has failed");
}
void owner(const Lease& lease) {
  live(lease);
  if (lease.owner != std::this_thread::get_id())
    throw std::runtime_error("collective lease calls must use the owning host thread");
}
void agreement(Lease& lease, std::string_view identity) {
  constexpr size_t capacity = 4096;
  std::array<char, capacity> local{};
  const bool overflow = identity.size() >= capacity;
  if (!overflow) std::copy(identity.begin(), identity.end(), local.begin());
  else std::strcpy(local.data(), "invalid-overlong-collective-identity");
  check(MPI_Allgather(local.data(), capacity, MPI_CHAR, lease.agreement_records.data(),
                     capacity, MPI_CHAR, lease.control), "collective control agreement");
  bool mismatch = overflow;
  for (size_t rank = 0; rank < lease.call_records.size(); ++rank)
    mismatch = mismatch || !std::equal(local.begin(), local.end(), lease.agreement_records.begin() + rank * capacity);
  if (mismatch) {
    lease.failed = true;
    throw std::runtime_error("all-rank collective configuration, preparation, or call order disagrees");
  }
}
void call_agreement(Lease& lease, uint32_t phase, int64_t ticket,
                    uint64_t sequence, std::string_view configuration = {},
                    std::string_view call = {}) {
  phx::collective::CallFrame frame;
  std::memset(&frame, 0, sizeof(frame));
  frame.phase = phase; frame.ticket = ticket; frame.sequence = sequence;
  if (configuration.size() > 128 || call.size() > 128)
    throw std::invalid_argument("collective configuration and call identities must fit 128 bytes");
  std::copy(configuration.begin(), configuration.end(), frame.configuration);
  std::copy(call.begin(), call.end(), frame.call);
  check(MPI_Allgather(&frame, sizeof(frame), MPI_BYTE, lease.call_records.data(),
                     sizeof(frame), MPI_BYTE, lease.control), "collective numeric order preflight");
  for (const auto& record : lease.call_records) {
    if (std::memcmp(&frame, &record, sizeof(frame)) != 0) {
      lease.failed = true;
      throw std::runtime_error("all-rank collective configuration, invocation ticket, or call order disagrees");
    }
  }
}
int64_t fresh_identity(int32_t root_process) {
  // JAX process ownership plus a never-reused local generation avoids stale
  // executable admission even after explicit retirement removes the registry.
  static auto* generations = new std::atomic<uint64_t>{0};
  const uint64_t generation = generations->fetch_add(1, std::memory_order_relaxed) + 1;
  if (generation > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("collective native lease generation space is exhausted");
  return static_cast<int64_t>((static_cast<uint64_t>(root_process) << 32) | generation);
}
template <class Function> int boundary(Function&& function) {
  try { function(); last_error.clear(); return 0; }
  catch (const std::exception& error) { last_error = error.what(); return 1; }
}
}  // namespace

namespace phx::collective {
CallGuard::CallGuard(int64_t id, std::string_view configuration,
                     std::string_view call_id, int64_t ticket)
    : lease_(find(id)), lock_(lease_->queue) {
  auto& lease = *lease_;
  live(lease);
  if (ticket <= 0 || ticket != lease.active_ticket)
    throw std::runtime_error("collective invocation ticket is absent, stale, or captured");
  if (configuration != lease.configuration_id)
    throw std::runtime_error("collective invocation configuration does not match its live lease");
  if (call_id.empty()) throw std::invalid_argument("collective call identity is required");
  if (lease.sequence == std::numeric_limits<uint64_t>::max())
    throw std::runtime_error("collective call sequence space is exhausted");
  call_agreement(lease, 1, ticket, lease.sequence++, configuration, call_id);
  if (call_id.size() != 64 || lease.schedule_position >= lease.expected_schedule.size() / 64 ||
      std::string_view(lease.expected_schedule.data() + lease.schedule_position * 64, 64) != call_id) {
    lease.failed = true;
    throw std::runtime_error("collective call does not match its admitted transformed schedule");
  }
  ++lease.schedule_position;
}
CallGuard::~CallGuard() {
  if (!completed_) {
    lease_->failed = true;
    lease_->supervisor.fail();
  }
}
MPI_Comm CallGuard::communicator() const { return lease_->numeric; }
void CallGuard::on_retire(const std::string& key, std::function<void(MPI_Comm)> callback) {
  if (key.empty() || !callback) throw std::invalid_argument("collective retirement requires a key and callback");
  if (!lease_->retirement.emplace(key, std::move(callback)).second)
    throw std::invalid_argument("collective retirement key is already owned");
}
void CallGuard::complete() {
  if (completed_) throw std::runtime_error("collective call was already completed");
  completed_ = true;
}
}  // namespace phx::collective

extern "C" {
const char* phx_collective_last_error() { return last_error.c_str(); }
const char* phx_collective_build_identity() { return PHX_COLLECTIVE_BUILD_IDENTITY; }
const char* phx_collective_mpi_library() { return PHX_COLLECTIVE_MPI_LIBRARY; }
const char* phx_collective_mpi_sha256() { return PHX_COLLECTIVE_MPI_SHA256; }
const char* phx_collective_mpi_header_sha256() { return PHX_COLLECTIVE_MPI_HEADER_SHA256; }
int phx_collective_validate_mpi_symbol(void* symbol) {
  return boundary([&] {
    if (!symbol || symbol != reinterpret_cast<void*>(&MPI_Get_library_version))
      throw std::runtime_error("mpi4py and the collective bridge do not share the same loaded site MPI ABI");
  });
}
const char* phx_collective_mpi_library_version() {
  static thread_local std::string text;
  try {
    require_initialized();
    char value[MPI_MAX_LIBRARY_VERSION_STRING]; int length = 0;
    check(MPI_Get_library_version(value, &length), "MPI_Get_library_version");
    text.assign(value, length);
  } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
  return text.c_str();
}
int phx_collective_create(int64_t source_fortran, int32_t process_index,
    const int32_t* processes, const int32_t* source_ranks, int32_t count, int32_t tag,
    const char* agreement_id, const char* configuration_id, double timeout,
    int64_t* output) {
  return boundary([&] {
    require_initialized();
    if (!processes || !source_ranks || count < 1 || !agreement_id || !*agreement_id ||
        !configuration_id || !*configuration_id || !output ||
        !std::isfinite(timeout) || timeout <= 0 || tag < 0 || tag > 32767 ||
        source_fortran < std::numeric_limits<MPI_Fint>::min() || source_fortran > std::numeric_limits<MPI_Fint>::max())
      throw std::invalid_argument("invalid collective group, identity, deadline, or communicator handle");
    MPI_Comm source = MPI_Comm_f2c(static_cast<MPI_Fint>(source_fortran));
    if (source == MPI_COMM_NULL) throw std::invalid_argument("source MPI communicator is null");
    int source_rank = -1, source_size = 0;
    check(MPI_Comm_rank(source, &source_rank), "MPI_Comm_rank");
    check(MPI_Comm_size(source, &source_size), "MPI_Comm_size");
    int member = -1;
    for (int i = 0; i < count; ++i) {
      if (source_ranks[i] < 0 || source_ranks[i] >= source_size || processes[i] < 0)
        throw std::invalid_argument("group mapping exceeds the initialized MPI communicator");
      for (int j = 0; j < i; ++j)
        if (source_ranks[j] == source_ranks[i] || processes[j] == processes[i])
          throw std::invalid_argument("group process and MPI rank mapping must be unique");
      if (processes[i] == process_index) member = i;
    }
    if (member < 0 || source_ranks[member] != source_rank)
      throw std::invalid_argument("local JAX process is not the declared initialized MPI rank participant");
    auto lease = std::make_shared<Lease>(source, timeout, configuration_id, count);
    lease->supervisor.arm();
    MPI_Group parent = MPI_GROUP_NULL, group = MPI_GROUP_NULL;
    check(MPI_Comm_group(source, &parent), "MPI_Comm_group");
    check(MPI_Group_incl(parent, count, source_ranks, &group), "MPI_Group_incl");
    check(MPI_Comm_create_group(source, group, tag, &lease->control), "MPI_Comm_create_group");
    check(MPI_Group_free(&group), "MPI_Group_free");
    check(MPI_Group_free(&parent), "MPI_Group_free");
    check(MPI_Comm_set_errhandler(lease->control, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
    std::vector<int32_t> actual(count);
    check(MPI_Allgather(&process_index, 1, MPI_INT32_T, actual.data(), 1, MPI_INT32_T, lease->control), "JAX process membership agreement");
    if (!std::equal(actual.begin(), actual.end(), processes))
      throw std::runtime_error("initialized MPI ranks disagree with JAX execution group membership");
    agreement(*lease, agreement_id);
    check(MPI_Comm_dup(lease->control, &lease->numeric), "MPI_Comm_dup numeric communicator");
    check(MPI_Comm_set_errhandler(lease->numeric, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
    int64_t id = member == 0 ? fresh_identity(processes[0]) : 0;
    check(MPI_Bcast(&id, 1, MPI_INT64_T, 0, lease->control), "collective lease identity broadcast");
    {
      std::lock_guard<std::mutex> lock(registry_mutex());
      if (!registry().emplace(id, lease).second) throw std::runtime_error("collective lease identity collision");
    }
    *output = id;
    lease->supervisor.disarm();
  });
}
int phx_collective_validate(int64_t id) {
  return boundary([&] { auto lease = find(id); std::lock_guard<std::mutex> lock(lease->queue); owner(*lease); });
}
int phx_collective_agree_program(int64_t id, const char* schedule_identity) {
  return boundary([&] {
    auto lease = find(id); std::lock_guard<std::mutex> lock(lease->queue); owner(*lease);
    if (lease->active_ticket || !schedule_identity || std::strlen(schedule_identity) != 64)
      throw std::invalid_argument("collective program staging requires an inactive lease and semantic schedule identity");
    lease->supervisor.arm();
    call_agreement(*lease, 5, 0, lease->sequence, lease->configuration_id, schedule_identity);
    lease->supervisor.disarm();
  });
}
int phx_collective_begin(int64_t id, const char* call_ids, int32_t count,
                         const char* schedule_identity, int64_t* ticket) {
  return boundary([&] {
    auto lease = find(id); std::lock_guard<std::mutex> lock(lease->queue); owner(*lease);
    if (!ticket || lease->active_ticket || lease->next_ticket == std::numeric_limits<int64_t>::max())
      throw std::runtime_error("collective invocation is nested or its ticket space is exhausted");
    if (!call_ids || count < 0 || !schedule_identity || std::strlen(schedule_identity) != 64 ||
        std::strlen(call_ids) != static_cast<size_t>(count) * 64)
      throw std::invalid_argument("invalid owner-generated collective invocation schedule");
    lease->supervisor.arm();
    const int64_t candidate = lease->next_ticket + 1;
    lease->expected_schedule.assign(call_ids, static_cast<size_t>(count) * 64);
    lease->schedule_position = 0;
    call_agreement(*lease, 0, candidate, lease->sequence,
                   lease->configuration_id, schedule_identity);
    lease->next_ticket = candidate;
    lease->active_ticket = candidate;
    *ticket = lease->active_ticket;
  });
}
int phx_collective_end(int64_t id, int64_t ticket) {
  return boundary([&] {
    auto lease = find(id); std::lock_guard<std::mutex> lock(lease->queue); owner(*lease);
    if (ticket <= 0 || ticket != lease->active_ticket) throw std::runtime_error("invalid collective invocation completion ticket");
    call_agreement(*lease, 3, ticket, lease->sequence);
    if (lease->schedule_position != lease->expected_schedule.size() / 64) {
      lease->failed = true;
      throw std::runtime_error("collective invocation ended before completing its transformed schedule");
    }
    lease->active_ticket = 0;
    lease->supervisor.disarm();
  });
}
int phx_collective_fail(int64_t id, const char* reason) {
  return boundary([&] {
    auto lease = find(id);
    if (!reason || !*reason) throw std::invalid_argument("collective failure reason is required");
    lease->failed = true;
    lease->supervisor.fail();
  });
}
int phx_collective_retire(int64_t id) {
  return boundary([&] {
    auto lease = find(id); std::lock_guard<std::mutex> lock(lease->queue); owner(*lease);
    if (lease->active_ticket) throw std::runtime_error("cannot retire an active collective invocation");
    lease->supervisor.arm();
    call_agreement(*lease, 4, 0, lease->sequence, lease->configuration_id, "retire");
    std::string keys = "retire";
    for (const auto& item : lease->retirement)
      keys += ":" + std::to_string(item.first.size()) + ":" + item.first;
    agreement(*lease, keys);
    for (const auto& item : lease->retirement) item.second(lease->numeric);
    agreement(*lease, "retired-vendor-resources");
    lease->retirement.clear();
    check(MPI_Comm_free(&lease->numeric), "MPI_Comm_free numeric");
    check(MPI_Comm_free(&lease->control), "MPI_Comm_free control");
    lease->closed = true;
    lease->supervisor.disarm();
    std::lock_guard<std::mutex> registry_lock(registry_mutex());
    registry().erase(id);
  });
}
}  // extern "C"
