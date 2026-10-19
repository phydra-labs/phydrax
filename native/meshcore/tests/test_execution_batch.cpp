//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "bounded_memory.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {
using Counts = std::array<uint64_t, 4>;

struct Receipt {
  std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> work{};
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
};

class Execution final {
 public:
  explicit Execution(Counts limits, double seconds = INFINITY) {
    PHX_CHECK(phx_mc_execution_begin(limits[0], limits[1], limits[2], limits[3],
                                     seconds, nullptr, &scope) == PHX_MC_OK);
  }
  ~Execution() {
    if (scope != nullptr) finish(expected_status);
  }
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;

  Counts remaining(int32_t expected = PHX_MC_OK) const {
    Counts result{};
    double seconds;
    PHX_CHECK(phx_mc_execution_remaining(scope, result.data(), &seconds) == expected);
    return result;
  }
  Receipt finish(int32_t expected = PHX_MC_OK) {
    Receipt result;
    double seconds;
    PHX_CHECK(phx_mc_execution_end(scope, result.work.data(), result.memory.data(),
                                   &seconds) == expected);
    scope = nullptr;
    return result;
  }
  void* scope = nullptr;
  int32_t expected_status = PHX_MC_OK;
};

class Batch final {
 public:
  Batch(void* scope, Counts upper, void* parent = nullptr) {
    PHX_CHECK(phx_mc_execution_batch_admit(scope, parent, upper[0], upper[1],
                                          upper[2], upper[3], &ticket) == PHX_MC_OK);
  }
  ~Batch() {
    if (ticket != nullptr) settle();
  }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  void settle() {
    PHX_CHECK(phx_mc_execution_batch_settle(ticket, actual[0], actual[1],
                                           actual[2], actual[3]) == PHX_MC_OK);
    ticket = nullptr;
  }
  void* ticket = nullptr;
  Counts actual{};
};

void admission_refuses_atomically(std::size_t resource) {
  Execution execution({10, 10, 10, 10});
  Batch admitted(execution.scope, {6, 6, 6, 6});
  const Counts before = execution.remaining();
  Counts request{1, 1, 1, 1};
  request[resource] = 5;
  void* refused = admitted.ticket;
  PHX_CHECK(phx_mc_execution_batch_admit(execution.scope, nullptr, request[0], request[1],
                                        request[2], request[3], &refused) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(refused == nullptr);
  PHX_CHECK(execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == before);
  std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES> refusal{};
  PHX_CHECK(phx_mc_execution_batch_refusal(execution.scope, refusal.data()) == PHX_MC_OK);
  PHX_CHECK(std::equal(before.begin(), before.end(), refusal.begin()));
  PHX_CHECK(refusal[PHX_MC_BATCH_CONSUMED_WORK] == 0 &&
            refusal[PHX_MC_BATCH_CONSUMED_QUERIES] == 0);
  // Refusal cannot prevent release or commit work that was already performed.
  admitted.actual = {2, 1, 3, 4};
  admitted.settle();
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{8, 9, 10, 10}));
  std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES> after_release{};
  PHX_CHECK(phx_mc_execution_batch_refusal(execution.scope, after_release.data()) == PHX_MC_OK);
  PHX_CHECK(after_release == refusal);
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 1);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 2 && receipt.work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 1);
  if (resource < 3) PHX_CHECK(receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS + resource] == 1);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void test_work_admission() { admission_refuses_atomically(0); }
void test_query_admission() { admission_refuses_atomically(1); }
void test_cavity_admission() { admission_refuses_atomically(2); }
void test_byte_admission() { admission_refuses_atomically(3); }

void test_unused_release() {
  Execution execution({10, 10, 10, 10});
  {
    Batch batch(execution.scope, {10, 10, 10, 10});
    PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
    batch.actual = {3, 2, 4, 5};
  }
  PHX_CHECK((execution.remaining() == Counts{7, 8, 10, 10}));
  {
    Batch next(execution.scope, {7, 8, 10, 10});
    next.actual = {1, 1, 2, 3};
  }
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 4 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 3 && receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 4);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 4 && receipt.work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 3);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 5);
}

void test_nested_isolation() {
  Execution execution({10, 10, 10, 100});
  Batch parent(execution.scope, {10, 10, 10, 100});
  Batch first(execution.scope, {6, 6, 6, 60}, parent.ticket);
  Batch second(execution.scope, {4, 4, 4, 40}, parent.ticket);
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  first.actual = {1, 2, 3, 20};
  first.settle();
  Batch third(execution.scope, {5, 4, 6, 60}, parent.ticket);
  second.actual = {1, 1, 2, 10};
  second.settle();
  third.actual = {2, 1, 4, 20};
  third.settle();
  // Settled children never debit the root a second time or free its reservation.
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  parent.actual = {2, 1, 2, 10};
  parent.settle();
  PHX_CHECK((execution.remaining() == Counts{4, 5, 10, 100}));
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 6 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 5);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 6 && receipt.work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 5);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 8 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 70);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void test_nested_refusal() {
  Execution execution({10, 10, 10, 10});
  Batch parent(execution.scope, {6, 6, 6, 6});
  Batch child(execution.scope, {4, 4, 4, 4}, parent.ticket);
  void* refused = nullptr;
  PHX_CHECK(phx_mc_execution_batch_admit(execution.scope, parent.ticket, 3, 0, 0, 0,
                                        &refused) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(refused == nullptr);
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{4, 4, 4, 4}));
  std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES> refusal{};
  PHX_CHECK(phx_mc_execution_batch_refusal(execution.scope, refusal.data()) == PHX_MC_OK);
  PHX_CHECK((refusal == std::array<uint64_t, 8>{2, 2, 2, 2, 0, 0, 0, 0}));
  child.actual = {1, 1, 1, 1};
  child.settle();
  parent.actual = {1, 1, 1, 1};
  parent.settle();
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{8, 8, 10, 10}));
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 2 && receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void native_charge_respects_reservation(std::size_t resource) {
  Execution execution({10, 10, 10, 10});
  Batch ticket(execution.scope, {6, 6, 6, 6});
  if (resource == 0 || resource == 1) {
    PHX_CHECK(phx_mc_execution_charge(execution.scope, resource == 0 ? 5 : 0,
                                     resource == 1 ? 5 : 0) == PHX_MC_CAPACITY_EXCEEDED);
  } else if (resource == 2) {
    PHX_CHECK(phx_mc_execution_admit_cavity(execution.scope, 5) == PHX_MC_CAPACITY_EXCEEDED);
  } else {
    bool refused = false;
    try {
      phx::mc::NativeVector<uint8_t> buffer(5, 0);
    } catch (const std::bad_alloc&) {
      refused = true;
    }
    PHX_CHECK(refused);
  }
  ticket.actual = {2, 2, 2, 2};
  ticket.settle();
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 2);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
  if (resource == 3) PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 1);
}

void test_native_work() { native_charge_respects_reservation(0); }
void test_native_queries() { native_charge_respects_reservation(1); }
void test_native_cavity() { native_charge_respects_reservation(2); }
void test_native_bytes() { native_charge_respects_reservation(3); }

void test_settlement_retry() {
  Execution execution({4, 3, 2, 10});
  Batch batch(execution.scope, {4, 3, 2, 10});
  constexpr std::array<Counts, 4> defects = {{{5, 1, 1, 5}, {2, 4, 1, 5},
                                            {2, 1, 3, 5}, {2, 1, 1, 11}}};
  for (const Counts& defect : defects) {
    PHX_CHECK(phx_mc_execution_batch_settle(batch.ticket, defect[0], defect[1],
                                           defect[2], defect[3]) == PHX_MC_INTERNAL_ERROR);
    PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  }
  batch.actual = {2, 1, 1, 5};
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{2, 2, 2, 10}));
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 1);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 0 && receipt.work[PHX_MC_EXECUTION_QUERY_REFUSALS] == 0 && receipt.work[PHX_MC_EXECUTION_CAVITY_REFUSALS] == 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 5);
}

void test_nested_overestimate_retry() {
  Execution execution({UINT64_MAX, 0, 0, 0});
  Batch parent(execution.scope, {UINT64_MAX, 0, 0, 0});
  Batch child(execution.scope, {UINT64_MAX - 1, 0, 0, 0}, parent.ticket);
  child.actual[0] = UINT64_MAX - 1;
  child.settle();
  PHX_CHECK(phx_mc_execution_batch_settle(parent.ticket, 2, 0, 0, 0) == PHX_MC_INTERNAL_ERROR);
  parent.actual[0] = 1;
  parent.settle();
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == UINT64_MAX && receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == UINT64_MAX);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
}

void test_live_ticket_lifetime() {
  Execution execution({10, 10, 10, 10});
  Batch parent(execution.scope, {6, 6, 6, 6});
  Batch child(execution.scope, {2, 2, 2, 2}, parent.ticket);
  Receipt untouched;
  untouched.work.fill(91);
  untouched.memory.fill(92);
  double seconds = 93;
  PHX_CHECK(phx_mc_execution_end(execution.scope, untouched.work.data(), untouched.memory.data(),
                                 &seconds) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(untouched.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 91 && untouched.memory[PHX_MC_EXECUTION_MEMORY_LIMIT_BYTES] == 92 && seconds == 93);
  PHX_CHECK(phx_mc_execution_batch_settle(parent.ticket, 0, 0, 0, 0) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK((execution.remaining() == Counts{4, 4, 4, 4}));
  child.actual = {1, 1, 1, 1};
  child.settle();
  parent.settle();
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 1 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 1 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void test_exception_settlement() {
  Execution execution({20, 20, 20, 20});
  bool escaped = false;
  try {
    Batch parent(execution.scope, {20, 20, 20, 20});
    parent.actual = {3, 2, 7, 9};
    Batch child(execution.scope, {10, 10, 10, 10}, parent.ticket);
    child.actual = {4, 3, 5, 6};
    throw std::runtime_error("operation failed after measured work");
  } catch (const std::runtime_error&) {
    escaped = true;
  }
  PHX_CHECK(escaped);
  PHX_CHECK((execution.remaining() == Counts{13, 15, 20, 20}));
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 7 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 5);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 7 && receipt.work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 5);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 7 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 9 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void test_concurrent_admission() {
  constexpr std::size_t workers = 12;
  Execution execution({8, 5, 10, 17});
  std::barrier gate(static_cast<std::ptrdiff_t>(workers + 1));
  std::atomic<unsigned> admitted = 0;
  std::array<std::thread, workers> threads;
  std::array<bool, workers> admission_valid{}, settlement_valid{};
  for (std::size_t index = 0; index < workers; ++index) {
    threads[index] = std::thread([&, index] {
      void* ticket = nullptr;
      gate.arrive_and_wait();
      const int32_t status = phx_mc_execution_batch_admit(
          execution.scope, nullptr, 3, 2, 4, 7, &ticket);
      admission_valid[index] = status == PHX_MC_OK ? ticket != nullptr
                                                  : status == PHX_MC_CAPACITY_EXCEEDED && ticket == nullptr;
      if (status == PHX_MC_OK) admitted.fetch_add(1);
      gate.arrive_and_wait();
      gate.arrive_and_wait();
      settlement_valid[index] = ticket == nullptr ||
          phx_mc_execution_batch_settle(ticket, 1, 1, 1, 2) == PHX_MC_OK;
    });
  }
  gate.arrive_and_wait();
  gate.arrive_and_wait();
  PHX_CHECK(admitted.load() == 2);
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{2, 1, 2, 3}));
  gate.arrive_and_wait();
  for (std::thread& thread : threads) thread.join();
  for (std::size_t index = 0; index < workers; ++index) {
    PHX_CHECK(admission_valid[index] && settlement_valid[index]);
  }
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{6, 3, 10, 17}));
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 2);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 2 && receipt.work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 2 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void test_nested_scope_reservation() {
  Execution parent({10, 10, 10, 10});
  Batch batch(parent.scope, {6, 6, 6, 6});
  Execution child({10, 10, 10, 10});
  PHX_CHECK((child.remaining() == Counts{4, 4, 4, 4}));
  PHX_CHECK(phx_mc_execution_charge(child.scope, 4, 4) == PHX_MC_OK);
  const Receipt child_receipt = child.finish();
  PHX_CHECK(child_receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 4 && child_receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 4);
  batch.actual = {2, 2, 2, 2};
  batch.settle();
  const Receipt receipt = parent.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 6 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 6);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 6 && receipt.work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 6 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void test_timeout_admission() {
  Execution execution({10, 10, 10, 10}, 0.0);
  void* ticket = nullptr;
  PHX_CHECK(phx_mc_execution_batch_admit(execution.scope, nullptr, 5, 5, 5, 5,
                                        &ticket) == PHX_MC_TIMEOUT);
  PHX_CHECK(ticket == nullptr);
  PHX_CHECK((execution.remaining(PHX_MC_TIMEOUT) == Counts{10, 10, 10, 10}));
  std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES> refusal{};
  PHX_CHECK(phx_mc_execution_batch_refusal(execution.scope, refusal.data()) == PHX_MC_OK);
  PHX_CHECK((refusal == std::array<uint64_t, 8>{10, 10, 10, 10, 0, 0, 0, 0}));
  const Receipt receipt = execution.finish(PHX_MC_TIMEOUT);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

void test_checkpoint_and_invalid_arguments() {
  Execution execution({10, 10, 10, 10});
  Batch batch(execution.scope, {6, 6, 6, 6});
  std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES> missing{};
  missing.fill(91);
  PHX_CHECK(phx_mc_execution_batch_refusal(execution.scope, missing.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(missing[0] == 91 && missing[7] == 91);
  PHX_CHECK(phx_mc_execution_batch_refusal(nullptr, missing.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_batch_refusal(execution.scope, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_batch_checkpoint(batch.ticket) == PHX_MC_OK);
  PHX_CHECK((execution.remaining() == Counts{4, 4, 4, 4}));
  PHX_CHECK(phx_mc_execution_batch_checkpoint(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_batch_settle(nullptr, 0, 0, 0, 0) == PHX_MC_INVALID_ARGUMENT);
  void* invalid = batch.ticket;
  PHX_CHECK(phx_mc_execution_batch_admit(nullptr, nullptr, 0, 0, 0, 0, &invalid) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(invalid == nullptr);
  PHX_CHECK(phx_mc_execution_batch_admit(execution.scope, nullptr, 0, 0, 0, 0, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
  batch.settle();
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 0);
}

void test_refusal_consumption_evidence() {
  Execution execution({10, 10, 10, 10});
  PHX_CHECK(phx_mc_execution_charge(execution.scope, 3, 2) == PHX_MC_OK);
  Batch admitted(execution.scope, {4, 3, 2, 2});
  void* refused = nullptr;
  PHX_CHECK(phx_mc_execution_batch_admit(execution.scope, nullptr, 4, 0, 0, 0,
                                        &refused) == PHX_MC_CAPACITY_EXCEEDED);
  std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES> evidence{};
  int32_t probe = PHX_MC_INTERNAL_ERROR;
  std::thread worker([&] {
    probe = phx_mc_execution_batch_refusal(execution.scope, evidence.data());
  });
  worker.join();
  PHX_CHECK(probe == PHX_MC_OK);
  PHX_CHECK((evidence == std::array<uint64_t, 8>{3, 5, 8, 8, 3, 2, 0, 0}));
  admitted.actual = {2, 1, 1, 1};
  admitted.settle();
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 5 && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 3 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
}

struct Case {
  std::string_view name;
  void (*run)();
};
constexpr std::array cases = {
    Case{"work_admission", test_work_admission},
    Case{"query_admission", test_query_admission},
    Case{"cavity_admission", test_cavity_admission},
    Case{"byte_admission", test_byte_admission},
    Case{"unused_release", test_unused_release},
    Case{"nested_isolation", test_nested_isolation},
    Case{"nested_refusal", test_nested_refusal},
    Case{"native_work", test_native_work},
    Case{"native_queries", test_native_queries},
    Case{"native_cavity", test_native_cavity},
    Case{"native_bytes", test_native_bytes},
    Case{"settlement_retry", test_settlement_retry},
    Case{"nested_overestimate_retry", test_nested_overestimate_retry},
    Case{"live_ticket_lifetime", test_live_ticket_lifetime},
    Case{"exception_settlement", test_exception_settlement},
    Case{"concurrent_admission", test_concurrent_admission},
    Case{"nested_scope_reservation", test_nested_scope_reservation},
    Case{"timeout_admission", test_timeout_admission},
    Case{"checkpoint_and_invalid_arguments", test_checkpoint_and_invalid_arguments},
    Case{"refusal_consumption_evidence", test_refusal_consumption_evidence},
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  for (const Case& test : cases) {
    if (test.name == argv[1]) {
      test.run();
      return phx::mc::test::finish(argv[1]);
    }
  }
  return 1;
}
