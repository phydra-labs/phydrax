//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <cmath>
#include <cstdint>
#include <string_view>
#include <thread>

#include "check.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"

namespace {

void invalid_arguments() {
  void* policy = nullptr;
  PHX_CHECK(phx_mc_parallel_begin(0, INFINITY, &policy) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(policy == nullptr);
  PHX_CHECK(phx_mc_parallel_begin(1, -1.0, &policy) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_parallel_begin(1, NAN, &policy) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_parallel_begin(1, INFINITY, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_parallel_end(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx::mc::current_parallelism() == 1);
}

void nested_restore() {
  void* outer = nullptr;
  void* inner = nullptr;
  PHX_CHECK(phx_mc_parallel_begin(2, INFINITY, &outer) == PHX_MC_OK);
  PHX_CHECK(phx::mc::current_parallelism() == 2);
  PHX_CHECK(phx_mc_parallel_begin(7, INFINITY, &inner) == PHX_MC_OK);
  PHX_CHECK(phx::mc::current_parallelism() == 7);
  PHX_CHECK(phx_mc_parallel_end(outer) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx::mc::current_parallelism() == 7);
  PHX_CHECK(phx_mc_parallel_end(inner) == PHX_MC_OK);
  PHX_CHECK(phx::mc::current_parallelism() == 2);
  PHX_CHECK(phx_mc_parallel_end(outer) == PHX_MC_OK);
  PHX_CHECK(phx::mc::current_parallelism() == 1);
}

void creating_thread() {
  void* policy = nullptr;
  PHX_CHECK(phx_mc_parallel_begin(7, INFINITY, &policy) == PHX_MC_OK);
  int32_t status = PHX_MC_OK;
  std::jthread worker([&] { status = phx_mc_parallel_end(policy); });
  worker.join();
  PHX_CHECK(status == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx::mc::current_parallelism() == 7);
  PHX_CHECK(phx_mc_parallel_end(policy) == PHX_MC_OK);
  PHX_CHECK(phx::mc::current_parallelism() == 1);
}

void expired_restore() {
  void* policy = nullptr;
  PHX_CHECK(phx_mc_parallel_begin(64, 0.0, &policy) == PHX_MC_OK);
  PHX_CHECK(phx::mc::current_parallelism() == 64);
  PHX_CHECK(phx_mc_parallel_end(policy) == PHX_MC_TIMEOUT);
  PHX_CHECK(phx::mc::current_parallelism() == 1);
}

void preparation_clock_independence() {
  phx::mc::NativeParallelPolicy policy(1, 30.0);
  const auto expiry = phx::mc::current_parallel_deadline();
  phx::mc::NativeExecutionScope scope(10, 10, 10, 4096, 120.0);
  PHX_CHECK(scope.import_preparation(2, 1, 45.0));
  PHX_CHECK(scope.prior_seconds() == 45.0);
  PHX_CHECK(phx::mc::current_parallel_deadline() == expiry);
  const double remaining = scope.remaining_wall_seconds();
  PHX_CHECK(remaining > 0.0 && remaining <= 30.0);
  PHX_CHECK(remaining >= phx::mc::current_parallel_lease_seconds());
  PHX_CHECK(scope.charge_external(1, 1));
  const auto counters = scope.evidence();
  PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 3);
  PHX_CHECK(counters[PHX_MC_EXECUTION_SOURCE_QUERIES] == 2);
}

void request_clock_still_debited() {
  phx::mc::NativeParallelPolicy policy(1, 120.0);
  phx::mc::NativeExecutionScope scope(10, 10, 10, 4096, 50.0);
  PHX_CHECK(scope.import_preparation(2, 1, 45.0));
  PHX_CHECK(scope.prior_seconds() == 45.0);
  const double remaining = scope.remaining_wall_seconds();
  PHX_CHECK(remaining > 0.0 && remaining <= 5.0);
}

void preparation_exhausts_request() {
  phx::mc::NativeParallelPolicy policy(1, 120.0);
  phx::mc::NativeExecutionScope scope(10, 10, 10, 4096, 5.0);
  PHX_CHECK(!scope.import_preparation(2, 1, 6.0));
  PHX_CHECK(scope.status() == PHX_MC_TIMEOUT);
  PHX_CHECK(scope.prior_seconds() == 0.0);
  PHX_CHECK(scope.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
}

void preparation_cannot_renew_expired_workflow() {
  phx::mc::NativeParallelPolicy policy(1, 0.0);
  const auto expiry = phx::mc::current_parallel_deadline();
  phx::mc::NativeExecutionScope scope(10, 10, 10, 4096, 120.0);
  PHX_CHECK(!scope.import_preparation(2, 1, 45.0));
  PHX_CHECK(scope.status() == PHX_MC_TIMEOUT);
  PHX_CHECK(scope.prior_seconds() == 0.0);
  PHX_CHECK(phx::mc::current_parallel_deadline() == expiry);
  PHX_CHECK(scope.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
}

void policy_entered_after_scope() {
  phx::mc::NativeExecutionScope scope(10, 10, 10, 4096, 120.0);
  phx::mc::NativeParallelPolicy policy(1, 0.0);
  PHX_CHECK(!scope.checkpoint());
  PHX_CHECK(scope.status() == PHX_MC_TIMEOUT);
}

void nested_scope_preparation_clock() {
  phx::mc::NativeParallelPolicy policy(1, 30.0);
  phx::mc::NativeExecutionScope parent(10, 10, 10, 4096, 120.0);
  {
    phx::mc::NativeExecutionScope child(10, 10, 10, 4096, 120.0);
    PHX_CHECK(child.import_preparation(2, 1, 45.0));
    PHX_CHECK(child.prior_seconds() == 45.0);
    PHX_CHECK(parent.prior_seconds() == 45.0);
    PHX_CHECK(child.remaining_wall_seconds() <= 30.0);
    PHX_CHECK(child.charge_external(1, 1));
  }
  PHX_CHECK(parent.remaining_wall_seconds() <= 30.0);
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 3);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string_view scenario = argv[1];
  if (scenario == "invalid_arguments")
    invalid_arguments();
  else if (scenario == "nested_restore")
    nested_restore();
  else if (scenario == "creating_thread")
    creating_thread();
  else if (scenario == "expired_restore")
    expired_restore();
  else if (scenario == "preparation_clock_independence")
    preparation_clock_independence();
  else if (scenario == "request_clock_still_debited")
    request_clock_still_debited();
  else if (scenario == "preparation_exhausts_request")
    preparation_exhausts_request();
  else if (scenario == "preparation_cannot_renew_expired_workflow")
    preparation_cannot_renew_expired_workflow();
  else if (scenario == "policy_entered_after_scope")
    policy_entered_after_scope();
  else if (scenario == "nested_scope_preparation_clock")
    nested_scope_preparation_clock();
  else
    return 1;
  return phx::mc::test::finish("test_parallel_policy");
}
