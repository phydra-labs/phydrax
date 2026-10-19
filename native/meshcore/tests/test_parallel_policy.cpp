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
  else
    return 1;
  return phx::mc::test::finish("test_parallel_policy");
}
