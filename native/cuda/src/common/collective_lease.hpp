#pragma once

#include <mpi.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#if defined(__GNUC__)
#define PHX_COLLECTIVE_EXPORT __attribute__((visibility("default")))
#else
#define PHX_COLLECTIVE_EXPORT
#endif

// This registry lives in one shared site-linked DSO. Consumers link that DSO;
// compiling collective_lease.cpp into each component would split ownership.
namespace phx::collective {
struct LeaseState;
class PHX_COLLECTIVE_EXPORT CallGuard final {
 public:
  CallGuard(int64_t lease_id, std::string_view configuration_id,
            std::string_view call_id, int64_t ticket);
  ~CallGuard();
  CallGuard(const CallGuard&) = delete;
  CallGuard& operator=(const CallGuard&) = delete;
  MPI_Comm communicator() const;
  // Keys are canonical component/preparation identities, equal on every rank.
  // Retire callbacks run collectively, in key order, before communicator free.
  void on_retire(const std::string& key, std::function<void(MPI_Comm)> callback);
  // Call only after stream completion, never merely after vendor enqueue.
  void complete();
 private:
  std::shared_ptr<LeaseState> lease_;
  std::unique_lock<std::mutex> lock_;
  bool completed_ = false;
};
}  // namespace phx::collective

extern "C" {
PHX_COLLECTIVE_EXPORT const char* phx_collective_last_error();
PHX_COLLECTIVE_EXPORT const char* phx_collective_build_identity();
PHX_COLLECTIVE_EXPORT const char* phx_collective_mpi_library();
PHX_COLLECTIVE_EXPORT const char* phx_collective_mpi_sha256();
PHX_COLLECTIVE_EXPORT const char* phx_collective_mpi_header_sha256();
PHX_COLLECTIVE_EXPORT const char* phx_collective_mpi_library_version();
PHX_COLLECTIVE_EXPORT int phx_collective_validate_mpi_symbol(void* library_version_symbol);
// The caller supplies an ALREADY initialized communicator and explicit JAX
// process -> source MPI rank mapping. This library never calls MPI_Init/Finalize.
// All participating ranks must belong to one MPI shared-memory domain;
// admission releases its temporary communicator before publishing a lease.
PHX_COLLECTIVE_EXPORT int phx_collective_create(
    int64_t source_fortran, int32_t process_index, const int32_t* processes,
    const int32_t* source_ranks, int32_t count, int32_t tag,
    const char* agreement_id, const char* configuration_id, double timeout_seconds,
    int64_t* lease_id);
PHX_COLLECTIVE_EXPORT int phx_collective_validate(int64_t lease_id);
// call_ids concatenates count canonical 64-byte call fingerprints. The schedule
// identity is derived from the actual transformed program, before numeric entry.
PHX_COLLECTIVE_EXPORT int phx_collective_agree_program(int64_t lease_id, const char* schedule_identity);
PHX_COLLECTIVE_EXPORT int phx_collective_begin(
    int64_t lease_id, const char* call_ids, int32_t count,
    const char* schedule_identity, int64_t* ticket);
PHX_COLLECTIVE_EXPORT int phx_collective_end(int64_t lease_id, int64_t ticket);
PHX_COLLECTIVE_EXPORT int phx_collective_fail(int64_t lease_id, const char* reason);
PHX_COLLECTIVE_EXPORT int phx_collective_retire(int64_t lease_id);
}
