// Native lease admission tests use real MPI collectives and PMPI fault injection.
#include <mpi.h>
#include <mpi-ext.h>
#include <dlfcn.h>
#include <cassert>
#include <cstring>
#include <string>

namespace {
std::string scenario;
int world_rank = -1;
int live_communicators = 0;
MPI_Comm shared_fixture = MPI_COMM_NULL;
const char* mpi_library() {
  static Dl_info info{};
  assert(dladdr(reinterpret_cast<void*>(&MPI_Get_library_version), &info));
  return info.dli_fname;
}
}

// CUDA buffers are never used: qualify host admission independently of GPUs.
extern "C" int MPIX_Query_cuda_support() { return 1; }
extern "C" int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm* out) {
  const int status = PMPI_Comm_create_group(comm, group, tag, out);
  if (*out != MPI_COMM_NULL) ++live_communicators;
  return status;
}
extern "C" int MPI_Comm_dup(MPI_Comm comm, MPI_Comm* out) {
  const int status = PMPI_Comm_dup(comm, out);
  if (*out != MPI_COMM_NULL) ++live_communicators;
  return status;
}
extern "C" int MPI_Comm_split_type(MPI_Comm comm, int type, int key, MPI_Info info, MPI_Comm* out) {
  int status;
  if (scenario == "partitioned")
    status = PMPI_Comm_split(comm, world_rank, key, out);
  else
    status = PMPI_Comm_split_type(comm, type, key, info, out);
  if (*out != MPI_COMM_NULL) ++live_communicators;
  shared_fixture = *out;
  if (scenario == "split-error" && world_rank == 0) return MPI_ERR_OTHER;
  return status;
}
extern "C" int MPI_Comm_size(MPI_Comm comm, int* size) {
  const int status = PMPI_Comm_size(comm, size);
  if (comm == shared_fixture && scenario == "size-error" && world_rank == 0)
    return MPI_ERR_OTHER;
  return status;
}
extern "C" int MPI_Comm_free(MPI_Comm* comm) {
  const bool shared = *comm == shared_fixture;
  const int status = PMPI_Comm_free(comm);
  if (status == MPI_SUCCESS) --live_communicators;
  if (shared) shared_fixture = MPI_COMM_NULL;
  if (shared && scenario == "free-error" && world_rank == 0) return MPI_ERR_OTHER;
  return status;
}

#define PHX_COLLECTIVE_MPI_LIBRARY mpi_library()
#define PHX_COLLECTIVE_BUILD_IDENTITY "native-lease-test"
#define PHX_COLLECTIVE_MPI_SHA256 "native-lease-test"
#define PHX_COLLECTIVE_MPI_HEADER_SHA256 "native-lease-test"
#include "../../native/cuda/src/common/collective_lease.cpp"

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(provided == MPI_THREAD_MULTIPLE);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  assert(size == 2 && argc == 2);
  scenario = argv[1];
  // Source rank order is deliberately reversed, independently of process IDs.
  const int32_t processes[] = {10, 20};
  const int32_t ranks[] = {1, 0};
  const int32_t process = world_rank == 1 ? 10 : 20;
  int64_t lease = 0;
  const int32_t wrong_process = world_rank == 1 ? 20 : 10;
  assert(phx_collective_create(MPI_Comm_c2f(MPI_COMM_WORLD), wrong_process,
      processes, ranks, 2, 41, "agreement", "configuration", 10.0, &lease) != 0);
  assert(std::string(phx_collective_last_error()) ==
      "local JAX process is not the declared initialized MPI rank participant");
  assert(lease == 0 && live_communicators == 0);
  const int status = phx_collective_create(MPI_Comm_c2f(MPI_COMM_WORLD), process,
      processes, ranks, 2, 42, "agreement", "configuration", 10.0, &lease);
  const std::string error = phx_collective_last_error();
  int statuses[2] = {-1, -1};
  MPI_Allgather(&status, 1, MPI_INT, statuses, 1, MPI_INT, MPI_COMM_WORLD);
  assert(statuses[0] == statuses[1]);
  if (scenario == "same-host") {
    assert(status == 0 && lease > 0);
    assert(live_communicators == 2);
    assert(phx_collective_validate(lease) == 0);
    assert(phx_collective_retire(lease) == 0);
    assert(phx_collective_validate(lease) != 0);
  } else {
    assert(status != 0 && lease == 0);
    assert(error == "collective lease requires every participating rank in one MPI shared-memory domain");
    assert(phx_collective_validate(lease) != 0);
  }
  assert(live_communicators == 0);
  // A refused attempt must not poison MPI or publish an orphaned lease.
  scenario = "same-host";
  assert(phx_collective_create(MPI_Comm_c2f(MPI_COMM_WORLD), process,
      processes, ranks, 2, 43, "agreement", "configuration", 10.0, &lease) == 0);
  assert(phx_collective_retire(lease) == 0);
  assert(live_communicators == 0);
  MPI_Finalize();
}
