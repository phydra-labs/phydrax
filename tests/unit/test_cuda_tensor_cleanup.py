"""Exercise the native owning-device teardown boundary without a CUDA context."""

import shutil
import subprocess
from pathlib import Path

import pytest


_SOURCE = r"""
#include "tensor_cleanup.hpp"
#include "tensor_registry.hpp"
#include <cstdlib>
#include <cstring>
#include <new>
#include <cassert>
#include <string>
#include <vector>
struct Operations {
  std::string failure;
  std::vector<std::string> calls;
  bool poisoned = false;
  bool invoke(const std::string& name) {
    calls.push_back(name);
    if (name == failure) { poisoned = true; return false; }
    return true;
  }
  bool get_device(int& device) { device = 3; return invoke("get"); }
  bool set_device(int device) { return invoke(device == 3 ? "restore" : "select"); }
  bool drain() { return invoke("drain"); }
};
bool refuse_allocations = false;
void* operator new(std::size_t bytes) {
  if (refuse_allocations) throw std::bad_alloc();
  if (void* value = std::malloc(bytes ? bytes : 1)) return value;
  throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
bool partition_poisoned = false;
int destructions = 0;
struct Entry {
  bool fail;
  explicit Entry(bool fail_) : fail(fail_) {}
  ~Entry() noexcept {
    ++destructions;
    if (fail) {
      partition_poisoned = true;
      phx::cuda::set_cleanup_error("destroy", "injected failure");
    }
  }
};
phx::cuda::Registry<Entry> entries;
void check_poison(bool all, int64_t partition) {
  if (partition_poisoned && (all || partition == 42))
    phx::cuda::set_cleanup_error("partition cleanup failed");
}
phx::cuda::RegistryBase* component_registry() {
  static phx::cuda::TensorRegistry view(entries, check_poison);
  return &view;
}
int32_t component_shutdown() {
  component_registry()->release_all();
  return partition_poisoned ? PHX_CUDA_PROVIDER_ERROR : PHX_CUDA_SUCCESS;
}
void registry_case(bool fail, bool leased) {
  auto entry = std::make_shared<Entry>(fail);
  const auto id = entries.insert(0, 42, 1, entry);
  if (!leased) entry.reset();
  int64_t released = -1;
  const auto expected = fail ? PHX_CUDA_PROVIDER_ERROR : PHX_CUDA_SUCCESS;
  assert(phx_cuda_release(42, &id, 1, &released) == (leased ? PHX_CUDA_SUCCESS : expected));
  if (fail && !leased)
    assert(!std::strcmp(phx_cuda_last_error(), "destroy: injected failure"));
  assert(released == 1);
  assert(destructions == (leased ? 0 : 1));
  entry.reset();
  assert(destructions == 1);
  assert(phx_cuda_release(42, &id, 1, &released) == expected);
  assert(released == 0);
  assert(phx_cuda_release(99, nullptr, 0, &released) == PHX_CUDA_SUCCESS);
  assert(phx_cuda_last_error()[0] == '\0');
  assert(phx_cuda_release_all(&released) == expected);
  assert(released == 0);
  assert(phx_cuda_shutdown() == expected);
  assert(destructions == 1);
  if (fail) assert(std::strstr(phx_cuda_last_error(), "cleanup failed"));
}
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string scenario = argv[1];
  if (scenario == "registry" || scenario == "registry_fail" || scenario == "leased") {
    registry_case(scenario != "registry", scenario == "leased");
    return 0;
  }
  if (scenario == "partition") {
    auto entry = std::make_shared<Entry>(false);
    const auto original = entries.insert(0, 42, 8, entry);
    const auto refreshed = entries.rekey(0, 42, 8, original, entry);
    entry.reset();
    entries.insert(0, 42, 8, std::make_shared<Entry>(false));
    entries.insert(1, 42, 8, std::make_shared<Entry>(false));
    const auto independent = entries.insert(0, 99, 8, std::make_shared<Entry>(false));
    entries.count_rebuild(0, 42);
    int64_t released = -1;
    assert(phx_cuda_release(42, &original, 1, &released) == PHX_CUDA_SUCCESS);
    assert(released == 0);
    assert(phx_cuda_release_partition(42, &released) == PHX_CUDA_SUCCESS);
    assert(released == 3 && destructions == 3);
    assert(!entries.find(0, 42, refreshed));
    assert(entries.find(0, 99, independent));
    for (const auto& stats : entries.stats()) {
      if (stats.partition == 42) {
        assert(stats.live == 0);
        if (stats.device == 0) assert(stats.rebuilds == 1);
      }
    }
    assert(phx_cuda_release_partition(42, &released) == PHX_CUDA_SUCCESS);
    assert(released == 0);
    entries.insert(0, 42, 8, std::make_shared<Entry>(false));
    assert(phx_cuda_release_partition(42, &released) == PHX_CUDA_SUCCESS);
    assert(released == 1 && destructions == 4);
    assert(phx_cuda_release(99, &independent, 1, &released) == PHX_CUDA_SUCCESS);
    assert(released == 1 && destructions == 5);
    return 0;
  }
  if (scenario == "noalloc") {
    component_registry();
    phx::cuda::set_last_error("old error");
    refuse_allocations = true;
    phx::cuda::set_cleanup_error("destroy", "allocation-free diagnostic");
    assert(!std::strcmp(phx_cuda_last_error(), "destroy: allocation-free diagnostic"));
    phx::cuda::set_cleanup_error("restore", "later failure");
    assert(!std::strcmp(phx_cuda_last_error(), "destroy: allocation-free diagnostic"));
    phx::cuda::set_last_error("");
    char detail[2048];
    std::memset(detail, 'x', sizeof(detail) - 1);
    detail[sizeof(detail) - 1] = '\0';
    phx::cuda::set_cleanup_error("destroy", detail);
    assert(std::strlen(phx_cuda_last_error()) < sizeof(detail));
    assert(std::strlen(phx_cuda_last_error()) == 1023);
    partition_poisoned = true;
    int64_t released = -1;
    assert(phx_cuda_release_all(&released) == PHX_CUDA_PROVIDER_ERROR);
    assert(released == 0);
    assert(phx_cuda_shutdown() == PHX_CUDA_PROVIDER_ERROR);
    refuse_allocations = false;
    phx::cuda::set_last_error("later error");
    assert(!std::strcmp(phx_cuda_last_error(), "later error"));
    return 0;
  }
  if (scenario == "free" || scenario == "free_fail" || scenario == "unallocated") {
    Operations owner{scenario == "free_fail" ? "free" : "success"};
    int storage = 0;
    void* pointer = scenario == "unallocated" ? nullptr : &storage;
    int charge = 128, retained = 128, frees = 0, retirements = 0;
    phx::cuda::TensorCleanup<Operations> cleanup(owner, 7);
    auto free = [&](void*) { ++frees; return owner.invoke("free"); };
    auto retire = [&](int bytes) { retained -= bytes; ++retirements; };
    const bool success = cleanup.release_scratch(pointer, charge, free, retire);
    assert(success == (scenario != "free_fail"));
    assert(charge == (success ? 0 : 128));
    assert(retained == charge);
    assert(pointer == (success ? nullptr : &storage));
    assert(owner.poisoned == !success);
    if (success) {
      assert(cleanup.release_scratch(pointer, charge, free, retire));
      assert(frees == (scenario == "unallocated" ? 0 : 1));
      assert(retirements == 1);
    }
    return 0;
  }
  Operations owner{scenario};
  int retained_bytes = 128;
  int releases = 0;
  {
    const int device = scenario == "host" ? -1 : scenario == "same" ? 3 : 7;
    phx::cuda::TensorCleanup<Operations> cleanup(owner, device);
    int storage = 0;
    void* pointer = &storage;
    int charge = retained_bytes;
    cleanup.release_scratch(pointer, charge, [](void*) { return true; },
                            [&](int bytes) { retained_bytes -= bytes; ++releases; });
  }
  if (scenario == "get") {
    assert((owner.calls == std::vector<std::string>{"get"}));
  } else if (scenario == "select") {
    assert((owner.calls == std::vector<std::string>{"get", "select", "restore"}));
  } else if (scenario == "host") {
    assert(owner.calls.empty());
  } else if (scenario == "same") {
    assert((owner.calls == std::vector<std::string>{"get", "drain"}));
  } else {
    assert((owner.calls == std::vector<std::string>{"get", "select", "drain", "restore"}));
  }
  const bool refused = scenario == "get" || scenario == "select" || scenario == "drain";
  assert(retained_bytes == (refused ? 128 : 0));
  assert(releases == (refused ? 0 : 1));
  assert(owner.poisoned == (refused || scenario == "restore"));
  Operations unrelated{"success"};
  { phx::cuda::TensorCleanup<Operations> cleanup(unrelated, 7); assert(cleanup.ready()); }
  assert(!unrelated.poisoned);
}
"""


@pytest.fixture(scope="module")
def cleanup_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.fail("Native tensor cleanup tests require a C++ compiler")
    directory = tmp_path_factory.mktemp("tensor_cleanup")
    source = directory / "cleanup.cpp"
    source.write_text(_SOURCE)
    executable = directory / "cleanup"
    include = Path(__file__).resolve().parents[2] / "native/cuda/src/common"
    # Only external CUDA/XLA types are stubbed; registry and C ABI behavior
    # comes from the production implementation.
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\n"
        "using cudaError_t = int;\n"
        "constexpr cudaError_t cudaSuccess = 0;\n"
        "inline cudaError_t cudaGetDevice(int* value) { *value = 0; return 0; }\n"
        'inline const char* cudaGetErrorString(cudaError_t) { return "test"; }\n'
    )
    ffi = directory / "xla/ffi/api"
    ffi.mkdir(parents=True)
    (ffi / "ffi.h").write_text(
        "#pragma once\n#include <string>\n"
        "using XLA_FFI_Handler = void();\n"
        "namespace xla::ffi { struct Error {\n"
        "static Error Success() { return {}; }\n"
        "static Error Internal(const std::string&) { return {}; }\n"
        "}; }\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-I",
            str(directory),
            "-I",
            str(include),
            "-I",
            str(include.parents[1] / "include"),
            str(source),
            str(include / "capi.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "host",
        "same",
        "get",
        "select",
        "drain",
        "restore",
        "free",
        "free_fail",
        "unallocated",
        "registry",
        "registry_fail",
        "leased",
        "noalloc",
        "partition",
    ],
)
def test_owning_device_teardown_boundary(cleanup_harness: Path, scenario: str) -> None:
    subprocess.run(
        [str(cleanup_harness), scenario], check=True, capture_output=True, text=True
    )
