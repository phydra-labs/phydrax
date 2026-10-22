"""Fault real native teardown calls in isolated CUDA subprocesses."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.cuda_native
_ROOT = Path(__file__).resolve().parents[2]
_FAILURES = {
    "cutensor": (
        "cutensorDestroyPlan",
        "cutensorDestroyOperationDescriptor",
        "cutensorDestroyPlanPreference",
        "cutensorDestroyTensorDescriptor",
        "cutensorDestroy",
        "cudaFree",
        "cudaGetDevice",
        "cudaDeviceSynchronize",
    ),
    "cutensornet": (
        "cutensornetDestroyWorkspaceDescriptor",
        "cutensornetDestroyContractionOptimizerInfo",
        "cutensornetDestroyNetwork",
        "cutensornetDestroy",
        "cutensornetDestroyContractionOptimizerConfig",
        "cudaGetDevice",
        "cudaDeviceSynchronize",
    ),
}
_INTERPOSER = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdlib.h>
#include <string.h>
static void* real_function;
static int attempts;
static int locate(struct dl_phdr_info* info, size_t size, void* ignored) {
  (void)size; (void)ignored;
  if (!strstr(info->dlpi_name, VENDOR)) return 0;
  void* library = dlopen(info->dlpi_name, RTLD_NOW | RTLD_NOLOAD);
  if (library) real_function = dlsym(library, SYMBOL);
  return real_function != NULL;
}
int teardown_attempts(void) { return attempts; }
#ifdef NO_ARGUMENT
int FUNCTION(void) {
#else
int FUNCTION(ARGUMENT_TYPE pointer) {
#endif
  const char* armed = getenv("PHYDRAX_TEST_TEARDOWN_ARMED");
  if (armed) {
    ++attempts;
    if (!strcmp(armed, "fail")) return 1;
  }
  if (!real_function) dl_iterate_phdr(locate, NULL);
  if (!real_function) abort();
#ifdef NO_ARGUMENT
  return ((int (*)(void))real_function)();
#else
  return ((int (*)(ARGUMENT_TYPE))real_function)(pointer);
#endif
}
"""
_WORKER = r"""
import ctypes
import os
import runpy
import sys
import pytest
helpers = runpy.run_path('tests/integration/test_cuda_tensor_contraction.py')
provider, mode, interposer_path, symbol = sys.argv[1:]
try:
    with helpers['_cuda_case'](provider) as runtime:
        values = (helpers['_values']((3, 4), 'float32'), helpers['_values']((2,), 'float32'))
        prepared = helpers['_prepare'](runtime, provider, 'ij,k->i', values)
        runtime.tn.execute_contraction(prepared).value.block_until_ready()
        library = ctypes.CDLL(runtime.component.path)
        release = library.phx_cuda_release_all
        release.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        release.restype = ctypes.c_int32
        shutdown = library.phx_cuda_shutdown
        shutdown.restype = ctypes.c_int32
        interposer = ctypes.CDLL(interposer_path)
        interposer.teardown_attempts.restype = ctypes.c_int
        os.environ['PHYDRAX_TEST_TEARDOWN_ARMED'] = mode
        if symbol == 'cutensornetDestroyContractionOptimizerConfig':
            planner = library.phydrax_cutensornet_plan
            planner.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
                                ctypes.POINTER(ctypes.c_int64), ctypes.c_char_p, ctypes.c_int64]
            planner.restype = ctypes.c_int32
            pairs = (ctypes.c_int32 * 2)()
            count = ctypes.c_int64()
            error = ctypes.create_string_buffer(1024)
            status = planner(b'0 2 2 3 4 0 1 2 4 2 1 2 2 3 2 0 2 0',
                             2**23, 1, 0, pairs, 2, ctypes.byref(count), error, 1024)
            assert interposer.teardown_attempts() == 1
            assert (status != 0) == (mode == 'fail'), (status, error.value)
            assert count.value == (0 if mode == 'fail' else 1)
            os.environ.pop('PHYDRAX_TEST_TEARDOWN_ARMED')
            assert shutdown() == 0, 'Host planner failure poisoned an unrelated memo partition'
            raise SystemExit(0)
        released = ctypes.c_int64()
        status = release(ctypes.byref(released))
        assert released.value > 0
        attempts = interposer.teardown_attempts()
        assert attempts > 0, 'The injected teardown category was not exercised'
        assert (status != 0) == (mode == 'fail'), status
        assert (release(ctypes.byref(released)) != 0) == (mode == 'fail')
        assert released.value == 0
        assert interposer.teardown_attempts() == attempts, 'Repeated release retried a retired resource'
        assert (shutdown() != 0) == (mode == 'fail')
        assert interposer.teardown_attempts() == attempts
        os.environ.pop('PHYDRAX_TEST_TEARDOWN_ARMED')
        from phydrax._cuda_native import release_cuda_component
        if mode == 'fail':
            with pytest.raises(RuntimeError, match='cleanup failed'):
                release_cuda_component(provider)
        else:
            assert release_cuda_component(provider) == 0
except pytest.skip.Exception as error:
    print(str(error))
    raise SystemExit(77)
"""


@pytest.mark.parametrize(
    ("provider", "symbol"),
    [(provider, symbol) for provider, symbols in _FAILURES.items() for symbol in symbols],
)
@pytest.mark.parametrize("mode", ["success", "fail"])
def test_native_tensor_teardown_status(
    tmp_path: Path, provider: str, symbol: str, mode: str
) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.fail("Native teardown fault injection requires a C compiler")
    source = tmp_path / "interpose.c"
    library = tmp_path / "interpose.so"
    vendor = "libcutensornet.so" if provider == "cutensornet" else "libcutensor.so"
    if symbol.startswith("cuda"):
        vendor = "libcudart.so"
    if symbol == "cudaDeviceSynchronize":
        signature = "#define NO_ARGUMENT\n"
    else:
        argument_type = "int*" if symbol == "cudaGetDevice" else "void*"
        signature = f"#define ARGUMENT_TYPE {argument_type}\n"
    source.write_text(
        f'#define VENDOR "{vendor}"\n#define SYMBOL "{symbol}"\n#define FUNCTION {symbol}\n'
        + signature
        + _INTERPOSER
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-ldl", "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [sys.executable, "-c", _WORKER, provider, mode, str(library), symbol],
        cwd=_ROOT,
        env={
            **os.environ,
            "LD_PRELOAD": str(library),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
