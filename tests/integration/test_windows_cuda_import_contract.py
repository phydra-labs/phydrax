"""Fresh-process Windows JAX bootstrap/refusal and installed-runtime contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows bootstrap contract"
)


@pytest.fixture
def loader_tree(tmp_path: Path) -> Path:
    package = tmp_path / "phydrax_cuda13_pjrt"
    shutil.copytree(ROOT / "native/cuda13_pjrt/python/phydrax_cuda13_pjrt", package)
    shutil.copyfile(ROOT / "phydrax/_pe_dependencies.py", package / "_pe_dependencies.py")
    return tmp_path


def _run(
    script: str, cwd: Path, *, pythonpath: Path | None = None, platforms: str = ""
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = platforms
    if pythonpath is not None:
        environment["PYTHONPATH"] = str(pythonpath)
    else:
        environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("platforms", ["", "cuda"])
def test_bootstrap_failure_survives_jax_discovery_catch(
    loader_tree: Path, platforms: str
) -> None:
    # The fault is injected at the native admission boundary, not as a GPU
    # provider. All source validation, JAX backend selection and refusal are real.
    completed = _run(
        """
import jax
from phydrax_cuda13_pjrt import plugin
original = OSError("driver fixture refused with native status 35")
def fail_native(bridge):
    raise original
plugin._register_native = fail_native
try:
    plugin.initialize()
except RuntimeError as error:
    assert error.__cause__ is original
else:
    raise AssertionError("bootstrap unexpectedly succeeded")
assert plugin.bootstrap_state().status == "failed"
import jax.numpy as jnp
for operation in (jax.devices, jax.devices, lambda: jnp.arange(2)):
    try:
        operation()
    except RuntimeError as error:
        assert "driver fixture refused with native status 35" in str(error), str(error)
    else:
        raise AssertionError("JAX silently selected CPU after CUDA bootstrap failure")
""",
        loader_tree,
        pythonpath=loader_tree,
        platforms=platforms,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_explicit_cpu_skips_native_bootstrap(loader_tree: Path) -> None:
    completed = _run(
        """
import jax
from phydrax_cuda13_pjrt import plugin
def forbidden(bridge):
    raise AssertionError("explicit CPU invoked native bootstrap")
plugin._register_native = forbidden
plugin.initialize()
assert plugin.bootstrap_state().status == "cpu_selected"
assert all(device.platform == "cpu" for device in jax.devices())
""",
        loader_tree,
        pythonpath=loader_tree,
        platforms="cpu",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_module_import_does_not_import_core_or_jax(loader_tree: Path) -> None:
    completed = _run(
        """
import sys
from phydrax_cuda13_pjrt import runtime
assert "jax" not in sys.modules
assert "phydrax" not in sys.modules
assert not runtime._LIBRARIES
""",
        loader_tree,
        pythonpath=loader_tree,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_TEST_WINDOWS_CUDA") != "1",
    reason="Requires built installed wheels and explicitly selected Windows GPU gate",
)
def test_installed_runtime_registers_before_callbacks_and_executes(
    tmp_path: Path,
) -> None:
    completed = _run(
        """
import sys
import jax
from jax._src import xla_bridge
seen = []
def callback(*, c_api):
    seen.append(c_api)
xla_bridge.register_plugin_callbacks(callback)
from phydrax_cuda13_pjrt import plugin, runtime
record = runtime.installation_record()
assert record.jax_version == record.jaxlib_version == "0.12.0"
assert not xla_bridge.backends_are_initialized()
devices = jax.devices("cuda")
assert len(devices) == 1 and devices[0].platform == "gpu"
assert plugin.bootstrap_state().status == "registered"
assert len(seen) == 1 and seen[0] is xla_bridge._backend_factories["cuda"].c_api
assert "phydrax" not in sys.modules
import jax.numpy as jnp
value = jax.jit(lambda x: x @ x)(jnp.eye(16, dtype=jnp.float32))
value.block_until_ready()
assert all(device.platform == "gpu" for device in value.devices())
assert float(jnp.trace(value)) == 16.0
assert any(item.distribution == "phydrax-cuda13-pjrt" for item in runtime.mapped_libraries())
""",
        tmp_path,
        platforms="cuda",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("hook", ["backends", "register_backend_factory", "source"])
def test_hook_admission_failure_repeatedly_refuses(
    loader_tree: Path, hook: str
) -> None:
    completed = _run(
        f"""
import functools
import jax
import jax.numpy as jnp
from jax._src import xla_bridge as xb
from phydrax_cuda13_pjrt import plugin
hook = {hook!r}
invoked = []
if hook == "source":
    def fail_source():
        raise OSError("source inspection denied")
    plugin._jax_contract.validate_installed_sources = fail_source
else:
    original = getattr(xb, hook)
    @functools.wraps(original)
    def replacement(*args, **kwargs):
        invoked.append(hook)
        raise AssertionError("untrusted hook invoked")
    setattr(xb, hook, replacement)
try:
    plugin.initialize()
except RuntimeError as error:
    admission_error = error.__cause__
    assert isinstance(admission_error, (OSError, RuntimeError))
else:
    raise AssertionError("modified hook was admitted")
assert not invoked
assert plugin.bootstrap_state().status == "failed"
assert plugin.bootstrap_state().error == str(admission_error)
for operation in (jax.devices, jax.devices, lambda: jnp.arange(2)):
    try:
        operation()
    except RuntimeError as error:
        assert error.__cause__ is admission_error
    else:
        raise AssertionError("hook admission failure silently selected CPU")
""",
        loader_tree,
        pythonpath=loader_tree,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("platforms", ["", "cuda", "cpu,cuda"])
def test_client_factory_failure_cannot_cache_cpu(
    loader_tree: Path, platforms: str
) -> None:
    completed = _run(
        """
import jax
import jax.numpy as jnp
from jax._src import xla_bridge as xb
from phydrax_cuda13_pjrt import plugin
registry = xb._backend_factories
existing = dict(registry)
def fail_client():
    raise RuntimeError("native PJRT client creation refused")
def register_failing_client(bridge):
    bridge.register_backend_factory("cuda", fail_client, priority=500, fail_quietly=False)
    return ()
plugin._register_native = register_failing_client
plugin.initialize()
assert registry is xb._backend_factories
assert all(registry[name] is value for name, value in existing.items())
for operation in (jax.devices, jax.devices, lambda: jnp.arange(2)):
    try:
        operation()
    except RuntimeError as error:
        assert "native PJRT client creation refused" in str(error), str(error)
    else:
        raise AssertionError("client refusal was bypassed by cached CPU")
""",
        loader_tree,
        pythonpath=loader_tree,
        platforms=platforms,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("failure", ["native", "register_backend_factory"])
def test_discovery_inflight_refusal_cannot_initialize_cpu(
    loader_tree: Path, failure: str
) -> None:
    info = loader_tree / "admission_regression-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Name: admission-regression\nVersion: 1.0\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(
        "[jax_plugins]\nadmission_regression = phydrax_cuda13_pjrt.plugin\n",
        encoding="utf-8",
    )
    completed = _run(
        f"""
import functools
import jax
import jax.numpy as jnp
from jax._src import xla_bridge as xb
from phydrax_cuda13_pjrt import plugin
failure = {failure!r}
invoked = []
if failure == "native":
    def refused(bridge):
        raise RuntimeError("native admission denied during discovery")
    plugin._register_native = refused
else:
    @functools.wraps(xb.register_backend_factory)
    def refused(*args, **kwargs):
        invoked.append(failure)
        raise AssertionError("untrusted hook must never execute")
    xb.register_backend_factory = refused
for operation in (jax.devices, jax.devices, lambda: jnp.arange(2)):
    try:
        operation()
    except RuntimeError as error:
        state = plugin.bootstrap_state()
        assert state.status == "failed"
        assert state.error is not None and state.error in str(error)
        if failure == "native":
            assert state.error == "native admission denied during discovery"
        assert not invoked
    else:
        raise AssertionError("in-flight discovery refusal silently initialized CPU")
assert plugin.bootstrap_state().status == "failed"
assert not xb._backends
""",
        loader_tree,
        pythonpath=loader_tree,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("observer", ["cuda_environment", "_cuda_facts"])
def test_observation_cannot_consume_cuda_refusal(
    loader_tree: Path, observer: str
) -> None:
    completed = _run(
        f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import jax
import jax.numpy as jnp
from phydrax._cuda_environment import cuda_environment
from phydrax._cuda_native import _cuda_facts, CudaNativeUnavailableError
from phydrax_cuda13_pjrt import plugin
def refused(bridge):
    raise RuntimeError("observer must not consume CUDA refusal")
plugin._register_native = refused
try:
    plugin.initialize()
except RuntimeError:
    pass
else:
    raise AssertionError("bootstrap unexpectedly succeeded")
if {observer!r} == "cuda_environment":
    report = cuda_environment()
    assert "observer must not consume CUDA refusal" in report.facts.backend_error
else:
    try:
        _cuda_facts()
    except CudaNativeUnavailableError as error:
        assert "observer must not consume CUDA refusal" in str(error), str(error)
    else:
        raise AssertionError("CUDA facts unexpectedly succeeded")
for operation in (jax.devices, lambda: jnp.arange(2)):
    try:
        operation()
    except RuntimeError as error:
        assert "observer must not consume CUDA refusal" in str(error), str(error)
    else:
        raise AssertionError("observer consumed the only CUDA refusal")
""",
        loader_tree,
        pythonpath=loader_tree,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
