"""Real fresh-process TPU inspection contracts; no synthetic provider clients.

These are host lifecycle tests, not TPU hardware qualification. CPU is selected
only to test that diagnostics preserve an existing caller-owned runtime.
"""

from __future__ import annotations

import sys

import pytest

from tests._support.metal_process import json_result, run_python


_HOST_INSPECTION = """
import json
import os
import platform
import sys
import sysconfig
import phydrax
from phydrax import config
from phydrax.backends import (
    assess_tpu_environment, collect_tpu_environment, tpu_environment,
)

before = config.snapshot()
assert not before.backends_initialized
facts = collect_tpu_environment()
assert facts.python_free_threaded == (sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1"))
assert facts.host_release == platform.release()
assert facts.wsl == (
    "microsoft" in platform.release().lower()
    or "wsl" in platform.release().lower()
    or "WSL_INTEROP" in os.environ
    or "WSL_DISTRO_NAME" in os.environ
)
report = tpu_environment()
after = config.snapshot()
assert not after.backends_initialized
assert not facts.device_probed and not report.facts.device_probed
assert facts.devices == () and report.facts.devices == ()
assert facts.runtime_platform is None and facts.runtime_version is None
assert facts.local_device_count is None and facts.global_device_count is None
assert facts.process_count is None and facts.process_index is None
assert assess_tpu_environment(report.facts) == report
assert not report.hardware_qualified
assert before.platforms == after.platforms
assert before.matmul_precision == after.matmul_precision
assert before.enable_x64 == after.enable_x64
print(json.dumps({"probed": report.facts.device_probed, "initialized": after.backends_initialized}))
"""


def test_default_inspection_is_device_inert_in_a_fresh_process() -> None:
    result = json_result(
        run_python(_HOST_INSPECTION, environment={"JAX_PLATFORMS": "cpu"})
    )
    assert result == {"probed": False, "initialized": False}


_CPU_SELECTION = """
import json
import jax
import phydrax
from phydrax import config
from phydrax.backends import tpu_environment

assert config.snapshot().platforms == ("cpu",)
assert not config.snapshot().backends_initialized
report = tpu_environment(initialize_backend=True)
assert not report.facts.device_probed
assert not config.snapshot().backends_initialized
assert not jax.distributed.is_initialized()
assert config.snapshot().platforms == ("cpu",)
assert "platform-not-selected" in [issue.code for issue in report.issues]
# A caller may initialize CPU later. Diagnostics must neither replace it nor
# mistake that real CPU runtime for observed TPU evidence.
devices = jax.devices("cpu")
assert devices and all(device.platform == "cpu" for device in devices)
report = tpu_environment(initialize_backend=True)
assert report.facts.backends_initialized
assert not report.facts.device_probed
assert report.facts.devices == ()
assert report.facts.runtime_platform is None
assert not report.hardware_qualified
assert config.snapshot().platforms == ("cpu",)
print(json.dumps({"platforms": config.snapshot().platforms, "probed": report.facts.device_probed}))
"""


def test_probe_request_respects_an_explicit_cpu_runtime() -> None:
    result = json_result(run_python(_CPU_SELECTION, environment={"JAX_PLATFORMS": "cpu"}))
    assert result == {"platforms": ["cpu"], "probed": False}


_INVALID_OVERRIDE = """
import json
import phydrax
from phydrax import config
from phydrax.backends import tpu_environment

before = config.snapshot()
report = tpu_environment(initialize_backend=True)
assert not report.successful
assert not report.facts.device_probed
assert not config.snapshot().backends_initialized
assert not report.hardware_qualified
assert report.facts.library_override == "/nonexistent/phydrax-test/libtpu.so"
assert any(issue.code == "library-invalid" for issue in report.issues)
assert config.snapshot().platforms == before.platforms
print(json.dumps({"probed": report.facts.device_probed, "successful": report.successful}))
"""


def test_invalid_override_refuses_before_backend_discovery() -> None:
    result = json_result(
        run_python(
            _INVALID_OVERRIDE,
            environment={
                "JAX_PLATFORMS": "cpu",
                "TPU_LIBRARY_PATH": "/nonexistent/phydrax-test/libtpu.so",
            },
        )
    )
    assert result == {"probed": False, "successful": False}


_DISTRIBUTED_LIFECYCLE = """
import json
import socket
import jax
import phydrax
from phydrax import config
from phydrax.backends import tpu_environment

assert not jax.distributed.is_initialized()
assert not config.snapshot().backends_initialized
# Inspection must leave the coordination phase available to the caller.
report = tpu_environment()
assert not report.facts.distributed_initialized
assert not jax.distributed.is_initialized()
assert not config.snapshot().backends_initialized
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
config.configure(config.RuntimeConfig(
    bootstrap=config.RuntimeBootstrap(
        coordinator_address=f"127.0.0.1:{port}",
        num_processes=1,
        process_id=0,
        initialization_timeout_seconds=30,
    ),
))
assert jax.distributed.is_initialized()
assert not config.snapshot().backends_initialized
report = tpu_environment(initialize_backend=True)
assert report.facts.distributed_initialized
assert not report.facts.device_probed
assert not config.snapshot().backends_initialized
assert jax.distributed.is_initialized()
assert config.snapshot().platforms == ("cpu",)
jax.distributed.shutdown()
print(json.dumps({"coordinated": True, "probed": report.facts.device_probed}))
"""


def test_inspection_preserves_caller_owned_distributed_bootstrap_order() -> None:
    result = json_result(
        run_python(
            _DISTRIBUTED_LIFECYCLE,
            environment={"JAX_PLATFORMS": "cpu"},
            timeout=90.0,
        )
    )
    assert result == {"coordinated": True, "probed": False}


_ADMISSION = """
import json
import sys
from jax._src.xla_bridge import backends_are_initialized

selection = sys.argv[1]
try:
    import phydrax
    from phydrax import config
    if selection == "configure":
        config.configure(config.RuntimeConfig(platforms=("tpu",)))
except RuntimeError as error:
    assert type(error).__name__ == "ConfigurationApplicationError", repr(error)
    assert getattr(error, "setting") == "platforms", error
    assert not backends_are_initialized()
    assert str(error)
    print(json.dumps({"admitted": False, "initialized": False}))
else:
    from phydrax.backends import tpu_environment
    report = tpu_environment()
    assert report.successful, report.issues
    assert not report.hardware_qualified
    assert not report.facts.device_probed
    assert config.snapshot().platforms == ("tpu",)
    assert not backends_are_initialized()
    print(json.dumps({"admitted": True, "initialized": False}))
"""


@pytest.mark.parametrize("selection", ["startup", "configure"])
def test_explicit_tpu_admission_precedes_backend_initialization(selection: str) -> None:
    result = json_result(
        run_python(
            _ADMISSION,
            selection,
            environment={"JAX_PLATFORMS": "tpu" if selection == "startup" else ""},
        )
    )
    assert isinstance(result["admitted"], bool)
    assert result["initialized"] is False


_UNATTRIBUTED_OVERRIDE = """
import hashlib
import json
import sys
from pathlib import Path
import phydrax
from phydrax import config
from phydrax.backends import tpu_environment

report = tpu_environment(initialize_backend=True)
path = Path(sys.executable).resolve()
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert report.facts.library_path == str(path)
assert report.facts.library_sha256 == digest
assert not report.facts.library_from_distribution
assert not report.successful
assert not report.facts.device_probed
assert not config.snapshot().backends_initialized
print(json.dumps({"attributed": False, "probed": False}))
"""


def test_unrelated_real_override_is_identified_but_never_loaded() -> None:
    # The actual interpreter is a real file, not a simulated libtpu provider.
    result = json_result(
        run_python(
            _UNATTRIBUTED_OVERRIDE,
            environment={"JAX_PLATFORMS": "cpu", "TPU_LIBRARY_PATH": sys.executable},
        )
    )
    assert result == {"attributed": False, "probed": False}
