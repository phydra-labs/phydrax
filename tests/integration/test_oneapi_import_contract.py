"""Real fresh-process host contracts; no Intel GPU or numerical claims."""

from __future__ import annotations

import pytest

from tests._support.metal_process import json_result, run_python


_HOST_ONLY = """
import json
import phydrax
from phydrax import config
from phydrax.backends import (
    assess_oneapi_environment,
    collect_oneapi_environment,
    oneapi_environment,
)

before = config.snapshot()
assert not before.backends_initialized
facts = collect_oneapi_environment()
report = oneapi_environment()
assert assess_oneapi_environment(report.facts) == report
assert facts == report.facts
assert not report.facts.device_probed
assert report.facts.devices == ()
assert report.facts.runtime_platform is None
assert report.facts.runtime_version is None
assert report.facts.backend_error is None
assert config.snapshot() == before
print(json.dumps({
    "platforms": before.platforms,
    "device_probed": report.facts.device_probed,
    "codes": [issue.code for issue in report.issues],
}))
"""


@pytest.mark.parametrize("selection", (None, "cpu"), ids=("automatic", "cpu"))
def test_default_diagnostics_are_device_inert(selection: str | None) -> None:
    environment = {} if selection is None else {"JAX_PLATFORMS": selection}
    result = json_result(run_python(_HOST_ONLY, environment=environment))
    assert result["platforms"] == (None if selection is None else ["cpu"])
    assert result["device_probed"] is False


_CPU = """
import json
import phydrax
from phydrax import config
from phydrax.backends import oneapi_environment

before = config.snapshot()
assert before.platforms == ("cpu",)
report = oneapi_environment(initialize_backend=True)
assert not report.facts.device_probed
assert not report.facts.backends_initialized
assert report.facts.devices == ()
assert report.facts.backend_error is None
assert config.snapshot() == before
print(json.dumps({"codes": [issue.code for issue in report.issues]}))
"""


def test_explicit_probe_never_overrides_cpu_selection() -> None:
    result = json_result(run_python(_CPU, environment={"JAX_PLATFORMS": "cpu"}))
    codes = result["codes"]
    assert isinstance(codes, list)
    assert "platform-not-selected" in codes


_ADMISSION = """
import json
import phydrax
from phydrax import config
from phydrax.backends import assess_oneapi_environment, collect_oneapi_environment

before = config.snapshot()
facts = collect_oneapi_environment()
assessment = assess_oneapi_environment(facts)
assert not before.backends_initialized
try:
    config.configure(config.RuntimeConfig(platforms=("oneapi",)))
except RuntimeError as error:
    assert not assessment.successful, (assessment, error)
    assert type(error).__name__ == "ConfigurationApplicationError", repr(error)
    assert error.setting == "platforms", error
    assert config.snapshot() == before
    admitted = False
else:
    assert assessment.successful, assessment.issues
    assert config.snapshot().platforms == ("oneapi",)
    admitted = True
assert not config.snapshot().backends_initialized
print(json.dumps({"admitted": admitted, "expected": assessment.successful}))
"""


def test_explicit_oneapi_configuration_uses_only_host_admission() -> None:
    result = json_result(run_python(_ADMISSION, environment={}))
    assert result["admitted"] is result["expected"]


_STARTUP = """
import json
from jax._src.xla_bridge import backends_are_initialized

try:
    import phydrax
except RuntimeError as error:
    assert type(error).__name__ == "ConfigurationApplicationError", repr(error)
    assert error.setting == "platforms", error
    admitted = False
else:
    from phydrax import config
    from phydrax.backends import oneapi_environment
    assert config.snapshot().platforms == ("oneapi",)
    report = oneapi_environment()
    assert report.successful, report.issues
    assert not report.facts.device_probed
    admitted = True
assert not backends_are_initialized()
print(json.dumps({"admitted": admitted}))
"""


def test_oneapi_startup_admission_never_initializes_a_backend() -> None:
    result = json_result(run_python(_STARTUP, environment={"JAX_PLATFORMS": "oneapi"}))
    assert isinstance(result["admitted"], bool)


_INVALID_ARGUMENT = """
import json
from phydrax import config
from phydrax.backends import oneapi_environment

before = config.snapshot()
try:
    oneapi_environment(initialize_backend="yes")
except TypeError as error:
    assert "boolean" in str(error)
else:
    raise AssertionError("A nonboolean probe request was accepted")
assert config.snapshot() == before
print(json.dumps({"unchanged": True}))
"""


def test_invalid_probe_flag_refuses_without_changing_runtime() -> None:
    result = json_result(
        run_python(_INVALID_ARGUMENT, environment={"JAX_PLATFORMS": "cpu"})
    )
    assert result["unchanged"] is True


_OVERRIDES = """
import json
import sysconfig
from phydrax import config
from phydrax.backends import collect_oneapi_environment, assess_oneapi_environment

before = config.snapshot()
facts = collect_oneapi_environment()
assert dict(facts.environment_overrides)["ONEAPI_ROOT"] == " /external sdk "
assert facts.python_free_threaded is bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
report = assess_oneapi_environment(facts)
assert any(issue.code == "library-overrides" for issue in report.issues)
assert not facts.device_probed
assert facts.devices == ()
assert config.snapshot() == before
print(json.dumps({"preserved": True}))
"""


def test_host_collection_preserves_real_sdk_override_values() -> None:
    result = json_result(
        run_python(
            _OVERRIDES,
            environment={"JAX_PLATFORMS": "cpu", "ONEAPI_ROOT": " /external sdk "},
        )
    )
    assert result["preserved"] is True
