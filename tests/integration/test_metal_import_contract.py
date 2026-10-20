"""Phydrax imports, host diagnostics and runtime selection with the Metal companion.

This is a host file: it never initializes a JAX backend in the test process.
Each case observes a fresh child interpreter, because import-time and
backend-initialization effects are only visible from a clean process.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import pytest

from tests._support.metal_process import json_result, run_python
from tests._support.metal_qualification import (
    METAL_RUNTIME_KEYS,
    METAL_STARTUP_ENVIRONMENT,
)


type RecordProperty = Callable[[str, object], None]

_AUTOMATIC: dict[str, str] = {}

# Loaded native images and the public JAX surface, observed in the child.
_OBSERVATION = '''
import ctypes
import json
import sys


def loaded_images():
    if sys.platform != "darwin":
        return []
    system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    system._dyld_image_count.restype = ctypes.c_uint32
    system._dyld_get_image_name.restype = ctypes.c_char_p
    system._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    names = (system._dyld_get_image_name(index) for index in range(system._dyld_image_count()))
    return [name.decode() for name in names if name]


def companion_images():
    return [name for name in loaded_images() if "phydrax_metal" in name]


def jax_surface():
    """Identity of every public JAX callable and the value of every JAX option."""
    import jax
    import jax.lax
    import jax.nn
    import jax.numpy
    import jax.random
    import jax.scipy.special

    callables = {
        f"{module.__name__}.{name}": id(value)
        for module in (jax, jax.lax, jax.nn, jax.numpy, jax.random, jax.scipy.special)
        for name, value in vars(module).items()
        if not name.startswith("_") and callable(value)
    }
    from phydrax._jax_config import configuration_values
    options = {name: repr(value) for name, value in configuration_values().items()}
    return callables, options


def surface_changes(before, after):
    changed = sorted(
        name for name in set(before[0]) | set(after[0]) if before[0].get(name) != after[0].get(name)
    )
    changed += sorted(
        name for name in set(before[1]) | set(after[1]) if before[1].get(name) != after[1].get(name)
    )
    return changed
'''


def _companion_installed() -> bool:
    try:
        metadata.distribution("phydrax-metal")
    except metadata.PackageNotFoundError:
        return False
    return True


def _require_companion() -> None:
    if not _companion_installed():
        pytest.skip("Install phydrax[metal] to run Apple GPU cases.")


_DIAGNOSTICS = (
    _OBSERVATION
    + """
import phydrax
from phydrax import config
from phydrax.backends import assess_metal_environment, metal_environment

assert not config.snapshot().backends_initialized
host_only = metal_environment(initialize_backend=False)
assert assess_metal_environment(host_only.facts) == host_only
snapshot = config.snapshot()
assert not snapshot.backends_initialized
assert not host_only.facts.device_probed and not host_only.facts.backends_initialized
assert host_only.facts.runtime_platform is None
assert "phydrax_metal.plugin" not in sys.modules
assert companion_images() == [], companion_images()
print(json.dumps({
    "codes": [issue.code for issue in host_only.issues],
    "platforms": snapshot.platforms,
}))
"""
)


def test_backend_diagnostics_import_is_device_inert() -> None:
    result = json_result(run_python(_DIAGNOSTICS, environment=_AUTOMATIC))
    codes = result["codes"]
    assert isinstance(codes, list)
    assert result["platforms"] is None
    if not _companion_installed():
        assert "companion-missing" in codes


_STARTUP = (
    _OBSERVATION
    + """
from jax._src.xla_bridge import backends_are_initialized

# Startup admission runs while `phydrax` is imported, so its refusal type is
# identified by name: importing the defining module would itself start Phydrax.
try:
    import phydrax
except RuntimeError as error:
    assert type(error).__name__ == "ConfigurationApplicationError", repr(error)
    assert getattr(error, "setting") == "platforms", error
    assert not backends_are_initialized()
    assert companion_images() == [], companion_images()
    print(json.dumps({"admitted": False, "message": str(error)}))
else:
    from phydrax import config
    from phydrax.backends import metal_environment

    snapshot = config.snapshot()
    assert snapshot.platforms == ("mps",) and snapshot.enable_x64
    assert not snapshot.backends_initialized
    report = metal_environment(initialize_backend=False)
    assert report.successful, report.issues
    assert not config.snapshot().backends_initialized
    assert companion_images() == [], companion_images()
    print(json.dumps({"admitted": True, "message": ""}))
"""
)


def test_startup_mps_selection_is_admitted_from_metadata_before_backend_use() -> None:
    result = json_result(run_python(_STARTUP, environment=METAL_STARTUP_ENVIRONMENT))
    if _companion_installed():
        assert result["admitted"] is True, result["message"]
    else:
        assert result["admitted"] is False
        assert "phydrax[metal]" in str(result["message"])


_CONFLICT = (
    _OBSERVATION
    + """
from pathlib import Path

import jax
from jax._src.xla_bridge import backends_are_initialized

marker = Path(sys.argv[1])
selection = sys.argv[2]
try:
    import phydrax
    from phydrax import config

    if selection == "configure":
        config.configure(config.RuntimeConfig(platforms=("mps",), enable_x64=True))
except RuntimeError as error:
    assert type(error).__name__ == "ConfigurationApplicationError", repr(error)
    message = str(error)
    setting = getattr(error, "setting")
else:
    raise AssertionError("A foreign mps plugin was admitted.")

assert setting == "platforms", setting
assert not backends_are_initialized()
# A refused explicit request is not applied; a startup selection came from the environment.
assert selection == "startup" or not config.snapshot().platforms
assert "foreign_mps_plugin" not in sys.modules
assert not marker.exists()
print(json.dumps({"message": message}))
"""
)


@pytest.mark.parametrize("selection", ["configure", "startup"])
def test_conflicting_mps_plugin_refuses_before_backend_use(
    tmp_path: Path, selection: str
) -> None:
    # A real installed distribution that registers the same platform token.
    marker = tmp_path / "foreign-plugin-used"
    (tmp_path / "foreign_mps_plugin.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "def initialize():\n"
        f"    Path({str(marker)!r}).write_text('initialized')\n",
        encoding="utf-8",
    )
    distribution = tmp_path / "foreign_mps_plugin-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: foreign-mps-plugin\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (distribution / "entry_points.txt").write_text(
        "[jax_plugins]\nmps = foreign_mps_plugin\n", encoding="utf-8"
    )
    environment = METAL_STARTUP_ENVIRONMENT if selection == "startup" else _AUTOMATIC
    result = json_result(
        run_python(
            _CONFLICT,
            str(marker),
            selection,
            environment=environment,
            import_roots=(tmp_path,),
        )
    )
    assert "foreign-mps-plugin" in str(result["message"])
    assert not marker.exists()


_EXPLICIT_CPU = (
    _OBSERVATION
    + """
import phydrax
from phydrax import config

config.configure(config.RuntimeConfig(platforms=("cpu",), enable_x64=True))
assert not config.snapshot().backends_initialized
before = jax_surface()

import jax
import jax.numpy as jnp

value = jnp.asarray([1.0, 2.0**-40, -1.0], jnp.float64)
total = jax.jit(jnp.sum)(value)
cpu = jax.devices("cpu")[0]
assert total.dtype == jnp.float64 and float(total) == 2.0**-40
assert total.devices() == {cpu}
assert {device.platform for device in jax.devices()} == {"cpu"}
try:
    jax.devices("mps")
except RuntimeError:
    pass
else:
    raise AssertionError("Explicit CPU selection initialized the Metal backend.")
changes = surface_changes(before, jax_surface())
assert changes == [], changes
snapshot = config.snapshot()
assert snapshot.platforms == ("cpu",) and snapshot.enable_x64
print(json.dumps({
    "plugin_module_loaded": "phydrax_metal.plugin" in sys.modules,
    "companion_images": companion_images(),
}))
"""
)


def test_explicit_cpu_with_companion_preserves_jax_semantics(
    record_property: RecordProperty,
) -> None:
    _require_companion()
    result = json_result(run_python(_EXPLICIT_CPU, environment=_AUTOMATIC))
    # JAX may discover (and load) an installed plugin during CPU initialization;
    # that is recorded, while numerical placement and JAX semantics are asserted.
    record_property("metal_plugin_loaded_during_cpu_initialization", json.dumps(result))


_SELECTED_METAL = (
    _OBSERVATION
    + """
import phydrax
from phydrax import config

config.configure(config.RuntimeConfig(platforms=("mps",), enable_x64=True))
selected = config.snapshot()
assert not selected.backends_initialized
assert companion_images() == [], companion_images()
before = jax_surface()

import jax
import jax.numpy as jnp
from phydrax.backends import metal_environment
from tests._support.metal_qualification import metal_runtime_evidence

report = metal_environment()
assert report.successful and report.facts.device_probed, report.issues
device = jax.devices()[0]
assert jax.devices() == [device] and device.platform == "mps"
doubled = jax.jit(lambda value: value * 2.0 + 0.5)(jnp.asarray([1.5, -2.25], jnp.float32))
assert doubled.dtype == jnp.float32 and doubled.devices() == {device}
assert [float(item) for item in doubled] == [3.5, -4.0]
changes = surface_changes(before, jax_surface())
assert changes == [], changes
after = config.snapshot()
for field in (
    "platforms", "enable_x64", "matmul_precision", "dtype_promotion", "rank_promotion",
    "cpu_device_count", "cpu_collectives_implementation", "default_prng_implementation",
    "transfer_guard_device_to_host", "jit_disabled",
):
    assert getattr(after, field) == getattr(selected, field), field
assert after.backends_initialized
assert companion_images(), "The selected Metal runtime did not load the companion library."
print(json.dumps({"evidence": metal_runtime_evidence(doubled)}))
"""
)


def test_selected_metal_initialization_preserves_configuration_and_callables(
    record_property: RecordProperty,
) -> None:
    _require_companion()
    result = json_result(run_python(_SELECTED_METAL, environment=_AUTOMATIC))
    evidence = result["evidence"]
    assert isinstance(evidence, dict) and set(evidence) == METAL_RUNTIME_KEYS, evidence
    assert evidence["platform"] == "mps" and evidence["x64_enabled"] is True
    version = metadata.version("phydrax-metal")
    assert str(evidence["platform_version"]).startswith(f"phydrax-metal {version} ")
    record_property("metal_child_runtime", json.dumps(evidence, sort_keys=True))
