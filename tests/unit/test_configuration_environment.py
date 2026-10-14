import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from phydrax import config as pxconfig


_ROOT = Path(__file__).resolve().parents[2]


def _launch_base() -> dict[str, str]:
    result = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PHYDRAX_", "JAX_"))
        and key
        not in {
            "XLA_FLAGS",
            "SLURM_NTASKS",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        }
    }
    result["PYTHONPATH"] = str(_ROOT)
    return result


def test_projection_serializes_declared_runtime_inputs_without_mutating_base() -> None:
    base = {"UNRELATED": "preserved", "JAX_PLATFORM_NAME": "gpu"}
    projected = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=False,
            platforms=("cpu",),
            cpu_device_count=3,
            cpu_collectives_implementation="gloo",
            launch_environment=pxconfig.LaunchEnvironmentConfig(
                xla_flags="--xla_cpu_enable_fast_math=false", preallocate=False
            ),
        ),
        base=base,
    )
    assert projected == {
        "UNRELATED": "preserved",
        "JAX_ENABLE_X64": "0",
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "3",
        "PHYDRAX_CPU_COLLECTIVES": "gloo",
        "XLA_FLAGS": "--xla_cpu_enable_fast_math=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    assert base == {"UNRELATED": "preserved", "JAX_PLATFORM_NAME": "gpu"}


def test_default_base_copies_current_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFIGURATION_UNRELATED", "preserved")
    before = dict(os.environ)
    projected = pxconfig.environment(pxconfig.RuntimeConfig(enable_x64=False))
    assert projected["CONFIGURATION_UNRELATED"] == "preserved"
    assert projected["JAX_ENABLE_X64"] == "0"
    projected["CONFIGURATION_UNRELATED"] = "changed"
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "defaults",
            {
                "JAX_ENABLE_X64": "0",
                "JAX_NUM_CPU_DEVICES": "4",
                "JAX_PLATFORM_NAME": "gpu",
            },
        ),
        (
            "override",
            {"JAX_ENABLE_X64": "1", "JAX_NUM_CPU_DEVICES": "2", "JAX_PLATFORMS": "cpu"},
        ),
    ],
    ids=["preserve-existing", "replace-existing"],
)
def test_merge_modes_respect_external_platform_selection(
    mode: pxconfig.EnvironmentMerge, expected: dict[str, str]
) -> None:
    base = {"JAX_ENABLE_X64": "0", "JAX_NUM_CPU_DEVICES": "4", "JAX_PLATFORM_NAME": "gpu"}
    assert (
        pxconfig.environment(
            pxconfig.RuntimeConfig(
                enable_x64=True, cpu_device_count=2, platforms=("cpu",)
            ),
            base=base,
            mode=mode,
        )
        == expected
    )


def test_defaults_mode_fills_unselected_settings() -> None:
    assert pxconfig.environment(
        pxconfig.RuntimeConfig(enable_x64=True, platforms=("cpu",)),
        base={},
        mode="defaults",
    ) == {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"}


@pytest.mark.parametrize(
    "bootstrap",
    [
        pxconfig.RuntimeBootstrap(),
        pxconfig.RuntimeBootstrap(autodetect=True),
        pxconfig.RuntimeBootstrap(cluster_detection_method="slurm"),
    ],
    ids=["empty", "autodetect", "named-detection"],
)
def test_nonexplicit_bootstrap_cannot_be_projected(
    bootstrap: pxconfig.RuntimeBootstrap,
) -> None:
    with pytest.raises(ValueError, match="explicit coordinator"):
        pxconfig.environment(pxconfig.RuntimeConfig(bootstrap=bootstrap), base={})


def test_bootstrap_projection_round_trips_through_child_startup() -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    bootstrap = pxconfig.RuntimeBootstrap(
        coordinator_address=f"127.0.0.1:{port}",
        num_processes=1,
        process_id=0,
        local_device_ids=(0,),
        coordinator_bind_address=f"127.0.0.1:{port}",
        heartbeat_timeout_seconds=29,
        initialization_timeout_seconds=47,
    )
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(platforms=("cpu",), bootstrap=bootstrap),
        base=_launch_base(),
    )
    source = f"from phydrax import config; import jax; expected = config.RuntimeBootstrap(coordinator_address={bootstrap.coordinator_address!r}, num_processes=1, process_id=0, local_device_ids=(0,), coordinator_bind_address={bootstrap.coordinator_bind_address!r}, heartbeat_timeout_seconds=29, initialization_timeout_seconds=47); assert config.snapshot().requested.bootstrap == expected; assert config.snapshot().distributed_initialized; assert jax.process_count() == 1; jax.distributed.shutdown()"
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1"}, "must be set together"),
        (
            {
                "PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1",
                "PHYDRAX_NUM_PROCESSES": "not-an-integer",
                "PHYDRAX_PROCESS_ID": "0",
            },
            "PHYDRAX_NUM_PROCESSES must be a nonnegative decimal integer",
        ),
        (
            {
                "PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1",
                "PHYDRAX_NUM_PROCESSES": "1",
                "PHYDRAX_PROCESS_ID": "0",
                "PHYDRAX_LOCAL_DEVICE_IDS": "bad",
            },
            "PHYDRAX_LOCAL_DEVICE_IDS must be a nonnegative decimal integer",
        ),
        ({"PHYDRAX_CPU_COLLECTIVES": "invalid"}, "PHYDRAX_CPU_COLLECTIVES"),
    ],
    ids=[
        "incomplete-coordinator",
        "invalid-count",
        "invalid-local-device",
        "invalid-collectives",
    ],
)
def test_malformed_launcher_environment_fails_at_import(
    inputs: dict[str, str], message: str
) -> None:
    environment = _launch_base()
    environment.update(inputs)
    result = subprocess.run(
        [sys.executable, "-c", "import phydrax"],
        env=environment,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert message in result.stderr


def test_native_runtime_environment_has_typed_aggregate_origins() -> None:
    environment = {
        "JAX_ENABLE_X64": "off",
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "3",
        "PHYDRAX_CPU_COLLECTIVES": "gloo",
        "PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1234",
        "PHYDRAX_NUM_PROCESSES": "2",
        "PHYDRAX_PROCESS_ID": "1",
        "PHYDRAX_LOCAL_DEVICE_IDS": "2,0",
        "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29",
        "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS": "47",
        "XLA_FLAGS": "--xla_cpu_enable_fast_math=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "0",
    }
    resolved = pxconfig.resolve(pxconfig.RuntimeConfig, environment=environment)
    assert resolved.value == pxconfig.RuntimeConfig(
        enable_x64=False,
        platforms=("cpu",),
        cpu_device_count=3,
        cpu_collectives_implementation="gloo",
        bootstrap=pxconfig.RuntimeBootstrap(
            coordinator_address="127.0.0.1:1234",
            num_processes=2,
            process_id=1,
            local_device_ids=(2, 0),
            heartbeat_timeout_seconds=29,
            initialization_timeout_seconds=47,
        ),
        launch_environment=pxconfig.LaunchEnvironmentConfig(
            xla_flags=environment["XLA_FLAGS"], preallocate=False
        ),
    )
    origins = {origin.path: origin for origin in resolved.origins}
    assert origins["bootstrap"].environment_keys == tuple(
        sorted(
            key
            for key in environment
            if key.startswith("PHYDRAX_") and key != "PHYDRAX_CPU_COLLECTIVES"
        )
    )
    assert origins["launch_environment"].environment_keys == (
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    assert all(origin.layer == "environment" for origin in resolved.origins)
    assert (
        pxconfig.ResolvedConfig.from_record(pxconfig.RuntimeConfig, resolved.to_record())
        == resolved
    )
    assert not pxconfig.snapshot().distributed_initialized


@pytest.mark.parametrize(
    "record_type",
    [pxconfig.RuntimeConfig, pxconfig.NumericsConfig],
    ids=["runtime", "numerics"],
)
@pytest.mark.parametrize(
    ("values", "overrides", "expected", "layer"),
    [
        ({}, [], False, "environment"),
        ({"enable_x64": True}, [], True, "value"),
        ({"enable_x64": True}, ["enable_x64=False"], False, "override"),
    ],
    ids=["environment", "values", "overrides"],
)
def test_native_environment_precedence(
    values: dict[str, object],
    overrides: list[str],
    expected: bool,
    layer: str,
    record_type: type[pxconfig.RuntimeConfig] | type[pxconfig.NumericsConfig],
) -> None:
    resolved = pxconfig.resolve(
        record_type,
        environment={"JAX_ENABLE_X64": "false"},
        values=values,
        overrides=overrides,
    )
    assert resolved.value.enable_x64 is expected
    origin = next(origin for origin in resolved.origins if origin.path == "enable_x64")
    assert origin.layer == layer
    assert origin.environment_keys == (
        ("JAX_ENABLE_X64",) if layer == "environment" else ()
    )


def test_native_numerics_environment_is_pure_and_restorable() -> None:
    before = pxconfig.snapshot()
    environment = {
        "JAX_ENABLE_X64": "false",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "JAX_NUMPY_DTYPE_PROMOTION": "strict",
        "JAX_NUMPY_RANK_PROMOTION": "raise",
        "JAX_TRANSFER_GUARD_DEVICE_TO_HOST": "disallow",
        "JAX_DISABLE_JIT": "ON",
    }
    resolved = pxconfig.resolve(pxconfig.NumericsConfig, environment=environment)
    assert resolved.value == pxconfig.NumericsConfig(
        enable_x64=False,
        matmul_precision="highest",
        dtype_promotion="strict",
        rank_promotion="raise",
        transfer_guard_device_to_host="disallow",
        disable_jit=True,
    )
    assert {origin.environment_keys[0] for origin in resolved.origins} == set(environment)
    assert (
        pxconfig.ResolvedConfig.from_record(pxconfig.NumericsConfig, resolved.to_record())
        == resolved
    )
    assert pxconfig.snapshot() == before


def test_native_provider_environment_retains_empty_text_and_aggregate_keys() -> None:
    environment = {
        "PHYDRAX_MESHCORE_LIBRARY": "",
        "PHYDRAX_METIS_LIBRARY": "/metis",
        "PHYDRAX_MMG_WORKER": "",
        "PHYDRAX_TIOGA_WORKER": "/tioga",
    }
    resolved = pxconfig.resolve(pxconfig.ProviderConfig, environment=environment)
    assert resolved.value == pxconfig.ProviderConfig(
        library_paths={"meshcore": "", "metis": "/metis"},
        workers={"mmg": "", "tioga": "/tioga"},
    )
    origins = {origin.path: origin.environment_keys for origin in resolved.origins}
    assert origins == {
        "library_paths": ("PHYDRAX_MESHCORE_LIBRARY", "PHYDRAX_METIS_LIBRARY"),
        "workers": ("PHYDRAX_MMG_WORKER", "PHYDRAX_TIOGA_WORKER"),
    }
    assert (
        pxconfig.ResolvedConfig.from_record(pxconfig.ProviderConfig, resolved.to_record())
        == resolved
    )
    overridden = pxconfig.resolve(
        pxconfig.ProviderConfig,
        environment=environment,
        values={"library_paths": {"metis": None}},
        overrides=["workers={'mmg': '/override'}"],
    )
    assert overridden.value.library_paths == {"metis": None}
    assert overridden.value.workers == {"mmg": "/override"}
    assert all(origin.environment_keys == () for origin in overridden.origins)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("YES", True),
        ("t", True),
        ("On", True),
        ("1", True),
        ("NO", False),
        ("f", False),
        ("Off", False),
        ("0", False),
    ],
    ids=["yes", "t", "on", "one", "no", "f", "off", "zero"],
)
def test_native_boolean_uses_upstream_policy(text: str, expected: bool) -> None:
    resolved = pxconfig.resolve(
        pxconfig.NumericsConfig,
        environment={"JAX_ENABLE_X64": text, "JAX_DISABLE_JIT": text},
    )
    assert resolved.value.enable_x64 is expected
    assert resolved.value.disable_jit is expected


@pytest.mark.parametrize(
    "text",
    [" true", "false ", "", "invalid"],
    ids=["leading-space", "trailing-space", "empty", "unknown"],
)
def test_native_boolean_refuses_non_upstream_text(text: str) -> None:
    with pytest.raises(ValueError, match="truth value"):
        pxconfig.resolve(pxconfig.RuntimeConfig, environment={"JAX_ENABLE_X64": text})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        ("false", False),
        ("False", False),
        ("0", False),
        ("FALSE", True),
        ("no", True),
        (" true ", True),
    ],
    ids=["empty", "lower-false", "title-false", "zero", "upper-false", "no", "other"],
)
def test_preallocation_uses_gpu_plugin_policy(text: str, expected: bool | None) -> None:
    resolved = pxconfig.resolve(
        pxconfig.LaunchEnvironmentConfig,
        environment={"XLA_PYTHON_CLIENT_PREALLOCATE": text},
    )
    assert resolved.value.preallocate is expected


@pytest.mark.parametrize(
    "key",
    ["PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS", "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS"],
    ids=["heartbeat", "initialization"],
)
@pytest.mark.parametrize(
    "text", ["0", "-1", "1.5", ""], ids=["zero", "negative", "fraction", "empty"]
)
def test_malformed_native_timeout_is_refused(key: str, text: str) -> None:
    environment = {
        "PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1234",
        "PHYDRAX_NUM_PROCESSES": "1",
        "PHYDRAX_PROCESS_ID": "0",
        key: text,
    }
    with pytest.raises(ValueError, match=key):
        pxconfig.resolve(pxconfig.RuntimeConfig, environment=environment)


def test_supplemental_bootstrap_and_foreign_launcher_do_not_activate_resolution() -> None:
    resolved = pxconfig.resolve(
        pxconfig.RuntimeConfig,
        environment={"SLURM_NTASKS": "2", "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29"},
    )
    assert resolved.value.bootstrap is None
    assert resolved.value.enable_x64 is None


def test_override_replaces_complete_bootstrap_and_clears_stale_options() -> None:
    base = {
        "PHYDRAX_COORDINATOR_ADDRESS": "old:1234",
        "PHYDRAX_NUM_PROCESSES": "9",
        "PHYDRAX_PROCESS_ID": "8",
        "PHYDRAX_LOCAL_DEVICE_IDS": "7",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS": "old:1234",
        "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "9",
        "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS": "11",
        "OTHER": "kept",
    }
    bootstrap = pxconfig.RuntimeBootstrap(
        coordinator_address="new:1234",
        num_processes=2,
        process_id=1,
        heartbeat_timeout_seconds=29,
        initialization_timeout_seconds=47,
    )
    result = pxconfig.environment(pxconfig.RuntimeConfig(bootstrap=bootstrap), base=base)
    assert result == {
        "PHYDRAX_COORDINATOR_ADDRESS": "new:1234",
        "PHYDRAX_NUM_PROCESSES": "2",
        "PHYDRAX_PROCESS_ID": "1",
        "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29",
        "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS": "47",
        "OTHER": "kept",
    }
    assert base["PHYDRAX_LOCAL_DEVICE_IDS"] == "7"


@pytest.mark.parametrize(
    "base",
    [
        {"PHYDRAX_PROCESS_ID": "8"},
        {"PHYDRAX_LOCAL_DEVICE_IDS": "7"},
        {"PHYDRAX_COORDINATOR_BIND_ADDRESS": "old:1234"},
        {"PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29"},
        {"PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS": "47"},
    ],
    ids=["partial-trio", "local-ids", "bind", "heartbeat", "initialization"],
)
def test_defaults_preserves_bootstrap_group_atomically(base: dict[str, str]) -> None:
    bootstrap = pxconfig.RuntimeBootstrap(
        coordinator_address="new:1234", num_processes=2, process_id=1
    )
    assert (
        pxconfig.environment(
            pxconfig.RuntimeConfig(bootstrap=bootstrap), base=base, mode="defaults"
        )
        == base
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ("--xla_force_host_platform_device_count=4", None),
        ("--xla_cpu_enable_fast_math=false", "2"),
        ('--unrelated="--xla_force_host_platform_device_count=4"', "2"),
        (r"--unrelated=prefix\ --xla_force_host_platform_device_count=4", "2"),
    ],
    ids=["topology-selected", "unrelated", "quoted-substring", "escaped-token"],
)
def test_defaults_recognizes_only_standalone_topology_tokens(
    flags: str, expected: str | None
) -> None:
    result = pxconfig.environment(
        pxconfig.RuntimeConfig(cpu_device_count=2),
        base={"XLA_FLAGS": flags},
        mode="defaults",
    )
    assert result.get("JAX_NUM_CPU_DEVICES") == expected
    assert result["XLA_FLAGS"] == flags


def test_override_removes_conflicting_topology_but_preserves_opaque_flags() -> None:
    flags = (
        '--one="with spaces" --xla_force_host_platform_device_count=4 --other=unchanged'
    )
    result = pxconfig.environment(
        pxconfig.RuntimeConfig(cpu_device_count=2), base={"XLA_FLAGS": flags}
    )
    assert result == {
        "JAX_NUM_CPU_DEVICES": "2",
        "XLA_FLAGS": '--one="with spaces"  --other=unchanged',
    }
    explicit = pxconfig.environment(
        pxconfig.RuntimeConfig(
            cpu_device_count=3,
            launch_environment=pxconfig.LaunchEnvironmentConfig(xla_flags=flags),
        ),
        base={},
    )
    assert explicit["JAX_NUM_CPU_DEVICES"] == "3"
    assert explicit["XLA_FLAGS"] == result["XLA_FLAGS"]


@pytest.mark.parametrize(
    ("environment", "expected", "key"),
    [
        (
            {
                "JAX_NUM_CPU_DEVICES": "2",
                "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            },
            2,
            "JAX_NUM_CPU_DEVICES",
        ),
        (
            {
                "XLA_FLAGS": "--xla_force_host_platform_device_count=2 --xla_force_host_platform_device_count=3"
            },
            3,
            "XLA_FLAGS",
        ),
        (
            {
                "JAX_NUM_CPU_DEVICES": "-1",
                "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            },
            4,
            "XLA_FLAGS",
        ),
    ],
    ids=["native-precedence", "last-forced-token", "native-unspecified"],
)
def test_cpu_topology_resolution_matches_upstream_precedence(
    environment: dict[str, str], expected: int, key: str
) -> None:
    resolved = pxconfig.resolve(pxconfig.RuntimeConfig, environment=environment)
    assert resolved.value.cpu_device_count == expected
    origin = next(
        origin for origin in resolved.origins if origin.path == "cpu_device_count"
    )
    assert origin.environment_keys == (key,)


def test_defaults_forced_topology_reaches_real_child() -> None:
    base = _launch_base()
    base["XLA_FLAGS"] = (
        "--xla_force_host_platform_device_count=2 --xla_force_host_platform_device_count=3"
    )
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(platforms=("cpu",), cpu_device_count=1),
        base=base,
        mode="defaults",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax; from phydrax import config; assert config.snapshot().cpu_device_count == 3; assert jax.device_count('cpu') == 3",
        ],
        env=environment,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "keys",
    [
        ["PHYDRAX_COORDINATOR_ADDRESS", "PHYDRAX_NUM_PROCESSES"],
        ["PHYDRAX_COORDINATOR_ADDRESS", "PHYDRAX_NUM_PROCESSES", "PHYDRAX_PROCESS_ID"],
        [
            "PHYDRAX_COORDINATOR_ADDRESS",
            "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS",
            "PHYDRAX_LOCAL_DEVICE_IDS",
            "PHYDRAX_NUM_PROCESSES",
            "PHYDRAX_PROCESS_ID",
        ],
    ],
    ids=["missing-required", "missing-nondefault-timeout", "unexpected-local-ids"],
)
def test_native_bootstrap_origin_refuses_impossible_key_groups(keys: list[str]) -> None:
    resolved = pxconfig.resolve(
        pxconfig.RuntimeConfig,
        environment={
            "PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1234",
            "PHYDRAX_NUM_PROCESSES": "1",
            "PHYDRAX_PROCESS_ID": "0",
            "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS": "29",
        },
    )
    record = resolved.to_record()
    origins = record["origins"]
    assert isinstance(origins, list)
    for origin in origins:
        assert isinstance(origin, dict)
        if origin["path"] == "bootstrap":
            origin["environment_keys"] = keys
    with pytest.raises(ValueError, match="environment"):
        pxconfig.ResolvedConfig.from_record(pxconfig.RuntimeConfig, record)


@pytest.mark.parametrize(
    ("environment", "expected", "key"),
    [
        (
            {"JAX_PLATFORMS": "cpu,gpu", "JAX_PLATFORM_NAME": "tpu"},
            ("cpu", "gpu"),
            "JAX_PLATFORMS",
        ),
        ({"JAX_PLATFORM_NAME": "cpu"}, ("cpu",), "JAX_PLATFORM_NAME"),
        (
            {"JAX_PLATFORMS": "", "JAX_PLATFORM_NAME": "cpu"},
            ("cpu",),
            "JAX_PLATFORM_NAME",
        ),
    ],
    ids=["native-priority", "legacy", "empty-native"],
)
def test_native_platform_environment_precedence(
    environment: dict[str, str], expected: tuple[str, ...], key: str
) -> None:
    resolved = pxconfig.resolve(pxconfig.RuntimeConfig, environment=environment)
    assert resolved.value.platforms == expected
    origin = next(origin for origin in resolved.origins if origin.path == "platforms")
    assert origin.environment_keys == (key,)
    assert (
        pxconfig.ResolvedConfig.from_record(pxconfig.RuntimeConfig, resolved.to_record())
        == resolved
    )


@pytest.mark.parametrize(
    ("record_type", "key", "value"),
    [
        (pxconfig.RuntimeConfig, "JAX_PLATFORMS", 3),
        (pxconfig.RuntimeConfig, "JAX_NUM_CPU_DEVICES", True),
        (pxconfig.RuntimeConfig, "XLA_FLAGS", None),
        (pxconfig.RuntimeConfig, "XLA_PYTHON_CLIENT_PREALLOCATE", 0),
        (pxconfig.RuntimeConfig, "PHYDRAX_LOCAL_DEVICE_IDS", 3),
        (pxconfig.ProviderConfig, "PHYDRAX_MESHCORE_LIBRARY", None),
        (pxconfig.NumericsConfig, "JAX_DEFAULT_MATMUL_PRECISION", None),
    ],
    ids=[
        "platforms",
        "cpu-count",
        "flags",
        "preallocation",
        "local-ids",
        "provider",
        "numerics",
    ],
)
def test_native_environment_refuses_nontext_present_input(
    record_type: (
        type[pxconfig.RuntimeConfig]
        | type[pxconfig.NumericsConfig]
        | type[pxconfig.ProviderConfig]
    ),
    key: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=key):
        pxconfig.resolve(
            record_type,
            environment={  # ty: ignore[invalid-argument-type]
                "PHYDRAX_COORDINATOR_ADDRESS": "127.0.0.1:1234",
                "PHYDRAX_NUM_PROCESSES": "1",
                "PHYDRAX_PROCESS_ID": "0",
                key: value,
            },
        )
