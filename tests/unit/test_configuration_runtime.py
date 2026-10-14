import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from phydrax import config as pxconfig, logging as pxlogging


_ROOT = Path(__file__).resolve().parents[2]


def _child(source: str, *, extra_environment: dict[str, str] | None = None) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("JAX_", "PHYDRAX_"))
        and key
        not in {
            "XLA_FLAGS",
            "SLURM_NTASKS",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
        }
    }
    environment["PYTHONPATH"] = str(_ROOT)
    environment.update(extra_environment or {})
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=environment,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("environment", "expected"),
    [({}, True), ({"JAX_ENABLE_X64": "0"}, False)],
    ids=["native-default", "external-disable"],
)
def test_startup_x64_policy(environment: dict[str, str], expected: bool) -> None:
    # Typed contracts load during startup, so they must observe the same choice.
    _child(
        f"""
        import jax.numpy as jnp
        from phydrax import config, typing as pt
        assert config.x64_enabled() is {expected!r}
        values = pt.parse(jnp.ones(2), pt.Float64[pt.AnyDim], "values")
        assert values.dtype == jnp.dtype({"float64" if expected else "float32"!r})
        """,
        extra_environment=environment,
    )


def test_deferred_load_ignores_the_callers_numerical_scope(tmp_path: Path) -> None:
    package = tmp_path / "deferred_constants"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from phydrax._deferred_exports import deferred_exports\n"
        "__getattr__, __dir__ = deferred_exports(\n"
        '    __name__, globals(), {"CONSTANT": "._leaf"}\n'
        ")\n"
    )
    # Import-time constants with x64 and implicit rank promotion.
    (package / "_leaf.py").write_text(
        "import jax.numpy as jnp\nCONSTANT = jnp.ones((2, 1)) + jnp.ones((2,))\n"
    )
    _child(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import jax.numpy as jnp
        from phydrax import config
        import deferred_constants
        scope = config.NumericsConfig(enable_x64=False, rank_promotion="raise")
        with config.context(scope):
            constant = deferred_constants.CONSTANT
            assert jnp.ones(1).dtype == jnp.dtype("float32")
        assert constant.dtype == jnp.dtype("float64")
        assert constant.shape == (2, 2)
    """)


@pytest.mark.parametrize(
    "exceptional", [False, True], ids=["normal-exit", "exception-exit"]
)
def test_nested_numerical_context_restores_all_fields(exceptional: bool) -> None:
    baseline = pxconfig.snapshot()
    try:
        with pxconfig.context(
            pxconfig.NumericsConfig(
                enable_x64=not baseline.enable_x64, dtype_promotion="strict"
            )
        ):
            outer = pxconfig.snapshot()
            assert pxconfig.x64_enabled() is not baseline.enable_x64
            assert outer.dtype_promotion == "strict"
            with pxconfig.context(
                pxconfig.NumericsConfig(
                    enable_x64=baseline.enable_x64,
                    rank_promotion="raise",
                    disable_jit=True,
                )
            ):
                inner = pxconfig.snapshot()
                assert pxconfig.x64_enabled() is baseline.enable_x64
                assert inner.dtype_promotion == "strict"
                assert inner.rank_promotion == "raise"
                assert inner.jit_disabled
            assert pxconfig.snapshot() == outer
            if exceptional:
                raise RuntimeError("context exit")
    except RuntimeError as error:
        assert exceptional and str(error) == "context exit"
    assert pxconfig.snapshot() == baseline


def test_cpu_device_count_applies_before_backend_initialization() -> None:
    _child("""
        from phydrax import config
        import jax
        assert not config.snapshot().backends_initialized
        config.configure(config.RuntimeConfig(platforms=("cpu",), cpu_device_count=2))
        assert config.snapshot().cpu_device_count == 2
        assert jax.device_count("cpu") == 2
    """)


def test_late_cpu_device_count_is_refused_without_partial_application() -> None:
    _child("""
        from phydrax import config
        import jax
        config.configure(config.RuntimeConfig(platforms=("cpu",), cpu_device_count=1))
        assert jax.device_count("cpu") == 1
        before = config.snapshot()
        try:
            config.configure(config.RuntimeConfig(cpu_device_count=2, enable_x64=False))
        except config.ConfigurationApplicationError as error:
            assert error.setting == "cpu_device_count"
            assert "cpu_device_count" in str(error)
            assert error.applied == ()
        else:
            raise AssertionError("late topology change was accepted")
        assert config.snapshot() == before
    """)


def test_identical_late_cpu_device_count_is_accepted() -> None:
    _child("""
        from phydrax import config
        import jax
        request = config.RuntimeConfig(platforms=("cpu",), cpu_device_count=1)
        config.configure(request)
        assert jax.device_count("cpu") == 1
        before = config.snapshot()
        config.configure(request)
        assert config.snapshot() == before
    """)


def test_launch_environment_is_refused_in_process() -> None:
    before = pxconfig.snapshot()
    with pytest.raises(
        pxconfig.ConfigurationApplicationError, match="launch_environment"
    ):
        pxconfig.configure(
            pxconfig.RuntimeConfig(
                enable_x64=not before.enable_x64,
                launch_environment=pxconfig.LaunchEnvironmentConfig(preallocate=False),
            )
        )
    assert pxconfig.snapshot() == before


def test_logging_configuration_gates_canonical_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    handler = pxlogging.add_json_sink(path, level="TRACE")
    try:
        pxconfig.configure(pxconfig.LoggingConfig(enabled=False))
        assert not pxconfig.logging_enabled()
        pxlogging.emit("INFO", "configuration.disabled", "Not emitted")
        pxconfig.configure(pxconfig.LoggingConfig(enabled=True))
        assert pxconfig.logging_enabled()
        pxlogging.emit("INFO", "configuration.enabled", "Emitted", count=3)
        pxconfig.configure(pxconfig.LoggingConfig(enabled=False))
        pxlogging.emit("INFO", "configuration.disabled", "Not emitted")
    finally:
        pxconfig.configure(pxconfig.LoggingConfig(enabled=False))
        pxlogging.remove_sink(handler)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "configuration.enabled"
    assert records[0]["fields"]["count"] == 3


@pytest.mark.parametrize(
    "value",
    [None, "", "/configured/provider"],
    ids=["none-fallback", "empty-text", "explicit-text"],
)
def test_provider_absent_none_and_text_inputs(value: str | None) -> None:
    _child(
        f"""
        from phydrax import config
        before = config.snapshot()
        assert before.library_paths["meshcore"] == "/environment/library"
        assert before.workers["mmg"] == "/environment/worker"
        config.configure(config.ProviderConfig(library_paths={{"meshcore": {value!r}}}, workers={{"mmg": {value!r}}}))
        after = config.snapshot()
        if {value!r} is None:
            assert "meshcore" not in after.library_paths
            assert "mmg" not in after.workers
        else:
            assert after.library_paths["meshcore"] == {value!r}
            assert after.workers["mmg"] == {value!r}
        assert after.library_paths["metis"] == "/unrelated/library"
    """,
        extra_environment={
            "PHYDRAX_MESHCORE_LIBRARY": "/environment/library",
            "PHYDRAX_MMG_WORKER": "/environment/worker",
            "PHYDRAX_METIS_LIBRARY": "/unrelated/library",
        },
    )


def test_failed_bootstrap_commits_successful_runtime_prefix() -> None:
    _child("""
        from phydrax import config
        before = config.snapshot()
        assert before.enable_x64
        try:
            config.configure(config.RuntimeConfig(
                cpu_collectives_implementation="mpi",
                platforms=("cpu",), cpu_device_count=2, enable_x64=False,
                bootstrap=config.RuntimeBootstrap(cluster_detection_method="phydrax-unsupported-launcher"),
            ))
        except config.ConfigurationApplicationError as error:
            assert error.setting == "bootstrap"
            assert error.applied == (
                "cpu_collectives_implementation", "platforms", "cpu_device_count", "enable_x64"
            )
        else:
            raise AssertionError("unsupported cluster detection unexpectedly succeeded")
        after = config.snapshot()
        assert after.requested == config.RuntimeConfig(
            cpu_collectives_implementation="mpi",
            platforms=("cpu",), cpu_device_count=2, enable_x64=False,
        )
        assert after.cpu_collectives_implementation == "mpi"
        assert after.platforms == ("cpu",)
        assert after.cpu_device_count == 2
        assert not after.enable_x64
        assert not after.distributed_initialized
        assert not after.backends_initialized
    """)


def test_identical_explicit_field_commits_without_claiming_an_effect() -> None:
    # JAX already uses gloo, so the explicit request needs no upstream effect.
    _child("""
        from phydrax import config
        before = config.snapshot()
        assert before.cpu_collectives_implementation == "gloo"
        assert before.requested.cpu_collectives_implementation is None
        try:
            config.configure(config.RuntimeConfig(
                cpu_collectives_implementation="gloo",
                bootstrap=config.RuntimeBootstrap(cluster_detection_method="phydrax-unsupported-launcher"),
            ))
        except config.ConfigurationApplicationError as error:
            assert error.setting == "bootstrap"
            assert error.applied == ()
        else:
            raise AssertionError("unsupported cluster detection unexpectedly succeeded")
        after = config.snapshot()
        assert after.requested.cpu_collectives_implementation == "gloo"
        assert after.requested.bootstrap is None
        assert after.cpu_collectives_implementation == "gloo"
    """)


def test_noncoordinating_bootstrap_records_explicit_intent() -> None:
    _child("""
        from phydrax import config
        bootstrap = config.RuntimeBootstrap(heartbeat_timeout_seconds=29)
        config.configure(config.RuntimeConfig(bootstrap=bootstrap))
        after = config.snapshot()
        assert after.requested.bootstrap == bootstrap
        assert not after.distributed_initialized
        assert not after.backends_initialized
    """)


@pytest.mark.parametrize("initial", [False, True], ids=["off-to-on", "on-to-off"])
def test_process_x64_changes_existing_process_policy(initial: bool) -> None:
    _child(
        f"""
        import jax.numpy as jnp
        from phydrax import config
        assert config.snapshot().enable_x64 is {initial!r}
        config.configure(config.RuntimeConfig(enable_x64={not initial!r}))
        changed = config.snapshot()
        assert changed.enable_x64 is {not initial!r}
        assert changed.requested.enable_x64 is {not initial!r}
        assert jnp.ones(2).dtype == jnp.dtype({"float32" if initial else "float64"!r})
        config.configure(config.RuntimeConfig(enable_x64={initial!r}))
        restored = config.snapshot()
        assert restored.enable_x64 is {initial!r}
        assert restored.requested.enable_x64 is {initial!r}
        assert jnp.ones(2).dtype == jnp.dtype({"float64" if initial else "float32"!r})
        """,
        extra_environment={"JAX_ENABLE_X64": "1" if initial else "0"},
    )


def test_process_x64_request_inside_a_matching_scope_takes_effect() -> None:
    # The scope already shows x64; the process-wide setting must still change.
    _child(
        """
        import jax.numpy as jnp
        from phydrax import config
        with config.context(config.NumericsConfig(enable_x64=True)):
            config.configure(config.RuntimeConfig(enable_x64=True))
        assert config.snapshot().enable_x64 is True
        assert jnp.ones(2).dtype == jnp.dtype("float64")
        """,
        extra_environment={"JAX_ENABLE_X64": "0"},
    )
