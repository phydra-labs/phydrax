import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


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
    environment.update({"PYTHONPATH": str(_ROOT), "JAX_PLATFORMS": "cpu"})
    environment.update(extra_environment or {})
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=environment,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cold_root_import_leaves_numerical_modules_and_backends_uninitialized() -> None:
    _child("""
        import sys
        import phydrax
        assert "phydrax.geometry" not in sys.modules
        assert "phydrax.operators" not in sys.modules
        assert "phydrax.solver" not in sys.modules
        assert not phydrax.config.snapshot().backends_initialized
    """)


@pytest.mark.parametrize(
    "load",
    [
        "import phydrax.geometry as geometry",
        "import phydrax; geometry = phydrax.geometry",
    ],
    ids=["explicit-import", "lazy-root-attribute"],
)
def test_geometry_import_has_x64_public_numerical_results(load: str) -> None:
    _child(
        load
        + "\n"
        + textwrap.dedent("""
        from phydrax import config
        import jax.numpy as jnp
        assert config.x64_enabled()
        result = geometry.radial_closest_point(
            jnp.asarray([[2.0, 0.0]], dtype=jnp.float64),
            jnp.asarray([0.0, 0.0], dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            represented_geometry_id="configuration-test-circle",
        )
        assert result.closest_point.dtype == jnp.dtype("float64")
        assert result.closest_point.tolist() == [[1.0, 0.0]]
        assert result.normal_coordinate.tolist() == [1.0]
    """)
    )


def test_startup_cpu_collectives_apply_without_device_discovery() -> None:
    _child(
        """
        import phydrax
        observed = phydrax.config.snapshot()
        assert observed.cpu_collectives_implementation == "gloo"
        assert observed.requested.cpu_collectives_implementation == "gloo"
        assert not observed.backends_initialized
    """,
        extra_environment={"PHYDRAX_CPU_COLLECTIVES": "gloo"},
    )


@pytest.mark.parametrize(
    ("extra_environment", "expected"),
    [
        ({}, "highest"),
        ({"JAX_DEFAULT_MATMUL_PRECISION": "tensorfloat32"}, "tensorfloat32"),
    ],
    ids=["unset", "environment-choice"],
)
def test_startup_defaults_float32_contractions_without_overriding_a_choice(
    extra_environment: dict[str, str], expected: str
) -> None:
    _child(
        f"""
        import phydrax
        observed = phydrax.config.snapshot()
        assert observed.matmul_precision == {expected!r}, observed.matmul_precision
        assert not observed.backends_initialized
    """,
        extra_environment=extra_environment,
    )


# The perimeter of a 2:1 ellipse: 4 a E(m) with m = 1 - (b / a) ** 2 = 3/4. The
# 48-point Gauss-Legendre rule is accurate to about 2e-10 relative here, while
# quadrature constants staged under an x64-off scope lose about seven digits.
_ELLIPSE_PERIMETER_CHECK = textwrap.dedent("""
    import jax.numpy as jnp
    import numpy as np
    from scipy.special import ellipe
    ellipse = phydrax.geometry.Ellipse((0.0, 0.0), (2.0, 1.0)).compile()
    perimeter = ellipse.boundary_measure
    assert perimeter.dtype == jnp.dtype("float64")
    np.testing.assert_allclose(
        float(perimeter), 4.0 * 2.0 * ellipe(0.75), rtol=1.0e-9, atol=0.0
    )
""")


@pytest.mark.parametrize(
    "access",
    [
        "phydrax.geometry.CSGContinuationPolicy",
        "phydrax.geometry.design.CSGContinuationPolicy",
    ],
    ids=["package-reexport", "qualified-nested-facade"],
)
def test_first_access_in_numerical_scope_keeps_process_precision(access: str) -> None:
    _child(
        textwrap.dedent(f"""
        import phydrax
        import phydrax.geometry.design
        from phydrax import config
        with config.context(config.NumericsConfig(enable_x64=False)):
            {access}
        assert config.x64_enabled()
        """)
        + _ELLIPSE_PERIMETER_CHECK
    )


def test_first_access_failure_restores_the_callers_numerical_scope() -> None:
    _child("""
        import jax.numpy as jnp
        import phydrax
        from phydrax import config
        with config.context(config.NumericsConfig(enable_x64=False)):
            try:
                phydrax.geometry.design.NotAnExport
            except AttributeError:
                pass
            else:
                raise AssertionError("unknown export resolved")
            assert jnp.ones(1).dtype == jnp.dtype("float32")
        assert jnp.ones(1).dtype == jnp.dtype("float64")
    """)


def test_first_access_under_a_device_scope_stages_nothing_on_that_device() -> None:
    _child(
        textwrap.dedent("""
        import jax
        import jax.numpy as jnp
        import phydrax
        from phydrax._jax_config import default_device
        second = jax.devices("cpu")[1]
        with default_device(second):
            phydrax.geometry.CSGContinuationPolicy
            assert jnp.zeros(1).device == second
        staged = [array for array in jax.live_arrays() if second in array.devices()]
        assert staged == [], staged
        """)
        + _ELLIPSE_PERIMETER_CHECK,
        extra_environment={"JAX_NUM_CPU_DEVICES": "2"},
    )
