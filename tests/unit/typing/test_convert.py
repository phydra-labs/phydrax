from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given

import phydrax.typing as pt
from phydrax import config as pxconfig
from tests._support.strategies import finite_arrays


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


class NodeDim(pt.Dim):
    pass


@given(finite_arrays(dtype=np.float32, shapes=((1,), (2,), (2, 2))))
def test_as_array_converts_host_values_once_and_preserves_category_dtypes(
    values: npt.NDArray[np.generic],
) -> None:
    converted = pt.as_array(values, pt.Float[pt.AnyShape], "values")
    assert isinstance(converted, jax.Array)
    assert converted.dtype == jnp.float32
    np.testing.assert_array_equal(converted, values)

    matrix = pt.as_array([[1, 2], [3, 4]], pt.Float64[NodeDim, NodeDim], "matrix")
    assert matrix.dtype == jnp.float64
    np.testing.assert_array_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])


def test_convert_scenario_1() -> None:
    for invalid in ("1.0", b"1.0", {"a": 1.0}, ["a", "b"]):
        with pytest.raises(TypeError):
            # ty: ignore[invalid-argument-type]
            pt.as_array(invalid, pt.Float64[NodeDim], "values")
    with pytest.raises(ValueError):
        pt.as_array([[1.0, 2.0], [3.0]], pt.Float64[NodeDim, pt.AnyDim], "ragged")
    with pytest.raises(TypeError):
        pt.as_array(np.asarray([1.0 + 1.0j]), pt.Float64[NodeDim], "complex")
    with pytest.raises(TypeError):
        pt.as_array(np.asarray([1.5]), pt.Int32[NodeDim], "float", casting="safe")
    with pytest.raises(TypeError):
        pt.as_array(np.zeros((2,), dtype=np.int32), pt.Float[NodeDim], "integers")

    flags = pt.as_array([True, False], pt.Float64[NodeDim], "flags")
    assert flags.dtype == jnp.float64
    with pytest.raises(TypeError):
        pt.as_array([1.0], pt.HostFloat64[NodeDim], "wrong")
    with pytest.raises(TypeError):
        pt.as_host_array([1.0], pt.Float64[NodeDim], "wrong")
    with pytest.raises(TypeError):
        pt.as_array([1.0], pt.Size[NodeDim], "wrong")


def test_device_and_host_conversion_boundaries_are_explicit() -> None:
    device = jnp.arange(3.0)
    with pxconfig.context(
        pxconfig.NumericsConfig(transfer_guard_device_to_host="disallow")
    ):
        same = pt.as_array(device, pt.Float64[NodeDim], "device")
        cast = pt.as_array(device.astype(jnp.float32), pt.Float64[NodeDim], "device")
    assert same is device
    assert cast.dtype == jnp.float64

    host = pt.as_host_array(device, pt.HostFloat64[NodeDim], "host")
    assert isinstance(host, np.ndarray)
    assert host.dtype == np.float64

    def traced(values: jax.Array) -> npt.NDArray[np.float64]:
        return pt.as_host_array(values, pt.HostFloat64[NodeDim], "traced")

    with pytest.raises(TypeError):
        jax.jit(traced)(device)


@pytest.mark.parametrize("startup", ["0", "1"], ids=["startup-off", "startup-on"])
def test_cached_conversion_tracks_process_and_scoped_x64(startup: str) -> None:
    root = Path(__file__).resolve().parents[3]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("JAX_", "PHYDRAX_"))
    }
    environment.update(PYTHONPATH=str(root), JAX_PLATFORMS="cpu", JAX_ENABLE_X64=startup)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import jax
                import numpy as np
                import pytest
                import phydrax.typing as pt
                from phydrax import config, StrictModule

                class NodeDim(pt.Dim):
                    pass

                class Field(StrictModule):
                    __strict_contract__ = True
                    values: pt.Float64[NodeDim]

                @pt.checked
                def identity(values: pt.Float64[NodeDim]) -> jax.Array:
                    return values

                compiled_check = jax.jit(identity)

                form = pt.Float64[NodeDim]
                host = np.ones(2, dtype=np.float64)
                assert pt.as_host_array(host, pt.HostFloat64[NodeDim], "host") is host
                initial = pt.as_array(host, form, "initial")
                initial_dtype = np.float64 if config.x64_enabled() else np.float32
                assert initial.dtype == np.dtype(initial_dtype)
                assert Field(initial).values is initial
                assert identity(initial) is initial
                assert compiled_check(initial).dtype == initial.dtype

                def convert(values: jax.Array) -> jax.Array:
                    return pt.as_array(values, form, "compiled")

                compiled = jax.jit(convert)
                assert compiled(initial).dtype == initial.dtype
                config.configure(config.RuntimeConfig(enable_x64=True))
                for enabled in (True, False, True):
                    with config.context(config.NumericsConfig(enable_x64=enabled)):
                        dtype = np.float64 if enabled else np.float32
                        values = pt.as_array(host, form, "values")
                        assert values.dtype == np.dtype(dtype)
                        assert pt.parse(values, form, "values") is values
                        assert pt.as_array(values, form, "values") is values
                        assert compiled(values).dtype == np.dtype(dtype)
                        assert Field(values).values is values
                        assert identity(values) is values
                        assert compiled_check(values).dtype == np.dtype(dtype)
                        assert pt.as_host_array(
                            host, pt.HostFloat64[NodeDim], "host"
                        ) is host
                        with pytest.raises(TypeError):
                            pt.parse(np.ones(2, dtype=np.float32),
                                     pt.HostFloat64[NodeDim], "host")
                        if not enabled:
                            with pytest.raises(TypeError, match="safe"):
                                pt.as_array(host, form, "host", casting="safe")
                assert config.x64_enabled()
                assert pt.as_array(host, form, "restored").dtype == np.dtype(np.float64)
            """),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
