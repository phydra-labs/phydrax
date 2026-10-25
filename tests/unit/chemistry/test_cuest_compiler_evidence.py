"""Raw cuEST evidence fails closed and keeps numerical recipes dynamic."""

from collections.abc import Callable
from inspect import getclosurevars
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from benchmarks import _cuest_kernels as kernels
from benchmarks.cuest_molecular import (
    _DF_FORCE_ATOL,
    _DF_FORCE_RTOL,
    _DF_JK_ATOL,
    _DF_JK_RTOL,
)
from phydrax import config
from phydrax.execution import default_device


def test_raw_kernel_closure_contains_only_static_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = "phydrax_cuest_jk"
    shapes = (
        jax.ShapeDtypeStruct((2, 2), np.float64),
        jax.ShapeDtypeStruct((2, 2), np.float64),
        jax.ShapeDtypeStruct((), np.int32),
    )
    attributes: dict[str, np.int64 | np.float64] = {
        "representation": np.int64(1),
        "fitting_cutoff": np.float64(1.0e-12),
    }
    observed: list[tuple[Array, ...]] = []

    def ffi_call(
        actual_target: str,
        actual_shapes: tuple[jax.ShapeDtypeStruct, ...],
        *,
        has_side_effect: bool,
    ) -> Callable[..., tuple[Array, Array, Array]]:
        assert actual_target == target
        assert actual_shapes == shapes
        assert has_side_effect is True

        def invoke(
            *arguments: Array, **actual_attributes: np.int64 | np.float64
        ) -> tuple[Array, Array, Array]:
            assert actual_attributes == attributes
            observed.append(arguments)
            return (arguments[-2], arguments[-2], jnp.asarray(0, dtype=jnp.int32))

        return invoke

    monkeypatch.setattr(jax.ffi, "ffi_call", ffi_call)
    kernel = kernels._make_kernel(target, shapes, attributes)
    assert isinstance(kernel, FunctionType)
    closure = getclosurevars(kernel).nonlocals
    assert set(closure) == {"target", "shapes", "attributes"}
    assert closure["attributes"] is attributes
    assert closure["shapes"] is shapes

    recipe = (
        np.zeros(4, dtype=np.int64),
        np.zeros((1, 4), dtype=np.int32),
        np.ones(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        np.zeros((1, 4), dtype=np.int32),
        np.ones(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        np.ones(2, dtype=np.float64),
    )
    first_host = recipe + (
        np.eye(2, dtype=np.float64),
        np.ones((1, 2), dtype=np.float64),
    )
    second_host = tuple(value.copy() for value in first_host)
    second_host[-2][0, 0] = 3.0
    cpu = jax.devices("cpu")[0]
    with config.context(config.NumericsConfig(enable_x64=True)), default_device(cpu):
        first = tuple(jax.device_put(value, cpu) for value in first_host)
        second = tuple(jax.device_put(value, cpu) for value in second_host)
        kernel(first)
        kernel(second)
    assert len(observed) == 2
    for expected, actual in zip((first, second), observed, strict=True):
        assert len(actual) == 11
        assert all(a is b for a, b in zip(actual, expected, strict=True))
    assert np.asarray(observed[0][-2])[0, 0] != np.asarray(observed[1][-2])[0, 0]


def test_compiler_sample_validation_reports_real_comparison() -> None:
    reference = np.eye(2, dtype=np.float64)
    actual = reference.copy()
    actual[0, 0] += 0.25 * _DF_JK_ATOL
    result = kernels._validate_outputs(
        0, (actual,), (reference,), ("coulomb",), rtol=_DF_JK_RTOL, atol=_DF_JK_ATOL
    )
    assert all(result["scientific_checks"].values())
    assert result["maximum_absolute_errors"]["coulomb"] > 0.0


@pytest.mark.parametrize("status", (1, -1, 32))
def test_compiler_sample_validation_rejects_native_failure(status: int) -> None:
    reference = np.eye(2, dtype=np.float64)
    with pytest.raises(RuntimeError, match="status_zero"):
        kernels._validate_outputs(
            status,
            (reference,),
            (reference,),
            ("coulomb",),
            rtol=_DF_JK_RTOL,
            atol=_DF_JK_ATOL,
        )


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, 1.0))
def test_compiler_sample_validation_rejects_bad_completed_sample(value: float) -> None:
    actual = np.zeros((2, 3), dtype=np.float64)
    actual[0, 0] = value
    with pytest.raises(RuntimeError, match="matches_host"):
        kernels._validate_outputs(
            0,
            (actual,),
            (np.zeros_like(actual),),
            ("one_electron",),
            rtol=_DF_FORCE_RTOL,
            atol=_DF_FORCE_ATOL,
        )


def test_compiler_sample_validation_rejects_nonfinite_host_reference() -> None:
    reference = np.full((2, 2), np.nan, dtype=np.float64)
    with pytest.raises(RuntimeError, match="coulomb_finite"):
        kernels._validate_outputs(
            0,
            (np.eye(2, dtype=np.float64),),
            (reference,),
            ("coulomb",),
            rtol=_DF_JK_RTOL,
            atol=_DF_JK_ATOL,
        )


def test_compiler_sample_validation_rejects_broadcast_shape_match() -> None:
    with pytest.raises(RuntimeError, match="matches_host"):
        kernels._validate_outputs(
            0,
            (np.zeros((2, 1), dtype=np.float64),),
            (np.zeros((2, 3), dtype=np.float64),),
            ("pulay",),
            rtol=_DF_FORCE_RTOL,
            atol=_DF_FORCE_ATOL,
        )
