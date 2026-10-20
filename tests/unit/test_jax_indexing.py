"""CPU reference contracts for shared lowering logic, not accelerator evidence."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_SOURCE = Path(__file__).resolve().parents[2] / "phydrax" / "_jax_indexing.py"
_LOAD = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("_indexing_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""


def _run_contract(source: str) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS", "PYTHONPATH")
    }
    environment.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    result = subprocess.run(
        [sys.executable, "-I", "-c", _LOAD + source, str(_SOURCE)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_registration_is_device_inert_and_preserves_jax_surface() -> None:
    _run_contract("""
assert "jax" not in sys.modules
import jax
import jax.lax
import jax.numpy
import jax.random
from jax._src import xla_bridge

owners = (jax, jax.lax, jax.numpy, jax.random)

def surface() -> dict:
    return {
        (owner.__name__, name): id(value)
        for owner in owners
        for name, value in vars(owner).items()
        if not name.startswith("_") and callable(value)
    }

before = surface()
options = dict(jax.config.values)
assert not xla_bridge.backends_are_initialized()
module.register_gather_lowering("cpu")
module.register_gather_lowering("cpu")
module.register_gather_lowering("tpu")
providers = {name for name in sys.modules if name.startswith("jax_plugins.")}
module.register_gather_lowering("tpu")
assert not xla_bridge.backends_are_initialized()
assert surface() == before
assert jax.config.values == options
assert "phydrax" not in sys.modules
module.install_accelerator_lowerings()
module.install_accelerator_lowerings()
assert {name for name in sys.modules if name.startswith("jax_plugins.")} == providers
assert not xla_bridge.backends_are_initialized()
assert surface() == before
assert jax.config.values == options
""")


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_fill_gather_keeps_wide_bounds_complete_windows_and_batched_axes(
    index_kind: str,
) -> None:
    _run_contract(
        f"index_kind = {index_kind!r}\n"
        + """
import jax
import numpy as np
from jax import lax

module.register_gather_lowering("cpu")
limits = np.iinfo(np.dtype(index_kind))
starts = np.asarray([limits.min, 0, 1, 3, 4, 2**32, limits.max], index_kind)
dimensions = lax.GatherDimensionNumbers(
    offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)
)
for kind, fill in (("int64", -(2**60) + 5), ("uint64", 2**63 + 5),
                   ("float32", -17.0), ("complex64", -7 + 3j)):
    values = np.asarray([1, 3, 7, 9, 11], kind)
    if kind == "int64":
        values[0] = 2**62 + 1
    elif kind == "uint64":
        values[0] = 2**63 + 1

    @jax.jit
    def gather(a: jax.Array, i: jax.Array) -> jax.Array:
        return lax.gather(a, i[:, None], dimensions, (2,), mode="fill", fill_value=fill)

    actual = gather(values, starts)
    expected = np.stack([
        values[int(i):int(i) + 2] if 0 <= int(i) <= 3 else np.full(2, fill, kind)
        for i in starts
    ])
    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(actual, expected)

values = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
indices = np.asarray([[[0], [2], [3]], [[limits.max], [1], [2**32]]], index_kind)
batched_dimensions = lax.GatherDimensionNumbers(
    offset_dims=(1, 3), collapsed_slice_dims=(), start_index_map=(2,),
    operand_batching_dims=(0,), start_indices_batching_dims=(0,)
)

@jax.jit
def batched(a: jax.Array, i: jax.Array) -> jax.Array:
    return lax.gather(a, i, batched_dimensions, (1, 2, 3), mode="fill", fill_value=-17.0)

expected = np.full((2, 2, 3, 3), -17, np.float32)
for batch in range(2):
    for window in range(3):
        start = int(indices[batch, window, 0])
        if 0 <= start <= 2:
            expected[batch, :, window, :] = values[batch, :2, start:start + 3]
np.testing.assert_array_equal(batched(values, indices), expected)
"""
    )


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_scatter_transpose_drops_wide_out_of_bounds_cotangents(index_kind: str) -> None:
    _run_contract(
        f"index_kind = {index_kind!r}\n"
        + """
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

module.register_gather_lowering("cpu")
values = np.arange(1, 7, dtype=np.float32)
limits = np.iinfo(np.dtype(index_kind))
dimensions = lax.ScatterDimensionNumbers(
    update_window_dims=(0,), inserted_window_dims=(), scatter_dims_to_operand_dims=(0,)
)
for size in (1, 3):
    updates = np.arange(2, size + 2, dtype=np.float32)
    for operation in (lax.scatter, lax.scatter_add, lax.scatter_mul):
        @jax.jit
        def gradients(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple:
            def loss(x: jax.Array, y: jax.Array) -> jax.Array:
                return jnp.sum(operation(x, i, y, dimensions, mode="drop", unique_indices=True))
            return jax.grad(loss, argnums=(0, 1))(a, u)

        for start in (limits.min, 0, 1, 3, 6, 2**32, limits.max):
            da, du = gradients(values, np.asarray([start], index_kind), updates)
            expected_a = np.ones_like(values)
            expected_u = np.zeros_like(updates)
            if 0 <= start <= len(values) - size:
                region = slice(start, start + size)
                expected_a[region] = (
                    updates if operation is lax.scatter_mul else 0 if operation is lax.scatter else 1
                )
                expected_u[:] = values[region] if operation is lax.scatter_mul else 1
            np.testing.assert_array_equal(da, expected_a)
            np.testing.assert_array_equal(du, expected_u)
"""
    )


def test_other_modes_extended_dtypes_and_empty_shapes_keep_jax_semantics() -> None:
    _run_contract("""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

module.register_gather_lowering("cpu")
dimensions = lax.GatherDimensionNumbers(
    offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
)

@jax.jit
def clip(a: jax.Array, i: jax.Array) -> jax.Array:
    return lax.gather(a, i, dimensions, (1,), mode="clip")

@jax.jit
def promise(a: jax.Array, i: jax.Array) -> jax.Array:
    return lax.gather(a, i, dimensions, (1,), mode="promise_in_bounds")

values = np.asarray([3, 5, 7], np.float32)
np.testing.assert_array_equal(clip(values, np.asarray([[-1], [3]], np.int64)), [3, 7])
np.testing.assert_array_equal(promise(values, np.asarray([[2], [0]], np.int64)), [7, 3])
keys = jax.vmap(jax.random.key)(jnp.arange(3, dtype=jnp.int64))

@jax.jit
def key_gather(a: jax.Array, i: jax.Array) -> jax.Array:
    return lax.gather(a, i, dimensions, (1,), mode="fill")

actual = key_gather(keys, np.asarray([[2], [0]], np.int64))
assert actual.dtype == keys.dtype
np.testing.assert_array_equal(jax.random.key_data(actual), np.asarray(jax.random.key_data(keys))[[2, 0]])

@jax.jit
def empty(a: jax.Array, i: jax.Array) -> jax.Array:
    window = lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,))
    return lax.gather(a, i, window, (0,), mode="fill", fill_value=-1.0)

actual = empty(np.empty(0, np.float32), np.asarray([[0], [2**32]], np.int64))
assert actual.shape == (2, 0)
assert actual.dtype == np.float32
""")
