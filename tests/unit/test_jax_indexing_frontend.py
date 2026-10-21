"""Fresh-process CPU consumer contracts; not accelerator qualification evidence."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_SOURCE = Path(__file__).resolve().parents[2] / "phydrax" / "_jax_indexing_frontend.py"
_LOAD = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("_frontend_under_test", sys.argv[1])
frontend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontend)
"""
_INSTALL = """
frontend.install_frontend_index_patch()
import jax
import jax.numpy as jnp
import numpy as np
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


def test_installation_is_host_only_locked_idempotent_and_honestly_identified() -> None:
    _run_contract("""
assert "jax" not in sys.modules
assert frontend.frontend_index_patch_identity() is None
assert "jax" not in sys.modules
import hashlib
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import jax
import jax.lax
import jax.numpy
import jax.random
from jax._src import xla_bridge
from jax._src.numpy import indexing

owners = (jax, jax.lax, jax.numpy, jax.random)
def surface():
    return {(owner.__name__, name): id(value)
            for owner in owners for name, value in vars(owner).items()
            if not name.startswith("_") and callable(value)}

before = surface()
options = dict(jax.config.values)
original = indexing._index_to_gather
original_dynamic = indexing.NDIndexer.to_dynamic_slice
assert not xla_bridge.backends_are_initialized()
with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(lambda _: frontend.install_frontend_index_patch(), range(16)))
patched = indexing._index_to_gather
assert patched is not original
assert not hasattr(patched, "__wrapped__")
assert patched.__name__ != original.__name__
assert indexing.NDIndexer.to_dynamic_slice is not original_dynamic
assert inspect.signature(indexing.NDIndexer.to_dynamic_slice).parameters.keys() == (
    inspect.signature(original_dynamic).parameters.keys()
)
assert surface() == before
assert jax.config.values == options
assert not xla_bridge.backends_are_initialized()
identity = frontend.frontend_index_patch_identity()
assert identity["jax_version"] == jax.__version__
assert identity["upstream_source_sha256"] == hashlib.sha256(
    Path(indexing.__file__).read_bytes()).hexdigest()
assert identity["patch_source_sha256"] == hashlib.sha256(
    Path(sys.argv[1]).read_bytes()).hexdigest()
assert identity["upstream_dynamic_slice_sha256"] == hashlib.sha256(
    inspect.getsource(original_dynamic).encode()).hexdigest()
assert identity["algorithm"] == "point-index-bounds-before-shape-width-narrowing"
identity["algorithm"] = "tampered snapshot"
assert frontend.frontend_index_patch_identity()["algorithm"] != identity["algorithm"]
jax.devices()
frontend.install_frontend_index_patch()
assert indexing._index_to_gather is patched
for owner, name, original_callable in (
    (indexing, "_index_to_gather", original),
    (indexing.NDIndexer, "to_dynamic_slice", original_dynamic),
):
    installed_callable = getattr(owner, name)
    setattr(owner, name, original_callable)
    for observe in (frontend.install_frontend_index_patch, frontend.frontend_index_patch_identity):
        try:
            observe()
        except RuntimeError as error:
            assert "replaced" in str(error)
        else:
            raise AssertionError("foreign post-install replacement was accepted")
    setattr(owner, name, installed_callable)
""")


def test_installation_guards_sources_and_callables_not_release_labels() -> None:
    _run_contract("""
import functools
from pathlib import Path
from unittest.mock import patch
import jax
from jax._src.numpy import indexing

original = indexing._index_to_gather

def rejected():
    try:
        frontend.install_frontend_index_patch()
    except RuntimeError:
        pass
    else:
        raise AssertionError("unreviewed compiler identity was accepted")
    assert frontend.frontend_index_patch_identity() is None

read_bytes = Path.read_bytes
def foreign_source(path):
    data = read_bytes(path)
    return data + b"\\n# foreign edit\\n" if str(path) == indexing.__file__ else data
with patch.object(Path, "read_bytes", foreign_source):
    rejected()

@functools.wraps(original)
def wrapped(*args, **kwargs):
    return original(*args, **kwargs)
with patch.object(indexing, "_index_to_gather", wrapped):
    rejected()

def foreign(indexer, *, x_sharding, normalize_indices=True):
    raise AssertionError("foreign code should never execute")
original_code = original.__code__
original.__code__ = foreign.__code__
try:
    rejected()
finally:
    original.__code__ = original_code
with patch.object(indexing.NDIndexer, "normalize_indices", foreign):
    rejected()
with patch.object(indexing.NDIndexer, "to_dynamic_slice", foreign):
    rejected()
assert indexing._index_to_gather is original
with patch.object(jax, "__version__", "0.12.1"):
    frontend.install_frontend_index_patch()
    assert frontend.frontend_index_patch_identity()["jax_version"] == "0.12.1"
""")


def test_first_installation_after_backend_initialization_is_rejected() -> None:
    _run_contract("""
import jax
from jax._src.numpy import indexing
original = indexing._index_to_gather
jax.devices()
try:
    frontend.install_frontend_index_patch()
except RuntimeError as error:
    assert "fresh process" in str(error)
else:
    raise AssertionError("late compiler patch installation was accepted")
assert indexing._index_to_gather is original
assert frontend.frontend_index_patch_identity() is None
""")


@pytest.mark.parametrize("kind", ["int64", "uint64", "uint32"])
def test_array_and_scalar_consumers_keep_full_width_bounds_and_all_modes(
    kind: str,
) -> None:
    _run_contract(
        _INSTALL
        + f"kind = {kind!r}\n"
        + """
limits = np.iinfo(kind)
raw = [0, 1, 4, 5, limits.max]
if limits.max > 2**32:
    raw += [2**32, 2**32 + 1]
if limits.min < 0:
    raw += [-1, -5, -6, -(2**32) + 1, limits.min]
indices = np.asarray(raw, kind)
values = np.arange(10, 15, dtype=np.float64)

for wrap in (True, False):
    normalized = [int(i) + len(values) if wrap and int(i) < 0 else int(i) for i in indices]
    for mode in ("fill", "drop", "clip"):
        def get(a, i):
            return a.at[i].get(mode=mode, fill_value=-99, wrap_negative_indices=wrap)
        expected = [values[min(max(i, 0), 4)] if mode == "clip" else
                    values[i] if 0 <= i < 5 else -99 for i in normalized]
        for execute in (get, jax.jit(get)):
            np.testing.assert_array_equal(execute(jnp.asarray(values), indices), expected)
        # Host integers, NumPy scalar integers, and rank-zero device arrays all
        # reach the real consumer with its default strategy.
        for raw_index, expected_value in zip(indices, expected):
            for scalar in (int(raw_index), raw_index, jnp.asarray(raw_index)):
                np.testing.assert_array_equal(get(jnp.asarray(values), scalar), expected_value)
        def add(a, i, u):
            return a.at[i].add(u, mode=mode, wrap_negative_indices=wrap)
        updates = np.arange(1, len(indices) + 1, dtype=np.float64)
        expected_add = values.copy()
        for index, update in zip(normalized, updates):
            if mode == "clip":
                expected_add[min(max(index, 0), 4)] += update
            elif 0 <= index < 5:
                expected_add[index] += update
        np.testing.assert_array_equal(jax.jit(add)(values, indices, updates), expected_add)
        for raw_index, index in zip(indices, normalized):
            expected_set = values.copy()
            if mode == "clip":
                expected_set[min(max(index, 0), 4)] = 71
            elif 0 <= index < 5:
                expected_set[index] = 71
            for scalar in (int(raw_index), raw_index, jnp.asarray(raw_index)):
                actual = jnp.asarray(values).at[scalar].set(71, mode=mode,
                                                           wrap_negative_indices=wrap)
                np.testing.assert_array_equal(actual, expected_set)
    matrix = np.arange(15, dtype=np.float64).reshape(5, 3)
    for raw_index, index in zip(indices, normalized):
        @jax.jit
        def scalar_clip(a, i):
            return a.at[i, :].get(mode="clip", wrap_negative_indices=wrap)
        np.testing.assert_array_equal(
            scalar_clip(matrix, np.asarray(raw_index)), matrix[min(max(index, 0), 4)]
        )

valid = np.asarray([0, 2, 4], kind)
for mode in (None, "promise_in_bounds"):
    np.testing.assert_array_equal(jnp.asarray(values).at[valid].get(mode=mode), values[valid])
    np.testing.assert_array_equal(jnp.asarray(values).at[valid].set(0, mode=mode), [0, 11, 0, 13, 0])
# Default public gather/scatter modes retain valid negative semantics too.
if limits.min < 0:
    np.testing.assert_array_equal(jnp.asarray(values)[jnp.asarray([-1, -5], kind)], [14, 10])
"""
    )


def test_multi_axis_broadcast_slices_newaxes_booleans_and_empty_indices() -> None:
    _run_contract(
        _INSTALL
        + """
values = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
rows = np.asarray([[0], [2**32 + 1]], np.int64)
cols = np.asarray([[0, -1, -(2**32) + 1]], np.int64)

@jax.jit
def gather(a, r, c):
    return a.at[r, :, c, None].get(mode="fill", fill_value=-99)
expected = np.full((2, 3, 4, 1), -99., np.float64)
expected[0, 0, :, 0] = values[0, :, 0]
expected[0, 1, :, 0] = values[0, :, 4]
np.testing.assert_array_equal(gather(values, rows, cols), expected)

@jax.jit
def scatter(a, r, c):
    return a.at[r, :, c, None].add(np.ones((2, 3, 4, 1)), mode="drop")
expected_scatter = values.copy()
expected_scatter[0, :, 0] += 1
expected_scatter[0, :, 4] += 1
np.testing.assert_array_equal(scatter(values, rows, cols), expected_scatter)

array = jnp.asarray(values)
for indices in (
    (None, np.asarray([0, 2], np.int64), Ellipsis, slice(None, None, -2)),
    (np.asarray([True, False, True]), slice(None, None, -1), None, slice(1, 5, 2)),
    (True, np.asarray([0, 2], np.int64), Ellipsis),
    (False, Ellipsis),
    (np.asarray([], np.int64), Ellipsis),
    (np.asarray(1, np.int64), Ellipsis),
    ([0, 2], slice(None), slice(None)),
):
    np.testing.assert_array_equal(array[indices], values[indices])
# Scalar boolean axis insertion must not move the bounds to a different axis.
np.testing.assert_array_equal(
    array.at[True, np.asarray([2**32 + 1], np.int64), ...].get(mode="fill", fill_value=-99),
    np.full((1, 4, 5), -99., np.float64),
)
"""
    )


def test_jit_batching_jvp_and_vjp_use_the_correct_address_mask() -> None:
    _run_contract(
        _INSTALL
        + """
indices = np.asarray([-1, 0, 2**32 + 1, np.iinfo(np.int64).min, 1], np.int64)
values = np.asarray([2, 3, 5, 7], np.float64)
updates = np.asarray([11, 13, 17, 19, 23], np.float64)

def gather(a):
    return a.at[indices].get(mode="fill", fill_value=-99)

def scatter(a, u):
    return a.at[indices].set(u, mode="drop")

primal, tangent = jax.jit(lambda a, t: jax.jvp(gather, (a,), (t,)))(values, np.ones(4))
np.testing.assert_array_equal(primal, [7, 2, -99, -99, 3])
np.testing.assert_array_equal(tangent, [1, 1, 0, 0, 1])
_, pullback = jax.vjp(gather, jnp.asarray(values))
np.testing.assert_array_equal(pullback(jnp.ones(5))[0], [1, 1, 0, 1])
_, tangent = jax.jvp(scatter, (jnp.asarray(values), jnp.asarray(updates)),
                     (jnp.ones(4), jnp.full(5, 2.)))
np.testing.assert_array_equal(tangent, [2, 2, 1, 2])
_, pullback = jax.vjp(scatter, jnp.asarray(values), jnp.asarray(updates))
da, du = pullback(jnp.ones(4))
np.testing.assert_array_equal(da, [0, 0, 1, 0])
np.testing.assert_array_equal(du, [1, 1, 0, 0, 1])
np.testing.assert_array_equal(jax.jit(jax.vmap(gather))(np.stack([values, values + 10])),
                              [[7, 2, -99, -99, 3], [17, 12, -99, -99, 13]])
np.testing.assert_array_equal(
    jax.jit(jax.vmap(scatter))(np.stack([values, values + 10]), np.stack([updates, updates + 1])),
    [[13, 23, 5, 11], [14, 24, 15, 12]],
)
"""
    )


def test_symbolic_shapes_preserve_signed_and_unsigned_point_indexing() -> None:
    _run_contract(
        _INSTALL
        + """
from jax import export
n, = export.symbolic_shape("n")
values = np.asarray([2, 3, 5, 7], np.float64)
for kind in (np.int64, np.uint64):
    indices = np.asarray([0, 1, 2**32 + 1, np.iinfo(kind).max], kind)
    def gather(a, i):
        return a.at[i].get(mode="fill", fill_value=-99)
    exported = export.export(jax.jit(gather))(
        jax.ShapeDtypeStruct((n,), np.float64), jax.ShapeDtypeStruct((4,), kind))
    np.testing.assert_array_equal(exported.call(values, indices), [2, 3, -99, -99])
    def scatter(a, i):
        return a.at[i].add(jnp.ones(4, np.float64), mode="drop")
    exported = export.export(jax.jit(scatter))(
        jax.ShapeDtypeStruct((n,), np.float64), jax.ShapeDtypeStruct((4,), kind))
    np.testing.assert_array_equal(exported.call(values, indices), [3, 4, 5, 7])
"""
    )


def test_device_indices_do_not_require_host_synchronization() -> None:
    _run_contract(
        _INSTALL
        + """
values = jax.device_put(np.arange(12, dtype=np.float64).reshape(4, 3))
indices = jax.device_put(np.asarray([0, 2**32 + 1, -1], np.int64))
scalar = jax.device_put(np.asarray(2**32 + 1, np.uint64))

@jax.jit
def indexed(a, i, s):
    return (
        a.at[i, :].get(mode="fill", fill_value=-99),
        a.at[i, :].set(71, mode="drop"),
        a.at[s, :].get(mode="clip"),
    )

with jax.transfer_guard_device_to_host("disallow"):
    gathered, scattered, clipped = indexed(values, indices, scalar)
    jax.block_until_ready((gathered, scattered, clipped))
np.testing.assert_array_equal(gathered, [[0, 1, 2], [-99, -99, -99], [9, 10, 11]])
np.testing.assert_array_equal(scattered, [[71, 71, 71], [3, 4, 5], [6, 7, 8], [71, 71, 71]])
np.testing.assert_array_equal(clipped, [9, 10, 11])
"""
    )
