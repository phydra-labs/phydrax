#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native interval-program branch semantics against the host interpreter.

The native branch opcodes repeat the host's comparisons, absolute value,
value-identity winner flags, decided comparisons and attainment bound bit for
bit, and the native ledger events reproduce `_BranchLedger`. Whole programs
built only from those operations agree with the host bit for bit, values and
branch evidence; the host's branch oracles also hold on the native route.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from functools import partial
from itertools import product
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, lax

from phydrax._meshcore import interval_program_execute, meshcore_available
from phydrax.geometry import _interval_enclosure as host
from phydrax.geometry._interval_enclosure import (
    IntervalExtensionError,
    prepare_interval_function,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]

_EQUAL, _ABSOLUTE, _EXTREMUM_PAIR, _EXTREMUM_REDUCE, _DECIDE, _ATTAIN = range(19, 25)
_ENDPOINTS = np.asarray(
    (-np.inf, -2.0, -1.0, -0.5, -0.0, 0.0, 5e-324, 0.5, 1.0, 2.0, np.inf, np.nan)
)


def _bits(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)


def _intervals(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint pairs over special values, ordered when neither is NaN."""
    random = np.random.default_rng(seed)
    first = random.choice(_ENDPOINTS, size=count)
    second = random.choice(_ENDPOINTS, size=count)
    swap = second < first
    return np.where(swap, second, first), np.where(swap, first, second)


def _run(
    rows: Sequence[tuple[int, ...]],
    lower: np.ndarray,
    upper: np.ndarray,
    extent: int,
    indices: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    workspace_lower = np.concatenate((lower, np.full(extent, 12345.0)))
    workspace_upper = np.concatenate((upper, np.full(extent, -12345.0)))
    interval_program_execute(
        np.asarray(rows, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        workspace_lower,
        workspace_upper,
    )
    return workspace_lower[lower.size :], workspace_upper[lower.size :]


def _assert_bits(native: tuple[object, ...], reference: tuple[object, ...]) -> None:
    for native_bound, reference_bound in zip(native, reference, strict=True):
        np.testing.assert_array_equal(_bits(native_bound), _bits(reference_bound))


def test_equal_and_absolute_opcodes_match_host_bits() -> None:
    count = 400
    left = _intervals(count, 1)
    right = _intervals(count, 2)
    lower = np.concatenate((left[0], right[0]))
    upper = np.concatenate((left[1], right[1]))
    rows = [
        (_EQUAL, 2 * count, count, 0, -1, count, -1, 0),
        (_EQUAL, 3 * count, count, 0, -1, count, -1, 1),
        (_ABSOLUTE, 4 * count, count, 0, -1, -1, -1, 0),
    ]
    out_lower, out_upper = _run(rows, lower, upper, 3 * count)
    for index, name in enumerate(("eq", "ne")):
        expected = host._comparison(name, left, right)
        span = slice(index * count, (index + 1) * count)
        _assert_bits(
            (out_lower[span], out_upper[span]),
            tuple(np.asarray(bound, dtype=np.float64) for bound in expected),
        )
    span = slice(2 * count, 3 * count)
    _assert_bits((out_lower[span], out_upper[span]), host.interval_absolute(left))


@pytest.mark.parametrize("name", ("min", "max"))
def test_extremum_pair_flags_match_host_winners(name: str) -> None:
    count = 400
    left = _intervals(count, 3)
    right = _intervals(count, 4)
    lower = np.concatenate((left[0], right[0]))
    upper = np.concatenate((left[1], right[1]))
    aux = 0 if name == "min" else 1
    rows = [(_EXTREMUM_PAIR, 2 * count, count, 0, -1, count, -1, aux)]
    flags_lower, flags_upper = _run(rows, lower, upper, 2 * count)
    select = lax.min if name == "min" else lax.max
    closed = jax.make_jaxpr(select)(jnp.zeros(count), jnp.zeros(count))
    (equation,) = closed.jaxpr.eqns
    values = host._ValueIdentity()
    values.bind_inputs(closed.jaxpr.invars)
    extremum = values.extremum(equation, (left, right))
    assert extremum is not None
    for operand in range(2):
        span = slice(operand * count, (operand + 1) * count)
        np.testing.assert_array_equal(flags_lower[span], extremum.decided)
        np.testing.assert_array_equal(flags_upper[span], extremum.winners[operand])


@pytest.mark.parametrize(
    ("minimum", "groups"),
    ((True, None), (False, None), (True, (0, 1, 0, 2)), (False, (0, 0, 1, 1))),
)
def test_extremum_reduction_flags_match_host_winners(
    minimum: bool, groups: tuple[int, ...] | None
) -> None:
    outputs, lanes = 300, 4
    flat_lower, flat_upper = _intervals(outputs * lanes, 5 if minimum else 6)
    lower = flat_lower.reshape((outputs, lanes))
    upper = flat_upper.reshape((outputs, lanes))
    if groups is not None:
        # Lanes of one identity carry one interval.
        first = [groups.index(group) for group in groups]
        lower, upper = lower[:, first], upper[:, first]
    representatives = -1 if groups is None else 0
    indices = () if groups is None else tuple(groups.index(group) for group in groups)
    rows = [
        (
            _EXTREMUM_REDUCE,
            outputs * lanes,
            outputs,
            0,
            -1,
            lanes,
            representatives,
            0 if minimum else 1,
        )
    ]
    flags_lower, flags_upper = _run(
        rows, lower.reshape(-1), upper.reshape(-1), outputs * lanes, indices
    )
    winners, decided = host._extremum_winners(minimum, lower, upper, (1,), groups)
    np.testing.assert_array_equal(
        flags_lower.reshape((outputs, lanes)),
        np.broadcast_to(decided, (outputs, lanes)),
    )
    np.testing.assert_array_equal(flags_upper.reshape((outputs, lanes)), winners)


def test_attainment_opcode_matches_the_host_lower_bound() -> None:
    lower, upper = _intervals(400, 7)
    rows = [(_ATTAIN, 400, 400, 0, -1, -1, -1, 0)]
    out_lower, out_upper = _run(rows, lower, upper, 400)
    _assert_bits((out_lower, out_upper), (np.maximum(lower, 1.0), upper))


def _branch_subset_programs() -> dict[str, Callable[[Array], Array]]:
    """Programs of bit-identical operations only (no rounded arithmetic)."""

    def decided_pair(point: Array) -> Array:
        high = jnp.maximum(point[0], point[1])
        low = jnp.minimum(point[1], point[2])
        return jnp.stack(
            (
                high,
                low,
                (point[0] == high).astype(jnp.float64),
                (point[1] != high).astype(jnp.float64),
                (low == point[2]).astype(jnp.float64),
            )
        )

    def decided_reduction(point: Array) -> Array:
        stacked = jnp.stack((point[0], point[1], point[0]))
        low = jnp.min(stacked)
        return jnp.concatenate(
            (low[None], (stacked == low).astype(jnp.float64), jnp.max(point)[None])
        )

    def same_operands(point: Array) -> Array:
        first, second = point[0], point[0]
        least = jnp.minimum(first, second)
        return jnp.stack((least, (first == least).astype(jnp.float64)))

    def selection(point: Array) -> Array:
        return jnp.where(point[0] > point[1], jnp.abs(point[2]), jnp.abs(point[0]))

    def comparisons(point: Array) -> Array:
        return jnp.stack(
            (
                (point[0] < point[1]).astype(jnp.float64),
                (point[0] >= point[2]).astype(jnp.float64),
                (point[1] == point[2]).astype(jnp.float64),
            )
        )

    return {
        "decided_pair": decided_pair,
        "decided_reduction": decided_reduction,
        "same_operands": same_operands,
        "selection": selection,
        "comparisons": comparisons,
    }


def _subset_boxes() -> tuple[np.ndarray, np.ndarray]:
    grid = (-1.0, -0.25, 0.0, 0.5, 1.0)
    centers = np.asarray(list(product(grid, repeat=3)), dtype=np.float64)
    widths = np.asarray((0.0, 0.1, 0.6), dtype=np.float64)
    radius = np.repeat(widths, centers.shape[0])[:, None] * np.ones((1, 3))
    tiled = np.tile(centers, (widths.size, 1))
    return tiled - radius, tiled + radius


@pytest.mark.parametrize("name", tuple(_branch_subset_programs()))
def test_branch_subset_programs_match_host_bits_and_evidence(name: str) -> None:
    function = _branch_subset_programs()[name]
    lower, upper = _subset_boxes()
    native = prepare_interval_function(function, 3, batch_capacity=16, execution="native")
    reference = prepare_interval_function(function, 3, batch_capacity=16)
    native_result = native.evaluate_branches(lower, upper)
    host_result = reference.evaluate_branches(lower, upper)
    _assert_bits(native_result[:2], host_result[:2])
    _assert_bits(native.evaluate(lower, upper), reference.evaluate(lower, upper))
    np.testing.assert_array_equal(native_result[2], host_result[2])
    np.testing.assert_array_equal(native_result[3], host_result[3])
    assert not np.all(host_result[2]) or name in ("same_operands",)


def _host_oracles() -> ModuleType:
    path = Path(__file__).with_name("test_interval_branch_evidence.py")
    spec = importlib.util.spec_from_file_location("_host_branch_oracles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "oracle",
    (
        "test_minimum_kinks_only_where_the_box_meets_its_tie",
        "test_equal_operands_decide_their_extremum",
        "test_absolute_value_kinks_and_selection_jumps_on_the_plane",
        "test_square_root_kinks_where_its_argument_reaches_zero",
        "test_decided_selections_keep_derivative_enclosures_finite",
    ),
)
def test_host_branch_oracles_hold_on_the_native_route(
    oracle: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _host_oracles()
    monkeypatch.setattr(
        module,
        "prepare_interval_function",
        partial(prepare_interval_function, execution="native"),
    )
    getattr(module, oracle)()


def test_sharp_union_evidence_and_gradients_on_the_native_route() -> None:
    from phydrax.geometry import Ellipsoid, Sphere
    from phydrax.geometry._contracts import AbstractGeometrySource

    def field(geometry: AbstractGeometrySource) -> Callable[[Array], Array]:
        compiled = geometry.compile()

        def value(point: Array) -> Array:
            return compiled.kernel.boundary_field(compiled.state, point[None, :])[0]

        return value

    ellipsoid = Ellipsoid((0.0123, 0.0037, -0.0102), (0.35, 0.28, 0.23))
    spheres = Sphere((-0.37, 0.013, -0.017), 0.19) | Sphere((0.36, -0.019, 0.021), 0.14)
    # Wholly on one sphere's side, straddling the bisector, on the self-union.
    lower = np.asarray(
        ((-0.6, -0.05, -0.05), (-0.02, 0.1, -0.05), (0.3, 0.0, 0.0)), dtype=np.float64
    )
    upper = lower + np.asarray((0.1, 0.1, 0.1))
    for geometry, smooth in (
        (ellipsoid | ellipsoid, (True, True, True)),
        (spheres, (True, False, True)),
    ):
        value = field(geometry)
        native = prepare_interval_function(value, 3, batch_capacity=4, execution="native")
        reference = prepare_interval_function(value, 3, batch_capacity=4)
        native_result = native.evaluate_branches(lower, upper)
        host_result = reference.evaluate_branches(lower, upper)
        np.testing.assert_array_equal(native_result[2], smooth)
        np.testing.assert_array_equal(native_result[2], host_result[2])
        np.testing.assert_array_equal(native_result[3], host_result[3])
        gradient = prepare_interval_function(
            jax.grad(value), 3, batch_capacity=4, execution="native"
        ).evaluate(lower, upper)
        assert np.all(np.isfinite(gradient[0])) and np.all(np.isfinite(gradient[1]))


def test_native_branch_evidence_requires_the_traced_map() -> None:
    def function(point: Array) -> Array:
        return jnp.minimum(point[0], point[1])

    prepared = prepare_interval_function(function, 2, execution="native")
    detached = host.PreparedIntervalFunction(
        closed_jaxpr=prepared.closed_jaxpr,
        input_dimension=2,
        output_shape=(),
        batch_capacity=prepared.batch_capacity,
        constant_intervals=prepared.constant_intervals,
        execution="native",
    )
    box = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(IntervalExtensionError, match="traced point map"):
        detached.evaluate_branches(box, box)
    program = prepared.native_program
    assert program is not None and program.events is None
    workspace = np.zeros((prepared.batch_capacity, 2), dtype=np.float64)
    with pytest.raises(IntervalExtensionError, match="box axes"):
        program.evaluate_branches(workspace, workspace)
