#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._geometry_precision import GeometryPrecisionPolicy
from phydrax._meshcore import meshcore_available, MeshcoreUnavailableError
from phydrax.geometry import (
    incircle,
    insphere,
    orient2d,
    orient3d,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
)


requires_meshcore = pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)

ULP = 2.0**-53


def _det(matrix: Any) -> Any:
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    total = Fraction(0)
    for column in range(size):
        if matrix[0][column] == 0:
            continue
        minor = [row[:column] + row[column + 1 :] for row in matrix[1:]]
        total += (-1) ** column * matrix[0][column] * _det(minor)
    return total


def _sign(value: Any) -> Any:
    return (value > 0) - (value < 0)


def _rows(*points: Any) -> Any:
    return [[Fraction(float(value)) for value in point] for point in points]


def reference_orient2d(a: Any, b: Any, c: Any) -> Any:
    rows = _rows(a, b, c)
    return _sign(_det([row + [Fraction(1)] for row in rows]))


def reference_orient3d(a: Any, b: Any, c: Any, d: Any) -> Any:
    rows = _rows(a, b, c, d)
    return -_sign(_det([row + [Fraction(1)] for row in rows]))


def reference_incircle(a: Any, b: Any, c: Any, d: Any) -> Any:
    rows = _rows(a, b, c, d)
    return _sign(_det([row + [row[0] ** 2 + row[1] ** 2, Fraction(1)] for row in rows]))


def reference_insphere(a: Any, b: Any, c: Any, d: Any, e: Any) -> Any:
    rows = _rows(a, b, c, d, e)
    lifted = [
        row + [row[0] ** 2 + row[1] ** 2 + row[2] ** 2, Fraction(1)] for row in rows
    ]
    return -_sign(_det(lifted))


def _grid(offsets: Any, dimension: Any) -> Any:
    return np.asarray(list(product(offsets, repeat=dimension)), dtype=np.float64)


def _orient2d_case() -> Any:
    # Kettner et al.: points near the line y = x with 2^-53 perturbations.
    a = 0.5 + _grid(np.arange(24) * ULP, 2)
    b = np.broadcast_to(np.asarray([12.0, 12.0]), a.shape)
    c = np.broadcast_to(np.asarray([24.0, 24.0]), a.shape)
    return (a, b, c), reference_orient2d, orient2d


def _orient3d_case() -> Any:
    # Plane z = y through three points; the query grid straddles it.
    d = 0.5 + _grid(np.arange(8) * ULP, 3)
    a = np.broadcast_to(np.asarray([1.0, 12.0, 12.0]), d.shape)
    b = np.broadcast_to(np.asarray([24.0, 24.0, 24.0]), d.shape)
    c = np.broadcast_to(np.asarray([-7.0, 3.0, 3.0]), d.shape)
    return (a, b, c, d), reference_orient3d, orient3d


def _incircle_case() -> Any:
    # Unit circle through (1, 0), (0, 1), (-1, 0); queries around (0, -1).
    offsets = _grid(np.arange(-8, 8) * ULP, 2)
    d = np.asarray([0.0, -1.0]) + offsets
    a = np.broadcast_to(np.asarray([1.0, 0.0]), d.shape)
    b = np.broadcast_to(np.asarray([0.0, 1.0]), d.shape)
    c = np.broadcast_to(np.asarray([-1.0, 0.0]), d.shape)
    return (a, b, c, d), reference_incircle, incircle


def _insphere_case() -> Any:
    # Unit sphere through four axis points; queries around (0, 0, -1).
    offsets = _grid(np.arange(-3, 3) * ULP, 3)
    e = np.asarray([0.0, 0.0, -1.0]) + offsets
    corners = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0])
    tetra = tuple(np.broadcast_to(np.asarray(corner), e.shape) for corner in corners)
    return tetra + (e,), reference_insphere, insphere


CASES = {
    "orient2d": _orient2d_case,
    "orient3d": _orient3d_case,
    "incircle": _incircle_case,
    "insphere": _insphere_case,
}


def _reference(reference: Any, arrays: Any) -> Any:
    return np.asarray([reference(*row) for row in zip(*arrays, strict=True)])


def _with_generic_rows(arrays: Any) -> Any:
    # Generic random rows that a sound filter must resolve.
    rng = np.random.default_rng(11)
    generic = tuple(rng.standard_normal((64, array.shape[1])) for array in arrays)
    return tuple(
        np.concatenate((array, extra), axis=0)
        for array, extra in zip(arrays, generic, strict=True)
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_filters_never_certify_a_wrong_sign_on_near_degenerate_grids(name: Any) -> None:
    near_degenerate, reference, predicate = CASES[name]()
    arrays = _with_generic_rows(near_degenerate)
    expected = _reference(reference, arrays)
    generic = slice(near_degenerate[0].shape[0], None)

    host = predicate(*arrays, mode=PredicateMode.FILTERED)
    assert host.signs.dtype == np.int8
    np.testing.assert_array_equal(host.signs[host.certain], expected[host.certain])
    assert np.all(host.signs[~host.certain] == PredicateSign.UNCERTAIN)
    assert np.mean(host.certain[generic]) > 0.9

    device = jax.jit(
        lambda *values: predicate(*values, mode=PredicateMode.FILTERED_DEVICE)
    )(*(jnp.asarray(array) for array in arrays))
    signs = np.asarray(device.signs)
    certain = np.asarray(device.certain)
    np.testing.assert_array_equal(signs[certain], expected[certain])
    assert np.all(signs[~certain] == PredicateSign.UNCERTAIN)
    assert np.mean(certain[generic]) > 0.9


@requires_meshcore
@pytest.mark.meshcore
@pytest.mark.parametrize("name", sorted(CASES))
def test_exact_mode_matches_rational_reference(name: Any) -> None:
    arrays, reference, predicate = CASES[name]()
    expected = _reference(reference, arrays)
    result = predicate(*arrays, mode=PredicateMode.EXACT)
    assert bool(np.all(result.certain))
    np.testing.assert_array_equal(result.signs, expected)
    # The adversarial grids contain exact zeros and both strict signs.
    assert set(expected.tolist()) == {-1, 0, 1}


@requires_meshcore
@pytest.mark.meshcore
def test_exact_signs_are_antisymmetric_under_argument_exchange() -> None:
    rng = np.random.default_rng(3)
    # A small integer lattice makes many configurations exactly degenerate.
    points = rng.integers(-2, 3, size=(400, 5, 3)).astype(np.float64)
    a, b, c, d, e = (points[:, index] for index in range(5))
    exact = PredicateMode.EXACT
    o2 = orient2d(a[:, :2], b[:, :2], c[:, :2], mode=exact).signs
    np.testing.assert_array_equal(
        orient2d(b[:, :2], a[:, :2], c[:, :2], mode=exact).signs, -o2
    )
    np.testing.assert_array_equal(
        orient2d(b[:, :2], c[:, :2], a[:, :2], mode=exact).signs, o2
    )
    o3 = orient3d(a, b, c, d, mode=exact).signs
    np.testing.assert_array_equal(orient3d(b, a, c, d, mode=exact).signs, -o3)
    np.testing.assert_array_equal(orient3d(a, b, d, c, mode=exact).signs, -o3)
    ic = incircle(a[:, :2], b[:, :2], c[:, :2], d[:, :2], mode=exact).signs
    np.testing.assert_array_equal(
        incircle(b[:, :2], a[:, :2], c[:, :2], d[:, :2], mode=exact).signs, -ic
    )
    sp = insphere(a, b, c, d, e, mode=exact).signs
    np.testing.assert_array_equal(insphere(b, a, c, d, e, mode=exact).signs, -sp)
    assert np.any(o2 == 0) and np.any(o3 == 0)


def test_structural_zeros_are_certified_without_meshcore() -> None:
    collinear = orient2d([0.0, 0.0], [1.0, 0.0], [5.0, 0.0], mode=PredicateMode.FILTERED)
    coplanar = orient3d(
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 2.0],
        [7.0, -1.0, 2.0],
        mode=PredicateMode.FILTERED,
    )
    duplicate = incircle(
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], mode=PredicateMode.FILTERED
    )
    for result in (collinear, coplanar, duplicate):
        assert bool(result.certain)
        assert int(result.signs) == PredicateSign.ZERO


def test_device_filter_refuses_subnormal_and_nonfinite_inputs() -> None:
    tiny = np.finfo(np.float64).tiny / 4.0
    a = jnp.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    b = jnp.asarray([[tiny, 0.0], [1.0, 0.0], [jnp.inf, 0.0]])
    c = jnp.asarray([[0.0, tiny], [0.0, 1.0], [0.0, 1.0]])
    result = orient2d(a, b, c, mode=PredicateMode.FILTERED_DEVICE)
    np.testing.assert_array_equal(np.asarray(result.certain), [False, True, False])
    assert int(result.signs[1]) == PredicateSign.POSITIVE


def test_device_filter_matches_float32_orientation_conventions() -> None:
    a = jnp.zeros((3,), dtype=jnp.float32)
    x = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
    y = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    z = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)
    inside = jnp.asarray([0.2, 0.2, 0.2], dtype=jnp.float32)
    mode = PredicateMode.FILTERED_DEVICE
    assert int(orient3d(a, x, y, z, mode=mode).signs) == 1
    assert int(orient3d(a, y, x, z, mode=mode).signs) == -1
    assert int(insphere(a, x, y, z, inside, mode=mode).signs) == 1
    assert int(incircle(a[:2], x[:2], y[:2], inside[:2], mode=mode).signs) == 1


def test_missing_library_is_explicit(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", str(tmp_path / "missing.dylib"))
    assert not meshcore_available()
    assert resolve_host_predicate_mode(PredicateMode.EXACT) is PredicateMode.FILTERED
    with pytest.raises(MeshcoreUnavailableError):
        orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], mode=PredicateMode.EXACT)
    filtered = orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], mode=PredicateMode.FILTERED)
    assert int(filtered.signs) == PredicateSign.POSITIVE
    with pytest.raises(ValueError, match="FILTERED or EXACT"):
        resolve_host_predicate_mode(PredicateMode.FILTERED_DEVICE)


def test_geometry_precision_policy_predicate_mode() -> None:
    default = GeometryPrecisionPolicy()
    filtered = GeometryPrecisionPolicy(predicate_mode=PredicateMode.FILTERED)
    assert default.predicate_mode is PredicateMode.EXACT
    assert filtered.predicate_mode is PredicateMode.FILTERED
    assert default.policy_id != filtered.policy_id
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        GeometryPrecisionPolicy(predicate_mode="exact")
