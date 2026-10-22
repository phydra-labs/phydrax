from __future__ import annotations

from dataclasses import fields
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._model._protocols import StructuredDerivativeProvider
from phydrax.geometry.brep import _patches as owner
from phydrax.geometry.brep._placed import PlacedSurface


_FLOAT = np.finfo(np.float64)
_SCALES = (
    _FLOAT.smallest_subnormal, _FLOAT.tiny, 1e-300, 1e-100,
    1.0, 1e100, 1e300, _FLOAT.max,
)


def assert_same_bits(actual: Any, expected: Any) -> None:
    left, right = np.asarray(actual), np.asarray(expected)
    assert left.dtype == right.dtype == np.float64
    assert left.shape == right.shape
    # Do not normalize NaNs or zeros: their signs and payloads are evidence.
    np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))


def ordinary_differential(patch: owner.AbstractSurfacePatch, parameters: jax.Array) -> jax.Array:
    points = jnp.asarray(parameters, dtype=jnp.float64)
    return jax.vmap(jax.jacfwd(patch.evaluate))(points.reshape((-1, 2))).reshape(
        (*points.shape[:-1], 3, 2)
    )


def ordinary_value_and_first_jets(
    patch: owner.AbstractSurfacePatch, parameter: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    def value_with_aux(point: jax.Array) -> tuple[jax.Array, jax.Array]:
        value = patch.evaluate(point)
        return value, value

    differential, value = jax.jacfwd(value_with_aux, has_aux=True)(parameter)
    return value, differential


def assert_batched_jets(patch: owner.AbstractSurfacePatch, points: jax.Array) -> None:
    # Both programs use the same eager vmap, rather than comparing fused JIT
    # arithmetic with eager primitives (especially important for placed values).
    actual = jax.vmap(lambda uv: owner._surface_value_and_first_jets(patch, uv))(points)
    expected = jax.vmap(lambda uv: ordinary_value_and_first_jets(patch, uv))(points)
    for left, right in zip(actual, expected, strict=True):
        assert_same_bits(left, right)


@pytest.fixture(scope="module")
def adversarial_parameters() -> jax.Array:
    rng = np.random.default_rng(81731)
    axis = np.asarray((
        0.0, -0.0, _FLOAT.smallest_subnormal, -_FLOAT.smallest_subnormal,
        _FLOAT.tiny, -_FLOAT.tiny, 1e-150, -1e-150, 1e150, -1e150,
        np.pi, -np.pi, np.pi / 2, -np.pi / 2, _FLOAT.max, -_FLOAT.max,
        np.inf, -np.inf, np.nan,
    ))
    u, v = np.meshgrid(axis, axis, indexing="ij")
    return jnp.asarray(np.concatenate((
        np.column_stack((u.ravel(), v.ravel())),
        rng.normal(size=(1024, 2)),
        rng.uniform(-1e300, 1e300, size=(1024, 2)),
    )), dtype=jnp.float64)


@pytest.fixture(scope="module")
def frames() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(78131)
    return (
        np.eye(3), -np.eye(3), np.diag((1.0, -1.0, 1.0)), np.eye(3)[[2, 0, 1]],
        *(np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(8)),
    )


def make_patch(kind: str, frame: np.ndarray, scale: float) -> owner.AbstractSurfacePatch:
    origin = (0.0, -0.0, 1.0)
    if kind == "plane":
        return owner.PlanePatch(origin, frame[:, 0] * scale, frame[:, 1] * scale)
    return owner.CylinderPatch(origin, frame[:, 0], frame[:, 1], frame[:, 2], scale)


def place(patch: owner.AbstractSurfacePatch, depth: int) -> owner.AbstractSurfacePatch:
    rng = np.random.default_rng(93211)
    for _ in range(depth):
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        patch = PlacedSurface(patch, rotation, (0.1, -0.2, 0.3))
    return patch


@pytest.mark.parametrize("kind", ("plane", "cylinder"))
@pytest.mark.parametrize("depth", (0, 1, 2), ids=("direct", "placed", "nested"))
@pytest.mark.parametrize("scale", _SCALES)
def test_structured_derivatives_preserve_ad_bits_random_and_adversarial(
    kind: str, depth: int, scale: float,
    frames: tuple[np.ndarray, ...], adversarial_parameters: jax.Array,
) -> None:
    # 576 patch/frame/scale/placement cases, each with 2409 parameter pairs.
    for frame in frames:
        patch = place(make_patch(kind, frame, scale), depth)
        expected = ordinary_differential(patch, adversarial_parameters)
        structured = owner._structured_surface_differential(patch, adversarial_parameters)
        assert structured is not None
        assert_same_bits(structured, expected)
        assert_same_bits(owner.surface_differential(patch, adversarial_parameters), expected)
        assert_batched_jets(patch, adversarial_parameters)


@pytest.mark.parametrize("kind", ("plane", "cylinder"))
@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("shape", ((2,), (0, 2), (6, 2), (2, 3, 2), (1, 2, 3, 2)))
def test_structured_derivatives_arbitrary_parameter_leading_shape(
    kind: str, depth: int, shape: tuple[int, ...],
) -> None:
    patch = place(make_patch(kind, -np.eye(3), 1.3), depth)
    points = jnp.asarray(np.arange(np.prod(shape)).reshape(shape) / 7, dtype=jnp.float64)
    expected = ordinary_differential(patch, points)
    assert_same_bits(owner._structured_surface_differential(patch, points), expected)
    assert_same_bits(owner.surface_differential(patch, points), expected)
    if shape == (2,):
        actual = owner._surface_value_and_first_jets(patch, points)
        reference = ordinary_value_and_first_jets(patch, points)
        for left, right in zip(actual, reference, strict=True):
            assert_same_bits(left, right)
    else:
        assert_batched_jets(patch, points.reshape((-1, 2)))


@eqx.filter_jit
def compiled_ordinary_value_and_differential(
    patch: owner.AbstractSurfacePatch, parameters: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    values, differentials = jax.vmap(
        lambda point: ordinary_value_and_first_jets(patch, point)
    )(parameters.reshape((-1, 2)))
    return (
        values.reshape((*parameters.shape[:-1], 3)),
        differentials.reshape((*parameters.shape[:-1], 3, 2)),
    )


@pytest.mark.parametrize("kind", ("plane", "cylinder"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_existing_compiled_value_boundary_matches_compiled_ordinary_ad(
    kind: str, depth: int, adversarial_parameters: jax.Array,
) -> None:
    patch = place(make_patch(kind, -np.eye(3), 1.3), depth)
    actual = owner.surface_value_and_differential(patch, adversarial_parameters)
    expected = compiled_ordinary_value_and_differential(patch, adversarial_parameters)
    for left, right in zip(actual, expected, strict=True):
        assert_same_bits(left, right)


@pytest.mark.parametrize("kind", ("plane", "cylinder"))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.int32))
def test_canonical_differential_coerces_parameters_to_float64(kind: str, dtype: Any) -> None:
    patch = place(make_patch(kind, np.eye(3), 1.3), 2)
    points = jnp.asarray(((0, 1), (2, 3)), dtype=dtype)
    assert_same_bits(owner.surface_differential(patch, points), ordinary_differential(patch, points))
    if dtype == jnp.float32:
        assert_batched_jets(patch, points)
        actual = owner.surface_value_and_differential(patch, points)
        expected = compiled_ordinary_value_and_differential(patch, points)
        for left, right in zip(actual, expected, strict=True):
            assert_same_bits(left, right)


@pytest.mark.parametrize("kind", ("plane", "cylinder"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_every_numerical_leaf_remains_dynamic_and_preserves_ad_bits(
    kind: str, depth: int,
) -> None:
    patch = place(make_patch(kind, np.eye(3), 1.3), depth)
    points = jnp.asarray(((0.0, -0.0), (0.2, -0.7), (-1.1, 2.3)), dtype=jnp.float64)
    leaves, structure = jax.tree_util.tree_flatten(patch)
    assert len(leaves) == (3 if kind == "plane" else 5) + 2 * depth
    assert all(isinstance(leaf, jax.Array) and leaf.dtype == jnp.float64 for leaf in leaves)
    current = patch
    for _ in range(depth + 1):
        assert all(not field.metadata.get("static", False) for field in fields(current))
        if isinstance(current, PlacedSurface):
            current = current.definition

    # Dynamic replacement deliberately bypasses constructor validation. Existing
    # array leaves are not promises of finite/orthogonal numerical contents.
    replacements = (
        0.0, -0.0, _FLOAT.smallest_subnormal, -_FLOAT.smallest_subnormal,
        _FLOAT.tiny, -_FLOAT.tiny, _FLOAT.max, -_FLOAT.max,
        np.inf, -np.inf, np.nan,
    )
    for index, leaf in enumerate(leaves):
        for value in replacements:
            changed = eqx.tree_at(
                lambda source: jax.tree_util.tree_leaves(source)[index],
                patch, jnp.full_like(leaf, value),
            )
            assert jax.tree_util.tree_structure(changed) == structure
            expected = ordinary_differential(changed, points)
            assert_same_bits(owner._structured_surface_differential(changed, points), expected)
            assert_same_bits(owner.surface_differential(changed, points), expected)
            assert_batched_jets(changed, points)


@pytest.mark.parametrize("kind", ("plane", "cylinder"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_finite_jvp_and_dynamic_pose_values(kind: str, depth: int) -> None:
    patch = place(make_patch(kind, np.eye(3), 1.3), depth)
    rng = np.random.default_rng(1917)
    points = jnp.asarray(rng.normal(size=(32, 2)))
    directions = jnp.asarray(rng.normal(size=(32, 2)))
    tangent = jax.vmap(lambda uv, duv: jax.jvp(patch.evaluate, (uv,), (duv,))[1])(
        points, directions,
    )
    differential = owner.surface_differential(patch, points)
    contraction = jnp.einsum("nij,nj->ni", differential, directions)
    assert np.all(np.isfinite(tangent))
    np.testing.assert_allclose(contraction, tangent, rtol=3e-14, atol=3e-14)
    if isinstance(patch, PlacedSurface):
        translated = eqx.tree_at(lambda source: source.translation, patch, patch.translation + 0.5)
        rotated = eqx.tree_at(lambda source: source.rotation, patch, -patch.rotation)
        assert_same_bits(owner.surface_differential(translated, points), differential)
        assert not np.array_equal(translated.evaluate(points), patch.evaluate(points))
        assert not np.array_equal(owner.surface_differential(rotated, points), differential)
        assert_batched_jets(translated, points)
        assert_batched_jets(rotated, points)


def unsupported_patch(kind: str) -> owner.AbstractSurfacePatch:
    frame = np.eye(3)
    arguments = ((0.1, -0.2, 0.3), frame[:, 0], frame[:, 1], frame[:, 2])
    if kind == "sphere":
        return owner.SpherePatch(*arguments, 1.3)
    if kind == "torus":
        return owner.TorusPatch(*arguments, 1.3, 0.4)
    if kind == "cone":
        return owner.ConePatch(*arguments, 1.3, 0.2)
    if kind == "bspline":
        return owner.BSplineSurfacePatch(
            (((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 1))),
            np.ones((2, 2), dtype=np.float64),
            (0, 0, 1, 1), (0, 0, 1, 1), 1, 1,
        )
    return owner.OffsetSurface(owner.PlanePatch(*arguments[:3]), 0.25)


@pytest.mark.parametrize("kind", ("sphere", "torus", "cone", "bspline", "offset"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_other_sources_and_their_placements_retain_ordinary_ad(
    kind: str, depth: int, adversarial_parameters: jax.Array,
) -> None:
    patch = place(unsupported_patch(kind), depth)
    if kind == "bspline":
        with pytest.raises(eqx.EquinoxRuntimeError, match="queries must be finite"):
            owner.surface_differential(patch, adversarial_parameters)
        with pytest.raises(eqx.EquinoxRuntimeError, match="queries must be finite"):
            ordinary_differential(patch, adversarial_parameters)
        adversarial_parameters = adversarial_parameters[
            np.all(np.isfinite(np.asarray(adversarial_parameters)), axis=1)
        ]
    assert owner._structured_surface_differential(patch, adversarial_parameters) is None
    assert_same_bits(
        owner.surface_differential(patch, adversarial_parameters),
        ordinary_differential(patch, adversarial_parameters),
    )
    assert_batched_jets(patch, adversarial_parameters)
    result, reason = patch.try_structured_partial(
        deps=("uv",), var="uv", axis=0, order=1,
        args=(adversarial_parameters,), key=None, kwargs={},
    )
    assert result is None and reason is not None
    assert patch.handle_structured_derivative_fallback(reason) is None


@pytest.mark.parametrize("kind", ("plane", "cylinder"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_existing_structured_provider_partial_and_fallback_contract(kind: str, depth: int) -> None:
    patch = place(make_patch(kind, -np.eye(3), 1.3), depth)
    assert isinstance(patch, StructuredDerivativeProvider)
    points = jnp.asarray(((0.0, -0.0), (0.2, -0.7)), dtype=jnp.float64)
    request: dict[str, Any] = dict(
        deps=("uv",), var="uv", axis=0, order=1, args=(points,), key=None, kwargs={},
    )
    expected = ordinary_differential(patch, points)
    for axis in (0, 1):
        request["axis"] = axis
        actual, reason = patch.try_structured_partial(**request)
        assert reason is None
        assert_same_bits(actual, expected[..., axis])
    for unsupported in (
        {"deps": ()}, {"deps": ("uv", "time")}, {"var": "other"},
        {"axis": -1}, {"axis": 2}, {"order": 0}, {"order": 2},
        {"args": ()}, {"args": (points, points)}, {"kwargs": {"extra": 1}},
    ):
        changed = request.copy()
        changed.update(unsupported)
        actual, reason = patch.try_structured_partial(**changed)
        assert actual is None and reason is not None
        assert patch.handle_structured_derivative_fallback(reason) is None
