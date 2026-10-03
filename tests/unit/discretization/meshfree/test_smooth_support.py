# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Smooth fixed-radius GMLS on a bounded candidate envelope and coordinate sensitivities.

Oracles are independent of the stencil kernel: GMLS weights from the closed
form ``w * P (P^T W P)^{-1} b`` with NumPy normal equations and closed-form
compact profiles, exact derivatives of quadratics, a nonpolynomial field,
one-sided and central coordinate differences, and JVP/VJP duality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import (
    FieldQueryEvidence,
    point_cloud_reconstruction_sensitivity,
    PointCloudPlan,
)
from phydrax.discretization.meshfree import (
    LocalStencilEvidence,
    LocalStencilPolicy,
    LocalStencilRefreshStatus,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    prepare_local_stencils,
    PreparedLocalStencils,
    refresh_local_stencils,
    SmoothSupportEnvelope,
)
from phydrax.discretization.meshfree._types import StencilWeightKernel


RADIUS = 1.0
# Fixed sources strictly inside the support, asymmetric so every row is generic.
FIXED = np.asarray(
    [
        [0.3, 0.1],
        [-0.4, 0.2],
        [0.1, -0.5],
        [-0.2, -0.3],
        [0.5, 0.45],
        [-0.6, -0.1],
        [0.05, 0.6],
        [0.65, -0.3],
        [-0.35, 0.55],
    ]
)
TARGET = np.zeros((1, 2))
# The moving source crosses the cutoff obliquely at t = 0 (|anchor| = R exactly).
ANCHOR = np.asarray([RADIUS, 0.0])
VELOCITY = np.asarray([-0.8, 0.6])


def _sources(t: float | Array) -> Array:
    return jnp.concatenate(
        (jnp.asarray(FIXED), (jnp.asarray(ANCHOR) + t * jnp.asarray(VELOCITY))[None])
    )


def _profile(ratio: np.ndarray, kernel: StencilWeightKernel, order: int) -> np.ndarray:
    inside = ratio < 1.0
    r = np.where(inside, ratio, 1.0)
    match kernel:
        case "wendland-c2":
            weight = (1.0 - r) ** 4 * (4.0 * r + 1.0)
        case "wendland-c4":
            weight = (1.0 - r) ** 6 * (35.0 * r**2 + 18.0 * r + 3.0)
        case "compact-polynomial":
            weight = (1.0 - r**2) ** (order + 1)
        case _:
            raise AssertionError(kernel)
    return np.where(inside, weight, 0.0)


def _oracle(sources: np.ndarray, kernel: StencilWeightKernel, order: int) -> np.ndarray:
    """d/dx GMLS weights at the origin by quadratic normal equations."""
    x, y = sources[:, 0] / RADIUS, sources[:, 1] / RADIUS
    weight = _profile(np.hypot(x, y), kernel, order)
    basis = np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=1)
    moments = np.asarray([0.0, 1.0 / RADIUS, 0.0, 0.0, 0.0, 0.0])
    coefficients = np.linalg.solve(basis.T @ (weight[:, None] * basis), moments)
    return weight * (basis @ coefficients)


def _by_source(stencils: PreparedLocalStencils, row: Array) -> Array:
    """Target-row weights scattered from relation order to source order."""
    relation = stencils.neighborhood.relation
    return (
        jnp.zeros((relation.source_size,), dtype=row.dtype)
        .at[relation.source_indices[0]]
        .add(jnp.where(relation.valid[0], row, 0.0))
    )


def _field(points: Array) -> Array:
    return jnp.exp(0.7 * points[:, 0]) * jnp.cos(points[:, 1] + 0.3 * points[:, 0])


def _prepared(
    kernel: StencilWeightKernel, order: int, displacement: float
) -> PreparedLocalStencils:
    support = SmoothSupportEnvelope(RADIUS, displacement)
    anchor = np.asarray(_sources(0.0))
    neighborhood = MeshfreeNeighborhoodPlan(
        anchor, anchor.shape[0], targets=TARGET, envelope=support
    ).prepare()
    return prepare_local_stencils(
        neighborhood,
        anchor,
        TARGET,
        (MeshfreeFunctional(((1, 0),), (1.0,)),),
        LocalStencilPolicy(weight_kernel=kernel, support=support, coordinate_order=order),
    )


_KERNELS = [
    ("wendland-c2", 1),
    ("wendland-c4", 2),
    ("compact-polynomial", 1),
]


@pytest.mark.parametrize(("kernel", "order"), _KERNELS, ids=lambda item: str(item))
def test_neighbor_crossing_the_cutoff_has_matching_two_sided_derivatives(
    kernel: StencilWeightKernel, order: int
) -> None:
    stencils = _prepared(kernel, order, displacement=0.2)

    def weights(t: Array) -> Array:
        refreshed = refresh_local_stencils(stencils, _sources(t), TARGET)
        return _by_source(stencils, refreshed.stencils.weights[0][0])

    def oracle(t: float) -> np.ndarray:
        return _oracle(np.asarray(_sources(t)), kernel, order)

    step = 1e-5
    for t in (-0.15, -0.01, -1e-3, 1e-3, 0.01, 0.15):
        refreshed = refresh_local_stencils(stencils, _sources(t), TARGET)
        assert bool(refreshed.accepted)
        np.testing.assert_array_equal(refreshed.stencils.evidence.rank, 6)
        value, tangent = jax.jvp(weights, (jnp.asarray(t),), (jnp.asarray(1.0),))
        np.testing.assert_allclose(value, oracle(t), rtol=0.0, atol=1e-11)
        central = (oracle(t + step) - oracle(t - step)) / (2.0 * step)
        np.testing.assert_allclose(tangent, central, rtol=0.0, atol=1e-7)
    # At the cutoff the moving source carries zero weight and the one-sided
    # second-order differences from inside and outside agree with the AD
    # derivative of the published map.
    h = 1e-4
    left = (3.0 * oracle(0.0) - 4.0 * oracle(-h) + oracle(-2.0 * h)) / (2.0 * h)
    right = (-3.0 * oracle(0.0) + 4.0 * oracle(h) - oracle(2.0 * h)) / (2.0 * h)
    value, tangent = jax.jvp(weights, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    assert float(value[-1]) == 0.0
    np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(tangent, left, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(tangent, right, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(("kernel", "order"), _KERNELS, ids=lambda item: str(item))
def test_crossing_stencil_differentiates_fields_along_the_trajectory(
    kernel: StencilWeightKernel, order: int
) -> None:
    stencils = _prepared(kernel, order, displacement=0.2)

    def derivative(t: Array) -> Array:
        sources = _sources(t)
        refreshed = refresh_local_stencils(stencils, sources, TARGET)
        return _by_source(stencils, refreshed.stencils.weights[0][0]) @ _field(sources)

    def quadratic(t: Array) -> Array:
        sources = _sources(t)
        x, y = sources[:, 0], sources[:, 1]
        values = 1.0 - 2.0 * x + 0.5 * y + 3.0 * x * x - x * y + 0.25 * y * y
        refreshed = refresh_local_stencils(stencils, sources, TARGET)
        return _by_source(stencils, refreshed.stencils.weights[0][0]) @ values

    def oracle(s: float) -> float:
        sources = np.asarray(_sources(s))
        return float(
            _oracle(sources, kernel, order) @ np.asarray(_field(jnp.asarray(sources)))
        )

    for t in (-0.12, -2e-3, 0.0, 2e-3, 0.12):
        # A quadratic is differentiated exactly on both sides of the crossing.
        np.testing.assert_allclose(quadratic(jnp.asarray(t)), -2.0, atol=1e-11)
    for t in (-0.12, -2e-3, 2e-3, 0.12):
        _, tangent = jax.jvp(derivative, (jnp.asarray(t),), (jnp.asarray(1.0),))
        central = (oracle(t + 1e-5) - oracle(t - 1e-5)) / 2e-5
        np.testing.assert_allclose(tangent, central, rtol=1e-6, atol=1e-8)
    # A C^1 kernel has a second-derivative jump at the cutoff, so only the
    # one-sided differences (each second order on its own side) are compared.
    h = 1e-4
    left = (3.0 * oracle(0.0) - 4.0 * oracle(-h) + oracle(-2.0 * h)) / (2.0 * h)
    right = (-3.0 * oracle(0.0) + 4.0 * oracle(h) - oracle(2.0 * h)) / (2.0 * h)
    _, tangent = jax.jvp(derivative, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(tangent, left, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(tangent, right, rtol=0.0, atol=1e-7)


def test_insufficient_envelope_refuses_with_nan_sensitivities() -> None:
    # A source just outside the 0.05 envelope's candidate radius would enter
    # the support after moving 0.2; the motion exceeds the envelope and the
    # refresh is refused rather than silently ignoring the entering neighbor.
    support = SmoothSupportEnvelope(RADIUS, 0.05)
    outsider = np.asarray([[0.0, -1.15]])
    sources = np.concatenate((FIXED, outsider))
    neighborhood = MeshfreeNeighborhoodPlan(
        sources, sources.shape[0], targets=TARGET, envelope=support
    ).prepare()
    assert not bool(np.asarray(neighborhood.relation.valid)[0, -1])
    stencils = prepare_local_stencils(
        neighborhood,
        sources,
        TARGET,
        (MeshfreeFunctional(((1, 0),), (1.0,)),),
        LocalStencilPolicy(support=support),
    )
    direction = np.zeros_like(sources)
    direction[-1] = (0.0, 1.0)

    def weights(t: Array) -> Array:
        moved = jnp.asarray(sources) + t * jnp.asarray(direction)
        return refresh_local_stencils(stencils, moved, TARGET).stencils.weights[0]

    moved = sources + 0.2 * direction
    refreshed = refresh_local_stencils(stencils, moved, TARGET)
    assert int(refreshed.status) == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED)
    value, tangent = jax.jvp(weights, (jnp.asarray(0.2),), (jnp.asarray(1.0),))
    assert np.all(np.isnan(value)) and np.all(np.isnan(tangent))
    _, pullback = jax.vjp(weights, jnp.asarray(0.2))
    assert np.isnan(pullback(jnp.ones_like(value))[0])
    # Inside the envelope the same refresh is admitted with finite tangents.
    value, tangent = jax.jvp(weights, (jnp.asarray(0.04),), (jnp.asarray(1.0),))
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(tangent))


def test_envelope_capacity_overflow_and_inconsistent_declarations_are_refused() -> None:
    support = SmoothSupportEnvelope(RADIUS, 0.2)
    anchor = np.asarray(_sources(0.0))
    with pytest.raises(ValueError, match="more than 5 candidates"):
        MeshfreeNeighborhoodPlan(anchor, 5, targets=TARGET, envelope=support).prepare()
    neighborhood = MeshfreeNeighborhoodPlan(
        anchor, anchor.shape[0], targets=TARGET, envelope=support
    ).prepare()
    functional = (MeshfreeFunctional(((1, 0),), (1.0,)),)
    with pytest.raises(ValueError, match="candidate-envelope"):
        prepare_local_stencils(
            neighborhood, anchor, TARGET, functional, LocalStencilPolicy()
        )
    nearest = MeshfreeNeighborhoodPlan(anchor, 8, targets=TARGET).prepare()
    with pytest.raises(ValueError, match="candidate-envelope"):
        prepare_local_stencils(
            nearest, anchor, TARGET, functional, LocalStencilPolicy(support=support)
        )
    with pytest.raises(ValueError, match="no smooth fixed-radius support"):
        LocalStencilPolicy(approximation="phs-rbf-fd", support=support)
    with pytest.raises(ValueError, match="not compactly supported"):
        LocalStencilPolicy(weight_kernel="inverse-square", support=support)
    with pytest.raises(ValueError, match="through coordinate order 3"):
        LocalStencilPolicy(
            weight_kernel="wendland-c2", support=support, coordinate_order=4
        )
    with pytest.raises(ValueError, match="needs a smooth support"):
        LocalStencilPolicy(weight_kernel="wendland-c4")


def _jittered(per_axis: int, seed: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, per_axis)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    jitter = np.random.default_rng(seed).uniform(-0.15, 0.15, grid.shape)
    return grid + jitter / (per_axis - 1)


def test_smooth_point_cloud_coordinate_sensitivity_matches_differences() -> None:
    points = _jittered(7, seed=3)
    spacing = 1.0 / 6.0
    support = SmoothSupportEnvelope(4.0 * spacing, 0.1 * spacing)
    cloud = PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        stencil=LocalStencilPolicy(support=support),
        neighbors=points.shape[0],
    ).prepare()
    direction = np.random.default_rng(5).normal(size=points.shape)
    direction *= 0.5 * support.displacement / np.max(np.linalg.norm(direction, axis=1))
    sensitivity = cloud.coordinate_sensitivity(points + 0.5 * direction)
    assert bool(sensitivity.accepted)
    assert isinstance(sensitivity.evidence, LocalStencilEvidence)
    np.testing.assert_array_equal(sensitivity.evidence.rank, 6)
    assert np.all(np.isfinite(np.asarray(sensitivity.evidence.condition)))
    tangent = sensitivity.linearization.jvp(jnp.asarray(direction))

    def weights(scale: float) -> tuple[Array, ...]:
        refreshed = cloud.refresh(points + scale * direction)
        return tuple(item for _, item in refreshed.discretization.mixed_weights)

    step = 1e-3
    for derivative, ahead, behind in zip(
        tangent, weights(0.5 + step), weights(0.5 - step), strict=True
    ):
        scale = float(jnp.max(jnp.abs(derivative)))
        np.testing.assert_allclose(
            derivative, (ahead - behind) / (2.0 * step), atol=1e-6 * scale
        )
    rng = np.random.default_rng(6)
    cotangent = tuple(jnp.asarray(rng.normal(size=item.shape)) for item in tangent)
    pulled = sensitivity.linearization.vjp(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(pulled, jnp.asarray(direction)),
        sum(jnp.vdot(c, t) for c, t in zip(cotangent, tangent, strict=True)),
        rtol=1e-9,
    )
    beyond = cloud.coordinate_sensitivity(points + 3.0 * direction)
    assert int(beyond.status) == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED)
    assert all(
        np.all(np.isnan(item))
        for item in beyond.linearization.jvp(jnp.asarray(direction))
    )


def test_point_cloud_reconstruction_coordinate_sensitivity() -> None:
    points = _jittered(9, seed=7)
    spacing = 1.0 / 8.0
    count = points.shape[0]
    cloud = PointCloudPlan(points, np.full(count, 1.0 / count)).prepare()
    support = phx.geometry.Rectangle((0.5, 0.5), (1.4, 1.4)).compile()
    envelope = 0.1 * spacing
    reconstruction = phx.discretization.prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=support,
        radius=2.5 * spacing,
        capacity=count,
        coordinate_envelope=envelope,
    )
    coefficients = _field(jnp.asarray(points))
    queries = np.asarray(((0.31, 0.42), (0.55, 0.61), (0.72, 0.28)))
    direction = np.random.default_rng(8).normal(size=points.shape)
    direction *= 0.4 * envelope / np.max(np.linalg.norm(direction, axis=1))
    sensitivity = point_cloud_reconstruction_sensitivity(
        reconstruction, coefficients, queries, points
    )
    assert bool(sensitivity.accepted)
    assert isinstance(sensitivity.evidence, FieldQueryEvidence)
    assert np.all(np.asarray(sensitivity.evidence.valid))
    tangent = sensitivity.linearization.jvp(jnp.asarray(direction))

    def values(scale: float) -> Array:
        moved = point_cloud_reconstruction_sensitivity(
            reconstruction, coefficients, queries, points + scale * direction
        )
        return moved.linearization.primal

    step = 1e-3
    central = (values(step) - values(-step)) / (2.0 * step)
    np.testing.assert_allclose(tangent, central, rtol=1e-6, atol=1e-9)
    cotangent = jnp.asarray(np.random.default_rng(9).normal(size=tangent.shape))
    np.testing.assert_allclose(
        jnp.vdot(sensitivity.linearization.vjp(cotangent), jnp.asarray(direction)),
        jnp.vdot(cotangent, tangent),
        rtol=1e-9,
    )
    beyond = point_cloud_reconstruction_sensitivity(
        reconstruction, coefficients, queries, points + 3.0 * direction
    )
    assert int(beyond.status) == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED)
    assert np.all(np.isnan(beyond.linearization.primal))
    assert np.all(np.isnan(beyond.linearization.jvp(jnp.asarray(direction))))
