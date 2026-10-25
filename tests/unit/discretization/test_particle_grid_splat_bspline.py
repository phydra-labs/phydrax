#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial, floor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.splatting._types import SplatBoundaryPolicy


_POSITIONS_1D = (0.0371, 0.1376, 0.2913, 0.4402, 0.5689, 0.7231, 0.8817, 0.9604)


def _cardinal(degree: int, t: Fraction) -> Fraction:
    """Centered cardinal B-spline from the exact truncated-power formula."""
    if degree == 0:
        return Fraction(1) if -Fraction(1, 2) <= t < Fraction(1, 2) else Fraction(0)
    half = Fraction(degree + 1, 2)
    total = sum(
        (-1) ** term
        * comb(degree + 1, term)
        * max(t + half - term, Fraction(0)) ** degree
        for term in range(degree + 2)
    )
    return total / factorial(degree)


def _cardinal_derivative(degree: int, t: Fraction) -> Fraction:
    half = Fraction(1, 2)
    return _cardinal(degree - 1, t + half) - _cardinal(degree - 1, t - half)


def _wrapped(t: Fraction, period: int) -> Fraction:
    """Nearest periodic image of a normalized offset."""
    half = Fraction(period, 2)
    return t + half - period * floor((t + half) / period) - half


def _bounded_grid(points: int, dimension: int = 1) -> Any:
    axes = tuple(
        phx.discretization.UniformAxisSpec(points, periodic=False, endpoint=True)
        for _ in range(dimension)
    )
    return phx.discretization.TensorGridPlan(
        axes, axis_names=tuple("xyz"[:dimension])
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))


def _dense_routes(prepared: Any, state: Any, count: int) -> tuple[Any, Any]:
    """Per-source dense weights and explicit x-gradients through channel payloads."""
    identity = jnp.eye(count)
    weights = prepared.deposit_content(state, identity).content
    gradients = prepared.scatter_route_payload(
        state, state.weight_gradients[..., 0][..., None] * identity[:, None, :]
    ).values
    return np.asarray(weights), np.asarray(gradients)


def _particles(count: int, dimension: int) -> Any:
    return phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()


def _periodic_grid(
    dimension: int, points: int = 16, *, cell_primary: bool = False
) -> Any:
    spec = (
        phx.discretization.UniformCellAxisSpec
        if cell_primary
        else phx.discretization.UniformAxisSpec
    )
    axes = tuple(
        spec(points, periodic=True, **({} if cell_primary else {"endpoint": False}))
        for _ in range(dimension)
    )
    return phx.discretization.TensorGridPlan(
        axes, axis_names=tuple("xyz"[:dimension])
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))


def test_particle_grid_splat_bspline_scenario_1() -> None:
    for degree in [1, 2, 3, 4, 5]:
        for dimension in [1, 2, 3]:
            grid = _periodic_grid(dimension)
            particles = _particles(4, dimension)
            base = jnp.asarray(
                [
                    [0.13, 0.27, 0.41],
                    [0.62, 0.18, 0.84],
                    [0.35, 0.79, 0.22],
                    [0.91, 0.68, 0.57],
                ]
            )[:, :dimension]
            assignment = phx.discretization.TensorBSplineSplatAssignment(degree)
            prepared = phx.discretization.ParticleGridSplatPlan(
                grid, assignment=assignment
            ).prepare(particles)
            state = prepared.build(base)
            result = prepared.deposit_content(state, jnp.asarray([1.0, 2.0, 3.0, 4.0]))

            assert state.stencil.indices.shape == (4, (degree + 1) ** dimension)
            assert jnp.allclose(state.partition_sums, 1.0, atol=1e-12)
            assert (
                jnp.min(jnp.where(state.stencil.valid, state.stencil.weights, jnp.inf))
                >= 0.0
            )
            assert jnp.max(jnp.abs(state.first_moments)) < 1e-12
            assert jnp.max(jnp.abs(state.gradient_sums)) < 1e-11
            assert jnp.all(jnp.diagonal(state.second_moments, axis1=-2, axis2=-1) > 0.0)
            assert jnp.allclose(jnp.sum(result.content), 10.0)
            assert result.balance.closed_domain_conservation_valid
    grid = _periodic_grid(2)
    particles = _particles(3, 2)
    position = jnp.asarray([[0.12, 0.27], [0.61, 0.83], [0.91, 0.06]])
    content = jnp.asarray([[1.0, 2.0], [-3.0, 0.5], [2.5, -1.0]])
    multilinear = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.MultilinearSplatAssignment()
    ).prepare(particles)
    bspline = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(1)
    ).prepare(particles)

    reference = multilinear.deposit_content(multilinear.build(position), content)
    result = bspline.deposit_content(bspline.build(position), content)

    assert jnp.allclose(result.content, reference.content)
    assert jnp.allclose(result.density, reference.density)
    assert jnp.allclose(result.balance.balance_defect, reference.balance.balance_defect)
    grid = _periodic_grid(2, points=8, cell_primary=True)
    particles = _particles(2, 2)
    position = jnp.asarray([[0.11, 0.23], [0.71, 0.82]])
    content = jnp.asarray([2.0, 3.0])
    layouts = (grid.vertices(), grid.cells(), grid.faces("x"), grid.faces("y"))

    for layout in layouts:
        location = grid.location(layout.offsets)
        prepared = phx.discretization.ParticleGridSplatPlan(
            grid,
            location=location,
            assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        ).prepare(particles)
        result = prepared.deposit_content(prepared.build(position), content)
        assert result.content.shape == layout.shape
        assert jnp.allclose(jnp.sum(result.content), jnp.sum(content))
        assert result.balance.closed_domain_conservation_valid


def test_particle_grid_splat_bspline_scenario_2() -> None:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = _particles(1, 1)
    position = jnp.asarray([[0.01]])
    content = jnp.asarray([4.0])
    assignment = phx.discretization.TensorBSplineSplatAssignment(2)
    rejecting = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=assignment, boundary="reject"
    ).prepare(particles)
    rejected_state = rejecting.build(position)
    assert rejected_state.truncated_support_mask[0]
    assert not rejected_state.successful

    dropping = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=assignment, boundary="drop"
    ).prepare(particles)
    state = dropping.build(position)
    result = dropping.deposit_content(state, content)
    expected_supported = content[0] * state.captured_fractions[0]

    assert state.successful
    assert 0.0 < state.captured_fractions[0] < 1.0
    assert jnp.allclose(jnp.sum(result.content), expected_supported)
    assert jnp.allclose(
        result.balance.dropped_source_total,
        content[0] * (1.0 - state.captured_fractions[0]),
    )
    assert result.balance.maximum_absolute_balance_defect < 1e-12
    assert not result.balance.closed_domain_conservation_valid
    grid = _periodic_grid(1)
    particles = _particles(1, 1)
    prepared = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(3)
    ).prepare(particles)
    position = jnp.asarray([[0.37]])
    state = prepared.build(position)

    jacobian = jax.jacfwd(lambda value: prepared.build(value).stencil.weights)(position)

    assert jnp.allclose(state.weight_gradients[..., 0], jacobian[:, :, 0, 0], atol=1e-12)
    assert jnp.max(jnp.abs(state.gradient_sums)) < 1e-12
    axis = phx.discretization.AxisDiscretization(
        nodes=jnp.asarray([0.0, 0.1, 0.4, 1.0]),
        quad_weights=None,
        basis="uniform",
        domain=phx.discretization.AxisDomain.interval(0.0, 1.0),
        lower_endpoint_included=True,
        upper_endpoint_included=True,
    )
    grid = phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))
    assignment = phx.discretization.TensorBSplineSplatAssignment(2)
    with pytest.raises(ValueError, match="uniformly spaced"):
        phx.discretization.ParticleGridSplatPlan(grid, assignment=assignment).prepare(
            _particles(1, 1)
        )

    periodic = _periodic_grid(3)
    tiny = phx.discretization.ParticleGridSplatBudget(maximum_routes=10)
    with pytest.raises(ValueError, match="routes"):
        phx.discretization.ParticleGridSplatPlan(
            periodic,
            assignment=phx.discretization.TensorBSplineSplatAssignment(3),
            budget=tiny,
        ).prepare(_particles(1, 3))


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5], ids=lambda p: f"order-{p + 1}")
@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "bounded"])
def test_cardinal_routes_match_exact_truncated_power_basis(
    degree: int, periodic: bool
) -> None:
    count = len(_POSITIONS_1D)
    if periodic:
        points, spacing = 12, Fraction(1, 12)
        grid = _periodic_grid(1, points)
        position = _POSITIONS_1D
    else:
        # Interior sources whose whole support lies inside a bounded axis.
        points, spacing = 25, Fraction(1, 24)
        grid = _bounded_grid(points)
        position = tuple(0.15 + 0.7 * value for value in _POSITIONS_1D)
    prepared = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(degree)
    ).prepare(_particles(count, 1))
    state = prepared.build(jnp.asarray(position)[:, None])
    weights, gradients = _dense_routes(prepared, state, count)

    expected = np.zeros((points, count))
    expected_gradient = np.zeros((points, count))
    for source, value in enumerate(position):
        for node in range(points):
            t = (Fraction(value) - node * spacing) / spacing
            if periodic:
                t = _wrapped(t, points)
            expected[node, source] = float(_cardinal(degree, t))
            expected_gradient[node, source] = float(
                _cardinal_derivative(degree, t) / spacing
            )

    assert bool(state.successful)
    assert state.stencil.indices.shape == (count, degree + 1)
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(gradients, expected_gradient, rtol=0.0, atol=1e-10)
    assert np.all(weights >= 0.0)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5], ids=lambda p: f"order-{p + 1}")
def test_cardinal_route_moments_are_position_independent_through_degree(
    degree: int,
) -> None:
    """Moments k <= p equal the centered Irwin-Hall moments; k = p + 1 does not."""
    count = len(_POSITIONS_1D)
    points = 16
    prepared = phx.discretization.ParticleGridSplatPlan(
        _periodic_grid(1, points),
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
    ).prepare(_particles(count, 1))
    state = prepared.build(jnp.asarray(_POSITIONS_1D)[:, None])
    weights = np.where(state.stencil.valid, state.stencil.weights, 0.0)
    offsets = np.asarray(state.route_offsets[..., 0]) * points
    moments = np.stack(
        [np.sum(weights * offsets**order, axis=1) for order in range(degree + 2)]
    )
    n = degree + 1
    # Centered moments of a sum of n uniform(-1/2, 1/2) variables (cumulants
    # n/12 and -n/120): independent of the spline evaluation under test.
    irwin_hall = (1.0, 0.0, n / 12.0, 0.0, n * n / 48.0 - n / 120.0, 0.0)

    for order in range(degree + 1):
        np.testing.assert_allclose(
            moments[order], irwin_hall[order], rtol=0.0, atol=1e-12, err_msg=str(order)
        )
    assert np.ptp(moments[degree + 1]) > 1e-4
    np.testing.assert_allclose(state.first_moments, 0.0, atol=1e-13)
    # Exact second moment from the truncated-power basis: the Irwin-Hall constant
    # n/12 for p >= 2, but the phase-dependent t(1 - t) of degree-one CIC.
    exact_second = [
        float(
            sum(
                _cardinal(degree, t) * t * t
                for t in (
                    _wrapped(Fraction(value) * points - node, points)
                    for node in range(points)
                )
            )
        )
        / points**2
        for value in _POSITIONS_1D
    ]
    np.testing.assert_allclose(state.second_moments[:, 0, 0], exact_second, rtol=1e-12)
    if degree >= 2:
        np.testing.assert_allclose(exact_second, n / 12.0 / points**2, rtol=1e-12)
    np.testing.assert_allclose(state.gradient_sums, 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "degree,dimension,points,periodic,boundary,channels",
    [
        (5, 3, 6, True, "reject", 3),
        ((4, 1), 2, 8, True, "reject", 2),
        ((0, 5), 2, 8, True, "reject", 2),
        (4, 2, 11, False, "drop", 2),
    ],
    ids=[
        "periodic-quintic-3d",
        "mixed-quartic-linear",
        "mixed-box-quintic",
        "bounded-drop",
    ],
)
def test_gather_is_masked_exact_transpose_of_channel_deposit(
    degree: int | tuple[int, ...],
    dimension: int,
    points: int,
    periodic: bool,
    boundary: SplatBoundaryPolicy,
    channels: int,
) -> None:
    rng = np.random.default_rng(6217)
    count = 7
    grid = (
        _periodic_grid(dimension, points)
        if periodic
        else _bounded_grid(points, dimension)
    )
    prepared = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
        boundary=boundary,
    ).prepare(_particles(count, dimension))
    # Bounded sources keep their complete support except the deliberately
    # truncated first source near the lower corner.
    position = rng.uniform(0.02, 0.98, size=(count, dimension))
    if not periodic:
        position = rng.uniform(0.3, 0.7, size=(count, dimension))
        position[0] = 0.03
    active = np.asarray([True, True, False, True, True, False, True], dtype=np.bool_)
    padded = position.copy()
    padded[2] = np.nan
    padded[5] = 1.0e30
    content = rng.normal(size=(count, channels))
    content[~active] = np.nan
    field = rng.normal(size=(points,) * dimension + (channels,))

    state = prepared.build(jnp.asarray(padded), active_mask=jnp.asarray(active))
    benign = prepared.build(jnp.asarray(position), active_mask=jnp.asarray(active))
    deposited = prepared.deposit_content(state, jnp.asarray(content))
    gathered = prepared.gather(state, jnp.asarray(field))

    assert bool(state.successful) and bool(deposited.successful)
    assert np.all(np.isfinite(np.asarray(state.stencil.weights)))
    indices = np.asarray(state.stencil.indices)
    assert np.all((indices >= 0) & (indices < points**dimension))
    assert not np.any(np.asarray(state.stencil.valid)[~active])
    np.testing.assert_array_equal(
        deposited.content, prepared.deposit_content(benign, jnp.asarray(content)).content
    )
    truncated = np.asarray(state.truncated_support_mask)
    complete = active & ~truncated
    # Gather interpolates only queries whose complete stencil is present; a
    # truncated query is unsupported rather than silently partial or renormalized.
    np.testing.assert_array_equal(np.asarray(gathered.support), complete)
    np.testing.assert_array_equal(np.asarray(gathered.values)[~complete], 0.0)
    supported_content = np.where(complete[:, None], content, 0.0)
    left = np.sum(
        np.asarray(
            prepared.deposit_content(state, jnp.asarray(supported_content)).content
        )
        * field
    )
    right = np.sum(supported_content * np.asarray(gathered.values))
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    # The deposition adjoint keeps every captured route, including those of a
    # truncated source, and coincides with gather on complete queries.
    active_content = jnp.asarray(np.where(active[:, None], content, 0.0))
    _, pullback = jax.vjp(
        lambda value: prepared.deposit_content(state, value).content, active_content
    )
    adjoint = np.asarray(pullback(jnp.asarray(field))[0])
    routed = np.where(
        np.asarray(state.stencil.valid)[..., None],
        np.asarray(state.stencil.weights)[..., None]
        * field.reshape((-1, channels))[indices],
        0.0,
    ).sum(axis=1)
    np.testing.assert_allclose(adjoint, routed, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        adjoint[complete], np.asarray(gathered.values)[complete], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(deposited.content) * field),
        np.sum(np.asarray(active_content) * adjoint),
        rtol=1e-12,
        atol=1e-12,
    )
    if periodic:
        assert not np.any(truncated)
        assert deposited.balance.closed_domain_conservation_valid
        np.testing.assert_allclose(
            np.sum(np.asarray(deposited.content), axis=tuple(range(dimension))),
            np.sum(content[active], axis=0),
            rtol=1e-12,
        )
    else:
        np.testing.assert_array_equal(truncated, np.arange(count) == 0)
        assert np.any(adjoint[0] != 0.0)
        assert not deposited.balance.closed_domain_conservation_valid


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5], ids=lambda p: f"order-{p + 1}")
def test_cardinal_deposit_is_c_p_minus_one_across_knots(degree: int) -> None:
    """Derivatives below p join continuously; the p-th jumps by binomial steps."""
    points = 12
    prepared = phx.discretization.ParticleGridSplatPlan(
        _periodic_grid(1, points),
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
    ).prepare(_particles(1, 1))

    def dense(x: Any) -> Any:
        state = prepared.build(jnp.reshape(x, (1, 1)))
        return prepared.deposit_content(state, jnp.ones((1,))).content

    derivatives = [jax.jit(dense)]
    for _ in range(degree):
        derivatives.append(jax.jit(jax.jacfwd(derivatives[-1])))
    # Odd degrees have knots on entities, even degrees between them.
    knot = Fraction(10 + (degree % 2 == 0), 2 * points)
    delta = 1e-9
    left_x = jnp.asarray(float(knot) - delta)
    right_x = jnp.asarray(float(knot) + delta)

    for order in range(degree):
        jump = np.asarray(derivatives[order](right_x) - derivatives[order](left_x))
        assert np.max(np.abs(jump)) <= 1e-5 * float(points) ** order, order
    jump = np.asarray(derivatives[degree](right_x) - derivatives[degree](left_x))
    expected = np.zeros((points,))
    for node in range(points):
        t = _wrapped(knot * points - node, points)
        step = t + Fraction(degree + 1, 2)
        if step.denominator == 1 and 0 <= step <= degree + 1:
            expected[node] = (-1) ** int(step) * comb(degree + 1, int(step))
    np.testing.assert_allclose(
        jump / float(points) ** degree, expected, rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize(
    "degree", [(5, 5, 5), (4, 5, 2)], ids=["quintic", "mixed-quartic-quintic-quadratic"]
)
def test_explicit_weight_gradients_match_position_jacobian_in_3d(
    degree: tuple[int, int, int],
) -> None:
    prepared = phx.discretization.ParticleGridSplatPlan(
        _periodic_grid(3, 8),
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
    ).prepare(_particles(3, 3))
    position = jnp.asarray(
        [[0.137, 0.552, 0.903], [0.481, 0.069, 0.315], [0.77, 0.94, 0.5]]
    )
    state = prepared.build(position)
    jacobian = jax.jacfwd(lambda value: prepared.build(value).stencil.weights)(position)
    diagonal = jnp.stack([jacobian[index, :, index, :] for index in range(3)])

    assert state.stencil.indices.shape == (3, int(np.prod([p + 1 for p in degree])))
    np.testing.assert_allclose(state.weight_gradients, diagonal, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(state.gradient_sums, 0.0, atol=1e-9)
    np.testing.assert_allclose(state.partition_sums, 1.0, rtol=0.0, atol=1e-13)


def test_box_axis_declares_nearest_entity_moment_and_no_affine_exactness() -> None:
    points = 8
    mixed = phx.discretization.TensorBSplineSplatAssignment((0, 5))
    quintic = phx.discretization.TensorBSplineSplatAssignment(5)
    prepared = phx.discretization.ParticleGridSplatPlan(
        _periodic_grid(2, points), assignment=mixed
    ).prepare(_particles(3, 2))
    position = np.asarray([[0.21, 0.37], [0.58, 0.93], [0.94, 0.11]])
    state = prepared.build(jnp.asarray(position))
    nearest = np.round(position[:, 0] * points) / points

    assert mixed.capabilities.polynomial_reproduction_order == 0
    assert prepared.properties.exact_on == ("constants",)
    assert mixed.continuity_orders(2) == (-1, 4)
    assert mixed.route_width(2) == 6
    assert quintic.capabilities.polynomial_reproduction_order == 1
    assert quintic.capabilities.maximum_explicit_derivative_order == 1
    assert quintic.capabilities.maximum_support_radius_cells == 3.0
    assert quintic.continuity_orders(3) == (4, 4, 4)
    np.testing.assert_allclose(state.first_moments[:, 0], nearest - position[:, 0])
    np.testing.assert_allclose(state.first_moments[:, 1], 0.0, atol=1e-13)
    np.testing.assert_array_equal(state.weight_gradients[..., 0], 0.0)
    np.testing.assert_allclose(state.partition_sums, 1.0, rtol=0.0, atol=1e-13)


def test_bounded_quintic_support_truncation_rejects_or_drops_exact_fraction() -> None:
    points, spacing = 13, Fraction(1, 12)
    grid = _bounded_grid(points)
    assignment = phx.discretization.TensorBSplineSplatAssignment(5)
    position = jnp.asarray([[0.1], [0.5], [1.0e30]])
    content = jnp.asarray([2.0, 3.0, 5.0])
    rejecting = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=assignment, boundary="reject"
    ).prepare(_particles(3, 1))
    dropping = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=assignment, boundary="drop"
    ).prepare(_particles(3, 1))

    rejected = rejecting.build(position)
    state = dropping.build(position)
    result = dropping.deposit_content(state, content)
    captured = float(
        sum(
            _cardinal(5, (Fraction(0.1) - node * spacing) / spacing)
            for node in range(points)
        )
    )

    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.truncated_support_mask, [True, False, True])
    assert bool(state.successful) and bool(result.successful)
    np.testing.assert_allclose(state.captured_fractions[0], captured, atol=1e-13)
    assert 0.0 < captured < 1.0
    np.testing.assert_allclose(state.captured_fractions[1:], [1.0, 0.0], atol=1e-13)
    assert np.all(np.isfinite(np.asarray(state.stencil.weights)))
    assert not np.any(np.asarray(state.stencil.valid)[2])
    np.testing.assert_allclose(
        result.balance.dropped_source_total, 2.0 * (1.0 - captured) + 5.0, rtol=1e-12
    )
    np.testing.assert_allclose(jnp.sum(result.content), 2.0 * captured + 3.0, rtol=1e-12)


@pytest.mark.parametrize(
    "degree,error",
    [
        (6, ValueError),
        (-1, ValueError),
        ((0, 0), ValueError),
        ((), ValueError),
        (True, TypeError),
        (2.0, TypeError),
        ((2, 6), ValueError),
    ],
    ids=["order-7", "negative", "all-box", "empty", "bool", "float", "axis-order-7"],
)
def test_bspline_assignment_refuses_unsupported_degrees(
    degree: Any, error: type[Exception]
) -> None:
    with pytest.raises(error, match="degree"):
        phx.discretization.TensorBSplineSplatAssignment(degree)


def test_bspline_preparation_refuses_short_axes_dimension_and_route_capacity() -> None:
    with pytest.raises(ValueError, match="degree plus one"):
        phx.discretization.ParticleGridSplatPlan(
            _periodic_grid(1, 5),
            assignment=phx.discretization.TensorBSplineSplatAssignment(5),
        ).prepare(_particles(1, 1))
    with pytest.raises(ValueError, match="degree plus one"):
        phx.discretization.ParticleGridSplatPlan(
            _bounded_grid(5, 2),
            assignment=phx.discretization.TensorBSplineSplatAssignment((1, 5)),
        ).prepare(_particles(1, 2))
    with pytest.raises(ValueError, match="match the dimension"):
        phx.discretization.ParticleGridSplatPlan(
            _periodic_grid(3, 8),
            assignment=phx.discretization.TensorBSplineSplatAssignment((2, 3)),
        ).prepare(_particles(1, 3))
    with pytest.raises(ValueError, match="routes"):
        phx.discretization.ParticleGridSplatPlan(
            _periodic_grid(3, 8),
            assignment=phx.discretization.TensorBSplineSplatAssignment(5),
            budget=phx.discretization.ParticleGridSplatBudget(maximum_routes=215),
        ).prepare(_particles(1, 3))


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5], ids=lambda p: f"order-{p + 1}")
@pytest.mark.parametrize("points", [6, 9, 16])
def test_periodic_symbol_is_dft_of_deposited_unit_source(
    degree: int, points: int
) -> None:
    prepared = phx.discretization.ParticleGridSplatPlan(
        _periodic_grid(1, points),
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
    ).prepare(_particles(1, 1))
    deposit = prepared.deposit_content(
        prepared.build(jnp.zeros((1, 1))), jnp.ones((1,))
    ).content
    transform = np.fft.fft(np.asarray(deposit))
    symbol = phx.discretization.cardinal_bspline_periodic_symbol(degree, points)

    assert symbol.shape == (points,) and symbol.dtype == np.float64
    np.testing.assert_allclose(transform.real, symbol, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(transform.imag, 0.0, atol=1e-13)
    assert symbol[0] == pytest.approx(1.0, abs=1e-15)
    assert np.min(symbol) > 0.0


def test_periodic_symbol_refuses_unsupported_degree_and_short_axis() -> None:
    with pytest.raises(ValueError, match="degrees zero to five"):
        phx.discretization.cardinal_bspline_periodic_symbol(6, 16)
    with pytest.raises(ValueError, match="degree plus one"):
        phx.discretization.cardinal_bspline_periodic_symbol(5, 5)
    with pytest.raises(TypeError, match="count"):
        phx.discretization.cardinal_bspline_periodic_symbol(3, 8.0)  # ty: ignore[invalid-argument-type]
