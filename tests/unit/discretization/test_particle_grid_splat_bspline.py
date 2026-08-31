#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _particles(count: int, dimension: int):
    return phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()


def _periodic_grid(dimension: int, points: int = 16, *, cell_primary: bool = False):
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


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_periodic_bspline_partition_moments_and_balance(degree: int, dimension: int):
    grid = _periodic_grid(dimension)
    particles = _particles(4, dimension)
    base = jnp.asarray(
        [[0.13, 0.27, 0.41], [0.62, 0.18, 0.84], [0.35, 0.79, 0.22], [0.91, 0.68, 0.57]]
    )[:, :dimension]
    assignment = phx.discretization.TensorBSplineSplatAssignment(degree)
    prepared = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=assignment
    ).prepare(particles)
    state = prepared.build(base)
    result = prepared.deposit_content(state, jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    assert state.stencil.indices.shape == (4, (degree + 1) ** dimension)
    assert jnp.allclose(state.partition_sums, 1.0, atol=1e-12)
    assert jnp.min(jnp.where(state.stencil.valid, state.stencil.weights, jnp.inf)) >= 0.0
    assert jnp.max(jnp.abs(state.first_moments)) < 1e-12
    assert jnp.max(jnp.abs(state.gradient_sums)) < 1e-11
    assert jnp.all(jnp.diagonal(state.second_moments, axis1=-2, axis2=-1) > 0.0)
    assert jnp.allclose(jnp.sum(result.content), 10.0)
    assert result.balance.closed_domain_conservation_valid


def test_degree_one_bspline_matches_multilinear_on_uniform_nodal_grid():
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


def test_uniform_cell_face_and_edge_layouts_are_supported():
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


def test_bounded_bspline_rejects_or_accounts_for_partial_support():
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


def test_route_weight_gradients_match_jax_jacobian():
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


def test_bspline_rejects_invalid_degree_nonuniform_axes_and_budget():
    with pytest.raises(ValueError, match="degrees"):
        phx.discretization.TensorBSplineSplatAssignment(4)
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
