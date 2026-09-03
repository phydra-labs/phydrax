#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import opt_einsum as oe
import pytest

import phydrax as phx


def _space(points, cells, *, component_shape=()):
    mesh = phx.discretization.CellMesh.from_polygons(jnp.asarray(points), cells)
    field = phx.discretization.ExplicitPolygonH1FieldSpec(
        "u", component_shape=component_shape
    )
    return phx.discretization.ExplicitPolygonH1Plan(mesh, field).prepare()


def test_condensed_basis_certifies_partition_affine_reproduction_and_rank():
    space = _space(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 0.4), (0.0, 1.0)),
        ((0, 1, 2, 3, 4),),
    )
    block = space.default_runtime.bases[0]
    evidence = block.evidence
    geometry = space.default_runtime.geometries[0]
    gradients = oe.contract(
        "cqnr,cqrd->cqnd", block.reference_gradients, block.inverse_jacobians
    )

    assert jnp.all(evidence.passed)
    assert jnp.allclose(jnp.sum(block.basis_values, axis=-1), 1.0, atol=1e-11)
    assert jnp.allclose(jnp.sum(gradients, axis=2), 0.0, atol=1e-11)
    assert jnp.allclose(
        oe.contract("cqn,cnd->cqd", block.basis_values, geometry.vertices),
        block.physical_points,
        atol=1e-11,
    )
    assert jnp.all(evidence.stiffness_rank == block.arity - 1)
    assert jnp.all(evidence.mass_minimum_eigenvalue > 0.0)


def test_triangle_reconstruction_is_affine_and_trace_is_orientation_independent():
    space = _space(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),))
    state = space.mesh.coordinates[:, 0] + 2.0 * space.mesh.coordinates[:, 1]
    reconstruction = phx.discretization.prepare_explicit_polygon_h1_reconstruction(
        space, state
    )
    points = jnp.asarray([[[0.2, 0.3], [0.1, 0.6]]])
    values, gradients = phx.discretization.evaluate_explicit_polygon_h1_reconstruction(
        reconstruction, space, 0, points
    )
    trace = phx.discretization.evaluate_explicit_polygon_h1_trace(
        reconstruction, space, jnp.asarray((0,)), jnp.asarray((0.0, 0.25, 1.0))
    )

    assert jnp.allclose(values, points[..., 0] + 2.0 * points[..., 1], atol=1e-11)
    assert jnp.allclose(gradients, jnp.asarray((1.0, 2.0)), atol=1e-11)
    assert jnp.allclose(trace, jnp.asarray([[0.0, 0.25, 1.0]]), atol=1e-11)


def test_mixed_arity_padding_and_component_axes_are_inert():
    points = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (3.2, 0.7),
        (2.5, 1.2),
        (1.8, 0.7),
    )
    space = _space(points, ((0, 1, 2), (3, 4, 5, 6, 7)), component_shape=(2,))
    assert space.dof_map.local_width == 5
    assert [basis.arity for basis in space.default_runtime.bases] == [3, 5]
    assert jnp.all(space.default_runtime.bases[0].basis_values[..., 3:] == 0.0)

    region = space.prepare_local_regions(
        space.cell_domain,
        field_names=("u",),
        maximum_derivative_order=1,
        kernel_mode="dense",
    )[0]
    local = jnp.ones((region.entity_indices.size, space.dof_map.local_width, 2))
    values = region.reference_actions[0].interpolate(space.default_runtime, local)
    assert values.shape[-1] == 2
    assert jnp.allclose(values, 1.0, atol=1e-11)


def test_unmatched_hanging_interface_is_rejected_but_matched_collinear_is_valid():
    points = jnp.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, -1.0),
            (1.0, -1.0),
        )
    )
    unmatched = phx.discretization.CellMesh.from_polygons(
        points, ((0, 2, 4, 3), (0, 5, 6, 2, 1))
    )
    field = phx.discretization.ExplicitPolygonH1FieldSpec("u")
    with pytest.raises(ValueError, match="T-junction|hanging-node"):
        phx.discretization.ExplicitPolygonH1Plan(unmatched, field).prepare()

    matched = phx.discretization.CellMesh.from_polygons(
        points, ((0, 1, 2, 4, 3), (0, 5, 6, 2, 1))
    )
    space = phx.discretization.ExplicitPolygonH1Plan(matched, field).prepare()
    assert all(jnp.all(block.evidence.passed) for block in space.default_runtime.bases)


def test_runtime_refresh_is_differentiable_and_preserves_layout_identity():
    space = _space(
        ((0.0, 0.0), (1.0, 0.0), (1.1, 1.0), (0.0, 1.0)),
        ((0, 1, 2, 3),),
    )
    coordinates = space.default_runtime.coordinates

    def response(value):
        runtime = space.prepare_runtime(value, numeric_version="shape-gradient")
        return jnp.sum(runtime.bases[0].prolongation ** 2)

    gradient = jax.grad(response)(coordinates)
    moved = coordinates.at[2, 0].add(0.05)
    runtime = space.prepare_runtime(moved, numeric_version="moved")

    assert jnp.all(jnp.isfinite(gradient))
    assert runtime.topology_id == space.default_runtime.topology_id
    assert runtime.runtime_id != space.default_runtime.runtime_id
    assert jnp.all(runtime.bases[0].evidence.passed)


def test_resource_and_outside_reconstruction_fail_closed():
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    mesh = phx.discretization.CellMesh.from_polygons(jnp.asarray(points), ((0, 1, 2, 3),))
    field = phx.discretization.ExplicitPolygonH1FieldSpec("u")
    with pytest.raises(ValueError, match="arity budget"):
        phx.discretization.ExplicitPolygonH1Plan(
            mesh,
            field,
            resource_budget=phx.discretization.ExplicitPolygonH1ResourceBudget(
                maximum_arity=3
            ),
        )
    space = phx.discretization.ExplicitPolygonH1Plan(mesh, field).prepare()
    reconstruction = phx.discretization.prepare_explicit_polygon_h1_reconstruction(
        space, jnp.zeros((4,))
    )
    with pytest.raises(Exception, match="outside an admissible polygon fan"):
        phx.discretization.evaluate_explicit_polygon_h1_reconstruction(
            reconstruction, space, 0, jnp.asarray([[[2.0, 2.0]]])
        )
