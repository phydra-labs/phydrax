#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(points: int = 3, *, periodic: bool = False):
    axis = phx.discretization.UniformAxisSpec(
        points,
        periodic=periodic,
        endpoint=not periodic,
    )
    return phx.discretization.TensorGridPlan((axis,), axis_names=("x",)).prepare(
        jnp.asarray([[0.0], [1.0]])
    )


def _particles(
    count: int = 2,
    *,
    dimension: int = 1,
    active_mask=None,
    particle_ids=None,
):
    ids = jnp.arange(count) if particle_ids is None else jnp.asarray(particle_ids)
    return phx.discretization.ParticleSetPlan(
        ids,
        jnp.ones((count,)),
        ambient_dimension=dimension,
        active_mask=active_mask,
    ).prepare()


def test_multilinear_deposit_distinguishes_content_and_density():
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(
        _particles(particle_ids=(2, 1))
    )
    state = prepared.build(jnp.asarray([[0.25], [0.75]]))
    result = prepared.deposit_content(state, jnp.asarray([2.0, 4.0]))

    assert jnp.allclose(result.content, jnp.asarray([1.0, 3.0, 2.0]))
    assert jnp.allclose(result.density, jnp.asarray([4.0, 6.0, 8.0]))
    assert result.successful
    assert result.balance.closed_domain_conservation_valid
    assert result.balance.maximum_absolute_balance_defect == 0.0
    assert result.balance.maximum_partition_defect == 0.0
    assert state.valid_route_count == 4
    assert prepared.properties.constant_preserving
    assert prepared.properties.adjoint_paired
    assert result.balance.source_support_id == prepared.particles.support.support_id
    assert result.balance.source_support_id != prepared.particles.measures[0].measure_id
    assert jnp.array_equal(result.require_success(result.content), result.content)
    assert jnp.array_equal(
        result.balance.require_closed_conservation(result.content), result.content
    )


def test_periodic_deposit_wraps_across_the_seam():
    prepared = phx.discretization.ParticleGridSplatPlan(_grid(4, periodic=True)).prepare(
        _particles(1)
    )
    result = prepared.deposit_content(
        prepared.build(jnp.asarray([[0.875]])), jnp.asarray([2.0])
    )

    assert jnp.allclose(result.content, jnp.asarray([1.0, 0.0, 0.0, 1.0]))
    assert jnp.sum(result.content) == 2.0
    assert result.balance.closed_domain_conservation_valid


def test_nonperiodic_boundary_rejects_or_accounts_for_dropped_content():
    particles = _particles(1)
    position = jnp.asarray([[-0.1]])
    rejecting = phx.discretization.ParticleGridSplatPlan(
        _grid(), boundary="reject"
    ).prepare(particles)
    rejected_state = rejecting.build(position)
    assert not rejected_state.successful
    assert rejected_state.dropped_source_count == 1
    with pytest.raises(RuntimeError, match="boundary checks"):
        rejecting.deposit_content(rejected_state, jnp.asarray([3.0]))

    dropping = phx.discretization.ParticleGridSplatPlan(_grid(), boundary="drop").prepare(
        particles
    )
    dropped = dropping.deposit_content(dropping.build(position), jnp.asarray([3.0]))
    assert dropped.successful
    assert jnp.all(dropped.content == 0.0)
    assert dropped.balance.dropped_source_total == 3.0
    assert dropped.balance.dropped_source_absolute_total == 3.0
    assert dropped.balance.maximum_absolute_balance_defect == 0.0
    assert not dropped.balance.closed_domain_conservation_valid


def test_inactive_nonfinite_storage_is_numerically_inert():
    particles = _particles(2, active_mask=jnp.asarray([True, False]))
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(particles)
    state = prepared.build(jnp.asarray([[0.5], [jnp.nan]]))
    result = prepared.deposit_content(state, jnp.asarray([2.0, jnp.nan]))

    assert state.successful
    assert state.invalid_geometry_count == 0
    assert jnp.allclose(result.content, jnp.asarray([0.0, 2.0, 0.0]))
    assert result.balance.active_source_total == 2.0


def test_vector_complex_payload_preserves_trailing_shape_and_balance():
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(_particles())
    state = prepared.build(jnp.asarray([[0.25], [0.75]]))
    payload = jnp.asarray([[1.0 + 2.0j, 3.0], [2.0 - 1.0j, -4.0]])
    result = prepared.deposit_content(state, payload)

    assert result.content.shape == (3, 2)
    assert jnp.allclose(jnp.sum(result.content, axis=0), jnp.sum(payload, axis=0))
    assert result.balance.maximum_absolute_balance_defect < 1e-12


def test_reconstruction_and_gather_keep_coverage_explicit():
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(_particles())
    state = prepared.build(jnp.asarray([[0.25], [0.75]]))
    reconstructed = prepared.reconstruct(
        state,
        jnp.asarray([2.0, 4.0]),
        jnp.ones((2,)),
    )
    gathered = prepared.gather(state, jnp.asarray([0.0, 1.0, 2.0]))

    assert jnp.allclose(reconstructed.values, jnp.asarray([2.0, 3.0, 4.0]))
    assert jnp.all(reconstructed.support)
    assert reconstructed.zero_coverage_count == 0
    assert jnp.allclose(gathered.values, jnp.asarray([0.5, 1.5]))
    assert jnp.all(gathered.support)


def test_reconstruction_zero_coverage_and_weight_validation():
    prepared = phx.discretization.ParticleGridSplatPlan(_grid(5)).prepare(_particles(1))
    state = prepared.build(jnp.asarray([[0.0]]))
    result = prepared.reconstruct(state, jnp.asarray([7.0]), jnp.asarray([1.0]))

    assert result.values[0] == 7.0
    assert jnp.all(result.values[1:] == 0.0)
    assert jnp.array_equal(
        result.support, jnp.asarray([True, False, False, False, False])
    )
    assert result.zero_coverage_count == 4
    with pytest.raises(ValueError, match="nonnegative"):
        prepared.reconstruct(state, jnp.asarray([7.0]), jnp.asarray([-1.0]))
    empty = prepared.reconstruct(state, jnp.asarray([7.0]), jnp.asarray([0.0]))
    assert jnp.all(empty.values == 0.0)
    assert not jnp.any(empty.support)
    assert empty.denominator_tolerance > 0.0


def test_vmap_keeps_independent_cases_and_shared_structure():
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(_particles())
    positions = jnp.asarray([[[0.25], [0.75]], [[0.5], [1.0]]])
    payloads = jnp.asarray([[2.0, 4.0], [3.0, 1.0]])

    def apply(position, payload):
        return prepared.deposit_content(prepared.build(position), payload).content

    batched = jax.jit(jax.vmap(apply))(positions, payloads)
    sequential = jnp.stack(
        [apply(position, payload) for position, payload in zip(positions, payloads)]
    )

    assert batched.shape == (2, 3)
    assert jnp.allclose(batched, sequential)


def test_deterministic_and_compensated_results_ignore_storage_permutation():
    grid = _grid(5)
    ids = jnp.asarray([30, 10, 20])
    position = jnp.asarray([[0.2], [0.55], [0.8]])
    content = jnp.asarray([[1.0, -2.0], [3.0, 0.5], [-4.0, 7.0]])
    permutation = jnp.asarray([1, 2, 0])

    for accumulation in ("deterministic", "compensated"):
        execution = phx.discretization.SplatExecutionPolicy(accumulation=accumulation)
        first_particles = _particles(3, particle_ids=ids)
        second_particles = _particles(3, particle_ids=ids[permutation])
        first = phx.discretization.ParticleGridSplatPlan(
            grid, execution=execution
        ).prepare(first_particles)
        second = phx.discretization.ParticleGridSplatPlan(
            grid, execution=execution
        ).prepare(second_particles)
        first_result = first.deposit_content(first.build(position), content)
        second_result = second.deposit_content(
            second.build(position[permutation]), content[permutation]
        )

        assert jnp.array_equal(first_result.content, second_result.content)
        assert jnp.array_equal(
            first_result.balance.balance_defect,
            second_result.balance.balance_defect,
        )


def test_plan_rejects_incompatible_dimension_resources_and_state():
    grid = _grid()
    with pytest.raises(ValueError, match="dimensions"):
        phx.discretization.ParticleGridSplatPlan(grid).prepare(_particles(1, dimension=2))
    tiny = phx.discretization.ParticleGridSplatBudget(maximum_routes=1)
    with pytest.raises(ValueError, match="routes"):
        phx.discretization.ParticleGridSplatPlan(grid, budget=tiny).prepare(_particles(1))

    particles = _particles()
    first = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    second = phx.discretization.ParticleGridSplatPlan(grid, boundary="drop").prepare(
        particles
    )
    state = first.build(jnp.asarray([[0.25], [0.75]]))
    with pytest.raises(ValueError, match="different prepared transfer"):
        second.deposit_content(state, jnp.ones((2,)))


def test_execution_policy_validation_and_frozen_geometry():
    with pytest.raises(ValueError, match="accumulation"):
        phx.discretization.SplatExecutionPolicy(accumulation="unknown")
    with pytest.raises(ValueError, match="geometry_ad"):
        phx.discretization.SplatExecutionPolicy(geometry_ad="unknown")

    policy = phx.discretization.SplatExecutionPolicy(geometry_ad="frozen")
    prepared = phx.discretization.ParticleGridSplatPlan(
        _grid(), execution=policy
    ).prepare(_particles())

    def loss(position):
        state = prepared.build(position)
        return jnp.sum(
            prepared.deposit_content(state, jnp.asarray([2.0, 4.0])).content ** 2
        )

    gradient = jax.grad(loss)(jnp.asarray([[0.25], [0.75]]))
    assert jnp.all(gradient == 0.0)


def test_splatting_public_api_is_provider_neutral():
    public = vars(phx.discretization)
    expected = {
        "AbstractStructuredSplatAssignment",
        "MultilinearSplatAssignment",
        "ParticleGridSplatPlan",
        "PreparedParticleGridSplat",
        "SplatAssignmentCapabilities",
        "TensorBSplineSplatAssignment",
    }

    assert expected <= public.keys()
    assert "ConservativeParticleGridTransferPlan" not in public
    assert "ParticleGridRelation" not in public
    assert "PreparedParticleGridTransfer" not in public
    assert "deposit_routes" not in public
    assert "splax" not in sys.modules
    assert "warp" not in sys.modules


def test_mesh_splat_barycentric_and_compact_routes_are_conservative():
    triangle = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "triangles",
                "triangle",
                jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
            ),
        ),
    )
    vertex_measure = phx.discretization.DiscreteMeasure(
        "vertex_measure",
        triangle.support.support_id,
        triangle.topology.entities(0).entity_set_id,
        jnp.ones((3,)),
    )
    vertex_target = phx.discretization.MeshSplatTarget(
        triangle, entity_dimension=0, measure=vertex_measure
    )
    assert vertex_target.entity_count == triangle.topology.entities(0).count
    np.testing.assert_array_equal(
        vertex_target.stable_entity_ids,
        triangle.topology.entities(0).entity_ids,
    )
    prepared = phx.discretization.SimplicialBarycentricSplatAssignment().prepare(
        vertex_target,
        jnp.asarray(((0.25, 0.25),)),
        jnp.asarray((True,)),
        jnp.asarray((7,), dtype=jnp.int64),
    )
    routes = prepared.routes(
        jnp.asarray(((0.25, 0.25),)),
        jnp.asarray((True,)),
    )
    deposited = prepared.deposit(
        jnp.asarray(((0.25, 0.25),)),
        jnp.asarray((True,)),
        jnp.asarray((2.0,)),
    )
    gathered = prepared.gather(
        jnp.asarray(((0.25, 0.25),)),
        jnp.asarray((True,)),
        triangle.coordinates[:, 0],
    )

    assert routes.valid.shape == routes.weights.shape == (1, 3)
    assert bool(deposited.successful)
    np.testing.assert_allclose(jnp.sum(deposited.content), 2.0)
    np.testing.assert_allclose(gathered.values, (0.25,))

    cell_measure = phx.discretization.DiscreteMeasure(
        "cell_measure",
        triangle.support.support_id,
        triangle.topology.entities(2).entity_set_id,
        jnp.ones((1,)),
    )
    cell_target = phx.discretization.MeshSplatTarget(
        triangle, entity_dimension=2, measure=cell_measure
    )
    compact = phx.discretization.MeshCompactKernelSplatAssignment(
        2.0, 1, partition_policy="normalize"
    ).prepare(
        cell_target,
        jnp.asarray(((0.25, 0.25),)),
        jnp.asarray((True,)),
        jnp.asarray((7,), dtype=jnp.int64),
    )
    compact_deposit = compact.deposit(
        jnp.asarray(((0.25, 0.25),)),
        jnp.asarray((True,)),
        jnp.asarray((3.0,)),
    )
    assert bool(compact_deposit.successful)
    np.testing.assert_allclose(compact_deposit.content, (3.0,))
