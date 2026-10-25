#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import sys
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(points: int = 3, *, periodic: bool = False) -> Any:
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
    active_mask: Any = None,
    particle_ids: Any = None,
) -> Any:
    ids = jnp.arange(count) if particle_ids is None else jnp.asarray(particle_ids)
    return phx.discretization.ParticleSetPlan(
        ids,
        jnp.ones((count,)),
        ambient_dimension=dimension,
        active_mask=active_mask,
    ).prepare()


def test_particle_grid_splat_scenario_1() -> None:
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
    support = prepared.physical_particles()
    assert result.balance.source_support_id == support.support.support_id
    assert result.balance.source_support_id != support.measures[0].measure_id
    assert jnp.array_equal(result.require_success(result.content), result.content)
    assert jnp.array_equal(
        result.balance.require_closed_conservation(result.content), result.content
    )
    prepared = phx.discretization.ParticleGridSplatPlan(_grid(4, periodic=True)).prepare(
        _particles(1)
    )
    result = prepared.deposit_content(
        prepared.build(jnp.asarray([[0.875]])), jnp.asarray([2.0])
    )

    assert jnp.allclose(result.content, jnp.asarray([1.0, 0.0, 0.0, 1.0]))
    assert jnp.sum(result.content) == 2.0
    assert result.balance.closed_domain_conservation_valid
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


def test_particle_grid_splat_scenario_2() -> None:
    particles = _particles(2, active_mask=jnp.asarray([True, False]))
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(particles)
    state = prepared.build(jnp.asarray([[0.5], [jnp.nan]]))
    result = prepared.deposit_content(state, jnp.asarray([2.0, jnp.nan]))

    assert state.successful
    assert state.invalid_geometry_count == 0
    assert jnp.allclose(result.content, jnp.asarray([0.0, 2.0, 0.0]))
    assert result.balance.active_source_total == 2.0
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(_particles())
    state = prepared.build(jnp.asarray([[0.25], [0.75]]))
    payload = jnp.asarray([[1.0 + 2.0j, 3.0], [2.0 - 1.0j, -4.0]])
    result = prepared.deposit_content(state, payload)

    assert result.content.shape == (3, 2)
    assert jnp.allclose(jnp.sum(result.content, axis=0), jnp.sum(payload, axis=0))
    assert result.balance.maximum_absolute_balance_defect < 1e-12
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
    prepared = phx.discretization.ParticleGridSplatPlan(_grid(5)).prepare(_particles(1))
    state = prepared.build(jnp.asarray([[0.0]]))
    result = prepared.reconstruct(state, jnp.asarray([7.0]), jnp.asarray([1.0]))

    assert result.values[0] == 7.0
    assert jnp.all(result.values[1:] == 0.0)
    assert jnp.array_equal(
        result.support, jnp.asarray([True, False, False, False, False])
    )
    assert result.zero_coverage_count == 4
    with pytest.raises(eqx.EquinoxRuntimeError, match="nonnegative"):
        prepared.reconstruct(state, jnp.asarray([7.0]), jnp.asarray([-1.0]))
    empty = prepared.reconstruct(state, jnp.asarray([7.0]), jnp.asarray([0.0]))
    assert jnp.all(empty.values == 0.0)
    assert not jnp.any(empty.support)
    assert empty.denominator_tolerance > 0.0


def test_vmap_keeps_independent_cases_and_shared_structure() -> None:
    prepared = phx.discretization.ParticleGridSplatPlan(_grid()).prepare(_particles())
    positions = jnp.asarray([[[0.25], [0.75]], [[0.5], [1.0]]])
    payloads = jnp.asarray([[2.0, 4.0], [3.0, 1.0]])

    def apply(position: Any, payload: Any) -> Any:
        return prepared.deposit_content(prepared.build(position), payload).content

    batched = jax.jit(jax.vmap(apply))(positions, payloads)
    sequential = jnp.stack(
        [apply(position, payload) for position, payload in zip(positions, payloads)]
    )

    assert batched.shape == (2, 3)
    assert jnp.allclose(batched, sequential)


def test_particle_grid_splat_scenario_3() -> None:
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
    public = set(dir(phx.discretization))
    expected = {
        "AbstractStructuredSplatAssignment",
        "MultilinearSplatAssignment",
        "ParticleGridSplatPlan",
        "PreparedParticleGridSplat",
        "SplatAssignmentCapabilities",
        "TensorBSplineSplatAssignment",
    }

    assert expected <= public
    assert "ConservativeParticleGridTransferPlan" not in public
    assert "ParticleGridRelation" not in public
    assert "PreparedParticleGridTransfer" not in public
    assert "deposit_routes" not in public
    assert "splax" not in sys.modules
    assert "warp" not in sys.modules


def test_execution_policy_validation_and_frozen_geometry() -> None:
    with pytest.raises(ValueError, match="accumulation"):
        # ty: ignore[invalid-argument-type]
        phx.discretization.SplatExecutionPolicy(accumulation="unknown")
    with pytest.raises(ValueError, match="geometry_ad"):
        # ty: ignore[invalid-argument-type]
        phx.discretization.SplatExecutionPolicy(geometry_ad="unknown")

    policy = phx.discretization.SplatExecutionPolicy(geometry_ad="frozen")
    prepared = phx.discretization.ParticleGridSplatPlan(
        _grid(), execution=policy
    ).prepare(_particles())

    def loss(position: Any) -> Any:
        state = prepared.build(position)
        return jnp.sum(
            prepared.deposit_content(state, jnp.asarray([2.0, 4.0])).content ** 2
        )

    gradient = jax.grad(loss)(jnp.asarray([[0.25], [0.75]]))
    assert jnp.all(gradient == 0.0)


def test_mesh_splat_barycentric_and_compact_routes_are_conservative() -> None:
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
    nonfinite_gather = prepared.gather(
        jnp.asarray(((0.25, 0.25),)),
        jnp.asarray((True,)),
        triangle.coordinates[:, 0].at[0].set(jnp.nan),
    )
    assert not bool(nonfinite_gather.successful)

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


def test_indexed_support_splat_matches_physical_particle_splat() -> None:
    from phydrax.discretization.splatting import PreparedParticleGridSplat

    ids = np.asarray((7, 2, 5, 3), dtype=np.int64)
    active = np.asarray((True, True, False, True))
    plan = phx.discretization.ParticleGridSplatPlan(_grid(6, periodic=True))
    physical = plan.prepare(_particles(4, particle_ids=ids, active_mask=active))
    schema = PreparedParticleGridSplat(
        plan,
        particle_ids=ids,
        active_mask=active,
        support_id="test-indexed-support-schema",
        coordinate_dtype="float64",
    )
    with pytest.raises(ValueError, match="indexed-support schema"):
        schema.physical_particles()
    assert schema.source_capacity == physical.source_capacity == 4
    assert schema.ambient_dimension == physical.ambient_dimension == 1
    assert schema.source_support_id == "test-indexed-support-schema"
    physical_support = physical.physical_particles().support
    assert physical.source_support_id == physical_support.support_id
    np.testing.assert_array_equal(
        schema.stable_source_order, physical.stable_source_order
    )
    np.testing.assert_array_equal(schema.source_active_mask, physical.source_active_mask)

    # The schema statics are content-free: other IDs and masks of the same
    # layout share the prepared identity and treedef.
    other = PreparedParticleGridSplat(
        plan,
        particle_ids=np.asarray((1, 9, 4, 6), dtype=np.int64),
        active_mask=np.asarray((True, False, True, True)),
        support_id="test-indexed-support-schema",
        coordinate_dtype="float64",
    )
    assert other.prepared_id == schema.prepared_id
    assert jax.tree.structure(other) == jax.tree.structure(schema)

    positions = jnp.asarray([[0.1], [0.45], [jnp.nan], [0.8]])
    values = jnp.asarray([2.0, -1.5, jnp.nan, 0.75])
    physical_state = physical.build(positions)
    schema_state = schema.build(positions)
    physical_deposit = physical.deposit_content(physical_state, values)
    schema_deposit = schema.deposit_content(schema_state, values)
    np.testing.assert_array_equal(schema_deposit.content, physical_deposit.content)
    np.testing.assert_array_equal(schema_deposit.density, physical_deposit.density)
    assert schema_deposit.balance.source_support_id == "test-indexed-support-schema"
    grid_values = jnp.linspace(-1.0, 1.0, 6)
    np.testing.assert_array_equal(
        schema.gather(schema_state, grid_values).values,
        physical.gather(physical_state, grid_values).values,
    )
    # Adjoint pairing <deposit(v), g> = <v, gather(g)> on active sources.
    active_values = jnp.where(active, values, 0.0)
    np.testing.assert_allclose(
        jnp.sum(schema_deposit.content * grid_values),
        jnp.sum(active_values * schema.gather(schema_state, grid_values).values),
        rtol=1e-13,
        atol=1e-14,
    )

    with pytest.raises(ValueError, match="unique and nonnegative"):
        PreparedParticleGridSplat(
            plan,
            particle_ids=np.asarray((1, 1, 2, 3), dtype=np.int64),
            active_mask=active,
            support_id="test-indexed-support-schema",
            coordinate_dtype="float64",
        )
    with pytest.raises(ValueError, match="not both"):
        PreparedParticleGridSplat(
            plan,
            _particles(4, particle_ids=ids, active_mask=active),
            support_id="test-indexed-support-schema",
        )
