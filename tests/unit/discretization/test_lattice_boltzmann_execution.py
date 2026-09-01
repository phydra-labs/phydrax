#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_difference._boundary import HaloPlan
from phydrax.discretization.finite_difference._distributed import DistributedHaloSchedule
from phydrax.discretization.finite_difference._stencil import StencilFootprint
from phydrax.discretization.lattice_boltzmann._aa import AALatticeBoltzmannPlan
from phydrax.discretization.lattice_boltzmann._distributed import (
    LatticeBoltzmannHaloSchedule,
    ShardedLatticeBoltzmannExecutionPlan,
)
from phydrax.discretization.lattice_boltzmann._distributed_dynamics import (
    PreparedDistributedLatticeBoltzmannDynamics,
)
from phydrax.discretization.lattice_boltzmann._execution import (
    LatticeBoltzmannExecutionStep,
    ReferenceLatticeBoltzmannExecutionPlan,
)
from phydrax.discretization.lattice_boltzmann._fused import (
    FusedLatticeBoltzmannExecutionPlan,
)
from phydrax.discretization.lattice_boltzmann._lattice import D2Q9, D3Q19, D3Q27


def _halo_schedule(dimension, *, shape=None):
    names = tuple("xyz"[:dimension])
    global_shape = (4,) * dimension if shape is None else tuple(shape)
    halo = HaloPlan(
        StencilFootprint(names, (1,) * dimension, (1,) * dimension),
        distributed_neighbors=True,
    )
    return DistributedHaloSchedule(
        global_shape,
        (1,) * dimension,
        halo,
        periodic_axes=(True,) * dimension,
    )


def _execution_step(step_index, time, populations, step_size, args):
    del time, step_size
    candidate = populations + args
    successful = step_index != 1
    accepted = jnp.where(successful, candidate, populations)
    diagnostics = {
        "mass": jnp.sum(accepted),
        "minimum": jnp.min(accepted),
        "failed": ~successful,
    }
    return LatticeBoltzmannExecutionStep(
        candidate,
        accepted,
        successful,
        jnp.where(successful, 0.0, 1.0),
        jnp.asarray(populations.size, dtype=jnp.int32),
        diagnostics,
    )


def test_lbm_halo_routes_cover_every_velocity_offset_and_codimension():
    for velocity_set in (D2Q9(), D3Q19(), D3Q27()):
        halo = LatticeBoltzmannHaloSchedule(
            velocity_set,
            _halo_schedule(velocity_set.dimension),
        )

        assert len(halo.routes) == velocity_set.population_count
        assert tuple(route.direction_index for route in halo.routes) == tuple(
            range(velocity_set.population_count)
        )
        assert tuple(route.velocity_offset for route in halo.routes) == (
            velocity_set.velocity_tuples
        )
        assert sum(route.local for route in halo.routes) == 1
        assert all(
            route.descriptor is None
            if route.local
            else route.descriptor.offset == route.source_offset
            for route in halo.routes
        )
        expected_codimensions = {
            sum(value != 0 for value in offset) for offset in velocity_set.velocity_tuples
        }
        assert {route.codimension for route in halo.routes} == expected_codimensions


def test_lbm_reference_halo_exchange_is_direction_selected_for_faces_and_corners():
    velocity_set = D2Q9()
    halo = LatticeBoltzmannHaloSchedule(
        velocity_set,
        _halo_schedule(2, shape=(2, 3)),
    )
    blocks = jnp.broadcast_to(
        jnp.arange(velocity_set.population_count, dtype=jnp.float64),
        (1, 1, 2, 3, velocity_set.population_count),
    )
    exchanged = halo.exchange_reference(blocks)
    face = velocity_set.velocity_tuples.index((1, 0))
    corner = velocity_set.velocity_tuples.index((1, 1))

    assert exchanged.shape == (1, 1, 4, 5, velocity_set.population_count)
    np.testing.assert_allclose(exchanged[0, 0, 0, 1:4, face], float(face))
    np.testing.assert_allclose(exchanged[0, 0, -1, 1:4, face], 0.0)
    np.testing.assert_allclose(exchanged[0, 0, 0, 0, corner], float(corner))
    np.testing.assert_allclose(exchanged[0, 0, 0, -1, corner], 0.0)
    np.testing.assert_allclose(exchanged[0, 0, 1:3, 1:4], blocks[0, 0])


def test_sharded_lbm_plan_records_unpartitioned_trailing_q_metadata():
    velocity_set = D2Q9()
    reference = ReferenceLatticeBoltzmannExecutionPlan(
        velocity_set,
        _execution_step,
        step_id="execution-test-step",
    )
    halo = LatticeBoltzmannHaloSchedule(velocity_set, _halo_schedule(2))
    plan = ShardedLatticeBoltzmannExecutionPlan(reference, halo)
    populations = jnp.ones((4, 4, velocity_set.population_count), dtype=jnp.float64)

    sharded = plan.shard(populations)
    realized = plan.realize(
        populations,
        step_count=1,
        step_size=1.0,
        args=jnp.asarray(0.25),
        verify_equivalence=False,
    )

    assert plan.metadata.global_population_shape == populations.shape
    assert plan.metadata.local_population_shape == populations.shape
    assert plan.metadata.partition_shape == (1, 1)
    assert not plan.metadata.population_axis_partitioned
    assert plan.metadata.route_count == velocity_set.population_count
    assert plan.population_sharding.spec[-1] is None
    assert sharded.sharding == plan.population_sharding
    assert realized.provenance.execution_kind == "sharded"
    np.testing.assert_allclose(realized.final_populations, populations + 0.25)
    with pytest.raises(ValueError, match="only the JAX backend"):
        ShardedLatticeBoltzmannExecutionPlan(reference, halo, backend="numpy")


def test_aa_even_odd_storage_is_canonical_equivalent_and_checkpoint_exact():
    velocity_set = D2Q9()
    plan = AALatticeBoltzmannPlan(velocity_set)
    canonical = jnp.arange(
        3 * 4 * velocity_set.population_count, dtype=jnp.float64
    ).reshape((3, 4, velocity_set.population_count))
    even = plan.encode(canonical, parity=0)
    odd = plan.encode(canonical, parity=1)

    np.testing.assert_array_equal(plan.canonical(even), canonical)
    np.testing.assert_array_equal(plan.canonical(odd), canonical)
    np.testing.assert_array_equal(
        plan.addressing(even).storage_direction_indices,
        jnp.arange(velocity_set.population_count),
    )
    np.testing.assert_array_equal(
        plan.addressing(odd).storage_direction_indices,
        velocity_set.opposite,
    )
    assert (
        plan.checkpoint(even, "same-state").identity
        != plan.checkpoint(odd, "same-state").identity
    )

    advanced = plan.advance(even, canonical + 1.0)
    checkpoint = plan.checkpoint(advanced, "step-17")
    restored = plan.restore(checkpoint)

    assert checkpoint.checkpoint_id == "step-17"
    assert checkpoint.identity
    np.testing.assert_array_equal(restored.storage, advanced.storage)
    np.testing.assert_array_equal(restored.parity, advanced.parity)
    np.testing.assert_array_equal(plan.canonical(restored), canonical + 1.0)


def test_fused_realization_preserves_failure_and_diagnostic_reference_behavior():
    velocity_set = D2Q9()
    reference = ReferenceLatticeBoltzmannExecutionPlan(
        velocity_set,
        _execution_step,
        step_id="fused-parity-step",
    )
    fused = FusedLatticeBoltzmannExecutionPlan(reference)
    initial = jnp.zeros((2, 3, velocity_set.population_count), dtype=jnp.float64)

    expected = reference.realize(
        initial,
        step_count=3,
        step_size=1.0,
        args=jnp.asarray(0.25),
    )
    realized = fused.realize(
        initial,
        step_count=3,
        step_size=1.0,
        args=jnp.asarray(0.25),
    )
    production = fused.realize(
        initial,
        step_count=3,
        step_size=1.0,
        args=jnp.asarray(0.25),
        verify_equivalence=False,
    )

    assert fused.implementation == "jax-jit-scan"
    assert fused.accelerated
    assert not realized.successful
    assert realized.equivalence.equivalent
    assert realized.equivalence.populations_equivalent
    assert realized.equivalence.failures_equivalent
    assert realized.equivalence.diagnostics_equivalent
    np.testing.assert_array_equal(realized.populations, expected.populations)
    np.testing.assert_array_equal(realized.valid, jnp.asarray([True, True, False, False]))
    np.testing.assert_array_equal(
        realized.diagnostics["failed"], expected.diagnostics["failed"]
    )
    assert realized.provenance.execution_kind == "fused"
    np.testing.assert_array_equal(production.populations, expected.populations)
    assert production.provenance.execution_kind == "fused"
    assert realized.provenance.backend == "jax"


def test_prepared_distributed_dynamics_matches_actual_hydrodynamic_reference():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    compiled = phx.equations.compile_lattice_boltzmann_problem(
        phx.equations.LatticeBoltzmannProblem("distributed", 2),
        discretization,
        phx.discretization.LatticeBoltzmannMethodPlan(
            phx.discretization.BGKCollisionPlan()
        ),
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    distributed = PreparedDistributedLatticeBoltzmannDynamics(
        compiled.dynamics,
        LatticeBoltzmannHaloSchedule(discretization.velocity_set, _halo_schedule(2)),
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(0.01)
    coordinates = grid.points.reshape(grid.shape + (2,))
    velocity = jnp.stack(
        (
            0.01 * jnp.sin(2.0 * jnp.pi * coordinates[..., 1]),
            jnp.zeros(grid.shape),
        ),
        axis=-1,
    )
    initial = compiled.initialize_state(1.0, velocity, parameters)
    qualified = distributed.realize(
        initial,
        step_count=3,
        args=parameters,
        verify_equivalence=True,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    production = distributed.realize(
        initial,
        step_count=3,
        args=parameters,
        verify_equivalence=False,
    )

    assert qualified.equivalence.equivalent
    assert qualified.equivalence.failures_equivalent
    np.testing.assert_allclose(
        production.final_populations,
        qualified.final_populations,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    assert distributed.dynamics.program_manifest.stages[2].exchange_fields == (
        "post_collision",
    )
    failed = distributed.realize(
        initial,
        step_count=1,
        args=phx.discretization.LatticeBoltzmannRuntimeParameters(0.0),
        verify_equivalence=True,
    )
    assert not failed.successful
    np.testing.assert_array_equal(failed.final_populations, initial)
