#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_cached_rolling_curved_wall_rollout_preserves_replay_and_energy_evidence():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.2]]),
        rolling_friction=jnp.asarray([[0.05]]),
    )
    barrier = phx.discretization.ImplicitDEMBarrier(
        phx.geometry.Circle((0.0, 0.0), 2.0).compile(),
        phx.discretization.DEMBarrierSide.INTERIOR,
        0,
        barrier_id="curved-container",
    )
    contact = phx.discretization.DEMContactModelPlan(
        phx.discretization.HertzNormalContactPlan(),
        tangential=phx.discretization.MindlinTangentialContactPlan(),
        rotational=phx.discretization.ConstantRollingResistancePlan(),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        contact, maximum_overlap_fraction=0.3
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "cached-curved-dem",
        materials,
        gravity=jnp.zeros((2,)),
        barriers=(barrier,),
    )
    box = phx.discretization.ParticleBox(
        jnp.asarray([-2.0, -2.0]),
        jnp.asarray([2.0, 2.0]),
        periodic_axes=(False, False),
    )
    base = phx.discretization.CellListParticleNeighborhoodPlan(1.2, 2, 1, box)
    neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(base, 1.0, 0.2)
    compiled = phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=neighborhood,
        execution=phx.discretization.ParticleExecutionPolicy(
            realization="cell_edge_list", kernel_backend="verlet_fused"
        ),
    )
    initial = compiled.initialize_state(
        0.0,
        jnp.asarray([[1.55, 0.0], [0.0, 0.0]]),
        jnp.zeros((2, 2)),
        jnp.asarray([[1.0], [0.0]]),
    )
    fixed = phx.solver.FixedStepProblem(
        phx.solver.DEMFixedStepMethod(compiled.dynamics),
        initial,
        t0=0.0,
        t1=5.0e-4,
        step_size=1.0e-4,
        state_geometry=compiled.dynamics.state_geometry,
        discretization_bundle=compiled.discretization_bundle,
    )
    solution = phx.solver.solve_fixed_step(fixed)
    final = jax.tree.map(lambda value: value[-1], solution.states)
    assert solution.successful
    assert final.energy.accepted_steps == 5
    assert jnp.isfinite(final.energy.cumulative_contact_balance_loss)
    assert final.neighborhood_cache.rebuild_count >= 1

    first = phx.discretization.checkpointed_dem_rollout(
        compiled.dynamics,
        initial,
        t0=0.0,
        step_size=1.0e-4,
        step_count=4,
        checkpoint=phx.discretization.DEMCheckpointPolicy(2),
    )
    second = phx.discretization.checkpointed_dem_rollout(
        compiled.dynamics,
        initial,
        t0=0.0,
        step_size=1.0e-4,
        step_count=4,
        checkpoint=phx.discretization.DEMCheckpointPolicy(2),
    )
    assert first.successful
    assert phx.discretization.dem_replay_matches(first.replay, second.replay)
