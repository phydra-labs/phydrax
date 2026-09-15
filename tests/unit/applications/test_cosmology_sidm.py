import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _gravity(particles):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    system = phx.equations.EulerSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "cosmology-sidm-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity = phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    return phx.solver.ParticleMeshGravityPlan(gravity, transfer)


def _case(*, masses=None, active=None, cross_section=0.0, policy=None):
    positions = jnp.asarray(
        [
            (0.25, 0.25, 0.25),
            (0.25, 0.25, 0.75),
            (0.25, 0.75, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.25),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
            (0.75, 0.75, 0.75),
        ]
    )
    count = positions.shape[0]
    masses = jnp.full((count,), 1.0 / count) if masses is None else jnp.asarray(masses)
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((101, 7, 83, 19, 211, 43, 59, 131)),
        masses,
        ambient_dimension=3,
        active_mask=active,
    ).prepare()
    gravity = _gravity(particles)
    kdk = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    pm = cosmology.CosmologicalParticleMeshPlan(kdk, gravity, (0.5, 0.55))
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2, box=box
    ).prepare(particles)
    smoothing = phx.discretization.CoupledSummationSmoothingLengthPlan(
        1.0,
        1.0e-3,
        2.0,
        maximum_iterations=80,
        tolerance=1.0e-6,
        relaxation=0.7,
    )
    kernel = phx.discretization.WendlandC2SPHKernel(3)
    policy = (
        cosmology.SIDMCollisionPolicy(
            maximum_pair_probability=0.9,
            maximum_particle_probability=0.9,
            minimum_knudsen_number=1.0e-3,
            maximum_events_per_half_step=count // 2,
        )
        if policy is None
        else policy
    )
    sidm = cosmology.CosmologicalSIDMPlan(
        pm,
        neighborhood,
        smoothing,
        kernel,
        cosmology.SIDMCrossSectionPlan(cross_section),
        policy,
    )
    velocity = jnp.asarray(
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.5, -0.5, 0.0),
            (-0.5, 0.5, 0.0),
        ]
    )
    state = kdk.initialize(positions, masses[:, None] * 0.5 * velocity, 0.5)
    return sidm, pm, state, neighborhood


def test_zero_cross_section_is_exactly_equivalent_to_particle_mesh():
    sidm, pm, state, _ = _case(cross_section=0.0)
    expected = pm.rollout(cosmology.FLRWBackground(1.0, 0.3), state)
    actual = sidm.rollout(cosmology.FLRWBackground(1.0, 0.3), state, jr.key(17))

    assert bool(expected.successful)
    assert bool(actual.successful)
    np.testing.assert_array_equal(actual.state.positions, expected.state.positions)
    np.testing.assert_array_equal(
        actual.state.canonical_momenta, expected.state.canonical_momenta
    )
    assert not jnp.any(actual.diagnostics.first_half_collisions.accepted_pairs)
    assert not jnp.any(actual.diagnostics.second_half_collisions.accepted_pairs)


def test_probability_evidence_has_explicit_a_time_and_kernel_scaling():
    sidm, _, state, _ = _case(cross_section=1.0e-4)
    short = sidm.collide(state, jr.key(2), 4, 0.01)
    long = sidm.collide(state, jr.key(2), 4, 0.02)
    at_one = sidm.collide(
        cosmology.CosmologicalParticleState(
            state.positions, 2.0 * state.canonical_momenta, jnp.asarray(1.0)
        ),
        jr.key(2),
        4,
        0.01,
    )
    valid = short.pairs.valid & (short.diagnostics.kernel_weight_comoving > 0.0)

    np.testing.assert_allclose(
        long.diagnostics.pair_probability[valid],
        2.0 * short.diagnostics.pair_probability[valid],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        short.diagnostics.kernel_weight_physical,
        short.diagnostics.kernel_weight_comoving / state.scale_factor**3,
    )
    np.testing.assert_allclose(
        at_one.diagnostics.pair_probability[valid],
        short.diagnostics.pair_probability[valid] / 8.0,
        rtol=2e-6,
    )
    background = cosmology.FLRWBackground(1.0, 0.3)
    np.testing.assert_allclose(
        sidm.physical_time_between(background, 0.5, 0.55),
        sidm.time.cosmic_time_between(background, 0.5, 0.55),
        rtol=0.0,
        atol=0.0,
    )


def test_pair_events_are_reorder_stable_endpoint_disjoint_and_conservative():
    sidm, pm, state, neighborhood = _case(cross_section=0.01)
    base = neighborhood.pair_relation
    permutation = jnp.arange(base.capacity - 1, -1, -1)
    relation = phx.sparse.EdgeRelation(
        base.left_indices[permutation],
        base.right_indices[permutation],
        source_size=base.relation.source_size,
        target_size=base.relation.target_size,
        valid=base.valid[permutation],
    )
    reordered_pairs = phx.discretization.ParticlePairRelation(
        relation,
        base.left_particle_ids[permutation],
        base.right_particle_ids[permutation],
        source_support_id=base.source_support_id,
        target_support_id=base.target_support_id,
        same_set=True,
        unordered=True,
        relation_schema_id=base.relation_schema_id,
    )
    reordered_neighborhood = eqx.tree_at(
        lambda prepared: prepared.pair_relation,
        neighborhood,
        reordered_pairs,
    )
    reordered = cosmology.CosmologicalSIDMPlan(
        pm,
        reordered_neighborhood,
        sidm.smoothing,
        sidm.kernel,
        sidm.cross_section,
        sidm.policy,
    )

    first = sidm.collide(state, jr.key(31), 9, 3.0)
    second = reordered.collide(state, jr.key(31), 9, 3.0)
    assert bool(first.successful)
    assert bool(second.successful)
    assert int(first.diagnostics.event_count) > 0
    assert bool(first.diagnostics.endpoint_disjoint)
    assert bool(first.diagnostics.conservative)
    np.testing.assert_allclose(first.diagnostics.total_momentum_defect, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        first.diagnostics.total_kinetic_energy_defect, 0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        first.accepted_state.canonical_momenta,
        second.accepted_state.canonical_momenta,
        atol=2e-14,
    )
    first_ids = {
        tuple(sorted((int(left), int(right))))
        for left, right, accepted in zip(
            np.asarray(first.pairs.left_particle_ids),
            np.asarray(first.pairs.right_particle_ids),
            np.asarray(first.diagnostics.accepted_pairs),
            strict=True,
        )
        if accepted
    }
    second_ids = {
        tuple(sorted((int(left), int(right))))
        for left, right, accepted in zip(
            np.asarray(second.pairs.left_particle_ids),
            np.asarray(second.pairs.right_particle_ids),
            np.asarray(second.diagnostics.accepted_pairs),
            strict=True,
        )
        if accepted
    }
    assert first_ids == second_ids


def test_inactive_particles_are_untouched_and_ignored_by_equal_mass_check():
    active = jnp.asarray((True, True, True, True, True, True, True, False))
    masses = jnp.asarray((0.125,) * 7 + (99.0,))
    sidm, _, initialized, _ = _case(masses=masses, active=active, cross_section=0.01)
    marker = jnp.asarray((123.0, -7.0, 9.0))
    state = cosmology.CosmologicalParticleState(
        initialized.positions,
        initialized.canonical_momenta.at[-1].set(marker),
        initialized.scale_factor,
    )
    result = sidm.collide(state, jr.key(5), 0, 1.0)

    assert bool(result.diagnostics.equal_active_mass)
    assert bool(result.diagnostics.inactive_preserved)
    np.testing.assert_array_equal(result.accepted_state.canonical_momenta[-1], marker)
    assert not jnp.any(
        result.diagnostics.accepted_pairs
        & ((result.pairs.left_indices == 7) | (result.pairs.right_indices == 7))
    )


def test_probability_knudsen_mass_and_capacity_violations_roll_back_atomically():
    high_probability, _, state, _ = _case(cross_section=1.0)
    probability_failure = high_probability.collide(state, jr.key(8), 0, 100.0)
    assert not bool(probability_failure.diagnostics.probability_valid)
    assert not bool(probability_failure.successful)
    np.testing.assert_array_equal(
        probability_failure.accepted_state.canonical_momenta,
        state.canonical_momenta,
    )
    aggregate_policy = cosmology.SIDMCollisionPolicy(
        maximum_pair_probability=0.9,
        maximum_particle_probability=1.0e-10,
        minimum_knudsen_number=1.0e-3,
        maximum_events_per_half_step=4,
    )
    aggregate_plan, _, aggregate_state, _ = _case(
        cross_section=1.0e-4, policy=aggregate_policy
    )
    aggregate_failure = aggregate_plan.collide(aggregate_state, jr.key(8), 0, 0.01)
    assert bool(aggregate_failure.diagnostics.probability_valid)
    assert not bool(aggregate_failure.diagnostics.aggregate_probability_valid)
    assert not bool(aggregate_failure.successful)
    np.testing.assert_array_equal(
        aggregate_failure.accepted_state.canonical_momenta,
        aggregate_state.canonical_momenta,
    )

    knudsen, _, state, _ = _case(cross_section=1.0e4)
    knudsen_failure = knudsen.collide(state, jr.key(8), 0, 0.0)
    assert not bool(knudsen_failure.diagnostics.knudsen_valid)
    assert not bool(knudsen_failure.successful)
    rolled_back = knudsen.rollout(cosmology.FLRWBackground(1.0, 0.3), state, jr.key(8))
    assert not bool(rolled_back.successful)
    np.testing.assert_array_equal(rolled_back.state.positions, state.positions)
    np.testing.assert_array_equal(
        rolled_back.state.canonical_momenta, state.canonical_momenta
    )

    unequal = jnp.asarray((0.12, 0.13, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125))
    unequal_plan, _, unequal_state, _ = _case(masses=unequal, cross_section=0.0)
    mass_failure = unequal_plan.collide(unequal_state, jr.key(8), 0, 0.0)
    assert not bool(mass_failure.diagnostics.equal_active_mass)
    assert not bool(mass_failure.successful)

    capacity_policy = cosmology.SIDMCollisionPolicy(
        maximum_pair_probability=0.9,
        maximum_particle_probability=0.9,
        minimum_knudsen_number=1.0e-3,
        maximum_events_per_half_step=0,
    )
    capacity_plan, _, capacity_state, _ = _case(
        cross_section=0.01, policy=capacity_policy
    )
    capacity_failure = capacity_plan.collide(capacity_state, jr.key(31), 9, 3.0)
    assert int(capacity_failure.diagnostics.event_count) > 0
    assert not bool(capacity_failure.diagnostics.capacity_valid)
    assert not bool(capacity_failure.successful)
    np.testing.assert_array_equal(
        capacity_failure.accepted_state.canonical_momenta,
        capacity_state.canonical_momenta,
    )

    nonfinite_state = cosmology.CosmologicalParticleState(
        capacity_state.positions,
        capacity_state.canonical_momenta.at[0, 0].set(jnp.nan),
        capacity_state.scale_factor,
    )
    finite_failure = capacity_plan.collide(nonfinite_state, jr.key(1), 0, 0.0)
    assert not bool(finite_failure.diagnostics.finite)
    assert not bool(finite_failure.successful)
    np.testing.assert_array_equal(
        finite_failure.accepted_state.canonical_momenta,
        nonfinite_state.canonical_momenta,
    )
