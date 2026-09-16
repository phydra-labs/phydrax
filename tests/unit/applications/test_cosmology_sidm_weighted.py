import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm import SIDMCollisionPolicy
from phydrax.applications.cosmology._sidm_kernels import TwoBodyDifferentialKernelPlan
from phydrax.applications.cosmology._sidm_weighted import (
    WeightedPacketResamplingPlan,
    WeightedSIDMPacketState,
    WeightedSIDMPlan,
)


cosmology = phx.applications.cosmology


def _assert_tree_equal(first, second):
    for first_leaf, second_leaf in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        np.testing.assert_array_equal(first_leaf, second_leaf)


def _particle_mesh(particles):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    system = phx.equations.EulerSystem(3)
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "weighted-sidm-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        finite_volume,
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
    gravity = phx.solver.ParticleMeshGravityPlan(
        phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
            phx.solver.prepare_balance_law_transport(runtime)
        ),
        phx.discretization.ParticleGridSplatPlan(grid).prepare(particles),
    )
    kinematics = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    return cosmology.CosmologicalParticleMeshPlan(
        kinematics, gravity, jnp.asarray((0.5, 0.51))
    )


def _case(weights, *, capacity=4, active_count=2, cross_section=0.01):
    ids = jnp.asarray((101, 7, 83, 19, 211, 43, 59, 131)[:capacity])
    particles = phx.discretization.ParticleSetPlan(
        ids,
        jnp.ones((capacity,)),
        ambient_dimension=3,
    ).prepare()
    particle_mesh = _particle_mesh(particles)
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        capacity * (capacity - 1) // 2, box=box
    ).prepare(particles)
    species = DarkSectorSpeciesPlan("chi", 1.0)
    differential = TwoBodyDifferentialKernelPlan.constant_isotropic(
        species, cross_section
    )
    plan = WeightedSIDMPlan(
        particle_mesh,
        neighborhood,
        phx.discretization.WendlandC2SPHKernel(3),
        differential,
        SIDMCollisionPolicy(
            maximum_pair_probability=1.0,
            maximum_particle_probability=1.0,
            minimum_knudsen_number=1.0e-12,
            maximum_events_per_half_step=capacity // 2,
        ),
        smoothing_length_comoving=0.5,
    )
    positions = jnp.zeros((capacity, 3))
    positions = positions.at[0].set(jnp.asarray((0.45, 0.5, 0.5)))
    positions = positions.at[1].set(jnp.asarray((0.55, 0.5, 0.5)))
    if capacity > 2:
        positions = positions.at[2:].set(jnp.asarray((0.8, 0.8, 0.8)))
    active = jnp.arange(capacity) < active_count
    packet_weight = jnp.zeros((capacity,)).at[: len(weights)].set(jnp.asarray(weights))
    microscopic = jnp.where(active, 1.0, 0.0)
    macro = microscopic * packet_weight
    velocity = jnp.zeros((capacity, 3))
    velocity = velocity.at[0, 0].set(1.0)
    velocity = velocity.at[1, 0].set(-1.0)
    momentum = macro[:, None] * 0.5 * velocity
    state = plan.initialize(
        positions,
        microscopic,
        packet_weight,
        momentum,
        0.5,
        active_mask=active,
    )
    return plan, state


def _certain_collision(plan, state, key=jr.key(3), epoch=4):
    unit = plan.collide(state, key, epoch, 1.0)
    rate = jnp.max(unit.diagnostics.pair_probability)
    return plan.collide(state, key, epoch, (1.0 - 1.0e-12) / rate)


def test_equal_weight_limit_is_unsplit_elastic_rare_scattering():
    plan, state = _case((2.0, 2.0))
    result = _certain_collision(plan, state)

    assert bool(result.successful)
    assert int(result.diagnostics.event_count) == 1
    assert int(result.diagnostics.child_required) == 0
    np.testing.assert_array_equal(result.accepted_state.active_mask, state.active_mask)
    np.testing.assert_allclose(result.diagnostics.mass_defect, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(result.diagnostics.momentum_defect, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        result.diagnostics.kinetic_energy_defect, 0.0, atol=1.0e-13
    )


def test_weighted_homogeneous_rate_uses_maximum_weight():
    equal_plan, equal_state = _case((1.0, 1.0))
    weighted_plan, weighted_state = _case((3.0, 1.0))
    equal = equal_plan.collide(equal_state, jr.key(9), 2, 1.0e-3)
    weighted = weighted_plan.collide(weighted_state, jr.key(9), 2, 1.0e-3)

    valid = equal.pairs.valid & (equal.diagnostics.kernel_weight_physical > 0.0)
    np.testing.assert_allclose(
        weighted.diagnostics.pair_probability[valid],
        3.0 * equal.diagnostics.pair_probability[valid],
        rtol=2.0e-12,
    )


def test_retained_subpacket_split_has_exact_ledger_lineage_and_restart_identity():
    plan, state = _case((3.0, 1.0))
    first = _certain_collision(plan, state, jr.key(81), 17)
    restarted = _certain_collision(plan, state, jr.key(81), 17)

    assert bool(first.successful)
    assert int(first.diagnostics.child_required) == 1
    assert int(first.diagnostics.child_slots_used) == 1
    _assert_tree_equal(first.accepted_state, restarted.accepted_state)
    np.testing.assert_array_equal(
        first.accepted_state.packet_ids, plan.particles.particle_ids
    )
    child = first.accepted_state.active_mask & ~state.active_mask
    assert int(jnp.sum(child)) == 1
    parent_id = first.accepted_state.parent_packet_ids[child][0]
    assert int(parent_id) == int(state.packet_ids[0])
    assert int(first.accepted_state.lineage_depth[child][0]) == 1
    np.testing.assert_allclose(first.diagnostics.mass_defect, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(first.diagnostics.momentum_defect, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(first.diagnostics.kinetic_energy_defect, 0.0, atol=2.0e-13)


def test_near_equal_weight_retains_exact_residual_in_a_child_packet():
    epsilon = jnp.finfo(jnp.float64).eps
    plan, state = _case((1.0 + 32.0 * epsilon, 1.0))
    result = _certain_collision(plan, state, jr.key(27), 6)

    assert bool(result.successful)
    assert int(result.diagnostics.child_required) == 1
    np.testing.assert_allclose(result.diagnostics.mass_defect, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        result.diagnostics.kinetic_energy_defect, 0.0, atol=1.0e-14
    )


def test_child_capacity_failure_rolls_back_every_state_leaf_atomically():
    plan, state = _case((3.0, 1.0), capacity=2)
    result = _certain_collision(plan, state)

    assert not bool(result.diagnostics.capacity_valid)
    assert not bool(result.successful)
    _assert_tree_equal(result.accepted_state, state)


def test_successful_weighted_rollout_uses_negative_one_failure_sentinel():
    plan, state = _case((1.0, 1.0), cross_section=0.0)
    result = plan.rollout(cosmology.FLRWBackground(1.0, 0.3), state, jr.key(41))

    assert bool(result.successful)
    assert int(result.diagnostics.first_failed_step) == -1
    assert result.profile_id == plan.plan_id


def test_runtime_macro_mass_is_the_pm_source_without_support_mutation():
    plan, state = _case((3.0, 1.0))
    prepared_mass = plan.particles.masses.copy()
    deposited, routes = plan.density(state)

    assert bool(deposited.successful)
    assert bool(routes.successful)
    np.testing.assert_allclose(
        deposited.balance.target_total,
        jnp.sum(state.gravitational_masses),
        rtol=2.0e-13,
    )
    np.testing.assert_array_equal(plan.particles.masses, prepared_mass)


def test_resampler_closes_declared_kinetic_and_covariance_moments():
    plan, initialized = _case(
        (1.0, 2.0, 0.5, 1.5, 0.75, 1.25), capacity=8, active_count=6
    )
    positions = jnp.asarray(
        (
            (0.1, 0.2, 0.3),
            (0.2, 0.7, 0.4),
            (0.3, 0.1, 0.9),
            (0.6, 0.5, 0.2),
            (0.8, 0.3, 0.7),
            (0.9, 0.8, 0.6),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    velocity = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (-0.5, 0.7, 0.1),
            (0.2, -0.3, 0.9),
            (-0.7, -0.2, 0.4),
            (0.4, 0.8, -0.5),
            (-0.1, -0.6, -0.4),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    state = plan.initialize(
        positions,
        jnp.where(initialized.active_mask, 1.0, 0.0),
        initialized.weights,
        initialized.gravitational_masses[:, None] * 0.5 * velocity,
        0.5,
        active_mask=initialized.active_mask,
    )
    kinetic = WeightedPacketResamplingPlan(
        4,
        periodic_box_size=(1.0, 1.0, 1.0),
        velocity_moment="kinetic_energy",
        maximum_packet_weight=4.0,
    ).apply(state, True)
    covariance = WeightedPacketResamplingPlan(
        6,
        periodic_box_size=(1.0, 1.0, 1.0),
        velocity_moment="covariance",
        maximum_packet_weight=4.0,
    ).apply(state, True)

    assert bool(kinetic.successful)
    assert bool(covariance.successful)
    np.testing.assert_allclose(kinetic.diagnostics.mass_defect, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(kinetic.diagnostics.centroid_defect, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(kinetic.diagnostics.momentum_defect, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        kinetic.diagnostics.kinetic_energy_defect, 0.0, atol=3.0e-13
    )
    np.testing.assert_allclose(
        covariance.diagnostics.velocity_covariance_defect, 0.0, atol=3.0e-13
    )
    assert float(covariance.diagnostics.position_covariance_loss) > 0.0

    refused = WeightedPacketResamplingPlan(
        4,
        periodic_box_size=(1.0, 1.0, 1.0),
        velocity_moment="kinetic_energy",
        maximum_packet_weight=0.1,
    ).apply(state, True)
    off_boundary = WeightedPacketResamplingPlan(
        4, periodic_box_size=(1.0, 1.0, 1.0)
    ).apply(state, False)
    assert not bool(refused.successful)
    assert not bool(off_boundary.successful)
    _assert_tree_equal(refused.accepted_state, state)
    _assert_tree_equal(off_boundary.accepted_state, state)


def test_resampler_periodic_centroid_and_large_capacity_use_linear_memory_identity():
    capacity = 2048
    active = jnp.arange(capacity) < 4
    positions = jnp.full((capacity, 3), jnp.nan)
    positions = positions.at[:4].set(
        jnp.asarray(
            (
                (0.99, 0.25, 0.5),
                (0.01, 0.25, 0.5),
                (0.98, 0.25, 0.5),
                (0.02, 0.25, 0.5),
            )
        )
    )
    microscopic = jnp.where(active, 1.0, jnp.nan)
    weights = jnp.where(active, 1.0, jnp.nan)
    macro = jnp.where(active, 1.0, jnp.nan)
    momentum = jnp.where(active[:, None], jnp.zeros((capacity, 3)), jnp.nan)
    state = WeightedSIDMPacketState(
        positions,
        microscopic,
        weights,
        macro,
        momentum,
        active,
        jnp.arange(capacity, dtype=jnp.int64),
        jnp.full((capacity,), -1, dtype=jnp.int64),
        jnp.where(active, 0, -1),
        0.5,
    )
    result = WeightedPacketResamplingPlan(2, periodic_box_size=(1.0, 1.0, 1.0)).apply(
        state, True
    )

    assert bool(result.successful)
    assert bool(result.diagnostics.identity_valid)
    assert bool(result.diagnostics.lineage_valid)
    assert bool(result.diagnostics.centroid_defined)
    active_positions = result.accepted_state.positions[result.accepted_state.active_mask]
    wrapped_distance = jnp.minimum(active_positions[:, 0], 1.0 - active_positions[:, 0])
    assert bool(jnp.all(wrapped_distance < 1.0e-12))
