import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_frequent import FrequentSmallAngleSIDMPlan
from phydrax.applications.cosmology._sidm_kernels import (
    SmallAngleSplitPlan,
    TwoBodyDifferentialKernelPlan,
)
from phydrax.applications.cosmology._sidm_weighted import WeightedSIDMPacketState


def _case(
    *,
    split_cosine=0.8,
    maximum_drag=0.1,
    moment_tolerance=0.2,
    microscopic_mass=1.0,
    weights=(1.0, 1.0),
):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((101, 7)),
        jnp.ones((2,)),
        ambient_dimension=3,
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=box).prepare(
        particles
    )
    species = DarkSectorSpeciesPlan("chi", microscopic_mass)
    kernel = TwoBodyDifferentialKernelPlan.constant_isotropic(species, 0.05)
    split = SmallAngleSplitPlan(kernel, split_cosine)
    plan = FrequentSmallAngleSIDMPlan(
        neighborhood,
        phx.discretization.WendlandC2SPHKernel(3),
        split,
        smoothing_length_comoving=0.5,
        maximum_drag_fraction_per_step=maximum_drag,
        maximum_transverse_variance_per_step=2.0 * maximum_drag,
        moment_tolerance=moment_tolerance,
    )
    positions = jnp.asarray(((0.45, 0.5, 0.5), (0.55, 0.5, 0.5)))
    packet_weight = jnp.asarray(weights)
    macro_mass = microscopic_mass * packet_weight
    velocity = jnp.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    state = WeightedSIDMPacketState(
        positions,
        jnp.full((2,), microscopic_mass),
        packet_weight,
        macro_mass,
        macro_mass[:, None] * 0.5 * velocity,
        jnp.ones((2,), dtype=bool),
        particles.particle_ids,
        jnp.full((2,), -1, dtype=jnp.int64),
        jnp.zeros((2,), dtype=jnp.int32),
        0.5,
    )
    return plan, state


def _step_for_drag(plan, state, drag_fraction):
    unit = plan.apply(state, jr.key(0), 0, 1.0)
    coefficient = jnp.max(unit.diagnostics.target_drag_fraction)
    return drag_fraction / coefficient


def _velocity(state):
    return state.canonical_momenta / (
        state.gravitational_masses[:, None] * state.scale_factor
    )


def test_pair_owned_drag_diffusion_is_momentum_energy_and_psd_conservative():
    plan, state = _case()
    dt = _step_for_drag(plan, state, 0.02)
    result = plan.apply(state, jr.key(5), 11, dt)

    assert bool(result.successful)
    assert bool(result.diagnostics.diffusion_psd)
    assert bool(result.diagnostics.all_pairs_covered)
    np.testing.assert_allclose(
        result.diagnostics.total_momentum_defect, 0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        result.diagnostics.total_kinetic_energy_defect, 0.0, atol=2.0e-13
    )
    covariance = result.diagnostics.pair_diffusion_covariance
    eigenvalues = jnp.linalg.eigvalsh(covariance)
    assert bool(jnp.all(eigenvalues >= -2.0e-14))


def test_sampled_first_and_second_kramers_moyal_moments_match_evidence():
    plan, state = _case()
    dt = _step_for_drag(plan, state, 0.01)
    keys = jr.split(jr.key(73), 2048)

    def sample(key):
        result = plan.apply(state, key, 9, dt)
        velocity = _velocity(result.accepted_state)
        return velocity[0] - velocity[1], result.successful

    relative_after, successful = jax.jit(jax.vmap(sample))(keys)
    assert bool(jnp.all(successful))
    relative_before = _velocity(state)[0] - _velocity(state)[1]
    increments = relative_after - relative_before
    reference = plan.apply(state, jr.key(0), 9, dt)
    pair = jnp.argmax(reference.diagnostics.selected_pairs.astype(jnp.int32))
    drag = reference.diagnostics.target_drag_fraction[pair]
    expected_mean = -drag * relative_before
    observed_mean = jnp.mean(increments, axis=0)
    np.testing.assert_allclose(observed_mean, expected_mean, atol=2.0e-2)

    centered = increments - observed_mean
    observed_covariance = jnp.einsum("ni,nj->ij", centered, centered) / keys.shape[0]
    expected_covariance = reference.diagnostics.pair_diffusion_covariance[pair]
    np.testing.assert_allclose(
        observed_covariance, expected_covariance, rtol=0.12, atol=2.0e-3
    )


def test_frequent_schedule_covers_every_supported_edge_once_with_aggregate_bound():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((3, 1, 2)), jnp.ones((3,)), ambient_dimension=3
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3, box=box).prepare(
        particles
    )
    species = DarkSectorSpeciesPlan("chi", 1.0)
    split = SmallAngleSplitPlan(
        TwoBodyDifferentialKernelPlan.constant_isotropic(species, 1.0e-3),
        0.8,
    )
    plan = FrequentSmallAngleSIDMPlan(
        neighborhood,
        phx.discretization.WendlandC2SPHKernel(3),
        split,
        smoothing_length_comoving=0.5,
        maximum_drag_fraction_per_step=0.1,
        maximum_transverse_variance_per_step=0.2,
        moment_tolerance=0.2,
    )
    positions = jnp.asarray(((0.45, 0.5, 0.5), (0.5, 0.55, 0.5), (0.55, 0.5, 0.5)))
    velocity = jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    state = WeightedSIDMPacketState(
        positions,
        jnp.ones((3,)),
        jnp.ones((3,)),
        jnp.ones((3,)),
        0.5 * velocity,
        jnp.ones((3,), dtype=bool),
        particles.particle_ids,
        jnp.full((3,), -1, dtype=jnp.int64),
        jnp.zeros((3,), dtype=jnp.int32),
        0.5,
    )
    result = plan.apply(state, jr.key(31), 7, 1.0e-5)

    assert bool(result.successful)
    assert bool(result.diagnostics.all_pairs_covered)
    assert int(jnp.sum(result.diagnostics.selected_pairs)) == 3
    np.testing.assert_allclose(result.diagnostics.selected_pair_fraction, 1.0, atol=0.0)
    assert bool(
        jnp.all(
            result.diagnostics.particle_aggregate_drag_fraction
            <= plan.maximum_drag_fraction_per_step
        )
    )


def test_split_angle_sweep_reconstructs_full_kernel_without_gap_or_overlap():
    species = DarkSectorSpeciesPlan("chi", 1.0)
    kernel = TwoBodyDifferentialKernelPlan.constant_isotropic(species, 0.2)
    small_transfer = []
    rare_transfer = []
    for split_cosine in (-0.5, 0.0, 0.5, 0.9):
        split = SmallAngleSplitPlan(kernel, split_cosine)
        moments = split.moments(jnp.asarray(2.0))
        assert bool(moments.successful)
        assert bool(split.no_gap)
        assert bool(split.no_overlap)
        np.testing.assert_allclose(
            moments.small.transfer + moments.rare.transfer,
            moments.total.transfer,
            rtol=2.0e-13,
        )
        np.testing.assert_allclose(
            moments.small.viscosity + moments.rare.viscosity,
            moments.total.viscosity,
            rtol=2.0e-13,
        )
        small_transfer.append(float(moments.small.transfer))
        rare_transfer.append(float(moments.rare.transfer))

    assert np.all(np.diff(small_transfer) < 0.0)
    assert np.all(np.diff(rare_transfer) > 0.0)


def test_frequent_profile_refuses_invalid_timestep_without_switching_regime():
    plan, state = _case(maximum_drag=0.05)
    dt = _step_for_drag(plan, state, 0.5)
    result = plan.apply(state, jr.key(4), 3, dt)

    assert not bool(result.diagnostics.timestep_valid)
    assert not bool(result.successful)
    for actual, expected in zip(
        jax.tree.leaves(result.accepted_state),
        jax.tree.leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_frequent_profile_refuses_unequal_packet_weights_without_partial_update():
    plan, state = _case(weights=(2.0, 1.0))
    result = plan.apply(state, jr.key(19), 2, 1.0e-3)

    assert not bool(result.diagnostics.equal_pair_weights)
    assert not bool(result.successful)
    for actual, expected in zip(
        jax.tree.leaves(result.accepted_state),
        jax.tree.leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_tiny_unit_mass_relation_rejects_order_unity_relative_error():
    plan, state = _case(microscopic_mass=1.0e-30)
    invalid = eqx.tree_at(
        lambda value: value.gravitational_masses,
        state,
        2.0 * state.gravitational_masses,
    )
    result = plan.apply(invalid, jr.key(8), 1, 1.0e-3)

    assert not bool(result.diagnostics.mass_relation_valid)
    assert not bool(result.successful)


def test_frequent_profile_refuses_nonidentical_species_kernel():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((1, 2)), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    box = phx.discretization.ParticleBox(jnp.zeros((3,)), jnp.ones((3,)))
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=box).prepare(
        particles
    )
    first = DarkSectorSpeciesPlan("a", 1.0)
    second = DarkSectorSpeciesPlan("b", 2.0)
    kernel = TwoBodyDifferentialKernelPlan.constant_isotropic(first, 0.01)
    kernel = eqx.tree_at(lambda value: value.second_species, kernel, second)
    split = SmallAngleSplitPlan(kernel, 0.5)

    with pytest.raises(ValueError, match="one elastic species"):
        FrequentSmallAngleSIDMPlan(
            neighborhood,
            phx.discretization.WendlandC2SPHKernel(3),
            split,
            smoothing_length_comoving=0.5,
        )


def test_frequent_profile_refuses_azimuth_dependent_kernel_without_tensor_moments():
    plan, _ = _case()
    anisotropic_split = eqx.tree_at(
        lambda value: value.kernel.azimuths,
        plan.split,
        jnp.asarray((0.0, jnp.pi, 2.0 * jnp.pi)),
        is_leaf=lambda value: value is None,
    )
    with pytest.raises(ValueError, match="axisymmetric kernel"):
        FrequentSmallAngleSIDMPlan(
            plan.neighborhood,
            plan.spatial_kernel,
            anisotropic_split,
            smoothing_length_comoving=plan.smoothing_length_comoving,
        )
