import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.polymer_liquids import reptation as rep


def test_particle_reptation_observables_use_unwrapped_time_origins():
    base = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    translation = jnp.arange(10.0)[:, None, None] * jnp.asarray([1.0, 0.0, 0.0])
    trajectory = base[None, :, :] + translation
    result = rep.reptation_observables(
        rep.ReptationObservablePlan(10, 6, 2, 3, maximum_modes=2),
        trajectory,
        [[0, 1, 2], [3, 4, 5]],
        [[True, True, True], [True, True, True]],
        [0, 1, 2],
        0.5,
        coordinate_representation="unwrapped",
    )
    assert result.successful
    np.testing.assert_allclose(result.monomer_msd, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(result.center_of_mass_msd, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(result.internal_msd, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.end_to_end_correlation, 1.0)
    np.testing.assert_allclose(result.rouse_mode_msd, 0.0, atol=1.0e-12)


def test_chain_scaling_and_linear_tube_spectra():
    scaling = rep.fit_chain_length_scaling(
        rep.ChainLengthScalingPlan(exponent_interval=(2.9, 3.1)),
        [10.0, 20.0, 40.0, 80.0],
        [1.0e3, 8.0e3, 64.0e3, 512.0e3],
    )
    assert scaling.successful
    np.testing.assert_allclose(scaling.exponent, 3.0, rtol=1.0e-12)

    doi_edwards = rep.doi_edwards_linear_rheology(
        rep.DoiEdwardsTubePlan(100.0, 2.0, odd_mode_count=32),
        [0.0, 1.0, 10.0],
        [0.01, 0.1, 1.0],
    )
    assert doi_edwards.successful
    np.testing.assert_allclose(doi_edwards.tube_survival[0], 1.0)
    assert np.all(np.diff(np.asarray(doi_edwards.tube_survival)) < 0.0)

    lm = rep.likhtman_mcleish_linear_rheology(
        rep.LikhtmanMcLeishPlan(10, 1.0, 2.0, odd_mode_count=32),
        [0.0, 1.0, 10.0],
        [0.01, 0.1, 1.0],
    )
    assert lm.successful
    np.testing.assert_allclose(lm.rheology.relaxation_modulus[0], lm.zero_time_modulus)
    assert np.all(np.asarray(lm.rheology.storage_modulus) >= 0.0)
    assert np.all(np.asarray(lm.rheology.loss_modulus) >= 0.0)


def test_slip_spring_birth_and_death_obey_metropolis_hastings_ratio():
    prepared = rep.SlipSpringPlan(2, 1, 1.0, 2.0, 1.0).prepare([[0, 1]])
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    initial = prepared.initialize(jax.random.key(13))
    birth = prepared.step(initial, positions)
    death = prepared.step(birth.accepted_state, positions)

    assert birth.accepted & death.accepted
    assert birth.attempted_birth & ~death.attempted_birth
    np.testing.assert_allclose(birth.log_hastings_ratio, 0.0)
    np.testing.assert_allclose(death.log_hastings_ratio, 0.0)
    assert int(jnp.sum(birth.accepted_state.active_mask)) == 1
    assert int(jnp.sum(death.accepted_state.active_mask)) == 0


def test_glamm_equilibrium_is_invariant_and_transactional():
    plan = rep.GLAMMPlan(
        5,
        0.01,
        2.0,
        10.0,
        1.0,
        contour_diffusivity=0.0,
    )
    state = plan.initialize(dtype=jnp.float64)
    result = rep.glamm_step(plan, state, jnp.zeros((3, 3), dtype=jnp.float64))
    assert result.accepted & result.successful
    np.testing.assert_allclose(result.accepted_state.conformation, state.conformation)
    np.testing.assert_allclose(result.accepted_state.stretch, state.stretch)
    np.testing.assert_allclose(result.cauchy_stress, 0.0)
    np.testing.assert_allclose(result.flow_power_density, 0.0)
