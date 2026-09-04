import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.incompressible_flow._boundary_turbulence import (
    StochasticTurbulentInflowPlan,
    StochasticTurbulentInflowState,
    VectorEquilibriumWallStressPlan,
)


def test_vector_wall_stress_is_orientation_invariant_and_opposes_slip():
    prepared = VectorEquilibriumWallStressPlan().prepare(3)
    velocity = jnp.asarray(((0.0, 3.0, 4.0), (0.0, 3.0, 4.0)))
    normals = jnp.asarray(((2.0, 0.0, 0.0), (-5.0, 0.0, 0.0)))
    result = prepared.evaluate(velocity, normals, 0.02, 1.2, 1.8e-5)

    assert bool(jnp.all(result.successful))
    np.testing.assert_allclose(result.traction[0], result.traction[1], rtol=1e-12)
    np.testing.assert_allclose(result.traction[:, 0], 0.0, atol=1e-14)
    assert bool(jnp.all(jnp.sum(result.traction * velocity, axis=-1) < 0.0))
    assert bool(jnp.all(result.boundary_power < 0.0))
    assert prepared.plan.pressure_gradient_support.startswith("none")

    rough = prepared.evaluate(
        velocity[:1], normals[:1], 0.02, 1.2, 1.8e-5, roughness_height=1.0e-3
    )
    assert bool(rough.successful[0])
    assert float(rough.wall_shear_magnitude[0]) > float(result.wall_shear_magnitude[0])


def test_vector_wall_stress_has_exact_zero_and_viscous_limits():
    prepared = VectorEquilibriumWallStressPlan(root_tolerance=1.0e-10).prepare(2)
    normal = jnp.asarray((1.0, 0.0))
    zero = prepared.evaluate(jnp.zeros((2,)), normal, 0.01, 1.2, 1.8e-5)
    np.testing.assert_array_equal(zero.traction, jnp.zeros((2,)))
    np.testing.assert_array_equal(zero.friction_velocity, 0.0)
    np.testing.assert_array_equal(zero.boundary_power, 0.0)
    assert bool(zero.converged)
    assert bool(zero.successful)

    speed = 1.0e-9
    laminar = prepared.evaluate(jnp.asarray((0.0, speed)), normal, 0.01, 1.2, 1.8e-5)
    expected_shear = 1.8e-5 * speed / 0.01
    np.testing.assert_allclose(
        laminar.wall_shear_magnitude, expected_shear, rtol=2.0e-3, atol=0.0
    )
    np.testing.assert_allclose(laminar.traction[1], -expected_shear, rtol=2.0e-3)
    assert float(laminar.evidence.y_plus) < 1.0e-2
    assert bool(laminar.successful)


def test_vector_wall_stress_refuses_nonconvergence_and_invalid_support():
    underresolved = VectorEquilibriumWallStressPlan(
        root_iterations=1,
        bracket_iterations=1,
        root_tolerance=1.0e-14,
    ).prepare(2)
    result = underresolved.evaluate(
        jnp.asarray((0.0, 12.0)), jnp.asarray((1.0, 0.0)), 0.01, 1.0, 1.0e-5
    )
    assert not bool(result.converged)
    assert not bool(result.successful)

    prepared = VectorEquilibriumWallStressPlan().prepare(2)
    nontangent = prepared.evaluate(
        jnp.asarray((0.1, 2.0)), jnp.asarray((1.0, 0.0)), 0.01, 1.0, 1.0e-5
    )
    assert not bool(nontangent.evidence.tangential_velocity_valid)
    assert not bool(nontangent.successful)
    unsupported_roughness = prepared.evaluate(
        jnp.asarray((0.0, 2.0)),
        jnp.asarray((1.0, 0.0)),
        0.01,
        1.0,
        1.0e-5,
        roughness_height=0.005,
    )
    assert not bool(unsupported_roughness.evidence.roughness_valid)
    assert not bool(unsupported_roughness.successful)


def test_vector_wall_stress_jit_preserves_result_and_evidence():
    prepared = VectorEquilibriumWallStressPlan().prepare(3)
    arguments = (
        jnp.asarray(((0.0, 2.0, -1.0), (0.0, -3.0, 0.5))),
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.01, 0.02)),
        jnp.asarray((1.0, 1.2)),
        jnp.asarray((1.0e-5, 1.8e-5)),
    )
    eager = prepared.evaluate(*arguments)
    compiled = eqx.filter_jit(prepared.evaluate)(*arguments)
    np.testing.assert_allclose(compiled.traction, eager.traction, rtol=1e-12)
    np.testing.assert_allclose(compiled.friction_velocity, eager.friction_velocity)
    np.testing.assert_array_equal(compiled.successful, eager.successful)


def _compact_prepared():
    coordinates = jnp.asarray(((0.0, 0.0), (0.0, 0.2), (0.0, 1.0), (0.0, 1.2)))
    velocity_covariance = jnp.asarray(((0.4, 0.1), (0.1, 0.3)))
    scalar_covariance = jnp.asarray(((0.2,),))
    cross_covariance = jnp.asarray(((0.05,), (-0.02,)))
    prepared = StochasticTurbulentInflowPlan(
        "compact", compact_support_radius=0.25
    ).prepare(
        coordinates,
        jnp.asarray((1.0, 0.0)),
        jnp.ones((4,)),
        velocity_covariance,
        scalar_covariance=scalar_covariance,
        velocity_scalar_covariance=cross_covariance,
    )
    target = np.block(
        [
            [np.asarray(velocity_covariance), np.asarray(cross_covariance)],
            [np.asarray(cross_covariance).T, np.asarray(scalar_covariance)],
        ]
    )
    return prepared, target


def test_compact_inflow_matches_joint_covariance_and_rejects_non_psd_inputs():
    prepared, target = _compact_prepared()
    assert bool(prepared.preparation.covariance_exact)
    assert bool(prepared.preparation.mass_compatible)
    assert not bool(prepared.preparation.divergence_available)

    def draw(state, _):
        result = prepared.sample(state)
        joint = jnp.concatenate(
            (result.velocity_fluctuation[0], result.scalar_fluctuation[0])
        )
        return result.state, joint

    _, draws = jax.lax.scan(
        draw, prepared.initialize(jax.random.key(83)), xs=None, length=20_000
    )
    empirical = np.cov(np.asarray(draws).T, bias=True)
    np.testing.assert_allclose(empirical, target, rtol=0.04, atol=0.012)

    plan = StochasticTurbulentInflowPlan("compact", compact_support_radius=1.0)
    geometry = (
        jnp.asarray(((0.0, 0.0), (0.0, 0.5))),
        jnp.asarray((1.0, 0.0)),
        jnp.ones((2,)),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        plan.prepare(*geometry, jnp.asarray(((1.0, 0.0), (0.0, -0.1))))
    with pytest.raises(ValueError, match="exactly symmetric"):
        plan.prepare(*geometry, jnp.asarray(((1.0, 0.2), (0.1, 1.0))))


def test_inflow_prng_lineage_reproducibility_and_restart_are_exact():
    prepared, _ = _compact_prepared()
    with pytest.raises(ValueError, match="typed JAX PRNG key"):
        prepared.initialize(jnp.asarray((0, 1), dtype=jnp.uint32))
    initial = prepared.initialize(jax.random.key(19))
    first = prepared.sample(initial)
    repeated = prepared.sample(initial)
    np.testing.assert_array_equal(first.velocity, repeated.velocity)
    np.testing.assert_array_equal(first.scalars, repeated.scalars)
    np.testing.assert_array_equal(
        jax.random.key_data(first.state.key), jax.random.key_data(repeated.state.key)
    )
    np.testing.assert_array_equal(
        jax.random.key_data(first.evidence.parent_key),
        jax.random.key_data(initial.key),
    )
    np.testing.assert_array_equal(
        jax.random.key_data(first.evidence.next_key),
        jax.random.key_data(first.state.key),
    )

    restored = StochasticTurbulentInflowState(
        key=first.state.key,
        sample_index=first.state.sample_index,
        prepared_id=first.state.prepared_id,
    )
    continued = prepared.sample(first.state)
    restarted = prepared.sample(restored)
    np.testing.assert_array_equal(continued.velocity, restarted.velocity)
    np.testing.assert_array_equal(continued.scalars, restarted.scalars)
    np.testing.assert_array_equal(
        jax.random.key_data(continued.state.key),
        jax.random.key_data(restarted.state.key),
    )
    assert int(continued.state.sample_index) == 2
    assert bool(continued.evidence.mass_compatible)
    np.testing.assert_allclose(
        continued.evidence.fluctuation_volume_flux, 0.0, atol=1e-12
    )


def test_spectral_inflow_certifies_surface_divergence_mass_and_jit():
    angles = 0.5 * jnp.pi * jnp.arange(4)
    coordinates = jnp.stack((jnp.zeros_like(angles), angles), axis=-1)
    velocity_covariance = jnp.asarray(((0.7, 0.0), (0.0, 0.0)))
    scalar_covariance = jnp.asarray(((0.2,),))
    cross_covariance = jnp.asarray(((0.1,), (0.0,)))
    prepared = StochasticTurbulentInflowPlan("spectral").prepare(
        coordinates,
        jnp.asarray((1.0, 0.0)),
        jnp.ones((4,)),
        velocity_covariance,
        scalar_covariance=scalar_covariance,
        velocity_scalar_covariance=cross_covariance,
        spectral_wavevectors=jnp.asarray(((0.0, 1.0), (0.0, 1.0))),
    )
    assert prepared.covariance_rank == 2
    assert bool(prepared.preparation.mass_compatible)
    assert bool(prepared.preparation.divergence_available)
    assert bool(prepared.preparation.divergence_compatible)
    assert "surface" in prepared.divergence_kind

    state = prepared.initialize(jax.random.key(7))
    eager = prepared.sample(
        state,
        mean_velocity=jnp.asarray((2.0, 0.0)),
        mean_scalars=jnp.asarray((10.0,)),
    )
    compiled = eqx.filter_jit(prepared.sample)(
        state,
        mean_velocity=jnp.asarray((2.0, 0.0)),
        mean_scalars=jnp.asarray((10.0,)),
    )
    np.testing.assert_array_equal(compiled.velocity, eager.velocity)
    np.testing.assert_array_equal(compiled.scalars, eager.scalars)
    np.testing.assert_allclose(eager.evidence.fluctuation_volume_flux, 0.0, atol=1e-12)
    np.testing.assert_allclose(eager.evidence.total_volume_flux, 8.0, atol=1e-12)
    np.testing.assert_allclose(eager.evidence.divergence_residual, 0.0, atol=1e-12)
    assert bool(eager.evidence.divergence_compatible)
    assert bool(eager.evidence.successful)
