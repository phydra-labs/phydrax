import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _context():
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0, 0.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )


def _circular_state():
    astro = phx.applications.astrodynamics
    return astro.CartesianOrbitState(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        _context(),
    )


def test_universal_kepler_closes_circular_orbit_and_jvp():
    astro = phx.applications.astrodynamics
    state = _circular_state()
    result = jax.jit(lambda dt: astro.propagate_universal_kepler(state, dt, 1.0))(
        2.0 * jnp.pi
    )
    assert bool(result.valid)
    np.testing.assert_allclose(result.state.position, state.position, atol=2.0e-11)
    np.testing.assert_allclose(result.state.velocity, state.velocity, atol=2.0e-11)
    assert float(result.angular_momentum_defect) < 1.0e-11

    tangent = jax.jvp(
        lambda dt: astro.propagate_universal_kepler(state, dt, 1.0).state.position,
        (jnp.asarray(0.3),),
        (jnp.asarray(1.0),),
    )[1]
    expected = jnp.asarray([-jnp.sin(0.3), jnp.cos(0.3), 0.0])
    np.testing.assert_allclose(tangent, expected, atol=2.0e-10)


def test_modified_equinoctial_round_trip_covers_circular_equatorial_state():
    astro = phx.applications.astrodynamics
    state = _circular_state()
    converted = astro.cartesian_to_modified_equinoctial(state, 1.0)
    restored, valid, status = astro.modified_equinoctial_to_cartesian(
        converted.elements, 1.0
    )
    assert bool(converted.valid)
    assert bool(valid)
    assert int(status) == int(astro.AstrodynamicsStatus.SUCCESS)
    np.testing.assert_allclose(restored.position, state.position, atol=1.0e-12)
    np.testing.assert_allclose(restored.velocity, state.velocity, atol=1.0e-12)

    classical = astro.cartesian_to_classical(state, 1.0)
    assert not bool(classical.valid)
    assert bool(classical.circular)
    assert bool(classical.equatorial)


def test_analytic_adaptive_and_symplectic_propagation_agree():
    astro = phx.applications.astrodynamics
    state = _circular_state()
    times = jnp.linspace(0.0, 1.0, 17)
    force = astro.PointMassGravity(1.0, state.context)
    adaptive = astro.AstrodynamicsPropagationPlan(
        force, times, relative_tolerance=1.0e-11, absolute_tolerance=1.0e-13
    ).solve(state)
    analytic = astro.AstrodynamicsPropagationPlan(force, times).solve_analytic_two_body(
        state
    )
    symplectic = astro.AstrodynamicsPropagationPlan(
        force,
        times,
        solver=phx.solver.StormerVerlet(3),
        dt0=times[1] - times[0],
    ).solve(state)
    assert bool(adaptive.successful)
    assert bool(analytic.successful)
    assert bool(symplectic.successful)
    np.testing.assert_allclose(
        adaptive.trajectory.states[-1], analytic.trajectory.states[-1], atol=2.0e-9
    )
    assert float(jnp.max(jnp.abs(symplectic.diagnostics.energy_drift))) < 2.0e-3


def test_invalid_two_body_inputs_return_status_without_shape_change():
    astro = phx.applications.astrodynamics
    state = _circular_state()
    result = jax.jit(lambda mu: astro.propagate_universal_kepler(state, 1.0, mu))(
        jnp.asarray(-1.0)
    )
    assert not bool(result.valid)
    assert int(result.status) == int(astro.AstrodynamicsStatus.INVALID_DOMAIN)
    assert result.state.position.shape == (3,)
