import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.compact_objects._advanced_perturbations import (
    ExcitationResiduePlan,
    MassiveFieldQuasiBoundPlan,
    QuadraticRingdownPlan,
)


jax.config.update("jax_enable_x64", True)


def test_massive_scalar_parameters_recover_neutral_schwarzschild_limit():
    mass = 2.0
    field_mass = 0.1
    plan = MassiveFieldQuasiBoundPlan(
        mass,
        0.0,
        0.0,
        field_mass,
        jnp.geomspace(4.01, 120.0, 64),
        ell=1,
        overtone=0,
        newton_steps=2,
    )
    seed = plan.hydrogenic_seed()
    parameters = plan.mode_parameters(seed)
    principal = 2.0

    np.testing.assert_allclose(parameters.horizon_radii, (0.0, 2.0 * mass))
    np.testing.assert_allclose(parameters.horizon_angular_velocity, 0.0)
    np.testing.assert_allclose(parameters.horizon_electric_potential, 0.0)
    np.testing.assert_allclose(parameters.surface_gravity, 1.0 / (4.0 * mass))
    np.testing.assert_allclose(
        jnp.real(seed),
        field_mass * np.sqrt(1.0 - (mass * field_mass / principal) ** 2),
    )
    assert bool(parameters.bound_state)
    assert bool(parameters.physically_valid)
    assert not bool(parameters.superradiant)


def test_charged_kerr_newman_shooting_returns_equation_residual_and_fixed_path():
    plan = MassiveFieldQuasiBoundPlan(
        1.0,
        0.2,
        0.1,
        0.2,
        jnp.geomspace(2.05, 80.0, 72),
        field_charge=0.05,
        ell=1,
        azimuthal=1,
        newton_steps=3,
        residual_tolerance=1.0e-6,
    )
    result = eqx.filter_jit(plan.solve)()

    assert result.radial_log_derivative.shape == plan.radial_nodes.shape
    np.testing.assert_allclose(
        result.shooting_residual,
        result.radial_log_derivative[-1]
        + result.parameters.radial_decay_rate
        - result.parameters.infinity_power / plan.radial_nodes[-1],
    )
    assert int(result.iterations) <= plan.newton_steps
    assert bool(result.finite)
    assert bool(result.parameters.bound_state)
    assert bool(result.converged)
    assert bool(result.physically_valid)
    assert bool(result.derivative_valid)
    assert bool(result.qualified)
    assert float(result.residual_norm) <= plan.residual_tolerance
    assert float(result.residual_norm) < abs(complex(result.hydrogenic_seed_residual))


def test_simple_pole_residues_assemble_causal_green_functions():
    poles = jnp.asarray((1.0 - 0.1j, 2.0 - 0.2j))
    numerators = jnp.asarray((2.0 + 1.0j, -1.0 + 0.5j))
    wronskian_derivatives = jnp.asarray((4.0 + 0.0j, 2.0 - 1.0j))
    plan = ExcitationResiduePlan(poles, numerators, wronskian_derivatives)
    residues = plan.residues()

    np.testing.assert_allclose(residues.residue, numerators / wronskian_derivatives)
    np.testing.assert_allclose(
        residues.nearest_pole_separation,
        np.full((2,), abs(complex(poles[0] - poles[1]))),
    )
    assert bool(residues.qualified)

    times = jnp.asarray((-1.0, 0.0, 0.75))
    time_green = plan.time_domain(times)
    np.testing.assert_allclose(time_green.values[0], 0.0)
    np.testing.assert_allclose(time_green.values[1], jnp.sum(residues.residue))
    assert bool(time_green.causal)
    assert bool(time_green.stable_poles)
    assert bool(time_green.qualified)
    assert not time_green.branch_cut_included

    frequencies = jnp.asarray((0.0, 0.5))
    frequency_green = plan.frequency_domain(frequencies)
    expected = jnp.sum(
        residues.residue[None, :] / (frequencies[:, None] - poles[None, :]),
        axis=1,
    )
    np.testing.assert_allclose(frequency_green.values, expected)
    assert bool(frequency_green.qualified)
    assert not frequency_green.branch_cut_included


def test_quadratic_ringdown_uses_regular_resonant_duhamel_limit():
    poles = jnp.asarray((1.0 - 0.1j, 2.0 - 0.2j))
    amplitudes = jnp.asarray((1.5 + 0.25j, 0.0j))
    coupling = jnp.zeros((2, 2, 2), dtype="complex128").at[1, 0, 0].set(0.3 - 0.1j)
    plan = QuadraticRingdownPlan(poles, amplitudes, coupling)
    times = jnp.asarray((0.0, 0.4, 1.0))
    result = eqx.filter_jit(plan.evaluate)(times)

    expected_child = (
        coupling[1, 0, 0] * amplitudes[0] ** 2 * times * jnp.exp(-1.0j * poles[1] * times)
    )
    np.testing.assert_allclose(result.quadratic_modes[:, 1], expected_child)
    np.testing.assert_allclose(result.quadratic_modes[:, 0], 0.0)
    assert bool(result.resonant_pairs[1, 0, 0])
    assert bool(result.finite)
    assert bool(result.stable)
    assert bool(result.qualified)
