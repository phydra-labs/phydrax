import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.compact_objects._advanced_perturbations import (
    ExcitationResiduePlan,
    MassiveFieldQuasiBoundPlan,
    QuadraticRingdownPlan,
)


jax.config.update("jax_enable_x64", True)


def test_advanced_black_hole_perturbations_scenario_1() -> None:
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


def test_quadratic_ringdown_uses_regular_resonant_duhamel_limit() -> None:
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


@pytest.mark.parametrize(("time_count", "mode_count"), [(1, 2), (7, 4), (31, 7)])
@pytest.mark.parametrize("chunk", [1, 8, 64])
def test_ringdown_matches_independent_duhamel_quadrature(
    time_count: int, mode_count: int, chunk: int
) -> None:
    # Integrate the driven ODE directly, without the implementation's sinc formula.
    poles = np.linspace(0.7, 2.4, mode_count) - 0.1j
    poles[1] = 2.0 * poles[0]
    if mode_count > 2:
        poles[2] = 2.0 * poles[0] + 1.0e-11
    amplitudes = np.linspace(0.1, 0.8, mode_count) + 0.2j
    rng = np.random.default_rng(410)
    coupling = rng.normal(size=(mode_count,) * 3) + 1.0j * rng.normal(
        size=(mode_count,) * 3
    )
    times = np.linspace(0.0, 3.0, time_count)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    expected = np.zeros((time_count, mode_count), dtype=np.complex128)
    detuning = np.empty((mode_count,) * 3, dtype=np.complex128)
    for k in range(mode_count):
        for i in range(mode_count):
            for j in range(mode_count):
                detuning[k, i, j] = poles[i] + poles[j] - poles[k]
                for t_index, time in enumerate(times):
                    source_times = 0.5 * time * (nodes + 1.0)
                    integrand = np.exp(
                        -1.0j * poles[k] * (time - source_times)
                        - 1.0j * (poles[i] + poles[j]) * source_times
                    )
                    expected[t_index, k] += (
                        coupling[k, i, j]
                        * amplitudes[i]
                        * amplitudes[j]
                        * 0.5
                        * time
                        * np.dot(weights, integrand)
                    )
    plan = QuadraticRingdownPlan(poles, amplitudes, coupling, time_chunk_size=chunk)
    result = eqx.filter_jit(plan.evaluate)(jnp.asarray(times))
    np.testing.assert_allclose(result.quadratic_modes, expected, rtol=2e-12, atol=2e-13)
    expected_linear = np.exp(-1.0j * times[:, None] * poles[None, :]) @ amplitudes
    np.testing.assert_allclose(
        result.linear_signal, expected_linear, rtol=2e-12, atol=2e-13
    )
    np.testing.assert_allclose(
        result.total_signal,
        expected_linear + expected.sum(axis=1),
        rtol=2e-12,
        atol=2e-13,
    )
    np.testing.assert_allclose(result.pair_detuning, detuning, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(result.resonant_pairs, np.abs(detuning) <= 1.0e-8)
    assert bool(result.finite)
    assert bool(result.converged)
    assert bool(result.stable)
    assert bool(result.physically_valid)
    assert bool(result.qualified)
    assert bool(result.derivative_valid)
    assert int(result.status) == 0
    assert result.time_chunk_size == chunk
    assert (
        result.logical_response_workspace_bytes
        == 16 * min(chunk, time_count) * mode_count**3
    )


def test_ringdown_workspace_stops_growing_after_time_chunk() -> None:
    plan = QuadraticRingdownPlan(
        np.array([0.7 - 0.1j, 1.4 - 0.2j]),
        np.ones(2, dtype=np.complex128),
        np.ones((2, 2, 2), dtype=np.complex128),
        time_chunk_size=8,
    )
    small = plan.evaluate(jnp.linspace(0.0, 2.0, 16))
    large = plan.evaluate(jnp.linspace(0.0, 2.0, 32))
    assert (
        large.logical_response_workspace_bytes / small.logical_response_workspace_bytes
        <= 2.25
    )
    assert large.quadratic_modes.shape == (32, 2)


@pytest.mark.parametrize("chunk", [0, -1])
def test_ringdown_refuses_nonpositive_chunk(chunk: int) -> None:
    with pytest.raises(ValueError, match="time_chunk_size must be positive"):
        QuadraticRingdownPlan(
            np.array([1.0 - 0.1j]),
            np.ones(1),
            np.ones((1, 1, 1)),
            time_chunk_size=chunk,
        )


def test_ringdown_unstable_poles_preserve_failure_evidence() -> None:
    plan = QuadraticRingdownPlan(np.array([1.0 + 0.1j]), np.ones(1), np.ones((1, 1, 1)))
    result = plan.evaluate(jnp.asarray([0.0, 1.0]))
    assert bool(result.finite)
    assert bool(result.converged)
    assert not bool(result.stable)
    assert not bool(result.physically_valid)
    assert not bool(result.qualified)
    assert not bool(result.derivative_valid)
    assert int(result.status) == 1
