import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.semiconductor._kinetics import (
    DynamicTrap,
    HBAR_SI,
    NonlocalTunnelingPath,
    WKBBarrierPath,
)
from phydrax.applications.semiconductor._quantities import ELEMENTARY_CHARGE_SI as Q


def _trap(kind="bulk", density=2e21):
    return DynamicTrap(
        density,
        2e-14,
        1e-14,
        1e6,
        3e5,
        0.1 * Q,
        empty_charge_number=0,
        population_kind=kind,
        temperature_range=(250.0, 400.0),
        energy_reference="synthetic zero",
        provenance="synthetic single-level rates; not trap/reliability calibration",
    )


def test_trap_exact_step_is_bounded_and_ledger_matches_inventory_change():
    trap = _trap()
    initial = jnp.array([0.0, 1.0, 0.2])
    dt = jnp.array([1e-14, 1e-8, 1e-2])
    result = trap.advance(initial, 3e20, 2e20, 300.0, 0.8 * Q, 0.4 * Q, dt)
    equilibrium = 6.3e6 / 9.3e6
    expected = equilibrium + (initial - equilibrium) * np.exp(-9.3e6 * dt)
    np.testing.assert_allclose(result.occupancy, expected, rtol=1e-9, atol=1e-15)
    assert np.all(result.occupancy >= 0)
    assert np.all(result.occupancy <= 1)
    exchange = result.average_exchange
    np.testing.assert_allclose(
        dt * exchange.trap_population_source / trap.density,
        result.occupancy - initial,
        rtol=1e-9,
        atol=1e-15,
    )
    charge_sum = (
        exchange.electron_charge_source
        + exchange.hole_charge_source
        + exchange.trap_charge_source
    )
    np.testing.assert_allclose(charge_sum / (Q * trap.density * 1e7), 0.0, atol=1e-15)
    energy_sum = (
        exchange.electron_energy_source
        + exchange.hole_energy_source
        + exchange.trap_energy_source
        + exchange.lattice_energy_source
    )
    np.testing.assert_allclose(energy_sum / (Q * trap.density * 1e7), 0.0, atol=1e-15)
    initial_charge = trap.storage(initial).charge_density
    final_charge = trap.storage(result.occupancy).charge_density
    np.testing.assert_allclose(
        (final_charge - initial_charge) / (Q * trap.density),
        dt * exchange.trap_charge_source / (Q * trap.density),
        rtol=1e-9,
        atol=1e-15,
    )
    # Relaxation cannot legitimize an invalid initial occupancy.
    invalid = trap.advance(1.1, 3e20, 2e20, 300.0, 0.8 * Q, 0.4 * Q, 1.0)
    assert not bool(invalid.successful)
    assert np.isnan(invalid.occupancy)


def test_trap_detailed_balance_and_energy_reference_invariance():
    trap = _trap()
    equilibrium = trap.evaluate(2 / 3, 1e20, 1.5e19, 300.0, 0.8 * Q, 0.4 * Q)
    np.testing.assert_allclose(
        equilibrium.electron_source / (trap.density * 1e6), 0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        equilibrium.hole_source / (trap.density * 1e6), 0.0, atol=1e-15
    )
    before = trap.evaluate(0.2, 3e20, 2e20, 300.0, 0.8 * Q, 0.4 * Q)
    shifted_trap = eqx.tree_at(
        lambda value: value.trap_energy, trap, trap.trap_energy + Q
    )
    after = shifted_trap.evaluate(0.2, 3e20, 2e20, 300.0, 1.8 * Q, -0.6 * Q)
    np.testing.assert_allclose(after.lattice_energy_source, before.lattice_energy_source)
    np.testing.assert_allclose(after.occupancy_rate, before.occupancy_rate)
    np.testing.assert_allclose(
        after.trap_energy_source - before.trap_energy_source,
        Q * before.trap_population_source,
    )


def test_surface_inventory_uses_area_not_neighbor_volume():
    bulk, surface = _trap(), _trap("surface", 2e15)
    volume, area = 3e-18, 3e-12
    args = (0.25, 3e20, 2e20, 300.0, 0.8 * Q, 0.4 * Q)
    bulk_exchange, surface_exchange = bulk.evaluate(*args), surface.evaluate(*args)
    np.testing.assert_allclose(
        volume * bulk_exchange.electron_source, area * surface_exchange.electron_source
    )
    np.testing.assert_allclose(
        volume * bulk.storage(0.25).charge_density,
        area * surface.storage(0.25).charge_density,
    )
    np.testing.assert_allclose(
        volume * bulk_exchange.lattice_energy_source,
        area * surface_exchange.lattice_energy_source,
    )


def _barrier(positions, energies, masses, minimum_action=0.0):
    return WKBBarrierPath(
        positions,
        energies,
        masses,
        minimum_action=minimum_action,
        energy_reference="synthetic zero",
        provenance="analytic scalar barrier fixture; no material complex-band claim",
    )


def test_wkb_rectangular_and_linear_turning_point_actions():
    mass, length, height, energy = 0.2 * 9.1093837139e-31, 4e-9, 1.0 * Q, 0.25 * Q
    rectangular = _barrier([0.0, length], [height, height], [mass])
    result = rectangular.evaluate(energy)
    expected = length * np.sqrt(2 * mass * (height - energy)) / HBAR_SI
    np.testing.assert_allclose(result.action, expected)
    np.testing.assert_allclose(result.transmission, np.exp(-2 * expected))
    linear = _barrier([0.0, length], [height, 0.0], [mass])
    result = linear.evaluate(energy)
    expected = (
        (2 / 3)
        * length
        * np.sqrt(2 * mass)
        * (height - energy) ** 1.5
        / (height * HBAR_SI)
    )
    np.testing.assert_allclose(result.action, expected)
    np.testing.assert_allclose(
        result.forbidden_length, length * (height - energy) / height
    )
    # The moving linear turning point's derivative is finite and analytic.
    derivative = jax.grad(lambda e: linear.evaluate(e).action)(energy)
    expected_derivative = (
        -length * np.sqrt(2 * mass * (height - energy)) / (height * HBAR_SI)
    )
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-10)
    over = rectangular.evaluate(2 * height)
    assert float(over.transmission) == 1.0
    assert float(over.forbidden_length) == 0.0


def test_wkb_rejects_resonant_multibarrier_and_unadmitted_small_action():
    mass = 0.2 * 9.1093837139e-31
    two_barriers = _barrier([0.0, 2e-9, 4e-9], [Q, 0.0, Q], [mass, mass])
    result = two_barriers.evaluate(0.5 * Q)
    assert not bool(result.successful)
    assert np.isnan(result.transmission)
    thin = _barrier([0.0, 1e-12], [Q, Q], [mass], minimum_action=2.0)
    assert not bool(thin.evaluate(0.5 * Q).successful)


def test_nonlocal_tunneling_has_nodewise_charge_and_energy_incidence():
    mass = 0.2 * 9.1093837139e-31
    barrier = _barrier([0.0, 2e-9, 4e-9], [Q, Q, Q], [mass, mass])
    path = NonlocalTunnelingPath(
        barrier, 5, [3, 1, 4], provenance="synthetic spectral channel"
    )
    result = path.evaluate(0.25 * Q, 0.8, 0.1, 1e12, 0.5 * Q, 0.0)
    assert bool(result.successful)
    assert float(result.pair_rate) > 0
    tails, heads = path.path_nodes[:-1], path.path_nodes[1:]
    current_incidence = (
        jnp.zeros(5)
        .at[tails]
        .add(-result.path_charge_current)
        .at[heads]
        .add(result.path_charge_current)
    )
    power_incidence = (
        jnp.zeros(5)
        .at[tails]
        .add(-result.path_energy_power)
        .at[heads]
        .add(result.path_energy_power)
    )
    np.testing.assert_allclose(current_incidence, result.charge_source)
    np.testing.assert_allclose(power_incidence, result.total_energy_source)
    np.testing.assert_allclose(
        result.electron_kinetic_energy_source
        + result.hole_kinetic_energy_source
        + result.band_storage_source,
        result.total_energy_source,
    )
    np.testing.assert_allclose(
        jnp.sum(result.total_energy_source + result.lattice_energy_source), 0.0, atol=0.0
    )
    reverse = path.evaluate(0.25 * Q, 0.1, 0.8, 1e12, 0.5 * Q, 0.0)
    np.testing.assert_allclose(reverse.path_charge_current, -result.path_charge_current)
    equilibrium = path.evaluate(0.25 * Q, 0.3, 0.3, 1e12, 0.5 * Q, 0.0)
    assert float(equilibrium.pair_rate) == 0.0
    shifted_barrier = eqx.tree_at(
        lambda value: value.barrier_energies, barrier, barrier.barrier_energies + Q
    )
    shifted = NonlocalTunnelingPath(
        shifted_barrier, 5, [3, 1, 4], provenance="same channel shifted energy datum"
    )
    after = shifted.evaluate(1.25 * Q, 0.8, 0.1, 1e12, 1.5 * Q, Q)
    np.testing.assert_allclose(after.path_charge_current, result.path_charge_current)
    np.testing.assert_allclose(
        after.electron_kinetic_energy_source, result.electron_kinetic_energy_source
    )
    np.testing.assert_allclose(
        after.hole_kinetic_energy_source, result.hole_kinetic_energy_source
    )
